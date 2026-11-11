"""Simulation loop: car physics, actuator delay, observation noise, crashes and restarts, lap timing."""

from __future__ import annotations

import collections
import json
import os
import random
import time
from math import fmod

import jax.numpy as jnp
import numpy as np

from race.config import GLOBAL_SPEED_LIMIT, ROOT, TIMESTEP_SIM, CarParameters
from race.dynamics import car_dynamics
from race.state import (
    ANGULAR_VEL_Z_IDX, LINEAR_VEL_X_IDX, LINEAR_VEL_Y_IDX, POSE_THETA_COS_IDX, POSE_THETA_IDX, POSE_THETA_SIN_IDX,
    POSE_X_IDX, POSE_Y_IDX, STATE_DIM,
)
from race.track import LapTimer, Track


def wrap_angle(angle: float) -> float:
    m = fmod(angle, 2 * np.pi)
    if m < -np.pi:
        return m + 2 * np.pi
    if m > np.pi:
        return m - 2 * np.pi
    return m


def normalize_yaw(state: np.ndarray) -> np.ndarray:
    state[POSE_THETA_IDX] = wrap_angle(float(state[POSE_THETA_IDX]))
    state[POSE_THETA_COS_IDX] = np.cos(state[POSE_THETA_IDX])
    state[POSE_THETA_SIN_IDX] = np.sin(state[POSE_THETA_IDX])
    return state


class Simulation:
    def __init__(self, cfg):
        self.cfg = cfg
        sim = cfg.simulation
        self.sim = sim
        random.seed(sim.seed)
        np.random.seed(sim.seed)
        import torch
        torch.manual_seed(sim.seed)
        torch.set_num_threads(1)

        self.car = CarParameters(friction=sim.friction)
        self.car_params = jnp.array(self.car.to_array())
        self.track = Track(sim.map_dir, sim.map, sim.waypoints, float(self.car.length), float(self.car.width))
        noise = sim.state_noise
        self.noise_level = np.array(noise if isinstance(noise, list) else [noise] * STATE_DIM, dtype=np.float64)

        if cfg.controller == "atmppi":
            from race.controllers.atmppi import AutoTuningMPPI
            self.controller = AutoTuningMPPI(cfg.atmppi, sim, self.car, self.track)
            self.controller_name = f"Auto-Tuning MPPI, {cfg.atmppi.rollouts} rollouts"
        else:
            from race.controllers.mppi import MPPI
            self.controller = MPPI(cfg.mppi, sim, self.car, self.track)
            self.controller_name = f"MPPI, {cfg.mppi.rollouts} rollouts"

        self.lap_timer = LapTimer(len(self.track.waypoints), self._lap_complete)
        self.viewer = None
        if sim.render:
            from race.render import NO_DISPLAY_HELP, Viewer, display_available
            if display_available():
                self.viewer = Viewer(self.track, f"{self.controller_name} on {sim.map}")
            else:
                print(NO_DISPLAY_HELP, flush=True)

        self.time = 0.0  # simulated time [s]
        self.clock = 0.0  # lap-timer clock, advanced once per control step
        self.laps: list[dict] = []
        self.crashes: list[float] = []
        self.state = np.zeros(STATE_DIM)

    # ------------------------------------------------------------------
    def reset(self) -> None:
        """Start (or restart after a crash) at a random pose near the waypoints, from rest."""
        pose = self.track.random_start()
        state = np.zeros(STATE_DIM)
        state[POSE_X_IDX], state[POSE_Y_IDX], state[POSE_THETA_IDX] = pose
        self.state = normalize_yaw(state)
        delay = int(self.sim.control_delay / TIMESTEP_SIM)
        self.delay_buffer = collections.deque([np.zeros(2) for _ in range(delay)])
        self.in_collision = False
        self.track.reset()
        self.lap_timer.reset()
        self.controller.reset()
        self.track.update_nearest(self.state.astype(np.float32))

    def _observe(self) -> np.ndarray:
        noisy = self.state + self.noise_level * np.random.uniform(-1, 1, STATE_DIM)
        noisy[POSE_THETA_COS_IDX] = np.cos(noisy[POSE_THETA_IDX])
        noisy[POSE_THETA_SIN_IDX] = np.sin(noisy[POSE_THETA_IDX])
        return noisy.astype(np.float32)

    def _physics_step(self, control) -> None:
        new_state = np.array(car_dynamics(jnp.array(self.state), jnp.array(control), self.car_params, TIMESTEP_SIM,
                                          intermediate_steps=1), dtype=np.float64)
        if np.all(np.isfinite(new_state)):
            self.state = normalize_yaw(new_state)
        else:
            self.state[[LINEAR_VEL_X_IDX, LINEAR_VEL_Y_IDX, ANGULAR_VEL_Z_IDX]] = 0.0
        self.state[LINEAR_VEL_X_IDX] = np.clip(self.state[LINEAR_VEL_X_IDX], -GLOBAL_SPEED_LIMIT, GLOBAL_SPEED_LIMIT)
        if self.track.body_in_wall(self.state[POSE_X_IDX], self.state[POSE_Y_IDX], self.state[POSE_THETA_IDX]):
            self.in_collision = True

    def _lap_complete(self, lap_time: float) -> None:
        self.laps.append({"t": round(self.time, 4), "lap_time": float(lap_time),
                          "frozen": bool(getattr(self.controller, "frozen", False))})
        print(f"Lap {len(self.laps)}: {lap_time:.2f} s  (t = {self.time:.1f} s)", flush=True)
        self.controller.report_lap_time(float(lap_time))

    def step(self) -> None:
        """One control step: observe, plan, then the physics steps with the actuator delay."""
        observation = self._observe()
        self.track.update_nearest(observation)
        steering, accel = self.controller.step(observation)
        self.lap_timer.update(self.track.nearest_index, self.clock)
        self.clock += self.sim.timestep_control
        control = np.array([steering, accel])
        for _ in range(self.sim.substeps):
            self.delay_buffer.append(control)
            self._physics_step(self.delay_buffer.popleft())
            self.time += TIMESTEP_SIM
        self.track.update_nearest(self.state.astype(np.float32))
        if self.viewer is not None:
            self.viewer.draw(self.state, self.controller.render_layers, self._info)
        if self.in_collision:
            self.crashes.append(round(self.time, 4))
            print(f"Crash at t = {self.time:.1f} s, restarting", flush=True)
            self.controller.on_crash()
            self.reset()

    def _info(self) -> dict:
        return {"controller": self.controller_name, "map": self.sim.map, "time": self.time,
                "speed": float(self.state[LINEAR_VEL_X_IDX]), "laps": [l["lap_time"] for l in self.laps],
                "crashes": len(self.crashes)}

    # ------------------------------------------------------------------
    def run(self) -> dict:
        print(f"{self.controller_name} on {self.sim.map} ({self.sim.waypoints}), friction {self.sim.friction}, "
              f"state noise {self.sim.state_noise}, seed {self.sim.seed}, {self.sim.duration} s", flush=True)
        self.reset()
        steps = int(round(self.sim.duration / self.sim.timestep_control))
        min_step_time = 0.0
        if self.viewer is not None and self.sim.speed:
            min_step_time = self.sim.timestep_control / float(self.sim.speed)
        wall_start = time.time()
        for _ in range(steps):
            t0 = time.time()
            self.step()
            if self.viewer is not None and self.viewer.closed:
                print("Viewer closed, ending the run", flush=True)
                break
            spare = min_step_time - (time.time() - t0)
            if spare > 0:
                time.sleep(spare)
        if self.viewer is not None:
            self.viewer.close()
        return self._save(time.time() - wall_start)

    def _save(self, wall_seconds: float) -> dict:
        lap_times = [l["lap_time"] for l in self.laps]
        result = {
            "controller": self.cfg.controller,
            "rollouts": getattr(self.cfg, self.cfg.controller).rollouts,
            "map": self.sim.map, "waypoints": self.sim.waypoints, "friction": self.sim.friction,
            "state_noise": self.sim.state_noise, "seed": self.sim.seed,
            "sim_seconds": round(self.time, 4), "wall_seconds": round(wall_seconds, 1),
            "laps": self.laps, "crashes": self.crashes,
            "settings": _plain(self.cfg),
        }
        print(f"\n{len(lap_times)} laps, {len(self.crashes)} crashes in {self.time:.0f} s")
        if lap_times:
            print(f"best {min(lap_times):.2f} s, median {float(np.median(lap_times)):.2f} s")
            frozen = [l["lap_time"] for l in self.laps if l["frozen"]]
            if frozen:
                print(f"after the memory froze: {len(frozen)} laps, median {float(np.median(frozen)):.2f} s")
        out_dir = os.path.join(ROOT, self.sim.results_dir)
        os.makedirs(out_dir, exist_ok=True)
        name = f"{time.strftime('%Y-%m-%d_%H-%M-%S')}_{self.cfg.controller}_{self.sim.map}_seed{self.sim.seed}.json"
        path = os.path.join(out_dir, name)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=1)
        print(f"wrote {path}")
        return result


def _plain(value):
    if hasattr(value, "__dict__"):
        return {k: _plain(v) for k, v in vars(value).items() if k not in ("map_dir",)}
    return value
