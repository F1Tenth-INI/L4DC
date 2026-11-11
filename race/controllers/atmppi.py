"""Auto-Tuning MPPI: sampling MPC seeded from a waypoint-indexed memory that it improves while driving.

Paper sections, and where they are here:
  Spatial memory (Sec. 3.1)              ``_init_memory``: per waypoint a target state and an action sequence
  Experience-guided initialization (3.2) ``_seed_from_states``, ``_fill_passed_waypoints``
  Driving objective (3.3)                ``_drive_rewards`` and ``drive_costs_jax.py``
  Lagged window updates (3.4)            ``_window_job``, ``_window_update``
  Local rollback, safeguards (3.5)       ``_revert_window_updates``, ``_update_best_lap_table``, ``_update_permanent_freeze``
  One control step (Alg. 1)              ``step``

Commands are sampled in normalized units: ``±1`` is ``±ACTION_DENORM`` on the car.
"""

from __future__ import annotations

import collections

import numpy as np
import torch

from race.controllers.drive_costs_jax import drive_costs, squared_distances
from race.controllers.walls import WallGeometry
from race.state import (
    ANGULAR_VEL_Z_IDX, LINEAR_VEL_X_IDX, LINEAR_VEL_Y_IDX, POSE_THETA_COS_IDX, POSE_THETA_IDX, POSE_THETA_SIN_IDX,
    POSE_X_IDX, POSE_Y_IDX, SLIP_ANGLE_IDX, STATE_DIM, STEERING_ANGLE_IDX,
)

# Memory table columns. 0..9 are those of the track's waypoint table.
WP_X_IDX = 1
WP_Y_IDX = 2
WP_HEADING_IDX = 3
WP_TARGET_SPEED_IDX = 5
WP_TARGET_DRIFT_IDX = 10  # m/s, speed across the car
WP_TARGET_STEER_IDX = 11  # rad
WP_TARGET_YAW_RATE_IDX = 12  # rad/s
WP_TARGET_SLIP_IDX = 13  # rad

# ----------------------------------------------------------------------------------------------------------
# Fixed parameters. The tuned parameters of the paper are in settings/settings.yml (atmppi section).
# ----------------------------------------------------------------------------------------------------------
ACTION_DENORM = (0.375, 5.0)  # normalized ±1 -> [steering rad, acceleration m/s²]; 0.375 * 1.6 = full lock
MAX_TRANSLATIONAL_CONTROL = 12.0  # [m/s²] clip of the acceleration command
INITIAL_SPEED_LIMIT = 1.5  # [m/s] target speed of every waypoint before the first lap
# Not in the paper's table, tuned with it:
TARGET_SPEED_FACTOR = 1.03  # the driving cost's target speeds are the stored ones times this
WALL_CLEARANCE = 0.25  # [m] driving rollout states closer to a wall than this pay a soft wall cost ...
WALL_WEIGHT = 0.2  # ... of up to this per state, rising linearly to the collision margin d_min
# Matching positions to waypoints
TRACK_WINDOW = 8  # waypoints on either side of the one the free-space path leads to that a position may match
IGNORE_STEPS = 1  # the first waypoint ahead of the car is skipped
DETOUR_START = 0.5  # [m] a free-space path this much longer than the straight line means a wall is in the way
DETOUR_SPAN = 0.5  # [m] over this much more, recovery turns fully toward the free-space path
# Recovery: the car is off the corridor [d_off, ψ_off], stalled or blocked
RECOVERY = {
    "window": 40, "distance": 4.0, "heading": 3.0, "steer_in": 1.5, "steer_in_max": 0.9, "speed_min": 1.5,
    "gate": 4.0, "noise_scale": 3.0, "speed_min_w": 1.5, "speed_max": 3.0, "speed_max_w": 2.0, "reverse": 1.0,
    "slip": 1.0, "yaw_rate": 0.3,
}
OFF_TRACK_GRACE = 3.0  # [s] after a (re)start, leaving the corridor does not roll back the memory
STALL_DISTANCE = 0.5  # [m] moved less than this in STALL_TIME: stalled
STALL_TIME = 2.0  # [s]
STALL_REVERSE_STEPS = 25  # max steps of backing up to the waypoint STALL_BACKTRACK_WAYPOINTS behind
STALL_BACKTRACK_WAYPOINTS = 3
# Fixed driving samples that replace the last random ones
FALLBACK_STEER_SCALES = (0.5, 0.75)  # the seed with its steering scaled down
CREEP_SPEED = 1.0  # [m/s] below this also pull-away and back-off samples
CREEP_ACCEL = 2.0  # [m/s²]
BACK_OFF_SPEED = 0.5  # [m/s]
BLOCKED_DISTANCE = 0.3  # [m] a slow car whose best rollout gets less far than this backs up at once
# Window updates
WINDOW_ERROR = {"lateral": 0.5, "heading": 0.5, "speed": 0.2, "drift": 0.2, "yaw_rate": 0.05}  # error weights
WINDOW_LINE_REACH = 0.5  # [m] a rollout crosses a waypoint's line only this close to the stored point
WINDOW_SHIFT_STEPS = (-2, -1, 1, 2)  # candidates with a channel of the stored sequence shifted by these steps
WINDOW_MAX_SLIP = 5.0  # [rad]
WINDOW_HISTORY = 40  # overwritten values kept per waypoint for the rollback
FILL_MIN_SPEED = 0.5  # [m/s] an empty waypoint is filled only when the car is at least this fast
DRIVEN_MEMORY = 400  # steps of issued commands kept for the "actually driven" window candidate


class AutoTuningMPPI:
    def __init__(self, cfg, sim_cfg, car, track):
        """``cfg``: the ``atmppi`` settings. ``car``: ``CarParameters`` of the run. ``track``: ``Track``."""
        self.p = cfg
        self.dt = float(sim_cfg.timestep_control)
        self.dt_sim = 0.01
        self.horizon = int(cfg.horizon)
        self.delay_steps = int(sim_cfg.delay_steps)
        self.car = car
        self.car_params = car.to_array()
        self.action_denorm = np.asarray(ACTION_DENORM, dtype=np.float32)
        self.walls = WallGeometry(track, cfg.collision_margin)

        self.angular_control = 0.0
        self.translational_control = 0.0
        self.total_timesteps = 0
        self.episode_timesteps = 0
        self._issued: collections.deque[np.ndarray] = collections.deque(maxlen=self.delay_steps)
        self.recover_mode = False
        self.off_track = False
        self.frozen = False  # True once the memory is frozen for good (Sec. 3.5)
        self._translational_range = None
        self._init_memory(track.waypoints)
        self.reset()

    # ------------------------------------------------------------------
    # Memory (Sec. 3.1)
    # ------------------------------------------------------------------
    def _init_memory(self, source: np.ndarray) -> None:
        owned = np.zeros((source.shape[0], WP_TARGET_SLIP_IDX + 1), dtype=np.float32)
        owned[:, :10] = source[:, :10]
        self.waypoints = torch.as_tensor(owned)  # (N, 14) reference states
        self._waypoint_xy = self.waypoints[:, 1:3]
        self._waypoints_n = int(self.waypoints.shape[0])
        self.waypoints[:, WP_TARGET_SPEED_IDX] = torch.minimum(
            self.waypoints[:, WP_TARGET_SPEED_IDX], torch.tensor(INITIAL_SPEED_LIMIT))
        # Direction of travel to the next point, including the segment that closes the loop.
        nxt = torch.roll(self._waypoint_xy, shifts=-1, dims=0)
        delta = nxt - self._waypoint_xy
        self.waypoints[:, WP_HEADING_IDX] = torch.atan2(delta[:, 1], delta[:, 0])
        # Fixed at load: a target position moves only along this cross-track line (Eq. 3).
        heading0 = self.waypoints[:, WP_HEADING_IDX]
        self._wp_anchor_xy = self._waypoint_xy.detach().clone()
        self._wp_tangent = torch.stack((torch.cos(heading0), torch.sin(heading0)), dim=1)
        self._wp_ortho = torch.stack((-torch.sin(heading0), torch.cos(heading0)), dim=1)
        self.walls.build_track_field(self._wp_anchor_xy, clearance=0.5 * self.walls.car_width)

        n = self._waypoints_n
        self._wp_actions = torch.zeros((n, self.horizon, 2), dtype=torch.float32)  # A_i
        self._wp_action_known = torch.zeros((n,), dtype=torch.bool)
        # Commands still in the actuator delay at each stored state.
        self._wp_pending = torch.zeros((n, self.delay_steps, 2), dtype=torch.float32)
        # History H_i: (control step, table row, actions, known, delay queue) of what each update overwrote.
        self._wp_window_hist = [[] for _ in range(n)]
        self._wp_vehicle_state_set = torch.zeros((n,), dtype=torch.bool)
        self._wp_off_track_count = torch.zeros((n,), dtype=torch.float32)  # display only
        self._wp_off_track_marked = torch.zeros((n,), dtype=torch.bool)
        self._wp_window_accepted = 0
        self._wp_window_tried = 0
        self._car_waypoint = None

    def reset(self) -> None:
        """New episode (start, or restart after a crash)."""
        self.episode_timesteps = 0
        self.angular_control = 0.0
        self.translational_control = 0.0
        self.action_buffer = torch.zeros(self.horizon, 2)
        self._mean_state_seed = None
        self._fill_previous_state = None
        self._driven = None
        self._issued.clear()
        self._reset_stall()
        self.mean_states = None  # executed plan, for display

    # ------------------------------------------------------------------
    # Matching positions to the memory
    # ------------------------------------------------------------------
    def _matchable_waypoints(self, pos: torch.Tensor) -> torch.Tensor:
        """``(B, N)`` bool: the waypoints positions ``(B, 2)`` may be matched to. The nearest waypoint can be
        behind a wall; only those around the one the free-space path leads to are allowed."""
        _, label, _ = self.walls.track_field_at(pos)
        n = self._waypoints_n
        gap = (torch.arange(n)[None, :] - label[:, None]).remainder(n)
        allowed = torch.minimum(gap, n - gap) <= TRACK_WINDOW
        return allowed | (label < 0)[:, None]

    def _raceline_segment(self, state: np.ndarray) -> tuple[int, float]:
        """``(start index, progress)`` of the reference segment the car is on."""
        pos = torch.as_tensor(state, dtype=torch.float32).reshape(-1)[[POSE_X_IDX, POSE_Y_IDX]]
        xy = self._waypoint_xy
        segment = torch.roll(xy, shifts=-1, dims=0) - xy
        length2 = segment.pow(2).sum(dim=-1).clamp_min(1e-12)
        progress = ((pos - xy) * segment).sum(dim=-1) / length2
        progress = progress.clamp(0.0, 1.0)
        projection = xy + progress[:, None] * segment
        distance2 = (pos - projection).pow(2).sum(dim=-1)
        distance2 = distance2.masked_fill(~self._matchable_waypoints(pos[None])[0], float("inf"))
        segment_index = int(distance2.argmin())
        return segment_index, float(progress[segment_index])

    def nearest_waypoint_index(self, state) -> int:
        """First waypoint ahead of the car along the reference (j_t), plus ``IGNORE_STEPS``."""
        state = torch.as_tensor(state, dtype=torch.float32)
        if state.ndim == 1:
            state = state.unsqueeze(0)
        pos = state[:, [POSE_X_IDX, POSE_Y_IDX]]
        xy = self._waypoint_xy
        p0 = xy
        p1 = torch.roll(xy, shifts=-1, dims=0)
        segment = p1 - p0
        length2 = segment.pow(2).sum(dim=-1).clamp_min(1e-12)
        offset = pos.unsqueeze(1) - p0.unsqueeze(0)
        progress = (offset * segment.unsqueeze(0)).sum(dim=-1) / length2
        progress = progress.clamp(0.0, 1.0)
        projection = p0.unsqueeze(0) + progress.unsqueeze(-1) * segment.unsqueeze(0)
        distance2 = (pos.unsqueeze(1) - projection).pow(2).sum(dim=-1)
        distance2 = distance2.masked_fill(~self._matchable_waypoints(pos), float("inf"))
        segment_index = distance2.argmin(dim=-1)
        along = progress.gather(1, segment_index.unsqueeze(1)).squeeze(1)
        # A projection clamped to the segment end means the car is past it: the next waypoint is ahead.
        ahead = torch.where(along <= 0, segment_index, torch.where(along >= 1, segment_index + 2, segment_index + 1))
        ahead = ahead + IGNORE_STEPS
        return int(ahead.remainder(xy.shape[0]).reshape(-1)[0])

    def _matched_waypoint_count(self, closest: int, steps: int) -> int:
        """M_t: how many waypoints the stored target speeds cover in the share η of the horizon, plus one."""
        duration = max(steps - 1, 0) * self.dt
        ahead = (int(closest) + torch.arange(self._waypoints_n)) % self._waypoints_n
        xy = self._waypoint_xy[ahead]
        length = (xy[1:] - xy[:-1]).norm(dim=-1)
        speed = self.waypoints[ahead[:-1], WP_TARGET_SPEED_IDX].clamp(min=0.5)
        covered = int((torch.cumsum(length / speed, dim=0) <= self.p.matched_horizon_share * duration).sum())
        return min(max(covered + 1, 1), self._waypoints_n)

    # ------------------------------------------------------------------
    # Experience-guided initialization (Sec. 3.2)
    # ------------------------------------------------------------------
    def _interpolated_waypoint_actions(self, state: np.ndarray) -> torch.Tensor | None:
        """Blend of the stored sequences of the two waypoints around the car. ``None`` while neither has one."""
        if not bool(self._wp_action_known.any()):
            return None
        start, alpha = self._raceline_segment(state)
        end = (start + 1) % self._waypoints_n
        has_start = bool(self._wp_action_known[start])
        has_end = bool(self._wp_action_known[end])
        if not (has_start or has_end):
            return None
        if has_start != has_end:
            return self._wp_actions[start if has_start else end]
        return (1.0 - alpha) * self._wp_actions[start] + alpha * self._wp_actions[end]

    def _seed_from_states(self, path: torch.Tensor) -> torch.Tensor:
        """Nominal sequence ``(H, 2)`` for the states ``path`` ``(H, D)`` (Eq. 5): input k is the action field
        (Eq. 4) at state k, corrected toward the stored speed with gain g_v. Where neither waypoint around
        a state has a sequence, input k of the shifted plan is used."""
        horizon = int(path.shape[0])
        pos = path[:, [POSE_X_IDX, POSE_Y_IDX]]
        xy = self._waypoint_xy
        segment = torch.roll(xy, shifts=-1, dims=0) - xy
        length2 = segment.pow(2).sum(dim=-1).clamp_min(1e-12)
        progress = (((pos[:, None, :] - xy[None]) * segment[None]).sum(dim=-1) / length2[None]).clamp(0.0, 1.0)
        projection = xy[None] + progress[..., None] * segment[None]
        start = (pos[:, None, :] - projection).pow(2).sum(dim=-1).argmin(dim=1)  # (H,)
        alpha = progress.gather(1, start[:, None])[:, 0]
        end = (start + 1) % self._waypoints_n
        has_start, has_end = self._wp_action_known[start], self._wp_action_known[end]
        first_start, first_end = self._wp_actions[start, 0], self._wp_actions[end, 0]  # (H, 2)
        mixed = (1.0 - alpha)[:, None] * first_start + alpha[:, None] * first_end
        mixed = torch.where((has_start & ~has_end)[:, None], first_start, mixed)
        mixed = torch.where((has_end & ~has_start)[:, None], first_end, mixed)
        # A stored action fits a car at the stored speed: a slower car must not also inherit its braking.
        speed = self.waypoints[:, WP_TARGET_SPEED_IDX] * float(TARGET_SPEED_FACTOR)
        target = (1.0 - alpha) * speed[start] + alpha * speed[end]
        target = torch.where(has_start & ~has_end, speed[start], target)
        target = torch.where(has_end & ~has_start, speed[end], target)
        gain = float(self.p.seed_speed_gain) / float(self.action_denorm[1])
        mixed = mixed.clone()
        mixed[:, 1] = mixed[:, 1] + gain * (target - path[:, LINEAR_VEL_X_IDX])
        return torch.where((has_start | has_end)[:, None], mixed, self.action_buffer[:horizon])

    def _waypoint_known(self, idx: torch.Tensor) -> torch.Tensor:
        return self._wp_action_known[idx] & self._wp_vehicle_state_set[idx]

    def _fill_passed_waypoints(self, state: np.ndarray, action_seq: torch.Tensor) -> None:
        """An empty waypoint whose line the car crossed since the last step stores the car state at the
        crossing, the executed plan and the commands still in the delay ("filling the memory")."""
        current = torch.as_tensor(state, dtype=torch.float32).reshape(1, -1)
        previous = self._fill_previous_state
        self._fill_previous_state = current
        if previous is None or self._car_waypoint is None or self._stall_phase == 1:
            return
        if self.frozen or self.recover_mode:
            return
        if float(current[0, LINEAR_VEL_X_IDX]) < FILL_MIN_SPEED:
            return
        idx = (int(self._car_waypoint) + torch.arange(-3, 4)) % self._waypoints_n
        path = torch.cat((previous, current))[None]
        idx = idx[~self._waypoint_known(idx)]
        if not idx.numel():
            return
        times, at = self._line_crossings(path, idx)
        pending = self._issued_actions()
        for j in torch.nonzero(~torch.isnan(times[0])).flatten().tolist():
            k = int(idx[j])
            self._write_waypoint_state(k, at[0, j])
            self._wp_actions[k] = self._fit_horizon(action_seq.reshape(-1, 2), int(self._wp_actions.shape[1]))
            self._wp_pending[k] = pending
            self._wp_action_known[k] = True

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------
    @staticmethod
    def _fit_horizon(seq: torch.Tensor, horizon: int) -> torch.Tensor:
        """Crop, or repeat the last action, so ``seq`` ``(H, 2)`` has ``horizon`` steps."""
        seq = seq.reshape(-1, 2)
        n = int(seq.shape[0])
        if n == horizon:
            return seq
        if n > horizon:
            return seq[:horizon].contiguous()
        return torch.cat((seq, seq[-1:].expand(horizon - n, -1)), dim=0)

    def _sample_actions(self, nominal: torch.Tensor, count: int, noise_scale: float = 1.0) -> torch.Tensor:
        """``count`` temporally correlated Gaussian samples around ``nominal`` (correlation ρ), clipped."""
        rho = min(max(float(self.p.noise_correlation), 0.0), 0.999)
        noise_std = noise_scale * torch.tensor(self.p.noise_std, dtype=torch.float32) / torch.as_tensor(self.action_denorm)
        if self.off_track or self._stall_phase == 1:
            noise_std = noise_std * float(RECOVERY["noise_scale"])
        eps = torch.randn(int(count), int(nominal.shape[0]), 2)
        actions = nominal + eps * noise_std
        if rho > 0.0:
            fresh = (1.0 - rho * rho) ** 0.5
            drift = ((1.0 - rho) * nominal).numpy()
            kick = (fresh * noise_std * eps).numpy().transpose(1, 0, 2).copy()
            out = actions.numpy()
            steps = out.transpose(1, 0, 2).copy()
            for t in range(1, int(eps.shape[1])):
                step = steps[t]
                np.multiply(steps[t - 1], np.float32(rho), out=step)
                np.add(step, drift[t], out=step)
                np.add(step, kick[t], out=step)
            out[:] = steps.transpose(1, 0, 2)
        brake_limit, accel_limit = self._translational_limits()
        actions[..., 0].clamp_(-1.0, 1.0)
        actions[..., 1].clamp_(brake_limit, accel_limit)
        return actions

    def _translational_limits(self) -> tuple[float, float]:
        """Normalized range of the acceleration command that the car does not saturate."""
        if self._translational_range is None:
            scale = float(self.action_denorm[1])
            grip = float(self.car.mu) * float(self.car.g)
            brake = max(-MAX_TRANSLATIONAL_CONTROL, max(float(self.car.a_min), -grip) / float(self.car.brake_multiplier))
            accel = min(MAX_TRANSLATIONAL_CONTROL, min(float(self.car.a_max), grip) / float(self.car.accel_multiplier))
            self._translational_range = (brake / scale, accel / scale)
        return self._translational_range

    def _with_fallback_samples(self, actions: torch.Tensor, nominal: torch.Tensor | None, speed: float) -> torch.Tensor:
        """Replace the last samples with fixed ones the noise rarely reaches: the previous plan, the seed with
        less steering and, for a slow car, pulling away and backing off at fixed steering."""
        if nominal is None:
            nominal = self.action_buffer
        horizon = int(nominal.shape[0])
        self.action_buffer = self._fit_horizon(self.action_buffer, horizon)
        fixed = [self.action_buffer]
        for scale in FALLBACK_STEER_SCALES:
            fixed.append(torch.stack((nominal[:, 0] * scale, nominal[:, 1]), dim=1))
        if speed < CREEP_SPEED:
            steps = int(np.ceil((CREEP_SPEED - speed) / (CREEP_ACCEL * self.dt)))
            throttle = torch.zeros(horizon)
            throttle[: min(steps, horizon)] = CREEP_ACCEL / float(self.action_denorm[1])
            for steer in (-1.0, -0.5, 0.5, 1.0):
                fixed.append(torch.stack((torch.full((horizon,), steer), throttle), dim=1))
            braking = CREEP_ACCEL * float(self.car.brake_multiplier)
            steps = int(np.ceil((max(speed, 0.0) + BACK_OFF_SPEED) / (braking * self.dt)))
            reverse = torch.zeros(horizon)
            reverse[: min(steps, horizon)] = -CREEP_ACCEL / float(self.action_denorm[1])
            for steer in (-1.0, 1.0):
                fixed.append(torch.stack((torch.full((horizon,), steer), reverse), dim=1))
        fixed = torch.stack(fixed, dim=0)
        if int(fixed.shape[0]) >= int(actions.shape[0]):
            return actions
        actions = actions.clone()
        actions[-int(fixed.shape[0]):] = fixed
        return actions

    # ------------------------------------------------------------------
    # Driving objective (Sec. 3.3)
    # ------------------------------------------------------------------
    def _drive_rewards(self, states: torch.Tensor, start: int) -> tuple[torch.Tensor, torch.Tensor]:
        """``(-J_drive without safety, -J_safe)`` of rollouts ``states`` ``(B, T, D)``. ``start``: j_t."""
        walls = self.walls
        batch, steps = int(states.shape[0]), int(states.shape[1])
        n = self._waypoints_n
        count = self._matched_waypoint_count(start, steps)
        width = -(-count // 16) * 16  # padded, so only a few widths are ever compiled
        idx = (start + torch.arange(width)) % n
        match_xy = self._waypoint_xy[idx].numpy()
        rows = self.waypoints[idx][:, [WP_X_IDX, WP_Y_IDX, WP_HEADING_IDX, WP_TARGET_SPEED_IDX, WP_TARGET_DRIFT_IDX]].numpy()
        if getattr(self, "_jax_walls", None) is None:
            import jax
            self._jax_walls = (jax.device_put(walls.wall_distance.numpy()),
                               walls._corner_x.numpy().astype(np.float64), walls._corner_y.numpy().astype(np.float64))
        grid, corner_x, corner_y = self._jax_walls
        p = self.p
        border = float(p.collision_margin)
        consts = (
            float(TARGET_SPEED_FACTOR), float(p.cost_overspeed), float(p.cost_arrival), border, float(WALL_CLEARANCE),
            max(float(WALL_CLEARANCE) - border, 1e-3),
            walls.origin_x, walls.origin_y, walls.origin_cos, walls.origin_sin, walls.resolution,
            float(walls.map_width), float(walls.map_height),
            float(p.cost_position), float(p.cost_heading), float(p.cost_speed_shortfall), float(p.cost_drift),
            float(p.cost_reverse),
        )
        s = np.ascontiguousarray(states.numpy(), dtype=np.float32)
        dist = squared_distances(s, match_xy)
        from jax.experimental import enable_x64
        with enable_x64():
            out = drive_costs(s, dist, rows, np.int32(count), grid, corner_x, corner_y, consts)
            drive_m, active, late, closeness, _ = (torch.from_numpy(np.array(o)) for o in out)

        if not self.off_track:
            drive = torch.where(active, drive_m, torch.zeros(batch))
        else:
            drive = self._recovery_cost(states, start)
        if self._stall_phase == 1:
            drive = self._backtrack_cost(states)
        # A crash costs at least as much per lost horizon step as the largest cost in the batch.
        crash_steps = late.to(drive.dtype)
        drive_crash = -(drive.abs().max().clamp_min(1.0) * 10 * crash_steps)
        drive_reward = -drive - float(WALL_WEIGHT) * closeness
        return drive_reward, drive_crash

    def _recovery_cost(self, states: torch.Tensor, start: int) -> torch.Tensor:
        """``(B,)`` cost of rollouts of a car outside the corridor: head back onto the reference and follow it."""
        cfg = RECOVERY
        n = self._waypoints_n
        window = (torch.arange(-3, int(cfg["window"])) + int(start)) % n
        wp = self.waypoints[window]
        pos = states[..., [POSE_X_IDX, POSE_Y_IDX]]
        distance = torch.linalg.norm(pos[:, :, None, :] - wp[None, None, :, [WP_X_IDX, WP_Y_IDX]], dim=-1)
        near_distance, near = distance.min(dim=-1)
        line = wp[near]
        track_heading = line[..., WP_HEADING_IDX]
        left = torch.stack((-torch.sin(track_heading), torch.cos(track_heading)), dim=-1)
        cross_track = ((pos - line[..., [WP_X_IDX, WP_Y_IDX]]) * left).sum(dim=-1)
        wanted = track_heading - (float(cfg["steer_in"]) * cross_track).clamp(-float(cfg["steer_in_max"]), float(cfg["steer_in_max"]))
        # Behind a wall the way back follows the free-space path.
        path, _, direction = self.walls.track_field_at(pos)
        detour = ((self._detour(pos).to(near_distance) - DETOUR_START) / DETOUR_SPAN).clamp(0.0, 1.0)
        turn = torch.atan2(direction[..., 1], direction[..., 0]).to(wanted) - wanted
        wanted = wanted + detour * torch.atan2(torch.sin(turn), torch.cos(turn))
        near_distance = torch.where(detour > 0, torch.maximum(near_distance, path.to(near_distance)), near_distance)
        heading_error = states[..., POSE_THETA_IDX] - wanted
        heading_error = torch.atan2(torch.sin(heading_error), torch.cos(heading_error)).abs()
        speed = states[..., LINEAR_VEL_X_IDX]
        pointing_along = (heading_error < float(cfg["gate"])).float()
        slow = (float(cfg["speed_min"]) - speed).clamp(min=0) * pointing_along
        fast = (speed - float(cfg["speed_max"])).clamp(min=0)
        reverse = (-speed).clamp(min=0)
        slip = (states[..., LINEAR_VEL_Y_IDX].abs() - 1.0).clamp(min=0)
        yaw = (states[..., ANGULAR_VEL_Z_IDX].abs() - 3.0).clamp(min=0)

        def over_time(x: torch.Tensor) -> torch.Tensor:  # the end of the rollout counts twice
            return x.mean(dim=1) + x[:, -1]

        return (
            float(cfg["distance"]) * over_time(near_distance)
            + float(cfg["heading"]) * over_time(heading_error)
            + float(cfg["speed_min_w"]) * slow.mean(dim=1)
            + float(cfg["speed_max_w"]) * fast.mean(dim=1)
            + float(cfg["reverse"]) * reverse.mean(dim=1)
            + float(cfg["slip"]) * slip.mean(dim=1)
            + float(cfg["yaw_rate"]) * yaw.mean(dim=1)
        )

    def _backtrack_cost(self, states: torch.Tensor) -> torch.Tensor:
        """``(B,)`` cost of a stalled car: distance to the waypoint behind it, forward or reverse."""
        cfg = RECOVERY
        target = self._waypoint_xy[self._backtrack_wp]
        pos = states[..., [POSE_X_IDX, POSE_Y_IDX]]
        distance = torch.linalg.norm(pos - target, dim=-1)
        slip = (states[..., LINEAR_VEL_Y_IDX].abs() - 1.0).clamp(min=0)
        yaw = (states[..., ANGULAR_VEL_Z_IDX].abs() - 3.0).clamp(min=0)
        fast = (states[..., LINEAR_VEL_X_IDX].abs() - float(cfg["speed_max"])).clamp(min=0)
        return (
            float(cfg["distance"]) * (distance.mean(dim=1) + distance[:, -1])
            + float(cfg["slip"]) * slip.mean(dim=1)
            + float(cfg["yaw_rate"]) * yaw.mean(dim=1)
            + float(cfg["speed_max_w"]) * fast.mean(dim=1)
        )

    def _detour(self, pos: torch.Tensor) -> torch.Tensor:
        """How much longer [m] the free-space path from ``pos`` ``(..., 2)`` to the reference is than the straight line."""
        path, label, _ = self.walls.track_field_at(pos)
        straight = (pos - self._wp_anchor_xy[label.clamp(min=0)]).norm(dim=-1)
        return torch.where(label >= 0, path.to(straight) - straight, torch.zeros_like(straight))

    def _raceline_distance(self, state: np.ndarray) -> float:
        """Distance of the car to the reference; the way around a wall when that is longer."""
        start, progress = self._raceline_segment(state)
        end = (start + 1) % self._waypoints_n
        pos = torch.as_tensor(state, dtype=torch.float32).reshape(-1)[[POSE_X_IDX, POSE_Y_IDX]]
        projection = self._waypoint_xy[start] + progress * (self._waypoint_xy[end] - self._waypoint_xy[start])
        distance = float((pos - projection).norm())
        if float(self._detour(pos)) > DETOUR_START:
            path, _, _ = self.walls.track_field_at(pos)
            return max(distance, float(path))
        return distance

    # ------------------------------------------------------------------
    # Corridor, stall and recovery (Sec. 3.5)
    # ------------------------------------------------------------------
    def _refresh_recovery(self, state: np.ndarray) -> None:
        """Gating events from the current pose, before this step plans."""
        car_xy = torch.as_tensor(state, dtype=torch.float32).reshape(-1)[[POSE_X_IDX, POSE_Y_IDX]]
        line_distance = self._raceline_distance(state)
        distance = (self._waypoint_xy - car_xy).norm(dim=-1)
        distance = distance.masked_fill(~self._matchable_waypoints(car_xy[None])[0], float("inf"))
        self._car_waypoint = int(distance.argmin())  # i_t
        self._capture_driven_table()
        on_track = line_distance <= float(self.p.corridor_width)
        limit = float(self.p.heading_limit)
        if on_track and limit > 0.0:
            heading = float(np.asarray(state).reshape(-1)[POSE_THETA_IDX])
            error = heading - float(self.waypoints[self._car_waypoint, WP_HEADING_IDX])
            on_track = abs(np.arctan2(np.sin(error), np.cos(error))) <= limit
        if on_track:
            self._seen_on_track = True
        self._update_stall(car_xy, self._car_waypoint)
        self.off_track = (not on_track) or self._stalled
        affects_waypoints = self.off_track and self.episode_timesteps * self.dt >= OFF_TRACK_GRACE
        if affects_waypoints and not self._was_off_track:
            self._revert_nearby_waypoints()
        self._was_off_track = affects_waypoints
        self.recover_mode = bool(self.off_track or self._stall_phase == 1)
        self._note_off_track_pass(affects_waypoints)

    def _note_off_track_pass(self, off: bool) -> None:
        """Count, once per visit, the waypoint where the car left the corridor (shown in the display)."""
        if self._car_waypoint is None or not self._seen_on_track:
            return
        n = self._waypoints_n
        car = int(self._car_waypoint)
        gap = (torch.arange(n) - car).remainder(n)
        gap = torch.minimum(gap, n - gap)
        self._wp_off_track_marked &= gap <= 10
        if off and not bool(self._wp_off_track_marked[car]):
            self._wp_off_track_marked[car] = True
            self._wp_off_track_count[car] += 1.0

    def _reset_stall(self) -> None:
        self._seen_on_track = False  # a car that starts off the reference does not roll back anything until it got there
        self._was_off_track = True
        self._stall_phase = 0  # 0 driving, 1 backing up, 2 stalled recovery
        self._reverse_left = 0
        self._stall_anchor = None
        self._backtrack_wp = None
        self._stalled = False
        self._pose_history = collections.deque()

    def _update_stall(self, car_xy: torch.Tensor, car_waypoint: int) -> None:
        """Not moving STALL_DISTANCE in STALL_TIME: back up to the waypoint behind (phase 1), then recover
        (phase 2) until STALL_DISTANCE away from there."""
        if self._stall_phase == 1:
            target = self._waypoint_xy[self._backtrack_wp]
            arrived = float((car_xy - target).norm()) < 0.5
            if self._reverse_left <= 0 or arrived:
                self._stall_phase = 2
                self._stall_anchor = car_xy.clone()
        elif self._stall_phase == 2:
            if float((car_xy - self._stall_anchor).norm()) >= STALL_DISTANCE:
                self._stall_phase = 0
                self._pose_history.clear()
        else:
            steps = max(1, int(round(STALL_TIME / self.dt)))
            history = self._pose_history
            history.append(car_xy.clone())
            while len(history) > steps + 1:
                history.popleft()
            if len(history) > steps and float((history[-1] - history[0]).norm()) < STALL_DISTANCE:
                self._start_backtrack(car_xy, car_waypoint)
        self._stalled = self._stall_phase == 2

    def _start_backtrack(self, car_xy: torch.Tensor, car_waypoint: int) -> None:
        self._pose_history.clear()
        self._stall_phase = 1
        self._reverse_left = STALL_REVERSE_STEPS
        self._backtrack_wp = (int(car_waypoint) - STALL_BACKTRACK_WAYPOINTS) % self._waypoints_n
        self._stalled = False

    def _blocked(self, speed: float, trajectories: torch.Tensor, crash: torch.Tensor, best: int) -> bool:
        """A slow car whose best rollout crashes or does not get going."""
        if speed >= CREEP_SPEED:
            return False
        if float(crash[best]) < 0:
            return True
        forward = float(trajectories[best, 1:, LINEAR_VEL_X_IDX].clamp(min=0).sum()) * self.dt
        return forward < BLOCKED_DISTANCE

    def _safe_mean(self, state, actions, mean, weights, total, crash):
        """The best sample instead of the weighted mean when the mean crashes and the best sample does not."""
        best = int(total.argmax())
        if float(crash[best]) < 0 or not bool((crash < 0).any()):
            return mean, weights
        path = self.rollout(state, mean)
        if not bool(self.walls.in_wall(path[..., [POSE_X_IDX, POSE_Y_IDX, POSE_THETA_IDX]]).any()):
            return mean, weights
        return actions[best].clone(), torch.nn.functional.one_hot(torch.tensor(best), int(weights.shape[0])).to(weights)

    # ------------------------------------------------------------------
    # Rollback and lap-level safeguards (Sec. 3.5)
    # ------------------------------------------------------------------
    def _revert_nearby_waypoints(self) -> None:
        if self._car_waypoint is None or self.frozen or not self._seen_on_track:
            return
        self._revert_window_updates()

    def _revert_window_updates(self) -> None:
        """Waypoints i_t - B .. i_t + F go back to their values from one lap ago, all to the same time."""
        lap_steps = getattr(self, "_last_lap_steps", None)
        if lap_steps is None:
            return
        since = int(self.total_timesteps) - int(lap_steps)
        n = self._waypoints_n
        center = int(self._car_waypoint)
        for offset in range(-int(self.p.rollback_behind), int(self.p.rollback_ahead) + 1):
            idx = (center + offset) % n
            history = self._wp_window_hist[idx]
            entry = None
            while history and history[-1][0] > since:
                entry = history.pop()
            if entry is None:
                continue
            _, row, actions, known, pending = entry
            self.waypoints[idx] = row
            self._wp_actions[idx] = actions
            self._wp_action_known[idx] = known
            self._wp_pending[idx] = pending

    def _snapshot_refs(self) -> dict:
        return {
            "waypoints": self.waypoints, "actions": self._wp_actions, "known": self._wp_action_known,
            "vehicle_state_set": self._wp_vehicle_state_set, "pending": self._wp_pending,
        }

    def _snapshot(self) -> dict:
        return {key: values.clone() for key, values in self._snapshot_refs().items()}

    def _capture_driven_table(self) -> None:
        """Keep every entry as it was when the car last reached it: at the end of a lap, the table it drove on."""
        car = self._car_waypoint
        if car is None:
            return
        n = self._waypoints_n
        table = getattr(self, "_driven_table", None)
        if table is None:
            self._driven_table, self._driven_last = self._snapshot(), car
            return
        ahead = (car - int(self._driven_last)) % n
        if ahead == 0:
            return
        # A jump of more than a quarter lap (restart, backing up) takes only the waypoint the car is at.
        idx = (int(self._driven_last) + 1 + torch.arange(ahead)) % n if ahead <= n // 4 else torch.tensor([car])
        self._driven_last = car
        for key, values in self._snapshot_refs().items():
            table[key][idx] = values[idx]

    def _best_lap_snapshot(self, start: dict) -> dict:
        driven = getattr(self, "_driven_table", None)
        if driven is None:
            return start
        return {key: values.clone() for key, values in driven.items()}

    def _restore_snapshot(self, snapshot: dict) -> None:
        self.waypoints.copy_(snapshot["waypoints"])
        self._wp_actions.copy_(snapshot["actions"])
        self._wp_action_known.copy_(snapshot["known"])
        self._wp_vehicle_state_set.copy_(snapshot["vehicle_state_set"])
        self._wp_pending.copy_(snapshot["pending"])
        for history in self._wp_window_hist:
            history.clear()

    def _update_permanent_freeze(self, lap_time: float) -> None:
        """P laps without a faster one after the reset to the best table: restore that table and freeze it."""
        patience = int(self.p.laps_without_improvement)
        if self.frozen or patience <= 0:
            return
        start = getattr(self, "_freeze_lap_start", None)
        self._freeze_lap_start = self._snapshot()  # the table the next lap starts on
        best = getattr(self, "_freeze_best_lap", None)
        if start is not None and (best is None or float(lap_time) < best):
            self._freeze_best_lap, self._freeze_best_table, self._freeze_laps_since_best = float(lap_time), self._best_lap_snapshot(start), 0
            return
        if best is None:
            return
        self._freeze_laps_since_best = int(getattr(self, "_freeze_laps_since_best", 0)) + 1
        if self._freeze_laps_since_best < patience:
            return
        self._restore_snapshot(self._freeze_best_table)
        self.frozen = True
        print(f"[atmppi] No lap faster than {best:.3f} s for {patience} laps: memory restored to that lap's and frozen")

    def _update_best_lap_table(self, lap_time: float) -> None:
        """P laps without a faster one: the memory goes back to the table of the fastest lap; updates go on."""
        patience = int(self.p.laps_without_improvement)
        if self.frozen:
            return
        start = getattr(self, "_lap_start_table", None)
        self._lap_start_table = self._snapshot()
        best = getattr(self, "_best_lap_time", None)
        if start is not None and (best is None or lap_time < best):
            self._best_lap_time, self._best_lap_table, self._laps_since_best = lap_time, self._best_lap_snapshot(start), 0
            return
        if best is None:
            return
        self._laps_since_best = int(getattr(self, "_laps_since_best", 0)) + 1
        if self._laps_since_best < patience:
            return
        self._restore_snapshot(self._best_lap_table)
        self._lap_start_table = self._best_lap_table
        self._laps_since_best = 0
        print(f"[atmppi] No lap faster than {best:.3f} s for {patience} laps: memory reset to that lap's")

    def report_lap_time(self, lap_time: float) -> None:
        previous = getattr(self, "_lap_end_step", None)
        self._lap_end_step = int(self.total_timesteps)
        if previous is not None:
            self._last_lap_steps = self._lap_end_step - int(previous)
        self._update_permanent_freeze(float(lap_time))
        self._update_best_lap_table(float(lap_time))
        known = self._waypoint_known(torch.arange(self._waypoints_n))
        speed = float(self.waypoints[known, WP_TARGET_SPEED_IDX].mean()) if bool(known.any()) else 0.0
        print(f"[atmppi] {self._wp_window_accepted}/{self._wp_window_tried} window updates accepted, "
              f"{int(known.sum())}/{self._waypoints_n} waypoints stored, mean stored speed {speed:.2f} m/s")
        self._wp_window_accepted = self._wp_window_tried = 0

    def on_crash(self) -> None:
        """The car hit a wall: roll back the waypoints around it, then the episode restarts."""
        self._revert_nearby_waypoints()
        self.reset()

    # ------------------------------------------------------------------
    # Lagged window updates (Sec. 3.4)
    # ------------------------------------------------------------------
    def _line_crossings(self, states: torch.Tensor, idx: torch.Tensor):
        """When (steps from the first state, NaN if not within ``WINDOW_LINE_REACH``) and in which interpolated
        state ``(B, J, D)`` each rollout ``(B, T, D)`` first crosses the cross-track line of each waypoint ``idx``."""
        anchor, tangent, ortho = self._wp_anchor_xy[idx], self._wp_tangent[idx], self._wp_ortho[idx]
        stored = ((self._waypoint_xy[idx] - anchor) * ortho).sum(dim=-1)  # (J,)
        rel = states[:, :, None, :][..., [POSE_X_IDX, POSE_Y_IDX]] - anchor[None, None]  # (B, T, J, 2)
        along = (rel * tangent).sum(dim=-1)
        near = ((rel * ortho).sum(dim=-1) - stored).abs() < float(WINDOW_LINE_REACH)
        cross = (along[:, :-1] < 0) & (along[:, 1:] >= 0) & near[:, :-1] & near[:, 1:]  # (B, T - 1, J)
        crossed = cross.any(dim=1)
        k = cross.float().argmax(dim=1)  # (B, J), first crossing
        a0 = along.gather(1, k[:, None]).squeeze(1)
        a1 = along.gather(1, (k + 1)[:, None]).squeeze(1)
        frac = (-a0 / (a1 - a0).clamp_min(1e-9)).clamp(0.0, 1.0)
        times = torch.where(crossed, k.to(frac.dtype) + frac, torch.full_like(frac, float("nan")))
        rows = torch.arange(int(states.shape[0]))[:, None]
        before, after = states[rows, k], states[rows, k + 1]
        at = before + frac[..., None] * (after - before)
        at[..., POSE_THETA_IDX] = torch.atan2(at[..., POSE_THETA_SIN_IDX], at[..., POSE_THETA_COS_IDX])
        return times, at

    def _window_error(self, at: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
        """``(B, J)`` weighted deviation of the crossing states ``at`` from the states stored at ``idx``."""
        wp = self.waypoints[idx]
        weight = WINDOW_ERROR
        offset = at[..., [POSE_X_IDX, POSE_Y_IDX]] - wp[None, :, [WP_X_IDX, WP_Y_IDX]]
        lateral = (offset * self._wp_ortho[idx][None]).sum(dim=-1).abs()
        heading = at[..., POSE_THETA_IDX] - wp[None, :, WP_HEADING_IDX]
        heading = torch.atan2(torch.sin(heading), torch.cos(heading)).abs()
        speed = (at[..., LINEAR_VEL_X_IDX] - wp[None, :, WP_TARGET_SPEED_IDX]).abs()
        drift = (at[..., LINEAR_VEL_Y_IDX] - wp[None, :, WP_TARGET_DRIFT_IDX]).abs()
        yaw_rate = (at[..., ANGULAR_VEL_Z_IDX] - wp[None, :, WP_TARGET_YAW_RATE_IDX]).abs()
        return (
            weight["lateral"] * lateral + weight["heading"] * heading + weight["speed"] * speed
            + weight["drift"] * drift + weight["yaw_rate"] * yaw_rate
        )

    @staticmethod
    def _timeline_at(values: torch.Tensor, start: float, times: torch.Tensor) -> torch.Tensor:
        """``values`` ``(L, 2)`` issued at times ``start, start + 1, ...``, interpolated at ``times``, held at the ends."""
        last = int(values.shape[0]) - 1
        values, times = values.numpy(), times.numpy()
        pos = np.minimum(np.maximum(times - times.dtype.type(start), 0.0), float(last))
        lo = np.minimum(np.floor(pos).astype(np.int64), last)
        hi = np.minimum(lo + 1, last)
        frac = (pos - lo.astype(pos.dtype))[:, None]
        return torch.from_numpy(values[lo] * (1.0 - frac) + values[hi] * frac)

    def _write_waypoint_state(self, idx: int, state: torch.Tensor) -> None:
        """Waypoint ``idx`` takes the car state ``state``, its position projected onto the cross-track line."""
        anchor, ortho = self._wp_anchor_xy[idx], self._wp_ortho[idx]
        xy = anchor + ((state[[POSE_X_IDX, POSE_Y_IDX]] - anchor) * ortho).sum() * ortho
        self.waypoints[idx, WP_X_IDX] = xy[0]
        self.waypoints[idx, WP_Y_IDX] = xy[1]
        self.waypoints[idx, WP_HEADING_IDX] = state[POSE_THETA_IDX]
        self.waypoints[idx, WP_TARGET_SPEED_IDX] = state[LINEAR_VEL_X_IDX].clamp(min=0.0)
        self.waypoints[idx, WP_TARGET_DRIFT_IDX] = state[LINEAR_VEL_Y_IDX]
        self.waypoints[idx, WP_TARGET_STEER_IDX] = state[STEERING_ANGLE_IDX]
        self.waypoints[idx, WP_TARGET_YAW_RATE_IDX] = state[ANGULAR_VEL_Z_IDX]
        self.waypoints[idx, WP_TARGET_SLIP_IDX] = state[SLIP_ANGLE_IDX]
        self._wp_vehicle_state_set[idx] = True

    def _waypoint_states(self, idx: torch.Tensor) -> torch.Tensor:
        """``(K, D)`` car states stored at waypoints ``idx``."""
        wp = self.waypoints[idx]
        start = torch.zeros((int(wp.shape[0]), STATE_DIM), dtype=torch.float32)
        heading = wp[:, WP_HEADING_IDX]
        start[:, POSE_X_IDX] = wp[:, WP_X_IDX]
        start[:, POSE_Y_IDX] = wp[:, WP_Y_IDX]
        start[:, POSE_THETA_IDX] = heading
        start[:, POSE_THETA_SIN_IDX] = torch.sin(heading)
        start[:, POSE_THETA_COS_IDX] = torch.cos(heading)
        start[:, LINEAR_VEL_X_IDX] = wp[:, WP_TARGET_SPEED_IDX]
        start[:, LINEAR_VEL_Y_IDX] = wp[:, WP_TARGET_DRIFT_IDX]
        start[:, STEERING_ANGLE_IDX] = wp[:, WP_TARGET_STEER_IDX]
        start[:, ANGULAR_VEL_Z_IDX] = wp[:, WP_TARGET_YAW_RATE_IDX]
        start[:, SLIP_ANGLE_IDX] = wp[:, WP_TARGET_SLIP_IDX]
        return start

    def _record_driven(self, state: np.ndarray, action: torch.Tensor) -> None:
        """Remember the command issued now, and when the car last crossed the lines of the waypoints around it."""
        driven = self._driven
        if driven is None:
            driven = {"step": -1, "first": 0, "actions": [], "previous": None,
                      "cross": torch.full((self._waypoints_n,), float("nan"))}
            self._driven = driven
        driven["step"] += 1
        driven["actions"].append(action.reshape(2).to(torch.float32).cpu())
        if len(driven["actions"]) > DRIVEN_MEMORY:
            driven["actions"].pop(0)
            driven["first"] += 1
        current = torch.as_tensor(state, dtype=torch.float32).reshape(1, -1)
        previous, driven["previous"] = driven["previous"], current
        if previous is None or self._car_waypoint is None or self.recover_mode:
            return
        idx = (int(self._car_waypoint) + torch.arange(-3, 4)) % self._waypoints_n
        times, _ = self._line_crossings(torch.cat((previous, current))[None], idx)
        hit = ~torch.isnan(times[0])
        driven["cross"][idx[hit]] = float(driven["step"] - 1) + times[0, hit]

    def _driven_candidate(self, w: int, horizon: int) -> torch.Tensor | None:
        """The commands actually issued since the car last crossed waypoint ``w``'s line, ``(H, 2)``, or None."""
        driven = self._driven
        if driven is None:
            return None
        crossed = float(driven["cross"][w])
        if np.isnan(crossed) or crossed < driven["first"] or crossed + horizon - 1 > driven["step"]:
            return None
        timeline = torch.stack(driven["actions"])
        return self._timeline_at(timeline, float(driven["first"]), crossed + torch.arange(horizon, dtype=torch.float32))

    def _window_job(self):
        """This step's window update ``(w, state stored at w, its delay queue, candidates (K_w, H, 2))``, or None.

        w = i_t - L. Candidate 0 is A_w itself, 1 the commands actually issued since the car crossed w, then
        A_w with a channel shifted by a few steps, the rest correlated noise around A_w.
        """
        if self.recover_mode or self._car_waypoint is None or self.frozen:
            return None
        n = self._waypoints_n
        span = max(1, int(self.p.window_size))
        w = (int(self._car_waypoint) - int(self.p.window_lag)) % n
        if not bool(self._waypoint_known((w + torch.arange(span + 2)) % n).all()):
            return None
        stored = self._wp_actions[w]
        samples = self._sample_actions(stored, int(self.p.window_samples), float(self.p.window_noise_scale))
        samples[0] = stored
        if int(samples.shape[0]) > 1:
            driven = self._driven_candidate(w, int(samples.shape[1]))
            if driven is not None:
                samples[1] = driven
        shifted = self._shifted_sequences(stored)
        if int(samples.shape[0]) > 2 + int(shifted.shape[0]):
            samples[2:2 + int(shifted.shape[0])] = shifted
        return w, self._waypoint_states(torch.tensor([w]))[0], self._wp_pending[w].clone(), samples

    @staticmethod
    def _shifted_sequences(stored: torch.Tensor) -> torch.Tensor:
        """Copies of ``stored`` with the acceleration, then the steering channel played earlier or later."""
        horizon = int(stored.shape[0])
        out = []
        for channel in (1, 0):
            for shift in WINDOW_SHIFT_STEPS:
                index = (torch.arange(horizon) + int(shift)).clamp(0, horizon - 1)
                seq = stored.clone()
                seq[:, channel] = stored[index, channel]
                out.append(seq)
        return torch.stack(out)

    def _window_update(self, window, states: torch.Tensor) -> None:
        """Accept the best valid candidate and overwrite the window W_win (Sec. 3.4).

        Compared are the window and up to D downstream waypoints that A_w crosses in a row. A candidate is
        valid if it crosses all of them in order, keeps d_w from the walls (or the clearance of A_w in a
        narrow passage, at least the minimum) and stays under the slip limit. Its error is the largest one
        at the downstream waypoints. While the error of A_w is within ε̄, the accepted candidate minimizes
        arrival time + c_ε * error among the valid ones within ε̄ that arrive earlier than A_w. Otherwise it
        is a repair: the valid candidate with the smallest error below that of A_w, however slow.
        """
        w, _, pending, actions = window
        n = self._waypoints_n
        span = max(1, int(self.p.window_size))
        down = max(1, int(self.p.window_downstream))
        idx = (w + 1 + torch.arange(span + down)) % n
        times, at = self._line_crossings(states, idx)
        usable = (~torch.isnan(times[0])) & self._waypoint_known(idx)
        count = int(usable.long().cumprod(dim=0).sum())
        if count < span + 1:
            return
        idx, times, at = idx[:count], times[:, :count], at[:, :count]
        self._wp_window_tried += 1
        crossed = (~torch.isnan(times)).all(dim=1) & (times[:, 1:] > times[:, :-1]).all(dim=1)
        # Only the waypoints after the window, which the candidate must rejoin, count for the error.
        error = self._window_error(at[:, span:], idx[span:]).amax(dim=1)
        arrival = times[:, -1]
        horizon = int(states.shape[1])
        poses = states[..., [POSE_X_IDX, POSE_Y_IDX, POSE_THETA_IDX]]
        corner_distance = self.walls.distance_at(self.walls.corners(poses)).amin(dim=-1).cpu().to(times.dtype)
        until = torch.nan_to_num(arrival, nan=float(horizon)).ceil() + 1
        compared = torch.arange(horizon)[None] <= until[:, None]
        clearance = torch.where(compared, corner_distance, torch.full_like(corner_distance, float("inf"))).amin(dim=1)
        margin = min(float(self.p.window_wall_clearance), float(clearance[0]))
        margin = max(margin, float(self.p.window_wall_clearance_min))
        valid = crossed & (clearance >= margin)
        slip = torch.atan2(states[..., LINEAR_VEL_Y_IDX].abs(), states[..., LINEAR_VEL_X_IDX].clamp(min=CREEP_SPEED))
        valid = valid & (torch.where(compared, slip, torch.zeros_like(slip)).amax(dim=1) <= WINDOW_MAX_SLIP)
        stored_arrival = float(arrival[0]) if bool(valid[0]) else float("inf")
        tolerance = float(self.p.window_error_tolerance)
        repair = float(error[0]) > tolerance
        if repair:
            eligible = valid & (error < float(error[0]))
            score = error
        else:
            eligible = valid & (error <= tolerance) & (arrival < stored_arrival)
            score = arrival + float(self.p.window_error_cost) * error
        eligible[0] = False
        if not bool(eligible.any()):
            return
        best = int(torch.where(eligible, score, torch.full_like(score, float("inf"))).argmin())

        horizon_steps = int(actions.shape[1])
        delay = int(pending.shape[0])
        sample_timeline = torch.cat((pending, actions[best]))  # issued at -delay .. H - 1
        # The candidate's commands until it crosses the first downstream waypoint, whose stored sequence
        # (and all after it) take over from there.
        handover = span
        last = int(idx[handover])
        reached = float(times[best, handover])
        stored_timeline = torch.cat((self._wp_pending[last], self._wp_actions[last]))

        def commands(at_times: torch.Tensor) -> torch.Tensor:
            own = self._timeline_at(sample_timeline, -delay, at_times)
            downstream = self._timeline_at(stored_timeline, reached - delay, at_times)
            return torch.where((at_times < reached)[:, None], own, downstream)

        steps = torch.arange(horizon_steps, dtype=torch.float32)
        queue = torch.arange(-delay, 0, dtype=torch.float32)
        updates = [(w, commands(steps), None, None)]
        for j in range(span):
            t = float(times[best, j])
            updates.append((int(idx[j]), commands(t + steps), commands(t + queue), at[best, j]))
        for k, seq, queue_k, state_k in updates:
            history = self._wp_window_hist[k]
            history.append((
                int(self.total_timesteps), self.waypoints[k].clone(), self._wp_actions[k].clone(),
                bool(self._wp_action_known[k]), self._wp_pending[k].clone(),
            ))
            del history[:-WINDOW_HISTORY]
            self._wp_actions[k] = seq
            if queue_k is not None:
                self._wp_pending[k] = queue_k
                self._write_waypoint_state(k, state_k)
        self._wp_window_accepted += 1

    # ------------------------------------------------------------------
    # One control step (Alg. 1)
    # ------------------------------------------------------------------
    def step(self, state: np.ndarray) -> tuple[float, float]:
        """Command ``(steering angle, acceleration)`` for the observed state ``state`` (float32)."""
        with torch.no_grad():
            action = self._select_action(state)
        steering, accel = np.asarray(action, dtype=np.float32).reshape(-1) * self.action_denorm
        self.angular_control = float(steering)
        self.translational_control = float(np.clip(float(accel), -MAX_TRANSLATIONAL_CONTROL, MAX_TRANSLATIONAL_CONTROL))
        self._issued.append(np.array(action, dtype=np.float32).reshape(-1))
        self.total_timesteps += 1
        self.episode_timesteps += 1
        return self.angular_control, self.translational_control

    def _select_action(self, state: np.ndarray) -> np.ndarray:
        self._refresh_recovery(state)
        # In recovery, and before any sequence is stored (first lap), MPPI samples around the shifted plan.
        nominal = None if self.recover_mode else self._interpolated_waypoint_actions(state)
        horizon = self.horizon
        self.action_buffer = self._fit_horizon(self.action_buffer, horizon)
        if nominal is not None:
            nominal = self._fit_horizon(nominal, horizon)
        if nominal is not None and self._mean_state_seed is not None and int(self._mean_state_seed.shape[0]) == horizon:
            nominal = self._mean_state_seed  # retrieved at the end of the previous step along its mean prediction
        actions = self._sample_actions(self.action_buffer if nominal is None else nominal, int(self.p.rollouts))
        flat = np.asarray(state).reshape(-1)
        actions = self._with_fallback_samples(actions, nominal, float(flat[LINEAR_VEL_X_IDX]))
        window = self._window_job()
        if window is not None:
            # Driving samples from the car and window candidates from the stored state w, in one batch.
            trajectories, window_states = self._rollout_with_window(state, actions, window)
        else:
            trajectories = self.rollout(state, actions)
        start = self.nearest_waypoint_index(state)
        drive_rewards, drive_crash = self._drive_rewards(trajectories, start)
        drive_total = drive_rewards + drive_crash
        weights = torch.softmax(drive_total / float(self.p.temperature), dim=0)  # Eq. 2
        action_seq = (weights[:, None, None] * actions).sum(dim=0)
        action_seq, weights = self._safe_mean(state, actions, action_seq, weights, drive_total, drive_crash)
        # The shifted plan U_shift for the next step.
        self.action_buffer = torch.cat((action_seq[1:], action_seq[-1:]), dim=0)
        drive_mean = self._weighted_state_trajectory(trajectories, weights)
        self.mean_states = drive_mean
        if self._stall_phase != 1 and self._blocked(
            float(flat[LINEAR_VEL_X_IDX]), trajectories, drive_crash, int(drive_total.argmax()),
        ):
            car_xy = torch.as_tensor(flat[[POSE_X_IDX, POSE_Y_IDX]], dtype=torch.float32)
            self._start_backtrack(car_xy, self._car_waypoint)
        if self._stall_phase == 1:
            self._reverse_left -= 1
            self._mean_state_seed = None
            return action_seq[0].cpu().numpy().astype(np.float32)
        if window is not None:
            self._window_update(window, window_states)
        self._record_driven(state, action_seq[0].detach())
        self._fill_passed_waypoints(state, action_seq.detach())
        # Next step's nominal sequence, retrieved along this step's mean prediction (Eq. 5).
        self._mean_state_seed = self._seed_from_states(drive_mean[1: 1 + horizon].detach())
        return action_seq[0].cpu().numpy().astype(np.float32)

    def _weighted_state_trajectory(self, trajectories: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        """Mean prediction X̄_t; the heading is rebuilt from the averaged sin and cos."""
        mean = torch.sum(weights[:, None, None] * trajectories, dim=0)
        mean[:, POSE_THETA_IDX] = torch.atan2(mean[:, POSE_THETA_SIN_IDX], mean[:, POSE_THETA_COS_IDX])
        return mean

    # ------------------------------------------------------------------
    # Rollouts with the planning model
    # ------------------------------------------------------------------
    def _rollout_with_window(self, state, actions, window):
        batch = int(actions.shape[0])
        _, start, pending, samples = window
        count = int(samples.shape[0])
        starts = torch.cat((torch.as_tensor(state, dtype=torch.float32).reshape(1, -1).expand(batch, -1),
                            start[None].expand(count, -1)))
        queues = torch.cat((self._issued_actions()[None].expand(batch, -1, -1), pending[None].expand(count, -1, -1)))
        out = self.rollout(starts.numpy(), torch.cat((actions, samples)).numpy(), pending_actions=queues.numpy())
        return out[:batch], out[batch:]

    def _scan_rollout(self):
        """Jitted rollout ``(state (B, D), actions (B, H, 2), pending (B, d, 2)) -> (B, H + 1, D)``: each step
        pushes the action into the delay queue and applies the oldest one, like the simulator."""
        fn = getattr(self, "_scan", None)
        if fn is None:
            import jax
            import jax.numpy as jnp
            from race.dynamics import car_dynamics

            car_params = jnp.asarray(self.car_params)
            dt = self.dt
            substeps = int(round(dt / self.dt_sim))

            def one(state, control):
                return car_dynamics(state, control, car_params, dt, substeps, ode_model='ODE:ks_pacejka')

            step = jax.jit(jax.vmap(one))
            denorm = jnp.asarray(np.asarray(self.action_denorm, dtype=np.float32))
            cap = float(MAX_TRANSLATIONAL_CONTROL)

            def run(state, actions, pending):
                def body(carry, action):
                    s, queue = carry
                    queue = jnp.concatenate((queue, action[:, None, :]), axis=1)
                    applied = queue[:, 0] * denorm
                    queue = queue[:, 1:]
                    applied = applied.at[:, 1].set(jnp.clip(applied[:, 1], -cap, cap))
                    s = step(s, applied)
                    s = s.at[:, POSE_THETA_IDX].set(jnp.arctan2(s[:, POSE_THETA_SIN_IDX], s[:, POSE_THETA_COS_IDX]))
                    return (s, queue), s

                _, trajectory = jax.lax.scan(body, (state, pending), jnp.swapaxes(actions, 0, 1))
                return jnp.concatenate((state[:, None], jnp.swapaxes(trajectory, 0, 1)), axis=1)

            fn = self._scan = jax.jit(run)
        return fn

    def rollout(self, state, actions, pending_actions=None) -> torch.Tensor:
        """States ``(B, H + 1, D)`` for action sequences ``(B, H, 2)`` from ``state``, with the car's delay queue
        or the given ``pending_actions`` ``(B, d, 2)``."""
        actions = np.asarray(actions, dtype=np.float32)
        if actions.ndim == 2:
            actions = actions[None]
        batch = actions.shape[0]
        if pending_actions is not None:
            pending = np.asarray(pending_actions, dtype=np.float32).reshape(batch, -1, 2).copy()
        else:
            pending = np.broadcast_to(self._issued_actions().numpy(), (batch, self.delay_steps, 2)).copy()
        s = np.broadcast_to(np.asarray(state, dtype=np.float32), (batch, STATE_DIM)).copy()
        out = self._scan_rollout()(s, actions, pending)
        return torch.as_tensor(np.array(out, dtype=np.float32, copy=True))

    def _issued_actions(self) -> torch.Tensor:
        """``(d, 2)`` the last commands of this episode, oldest first, zeros before the first one."""
        pending = torch.zeros((self.delay_steps, 2), dtype=torch.float32)
        if self._issued:
            pending[-len(self._issued):] = torch.from_numpy(np.stack(self._issued))
        return pending

    # ------------------------------------------------------------------
    # Display
    # ------------------------------------------------------------------
    def render_layers(self) -> dict:
        """What the viewer draws: memory arrows colored by stored speed, roll-back counts and the plan."""
        xy = self.waypoints[:, [WP_X_IDX, WP_Y_IDX]].numpy()
        heading = self.waypoints[:, WP_HEADING_IDX].numpy()
        speed = self.waypoints[:, WP_TARGET_SPEED_IDX].numpy()
        counts = self._wp_off_track_count.numpy()
        plan = None if self.mean_states is None else self.mean_states[:, [POSE_X_IDX, POSE_Y_IDX]].numpy()
        return {"arrows": (xy, heading, speed, float(self.car.v_max)), "counts": (xy, counts), "plan": plan,
                "known": self._waypoint_known(torch.arange(self._waypoints_n)).numpy(), "frozen": self.frozen}
