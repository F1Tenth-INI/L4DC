"""MPPI baseline: warm-started with the shifted previous plan, cost on the raceline, walls and progress.

Only the number of rollouts is a setting (settings.yml, ``mppi.rollouts``); everything else is fixed here.
"""

from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

from race.dynamics import car_steps_sequential
from race.state import LINEAR_VEL_X_IDX, POSE_THETA_IDX, POSE_X_IDX, POSE_Y_IDX
from race.track import WP_D_LEFT_IDX, WP_D_RIGHT_IDX, WP_PSI_IDX, WP_VX_IDX, WP_X_IDX, WP_Y_IDX

HORIZON = 35  # steps (1.4 s)
INTERMEDIATE_STEPS = 2  # Euler steps of the prediction model per control step
NOISE_STD = (0.2, 0.6)  # sampling noise: steering [rad], acceleration [m/s²]
LOOK_AHEAD_WAYPOINTS = 30  # waypoints ahead of the car the cost is computed on

# Cost weights. A pytree argument of the jitted functions.
COST_WEIGHTS = {
    'lateral': 5.0,           # (lateral offset from raceline [m])^2
    'heading': 2.0,           # (heading error to raceline [rad])^2
    'speed': 0.0,             # (v - speed_factor * raceline speed)^2. Off
    'speed_factor': 0.0,
    'progress': 3.0,          # reward per m/s driven along the raceline
    'border': 200.0,          # (penetration into the border margin [m])^2
    'border_margin': 0.1,     # [m] distance of the car centre to the track edge (d_left / d_right) where it starts
    'wall': 200.0,            # (penetration of the closest body corner into the wall margin [m])^2
    'wall_margin': 0.3,       # [m]
    'crash': 1000.0,          # per step once a body corner is closer than crash_margin to a wall
    'crash_margin': 0.15,     # [m]
    'angular_control': 0.1,   # |steering command|
    'translational_control': 0.01,  # |acceleration command|
    'angular_smoothness': 1.0,      # (steering change per step)^2
    'translational_smoothness': 0.01,  # (acceleration change per step)^2
    # Terminal cost on the last state of the horizon
    'terminal_lateral': 25.0,
    'terminal_heading': 5.0,
    'terminal_speed': 1000.0,        # (v - terminal_speed_factor * waypoint speed)^2, only above that speed
    'terminal_speed_factor': 1.0,
    'terminal_heading_bound': 500.0, # (|heading error| - terminal_max_heading)^2, only beyond that bound
    'terminal_max_heading': 0.087,   # [rad]
    'temperature': 3.0,              # softmax temperature on (cost - min cost)
}


class MPPI:
    def __init__(self, cfg, sim_cfg, car, track):
        self.batch_size = int(cfg.rollouts)
        self.horizon = HORIZON
        self.dt = float(sim_cfg.timestep_control)
        self.delay_steps = int(sim_cfg.delay_steps)
        self.cost_weights = {k: jnp.float32(v) for k, v in COST_WEIGHTS.items()}
        self.car_params = jnp.array(car.to_array())
        # Commands beyond these limits saturate in the car model.
        mu = float(car.to_array()[0])
        steering_limit = car.s_max / car.steering_multiplier
        accel_limit = min(car.a_max, mu * car.g) / car.accel_multiplier
        brake_limit = max(car.a_min, -mu * car.g) / car.brake_multiplier
        self.control_limits = jnp.array([[-steering_limit, brake_limit], [steering_limit, accel_limit]], dtype=jnp.float32)
        self.wall_map = wall_map(track)
        self.key = jax.random.PRNGKey(int(sim_cfg.seed))
        self.track = track
        self.reset()

    def reset(self) -> None:
        self.last_Q_sq = np.zeros((self.horizon, 2), dtype=np.float32)
        self.sent_controls = np.zeros((self.delay_steps, 2), dtype=np.float32)  # in the actuator delay
        self.rollouts = None
        self.optimal_trajectory = None

    def step(self, state: np.ndarray) -> tuple[float, float]:
        s = jnp.array(state, dtype=jnp.float32)
        waypoints = jnp.array(self.track.next_waypoints(LOOK_AHEAD_WAYPOINTS), dtype=jnp.float32)
        self.key, subkey = jax.random.split(self.key)
        Q_sequence, optimal_traj, state_batch, _ = mppi_step(
            s, jnp.array(self.last_Q_sq), jnp.array(self.sent_controls), self.batch_size, self.horizon,
            self.car_params, waypoints, subkey, self.cost_weights, self.wall_map, self.control_limits,
            self.delay_steps, self.dt, INTERMEDIATE_STEPS, NOISE_STD,
        )
        self.rollouts = state_batch
        self.optimal_trajectory = optimal_traj
        Q_sequence = np.array(Q_sequence)
        # Q_sequence[:delay_steps] are the commands already in the delay; this one is applied next.
        steering, accel = Q_sequence[self.delay_steps]
        if self.delay_steps > 0:
            self.sent_controls = np.concatenate([self.sent_controls[1:], Q_sequence[self.delay_steps][None]], axis=0)
        self.last_Q_sq = Q_sequence
        return float(steering), float(accel)

    def on_crash(self) -> None:
        pass

    def report_lap_time(self, lap_time: float) -> None:
        pass

    def render_layers(self) -> dict:
        if self.rollouts is None:
            return {}
        rollouts = np.array(self.rollouts[:20, :, :])[..., [POSE_X_IDX, POSE_Y_IDX]]
        plan = np.array(self.optimal_trajectory)[:, [POSE_X_IDX, POSE_Y_IDX]]
        return {"rollouts": rollouts, "plan": plan,
                "next_waypoints": self.track.next_waypoints(LOOK_AHEAD_WAYPOINTS)[:, [WP_X_IDX, WP_Y_IDX]]}


@partial(jax.jit, static_argnames=["batch_size", "horizon", "delay_steps", "dt", "intermediate_steps", "noise_std"])
def mppi_step(state, last_Q_sq, sent_controls, batch_size, horizon, car_params, waypoints, key, w, wall_map,
              control_limits, delay_steps, dt, intermediate_steps, noise_std):
    s_batch = jnp.repeat(state[None, :], batch_size, axis=0)
    # Warm start: the last plan shifted by one step, its last control repeated
    base_Q = jnp.concatenate([last_Q_sq[1:], last_Q_sq[-1:]], axis=0)
    Q_batch_sequence = jnp.repeat(base_Q[None, :, :], batch_size, axis=0)
    key1, key2 = jax.random.split(key)
    noise = jnp.stack([
        jax.random.normal(key1, (batch_size, horizon)) * noise_std[0],
        jax.random.normal(key2, (batch_size, horizon)) * noise_std[1]
    ], axis=-1)
    noise = noise.at[0].set(0.0)  # sample 0 re-evaluates the shifted last plan
    Q_batch_sequence = jnp.clip(Q_batch_sequence + noise, control_limits[0], control_limits[1])
    # The first delay_steps controls are already in the delay buffer
    Q_batch_sequence = Q_batch_sequence.at[:, :delay_steps].set(sent_controls)

    traj_batch = jax.vmap(lambda s_single, Q_single: car_steps_sequential(
        s_single, Q_single, car_params, dt, horizon, intermediate_steps=intermediate_steps
    ))(s_batch, Q_batch_sequence)

    cost_batch = cost_function_batch_sequence(traj_batch, Q_batch_sequence, waypoints, w, wall_map)
    total_cost = jnp.sum(cost_batch, axis=1)
    weights = jax.nn.softmax(-(total_cost - jnp.min(total_cost)) / w['temperature'])
    Q_sequence = jnp.sum(Q_batch_sequence * weights[:, None, None], axis=0)
    optimal_trajectory = car_steps_sequential(state, Q_sequence, car_params, dt, horizon,
                                              intermediate_steps=intermediate_steps)
    return Q_sequence, optimal_trajectory, traj_batch, total_cost


def wall_map(track) -> dict:
    """Wall distance grid of the track, as the simulator's collision check sees it."""
    half_l, half_w = 0.5 * track.car_length, 0.5 * track.car_width
    return {
        'distance': jnp.array(track.wall_distance, dtype=jnp.float32),  # row 0 = lowest y
        'origin': jnp.array([track.origin[0], track.origin[1]], dtype=jnp.float32),
        'rotation': jnp.array([np.cos(track.origin[2]), np.sin(track.origin[2])], dtype=jnp.float32),
        'resolution': jnp.float32(track.resolution),
        # Body corners in the car frame: rl, rr, fr, fl
        'corners': jnp.array([[-half_l, half_w], [-half_l, -half_w], [half_l, -half_w], [half_l, half_w]], dtype=jnp.float32),
    }


@jax.jit
def wall_distance(state, wall_map):
    """Distance from the closest body corner to a wall, bilinear between cell centres. Off the map: 0."""
    c, s = jnp.cos(state[POSE_THETA_IDX]), jnp.sin(state[POSE_THETA_IDX])
    corners = wall_map['corners']
    x = state[POSE_X_IDX] + c * corners[:, 0] - s * corners[:, 1] - wall_map['origin'][0]
    y = state[POSE_Y_IDX] + s * corners[:, 0] + c * corners[:, 1] - wall_map['origin'][1]
    oc, os_ = wall_map['rotation']
    u = (x * oc + y * os_) / wall_map['resolution'] - 0.5
    v = (-x * os_ + y * oc) / wall_map['resolution'] - 0.5
    height, width = wall_map['distance'].shape
    inside = (u >= -0.5) & (u < width - 0.5) & (v >= -0.5) & (v < height - 0.5)
    col, row = jnp.floor(u).astype(jnp.int32), jnp.floor(v).astype(jnp.int32)
    fu, fv = u - col, v - row
    grid = lambda r, c: wall_map['distance'][jnp.clip(r, 0, height - 1), jnp.clip(c, 0, width - 1)]
    distance = ((1 - fv) * ((1 - fu) * grid(row, col) + fu * grid(row, col + 1))
                + fv * ((1 - fu) * grid(row + 1, col) + fu * grid(row + 1, col + 1)))
    return jnp.min(jnp.where(inside, distance, 0.0))


@jax.jit
def compute_waypoint_distance(state, waypoints):
    dx = state[POSE_X_IDX] - waypoints[:, WP_X_IDX]
    dy = state[POSE_Y_IDX] - waypoints[:, WP_Y_IDX]
    dist_sq = dx ** 2 + dy ** 2
    return jnp.min(dist_sq), jnp.argmin(dist_sq)


@jax.jit
def raceline_errors(state, waypoints):
    """Nearest waypoint, signed lateral offset (left positive) and heading error to the raceline."""
    _, i = compute_waypoint_distance(state, waypoints)
    wp = waypoints[i]
    psi = wp[WP_PSI_IDX]
    dx = state[POSE_X_IDX] - wp[WP_X_IDX]
    dy = state[POSE_Y_IDX] - wp[WP_Y_IDX]
    lateral = -jnp.sin(psi) * dx + jnp.cos(psi) * dy
    heading_error = jnp.arctan2(jnp.sin(state[POSE_THETA_IDX] - psi), jnp.cos(state[POSE_THETA_IDX] - psi))
    return wp, lateral, heading_error


@jax.jit
def cost_function(state, control, waypoints, w, wall_map):
    wp, lateral, heading_error = raceline_errors(state, waypoints)

    edge_distance = jnp.minimum(wp[WP_D_LEFT_IDX] - lateral, wp[WP_D_RIGHT_IDX] + lateral)
    border_cost = w['border'] * jax.nn.relu(w['border_margin'] - edge_distance) ** 2

    wall_dist = wall_distance(state, wall_map)
    wall_cost = w['wall'] * jax.nn.relu(w['wall_margin'] - wall_dist) ** 2
    crash_cost = w['crash'] * (wall_dist < w['crash_margin'])

    v = state[LINEAR_VEL_X_IDX]
    target_speed = w['speed_factor'] * wp[WP_VX_IDX]
    speed_cost = w['speed'] * (v - target_speed) ** 2
    progress_cost = -w['progress'] * v * jnp.cos(heading_error)

    tracking_cost = w['lateral'] * lateral ** 2 + w['heading'] * heading_error ** 2
    control_cost = w['angular_control'] * jnp.abs(control[0]) + w['translational_control'] * jnp.abs(control[1])

    return tracking_cost + speed_cost + progress_cost + border_cost + wall_cost + crash_cost + control_cost


@jax.jit
def terminal_cost(state, waypoints, w):
    """Near the raceline, heading within terminal_max_heading and not faster than the waypoint speed."""
    wp, lateral, heading_error = raceline_errors(state, waypoints)
    tracking_cost = w['terminal_lateral'] * lateral ** 2 + w['terminal_heading'] * heading_error ** 2
    heading_bound_cost = w['terminal_heading_bound'] * jax.nn.relu(jnp.abs(heading_error) - w['terminal_max_heading']) ** 2
    speed_cost = w['terminal_speed'] * jax.nn.relu(state[LINEAR_VEL_X_IDX] - w['terminal_speed_factor'] * wp[WP_VX_IDX]) ** 2
    return tracking_cost + heading_bound_cost + speed_cost


@jax.jit
def cost_function_sequence(state_sequence, control_sequence, waypoints, w, wall_map):
    standard_costs = jax.vmap(lambda s, u: cost_function(s, u, waypoints, w, wall_map))(state_sequence, control_sequence)
    control_diff = control_sequence[1:] - control_sequence[:-1]
    smoothness_costs = (w['angular_smoothness'] * control_diff[:, 0] ** 2
                        + w['translational_smoothness'] * control_diff[:, 1] ** 2)
    costs = standard_costs.at[1:].add(smoothness_costs)
    return costs.at[-1].add(terminal_cost(state_sequence[-1], waypoints, w))


@jax.jit
def cost_function_batch_sequence(state_batch_sequence, Q_batch_sequence, waypoints, w, wall_map):
    return jax.vmap(lambda states, controls: cost_function_sequence(states, controls, waypoints, w, wall_map))(
        state_batch_sequence, Q_batch_sequence)
