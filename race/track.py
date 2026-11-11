"""Track: occupancy map, wall distances, waypoints, nearest-waypoint search and lap timing."""

from __future__ import annotations

import os
import random

import numpy as np
import pandas as pd
import yaml
from numba import njit
from PIL import Image
from scipy.ndimage import distance_transform_edt

from race.state import POSE_X_IDX, POSE_Y_IDX

# Waypoint table columns
WP_S_IDX = 0  # distance along the line [m]
WP_X_IDX = 1
WP_Y_IDX = 2
WP_PSI_IDX = 3  # heading [rad], 0 along +x
WP_KAPPA_IDX = 4  # curvature [1/m]
WP_VX_IDX = 5  # target speed [m/s]
WP_A_X_IDX = 6
WP_GLOBID_IDX = 7
WP_D_RIGHT_IDX = 8  # distance to the right track border [m]
WP_D_LEFT_IDX = 9

DECREASE_RESOLUTION_FACTOR = 4  # every 4th point of the waypoint file is used
INITIAL_SPEED_LIMIT = 1.5  # [m/s] target speed of the midline waypoints
WALL_COLLISION_MARGIN = 0.005  # [m] a body corner closer to a wall than this is a crash

RANDOM_START_JITTER_XY = 1.0  # [m]
RANDOM_START_JITTER_YAW = 0.3  # [rad]
RANDOM_START_MIN_WALL_CLEARANCE = 0.15  # [m]
RANDOM_START_ATTEMPTS = 200


class Track:
    def __init__(self, map_dir: str, map_name: str, waypoint_source: str, car_length: float, car_width: float):
        self.map_dir = map_dir
        self.map_name = map_name
        self.car_length = car_length
        self.car_width = car_width
        with open(os.path.join(map_dir, map_name + ".yaml"), encoding="utf-8") as f:
            metadata = yaml.safe_load(f)
        self.resolution = float(metadata["resolution"])
        self.origin = [float(v) for v in metadata["origin"]]
        self.image_path = os.path.join(map_dir, map_name + ".png")
        image = np.array(Image.open(self.image_path).transpose(Image.FLIP_TOP_BOTTOM)).astype(np.float64)
        if image.ndim == 3:
            image = image[..., 0]
        image[image <= 128.0] = 0.0
        image[image > 128.0] = 255.0
        self.free = image > 0  # row 0 = lowest y
        self.height, self.width = image.shape
        self.wall_distance = self.resolution * distance_transform_edt(image)  # [m] from every cell to the nearest wall
        self.orig_c, self.orig_s = np.cos(self.origin[2]), np.sin(self.origin[2])
        self.waypoint_source = waypoint_source
        self.waypoints = load_waypoints(map_dir, map_name, waypoint_source)
        self.nearest_index = None

    # ------------------------------------------------------------------
    # Walls
    # ------------------------------------------------------------------
    def body_in_wall(self, x: float, y: float, theta: float) -> bool:
        """A body corner of a car at this pose within ``WALL_COLLISION_MARGIN`` of a wall."""
        pose = np.array([x, y, theta], dtype=np.float64)
        if not np.all(np.isfinite(pose)):
            return False
        vertices = get_vertices(pose, self.car_length, self.car_width)
        return bool(check_body_map_collision(
            vertices, self.origin[0], self.origin[1], self.orig_c, self.orig_s, self.height, self.width,
            self.resolution, self.wall_distance, 30.0, WALL_COLLISION_MARGIN))

    def start_clearance(self, pose) -> float:
        """Smallest distance of a body corner of a car at ``pose`` ``[x, y, theta]`` to a wall. Off the map: 0."""
        x, y, theta = (float(v) for v in pose)
        corner_x = np.array([-0.5, -0.5, 0.5, 0.5]) * self.car_length
        corner_y = np.array([0.5, -0.5, -0.5, 0.5]) * self.car_width
        px = x + np.cos(theta) * corner_x - np.sin(theta) * corner_y - self.origin[0]
        py = y + np.sin(theta) * corner_x + np.cos(theta) * corner_y - self.origin[1]
        cols = np.floor((px * self.orig_c + py * self.orig_s) / self.resolution).astype(np.int64)
        rows = np.floor((-px * self.orig_s + py * self.orig_c) / self.resolution).astype(np.int64)
        inside = (rows >= 0) & (rows < self.height) & (cols >= 0) & (cols < self.width)
        if not bool(inside.all()):
            return 0.0
        return float(self.wall_distance[rows, cols].min())

    def random_start(self) -> np.ndarray:
        """``[x, y, theta]`` near a random waypoint, clear of the walls (python ``random``)."""
        n = len(self.waypoints)
        base_idx = random.randint(0, n - 1)
        for attempt in range(RANDOM_START_ATTEMPTS):
            if attempt and attempt % 20 == 0:
                base_idx = random.randint(0, n - 1)
            wp = np.array(self.waypoints[base_idx], copy=True)
            wp[WP_X_IDX] += random.uniform(0.0, RANDOM_START_JITTER_XY)
            wp[WP_Y_IDX] += random.uniform(0.0, RANDOM_START_JITTER_XY)
            wp[WP_PSI_IDX] += random.uniform(0.0, RANDOM_START_JITTER_YAW)
            if self.start_clearance(wp[1:4]) >= RANDOM_START_MIN_WALL_CLEARANCE:
                return wp[1:4]
        print("Warning: no random start clear of the walls; starting on the waypoint itself.")
        return np.array(self.waypoints[base_idx], copy=True)[1:4]

    # ------------------------------------------------------------------
    # Waypoints
    # ------------------------------------------------------------------
    def reset(self) -> None:
        self.nearest_index = None

    def update_nearest(self, car_state: np.ndarray) -> int:
        """Nearest waypoint to the car: a local search around the last one, the whole track after a jump."""
        if self.nearest_index is None:
            index, _ = get_nearest_waypoint(car_state, self.waypoints)
        else:
            index, dist = get_nearest_waypoint(car_state, self.waypoints, self.nearest_index, -3, 5)
            if dist > 100.0:  # 10 m
                index, _ = get_nearest_waypoint(car_state, self.waypoints)
        self.nearest_index = int(index)
        return self.nearest_index

    def next_waypoints(self, count: int, skip: int = 1) -> np.ndarray:
        """``(count, 10)`` waypoints ahead of the nearest one, the first ``skip`` left out."""
        idx = (np.arange(count + skip) + self.nearest_index) % len(self.waypoints)
        return self.waypoints[idx][skip:]


def load_waypoints(map_dir: str, map_name: str, source: str) -> np.ndarray:
    """``(N, 10)`` float32 waypoint table: the raceline file ``<map>_wp.csv`` or the track's ``centerline.csv``."""
    if source == "midline":
        table = _midline_table(os.path.join(map_dir, "centerline.csv"))
    else:
        df = pd.read_csv(os.path.join(map_dir, map_name + "_wp.csv"), comment="#", encoding="utf-8")
        table = np.zeros((df.shape[0], 10), dtype=np.float32)
        table[:, WP_S_IDX] = df["s_m"].to_numpy()
        table[:, WP_X_IDX] = df["x_m"].to_numpy()
        table[:, WP_Y_IDX] = df["y_m"].to_numpy()
        table[:, WP_PSI_IDX] = df["psi_rad"].to_numpy()
        table[:, WP_KAPPA_IDX] = df["kappa_radpm"].to_numpy()
        table[:, WP_VX_IDX] = df["vx_mps"].to_numpy()
        table[:, WP_A_X_IDX] = df["ax_mps2"].to_numpy()
        table[:, WP_GLOBID_IDX] = np.arange(df.shape[0])
        table[:, WP_D_RIGHT_IDX] = df["d_right_iqp"].to_numpy()
        table[:, WP_D_LEFT_IDX] = df["d_left_iqp"].to_numpy()
        table[:, WP_PSI_IDX] += 0.5 * np.pi  # the file has the normal angle
    table = np.array(table[::DECREASE_RESOLUTION_FACTOR])
    # Drop points at the same position
    _, unique = np.unique(table[:, 1:3], axis=0, return_index=True)
    table = np.array(table[np.sort(unique), :])
    if source == "midline":
        table[:, WP_VX_IDX] = INITIAL_SPEED_LIMIT
        table[:, WP_A_X_IDX] = 0.0
    return table


def _midline_table(path: str) -> np.ndarray:
    rows = np.loadtxt(path, delimiter=",", comments="#", dtype=np.float64, ndmin=2, encoding="utf-8")[:, :4]
    step = np.linalg.norm(np.diff(rows[:, :2], axis=0), axis=1)
    keep = np.ones(len(rows), dtype=bool)
    keep[1:] = step > 1e-4
    if np.linalg.norm(rows[0, :2] - rows[-1, :2]) < 1e-3:
        keep[-1] = False
    rows = rows[keep]
    xy = rows[:, :2]
    nxt, prev = np.roll(xy, -1, axis=0), np.roll(xy, 1, axis=0)
    forward = nxt - xy
    arc = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(xy, axis=0), axis=1))])
    a = np.linalg.norm(xy - prev, axis=1)
    b = np.linalg.norm(forward, axis=1)
    c = np.linalg.norm(nxt - prev, axis=1)
    cross = (xy[:, 0] - prev[:, 0]) * forward[:, 1] - (xy[:, 1] - prev[:, 1]) * forward[:, 0]
    table = np.zeros((len(xy), 10), dtype=np.float32)
    table[:, WP_S_IDX] = arc
    table[:, WP_X_IDX] = xy[:, 0]
    table[:, WP_Y_IDX] = xy[:, 1]
    table[:, WP_PSI_IDX] = np.arctan2(forward[:, 1], forward[:, 0])
    table[:, WP_KAPPA_IDX] = 2.0 * cross / np.maximum(a * b * c, 1e-9)
    table[:, WP_GLOBID_IDX] = np.arange(len(xy))
    table[:, WP_D_RIGHT_IDX] = rows[:, 2]
    table[:, WP_D_LEFT_IDX] = rows[:, 3]
    return table


class LapTimer:
    """Lap times from the sequence of nearest waypoints: a lap ends when the car is back where it started."""

    def __init__(self, total_waypoints: int, callback):
        self.total_waypoints = total_waypoints
        self.callback = callback
        self.reset()

    def reset(self) -> None:
        self.waypoint_log, self.time_log, self.ready = [], [], []
        self.start_time = None

    def _restart(self, index: int, time_now: float) -> None:
        self.waypoint_log, self.time_log, self.ready = [index], [time_now], [False]
        self.start_time = time_now

    def update(self, index: int, time_now: float) -> None:
        n = self.total_waypoints
        if not self.waypoint_log:
            self._restart(index, time_now)
            return
        while self.waypoint_log:  # driving backwards drops what was logged after the current waypoint
            diff = (index - self.waypoint_log[-1]) % n
            if diff > 3 * n // 4:
                self.waypoint_log.pop()
                self.time_log.pop()
                self.ready.pop()
            else:
                break
        if not self.waypoint_log:
            self._restart(index, time_now)
            return
        distance = (index - np.array(self.waypoint_log)) % n
        if distance[-1] == 0 and not self.ready[-1]:
            return
        for i, d in enumerate(distance):
            if d > n // 2:
                self.ready[i] = True
        done = [i for i, (d, r) in enumerate(zip(distance, self.ready)) if d < n // 4 and r]
        if done:
            last = max(done)
            if self.time_log[last] >= self.start_time:
                lap_time = time_now - self.time_log[last]
                self.start_time = time_now
                self.callback(lap_time)
            self.waypoint_log = self.waypoint_log[last + 1:]
            self.time_log = self.time_log[last + 1:]
            self.ready = self.ready[last + 1:]
        self.waypoint_log.append(index)
        self.time_log.append(time_now)
        self.ready.append(False)


# ----------------------------------------------------------------------
# Compiled helpers
# ----------------------------------------------------------------------
@njit(fastmath=True)
def get_nearest_waypoint(car_state, waypoints, last_index=None, lower=None, upper=None):
    car_position = np.array([car_state[POSE_X_IDX], car_state[POSE_Y_IDX]])
    num_waypoints = len(waypoints)
    if last_index is None or lower is None or upper is None:
        candidates = np.arange(num_waypoints)
    else:
        candidates = np.mod(np.arange(last_index + lower, last_index + upper + 1), num_waypoints)
    best_index = candidates[0]
    best_dist = np.sum((car_position - waypoints[best_index, WP_X_IDX:WP_Y_IDX + 1]) ** 2)
    for idx in candidates:
        dist = np.sum((car_position - waypoints[idx, WP_X_IDX:WP_Y_IDX + 1]) ** 2)
        if dist < best_dist:
            best_dist = dist
            best_index = idx
    return best_index, best_dist


@njit(cache=True)
def get_trmtx(pose):
    cos = np.cos(pose[2])
    sin = np.sin(pose[2])
    return np.array([[cos, -sin, 0., pose[0]], [sin, cos, 0., pose[1]], [0., 0., 1., 0.], [0., 0., 0., 1.]])


@njit(cache=True)
def get_vertices(pose, length, width):
    """Body corners ``(4, 2)``: rear left, rear right, front right, front left."""
    H = get_trmtx(pose)
    rl = H.dot(np.asarray([[-length / 2], [width / 2], [0.], [1.]])).flatten()
    rr = H.dot(np.asarray([[-length / 2], [-width / 2], [0.], [1.]])).flatten()
    fl = H.dot(np.asarray([[length / 2], [width / 2], [0.], [1.]])).flatten()
    fr = H.dot(np.asarray([[length / 2], [-width / 2], [0.], [1.]])).flatten()
    rl = rl / rl[3]
    rr = rr / rr[3]
    fl = fl / fl[3]
    fr = fr / fr[3]
    return np.asarray([[rl[0], rl[1]], [rr[0], rr[1]], [fr[0], fr[1]], [fl[0], fl[1]]])


@njit(cache=True)
def _distance_lookup(x, y, orig_x, orig_y, orig_c, orig_s, height, width, resolution, dt, max_range):
    x_trans = x - orig_x
    y_trans = y - orig_y
    x_rot = x_trans * orig_c + y_trans * orig_s
    y_rot = -x_trans * orig_s + y_trans * orig_c
    if x_rot < 0 or x_rot >= width * resolution or y_rot < 0 or y_rot >= height * resolution:
        return max_range
    c = int(x_rot / resolution)
    r = int(y_rot / resolution)
    if r < 0 or c < 0 or r >= height or c >= width:
        return max_range
    return dt[r, c]


@njit(cache=True)
def check_body_map_collision(vertices, orig_x, orig_y, orig_c, orig_s, height, width, resolution, dt, max_range, margin):
    for i in range(vertices.shape[0]):
        if _distance_lookup(vertices[i, 0], vertices[i, 1], orig_x, orig_y, orig_c, orig_s, height, width,
                            resolution, dt, max_range) < margin:
            return True
    return False
