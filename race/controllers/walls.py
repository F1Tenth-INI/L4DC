"""Wall distances and free-space paths for Auto-Tuning MPPI, on the same grid as the simulator's collision check."""

from __future__ import annotations

import math

import numpy as np
import torch
from scipy import ndimage


class WallGeometry:
    def __init__(self, track, collision_margin: float):
        """``collision_margin`` [d_min]: a body corner closer to a wall than this counts as a crash in the planner."""
        self.collision_margin = float(collision_margin)
        self.car_length, self.car_width = float(track.car_length), float(track.car_width)
        half_l, half_w = 0.5 * self.car_length, 0.5 * self.car_width
        # Body corners in the car frame: rear left, rear right, front right, front left.
        self._corner_x = torch.tensor([-half_l, -half_l, half_l, half_l], dtype=torch.float64)
        self._corner_y = torch.tensor([half_w, -half_w, -half_w, half_w], dtype=torch.float64)
        self.resolution = float(track.resolution)
        self.origin_x, self.origin_y = float(track.origin[0]), float(track.origin[1])
        self.origin_cos, self.origin_sin = math.cos(float(track.origin[2])), math.sin(float(track.origin[2]))
        self.map_height, self.map_width = int(track.height), int(track.width)
        distance = self.resolution * ndimage.distance_transform_edt(track.free)
        self.wall_distance = torch.as_tensor(distance, dtype=torch.float32)

    def corners(self, poses) -> torch.Tensor:
        """Body corners ``(..., 4, 2)`` for poses ``(..., 3)`` ``[x, y, theta]`` (float64)."""
        p = torch.as_tensor(poses).to(torch.float64)
        cos_t = torch.cos(p[..., 2:3])
        sin_t = torch.sin(p[..., 2:3])
        x = p[..., 0:1] + cos_t * self._corner_x - sin_t * self._corner_y
        y = p[..., 1:2] + sin_t * self._corner_x + cos_t * self._corner_y
        return torch.stack((x, y), dim=-1)

    def _world_to_cells(self, xy: np.ndarray):
        x_t = xy[:, 0] - self.origin_x
        y_t = xy[:, 1] - self.origin_y
        x_rot = x_t * self.origin_cos + y_t * self.origin_sin
        y_rot = -x_t * self.origin_sin + y_t * self.origin_cos
        inside = (
            (x_rot >= 0) & (x_rot < self.map_width * self.resolution)
            & (y_rot >= 0) & (y_rot < self.map_height * self.resolution)
        )
        cols = np.clip((x_rot / self.resolution).astype(np.int64), 0, self.map_width - 1)
        rows = np.clip((y_rot / self.resolution).astype(np.int64), 0, self.map_height - 1)
        return rows, cols, inside

    def _grid_cells(self, p: torch.Tensor):
        """``(rows, cols, inside)`` of the map cells under world points ``(..., 2)``, clamped to the image."""
        q = p.detach().numpy()
        x_t = q[..., 0] - self.origin_x
        y_t = q[..., 1] - self.origin_y
        x_rot = x_t * self.origin_cos + y_t * self.origin_sin
        y_rot = -x_t * self.origin_sin + y_t * self.origin_cos
        inside = (
            (x_rot >= 0) & (x_rot < self.map_width * self.resolution)
            & (y_rot >= 0) & (y_rot < self.map_height * self.resolution)
        )
        cols = np.fmin(np.fmax(np.floor(x_rot / self.resolution), 0.0), self.map_width - 1)
        rows = np.fmin(np.fmax(np.floor(y_rot / self.resolution), 0.0), self.map_height - 1)
        return (
            torch.from_numpy(np.asarray(rows, dtype=np.int64)), torch.from_numpy(np.asarray(cols, dtype=np.int64)),
            torch.from_numpy(np.asarray(inside)),
        )

    def in_wall(self, poses) -> torch.Tensor:
        """``(...)`` bool: a body corner closer to a wall than the collision margin, or off the map."""
        rows, cols, inside = self._grid_cells(self.corners(poses))
        return (~inside | (self.wall_distance[rows, cols] < self.collision_margin)).any(dim=-1)

    def distance_at(self, xy) -> torch.Tensor:
        """Distance [m] from points ``(..., 2)`` to the nearest wall. Off the map: 0."""
        p = torch.as_tensor(xy).to(torch.float64)
        rows, cols, inside = self._grid_cells(p)
        distance = self.wall_distance[rows, cols]
        return torch.where(inside, distance, torch.zeros_like(distance))

    def build_track_field(self, points, clearance: float, lookahead: float = 0.5) -> None:
        """Shortest paths through free space from every map cell to the nearest of ``points`` ``(N, 2)``.

        A cell is passable when it is at least ``clearance`` from a wall. ``track_field_at`` then gives, for
        any position, the path length, the index of the point it ends at, and the unit direction the path
        takes ``lookahead`` m along it.
        """
        from scipy.sparse import coo_matrix
        from scipy.sparse.csgraph import dijkstra

        pts = torch.as_tensor(points).detach().cpu().numpy().astype(np.float64)
        height, width = self.map_height, self.map_width
        passable = self.wall_distance.cpu().numpy() >= float(clearance)
        p_rows, p_cols, p_inside = self._world_to_cells(pts)
        passable[p_rows[p_inside], p_cols[p_inside]] = True
        cells = np.flatnonzero(passable)
        node = np.full(height * width, -1, dtype=np.int64)
        node[cells] = np.arange(cells.size)
        rows, cols = np.divmod(cells, width)
        edges_from, edges_to, weights = [], [], []
        for d_row, d_col in ((0, 1), (1, 0), (1, 1), (1, -1)):
            n_rows, n_cols = rows + d_row, cols + d_col
            ok = (n_rows >= 0) & (n_rows < height) & (n_cols >= 0) & (n_cols < width)
            to = np.full(cells.size, -1, dtype=np.int64)
            to[ok] = node[n_rows[ok] * width + n_cols[ok]]
            ok &= to >= 0
            edges_from.append(np.flatnonzero(ok))
            edges_to.append(to[ok])
            weights.append(np.full(int(ok.sum()), self.resolution * math.hypot(d_row, d_col)))
        graph = coo_matrix(
            (np.concatenate(weights), (np.concatenate(edges_from), np.concatenate(edges_to))),
            shape=(cells.size, cells.size),
        ).tocsr()
        point_of_node = np.full(cells.size, -1, dtype=np.int64)
        source_nodes = node[p_rows[p_inside] * width + p_cols[p_inside]]
        point_of_node[source_nodes] = np.flatnonzero(p_inside)
        sources = np.unique(source_nodes)
        length, predecessor, source = dijkstra(
            graph, directed=False, indices=sources, min_only=True, return_predecessors=True
        )
        reached = np.isfinite(length)
        label = np.where(reached, point_of_node[np.clip(source, 0, None)], -1)
        ahead = np.arange(cells.size)
        for _ in range(max(1, int(round(float(lookahead) / self.resolution)))):
            step = predecessor[ahead]
            ahead = np.where(step >= 0, step, ahead)
        d_col = (cols[ahead] - cols).astype(np.float64)
        d_row = (rows[ahead] - rows).astype(np.float64)
        dx = d_col * self.origin_cos - d_row * self.origin_sin
        dy = d_col * self.origin_sin + d_row * self.origin_cos
        norm = np.maximum(np.hypot(dx, dy), 1e-9)
        direction = np.stack((dx / norm, dy / norm), axis=-1) * (np.hypot(dx, dy) > 0)[:, None]

        grid_length = np.full(height * width, np.inf)
        grid_label = np.full(height * width, -1, dtype=np.int64)
        grid_direction = np.zeros((height * width, 2))
        grid_length[cells] = length
        grid_label[cells] = label
        grid_direction[cells] = direction
        grid_reached = np.isfinite(grid_length).reshape(height, width)
        if grid_reached.any():
            # Cells the search does not reach copy the nearest reached cell, plus the straight distance to it.
            gap, (near_rows, near_cols) = ndimage.distance_transform_edt(~grid_reached, return_indices=True)
            near = (near_rows * width + near_cols).ravel()
            grid_length = grid_length[near] + self.resolution * gap.ravel()
            grid_label = grid_label[near]
            grid_direction = grid_direction[near]
        self.track_length = torch.as_tensor(grid_length.reshape(height, width), dtype=torch.float32)
        self.track_label = torch.as_tensor(grid_label.reshape(height, width))
        self.track_direction = torch.as_tensor(grid_direction.reshape(height, width, 2), dtype=torch.float32)

    def track_field_at(self, xy):
        """``(length, index, direction)`` of the free-space path from points ``(..., 2)`` (see ``build_track_field``)."""
        rows, cols, _ = self._grid_cells(torch.as_tensor(xy).to(torch.float64))
        return self.track_length[rows, cols], self.track_label[rows, cols], self.track_direction[rows, cols]
