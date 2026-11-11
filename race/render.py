"""Pygame viewer. Mouse wheel: zoom, drag: pan, F: follow the car, Esc / closing the window: end the run.

Auto-Tuning MPPI shows its memory as one arrow per waypoint, along the stored heading, 0.08 m long per m/s of
stored speed and colored from green (slow) to red (v_max), with the number of times the car left the corridor
there; the orange line is the executed plan (mean prediction). MPPI shows 20 of its rollouts (red), its plan
(orange) and the waypoints its cost uses (green).
"""

from __future__ import annotations

import time

import numpy as np
import pygame

from race.track import WP_X_IDX, WP_Y_IDX, get_vertices

BACKGROUND = (9, 32, 87)
WALL = (183, 193, 222)
WAYPOINTS = (64, 190, 255)
CAR = (172, 97, 185)
PLAN = (255, 165, 0)
ROLLOUTS = (250, 25, 30)
NEXT_WAYPOINTS = (0, 160, 0)
TEXT = (255, 255, 255)
SPEED_STOPS = np.array(((0, 190, 0), (240, 220, 0), (255, 140, 0), (210, 30, 20)), dtype=np.float64)
FPS = 60.0

NO_DISPLAY_HELP = """No display found (e.g. logged in over ssh), so no window can be shown. The run continues without it.
To see the window: connect with X forwarding, `ssh -X user@host` (macOS needs XQuartz, Windows an X server
such as VcXsrv or MobaXterm), or run on a computer with a screen. Set `render: false` to hide this message."""


def display_available() -> bool:
    """True if pygame can open a real window (not just its invisible offscreen fallback)."""
    try:
        pygame.display.init()
        available = pygame.display.get_driver() not in ("offscreen", "dummy")
        if available:
            pygame.display.set_mode((1, 1), pygame.HIDDEN)
    except pygame.error:  # some SDL versions raise instead of falling back to offscreen
        available = False
    pygame.display.quit()
    return available


class Viewer:
    def __init__(self, track, title: str):
        pygame.init()
        info = pygame.display.Info()
        self.width = int(0.7 * info.current_w) if info.current_w > 0 else 1200
        self.height = int(self.width / 1.5)
        self.screen = pygame.display.set_mode((self.width, self.height), pygame.RESIZABLE)
        pygame.display.set_caption(title)
        self.font = pygame.font.SysFont(None, 22)
        self.small_font = pygame.font.SysFont(None, 16)
        self.track = track
        self.follow = False
        self.dragging = False
        self.closed = False
        self._last_draw = 0.0

        # Walls as an image, row 0 = lowest y. Screen rows go down, so the image is flipped back.
        rgba = np.zeros((track.height, track.width, 4), dtype=np.uint8)
        wall = ~track.free[::-1]
        rgba[wall, :3] = WALL
        rgba[wall, 3] = 255
        self.map_surface = pygame.image.frombuffer(rgba.tobytes(), (track.width, track.height), "RGBA").convert_alpha()
        self._scaled_map, self._scaled_zoom = None, None
        # Fit the whole map into the window.
        extent = np.array([track.width, track.height]) * track.resolution
        self.zoom = 0.9 * min(self.width / extent[0], self.height / extent[1])
        self.center = np.array(track.origin[:2]) + 0.5 * extent

    # ------------------------------------------------------------------
    def _to_screen(self, xy) -> np.ndarray:
        xy = np.asarray(xy, dtype=np.float64).reshape(-1, 2)
        sx = (xy[:, 0] - self.center[0]) * self.zoom + self.width / 2
        sy = -(xy[:, 1] - self.center[1]) * self.zoom + self.height / 2
        return np.stack((sx, sy), axis=1)

    def _handle_events(self) -> None:
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                self.closed = True
            elif event.type == pygame.VIDEORESIZE:
                self.width, self.height = event.w, event.h
                self.screen = pygame.display.set_mode((self.width, self.height), pygame.RESIZABLE)
            elif event.type == pygame.MOUSEWHEEL:
                self._zoom_at(pygame.mouse.get_pos(), 1.2 if event.y > 0 else 1 / 1.2)
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                self.dragging, self.follow = True, False
            elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
                self.dragging = False
            elif event.type == pygame.MOUSEMOTION and self.dragging:
                dx, dy = event.rel
                self.center -= np.array([dx, -dy]) / self.zoom
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_f:
                self.follow = not self.follow

    def _zoom_at(self, mouse, factor: float) -> None:
        before = self._to_world(mouse)
        self.zoom = float(np.clip(self.zoom * factor, 2.0, 1000.0))
        self.center += before - self._to_world(mouse)

    def _to_world(self, point) -> np.ndarray:
        return np.array([(point[0] - self.width / 2) / self.zoom + self.center[0],
                         -(point[1] - self.height / 2) / self.zoom + self.center[1]])

    # ------------------------------------------------------------------
    def _draw_map(self) -> None:
        if self._scaled_zoom != self.zoom:
            size = (max(1, int(self.track.width * self.track.resolution * self.zoom)),
                    max(1, int(self.track.height * self.track.resolution * self.zoom)))
            self._scaled_map = pygame.transform.smoothscale(self.map_surface, size)
            self._scaled_zoom = self.zoom
        top_left = self._to_screen([self.track.origin[0], self.track.origin[1] + self.track.height * self.track.resolution])[0]
        self.screen.blit(self._scaled_map, (int(top_left[0]), int(top_left[1])))

    def _dots(self, xy, color, radius: float) -> None:
        if xy is None or len(xy) == 0:
            return
        r = max(1, int(round(radius)))
        for sx, sy in self._to_screen(xy):
            if -r <= sx < self.width + r and -r <= sy < self.height + r:
                pygame.draw.circle(self.screen, color, (int(sx), int(sy)), r)

    def _line(self, xy, color, width: int) -> None:
        if xy is None or len(xy) < 2:
            return
        pygame.draw.lines(self.screen, color, False, [tuple(p) for p in self._to_screen(xy)], width)

    def _arrows(self, xy, heading, speed, v_max) -> None:
        length = np.abs(speed) * 0.08
        tip = xy + np.stack((length * np.cos(heading), length * np.sin(heading)), axis=1)
        head, wing = length * 0.35, 0.5
        left = tip - np.stack((head * np.cos(heading - wing), head * np.sin(heading - wing)), axis=1)
        right = tip - np.stack((head * np.cos(heading + wing), head * np.sin(heading + wing)), axis=1)
        scale = np.clip(np.abs(speed) / max(v_max, 1e-3), 0.0, 1.0) * (len(SPEED_STOPS) - 1)
        i = np.minimum(scale.astype(np.int64), len(SPEED_STOPS) - 2)
        colors = np.rint(SPEED_STOPS[i] + (SPEED_STOPS[i + 1] - SPEED_STOPS[i]) * (scale - i)[:, None]).astype(int)
        points = self._to_screen(np.stack((xy, tip, left, tip, right), axis=1).reshape(-1, 2)).reshape(-1, 5, 2)
        for pts, color in zip(points, colors):
            pygame.draw.lines(self.screen, tuple(color), False, [tuple(p) for p in pts], 2)

    def _car(self, state) -> None:
        from race.state import POSE_THETA_IDX, POSE_X_IDX, POSE_Y_IDX
        pose = np.array([state[POSE_X_IDX], state[POSE_Y_IDX], state[POSE_THETA_IDX]], dtype=np.float64)
        corners = get_vertices(pose, self.track.car_length, self.track.car_width)
        pygame.draw.polygon(self.screen, CAR, [tuple(p) for p in self._to_screen(corners)])

    def _text(self, lines, x, y, font=None) -> None:
        font = font or self.font
        width = max(font.size(line)[0] for line in lines) + 12
        box = pygame.Surface((width, font.get_linesize() * len(lines) + 6), pygame.SRCALPHA)
        box.fill((4, 12, 38, 200))
        self.screen.blit(box, (x - 6, y - 3))
        for line in lines:
            self.screen.blit(font.render(line, True, TEXT), (x, y))
            y += font.get_linesize()

    # ------------------------------------------------------------------
    def draw(self, state, layers_fn, info_fn) -> None:
        """One frame. Skipped (apart from the events) above ``FPS`` frames per wall second.

        ``layers_fn()`` returns the controller's display layers, ``info_fn()`` the status line values."""
        self._handle_events()
        now = time.monotonic()
        if now - self._last_draw < 1.0 / FPS:
            return
        self._last_draw = now
        layers, info = layers_fn(), info_fn()
        from race.state import POSE_X_IDX, POSE_Y_IDX
        if self.follow:
            self.center = np.array([state[POSE_X_IDX], state[POSE_Y_IDX]], dtype=np.float64)

        self.screen.fill(BACKGROUND)
        self._draw_map()
        if "arrows" in layers:
            xy, heading, speed, v_max = layers["arrows"]
            self._arrows(xy, heading, speed, v_max)
            count_xy, counts = layers["counts"]
            for (sx, sy), c in zip(self._to_screen(count_xy), counts):
                if c > 0:
                    self.screen.blit(self.small_font.render(str(int(c)), True, TEXT), (int(sx) + 3, int(sy) - 12))
        else:
            self._dots(self.track.waypoints[:, [WP_X_IDX, WP_Y_IDX]], WAYPOINTS, 2)
            self._dots(layers.get("next_waypoints"), NEXT_WAYPOINTS, 2.5)
        rollouts = layers.get("rollouts")
        if rollouts is not None:
            self._dots(rollouts.reshape(-1, 2), ROLLOUTS, 1.3)
        self._line(layers.get("plan"), PLAN, 3)
        self._car(state)

        laps = info["laps"]
        lines = [
            f"{info['controller']}   map {info['map']}   t = {info['time']:.1f} s   v = {info['speed']:.2f} m/s",
            f"laps {len(laps)}   last {laps[-1]:.2f} s   best {min(laps):.2f} s" if laps else "laps 0",
            f"crashes {info['crashes']}" + ("   memory frozen" if layers.get("frozen") else ""),
        ]
        self._text(lines, 10, 8)
        self._text(["wheel: zoom   drag: pan   F: follow car   Esc: quit"], 10, self.height - 20, self.small_font)
        pygame.display.flip()

    def close(self) -> None:
        pygame.quit()
