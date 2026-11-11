"""Pygame viewer. Mouse wheel: zoom, drag: pan, F: follow the car, Esc / closing the window: end the run.

Auto-Tuning MPPI shows its memory as one arrow per waypoint, along the stored heading, 0.08 m long per m/s of
stored speed and colored from green (slow) to red (v_max), with the number of times the car left the corridor
there; the orange line is the executed plan (mean prediction). MPPI shows 20 of its rollouts (red), its plan
(orange) and the waypoints its cost uses (green).
"""

from __future__ import annotations

import sys
import time

import numpy as np
import pygame
from pygame import gfxdraw

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
MAP_UPSAMPLE = 4  # wall image pixels per map cell
MAP_SMOOTHING = 0.6  # [cells] blur before upsampling: rounds the steps of the grid, keeps straight walls in place
COORD_LIMIT = 30000  # gfxdraw takes 16-bit coordinates


def _windows_dpi_scale() -> float:
    """On Windows, draw at the screen's real resolution instead of letting Windows stretch (blur) the window."""
    if sys.platform != "win32":
        return 1.0
    try:
        import ctypes
        try:
            ctypes.windll.shcore.SetProcessDpiAwareness(2)
        except Exception:
            ctypes.windll.user32.SetProcessDPIAware()
        return max(1.0, ctypes.windll.user32.GetDpiForSystem() / 96.0)
    except Exception:
        return 1.0


UI_SCALE = _windows_dpi_scale()

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
        self.font = pygame.font.SysFont(None, round(22 * UI_SCALE))
        self.small_font = pygame.font.SysFont(None, round(16 * UI_SCALE))
        self.track = track
        self.follow = False
        self.dragging = False
        self.closed = False
        self._last_draw = 0.0

        # Walls as an image with smooth edges: the occupancy grid is blurred slightly and upsampled, and the edge
        # is anti-aliased. Row 0 of track.free is the lowest y; image rows go down, so it is flipped.
        from scipy.ndimage import gaussian_filter, zoom as upsample
        wall = gaussian_filter((~track.free[::-1]).astype(np.float32), MAP_SMOOTHING)
        wall = upsample(wall, MAP_UPSAMPLE, order=1)
        alpha = np.clip((wall - 0.5) * 2.0 * MAP_UPSAMPLE + 0.5, 0.0, 1.0)
        rgba = np.empty(alpha.shape + (4,), dtype=np.uint8)
        rgba[..., :3] = WALL
        rgba[..., 3] = np.rint(255 * alpha)
        self.map_surface = pygame.image.frombuffer(rgba.tobytes(), alpha.shape[::-1], "RGBA").convert_alpha()
        self._map_cache = None
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
        """Scale only the visible part of the wall image (cached while the view does not move)."""
        key = (self.zoom, *self.center, self.width, self.height)
        if self._map_cache is None or self._map_cache[0] != key:
            self._map_cache = (key, self._visible_map())
        if self._map_cache[1] is not None:
            surface, position = self._map_cache[1]
            self.screen.blit(surface, position)

    def _visible_map(self):
        k = MAP_UPSAMPLE / self.track.resolution  # image pixels per m
        ox, oy = self.track.origin[:2]
        top = oy + self.track.height * self.track.resolution  # y of image row 0
        left, right = self.center[0] - self.width / 2 / self.zoom, self.center[0] + self.width / 2 / self.zoom
        high, low = self.center[1] + self.height / 2 / self.zoom, self.center[1] - self.height / 2 / self.zoom
        w, h = self.map_surface.get_size()
        c0, c1 = max(0, int(np.floor((left - ox) * k))), min(w, int(np.ceil((right - ox) * k)))
        r0, r1 = max(0, int(np.floor((top - high) * k))), min(h, int(np.ceil((top - low) * k)))
        if c1 <= c0 or r1 <= r0:
            return None
        x0, y0 = self._to_screen([ox + c0 / k, top - r0 / k])[0]
        x1, y1 = self._to_screen([ox + c1 / k, top - r1 / k])[0]
        position = (int(round(x0)), int(round(y0)))
        size = (max(1, int(round(x1)) - position[0]), max(1, int(round(y1)) - position[1]))
        part = self.map_surface.subsurface(pygame.Rect(c0, r0, c1 - c0, r1 - r0))
        return pygame.transform.smoothscale(part, size), position

    # Shapes are anti-aliased polygons (gfxdraw). Geometry is computed for all shapes at once; off-screen ones
    # are skipped.
    def _on_screen(self, polygons, margin: float = 20.0) -> np.ndarray:
        lo, hi = polygons.min(axis=1), polygons.max(axis=1)
        return ((hi[:, 0] >= -margin) & (lo[:, 0] < self.width + margin)
                & (hi[:, 1] >= -margin) & (lo[:, 1] < self.height + margin))

    def _polygons(self, polygons, colors) -> None:
        """polygons: (n, corners, 2) screen points; colors: one color or (n, 3)."""
        if len(polygons) == 0:
            return
        keep = self._on_screen(polygons)
        points = np.rint(np.clip(polygons[keep], -COORD_LIMIT, COORD_LIMIT)).astype(int).tolist()
        if np.ndim(colors) == 1:
            colors = [tuple(colors)] * len(points)
        else:
            colors = [tuple(c) for c in np.asarray(colors)[keep].tolist()]
        for pts, color in zip(points, colors):
            gfxdraw.filled_polygon(self.screen, pts, color)
            gfxdraw.aapolygon(self.screen, pts, color)

    @staticmethod
    def _quads(a, b, width: float) -> np.ndarray:
        """Rectangles of the given width [px] along the segments a[i] -> b[i]."""
        d = b - a
        n = np.maximum(np.hypot(d[:, 0], d[:, 1]), 1e-9)[:, None]
        off = np.stack((-d[:, 1], d[:, 0]), axis=1) / n * (width / 2)
        return np.stack((a + off, b + off, b - off, a - off), axis=1)

    def _polygon(self, pts, color) -> None:
        self._polygons(np.asarray(pts)[None], color)

    def _dots(self, xy, color, radius: float) -> None:
        if xy is None or len(xy) == 0:
            return
        r = max(1, int(round(radius * UI_SCALE)))
        for sx, sy in self._to_screen(xy):
            if -r <= sx < self.width + r and -r <= sy < self.height + r:
                gfxdraw.filled_circle(self.screen, int(sx), int(sy), r, color)
                gfxdraw.aacircle(self.screen, int(sx), int(sy), r, color)

    def _line(self, xy, color, width: float) -> None:
        if xy is None or len(xy) < 2:
            return
        pts = self._to_screen(xy)
        w = width * UI_SCALE
        self._polygons(self._quads(pts[:-1], pts[1:], w), color)
        r = int(w / 2)
        if r >= 1:  # round joints
            for x, y in pts[1:-1].astype(int).tolist():
                if 0 <= x < self.width and 0 <= y < self.height:
                    gfxdraw.filled_circle(self.screen, x, y, r, color)

    def _arrows(self, xy, heading, speed, v_max) -> None:
        """One arrow per memory entry: shaft plus filled head, length 0.08 m per m/s, colored by speed."""
        scale = np.clip(np.abs(speed) / max(v_max, 1e-3), 0.0, 1.0) * (len(SPEED_STOPS) - 1)
        i = np.minimum(scale.astype(np.int64), len(SPEED_STOPS) - 2)
        colors = np.rint(SPEED_STOPS[i] + (SPEED_STOPS[i + 1] - SPEED_STOPS[i]) * (scale - i)[:, None]).astype(int)
        length = np.abs(speed) * 0.08 * self.zoom  # px
        keep = length >= 1.0
        base, length, colors = self._to_screen(xy)[keep], length[keep, None], colors[keep]
        d = np.stack((np.cos(heading[keep]), -np.sin(heading[keep])), axis=1)  # screen y points down
        normal = np.stack((-d[:, 1], d[:, 0]), axis=1)
        shaft_w = float(np.clip(0.012 * self.zoom, 1.2, 4.0)) * UI_SCALE
        head = np.minimum(0.4 * length, np.maximum(4.0 * shaft_w, 0.25 * length))
        tip = base + d * length
        neck = tip - d * head
        self._polygons(self._quads(base, neck + d * 0.5, shaft_w), colors)
        self._polygons(np.stack((tip, neck + normal * head * 0.45, neck - normal * head * 0.45), axis=1), colors)

    def _car(self, state) -> None:
        from race.state import POSE_THETA_IDX, POSE_X_IDX, POSE_Y_IDX
        pose = np.array([state[POSE_X_IDX], state[POSE_Y_IDX], state[POSE_THETA_IDX]], dtype=np.float64)
        corners = get_vertices(pose, self.track.car_length, self.track.car_width)
        self._polygon(self._to_screen(corners), CAR)

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
                    self.screen.blit(self.small_font.render(str(int(c)), True, TEXT), (int(sx) + 3, int(sy) - self.small_font.get_linesize()))
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
        self._text(["wheel: zoom   drag: pan   F: follow car   Esc: quit"], 10,
                   self.height - self.small_font.get_linesize() - 6, self.small_font)
        pygame.display.flip()

    def close(self) -> None:
        pygame.quit()
