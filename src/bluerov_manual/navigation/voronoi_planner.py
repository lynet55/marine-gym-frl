from __future__ import annotations

from dataclasses import dataclass
from heapq import heappop, heappush
from typing import List, Sequence, Tuple

import math
import random

import numpy as np

from .global_planner import GlobalPlanner, GlobalPlannerConfig
from .obstacles import Obstacle


@dataclass
class VoronoiPlannerConfig:
    grid_resolution: float = 0.20
    fixed_depth: float | None = None
    bounds_min: Tuple[float, float, float] = (-10.0, -10.0, -5.0)
    bounds_max: Tuple[float, float, float] = (10.0, 10.0, 5.0)
    wall_margin: float = 0.0
    goal_tolerance: float = 1.0
    # Occupancy / clearance behavior.
    occ_threshold: float = 0.5
    occ_step: float | None = None
    clearance_eps: float = 0.05
    # Hard minimum clearance from occupied cells/walls (meters). Cells below this are blocked.
    min_clearance: float = 0.0
    # Path post-processing.
    shortcut_enable: bool = True
    shortcut_trials: int = 120
    resample_enable: bool = True
    resample_step: float = 0.4
    # Fallback RRT if grid search fails.
    fallback_enable: bool = True
    fallback_step_size: float = 1.0
    fallback_max_iters: int = 500
    fallback_goal_sample_rate: float = 0.1
    fallback_neighbor_radius: float = 2.0


class VoronoiPlanner:
    """Clearance-field + A* global replanner (Voronoi-style high-clearance routing)."""

    def __init__(self, cfg: VoronoiPlannerConfig):
        self.cfg = cfg
        self._fallback = None
        if self.cfg.fallback_enable:
            self._fallback = GlobalPlanner(
                GlobalPlannerConfig(
                    bounds_min=self.cfg.bounds_min,
                    bounds_max=self.cfg.bounds_max,
                    wall_margin=self.cfg.wall_margin,
                    fixed_depth=self.cfg.fixed_depth,
                    step_size=float(self.cfg.fallback_step_size),
                    max_iters=int(self.cfg.fallback_max_iters),
                    goal_sample_rate=float(self.cfg.fallback_goal_sample_rate),
                    neighbor_radius=float(self.cfg.fallback_neighbor_radius),
                    goal_tolerance=float(self.cfg.goal_tolerance),
                    shortcut_enable=bool(self.cfg.shortcut_enable),
                    shortcut_trials=int(self.cfg.shortcut_trials),
                    resample_enable=bool(self.cfg.resample_enable),
                    resample_step=float(self.cfg.resample_step),
                    use_occ_grid=True,
                    occ_threshold=float(self.cfg.occ_threshold),
                    occ_step=self.cfg.occ_step,
                    occ_clearance=float(max(0.0, self.cfg.min_clearance)),
                )
            )

    def plan(
        self,
        start_w: Sequence[float],
        goal_w: Sequence[float],
        obstacles: Sequence[Obstacle] | None = None,
        occ_grid: "list[list[float]] | None" = None,
        occ_bounds_xy: Tuple[float, float, float, float] | None = None,
    ) -> List[Tuple[float, float, float]]:
        obstacles = list(obstacles or [])
        start = self._clamp_to_bounds(
            (float(start_w[0]), float(start_w[1]), float(start_w[2]))
        )
        if self.cfg.fixed_depth is not None:
            goal = self._clamp_to_bounds(
                (float(goal_w[0]), float(goal_w[1]), float(self.cfg.fixed_depth))
            )
        else:
            goal = self._clamp_to_bounds(
                (float(goal_w[0]), float(goal_w[1]), float(goal_w[2]))
            )

        # If already near goal, return direct tiny path.
        if self._dist3(start, goal) <= self.cfg.goal_tolerance:
            return [start, goal]

        blocked, x_min, x_max, y_min, y_max, cell_x, cell_y = self._build_occupancy_grid(
            obstacles=obstacles, occ_grid=occ_grid, occ_bounds_xy=occ_bounds_xy
        )
        h, w = blocked.shape
        if h < 2 or w < 2:
            return self._fallback_plan(start, goal, obstacles, occ_grid, occ_bounds_xy)

        s_idx = self._xy_to_idx(start[0], start[1], x_min, x_max, y_min, y_max, w, h)
        g_idx = self._xy_to_idx(goal[0], goal[1], x_min, x_max, y_min, y_max, w, h)
        if s_idx is None or g_idx is None:
            return self._fallback_plan(start, goal, obstacles, occ_grid, occ_bounds_xy)
        s_idx = self._nearest_free(blocked, s_idx)
        g_idx = self._nearest_free(blocked, g_idx)
        if s_idx is None or g_idx is None:
            return self._fallback_plan(start, goal, obstacles, occ_grid, occ_bounds_xy)

        clearance = self._compute_clearance_field(
            blocked=blocked,
            x_min=x_min,
            x_max=x_max,
            y_min=y_min,
            y_max=y_max,
            cell_x=cell_x,
            cell_y=cell_y,
        )

        blocked_eff = blocked.copy()
        min_clear = float(max(0.0, self.cfg.min_clearance))
        if min_clear > 0.0:
            blocked_eff = blocked_eff | (clearance < min_clear)
            s_idx = self._nearest_free(blocked_eff, s_idx)
            g_idx = self._nearest_free(blocked_eff, g_idx)
            if s_idx is None or g_idx is None:
                return self._fallback_plan(start, goal, obstacles, occ_grid, occ_bounds_xy)

        path_idx = self._astar(blocked_eff, clearance, s_idx, g_idx, cell_x, cell_y)
        if not path_idx:
            return self._fallback_plan(start, goal, obstacles, occ_grid, occ_bounds_xy)

        path_xy = [
            self._idx_to_xy(ix, iy, x_min, x_max, y_min, y_max, w, h) for iy, ix in path_idx
        ]
        path = self._attach_depth(path_xy, start[2], goal[2])
        if path and path[0] != start:
            path.insert(0, start)
        if path and path[-1] != goal:
            path.append(goal)

        if self.cfg.shortcut_enable and len(path) >= 3:
            # Shortcutting must respect the inflated occupancy used for planning.
            # Using the raw blocked grid can cut a high-clearance A* path back
            # through cells that were excluded by min_clearance.
            path = self._shortcut_path(path, obstacles, blocked_eff, x_min, x_max, y_min, y_max)
        if self.cfg.resample_enable and len(path) >= 2:
            path = self._resample_path(path)
        return path

    def _build_occupancy_grid(
        self,
        obstacles: Sequence[Obstacle],
        occ_grid: "list[list[float]] | None",
        occ_bounds_xy: Tuple[float, float, float, float] | None,
    ) -> Tuple[np.ndarray, float, float, float, float, float, float]:
        if occ_grid is not None and occ_bounds_xy is not None:
            occ = np.asarray(occ_grid, dtype=np.float32)
            if occ.ndim != 2:
                raise ValueError("occ_grid must be 2D")
            blocked = occ >= float(self.cfg.occ_threshold)
            x_min, x_max, y_min, y_max = [float(v) for v in occ_bounds_xy]
            h, w = blocked.shape
            cell_x = (x_max - x_min) / max(w - 1, 1)
            cell_y = (y_max - y_min) / max(h - 1, 1)
        else:
            (x_min, y_min, _), (x_max, y_max, _) = self._bounds_with_margin()
            res = max(0.02, float(self.cfg.grid_resolution))
            w = int(max(2, round((x_max - x_min) / res) + 1))
            h = int(max(2, round((y_max - y_min) / res) + 1))
            blocked = np.zeros((h, w), dtype=bool)
            cell_x = (x_max - x_min) / max(w - 1, 1)
            cell_y = (y_max - y_min) / max(h - 1, 1)

        # Rasterize obstacle circles.
        for obs in obstacles:
            cx, cy, _ = obs.position_w
            r = max(0.0, float(obs.radius))
            cidx = self._xy_to_idx(cx, cy, x_min, x_max, y_min, y_max, w, h)
            if cidx is None:
                continue
            ciy, cix = cidx
            rad_x = int(math.ceil(r / max(cell_x, 1e-6)))
            rad_y = int(math.ceil(r / max(cell_y, 1e-6)))
            y0 = max(0, ciy - rad_y)
            y1 = min(h - 1, ciy + rad_y)
            x0 = max(0, cix - rad_x)
            x1 = min(w - 1, cix + rad_x)
            for iy in range(y0, y1 + 1):
                for ix in range(x0, x1 + 1):
                    x, y = self._idx_to_xy(ix, iy, x_min, x_max, y_min, y_max, w, h)
                    if (x - cx) * (x - cx) + (y - cy) * (y - cy) <= r * r:
                        blocked[iy, ix] = True

        return blocked, x_min, x_max, y_min, y_max, cell_x, cell_y

    def _compute_clearance_field(
        self,
        blocked: np.ndarray,
        x_min: float,
        x_max: float,
        y_min: float,
        y_max: float,
        cell_x: float,
        cell_y: float,
    ) -> np.ndarray:
        h, w = blocked.shape
        inf = float("inf")
        d_obs = np.full((h, w), inf, dtype=np.float32)
        pq: list[tuple[float, int, int]] = []

        # Multi-source Dijkstra from occupied cells.
        for iy in range(h):
            for ix in range(w):
                if blocked[iy, ix]:
                    d_obs[iy, ix] = 0.0
                    heappush(pq, (0.0, iy, ix))

        neighbors = [
            (-1, 0, cell_y),
            (1, 0, cell_y),
            (0, -1, cell_x),
            (0, 1, cell_x),
            (-1, -1, math.sqrt(cell_x * cell_x + cell_y * cell_y)),
            (-1, 1, math.sqrt(cell_x * cell_x + cell_y * cell_y)),
            (1, -1, math.sqrt(cell_x * cell_x + cell_y * cell_y)),
            (1, 1, math.sqrt(cell_x * cell_x + cell_y * cell_y)),
        ]
        while pq:
            d, iy, ix = heappop(pq)
            if d > d_obs[iy, ix]:
                continue
            for dy, dx, wgt in neighbors:
                ny = iy + dy
                nx = ix + dx
                if ny < 0 or ny >= h or nx < 0 or nx >= w:
                    continue
                nd = d + wgt
                if nd < d_obs[ny, nx]:
                    d_obs[ny, nx] = nd
                    heappush(pq, (nd, ny, nx))

        # Boundary clearance (distance to inner map bounds).
        d_bnd = np.zeros((h, w), dtype=np.float32)
        for iy in range(h):
            for ix in range(w):
                x, y = self._idx_to_xy(ix, iy, x_min, x_max, y_min, y_max, w, h)
                d_bnd[iy, ix] = float(
                    min(
                        x - x_min,
                        x_max - x,
                        y - y_min,
                        y_max - y,
                    )
                )

        d_obs = np.where(np.isfinite(d_obs), d_obs, 1e6).astype(np.float32)
        clearance = np.minimum(d_obs, d_bnd)
        clearance[blocked] = 0.0
        return clearance

    def _astar(
        self,
        blocked: np.ndarray,
        clearance: np.ndarray,
        start: Tuple[int, int],
        goal: Tuple[int, int],
        cell_x: float,
        cell_y: float,
    ) -> list[Tuple[int, int]]:
        h, w = blocked.shape
        sy, sx = start
        gy, gx = goal
        if blocked[sy, sx] or blocked[gy, gx]:
            return []

        g_cost = np.full((h, w), float("inf"), dtype=np.float64)
        came_y = np.full((h, w), -1, dtype=np.int32)
        came_x = np.full((h, w), -1, dtype=np.int32)
        visited = np.zeros((h, w), dtype=bool)
        g_cost[sy, sx] = 0.0

        def _heur(ix: int, iy: int) -> float:
            dx = (gx - ix) * cell_x
            dy = (gy - iy) * cell_y
            return math.sqrt(dx * dx + dy * dy)

        pq: list[tuple[float, float, int, int]] = []
        heappush(pq, (_heur(sx, sy), 0.0, sy, sx))

        neighbors = [
            (-1, 0, cell_y),
            (1, 0, cell_y),
            (0, -1, cell_x),
            (0, 1, cell_x),
            (-1, -1, math.sqrt(cell_x * cell_x + cell_y * cell_y)),
            (-1, 1, math.sqrt(cell_x * cell_x + cell_y * cell_y)),
            (1, -1, math.sqrt(cell_x * cell_x + cell_y * cell_y)),
            (1, 1, math.sqrt(cell_x * cell_x + cell_y * cell_y)),
        ]
        eps = max(1e-4, float(self.cfg.clearance_eps))

        while pq:
            _, g_now, iy, ix = heappop(pq)
            if visited[iy, ix]:
                continue
            visited[iy, ix] = True
            if iy == gy and ix == gx:
                break
            if g_now > g_cost[iy, ix]:
                continue
            for dy, dx, dl in neighbors:
                ny = iy + dy
                nx = ix + dx
                if ny < 0 or ny >= h or nx < 0 or nx >= w:
                    continue
                if blocked[ny, nx]:
                    continue
                c = max(eps, 0.5 * (float(clearance[iy, ix]) + float(clearance[ny, nx])))
                ng = g_now + (dl / c)
                if ng < g_cost[ny, nx]:
                    g_cost[ny, nx] = ng
                    came_y[ny, nx] = iy
                    came_x[ny, nx] = ix
                    nf = ng + _heur(nx, ny)
                    heappush(pq, (nf, ng, ny, nx))

        if not math.isfinite(float(g_cost[gy, gx])):
            return []

        path: list[Tuple[int, int]] = []
        cy, cx = gy, gx
        while True:
            path.append((cy, cx))
            if cy == sy and cx == sx:
                break
            py = int(came_y[cy, cx])
            px = int(came_x[cy, cx])
            if py < 0 or px < 0:
                return []
            cy, cx = py, px
        path.reverse()
        return path

    def _attach_depth(
        self,
        path_xy: list[Tuple[float, float]],
        z_start: float,
        z_goal: float,
    ) -> list[Tuple[float, float, float]]:
        if not path_xy:
            return []
        if self.cfg.fixed_depth is not None:
            z_const = float(self.cfg.fixed_depth)
            return [(x, y, z_const) for x, y in path_xy]

        # Linear z interpolation by arclength.
        s = [0.0]
        for i in range(1, len(path_xy)):
            dx = path_xy[i][0] - path_xy[i - 1][0]
            dy = path_xy[i][1] - path_xy[i - 1][1]
            s.append(s[-1] + math.sqrt(dx * dx + dy * dy))
        total = max(1e-6, s[-1])
        out = []
        for i, (x, y) in enumerate(path_xy):
            t = s[i] / total
            z = z_start + (z_goal - z_start) * t
            out.append((x, y, float(z)))
        return out

    def _shortcut_path(
        self,
        path: list[Tuple[float, float, float]],
        obstacles: Sequence[Obstacle],
        blocked: np.ndarray,
        x_min: float,
        x_max: float,
        y_min: float,
        y_max: float,
    ) -> list[Tuple[float, float, float]]:
        p = list(path)
        if len(p) < 3:
            return p
        trials = max(0, int(self.cfg.shortcut_trials))
        for _ in range(trials):
            if len(p) < 3:
                break
            i = random.randint(0, len(p) - 3)
            j = random.randint(i + 2, len(p) - 1)
            if self._collision_free(p[i], p[j], obstacles, blocked, x_min, x_max, y_min, y_max):
                p = p[: i + 1] + p[j:]
        return p

    def _resample_path(self, path: list[Tuple[float, float, float]]) -> list[Tuple[float, float, float]]:
        step = float(self.cfg.resample_step)
        if step <= 0.0 or len(path) < 2:
            return path
        out = [path[0]]
        for k in range(len(path) - 1):
            ax, ay, az = out[-1]
            bx, by, bz = path[k + 1]
            dx, dy, dz = bx - ax, by - ay, bz - az
            d = math.sqrt(dx * dx + dy * dy + dz * dz)
            if d <= 1e-9:
                continue
            n = int(max(1, math.floor(d / step)))
            for i in range(1, n + 1):
                t = min(1.0, (i * step) / d)
                out.append((ax + dx * t, ay + dy * t, az + dz * t))
        if out[-1] != path[-1]:
            out[-1] = path[-1]
        return out

    def _collision_free(
        self,
        a: Tuple[float, float, float],
        b: Tuple[float, float, float],
        obstacles: Sequence[Obstacle],
        blocked: np.ndarray,
        x_min: float,
        x_max: float,
        y_min: float,
        y_max: float,
    ) -> bool:
        if not (self._within_bounds(a) and self._within_bounds(b)):
            return False
        if self._segment_hits_occ_grid_xy(a, b, blocked, x_min, x_max, y_min, y_max):
            return False
        for obs in obstacles:
            if self._segment_circle_intersect_xy(a, b, obs.position_w, float(obs.radius)):
                return False
        return True

    def _segment_hits_occ_grid_xy(
        self,
        a: Tuple[float, float, float],
        b: Tuple[float, float, float],
        blocked: np.ndarray,
        x_min: float,
        x_max: float,
        y_min: float,
        y_max: float,
    ) -> bool:
        h, w = blocked.shape
        ax, ay, _ = a
        bx, by, _ = b
        dx = bx - ax
        dy = by - ay
        dist = math.sqrt(dx * dx + dy * dy)
        if dist <= 1e-9:
            return False
        if self.cfg.occ_step is not None and self.cfg.occ_step > 0.0:
            step = float(self.cfg.occ_step)
        else:
            cell_x = (x_max - x_min) / max(w - 1, 1)
            cell_y = (y_max - y_min) / max(h - 1, 1)
            step = float(max(min(cell_x, cell_y), 1e-3))
        n = int(max(2, math.ceil(dist / step) + 1))
        for i in range(n + 1):
            t = i / n
            x = ax + dx * t
            y = ay + dy * t
            idx = self._xy_to_idx(x, y, x_min, x_max, y_min, y_max, w, h)
            if idx is None:
                return True
            iy, ix = idx
            if bool(blocked[iy, ix]):
                return True
        return False

    @staticmethod
    def _segment_circle_intersect_xy(
        a: Tuple[float, float, float],
        b: Tuple[float, float, float],
        center: Tuple[float, float, float],
        radius: float,
    ) -> bool:
        ax, ay, _ = a
        bx, by, _ = b
        cx, cy, _ = center
        abx, aby = (bx - ax), (by - ay)
        acx, acy = (cx - ax), (cy - ay)
        ab_len2 = abx * abx + aby * aby
        r2 = radius * radius
        if ab_len2 == 0.0:
            dist2 = (ax - cx) ** 2 + (ay - cy) ** 2
            return dist2 <= r2
        t = max(0.0, min(1.0, (acx * abx + acy * aby) / ab_len2))
        closest_x = ax + abx * t
        closest_y = ay + aby * t
        dist2 = (closest_x - cx) ** 2 + (closest_y - cy) ** 2
        return dist2 <= r2

    def _fallback_plan(
        self,
        start: Tuple[float, float, float],
        goal: Tuple[float, float, float],
        obstacles: Sequence[Obstacle],
        occ_grid: "list[list[float]] | None",
        occ_bounds_xy: Tuple[float, float, float, float] | None,
    ) -> list[Tuple[float, float, float]]:
        if self._fallback is None:
            return [start, goal]
        return self._fallback.plan(
            start_w=start,
            goal_w=goal,
            obstacles=obstacles,
            occ_grid=occ_grid,
            occ_bounds_xy=occ_bounds_xy,
        )

    def _bounds_with_margin(self) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
        m = float(self.cfg.wall_margin)
        mn = (self.cfg.bounds_min[0] + m, self.cfg.bounds_min[1] + m, self.cfg.bounds_min[2])
        mx = (self.cfg.bounds_max[0] - m, self.cfg.bounds_max[1] - m, self.cfg.bounds_max[2])
        return mn, mx

    def _within_bounds(self, p: Tuple[float, float, float]) -> bool:
        mn, mx = self._bounds_with_margin()
        return mn[0] <= p[0] <= mx[0] and mn[1] <= p[1] <= mx[1] and mn[2] <= p[2] <= mx[2]

    def _clamp_to_bounds(self, p: Tuple[float, float, float]) -> Tuple[float, float, float]:
        mn, mx = self._bounds_with_margin()
        return (
            min(max(p[0], mn[0]), mx[0]),
            min(max(p[1], mn[1]), mx[1]),
            min(max(p[2], mn[2]), mx[2]),
        )

    @staticmethod
    def _dist3(a: Tuple[float, float, float], b: Tuple[float, float, float]) -> float:
        return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2)

    @staticmethod
    def _xy_to_idx(
        x: float,
        y: float,
        x_min: float,
        x_max: float,
        y_min: float,
        y_max: float,
        w: int,
        h: int,
    ) -> Tuple[int, int] | None:
        if x < x_min or x > x_max or y < y_min or y > y_max:
            return None
        gx = (x - x_min) / max(x_max - x_min, 1e-6)
        gy = (y - y_min) / max(y_max - y_min, 1e-6)
        ix = int(min(max(round(gx * (w - 1)), 0), w - 1))
        iy = int(min(max(round(gy * (h - 1)), 0), h - 1))
        return iy, ix

    @staticmethod
    def _idx_to_xy(
        ix: int,
        iy: int,
        x_min: float,
        x_max: float,
        y_min: float,
        y_max: float,
        w: int,
        h: int,
    ) -> Tuple[float, float]:
        gx = float(ix) / max(w - 1, 1)
        gy = float(iy) / max(h - 1, 1)
        x = x_min + gx * (x_max - x_min)
        y = y_min + gy * (y_max - y_min)
        return x, y

    @staticmethod
    def _nearest_free(blocked: np.ndarray, idx: Tuple[int, int]) -> Tuple[int, int] | None:
        h, w = blocked.shape
        sy, sx = idx
        if sy < 0 or sy >= h or sx < 0 or sx >= w:
            return None
        if not blocked[sy, sx]:
            return sy, sx
        max_r = max(h, w)
        for r in range(1, max_r + 1):
            y0 = max(0, sy - r)
            y1 = min(h - 1, sy + r)
            x0 = max(0, sx - r)
            x1 = min(w - 1, sx + r)
            # Ring scan.
            for ix in range(x0, x1 + 1):
                if not blocked[y0, ix]:
                    return y0, ix
                if not blocked[y1, ix]:
                    return y1, ix
            for iy in range(y0 + 1, y1):
                if not blocked[iy, x0]:
                    return iy, x0
                if not blocked[iy, x1]:
                    return iy, x1
        return None
