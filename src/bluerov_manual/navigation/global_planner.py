from dataclasses import dataclass
from typing import List, Sequence, Tuple

import math
import random

from .obstacles import Obstacle


@dataclass
class GlobalPlannerConfig:
    grid_resolution: float = 0.5
    fixed_depth: float | None = None
    bounds_min: Tuple[float, float, float] = (-10.0, -10.0, -5.0)
    bounds_max: Tuple[float, float, float] = (10.0, 10.0, 5.0)
    wall_margin: float = 0.0
    step_size: float = 1.0
    max_iters: int = 400
    goal_sample_rate: float = 0.1
    neighbor_radius: float = 2.0
    goal_tolerance: float = 1.0
    # Post-processing for smoother, shorter paths.
    shortcut_enable: bool = True
    shortcut_trials: int = 200
    resample_enable: bool = True
    resample_step: float = 0.4
    # Optional occupancy-grid collision checking (2D x/y). If enabled, any segment that passes
    # through occupied cells is rejected, even if no explicit Obstacle circles are provided.
    use_occ_grid: bool = False
    occ_threshold: float = 0.5
    occ_step: float | None = None
    # Extra XY safety margin from occupancy obstacles (meters).
    occ_clearance: float = 0.0


@dataclass
class _Node:
    pos: Tuple[float, float, float]
    parent: int | None
    cost: float


class GlobalPlanner:
    def __init__(self, cfg: GlobalPlannerConfig):
        self.cfg = cfg

    def plan(
        self,
        start_w: Sequence[float],
        goal_w: Sequence[float],
        obstacles: Sequence[Obstacle] | None = None,
        occ_grid: "list[list[float]] | None" = None,
        occ_bounds_xy: Tuple[float, float, float, float] | None = None,
    ) -> List[Tuple[float, float, float]]:
        """Return a coarse list of waypoints in world frame."""
        obstacles = list(obstacles or [])
        start = (float(start_w[0]), float(start_w[1]), float(start_w[2]))
        if self.cfg.fixed_depth is not None:
            goal = (float(goal_w[0]), float(goal_w[1]), float(self.cfg.fixed_depth))
        else:
            goal = (float(goal_w[0]), float(goal_w[1]), float(goal_w[2]))
        start = self._clamp_to_bounds(start)
        goal = self._clamp_to_bounds(goal)

        nodes = [_Node(pos=start, parent=None, cost=0.0)]
        goal_idx = None

        for _ in range(self.cfg.max_iters):
            sample = self._sample(goal)
            nearest_idx = self._nearest(nodes, sample)
            new_pos = self._steer(nodes[nearest_idx].pos, sample)
            if not self._collision_free(nodes[nearest_idx].pos, new_pos, obstacles):
                continue

            new_cost = nodes[nearest_idx].cost + self._dist(nodes[nearest_idx].pos, new_pos)
            new_node = _Node(pos=new_pos, parent=nearest_idx, cost=new_cost)

            neighbor_idxs = self._near(nodes, new_pos)
            best_parent = nearest_idx
            best_cost = new_cost
            for idx in neighbor_idxs:
                cand_cost = nodes[idx].cost + self._dist(nodes[idx].pos, new_pos)
                if cand_cost < best_cost and self._collision_free(nodes[idx].pos, new_pos, obstacles):
                    best_parent = idx
                    best_cost = cand_cost
            new_node.parent = best_parent
            new_node.cost = best_cost
            nodes.append(new_node)
            new_idx = len(nodes) - 1

            for idx in neighbor_idxs:
                cand_cost = new_node.cost + self._dist(nodes[idx].pos, new_pos)
                if cand_cost < nodes[idx].cost and self._collision_free(nodes[idx].pos, new_pos, obstacles):
                    nodes[idx].parent = new_idx
                    nodes[idx].cost = cand_cost

            if self._dist(new_pos, goal) <= self.cfg.goal_tolerance:
                goal_idx = new_idx
                break

        if goal_idx is None:
            # If we fail to directly connect to the goal, return the best-so-far safe path
            # toward the goal (instead of a straight line through obstacles).
            nearest_idx = self._nearest(nodes, goal)
            path = self._extract_path(nodes, nearest_idx, nodes[nearest_idx].pos)
            if self._collision_free(path[-1], goal, obstacles, occ_grid, occ_bounds_xy):
                path = path + [goal]
        else:
            path = self._extract_path(nodes, goal_idx, goal)

        if self.cfg.shortcut_enable and len(path) >= 3:
            path = self._shortcut_path(path, obstacles, occ_grid, occ_bounds_xy)
        if self.cfg.resample_enable and len(path) >= 2:
            path = self._resample_path(path)
        return path

    def _sample(self, goal: Tuple[float, float, float]) -> Tuple[float, float, float]:
        if random.random() < self.cfg.goal_sample_rate:
            return goal
        mn, mx = self._bounds_with_margin()
        return (
            random.uniform(mn[0], mx[0]),
            random.uniform(mn[1], mx[1]),
            random.uniform(mn[2], mx[2]),
        )

    def _nearest(self, nodes: List[_Node], point: Tuple[float, float, float]) -> int:
        dists = [self._dist(n.pos, point) for n in nodes]
        return int(min(range(len(dists)), key=dists.__getitem__))

    def _near(self, nodes: List[_Node], point: Tuple[float, float, float]) -> List[int]:
        return [
            i
            for i, n in enumerate(nodes)
            if self._dist(n.pos, point) <= self.cfg.neighbor_radius
        ]

    def _steer(
        self, from_pos: Tuple[float, float, float], to_pos: Tuple[float, float, float]
    ) -> Tuple[float, float, float]:
        d = self._dist(from_pos, to_pos)
        if d <= self.cfg.step_size:
            return to_pos
        scale = self.cfg.step_size / d
        return (
            from_pos[0] + (to_pos[0] - from_pos[0]) * scale,
            from_pos[1] + (to_pos[1] - from_pos[1]) * scale,
            from_pos[2] + (to_pos[2] - from_pos[2]) * scale,
        )

    @staticmethod
    def _dist(a: Tuple[float, float, float], b: Tuple[float, float, float]) -> float:
        return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2)

    def _collision_free(
        self,
        a: Tuple[float, float, float],
        b: Tuple[float, float, float],
        obstacles: Sequence[Obstacle],
        occ_grid: "list[list[float]] | None" = None,
        occ_bounds_xy: Tuple[float, float, float, float] | None = None,
    ) -> bool:
        if not (self._within_bounds(a) and self._within_bounds(b)):
            return False
        if self.cfg.use_occ_grid and occ_grid is not None and occ_bounds_xy is not None:
            if self._segment_hits_occ_grid_xy(a, b, occ_grid, occ_bounds_xy):
                return False
        for obs in obstacles:
            # Obstacles in the tank are effectively vertical cylinders; use a conservative
            # 2D (x/y) segment-circle intersection so we don't "thread" through them by
            # exploiting z differences.
            if self._segment_circle_intersect_xy(a, b, obs.position_w, obs.radius):
                return False
        return True

    def _bounds_with_margin(self) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
        m = float(self.cfg.wall_margin)
        # Apply wall margin in x/y only. Z bounds are controlled directly via bounds_min/max.
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

    def _extract_path(
        self, nodes: List[_Node], goal_idx: int, goal: Tuple[float, float, float]
    ) -> List[Tuple[float, float, float]]:
        path = [goal]
        idx = goal_idx
        while idx is not None:
            node = nodes[idx]
            path.append(node.pos)
            idx = node.parent
        path.reverse()
        return path

    def _shortcut_path(
        self,
        path: List[Tuple[float, float, float]],
        obstacles: Sequence[Obstacle],
        occ_grid: "list[list[float]] | None" = None,
        occ_bounds_xy: Tuple[float, float, float, float] | None = None,
    ) -> List[Tuple[float, float, float]]:
        """Randomized shortcutting: removes unnecessary waypoints while staying collision-free."""
        if len(path) < 3:
            return path
        import random

        p = list(path)
        trials = max(0, int(self.cfg.shortcut_trials))
        for _ in range(trials):
            if len(p) < 3:
                break
            i = random.randint(0, len(p) - 3)
            j = random.randint(i + 2, len(p) - 1)
            a, b = p[i], p[j]
            if self._collision_free(a, b, obstacles, occ_grid, occ_bounds_xy):
                # Remove intermediate points.
                p = p[: i + 1] + p[j:]
        return p

    def _resample_path(self, path: List[Tuple[float, float, float]]) -> List[Tuple[float, float, float]]:
        """Densify the path with roughly fixed spacing for smoother downstream tracking."""
        step = float(self.cfg.resample_step)
        if step <= 0.0 or len(path) < 2:
            return path
        out: List[Tuple[float, float, float]] = [path[0]]
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
        # Ensure exact final waypoint.
        if out[-1] != path[-1]:
            out[-1] = path[-1]
        return out

    def _segment_hits_occ_grid_xy(
        self,
        a: Tuple[float, float, float],
        b: Tuple[float, float, float],
        occ_grid: "list[list[float]]",
        occ_bounds_xy: Tuple[float, float, float, float],
    ) -> bool:
        """Check whether the segment crosses occupied cells in a 2D occupancy grid."""
        x_min, x_max, y_min, y_max = occ_bounds_xy
        h = len(occ_grid)
        w = len(occ_grid[0]) if h > 0 else 0
        if h == 0 or w == 0:
            return False

        ax, ay, _ = a
        bx, by, _ = b
        dx = bx - ax
        dy = by - ay
        dist = math.sqrt(dx * dx + dy * dy)
        if dist <= 1e-9:
            return False

        # Step defaults to ~1 cell.
        if self.cfg.occ_step is not None and self.cfg.occ_step > 0.0:
            step = float(self.cfg.occ_step)
        else:
            cell_x = (x_max - x_min) / max(w - 1, 1)
            cell_y = (y_max - y_min) / max(h - 1, 1)
            step = float(max(min(cell_x, cell_y), 1e-3))

        n = int(max(2, math.ceil(dist / step) + 1))
        thresh = float(self.cfg.occ_threshold)
        cell_x = (x_max - x_min) / max(w - 1, 1)
        cell_y = (y_max - y_min) / max(h - 1, 1)
        occ_clear = float(max(0.0, self.cfg.occ_clearance))
        rad_ix = int(math.ceil(occ_clear / max(cell_x, 1e-6))) if occ_clear > 0.0 else 0
        rad_iy = int(math.ceil(occ_clear / max(cell_y, 1e-6))) if occ_clear > 0.0 else 0

        for i in range(n + 1):
            t = i / n
            x = ax + dx * t
            y = ay + dy * t
            gx = (x - x_min) / max(x_max - x_min, 1e-6)
            gy = (y - y_min) / max(y_max - y_min, 1e-6)
            ix = int(min(max(round(gx * (w - 1)), 0), w - 1))
            iy = int(min(max(round(gy * (h - 1)), 0), h - 1))
            if rad_ix == 0 and rad_iy == 0:
                if float(occ_grid[iy][ix]) >= thresh:
                    return True
                continue
            y0 = max(0, iy - rad_iy)
            y1 = min(h - 1, iy + rad_iy)
            x0 = max(0, ix - rad_ix)
            x1 = min(w - 1, ix + rad_ix)
            hit = False
            for ny in range(y0, y1 + 1):
                dy_m = abs(ny - iy) * cell_y
                for nx in range(x0, x1 + 1):
                    dx_m = abs(nx - ix) * cell_x
                    if (dx_m * dx_m + dy_m * dy_m) > (occ_clear * occ_clear):
                        continue
                    if float(occ_grid[ny][nx]) >= thresh:
                        hit = True
                        break
                if hit:
                    break
            if hit:
                return True
        return False
