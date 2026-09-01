"""Run BlueROV with a robust behavior-tree controller over maze mapping + RRT (no RL checkpoint)."""

from __future__ import annotations

import collections
import csv
import json
import math
import os
import random
import sys
from dataclasses import dataclass

import hydra
import numpy as np
import torch
from omegaconf import OmegaConf

# Isaac Sim bundled torch can miss ForkingPickler required by tensordict/torchrl.
try:
    import torch.multiprocessing.reductions as _torch_reductions

    if not hasattr(_torch_reductions, "ForkingPickler"):
        from multiprocessing.reduction import ForkingPickler as _MPForkingPickler

        _torch_reductions.ForkingPickler = _MPForkingPickler
except Exception:
    pass

# Make local package importable.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))


def _append_summary_csv(csv_path: str, row: dict) -> None:
    if not csv_path:
        return
    out_dir = os.path.dirname(os.path.abspath(csv_path))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    fields = [
        "label",
        "algo",
        "checkpoint_path",
        "attempts",
        "success_rate",
        "mean_reward",
        "mean_final_dist",
        "mean_episode_length",
        "mean_time_sec",
        "mean_time_to_success_sec",
        "success_count",
        "rate_denominator",
        "rate_denominator_count",
        "excluded_non_outcome_count",
        "collision_count",
        "collision_rate",
        "ttc_collision_count",
        "ttc_collision_rate",
        "collision_any_count",
        "collision_any_rate",
        "mean_policy_latency_ms",
        "mean_map_latency_ms",
        "mean_planner_latency_ms",
        "mean_obs_latency_ms",
        "mean_step_latency_ms",
        "planner_replans_total",
        "planner_replans_per_episode",
        "termination_reasons",
    ]
    exists = os.path.isfile(csv_path)
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        if not exists:
            writer.writeheader()
        writer.writerow({k: row.get(k, "") for k in fields})


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _quat_rotate_inverse(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Rotate vectors v from world to body frame using quaternion q=(w,x,y,z)."""
    q_w = q[..., 0:1]
    q_vec = q[..., 1:4]
    a = v * (2.0 * q_w * q_w - 1.0)
    b = torch.cross(q_vec, v, dim=-1) * (2.0 * q_w)
    c = q_vec * (2.0 * torch.sum(q_vec * v, dim=-1, keepdim=True))
    return a - b + c


@dataclass
class EnvState:
    rel_b: torch.Tensor
    yaw_err: float
    vel_b: torch.Tensor
    yaw_rate: float
    depth: float
    roll: float
    pitch: float
    min_range: float
    front_min: float
    left_min: float
    right_min: float

    @property
    def goal_dist(self) -> float:
        return float(torch.norm(self.rel_b).item())


@dataclass
class DynamicRisk:
    static_count: int
    dynamic_count: int
    mean_dynamic_speed: float
    min_static_dist: float
    min_dynamic_dist: float
    min_ttc: float
    ttc_obstacle_idx: int
    imminent: bool
    approaching: bool
    turn_preference: float
    heave_preference: float
    proceed_safe: bool
    speed_scale: float
    crossing_preference: float
    source: str


class RobustMazeBTController:
    # hover_tank_ppo observation layout starts with this 12D block:
    # [rel_b(3), yaw_err_sin/cos(2), vel_b(3), yaw_rate(1), depth(1), roll_pitch(2), ...]
    IDX_REL = slice(0, 3)
    IDX_YAW_SIN = 3
    IDX_YAW_COS = 4
    IDX_VEL = slice(5, 8)
    IDX_YAW_RATE = 8
    IDX_DEPTH = 9
    IDX_ROLL = 10
    IDX_PITCH = 11

    def __init__(self, cfg, env):
        self.cfg = cfg
        self.env = env

        self.goal_tolerance = float(cfg.get("goal_tolerance", 0.7))
        self.align_yaw_threshold = float(cfg.get("align_yaw_threshold", 0.45))
        self.align_yaw_threshold_far = float(cfg.get("align_yaw_threshold_far", 1.10))
        self.align_relax_far_dist = float(cfg.get("align_relax_far_dist", 3.0))
        # Waypoint-local "close" threshold (for diagnostics only); final stopping uses
        # final-goal distance from env target_pos, not waypoint distance.
        self.waypoint_close_threshold = float(cfg.get("waypoint_close_threshold", 0.35))

        self.stop_dist = float(cfg.get("stop_dist", 0.40))
        self.avoid_dist = float(cfg.get("avoid_dist", 0.80))
        self.replan_dist = float(cfg.get("replan_dist", 0.95))
        # Front clearance should drive "afraid to move" behavior. Side clearance
        # only becomes a blocker when it is genuinely close; otherwise corridors
        # with nearby side walls look blocked even when the forward gap is clear.
        self.side_block_dist = float(cfg.get("side_block_dist", max(0.18, self.stop_dist)))

        self.k_pos_xy = float(cfg.get("k_pos_xy", 0.55))
        self.k_pos_z = float(cfg.get("k_pos_z", 0.50))
        self.k_vel = float(cfg.get("k_vel", 0.45))
        self.k_yaw = float(cfg.get("k_yaw", 1.40))
        self.k_yaw_rate = float(cfg.get("k_yaw_rate", 0.35))

        self.max_fwd = float(cfg.get("max_fwd", 0.35))
        self.max_lat = float(cfg.get("max_lat", 0.25))
        self.max_heave = float(cfg.get("max_heave", 0.20))
        self.max_yaw = float(cfg.get("max_yaw", 0.35))

        self.align_fwd_scale = float(cfg.get("align_fwd_scale", 0.35))
        self.stop_backoff = float(cfg.get("stop_backoff", 0.08))
        self.avoid_fwd = float(cfg.get("avoid_fwd", 0.16))
        self.avoid_yaw = float(cfg.get("avoid_yaw", 0.35))
        self.force_replan_on_avoid = bool(cfg.get("force_replan_on_avoid", True))
        self.min_replan_interval_steps = int(cfg.get("min_replan_interval_steps", 50))
        self.replan_consecutive_trigger_steps = int(cfg.get("replan_consecutive_trigger_steps", 8))
        self.replan_live_debug = bool(cfg.get("replan_live_debug", True))

        # Dynamic obstacle understanding (classification + TTC gating).
        self.dynamic_enable = bool(cfg.get("dynamic_enable", True))
        self.dynamic_from_observation_only = bool(cfg.get("dynamic_from_observation_only", True))
        self.dynamic_speed_threshold = float(cfg.get("dynamic_speed_threshold", 0.06))
        self.dynamic_speed_hysteresis = float(cfg.get("dynamic_speed_hysteresis", 0.02))
        self.dynamic_vel_ema = float(cfg.get("dynamic_vel_ema", 0.35))
        self.ttc_horizon = float(cfg.get("ttc_horizon", 3.0))
        self.ttc_stop = float(cfg.get("ttc_stop", 0.9))
        self.ttc_replan = float(cfg.get("ttc_replan", 1.6))
        self.dynamic_cpa_margin = float(cfg.get("dynamic_cpa_margin", 0.12))
        self.dynamic_proximity_trigger = float(cfg.get("dynamic_proximity_trigger", 0.85))
        self.closing_rate_thresh = float(cfg.get("closing_rate_thresh", 0.08))
        self.side_rate_margin = float(cfg.get("side_rate_margin", 0.04))
        self.robot_radius = float(cfg.get("robot_radius", 0.18))
        self.dynamic_safety_margin = float(cfg.get("dynamic_safety_margin", 0.08))
        self.predictive_avoid_fwd = float(cfg.get("predictive_avoid_fwd", 0.14))
        self.predictive_lateral_gain = float(cfg.get("predictive_lateral_gain", 0.40))
        self.predictive_heave_gain = float(cfg.get("predictive_heave_gain", 0.25))
        self.dynamic_go_enable = bool(cfg.get("dynamic_go_enable", True))
        self.dynamic_go_ttc = float(cfg.get("dynamic_go_ttc", 2.2))
        self.dynamic_yield_ttc = float(cfg.get("dynamic_yield_ttc", 1.0))
        self.dynamic_go_front_dist = float(cfg.get("dynamic_go_front_dist", 0.55))
        self.dynamic_slow_front_dist = float(cfg.get("dynamic_slow_front_dist", 1.15))
        self.dynamic_go_speed_scale = float(cfg.get("dynamic_go_speed_scale", 0.62))
        self.dynamic_corridor_half_width = float(cfg.get("dynamic_corridor_half_width", 0.42))
        self.dynamic_crossing_margin = float(cfg.get("dynamic_crossing_margin", 0.10))
        self.dynamic_crossing_avoid_enable = bool(cfg.get("dynamic_crossing_avoid_enable", True))
        self.dynamic_crossing_speed_thresh = float(cfg.get("dynamic_crossing_speed_thresh", 0.04))
        self.dynamic_crossing_avoid_fwd = float(cfg.get("dynamic_crossing_avoid_fwd", 0.04))
        self.dynamic_crossing_avoid_lat = float(cfg.get("dynamic_crossing_avoid_lat", 0.25))
        self.dynamic_crossing_avoid_yaw = float(cfg.get("dynamic_crossing_avoid_yaw", 0.28))
        # Crowded dynamic-obstacle recovery: back off briefly, then resume forward progress.
        self.crowd_backup_enable = bool(cfg.get("crowd_backup_enable", True))
        self.crowd_dyn_min = int(cfg.get("crowd_dyn_min", 2))
        self.crowd_front_dist = float(cfg.get("crowd_front_dist", 0.95))
        self.crowd_ttc_trigger = float(cfg.get("crowd_ttc_trigger", 1.4))
        self.crowd_backup_steps = int(cfg.get("crowd_backup_steps", 14))
        self.crowd_backoff_u = float(cfg.get("crowd_backoff_u", 0.18))
        self.crowd_backoff_v = float(cfg.get("crowd_backoff_v", 0.12))
        self.crowd_backoff_yaw = float(cfg.get("crowd_backoff_yaw", 0.55))
        self.escape_enable = bool(cfg.get("escape_enable", True))
        self.escape_steps = int(cfg.get("escape_steps", 34))
        self.escape_no_progress_steps = int(cfg.get("escape_no_progress_steps", 42))
        self.escape_side_switches = int(cfg.get("escape_side_switches", 3))
        self.escape_front_dist = float(cfg.get("escape_front_dist", 1.20))
        self.escape_min_dist = float(cfg.get("escape_min_dist", 0.85))
        self.escape_backoff_u = float(cfg.get("escape_backoff_u", 0.24))
        self.escape_lateral_v = float(cfg.get("escape_lateral_v", 0.22))
        self.escape_yaw = float(cfg.get("escape_yaw", 0.60))
        self.escape_heave = float(cfg.get("escape_heave", 0.10))
        self.stale_wp_dist = float(cfg.get("stale_wp_dist", 0.08))
        self.stale_wp_margin = float(cfg.get("stale_wp_margin", 0.35))
        self.no_progress_epsilon = float(cfg.get("no_progress_epsilon", 0.002))
        self.no_progress_steps_replan = int(cfg.get("no_progress_steps_replan", 55))
        self.no_progress_steps_force = int(cfg.get("no_progress_steps_force", 25))
        # Stall recovery: if progress is stuck for long, explicitly recover instead of repeating move+replan.
        self.stall_recovery_enable = bool(cfg.get("stall_recovery_enable", True))
        self.stall_recovery_steps = int(cfg.get("stall_recovery_steps", 30))
        self.stall_recovery_front_dist = float(cfg.get("stall_recovery_front_dist", 1.20))
        self.stall_recovery_align_yaw = float(cfg.get("stall_recovery_align_yaw", 0.75))
        self.progress_commit_enable = bool(cfg.get("progress_commit_enable", True))
        self.progress_commit_steps = int(cfg.get("progress_commit_steps", 85))
        self.progress_commit_front_min = float(cfg.get("progress_commit_front_min", 1.15))
        self.progress_commit_min_range = float(cfg.get("progress_commit_min_range", 0.70))
        self.progress_commit_yaw_err = float(cfg.get("progress_commit_yaw_err", 1.25))
        self.progress_commit_speed_scale = float(cfg.get("progress_commit_speed_scale", 0.70))
        self.progress_commit_dynamic_ttc_min = float(cfg.get("progress_commit_dynamic_ttc_min", 2.50))
        self.goal_closeout_enable = bool(cfg.get("goal_closeout_enable", True))
        self.goal_closeout_dist = float(cfg.get("goal_closeout_dist", 2.50))
        self.goal_closeout_front_min = float(cfg.get("goal_closeout_front_min", 0.85))
        self.goal_closeout_min_range = float(cfg.get("goal_closeout_min_range", 0.55))
        self.goal_closeout_yaw_err = float(cfg.get("goal_closeout_yaw_err", 1.45))
        self.goal_closeout_speed_scale = float(cfg.get("goal_closeout_speed_scale", 0.85))
        self.goal_closeout_dynamic_ttc_min = float(cfg.get("goal_closeout_dynamic_ttc_min", 1.80))
        # Smoother tracking along global path.
        self.path_lookahead_points = int(cfg.get("path_lookahead_points", 2))
        self.path_priority_enable = bool(cfg.get("path_priority_enable", True))
        self.path_only_mode = bool(cfg.get("path_only_mode", False))
        self.path_prefer_map_overlay = bool(cfg.get("path_prefer_map_overlay", True))
        self.path_lookahead_dist = float(cfg.get("path_lookahead_dist", 0.75))
        self.path_lateral_gain = float(cfg.get("path_lateral_gain", 1.25))
        self.path_yaw_gain = float(cfg.get("path_yaw_gain", 1.15))
        self.path_lateral_slow_gain = float(cfg.get("path_lateral_slow_gain", 1.10))
        self.path_yaw_slow_gain = float(cfg.get("path_yaw_slow_gain", 0.65))
        self.path_min_u_scale = float(cfg.get("path_min_u_scale", 0.20))
        self.corridor_commit_enable = bool(cfg.get("corridor_commit_enable", True))
        self.corridor_commit_front_min = float(cfg.get("corridor_commit_front_min", 0.75))
        self.corridor_commit_lateral_err = float(cfg.get("corridor_commit_lateral_err", 0.35))
        self.corridor_commit_yaw_err = float(cfg.get("corridor_commit_yaw_err", 0.45))
        self.corridor_commit_min_speed_scale = float(cfg.get("corridor_commit_min_speed_scale", 0.45))
        self.cmd_smooth_alpha = float(cfg.get("cmd_smooth_alpha", 0.35))
        self.cmd_max_delta = float(cfg.get("cmd_max_delta", 0.10))
        # Yaw-priority: when heading error is large, reduce translational drive and rotate first.
        self.yaw_priority_thresh = float(cfg.get("yaw_priority_thresh", 0.55))
        self.yaw_priority_u_scale = float(cfg.get("yaw_priority_u_scale", 0.55))
        self.yaw_priority_v_scale = float(cfg.get("yaw_priority_v_scale", 0.20))
        self.yaw_priority_r_boost = float(cfg.get("yaw_priority_r_boost", 1.35))
        # Simpler staged policy:
        # 1) align yaw to waypoint, 2) surge-dominant move, 3) local avoid (sway/heave/yaw), then back to move.
        self.simple_mode = bool(cfg.get("simple_mode", False))
        self.simple_use_dynamic = bool(cfg.get("simple_use_dynamic", False))
        self.simple_align_yaw_threshold = float(cfg.get("simple_align_yaw_threshold", 0.35))
        self.simple_move_sway_gain = float(cfg.get("simple_move_sway_gain", 0.35))
        self.simple_move_yaw_gain = float(cfg.get("simple_move_yaw_gain", 0.65))
        self.simple_avoid_lat = float(cfg.get("simple_avoid_lat", 0.25))
        self.simple_avoid_heave = float(cfg.get("simple_avoid_heave", 0.12))
        self.simple_avoid_yaw = float(cfg.get("simple_avoid_yaw", 0.50))
        self.simple_avoid_fwd = float(cfg.get("simple_avoid_fwd", 0.10))
        self.simple_avoid_latch_steps = int(cfg.get("simple_avoid_latch_steps", 20))

        # Convert velocity-domain command to normalized policy action [-1, 1].
        ctrl_cfg = getattr(env, "ctrl_cfg", {}) or {}
        self.u_max = float(cfg.get("u_max", ctrl_cfg.get("u_max", 0.8)))
        self.v_max = float(cfg.get("v_max", ctrl_cfg.get("v_max", 0.5)))
        self.w_max = float(cfg.get("w_max", ctrl_cfg.get("w_max", 0.3)))
        self.r_max = float(cfg.get("r_max", ctrl_cfg.get("r_max", 0.35)))

        sonar_cfg = getattr(env, "sonar_cfg", {}) or {}
        self.num_rays = int(cfg.get("num_rays", sonar_cfg.get("num_rays", 64)))

        front_half_angle_deg = float(cfg.get("front_half_angle_deg", 55.0))
        side_sector_min_deg = float(cfg.get("side_sector_min_deg", 25.0))
        side_sector_max_deg = float(cfg.get("side_sector_max_deg", 115.0))

        self.angles = self._build_angles(env, self.num_rays)
        abs_angles_deg = self.angles.abs() * (180.0 / math.pi)
        self.front_mask = abs_angles_deg <= front_half_angle_deg
        self.left_mask = (self.angles > 0.0) & (abs_angles_deg >= side_sector_min_deg) & (abs_angles_deg <= side_sector_max_deg)
        self.right_mask = (self.angles < 0.0) & (abs_angles_deg >= side_sector_min_deg) & (abs_angles_deg <= side_sector_max_deg)

        self._replan_env_ids: list[int] = []
        self._last_debug: list[dict] = []
        self._step_count = 0
        self._last_replan_step: torch.Tensor | None = None
        self._replan_pressure: torch.Tensor | None = None
        self._prev_obst_pos_w: torch.Tensor | None = None
        self._obst_vel_w: torch.Tensor | None = None
        self._obst_is_dynamic: torch.Tensor | None = None
        self._dt = float(max(1e-4, getattr(env, "dt", 0.02)))
        self._prev_front_min: torch.Tensor | None = None
        self._prev_left_min: torch.Tensor | None = None
        self._prev_right_min: torch.Tensor | None = None
        self._front_rate_ema: torch.Tensor | None = None
        self._left_rate_ema: torch.Tensor | None = None
        self._right_rate_ema: torch.Tensor | None = None
        self.dynamic_obs_est_speed_thresh = float(cfg.get("dynamic_obs_est_speed_thresh", 0.08))
        self.dynamic_obs_est_min_rays = int(cfg.get("dynamic_obs_est_min_rays", 2))
        self.dynamic_obs_est_max_markers = int(cfg.get("dynamic_obs_est_max_markers", 8))
        self.dynamic_obs_est_center_offset_radius = float(cfg.get("dynamic_obs_est_center_offset_radius", 0.15))
        self.dynamic_obs_est_assoc_dist = float(cfg.get("dynamic_obs_est_assoc_dist", 0.75))
        self.dynamic_obs_est_pos_ema = float(cfg.get("dynamic_obs_est_pos_ema", 0.35))
        self.dynamic_obs_est_vel_ema = float(cfg.get("dynamic_obs_est_vel_ema", 0.30))
        self.dynamic_obs_est_ttl_steps = int(cfg.get("dynamic_obs_est_ttl_steps", 12))
        self.dynamic_obs_est_marker_min_speed = float(cfg.get("dynamic_obs_est_marker_min_speed", 0.04))
        self.dynamic_obs_est_map_threshold = float(cfg.get("dynamic_obs_est_map_threshold", 0.20))
        self.dynamic_obs_est_min_cells = int(cfg.get("dynamic_obs_est_min_cells", 3))
        self.dynamic_obs_est_exclude_robot_radius = float(
            cfg.get("dynamic_obs_est_exclude_robot_radius", 0.85)
        )
        self.dynamic_obs_est_show_static_tracks = bool(cfg.get("dynamic_obs_est_show_static_tracks", False))
        self._obs_dyn_tracks: list[dict] = []
        self._obs_dyn_next_id = 0
        self._last_final_dist: torch.Tensor | None = None
        self._no_progress_steps: torch.Tensor | None = None
        self._prev_cmd: torch.Tensor | None = None
        self._simple_avoid_dir: torch.Tensor | None = None
        self._simple_avoid_hold: torch.Tensor | None = None
        self._crowd_backup_hold: torch.Tensor | None = None
        self._crowd_backup_dir: torch.Tensor | None = None
        self._escape_hold: torch.Tensor | None = None
        self._escape_dir: torch.Tensor | None = None
        self._avoid_side_prev: torch.Tensor | None = None
        self._avoid_side_switches: torch.Tensor | None = None

    @staticmethod
    def _clampf(x: float, lo: float, hi: float) -> float:
        return max(lo, min(hi, x))

    @staticmethod
    def _safe_min(x: torch.Tensor | None, mask: torch.Tensor | None = None) -> float:
        if x is None:
            return float("inf")
        if mask is not None:
            if mask.numel() != x.numel():
                n = min(int(mask.numel()), int(x.numel()))
                if n <= 0:
                    return float("inf")
                mask = mask[:n]
                x = x[:n]
            sel = x[mask]
        else:
            sel = x
        if sel.numel() == 0:
            return float("inf")
        return float(torch.min(sel).item())

    def _get_sonar_ranges(self, env_id: int, obs_1d: torch.Tensor) -> torch.Tensor | None:
        # Preferred path: sonar rays are appended to obs (some envs do this).
        start = 12
        end = start + self.num_rays
        if obs_1d.numel() >= end:
            return obs_1d[start:end].detach().float().cpu().view(-1)

        # Fallback path: env-provided sonar buffers.
        for attr in ("_last_map_sonar", "_last_sonar"):
            try:
                src = getattr(self.env, attr, None)
                if src is None:
                    continue
                src_t = torch.as_tensor(src)
                if src_t.ndim != 2 or env_id < 0 or env_id >= int(src_t.shape[0]):
                    continue
                row = src_t[env_id].detach().float().cpu().view(-1)
                if row.numel() <= 0:
                    continue
                if row.numel() >= self.num_rays:
                    return row[: self.num_rays]
                max_range = float(getattr(getattr(self.env, "sonar", None), "cfg", {}).max_range) if hasattr(getattr(self.env, "sonar", None), "cfg") else 10.0
                out = torch.full((self.num_rays,), max_range, dtype=torch.float32)
                out[: row.numel()] = row
                return out
            except Exception:
                continue

        # Last-resort: one direct sonar scan using current pose.
        try:
            pos_w = self.env.drone.pos[:, 0, :]
            rot_w = self.env.drone.rot[:, 0, :]
            live = self.env.sonar.scan(pos_w, rot_w)["ranges"]
            if hasattr(self.env, "_sonar_ranges_from_base_link"):
                live = self.env._sonar_ranges_from_base_link(live)
            else:
                try:
                    max_range = float(getattr(getattr(self.env, "sonar", None), "cfg", {}).max_range)
                    origin_offset = float(getattr(getattr(self.env, "sonar", None), "cfg", {}).origin_offset)
                    if origin_offset > 0.0:
                        live = torch.where(live < (max_range - 1e-6), torch.clamp(live + origin_offset, max=max_range), live)
                except Exception:
                    pass
            if live.ndim == 2 and 0 <= env_id < int(live.shape[0]):
                row = live[env_id].detach().float().cpu().view(-1)
                if row.numel() >= self.num_rays:
                    return row[: self.num_rays]
                if row.numel() > 0:
                    max_range = float(getattr(getattr(self.env, "sonar", None), "cfg", {}).max_range) if hasattr(getattr(self.env, "sonar", None), "cfg") else 10.0
                    out = torch.full((self.num_rays,), max_range, dtype=torch.float32)
                    out[: row.numel()] = row
                    return out
        except Exception:
            pass
        return None

    @staticmethod
    def _build_angles(env, num_rays: int) -> torch.Tensor:
        try:
            angles = env.sonar.angles.detach().float().cpu().view(-1)
            if angles.numel() == num_rays:
                return angles
        except Exception:
            pass

        sonar_cfg = getattr(env, "sonar_cfg", {}) or {}
        fov_deg = float(sonar_cfg.get("fov_deg", 120.0))
        return torch.linspace(-0.5 * fov_deg, 0.5 * fov_deg, num_rays) * (math.pi / 180.0)

    def reset_internal_state(self, num_envs: int, device: torch.device) -> None:
        """Reset all controller memories that should not leak across trial resets."""
        self._step_count = 0
        self._replan_env_ids = []
        self._last_debug = []
        self._last_replan_step = torch.full((num_envs,), -10_000, dtype=torch.long, device=device)
        self._replan_pressure = torch.zeros((num_envs,), dtype=torch.long, device=device)
        self._prev_obst_pos_w = None
        self._obst_vel_w = None
        self._obst_is_dynamic = None
        self._prev_front_min = None
        self._prev_left_min = None
        self._prev_right_min = None
        self._front_rate_ema = None
        self._left_rate_ema = None
        self._right_rate_ema = None
        self._obs_dyn_tracks = []
        self._obs_dyn_next_id = 0
        self._last_final_dist = torch.full((num_envs,), float("inf"), dtype=torch.float32, device=device)
        self._no_progress_steps = torch.zeros((num_envs,), dtype=torch.long, device=device)
        self._prev_cmd = torch.zeros((num_envs, 4), dtype=torch.float32, device=device)
        self._simple_avoid_dir = torch.zeros((num_envs,), dtype=torch.float32, device=device)
        self._simple_avoid_hold = torch.zeros((num_envs,), dtype=torch.long, device=device)
        self._crowd_backup_hold = torch.zeros((num_envs,), dtype=torch.long, device=device)
        self._crowd_backup_dir = torch.zeros((num_envs,), dtype=torch.float32, device=device)
        self._escape_hold = torch.zeros((num_envs,), dtype=torch.long, device=device)
        self._escape_dir = torch.zeros((num_envs,), dtype=torch.float32, device=device)
        self._avoid_side_prev = torch.zeros((num_envs,), dtype=torch.float32, device=device)
        self._avoid_side_switches = torch.zeros((num_envs,), dtype=torch.long, device=device)

    def reset_env_state(self, env_ids: torch.Tensor | list[int]) -> None:
        """Reset controller memories for envs that just auto-reset."""
        if env_ids is None:
            return
        try:
            if not torch.is_tensor(env_ids):
                device = self._prev_cmd.device if self._prev_cmd is not None else torch.device("cpu")
                env_ids = torch.as_tensor(env_ids, device=device, dtype=torch.long)
            else:
                env_ids = env_ids.reshape(-1).to(dtype=torch.long)
            if env_ids.numel() <= 0:
                return
            for name, value in (
                ("_last_replan_step", -10_000),
                ("_replan_pressure", 0),
                ("_last_final_dist", float("inf")),
                ("_no_progress_steps", 0),
                ("_simple_avoid_dir", 0.0),
                ("_simple_avoid_hold", 0),
                ("_crowd_backup_hold", 0),
                ("_crowd_backup_dir", 0.0),
                ("_escape_hold", 0),
                ("_escape_dir", 0.0),
                ("_avoid_side_prev", 0.0),
                ("_avoid_side_switches", 0),
            ):
                arr = getattr(self, name, None)
                if arr is not None and torch.is_tensor(arr):
                    valid = env_ids[(env_ids >= 0) & (env_ids < int(arr.shape[0]))]
                    if valid.numel() > 0:
                        arr[valid] = value
            if self._prev_cmd is not None and torch.is_tensor(self._prev_cmd):
                valid = env_ids[(env_ids >= 0) & (env_ids < int(self._prev_cmd.shape[0]))]
                if valid.numel() > 0:
                    self._prev_cmd[valid] = 0.0
            # These trackers are shared controller memory. Clear them on reset
            # so old obstacle motion does not leak into a new episode.
            self._prev_obst_pos_w = None
            self._obst_vel_w = None
            self._obst_is_dynamic = None
            self._prev_front_min = None
            self._prev_left_min = None
            self._prev_right_min = None
            self._front_rate_ema = None
            self._left_rate_ema = None
            self._right_rate_ema = None
            self._obs_dyn_tracks = []
            self._obs_dyn_next_id = 0
        except Exception:
            return

    def _to_action(self, u: float, v: float, w: float, r: float) -> torch.Tensor:
        au = self._clampf(u / max(self.u_max, 1e-6), -1.0, 1.0)
        av = self._clampf(v / max(self.v_max, 1e-6), -1.0, 1.0)
        aw = self._clampf(w / max(self.w_max, 1e-6), -1.0, 1.0)
        ar = self._clampf(r / max(self.r_max, 1e-6), -1.0, 1.0)
        return torch.tensor([au, av, aw, ar], dtype=torch.float32)

    @staticmethod
    def _yaw_from_quat_wxyz(q: torch.Tensor) -> torch.Tensor:
        w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
        return torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))

    def _update_dynamic_tracks(self, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | None:
        if not self.dynamic_enable:
            return None
        if not getattr(self.env, "enable_obstacles", False):
            return None
        if getattr(self.env, "obstacles", None) is None:
            return None
        if int(getattr(self.env, "num_obstacles", 0)) <= 0:
            return None
        try:
            pos_w, _ = self.env.obstacles.get_world_poses()
            pos_w = pos_w.to(device)
            if pos_w.ndim == 2:
                pos_w = pos_w.view(self.env.num_envs, self.env.num_obstacles, 3)
        except Exception:
            return None

        n, m, _ = pos_w.shape
        reset_mask = None
        try:
            pb = getattr(self.env, "progress_buf", None)
            if pb is not None:
                reset_mask = (pb.to(device).view(-1) <= 1)
        except Exception:
            reset_mask = None

        if (
            self._prev_obst_pos_w is None
            or self._obst_vel_w is None
            or self._obst_is_dynamic is None
            or self._prev_obst_pos_w.shape != pos_w.shape
        ):
            self._prev_obst_pos_w = pos_w.detach().clone()
            self._obst_vel_w = torch.zeros_like(pos_w)
            self._obst_is_dynamic = torch.zeros((n, m), dtype=torch.bool, device=device)
            return pos_w, self._obst_vel_w, self._obst_is_dynamic, torch.zeros((n, m), device=device)

        vel_inst = (pos_w - self._prev_obst_pos_w) / self._dt
        ema = float(self.dynamic_vel_ema)
        self._obst_vel_w = (1.0 - ema) * self._obst_vel_w + ema * vel_inst
        speed_xy = torch.norm(self._obst_vel_w[..., :2], dim=-1)
        dyn_hi = speed_xy >= (self.dynamic_speed_threshold + self.dynamic_speed_hysteresis)
        dyn_lo = speed_xy <= max(0.0, self.dynamic_speed_threshold - self.dynamic_speed_hysteresis)
        self._obst_is_dynamic = torch.where(dyn_hi, torch.ones_like(dyn_hi), self._obst_is_dynamic)
        self._obst_is_dynamic = torch.where(dyn_lo, torch.zeros_like(dyn_lo), self._obst_is_dynamic)

        if reset_mask is not None and bool(reset_mask.any().item()):
            self._obst_vel_w[reset_mask] = 0.0
            self._obst_is_dynamic[reset_mask] = False

        self._prev_obst_pos_w = pos_w.detach().clone()
        return pos_w, self._obst_vel_w, self._obst_is_dynamic, speed_xy

    def dynamic_estimates_payload(self, env_id: int = 0) -> list[dict]:
        """Return estimated dynamic obstacle pose/velocity in local map coordinates."""
        if self.dynamic_from_observation_only:
            return self._dynamic_estimates_payload_from_observation(env_id=env_id)
        try:
            device = self.env.device
            tracks = self._update_dynamic_tracks(device)
            if tracks is None:
                return []
            pos_w, vel_w, is_dyn, speed_xy = tracks
            if env_id < 0 or env_id >= int(pos_w.shape[0]):
                return []
            env_offset = self.env.envs_positions[env_id].to(device)
            pos_local = pos_w[env_id] - env_offset.view(1, 3)
            vel = vel_w[env_id]
            dyn = is_dyn[env_id]
            speed = speed_xy[env_id]
            out = []
            for k in range(int(pos_local.shape[0])):
                out.append(
                    {
                        "id": int(k),
                        "position": pos_local[k].detach().float().cpu().tolist(),
                        "velocity": vel[k].detach().float().cpu().tolist(),
                        "speed": float(speed[k].detach().float().cpu().item()),
                        "is_dynamic": bool(dyn[k].detach().cpu().item()),
                    }
                )
            return out
        except Exception:
            return []

    def _update_observation_dynamic_tracks(self, detections: list[dict], device: torch.device, source: str) -> list[dict]:
        pos_alpha = min(1.0, max(0.0, float(self.dynamic_obs_est_pos_ema)))
        vel_alpha = min(1.0, max(0.0, float(self.dynamic_obs_est_vel_ema)))
        assoc_dist = max(0.01, float(self.dynamic_obs_est_assoc_dist))
        robot_xy = None
        try:
            robot_xy = self.env.drone.pos[0, 0, :2].to(device) - self.env.envs_positions[0, :2].to(device)
        except Exception:
            robot_xy = None
        exclude_r = max(0.0, float(self.dynamic_obs_est_exclude_robot_radius))
        used_tracks: set[int] = set()
        now_step = int(self._step_count)

        for det in detections:
            center = torch.as_tensor(det["center"], device=device, dtype=torch.float32)
            if robot_xy is not None and exclude_r > 0.0:
                if float(torch.norm(center[:2] - robot_xy).item()) <= exclude_r:
                    continue
            best_i = -1
            best_d = float("inf")
            for ti, tr in enumerate(self._obs_dyn_tracks):
                if ti in used_tracks:
                    continue
                tr_pos = torch.as_tensor(tr["position"], device=device, dtype=center.dtype)
                d = float(torch.norm(center[:2] - tr_pos[:2]).item())
                if d < best_d:
                    best_d = d
                    best_i = ti
            if best_i >= 0 and best_d <= assoc_dist:
                tr = self._obs_dyn_tracks[best_i]
                old_pos = torch.as_tensor(tr["position"], device=device, dtype=center.dtype)
                inst_vel = (center - old_pos) / self._dt
                old_vel = torch.as_tensor(tr["velocity"], device=device, dtype=center.dtype)
                new_pos = (1.0 - pos_alpha) * old_pos + pos_alpha * center
                new_vel = (1.0 - vel_alpha) * old_vel + vel_alpha * inst_vel
                tr["position"] = new_pos.detach().cpu().tolist()
                tr["surface_position"] = det.get("surface", center).detach().cpu().tolist()
                tr["velocity"] = new_vel.detach().cpu().tolist()
                tr["speed"] = float(torch.norm(new_vel[:2]).item())
                tr["last_seen"] = now_step
                tr["age"] = int(tr.get("age", 0)) + 1
                tr["ray_count"] = int(det.get("ray_count", 0))
                tr["cell_count"] = int(det.get("cell_count", 0))
                tr["source"] = source
                used_tracks.add(best_i)
            else:
                self._obs_dyn_tracks.append(
                    {
                        "id": int(self._obs_dyn_next_id),
                        "position": center.detach().cpu().tolist(),
                        "surface_position": det.get("surface", center).detach().cpu().tolist(),
                        "velocity": [0.0, 0.0, 0.0],
                        "speed": 0.0,
                        "last_seen": now_step,
                        "age": 1,
                        "ray_count": int(det.get("ray_count", 0)),
                        "cell_count": int(det.get("cell_count", 0)),
                        "source": source,
                    }
                )
                self._obs_dyn_next_id += 1

        ttl = max(1, int(self.dynamic_obs_est_ttl_steps))
        self._obs_dyn_tracks = [
            tr for tr in self._obs_dyn_tracks if (now_step - int(tr.get("last_seen", now_step))) <= ttl
        ]
        self._obs_dyn_tracks.sort(key=lambda tr: float(tr.get("speed", 0.0)), reverse=True)

        out = []
        for tr in self._obs_dyn_tracks[: int(self.dynamic_obs_est_max_markers)]:
            speed = float(tr.get("speed", 0.0))
            is_dyn = speed >= float(self.dynamic_obs_est_marker_min_speed) and int(tr.get("age", 0)) >= 2
            if (not is_dyn) and (not self.dynamic_obs_est_show_static_tracks):
                continue
            out.append(
                {
                    "id": int(tr.get("id", 0)),
                    "position": tr.get("position", [0.0, 0.0, 0.0]),
                    "surface_position": tr.get("surface_position", tr.get("position", [0.0, 0.0, 0.0])),
                    "velocity": tr.get("velocity", [0.0, 0.0, 0.0]),
                    "speed": speed,
                    "is_dynamic": bool(is_dyn),
                    "source": str(tr.get("source", source)),
                }
            )
        return out

    def _dynamic_estimates_payload_from_dynamic_map(self, env_id: int = 0) -> list[dict]:
        try:
            dyn_map = getattr(self.env, "_map_dynamic", None)
            if dyn_map is None:
                return []
            device = self.env.device
            dyn = torch.as_tensor(dyn_map, device=device).detach().float()
            if dyn.ndim == 3:
                if env_id < 0 or env_id >= int(dyn.shape[0]):
                    return []
                dyn = dyn[env_id]
            if dyn.ndim != 2:
                return []
            mask = dyn >= float(self.dynamic_obs_est_map_threshold)
            if not bool(mask.any().item()):
                return []
            occ = mask.detach().cpu().numpy().astype(bool)
            h, w = occ.shape
            x_min = float(getattr(self.env, "_map_x_min", 0.0))
            x_max = float(getattr(self.env, "_map_x_max", float(w - 1)))
            y_min = float(getattr(self.env, "_map_y_min", 0.0))
            y_max = float(getattr(self.env, "_map_y_max", float(h - 1)))
            dx = (x_max - x_min) / max(1, w - 1)
            dy = (y_max - y_min) / max(1, h - 1)
            seen = np.zeros_like(occ, dtype=bool)
            detections: list[dict] = []
            min_cells = max(1, int(self.dynamic_obs_est_min_cells))
            for sy in range(h):
                for sx in range(w):
                    if not occ[sy, sx] or seen[sy, sx]:
                        continue
                    stack = [(sy, sx)]
                    seen[sy, sx] = True
                    cells = []
                    while stack:
                        cy, cx = stack.pop()
                        cells.append((cy, cx))
                        for ny in (cy - 1, cy, cy + 1):
                            for nx in (cx - 1, cx, cx + 1):
                                if ny < 0 or ny >= h or nx < 0 or nx >= w:
                                    continue
                                if seen[ny, nx] or not occ[ny, nx]:
                                    continue
                                seen[ny, nx] = True
                                stack.append((ny, nx))
                    if len(cells) < min_cells:
                        continue
                    arr = np.asarray(cells, dtype=np.float32)
                    ys = arr[:, 0]
                    xs = arr[:, 1]
                    vals = dyn.detach().cpu().numpy()[ys.astype(np.int64), xs.astype(np.int64)]
                    weights = np.maximum(vals, 1e-3)
                    wx = float(np.sum((x_min + xs * dx) * weights) / np.sum(weights))
                    wy = float(np.sum((y_min + ys * dy) * weights) / np.sum(weights))
                    z = 0.0
                    try:
                        z = float((self.env.drone.pos[env_id, 0, 2] - self.env.envs_positions[env_id, 2]).item())
                    except Exception:
                        pass
                    center = torch.tensor([wx, wy, z], device=device, dtype=torch.float32)
                    detections.append({"center": center, "surface": center, "cell_count": len(cells)})
                    if len(detections) >= int(self.dynamic_obs_est_max_markers) * 2:
                        break
                if len(detections) >= int(self.dynamic_obs_est_max_markers) * 2:
                    break
            if not detections:
                return []
            return self._update_observation_dynamic_tracks(
                detections=detections,
                device=device,
                source="dynamic_occupancy_map",
            )
        except Exception:
            return []

    def _dynamic_estimates_payload_from_observation(self, env_id: int = 0) -> list[dict]:
        """Estimate dynamic obstacle markers from sonar hit motion only.

        This intentionally avoids simulator obstacle poses. The returned marker
        position is a center estimate; the occupancy map remains based on sonar
        surface hits, which should appear closer than a center marker.
        """
        try:
            map_out = self._dynamic_estimates_payload_from_dynamic_map(env_id=env_id)
            if map_out:
                return map_out

            device = self.env.device
            sonar = None
            for attr in ("_last_map_sonar", "_last_sonar"):
                src = getattr(self.env, attr, None)
                if src is None:
                    continue
                src_t = torch.as_tensor(src, device=device).detach().float()
                if src_t.ndim == 2 and 0 <= env_id < int(src_t.shape[0]):
                    sonar = src_t[env_id].reshape(-1)
                    break
            if sonar is None or sonar.numel() <= 0:
                return []

            n = min(int(sonar.numel()), int(self.angles.numel()))
            ranges = sonar[:n].clamp(min=0.0)
            angles = self.angles[:n].to(device)
            max_range = float(getattr(getattr(self.env, "sonar", None), "cfg", {}).max_range) if hasattr(getattr(self.env, "sonar", None), "cfg") else 10.0
            hit_mask = (ranges < (max_range - 1e-3)) & torch.isfinite(ranges)

            pos_w = self.env.drone.pos[env_id, 0, :].to(device)
            rot_w = self.env.drone.rot[env_id, 0, :].to(device)
            env_offset = self.env.envs_positions[env_id].to(device)
            pos_local = pos_w - env_offset
            yaw = self._yaw_from_quat_wxyz(rot_w.view(1, 4))[0]
            theta = yaw + angles
            dirs = torch.stack([torch.cos(theta), torch.sin(theta), torch.zeros_like(theta)], dim=-1)
            points = pos_local.view(1, 3) + ranges.view(-1, 1) * dirs

            idxs = torch.nonzero(hit_mask, as_tuple=False).reshape(-1).detach().cpu().tolist()
            clusters: list[list[int]] = []
            cur: list[int] = []
            for idx in idxs:
                if cur and idx > cur[-1] + 1:
                    clusters.append(cur)
                    cur = []
                cur.append(int(idx))
            if cur:
                clusters.append(cur)

            min_rays = max(1, int(self.dynamic_obs_est_min_rays))
            radius = max(0.0, float(self.dynamic_obs_est_center_offset_radius))
            detections: list[dict] = []
            for cid, cluster in enumerate(clusters):
                if len(cluster) < min_rays:
                    continue
                cidx = torch.as_tensor(cluster, device=device, dtype=torch.long)
                pts = points[cidx]
                centroid = pts.mean(dim=0)
                ray_dir = dirs[cidx].mean(dim=0)
                ray_norm = torch.norm(ray_dir[:2]).clamp_min(1e-6)
                ray_dir = ray_dir / ray_norm
                center = centroid + radius * ray_dir
                detections.append({"center": center.detach(), "surface": centroid.detach(), "ray_count": len(cluster)})
                if len(detections) >= int(self.dynamic_obs_est_max_markers) * 2:
                    break

            return self._update_observation_dynamic_tracks(
                detections=detections,
                device=device,
                source="sonar_observation_cluster",
            )
        except Exception:
            return []

    def _compute_dynamic_risk(self, device: torch.device) -> list[DynamicRisk]:
        n = int(self.env.num_envs)
        default = DynamicRisk(
            static_count=0,
            dynamic_count=0,
            mean_dynamic_speed=0.0,
            min_static_dist=float("inf"),
            min_dynamic_dist=float("inf"),
            min_ttc=float("inf"),
            ttc_obstacle_idx=-1,
            imminent=False,
            approaching=False,
            turn_preference=0.0,
            heave_preference=0.0,
            proceed_safe=True,
            speed_scale=1.0,
            crossing_preference=0.0,
            source="none",
        )
        out = [default for _ in range(n)]
        tracks = self._update_dynamic_tracks(device)
        if tracks is None:
            return out

        pos_o, vel_o, is_dyn, speed_xy = tracks
        pos_r = self.env.drone.pos[:, 0, :].to(device)
        vel_r = self.env.drone.vel_w[:, 0, :3].to(device)
        rot_r = self.env.drone.rot[:, 0, :].to(device)
        yaw_r = self._yaw_from_quat_wxyz(rot_r)
        cos_y = torch.cos(yaw_r)
        sin_y = torch.sin(yaw_r)

        radius_o = float(getattr(self.env, "obst_cfg", {}).get("radius", 0.15))
        safety_rad = max(0.0, self.robot_radius + radius_o + self.dynamic_safety_margin)

        p = pos_o - pos_r[:, None, :]
        v = vel_o - vel_r[:, None, :]
        d_now = torch.norm(p, dim=-1) - safety_rad

        pv = (p * v).sum(dim=-1)
        vv = (v * v).sum(dim=-1).clamp_min(1e-6)
        t_cpa = -pv / vv
        t_valid = is_dyn & (pv < 0.0) & (t_cpa > 0.0) & (t_cpa <= self.ttc_horizon)
        p_cpa = p + t_cpa.unsqueeze(-1) * v
        d_cpa = torch.norm(p_cpa, dim=-1) - safety_rad
        coll_pred = t_valid & (d_cpa <= self.dynamic_cpa_margin)

        for i in range(n):
            dyn_mask = is_dyn[i]
            st_mask = ~dyn_mask
            dyn_count = int(dyn_mask.sum().item())
            st_count = int(st_mask.sum().item())
            mean_dyn_spd = float(speed_xy[i][dyn_mask].mean().item()) if dyn_count > 0 else 0.0
            min_dyn_dist = float(d_now[i][dyn_mask].min().item()) if dyn_count > 0 else float("inf")
            min_st_dist = float(d_now[i][st_mask].min().item()) if st_count > 0 else float("inf")

            min_ttc = float("inf")
            ttc_obstacle_idx = -1
            imminent = False
            approaching = False
            turn_pref = 0.0
            heave_pref = 0.0
            proceed_safe = True
            speed_scale = 1.0
            crossing_pref = 0.0
            pick_idx = -1
            approach_mask = t_valid[i]
            if bool(approach_mask.any().item()):
                approaching = True
                t_vals = t_cpa[i][approach_mask]
                min_ttc = float(t_vals.min().item()) if t_vals.numel() > 0 else float("inf")
            if bool(coll_pred[i].any().item()):
                imminent = True
                t_vals = t_cpa[i].clone()
                t_vals[~coll_pred[i]] = float("inf")
                pick_idx = int(torch.argmin(t_vals).item())
                min_ttc = float(t_vals[pick_idx].item())
                ttc_obstacle_idx = pick_idx
            elif dyn_count > 0:
                # Fallback: closest dynamic obstacle for soft predictive bias.
                d_vals = d_now[i].clone()
                d_vals[~dyn_mask] = float("inf")
                pick_idx = int(torch.argmin(d_vals).item())
                if min_ttc == float("inf"):
                    min_ttc = float(t_cpa[i, pick_idx].item()) if bool(t_valid[i, pick_idx].item()) else float("inf")
                if pick_idx >= 0 and bool(dyn_mask[pick_idx].item()):
                    ttc_obstacle_idx = pick_idx

            if dyn_count > 0 and bool(approach_mask.any().item()):
                # Aggregate turn/heave bias from all approaching dynamic obstacles.
                p_i = p[i]
                weights = 1.0 / (d_now[i].clamp_min(0.05) + 0.2)
                weights = torch.where(approach_mask, weights, torch.zeros_like(weights))
                x_b = cos_y[i] * p_i[:, 0] + sin_y[i] * p_i[:, 1]
                y_b = -sin_y[i] * p_i[:, 0] + cos_y[i] * p_i[:, 1]
                side_push = (-torch.sign(y_b)) * weights
                heave_push = (-torch.sign(p_i[:, 2])) * weights
                sw = float(side_push.sum().item())
                hw = float(heave_push.sum().item())
                if abs(sw) > 1e-6:
                    turn_pref = 1.0 if sw > 0.0 else -1.0
                if abs(hw) > 1e-6:
                    heave_pref = 1.0 if hw > 0.0 else -1.0

            if pick_idx >= 0:
                p_sel = p[i, pick_idx]
                v_sel = v[i, pick_idx]
                # Relative position in robot body frame (yaw-only transform).
                x_b = cos_y[i] * p_sel[0] + sin_y[i] * p_sel[1]
                y_b = -sin_y[i] * p_sel[0] + cos_y[i] * p_sel[1]
                vx_b = cos_y[i] * v_sel[0] + sin_y[i] * v_sel[1]
                vy_b = -sin_y[i] * v_sel[0] + cos_y[i] * v_sel[1]
                if abs(float(vy_b.item())) >= self.dynamic_crossing_speed_thresh:
                    # Obstacle moving left leaves space on the right, and vice versa.
                    crossing_pref = float(-torch.sign(vy_b).item())
                forward_speed = max(0.12, float(vel_r[i, 0].item()), self.max_fwd * self.dynamic_go_speed_scale)
                if float(x_b.item()) > 0.0:
                    t_arrive = float(x_b.item()) / forward_speed
                    y_future = float(y_b.item()) + float(vy_b.item()) * t_arrive
                    corridor = self.dynamic_corridor_half_width + safety_rad + self.dynamic_crossing_margin
                    crossing_clear = abs(y_future) > corridor
                    moving_away = float(vx_b.item()) >= -0.02 and abs(float(y_b.item())) > corridor
                    if crossing_clear or moving_away:
                        proceed_safe = True
                        speed_scale = min(1.0, max(0.45, self.dynamic_go_speed_scale))
                    elif min_ttc <= self.dynamic_yield_ttc or imminent:
                        proceed_safe = False
                        speed_scale = 0.0
                    elif min_ttc >= self.dynamic_go_ttc:
                        proceed_safe = True
                        speed_scale = min(1.0, max(0.35, self.dynamic_go_speed_scale))
                    else:
                        proceed_safe = False
                        speed_scale = 0.35
                if abs(turn_pref) < 1e-4:
                    turn_pref = crossing_pref
                if abs(turn_pref) < 1e-4:
                    turn_pref = float(-torch.sign(y_b).item()) if abs(float(y_b.item())) > 1e-3 else 0.0
                if abs(heave_pref) < 1e-4:
                    heave_pref = float(-torch.sign(p_sel[2]).item()) if abs(float(p_sel[2].item())) > 1e-3 else 0.0

            if (not imminent) and approaching and (min_dyn_dist <= self.dynamic_proximity_trigger):
                imminent = True
                proceed_safe = False
                speed_scale = 0.0

            out[i] = DynamicRisk(
                static_count=st_count,
                dynamic_count=dyn_count,
                mean_dynamic_speed=mean_dyn_spd,
                min_static_dist=min_st_dist,
                min_dynamic_dist=min_dyn_dist,
                min_ttc=min_ttc,
                ttc_obstacle_idx=int(ttc_obstacle_idx),
                imminent=imminent,
                approaching=approaching,
                turn_preference=turn_pref,
                heave_preference=heave_pref,
                proceed_safe=proceed_safe,
                speed_scale=speed_scale,
                crossing_preference=crossing_pref,
                source="oracle",
            )
        return out

    def _compute_dynamic_risk_from_obs(self, states: list[EnvState], device: torch.device) -> list[DynamicRisk]:
        n = len(states)
        out: list[DynamicRisk] = []
        if (
            self._prev_front_min is None
            or self._prev_left_min is None
            or self._prev_right_min is None
            or int(self._prev_front_min.numel()) != n
        ):
            self._prev_front_min = torch.tensor([s.front_min for s in states], device=device, dtype=torch.float32)
            self._prev_left_min = torch.tensor([s.left_min for s in states], device=device, dtype=torch.float32)
            self._prev_right_min = torch.tensor([s.right_min for s in states], device=device, dtype=torch.float32)
            self._front_rate_ema = torch.zeros(n, device=device)
            self._left_rate_ema = torch.zeros(n, device=device)
            self._right_rate_ema = torch.zeros(n, device=device)

        prev_f = self._prev_front_min
        prev_l = self._prev_left_min
        prev_r = self._prev_right_min
        front = torch.tensor([s.front_min for s in states], device=device, dtype=torch.float32)
        left = torch.tensor([s.left_min for s in states], device=device, dtype=torch.float32)
        right = torch.tensor([s.right_min for s in states], device=device, dtype=torch.float32)

        # Positive rate means obstacle is getting closer.
        rf = (prev_f - front) / self._dt
        rl = (prev_l - left) / self._dt
        rr = (prev_r - right) / self._dt

        alpha = float(self.dynamic_vel_ema)
        self._front_rate_ema = (1.0 - alpha) * self._front_rate_ema + alpha * rf
        self._left_rate_ema = (1.0 - alpha) * self._left_rate_ema + alpha * rl
        self._right_rate_ema = (1.0 - alpha) * self._right_rate_ema + alpha * rr

        self._prev_front_min = front
        self._prev_left_min = left
        self._prev_right_min = right

        for i, s in enumerate(states):
            f = float(front[i].item())
            l = float(left[i].item())
            r = float(right[i].item())
            c_front = max(0.0, float(self._front_rate_ema[i].item()))
            c_left = max(0.0, float(self._left_rate_ema[i].item()))
            c_right = max(0.0, float(self._right_rate_ema[i].item()))
            closing = max(c_front, c_left, c_right)
            approaching = closing > self.closing_rate_thresh
            min_ttc = (f / max(c_front, 1e-6)) if c_front > self.closing_rate_thresh else float("inf")

            # Estimate dynamic/static from temporal change in sectors.
            dynamic_count = int(approaching)
            static_count = int((f < self.avoid_dist) and not approaching)
            min_dynamic_dist = f if dynamic_count > 0 else float("inf")
            min_static_dist = f if static_count > 0 else float("inf")
            imminent = (min_ttc <= self.ttc_stop) or (approaching and f <= self.dynamic_proximity_trigger)
            proceed_safe = True
            speed_scale = 1.0
            if approaching:
                if imminent or min_ttc <= self.dynamic_yield_ttc or f <= self.dynamic_go_front_dist:
                    proceed_safe = False
                    speed_scale = 0.0
                elif min_ttc >= self.dynamic_go_ttc or f >= self.dynamic_slow_front_dist:
                    proceed_safe = True
                    speed_scale = min(1.0, max(0.35, self.dynamic_go_speed_scale))
                else:
                    proceed_safe = False
                    speed_scale = 0.35

            # Turn toward more open/slower-closing side.
            turn_pref = 1.0 if l > r else -1.0
            crossing_pref = 0.0
            if (c_left - c_right) > self.side_rate_margin:
                # Left sector is getting occupied; use the right-side gap.
                turn_pref = -1.0
                crossing_pref = -1.0
            elif (c_right - c_left) > self.side_rate_margin:
                # Right sector is getting occupied; use the left-side gap.
                turn_pref = 1.0
                crossing_pref = 1.0
            heave_pref = float(-math.copysign(1.0, float(s.rel_b[2].item()))) if abs(float(s.rel_b[2].item())) > 0.05 else 0.0

            out.append(
                DynamicRisk(
                    static_count=static_count,
                    dynamic_count=dynamic_count,
                    mean_dynamic_speed=closing,
                    min_static_dist=min_static_dist,
                    min_dynamic_dist=min_dynamic_dist,
                    min_ttc=min_ttc,
                    ttc_obstacle_idx=-1,
                    imminent=imminent,
                    approaching=approaching,
                    turn_preference=turn_pref,
                    heave_preference=heave_pref,
                    proceed_safe=proceed_safe,
                    speed_scale=speed_scale,
                    crossing_preference=crossing_pref,
                    source="obs",
                )
            )
        return out

    def _parse_state(self, env_id: int, obs_1d: torch.Tensor) -> EnvState:
        if obs_1d.numel() < 12:
            raise ValueError(f"Expected obs dim >= 12, got {obs_1d.numel()}")

        rel_b = obs_1d[self.IDX_REL]
        yaw_err = float(torch.atan2(obs_1d[self.IDX_YAW_SIN], obs_1d[self.IDX_YAW_COS]).item())
        vel_b = obs_1d[self.IDX_VEL]
        yaw_rate = float(obs_1d[self.IDX_YAW_RATE].item())
        depth = float(obs_1d[self.IDX_DEPTH].item())
        roll = float(obs_1d[self.IDX_ROLL].item())
        pitch = float(obs_1d[self.IDX_PITCH].item())

        sonar_ranges = self._get_sonar_ranges(env_id, obs_1d)

        min_range = self._safe_min(sonar_ranges)
        front_min = self._safe_min(sonar_ranges, self.front_mask)
        left_min = self._safe_min(sonar_ranges, self.left_mask)
        right_min = self._safe_min(sonar_ranges, self.right_mask)

        return EnvState(
            rel_b=rel_b,
            yaw_err=yaw_err,
            vel_b=vel_b,
            yaw_rate=yaw_rate,
            depth=depth,
            roll=roll,
            pitch=pitch,
            min_range=min_range,
            front_min=front_min,
            left_min=left_min,
            right_min=right_min,
        )

    def _final_goal_state(self, env_id: int, device: torch.device, fallback: EnvState) -> EnvState:
        try:
            pos_w = self.env.drone.pos[env_id, 0, :].to(device)
            rot_w = self.env.drone.rot[env_id, 0, :].to(device)
            goal_w = self.env.target_pos[env_id, 0, :].to(device)
            rel_w = goal_w - pos_w
            rel_b = _quat_rotate_inverse(rot_w.unsqueeze(0), rel_w.unsqueeze(0)).squeeze(0)
            yaw_err = float(math.atan2(float(rel_b[1].item()), float(rel_b[0].item())))
            return EnvState(
                rel_b=rel_b,
                yaw_err=yaw_err,
                vel_b=fallback.vel_b,
                yaw_rate=fallback.yaw_rate,
                depth=fallback.depth,
                roll=fallback.roll,
                pitch=fallback.pitch,
                min_range=fallback.min_range,
                front_min=fallback.front_min,
                left_min=fallback.left_min,
                right_min=fallback.right_min,
            )
        except Exception:
            return fallback

    def _hover(self, s: EnvState) -> torch.Tensor:
        u = -0.8 * float(s.vel_b[0].item())
        v = -0.8 * float(s.vel_b[1].item())
        w = -0.8 * float(s.vel_b[2].item())
        r = -0.7 * s.yaw_rate
        return self._to_action(u, v, w, r)

    def _apply_yaw_priority(self, u: float, v: float, r: float, yaw_err: float) -> tuple[float, float, float]:
        ae = abs(float(yaw_err))
        thr = max(1e-3, self.yaw_priority_thresh)
        if ae <= thr:
            return u, v, r
        k = min(1.0, max(0.0, (ae - thr) / max(math.pi - thr, 1e-3)))
        su = 1.0 - k * (1.0 - self.yaw_priority_u_scale)
        sv = 1.0 - k * (1.0 - self.yaw_priority_v_scale)
        sr = 1.0 + k * max(0.0, self.yaw_priority_r_boost - 1.0)
        return u * su, v * sv, r * sr

    def _align(self, s: EnvState) -> torch.Tensor:
        u_des = self._clampf(self.k_pos_xy * float(torch.tanh(0.8 * s.rel_b[0]).item()), -self.max_fwd, self.max_fwd)
        v_des = self._clampf(self.k_pos_xy * float(torch.tanh(0.8 * s.rel_b[1]).item()), -self.max_lat, self.max_lat)
        w_des = self._clampf(self.k_pos_z * float(torch.tanh(0.8 * s.rel_b[2]).item()), -self.max_heave, self.max_heave)

        u = self.align_fwd_scale * u_des - self.k_vel * float(s.vel_b[0].item())
        v = self.align_fwd_scale * v_des - self.k_vel * float(s.vel_b[1].item())
        w = self.align_fwd_scale * w_des - self.k_vel * float(s.vel_b[2].item())
        r = self.k_yaw * s.yaw_err - self.k_yaw_rate * s.yaw_rate

        u = self._clampf(u, -self.max_fwd, self.max_fwd)
        v = self._clampf(v, -self.max_lat, self.max_lat)
        w = self._clampf(w, -self.max_heave, self.max_heave)
        r = self._clampf(r, -self.max_yaw, self.max_yaw)
        u, v, r = self._apply_yaw_priority(u, v, r, s.yaw_err)
        r = self._clampf(r, -self.max_yaw, self.max_yaw)
        return self._to_action(u, v, w, r)

    def _move(self, s: EnvState) -> torch.Tensor:
        u_des = self._clampf(self.k_pos_xy * float(torch.tanh(s.rel_b[0]).item()), -self.max_fwd, self.max_fwd)
        v_des = self._clampf(self.k_pos_xy * float(torch.tanh(s.rel_b[1]).item()), -self.max_lat, self.max_lat)
        w_des = self._clampf(self.k_pos_z * float(torch.tanh(s.rel_b[2]).item()), -self.max_heave, self.max_heave)

        u = u_des - self.k_vel * float(s.vel_b[0].item())
        v = v_des - self.k_vel * float(s.vel_b[1].item())
        w = w_des - self.k_vel * float(s.vel_b[2].item())
        r = self.k_yaw * s.yaw_err - self.k_yaw_rate * s.yaw_rate

        u = self._clampf(u, -self.max_fwd, self.max_fwd)
        v = self._clampf(v, -self.max_lat, self.max_lat)
        w = self._clampf(w, -self.max_heave, self.max_heave)
        r = self._clampf(r, -self.max_yaw, self.max_yaw)
        u, v, r = self._apply_yaw_priority(u, v, r, s.yaw_err)
        r = self._clampf(r, -self.max_yaw, self.max_yaw)
        return self._to_action(u, v, w, r)

    def _move_scaled(self, s: EnvState, speed_scale: float) -> torch.Tensor:
        scale = min(1.0, max(0.0, float(speed_scale)))
        cmd = self._move(s)
        cmd = cmd.clone()
        cmd[0] *= scale
        cmd[1] *= max(0.45, scale)
        return cmd

    def _move_path(self, s: EnvState, speed_scale: float = 1.0) -> torch.Tensor:
        if not self.path_priority_enable:
            return self._move_scaled(s, speed_scale)

        scale = min(1.0, max(0.0, float(speed_scale)))
        x_err = float(s.rel_b[0].item())
        y_err = float(s.rel_b[1].item())
        z_err = float(s.rel_b[2].item())
        yaw_err = float(s.yaw_err)

        u_des = self._clampf(self.k_pos_xy * math.tanh(x_err), 0.0, self.max_fwd)
        # Prioritize lateral convergence to the path waypoint over straight-ahead progress.
        v_des = self._clampf(
            self.path_lateral_gain * self.k_pos_xy * math.tanh(y_err),
            -self.max_lat,
            self.max_lat,
        )
        w_des = self._clampf(self.k_pos_z * math.tanh(z_err), -self.max_heave, self.max_heave)

        lateral_abs = abs(y_err)
        yaw_abs = abs(yaw_err)
        u_scale = 1.0 - self.path_lateral_slow_gain * lateral_abs - self.path_yaw_slow_gain * yaw_abs
        u_scale = min(1.0, max(self.path_min_u_scale, u_scale)) * scale

        u = u_scale * u_des - self.k_vel * float(s.vel_b[0].item())
        v = v_des - self.k_vel * float(s.vel_b[1].item())
        w = w_des - self.k_vel * float(s.vel_b[2].item())
        r = self.path_yaw_gain * self.k_yaw * yaw_err - self.k_yaw_rate * s.yaw_rate

        u = self._clampf(u, -0.10 * self.max_fwd, self.max_fwd)
        v = self._clampf(v, -self.max_lat, self.max_lat)
        w = self._clampf(w, -self.max_heave, self.max_heave)
        r = self._clampf(r, -self.max_yaw, self.max_yaw)
        return self._to_action(u, v, w, r)

    def _path_target_world(self, path_w: torch.Tensor, pos_w: torch.Tensor, n_look: int) -> torch.Tensor:
        if path_w.ndim != 2 or path_w.shape[0] <= 1:
            return path_w[-1]

        look_dist = float(max(0.0, self.path_lookahead_dist))
        if look_dist <= 1e-4:
            d = torch.norm(path_w - pos_w.unsqueeze(0), dim=-1)
            nearest = int(torch.argmin(d).item())
            return path_w[min(int(path_w.shape[0]) - 1, max(0, nearest + n_look))]

        seg = path_w[1:] - path_w[:-1]
        seg_len = torch.norm(seg[:, :2], dim=-1).clamp_min(1e-6)
        rel = pos_w.unsqueeze(0) - path_w[:-1]
        t = ((rel[:, :2] * seg[:, :2]).sum(dim=-1) / (seg_len * seg_len)).clamp(0.0, 1.0)
        proj = path_w[:-1] + t.unsqueeze(-1) * seg
        d_proj = torch.norm(proj[:, :2] - pos_w[:2].unsqueeze(0), dim=-1)
        seg_idx = int(torch.argmin(d_proj).item())

        remaining = look_dist
        cur = proj[seg_idx]
        for j in range(seg_idx, int(path_w.shape[0]) - 1):
            end = path_w[j + 1]
            span = torch.norm((end - cur)[:2]).item()
            if span >= remaining:
                alpha = remaining / max(span, 1e-6)
                return cur + alpha * (end - cur)
            remaining -= span
            cur = end
        return path_w[-1]

    def _align_simple(self, s: EnvState) -> torch.Tensor:
        # Pure heading alignment + depth hold (no x/y translation).
        u = -0.5 * self.k_vel * float(s.vel_b[0].item())
        v = -0.5 * self.k_vel * float(s.vel_b[1].item())
        w_des = self._clampf(self.k_pos_z * float(torch.tanh(0.9 * s.rel_b[2]).item()), -self.max_heave, self.max_heave)
        w = w_des - self.k_vel * float(s.vel_b[2].item())
        r = self.k_yaw * s.yaw_err - self.k_yaw_rate * s.yaw_rate
        u = self._clampf(u, -0.20 * self.max_fwd, 0.20 * self.max_fwd)
        v = self._clampf(v, -0.20 * self.max_lat, 0.20 * self.max_lat)
        w = self._clampf(w, -self.max_heave, self.max_heave)
        r = self._clampf(r, -self.max_yaw, self.max_yaw)
        return self._to_action(u, v, w, r)

    def _move_simple(self, s: EnvState) -> torch.Tensor:
        # Surge-dominant tracking: move forward along current heading while gently correcting sway/heave/yaw.
        u_des = self._clampf(self.k_pos_xy * float(torch.tanh(0.9 * s.rel_b[0]).item()), 0.0, self.max_fwd)
        v_des = self._clampf(
            self.simple_move_sway_gain * self.k_pos_xy * float(torch.tanh(0.9 * s.rel_b[1]).item()),
            -self.max_lat,
            self.max_lat,
        )
        w_des = self._clampf(self.k_pos_z * float(torch.tanh(0.9 * s.rel_b[2]).item()), -self.max_heave, self.max_heave)
        u = u_des - self.k_vel * float(s.vel_b[0].item())
        v = v_des - self.k_vel * float(s.vel_b[1].item())
        w = w_des - self.k_vel * float(s.vel_b[2].item())
        r = self.simple_move_yaw_gain * self.k_yaw * s.yaw_err - self.k_yaw_rate * s.yaw_rate
        u = self._clampf(u, 0.0, self.max_fwd)
        v = self._clampf(v, -self.max_lat, self.max_lat)
        w = self._clampf(w, -self.max_heave, self.max_heave)
        r = self._clampf(r, -self.max_yaw, self.max_yaw)
        return self._to_action(u, v, w, r)

    def _avoid_simple(self, s: EnvState, turn_pref: float, heave_pref: float) -> torch.Tensor:
        turn_left = (turn_pref >= 0.0)
        lat_sign = 1.0 if turn_left else -1.0
        v = lat_sign * self.simple_avoid_lat - 0.4 * float(s.vel_b[1].item())
        if abs(heave_pref) < 1e-4:
            heave_pref = float(math.copysign(1.0, float(s.rel_b[2].item()))) if abs(float(s.rel_b[2].item())) > 0.10 else 0.0
        w = heave_pref * self.simple_avoid_heave + 0.2 * self.k_pos_z * float(torch.tanh(s.rel_b[2]).item()) - 0.5 * float(s.vel_b[2].item())
        u = self.simple_avoid_fwd - 0.3 * float(s.vel_b[0].item())
        r = lat_sign * self.simple_avoid_yaw + 0.2 * self.k_yaw * s.yaw_err - 0.4 * self.k_yaw_rate * s.yaw_rate
        u = self._clampf(u, -0.20 * self.max_fwd, self.max_fwd)
        v = self._clampf(v, -self.max_lat, self.max_lat)
        w = self._clampf(w, -self.max_heave, self.max_heave)
        r = self._clampf(r, -self.max_yaw, self.max_yaw)
        return self._to_action(u, v, w, r)

    def _stop_and_turn(self, s: EnvState, turn_pref: float = 0.0) -> torch.Tensor:
        turn_left = s.left_min > s.right_min if abs(turn_pref) < 1e-4 else (turn_pref > 0.0)
        r = self.avoid_yaw if turn_left else -self.avoid_yaw
        u = -self.stop_backoff
        v = 0.0
        w = -0.6 * float(s.vel_b[2].item())
        r = self._clampf(r + 0.25 * self.k_yaw * s.yaw_err - 0.35 * s.yaw_rate, -self.max_yaw, self.max_yaw)
        return self._to_action(u, v, w, r)

    def _avoid(self, s: EnvState, turn_pref: float = 0.0, heave_pref: float = 0.0, predictive: bool = False) -> torch.Tensor:
        turn_left = s.left_min > s.right_min if abs(turn_pref) < 1e-4 else (turn_pref > 0.0)
        r_avoid = self.avoid_yaw if turn_left else -self.avoid_yaw
        r_goal = 0.45 * self.k_yaw * s.yaw_err - self.k_yaw_rate * s.yaw_rate
        r = self._clampf(0.65 * r_avoid + 0.35 * r_goal, -self.max_yaw, self.max_yaw)
        if predictive:
            u = min(self.avoid_fwd, self.predictive_avoid_fwd)
            v = self.predictive_lateral_gain * (1.0 if turn_left else -1.0) - 0.4 * float(s.vel_b[1].item())
            w = (
                self.predictive_heave_gain * heave_pref
                + 0.3 * self.k_pos_z * float(torch.tanh(s.rel_b[2]).item())
                - 0.5 * float(s.vel_b[2].item())
            )
            v = self._clampf(v, -self.max_lat, self.max_lat)
        else:
            u = self.avoid_fwd
            v = 0.0
            w = 0.4 * self.k_pos_z * float(torch.tanh(s.rel_b[2]).item()) - 0.5 * float(s.vel_b[2].item())
        w = self._clampf(w, -self.max_heave, self.max_heave)
        return self._to_action(u, v, w, r)

    def _avoid_crossing_dynamic(self, s: EnvState, turn_pref: float, heave_pref: float = 0.0) -> torch.Tensor:
        lat_sign = 1.0 if turn_pref >= 0.0 else -1.0
        u = min(self.dynamic_crossing_avoid_fwd, self.predictive_avoid_fwd) - 0.35 * float(s.vel_b[0].item())
        v = lat_sign * abs(self.dynamic_crossing_avoid_lat) - 0.45 * float(s.vel_b[1].item())
        if abs(heave_pref) < 1e-4:
            heave_pref = float(math.copysign(1.0, float(s.rel_b[2].item()))) if abs(float(s.rel_b[2].item())) > 0.08 else 0.0
        w = heave_pref * (0.35 * self.max_heave) - 0.45 * float(s.vel_b[2].item())
        r = lat_sign * abs(self.dynamic_crossing_avoid_yaw) + 0.20 * self.k_yaw * s.yaw_err - 0.45 * self.k_yaw_rate * s.yaw_rate
        u = self._clampf(u, -0.35 * self.max_fwd, 0.20 * self.max_fwd)
        v = self._clampf(v, -self.max_lat, self.max_lat)
        w = self._clampf(w, -self.max_heave, self.max_heave)
        r = self._clampf(r, -self.max_yaw, self.max_yaw)
        return self._to_action(u, v, w, r)

    def _backup_and_turn(self, s: EnvState, turn_pref: float = 0.0, heave_pref: float = 0.0) -> torch.Tensor:
        turn_left = s.left_min > s.right_min if abs(turn_pref) < 1e-4 else (turn_pref > 0.0)
        lat_sign = 1.0 if turn_left else -1.0
        u = -abs(self.crowd_backoff_u) - 0.15 * float(s.vel_b[0].item())
        v = lat_sign * abs(self.crowd_backoff_v) - 0.25 * float(s.vel_b[1].item())
        if abs(heave_pref) < 1e-4:
            heave_pref = float(math.copysign(1.0, float(s.rel_b[2].item()))) if abs(float(s.rel_b[2].item())) > 0.08 else 0.0
        w = heave_pref * (0.35 * self.max_heave) + 0.20 * self.k_pos_z * float(torch.tanh(s.rel_b[2]).item()) - 0.35 * float(s.vel_b[2].item())
        r = lat_sign * abs(self.crowd_backoff_yaw) + 0.15 * self.k_yaw * s.yaw_err - 0.35 * self.k_yaw_rate * s.yaw_rate
        u = self._clampf(u, -self.max_fwd, 0.15 * self.max_fwd)
        v = self._clampf(v, -self.max_lat, self.max_lat)
        w = self._clampf(w, -self.max_heave, self.max_heave)
        r = self._clampf(r, -self.max_yaw, self.max_yaw)
        return self._to_action(u, v, w, r)

    def _escape_corridor(self, s: EnvState, turn_pref: float, heave_pref: float = 0.0) -> torch.Tensor:
        lat_sign = 1.0 if turn_pref >= 0.0 else -1.0
        if abs(heave_pref) < 1e-4:
            heave_pref = 1.0 if float(s.rel_b[2].item()) >= 0.0 else -1.0
        u = -abs(self.escape_backoff_u) - 0.12 * float(s.vel_b[0].item())
        v = lat_sign * abs(self.escape_lateral_v) - 0.20 * float(s.vel_b[1].item())
        w = heave_pref * abs(self.escape_heave) - 0.25 * float(s.vel_b[2].item())
        r = lat_sign * abs(self.escape_yaw) - 0.30 * self.k_yaw_rate * s.yaw_rate
        u = self._clampf(u, -self.max_fwd, 0.10 * self.max_fwd)
        v = self._clampf(v, -self.max_lat, self.max_lat)
        w = self._clampf(w, -self.max_heave, self.max_heave)
        r = self._clampf(r, -self.max_yaw, self.max_yaw)
        return self._to_action(u, v, w, r)

    def _lookahead_waypoint_state(self, env_id: int, fallback: EnvState) -> EnvState:
        n_look = int(max(0, self.path_lookahead_points))
        if n_look <= 0:
            return fallback
        prefer_overlay = bool(self.path_prefer_map_overlay) and hasattr(
            self.env, "_compute_map_overlay_path_local"
        )
        try:
            paths = getattr(self.env, "_paths", None)
            path_idx = getattr(self.env, "_path_idx", None)
            if prefer_overlay:
                raise RuntimeError("prefer map-overlay path")
            if paths is None or path_idx is None:
                return fallback
            if env_id < 0 or env_id >= len(paths):
                return fallback
            path = paths[env_id]
            if not path:
                return fallback
            pos_w = self.env.drone.pos[env_id, 0, :]
            rot_w = self.env.drone.rot[env_id, 0, :]
            path_w = torch.as_tensor(path, device=pos_w.device, dtype=pos_w.dtype)
            tgt_w = self._path_target_world(path_w, pos_w, n_look)
            rel_w = tgt_w - pos_w
            rel_b = _quat_rotate_inverse(rot_w.unsqueeze(0), rel_w.unsqueeze(0)).squeeze(0)
            yaw_err = float(math.atan2(float(rel_b[1].item()), float(rel_b[0].item())))
            return EnvState(
                rel_b=rel_b,
                yaw_err=yaw_err,
                vel_b=fallback.vel_b,
                yaw_rate=fallback.yaw_rate,
                depth=fallback.depth,
                roll=fallback.roll,
                pitch=fallback.pitch,
                min_range=fallback.min_range,
                front_min=fallback.front_min,
                left_min=fallback.left_min,
                right_min=fallback.right_min,
            )
        except Exception:
            pass

        # Fallback for HoverVelStableGoal: follow map overlay path if available.
        try:
            # Use the cached overlay path first so RViz and the BT controller agree.
            path_local = None
            if hasattr(self.env, "_maybe_refresh_overlay_path"):
                try:
                    path_local = self.env._maybe_refresh_overlay_path(env_id, force=False)
                except Exception:
                    path_local = None
            if path_local is None:
                path_local = getattr(self.env, "_map_last_overlay_path_local", None)
            if path_local is None and hasattr(self.env, "_compute_map_overlay_path_local"):
                try:
                    path_local = self.env._compute_map_overlay_path_local(env_id)
                except Exception:
                    path_local = None
            if path_local is None:
                return fallback
            if env_id < 0:
                return fallback

            pos_w = self.env.drone.pos[env_id, 0, :]
            rot_w = self.env.drone.rot[env_id, 0, :]
            env_off = self.env.envs_positions[env_id]

            path_w = []
            for p in path_local:
                if p is None or len(p) < 3:
                    continue
                # Overlay path is in env-local coordinates.
                path_w.append(
                    torch.tensor(
                        [
                            float(p[0]) + float(env_off[0].item()),
                            float(p[1]) + float(env_off[1].item()),
                            float(p[2]) + float(env_off[2].item()),
                        ],
                        device=pos_w.device,
                        dtype=pos_w.dtype,
                    )
                )
            if not path_w:
                return fallback

            pw = torch.stack(path_w, dim=0)
            tgt_w = self._path_target_world(pw, pos_w, n_look)

            rel_w = tgt_w - pos_w
            rel_b = _quat_rotate_inverse(rot_w.unsqueeze(0), rel_w.unsqueeze(0)).squeeze(0)
            yaw_err = float(math.atan2(float(rel_b[1].item()), float(rel_b[0].item())))
            return EnvState(
                rel_b=rel_b,
                yaw_err=yaw_err,
                vel_b=fallback.vel_b,
                yaw_rate=fallback.yaw_rate,
                depth=fallback.depth,
                roll=fallback.roll,
                pitch=fallback.pitch,
                min_range=fallback.min_range,
                front_min=fallback.front_min,
                left_min=fallback.left_min,
                right_min=fallback.right_min,
            )
        except Exception:
            return fallback

    def act(self, obs: torch.Tensor, final_goal_dist: torch.Tensor | None = None) -> torch.Tensor:
        n = int(obs.shape[0])
        out = torch.zeros((n, 4), device=obs.device, dtype=torch.float32)
        self._replan_env_ids = []
        self._last_debug = []
        self._step_count += 1
        if self._last_replan_step is None or int(self._last_replan_step.numel()) != n:
            self._last_replan_step = torch.full((n,), -10_000, dtype=torch.long, device=obs.device)
        if self._replan_pressure is None or int(self._replan_pressure.numel()) != n:
            self._replan_pressure = torch.zeros((n,), dtype=torch.long, device=obs.device)
        if self._last_final_dist is None or int(self._last_final_dist.numel()) != n:
            self._last_final_dist = torch.full((n,), float("inf"), dtype=torch.float32, device=obs.device)
        if self._no_progress_steps is None or int(self._no_progress_steps.numel()) != n:
            self._no_progress_steps = torch.zeros((n,), dtype=torch.long, device=obs.device)
        if self._prev_cmd is None or int(self._prev_cmd.shape[0]) != n:
            self._prev_cmd = torch.zeros((n, 4), dtype=torch.float32, device=obs.device)
        if self._simple_avoid_dir is None or int(self._simple_avoid_dir.numel()) != n:
            self._simple_avoid_dir = torch.zeros((n,), dtype=torch.float32, device=obs.device)
        if self._simple_avoid_hold is None or int(self._simple_avoid_hold.numel()) != n:
            self._simple_avoid_hold = torch.zeros((n,), dtype=torch.long, device=obs.device)
        if self._crowd_backup_hold is None or int(self._crowd_backup_hold.numel()) != n:
            self._crowd_backup_hold = torch.zeros((n,), dtype=torch.long, device=obs.device)
        if self._crowd_backup_dir is None or int(self._crowd_backup_dir.numel()) != n:
            self._crowd_backup_dir = torch.zeros((n,), dtype=torch.float32, device=obs.device)
        if self._escape_hold is None or int(self._escape_hold.numel()) != n:
            self._escape_hold = torch.zeros((n,), dtype=torch.long, device=obs.device)
        if self._escape_dir is None or int(self._escape_dir.numel()) != n:
            self._escape_dir = torch.zeros((n,), dtype=torch.float32, device=obs.device)
        if self._avoid_side_prev is None or int(self._avoid_side_prev.numel()) != n:
            self._avoid_side_prev = torch.zeros((n,), dtype=torch.float32, device=obs.device)
        if self._avoid_side_switches is None or int(self._avoid_side_switches.numel()) != n:
            self._avoid_side_switches = torch.zeros((n,), dtype=torch.long, device=obs.device)
        states = [self._parse_state(i, obs[i]) for i in range(n)]
        if self.simple_mode and (not self.simple_use_dynamic):
            dyn_risks = [
                DynamicRisk(
                    static_count=0,
                    dynamic_count=0,
                    mean_dynamic_speed=0.0,
                    min_static_dist=float("inf"),
                    min_dynamic_dist=float("inf"),
                    min_ttc=float("inf"),
                    ttc_obstacle_idx=-1,
                    imminent=False,
                    approaching=False,
                    turn_preference=0.0,
                    heave_preference=0.0,
                    proceed_safe=True,
                    speed_scale=1.0,
                    crossing_preference=0.0,
                    source="simple",
                )
                for _ in range(n)
            ]
        elif self.dynamic_from_observation_only:
            dyn_risks = self._compute_dynamic_risk_from_obs(states, obs.device)
        else:
            dyn_risks = self._compute_dynamic_risk(obs.device)

        for i in range(n):
            s = states[i]
            dyn = dyn_risks[i]
            dyn_min_ttc = float(dyn.min_ttc)
            dyn_ttc_finite = math.isfinite(dyn_min_ttc)

            fdist = float(final_goal_dist[i].item()) if final_goal_dist is not None else float("inf")
            s_goal = self._final_goal_state(i, obs.device, s)

            # Anti-stall: detect when final distance is not improving.
            prev_f = float(self._last_final_dist[i].item())
            if math.isfinite(fdist) and math.isfinite(prev_f):
                prog = prev_f - fdist
                if prog < self.no_progress_epsilon:
                    self._no_progress_steps[i] = self._no_progress_steps[i] + 1
                else:
                    self._no_progress_steps[i] = torch.clamp(self._no_progress_steps[i] - 2, min=0)
            else:
                self._no_progress_steps[i] = torch.zeros_like(self._no_progress_steps[i])
                prog = 0.0
            self._last_final_dist[i] = float(fdist)
            no_prog = int(self._no_progress_steps[i].item())

            stale_wp = (s.goal_dist <= self.stale_wp_dist) and (
                fdist > (self.goal_tolerance + self.stale_wp_margin)
            )
            want_replan = False
            replan_causes: list[str] = []
            if stale_wp:
                want_replan = True
                replan_causes.append("stale_wp")

            if no_prog >= self.no_progress_steps_replan:
                want_replan = True
                replan_causes.append("no_progress")
            stall_recover = (
                self.stall_recovery_enable
                and (no_prog >= self.stall_recovery_steps)
                and (fdist > self.goal_tolerance)
            )
            if stall_recover:
                want_replan = True
                replan_causes.append("stall_recovery")

            # When path/waypoint tracking stalls, steer by final-goal vector but keep normal BT branches.
            force_goal_nav = stale_wp or (no_prog >= self.no_progress_steps_force)
            s_nav = s_goal if force_goal_nav else s
            s_track = self._lookahead_waypoint_state(i, s_nav)
            path_lateral_err = abs(float(s_track.rel_b[1].item()))
            corridor_commit = (
                self.corridor_commit_enable
                and (fdist > self.goal_tolerance)
                and (s.front_min > self.corridor_commit_front_min)
                and (path_lateral_err < self.corridor_commit_lateral_err)
                and (abs(float(s_track.yaw_err)) < self.corridor_commit_yaw_err)
            )
            dynamic_safe_to_progress = (not bool(dyn.approaching)) or bool(dyn.proceed_safe)
            dynamic_safe_to_commit = dynamic_safe_to_progress and (
                (not bool(dyn.approaching))
                or (not dyn_ttc_finite)
                or (dyn_min_ttc >= self.progress_commit_dynamic_ttc_min)
            )
            dynamic_safe_to_closeout = dynamic_safe_to_progress and (
                (not bool(dyn.approaching))
                or (not dyn_ttc_finite)
                or (dyn_min_ttc >= self.goal_closeout_dynamic_ttc_min)
            )
            progress_commit = (
                self.progress_commit_enable
                and (fdist > self.goal_tolerance)
                and (no_prog >= self.progress_commit_steps)
                and (s.front_min > self.progress_commit_front_min)
                and (s.min_range > self.progress_commit_min_range)
                and (abs(float(s_track.yaw_err)) < self.progress_commit_yaw_err)
                and (not bool(dyn.imminent))
                and dynamic_safe_to_commit
            )
            path_blocked = (
                (s.front_min < self.avoid_dist)
                or ((s.min_range < self.side_block_dist) and not corridor_commit)
                or bool(dyn.imminent)
            )
            goal_closeout = (
                self.goal_closeout_enable
                and (self.goal_tolerance < fdist <= self.goal_closeout_dist)
                and (s.front_min > self.goal_closeout_front_min)
                and (s.min_range > self.goal_closeout_min_range)
                and (abs(float(s_goal.yaw_err)) < self.goal_closeout_yaw_err)
                and (not bool(dyn.imminent))
                and dynamic_safe_to_closeout
            )

            if self.path_only_mode:
                if fdist <= self.goal_tolerance:
                    zone = "goal"
                    cmd = self._hover(s)
                elif abs(s_track.yaw_err) > self.align_yaw_threshold:
                    zone = "path_align"
                    cmd = self._align(s_track)
                else:
                    zone = "path_move"
                    cmd = self._move_path(s_track)
            elif self.simple_mode:
                # Simple staged behavior:
                # goal -> avoid (if blocked) -> align yaw -> surge-dominant move.
                if fdist <= self.goal_tolerance:
                    zone = "goal"
                    cmd = self._hover(s)
                    self._simple_avoid_hold[i] = 0
                else:
                    blocked = (s.front_min < self.avoid_dist) or (s.min_range < self.side_block_dist)
                    hard_blocked = (s.front_min < self.stop_dist)
                    hold = int(self._simple_avoid_hold[i].item())
                    if blocked or hold > 0:
                        # Latch avoid direction for a few steps to prevent oscillation.
                        if hold <= 0:
                            turn_pref = 1.0 if s.left_min >= s.right_min else -1.0
                            self._simple_avoid_dir[i] = turn_pref
                            self._simple_avoid_hold[i] = self.simple_avoid_latch_steps
                        turn_pref = float(self._simple_avoid_dir[i].item())
                        if hard_blocked:
                            zone = "avoid_hard"
                        else:
                            zone = "avoid"
                        heave_pref = 0.0
                        cmd = self._avoid_simple(s_track, turn_pref=turn_pref, heave_pref=heave_pref)
                        self._simple_avoid_hold[i] = torch.clamp(self._simple_avoid_hold[i] - 1, min=0)
                        if self.force_replan_on_avoid and (s.front_min < self.replan_dist):
                            want_replan = True
                            replan_causes.append("avoid_blocked")
                    elif abs(s_track.yaw_err) > self.simple_align_yaw_threshold:
                        zone = "align"
                        cmd = self._align_simple(s_track)
                    else:
                        zone = "move"
                        cmd = self._move_simple(s_track)
            else:
                ttc_stop = dyn.imminent and dyn.min_ttc <= self.ttc_stop
                ttc_finite = math.isfinite(float(dyn.min_ttc))
                dynamic_evidence = (
                    (ttc_finite and float(dyn.min_ttc) < max(self.ttc_horizon, self.dynamic_go_ttc) + 1.5)
                    or (s.front_min < self.dynamic_slow_front_dist)
                )
                dynamic_go = (
                    self.dynamic_go_enable
                    and bool(dyn.approaching)
                    and bool(dyn.proceed_safe)
                    and (s.front_min > self.dynamic_go_front_dist)
                    and dynamic_evidence
                )
                ttc_avoid = (
                    ((dyn.approaching and dyn.min_ttc <= self.ttc_horizon) or dyn.imminent)
                    and not dynamic_go
                )
                crowd_hold = int(self._crowd_backup_hold[i].item())
                approaching_close = bool(dyn.approaching) and (
                    (math.isfinite(float(dyn.min_dynamic_dist)) and float(dyn.min_dynamic_dist) < self.escape_front_dist)
                    or (ttc_finite and float(dyn.min_ttc) <= self.ttc_replan)
                ) and not dynamic_go
                blocked_for_escape = (
                    (s.front_min < self.escape_front_dist)
                    or ((s.min_range < self.side_block_dist) and not corridor_commit)
                    or bool(dyn.imminent)
                    or approaching_close
                ) and not dynamic_go
                avoid_side = float(dyn.crossing_preference)
                if abs(avoid_side) < 1e-4:
                    avoid_side = float(dyn.turn_preference)
                if abs(avoid_side) < 1e-4 and blocked_for_escape:
                    avoid_side = 1.0 if s.left_min >= s.right_min else -1.0
                prev_side = float(self._avoid_side_prev[i].item())
                if blocked_for_escape and abs(avoid_side) > 1e-4:
                    if abs(prev_side) > 1e-4 and (avoid_side * prev_side) < 0.0:
                        self._avoid_side_switches[i] = self._avoid_side_switches[i] + 1
                    else:
                        self._avoid_side_switches[i] = torch.clamp(self._avoid_side_switches[i] - 1, min=0)
                    self._avoid_side_prev[i] = avoid_side
                elif not blocked_for_escape:
                    self._avoid_side_switches[i] = torch.clamp(self._avoid_side_switches[i] - 1, min=0)
                crowded_now = (
                    self.crowd_backup_enable
                    and (int(dyn.dynamic_count) >= self.crowd_dyn_min)
                    and (
                        (s.front_min < self.crowd_front_dist)
                        or ((s.min_range < self.side_block_dist) and not corridor_commit)
                        or (ttc_finite and float(dyn.min_ttc) <= self.crowd_ttc_trigger)
                    )
                    and not dynamic_go
                )
                escape_hold = int(self._escape_hold[i].item())
                escape_switch = int(self._avoid_side_switches[i].item()) >= self.escape_side_switches
                escape_stall = no_prog >= self.escape_no_progress_steps and blocked_for_escape
                escape_now = (
                    self.escape_enable
                    and (fdist > self.goal_tolerance)
                    and (escape_hold > 0 or escape_switch or escape_stall)
                )
                # IMPORTANT: stop/hover only when final global goal is reached.
                if fdist <= self.goal_tolerance:
                    zone = "goal"
                    cmd = self._hover(s)
                    self._crowd_backup_hold[i] = 0
                    self._escape_hold[i] = 0
                elif goal_closeout:
                    zone = "goal_closeout"
                    cmd = self._move_path(s_goal, self.goal_closeout_speed_scale)
                    self._crowd_backup_hold[i] = 0
                    self._escape_hold[i] = 0
                    want_replan = False
                    replan_causes = []
                    self._replan_pressure[i] = torch.zeros_like(self._replan_pressure[i])
                elif progress_commit:
                    zone = "progress_commit"
                    cmd = self._move_path(
                        s_track,
                        max(self.progress_commit_speed_scale, float(dyn.speed_scale)),
                    )
                    self._crowd_backup_hold[i] = 0
                    self._escape_hold[i] = 0
                    want_replan = False
                    replan_causes = []
                    self._replan_pressure[i] = torch.zeros_like(self._replan_pressure[i])
                elif dynamic_go:
                    zone = "dynamic_go"
                    cmd = self._move_path(s_track, dyn.speed_scale)
                    self._crowd_backup_hold[i] = 0
                    self._escape_hold[i] = 0
                elif s.front_min < self.stop_dist:
                    zone = "stop"
                    cmd = self._stop_and_turn(s)
                elif ttc_stop:
                    if self.dynamic_crossing_avoid_enable and abs(float(dyn.crossing_preference)) > 1e-4:
                        zone = "escape_crossing"
                        cmd = self._avoid_crossing_dynamic(
                            s,
                            turn_pref=avoid_side,
                            heave_pref=dyn.heave_preference,
                        )
                    else:
                        zone = "stop_ttc"
                        cmd = self._stop_and_turn(s, turn_pref=avoid_side)
                    if self.force_replan_on_avoid:
                        want_replan = True
                        replan_causes.append("ttc_stop")
                elif ttc_avoid:
                    if self.dynamic_crossing_avoid_enable and abs(float(dyn.crossing_preference)) > 1e-4:
                        zone = "avoid_crossing"
                        cmd = self._avoid_crossing_dynamic(
                            s,
                            turn_pref=avoid_side,
                            heave_pref=dyn.heave_preference,
                        )
                    else:
                        zone = "avoid_ttc"
                        cmd = self._avoid(
                            s,
                            turn_pref=avoid_side,
                            heave_pref=dyn.heave_preference,
                            predictive=True,
                        )
                    if self.force_replan_on_avoid and dyn.min_ttc <= self.ttc_replan:
                        want_replan = True
                        replan_causes.append("ttc_avoid")
                elif escape_now:
                    zone = "escape_corridor"
                    if escape_hold <= 0:
                        turn_pref = avoid_side
                        if abs(turn_pref) < 1e-4:
                            turn_pref = float(dyn.turn_preference)
                        if abs(turn_pref) < 1e-4:
                            turn_pref = 1.0 if s.left_min >= s.right_min else -1.0
                        self._escape_dir[i] = float(turn_pref)
                        self._escape_hold[i] = int(max(1, self.escape_steps))
                        self._avoid_side_switches[i] = 0
                        want_replan = True
                        replan_causes.append("escape_corridor")
                    turn_pref = float(self._escape_dir[i].item())
                    cmd = self._escape_corridor(s, turn_pref=turn_pref, heave_pref=dyn.heave_preference)
                    self._escape_hold[i] = torch.clamp(self._escape_hold[i] - 1, min=0)
                elif stall_recover:
                    # Prioritize heading recovery first; blind backup often keeps robot trapped.
                    if abs(s_track.yaw_err) > self.stall_recovery_align_yaw:
                        zone = "stall_align"
                        cmd = self._align(s_track)
                    elif (s.front_min < self.stall_recovery_front_dist) or (s.min_range < self.stall_recovery_front_dist):
                        zone = "stall_avoid"
                        cmd = self._avoid(
                            s,
                            turn_pref=avoid_side,
                            heave_pref=dyn.heave_preference,
                            predictive=True,
                        )
                    else:
                        zone = "stall_move"
                        cmd = self._move_path(s_track)
                elif crowd_hold > 0 or crowded_now:
                    zone = "crowd_backup"
                    if crowd_hold <= 0:
                        turn_pref = avoid_side
                        if abs(turn_pref) < 1e-4:
                            turn_pref = 1.0 if (s.left_min >= s.right_min) else -1.0
                        self._crowd_backup_dir[i] = float(turn_pref)
                        self._crowd_backup_hold[i] = int(max(1, self.crowd_backup_steps))
                        want_replan = True
                        replan_causes.append("crowd_backup")
                    turn_pref = float(self._crowd_backup_dir[i].item())
                    cmd = self._backup_and_turn(s, turn_pref=turn_pref, heave_pref=dyn.heave_preference)
                    self._crowd_backup_hold[i] = torch.clamp(self._crowd_backup_hold[i] - 1, min=0)
                elif s.front_min < self.avoid_dist or s.min_range < self.side_block_dist:
                    if corridor_commit and s.front_min >= self.avoid_dist:
                        zone = "corridor_commit"
                        cmd = self._move_path(
                            s_track,
                            max(self.corridor_commit_min_speed_scale, float(dyn.speed_scale)),
                        )
                    else:
                        zone = "avoid"
                        if abs(avoid_side) < 1e-4:
                            avoid_side = 1.0 if s.left_min >= s.right_min else -1.0
                        cmd = self._avoid(
                            s_track,
                            turn_pref=avoid_side,
                            heave_pref=dyn.heave_preference,
                            predictive=True,
                        )
                        if self.force_replan_on_avoid and (
                            (s.front_min < self.replan_dist) or (s.min_range < self.side_block_dist)
                        ):
                            want_replan = True
                            replan_causes.append("avoid_range")
                elif abs(s_track.yaw_err) > self.align_yaw_threshold:
                    zone = "align"
                    cmd = self._align(s_track)
                else:
                    zone = "move"
                    cmd = self._move_path(s_track)

            # Replan hysteresis: require persistent pressure + cooldown.
            if want_replan:
                self._replan_pressure[i] = self._replan_pressure[i] + 1
            else:
                self._replan_pressure[i] = torch.clamp(self._replan_pressure[i] - 1, min=0)
            since = self._step_count - int(self._last_replan_step[i].item())
            if (
                int(self._replan_pressure[i].item()) >= self.replan_consecutive_trigger_steps
                and since >= self.min_replan_interval_steps
            ):
                self._replan_env_ids.append(i)
                if self.replan_live_debug:
                    reason = "+".join(replan_causes) if replan_causes else "unspecified"
                    ttc_val = float(dyn.min_ttc)
                    obst_idx = int(dyn.ttc_obstacle_idx)
                    pb = getattr(self.env, "progress_buf", None)
                    if pb is not None and int(pb.numel()) > i:
                        step_env = int(pb[i].item())
                    else:
                        step_env = int(self._step_count)
                    print(
                        "[BT][replan]",
                        f"env={i}",
                        f"step={step_env}",
                        f"reason={reason}",
                        f"source={dyn.source}",
                        f"pred_ttc={ttc_val:.2f}" if math.isfinite(ttc_val) else "pred_ttc=inf",
                        f"obst_idx={obst_idx}" if obst_idx >= 0 else "obst_idx=na",
                        f"front_min={s.front_min:.3f}",
                        f"goal_dist={fdist:.3f}",
                        f"zone={zone}",
                    )
                self._last_replan_step[i] = self._step_count
                self._replan_pressure[i] = 0

            cmd_now = torch.clamp(cmd.to(obs.device), -1.0, 1.0)
            prev_cmd = self._prev_cmd[i]
            max_d = max(0.0, float(self.cmd_max_delta))
            if max_d > 0.0:
                delta = torch.clamp(cmd_now - prev_cmd, -max_d, max_d)
                cmd_now = prev_cmd + delta
            alpha = min(1.0, max(0.0, float(self.cmd_smooth_alpha)))
            cmd_now = (1.0 - alpha) * prev_cmd + alpha * cmd_now
            out[i] = cmd_now
            self._prev_cmd[i] = cmd_now
            self._last_debug.append(
                {
                    "zone": zone,
                    "goal_dist": s.goal_dist,
                    "final_goal_dist": fdist,
                    "front_min": s.front_min,
                    "min_range": s.min_range,
                    "left_min": s.left_min,
                    "right_min": s.right_min,
                    "yaw_err": s.yaw_err,
                    "path_rel_x": float(s_track.rel_b[0].item()),
                    "path_rel_y": float(s_track.rel_b[1].item()),
                    "path_rel_z": float(s_track.rel_b[2].item()),
                    "path_lateral_error": abs(float(s_track.rel_b[1].item())),
                    "path_yaw_err": float(s_track.yaw_err),
                    "path_blocked": bool(path_blocked),
                    "corridor_commit": bool(corridor_commit),
                    "progress_commit": bool(progress_commit),
                    "goal_closeout": bool(goal_closeout),
                    "roll": s.roll,
                    "pitch": s.pitch,
                    "min_ttc": dyn.min_ttc,
                    "imminent_dyn": bool(dyn.imminent),
                    "approaching_dyn": bool(dyn.approaching),
                    "dyn_count": int(dyn.dynamic_count),
                    "st_count": int(dyn.static_count),
                    "dyn_speed": dyn.mean_dynamic_speed,
                    "dyn_min_dist": dyn.min_dynamic_dist,
                    "st_min_dist": dyn.min_static_dist,
                    "dyn_proceed_safe": bool(dyn.proceed_safe),
                    "dyn_speed_scale": dyn.speed_scale,
                    "dyn_crossing_pref": dyn.crossing_preference,
                    "dyn_source": dyn.source,
                    "ttc_obstacle_idx": int(dyn.ttc_obstacle_idx),
                    "no_progress_steps": no_prog,
                    "progress_delta": float(prog),
                    "stale_wp": bool(stale_wp),
                    "force_goal_nav": bool(force_goal_nav),
                    "replan_pressure": int(self._replan_pressure[i].item()),
                    "crowd_backup_hold": int(self._crowd_backup_hold[i].item()),
                    "escape_hold": int(self._escape_hold[i].item()),
                    "avoid_side_switches": int(self._avoid_side_switches[i].item()),
                }
            )

        return out

    def pop_replan_env_ids(self, device: torch.device) -> torch.Tensor:
        if not self._replan_env_ids:
            return torch.empty(0, dtype=torch.long, device=device)
        env_ids = torch.tensor(sorted(set(self._replan_env_ids)), dtype=torch.long, device=device)
        self._replan_env_ids = []
        return env_ids


@hydra.main(version_base=None, config_path="../cfg", config_name="play_bluerov_behavior_tree_maze")
def main(cfg):
    OmegaConf.register_new_resolver("eval", eval)
    OmegaConf.resolve(cfg)
    OmegaConf.set_struct(cfg, False)
    cfg.mode = str(cfg.get("mode", "play"))
    cfg.task.mapping = cfg.task.get("mapping", {}) or {}
    use_external_mapper = bool(cfg.task.mapping.get("use_external_mapper", False))
    if use_external_mapper:
        # Prevent double-updating map state when external mapper is active.
        cfg.task.mapping.enable = False
        cfg.task.mapping.use_in_obs = False

    seed = int(cfg.get("seed", 0))
    _seed_everything(seed)

    from bluerov_manual.sim_app import init_app

    sim_app = init_app(cfg)

    # Delayed imports after SimulationApp creation.
    import bluerov_manual.envs  # noqa: F401
    from bluerov_manual.mapping import PersistentOccupancyMapper
    from marinegym.envs.isaac_env import IsaacEnv
    from marinegym.utils.torch import quat_rotate
    from bluerov_manual.isaaclab_ppo import TorchRlIsaacRslVecEnv

    env_class = IsaacEnv.REGISTRY[cfg.task.name]
    env = env_class(cfg, headless=cfg.headless)
    vec_env = TorchRlIsaacRslVecEnv(env, auto_reset=False)

    # Multi-env safety guard: sonar must be env-scoped to avoid cross-env hit contamination.
    bt_cfg = cfg.get("bt", {}) or {}
    force_filter_by_env_multi = bool(bt_cfg.get("force_filter_by_env_multi_env", True))
    auto_relax_filter_on_empty_hits = bool(
        bt_cfg.get("auto_relax_filter_by_env_on_empty_hits", False)
    )
    if force_filter_by_env_multi and int(env.num_envs) > 1:
        try:
            if hasattr(env, "sonar") and hasattr(env.sonar, "cfg"):
                if not bool(getattr(env.sonar.cfg, "filter_by_env", True)):
                    env.sonar.cfg.filter_by_env = True
                    print(
                        "[BT][multi-env] forcing sonar.filter_by_env=true "
                        f"for num_envs={int(env.num_envs)}"
                    )
                # Probe once: if strict env filtering yields no sonar hits at all, relax it.
                if auto_relax_filter_on_empty_hits:
                    try:
                        pos_w = env.drone.pos[:, 0, :]
                        rot_w = env.drone.rot[:, 0, :]
                        sonar_probe = env.sonar.scan(pos_w, rot_w)["ranges"]
                        max_r = float(getattr(env.sonar.cfg, "max_range", 10.0))
                        n_hit = int((sonar_probe < (max_r - 1e-3)).sum().item())
                        if n_hit <= 0:
                            env.sonar.cfg.filter_by_env = False
                            print(
                                "[BT][multi-env] no sonar hits with filter_by_env=true; "
                                "relaxing to filter_by_env=false"
                            )
                    except Exception:
                        pass
        except Exception:
            pass
    # If map-overlay path is used, update map for all envs in multi-env runs.
    if int(env.num_envs) > 1:
        try:
            if (
                bool(getattr(env, "mapping_enabled", False))
                and bool(getattr(env, "_map_path_overlay_enable", False))
                and not bool(getattr(env, "_map_update_all", False))
            ):
                env._map_update_all = True
                print(
                    "[BT][multi-env] forcing mapping.update_all=true "
                    "for per-env path-overlay consistency"
                )
        except Exception:
            pass

    controller = RobustMazeBTController(cfg.get("bt", {}) or {}, env)
    mapper = None
    if use_external_mapper:
        mapper_cfg = dict(cfg.task.mapping)
        mapper_cfg["enable"] = True
        mapper = PersistentOccupancyMapper(
            mapping_cfg=mapper_cfg,
            sonar_cfg=cfg.task.get("sonar", {}) or {},
            device=env.device,
            num_envs=env.num_envs,
        )

    num_trials = max(1, int(cfg.get("num_trials", 5)))
    max_steps = max(1, int(cfg.get("max_steps", env.max_episode_length)))
    log_every = max(1, int(cfg.get("log_every", 50)))

    goal_tolerance = float(cfg.get("goal_tolerance", -1.0))
    if goal_tolerance <= 0.0:
        goal_tolerance = float(getattr(env, "success_distance", 1.0))
    env_success_distance = float(getattr(env, "success_distance", goal_tolerance))
    bt_cfg = cfg.get("bt", {}) or {}
    force_success_done_on_eval_tol = bool(bt_cfg.get("force_success_done_on_eval_tol", True))
    manual_collision_enable = bool(bt_cfg.get("manual_collision_enable", True))
    manual_collision_obstacle_only = bool(bt_cfg.get("manual_collision_obstacle_only", True))
    manual_collision_dist = float(
        bt_cfg.get("manual_collision_dist", getattr(env, "collision_distance", 0.10))
    )
    print_termination_reasons = bool(bt_cfg.get("print_termination_reasons", True))

    wandb_run = None
    wandb_enable = bool(cfg.get("wandb", {}).get("enable", False))
    if wandb_enable:
        import wandb

        wandb_project = str(cfg.get("wandb", {}).get("project", "bluerov_bt_maze_play")).strip()
        wandb_entity = str(cfg.get("wandb", {}).get("username", "")).strip() or None
        wandb_group = str(cfg.get("wandb", {}).get("group", "")).strip() or None
        wandb_run_name = str(cfg.get("wandb", {}).get("run_name", "")).strip() or None
        wandb_job_type = str(cfg.get("wandb", {}).get("job_type", "bt_only")).strip() or "bt_only"
        tags_raw = cfg.get("wandb", {}).get("tags", [])
        if isinstance(tags_raw, str):
            wandb_tags = [t.strip() for t in tags_raw.split(",") if t.strip()]
        else:
            wandb_tags = [str(t).strip() for t in (tags_raw or []) if str(t).strip()]
        wandb_mode = str(cfg.get("wandb", {}).get("mode", "online")).strip().lower()
        if wandb_mode not in {"online", "offline", "disabled"}:
            wandb_mode = "online"
        wandb_run = wandb.init(
            project=wandb_project,
            entity=wandb_entity,
            group=wandb_group,
            name=wandb_run_name,
            job_type=wandb_job_type,
            tags=wandb_tags if wandb_tags else None,
            mode=wandb_mode,
            config=OmegaConf.to_container(cfg, resolve=True),
        )
        try:
            wandb.define_metric("paper/iteration")
            wandb.define_metric("paper/*", step_metric="paper/iteration")
        except Exception:
            pass

    print("\n" + "=" * 78)
    print("BLUEROV ROBUST BT MAZE PLAY")
    print("=" * 78)
    print(f"task={cfg.task.name} num_trials={num_trials} num_envs={env.num_envs} max_steps={max_steps}")
    print(f"goal_tolerance={goal_tolerance:.3f} log_every={log_every}")
    if use_external_mapper and mapper is not None:
        print(f"mapper_source=external save_every={mapper.save_every} save_dir={mapper.save_dir}")
    else:
        print("mapper_source=env_internal")
    print("=" * 78)

    total_attempts = 0
    total_success = 0
    total_rewards = 0.0
    total_final_dist = 0.0
    total_steps_taken = 0
    total_reason_counts = collections.defaultdict(int)

    for trial_i in range(num_trials):
        _ = vec_env.reset()
        controller.reset_internal_state(env.num_envs, env.device)
        if use_external_mapper and mapper is not None and mapper.reset_on_reset:
            mapper.reset()
        done = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        ep_reward = torch.zeros(env.num_envs, dtype=torch.float32, device=env.device)
        steps_taken = 0

        zone_counts = collections.defaultdict(int)
        ttc_sum = 0.0
        ttc_count = 0
        dyn_count_sum = 0.0
        dyn_speed_sum = 0.0
        dyn_stat_count = 0
        term_reason_counts = collections.defaultdict(int)

        for step in range(max_steps):
            obs_td = vec_env.get_observations().to(env.device)
            obs = obs_td["policy"]
            final_pos = env.drone.pos[:, 0, :]
            goal_pos = env.target_pos[:, 0, :]
            final_goal_dist = torch.norm(goal_pos - final_pos, dim=-1)
            with torch.inference_mode():
                actions = controller.act(obs, final_goal_dist=final_goal_dist)
            # Keep already-finished envs quiescent; avoids post-done drift/noise.
            if bool(done.any().item()):
                actions = actions.clone()
                actions[done] = 0.0

            # Optional proactive replan from BT safety state.
            try:
                replan_env_ids = controller.pop_replan_env_ids(env.device)
                if replan_env_ids.numel() > 0 and bool(done.any().item()):
                    alive_mask = ~done[replan_env_ids]
                    replan_env_ids = replan_env_ids[alive_mask]
                if replan_env_ids.numel() > 0 and hasattr(env, "_plan_paths"):
                    env._plan_paths(replan_env_ids)
                elif replan_env_ids.numel() > 0 and hasattr(env, "_maybe_refresh_overlay_path"):
                    # HoverVelStableGoal has no _plan_paths; force-refresh map overlay path per env.
                    for e in replan_env_ids.tolist():
                        try:
                            env._maybe_refresh_overlay_path(int(e), force=True)
                        except Exception:
                            continue
            except Exception:
                pass

            done_prev = done.clone()
            _, rewards, dones, _extras = vec_env.step(actions)
            rewards = rewards.to(env.device)
            dones = dones.to(env.device)
            time_outs = _extras.get("time_outs", torch.zeros(env.num_envs, device=env.device))
            time_outs = torch.as_tensor(time_outs, device=env.device).reshape(env.num_envs).to(torch.bool)

            if use_external_mapper and mapper is not None and mapper.enabled:
                scan_pos = env.drone.pos[:, 0, :]
                scan_rot = env.drone.rot[:, 0, :]
                valid_pose = torch.isfinite(scan_pos).all(dim=-1) & torch.isfinite(scan_rot).all(dim=-1)
                if not bool(valid_pose.all().item()):
                    scan_pos = scan_pos.clone()
                    scan_rot = scan_rot.clone()
                    scan_pos[~valid_pose] = 0.0
                    scan_rot[~valid_pose] = torch.tensor([1.0, 0.0, 0.0, 0.0], device=env.device)

                sonar_obs = env.sonar.scan(scan_pos, scan_rot)["ranges"]
                if hasattr(env, "_sonar_ranges_from_base_link"):
                    sonar_obs = env._sonar_ranges_from_base_link(sonar_obs)
                else:
                    try:
                        max_range = float(getattr(getattr(env, "sonar", None), "cfg", {}).max_range)
                        origin_offset = float(getattr(getattr(env, "sonar", None), "cfg", {}).origin_offset)
                        if origin_offset > 0.0:
                            sonar_obs = torch.where(
                                sonar_obs < (max_range - 1e-6),
                                torch.clamp(sonar_obs + origin_offset, max=max_range),
                                sonar_obs,
                            )
                    except Exception:
                        pass
                # If env provides its own sonar buffer (e.g., analytic fallback with maze support),
                # fuse conservatively by taking nearest return per ray.
                try:
                    env_last_sonar = getattr(env, "_last_sonar", None)
                    if env_last_sonar is not None:
                        env_last_sonar = torch.as_tensor(env_last_sonar, device=env.device)
                        if env_last_sonar.shape == sonar_obs.shape:
                            sonar_obs = torch.minimum(sonar_obs, env_last_sonar)
                except Exception:
                    pass
                dyn_centers_w = None
                dyn_radius = float(cfg.task.get("obstacles", {}).get("radius", 0.15))
                dyn_height = float(cfg.task.get("obstacles", {}).get("height", 0.30))
                if bool(getattr(env, "enable_obstacles", False)) and getattr(env, "obstacles", None) is not None:
                    try:
                        obst_pos_w, _ = env.obstacles.get_world_poses()
                        obst_pos_w = obst_pos_w.to(env.device)
                        if obst_pos_w.ndim == 2:
                            obst_pos_w = obst_pos_w.view(
                                env.num_envs,
                                int(getattr(env, "num_obstacles", 1)),
                                3,
                            )
                        dyn_centers_w = obst_pos_w
                    except Exception:
                        dyn_centers_w = None
                maze_centers_local = None
                maze_yaw = None
                maze_length = None
                maze_thickness = None
                if bool(getattr(env, "enable_maze", False)):
                    try:
                        if getattr(env, "maze_centers", None) is not None:
                            maze_centers_local = env.maze_centers.to(env.device)
                        if getattr(env, "maze_yaw", None) is not None:
                            maze_yaw = env.maze_yaw.to(env.device)
                        if hasattr(env, "maze_length"):
                            maze_length = float(env.maze_length)
                        if hasattr(env, "maze_thickness"):
                            maze_thickness = float(env.maze_thickness)
                    except Exception:
                        maze_centers_local = None
                        maze_yaw = None

                sonar_obs = mapper.inject_dynamic_obstacle_sonar_hits(
                    scan_pos_w=scan_pos,
                    scan_rot_w=scan_rot,
                    sonar_obs=sonar_obs,
                    env_offsets=env.envs_positions.to(env.device),
                    dyn_centers_w=dyn_centers_w,
                    dyn_radius=dyn_radius,
                    dyn_height=dyn_height,
                )
                env._last_map_sonar = sonar_obs.detach()

                depth_img_for_occ = None
                if mapper.use_depth and getattr(env, "depth_camera", None) is not None:
                    try:
                        if getattr(env, "depth_camera_view", None) is not None:
                            depth_cfg = getattr(env, "depth_cfg", cfg.task.get("depth_camera", {}) or {})
                            cam_offset = torch.tensor(
                                depth_cfg.get("offset", [0.2, 0.0, 0.0]),
                                device=env.device,
                                dtype=scan_pos.dtype,
                            ).unsqueeze(0).expand(scan_pos.shape[0], 3)
                            cam_pos = scan_pos + quat_rotate(scan_rot, cam_offset)
                            cam_rot = scan_rot
                            if (
                                hasattr(env, "_depth_rot_offset_q")
                                and env._depth_rot_offset_q is not None
                                and hasattr(env, "_quat_mul")
                            ):
                                cam_rot = env._quat_mul(scan_rot, env._depth_rot_offset_q.expand_as(scan_rot))
                            env.depth_camera_view.set_world_poses(positions=cam_pos, orientations=cam_rot)
                    except Exception:
                        pass

                    try:
                        imgs = env.depth_camera.get_images()
                        depth_cfg = getattr(env, "depth_cfg", cfg.task.get("depth_camera", {}) or {})
                        data_type = str(depth_cfg.get("data_type", "distance_to_camera"))
                        if data_type == "depth":
                            data_type = "distance_to_camera"
                        if imgs is not None and data_type in imgs:
                            depth_img = imgs[data_type].squeeze(1)
                            if depth_img.dim() == 2:
                                depth_img = depth_img.unsqueeze(0)
                            if depth_img.dim() == 3:
                                max_range = float(depth_cfg.get("max_range", mapper.max_range))
                                depth_img = depth_img.clamp(min=0.0, max=max_range)
                                if torch.isfinite(depth_img).all():
                                    depth_img_for_occ = depth_img
                    except Exception:
                        depth_img_for_occ = None

                mapper.update(
                    pos_w=scan_pos,
                    rot_w=scan_rot,
                    sonar_obs=sonar_obs,
                    env_offsets=env.envs_positions.to(env.device),
                    depth_img_for_occ=depth_img_for_occ,
                    dyn_centers_w=dyn_centers_w,
                    dyn_radius=dyn_radius,
                    dyn_height=dyn_height,
                    maze_centers_local=maze_centers_local,
                    maze_yaw=maze_yaw,
                    maze_length=maze_length,
                    maze_thickness=maze_thickness,
                )

                try:
                    env_id = int(max(0, min(env.num_envs - 1, mapper.save_env_id)))
                    r_local = env.drone.pos[env_id, 0, :] - env.envs_positions[env_id]
                    g_local = env.target_pos[env_id, 0, :] - env.envs_positions[env_id]
                    mapper.maybe_save_png(
                        robot_local_xy=(float(r_local[0].item()), float(r_local[1].item())),
                        goal_local_xy=(float(g_local[0].item()), float(g_local[1].item())),
                    )
                except Exception:
                    pass

            # Build explicit done reasons (including BT-side strict obstacle overlap).
            pos_now = env.drone.pos[:, 0, :]
            goal_now = env.target_pos[:, 0, :]
            final_goal_dist_now = torch.norm(goal_now - pos_now, dim=-1)
            success_now = final_goal_dist_now <= goal_tolerance
            success_env_now = final_goal_dist_now <= env_success_distance

            surface_now = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
            term_max_z = getattr(env, "termination_max_z", None)
            if term_max_z is not None:
                surface_now = pos_now[:, 2] > float(term_max_z)
            # Mirror env-side termination categories to reduce "done_unknown".
            out_of_tank_now = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
            misbehave_now = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
            try:
                pos_local_now = pos_now - env.envs_positions.to(env.device)
                grace_steps = int(getattr(env, "termination_grace_steps", 0))
                grace_mask = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
                if grace_steps > 0:
                    grace_mask = (env.progress_buf >= grace_steps)

                tx_min = getattr(env, "tank_x_min", None)
                tx_max = getattr(env, "tank_x_max", None)
                ty_min = getattr(env, "tank_y_min", None)
                ty_max = getattr(env, "tank_y_max", None)
                tz_min = getattr(env, "tank_z_min", None)
                tz_max = getattr(env, "tank_z_max", None)
                if None not in (tx_min, tx_max, ty_min, ty_max, tz_min, tz_max):
                    x = pos_local_now[:, 0]
                    y = pos_local_now[:, 1]
                    z = pos_local_now[:, 2]
                    out_of_tank_now = (
                        (x < float(tx_min))
                        | (x > float(tx_max))
                        | (y < float(ty_min))
                        | (y > float(ty_max))
                        | (z < float(tz_min))
                        | (z > float(tz_max))
                    ) & grace_mask

                term_min_z = getattr(env, "termination_min_z", None)
                term_max_z2 = getattr(env, "termination_max_z", None)
                term_max_dist = getattr(env, "termination_max_distance", None)
                if None not in (term_min_z, term_max_z2, term_max_dist):
                    z = pos_local_now[:, 2]
                    misbehave_now = (
                        (z < float(term_min_z))
                        | (z > float(term_max_z2))
                        | (final_goal_dist_now > float(term_max_dist))
                    ) & grace_mask
            except Exception:
                pass

            nonfinite_now = (
                ~torch.isfinite(env.drone.pos[:, 0, :]).all(dim=-1)
                | ~torch.isfinite(env.drone.rot[:, 0, :]).all(dim=-1)
                | ~torch.isfinite(env.drone.vel_w[:, 0, :]).all(dim=-1)
            )

            min_range_now = None
            collision_env_now = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
            if bool(getattr(env, "terminate_on_collision", False)):
                last_sonar = getattr(env, "_last_sonar", None)
                if last_sonar is not None:
                    min_range_now = torch.min(last_sonar.to(env.device), dim=1)[0]
                for attr in ("_last_depth_min", "_last_wall_dist", "_last_maze_dist", "_last_obst_dist"):
                    val = getattr(env, attr, None)
                    if val is None:
                        continue
                    val_t = torch.as_tensor(val, device=env.device).reshape(env.num_envs)
                    if min_range_now is None:
                        min_range_now = val_t
                    else:
                        min_range_now = torch.minimum(min_range_now, val_t)
                if min_range_now is not None:
                    collision_env_now = min_range_now <= float(getattr(env, "collision_distance", manual_collision_dist))

            obst_clear_now = getattr(env, "_last_obst_dist", None)
            if obst_clear_now is None and hasattr(env, "_obstacle_clearance"):
                try:
                    obst_clear_now = env._obstacle_clearance(pos_now)
                except Exception:
                    obst_clear_now = None
            if obst_clear_now is not None:
                obst_clear_now = torch.as_tensor(obst_clear_now, device=env.device).reshape(env.num_envs)

            collision_manual_now = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
            if manual_collision_enable:
                if obst_clear_now is not None:
                    collision_manual_now = collision_manual_now | (obst_clear_now <= manual_collision_dist)
                if (not manual_collision_obstacle_only) and (min_range_now is not None):
                    collision_manual_now = collision_manual_now | (min_range_now <= manual_collision_dist)

            # Keep BT evaluation stable: terminate as soon as eval tolerance is reached.
            if force_success_done_on_eval_tol:
                dones = dones | success_now
            dones = dones | collision_manual_now
            ep_reward += rewards
            done = done | dones
            steps_taken += 1

            newly_done = (~done_prev) & done
            if bool(newly_done.any().item()):
                done_ids = torch.where(newly_done)[0].tolist()
                env_reason_masks = getattr(env, "_last_done_reason_masks", None)

                def _env_reason_hit(name: str, env_idx: int) -> bool:
                    if not isinstance(env_reason_masks, dict) or name not in env_reason_masks:
                        return False
                    try:
                        mask = torch.as_tensor(env_reason_masks[name], device=env.device).reshape(env.num_envs, -1)
                        return bool(mask[env_idx].any().item())
                    except Exception:
                        return False

                for env_id in done_ids:
                    if bool(success_now[env_id].item()):
                        reason = "success"
                    elif bool(success_env_now[env_id].item()):
                        reason = "success_env_tol"
                    elif bool(collision_manual_now[env_id].item()):
                        reason = "collision_manual_obstacle"
                    elif _env_reason_hit("collision", env_id):
                        reason = "collision_env"
                    elif _env_reason_hit("ttc_collision", env_id):
                        reason = "ttc_collision"
                    elif _env_reason_hit("out_of_tank", env_id):
                        reason = "out_of_tank"
                    elif _env_reason_hit("misbehave", env_id):
                        reason = "misbehave"
                    elif _env_reason_hit("invalid_state", env_id):
                        reason = "nonfinite"
                    elif _env_reason_hit("truncated", env_id):
                        reason = "timeout"
                    elif bool(collision_env_now[env_id].item()):
                        reason = "collision_env"
                    elif bool(out_of_tank_now[env_id].item()):
                        reason = "out_of_tank"
                    elif bool(misbehave_now[env_id].item()):
                        reason = "misbehave"
                    elif bool(surface_now[env_id].item()):
                        reason = "surface"
                    elif bool(nonfinite_now[env_id].item()):
                        reason = "nonfinite"
                    elif bool(time_outs[env_id].item()):
                        reason = "timeout"
                    elif bool(dones[env_id].item()):
                        reason = "done_unknown"
                    else:
                        reason = "unknown"
                    term_reason_counts[reason] = term_reason_counts.get(reason, 0) + 1
                    if print_termination_reasons:
                        obs_clear = float(obst_clear_now[env_id].item()) if obst_clear_now is not None else float("nan")
                        print(
                            f"[trial {trial_i + 1:03d} term] env={env_id} "
                            f"reason={reason} final_dist={float(final_goal_dist_now[env_id].item()):.3f} "
                            f"obst_clear={obs_clear:.3f} "
                            f"(eval_tol={goal_tolerance:.3f}, env_tol={env_success_distance:.3f})"
                        )

            for dbg in controller._last_debug:
                zone_counts[dbg["zone"]] = zone_counts.get(dbg["zone"], 0) + 1
                ttc_val = float(dbg.get("min_ttc", float("inf")))
                if math.isfinite(ttc_val):
                    ttc_sum += ttc_val
                    ttc_count += 1
                dyn_count_sum += float(dbg.get("dyn_count", 0))
                dyn_speed_sum += float(dbg.get("dyn_speed", 0.0))
                dyn_stat_count += 1

            if controller._last_debug and (step % log_every == 0):
                sonar_hit_path = getattr(env.sonar, "last_hit_path", None)
                sonar_hit_dist = getattr(env.sonar, "last_hit_distance", None)
                sonar_hit_s = "none"
                if sonar_hit_path is not None:
                    sonar_hit_s = (
                        f"{sonar_hit_path}:{float(sonar_hit_dist):.2f}"
                        if sonar_hit_dist is not None
                        else str(sonar_hit_path)
                    )
                # Print one line per env (up to 2) for multi-env debugging.
                n_dbg = len(controller._last_debug)
                n_show = 1 if env.num_envs <= 1 else min(2, n_dbg)
                for env_j in range(n_show):
                    dj = controller._last_debug[env_j]
                    print(
                        f"[trial {trial_i + 1:03d} step {step:04d} env={env_j}] "
                        f"zone={dj['zone']} "
                        f"wp_dist={dj['goal_dist']:.3f} "
                        f"final_dist={dj['final_goal_dist']:.3f} "
                        f"front_min={dj['front_min']:.3f} "
                        f"ttc={dj.get('min_ttc', float('inf')):.2f} "
                        f"dyn={dj.get('dyn_count', 0)} "
                        f"yaw_err={dj['yaw_err']:+.3f} "
                        f"rp=({dj['roll']:+.3f},{dj['pitch']:+.3f}) "
                        f"first_hit={sonar_hit_s}"
                    )

            if bool(done.all().item()):
                break

        final_pos = env.drone.pos[:, 0, :]
        goal_pos = env.target_pos[:, 0, :]
        final_dist = torch.norm(goal_pos - final_pos, dim=-1)
        success = final_dist <= goal_tolerance
        still_alive = ~done
        if bool(still_alive.any().item()):
            term_reason_counts["max_steps"] = term_reason_counts.get("max_steps", 0) + int(still_alive.sum().item())

        total_attempts += int(env.num_envs)
        total_success += int(success.sum().item())
        total_rewards += float(ep_reward.sum().item())
        total_final_dist += float(final_dist.sum().item())
        total_steps_taken += int(steps_taken) * int(env.num_envs)
        for k, v in term_reason_counts.items():
            total_reason_counts[k] = total_reason_counts.get(k, 0) + int(v)

        print(
            f"[trial {trial_i + 1:03d}] "
            f"steps={steps_taken:04d} "
            f"success={int(success.sum().item())}/{env.num_envs} "
            f"mean_reward={float(ep_reward.mean().item()):+.3f} "
            f"mean_final_dist={float(final_dist.mean().item()):.3f} "
            f"zones={zone_counts} "
            f"term_reasons={dict(term_reason_counts)}"
        )
        if wandb_run is not None:
            tot_z = float(max(1, sum(zone_counts.values())))
            tot_term = float(max(1, sum(term_reason_counts.values())))
            mean_ttc = float(ttc_sum / max(1, ttc_count))
            mean_dyn_count = float(dyn_count_sum / max(1, dyn_stat_count))
            mean_dyn_speed = float(dyn_speed_sum / max(1, dyn_stat_count))
            wandb_run.log(
                {
                    "paper/iteration": float(trial_i + 1),
                    "paper/trial_index": int(trial_i + 1),
                    "paper/success_rate": float(success.float().mean().item()),
                    "paper/episode_return": float(ep_reward.mean().item()),
                    "paper/final_dist": float(final_dist.mean().item()),
                    "paper/episode_length": float(steps_taken),
                    "bt/zone_goal_ratio": float(zone_counts.get("goal", 0)) / tot_z,
                    "bt/zone_stop_ratio": float(zone_counts.get("stop", 0)) / tot_z,
                    "bt/zone_stop_ttc_ratio": float(zone_counts.get("stop_ttc", 0)) / tot_z,
                    "bt/zone_avoid_ratio": float(zone_counts.get("avoid", 0)) / tot_z,
                    "bt/zone_avoid_ttc_ratio": float(zone_counts.get("avoid_ttc", 0)) / tot_z,
                    "bt/zone_align_ratio": float(zone_counts.get("align", 0)) / tot_z,
                    "bt/zone_move_ratio": float(zone_counts.get("move", 0)) / tot_z,
                    "bt/mean_ttc": mean_ttc,
                    "bt/mean_dynamic_count": mean_dyn_count,
                    "bt/mean_dynamic_speed": mean_dyn_speed,
                    "bt/term_success_ratio": float(term_reason_counts.get("success", 0)) / tot_term,
                    "bt/term_success_env_tol_ratio": float(term_reason_counts.get("success_env_tol", 0)) / tot_term,
                    "bt/term_collision_manual_ratio": float(term_reason_counts.get("collision_manual_obstacle", 0)) / tot_term,
                    "bt/term_collision_env_ratio": float(term_reason_counts.get("collision_env", 0)) / tot_term,
                    "bt/term_timeout_ratio": float(term_reason_counts.get("timeout", 0)) / tot_term,
                    "bt/term_surface_ratio": float(term_reason_counts.get("surface", 0)) / tot_term,
                    "bt/term_nonfinite_ratio": float(term_reason_counts.get("nonfinite", 0)) / tot_term,
                    "bt/term_max_steps_ratio": float(term_reason_counts.get("max_steps", 0)) / tot_term,
                },
                step=int(trial_i + 1),
            )

    attempts_f = float(max(1, total_attempts))
    print("\n" + "=" * 78)
    print("SUMMARY")
    print("=" * 78)
    print(f"attempts={total_attempts} success_rate={total_success / attempts_f:.3f}")
    print(f"mean_reward={total_rewards / attempts_f:+.3f}")
    print(f"mean_final_dist={total_final_dist / attempts_f:.3f}")
    print(f"termination_reasons={dict(total_reason_counts)}")
    print("=" * 78)
    csv_out = str(cfg.get("csv_out", "")).strip()
    if csv_out:
        collision_count = int(total_reason_counts.get("collision_manual_obstacle", 0)) + int(
            total_reason_counts.get("collision_env", 0)
        )
        ttc_collision_count = int(total_reason_counts.get("ttc_collision", 0))
        success_count = int(total_success)
        outcome_count = success_count + collision_count + ttc_collision_count
        rate_mode = str(cfg.get("rate_denominator", "physical_outcomes")).strip() or "physical_outcomes"
        if rate_mode in {"physical_outcomes", "outcomes"}:
            rate_den_count = max(1, outcome_count)
        else:
            rate_den_count = max(1, int(total_attempts))
        excluded_non_outcome_count = max(0, int(total_attempts) - int(outcome_count))
        mean_episode_length = float(total_steps_taken / attempts_f)
        dt = float(cfg.get("dt", cfg.task.sim.get("dt", 0.02)))
        mean_time_sec = mean_episode_length * dt
        success_rate = float(success_count / float(rate_den_count))
        mean_time_to_success_sec = float("inf") if success_count <= 0 else float(mean_time_sec / max(success_rate, 1e-9))
        csv_row = {
            "label": str(cfg.get("csv_label", "bt_autonomous")).strip() or "bt_autonomous",
            "algo": "bt_autonomous",
            "checkpoint_path": "",
            "attempts": int(total_attempts),
            "success_rate": success_rate,
            "mean_reward": float(total_rewards / attempts_f),
            "mean_final_dist": float(total_final_dist / attempts_f),
            "mean_episode_length": mean_episode_length,
            "mean_time_sec": mean_time_sec,
            "mean_time_to_success_sec": mean_time_to_success_sec,
            "success_count": success_count,
            "rate_denominator": rate_mode,
            "rate_denominator_count": int(rate_den_count),
            "excluded_non_outcome_count": int(excluded_non_outcome_count),
            "collision_count": collision_count,
            "collision_rate": float(collision_count / float(rate_den_count)),
            "ttc_collision_count": ttc_collision_count,
            "ttc_collision_rate": float(ttc_collision_count / attempts_f),
            "collision_any_count": collision_count + ttc_collision_count,
            "collision_any_rate": float((collision_count + ttc_collision_count) / float(rate_den_count)),
            "mean_policy_latency_ms": 0.0,
            "mean_map_latency_ms": 0.0,
            "mean_planner_latency_ms": 0.0,
            "mean_obs_latency_ms": 0.0,
            "mean_step_latency_ms": 0.0,
            "planner_replans_total": 0,
            "planner_replans_per_episode": 0.0,
            "termination_reasons": json.dumps(dict(total_reason_counts), sort_keys=True),
        }
        _append_summary_csv(csv_out, csv_row)
        print(f"[bt-eval] appended CSV row: {csv_out}")
    if wandb_run is not None:
        tot_term_all = float(max(1, sum(total_reason_counts.values())))
        wandb_run.log(
            {
                "paper/iteration": float(num_trials + 1),
                "paper/summary/success_rate": float(total_success / attempts_f),
                "paper/summary/episode_return": float(total_rewards / attempts_f),
                "paper/summary/final_dist": float(total_final_dist / attempts_f),
                "paper/summary/trials": int(total_attempts),
                "paper/summary/term_collision_manual_ratio": float(total_reason_counts.get("collision_manual_obstacle", 0)) / tot_term_all,
                "paper/summary/term_collision_env_ratio": float(total_reason_counts.get("collision_env", 0)) / tot_term_all,
                "paper/summary/term_success_env_tol_ratio": float(total_reason_counts.get("success_env_tol", 0)) / tot_term_all,
                "paper/summary/term_timeout_ratio": float(total_reason_counts.get("timeout", 0)) / tot_term_all,
                "paper/summary/term_max_steps_ratio": float(total_reason_counts.get("max_steps", 0)) / tot_term_all,
            },
            step=int(num_trials + 1),
        )
        wandb_run.finish()

    try:
        sim_app.close()
    except Exception:
        pass


if __name__ == "__main__":
    main()
