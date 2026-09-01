import math
import os
import time
from collections import deque

import numpy as np
import torch
from tensordict.tensordict import TensorDict
from torchrl.data import Unbounded

import omni.isaac.core.utils.prims as prim_utils
from marinegym.sensors.camera import Camera
from marinegym.sensors.config import PinholeCameraCfg
from marinegym.utils.torch import (
    euler_to_quaternion,
    quat_rotate,
    quat_rotate_inverse,
    quaternion_to_euler,
)
from omni.isaac.core.prims import XFormPrimView

from ..navigation import (
    GlobalPlanner,
    GlobalPlannerConfig,
    Obstacle,
    VoronoiPlanner,
    VoronoiPlannerConfig,
)
from .hover_vel_policy import HoverVel


class HoverVelStableGoal(HoverVel):
    """Minimal velocity-control task: keep stable and move to a goal."""

    def __init__(self, cfg, headless):
        # Task-specific reward/goal settings.
        self.goal_cfg = cfg.task.get("goal", {}) or {}
        self.reset_cfg = cfg.task.get("reset", {}) or {}
        self.reset_stable_pose = bool(self.reset_cfg.get("stable_pose", True))
        self.reset_zero_velocity = bool(self.reset_cfg.get("zero_velocity", True))
        self.reset_yaw_deg = float(self.reset_cfg.get("yaw_deg", 0.0))
        self.reset_random_yaw = bool(self.reset_cfg.get("random_yaw", False))
        yaw_range = self.reset_cfg.get("yaw_range_deg", [-180.0, 180.0])
        yaw_lo = float(yaw_range[0])
        yaw_hi = float(yaw_range[1])
        self.reset_yaw_min_deg = min(yaw_lo, yaw_hi)
        self.reset_yaw_max_deg = max(yaw_lo, yaw_hi)
        self.reset_collision_free_sampling = bool(
            self.reset_cfg.get("collision_free_sampling", True)
        )
        self.reset_collision_clearance = float(self.reset_cfg.get("collision_clearance", 0.45))
        self.reset_collision_resample_tries = int(self.reset_cfg.get("collision_resample_tries", 24))
        maze_cfg = cfg.task.get("maze_walls", {}) or {}
        self.maze_feasibility_filter = bool(maze_cfg.get("feasibility_filter", False))
        self.maze_feasibility_tries = int(maze_cfg.get("feasibility_tries", 24))
        self.maze_feasibility_clearance = float(maze_cfg.get("feasibility_clearance", 0.45))
        self.maze_feasibility_resolution = float(maze_cfg.get("feasibility_resolution", 0.30))
        self.maze_feasibility_max_path_ratio = float(maze_cfg.get("feasibility_max_path_ratio", 0.0))
        self.success_distance = float(cfg.task.get("success_distance", 0.6))
        self.success_hold_steps = int(cfg.task.get("success_hold_steps", 10))
        self.success_min_steps = int(cfg.task.get("success_min_steps", 0))
        self.success_min_upright = float(cfg.task.get("success_min_upright", 0.0))
        self.success_min_heading = float(cfg.task.get("success_min_heading", -1.0))
        self.success_bonus = float(cfg.task.get("success_bonus", 10.0))
        self.reward_alive = float(cfg.task.get("reward_alive", 0.02))
        self.reward_progress_weight = float(cfg.task.get("reward_progress_weight", 5.0))
        self.reward_distance_weight = float(cfg.task.get("reward_distance_weight", 0.25))
        self.reward_goal_progress_weight = float(cfg.task.get("reward_goal_progress_weight", 0.0))
        self.goal_progress_clip = float(max(0.01, cfg.task.get("goal_progress_clip", 0.5)))
        self.reward_goal_proximity_weight = float(
            cfg.task.get("reward_goal_proximity_weight", 0.0)
        )
        self.reward_goal_distance_weight = float(
            cfg.task.get("reward_goal_distance_weight", 0.0)
        )
        self.reward_goal_proximity_scale = float(
            max(1e-3, cfg.task.get("reward_goal_proximity_scale", 1.0))
        )
        self.reward_upright_weight = float(cfg.task.get("reward_upright_weight", 0.5))
        self.reward_yaw_weight = float(cfg.task.get("reward_yaw_weight", 0.2))
        self.reward_tilt_penalty_weight = float(cfg.task.get("reward_tilt_penalty_weight", 0.05))
        self.reward_effort_penalty_weight = float(cfg.task.get("reward_effort_penalty_weight", 0.01))
        self.reward_aggressive_vel_penalty_weight = float(
            cfg.task.get("reward_aggressive_vel_penalty_weight", 0.0)
        )
        self.reward_clearance_penalty_weight = float(
            cfg.task.get("reward_clearance_penalty_weight", 0.0)
        )
        self.reward_ttc_penalty_weight = float(cfg.task.get("reward_ttc_penalty_weight", 0.0))
        self.clearance_soft_distance = float(cfg.task.get("clearance_soft_distance", 0.8))
        self.ttc_horizon = float(cfg.task.get("ttc_horizon", 3.0))
        self.ttc_min_forward_speed = float(cfg.task.get("ttc_min_forward_speed", 0.05))
        self.max_body_speed = float(cfg.task.get("max_body_speed", 0.5))
        safety_shield_cfg = cfg.task.get("safety_shield", {}) or {}
        self.safety_shield_terminate_on_clearance = bool(
            safety_shield_cfg.get("terminate_on_clearance", False)
        )
        self.safety_shield_termination_clearance = float(
            safety_shield_cfg.get("termination_clearance", 0.35)
        )
        self.reward_termination_penalty = float(cfg.task.get("reward_termination_penalty", 5.0))
        self.reward_timeout_penalty = float(cfg.task.get("reward_timeout_penalty", 0.0))
        # Optional path-following reward mode.
        self.path_follow_reward_enable = bool(cfg.task.get("path_follow_reward_enable", False))
        self.path_follow_use_goal_fallback = bool(cfg.task.get("path_follow_use_goal_fallback", True))
        self.path_follow_replan_interval = int(max(1, cfg.task.get("path_follow_replan_interval", 8)))
        self.path_follow_lookahead_steps = int(max(1, cfg.task.get("path_follow_lookahead_steps", 2)))
        self.path_progress_clip = float(max(0.1, cfg.task.get("path_progress_clip", 2.0)))
        self.reward_path_progress_weight = float(
            cfg.task.get("reward_path_progress_weight", self.reward_progress_weight)
        )
        self.reward_path_distance_weight = float(
            cfg.task.get("reward_path_distance_weight", self.reward_distance_weight)
        )
        self.reward_path_heading_weight = float(
            cfg.task.get("reward_path_heading_weight", self.reward_yaw_weight)
        )
        self.reward_path_crosstrack_weight = float(cfg.task.get("reward_path_crosstrack_weight", 0.0))
        # Optional fixed start-goal override for reproducible qualitative trials.
        # When enabled, reset sampling is bypassed and each reset uses these exact local coordinates.
        self.fixed_start_goal_enable = bool(cfg.task.get("fixed_start_goal_enable", False))
        self.fixed_start_local_cfg = cfg.task.get("fixed_start_local", [2.0, 1.0, -0.8])
        self.fixed_goal_local_cfg = cfg.task.get("fixed_goal_local", [18.0, 5.0, -0.8])
        self.fixed_start_goal_align_yaw = bool(cfg.task.get("fixed_start_goal_align_yaw", True))

        # Optional obstacle/path-aware observation augmentation for RL.
        obs_cfg = cfg.task.get("observation", {}) or {}
        self.obs_sonar_features_enable = bool(obs_cfg.get("sonar_features_enable", False))
        self.obs_sonar_num_sectors = int(obs_cfg.get("sonar_num_sectors", 12))
        self.obs_include_front_min = bool(obs_cfg.get("include_front_min", True))
        self.obs_include_front_ttc = bool(obs_cfg.get("include_front_ttc", True))
        self.obs_front_angle_deg = float(obs_cfg.get("front_angle_deg", 30.0))
        self.obs_use_path_lookahead = bool(obs_cfg.get("use_path_lookahead", False))
        self.obs_path_lookahead_points = int(obs_cfg.get("path_lookahead_points", 2))
        self.obs_path_fallback_to_goal = bool(obs_cfg.get("path_fallback_to_goal", True))
        self._obs_extra_dim = 0
        if self.obs_sonar_features_enable:
            self._obs_extra_dim += max(1, self.obs_sonar_num_sectors)
            if self.obs_include_front_min:
                self._obs_extra_dim += 1
            if self.obs_include_front_ttc:
                self._obs_extra_dim += 1
        if self.obs_use_path_lookahead:
            self._obs_extra_dim += 3 * max(1, self.obs_path_lookahead_points)

        # Safer/faster termination defaults for a simple task.
        term_cfg = cfg.task.get("termination", {}) or {}
        self.termination_enabled = bool(term_cfg.get("enable", True))
        self.termination_on_invalid_state = bool(term_cfg.get("terminate_on_invalid_state", False))
        self.termination_out_of_tank = bool(term_cfg.get("terminate_out_of_tank", True))
        self.termination_tank_margin = float(term_cfg.get("tank_margin", 0.0))
        self.termination_grace_steps = int(term_cfg.get("grace_steps", 20))
        self.termination_hasnan_grace_steps = int(term_cfg.get("hasnan_grace_steps", 80))
        self.termination_min_z = float(term_cfg.get("min_z", -2.0))
        self.termination_max_z = float(term_cfg.get("max_z", 0.2))
        self.termination_max_distance = float(term_cfg.get("max_distance", 30.0))

        spawn_cfg = cfg.task.get("spawn_ranges", {}) or {}
        tank_bounds_cfg = term_cfg.get("tank_bounds", {}) or {}

        def _axis_range(src: dict, key: str, default):
            raw = src.get(key, default)
            lo, hi = float(raw[0]), float(raw[1])
            return min(lo, hi), max(lo, hi)

        x_default = spawn_cfg.get("x_range", [0.0, 20.0])
        y_default = spawn_cfg.get("y_range", [0.0, 6.0])
        z_default = spawn_cfg.get("z_range", [self.termination_min_z, self.termination_max_z])
        tx_min, tx_max = _axis_range(tank_bounds_cfg, "x_range", x_default)
        ty_min, ty_max = _axis_range(tank_bounds_cfg, "y_range", y_default)
        tz_min, tz_max = _axis_range(tank_bounds_cfg, "z_range", z_default)
        margin = self.termination_tank_margin
        self.tank_x_min = tx_min - margin
        self.tank_x_max = tx_max + margin
        self.tank_y_min = ty_min - margin
        self.tank_y_max = ty_max + margin
        self.tank_z_min = tz_min - margin
        self.tank_z_max = tz_max + margin
        self._last_done_reason_debug = {}
        self.runtime_timing_enable = bool(cfg.task.get("runtime_timing_enable", False))
        self._rt_last_map_ms = 0.0
        self._rt_last_planner_ms = 0.0
        self._rt_last_obs_ms = 0.0
        self._rt_last_replans = 0

        # Optional sonar+camera persistent map hooks (controller/motion unchanged).
        self.depth_cfg = cfg.task.get("depth_camera", {}) or {}
        self.depth_enabled = bool(self.depth_cfg.get("enable", False))
        self.mapping_cfg = cfg.task.get("mapping", {}) or {}
        self.mapping_enabled = bool(self.mapping_cfg.get("enable", False))
        self.mapping_use_in_obs = bool(self.mapping_cfg.get("use_in_obs", False))
        self._map_grid_size = int(self.mapping_cfg.get("grid_size", 128))
        self._map_decay = float(self.mapping_cfg.get("decay", 0.0))
        self._map_occ_inc = float(self.mapping_cfg.get("occ_inc", 1.0))
        self._map_save_every = int(self.mapping_cfg.get("save_every", 0))
        self._map_save_dir = str(self.mapping_cfg.get("save_dir", "maps"))
        self._map_save_env_id = int(self.mapping_cfg.get("save_env_id", 0))
        self._map_use_depth = bool(self.mapping_cfg.get("use_depth", True))
        self._map_mark_free = bool(self.mapping_cfg.get("mark_free", True))
        self._map_free_dec = float(self.mapping_cfg.get("free_dec", 0.05))
        self._map_classify_dynamic_from_gt = bool(
            self.mapping_cfg.get("classify_dynamic_from_gt", True)
        )
        self._map_dyn_decay = float(self.mapping_cfg.get("dynamic_decay", max(self._map_decay, 0.05)))
        self._map_dyn_occ_inc = float(self.mapping_cfg.get("dynamic_occ_inc", self._map_occ_inc))
        self._map_dyn_free_dec = float(self.mapping_cfg.get("dynamic_free_dec", self._map_free_dec))
        self._map_dyn_classify_margin = float(self.mapping_cfg.get("dynamic_classify_margin", 0.15))
        self._map_dyn_stamp_from_gt = bool(self.mapping_cfg.get("dynamic_stamp_from_gt", True))
        self._map_dyn_stamp_radius_m = float(self.mapping_cfg.get("dynamic_stamp_radius_m", 0.35))
        self._map_dyn_stamp_inc = float(self.mapping_cfg.get("dynamic_stamp_inc", 0.35))
        self._map_static_hit_stamp_radius_m = float(
            self.mapping_cfg.get("static_hit_stamp_radius_m", 0.0)
        )
        self._map_static_hit_stamp_inc = float(self.mapping_cfg.get("static_hit_stamp_inc", 0.0))
        self._map_dynamic_hit_stamp_radius_m = float(
            self.mapping_cfg.get("dynamic_hit_stamp_radius_m", 0.0)
        )
        self._map_dynamic_hit_stamp_inc = float(self.mapping_cfg.get("dynamic_hit_stamp_inc", 0.0))
        self._map_show_dynamic = bool(self.mapping_cfg.get("show_dynamic", True))
        self._map_show_static = bool(self.mapping_cfg.get("show_static", True))
        self._map_self_clear_radius = float(self.mapping_cfg.get("self_clear_radius", 0.35))
        depth_row_band = self.mapping_cfg.get("depth_row_band", [0.35, 0.65])
        self._map_depth_row_min = float(depth_row_band[0])
        self._map_depth_row_max = float(depth_row_band[1])
        self._map_plot_pad_x = float(self.mapping_cfg.get("plot_pad_x", 0.5))
        self._map_plot_pad_y = float(self.mapping_cfg.get("plot_pad_y", 0.5))
        self._map_depth_occ_weight = float(self.mapping_cfg.get("depth_occ_weight", 0.25))
        self._map_depth_gate_with_sonar = bool(self.mapping_cfg.get("depth_gate_with_sonar", True))
        self._map_depth_sonar_gate_tol = float(self.mapping_cfg.get("depth_sonar_gate_tol", 0.6))
        self._map_dynamic_obstacle_sonar_fallback = bool(
            self.mapping_cfg.get("dynamic_obstacle_sonar_fallback", True)
        )
        self._map_maze_wall_sonar_fallback = bool(
            self.mapping_cfg.get("maze_wall_sonar_fallback", True)
        )
        self._map_maze_margin_xy = float(self.mapping_cfg.get("maze_wall_margin_xy", 0.06))
        self._map_maze_margin_z = float(self.mapping_cfg.get("maze_wall_margin_z", 0.08))
        self._map_min_hit_range = float(
            self.mapping_cfg.get(
                "min_hit_range",
                max(float(cfg.task.get("sonar", {}).get("min_range_filter", 0.2)), 0.3),
            )
        )
        self._map_max_samples = int(self.mapping_cfg.get("max_samples", 256))
        self._map_update_all = bool(self.mapping_cfg.get("update_all", False))
        self._map_update_env_id = int(self.mapping_cfg.get("update_env_id", self._map_save_env_id))
        self._map_reset_on_reset = bool(self.mapping_cfg.get("reset_on_reset", True))
        self._map_step = 0
        self._map_occ = None
        self._map_static = None
        self._map_dynamic = None
        self._map_x_min = 0.0
        self._map_x_max = 1.0
        self._map_y_min = 0.0
        self._map_y_max = 1.0
        self._map_z_min = -2.0
        self._map_z_max = 2.0
        self._map_max_range = float(cfg.task.get("sonar", {}).get("max_range", 10.0))
        # Optional map-only path overlay (does not drive control/actions).
        self._map_path_overlay_enable = bool(self.mapping_cfg.get("path_overlay_enable", False))
        self._map_path_overlay_planner = str(
            self.mapping_cfg.get("path_overlay_planner", "voronoi")
        ).strip().lower()
        if self._map_path_overlay_planner not in {"voronoi", "rrt"}:
            self._map_path_overlay_planner = "voronoi"
        self._map_path_overlay_occ_threshold = float(
            self.mapping_cfg.get("path_overlay_occ_threshold", 0.45)
        )
        _overlay_occ_step = float(self.mapping_cfg.get("path_overlay_occ_step", 0.07))
        self._map_path_overlay_occ_step = _overlay_occ_step if _overlay_occ_step > 0.0 else None
        self._map_path_overlay_goal_tolerance = float(
            self.mapping_cfg.get("path_overlay_goal_tolerance", self.success_distance)
        )
        self._map_path_overlay_resample_step = float(
            self.mapping_cfg.get("path_overlay_resample_step", 0.35)
        )
        self._map_path_overlay_grid_resolution = float(
            self.mapping_cfg.get("path_overlay_grid_resolution", 0.20)
        )
        self._map_path_overlay_clearance_eps = float(
            self.mapping_cfg.get("path_overlay_clearance_eps", 0.05)
        )
        self._map_path_overlay_rrt_step_size = float(
            self.mapping_cfg.get("path_overlay_rrt_step_size", 0.8)
        )
        self._map_path_overlay_rrt_max_iters = int(
            self.mapping_cfg.get("path_overlay_rrt_max_iters", 700)
        )
        self._map_path_overlay_rrt_goal_sample_rate = float(
            self.mapping_cfg.get("path_overlay_rrt_goal_sample_rate", 0.12)
        )
        self._map_path_overlay_rrt_neighbor_radius = float(
            self.mapping_cfg.get("path_overlay_rrt_neighbor_radius", 2.0)
        )
        self._map_path_overlay_wall_margin = float(
            self.mapping_cfg.get("path_overlay_wall_margin", 0.0)
        )
        self._map_path_overlay_min_clearance = float(
            self.mapping_cfg.get("path_overlay_min_clearance", 0.0)
        )
        self._map_path_overlay_occ_clearance = float(
            self.mapping_cfg.get(
                "path_overlay_occ_clearance",
                self._map_path_overlay_min_clearance,
            )
        )
        self._map_path_overlay_dynamic_predict_enable = bool(
            self.mapping_cfg.get("path_overlay_dynamic_predict_enable", True)
        )
        self._map_path_overlay_dynamic_horizon_sec = float(
            self.mapping_cfg.get("path_overlay_dynamic_horizon_sec", 2.5)
        )
        self._map_path_overlay_dynamic_samples = int(
            self.mapping_cfg.get("path_overlay_dynamic_samples", 5)
        )
        self._map_path_overlay_dynamic_vel_ema = float(
            self.mapping_cfg.get("path_overlay_dynamic_vel_ema", 0.35)
        )
        self._map_path_overlay_dynamic_min_speed = float(
            self.mapping_cfg.get("path_overlay_dynamic_min_speed", 0.05)
        )
        self._map_path_overlay_dynamic_speed_clip = float(
            self.mapping_cfg.get("path_overlay_dynamic_speed_clip", 2.0)
        )
        self._map_path_overlay_shortcut_enable = bool(
            self.mapping_cfg.get("path_overlay_shortcut_enable", False)
        )
        self._map_path_overlay_dynamic_inflate_add = float(
            self.mapping_cfg.get("path_overlay_dynamic_inflate_add", 0.03)
        )
        self._map_path_overlay_dynamic_inflate_per_mps = float(
            self.mapping_cfg.get("path_overlay_dynamic_inflate_per_mps", 0.12)
        )
        self._map_last_overlay_path_local = None
        self._map_overlay_prev_dyn_local = None
        self._map_overlay_vel_dyn_local = None
        self._map_overlay_prev_step = None

        self.depth_camera = None
        self.depth_camera_view = None
        self._depth_rot_offset_q = None
        self._depth_fail_count = 0
        self._depth_fail_limit = int(self.depth_cfg.get("fail_limit", 10))
        self._depth_disabled = False
        self._last_map_sonar = None
        self._map_last_dynamic_sonar_mask = None
        self._map_dyn_inject_ratio = 0.0
        self._map_dyn_inject_min = float("nan")
        # Sonar debug overlay on saved occupancy figures.
        self._map_show_sonar_hits = bool(self.mapping_cfg.get("show_sonar_hits", True))
        self._map_show_sonar_rays = bool(self.mapping_cfg.get("show_sonar_rays", False))
        self._map_sonar_hit_marker_size = float(self.mapping_cfg.get("sonar_hit_marker_size", 7.0))
        self._map_sonar_hit_alpha = float(self.mapping_cfg.get("sonar_hit_alpha", 0.85))
        self._map_sonar_ray_stride = int(self.mapping_cfg.get("sonar_ray_stride", 4))
        self._map_last_hit_points_local = None
        self._map_last_hit_points_dyn_local = None
        self._map_last_scan_origin_local = None
        self._map_last_hit_count = 0
        self._map_last_hit_ratio = 0.0

        super().__init__(cfg, headless)

        # Collision termination controls (separate from out-of-tank bounds).
        # Keep this explicit so HoverVelStableGoal can terminate on maze/obstacle contact-like proximity.
        self.terminate_on_collision = bool(
            cfg.task.get("terminate_on_collision", getattr(self, "terminate_on_collision", True))
        )
        self.collision_distance = float(
            cfg.task.get("collision_distance", getattr(self, "collision_distance", 0.10))
        )
        self.collision_check_mode = str(cfg.task.get("collision_check_mode", "proximity")).strip().lower()
        self.collision_overlap_clearance = float(cfg.task.get("collision_overlap_clearance", 0.0))
        self.ground_collision_z = cfg.task.get("ground_collision_z", None)
        if self.ground_collision_z is not None:
            self.ground_collision_z = float(self.ground_collision_z)
        self.terminate_on_ttc = bool(cfg.task.get("terminate_on_ttc", False))
        self.termination_ttc_threshold = float(cfg.task.get("termination_ttc_threshold", 0.25))

        if self.mapping_enabled:
            bounds_min = self.mapping_cfg.get("bounds_min", [0.0, 0.0, -2.0])
            bounds_max = self.mapping_cfg.get("bounds_max", [20.0, 6.0, 2.0])
            self._map_x_min = float(bounds_min[0])
            self._map_x_max = float(bounds_max[0])
            self._map_y_min = float(bounds_min[1])
            self._map_y_max = float(bounds_max[1])
            self._map_z_min = float(bounds_min[2]) if len(bounds_min) > 2 else self._map_z_min
            self._map_z_max = float(bounds_max[2]) if len(bounds_max) > 2 else self._map_z_max
            self._map_max_range = float(
                self.mapping_cfg.get("max_range", float(self.sonar.cfg.max_range))
            )
            self._map_occ = torch.zeros(
                (self.num_envs, self._map_grid_size, self._map_grid_size),
                device=self.device,
            )
            self._map_static = torch.zeros_like(self._map_occ)
            self._map_dynamic = torch.zeros_like(self._map_occ)

        if self.depth_enabled:
            data_type = str(self.depth_cfg.get("data_type", "distance_to_camera"))
            if data_type == "depth":
                data_type = "distance_to_camera"
            resolution = tuple(self.depth_cfg.get("resolution", [128, 72]))
            clip_range = self.depth_cfg.get("clip_range", None)
            extra_types = self.depth_cfg.get("extra_data_types", ["depth"])
            extra_types = [t for t in extra_types if t != "depth"]
            data_types = list({data_type, *extra_types})
            cam_cfg = PinholeCameraCfg(
                sensor_tick=0,
                resolution=resolution,
                data_types=data_types,
            )
            if clip_range is not None:
                cam_cfg.usd_params.clipping_range = (
                    float(clip_range[0]),
                    float(clip_range[1]),
                )
            self.depth_camera = Camera(cfg=cam_cfg)
            cam_paths = prim_utils.find_matching_prim_paths("/World/envs/.*/Camera_depth")
            for cam_path in cam_paths:
                self.depth_camera._define_usd_camera_attributes(cam_path)
            self.depth_camera.initialize(prim_paths_expr="/World/envs/.*/Camera_depth")
            self.depth_camera_view = XFormPrimView(
                prim_paths_expr="/World/envs/env_*/Camera_depth",
                reset_xform_properties=False,
            )
            self.depth_camera_view.initialize()
            rot_offset_deg = self.depth_cfg.get("rot_offset_deg", [0.0, -90.0, 0.0])
            rot_offset = torch.tensor(rot_offset_deg, device=self.device) * (torch.pi / 180.0)
            self._depth_rot_offset_q = euler_to_quaternion(rot_offset)

        # Per-env goal in world frame.
        self.target_pos = torch.zeros(self.num_envs, 1, 3, device=self.device)

        goal_fixed = self.goal_cfg.get("fixed", [10.0, 3.0, -1.0])
        self._goal_fixed_local = torch.tensor(
            [float(goal_fixed[0]), float(goal_fixed[1]), float(goal_fixed[2])],
            device=self.device,
        )
        self.goal_min_start_goal_dist = float(
            self.goal_cfg.get("min_start_goal_dist", cfg.task.get("min_start_goal_dist", 0.0))
        )
        self.goal_resample_tries = int(self.goal_cfg.get("resample_tries", 8))
        self.goal_collision_free_sampling = bool(
            self.goal_cfg.get("collision_free_sampling", True)
        )
        self.goal_collision_clearance = float(self.goal_cfg.get("collision_clearance", 0.45))

        goal_spawn = self.goal_cfg.get("spawn_ranges", {}) or {}
        if goal_spawn:
            def _axis_range(key, default):
                raw = goal_spawn.get(key, default)
                lo, hi = float(raw[0]), float(raw[1])
                return min(lo, hi), max(lo, hi)

            gx_min, gx_max = _axis_range("x_range", (self._goal_fixed_local[0], self._goal_fixed_local[0]))
            gy_min, gy_max = _axis_range("y_range", (self._goal_fixed_local[1], self._goal_fixed_local[1]))
            gz_min, gz_max = _axis_range("z_range", (self._goal_fixed_local[2], self._goal_fixed_local[2]))
            self.goal_spawn_low = torch.tensor([gx_min, gy_min, gz_min], device=self.device)
            self.goal_spawn_high = torch.tensor([gx_max, gy_max, gz_max], device=self.device)
        else:
            self.goal_spawn_low = None
            self.goal_spawn_high = None

        self.goal_randomize = bool(self.goal_cfg.get("randomize", True))
        self._prev_goal_dist = torch.zeros(self.num_envs, 1, device=self.device)
        self._prev_path_ref_dist = torch.zeros(self.num_envs, 1, device=self.device)
        self._success_counter = torch.zeros(self.num_envs, 1, dtype=torch.long, device=self.device)
        self._last_done_reason_rates = {
            "success": 0.0,
            "collision": 0.0,
            "ttc_collision": 0.0,
            "out_of_tank": 0.0,
            "misbehave": 0.0,
            "invalid_state": 0.0,
            "truncated": 0.0,
            "done_unknown": 0.0,
        }
        self._last_min_clearance_mean = float("nan")
        self._last_front_ttc_mean = float("nan")
        self._last_progress_mean = 0.0
        self._last_goal_progress_mean = 0.0
        self._last_clearance_penalty_mean = 0.0
        self._last_ttc_penalty_mean = 0.0
        self._last_path_available_rate = 0.0
        self._last_path_crosstrack_mean = 0.0
        self._last_path_ref_dist_mean = float("nan")
        self._path_overlay_cache_local = [None for _ in range(self.num_envs)]
        self._path_overlay_cache_step = torch.full(
            (self.num_envs,), -1_000_000, dtype=torch.long, device=self.device
        )

    def _set_specs(self):
        super()._set_specs()
        if self._obs_extra_dim <= 0:
            return
        base_dim = int(self.observation_spec["agents"]["observation"].shape[-1])
        obs_dim = base_dim + int(self._obs_extra_dim)
        # Preserve leading env dimension for vectorized runs (num_envs >= 1).
        self.observation_spec["agents"]["observation"] = Unbounded(
            (self.num_envs, obs_dim), device=self.device
        )

    @staticmethod
    def _quat_mul(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
        """Quaternion multiplication q = q1 * q2."""
        w1, x1, y1, z1 = torch.unbind(q1, dim=-1)
        w2, x2, y2, z2 = torch.unbind(q2, dim=-1)
        w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
        x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
        y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
        z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
        return torch.stack([w, x, y, z], dim=-1)

    def _design_scene(self):
        global_paths = super()._design_scene()
        if self.depth_enabled:
            for env_idx in range(self.num_envs):
                cam_path = f"/World/envs/env_{env_idx}/Camera_depth"
                if not prim_utils.is_prim_path_valid(cam_path):
                    prim_utils.create_prim(
                        prim_path=cam_path,
                        prim_type="Camera",
                        translation=(0.0, 0.0, 0.0),
                    )
        return global_paths

    def _points_blocked_local(
        self,
        env_ids: torch.Tensor,
        points_local: torch.Tensor,
        clearance: float,
    ) -> torch.Tensor:
        """Check overlap of local points with tank boundaries, maze walls and obstacles."""
        n = int(points_local.shape[0])
        if n <= 0:
            return torch.zeros((0,), dtype=torch.bool, device=self.device)

        p = points_local.to(self.device)
        eids = env_ids.to(self.device)
        clr = max(0.0, float(clearance))
        blocked = torch.zeros((n,), dtype=torch.bool, device=self.device)

        # Keep a configurable margin from tank boundaries.
        blocked |= (
            (p[:, 0] <= float(self.tank_x_min) + clr)
            | (p[:, 0] >= float(self.tank_x_max) - clr)
            | (p[:, 1] <= float(self.tank_y_min) + clr)
            | (p[:, 1] >= float(self.tank_y_max) - clr)
            | (p[:, 2] <= float(self.tank_z_min) + clr)
            | (p[:, 2] >= float(self.tank_z_max) - clr)
        )

        # Maze wall overlap check (oriented rectangles in local frame).
        if bool(getattr(self, "enable_maze", False)) and getattr(self, "maze_centers", None) is not None:
            try:
                centers_all = self.maze_centers[eids].to(self.device)  # [N,W,3]
                yaw_all = self.maze_yaw[eids].to(self.device)  # [N,W]
                L = 0.5 * float(getattr(self, "maze_length", 1.0)) + clr
                T = 0.5 * float(getattr(self, "maze_thickness", 0.2)) + clr
                H = 0.5 * float(getattr(self, "maze_height", 0.6)) + clr
                for i in range(n):
                    c = centers_all[i]
                    if c.numel() <= 0:
                        continue
                    active = (
                        (c[:, 0] > (self.tank_x_min - 2.0))
                        & (c[:, 0] < (self.tank_x_max + 2.0))
                        & (c[:, 1] > (self.tank_y_min - 2.0))
                        & (c[:, 1] < (self.tank_y_max + 2.0))
                    )
                    if not bool(active.any().item()):
                        continue
                    c = c[active]
                    y = yaw_all[i][active]
                    rel = p[i : i + 1, :2] - c[:, :2]
                    cy = torch.cos(y)
                    sy = torch.sin(y)
                    x_l = cy * rel[:, 0] + sy * rel[:, 1]
                    y_l = -sy * rel[:, 0] + cy * rel[:, 1]
                    z_ok = torch.abs(p[i, 2] - c[:, 2]) <= H
                    hit = (torch.abs(x_l) <= L) & (torch.abs(y_l) <= T) & z_ok
                    if bool(hit.any().item()):
                        blocked[i] = True
            except Exception:
                pass

        # Dynamic/static obstacle overlap check.
        if bool(getattr(self, "enable_obstacles", False)) and int(getattr(self, "num_obstacles", 0)) > 0:
            try:
                obst_local = None
                if getattr(self, "obstacles", None) is not None:
                    obst_pos_w, _ = self.obstacles.get_world_poses()
                    obst_pos_w = obst_pos_w.to(self.device)
                    if obst_pos_w.ndim == 2:
                        obst_pos_w = obst_pos_w.view(self.num_envs, self.num_obstacles, 3)
                    obst_local = obst_pos_w[eids] - self.envs_positions[eids].unsqueeze(1)
                if obst_local is None and getattr(self, "obst_center", None) is not None:
                    obst_local = self.obst_center[eids].to(self.device)
                if obst_local is not None:
                    r = float(self.obst_cfg.get("radius", 0.15)) + clr
                    h = 0.5 * float(self.obst_cfg.get("height", 0.30)) + clr
                    for i in range(n):
                        c = obst_local[i]
                        dxy = torch.norm(c[:, :2] - p[i : i + 1, :2], dim=-1)
                        dz = torch.abs(c[:, 2] - p[i, 2])
                        hit = (dxy <= r) & (dz <= h)
                        if bool(hit.any().item()):
                            blocked[i] = True
            except Exception:
                pass

        return blocked

    def _maze_path_feasible_local(
        self,
        env_id: int,
        start_local: torch.Tensor,
        goal_local: torch.Tensor,
    ) -> bool:
        """Coarse 2-D reachability test for sampled maze layouts."""
        if not bool(getattr(self, "enable_maze", False)) or int(getattr(self, "num_maze_walls", 0)) <= 0:
            return True

        res = max(0.10, float(self.maze_feasibility_resolution))
        x_min = float(self.tank_x_min) + float(self.maze_feasibility_clearance)
        x_max = float(self.tank_x_max) - float(self.maze_feasibility_clearance)
        y_min = float(self.tank_y_min) + float(self.maze_feasibility_clearance)
        y_max = float(self.tank_y_max) - float(self.maze_feasibility_clearance)
        if x_max <= x_min or y_max <= y_min:
            return False

        xs = torch.arange(x_min, x_max + 0.5 * res, res, device=self.device)
        ys = torch.arange(y_min, y_max + 0.5 * res, res, device=self.device)
        if xs.numel() < 2 or ys.numel() < 2:
            return False
        yy, xx = torch.meshgrid(ys, xs, indexing="ij")
        zz = torch.full_like(xx, float(start_local[2].item()))
        pts = torch.stack([xx.reshape(-1), yy.reshape(-1), zz.reshape(-1)], dim=-1)
        eids = torch.full((pts.shape[0],), int(env_id), device=self.device, dtype=torch.long)
        blocked = self._points_blocked_local(
            env_ids=eids,
            points_local=pts,
            clearance=float(self.maze_feasibility_clearance),
        ).reshape(int(ys.numel()), int(xs.numel()))

        def _idx(p: torch.Tensor) -> tuple[int, int]:
            ix = int(torch.argmin(torch.abs(xs - p[0])).item())
            iy = int(torch.argmin(torch.abs(ys - p[1])).item())
            return iy, ix

        sy, sx = _idx(start_local)
        gy, gx = _idx(goal_local)
        if bool(blocked[sy, sx].item()) or bool(blocked[gy, gx].item()):
            return False

        h, w = int(blocked.shape[0]), int(blocked.shape[1])
        visited = np.zeros((h, w), dtype=bool)
        dist_grid = np.full((h, w), np.inf, dtype=np.float32)
        queue: deque[tuple[int, int]] = deque([(sy, sx)])
        visited[sy, sx] = True
        dist_grid[sy, sx] = 0.0
        neighbors = [
            (-1, 0, 1.0),
            (1, 0, 1.0),
            (0, -1, 1.0),
            (0, 1, 1.0),
            (-1, -1, math.sqrt(2.0)),
            (-1, 1, math.sqrt(2.0)),
            (1, -1, math.sqrt(2.0)),
            (1, 1, math.sqrt(2.0)),
        ]
        blocked_np = blocked.detach().cpu().numpy().astype(bool)
        while queue:
            cy, cx = queue.popleft()
            if cy == gy and cx == gx:
                path_len = float(dist_grid[cy, cx]) * res
                max_ratio = float(self.maze_feasibility_max_path_ratio)
                if max_ratio > 0.0:
                    direct_len = float(torch.norm(goal_local[:2] - start_local[:2]).item())
                    if direct_len > 1e-6 and path_len > direct_len * max_ratio:
                        return False
                return True
            for dy, dx, step_cost in neighbors:
                ny, nx = cy + dy, cx + dx
                if ny < 0 or ny >= h or nx < 0 or nx >= w:
                    continue
                if visited[ny, nx] or blocked_np[ny, nx]:
                    continue
                visited[ny, nx] = True
                dist_grid[ny, nx] = dist_grid[cy, cx] + float(step_cost)
                queue.append((ny, nx))
        return False

    def _apply_maze_wall_poses(self, env_ids: torch.Tensor) -> None:
        if not bool(getattr(self, "enable_maze", False)) or getattr(self, "maze_walls", None) is None:
            return
        if int(getattr(self, "num_maze_walls", 0)) <= 0:
            return
        wall_pos = self.maze_centers[env_ids] + self.envs_positions[env_ids].unsqueeze(1)
        yaw = self.maze_yaw[env_ids].unsqueeze(-1)
        rpy = torch.zeros(len(env_ids), self.num_maze_walls, 3, device=self.device)
        rpy[..., 2:3] = yaw
        wall_rot = euler_to_quaternion(rpy)
        flat_indices = (
            env_ids.to(self.device).unsqueeze(1) * self.num_maze_walls
            + torch.arange(self.num_maze_walls, device=self.device).unsqueeze(0)
        ).reshape(-1)
        full_pos, full_rot = self.maze_walls.get_world_poses()
        full_pos = full_pos.clone()
        full_rot = full_rot.clone()
        full_pos[flat_indices] = wall_pos.reshape(-1, 3)
        full_rot[flat_indices] = wall_rot.reshape(-1, 4)
        self.maze_walls.set_world_poses(positions=full_pos, orientations=full_rot)

    def _resample_maze_for_feasible_paths(
        self,
        env_ids: torch.Tensor,
        start_local: torch.Tensor,
        goal_local: torch.Tensor,
    ) -> None:
        if not self.maze_feasibility_filter:
            return
        if not bool(getattr(self, "enable_maze", False)) or int(getattr(self, "num_maze_walls", 0)) <= 0:
            return
        if getattr(self, "wall_spawn_low", None) is None or getattr(self, "wall_spawn_high", None) is None:
            return

        active_walls = int(getattr(self, "_maze_active_num", self.num_maze_walls))
        active_walls = max(0, min(int(self.num_maze_walls), active_walls))
        if active_walls <= 0:
            return

        yaw_options = self.maze_cfg.get("yaw_options_deg", [0.0, 90.0])
        yaw_tensor = torch.tensor(yaw_options, device=self.device, dtype=torch.float32) * (torch.pi / 180.0)
        changed: list[int] = []
        for row, env_id_tensor in enumerate(env_ids.detach().to(self.device)):
            env_id = int(env_id_tensor.item())
            if self._maze_path_feasible_local(env_id, start_local[row], goal_local[row]):
                continue
            for _ in range(max(1, int(self.maze_feasibility_tries))):
                rand = torch.rand(active_walls, 3, device=self.device)
                self.maze_centers[env_id, :active_walls] = self.wall_spawn_low + rand * (
                    self.wall_spawn_high - self.wall_spawn_low
                )
                yaw_choices = torch.randint(0, len(yaw_options), (active_walls,), device=self.device)
                self.maze_yaw[env_id, :active_walls] = yaw_tensor[yaw_choices]
                if active_walls < int(self.num_maze_walls):
                    far_center = torch.tensor([1000.0, 1000.0, 0.0], device=self.device)
                    self.maze_centers[env_id, active_walls:] = far_center
                    self.maze_yaw[env_id, active_walls:] = 0.0
                if self._maze_path_feasible_local(env_id, start_local[row], goal_local[row]):
                    changed.append(env_id)
                    break

        if changed:
            changed_ids = torch.tensor(changed, device=self.device, dtype=torch.long)
            self._apply_maze_wall_poses(changed_ids)
            for env_id in changed:
                self._path_overlay_cache_local[int(env_id)] = None
            self._path_overlay_cache_step[changed_ids] = -1_000_000

    def _sample_goal_local(
        self,
        n: int,
        start_local: torch.Tensor | None = None,
        env_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.goal_randomize and self.goal_spawn_low is not None and self.goal_spawn_high is not None:
            rand = torch.rand(n, 3, device=self.device)
            goal_local = self.goal_spawn_low + rand * (self.goal_spawn_high - self.goal_spawn_low)

            if (
                start_local is not None
                and self.goal_min_start_goal_dist > 0.0
                and start_local.shape[0] == n
            ):
                for _ in range(max(1, self.goal_resample_tries)):
                    dist = torch.norm(goal_local - start_local, dim=-1)
                    bad = dist < self.goal_min_start_goal_dist
                    if not bool(bad.any().item()):
                        break
                    rand = torch.rand(int(bad.sum().item()), 3, device=self.device)
                    goal_local[bad] = self.goal_spawn_low + rand * (self.goal_spawn_high - self.goal_spawn_low)

            if (
                self.goal_collision_free_sampling
                and env_ids is not None
                and int(env_ids.numel()) == n
            ):
                for _ in range(max(1, self.goal_resample_tries)):
                    bad = self._points_blocked_local(
                        env_ids=env_ids,
                        points_local=goal_local,
                        clearance=self.goal_collision_clearance,
                    )
                    if not bool(bad.any().item()):
                        break
                    rand = torch.rand(int(bad.sum().item()), 3, device=self.device)
                    goal_local[bad] = self.goal_spawn_low + rand * (self.goal_spawn_high - self.goal_spawn_low)
            return goal_local
        return self._goal_fixed_local.unsqueeze(0).repeat(n, 1)

    def _reset_idx(self, env_ids: torch.Tensor):
        super()._reset_idx(env_ids)
        if env_ids.device != self.device:
            env_ids = env_ids.to(self.device)

        if self.mapping_enabled and self._map_reset_on_reset and self._map_occ is not None:
            self._map_occ[env_ids] = 0.0
            if self._map_static is not None and self._map_dynamic is not None:
                self._map_static[env_ids] = 0.0
                self._map_dynamic[env_ids] = 0.0

        if self.reset_stable_pose or self.reset_zero_velocity:
            pos_w, _ = self.drone.get_world_poses()
            pos_sel = pos_w[env_ids]

            if self.reset_stable_pose:
                # Force a level orientation at reset: roll=0, pitch=0, yaw=reset_yaw_deg.
                if self.reset_random_yaw:
                    yaw_deg = self.reset_yaw_min_deg + (
                        self.reset_yaw_max_deg - self.reset_yaw_min_deg
                    ) * torch.rand(len(env_ids), device=self.device, dtype=pos_sel.dtype)
                else:
                    yaw_deg = torch.full(
                        (len(env_ids),),
                        self.reset_yaw_deg,
                        device=self.device,
                        dtype=pos_sel.dtype,
                    )
                yaw_rad = yaw_deg * (torch.pi / 180.0)
                half = 0.5 * yaw_rad
                q = torch.zeros(len(env_ids), 4, device=self.device, dtype=pos_sel.dtype)
                q[:, 0] = torch.cos(half)
                q[:, 3] = torch.sin(half)
                self.drone.set_world_poses(positions=pos_sel, orientations=q, env_indices=env_ids)

            if self.reset_zero_velocity:
                zero_vel = torch.zeros_like(self.init_vels[env_ids])
                self.drone.set_velocities(zero_vel, env_ids)

        n = int(env_ids.shape[0])
        if self.reset_collision_free_sampling and n > 0:
            for _ in range(max(1, self.reset_collision_resample_tries)):
                start_local_now = self.drone.pos[env_ids, 0, :] - self.envs_positions[env_ids]
                bad = self._points_blocked_local(
                    env_ids=env_ids,
                    points_local=start_local_now,
                    clearance=self.reset_collision_clearance,
                )
                if not bool(bad.any().item()):
                    break
                bad_env_ids = env_ids[bad]
                if bad_env_ids.numel() <= 0:
                    break
                pos_new_local = self.init_pos_dist.sample((*bad_env_ids.shape, 1)).to(self.device)
                if pos_new_local.ndim == 3 and pos_new_local.shape[1] == 1:
                    pos_new_local = pos_new_local[:, 0, :]
                pos_new_world = pos_new_local + self.envs_positions[bad_env_ids]
                _, rot_w = self.drone.get_world_poses()
                rot_bad = rot_w[bad_env_ids]
                self.drone.set_world_poses(
                    positions=pos_new_world,
                    orientations=rot_bad,
                    env_indices=bad_env_ids,
                )
                if self.reset_zero_velocity:
                    self.drone.set_velocities(torch.zeros_like(self.init_vels[bad_env_ids]), bad_env_ids)

        if self.fixed_start_goal_enable and n > 0:
            try:
                start_local = torch.as_tensor(
                    self.fixed_start_local_cfg,
                    device=self.device,
                    dtype=self.drone.pos.dtype,
                ).reshape(1, 3).repeat(n, 1)
                goal_local = torch.as_tensor(
                    self.fixed_goal_local_cfg,
                    device=self.device,
                    dtype=self.drone.pos.dtype,
                ).reshape(1, 3).repeat(n, 1)

                start_world = start_local + self.envs_positions[env_ids]
                _, rot_w = self.drone.get_world_poses()
                rot_sel = rot_w[env_ids]
                if self.fixed_start_goal_align_yaw:
                    rel = goal_local - start_local
                    yaw = torch.atan2(rel[:, 1], rel[:, 0])
                    half = 0.5 * yaw
                    q = torch.zeros((n, 4), device=self.device, dtype=start_world.dtype)
                    q[:, 0] = torch.cos(half)
                    q[:, 3] = torch.sin(half)
                    rot_sel = q
                self.drone.set_world_poses(
                    positions=start_world,
                    orientations=rot_sel,
                    env_indices=env_ids,
                )
                self.drone.set_velocities(torch.zeros_like(self.init_vels[env_ids]), env_ids)
                self._resample_maze_for_feasible_paths(env_ids, start_local, goal_local)

                goal_world = goal_local + self.envs_positions[env_ids]
                self.target_pos[env_ids, 0, :] = goal_world
                self._set_target_poses(positions=goal_world, env_ids=env_ids)

                rel_w = goal_world - self.drone.pos[env_ids, 0, :]
                self._prev_goal_dist[env_ids] = torch.norm(rel_w, dim=-1, keepdim=True)
                self._prev_path_ref_dist[env_ids] = self._prev_goal_dist[env_ids]
                self._success_counter[env_ids] = 0
                for i in env_ids.detach().cpu().tolist():
                    self._path_overlay_cache_local[int(i)] = None
                self._path_overlay_cache_step[env_ids] = -1_000_000
                return
            except Exception:
                pass

        start_local = self.drone.pos[env_ids, 0, :] - self.envs_positions[env_ids]
        goal_local = self._sample_goal_local(n, start_local=start_local, env_ids=env_ids)
        self._resample_maze_for_feasible_paths(env_ids, start_local, goal_local)
        goal_world = goal_local + self.envs_positions[env_ids]

        self.target_pos[env_ids, 0, :] = goal_world
        self._set_target_poses(positions=goal_world, env_ids=env_ids)

        rel = goal_world - self.drone.pos[env_ids, 0, :]
        self._prev_goal_dist[env_ids] = torch.norm(rel, dim=-1, keepdim=True)
        self._prev_path_ref_dist[env_ids] = self._prev_goal_dist[env_ids]
        self._success_counter[env_ids] = 0
        for i in env_ids.detach().cpu().tolist():
            self._path_overlay_cache_local[int(i)] = None
        self._path_overlay_cache_step[env_ids] = -1_000_000

    def _map_world_to_grid(self, x: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Map local (x,y) in meters to grid indices (ix,iy)."""
        gx = (x - self._map_x_min) / max(self._map_x_max - self._map_x_min, 1e-6)
        gy = (y - self._map_y_min) / max(self._map_y_max - self._map_y_min, 1e-6)
        ix = (gx * (self._map_grid_size - 1)).long().clamp(0, self._map_grid_size - 1)
        iy = (gy * (self._map_grid_size - 1)).long().clamp(0, self._map_grid_size - 1)
        return ix, iy

    def _stamp_map_hits(
        self,
        layer: torch.Tensor,
        ix_hits: torch.Tensor,
        iy_hits: torch.Tensor,
        radius_m: float,
        inc: float,
    ) -> None:
        """Stamp sparse sensor endpoints over a small grid footprint."""
        if radius_m <= 0.0 or inc <= 0.0 or ix_hits.numel() == 0:
            return
        dx = (self._map_x_max - self._map_x_min) / max(self._map_grid_size - 1, 1)
        dy = (self._map_y_max - self._map_y_min) / max(self._map_grid_size - 1, 1)
        cell = max(min(dx, dy), 1e-6)
        r = max(1, int(math.ceil(float(radius_m) / cell)))
        pts = torch.stack([ix_hits.reshape(-1), iy_hits.reshape(-1)], dim=1)
        pts = torch.unique(pts, dim=0)
        for p in pts:
            cx = int(p[0].item())
            cy = int(p[1].item())
            ix0 = max(0, cx - r)
            ix1 = min(self._map_grid_size - 1, cx + r)
            iy0 = max(0, cy - r)
            iy1 = min(self._map_grid_size - 1, cy + r)
            if ix1 < ix0 or iy1 < iy0:
                continue
            ys = torch.arange(iy0, iy1 + 1, device=self.device)
            xs = torch.arange(ix0, ix1 + 1, device=self.device)
            yy, xx = torch.meshgrid(ys, xs, indexing="ij")
            d2 = (xx - cx) ** 2 + (yy - cy) ** 2
            mask = d2 <= (r * r)
            patch = layer[iy0 : iy1 + 1, ix0 : ix1 + 1]
            patch[mask] = torch.clamp(patch[mask] + float(inc), 0.0, 1.0)

    def _sonar_sector_mins(self, ranges: torch.Tensor, num_sectors: int) -> torch.Tensor:
        """Split rays into contiguous angular sectors and return per-sector min range."""
        if ranges.dim() != 2:
            return torch.full(
                (self.num_envs, max(1, num_sectors)),
                float(self.sonar.cfg.max_range),
                device=self.device,
                dtype=torch.float32,
            )
        n_envs, n_rays = ranges.shape
        n_sec = max(1, int(num_sectors))
        out = []
        for s in range(n_sec):
            i0 = int((s * n_rays) // n_sec)
            i1 = int(((s + 1) * n_rays) // n_sec)
            if i1 <= i0:
                i1 = min(n_rays, i0 + 1)
            seg = ranges[:, i0:i1]
            if seg.numel() == 0:
                out.append(
                    torch.full(
                        (n_envs,),
                        float(self.sonar.cfg.max_range),
                        device=self.device,
                        dtype=ranges.dtype,
                    )
                )
            else:
                out.append(torch.min(seg, dim=1).values)
        return torch.stack(out, dim=1)

    def _sonar_ranges_from_base_link(self, ranges: torch.Tensor) -> torch.Tensor:
        """Convert raw sonar raycast ranges to distances from the robot reference frame.

        The PhysX raycast starts each ray slightly along its direction to avoid self hits.
        Mapping, TTC, BT, and ROS overlays use the robot/base-link frame, so valid hit
        ranges must include that start offset. Max-range no-hit values stay unchanged.
        """
        if ranges.dim() != 2:
            return ranges
        origin_offset = float(getattr(self.sonar.cfg, "origin_offset", 0.0))
        if origin_offset <= 0.0:
            return ranges
        max_range = float(getattr(self.sonar.cfg, "max_range", 10.0))
        hit_mask = ranges < (max_range - 1e-6)
        return torch.where(hit_mask, torch.clamp(ranges + origin_offset, max=max_range), ranges)

    def _front_min_and_ttc_risk(self, ranges: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute front clearance and a bounded TTC risk score in [0,1]."""
        if ranges.dim() != 2:
            n = self.num_envs
            inf = torch.full((n,), float(self.sonar.cfg.max_range), device=self.device)
            return inf, torch.zeros((n,), device=self.device)

        angles = self.sonar.angles.to(self.device)
        half = float(abs(self.obs_front_angle_deg)) * (torch.pi / 180.0)
        front_mask = torch.abs(angles) <= max(1e-3, half)
        if not bool(front_mask.any().item()):
            front_min = torch.min(ranges, dim=1).values
        else:
            front_min = torch.min(ranges[:, front_mask], dim=1).values

        vel_b = torch.nan_to_num(self.drone.vel_b[:, 0, :3], nan=0.0, posinf=0.0, neginf=0.0)
        fwd_speed = torch.clamp(vel_b[:, 0], min=self.ttc_min_forward_speed)
        ttc = front_min / fwd_speed
        ttc = torch.nan_to_num(ttc, nan=self.ttc_horizon, posinf=self.ttc_horizon, neginf=0.0)
        ttc_risk = torch.clamp(1.0 - (ttc / max(1e-3, self.ttc_horizon)), 0.0, 1.0)
        return front_min, ttc_risk

    def _path_lookahead_features(self, scan_pos: torch.Tensor, scan_rot: torch.Tensor) -> torch.Tensor:
        """Return lookahead points in body frame, flattened as [E, 3*N]."""
        n_pts = max(1, int(self.obs_path_lookahead_points))
        feat = torch.zeros((self.num_envs, 3 * n_pts), device=self.device, dtype=scan_pos.dtype)
        if not self.obs_use_path_lookahead:
            return feat

        env_offsets = self.envs_positions.to(scan_pos.device)
        goal_local = self.target_pos[:, 0, :] - env_offsets

        def _fill_from_points(env_id: int, pts_local: torch.Tensor):
            rel = pts_local - (scan_pos[env_id : env_id + 1] - env_offsets[env_id : env_id + 1])
            rot_rep = scan_rot[env_id : env_id + 1].expand(rel.shape[0], -1)
            rel_b = quat_rotate_inverse(rot_rep, rel)
            feat[env_id] = rel_b.reshape(-1)

        for env_id in range(self.num_envs):
            used_overlay = False
            if self.mapping_enabled and self._map_path_overlay_enable:
                p_local = self._maybe_refresh_overlay_path(env_id)
                if p_local is not None:
                    p = torch.as_tensor(p_local, device=self.device, dtype=scan_pos.dtype)
                else:
                    p = None
                if p is not None and p.ndim == 2 and p.shape[0] >= 2:
                    cur_xy = (scan_pos[env_id, :2] - env_offsets[env_id, :2]).unsqueeze(0)
                    d2 = ((p[:, :2] - cur_xy) ** 2).sum(dim=1)
                    near = int(torch.argmin(d2).item())
                    idx = torch.clamp(
                        torch.arange(near + 1, near + 1 + n_pts, device=self.device),
                        max=p.shape[0] - 1,
                    )
                    pts = p[idx, :3]
                    _fill_from_points(env_id, pts)
                    used_overlay = True
            if not used_overlay and self.obs_path_fallback_to_goal:
                pts = goal_local[env_id : env_id + 1].expand(n_pts, 3)
                _fill_from_points(env_id, pts)
        return feat

    def _maybe_refresh_overlay_path(self, env_id: int, force: bool = False):
        if (
            (not self.mapping_enabled)
            or (not self._map_path_overlay_enable)
            or (self._map_occ is None)
            or env_id < 0
            or env_id >= self.num_envs
        ):
            return None
        cur_step = int(self._map_step)
        last_step = int(self._path_overlay_cache_step[env_id].item())
        cached = self._path_overlay_cache_local[env_id]
        need_replan = force or (cached is None) or ((cur_step - last_step) >= self.path_follow_replan_interval)
        if not need_replan:
            return cached
        t0 = time.perf_counter() if self.runtime_timing_enable else None
        plan = self._compute_map_overlay_path_local(env_id)
        if t0 is not None:
            self._rt_last_planner_ms += 1000.0 * (time.perf_counter() - t0)
            self._rt_last_replans += 1
        if plan is not None and len(plan) >= 2:
            cached = np.asarray(plan, dtype=np.float32)
        self._path_overlay_cache_local[env_id] = cached
        self._path_overlay_cache_step[env_id] = cur_step
        return cached

    def _path_tracking_terms(
        self,
        pos_w: torch.Tensor,
        yaw: torch.Tensor,
        goal_w: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # Defaults: goal-directed shaping.
        rel_goal = goal_w - pos_w
        ref_dist = torch.norm(rel_goal, dim=-1, keepdim=True)
        yaw_goal = torch.atan2(rel_goal[:, 1], rel_goal[:, 0])
        yaw_err = (yaw_goal - yaw + torch.pi) % (2 * torch.pi) - torch.pi
        yaw_reward = torch.cos(yaw_err).unsqueeze(-1)
        cross_track = torch.zeros_like(ref_dist)
        path_available = torch.zeros_like(ref_dist, dtype=torch.bool)

        if not self.path_follow_reward_enable:
            return ref_dist, yaw_reward, cross_track, path_available
        if (not self.mapping_enabled) or (not self._map_path_overlay_enable):
            return ref_dist, yaw_reward, cross_track, path_available

        env_offsets = self.envs_positions.to(pos_w.device)
        pos_local = pos_w - env_offsets
        ref_w = goal_w.clone()

        for env_id in range(self.num_envs):
            p_local = self._maybe_refresh_overlay_path(env_id)
            if p_local is None:
                continue
            p = torch.as_tensor(p_local, device=self.device, dtype=pos_w.dtype)
            if p.ndim != 2 or p.shape[0] < 2:
                continue

            cur_xy = pos_local[env_id, :2].unsqueeze(0)
            d2 = ((p[:, :2] - cur_xy) ** 2).sum(dim=1)
            near = int(torch.argmin(d2).item())
            look = min(p.shape[0] - 1, near + self.path_follow_lookahead_steps)
            ref_local = p[look, :3]
            ref_w[env_id] = ref_local + env_offsets[env_id]
            cross_track[env_id, 0] = torch.sqrt(torch.clamp(d2[near], min=0.0))

            i0 = max(0, look - 1)
            i1 = min(p.shape[0] - 1, look + 1)
            tangent = p[i1, :2] - p[i0, :2]
            if torch.norm(tangent) < 1e-6:
                tangent = p[-1, :2] - p[0, :2]
            if torch.norm(tangent) >= 1e-6:
                path_yaw = torch.atan2(tangent[1], tangent[0])
                yaw_err_i = (path_yaw - yaw[env_id] + torch.pi) % (2 * torch.pi) - torch.pi
                yaw_reward[env_id, 0] = torch.cos(yaw_err_i)
            path_available[env_id, 0] = True

        rel_ref = ref_w - pos_w
        ref_dist = torch.norm(rel_ref, dim=-1, keepdim=True)
        return ref_dist, yaw_reward, cross_track, path_available

    def _inject_dynamic_obstacle_sonar_hits(
        self,
        scan_pos_w: torch.Tensor,
        scan_rot_w: torch.Tensor,
        sonar_obs: torch.Tensor,
    ) -> torch.Tensor:
        """Analytic fallback: inject obstacle-circle intersections into sonar ranges."""
        self._map_last_dynamic_sonar_mask = None
        if not self._map_dynamic_obstacle_sonar_fallback:
            return sonar_obs
        if not getattr(self, "enable_obstacles", False):
            return sonar_obs
        try:
            env_pos = self.envs_positions.to(self.device)
            pos_local = scan_pos_w - env_pos
            obst_local = None
            if getattr(self, "obstacles", None) is not None:
                obst_pos_w, _ = self.obstacles.get_world_poses()
                obst_pos_w = obst_pos_w.to(self.device)
                if obst_pos_w.ndim == 2:
                    obst_pos_w = obst_pos_w.view(self.num_envs, self.num_obstacles, 3)
                obst_local = obst_pos_w - env_pos.unsqueeze(1)
            if obst_local is None and getattr(self, "obst_center", None) is not None:
                obst_local = self.obst_center.to(self.device)
            if obst_local is None or obst_local.numel() == 0:
                return sonar_obs
            yaw = quaternion_to_euler(scan_rot_w)[:, 2]
            angles = self.sonar.angles.to(self.device)
            theta = yaw.unsqueeze(1) + angles.unsqueeze(0)  # [E,R]
            dirs = torch.stack([torch.cos(theta), torch.sin(theta)], dim=-1)  # [E,R,2]
            o = pos_local[:, :2]  # [E,2]
            c = obst_local[:, :, :2]  # [E,O,2]
            r = float(self.obst_cfg.get("radius", 0.15))
            h = float(self.obst_cfg.get("height", 0.30))
            z_ok = (
                torch.abs(obst_local[:, :, 2] - pos_local[:, None, 2])
                <= (0.5 * h + r + 0.10)
            )  # [E,O]

            # Ray-circle intersection per env/ray/obstacle:
            # p(t)=o+t*d ; |p-c|^2=r^2 ; choose smallest t>0.
            m = o[:, None, None, :] - c[:, None, :, :]  # [E,1,O,2]
            d = dirs[:, :, None, :]  # [E,R,1,2]
            b = 2.0 * torch.sum(d * m, dim=-1)  # [E,R,O]
            cterm = torch.sum(m * m, dim=-1) - (r * r)  # [E,1,O]
            disc = b * b - 4.0 * cterm  # [E,R,O]
            valid = disc >= 0.0
            sqrt_disc = torch.sqrt(torch.clamp(disc, min=0.0))
            t1 = (-b - sqrt_disc) * 0.5
            t2 = (-b + sqrt_disc) * 0.5
            inf = torch.full_like(t1, float("inf"))
            t = torch.where(t1 > 0.0, t1, torch.where(t2 > 0.0, t2, inf))
            t = torch.where(valid, t, inf)
            t = torch.where(z_ok[:, None, :], t, inf)
            tmin = torch.min(t, dim=-1).values  # [E,R]
            tmin = torch.clamp(tmin, min=0.0, max=self._map_max_range)
            fused = torch.minimum(sonar_obs, tmin)
            injected = (fused + 1e-6) < sonar_obs
            self._map_last_dynamic_sonar_mask = injected.detach()
            self._map_dyn_inject_ratio = float(injected.float().mean().item())
            if bool(injected.any().item()):
                self._map_dyn_inject_min = float(fused[injected].min().item())
            else:
                self._map_dyn_inject_min = float("nan")
            return fused
        except Exception:
            self._map_last_dynamic_sonar_mask = None
            self._map_dyn_inject_ratio = 0.0
            self._map_dyn_inject_min = float("nan")
            return sonar_obs

    def _inject_maze_wall_sonar_hits(
        self,
        scan_pos_w: torch.Tensor,
        scan_rot_w: torch.Tensor,
        sonar_obs: torch.Tensor,
    ) -> torch.Tensor:
        """Analytic fallback: inject oriented-rectangle (maze wall) intersections into sonar."""
        if not self._map_maze_wall_sonar_fallback:
            return sonar_obs
        if not getattr(self, "enable_maze", False) or getattr(self, "num_maze_walls", 0) <= 0:
            return sonar_obs
        if getattr(self, "maze_centers", None) is None or getattr(self, "maze_yaw", None) is None:
            return sonar_obs
        try:
            env_pos = self.envs_positions.to(self.device)
            pos_local = scan_pos_w - env_pos  # [E,3]
            centers = self.maze_centers.to(self.device)  # [E,W,3]
            wall_yaw = self.maze_yaw.to(self.device)  # [E,W]

            yaw = quaternion_to_euler(scan_rot_w)[:, 2]  # [E]
            angles = self.sonar.angles.to(self.device)  # [R]
            theta = yaw.unsqueeze(1) + angles.unsqueeze(0)  # [E,R]
            dirs = torch.stack([torch.cos(theta), torch.sin(theta)], dim=-1)  # [E,R,2]

            o = pos_local[:, :2]  # [E,2]
            c = centers[:, :, :2]  # [E,W,2]
            zc = centers[:, :, 2]  # [E,W]
            oz = pos_local[:, 2:3]  # [E,1]

            # Active walls only (skip walls moved far away when inactive).
            active_xy = (
                (c[:, :, 0] > (self._map_x_min - 2.0))
                & (c[:, :, 0] < (self._map_x_max + 2.0))
                & (c[:, :, 1] > (self._map_y_min - 2.0))
                & (c[:, :, 1] < (self._map_y_max + 2.0))
            )

            L = 0.5 * float(getattr(self, "maze_length", 1.0)) + self._map_maze_margin_xy
            T = 0.5 * float(getattr(self, "maze_thickness", 0.2)) + self._map_maze_margin_xy
            H = 0.5 * float(getattr(self, "maze_height", 0.6)) + self._map_maze_margin_z
            z_ok = torch.abs(zc - oz) <= H
            valid_wall = active_xy & z_ok  # [E,W]

            # Transform ray origin/direction into each wall local frame.
            rel = o[:, None, :] - c  # [E,W,2]
            cy = torch.cos(wall_yaw)
            sy = torch.sin(wall_yaw)
            ox = cy * rel[:, :, 0] + sy * rel[:, :, 1]  # [E,W]
            oy = -sy * rel[:, :, 0] + cy * rel[:, :, 1]  # [E,W]

            dx = dirs[:, :, 0].unsqueeze(-1)  # [E,R,1]
            dy = dirs[:, :, 1].unsqueeze(-1)  # [E,R,1]
            dxl = cy.unsqueeze(1) * dx + sy.unsqueeze(1) * dy  # [E,R,W]
            dyl = -sy.unsqueeze(1) * dx + cy.unsqueeze(1) * dy  # [E,R,W]

            ox3 = ox.unsqueeze(1)  # [E,1,W]
            oy3 = oy.unsqueeze(1)  # [E,1,W]
            eps = 1e-6
            inf = torch.full_like(dxl, float("inf"))
            ninf = torch.full_like(dxl, -float("inf"))

            # X slab.
            par_x = torch.abs(dxl) <= eps
            valid_x = (~par_x) | (torch.abs(ox3) <= L)
            dx_safe = torch.where(par_x, torch.ones_like(dxl), dxl)
            tx1 = (-L - ox3) / dx_safe
            tx2 = (+L - ox3) / dx_safe
            tx_lo = torch.minimum(tx1, tx2)
            tx_hi = torch.maximum(tx1, tx2)
            tx_lo = torch.where(par_x, ninf, tx_lo)
            tx_hi = torch.where(par_x, inf, tx_hi)

            # Y slab.
            par_y = torch.abs(dyl) <= eps
            valid_y = (~par_y) | (torch.abs(oy3) <= T)
            dy_safe = torch.where(par_y, torch.ones_like(dyl), dyl)
            ty1 = (-T - oy3) / dy_safe
            ty2 = (+T - oy3) / dy_safe
            ty_lo = torch.minimum(ty1, ty2)
            ty_hi = torch.maximum(ty1, ty2)
            ty_lo = torch.where(par_y, ninf, ty_lo)
            ty_hi = torch.where(par_y, inf, ty_hi)

            t_enter = torch.maximum(tx_lo, ty_lo)
            t_exit = torch.minimum(tx_hi, ty_hi)
            valid_hit = valid_x & valid_y & (t_exit >= torch.maximum(t_enter, torch.zeros_like(t_enter)))
            t = torch.where(t_enter > 0.0, t_enter, t_exit)
            t = torch.where(valid_hit & valid_wall.unsqueeze(1), t, inf)
            tmin = torch.min(t, dim=-1).values  # [E,R]
            tmin = torch.clamp(tmin, min=0.0, max=self._map_max_range)
            return torch.minimum(sonar_obs, tmin)
        except Exception:
            return sonar_obs

    def _update_persistent_map(
        self,
        pos_w: torch.Tensor,
        rot_w: torch.Tensor,
        sonar_obs: torch.Tensor,
        depth_img_for_occ: torch.Tensor | None,
    ) -> None:
        if (
            not self.mapping_enabled
            or self._map_occ is None
            or self._map_static is None
            or self._map_dynamic is None
        ):
            return

        if self._map_update_all:
            env_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.long)
        else:
            env_id = int(max(0, min(self.num_envs - 1, self._map_update_env_id)))
            env_ids = torch.tensor([env_id], device=self.device, dtype=torch.long)
        if env_ids.numel() == 0:
            return

        if self._map_decay > 0.0:
            self._map_static[env_ids] *= max(0.0, 1.0 - self._map_decay)
        if self._map_dyn_decay > 0.0:
            self._map_dynamic[env_ids] *= max(0.0, 1.0 - self._map_dyn_decay)

        pos_local_all = pos_w - self.envs_positions.to(pos_w.device)
        pos_local = pos_local_all[env_ids]
        yaw = quaternion_to_euler(rot_w[env_ids])[:, 2]
        angles = self.sonar.angles.to(pos_w.device)
        theta = yaw.unsqueeze(1) + angles.unsqueeze(0)
        ranges = sonar_obs[env_ids].clamp(min=0.0, max=self._map_max_range)
        hit = (ranges < (self._map_max_range - 1e-6)) & (ranges >= self._map_min_hit_range)

        dyn_centers = None
        dyn_r = float(self.obst_cfg.get("radius", 0.15))
        if self._map_classify_dynamic_from_gt and getattr(self, "enable_obstacles", False):
            try:
                if getattr(self, "obstacles", None) is not None:
                    obst_pos_w, _ = self.obstacles.get_world_poses()
                    obst_pos_w = obst_pos_w.to(self.device)
                    if obst_pos_w.ndim == 2:
                        obst_pos_w = obst_pos_w.view(self.num_envs, self.num_obstacles, 3)
                    obst_local = obst_pos_w - self.envs_positions.unsqueeze(1)
                    dyn_centers = obst_local[env_ids]
                elif getattr(self, "obst_center", None) is not None:
                    dyn_centers = self.obst_center.to(self.device)[env_ids]
            except Exception:
                dyn_centers = None

        if self._map_mark_free and self._map_free_dec > 0.0:
            dx = (self._map_x_max - self._map_x_min) / max(self._map_grid_size - 1, 1)
            dy = (self._map_y_max - self._map_y_min) / max(self._map_grid_size - 1, 1)
            step = float(max(min(dx, dy), 1e-3))
            num_samples = int(min(self._map_max_samples, max(2, (self._map_max_range / step) + 1)))
            t = torch.linspace(0.0, self._map_max_range, num_samples, device=self.device).view(1, 1, -1)

            x = pos_local[:, 0].view(-1, 1, 1) + t * torch.cos(theta).unsqueeze(-1)
            y = pos_local[:, 1].view(-1, 1, 1) + t * torch.sin(theta).unsqueeze(-1)
            ix_s, iy_s = self._map_world_to_grid(x, y)

            free_limit = (ranges - step).clamp_min(0.0).unsqueeze(-1)
            free_mask = t < free_limit

            G = self._map_grid_size
            for j, env_id in enumerate(env_ids.tolist()):
                if not free_mask[j].any():
                    continue
                map_flat_s = self._map_static[env_id].view(-1)
                map_flat_d = self._map_dynamic[env_id].view(-1)
                idx_flat = (iy_s[j] * G + ix_s[j]).reshape(-1)
                sel = idx_flat[free_mask[j].reshape(-1)]
                if sel.numel() > 0:
                    delta_s = torch.full(
                        (sel.numel(),),
                        -float(self._map_free_dec),
                        device=self.device,
                        dtype=map_flat_s.dtype,
                    )
                    delta_d = torch.full(
                        (sel.numel(),),
                        -float(self._map_dyn_free_dec),
                        device=self.device,
                        dtype=map_flat_d.dtype,
                    )
                    map_flat_s.index_add_(0, sel, delta_s)
                    map_flat_d.index_add_(0, sel, delta_d)
                    self._map_static[env_id].clamp_(0.0, 1.0)
                    self._map_dynamic[env_id].clamp_(0.0, 1.0)

        x_end = pos_local[:, 0:1] + ranges * torch.cos(theta)
        y_end = pos_local[:, 1:2] + ranges * torch.sin(theta)
        ix, iy = self._map_world_to_grid(x_end, y_end)
        G = self._map_grid_size
        for j, env_id in enumerate(env_ids.tolist()):
            # Cache per-step sonar endpoints for map debug overlay.
            if env_id == int(self._map_save_env_id):
                hit_mask = hit[j]
                if hit_mask.any():
                    pts_all = torch.stack([x_end[j], y_end[j]], dim=-1)
                    pts_hit = pts_all[hit_mask]
                    self._map_last_hit_points_local = pts_hit.detach().cpu().numpy()
                    self._map_last_hit_count = int(pts_hit.shape[0])
                else:
                    self._map_last_hit_points_local = np.zeros((0, 2), dtype=np.float32)
                    self._map_last_hit_count = 0
                self._map_last_hit_ratio = float(hit_mask.float().mean().item())
                self._map_last_scan_origin_local = (
                    pos_local[j, :2].detach().cpu().numpy().astype(np.float32)
                )
                self._map_last_hit_points_dyn_local = np.zeros((0, 2), dtype=np.float32)

            if hit[j].any():
                dyn_mask_from_sonar = None
                try:
                    dyn_sonar_mask = getattr(self, "_map_last_dynamic_sonar_mask", None)
                    if dyn_sonar_mask is not None:
                        dyn_sonar_mask = torch.as_tensor(dyn_sonar_mask, device=self.device, dtype=torch.bool)
                        if dyn_sonar_mask.shape == hit.shape:
                            dyn_mask_from_sonar = dyn_sonar_mask[env_ids[j]]
                except Exception:
                    dyn_mask_from_sonar = None

                if dyn_centers is not None:
                    pts = torch.stack([x_end[j], y_end[j]], dim=-1)
                    centers_xy = dyn_centers[j, :, :2]
                    d2 = ((pts[:, None, :] - centers_xy[None, :, :]) ** 2).sum(dim=-1)
                    is_dyn = (d2.min(dim=1).values.sqrt() <= (dyn_r + self._map_dyn_classify_margin))
                    is_dyn = is_dyn & hit[j]
                else:
                    is_dyn = torch.zeros_like(hit[j])
                if dyn_mask_from_sonar is not None:
                    is_dyn = is_dyn | (dyn_mask_from_sonar & hit[j])

                if env_id == int(self._map_save_env_id):
                    if is_dyn.any():
                        pts_dyn = torch.stack([x_end[j], y_end[j]], dim=-1)[is_dyn]
                        self._map_last_hit_points_dyn_local = pts_dyn.detach().cpu().numpy()
                    else:
                        self._map_last_hit_points_dyn_local = np.zeros((0, 2), dtype=np.float32)

                map_flat_s = self._map_static[env_id].view(-1)
                map_flat_d = self._map_dynamic[env_id].view(-1)
                if is_dyn.any():
                    dyn_idx = (iy[j] * G + ix[j]).reshape(-1)[is_dyn.reshape(-1)]
                    delta_d = torch.full(
                        (dyn_idx.numel(),),
                        float(self._map_dyn_occ_inc),
                        device=self.device,
                        dtype=map_flat_d.dtype,
                    )
                    map_flat_d.index_add_(0, dyn_idx, delta_d)
                    self._stamp_map_hits(
                        self._map_dynamic[env_id],
                        ix[j][is_dyn],
                        iy[j][is_dyn],
                        self._map_dynamic_hit_stamp_radius_m,
                        self._map_dynamic_hit_stamp_inc,
                    )
                stat_mask = hit[j] & ~is_dyn
                if stat_mask.any():
                    stat_idx = (iy[j] * G + ix[j]).reshape(-1)[stat_mask.reshape(-1)]
                    delta_s = torch.full(
                        (stat_idx.numel(),),
                        float(self._map_occ_inc),
                        device=self.device,
                        dtype=map_flat_s.dtype,
                    )
                    map_flat_s.index_add_(0, stat_idx, delta_s)
                    self._stamp_map_hits(
                        self._map_static[env_id],
                        ix[j][stat_mask],
                        iy[j][stat_mask],
                        self._map_static_hit_stamp_radius_m,
                        self._map_static_hit_stamp_inc,
                    )
                self._map_static[env_id].clamp_(0.0, 1.0)
                self._map_dynamic[env_id].clamp_(0.0, 1.0)

        if self._map_use_depth and depth_img_for_occ is not None:
            try:
                depth_range = float(self.mapping_cfg.get("depth_range", self._map_max_range))
                min_filter = float(self.depth_cfg.get("min_range_filter", 0.0))
                fov_deg = float(self.depth_cfg.get("fov_deg", 90.0))
                width = int(depth_img_for_occ.shape[2])
                angles_d = torch.linspace(
                    -0.5 * fov_deg, 0.5 * fov_deg, width, device=self.device
                ) * (math.pi / 180.0)
                depth_img = depth_img_for_occ[env_ids].clamp(min=0.0, max=depth_range)
                if min_filter > 0.0:
                    depth_img = torch.where(
                        depth_img > min_filter,
                        depth_img,
                        torch.full_like(depth_img, depth_range),
                    )
                h = int(depth_img.shape[1])
                r0 = int(round(max(0.0, min(1.0, self._map_depth_row_min)) * float(h)))
                r1 = int(round(max(0.0, min(1.0, self._map_depth_row_max)) * float(h)))
                if r1 <= r0:
                    r0, r1 = 0, h
                depth_band = depth_img[:, r0:r1, :]
                depth_cols = depth_band.median(dim=1).values
                theta_d = yaw.unsqueeze(1) + angles_d.unsqueeze(0)
                hit_d = (depth_cols < depth_range) & (depth_cols >= self._map_min_hit_range)
                if self._map_depth_gate_with_sonar:
                    sonar_angles = self.sonar.angles.to(self.device)  # [R]
                    # Relative front-camera angles to body frame in [-pi, pi].
                    rel = torch.atan2(torch.sin(angles_d), torch.cos(angles_d)).unsqueeze(1)  # [W,1]
                    diff = torch.abs(rel - sonar_angles.unsqueeze(0))  # [W,R]
                    diff = torch.minimum(diff, (2.0 * math.pi) - diff)
                    nearest = torch.argmin(diff, dim=1)  # [W]
                    sonar_near = ranges[:, nearest]  # [E',W]
                    gate = torch.abs(depth_cols - sonar_near) <= self._map_depth_sonar_gate_tol
                    # If sonar has no return (max-range), do not force depth to zero.
                    sonar_open = sonar_near >= (self._map_max_range - 1e-3)
                    hit_d = hit_d & (gate | sonar_open)
                x_d = pos_local[:, 0:1] + depth_cols * torch.cos(theta_d)
                y_d = pos_local[:, 1:2] + depth_cols * torch.sin(theta_d)
                ix_d, iy_d = self._map_world_to_grid(x_d, y_d)
                for j, env_id in enumerate(env_ids.tolist()):
                    if hit_d[j].any():
                        occ_idx = (iy_d[j] * G + ix_d[j]).reshape(-1)[hit_d[j].reshape(-1)]
                        map_flat_s = self._map_static[env_id].view(-1)
                        delta = torch.full(
                            (occ_idx.numel(),),
                            float(self._map_occ_inc) * float(max(0.0, self._map_depth_occ_weight)),
                            device=self.device,
                            dtype=map_flat_s.dtype,
                        )
                        map_flat_s.index_add_(0, occ_idx, delta)
                        self._map_static[env_id].clamp_(0.0, 1.0)
            except Exception:
                pass

        # Robust dynamic-layer visibility: stamp current GT dynamic obstacle centers.
        # This is map-only logic and does not affect control/motion code.
        if (
            self._map_dyn_stamp_from_gt
            and dyn_centers is not None
            and dyn_centers.numel() > 0
            and self._map_dyn_occ_inc > 0.0
        ):
            dx = (self._map_x_max - self._map_x_min) / max(self._map_grid_size - 1, 1)
            dy = (self._map_y_max - self._map_y_min) / max(self._map_grid_size - 1, 1)
            cell = max(min(dx, dy), 1e-6)
            if self._map_dyn_stamp_radius_m > 0.0:
                stamp_r_m = float(self._map_dyn_stamp_radius_m)
            else:
                stamp_r_m = max(0.05, dyn_r + 0.05)
            stamp_r = max(1, int(math.ceil(stamp_r_m / cell)))
            inc = float(max(0.05, self._map_dyn_stamp_inc))
            for j, env_id in enumerate(env_ids.tolist()):
                centers = dyn_centers[j, :, :2]
                for k in range(int(centers.shape[0])):
                    cx = float(centers[k, 0].item())
                    cy = float(centers[k, 1].item())
                    ix_c, iy_c = self._map_world_to_grid(
                        torch.tensor([cx], device=self.device),
                        torch.tensor([cy], device=self.device),
                    )
                    ix0 = max(0, int(ix_c.item()) - stamp_r)
                    ix1 = min(self._map_grid_size - 1, int(ix_c.item()) + stamp_r)
                    iy0 = max(0, int(iy_c.item()) - stamp_r)
                    iy1 = min(self._map_grid_size - 1, int(iy_c.item()) + stamp_r)
                    if ix1 < ix0 or iy1 < iy0:
                        continue
                    ys = torch.arange(iy0, iy1 + 1, device=self.device)
                    xs = torch.arange(ix0, ix1 + 1, device=self.device)
                    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
                    d2 = (xx - int(ix_c.item())) ** 2 + (yy - int(iy_c.item())) ** 2
                    mask = d2 <= (stamp_r * stamp_r)
                    patch = self._map_dynamic[env_id, iy0 : iy1 + 1, ix0 : ix1 + 1]
                    patch[mask] = torch.clamp(patch[mask] + inc, 0.0, 1.0)

        if self._map_self_clear_radius > 0.0:
            rr2 = float(self._map_self_clear_radius) ** 2
            gx = torch.linspace(self._map_x_min, self._map_x_max, self._map_grid_size, device=self.device)
            gy = torch.linspace(self._map_y_min, self._map_y_max, self._map_grid_size, device=self.device)
            xx, yy = torch.meshgrid(gx, gy, indexing="xy")
            for j, env_id in enumerate(env_ids.tolist()):
                px = float(pos_local[j, 0].item())
                py = float(pos_local[j, 1].item())
                mask = ((xx - px) ** 2 + (yy - py) ** 2) <= rr2
                self._map_static[env_id][mask] = 0.0
                self._map_dynamic[env_id][mask] = 0.0

        self._map_occ[env_ids] = torch.maximum(self._map_static[env_ids], self._map_dynamic[env_ids])

    def _map_overlay_predictive_obstacles(
        self, env_id: int, base_radius: float
    ) -> list[Obstacle]:
        out: list[Obstacle] = []
        if (
            (not self._map_path_overlay_dynamic_predict_enable)
            or (not getattr(self, "enable_obstacles", False))
            or getattr(self, "obstacles", None) is None
            or int(getattr(self, "num_obstacles", 0)) <= 0
        ):
            return out
        try:
            obst_pos_w, _ = self.obstacles.get_world_poses()
            obst_pos_w = obst_pos_w.to(self.device)
            if obst_pos_w.ndim == 2:
                obst_pos_w = obst_pos_w.view(self.num_envs, self.num_obstacles, 3)
            obst_local = obst_pos_w - self.envs_positions.unsqueeze(1)

            if (
                self._map_overlay_prev_dyn_local is None
                or self._map_overlay_vel_dyn_local is None
                or self._map_overlay_prev_step is None
                or self._map_overlay_prev_dyn_local.shape != obst_local.shape
            ):
                self._map_overlay_prev_dyn_local = obst_local.detach().clone()
                self._map_overlay_vel_dyn_local = torch.zeros_like(obst_local)
                pb = getattr(self, "progress_buf", None)
                if pb is not None:
                    self._map_overlay_prev_step = pb.view(-1).to(self.device).clone().long()
                else:
                    self._map_overlay_prev_step = torch.zeros(
                        (self.num_envs,), device=self.device, dtype=torch.long
                    )
                vel_local = self._map_overlay_vel_dyn_local
            else:
                pb = getattr(self, "progress_buf", None)
                if pb is not None:
                    cur_step = pb.view(-1).to(self.device).clone().long()
                else:
                    cur_step = self._map_overlay_prev_step + 1
                dt_steps = (cur_step - self._map_overlay_prev_step).clamp(min=1).view(-1, 1, 1).float()
                dt = dt_steps * float(max(getattr(self, "dt", 0.02), 1e-4))
                vel_inst = (obst_local - self._map_overlay_prev_dyn_local) / dt
                clip = float(self._map_path_overlay_dynamic_speed_clip)
                if clip > 0.0:
                    speed_xy = torch.norm(vel_inst[..., :2], dim=-1, keepdim=True).clamp_min(1e-6)
                    scale = torch.clamp(clip / speed_xy, max=1.0)
                    vel_inst[..., :2] = vel_inst[..., :2] * scale
                ema = float(min(1.0, max(0.0, self._map_path_overlay_dynamic_vel_ema)))
                self._map_overlay_vel_dyn_local = (
                    (1.0 - ema) * self._map_overlay_vel_dyn_local + ema * vel_inst
                )
                reset_mask = (cur_step <= 1) | (cur_step < self._map_overlay_prev_step)
                if bool(reset_mask.any().item()):
                    self._map_overlay_vel_dyn_local[reset_mask] = 0.0
                self._map_overlay_prev_dyn_local = obst_local.detach().clone()
                self._map_overlay_prev_step = cur_step
                vel_local = self._map_overlay_vel_dyn_local

            centers = obst_local[env_id]
            vels = vel_local[env_id]
            min_speed = float(max(0.0, self._map_path_overlay_dynamic_min_speed))
            n_samples = int(max(1, self._map_path_overlay_dynamic_samples))
            horizon = float(max(0.0, self._map_path_overlay_dynamic_horizon_sec))
            x_min, x_max = self._map_x_min, self._map_x_max
            y_min, y_max = self._map_y_min, self._map_y_max
            for j in range(int(centers.shape[0])):
                p = centers[j]
                v = vels[j]
                speed_xy = float(torch.norm(v[:2]).item())
                if speed_xy < min_speed:
                    continue
                for k in range(1, n_samples + 1):
                    t = horizon * (float(k) / float(n_samples))
                    pred = p + v * t
                    px = float(pred[0].item())
                    py = float(pred[1].item())
                    if (px < x_min) or (px > x_max) or (py < y_min) or (py > y_max):
                        continue
                    rr = (
                        float(base_radius)
                        + float(max(0.0, self._map_path_overlay_min_clearance))
                        + float(self._map_path_overlay_dynamic_inflate_add)
                        + float(self._map_path_overlay_dynamic_inflate_per_mps) * speed_xy * t
                    )
                    out.append(Obstacle((px, py, float(pred[2].item())), max(0.0, rr)))
        except Exception:
            return out
        return out

    def _compute_map_overlay_path_local(self, env_id: int) -> list[tuple[float, float, float]] | None:
        if not self.mapping_enabled or self._map_occ is None:
            return None
        try:
            env_offset = self.envs_positions[env_id].detach()
            start_t = self.drone.pos[env_id, 0, :].detach() - env_offset
            goal_t = self.target_pos[env_id, 0, :].detach() - env_offset
            start_local = (float(start_t[0].item()), float(start_t[1].item()), float(start_t[2].item()))
            goal_local = (float(goal_t[0].item()), float(goal_t[1].item()), float(goal_t[2].item()))
            occ_grid = self._map_occ[env_id].detach().cpu().float().numpy()
            occ_bounds_xy = (self._map_x_min, self._map_x_max, self._map_y_min, self._map_y_max)
            fixed_depth = float(start_local[2])
            base_radius = float(getattr(self, "obst_cfg", {}).get("radius", 0.15))
            pred_obstacles = self._map_overlay_predictive_obstacles(env_id, base_radius)

            if self._map_path_overlay_planner == "rrt":
                planner = GlobalPlanner(
                    GlobalPlannerConfig(
                        bounds_min=(self._map_x_min, self._map_y_min, self._map_z_min),
                        bounds_max=(self._map_x_max, self._map_y_max, self._map_z_max),
                        wall_margin=self._map_path_overlay_wall_margin,
                        fixed_depth=fixed_depth,
                        step_size=max(0.05, self._map_path_overlay_rrt_step_size),
                        max_iters=max(10, self._map_path_overlay_rrt_max_iters),
                        goal_sample_rate=min(
                            1.0, max(0.0, self._map_path_overlay_rrt_goal_sample_rate)
                        ),
                        neighbor_radius=max(0.1, self._map_path_overlay_rrt_neighbor_radius),
                        goal_tolerance=max(0.05, self._map_path_overlay_goal_tolerance),
                        shortcut_enable=bool(self._map_path_overlay_shortcut_enable),
                        resample_enable=True,
                        resample_step=max(0.05, self._map_path_overlay_resample_step),
                        use_occ_grid=True,
                        occ_threshold=self._map_path_overlay_occ_threshold,
                        occ_step=self._map_path_overlay_occ_step,
                        occ_clearance=max(0.0, self._map_path_overlay_occ_clearance),
                    )
                )
                plan = planner.plan(
                    start_w=start_local,
                    goal_w=goal_local,
                    obstacles=pred_obstacles,
                    occ_grid=occ_grid,
                    occ_bounds_xy=occ_bounds_xy,
                )
            else:
                planner = VoronoiPlanner(
                    VoronoiPlannerConfig(
                        bounds_min=(self._map_x_min, self._map_y_min, self._map_z_min),
                        bounds_max=(self._map_x_max, self._map_y_max, self._map_z_max),
                        wall_margin=self._map_path_overlay_wall_margin,
                        fixed_depth=fixed_depth,
                        goal_tolerance=max(0.05, self._map_path_overlay_goal_tolerance),
                        grid_resolution=max(0.05, self._map_path_overlay_grid_resolution),
                        occ_threshold=self._map_path_overlay_occ_threshold,
                        occ_step=self._map_path_overlay_occ_step,
                        clearance_eps=max(0.0, self._map_path_overlay_clearance_eps),
                        min_clearance=max(0.0, self._map_path_overlay_min_clearance),
                        shortcut_enable=bool(self._map_path_overlay_shortcut_enable),
                        resample_enable=True,
                        resample_step=max(0.05, self._map_path_overlay_resample_step),
                    )
                )
                plan = planner.plan(
                    start_w=start_local,
                    goal_w=goal_local,
                    obstacles=pred_obstacles,
                    occ_grid=occ_grid,
                    occ_bounds_xy=occ_bounds_xy,
                )
            if not plan:
                return None
            return [(float(p[0]), float(p[1]), float(p[2])) for p in plan]
        except Exception:
            return None

    def _save_persistent_map_fig(self) -> None:
        if not self.mapping_enabled or self._map_occ is None:
            return
        if self._map_save_every <= 0:
            return
        if (self._map_step % self._map_save_every) != 0:
            return

        env_id = int(max(0, min(self.num_envs - 1, self._map_save_env_id)))
        try:
            os.makedirs(self._map_save_dir, exist_ok=True)
        except Exception:
            return

        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except Exception:
            try:
                occ = self._map_occ[env_id].detach().float().cpu().numpy()
                occ_s = (
                    None
                    if self._map_static is None
                    else self._map_static[env_id].detach().float().cpu().numpy()
                )
                occ_d = (
                    None
                    if self._map_dynamic is None
                    else self._map_dynamic[env_id].detach().float().cpu().numpy()
                )
                out_npz = os.path.join(
                    self._map_save_dir,
                    f"persistent_occ_env{env_id:03d}_step{self._map_step:06d}.npz",
                )
                np.savez_compressed(
                    out_npz,
                    occ=occ,
                    occ_static=occ_s,
                    occ_dynamic=occ_d,
                    bounds=np.array(
                        [self._map_x_min, self._map_x_max, self._map_y_min, self._map_y_max],
                        dtype=np.float32,
                    ),
                )
            except Exception:
                pass
            return

        robot_xy_np = np.zeros((2,), dtype=np.float32)
        goal_xy_np = np.zeros((2,), dtype=np.float32)
        obst_xy_np = np.zeros((0, 2), dtype=np.float32)
        robot_xyz_np = np.zeros((3,), dtype=np.float32)
        goal_xyz_np = np.zeros((3,), dtype=np.float32)
        obst_xyz_np = np.zeros((0, 3), dtype=np.float32)
        try:
            env_offset = self.envs_positions[env_id].detach().to(self.device)
            robot_local = (self.drone.pos[env_id, 0, :].detach().to(self.device) - env_offset).cpu().numpy()
            goal_local = (self.target_pos[env_id, 0, :].detach().to(self.device) - env_offset).cpu().numpy()
            robot_xyz_np = np.asarray(robot_local[:3], dtype=np.float32)
            goal_xyz_np = np.asarray(goal_local[:3], dtype=np.float32)
            robot_xy_np = np.asarray(robot_local[:2], dtype=np.float32)
            goal_xy_np = np.asarray(goal_local[:2], dtype=np.float32)
            if getattr(self, "enable_obstacles", False):
                if getattr(self, "obstacles", None) is not None:
                    obst_pos_w, _ = self.obstacles.get_world_poses()
                    obst_pos_w = obst_pos_w.to(self.device)
                    if obst_pos_w.ndim == 2:
                        obst_pos_w = obst_pos_w.view(self.num_envs, self.num_obstacles, 3)
                    obst_local = obst_pos_w[env_id] - env_offset.unsqueeze(0)
                    obst_xyz_np = obst_local[:, :3].detach().cpu().numpy().astype(np.float32)
                    obst_xy_np = obst_local[:, :2].detach().cpu().numpy().astype(np.float32)
                elif getattr(self, "obst_center", None) is not None:
                    obst_c = self.obst_center[env_id].detach().to(self.device)
                    obst_xyz_np = obst_c[:, :3].detach().cpu().numpy().astype(np.float32)
                    obst_xy_np = obst_c[:, :2].detach().cpu().numpy().astype(np.float32)
        except Exception:
            pass

        if self._map_static is not None and self._map_dynamic is not None:
            occ_s = self._map_static[env_id].detach().float().cpu().numpy()
            occ_d = self._map_dynamic[env_id].detach().float().cpu().numpy()
            occ = np.maximum(occ_s, occ_d)
        else:
            occ_s = None
            occ_d = None
            occ = self._map_occ[env_id].detach().float().cpu().numpy()
        # Always export numeric occupancy tensors for offline paper plots
        # (risk-field reconstruction, ablations, and reproducibility).
        try:
            out_npz = os.path.join(
                self._map_save_dir,
                f"persistent_occ_env{env_id:03d}_step{self._map_step:06d}.npz",
            )
            np.savez_compressed(
                out_npz,
                occ=occ,
                occ_static=occ_s,
                occ_dynamic=occ_d,
                bounds=np.array(
                    [self._map_x_min, self._map_x_max, self._map_y_min, self._map_y_max],
                    dtype=np.float32,
                ),
                map_step=np.array([int(self._map_step)], dtype=np.int32),
                robot_local_xy=robot_xy_np,
                robot_local_xyz=robot_xyz_np,
                goal_local_xy=goal_xy_np,
                goal_local_xyz=goal_xyz_np,
                obstacles_local_xy=obst_xy_np,
                obstacles_local_xyz=obst_xyz_np,
            )
        except Exception:
            pass
        path_pts = 0
        self._map_last_overlay_path_local = None
        fig, ax = plt.subplots(figsize=(7, 3))
        if occ_s is None or bool(self._map_show_static):
            ax.imshow(
                occ_s if occ_s is not None else occ,
                origin="lower",
                cmap="gray_r",
                vmin=0.0,
                vmax=1.0,
                extent=(self._map_x_min, self._map_x_max, self._map_y_min, self._map_y_max),
                aspect="auto",
            )
        if occ_d is not None and bool(self._map_show_dynamic):
            ax.imshow(
                occ_d,
                origin="lower",
                cmap="Reds",
                vmin=0.0,
                vmax=1.0,
                alpha=0.45,
                extent=(self._map_x_min, self._map_x_max, self._map_y_min, self._map_y_max),
                aspect="auto",
            )
        try:
            robot_local = np.array([robot_xy_np[0], robot_xy_np[1], 0.0], dtype=np.float32)
            goal_local = np.array([goal_xy_np[0], goal_xy_np[1], 0.0], dtype=np.float32)
            ax.plot([robot_local[0]], [robot_local[1]], "o", color="dodgerblue", markersize=5, label="robot")
            ax.plot([goal_local[0]], [goal_local[1]], "*", color="red", markersize=7, label="goal")
            if self._map_path_overlay_enable:
                overlay_path = self._compute_map_overlay_path_local(env_id)
                if overlay_path is not None and len(overlay_path) >= 2:
                    p = np.asarray(overlay_path, dtype=np.float32)
                    if p.ndim == 2 and p.shape[1] >= 2:
                        ax.plot(
                            p[:, 0],
                            p[:, 1],
                            "-",
                            color="limegreen",
                            linewidth=1.6,
                            alpha=0.95,
                            label=f"path_{self._map_path_overlay_planner}",
                        )
                        if p.shape[0] > 2:
                            ax.scatter(
                                p[1:-1, 0],
                                p[1:-1, 1],
                                s=6.0,
                                c="limegreen",
                                alpha=0.6,
                                marker="o",
                            )
                        path_pts = int(p.shape[0])
                        self._map_last_overlay_path_local = p
            if self._map_show_sonar_hits and self._map_last_hit_points_local is not None:
                pts = np.asarray(self._map_last_hit_points_local)
                if pts.ndim == 2 and pts.shape[0] > 0:
                    ax.scatter(
                        pts[:, 0],
                        pts[:, 1],
                        s=self._map_sonar_hit_marker_size,
                        c="deepskyblue",
                        alpha=self._map_sonar_hit_alpha,
                        marker=".",
                        label="sonar_hits",
                    )
                    pts_dyn = (
                        None
                        if self._map_last_hit_points_dyn_local is None
                        else np.asarray(self._map_last_hit_points_dyn_local)
                    )
                    if pts_dyn is not None and pts_dyn.ndim == 2 and pts_dyn.shape[0] > 0:
                        ax.scatter(
                            pts_dyn[:, 0],
                            pts_dyn[:, 1],
                            s=max(10.0, self._map_sonar_hit_marker_size * 1.8),
                            c="orange",
                            alpha=min(1.0, self._map_sonar_hit_alpha + 0.05),
                            marker="x",
                            label="sonar_dyn_hits",
                        )
                    if (
                        self._map_show_sonar_rays
                        and self._map_last_scan_origin_local is not None
                    ):
                        o = np.asarray(self._map_last_scan_origin_local).reshape(2)
                        stride = max(1, int(self._map_sonar_ray_stride))
                        for p in pts[::stride]:
                            ax.plot(
                                [o[0], p[0]],
                                [o[1], p[1]],
                                "-",
                                color="deepskyblue",
                                alpha=0.15,
                                linewidth=0.5,
                            )
            ax.legend(loc="upper right", fontsize=8, framealpha=0.7)
        except Exception:
            pass
        ax.set_xlim(self._map_x_min - self._map_plot_pad_x, self._map_x_max + self._map_plot_pad_x)
        ax.set_ylim(self._map_y_min - self._map_plot_pad_y, self._map_y_max + self._map_plot_pad_y)
        ax.set_title(
            f"persistent_occ env={env_id} step={self._map_step} "
            f"sonar_hits={int(self._map_last_hit_count)} "
            f"hit_ratio={float(self._map_last_hit_ratio):.2f} "
            f"path_pts={int(path_pts)}"
        )
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        out_path = os.path.join(
            self._map_save_dir,
            f"persistent_occ_env{env_id:03d}_step{self._map_step:06d}.png",
        )
        fig.tight_layout()
        fig.savefig(out_path, dpi=150)
        plt.close(fig)

    def _compute_state_and_obs(self):
        obs_t0 = time.perf_counter() if self.runtime_timing_enable else None
        if self.runtime_timing_enable:
            self._rt_last_map_ms = 0.0
            self._rt_last_planner_ms = 0.0
            self._rt_last_obs_ms = 0.0
            self._rt_last_replans = 0
        td = super()._compute_state_and_obs()
        need_scan = self.mapping_enabled or self.obs_sonar_features_enable or self.obs_use_path_lookahead
        if not need_scan:
            if obs_t0 is not None:
                self._rt_last_obs_ms = 1000.0 * (time.perf_counter() - obs_t0)
            return td

        pos_w = self.drone.pos[:, 0, :]
        rot_w = self.drone.rot[:, 0, :]
        valid_pose = torch.isfinite(pos_w).all(dim=-1) & torch.isfinite(rot_w).all(dim=-1)
        scan_pos = pos_w
        scan_rot = rot_w
        if not bool(valid_pose.all().item()):
            scan_pos = pos_w.clone()
            scan_rot = rot_w.clone()
            scan_pos[~valid_pose] = 0.0
            scan_rot[~valid_pose] = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device)

        if self.mapping_enabled and self.depth_enabled and self.depth_camera_view is not None:
            cam_offset = torch.tensor(
                self.depth_cfg.get("offset", [0.2, 0.0, 0.0]),
                device=self.device,
                dtype=scan_pos.dtype,
            ).unsqueeze(0).expand(scan_pos.shape[0], 3)
            cam_pos = scan_pos + quat_rotate(scan_rot, cam_offset)
            cam_rot = scan_rot
            if self._depth_rot_offset_q is not None:
                cam_rot = self._quat_mul(scan_rot, self._depth_rot_offset_q.expand_as(scan_rot))
            self.depth_camera_view.set_world_poses(
                positions=cam_pos,
                orientations=cam_rot,
            )

        sonar_obs = self.sonar.scan(scan_pos, scan_rot)["ranges"]
        sonar_obs = self._sonar_ranges_from_base_link(sonar_obs)
        sonar_obs = self._inject_maze_wall_sonar_hits(scan_pos, scan_rot, sonar_obs)
        sonar_obs = self._inject_dynamic_obstacle_sonar_hits(scan_pos, scan_rot, sonar_obs)
        expected_rays = int(self.sonar_cfg.get("num_rays", 32))
        if sonar_obs.dim() != 2 or sonar_obs.shape[1] != expected_rays:
            sonar_obs = torch.full(
                (scan_pos.shape[0], expected_rays),
                float(self.sonar.cfg.max_range),
                device=self.device,
            )
        self._last_sonar = sonar_obs.detach()
        self._last_map_sonar = sonar_obs.detach()

        depth_img_for_occ = None
        if self.mapping_enabled and self.depth_enabled and self.depth_camera is not None and not self._depth_disabled:
            try:
                imgs = self.depth_camera.get_images()
            except Exception:
                self._depth_fail_count += 1
                if self._depth_fail_count >= self._depth_fail_limit:
                    self._depth_disabled = True
                imgs = None
            data_type = str(self.depth_cfg.get("data_type", "distance_to_camera"))
            if imgs is not None and data_type in imgs:
                depth_img = imgs[data_type].squeeze(1)
                if depth_img.dim() == 2:
                    depth_img = depth_img.unsqueeze(0)
                if depth_img.dim() == 3:
                    if depth_img.shape[0] == 1 and scan_pos.shape[0] > 1:
                        depth_img = depth_img.expand(scan_pos.shape[0], -1, -1)
                    max_range = float(self.depth_cfg.get("max_range", 10.0))
                    depth_img = depth_img.clamp(min=0.0, max=max_range)
                    if torch.isfinite(depth_img).all():
                        depth_img_for_occ = depth_img

        if self._obs_extra_dim > 0:
            base_obs = td["agents"]["observation"].squeeze(1)
            extra_list: list[torch.Tensor] = []
            sonar_max = max(1e-3, float(self.sonar.cfg.max_range))
            if self.obs_sonar_features_enable:
                sec = self._sonar_sector_mins(sonar_obs, self.obs_sonar_num_sectors)
                extra_list.append(torch.clamp(sec / sonar_max, 0.0, 1.0))
                front_min, ttc_risk = self._front_min_and_ttc_risk(sonar_obs)
                if self.obs_include_front_min:
                    extra_list.append(torch.clamp((front_min / sonar_max).unsqueeze(-1), 0.0, 1.0))
                if self.obs_include_front_ttc:
                    extra_list.append(ttc_risk.unsqueeze(-1))
            if self.obs_use_path_lookahead:
                extra_list.append(self._path_lookahead_features(scan_pos, scan_rot))
            if extra_list:
                extra = torch.cat(extra_list, dim=-1).unsqueeze(1)
                td["agents"]["observation"] = torch.cat([base_obs.unsqueeze(1), extra], dim=-1)

        if self.mapping_enabled:
            self._map_step += 1
            map_t0 = time.perf_counter() if self.runtime_timing_enable else None
            self._update_persistent_map(scan_pos, scan_rot, sonar_obs, depth_img_for_occ)
            if map_t0 is not None:
                self._rt_last_map_ms += 1000.0 * (time.perf_counter() - map_t0)
            self._save_persistent_map_fig()

        if obs_t0 is not None:
            self._rt_last_obs_ms = 1000.0 * (time.perf_counter() - obs_t0)
        return td

    def _compute_reward_and_done(self):
        pos_w_raw = self.drone.pos[:, 0, :]
        rot_w_raw = self.drone.rot[:, 0, :]

        valid_pose = torch.isfinite(pos_w_raw).all(dim=-1) & torch.isfinite(rot_w_raw).all(dim=-1)
        pos_w = pos_w_raw
        rot_w = rot_w_raw
        if not bool(valid_pose.all().item()):
            pos_w = pos_w_raw.clone()
            rot_w = rot_w_raw.clone()
            pos_w[~valid_pose] = 0.0
            rot_w[~valid_pose] = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device)

        env_offset = self.envs_positions.to(pos_w.device)
        pos_local = pos_w - env_offset

        goal_w = self.target_pos[:, 0, :]
        rel_w = goal_w - pos_w
        dist = torch.norm(rel_w, dim=-1, keepdim=True)
        dist = torch.nan_to_num(dist, nan=self.termination_max_distance, posinf=self.termination_max_distance, neginf=0.0)

        rpy = quaternion_to_euler(rot_w)
        roll = rpy[:, 0:1]
        pitch = rpy[:, 1:2]
        yaw = rpy[:, 2]
        tilt = torch.sqrt(roll * roll + pitch * pitch)
        tilt = torch.nan_to_num(tilt, nan=1.0, posinf=1.0, neginf=1.0)

        ref_dist, yaw_reward_ref, cross_track, path_available = self._path_tracking_terms(
            pos_w=pos_w,
            yaw=yaw,
            goal_w=goal_w,
        )
        if self.path_follow_reward_enable and not self.path_follow_use_goal_fallback:
            use_path = path_available.float()
            dist_ref = use_path * ref_dist + (1.0 - use_path) * dist.detach()
            yaw_reward = use_path * yaw_reward_ref
            cross_track_eff = use_path * cross_track
        else:
            use_path = path_available.float()
            dist_ref = use_path * ref_dist + (1.0 - use_path) * dist
            yaw_reward = use_path * yaw_reward_ref + (1.0 - use_path) * torch.cos(
                ((torch.atan2(rel_w[:, 1], rel_w[:, 0]) - yaw + torch.pi) % (2 * torch.pi) - torch.pi)
            ).unsqueeze(-1)
            cross_track_eff = use_path * cross_track

        dist_ref = torch.nan_to_num(
            dist_ref,
            nan=self.termination_max_distance,
            posinf=self.termination_max_distance,
            neginf=0.0,
        )
        yaw_reward = torch.nan_to_num(yaw_reward, nan=0.0, posinf=0.0, neginf=0.0)
        prev_path_dist = torch.where(torch.isfinite(self._prev_path_ref_dist), self._prev_path_ref_dist, dist_ref)
        progress = (prev_path_dist - dist_ref).clamp(-self.path_progress_clip, self.path_progress_clip)
        self._prev_path_ref_dist = dist_ref.detach()
        prev_goal_dist = torch.where(torch.isfinite(self._prev_goal_dist), self._prev_goal_dist, dist)
        goal_progress = (prev_goal_dist - dist).clamp(-self.goal_progress_clip, self.goal_progress_clip)
        self._prev_goal_dist = dist.detach()

        upright = ((self.drone.up[:, 0, 2:3] + 1.0) * 0.5).clamp(0.0, 1.0)
        upright = torch.nan_to_num(upright, nan=0.0, posinf=0.0, neginf=0.0)
        effort = self.effort.reshape(self.num_envs, -1).mean(dim=-1, keepdim=True)
        effort = torch.nan_to_num(effort, nan=0.0, posinf=1.0, neginf=0.0)
        vel_b = torch.nan_to_num(self.drone.vel_b[:, 0, :3], nan=0.0, posinf=0.0, neginf=0.0)
        body_speed = torch.norm(vel_b, dim=-1, keepdim=True)
        body_speed = torch.nan_to_num(body_speed, nan=0.0, posinf=10.0, neginf=0.0)
        overspeed = torch.clamp(body_speed - self.max_body_speed, min=0.0)
        aggressive_vel_penalty = overspeed * overspeed
        min_clear = None
        last_sonar = getattr(self, "_last_sonar", None)
        if last_sonar is not None:
            try:
                sonar_min = torch.min(torch.as_tensor(last_sonar, device=self.device), dim=1).values
                min_clear = sonar_min if min_clear is None else torch.minimum(min_clear, sonar_min)
            except Exception:
                pass
        last_map_sonar = getattr(self, "_last_map_sonar", None)
        if last_map_sonar is not None:
            try:
                map_min = torch.min(torch.as_tensor(last_map_sonar, device=self.device), dim=1).values
                min_clear = map_min if min_clear is None else torch.minimum(min_clear, map_min)
            except Exception:
                pass
        try:
            obst_clear = self._obstacle_clearance(pos_w).reshape(-1)
            obst_clear = torch.nan_to_num(
                obst_clear,
                nan=float("inf"),
                posinf=float("inf"),
                neginf=0.0,
            )
            min_clear = obst_clear if min_clear is None else torch.minimum(min_clear, obst_clear)
        except Exception:
            pass
        if min_clear is None:
            min_clear = torch.full((self.num_envs,), float("inf"), device=self.device)
        clear_pen = torch.clamp(self.clearance_soft_distance - min_clear, min=0.0).unsqueeze(-1)
        front_min = min_clear
        fwd_speed = torch.clamp(vel_b[:, 0], min=self.ttc_min_forward_speed)
        ttc_front = torch.nan_to_num(
            front_min / fwd_speed,
            nan=self.ttc_horizon,
            posinf=self.ttc_horizon,
            neginf=0.0,
        )
        ttc_pen = torch.clamp(1.0 - (ttc_front / max(1e-3, self.ttc_horizon)), min=0.0, max=1.0).unsqueeze(-1)
        try:
            self._last_min_clearance_mean = float(torch.nanmean(min_clear).item())
            self._last_front_ttc_mean = float(torch.nanmean(ttc_front).item())
            self._last_progress_mean = float(torch.nanmean(progress).item())
            self._last_goal_progress_mean = float(torch.nanmean(goal_progress).item())
            self._last_clearance_penalty_mean = float(torch.nanmean(clear_pen).item())
            self._last_ttc_penalty_mean = float(torch.nanmean(ttc_pen).item())
            self._last_path_available_rate = float(path_available.float().mean().item())
            self._last_path_crosstrack_mean = float(torch.nanmean(cross_track_eff).item())
            self._last_path_ref_dist_mean = float(torch.nanmean(dist_ref).item())
        except Exception:
            pass

        progress_w = self.reward_progress_weight
        distance_w = self.reward_distance_weight
        yaw_w = self.reward_yaw_weight
        if self.path_follow_reward_enable:
            progress_w = self.reward_path_progress_weight
            distance_w = self.reward_path_distance_weight
            yaw_w = self.reward_path_heading_weight

        reward = (
            self.reward_alive
            + progress_w * progress
            + self.reward_goal_progress_weight * goal_progress
            - distance_w * dist_ref
            + self.reward_goal_proximity_weight
            * torch.exp(-dist / self.reward_goal_proximity_scale)
            - self.reward_goal_distance_weight * dist
            + self.reward_upright_weight * upright
            + yaw_w * yaw_reward
            - self.reward_tilt_penalty_weight * tilt
            - self.reward_effort_penalty_weight * effort
            - self.reward_aggressive_vel_penalty_weight * aggressive_vel_penalty
            - self.reward_clearance_penalty_weight * clear_pen
            - self.reward_ttc_penalty_weight * ttc_pen
            - self.reward_path_crosstrack_weight * cross_track_eff
        )
        reward = torch.nan_to_num(reward, nan=-1.0, posinf=10.0, neginf=-10.0).clamp(-20.0, 20.0)

        success_dist_ok = dist <= self.success_distance
        success_step_ok = self.progress_buf.unsqueeze(-1) >= max(0, int(self.success_min_steps))
        success_upright_ok = upright >= float(self.success_min_upright)
        success_heading_ok = yaw_reward >= float(self.success_min_heading)
        success_now = success_dist_ok & success_step_ok & success_upright_ok & success_heading_ok
        self._last_goal_dist = dist.detach()
        self._last_success_now = success_now.detach()
        self._last_success_dist_ok_rate = float(success_dist_ok.float().mean().item())
        self._last_success_step_ok_rate = float(success_step_ok.float().mean().item())
        self._last_success_upright_ok_rate = float(success_upright_ok.float().mean().item())
        self._last_success_heading_ok_rate = float(success_heading_ok.float().mean().item())
        self._success_counter = torch.where(
            success_now,
            self._success_counter + 1,
            torch.zeros_like(self._success_counter),
        )
        success = self._success_counter >= self.success_hold_steps
        reward = reward + self.success_bonus * success.float()

        if self.termination_enabled:
            z = pos_local[:, 2:3]
            misbehave = (
                (z < self.termination_min_z)
                | (z > self.termination_max_z)
                | (dist > self.termination_max_distance)
            )
            if self.termination_grace_steps > 0:
                grace_done = (self.progress_buf.unsqueeze(-1) >= self.termination_grace_steps)
                misbehave = misbehave & grace_done
        else:
            misbehave = torch.zeros_like(dist, dtype=torch.bool)
        if self.termination_on_invalid_state:
            hasnan = (~valid_pose).unsqueeze(-1)
            if self.termination_hasnan_grace_steps > 0:
                hasnan_gate = (self.progress_buf.unsqueeze(-1) >= self.termination_hasnan_grace_steps)
                hasnan = hasnan & hasnan_gate
        else:
            hasnan = torch.zeros_like(dist, dtype=torch.bool)

        if self.termination_out_of_tank:
            # Tank limits are defined in per-env local frame, not world frame.
            x = pos_local[:, 0:1]
            y = pos_local[:, 1:2]
            z = pos_local[:, 2:3]
            out_of_tank = (
                (x < self.tank_x_min)
                | (x > self.tank_x_max)
                | (y < self.tank_y_min)
                | (y > self.tank_y_max)
                | (z < self.tank_z_min)
                | (z > self.tank_z_max)
            )
            if self.termination_grace_steps > 0:
                out_of_tank = out_of_tank & (self.progress_buf.unsqueeze(-1) >= self.termination_grace_steps)
        else:
            out_of_tank = torch.zeros_like(dist, dtype=torch.bool)

        # Optional collision termination. The default proximity mode preserves
        # older sweeps; overlap/contact mode counts geometric overlap only.
        collision_now = torch.zeros_like(dist, dtype=torch.bool)
        if self.terminate_on_collision:
            if self.collision_check_mode in {"overlap", "contact", "actual"}:
                try:
                    env_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.long)
                    collision_now = self._points_blocked_local(
                        env_ids=env_ids,
                        points_local=pos_local,
                        clearance=max(0.0, float(self.collision_overlap_clearance)),
                    ).unsqueeze(-1)
                except Exception:
                    collision_now = torch.zeros_like(dist, dtype=torch.bool)
            elif min_clear is not None:
                collision_now = (min_clear <= float(self.collision_distance)).unsqueeze(-1)
            if self.ground_collision_z is not None:
                ground_collision = pos_local[:, 2:3] <= float(self.ground_collision_z)
                collision_now = collision_now | ground_collision
            if self.termination_grace_steps > 0:
                collision_now = collision_now & (
                    self.progress_buf.unsqueeze(-1) >= self.termination_grace_steps
                )
        ttc_collision_now = torch.zeros_like(dist, dtype=torch.bool)
        if self.terminate_on_ttc:
            ttc_collision_now = (ttc_front <= float(self.termination_ttc_threshold)).unsqueeze(-1)
            if self.safety_shield_terminate_on_clearance:
                ttc_collision_now = ttc_collision_now | (
                    min_clear <= float(self.safety_shield_termination_clearance)
                ).unsqueeze(-1)
            if self.termination_grace_steps > 0:
                ttc_collision_now = ttc_collision_now & (
                    self.progress_buf.unsqueeze(-1) >= self.termination_grace_steps
                )

        truncated = (self.progress_buf >= self.max_episode_length).unsqueeze(-1)
        terminated_fail = misbehave | hasnan | out_of_tank | collision_now | ttc_collision_now
        timeout_fail = truncated & (~success)
        reward = reward - self.reward_termination_penalty * terminated_fail.float()
        reward = reward - self.reward_timeout_penalty * timeout_fail.float()
        terminated = terminated_fail | success
        try:
            done = terminated | truncated
            assigned = torch.zeros_like(done, dtype=torch.bool)
            reason_collision = done & collision_now
            assigned = assigned | reason_collision
            reason_ttc_collision = done & (~assigned) & ttc_collision_now
            assigned = assigned | reason_ttc_collision
            reason_out_of_tank = done & (~assigned) & out_of_tank
            assigned = assigned | reason_out_of_tank
            reason_invalid_state = done & (~assigned) & hasnan
            assigned = assigned | reason_invalid_state
            reason_misbehave = done & (~assigned) & misbehave
            assigned = assigned | reason_misbehave
            reason_success = done & (~assigned) & success
            assigned = assigned | reason_success
            reason_truncated = done & (~assigned) & truncated
            assigned = assigned | reason_truncated
            done_unknown = done & (~assigned)
            self._last_done_reason_masks = {
                "success": reason_success.detach(),
                "collision": reason_collision.detach(),
                "ttc_collision": reason_ttc_collision.detach(),
                "out_of_tank": reason_out_of_tank.detach(),
                "misbehave": reason_misbehave.detach(),
                "invalid_state": reason_invalid_state.detach(),
                "truncated": reason_truncated.detach(),
                "done_unknown": done_unknown.detach(),
            }
            self._last_done_reason_rates = {
                "success": float(reason_success.float().mean().item()),
                "collision": float(reason_collision.float().mean().item()),
                "ttc_collision": float(reason_ttc_collision.float().mean().item()),
                "out_of_tank": float(reason_out_of_tank.float().mean().item()),
                "misbehave": float(reason_misbehave.float().mean().item()),
                "invalid_state": float(reason_invalid_state.float().mean().item()),
                "truncated": float(reason_truncated.float().mean().item()),
                "done_unknown": float(done_unknown.float().mean().item()),
            }
            debug_info = {}
            if bool(out_of_tank.any().item()):
                out_entries = []
                out_mask = out_of_tank.squeeze(-1)
                x_flat = x.squeeze(-1)
                y_flat = y.squeeze(-1)
                z_flat = z.squeeze(-1)
                for env_id in torch.nonzero(out_mask, as_tuple=False).flatten().tolist():
                    axes = []
                    xv = float(x_flat[env_id].item())
                    yv = float(y_flat[env_id].item())
                    zv = float(z_flat[env_id].item())
                    if xv < float(self.tank_x_min):
                        axes.append("x<min")
                    if xv > float(self.tank_x_max):
                        axes.append("x>max")
                    if yv < float(self.tank_y_min):
                        axes.append("y<min")
                    if yv > float(self.tank_y_max):
                        axes.append("y>max")
                    if zv < float(self.tank_z_min):
                        axes.append("z<min")
                    if zv > float(self.tank_z_max):
                        axes.append("z>max")
                    out_entries.append(
                        {
                            "env_id": int(env_id),
                            "pos_local": [xv, yv, zv],
                            "axes": axes,
                            "bounds": {
                                "x": [float(self.tank_x_min), float(self.tank_x_max)],
                                "y": [float(self.tank_y_min), float(self.tank_y_max)],
                                "z": [float(self.tank_z_min), float(self.tank_z_max)],
                            },
                        }
                    )
                debug_info["out_of_tank"] = out_entries
            self._last_done_reason_debug = debug_info
        except Exception:
            pass

        self.stats["pos_error"].lerp_(dist, (1 - self.alpha))
        self.stats["heading_alignment"].lerp_(yaw_reward, (1 - self.alpha))
        self.stats["uprightness"].lerp_(upright, (1 - self.alpha))
        self.stats["action_smoothness"].lerp_(-torch.abs(tilt), (1 - self.alpha))
        self.stats["return"] += reward
        self.stats["episode_len"][:] = self.progress_buf.unsqueeze(1)

        return TensorDict(
            {
                "agents": {"reward": reward.unsqueeze(-1)},
                "done": terminated | truncated,
                "terminated": terminated,
                "truncated": truncated,
                "termination_reason": self._last_done_reason_masks,
            },
            self.batch_size,
        )
