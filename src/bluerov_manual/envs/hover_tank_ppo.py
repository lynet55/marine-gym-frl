import os
import torch
import numpy as np
import math
import re

import omni.isaac.core.utils.prims as prim_utils
import marinegym.utils.kit as kit_utils

from tensordict.tensordict import TensorDict
from torchrl.data import Bounded, Composite, Unbounded

from marinegym.envs.isaac_env import AgentSpec
from marinegym.utils.torch import quat_rotate, quat_rotate_inverse, quaternion_to_euler, quat_axis, euler_to_quaternion
from marinegym.sensors.camera import Camera
from marinegym.sensors.config import PinholeCameraCfg
from marinegym.controllers.hardcoded_mixer import HardcodedMixer
from omni.isaac.core.prims import XFormPrimView
from bluerov_manual.navigation import (
    GlobalPlanner,
    GlobalPlannerConfig,
    Sonar,
    SonarConfig,
    VoronoiPlanner,
    VoronoiPlannerConfig,
)
from bluerov_manual.navigation.obstacles import Obstacle

from .hover_tank import HoverTank


class HoverTankPPO(HoverTank):
    """HoverTank with PPO local velocity actions (continuous)."""

    def _wandb_log(self, data: dict, step: int | None = None) -> None:
        try:
            import wandb  # type: ignore
        except Exception:
            return
        if wandb.run is None:
            return
        if step is None:
            wandb.log(data)
        else:
            wandb.log(data, step=step)

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

    def __init__(self, cfg, headless):
        self.ctrl_cfg = cfg.task.get("controller", {}) or {}
        self.run_mode = str(getattr(cfg, "mode", "train")).lower()
        if self.run_mode == "train":
            legacy_post_mode = self.ctrl_cfg.get("action_postprocess_mode", None)
            if legacy_post_mode is not None:
                legacy_post_mode = str(legacy_post_mode).lower()
                if legacy_post_mode not in {"never", "train_only"}:
                    raise RuntimeError(
                        "Training must run with env action postprocess disabled. "
                        "Set controller.action_postprocess_mode=never or remove it."
                    )
        planner_stack = cfg.task.get("planner_stack", {}) or {}
        if planner_stack.get("enable", False):
            self.sonar_cfg = planner_stack.get("sonar", {}) or {}
            self.planner_cfg = planner_stack.get("rrt", {}) or {}
            self.fusion_cfg = planner_stack.get("fusion", {}) or {}
            self.policy_cfg = planner_stack.get("policy", {}) or {}
        else:
            self.sonar_cfg = cfg.task.get("sonar", {}) or {}
            self.planner_cfg = cfg.task.get("planner", {}) or {}
            self.fusion_cfg = self.planner_cfg
            # Allow configuring safety override thresholds even when the planner stack is disabled.
            # This is useful for "sanity" training runs where you want to ensure PPO actions
            # aren't being clamped to zero by wall-proximity safety logic.
            self.policy_cfg = planner_stack.get("policy", {}) or {}
        self.success_distance = float(cfg.task.get("success_distance", 0.5))
        self.success_bonus = float(cfg.task.get("success_bonus", 5.0))
        self.goal_min_start_dist = float(cfg.task.get("goal_min_start_dist", 0.0))
        self.goal_sample_tries = int(cfg.task.get("goal_sample_tries", 32))
        self.reward_scale = float(cfg.task.get("reward_scale", 1.0))
        self.reward_clip = float(cfg.task.get("reward_clip", 0.0))
        self.terminate_on_nonfinite = bool(cfg.task.get("terminate_on_nonfinite", True))
        self.reward_alive = float(cfg.task.get("reward_alive", 0.0))
        self.reward_goal_progress = float(cfg.task.get("reward_goal_progress", 1.0))
        self.reward_final_goal_progress = float(cfg.task.get("reward_final_goal_progress_weight", 0.0))
        self.obstacle_penalty = float(cfg.task.get("obstacle_penalty", 0.5))
        self.obstacle_threshold = float(cfg.task.get("obstacle_threshold", 1.0))
        self.terminate_on_collision = bool(cfg.task.get("terminate_on_collision", True))
        self.collision_distance = float(cfg.task.get("collision_distance", 0.15))
        self.collision_penalty = float(cfg.task.get("collision_penalty", 1.0))
        self.reset_cooldown_steps = int(cfg.task.get("reset_cooldown_steps", 5))
        self.termination_max_z = float(cfg.task.get("termination", {}).get("max_z", 0.0))
        self.surface_penalty = float(cfg.task.get("surface_penalty", 0.2))
        self.use_waypoint_reward = bool(cfg.task.get("use_waypoint_reward", True))
        self.safety_cfg = cfg.task.get("safety", {}) or {}
        self.debug = bool(cfg.task.get("debug", False))
        self.debug_interval = int(cfg.task.get("debug_interval", 50))
        self.depth_cfg = cfg.task.get("depth_camera", {}) or {}
        self.depth_enabled = bool(self.depth_cfg.get("enable", False))
        self.depth_obs_cfg = cfg.task.get("depth_obs", {}) or {}
        self.depth_obs_enabled = bool(self.depth_obs_cfg.get("enable", False))
        self.occ_cfg = cfg.task.get("occ_grid", {}) or {}
        self.occ_enabled = bool(self.occ_cfg.get("enable", False))
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
        # If false, mapping uses only sensor returns (sonar/depth) and does not use
        # ground-truth obstacle states to classify dynamic/static hits.
        self._map_classify_dynamic_from_gt = bool(
            self.mapping_cfg.get("classify_dynamic_from_gt", True)
        )
        self._map_mark_free = bool(self.mapping_cfg.get("mark_free", True))
        self._map_free_dec = float(self.mapping_cfg.get("free_dec", 0.05))
        # Clear a small disk around the vehicle each update to suppress self-observation artifacts.
        self._map_self_clear_radius = float(self.mapping_cfg.get("self_clear_radius", 0.35))
        # Ignore very-close hit points when integrating map occupancy.
        self._map_min_hit_range = float(
            self.mapping_cfg.get(
                "min_hit_range",
                max(float(self.sonar_cfg.get("min_range_filter", 0.2)), 0.3),
            )
        )
        self._map_max_samples = int(self.mapping_cfg.get("max_samples", 256))
        self._map_update_all = bool(self.mapping_cfg.get("update_all", False))
        self._map_update_env_id = int(self.mapping_cfg.get("update_env_id", self._map_save_env_id))
        # Clear accumulated occupancy on episode reset by default to avoid stale "ghost obstacles"
        # carrying across episodes (especially important when using the map for planning).
        self._map_reset_on_reset = bool(self.mapping_cfg.get("reset_on_reset", True))
        self._map_step = 0

        # 3D voxel map from depth camera (optional).
        self.mapping3d_cfg = cfg.task.get("mapping3d", {}) or {}
        self.mapping3d_enabled = bool(self.mapping3d_cfg.get("enable", False))
        self._vox_step = 0
        self._vox_occ = None
        self._vox_dirs_cam = None
        self._vox_grid_size = tuple(self.mapping3d_cfg.get("grid_size", [96, 64, 32]))  # x,y,z
        self._vox_decay = float(self.mapping3d_cfg.get("decay", 0.02))
        self._vox_occ_inc = float(self.mapping3d_cfg.get("occ_inc", 1.0))
        self._vox_save_every = int(self.mapping3d_cfg.get("save_every", 200))
        self._vox_save_dir = str(self.mapping3d_cfg.get("save_dir", "maps"))
        self._vox_save_env_id = int(self.mapping3d_cfg.get("save_env_id", 0))
        self._vox_max_range = float(self.mapping3d_cfg.get("max_range", float(self.depth_cfg.get("max_range", 10.0))))
        self._vox_stride = tuple(self.mapping3d_cfg.get("stride", [4, 4]))  # w,h stride
        self.depth_camera = None
        self.depth_camera_view = None
        self._last_depth_min = None
        self._depth_viz_every = int(self.depth_cfg.get("viz_every", 50))
        self._depth_viz_count = 0
        self._depth_rot_offset_q = None
        self._depth_fail_count = 0
        self._depth_fail_limit = int(self.depth_cfg.get("fail_limit", 10))
        self._depth_disabled = False
        self._depth_log_wandb = bool(self.depth_cfg.get("log_wandb", False))
        self._depth_fuse_occ = bool(self.depth_cfg.get("fuse_occ", False))
        super().__init__(cfg, headless)
        self._reset_cooldown = torch.zeros(self.num_envs, dtype=torch.int, device=self.device)
        self._cooldown_z_ref = torch.zeros(self.num_envs, device=self.device)
        print(
            "[HoverTankPPO] architecture="
            "PPO_velocity->VelocityController->Thrusters | "
            "GlobalPlanner=configurable(rrt|voronoi) | Sonar=PhysX+Fallback"
        )
        try:
            print(
                "[HoverTankPPO] controller:",
                f"axis_test_open_loop={bool(self.ctrl_cfg.get('axis_test_open_loop', False))}",
                f"hardcoded_axis_mix_enable={bool(self.ctrl_cfg.get('hardcoded_axis_mix_enable', False))}",
                f"u_max={float(self.ctrl_cfg.get('u_max', float('nan'))):.3f}",
                f"v_max={float(self.ctrl_cfg.get('v_max', float('nan'))):.3f}",
                f"w_max={float(self.ctrl_cfg.get('w_max', float('nan'))):.3f}",
                f"r_max={float(self.ctrl_cfg.get('r_max', float('nan'))):.3f}",
            )
        except Exception:
            pass
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
                cam_cfg.usd_params.clipping_range = (float(clip_range[0]), float(clip_range[1]))
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

        self.target_pos = torch.zeros(self.num_envs, 1, 3, device=self.device)
        self._prev_goal_dist = torch.zeros(self.num_envs, 1, device=self.device)
        self._prev_final_goal_dist = torch.zeros(self.num_envs, 1, device=self.device)
        self._last_sonar = None
        self._last_action_executed = None
        self._last_target_vel_w = None
        self._last_wall_dist = None
        self._last_maze_dist = None
        self._last_obst_dist = None
        self._last_reset_goal_dist_mean = 0.0
        self._last_reset_waypoint_dist_mean = 0.0
        self._last_done_reason_rates = {
            "success": 0.0,
            "collision": 0.0,
            "surface": 0.0,
            "nonfinite": 0.0,
            "truncated": 0.0,
        }
        self._waypoint_tol = float(self.planner_cfg.get("waypoint_tol", 0.5))
        self._stuck_steps = torch.zeros(self.num_envs, dtype=torch.int, device=self.device)
        self._stuck_replan_steps = int(self.policy_cfg.get("stuck_replan_steps", 60))
        self._stuck_speed_thresh = float(self.policy_cfg.get("stuck_speed_thresh", 0.02))
        # Predictive dynamic-obstacle global planning (online, planner-only).
        self._dyn_pred_enable = bool(self.planner_cfg.get("dynamic_predict_enable", True))
        self._dyn_pred_horizon_sec = float(self.planner_cfg.get("dynamic_predict_horizon_sec", 2.5))
        self._dyn_pred_samples = int(self.planner_cfg.get("dynamic_predict_samples", 5))
        self._dyn_pred_vel_ema = float(self.planner_cfg.get("dynamic_predict_vel_ema", 0.35))
        self._dyn_pred_min_speed = float(self.planner_cfg.get("dynamic_predict_min_speed", 0.05))
        self._dyn_pred_speed_clip = float(self.planner_cfg.get("dynamic_predict_speed_clip", 2.0))
        self._dyn_pred_inflate_add = float(self.planner_cfg.get("dynamic_predict_inflate_add", 0.03))
        self._dyn_pred_inflate_per_mps = float(
            self.planner_cfg.get("dynamic_predict_inflate_per_mps", 0.12)
        )
        self._dyn_pred_replan_enable = bool(
            self.planner_cfg.get("dynamic_predict_replan_enable", True)
        )
        self._dyn_pred_replan_margin = float(
            self.planner_cfg.get("dynamic_predict_replan_margin", 0.05)
        )
        self._dyn_pred_replan_horizon_sec = float(
            self.planner_cfg.get("dynamic_predict_replan_horizon_sec", self._dyn_pred_horizon_sec)
        )
        self._dyn_pred_assumed_robot_speed = float(
            self.planner_cfg.get("dynamic_predict_assumed_robot_speed", 0.30)
        )
        self._replan_live_debug = bool(self.planner_cfg.get("replan_live_debug", False))
        self._plan_counts = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._plot_rrt_every = int(self.planner_cfg.get("plot_every", 0))
        self._plot_rrt_gif = bool(self.planner_cfg.get("plot_gif", False))
        self._plot_rrt_gif_every = int(self.planner_cfg.get("plot_gif_every", 10))
        self._plot_rrt_history = int(self.planner_cfg.get("plot_history", 50))
        self._plot_rrt_3d = bool(self.planner_cfg.get("plot_3d", False))
        self._plot_rrt_log_wandb = bool(self.planner_cfg.get("log_wandb", False))
        self._plot_rrt_log_gif_wandb = bool(self.planner_cfg.get("log_gif_wandb", False))
        self._rrt_plot_paths = {}
        self._dyn_pred_prev_obst_local = None
        self._dyn_pred_vel_obst_local = None
        self._dyn_pred_prev_step = None
        bounds_min = self.planner_cfg.get("bounds_min", [0.0, 0.0, -2.0])
        bounds_max = self.planner_cfg.get("bounds_max", [20.0, 6.0, 2.0])
        self._rrt_bounds_min = (float(bounds_min[0]), float(bounds_min[1]))
        self._rrt_bounds_max = (float(bounds_max[0]), float(bounds_max[1]))
        self._rrt_bounds_z = (float(bounds_min[2]), float(bounds_max[2]))
        goal_spawn = cfg.task.get("goal_spawn_ranges", cfg.task.get("spawn_ranges", {})) or {}
        if goal_spawn:
            def _axis_range(key, default):
                raw = goal_spawn.get(key, default)
                lo, hi = float(raw[0]), float(raw[1])
                return min(lo, hi), max(lo, hi)

            gx_min, gx_max = _axis_range("x_range", (-2.5, 2.5))
            gy_min, gy_max = _axis_range("y_range", (-2.5, 2.5))
            gz_min, gz_max = _axis_range("z_range", (1.5, 2.5))
            self.goal_spawn_low = torch.tensor([gx_min, gy_min, gz_min], device=self.device)
            self.goal_spawn_high = torch.tensor([gx_max, gy_max, gz_max], device=self.device)
        else:
            self.goal_spawn_low = None
            self.goal_spawn_high = None

        self.sonar = Sonar(
            SonarConfig(
                num_rays=int(self.sonar_cfg.get("num_rays", 32)),
                fov_deg=float(self.sonar_cfg.get("fov_deg", 180.0)),
                max_range=float(self.sonar_cfg.get("max_range", 10.0)),
                include_angle_encoding=bool(self.sonar_cfg.get("include_angle_encoding", False)),
                min_range_filter=float(self.sonar_cfg.get("min_range_filter", 0.2)),
            ),
            device=self.device,
        )
        self.planner_type = str(self.planner_cfg.get("planner_type", "rrt")).strip().lower()
        strict_min_clear = float(self.planner_cfg.get("min_clearance", 0.0))
        occ_clear = float(self.planner_cfg.get("occ_clearance", strict_min_clear))
        if self.planner_type == "voronoi":
            self.planner = VoronoiPlanner(
                VoronoiPlannerConfig(
                    bounds_min=tuple(self.planner_cfg.get("bounds_min", [-10.0, -10.0, -5.0])),
                    bounds_max=tuple(self.planner_cfg.get("bounds_max", [10.0, 10.0, 5.0])),
                    wall_margin=float(self.planner_cfg.get("wall_margin", 0.0)),
                    fixed_depth=self.planner_cfg.get("fixed_depth", None),
                    goal_tolerance=float(self.planner_cfg.get("goal_tolerance", 1.0)),
                    grid_resolution=float(self.planner_cfg.get("grid_resolution", 0.20)),
                    occ_threshold=float(self.planner_cfg.get("map_occ_threshold", 0.5)),
                    occ_step=float(self.planner_cfg.get("map_occ_step", 0.0)) or None,
                    clearance_eps=float(self.planner_cfg.get("voronoi_clearance_eps", 0.05)),
                    min_clearance=strict_min_clear,
                    shortcut_enable=bool(self.planner_cfg.get("shortcut_enable", True)),
                    shortcut_trials=int(self.planner_cfg.get("shortcut_trials", 120)),
                    resample_enable=bool(self.planner_cfg.get("resample_enable", True)),
                    resample_step=float(self.planner_cfg.get("resample_step", 0.4)),
                    fallback_enable=bool(self.planner_cfg.get("voronoi_fallback_rrt_enable", True)),
                    fallback_step_size=float(
                        self.planner_cfg.get(
                            "voronoi_fallback_step_size",
                            self.planner_cfg.get("step_size", 1.0),
                        )
                    ),
                    fallback_max_iters=int(
                        self.planner_cfg.get(
                            "voronoi_fallback_max_iters",
                            self.planner_cfg.get("max_iters", 500),
                        )
                    ),
                    fallback_goal_sample_rate=float(
                        self.planner_cfg.get(
                            "voronoi_fallback_goal_sample_rate",
                            self.planner_cfg.get("goal_sample_rate", 0.1),
                        )
                    ),
                    fallback_neighbor_radius=float(
                        self.planner_cfg.get(
                            "voronoi_fallback_neighbor_radius",
                            self.planner_cfg.get("neighbor_radius", 2.0),
                        )
                    ),
                )
            )
        else:
            self.planner = GlobalPlanner(
                GlobalPlannerConfig(
                    bounds_min=tuple(self.planner_cfg.get("bounds_min", [-10.0, -10.0, -5.0])),
                    bounds_max=tuple(self.planner_cfg.get("bounds_max", [10.0, 10.0, 5.0])),
                    step_size=float(self.planner_cfg.get("step_size", 1.0)),
                    max_iters=int(self.planner_cfg.get("max_iters", 400)),
                    goal_sample_rate=float(self.planner_cfg.get("goal_sample_rate", 0.1)),
                    neighbor_radius=float(self.planner_cfg.get("neighbor_radius", 2.0)),
                    goal_tolerance=float(self.planner_cfg.get("goal_tolerance", 1.0)),
                    shortcut_enable=bool(self.planner_cfg.get("shortcut_enable", True)),
                    shortcut_trials=int(self.planner_cfg.get("shortcut_trials", 200)),
                    resample_enable=bool(self.planner_cfg.get("resample_enable", True)),
                    resample_step=float(self.planner_cfg.get("resample_step", 0.4)),
                    use_occ_grid=bool(self.planner_cfg.get("use_map", False)),
                    occ_threshold=float(self.planner_cfg.get("map_occ_threshold", 0.5)),
                    occ_step=float(self.planner_cfg.get("map_occ_step", 0.0)) or None,
                    occ_clearance=occ_clear,
                )
            )
        self._paths = [[] for _ in range(self.num_envs)]
        self._path_idx = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._A_pinv = None
        self._A_pinv_default = None
        self._prev_action = None
        self._alloc_step = 0
        self._thr_lut_ready = False
        self._thr_cmd_mode = str(self.ctrl_cfg.get("thruster_cmd_mode", "model_inverse")).lower()
        self._axis_mix_warned_invalid = False
        
        # Hard-coded mixer: initialized lazily when first needed
        self._hardcoded_mixer = None
        self._use_hardcoded_mixer = bool(self.ctrl_cfg.get("hardcoded_mixer_enable", False))
        if self._use_hardcoded_mixer:
            self._hardcoded_mixer = HardcodedMixer(device=self.device, dtype=torch.float32)
            print("[HoverTankPPO] Enabled HARD-CODED mixer (6x4 from calibrated scan data)")

        # Persistent 2D occupancy map in local tank coordinates (x/y only).
        self._map_occ = None
        self._map_static = None
        self._map_dynamic = None
        if self.mapping_enabled:
            bounds_min = self.mapping_cfg.get("bounds_min", self.planner_cfg.get("bounds_min", [0.0, 0.0, -2.0]))
            bounds_max = self.mapping_cfg.get("bounds_max", self.planner_cfg.get("bounds_max", [20.0, 6.0, 2.0]))
            self._map_x_min = float(bounds_min[0])
            self._map_x_max = float(bounds_max[0])
            self._map_y_min = float(bounds_min[1])
            self._map_y_max = float(bounds_max[1])
            self._map_max_range = float(self.mapping_cfg.get("max_range", float(self.sonar.cfg.max_range)))
            self._map_occ = torch.zeros(
                (self.num_envs, self._map_grid_size, self._map_grid_size),
                device=self.device,
            )
            self._map_static = torch.zeros_like(self._map_occ)
            self._map_dynamic = torch.zeros_like(self._map_occ)
            self._map_dyn_decay = float(self.mapping_cfg.get("dynamic_decay", max(self._map_decay, 0.05)))
            self._map_dyn_occ_inc = float(self.mapping_cfg.get("dynamic_occ_inc", self._map_occ_inc))
            self._map_dyn_free_dec = float(self.mapping_cfg.get("dynamic_free_dec", self._map_free_dec))
            self._map_dyn_classify_margin = float(self.mapping_cfg.get("dynamic_classify_margin", 0.15))
            self._map_show_dynamic = bool(self.mapping_cfg.get("show_dynamic", True))
            self._map_show_static = bool(self.mapping_cfg.get("show_static", True))

        if self.mapping3d_enabled:
            bounds_min = self.mapping3d_cfg.get(
                "bounds_min",
                self.planner_cfg.get("bounds_min", [0.0, 0.0, -2.0]),
            )
            bounds_max = self.mapping3d_cfg.get(
                "bounds_max",
                self.planner_cfg.get("bounds_max", [20.0, 6.0, 2.0]),
            )
            self._vox_x_min = float(bounds_min[0])
            self._vox_x_max = float(bounds_max[0])
            self._vox_y_min = float(bounds_min[1])
            self._vox_y_max = float(bounds_max[1])
            self._vox_z_min = float(bounds_min[2])
            self._vox_z_max = float(bounds_max[2])
            gx, gy, gz = int(self._vox_grid_size[0]), int(self._vox_grid_size[1]), int(self._vox_grid_size[2])
            gx = max(gx, 8)
            gy = max(gy, 8)
            gz = max(gz, 4)
            self._vox_grid_size = (gx, gy, gz)
            self._vox_occ = torch.zeros((gz, gy, gx), device=self.device)

    def _set_specs(self):
        num_rays = int(self.sonar_cfg.get("num_rays", 32))
        obs_dim = 3 + 2 + 3 + 1 + 1 + 2 + num_rays
        if self.occ_enabled:
            grid_size = int(self.occ_cfg.get("grid_size", 16))
            obs_dim += grid_size * grid_size
        if self.mapping_enabled and self.mapping_use_in_obs:
            obs_dim += self._map_grid_size * self._map_grid_size
        if self.depth_obs_enabled:
            down = self.depth_obs_cfg.get("downsample", [16, 9])
            obs_dim += int(down[0]) * int(down[1])
        self.observation_spec = Composite(
            {
                "agents": Composite(
                    {
                        "observation": Unbounded((1, obs_dim), device=self.device),
                        "intrinsics": self.drone.intrinsics_spec.unsqueeze(0).to(self.device),
                    }
                )
            }
        ).expand(self.num_envs).to(self.device)
        self.action_spec = Composite(
            {
                "agents": Composite(
                    {
                        "action": Bounded(
                            low=-1.0,
                            high=1.0,
                            shape=(1, 4),
                            device=self.device,
                        ),
                    }
                )
            }
        ).expand(self.num_envs).to(self.device)
        self.reward_spec = Composite(
            {"agents": Composite({"reward": Unbounded((1, 1))})}
        ).expand(self.num_envs).to(self.device)
        self.agent_spec["drone"] = AgentSpec(
            "drone", 1,
            observation_key=("agents", "observation"),
            action_key=("agents", "action"),
            reward_key=("agents", "reward"),
            state_key=("agents", "intrinsics"),
        )
        stats_spec = Composite(
            {
                "return": Unbounded(1),
                "episode_len": Unbounded(1),
                "pos_error": Unbounded(1),
                "heading_alignment": Unbounded(1),
                "uprightness": Unbounded(1),
                "action_smoothness": Unbounded(1),
            }
        ).expand(self.num_envs).to(self.device)
        self.observation_spec["stats"] = stats_spec
        self.stats = stats_spec.zero()

    def _build_depth_dirs_cam(self, width: int, height: int, stride_w: int, stride_h: int) -> torch.Tensor:
        """Precompute camera-frame ray directions for a pinhole model (+Z forward)."""
        stride_w = max(int(stride_w), 1)
        stride_h = max(int(stride_h), 1)
        u = torch.arange(0, width, stride_w, device=self.device, dtype=torch.float32)
        v = torch.arange(0, height, stride_h, device=self.device, dtype=torch.float32)
        uu, vv = torch.meshgrid(u, v, indexing="xy")  # [W',H']
        # Treat depth_cfg.fov_deg as horizontal FOV.
        hfov = float(self.depth_cfg.get("fov_deg", 90.0)) * (math.pi / 180.0)
        fx = (width * 0.5) / max(math.tan(hfov * 0.5), 1e-6)
        vfov = 2.0 * math.atan((height * 0.5) / max(fx, 1e-6))
        fy = (height * 0.5) / max(math.tan(vfov * 0.5), 1e-6)
        cx = (width - 1) * 0.5
        cy = (height - 1) * 0.5
        x = (uu - cx) / max(fx, 1e-6)
        y = (vv - cy) / max(fy, 1e-6)
        z = torch.ones_like(x)
        dirs = torch.stack([x, y, z], dim=-1)  # [W',H',3]
        dirs = dirs / (dirs.norm(dim=-1, keepdim=True).clamp_min(1e-6))
        # Return flattened [N,3]
        return dirs.reshape(-1, 3)

    def _update_voxel_map(self, pos_w: torch.Tensor, rot_w: torch.Tensor, depth_img: torch.Tensor | None) -> None:
        if not self.mapping3d_enabled or self._vox_occ is None or depth_img is None:
            return
        if depth_img.dim() != 3:
            return
        env_id = int(max(0, min(self.num_envs - 1, self._vox_save_env_id)))

        # Decay.
        if self._vox_decay > 0.0:
            self._vox_occ.mul_(max(0.0, 1.0 - self._vox_decay))

        H = int(depth_img.shape[1])
        W = int(depth_img.shape[2])
        stride_w, stride_h = int(self._vox_stride[0]), int(self._vox_stride[1])
        if self._vox_dirs_cam is None or getattr(self, "_vox_dirs_shape", None) != (W, H, stride_w, stride_h):
            self._vox_dirs_cam = self._build_depth_dirs_cam(W, H, stride_w=stride_w, stride_h=stride_h)
            self._vox_dirs_shape = (W, H, stride_w, stride_h)

        # Camera pose in world.
        cam_offset = torch.tensor(
            self.depth_cfg.get("offset", [0.2, 0.0, 0.0]),
            device=self.device,
            dtype=pos_w.dtype,
        )
        # `quat_rotate` requires a non-empty batch shape (manual_batch wrapper),
        # so keep everything at least 1D-batched.
        cam_pos = pos_w[env_id] + quat_rotate(rot_w[env_id : env_id + 1], cam_offset.unsqueeze(0))[0]
        cam_rot = rot_w[env_id]
        if self._depth_rot_offset_q is not None:
            cam_rot = self._quat_mul(rot_w[env_id : env_id + 1], self._depth_rot_offset_q.unsqueeze(0))[0]

        # Subsample depth and project to points.
        d = depth_img[env_id, ::stride_h, ::stride_w].reshape(-1)
        d = d.clamp(min=0.0, max=self._vox_max_range)
        valid = d > float(self.depth_cfg.get("min_range_filter", 0.2))
        if not valid.any():
            return
        dirs_cam = self._vox_dirs_cam  # [N,3]
        dirs_cam = dirs_cam[: d.shape[0]]
        q = cam_rot.unsqueeze(0).expand(dirs_cam.shape[0], 4)
        dirs_w = quat_rotate(q, dirs_cam)  # [N,3]
        pts_w = cam_pos.unsqueeze(0) + dirs_w * d.unsqueeze(-1)  # [N,3]

        # Convert to local tank coords and bin into voxels.
        env_offset = self.envs_positions[env_id].to(pts_w.device)
        pts_l = pts_w - env_offset.unsqueeze(0)
        gx, gy, gz = self._vox_grid_size
        x = pts_l[:, 0]
        y = pts_l[:, 1]
        z = pts_l[:, 2]
        mx = (x - self._vox_x_min) / max(self._vox_x_max - self._vox_x_min, 1e-6)
        my = (y - self._vox_y_min) / max(self._vox_y_max - self._vox_y_min, 1e-6)
        mz = (z - self._vox_z_min) / max(self._vox_z_max - self._vox_z_min, 1e-6)
        ix = (mx * (gx - 1)).long().clamp(0, gx - 1)
        iy = (my * (gy - 1)).long().clamp(0, gy - 1)
        iz = (mz * (gz - 1)).long().clamp(0, gz - 1)
        idx = (iz * (gy * gx) + iy * gx + ix).view(-1)
        occ_flat = self._vox_occ.view(-1)
        delta = torch.full((idx.numel(),), float(self._vox_occ_inc), device=self.device, dtype=occ_flat.dtype)
        occ_flat.index_add_(0, idx, delta)
        self._vox_occ.clamp_(0.0, 1.0)

    def _save_voxel_map_fig(self) -> None:
        if not self.mapping3d_enabled or self._vox_occ is None:
            return
        if self._vox_save_every <= 0 or (self._vox_step % self._vox_save_every) != 0:
            return
        env_id = int(max(0, min(self.num_envs - 1, self._vox_save_env_id)))
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except Exception:
            return
        try:
            os.makedirs(self._vox_save_dir, exist_ok=True)
        except Exception:
            return

        gx, gy, gz = self._vox_grid_size
        vox = self._vox_occ.detach().float().cpu().numpy()  # [Z,Y,X]
        # Indices near robot local pose.
        try:
            env_offset = self.envs_positions[env_id].detach()
            robot_local = (self.drone.pos[env_id, 0, :].detach() - env_offset).cpu().numpy()
            goal_local = (self.target_pos[env_id, 0, :].detach() - env_offset).cpu().numpy()
            x0 = float(robot_local[0])
            y0 = float(robot_local[1])
            z0 = float(robot_local[2])
        except Exception:
            robot_local = None
            goal_local = None
            x0 = y0 = z0 = 0.0

        mx = (x0 - self._vox_x_min) / max(self._vox_x_max - self._vox_x_min, 1e-6)
        my = (y0 - self._vox_y_min) / max(self._vox_y_max - self._vox_y_min, 1e-6)
        mz = (z0 - self._vox_z_min) / max(self._vox_z_max - self._vox_z_min, 1e-6)
        ix = int(min(max(round(mx * (gx - 1)), 0), gx - 1))
        iy = int(min(max(round(my * (gy - 1)), 0), gy - 1))
        iz = int(min(max(round(mz * (gz - 1)), 0), gz - 1))

        views = self.mapping3d_cfg.get("save_views", ["xy", "xz", "yz"])
        if isinstance(views, str):
            views = [views]

        def _overlay_path(ax, mode: str) -> None:
            try:
                path = self._paths[env_id] if hasattr(self, "_paths") else None
                if not path:
                    return
                xs = [p[0] - float(env_offset[0].cpu().item()) for p in path]
                ys = [p[1] - float(env_offset[1].cpu().item()) for p in path]
                zs = [p[2] - float(env_offset[2].cpu().item()) for p in path]
                if mode == "xy":
                    ax.plot(xs, ys, "-", color="cyan", linewidth=1.0, alpha=0.9, label="rrt_path")
                elif mode == "xz":
                    ax.plot(xs, zs, "-", color="cyan", linewidth=1.0, alpha=0.9, label="rrt_path")
                elif mode == "yz":
                    ax.plot(ys, zs, "-", color="cyan", linewidth=1.0, alpha=0.9, label="rrt_path")
            except Exception:
                return

        def _save_view(mode: str) -> None:
            if mode == "xy":
                img = vox[iz]  # [Y,X]
                extent = (self._vox_x_min, self._vox_x_max, self._vox_y_min, self._vox_y_max)
                xlabel, ylabel = "x (m)", "y (m)"
                title = f"voxel_xy env={env_id} step={self._vox_step} z={z0:.2f}"
                robot_pt = (x0, y0)
                goal_pt = (float(goal_local[0]), float(goal_local[1])) if goal_local is not None else None
            elif mode == "xz":
                img = vox[:, iy, :]  # [Z,X]
                extent = (self._vox_x_min, self._vox_x_max, self._vox_z_min, self._vox_z_max)
                xlabel, ylabel = "x (m)", "z (m)"
                title = f"voxel_xz env={env_id} step={self._vox_step} y={y0:.2f}"
                robot_pt = (x0, z0)
                goal_pt = (float(goal_local[0]), float(goal_local[2])) if goal_local is not None else None
            elif mode == "yz":
                img = vox[:, :, ix]  # [Z,Y]
                extent = (self._vox_y_min, self._vox_y_max, self._vox_z_min, self._vox_z_max)
                xlabel, ylabel = "y (m)", "z (m)"
                title = f"voxel_yz env={env_id} step={self._vox_step} x={x0:.2f}"
                robot_pt = (y0, z0)
                goal_pt = (float(goal_local[1]), float(goal_local[2])) if goal_local is not None else None
            else:
                return

            fig, ax = plt.subplots(figsize=(7, 3))
            ax.imshow(
                img,
                origin="lower",
                cmap="viridis",
                vmin=0.0,
                vmax=1.0,
                extent=extent,
                aspect="auto",
            )
            ax.plot([robot_pt[0]], [robot_pt[1]], "o", color="dodgerblue", markersize=5, label="robot")
            if goal_pt is not None:
                ax.plot([goal_pt[0]], [goal_pt[1]], "*", color="red", markersize=7, label="goal")
            _overlay_path(ax, mode)
            ax.set_title(title)
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            try:
                ax.legend(loc="upper right", fontsize=8, framealpha=0.7)
            except Exception:
                pass
            out_path = os.path.join(self._vox_save_dir, f"voxel_{mode}_env{env_id:03d}_step{self._vox_step:06d}.png")
            fig.tight_layout()
            fig.savefig(out_path, dpi=150)
            plt.close(fig)

        for mode in views:
            _save_view(str(mode).lower())

        # Optional 3D rendering (single XYZ view).
        if bool(self.mapping3d_cfg.get("save_3d", False)):
            try:
                thresh = float(self.mapping3d_cfg.get("save_3d_threshold", 0.5))
                max_pts = int(self.mapping3d_cfg.get("save_3d_max_points", 20000))
                max_pts = max(1000, max_pts)
                zz, yy, xx = np.where(vox >= thresh)
                if zz.size == 0:
                    return
                # Subsample for performance.
                if zz.size > max_pts:
                    idx = np.random.choice(zz.size, size=max_pts, replace=False)
                    zz, yy, xx = zz[idx], yy[idx], xx[idx]
                # Convert voxel indices to metric coordinates in local tank frame.
                xs = self._vox_x_min + (xx.astype(np.float32) / max(gx - 1, 1)) * (self._vox_x_max - self._vox_x_min)
                ys = self._vox_y_min + (yy.astype(np.float32) / max(gy - 1, 1)) * (self._vox_y_max - self._vox_y_min)
                zs = self._vox_z_min + (zz.astype(np.float32) / max(gz - 1, 1)) * (self._vox_z_max - self._vox_z_min)

                fig = plt.figure(figsize=(7, 5))
                ax3 = fig.add_subplot(111, projection="3d")
                ax3.scatter(xs, ys, zs, s=1, c=zs, cmap="viridis", alpha=0.35)
                if robot_local is not None:
                    ax3.scatter([x0], [y0], [z0], c="dodgerblue", s=30, label="robot")
                if goal_local is not None:
                    ax3.scatter([float(goal_local[0])], [float(goal_local[1])], [float(goal_local[2])], c="red", s=40, marker="*", label="goal")
                try:
                    path = self._paths[env_id] if hasattr(self, "_paths") else None
                    if path:
                        xs_p = [p[0] - float(env_offset[0].cpu().item()) for p in path]
                        ys_p = [p[1] - float(env_offset[1].cpu().item()) for p in path]
                        zs_p = [p[2] - float(env_offset[2].cpu().item()) for p in path]
                        ax3.plot(xs_p, ys_p, zs_p, color="cyan", linewidth=1.0, alpha=0.9, label="rrt_path")
                except Exception:
                    pass
                ax3.set_xlim(self._vox_x_min, self._vox_x_max)
                ax3.set_ylim(self._vox_y_min, self._vox_y_max)
                ax3.set_zlim(self._vox_z_min, self._vox_z_max)
                ax3.set_xlabel("x (m)")
                ax3.set_ylabel("y (m)")
                ax3.set_zlabel("z (m)")
                ax3.set_title(f"voxel_xyz env={env_id} step={self._vox_step}")
                try:
                    ax3.legend(loc="upper right", fontsize=8)
                except Exception:
                    pass
                out_path = os.path.join(self._vox_save_dir, f"voxel_xyz_env{env_id:03d}_step{self._vox_step:06d}.png")
                fig.tight_layout()
                fig.savefig(out_path, dpi=150)
                plt.close(fig)
            except Exception:
                pass

    def _map_world_to_grid(self, x: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Map local (x,y) in meters to grid indices (ix,iy)."""
        gx = (x - self._map_x_min) / max(self._map_x_max - self._map_x_min, 1e-6)
        gy = (y - self._map_y_min) / max(self._map_y_max - self._map_y_min, 1e-6)
        ix = (gx * (self._map_grid_size - 1)).long().clamp(0, self._map_grid_size - 1)
        iy = (gy * (self._map_grid_size - 1)).long().clamp(0, self._map_grid_size - 1)
        return ix, iy

    def _update_persistent_map(
        self,
        pos_w: torch.Tensor,
        rot_w: torch.Tensor,
        sonar_obs: torch.Tensor,
        depth_img_for_occ: torch.Tensor | None,
    ) -> None:
        if not self.mapping_enabled or self._map_occ is None or self._map_static is None or self._map_dynamic is None:
            return
        # Optionally update only one env (debugging) to keep compute cheap.
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
        yaw = quaternion_to_euler(rot_w[env_ids])[:, 2]  # [E']
        angles = self.sonar.angles.to(pos_w.device)  # [R]
        theta = yaw.unsqueeze(1) + angles.unsqueeze(0)  # [E',R]
        ranges = sonar_obs[env_ids].clamp(min=0.0, max=self._map_max_range)
        hit = (ranges < (self._map_max_range - 1e-6)) & (ranges >= self._map_min_hit_range)

        # Current dynamic obstacle centers in local frame for classification.
        dyn_centers = None
        dyn_r = float(self.obst_cfg.get("radius", 0.15))
        if (
            self._map_classify_dynamic_from_gt
            and getattr(self, "enable_obstacles", False)
            and getattr(self, "obstacles", None) is not None
        ):
            try:
                obst_pos_w, _ = self.obstacles.get_world_poses()
                obst_pos_w = obst_pos_w.to(self.device)
                if obst_pos_w.ndim == 2:
                    obst_pos_w = obst_pos_w.view(self.num_envs, self.num_obstacles, 3)
                obst_local = obst_pos_w - self.envs_positions.unsqueeze(1)
                dyn_centers = obst_local[env_ids]  # [E',O,3]
            except Exception:
                dyn_centers = None

        # Free-space clearing along rays (prevents “ghost obstacles” from dynamic objects).
        if self._map_mark_free and self._map_free_dec > 0.0:
            # Step ~ 1 grid cell.
            dx = (self._map_x_max - self._map_x_min) / max(self._map_grid_size - 1, 1)
            dy = (self._map_y_max - self._map_y_min) / max(self._map_grid_size - 1, 1)
            step = float(max(min(dx, dy), 1e-3))
            num_samples = int(min(self._map_max_samples, max(2, (self._map_max_range / step) + 1)))
            t = torch.linspace(0.0, self._map_max_range, num_samples, device=self.device).view(1, 1, -1)  # [1,1,S]

            cos_t = torch.cos(theta).unsqueeze(-1)  # [E',R,1]
            sin_t = torch.sin(theta).unsqueeze(-1)  # [E',R,1]
            x = pos_local[:, 0].view(-1, 1, 1) + t * cos_t  # [E',R,S]
            y = pos_local[:, 1].view(-1, 1, 1) + t * sin_t  # [E',R,S]
            ix_s, iy_s = self._map_world_to_grid(x, y)

            free_limit = (ranges - step).clamp_min(0.0).unsqueeze(-1)  # [E',R,1]
            free_mask = t < free_limit  # [E',R,S]

            G = self._map_grid_size
            for j, env_id in enumerate(env_ids.tolist()):
                if not free_mask[j].any():
                    continue
                # Free-space is treated as "no obstacle": clear both static and dynamic layers.
                map_flat_s = self._map_static[env_id].view(-1)
                map_flat_d = self._map_dynamic[env_id].view(-1)
                idx_flat = (iy_s[j] * G + ix_s[j]).reshape(-1)
                mask_flat = free_mask[j].reshape(-1)
                sel = idx_flat[mask_flat]
                if sel.numel() > 0:
                    delta_s = torch.full((sel.numel(),), -float(self._map_free_dec), device=self.device, dtype=map_flat_s.dtype)
                    delta_d = torch.full((sel.numel(),), -float(self._map_dyn_free_dec), device=self.device, dtype=map_flat_d.dtype)
                    map_flat_s.index_add_(0, sel, delta_s)
                    map_flat_d.index_add_(0, sel, delta_d)
                    self._map_static[env_id].clamp_(0.0, 1.0)
                    self._map_dynamic[env_id].clamp_(0.0, 1.0)

        # Occupied endpoints from sonar hits.
        x_end = pos_local[:, 0:1] + ranges * torch.cos(theta)
        y_end = pos_local[:, 1:2] + ranges * torch.sin(theta)
        ix, iy = self._map_world_to_grid(x_end, y_end)
        G = self._map_grid_size
        for j, env_id in enumerate(env_ids.tolist()):
            if hit[j].any():
                occ_idx = (iy[j] * G + ix[j]).reshape(-1)[hit[j].reshape(-1)]
                # Classify hit as dynamic if close to a dynamic obstacle center in x/y.
                if dyn_centers is not None:
                    pts = torch.stack([x_end[j], y_end[j]], dim=-1)  # [R,2]
                    centers_xy = dyn_centers[j, :, :2]  # [O,2]
                    d2 = ((pts[:, None, :] - centers_xy[None, :, :]) ** 2).sum(dim=-1)  # [R,O]
                    is_dyn = (d2.min(dim=1).values.sqrt() <= (dyn_r + self._map_dyn_classify_margin))
                    is_dyn = is_dyn & hit[j]
                else:
                    is_dyn = torch.zeros_like(hit[j])

                map_flat_s = self._map_static[env_id].view(-1)
                map_flat_d = self._map_dynamic[env_id].view(-1)
                if is_dyn.any():
                    dyn_idx = (iy[j] * G + ix[j]).reshape(-1)[is_dyn.reshape(-1)]
                    delta_d = torch.full((dyn_idx.numel(),), float(self._map_dyn_occ_inc), device=self.device, dtype=map_flat_d.dtype)
                    map_flat_d.index_add_(0, dyn_idx, delta_d)
                # Remaining hits go to static.
                stat_mask = hit[j] & ~is_dyn
                if stat_mask.any():
                    stat_idx = (iy[j] * G + ix[j]).reshape(-1)[stat_mask.reshape(-1)]
                    delta_s = torch.full((stat_idx.numel(),), float(self._map_occ_inc), device=self.device, dtype=map_flat_s.dtype)
                    map_flat_s.index_add_(0, stat_idx, delta_s)
                self._map_static[env_id].clamp_(0.0, 1.0)
                self._map_dynamic[env_id].clamp_(0.0, 1.0)

        if self._map_use_depth and depth_img_for_occ is not None:
            try:
                depth_range = float(self.mapping_cfg.get("depth_range", self._map_max_range))
                min_filter = float(self.depth_cfg.get("min_range_filter", 0.0))
                fov_deg = float(self.depth_cfg.get("fov_deg", 90.0))
                width = int(depth_img_for_occ.shape[2])
                angles_d = torch.linspace(
                    -0.5 * fov_deg,
                    0.5 * fov_deg,
                    width,
                    device=self.device,
                ) * (math.pi / 180.0)
                depth_img = depth_img_for_occ[env_ids].clamp(min=0.0, max=depth_range)
                if min_filter > 0.0:
                    # Drop very-close returns (typically self-geometry) before projecting into 2D.
                    depth_img = torch.where(
                        depth_img > min_filter,
                        depth_img,
                        torch.full_like(depth_img, depth_range),
                    )
                depth_cols = depth_img.median(dim=1).values  # [E',W]
                theta_d = yaw.unsqueeze(1) + angles_d.unsqueeze(0)  # [E',W]
                hit_d = (depth_cols < depth_range) & (depth_cols >= self._map_min_hit_range)
                x_d = pos_local[:, 0:1] + depth_cols * torch.cos(theta_d)
                y_d = pos_local[:, 1:2] + depth_cols * torch.sin(theta_d)
                ix_d, iy_d = self._map_world_to_grid(x_d, y_d)
                for j, env_id in enumerate(env_ids.tolist()):
                    if hit_d[j].any():
                        occ_idx = (iy_d[j] * G + ix_d[j]).reshape(-1)[hit_d[j].reshape(-1)]
                        # Depth hits are treated as static by default (can include walls/maze).
                        map_flat_s = self._map_static[env_id].view(-1)
                        delta_s = torch.full((occ_idx.numel(),), float(self._map_occ_inc), device=self.device, dtype=map_flat_s.dtype)
                        map_flat_s.index_add_(0, occ_idx, delta_s)
                        self._map_static[env_id].clamp_(0.0, 1.0)
            except Exception:
                pass

        # Always clear a small disk around current robot position in the persistent map.
        # This prevents robot-body/camera near-field returns from accumulating into false obstacles.
        if self._map_self_clear_radius > 0.0:
            rr2 = float(self._map_self_clear_radius) ** 2
            gx = torch.linspace(
                self._map_x_min, self._map_x_max, self._map_grid_size, device=self.device
            )
            gy = torch.linspace(
                self._map_y_min, self._map_y_max, self._map_grid_size, device=self.device
            )
            xx, yy = torch.meshgrid(gx, gy, indexing="xy")
            for j, env_id in enumerate(env_ids.tolist()):
                px = float(pos_local[j, 0].item())
                py = float(pos_local[j, 1].item())
                mask = ((xx - px) ** 2 + (yy - py) ** 2) <= rr2
                self._map_static[env_id][mask] = 0.0
                self._map_dynamic[env_id][mask] = 0.0

        # Fused map for planning/visualization: static OR dynamic (max).
        self._map_occ[env_ids] = torch.maximum(self._map_static[env_ids], self._map_dynamic[env_ids])

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

        # Fallback when matplotlib is unavailable in runtime: still persist occupancy map data.
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
                        [
                            self._map_x_min,
                            self._map_x_max,
                            self._map_y_min,
                            self._map_y_max,
                        ],
                        dtype=np.float32,
                    ),
                )
            except Exception:
                pass
            return
        if self._map_static is not None and self._map_dynamic is not None:
            occ_s = self._map_static[env_id].detach().float().cpu().numpy()
            occ_d = self._map_dynamic[env_id].detach().float().cpu().numpy()
            occ = np.maximum(occ_s, occ_d)
        else:
            occ_s = None
            occ_d = None
            occ = self._map_occ[env_id].detach().float().cpu().numpy()
        fig, ax = plt.subplots(figsize=(7, 3))
        if occ_s is None or bool(getattr(self, "_map_show_static", True)):
            ax.imshow(
                occ_s if occ_s is not None else occ,
                origin="lower",
                cmap="gray",
                vmin=0.0,
                vmax=1.0,
                extent=(self._map_x_min, self._map_x_max, self._map_y_min, self._map_y_max),
                aspect="auto",
            )
        if occ_d is not None and bool(getattr(self, "_map_show_dynamic", True)):
            ax.imshow(
                occ_d,
                origin="lower",
                cmap="Reds",
                vmin=0.0,
                vmax=1.0,
                alpha=0.35,
                extent=(self._map_x_min, self._map_x_max, self._map_y_min, self._map_y_max),
                aspect="auto",
            )
        # Overlay robot pose, goal, and planned RRT waypoints (in local tank coordinates).
        try:
            env_offset = self.envs_positions[env_id].detach()
            robot_local = (self.drone.pos[env_id, 0, :].detach() - env_offset).cpu().numpy()
            goal_local = (self.target_pos[env_id, 0, :].detach() - env_offset).cpu().numpy()
            ax.plot([robot_local[0]], [robot_local[1]], "o", color="dodgerblue", markersize=5, label="robot")
            ax.plot([goal_local[0]], [goal_local[1]], "*", color="red", markersize=7, label="goal")
            path = self._paths[env_id] if hasattr(self, "_paths") else None
            if path:
                xs = [p[0] - float(env_offset[0].cpu().item()) for p in path]
                ys = [p[1] - float(env_offset[1].cpu().item()) for p in path]
                ax.plot(xs, ys, "-", color="cyan", linewidth=1.0, alpha=0.9, label="rrt_path")
        except Exception:
            pass

        # Dynamic obstacle overlay: position + velocity vector (heading/speed).
        try:
            if getattr(self, "enable_obstacles", False) and getattr(self, "obstacles", None) is not None:
                obst_pos_w, _ = self.obstacles.get_world_poses()
                obst_pos_w = obst_pos_w.to(self.device)
                if obst_pos_w.ndim == 2:
                    obst_pos_w = obst_pos_w.view(self.num_envs, self.num_obstacles, 3)
                obst_w = obst_pos_w[env_id]
                obst_local = (obst_w - self.envs_positions[env_id].unsqueeze(0)).detach().cpu().numpy()

                # Velocity from finite difference across saves (good enough for visualization).
                prev = getattr(self, "_map_prev_obst_pos_w", None)
                if prev is None or not isinstance(prev, dict):
                    prev = {}
                dt = float(getattr(self, "dt", 0.02))
                vel_local = None
                if env_id in prev and dt > 0.0:
                    prev_w = prev[env_id]
                    if isinstance(prev_w, np.ndarray) and prev_w.shape == obst_w.detach().cpu().numpy().shape:
                        vel_w = (obst_w.detach().cpu().numpy() - prev_w) / dt
                        vel_local = vel_w  # env offset cancels in finite difference
                prev[env_id] = obst_w.detach().cpu().numpy().copy()
                self._map_prev_obst_pos_w = prev

                r = float(self.obst_cfg.get("radius", 0.15))
                for k in range(obst_local.shape[0]):
                    ax.add_patch(
                        plt.Circle((obst_local[k, 0], obst_local[k, 1]), r, fill=False, color="orange", linewidth=1.0)
                    )
                if vel_local is not None:
                    # Arrow scaling for visibility.
                    scale = float(self.mapping_cfg.get("vel_arrow_scale", 0.5))
                    u = vel_local[:, 0] * scale
                    v = vel_local[:, 1] * scale
                    ax.quiver(
                        obst_local[:, 0],
                        obst_local[:, 1],
                        u,
                        v,
                        angles="xy",
                        scale_units="xy",
                        scale=1.0,
                        color="orange",
                        width=0.003,
                        alpha=0.8,
                    )
        except Exception:
            pass
        ax.set_title(f"persistent_occ env={env_id} step={self._map_step}")
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        try:
            ax.legend(loc="upper right", fontsize=8, framealpha=0.7)
        except Exception:
            pass
        out_path = os.path.join(
            self._map_save_dir, f"persistent_occ_env{env_id:03d}_step{self._map_step:06d}.png"
        )
        fig.tight_layout()
        fig.savefig(out_path, dpi=150)
        plt.close(fig)

    def _reset_idx(self, env_ids: torch.Tensor):
        if env_ids.device != self.device:
            env_ids = env_ids.to(self.device)
        super()._reset_idx(env_ids)
        self.drone.get_state(env_frame=False)
        if (
            self.mapping_enabled
            and self._map_reset_on_reset
            and self._map_occ is not None
            and self._map_static is not None
            and self._map_dynamic is not None
        ):
            self._map_occ[env_ids] = 0.0
            self._map_static[env_ids] = 0.0
            self._map_dynamic[env_ids] = 0.0
        # Clear action smoothing/history per-env so we don't carry control commands across episodes.
        if self._prev_action is not None and bool(self.ctrl_cfg.get("reset_action_smoother", True)):
            self._prev_action[env_ids] = 0.0
        if hasattr(self, "_last_action_clamped"):
            try:
                self._last_action_clamped[env_ids] = 0.0
            except Exception:
                pass
        if getattr(self, "_stuck_steps", None) is not None:
            self._stuck_steps[env_ids] = 0
        # If kinematic obstacles are enabled, ensure they don't spawn overlapping the robot.
        # Overlaps can generate large separation impulses and immediately trigger "surface" termination,
        # which looks like the env "flying up" and resetting every few steps.
        if self.enable_obstacles and self.obst_spawn_low is not None and self.obst_spawn_high is not None:
            radius = float(self.obst_cfg.get("radius", 0.15))
            clearance = float(self.obst_cfg.get("reset_clearance", radius + 0.35))
            pos_local = self.drone.pos[env_ids, 0, :] - self.envs_positions[env_ids]
            # Re-sample only the violating obstacle positions, a few attempts per reset.
            for _ in range(8):
                d = torch.norm(self.obst_center[env_ids] - pos_local.unsqueeze(1), dim=-1)
                viol = d < clearance
                if not viol.any():
                    break
                # Sample replacements for all violating entries.
                repl = self.obst_spawn_low + torch.rand_like(self.obst_center[env_ids]) * (self.obst_spawn_high - self.obst_spawn_low)
                self.obst_center[env_ids] = torch.where(viol.unsqueeze(-1), repl, self.obst_center[env_ids])
            # Immediately update obstacle poses in the stage/physics.
            self._update_obstacles()
        if self._reset_cooldown is not None:
            self._reset_cooldown[env_ids] = self.reset_cooldown_steps
            # Keep the post-reset depth as a short-term reference during cooldown.
            self._cooldown_z_ref[env_ids] = self.drone.pos[env_ids, 0, 2]
        if self.goal_spawn_low is not None:
            n_env = len(env_ids)
            rand = torch.rand(n_env, 3, device=self.device)
            goal_local = self.goal_spawn_low + rand * (self.goal_spawn_high - self.goal_spawn_low)
            if self.goal_min_start_dist > 0.0:
                start_w = self.drone.pos[env_ids, 0, :]
                env_offsets = self.envs_positions[env_ids]
                for _ in range(max(1, self.goal_sample_tries)):
                    goal_world_try = goal_local + env_offsets
                    d = torch.norm(goal_world_try - start_w, dim=-1)
                    viol = d < self.goal_min_start_dist
                    if not viol.any():
                        break
                    rand_re = torch.rand(n_env, 3, device=self.device)
                    goal_re = self.goal_spawn_low + rand_re * (self.goal_spawn_high - self.goal_spawn_low)
                    goal_local = torch.where(viol.unsqueeze(-1), goal_re, goal_local)
            goal_world = goal_local + self.envs_positions[env_ids]
            self.target_pos[env_ids, 0, :] = goal_world
            target_rot = euler_to_quaternion(torch.zeros(len(env_ids), 1, 3, device=self.device))
            self._set_target_poses(positions=goal_world, orientations=target_rot, env_ids=env_ids)
            rel = goal_world - self.drone.pos[env_ids, 0, :]
            rel_norm = torch.norm(rel, dim=-1, keepdim=True).clamp_min(1e-6)
            self.target_heading[env_ids] = (rel / rel_norm).unsqueeze(1)
            self._prev_goal_dist[env_ids] = torch.norm(rel, dim=-1, keepdim=True)
            self._prev_final_goal_dist[env_ids] = torch.norm(rel, dim=-1, keepdim=True)
        if self.debug:
            start_pos = self.drone.pos[env_ids, 0, :].detach().cpu().numpy()
            goals = self.target_pos[env_ids, 0, :].detach().cpu().numpy()
            env_offset = self.envs_positions[env_ids].detach().cpu().numpy()
            start_local = (self.drone.pos[env_ids, 0, :] - self.envs_positions[env_ids]).detach().cpu().numpy()
            goals_local = (self.target_pos[env_ids, 0, :] - self.envs_positions[env_ids]).detach().cpu().numpy()
            wall_min = self._wall_clearance(self.drone.pos[:, 0, :])[env_ids].detach().cpu().numpy()
            print(
                f"[HoverTankPPO] reset env_ids={env_ids.tolist()} start_pos={start_pos} goals={goals}"
            )
            print(
                f"[HoverTankPPO] reset env_ids={env_ids.tolist()} env_offset={env_offset} "
                f"start_local={start_local} goal_local={goals_local} wall_min={wall_min}"
            )
        self._plan_paths(env_ids)
        # Initialize the progress reference distance for reward shaping.
        # When using waypoint reward, measure distance to the first *next* waypoint (not the goal).
        try:
            if self.use_waypoint_reward:
                pos_w = self.drone.pos[env_ids, 0, :]
                waypoint_w = self._current_waypoints()[env_ids]
                self._prev_goal_dist[env_ids] = torch.norm(waypoint_w - pos_w, dim=-1, keepdim=True)
            # Keep an independent distance reference to the final goal for direct-goal shaping.
            goal_w = self.target_pos[env_ids, 0, :]
            pos_w = self.drone.pos[env_ids, 0, :]
            final_dist = torch.norm(goal_w - pos_w, dim=-1, keepdim=True)
            self._prev_final_goal_dist[env_ids] = final_dist
            self._last_reset_goal_dist_mean = float(final_dist.mean().item())
            try:
                wp_dist = torch.norm(self._current_waypoints()[env_ids] - pos_w, dim=-1)
                self._last_reset_waypoint_dist_mean = float(wp_dist.mean().item())
            except Exception:
                self._last_reset_waypoint_dist_mean = self._last_reset_goal_dist_mean
        except Exception:
            pass
        if self._prev_action is None:
            self._prev_action = torch.zeros(self.num_envs, 1, 4, device=self.device)

    def _update_dynamic_obstacle_prediction(self, obst_pos_local: torch.Tensor | None) -> torch.Tensor | None:
        """Estimate dynamic-obstacle velocity in local frame for predictive path planning."""
        if (
            (not self._dyn_pred_enable)
            or obst_pos_local is None
            or (not torch.is_tensor(obst_pos_local))
            or obst_pos_local.numel() == 0
        ):
            return None
        try:
            cur = obst_pos_local.detach()
            if (
                self._dyn_pred_prev_obst_local is None
                or self._dyn_pred_vel_obst_local is None
                or self._dyn_pred_prev_step is None
                or self._dyn_pred_prev_obst_local.shape != cur.shape
            ):
                self._dyn_pred_prev_obst_local = cur.clone()
                self._dyn_pred_vel_obst_local = torch.zeros_like(cur)
                pb = getattr(self, "progress_buf", None)
                if pb is not None:
                    self._dyn_pred_prev_step = pb.view(-1).to(self.device).clone().long()
                else:
                    self._dyn_pred_prev_step = torch.zeros((self.num_envs,), device=self.device, dtype=torch.long)
                return self._dyn_pred_vel_obst_local

            pb = getattr(self, "progress_buf", None)
            if pb is not None:
                cur_step = pb.view(-1).to(self.device).clone().long()
            else:
                cur_step = self._dyn_pred_prev_step + 1
            dt_steps = (cur_step - self._dyn_pred_prev_step).clamp(min=1).view(-1, 1, 1).float()
            dt = dt_steps * float(max(getattr(self, "dt", 0.02), 1e-4))
            vel_inst = (cur - self._dyn_pred_prev_obst_local) / dt

            clip = float(self._dyn_pred_speed_clip)
            if clip > 0.0:
                speed_xy = torch.norm(vel_inst[..., :2], dim=-1, keepdim=True).clamp_min(1e-6)
                scale = torch.clamp(clip / speed_xy, max=1.0)
                vel_inst[..., :2] = vel_inst[..., :2] * scale
            ema = float(min(1.0, max(0.0, self._dyn_pred_vel_ema)))
            self._dyn_pred_vel_obst_local = (
                (1.0 - ema) * self._dyn_pred_vel_obst_local + ema * vel_inst
            )

            reset_mask = (cur_step <= 1) | (cur_step < self._dyn_pred_prev_step)
            if bool(reset_mask.any().item()):
                self._dyn_pred_vel_obst_local[reset_mask] = 0.0
            self._dyn_pred_prev_obst_local = cur.clone()
            self._dyn_pred_prev_step = cur_step
            return self._dyn_pred_vel_obst_local
        except Exception:
            return None

    def _predictive_obstacles_for_env(
        self,
        env_id: int,
        centers_local: torch.Tensor,
        vel_local: torch.Tensor,
        base_radius: float,
    ) -> list[Obstacle]:
        """Create predicted obstacle samples over a short horizon for safer global plans."""
        out: list[Obstacle] = []
        if (
            (not self._dyn_pred_enable)
            or centers_local.numel() == 0
            or vel_local.numel() == 0
            or self._dyn_pred_horizon_sec <= 0.0
            or self._dyn_pred_samples <= 0
        ):
            return out

        x_min, y_min = self._rrt_bounds_min
        x_max, y_max = self._rrt_bounds_max
        min_speed = float(max(0.0, self._dyn_pred_min_speed))
        for j in range(int(centers_local.shape[0])):
            p = centers_local[j]
            v = vel_local[j]
            speed_xy = float(torch.norm(v[:2]).item())
            if speed_xy < min_speed:
                continue
            for k in range(1, int(self._dyn_pred_samples) + 1):
                t = float(self._dyn_pred_horizon_sec) * (float(k) / float(self._dyn_pred_samples))
                pred = p + v * t
                px = float(pred[0].item())
                py = float(pred[1].item())
                if (px < x_min) or (px > x_max) or (py < y_min) or (py > y_max):
                    continue
                rr_t = float(base_radius) + float(self._dyn_pred_inflate_add) + float(
                    self._dyn_pred_inflate_per_mps
                ) * speed_xy * t
                out.append(Obstacle((px, py, float(pred[2].item())), max(0.0, rr_t)))
        return out

    def _predictive_path_collision(
        self,
        pts_local_xy: torch.Tensor,
        obst_local_xy: torch.Tensor,
        obst_vel_xy: torch.Tensor,
        base_radius: float,
    ) -> tuple[bool, float, int]:
        """Check predicted dynamic-path collision; return (hit, earliest_ttc, obstacle_idx)."""
        if pts_local_xy.numel() == 0 or obst_local_xy.numel() == 0:
            return False, float("inf"), -1
        if not self._dyn_pred_replan_enable:
            return False, float("inf"), -1

        speed_assumed = float(max(0.05, self._dyn_pred_assumed_robot_speed))
        seg = pts_local_xy[1:] - pts_local_xy[:-1]
        seg_len = torch.norm(seg, dim=-1)
        cum = torch.zeros((pts_local_xy.shape[0],), device=pts_local_xy.device, dtype=pts_local_xy.dtype)
        if seg_len.numel() > 0:
            cum[1:] = torch.cumsum(seg_len, dim=0)
        t_arrive = cum / speed_assumed
        t_h = float(max(0.0, self._dyn_pred_replan_horizon_sec))
        margin = float(max(0.0, self._dyn_pred_replan_margin))
        inflate_gain = float(max(0.0, self._dyn_pred_inflate_per_mps))
        speed_obs = torch.norm(obst_vel_xy, dim=-1)
        best_t = float("inf")
        best_idx = -1

        for k in range(int(pts_local_xy.shape[0])):
            t = float(t_arrive[k].item())
            if t_h > 0.0 and t > t_h:
                break
            pred_xy = obst_local_xy + obst_vel_xy * t
            rr = base_radius + margin + (inflate_gain * t * speed_obs)
            d = torch.norm(pred_xy - pts_local_xy[k].unsqueeze(0), dim=-1)
            hit_mask = d <= rr
            if bool(hit_mask.any().item()):
                idx = torch.where(hit_mask)[0]
                if idx.numel() > 0 and t < best_t:
                    best_t = t
                    best_idx = int(idx[0].item())
        return (best_idx >= 0), best_t, best_idx

    def _plan_paths(self, env_ids: torch.Tensor):
        pos_w = self.drone.pos[env_ids, 0, :]
        # Fetch current dynamic obstacle positions once (world frame), then convert to local per env.
        obst_pos_local = None
        obst_vel_local = None
        if getattr(self, "enable_obstacles", False) and getattr(self, "obstacles", None) is not None:
            try:
                obst_pos_w, _ = self.obstacles.get_world_poses()
                obst_pos_w = obst_pos_w.to(self.device)
                if obst_pos_w.ndim == 2:
                    obst_pos_w = obst_pos_w.view(self.num_envs, self.num_obstacles, 3)
                obst_pos_local = obst_pos_w - self.envs_positions.unsqueeze(1)
                obst_vel_local = self._update_dynamic_obstacle_prediction(obst_pos_local)
            except Exception:
                obst_pos_local = None
                obst_vel_local = None

        for i, env_id in enumerate(env_ids.tolist()):
            # Plan in local coordinates per-env, then convert path back to world.
            env_offset = self.envs_positions[env_id]
            start_local = (pos_w[i] - env_offset).tolist()
            goal_local = (self.target_pos[env_id, 0, :] - env_offset).tolist()
            if self.debug and (int(self._plan_counts[env_id].item()) < 3):
                print(
                    "[HoverTankPPO] plan_local",
                    f"env={env_id}",
                    f"start_local={start_local}",
                    f"goal_local={goal_local}",
                    f"bounds_min={self.planner_cfg.get('bounds_min')}",
                    f"bounds_max={self.planner_cfg.get('bounds_max')}",
                )
            obstacles = []
            if self.enable_obstacles:
                radius = float(self.obst_cfg.get("radius", 0.15))
                inflate = float(self.planner_cfg.get("obstacle_inflate", 0.0))
                rr = max(0.0, radius + inflate)
                if obst_pos_local is not None:
                    for pos in obst_pos_local[env_id].tolist():
                        obstacles.append(Obstacle(tuple(pos), rr))
                    if self._dyn_pred_enable and obst_vel_local is not None:
                        obstacles.extend(
                            self._predictive_obstacles_for_env(
                                env_id=env_id,
                                centers_local=obst_pos_local[env_id],
                                vel_local=obst_vel_local[env_id],
                                base_radius=rr,
                            )
                        )
                else:
                    for pos in self.obst_center[env_id].tolist():
                        obstacles.append(Obstacle(tuple(pos), rr))
            plan = self.planner.plan(
                start_w=start_local,
                goal_w=goal_local,
                obstacles=obstacles,
                occ_grid=(
                    None
                    if not bool(self.planner_cfg.get("use_map", False))
                    or self._map_occ is None
                    else self._map_occ[env_id].detach().cpu().float().numpy().tolist()
                ),
                occ_bounds_xy=(self._map_x_min, self._map_x_max, self._map_y_min, self._map_y_max)
                if bool(self.planner_cfg.get("use_map", False)) and self.mapping_enabled
                else None,
            )
            goal_w = self.target_pos[env_id, 0, :].detach().cpu().numpy().tolist()
            # Convert path to world coordinates
            if plan:
                plan = [(p[0] + env_offset[0].item(), p[1] + env_offset[1].item(), p[2] + env_offset[2].item()) for p in plan]
                # Planner paths include the start pose as the first node; drop it so the "current waypoint"
                # is a point ahead of the robot rather than the robot's exact current position.
                if len(plan) > 1:
                    plan = plan[1:]
            self._paths[env_id] = plan
            self._path_idx[env_id] = 0
            self._plan_counts[env_id] += 1
            if self._plot_rrt_every > 0 and (self._plan_counts[env_id] % self._plot_rrt_every == 0):
                try:
                    import matplotlib.pyplot as plt
                    xs = [p[0] for p in plan] if plan else []
                    ys = [p[1] for p in plan] if plan else []
                    zs = [p[2] for p in plan] if plan else []
                    x_min, y_min = self._rrt_bounds_min
                    x_max, y_max = self._rrt_bounds_max
                    z_min, z_max = self._rrt_bounds_z
                    width = max(x_max - x_min, 1.0)
                    height = max(y_max - y_min, 1.0)
                    fig_w = 6.0
                    fig_h = max(3.0, fig_w * (height / width))
                    if self._plot_rrt_3d:
                        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
                        fig = plt.figure(figsize=(fig_w, fig_h))
                        ax = fig.add_subplot(111, projection="3d")
                        ax.plot(xs, ys, zs, "-o", markersize=2, linewidth=1.0, label="RRT")
                        ax.scatter(
                            [pos_w[i, 0].item()],
                            [pos_w[i, 1].item()],
                            [pos_w[i, 2].item()],
                            c="blue",
                            s=20,
                            label="start",
                        )
                        ax.scatter([goal_w[0]], [goal_w[1]], [goal_w[2]], c="red", s=20, label="goal")
                        ax.set_xlim(x_min, x_max)
                        ax.set_ylim(y_min, y_max)
                        ax.set_zlim(z_min, z_max)
                    else:
                        fig, ax = plt.subplots(figsize=(fig_w, fig_h))
                        ax.plot(xs, ys, "-o", markersize=2, linewidth=1.0, label="RRT")
                        ax.scatter([pos_w[i, 0].item()], [pos_w[i, 1].item()], c="blue", s=20, label="start")
                        ax.scatter([goal_w[0]], [goal_w[1]], c="red", s=20, label="goal")
                        ax.set_aspect("equal", adjustable="box")
                        ax.set_xlim(x_min, x_max)
                        ax.set_ylim(y_min, y_max)
                    ax.set_title(f"env_{env_id} plan_{int(self._plan_counts[env_id].item())}")
                    ax.legend(loc="best")
                    fig.canvas.draw()
                    w, h = fig.canvas.get_width_height()
                    rgba = np.asarray(fig.canvas.buffer_rgba())
                    if rgba.shape[0] != h or rgba.shape[1] != w:
                        rgba = rgba.reshape(h, w, 4)
                    frame = rgba[:, :, :3].copy()
                    plt.close(fig)
                    history = self._rrt_plot_paths.get(env_id, [])
                    history.append(frame)
                    if len(history) > self._plot_rrt_history:
                        history = history[-self._plot_rrt_history:]
                    self._rrt_plot_paths[env_id] = history
                    if self._plot_rrt_log_wandb:
                        import wandb
                        self._wandb_log(
                            {
                                "viz/rrt_path": wandb.Image(
                                    frame,
                                    caption=f"env_{env_id}_plan_{int(self._plan_counts[env_id].item())}",
                                )
                            }
                        )
                    if self._plot_rrt_gif and (self._plan_counts[env_id] % self._plot_rrt_gif_every == 0):
                        if self._plot_rrt_log_gif_wandb and history:
                            try:
                                import wandb
                                frames = np.stack(history, axis=0)
                                self._wandb_log(
                                    {"viz/rrt_gif": wandb.Video(frames, fps=5, format="gif")}
                                )
                            except Exception as gif_exc:
                                if self.debug:
                                    print(f"[HoverTankPPO] rrt gif failed: {gif_exc}")
                except Exception as exc:
                    if self.debug:
                        print(f"[HoverTankPPO] rrt plot failed: {exc}")

    def _current_waypoints(self) -> torch.Tensor:
        waypoints = []
        for env_idx in range(self.num_envs):
            idx = int(self._path_idx[env_idx].item())
            path = self._paths[env_idx]
            idx = min(idx, len(path) - 1) if path else 0
            waypoints.append(path[idx] if path else self.target_pos[env_idx, 0, :].tolist())
        return torch.tensor(waypoints, device=self.device)

    def _build_allocation(self):
        rotor_pos_w, rotor_rot_w = self.drone.rotors_view.get_world_poses()
        r_w = rotor_pos_w[:, 0]
        q_w = rotor_rot_w[:, 0]

        base_pos_w, base_rot_w = self.drone.base_link.get_world_poses()
        base_pos_w = base_pos_w[:, 0]
        base_rot_w = base_rot_w[:, 0]

        thrust_axis = int(self.ctrl_cfg.get("thrust_axis", 0))
        thrust_sign = float(self.ctrl_cfg.get("thrust_sign", 1.0))
        # Keep the vehicle’s thrust-axis convention consistent with the allocation matrix.
        # (UnderwaterVehicle uses this to interpret rotor USD orientations.)
        try:
            setattr(self.drone, "ctrl_cfg", self.ctrl_cfg)
            setattr(self.drone, "thrust_axis", int(thrust_axis))
            setattr(self.drone, "thrust_sign", float(thrust_sign))
        except Exception:
            pass
        d_w = quat_axis(q_w, axis=thrust_axis) * thrust_sign
        r_rel_w = r_w - base_pos_w.unsqueeze(1)
        base_q = base_rot_w.unsqueeze(1).expand_as(q_w)
        r_b = quat_rotate_inverse(base_q, r_rel_w)
        d_b = quat_rotate_inverse(base_q, d_w)
        tau_b = torch.cross(r_b, d_b, dim=-1)

        alloc_mode = str(self.ctrl_cfg.get("allocation_mode", "4dof")).lower()
        if alloc_mode in {"4dof", "4d", "marine", "4dof_balanced"}:
            # BlueROV control space: [Fx, Fy, Fz, Tz].
            if alloc_mode == "4dof_balanced":
                # Weighted 6-DOF solve for 4-DOF command:
                # track Fx/Fy/Fz/Tz while suppressing Tx/Ty to reduce tilt-induced cross coupling.
                w_force = float(self.ctrl_cfg.get("allocation_weight_force", 1.0))
                w_rp = float(self.ctrl_cfg.get("allocation_weight_rp", 3.0))
                w_yaw = float(self.ctrl_cfg.get("allocation_weight_yaw", 1.0))
                A = torch.zeros(self.num_envs, 6, r_b.shape[1], device=self.device)
                A[:, 0, :] = d_b[:, :, 0] * w_force
                A[:, 1, :] = d_b[:, :, 1] * w_force
                A[:, 2, :] = d_b[:, :, 2] * w_force
                A[:, 3, :] = tau_b[:, :, 0] * w_rp
                A[:, 4, :] = tau_b[:, :, 1] * w_rp
                A[:, 5, :] = tau_b[:, :, 2] * w_yaw
                self._alloc_ctrl_rows = (0, 1, 2, 3, 4, 5)
                self._alloc_mode_runtime = "4dof_balanced"
                self._alloc_weights_runtime = (w_force, w_rp, w_yaw)
            else:
                A = torch.zeros(self.num_envs, 4, r_b.shape[1], device=self.device)
                A[:, 0, :] = d_b[:, :, 0]
                A[:, 1, :] = d_b[:, :, 1]
                A[:, 2, :] = d_b[:, :, 2]
                A[:, 3, :] = tau_b[:, :, 2]
                self._alloc_ctrl_rows = (0, 1, 2, 5)
                self._alloc_mode_runtime = "4dof"
                self._alloc_weights_runtime = (1.0, 1.0, 1.0)
        else:
            A = torch.zeros(self.num_envs, 6, r_b.shape[1], device=self.device)
            A[:, 0:3, :] = d_b.transpose(1, 2)
            A[:, 3:6, :] = tau_b.transpose(1, 2)
            self._alloc_ctrl_rows = (0, 1, 2, 3, 4, 5)
            self._alloc_mode_runtime = "6dof"
            self._alloc_weights_runtime = (1.0, 1.0, 1.0)

        # Optional rotor channel calibration: solve in calibrated channel order, then
        # map commands back to the simulator's native rotor order before apply_action().
        n_rotors = A.shape[-1]
        perm_tensor = None
        perm_cfg = self.ctrl_cfg.get("allocation_rotor_permutation", None)
        if perm_cfg is not None:
            try:
                p = torch.as_tensor(perm_cfg, device=self.device, dtype=torch.long).view(-1)
                if p.numel() == n_rotors:
                    perm_tensor = p
            except Exception:
                perm_tensor = None
        elif bool(self.ctrl_cfg.get("allocation_auto_rotor_permutation", True)):
            try:
                prim_paths = list(getattr(self.drone.rotors_view, "prim_paths", []))
                if len(prim_paths) == n_rotors:
                    rotor_ids = []
                    for p in prim_paths:
                        m = re.search(r"rotor_(\d+)", str(p))
                        rotor_ids.append(int(m.group(1)) if m else -1)
                    if min(rotor_ids) >= 0:
                        perm_tensor = torch.argsort(torch.as_tensor(rotor_ids, device=self.device, dtype=torch.long))
            except Exception:
                perm_tensor = None
        if perm_tensor is None:
            perm_tensor = torch.arange(n_rotors, device=self.device, dtype=torch.long)

        sign_tensor = torch.ones(n_rotors, device=self.device)
        sign_cfg = self.ctrl_cfg.get("allocation_rotor_sign", None)
        if sign_cfg is not None:
            try:
                s = torch.as_tensor(sign_cfg, device=self.device, dtype=torch.float32).view(-1)
                if s.numel() == n_rotors:
                    sign_tensor = torch.where(s >= 0.0, torch.ones_like(s), -torch.ones_like(s))
            except Exception:
                pass
        # Apply calibration in solver space.
        A = A.index_select(-1, perm_tensor) * sign_tensor.view(1, 1, -1)
        inv_perm = torch.empty_like(perm_tensor)
        inv_perm[perm_tensor] = torch.arange(n_rotors, device=self.device, dtype=torch.long)
        self._alloc_perm_runtime = perm_tensor
        self._alloc_inv_perm_runtime = inv_perm
        self._alloc_sign_runtime = sign_tensor
        finite_mask = torch.isfinite(A).all(dim=(1, 2))
        if self._A_pinv is None or self._A_pinv.shape[-1] != A.shape[1]:
            self._A_pinv = torch.zeros(self.num_envs, r_b.shape[1], A.shape[1], device=self.device)

        if finite_mask.any():
            A_good = A[finite_mask]
            damping = float(self.ctrl_cfg.get("allocation_damping", 0.0))
            if damping > 0.0:
                # Damped least-squares inverse improves numerical stability when A is near-singular.
                m = A_good.shape[1]
                eye = torch.eye(m, device=self.device, dtype=A_good.dtype).unsqueeze(0).expand(A_good.shape[0], -1, -1)
                lhs = A_good @ A_good.transpose(-1, -2) + (damping * damping) * eye
                rhs = A_good.transpose(-1, -2)
                pinv_good = rhs @ torch.linalg.inv(lhs)
            else:
                pinv_good = torch.linalg.pinv(A_good)
            self._A_pinv[finite_mask] = pinv_good
            if self._A_pinv_default is None:
                # All envs share the same vehicle model; keep a stable fallback for envs whose
                # rotor poses may briefly go non-finite (e.g., right after reset / sim hiccups).
                self._A_pinv_default = pinv_good[0].detach().clone()

        if not finite_mask.all():
            bad_envs = torch.where(~finite_mask)[0]
            if self._A_pinv_default is not None:
                self._A_pinv[bad_envs] = self._A_pinv_default
            else:
                self._A_pinv[bad_envs] = 0.0
            if self.debug:
                print(f"[HoverTankPPO] allocation non-finite envs={bad_envs.tolist()}")

    def _build_thruster_inverse_lut(self):
        if self._thr_lut_ready:
            return
        # Steady-state inversion of T200 model used in marinegym/actuators/t200.py.
        # This avoids the old linear f_max scaling, which over-saturates commands.
        u = torch.linspace(-1.0, 1.0, 4097, device=self.device)
        rpm = torch.where(
            u > 0.075,
            3.6599e3 * u + 3.4521e2,
            torch.where(u < -0.075, 3.4944e3 * u - 4.3350e2, torch.zeros_like(u)),
        )
        poly_pos = 4.7368e-07 * rpm.square() - 1.9275e-04 * rpm + 8.4452e-02
        poly_neg = -3.8442e-07 * rpm.square() - 1.6186e-04 * rpm - 3.9139e-02
        poly = torch.where(rpm > 0.0, poly_pos, torch.where(rpm < 0.0, poly_neg, torch.zeros_like(rpm)))
        force_scale = float((self.drone.FORCE_CONSTANTS_0[0] / 4.4e-7).detach().cpu().item())
        thrust = force_scale * 9.81 * poly

        pos_mask = u > 0.075
        neg_mask = u < -0.075
        self._thr_u_pos = u[pos_mask]
        self._thr_f_pos = thrust[pos_mask]
        self._thr_u_neg = u[neg_mask]
        self._thr_f_neg = thrust[neg_mask]
        self._thr_lut_ready = True

    @staticmethod
    def _interp_1d_clamped(x: torch.Tensor, xp: torch.Tensor, fp: torch.Tensor) -> torch.Tensor:
        if xp.numel() < 2:
            return torch.zeros_like(x)
        xq = x.clamp(xp[0], xp[-1])
        idx = torch.searchsorted(xp, xq)
        idx1 = idx.clamp(1, xp.numel() - 1)
        idx0 = idx1 - 1
        x0 = xp[idx0]
        x1 = xp[idx1]
        y0 = fp[idx0]
        y1 = fp[idx1]
        t = (xq - x0) / (x1 - x0 + 1e-9)
        return y0 + t * (y1 - y0)

    def _forces_to_thr_action(self, forces: torch.Tensor) -> torch.Tensor:
        # forces: [E, N] thrust in Newton per thruster.
        self._build_thruster_inverse_lut()
        out = torch.zeros_like(forces)
        pos = forces > 0.0
        neg = forces < 0.0
        if pos.any():
            out[pos] = self._interp_1d_clamped(forces[pos], self._thr_f_pos, self._thr_u_pos)
        if neg.any():
            out[neg] = self._interp_1d_clamped(forces[neg], self._thr_f_neg, self._thr_u_neg)
        return out.clamp(-1.0, 1.0)

    def _get_hardcoded_axis_mix(self, dtype: torch.dtype) -> torch.Tensor | None:
        """Return configured 4x4 low-level action mix matrix on the current device."""
        mix_cfg = self.ctrl_cfg.get("hardcoded_axis_mix", None)
        if mix_cfg is None:
            return None
        cache_key = repr(mix_cfg)
        if getattr(self, "_axis_mix_cache_key", None) == cache_key:
            cached = getattr(self, "_axis_mix_cache_tensor", None)
            if torch.is_tensor(cached):
                return cached.to(device=self.device, dtype=dtype)
        try:
            mix = torch.as_tensor(mix_cfg, device=self.device, dtype=dtype)
            if mix.shape != (4, 4) or (not torch.isfinite(mix).all()):
                return None
            self._axis_mix_cache_key = cache_key
            self._axis_mix_cache_tensor = mix.detach().clone()
            return mix
        except Exception:
            return None

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

    def _pre_sim_step(self, tensordict):
        self._update_obstacles()
        self.drone.get_state(env_frame=False)

        pos_w = self.drone.pos[:, 0, :]
        rot_w = self.drone.rot[:, 0, :]
        bad_state = ~torch.isfinite(pos_w).all(dim=-1) | ~torch.isfinite(rot_w).all(dim=-1)
        if bad_state.any():
            bad_envs = torch.where(bad_state)[0]
            if self.debug:
                print(f"[HoverTankPPO] reset non-finite (pre_step) envs={bad_envs.tolist()}")
            self._reset_idx(bad_envs)
            self.drone.get_state(env_frame=False)
            if self._reset_cooldown is not None:
                self._reset_cooldown[bad_envs] = self.reset_cooldown_steps

        if self.depth_enabled and self.depth_camera_view is not None:
            cam_offset = torch.tensor(
                self.depth_cfg.get("offset", [0.2, 0.0, 0.0]),
                device=self.device,
                dtype=pos_w.dtype,
            )
            cam_offset = cam_offset.unsqueeze(0).expand(pos_w.shape[0], 3)
            cam_pos = pos_w + quat_rotate(rot_w, cam_offset)
            cam_rot = rot_w
            if self._depth_rot_offset_q is not None:
                cam_rot = self._quat_mul(rot_w, self._depth_rot_offset_q.expand_as(rot_w))
            self.depth_camera_view.set_world_poses(
                positions=cam_pos,
                orientations=cam_rot,
            )

        apply_post = self.run_mode != "train"

        def _apply_action_remap(a: torch.Tensor) -> torch.Tensor:
            """Optionally remap PPO action channels.

            The policy always outputs a 4D vector, but in practice it’s easy to end up
            with an axis/sign mismatch (e.g., sway sign flipped). This hook lets us
            reorder and sign-flip actions without retraining the controller stack.
            """
            order = self.ctrl_cfg.get("action_order", None)
            sign = self.ctrl_cfg.get("action_sign", None)
            if order is None and sign is None:
                return a
            out = a
            if order is not None:
                try:
                    idx = torch.as_tensor(order, device=out.device, dtype=torch.long).view(-1)
                    if idx.numel() == out.shape[-1]:
                        out = out.index_select(-1, idx)
                except Exception:
                    pass
            if sign is not None:
                try:
                    s = torch.as_tensor(sign, device=out.device, dtype=out.dtype).view(-1)
                    if s.numel() == out.shape[-1]:
                        out = out * s
                except Exception:
                    pass
            return out

        actions_policy = tensordict[("agents", "action")]
        # Keep a copy of the policy output (before any env-side remap/smoothing/clamping).
        # IMPORTANT: do not mutate `tensordict[("agents","action")]` during training, otherwise
        # PPO's stored `sample_log_prob` no longer matches the action in the rollout.
        try:
            self._last_action_policy = actions_policy.detach().clone()
        except Exception:
            self._last_action_policy = None
        # Keep NaN/Inf sanitization in all modes to avoid sim instability.
        actions_policy_safe = torch.nan_to_num(actions_policy, nan=0.0, posinf=0.0, neginf=0.0)
        if self.run_mode == "train":
            if actions_policy_safe.ndim != 3 or tuple(actions_policy_safe.shape[1:]) != (1, 4):
                raise RuntimeError(
                    f"Bad action shape {tuple(actions_policy_safe.shape)} expected [E,1,4]"
                )
        a_exec = actions_policy_safe

        # Debug: force a constant action to validate axis conventions (surge/sway/heave/yaw).
        fixed_action = self.ctrl_cfg.get("debug_fixed_action", None)
        fixed_steps = int(self.ctrl_cfg.get("debug_fixed_steps", 0))
        used_fixed_action = False
        if self.run_mode == "play" and fixed_action is not None and fixed_steps > 0:
            if not hasattr(self, "_debug_fixed_count"):
                self._debug_fixed_count = 0
            if self._debug_fixed_count < fixed_steps:
                try:
                    fa = torch.tensor(fixed_action, device=self.device, dtype=a_exec.dtype).view(1, 1, 4)
                    a_exec = fa.expand_as(a_exec)
                    used_fixed_action = True
                except Exception:
                    pass
                self._debug_fixed_count += 1

        # Optional action remap (only for policy actions; fixed-action tests should stay literal).
        if apply_post and (not used_fixed_action or bool(self.ctrl_cfg.get("remap_fixed_action", False))):
            a_exec = _apply_action_remap(a_exec)
        try:
            self._last_action_remapped = a_exec.detach().clone()
        except Exception:
            self._last_action_remapped = None

        if self._prev_action is None:
            self._prev_action = torch.zeros_like(a_exec)
        if apply_post:
            alpha = float(self.ctrl_cfg.get("action_smooth_alpha", 0.0))
            a_exec = alpha * self._prev_action + (1.0 - alpha) * a_exec
            self._prev_action = a_exec.detach()
        else:
            # Training path: keep action mapping memoryless.
            a_exec = actions_policy_safe
        try:
            self._last_action_smoothed = a_exec.detach().clone()
        except Exception:
            self._last_action_smoothed = None

        # Safety override using directional sonar (front cone) + analytic clearance.
        min_range = self._wall_clearance(pos_w)
        maze_clear = self._maze_clearance(pos_w)
        if maze_clear is not None:
            min_range = torch.minimum(min_range, maze_clear)
        obst_clear = self._obstacle_clearance(pos_w)
        if obst_clear is not None:
            min_range = torch.minimum(min_range, obst_clear)

        stop_dist = float(self.policy_cfg.get("min_range_stop", 0.5))
        replan_dist = float(self.policy_cfg.get("min_range_replan", 0.8))
        avoid_dist = float(self.policy_cfg.get("min_range_avoid", 1.2))
        front_deg = float(self.policy_cfg.get("front_safety_deg", 90.0))
        use_front = bool(self.policy_cfg.get("use_front_safety", True))

        safety_stop = min_range < stop_dist
        safety_avoid = min_range < avoid_dist
        if use_front and front_deg > 0.0:
            fallback = self._fallback_sonar_ranges(pos_w, rot_w)
            if fallback is not None:
                angles = self.sonar.angles.to(self.device)
                front_mask = angles.abs() <= (front_deg * torch.pi / 180.0) * 0.5
                front_min = torch.min(fallback[:, front_mask], dim=1)[0]
                safety_stop = front_min < stop_dist
                safety_avoid = front_min < avoid_dist
                self._last_front_min = front_min
        else:
            self._last_front_min = None
        safety_replan = (min_range < replan_dist) & ~safety_stop
        if not apply_post:
            safety_stop = torch.zeros_like(safety_stop)
            safety_avoid = torch.zeros_like(safety_avoid)
            safety_replan = torch.zeros_like(safety_replan)
        self._last_min_range = min_range
        self._last_safety_stop = safety_stop
        self._last_safety_avoid = safety_avoid
        self._last_safety_replan = safety_replan

        # Stuck replan trigger: low speed + not at goal for N steps.
        dist_goal = torch.norm(self.target_pos[:, 0, :] - pos_w, dim=-1)
        speed = torch.norm(self.drone.vel_b[:, 0, :3], dim=-1)
        stuck = (speed < self._stuck_speed_thresh) & (dist_goal > self.success_distance)
        self._stuck_steps = torch.where(
            stuck, self._stuck_steps + 1, torch.zeros_like(self._stuck_steps)
        )
        replan_stuck = self._stuck_steps >= self._stuck_replan_steps
        if replan_stuck.any():
            self._stuck_steps = torch.where(
                replan_stuck, torch.zeros_like(self._stuck_steps), self._stuck_steps
            )

        # Path validity replan: if dynamic obstacles move into the current global path, replan early
        # (not only when the robot is already close to them).
        path_replan = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        path_replan_reason: list[str] = ["" for _ in range(self.num_envs)]
        path_replan_ttc = torch.full((self.num_envs,), float("inf"), device=self.device)
        path_replan_obst_idx = torch.full((self.num_envs,), -1, dtype=torch.long, device=self.device)
        try:
            enable_path_replan = bool(self.planner_cfg.get("path_replan_enable", True))
            lookahead = int(self.planner_cfg.get("path_replan_lookahead", 30))
            if enable_path_replan and lookahead > 0 and getattr(self, "_paths", None) is not None:
                # 1) Map-based check (if enabled).
                use_map = bool(self.planner_cfg.get("use_map", False)) and self._map_occ is not None
                map_thresh = float(self.planner_cfg.get("map_path_replan_threshold", self.planner_cfg.get("map_occ_threshold", 0.5)))

                # 2) Dynamic obstacle-based check (always available if obstacles enabled).
                use_dyn = getattr(self, "enable_obstacles", False) and getattr(self, "obstacles", None) is not None
                rr = float(self.obst_cfg.get("radius", 0.15)) + float(self.planner_cfg.get("obstacle_inflate", 0.0))
                obst_local_xyz = None
                obst_vel_xyz = None
                if use_dyn:
                    obst_pos_w, _ = self.obstacles.get_world_poses()
                    obst_pos_w = obst_pos_w.to(self.device)
                    if obst_pos_w.ndim == 2:
                        obst_pos_w = obst_pos_w.view(self.num_envs, self.num_obstacles, 3)
                    obst_local_xyz = (obst_pos_w - self.envs_positions.unsqueeze(1))
                    obst_local_xy = obst_local_xyz[:, :, :2]
                    obst_vel_xyz = self._update_dynamic_obstacle_prediction(obst_local_xyz)
                    obst_vel_xy = (
                        None if obst_vel_xyz is None else obst_vel_xyz[:, :, :2]
                    )

                for e in range(self.num_envs):
                    path = self._paths[e]
                    if not path:
                        continue
                    start_i = int(self._path_idx[e].item()) if hasattr(self, "_path_idx") else 0
                    if start_i < 0:
                        start_i = 0
                    end_i = min(len(path), start_i + lookahead)
                    if end_i <= start_i:
                        continue
                    pts_w = torch.tensor(path[start_i:end_i], device=self.device, dtype=pos_w.dtype)
                    pts_local = pts_w - self.envs_positions[e].unsqueeze(0)
                    if use_map:
                        ix, iy = self._map_world_to_grid(pts_local[:, 0], pts_local[:, 1])
                        if (self._map_occ[e, iy, ix] >= map_thresh).any():
                            path_replan[e] = True
                            path_replan_reason[e] = "map_occ"
                            continue
                    if use_dyn:
                        d = torch.cdist(pts_local[:, :2], obst_local_xy[e])
                        if (d <= rr).any():
                            path_replan[e] = True
                            path_replan_reason[e] = "dyn_path_overlap"
                            ii = torch.where(d <= rr)
                            if len(ii) >= 2 and ii[1].numel() > 0:
                                path_replan_obst_idx[e] = int(ii[1][0].item())
                            continue
                        if (
                            self._dyn_pred_enable
                            and obst_vel_xyz is not None
                        ):
                            pred_hit, pred_ttc, pred_obst_idx = self._predictive_path_collision(
                                pts_local_xy=pts_local[:, :2],
                                obst_local_xy=obst_local_xy[e],
                                obst_vel_xy=obst_vel_xy[e],
                                base_radius=rr,
                            )
                            if pred_hit:
                                path_replan[e] = True
                                path_replan_reason[e] = "dyn_predictive"
                                path_replan_ttc[e] = float(pred_ttc)
                                path_replan_obst_idx[e] = int(pred_obst_idx)
                                continue
        except Exception:
            pass

        replan_mask = safety_replan | replan_stuck | path_replan
        if replan_mask.any():
            env_ids = torch.where(replan_mask)[0]
            if self._replan_live_debug:
                for e in env_ids.tolist():
                    reasons: list[str] = []
                    if bool(safety_replan[e].item()):
                        reasons.append("safety_range")
                    if bool(replan_stuck[e].item()):
                        reasons.append("stuck")
                    if bool(path_replan[e].item()):
                        reasons.append(path_replan_reason[e] if path_replan_reason[e] else "path")
                    reason_str = "+".join(reasons) if reasons else "unknown"
                    ttc_val = float(path_replan_ttc[e].item())
                    obst_idx = int(path_replan_obst_idx[e].item())
                    step_e = int(self.progress_buf[e].item()) if hasattr(self, "progress_buf") else -1
                    min_r_e = float(min_range[e].item()) if min_range is not None else float("nan")
                    goal_d_e = float(dist_goal[e].item()) if dist_goal is not None else float("nan")
                    pidx_e = int(self._path_idx[e].item()) if hasattr(self, "_path_idx") else -1
                    plen_e = len(self._paths[e]) if getattr(self, "_paths", None) is not None else 0
                    print(
                        "[HoverTankPPO][replan]",
                        f"env={e}",
                        f"step={step_e}",
                        f"reason={reason_str}",
                        f"pred_ttc={ttc_val:.2f}" if math.isfinite(ttc_val) else "pred_ttc=inf",
                        f"obst_idx={obst_idx}" if obst_idx >= 0 else "obst_idx=na",
                        f"min_range={min_r_e:.3f}",
                        f"goal_dist={goal_d_e:.3f}",
                        f"path_idx={pidx_e}/{plen_e}",
                    )
            self._plan_paths(env_ids)

        waypoint_w = self._current_waypoints()
        dist_to_wp = torch.norm(waypoint_w - self.drone.pos[:, 0, :], dim=-1)
        reached = dist_to_wp < self._waypoint_tol
        if reached.any():
            for env_idx in torch.where(reached)[0].tolist():
                new_idx = min(
                    int(self._path_idx[env_idx].item()) + 1,
                    max(len(self._paths[env_idx]) - 1, 0),
                )
                self._path_idx[env_idx] = new_idx
            # Prevent large negative "progress" spikes when switching to a new waypoint by
            # immediately resetting the progress reference distance to the new waypoint.
            if self.use_waypoint_reward and self._prev_goal_dist is not None:
                try:
                    env_ids = torch.where(reached)[0]
                    wp_new = self._current_waypoints()[env_ids]
                    pos_new = self.drone.pos[env_ids, 0, :]
                    self._prev_goal_dist[env_ids] = torch.norm(wp_new - pos_new, dim=-1, keepdim=True)
                except Exception:
                    pass
        self._last_waypoint_reached = reached

        act = a_exec.squeeze(1)
        if act.dim() == 1:
            act = act.unsqueeze(0)
        deadband = float(self.ctrl_cfg.get("action_deadband", 0.0)) if apply_post else 0.0
        if apply_post and deadband > 0.0:
            act = torch.where(act.abs() < deadband, torch.zeros_like(act), act)
        # Optional hardcoded 4x4 action mixer to compensate persistent low-level coupling.
        # a_phys = M * a_cmd, where rows are [surge,sway,heave,yaw] physical commands.
        if bool(self.ctrl_cfg.get("hardcoded_axis_mix_enable", False)):
            mix = self._get_hardcoded_axis_mix(act.dtype)
            if torch.is_tensor(mix):
                act = act @ mix.transpose(0, 1)
            elif not self._axis_mix_warned_invalid:
                self._axis_mix_warned_invalid = True
                print(
                    "[HoverTankPPO] WARNING: controller.hardcoded_axis_mix_enable=true "
                    "but controller.hardcoded_axis_mix is missing/invalid (expected finite 4x4). "
                    "Ignoring hardcoded axis mix."
                )
        # Optional dynamic RP suppression scheduling: use stronger roll/pitch suppression for
        # horizontal motion, and lighter suppression for heave-dominant commands.
        if bool(self.ctrl_cfg.get("allocation_dynamic_rp_enable", False)):
            ratio = float(self.ctrl_cfg.get("allocation_dynamic_rp_ratio", 1.2))
            dead = float(self.ctrl_cfg.get("allocation_dynamic_rp_deadband", 0.05))
            rp_xy = float(self.ctrl_cfg.get("allocation_weight_rp_xy", self.ctrl_cfg.get("allocation_weight_rp", 1.0)))
            rp_z = float(self.ctrl_cfg.get("allocation_weight_rp_z", self.ctrl_cfg.get("allocation_weight_rp", 1.0)))
            aabs = act.abs()
            max_xy = torch.maximum(aabs[:, 0], aabs[:, 1])
            z_dom = (aabs[:, 2] > dead) & (aabs[:, 2] > ratio * max_xy) & (aabs[:, 2] > ratio * aabs[:, 3])
            target_rp = rp_z if bool(z_dom.float().mean() > 0.5) else rp_xy
            current_rp = float(self.ctrl_cfg.get("allocation_weight_rp", rp_xy))
            if abs(current_rp - target_rp) > 1e-6:
                self.ctrl_cfg["allocation_weight_rp"] = target_rp
                # Force reallocation rebuild when the objective weights change.
                self._A_pinv = None
        if apply_post:
            act = act.clamp(-1.0, 1.0)
        self._last_action_clamped = act
        u_max = float(self.ctrl_cfg.get("u_max", 0.6))
        v_max = float(self.ctrl_cfg.get("v_max", 0.4))
        w_max = float(self.ctrl_cfg.get("w_max", 0.3))
        r_max = float(self.ctrl_cfg.get("r_max", 0.6))
        v_cmd_b = torch.zeros(self.num_envs, 3, device=self.device)
        v_cmd_b[:, 0] = act[:, 0] * u_max
        v_cmd_b[:, 1] = act[:, 1] * v_max
        v_cmd_b[:, 2] = act[:, 2] * w_max
        r_cmd = act[:, 3] * r_max
        # Optional low-level smoothing in physical command units (m/s, rad/s).
        # This is useful for BT play where aggressive command jumps can excite cross-axis coupling.
        if apply_post:
            v_alpha = float(self.ctrl_cfg.get("v_cmd_smooth_alpha", 0.0))
            r_alpha = float(self.ctrl_cfg.get("r_cmd_smooth_alpha", v_alpha))
            if v_alpha > 0.0 or r_alpha > 0.0:
                if (not hasattr(self, "_prev_v_cmd_b")) or self._prev_v_cmd_b is None:
                    self._prev_v_cmd_b = torch.zeros_like(v_cmd_b)
                if (not hasattr(self, "_prev_r_cmd")) or self._prev_r_cmd is None:
                    self._prev_r_cmd = torch.zeros_like(r_cmd)
                v_alpha = max(0.0, min(1.0, v_alpha))
                r_alpha = max(0.0, min(1.0, r_alpha))
                v_cmd_b = (1.0 - v_alpha) * v_cmd_b + v_alpha * self._prev_v_cmd_b
                r_cmd = (1.0 - r_alpha) * r_cmd + r_alpha * self._prev_r_cmd
                self._prev_v_cmd_b = v_cmd_b.detach().clone()
                self._prev_r_cmd = r_cmd.detach().clone()
        self._last_v_cmd_b = v_cmd_b
        # If tilted too much, slow commands to let stabilizer recover.
        tilt_limit_deg = float(self.cfg.task.get("tilt_limit_deg", 10.0)) if apply_post else 0.0
        tilt_slow = float(self.cfg.task.get("tilt_slow_scale", 0.2)) if apply_post else 1.0
        if apply_post and tilt_limit_deg > 0.0 and tilt_slow < 1.0:
            rpy = quaternion_to_euler(rot_w)
            tilt = rpy[:, :2].abs().max(dim=-1).values
            tilt_limit = tilt_limit_deg * torch.pi / 180.0
            tilt_mask = tilt > tilt_limit
            if tilt_mask.any():
                v_cmd_b[tilt_mask] *= tilt_slow
                r_cmd[tilt_mask] *= tilt_slow
        # Prevent commanding upward motion above the water surface.
        if apply_post and self.termination_max_z is not None:
            above_surface = pos_w[:, 2] > self.termination_max_z
            if above_surface.any():
                v_cmd_b[above_surface, 2] = torch.minimum(
                    v_cmd_b[above_surface, 2],
                    torch.zeros_like(v_cmd_b[above_surface, 2]),
                )
            self._last_above_surface = above_surface
            # Soft surface avoidance: if we get close to the surface, bias commands downward to
            # counter buoyancy before we trip the termination condition.
            surface_margin = float(self.cfg.task.get("surface_avoid_margin", 0.15))
            surface_push_w = float(self.cfg.task.get("surface_avoid_w", 0.15))
            if surface_margin > 0.0 and surface_push_w > 0.0:
                near_surface = pos_w[:, 2] > (self.termination_max_z - surface_margin)
                if near_surface.any():
                    v_cmd_b[near_surface, 2] = torch.minimum(
                        v_cmd_b[near_surface, 2],
                        v_cmd_b.new_full(v_cmd_b[near_surface, 2].shape, -surface_push_w),
                    )
        else:
            self._last_above_surface = None
        if apply_post and safety_stop.any():
            v_cmd_b[safety_stop] = 0.0
            r_cmd[safety_stop] = 0.0

        # During cooldown (right after reset), suppress lateral/yaw commands and keep depth near the
        # reset depth. This prevents buoyancy-driven "rocket to surface" loops in multi-env runs.
        if apply_post and self._reset_cooldown is not None:
            cooldown_mask = self._reset_cooldown > 0
            if cooldown_mask.any():
                v_cmd_b[cooldown_mask, 0:2] = 0.0
                r_cmd[cooldown_mask] = 0.0
                cooldown_kp = float(self.ctrl_cfg.get("cooldown_depth_kp", 1.0))
                z_err = self._cooldown_z_ref[cooldown_mask] - pos_w[cooldown_mask, 2]
                v_cmd_b[cooldown_mask, 2] = torch.clamp(z_err * cooldown_kp, -w_max, w_max)
        target_vel_w = quat_rotate(rot_w, v_cmd_b)
        self._last_target_vel_w = target_vel_w
        vel_b = self.drone.vel_b[:, 0, :3]
        ang_b = self.drone.vel_b[:, 0, 3:6]
        yaw = quaternion_to_euler(self.drone.rot[:, 0, :])[:, 2]
        yaw_goal = torch.atan2(
            (waypoint_w[:, 1] - pos_w[:, 1]),
            (waypoint_w[:, 0] - pos_w[:, 0]),
        )
        yaw_err = (yaw_goal - yaw + torch.pi) % (2 * torch.pi) - torch.pi
        yaw_blend_gain = float(self.cfg.task.get("yaw_blend_gain", 0.0)) if apply_post else 0.0
        yaw_blend_thresh = float(self.cfg.task.get("yaw_blend_thresh", 0.1)) if apply_post else -1.0
        if apply_post and yaw_blend_gain > 0.0 and yaw_blend_thresh >= 0.0:
            weak_mask = r_cmd.abs() < yaw_blend_thresh
            r_cmd = torch.where(weak_mask, r_cmd + yaw_blend_gain * yaw_err, r_cmd)
            r_cmd = r_cmd.clamp(-r_max, r_max)
        act_exec = torch.zeros(self.num_envs, 1, 4, device=self.device, dtype=act.dtype)
        act_exec[:, 0, 0] = v_cmd_b[:, 0] / max(u_max, 1e-6)
        act_exec[:, 0, 1] = v_cmd_b[:, 1] / max(v_max, 1e-6)
        act_exec[:, 0, 2] = v_cmd_b[:, 2] / max(w_max, 1e-6)
        act_exec[:, 0, 3] = r_cmd / max(r_max, 1e-6)
        if apply_post:
            self._last_action_executed = act_exec.clamp(-1.0, 1.0)
        else:
            self._last_action_executed = a_exec.detach().clone()
            diff = (self._last_action_executed - actions_policy_safe).abs().max()
            if float(diff.detach().cpu().item()) > 1e-6:
                raise RuntimeError(f"PPO mismatch: executed action differs by {float(diff.detach().cpu().item())}")
        self._last_yaw_err = yaw_err
        self._last_r_cmd = r_cmd

        # Axis-test open-loop mode: bypass velocity/yaw/RP feedback gains and map
        # [surge, sway, heave, yaw] actions directly to body wrench commands.
        axis_test_open_loop = bool(self.ctrl_cfg.get("axis_test_open_loop", False))
        F_b = torch.zeros(self.num_envs, 3, device=self.device)
        T_b = torch.zeros(self.num_envs, 3, device=self.device)
        if axis_test_open_loop:
            force_x = float(self.ctrl_cfg.get("axis_test_force_x", self.ctrl_cfg.get("axis_test_force", 12.0)))
            force_y = float(self.ctrl_cfg.get("axis_test_force_y", self.ctrl_cfg.get("axis_test_force", 12.0)))
            force_z = float(self.ctrl_cfg.get("axis_test_force_z", self.ctrl_cfg.get("axis_test_force", 12.0)))
            torque_yaw = float(self.ctrl_cfg.get("axis_test_torque_yaw", 0.05))
            F_b[:, 0] = act[:, 0] * force_x
            F_b[:, 1] = act[:, 1] * force_y
            F_b[:, 2] = act[:, 2] * force_z
            T_b[:, 2] = act[:, 3] * torque_yaw
        else:
            e_v = v_cmd_b - vel_b
            # Axis-wise velocity gains improve decoupling:
            # surge/sway/heave can have different dynamics and should not share a single kp.
            kp_u = float(self.ctrl_cfg.get("vel_kp_u", self.ctrl_cfg.get("vel_kp", 5.0)))
            kp_v = float(self.ctrl_cfg.get("vel_kp_v", self.ctrl_cfg.get("vel_kp", 5.0)))
            kp_w = float(self.ctrl_cfg.get("vel_kp_w", self.ctrl_cfg.get("vel_kp", 5.0)))
            kp_vec = torch.tensor([kp_u, kp_v, kp_w], device=self.device, dtype=e_v.dtype).view(1, 3)
            F_b = kp_vec * e_v
            gravity_scale = float(self.ctrl_cfg.get("gravity_comp_scale", 0.0))
            if gravity_scale != 0.0:
                masses = self.drone.masses[:, 0].squeeze(-1)
                F_b[:, 2] += masses * 9.81 * gravity_scale
            T_b[:, 2] = float(self.ctrl_cfg.get("yaw_rate_kp", 1.5)) * (r_cmd - ang_b[:, 2])
            if bool(self.ctrl_cfg.get("stabilize_rp", True)):
                rpy = quaternion_to_euler(self.drone.rot[:, 0, :])
                T_b[:, 0] = -float(self.ctrl_cfg.get("rp_kp", 2.0)) * rpy[:, 0] - float(self.ctrl_cfg.get("rp_kd", 0.2)) * ang_b[:, 0]
                T_b[:, 1] = -float(self.ctrl_cfg.get("rp_kp", 2.0)) * rpy[:, 1] - float(self.ctrl_cfg.get("rp_kd", 0.2)) * ang_b[:, 1]

        # Optional wrench clamps to avoid allocator saturation and improve high-level command fidelity.
        max_force_cfg = self.ctrl_cfg.get("max_force", None)
        if max_force_cfg is not None:
            try:
                mf = torch.as_tensor(max_force_cfg, device=self.device, dtype=F_b.dtype).view(-1)
                if mf.numel() == 1:
                    F_b = F_b.clamp(min=-float(mf.item()), max=float(mf.item()))
                elif mf.numel() == 3:
                    F_b = torch.maximum(torch.minimum(F_b, mf.view(1, 3)), -mf.view(1, 3))
            except Exception:
                pass
        max_torque_cfg = self.ctrl_cfg.get("max_torque", None)
        if max_torque_cfg is not None:
            try:
                mt = torch.as_tensor(max_torque_cfg, device=self.device, dtype=T_b.dtype).view(-1)
                if mt.numel() == 1:
                    T_b = T_b.clamp(min=-float(mt.item()), max=float(mt.item()))
                elif mt.numel() == 3:
                    T_b = torch.maximum(torch.minimum(T_b, mt.view(1, 3)), -mt.view(1, 3))
            except Exception:
                pass

        if self._A_pinv is None or (self._alloc_step % int(self.ctrl_cfg.get("allocation_update_every", 1)) == 0):
            self._build_allocation()
        self._alloc_step += 1

        wrench_full = torch.cat([F_b, T_b], dim=-1)
        alloc_mode_runtime = str(getattr(self, "_alloc_mode_runtime", "4dof"))
        alloc_rows = getattr(self, "_alloc_ctrl_rows", (0, 1, 2, 5))
        if alloc_mode_runtime == "4dof_balanced":
            w_force, w_rp, w_yaw = getattr(self, "_alloc_weights_runtime", (1.0, 3.0, 1.0))
            wrench = torch.zeros(self.num_envs, 1, 6, device=self.device, dtype=wrench_full.dtype)
            wrench[:, 0, 0] = F_b[:, 0] * w_force
            wrench[:, 0, 1] = F_b[:, 1] * w_force
            wrench[:, 0, 2] = F_b[:, 2] * w_force
            wrench[:, 0, 3] = 0.0
            wrench[:, 0, 4] = 0.0
            wrench[:, 0, 5] = T_b[:, 2] * w_yaw
        elif alloc_rows == (0, 1, 2, 5):
            wrench = torch.stack([F_b[:, 0], F_b[:, 1], F_b[:, 2], T_b[:, 2]], dim=-1).unsqueeze(1)
        else:
            wrench = wrench_full.unsqueeze(1)
        # Debug taps for axis testing / diagnosis.
        self._last_wrench_b = wrench_full
        if bool(self.ctrl_cfg.get("direct_force", False)):
            self.drone.base_link.apply_forces_and_torques_at_pos(
                F_b.reshape(-1, 3),
                T_b.reshape(-1, 3),
                is_global=False,
            )
            self.effort = torch.norm(F_b, dim=-1)
            self._last_thr_action = None
            self._last_thr_forces = None
        else:
            # Apply hard-coded mixer if enabled, otherwise use dynamic allocation
            if self._use_hardcoded_mixer and self._hardcoded_mixer is not None:
                # Hard-coded 6x4 mixer: (batch, 1, 4) -> (batch, 1, 6)
                # Maps 4-DOF wrench [Fx, Fy, Fz, Tz] directly to 6 thruster commands
                forces_mixed = self._hardcoded_mixer.mix(wrench.squeeze(1))  # (batch, 4) -> (batch, 6)
                forces = forces_mixed.unsqueeze(1)  # -> (batch, 1, 6)
            else:
                # Dynamic allocation: solve A @ u = wrench using pseudoinverse
                forces = wrench @ self._A_pinv.transpose(-1, -2)
            
            if bool(self.ctrl_cfg.get("axis_test_linear_thruster", False)):
                max_force = float(self.ctrl_cfg.get("axis_test_thruster_max_force", 20.0))
                thr_action = torch.clamp(forces / max(max_force, 1e-6), -1.0, 1.0)
            else:
                cmd_mode = str(self.ctrl_cfg.get("thruster_cmd_mode", self._thr_cmd_mode)).lower()
                if cmd_mode in {"model_inverse", "inverse", "lut"}:
                    thr_action = self._forces_to_thr_action(forces.squeeze(1)).unsqueeze(1)
                else:
                    f_max = float(self.drone.FORCE_CONSTANTS_0[0].item()) * (3900.0 ** 2)
                    thr_action = torch.clamp(forces / f_max, -1.0, 1.0)
            # Undo calibration-space mapping back to simulator rotor-channel order.
            sign_map = getattr(self, "_alloc_sign_runtime", None)
            inv_perm = getattr(self, "_alloc_inv_perm_runtime", None)
            if torch.is_tensor(sign_map) and sign_map.numel() == thr_action.shape[-1]:
                thr_action = thr_action * sign_map.view(1, 1, -1)
            if torch.is_tensor(inv_perm) and inv_perm.numel() == thr_action.shape[-1]:
                thr_action = thr_action.index_select(-1, inv_perm)
            # Diagnostic mode: command exactly one rotor channel and zero all others.
            if bool(self.ctrl_cfg.get("axis_test_single_thruster_enable", False)):
                idx = int(self.ctrl_cfg.get("axis_test_single_thruster_idx", -1))
                cmd = float(self.ctrl_cfg.get("axis_test_single_thruster_cmd", 0.0))
                cmd = max(-1.0, min(1.0, cmd))
                if 0 <= idx < thr_action.shape[-1]:
                    thr_single = torch.zeros_like(thr_action)
                    thr_single[..., idx] = cmd
                    thr_action = thr_single
            thr_action = torch.nan_to_num(thr_action, nan=0.0, posinf=0.0, neginf=0.0)
            # Save per-thruster commands (normalized) and raw forces for debugging.
            try:
                self._last_thr_action = thr_action.squeeze(1).detach()
                self._last_thr_forces = forces.squeeze(1).detach()
            except Exception:
                self._last_thr_action = None
                self._last_thr_forces = None
            bad_thr = ~torch.isfinite(thr_action).all(dim=-1)
            if bad_thr.any():
                thr_action[bad_thr] = 0.0
                if self.debug:
                    bad_envs = torch.where(bad_thr)[0].tolist()
                    print(f"[HoverTankPPO] zero thrusters for envs={bad_envs}")
            if self._reset_cooldown is not None:
                cooldown_mask = self._reset_cooldown > 0
                if cooldown_mask.any():
                    self._reset_cooldown[cooldown_mask] -= 1
            self.effort = torch.abs(self.drone.apply_action(thr_action))

    def _fallback_sonar_ranges(self, pos_w: torch.Tensor, rot_w: torch.Tensor) -> torch.Tensor | None:
        """Fallback sonar: ray distances to walls/obstacles using analytic geometry."""
        num_envs = pos_w.shape[0]
        num_rays = int(self.sonar.cfg.num_rays)
        if num_rays <= 0:
            return None

        max_range = float(self.sonar.cfg.max_range)
        angles = self.sonar.angles.to(pos_w.device)
        dir_body = torch.stack(
            [torch.cos(angles), torch.sin(angles), torch.zeros_like(angles)], dim=-1
        )
        dir_body = dir_body.unsqueeze(0).expand(num_envs, -1, -1)
        rot_expanded = rot_w.unsqueeze(1).expand(-1, num_rays, -1)
        dir_world = quat_rotate(rot_expanded.reshape(-1, 4), dir_body.reshape(-1, 3)).reshape(
            num_envs, num_rays, 3
        )
        dir_world = dir_world / (dir_world.norm(dim=-1, keepdim=True).clamp_min(1e-6))

        ranges = torch.full((num_envs, num_rays), max_range, device=pos_w.device)

        # Wall intersection (2D bounds).
        bounds_min = torch.as_tensor(self.planner_cfg.get("bounds_min", [0.0, 0.0, -2.0]), device=pos_w.device)
        bounds_max = torch.as_tensor(self.planner_cfg.get("bounds_max", [20.0, 6.0, 2.0]), device=pos_w.device)
        wall_margin = float(self.planner_cfg.get("wall_margin", 0.0))
        x_min = float(bounds_min[0] + wall_margin)
        x_max = float(bounds_max[0] - wall_margin)
        y_min = float(bounds_min[1] + wall_margin)
        y_max = float(bounds_max[1] - wall_margin)

        pos_local = pos_w - self.envs_positions.to(pos_w.device)
        x0 = pos_local[:, 0].unsqueeze(1)
        y0 = pos_local[:, 1].unsqueeze(1)
        dx = dir_world[:, :, 0]
        dy = dir_world[:, :, 1]

        # Avoid division by zero.
        eps = 1e-6
        dx_safe = torch.where(dx.abs() < eps, torch.full_like(dx, eps), dx)
        dy_safe = torch.where(dy.abs() < eps, torch.full_like(dy, eps), dy)

        t_xmin = (x_min - x0) / dx_safe
        t_xmax = (x_max - x0) / dx_safe
        t_ymin = (y_min - y0) / dy_safe
        t_ymax = (y_max - y0) / dy_safe

        def _valid_wall_t(t, x_hit, y_hit):
            return (t > 0.0) & (x_hit >= x_min) & (x_hit <= x_max) & (y_hit >= y_min) & (y_hit <= y_max)

        y_hit_xmin = y0 + t_xmin * dy
        y_hit_xmax = y0 + t_xmax * dy
        x_hit_ymin = x0 + t_ymin * dx
        x_hit_ymax = x0 + t_ymax * dx

        wall_ts = []
        wall_ts.append(torch.where(_valid_wall_t(t_xmin, x_min, y_hit_xmin), t_xmin, torch.full_like(t_xmin, max_range)))
        wall_ts.append(torch.where(_valid_wall_t(t_xmax, x_max, y_hit_xmax), t_xmax, torch.full_like(t_xmax, max_range)))
        wall_ts.append(torch.where(_valid_wall_t(t_ymin, x_hit_ymin, y_min), t_ymin, torch.full_like(t_ymin, max_range)))
        wall_ts.append(torch.where(_valid_wall_t(t_ymax, x_hit_ymax, y_max), t_ymax, torch.full_like(t_ymax, max_range)))
        wall_t = torch.minimum(torch.minimum(wall_ts[0], wall_ts[1]), torch.minimum(wall_ts[2], wall_ts[3]))
        ranges = torch.minimum(ranges, wall_t.clamp(min=0.0, max=max_range))

        # Obstacle intersection (spheres/capsules).
        if self.enable_obstacles and self.obstacles is not None:
            obst_pos, _ = self.obstacles.get_world_poses()
            obst_pos = obst_pos.to(pos_w.device)
            if obst_pos.ndim == 2:
                obst_pos = obst_pos.view(self.num_envs, self.num_obstacles, 3)
            radius = float(self.obst_cfg.get("radius", 0.15))
            height = float(self.obst_cfg.get("height", 0.30))
            z_tol = height * 0.5 + radius

            O = pos_w[:, None, None, :]  # [E,1,1,3]
            D = dir_world[:, :, None, :]  # [E,R,1,3]
            C = obst_pos[:, None, :, :]   # [E,1,O,3]

            L = C - O
            t_ca = (L * D).sum(dim=-1)
            d2 = (L * L).sum(dim=-1) - t_ca**2
            z_ok = (L[..., 2].abs() <= z_tol)
            r2 = radius ** 2
            hit = (t_ca > 0.0) & (d2 <= r2) & z_ok
            thc = torch.sqrt(torch.clamp(r2 - d2, min=0.0))
            t_hit = torch.where(hit, t_ca - thc, torch.full_like(t_ca, max_range))
            t_min = t_hit.min(dim=-1).values  # [E,R]
            ranges = torch.minimum(ranges, t_min.clamp(min=0.0, max=max_range))

        # Maze walls (axis-aligned rectangles per yaw 0/90 deg).
        if getattr(self, "enable_maze", False) and getattr(self, "num_maze_walls", 0) > 0:
            pos_local = pos_w - self.envs_positions
            centers = self.maze_centers
            yaw = self.maze_yaw
            is_vertical = torch.abs(torch.sin(yaw)) > torch.abs(torch.cos(yaw))
            half_x = torch.where(
                is_vertical,
                torch.full_like(centers[..., 0], self.maze_thickness * 0.5),
                torch.full_like(centers[..., 0], self.maze_length * 0.5),
            )
            half_y = torch.where(
                is_vertical,
                torch.full_like(centers[..., 1], self.maze_length * 0.5),
                torch.full_like(centers[..., 1], self.maze_thickness * 0.5),
            )
            x_min_w = (centers[..., 0] - half_x).unsqueeze(1)
            x_max_w = (centers[..., 0] + half_x).unsqueeze(1)
            y_min_w = (centers[..., 1] - half_y).unsqueeze(1)
            y_max_w = (centers[..., 1] + half_y).unsqueeze(1)
            z_min = (centers[..., 2] - self.maze_height * 0.5).unsqueeze(1)
            z_max = (centers[..., 2] + self.maze_height * 0.5).unsqueeze(1)

            ox = pos_local[:, 0].unsqueeze(1).unsqueeze(-1)
            oy = pos_local[:, 1].unsqueeze(1).unsqueeze(-1)
            oz = pos_local[:, 2].unsqueeze(1).unsqueeze(-1)
            dx = dir_world[:, :, 0].unsqueeze(-1)
            dy = dir_world[:, :, 1].unsqueeze(-1)
            eps = 1e-6
            dx_safe = torch.where(dx.abs() < eps, torch.full_like(dx, eps), dx)
            dy_safe = torch.where(dy.abs() < eps, torch.full_like(dy, eps), dy)

            tx1 = (x_min_w - ox) / dx_safe
            tx2 = (x_max_w - ox) / dx_safe
            tmin_x = torch.minimum(tx1, tx2)
            tmax_x = torch.maximum(tx1, tx2)

            ty1 = (y_min_w - oy) / dy_safe
            ty2 = (y_max_w - oy) / dy_safe
            tmin_y = torch.minimum(ty1, ty2)
            tmax_y = torch.maximum(ty1, ty2)

            tmin = torch.maximum(tmin_x, tmin_y)
            tmax = torch.minimum(tmax_x, tmax_y)
            z_ok = (oz >= z_min) & (oz <= z_max)
            hit = (tmax >= tmin) & (tmax > 0.0) & z_ok
            t_hit = torch.where(hit, tmin, torch.full_like(tmin, max_range))
            t_min = t_hit.min(dim=-1).values  # [E,R]
            ranges = torch.minimum(ranges, t_min.clamp(min=0.0, max=max_range))

        return ranges

    def _wall_clearance(self, pos_w: torch.Tensor) -> torch.Tensor:
        """Compute min distance to world bounds (x/y walls)."""
        bounds_min = torch.as_tensor(self.planner_cfg.get("bounds_min", [0.0, 0.0, -2.0]), device=pos_w.device)
        bounds_max = torch.as_tensor(self.planner_cfg.get("bounds_max", [20.0, 6.0, 2.0]), device=pos_w.device)
        wall_margin = float(self.planner_cfg.get("wall_margin", 0.0))
        x_min = bounds_min[0] + wall_margin
        x_max = bounds_max[0] - wall_margin
        y_min = bounds_min[1] + wall_margin
        y_max = bounds_max[1] - wall_margin
        pos_local = pos_w - self.envs_positions.to(pos_w.device)
        dx = torch.minimum(pos_local[:, 0] - x_min, x_max - pos_local[:, 0])
        dy = torch.minimum(pos_local[:, 1] - y_min, y_max - pos_local[:, 1])
        return torch.minimum(dx, dy).clamp(min=0.0)

    def _maze_clearance(self, pos_w: torch.Tensor) -> torch.Tensor | None:
        """Compute min planar distance to maze walls (approx axis-aligned)."""
        if not getattr(self, "enable_maze", False) or getattr(self, "num_maze_walls", 0) == 0:
            return None
        pos_local = pos_w - self.envs_positions
        centers = self.maze_centers
        yaw = self.maze_yaw
        is_vertical = torch.abs(torch.sin(yaw)) > torch.abs(torch.cos(yaw))
        half_x = torch.where(
            is_vertical,
            torch.full_like(centers[..., 0], self.maze_thickness * 0.5),
            torch.full_like(centers[..., 0], self.maze_length * 0.5),
        )
        half_y = torch.where(
            is_vertical,
            torch.full_like(centers[..., 1], self.maze_length * 0.5),
            torch.full_like(centers[..., 1], self.maze_thickness * 0.5),
        )
        dx = torch.abs(pos_local[:, None, 0] - centers[..., 0]) - half_x
        dy = torch.abs(pos_local[:, None, 1] - centers[..., 1]) - half_y
        dx = dx.clamp_min(0.0)
        dy = dy.clamp_min(0.0)
        dist = torch.sqrt(dx * dx + dy * dy)
        return dist.min(dim=1).values

    def _obstacle_clearance(self, pos_w: torch.Tensor) -> torch.Tensor | None:
        """Compute min distance to dynamic obstacles (center distance minus radius)."""
        if not getattr(self, "enable_obstacles", False) or getattr(self, "num_obstacles", 0) == 0:
            return None
        radius = float(self.obst_cfg.get("radius", 0.15))
        height = float(self.obst_cfg.get("height", 0.30))
        z_tol = height * 0.5 + radius

        # Prefer current obstacle poses (they move over time); fall back to the spawn centers.
        if getattr(self, "obstacles", None) is not None:
            obst_pos, _ = self.obstacles.get_world_poses()
            obst_pos = obst_pos.to(pos_w.device)
            if obst_pos.ndim == 2:
                obst_pos = obst_pos.view(self.num_envs, self.num_obstacles, 3)
        else:
            obst_pos = self.obst_center + self.envs_positions.unsqueeze(1)

        dz = (pos_w[:, None, 2] - obst_pos[:, :, 2]).abs()
        z_ok = dz <= z_tol
        center_dist = torch.norm(pos_w[:, None, :] - obst_pos, dim=-1)
        d = torch.where(z_ok, center_dist - radius, torch.full_like(center_dist, 1e6))
        return d.clamp_min(0.0).min(dim=1).values

    def _compute_state_and_obs(self):
        self.drone_state = self.drone.get_state(env_frame=False)
        pos_w = self.drone.pos[:, 0, :]
        rot_w = self.drone.rot[:, 0, :]
        bad_state = ~torch.isfinite(pos_w).all(dim=-1) | ~torch.isfinite(rot_w).all(dim=-1)
        if bad_state.any():
            bad_envs = torch.where(bad_state)[0]
            if self.debug:
                print(f"[HoverTankPPO] reset non-finite envs={bad_envs.tolist()}")
            self._reset_idx(bad_envs)
            self.drone_state = self.drone.get_state(env_frame=False)
            pos_w = self.drone.pos[:, 0, :]
            rot_w = self.drone.rot[:, 0, :]
        vel_b = self.drone.vel_b[:, 0, :3]
        ang_b = self.drone.vel_b[:, 0, 3:6]

        # Refresh target heading based on current position -> goal direction.
        rel_goal = self.target_pos[:, 0, :] - pos_w
        rel_norm = torch.norm(rel_goal, dim=-1, keepdim=True).clamp_min(1e-6)
        self.target_heading = (rel_goal / rel_norm).unsqueeze(1)

        waypoint_w = self._current_waypoints()
        rel_w = waypoint_w - pos_w
        rel_b = quat_rotate_inverse(rot_w, rel_w)
        self.rpos = self.target_pos - self.drone_state[..., :3]
        self.rheading = self.target_heading - self.drone_state[..., 13:16]

        yaw_goal = torch.atan2(rel_w[:, 1], rel_w[:, 0])
        yaw = quaternion_to_euler(rot_w)[:, 2]
        yaw_err = (yaw_goal - yaw + torch.pi) % (2 * torch.pi) - torch.pi
        yaw_sin = torch.sin(yaw_err).unsqueeze(-1)
        yaw_cos = torch.cos(yaw_err).unsqueeze(-1)

        rpy = quaternion_to_euler(rot_w)
        roll_pitch = rpy[:, :2]
        depth = pos_w[:, 2:3]

        raw_sonar = self.sonar.scan(pos_w, rot_w)["ranges"]
        sonar_obs = raw_sonar
        fallback = self._fallback_sonar_ranges(pos_w, rot_w)
        self._last_sonar_fallback = False
        if fallback is not None:
            sonar_obs = torch.minimum(sonar_obs, fallback)
            self._last_sonar_fallback = True
        expected_rays = int(self.sonar_cfg.get("num_rays", 32))
        if sonar_obs.dim() != 2 or sonar_obs.shape[1] != expected_rays:
            sonar_obs = torch.full(
                (pos_w.shape[0], expected_rays),
                float(self.sonar.cfg.max_range),
                device=self.device,
            )
        self._last_sonar = sonar_obs
        self._last_depth_min = None
        self._last_depth_obs = None
        depth_img_for_occ = None
        self._last_wall_dist = self._wall_clearance(pos_w)
        self._last_maze_dist = self._maze_clearance(pos_w)
        self._last_obst_dist = self._obstacle_clearance(pos_w)
        depth_obs = None
        if self.depth_enabled and self.depth_camera is not None and not self._depth_disabled:
            try:
                imgs = self.depth_camera.get_images()
            except Exception as exc:
                if self.debug:
                    print(f"[HoverTankPPO] depth camera read failed: {exc}")
                self._depth_fail_count += 1
                if self._depth_fail_count >= self._depth_fail_limit:
                    self._depth_disabled = True
                    if self.debug:
                        print("[HoverTankPPO] depth camera disabled after repeated failures")
                imgs = None
            data_type = str(self.depth_cfg.get("data_type", "distance_to_camera"))
            depth_img = None
            if imgs is not None and data_type in imgs.keys(True, True):
                depth_img = imgs[data_type]
                depth_img = depth_img.squeeze(1)
                if depth_img.dim() == 2:
                    depth_img = depth_img.unsqueeze(0)
                elif depth_img.dim() != 3:
                    # Invalid frame, skip.
                    depth_img = None
                if depth_img is not None and depth_img.shape[0] != pos_w.shape[0]:
                    if depth_img.shape[0] == 1:
                        depth_img = depth_img.expand(pos_w.shape[0], -1, -1)
                    else:
                        depth_img = depth_img[: pos_w.shape[0]]
                if depth_img is not None:
                    if not torch.isfinite(depth_img).all():
                        depth_img = None
                if depth_img is not None:
                    max_range = float(self.depth_cfg.get("max_range", 10.0))
                    min_filter = float(self.depth_cfg.get("min_range_filter", 0.2))
                    depth_img = depth_img.clamp(min=0.0, max=max_range)
                    depth_img_for_occ = depth_img
                    if float(depth_img.max() - depth_img.min()) < 1e-3:
                        alt_type = "depth" if data_type != "depth" else "distance_to_camera"
                        if imgs is not None and alt_type in imgs.keys(True, True):
                            alt_img = imgs[alt_type].squeeze(1)
                            if alt_img.dim() == 2:
                                alt_img = alt_img.unsqueeze(0)
                            if alt_img.dim() == 3:
                                depth_img = alt_img.clamp(min=0.0, max=max_range)
                    if depth_img is None:
                        if self.debug:
                            print("[HoverTankPPO] depth camera read failed: empty frame")
                        imgs = None
                    valid = depth_img > min_filter
                    depth_min = torch.where(
                        valid.any(dim=(1, 2)),
                        torch.where(valid, depth_img, torch.full_like(depth_img, max_range)).amin(dim=(1, 2)),
                        torch.full((depth_img.shape[0],), max_range, device=depth_img.device),
                    )
                    self._last_depth_min = depth_min
                    if self.depth_obs_enabled:
                        down = self.depth_obs_cfg.get("downsample", [16, 9])
                        dh, dw = int(down[1]), int(down[0])
                        # Simple stride downsample to keep it cheap.
                        stride_h = max(1, depth_img.shape[1] // dh)
                        stride_w = max(1, depth_img.shape[2] // dw)
                        depth_obs = depth_img[:, ::stride_h, ::stride_w]
                        depth_obs = depth_obs[:, :dh, :dw]
                        depth_obs = depth_obs.reshape(depth_obs.shape[0], -1)
                        self._last_depth_obs = depth_obs
                    if self._depth_viz_every > 0:
                        self._depth_viz_count += 1
                        if self._depth_viz_count % self._depth_viz_every == 0 and self._depth_log_wandb:
                            import matplotlib.pyplot as plt
                            import wandb
                            depth_np = depth_img[0].detach().cpu().numpy()
                            denom = max(float(depth_np.max() - depth_np.min()), 1e-6)
                            norm = (depth_np - depth_np.min()) / denom
                            cmap = np.uint8(plt.get_cmap("jet")(norm)[:, :, :3] * 255)
                            self._wandb_log(
                                {
                                    "viz/depth": wandb.Image(
                                        cmap,
                                        caption=f"depth_step_{self._depth_viz_count:06d}",
                                    )
                                }
                            )

        if self.debug and self.num_envs > 0:
            step = int(self.progress_buf[0].item())
            if step % max(self.debug_interval, 1) == 0:
                dist = torch.norm(rel_w[0]).item()
                wp = self._current_waypoints()[0].detach().cpu().numpy()
                wp_dist = float(torch.norm(self._current_waypoints()[0] - pos_w[0]).item())
                path_len = len(self._paths[0]) if self._paths[0] is not None else 0
                path_idx = int(self._path_idx[0].item())
                action_dbg = (
                    [0.0, 0.0, 0.0, 0.0]
                    if not hasattr(self, "_last_action_clamped") or self._last_action_clamped is None
                    else self._last_action_clamped[0].detach().cpu().numpy().tolist()
                )
                action_raw_dbg = (
                    [0.0, 0.0, 0.0, 0.0]
                    if self._prev_action is None
                    else self._prev_action[0, 0].detach().cpu().numpy().tolist()
                )
                v_cmd_b_dbg = (
                    [0.0, 0.0, 0.0]
                    if not hasattr(self, "_last_v_cmd_b") or self._last_v_cmd_b is None
                    else self._last_v_cmd_b[0].detach().cpu().numpy().tolist()
                )
                vel_b_dbg = vel_b[0].detach().cpu().numpy().tolist()
                tgt_vel = (
                    [0.0, 0.0, 0.0]
                    if self._last_target_vel_w is None
                    else self._last_target_vel_w[0].detach().cpu().numpy().tolist()
                )
                yaw_err_val = 0.0 if not hasattr(self, "_last_yaw_err") else float(self._last_yaw_err[0].item())
                r_cmd_val = 0.0 if not hasattr(self, "_last_r_cmd") else float(self._last_r_cmd[0].item())
                min_range = None
                depth_min_dbg = None
                wall_min_dbg = None
                maze_min_dbg = None
                obst_min_dbg = None
                safety_min_dbg = None
                front_min_dbg = None
                zone = "clear"
                if self._last_sonar is not None:
                    min_range = float(torch.min(self._last_sonar[0]).item())
                    if self._last_depth_min is not None:
                        depth_min_dbg = float(self._last_depth_min[0].item())
                        min_range = min(min_range, depth_min_dbg)
                    if self._last_wall_dist is not None:
                        wall_min_dbg = float(self._last_wall_dist[0].item())
                        min_range = min(min_range, wall_min_dbg)
                        safety_min_dbg = wall_min_dbg
                    if self._last_maze_dist is not None:
                        maze_min_dbg = float(self._last_maze_dist[0].item())
                        min_range = min(min_range, maze_min_dbg)
                        safety_min_dbg = maze_min_dbg if safety_min_dbg is None else min(safety_min_dbg, maze_min_dbg)
                    if self._last_obst_dist is not None:
                        obst_min_dbg = float(self._last_obst_dist[0].item())
                        min_range = min(min_range, obst_min_dbg)
                        safety_min_dbg = obst_min_dbg if safety_min_dbg is None else min(safety_min_dbg, obst_min_dbg)
                    angles = self.sonar.angles.to(self.device)
                    front_deg = float(self.policy_cfg.get("front_safety_deg", 90.0))
                    front_mask = angles.abs() <= (front_deg * torch.pi / 180.0) * 0.5
                    front_min_dbg = float(torch.min(self._last_sonar[0, front_mask]).item())
                    stop_dist = float(self.policy_cfg.get("min_range_stop", 0.5))
                    avoid_dist = float(self.policy_cfg.get("min_range_avoid", 1.2))
                    replan_dist = float(self.policy_cfg.get("min_range_replan", 0.8))
                    if min_range < stop_dist:
                        zone = "stop"
                    elif min_range < avoid_dist:
                        zone = "avoid"
                    elif min_range < replan_dist:
                        zone = "replan"
                print(
                    "[HoverTankPPO]",
                    f"step={step}",
                    f"pos={pos_w[0].detach().cpu().numpy()}",
                    f"goal={self.target_pos[0, 0, :].detach().cpu().numpy()}",
                    f"dist={dist:.3f}",
                    f"wp={wp}",
                    f"wp_dist={wp_dist:.3f}",
                    f"wp_tol={self._waypoint_tol:.3f}",
                    f"path_idx={path_idx}/{path_len}",
                    f"action_raw={action_raw_dbg}",
                    f"action={action_dbg}",
                    f"min_range={min_range:.3f}" if min_range is not None else "min_range=na",
                    f"depth_min={depth_min_dbg:.3f}" if depth_min_dbg is not None else "depth_min=na",
                    f"wall_min={wall_min_dbg:.3f}" if wall_min_dbg is not None else "wall_min=na",
                    f"maze_min={maze_min_dbg:.3f}" if maze_min_dbg is not None else "maze_min=na",
                    f"obst_min={obst_min_dbg:.3f}" if obst_min_dbg is not None else "obst_min=na",
                    f"safety_min={safety_min_dbg:.3f}" if safety_min_dbg is not None else "safety_min=na",
                    f"front_min={front_min_dbg:.3f}" if front_min_dbg is not None else "front_min=na",
                    f"obst_ids={list(range(self.num_obstacles)) if getattr(self, 'enable_obstacles', False) else []}",
                    f"zone={zone}",
                    f"yaw_err={yaw_err_val:.3f}",
                    f"r_cmd={r_cmd_val:.3f}",
                    f"v_cmd_b={v_cmd_b_dbg}",
                    f"vel_b={vel_b_dbg}",
                    f"target_vel_w={tgt_vel}",
                )

        parts = [
            rel_b,
            yaw_sin,
            yaw_cos,
            vel_b,
            ang_b[:, 2:3],
            depth,
            roll_pitch,
            sonar_obs,
        ]
        if self.depth_obs_enabled:
            if depth_obs is None:
                down = self.depth_obs_cfg.get("downsample", [16, 9])
                depth_obs = torch.zeros((pos_w.shape[0], int(down[0]) * int(down[1])), device=self.device)
            parts.append(depth_obs)
        if self.occ_enabled:
            grid_size = int(self.occ_cfg.get("grid_size", 16))
            grid_range = float(self.occ_cfg.get("grid_range", 5.0))
            occ = torch.zeros((pos_w.shape[0], grid_size, grid_size), device=self.device)
            angles = self.sonar.angles.to(self.device)
            ranges = sonar_obs.clamp(max=grid_range)
            hit = ranges < grid_range
            cos_a = torch.cos(angles)
            sin_a = torch.sin(angles)
            x = ranges * cos_a
            y = ranges * sin_a
            ix = ((x + grid_range) / (2 * grid_range) * (grid_size - 1)).long().clamp(0, grid_size - 1)
            iy = ((y + grid_range) / (2 * grid_range) * (grid_size - 1)).long().clamp(0, grid_size - 1)
            for e in range(pos_w.shape[0]):
                if hit[e].any():
                    occ[e, iy[e, hit[e]], ix[e, hit[e]]] = 1.0
            if depth_img_for_occ is not None and self._depth_fuse_occ:
                fov_deg = float(self.depth_cfg.get("fov_deg", 90.0))
                min_filter = float(self.depth_cfg.get("min_range_filter", 0.0))
                width = depth_img_for_occ.shape[2]
                angles_d = torch.linspace(
                    -0.5 * fov_deg, 0.5 * fov_deg, width, device=self.device
                ) * (torch.pi / 180.0)
                depth_img = depth_img_for_occ.clamp(min=0.0, max=grid_range)
                if min_filter > 0.0:
                    depth_img = torch.where(
                        depth_img > min_filter,
                        depth_img,
                        torch.full_like(depth_img, grid_range),
                    )
                depth_cols = depth_img.median(dim=1).values
                hit_d = depth_cols < grid_range
                cos_d = torch.cos(angles_d)
                sin_d = torch.sin(angles_d)
                x_d = depth_cols * cos_d
                y_d = depth_cols * sin_d
                ix_d = ((x_d + grid_range) / (2 * grid_range) * (grid_size - 1)).long().clamp(0, grid_size - 1)
                iy_d = ((y_d + grid_range) / (2 * grid_range) * (grid_size - 1)).long().clamp(0, grid_size - 1)
                for e in range(pos_w.shape[0]):
                    if hit_d[e].any():
                        occ[e, iy_d[e, hit_d[e]], ix_d[e, hit_d[e]]] = 1.0
            self._last_occ_grid = occ
            parts.append(occ.view(pos_w.shape[0], -1))

        # Persistent mapping (single, accumulated occupancy map per env) + optional figure saving.
        if self.mapping_enabled:
            self._map_step += 1
            self._update_persistent_map(pos_w, rot_w, sonar_obs, depth_img_for_occ)
            self._save_persistent_map_fig()
            if self.mapping_use_in_obs and self._map_occ is not None:
                parts.append(self._map_occ.view(pos_w.shape[0], -1))
        if self.mapping3d_enabled:
            self._vox_step += 1
            self._update_voxel_map(pos_w, rot_w, depth_img_for_occ)
            self._save_voxel_map_fig()
        fixed = []
        for part in parts:
            if part.dim() == 1:
                part = part.unsqueeze(-1)
            if part.dim() > 2:
                part = part.reshape(part.shape[0], -1)
            fixed.append(part)
        obs = torch.cat(fixed, dim=-1)
        expected_dim = int(self.observation_spec["agents", "observation"].shape[-1])
        if obs.shape[-1] != expected_dim:
            if obs.shape[-1] < expected_dim:
                pad = torch.zeros((obs.shape[0], expected_dim - obs.shape[-1]), device=obs.device)
                obs = torch.cat([obs, pad], dim=-1)
            else:
                obs = obs[:, :expected_dim]
        obs = obs.unsqueeze(1)

        return TensorDict(
            {
                "agents": {
                    "observation": obs,
                    "intrinsics": self.drone.intrinsics,
                },
                "stats": self.stats.clone(),
            },
            self.batch_size,
        )

    def _compute_reward_and_done(self):
        td = super()._compute_reward_and_done()
        base_reward = td["agents", "reward"]
        reward = base_reward
        if self.reward_alive != 0.0:
            reward = reward + self.reward_alive
        pos_w = self.drone.pos[:, 0, :]
        dist_goal = torch.norm(self.target_pos[:, 0, :] - pos_w, dim=-1).view(-1, 1)
        dist_goal = torch.nan_to_num(dist_goal, nan=1.0e6, posinf=1.0e6, neginf=0.0)
        waypoint_w = self._current_waypoints()
        if self.use_waypoint_reward:
            dist = torch.norm(waypoint_w - pos_w, dim=-1).view(-1, 1)
        else:
            # Prefer final-goal progress for learning; keep waypoint only for path-shaping terms.
            dist = dist_goal
        dist = torch.nan_to_num(dist, nan=1.0e6, posinf=1.0e6, neginf=0.0)

        # Dense path-following shaping: encourage motion aligned with the next waypoint direction.
        # This helps PPO learn to track the global RRT path while still learning obstacle avoidance.
        path_align_w = float(self.cfg.task.get("reward_path_align_weight", 0.0))
        path_speed_w = float(self.cfg.task.get("reward_path_speed_weight", 0.0))
        path_lat_w = float(self.cfg.task.get("penalty_path_lateral_weight", 0.0))
        if (path_align_w > 0.0 or path_speed_w > 0.0 or path_lat_w > 0.0):
            vel_w = self.drone.vel_w[:, 0, :3]
            dir_w = (waypoint_w - pos_w)
            dir_n = torch.norm(dir_w, dim=-1, keepdim=True).clamp_min(1e-6)
            dir_u = dir_w / dir_n
            speed = torch.norm(vel_w, dim=-1, keepdim=True).clamp_min(1e-6)
            cos = (vel_w * dir_u).sum(dim=-1, keepdim=True) / speed
            if path_speed_w > 0.0:
                v_along = (vel_w * dir_u).sum(dim=-1, keepdim=True).clamp_min(0.0)
                reward = reward + path_speed_w * v_along.view(-1, 1, 1)
            if path_align_w > 0.0:
                reward = reward + path_align_w * cos.view(-1, 1, 1)
            if path_lat_w > 0.0:
                v_lat = vel_w - dir_u * (vel_w * dir_u).sum(dim=-1, keepdim=True)
                lat_mag = torch.norm(v_lat, dim=-1, keepdim=True)
                reward = reward - path_lat_w * lat_mag.view(-1, 1, 1)

        progress = (self._prev_goal_dist - dist).clamp_min(-5.0)
        self._prev_goal_dist = dist.detach()
        reward = reward + self.reward_goal_progress * progress.view(-1, 1, 1)
        # Always provide a direct final-goal progress signal, even when waypoint progress is used.
        final_progress = (self._prev_final_goal_dist - dist_goal).clamp_min(-5.0)
        self._prev_final_goal_dist = dist_goal.detach()
        if self.reward_final_goal_progress > 0.0:
            reward = reward + self.reward_final_goal_progress * final_progress.view(-1, 1, 1)
        waypoint_bonus = float(self.cfg.task.get("reward_waypoint_bonus", 0.0))
        if waypoint_bonus > 0.0 and hasattr(self, "_last_waypoint_reached"):
            reward = reward + waypoint_bonus * self._last_waypoint_reached.view(-1, 1, 1).float()
        yaw_align_weight = float(self.cfg.task.get("yaw_align_reward_weight", 0.0))
        if yaw_align_weight > 0.0:
            pos_w = self.drone.pos[:, 0, :]
            yaw = quaternion_to_euler(self.drone.rot[:, 0, :])[:, 2]
            yaw_goal = torch.atan2(
                (self.target_pos[:, 0, 1] - pos_w[:, 1]),
                (self.target_pos[:, 0, 0] - pos_w[:, 0]),
            )
            yaw_err = (yaw_goal - yaw + torch.pi) % (2 * torch.pi) - torch.pi
            yaw_align = 1.0 - (yaw_err.abs() / torch.pi)
            reward = reward + yaw_align_weight * yaw_align.view(-1, 1, 1)

        if self._last_sonar is not None:
            min_range = torch.min(self._last_sonar, dim=1)[0]
            if self._last_depth_min is not None:
                min_range = torch.minimum(min_range, self._last_depth_min)
            if self._last_wall_dist is not None:
                min_range = torch.minimum(min_range, self._last_wall_dist)
            if self._last_maze_dist is not None:
                min_range = torch.minimum(min_range, self._last_maze_dist)
            if self._last_obst_dist is not None:
                min_range = torch.minimum(min_range, self._last_obst_dist)
            proximity = (self.obstacle_threshold - min_range).clamp_min(0.0) / max(self.obstacle_threshold, 1e-6)
            reward = reward - self.obstacle_penalty * proximity.view(-1, 1, 1)
            avoid_dist = float(self.policy_cfg.get("min_range_avoid", 1.2))
            safe_mask = (min_range >= avoid_dist).view(-1, 1, 1)
            forward_speed = self.drone.vel_b[:, 0, 0].clamp_min(0.0).view(-1, 1, 1)
            reward_forward = float(self.cfg.task.get("reward_forward_weight", 0.0))
            if reward_forward > 0.0:
                reward = reward + reward_forward * forward_speed * safe_mask

        if self.termination_max_z is not None:
            pos_w = self.drone.pos[:, 0, :]
            surface_violation = (pos_w[:, 2] > self.termination_max_z).view(-1, 1, 1)
            if self.surface_penalty > 0.0:
                reward = reward - self.surface_penalty * surface_violation.float()
        tilt_penalty = float(self.cfg.task.get("tilt_penalty_weight", 0.0))
        if tilt_penalty > 0.0:
            rpy = quaternion_to_euler(self.drone.rot[:, 0, :])
            tilt = rpy[:, :2].abs().sum(dim=-1).view(-1, 1, 1)
            reward = reward - tilt_penalty * tilt
        sat_penalty = float(self.cfg.task.get("action_saturation_penalty", 0.0))
        if sat_penalty > 0.0 and hasattr(self, "_last_action_clamped"):
            raw = (
                self._prev_action.squeeze(1)
                if self._prev_action is not None
                else torch.zeros_like(self._last_action_clamped)
            )
            sat = (raw.abs() - 1.0).clamp_min(0.0).mean(dim=-1).view(-1, 1, 1)
            reward = reward - sat_penalty * sat
        # Keep reward numerically stable for all algorithms (especially SAC/A2C).
        reward = torch.nan_to_num(reward, nan=-1.0, posinf=50.0, neginf=-50.0)
        if self.reward_clip > 0.0:
            reward = reward.clamp(-self.reward_clip, self.reward_clip)
        if self.reward_scale != 1.0:
            reward = reward * self.reward_scale

        # Success is defined by reaching the final goal, not the intermediate waypoint.
        success = dist_goal < self.success_distance
        success_mask = success.unsqueeze(-1)
        if success.any():
            reward = reward + success_mask * self.success_bonus
        collision = torch.zeros_like(success)
        if self.terminate_on_collision and self._last_sonar is not None:
            min_range = torch.min(self._last_sonar, dim=1)[0]
            if self._last_depth_min is not None:
                min_range = torch.minimum(min_range, self._last_depth_min)
            if self._last_wall_dist is not None:
                min_range = torch.minimum(min_range, self._last_wall_dist)
            if self._last_maze_dist is not None:
                min_range = torch.minimum(min_range, self._last_maze_dist)
            if self._last_obst_dist is not None:
                min_range = torch.minimum(min_range, self._last_obst_dist)
            collision = (min_range <= self.collision_distance).unsqueeze(-1)
            if collision.any():
                reward = reward - collision.unsqueeze(-1) * self.collision_penalty

        # Hard collision termination using PhysX contact forces (covers tank walls + floor).
        contact_cfg = self.cfg.task.get("contact_termination", {}) or {}
        if (
            self.terminate_on_collision
            and bool(contact_cfg.get("enable", False))
            and bool(contact_cfg.get("track_contact_forces", False))
        ):
            try:
                cf = self.drone.base_link.get_net_contact_forces(clone=False)
                # cf shape: [E, 1, 3] for base_link; be robust to other shapes.
                if torch.is_tensor(cf):
                    cf_mag = cf.reshape(cf.shape[0], -1).norm(dim=-1)
                    thresh = float(contact_cfg.get("force_thresh", 1.0))
                    contact_hit = (cf_mag > thresh).view(-1, 1)
                    if contact_hit.any():
                        collision = collision | contact_hit
                        reward = reward - contact_hit.unsqueeze(-1) * self.collision_penalty
            except Exception:
                pass
        nonfinite_state = torch.zeros_like(success)
        if self.terminate_on_nonfinite:
            try:
                nonfinite_state = (
                    ~torch.isfinite(self.drone.pos[:, 0, :]).all(dim=-1)
                    | ~torch.isfinite(self.drone.rot[:, 0, :]).all(dim=-1)
                    | ~torch.isfinite(self.drone.vel_w[:, 0, :]).all(dim=-1)
                ).unsqueeze(-1)
            except Exception:
                nonfinite_state = torch.zeros_like(success)
        terminated = td["terminated"] | success | collision | nonfinite_state
        if self.termination_max_z is not None:
            pos_w = self.drone.pos[:, 0, :]
            terminated = terminated | (pos_w[:, 2] > self.termination_max_z).unsqueeze(-1)
        truncated = td["truncated"]
        done = terminated | truncated
        # Lightweight done-reason telemetry for training diagnostics.
        try:
            self._last_done_reason_rates = {
                "success": float(success.float().mean().item()),
                "collision": float(collision.float().mean().item()),
                "surface": float(
                    (pos_w[:, 2] > self.termination_max_z).float().mean().item()
                ) if self.termination_max_z is not None else 0.0,
                "nonfinite": float(nonfinite_state.float().mean().item()),
                "truncated": float(truncated.float().mean().item()),
            }
        except Exception:
            pass
        # Ensure stats.return reflects the *final* reward (base reward + PPO shaping).
        # The base Hover env already added `base_reward` to stats.return; we add the delta here.
        try:
            delta = (reward - base_reward).squeeze(-1)
            self.stats["return"] += delta
        except Exception:
            pass
        base_shape = td["terminated"].shape
        if terminated.numel() != td["terminated"].numel():
            terminated = terminated.reshape(-1)[: td["terminated"].numel()].reshape(base_shape)
        else:
            terminated = terminated.reshape(base_shape)
        if done.numel() != td["done"].numel():
            done = done.reshape(-1)[: td["done"].numel()].reshape(td["done"].shape)
        else:
            done = done.reshape(td["done"].shape)
        td["agents", "reward"] = reward
        td["terminated"] = terminated
        td["done"] = done
        if self.debug:
            done_mask = td["done"].reshape(-1).bool()
            if done_mask.any():
                reason_success = success.reshape(-1)[done_mask]
                reason_collision = collision.reshape(-1)[done_mask]
                reason_surface = None
                if self.termination_max_z is not None:
                    pos_w = self.drone.pos[:, 0, :]
                    reason_surface = (pos_w[:, 2] > self.termination_max_z).reshape(-1)[done_mask]
                pos_error = torch.norm(self.rpos, dim=-1)
                distance = torch.norm(torch.cat([self.rpos, self.rheading], dim=-1), dim=-1)
                pos_error_flat = pos_error.reshape(-1)
                distance_flat = distance.reshape(-1)
                if done_mask.numel() != pos_error_flat.numel():
                    done_mask = done_mask[:pos_error_flat.numel()]
                print(
                    "[HoverTankPPO] done",
                    "env_ids=",
                    torch.where(done_mask)[0].tolist(),
                    "pos_error=",
                    pos_error_flat[done_mask].detach().cpu().numpy(),
                    "distance=",
                    distance_flat[done_mask].detach().cpu().numpy(),
                    "progress=",
                    self.progress_buf.reshape(-1)[done_mask].detach().cpu().numpy(),
                    "success=",
                    success.reshape(-1)[done_mask].detach().cpu().numpy(),
                    "collision=",
                    reason_collision.detach().cpu().numpy(),
                    "surface=",
                    None if reason_surface is None else reason_surface.detach().cpu().numpy(),
                )
                if reason_surface is not None and reason_surface.any():
                    done_ids = torch.where(done_mask)[0]
                    surf_ids = done_ids[reason_surface]
                    z_vals = self.drone.pos[surf_ids, 0, 2].detach().cpu().numpy()
                    min_range = None
                    if hasattr(self, "_last_min_range") and self._last_min_range is not None:
                        min_range = self._last_min_range[surf_ids].detach().cpu().numpy()
                    safety_stop = None
                    if hasattr(self, "_last_safety_stop") and self._last_safety_stop is not None:
                        safety_stop = self._last_safety_stop[surf_ids].detach().cpu().numpy()
                    v_cmd = None
                    if hasattr(self, "_last_v_cmd_b") and self._last_v_cmd_b is not None:
                        v_cmd = self._last_v_cmd_b[surf_ids].detach().cpu().numpy()
                    masses = None
                    volumes = None
                    try:
                        masses = self.drone.masses[surf_ids, 0].detach().cpu().numpy()
                        volumes = self.drone.volumes[surf_ids, 0].detach().cpu().numpy()
                    except Exception:
                        masses = None
                        volumes = None
                    print(
                        "[HoverTankPPO] surface_dbg",
                        "env_ids=",
                        surf_ids.tolist(),
                        "z=",
                        z_vals,
                        "min_range=",
                        min_range,
                        "safety_stop=",
                        safety_stop,
                        "v_cmd_b=",
                        v_cmd,
                        "mass=",
                        masses,
                        "volume=",
                        volumes,
                    )
        td["agents", "reward"] = reward.squeeze(-1)
        return td
