import torch

from torchrl.data import Bounded, Composite, Unbounded
from tensordict.tensordict import TensorDict

from marinegym.envs.isaac_env import AgentSpec
from marinegym.envs.single.hover import Hover
from marinegym.utils.torch import quat_rotate, quat_rotate_inverse, quaternion_to_euler
from bluerov_manual.navigation import Sonar, SonarConfig


class HoverVel(Hover):
    """Hover task with velocity + yaw-rate actions (4D)."""

    def __init__(self, cfg, headless):
        self.ctrl_cfg = cfg.task.get("controller", {}) or {}
        self.sonar_cfg = cfg.task.get("sonar", {}) or {}
        self.safety_shield_cfg = cfg.task.get("safety_shield", {}) or {}
        super().__init__(cfg, headless)
        self.sonar = Sonar(
            SonarConfig(
                num_rays=int(self.sonar_cfg.get("num_rays", 32)),
                fov_deg=float(self.sonar_cfg.get("fov_deg", 180.0)),
                max_range=float(self.sonar_cfg.get("max_range", 10.0)),
                include_angle_encoding=bool(self.sonar_cfg.get("include_angle_encoding", False)),
                min_range_filter=float(self.sonar_cfg.get("min_range_filter", 0.2)),
                hit_path_allow=tuple(
                    self.sonar_cfg.get(
                        "hit_path_allow",
                        ("/tank", "/obstacles", "/maze_walls", "/wall_"),
                    )
                ),
                hit_path_deny=tuple(
                    self.sonar_cfg.get(
                        "hit_path_deny",
                        ("/target", "/BlueROV", "/bluerov", "/drone", "/robot"),
                    )
                ),
                filter_by_env=bool(self.sonar_cfg.get("filter_by_env", True)),
                origin_offset=float(self.sonar_cfg.get("origin_offset", 0.05)),
                ignore_empty_path=bool(self.sonar_cfg.get("ignore_empty_path", True)),
                debug_first_hit=bool(self.sonar_cfg.get("debug_first_hit", False)),
            ),
            device=self.device,
        )

        self.u_max = float(self.ctrl_cfg.get("u_max", 0.6))
        self.v_max = float(self.ctrl_cfg.get("v_max", 0.4))
        self.w_max = float(self.ctrl_cfg.get("w_max", 0.3))
        self.r_max = float(self.ctrl_cfg.get("r_max", 0.6))
        self.vel_kp = float(self.ctrl_cfg.get("vel_kp", 5.0))
        self.yaw_rate_kp = float(self.ctrl_cfg.get("yaw_rate_kp", 1.5))
        self.stabilize_rp = bool(self.ctrl_cfg.get("stabilize_rp", True))
        self.rp_kp = float(self.ctrl_cfg.get("rp_kp", 2.0))
        self.rp_kd = float(self.ctrl_cfg.get("rp_kd", 0.2))
        self.max_accel = self.ctrl_cfg.get("max_accel", None)
        self.max_force = self.ctrl_cfg.get("max_force", None)
        self.max_torque = self.ctrl_cfg.get("max_torque", None)

        self.thrust_axis = int(self.ctrl_cfg.get("thrust_axis", 0))
        self.thrust_sign = float(self.ctrl_cfg.get("thrust_sign", 1.0))
        self.alloc_update_every = int(self.ctrl_cfg.get("allocation_update_every", 1))
        self.action_smooth_alpha = float(self.ctrl_cfg.get("action_smooth_alpha", 0.8))
        # Simple "4D block" mode: bypass thruster allocation and directly command
        # linear velocity (u,v,w) + yaw rate. Useful to debug PPO learning first.
        self.simple_block_4d = bool(self.ctrl_cfg.get("simple_block_4d", False))
        self.simple_block_zero_thrusters = bool(self.ctrl_cfg.get("simple_block_zero_thrusters", True))
        self.simple_block_vel_blend = float(self.ctrl_cfg.get("simple_block_vel_blend", 1.0))
        self.safety_shield_enable = bool(self.safety_shield_cfg.get("enable", False))
        self.safety_shield_slow_distance = float(self.safety_shield_cfg.get("slow_distance", 1.0))
        self.safety_shield_stop_distance = float(self.safety_shield_cfg.get("stop_distance", 0.45))
        self.safety_shield_brake_gain = float(self.safety_shield_cfg.get("brake_gain", 0.0))
        self.safety_shield_use_obstacle_clearance = bool(
            self.safety_shield_cfg.get("use_obstacle_clearance", True)
        )
        self._alloc_step = 0
        self._A_pinv = None
        self._prev_action = None

    def _build_allocation(self):
        from marinegym.utils.torch import quat_axis

        rotor_pos_w, rotor_rot_w = self.drone.rotors_view.get_world_poses()
        r_w = rotor_pos_w[:, 0]
        q_w = rotor_rot_w[:, 0]

        base_pos_w, base_rot_w = self.drone.base_link.get_world_poses()
        base_pos_w = base_pos_w[:, 0]
        base_rot_w = base_rot_w[:, 0]

        d_w = quat_axis(q_w, axis=self.thrust_axis) * self.thrust_sign
        r_rel_w = r_w - base_pos_w.unsqueeze(1)

        base_q = base_rot_w.unsqueeze(1).expand_as(q_w)
        r_b = quat_rotate_inverse(base_q, r_rel_w)
        d_b = quat_rotate_inverse(base_q, d_w)
        tau_b = torch.cross(r_b, d_b, dim=-1)

        A = torch.zeros(self.num_envs, 6, r_b.shape[1], device=self.device)
        A[:, 0:3, :] = d_b.transpose(1, 2)
        A[:, 3:6, :] = tau_b.transpose(1, 2)
        self._A_pinv = torch.linalg.pinv(A)

    def _set_specs(self):
        # Compact policy observation:
        # [rel_b(3), yaw_err_sin/cos(2), body_vel_b(3), yaw_rate(1), depth(1), roll_pitch(2)].
        obs_dim = 3 + 2 + 3 + 1 + 1 + 2
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
                        "action": Bounded(-1.0, 1.0, (1, 4), device=self.device),
                    }
                )
            }
        ).expand(self.num_envs).to(self.device)

        self.reward_spec = Composite(
            {"agents": Composite({"reward": Unbounded((1, 1))})}
        ).expand(self.num_envs).to(self.device)

        stats_spec = Composite({
            "return": Unbounded(1),
            "episode_len": Unbounded(1),
            "pos_error": Unbounded(1),
            "heading_alignment": Unbounded(1),
            "uprightness": Unbounded(1),
            "action_smoothness": Unbounded(1),
        }).expand(self.num_envs).to(self.device)
        self.observation_spec["stats"] = stats_spec
        self.stats = stats_spec.zero()

        self.agent_spec["drone"] = AgentSpec(
            "drone", 1,
            observation_key=("agents", "observation"),
            action_key=("agents", "action"),
            reward_key=("agents", "reward"),
            state_key=("agents", "intrinsics"),
        )

    def _pre_sim_step(self, tensordict):
        self._update_obstacles()
        # Keep robot state in world frame so downstream task logic can
        # consistently convert to local with env offsets in multi-env runs.
        self.drone.get_state(env_frame=False)
        actions = tensordict[("agents", "action")]
        if self._prev_action is None:
            self._prev_action = torch.zeros_like(actions)
        actions = self.action_smooth_alpha * self._prev_action + (1.0 - self.action_smooth_alpha) * actions
        self._prev_action = actions

        a = torch.nan_to_num(actions.squeeze(1), nan=0.0, posinf=0.0, neginf=0.0)
        u_cmd = torch.clamp(a[:, 0], -1.0, 1.0) * self.u_max
        v_cmd = torch.clamp(a[:, 1], -1.0, 1.0) * self.v_max
        w_cmd = torch.clamp(a[:, 2], -1.0, 1.0) * self.w_max
        r_cmd = torch.clamp(a[:, 3], -1.0, 1.0) * self.r_max

        v_cmd_b = torch.stack([u_cmd, v_cmd, w_cmd], dim=-1)
        vel_b = torch.nan_to_num(self.drone.vel_b[:, 0, :3], nan=0.0, posinf=0.0, neginf=0.0)
        ang_b = torch.nan_to_num(self.drone.vel_b[:, 0, 3:6], nan=0.0, posinf=0.0, neginf=0.0)

        if self.safety_shield_enable:
            min_clear = None
            for attr in ("_last_sonar", "_last_map_sonar"):
                ranges = getattr(self, attr, None)
                if ranges is None:
                    continue
                try:
                    clear = torch.min(torch.as_tensor(ranges, device=self.device), dim=1).values
                    min_clear = clear if min_clear is None else torch.minimum(min_clear, clear)
                except Exception:
                    pass
            if self.safety_shield_use_obstacle_clearance and hasattr(self, "_obstacle_clearance"):
                try:
                    obs_clear = self._obstacle_clearance(self.drone.pos[:, 0, :]).reshape(-1)
                    obs_clear = torch.nan_to_num(
                        obs_clear,
                        nan=float("inf"),
                        posinf=float("inf"),
                        neginf=0.0,
                    )
                    min_clear = obs_clear if min_clear is None else torch.minimum(min_clear, obs_clear)
                except Exception:
                    pass
            if min_clear is not None:
                slow_d = max(float(self.safety_shield_slow_distance), 1e-3)
                stop_d = max(float(self.safety_shield_stop_distance), 0.0)
                denom = max(slow_d - stop_d, 1e-3)
                scale = torch.clamp((min_clear - stop_d) / denom, min=0.0, max=1.0).unsqueeze(-1)
                v_cmd_b = v_cmd_b * scale
                r_cmd = r_cmd * scale.squeeze(-1)
                if self.safety_shield_brake_gain > 0.0:
                    stop_mask = (min_clear <= stop_d).unsqueeze(-1)
                    brake_cmd = -float(self.safety_shield_brake_gain) * vel_b
                    limit = torch.tensor(
                        [self.u_max, self.v_max, self.w_max],
                        device=self.device,
                        dtype=v_cmd_b.dtype,
                    ).unsqueeze(0)
                    brake_cmd = torch.maximum(torch.minimum(brake_cmd, limit), -limit)
                    v_cmd_b = torch.where(stop_mask, brake_cmd, v_cmd_b)

        e_v = v_cmd_b - vel_b
        F_b = self.vel_kp * e_v
        T_b = torch.zeros(self.num_envs, 3, device=self.device)
        T_b[:, 2] = self.yaw_rate_kp * (r_cmd - ang_b[:, 2])
        if self.stabilize_rp:
            rpy = torch.nan_to_num(quaternion_to_euler(self.drone.rot[:, 0, :]), nan=0.0, posinf=0.0, neginf=0.0)
            T_b[:, 0] = -self.rp_kp * rpy[:, 0] - self.rp_kd * ang_b[:, 0]
            T_b[:, 1] = -self.rp_kp * rpy[:, 1] - self.rp_kd * ang_b[:, 1]
        F_b = torch.nan_to_num(F_b, nan=0.0, posinf=0.0, neginf=0.0)
        T_b = torch.nan_to_num(T_b, nan=0.0, posinf=0.0, neginf=0.0)

        if self.simple_block_4d:
            rot_w = torch.nan_to_num(self.drone.rot[:, 0, :], nan=0.0, posinf=0.0, neginf=0.0)
            rot_invalid = ~torch.isfinite(self.drone.rot[:, 0, :]).all(dim=-1)
            if bool(rot_invalid.any().item()):
                rot_w = rot_w.clone()
                rot_w[rot_invalid] = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device)

            vel_cmd_w = quat_rotate(rot_w, v_cmd_b)
            omega_cmd_w = torch.zeros(self.num_envs, 3, device=self.device)
            omega_cmd_w[:, 2] = r_cmd

            root_vel = self.drone.get_velocities(clone=False)
            if root_vel.ndim == 2:
                root_vel = root_vel.unsqueeze(1)
            root_vel = root_vel.clone()
            blend = float(max(0.0, min(1.0, self.simple_block_vel_blend)))
            root_vel[:, 0, :3] = (1.0 - blend) * root_vel[:, 0, :3] + blend * vel_cmd_w
            root_vel[:, 0, 3:6] = (1.0 - blend) * root_vel[:, 0, 3:6] + blend * omega_cmd_w
            root_vel = torch.nan_to_num(root_vel, nan=0.0, posinf=0.0, neginf=0.0)
            self.drone.set_velocities(root_vel)

            if self.simple_block_zero_thrusters:
                zero_thr = torch.zeros(self.num_envs, 1, self.drone.num_rotors, device=self.device)
                self.drone.apply_action(zero_thr)

            self.effort = torch.abs(a).mean(dim=-1, keepdim=True)
            self.effort = torch.nan_to_num(self.effort, nan=0.0, posinf=1.0, neginf=0.0)
            return

        if self.max_accel is not None:
            masses = self.drone.masses[:, 0].squeeze(-1)
            max_force = masses * float(self.max_accel)
        else:
            max_force = self.max_force
        if max_force is not None:
            max_force_val = torch.as_tensor(max_force, device=self.device)
            force_norm = torch.norm(F_b, dim=-1, keepdim=True).clamp_min(1e-6)
            scale = torch.clamp(max_force_val / force_norm.squeeze(-1), max=1.0).unsqueeze(-1)
            F_b = F_b * scale
        if self.max_torque is not None:
            max_torque_val = float(self.max_torque)
            torque_norm = torch.norm(T_b, dim=-1, keepdim=True).clamp_min(1e-6)
            scale = torch.clamp(max_torque_val / torque_norm.squeeze(-1), max=1.0).unsqueeze(-1)
            T_b = T_b * scale

        if self._A_pinv is None or (self.alloc_update_every > 0 and self._alloc_step % self.alloc_update_every == 0):
            self._build_allocation()
        self._alloc_step += 1

        wrench = torch.cat([F_b, T_b], dim=-1).unsqueeze(1)
        forces = wrench @ self._A_pinv.transpose(-1, -2)
        f_max = float(self.drone.FORCE_CONSTANTS_0[0].item()) * (3900.0 ** 2)
        thr_action = torch.clamp(forces / f_max, -1.0, 1.0)
        thr_action = torch.nan_to_num(thr_action, nan=0.0, posinf=0.0, neginf=0.0)
        self.effort = torch.nan_to_num(torch.abs(self.drone.apply_action(thr_action)), nan=0.0, posinf=1.0, neginf=0.0)

    def _compute_state_and_obs(self):
        # Keep state in world frame; observation building below explicitly
        # handles world->body transforms.
        self.drone_state = self.drone.get_state(env_frame=False)
        pos_w = self.drone.pos[:, 0, :]
        rot_w = self.drone.rot[:, 0, :]
        vel_b = torch.nan_to_num(self.drone.vel_b[:, 0, :3], nan=0.0, posinf=0.0, neginf=0.0)
        ang_b = torch.nan_to_num(self.drone.vel_b[:, 0, 3:6], nan=0.0, posinf=0.0, neginf=0.0)

        # Guard against occasional invalid simulator states to keep PPO inputs finite.
        scan_pos = pos_w
        scan_rot = rot_w
        valid_pose = torch.isfinite(pos_w).all(dim=-1) & torch.isfinite(rot_w).all(dim=-1)
        if not bool(valid_pose.all().item()):
            scan_pos = pos_w.clone()
            scan_rot = rot_w.clone()
            scan_pos[~valid_pose] = 0.0
            scan_rot[~valid_pose] = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device)

        self.rpos = self.target_pos - self.drone_state[..., :3]
        self.rheading = self.target_heading - self.drone_state[..., 13:16]

        goal_w = self.target_pos.expand(self.num_envs, 1, 3)[:, 0, :]
        rel_w = goal_w - pos_w
        rel_b = quat_rotate_inverse(scan_rot, rel_w)

        yaw_goal = torch.atan2(rel_w[:, 1], rel_w[:, 0])
        yaw = quaternion_to_euler(scan_rot)[:, 2]
        yaw_err = (yaw_goal - yaw + torch.pi) % (2 * torch.pi) - torch.pi
        yaw_sin = torch.sin(yaw_err).unsqueeze(-1)
        yaw_cos = torch.cos(yaw_err).unsqueeze(-1)

        rpy = quaternion_to_euler(scan_rot)
        roll_pitch = rpy[:, :2]
        depth = pos_w[:, 2:3]

        obs = torch.cat(
            [
                rel_b,
                yaw_sin,
                yaw_cos,
                vel_b,
                ang_b[:, 2:3],
                depth,
                roll_pitch,
            ],
            dim=-1,
        ).unsqueeze(1)
        obs = torch.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0)

        return TensorDict(
            {
                "agents": {
                    "observation": obs,
                },
                "stats": self.stats.clone(),
            },
            self.batch_size,
        )
