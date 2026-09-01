"""Open the HoverVelStableGoal tank scene for visual debugging.

This is intentionally not a training/evaluation script. It exists to inspect
the tank, BlueROV spawn, goal, maze walls, dynamic obstacles, sonar/mapping, and
termination bookkeeping in one GUI run.
"""

from __future__ import annotations

import os
import base64
import json
import socket
import sys
import time
import zlib

import hydra
import torch
from omegaconf import OmegaConf
from tensordict import TensorDict

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))


class UdpActionReceiver:
    def __init__(self, host: str, port: int, timeout_sec: float) -> None:
        self.timeout_sec = float(timeout_sec)
        self.last_action: list[float] = [0.0, 0.0, 0.0, 0.0]
        self.last_packet_time = 0.0
        self.reset_requested = False
        self.stop_requested = False
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock.setblocking(False)
        self._sock.bind((host, int(port)))

    def close(self) -> None:
        self._sock.close()

    def poll(self) -> None:
        while True:
            try:
                data, _ = self._sock.recvfrom(4096)
            except BlockingIOError:
                break
            payload = json.loads(data.decode("utf-8"))
            action = payload.get("action", [0.0, 0.0, 0.0, 0.0])
            if len(action) != 4:
                continue
            self.last_action = [float(v) for v in action]
            buttons = payload.get("buttons", {}) or {}
            self.reset_requested = self.reset_requested or bool(buttons.get("reset", False))
            self.stop_requested = bool(buttons.get("stop", False))
            self.last_packet_time = time.time()

    def action_tensor(self, num_envs: int, device) -> torch.Tensor:
        self.poll()
        age = time.time() - self.last_packet_time if self.last_packet_time else float("inf")
        values = self.last_action if age <= self.timeout_sec and not self.stop_requested else [0.0, 0.0, 0.0, 0.0]
        action = torch.as_tensor(values, device=device, dtype=torch.float32).view(1, 4).repeat(num_envs, 1)
        return action.clamp(-1.0, 1.0)

    def consume_reset_requested(self) -> bool:
        requested = self.reset_requested
        self.reset_requested = False
        return requested

    def status(self) -> str:
        age = time.time() - self.last_packet_time if self.last_packet_time else float("inf")
        age_text = "none" if age == float("inf") else f"{age:.2f}s"
        return f"udp_action={ [round(v, 3) for v in self.last_action] } age={age_text} stop={self.stop_requested}"


class UdpTelemetrySender:
    def __init__(
        self,
        host: str,
        port: int,
        publish_every: int,
        send_depth: bool,
        depth_every: int,
        send_rgb: bool,
        rgb_every: int,
        send_map: bool,
        map_every: int,
        image_rotate: str,
    ) -> None:
        self.target = (host, int(port))
        self.publish_every = max(1, int(publish_every))
        self.send_depth = bool(send_depth)
        self.depth_every = max(1, int(depth_every))
        self.send_rgb = bool(send_rgb)
        self.rgb_every = max(1, int(rgb_every))
        self.send_map = bool(send_map)
        self.map_every = max(1, int(map_every))
        self.image_rotate = str(image_rotate).strip().lower()
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    def close(self) -> None:
        self._sock.close()

    def maybe_send(self, env, step_i: int, bt_controller=None) -> None:
        if (step_i % self.publish_every) != 0:
            return
        include_depth = self.send_depth and ((step_i % self.depth_every) == 0)
        include_rgb = self.send_rgb and ((step_i % self.rgb_every) == 0)
        include_map = self.send_map and ((step_i % self.map_every) == 0)
        if include_depth and include_rgb:
            self._send_payload(env, include_depth=False, include_rgb=True, bt_controller=bt_controller)
            self._send_payload(env, include_depth=True, include_rgb=False, include_map=include_map, bt_controller=bt_controller)
            return
        self._send_payload(env, include_depth=include_depth, include_rgb=include_rgb, include_map=include_map, bt_controller=bt_controller)

    def _send_payload(self, env, include_depth: bool, include_rgb: bool, include_map: bool = False, bt_controller=None) -> None:
        payload = _build_telemetry_payload(
            env,
            include_depth=include_depth,
            include_rgb=include_rgb,
            include_map=include_map,
            image_rotate=self.image_rotate,
            bt_controller=bt_controller,
        )
        data = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        try:
            self._sock.sendto(data, self.target)
        except OSError as exc:
            print(f"[debug_open_env] telemetry send failed: {exc}")


class UdpGoalReceiver:
    def __init__(self, host: str, port: int) -> None:
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock.setblocking(False)
        self._sock.bind((host, int(port)))

    def poll(self, env) -> bool:
        updated = False
        while True:
            try:
                data, _ = self._sock.recvfrom(2048)
            except BlockingIOError:
                break
            payload = json.loads(data.decode("utf-8"))
            position = payload.get("position")
            if not position or len(position) < 3:
                continue
            goal_local = torch.tensor(
                [float(position[0]), float(position[1]), float(position[2])],
                device=env.device,
                dtype=env.target_pos.dtype,
            )
            goal_w = goal_local.view(1, 3).repeat(env.num_envs, 1) + env.envs_positions
            env.target_pos[:, 0, :] = goal_w
            try:
                env._set_target_poses(positions=goal_w, env_ids=torch.arange(env.num_envs, device=env.device))
            except Exception:
                pass
            updated = True
        return updated


def _tensor_list(value, env_id: int = 0):
    if value is None:
        return None
    try:
        tensor = torch.as_tensor(value)
        if tensor.ndim > 1:
            tensor = tensor[env_id]
        return torch.nan_to_num(tensor.detach().cpu()).reshape(-1).tolist()
    except Exception:
        return None


def _rotate_image_tensor(img: torch.Tensor, rotate: str) -> torch.Tensor:
    if rotate in {"ccw", "left", "counterclockwise"}:
        return torch.rot90(img, k=1, dims=(-2, -1)) if img.dim() == 2 else torch.rot90(img, k=1, dims=(0, 1))
    if rotate in {"cw", "right", "clockwise"}:
        return torch.rot90(img, k=-1, dims=(-2, -1)) if img.dim() == 2 else torch.rot90(img, k=-1, dims=(0, 1))
    if rotate in {"180", "half"}:
        return torch.rot90(img, k=2, dims=(-2, -1)) if img.dim() == 2 else torch.rot90(img, k=2, dims=(0, 1))
    return img


def _get_depth_payload(env, env_id: int = 0, image_rotate: str = "ccw") -> dict | None:
    if getattr(env, "depth_camera", None) is None:
        return None
    if bool(getattr(env, "_depth_disabled", False)):
        return None
    try:
        imgs = env.depth_camera.get_images()
    except Exception:
        return None
    data_type = str(getattr(env, "depth_cfg", {}).get("data_type", "distance_to_camera"))
    if imgs is None or data_type not in imgs:
        return None
    depth_img = imgs[data_type]
    try:
        depth_img = torch.as_tensor(depth_img).detach()
        depth_img = depth_img.squeeze()
        if depth_img.dim() == 3:
            depth_img = depth_img[env_id]
        if depth_img.dim() != 2:
            return None
        max_range = float(getattr(env, "depth_cfg", {}).get("max_range", 10.0))
        depth_img = torch.nan_to_num(depth_img.float(), nan=max_range, posinf=max_range, neginf=0.0)
        depth_img = depth_img.clamp(min=0.0, max=max_range).cpu().contiguous()
        depth_img = _rotate_image_tensor(depth_img, image_rotate).contiguous()
        array = depth_img.numpy().astype("float32", copy=False)
        raw = zlib.compress(array.tobytes(order="C"), level=1)
        return {
            "height": int(array.shape[0]),
            "width": int(array.shape[1]),
            "encoding": "32FC1",
            "is_bigendian": 0,
            "step": int(array.shape[1] * 4),
            "zlib_base64": base64.b64encode(raw).decode("ascii"),
        }
    except Exception:
        return None


def _get_rgb_payload(env, env_id: int = 0, image_rotate: str = "ccw") -> dict | None:
    if getattr(env, "depth_camera", None) is None:
        return None
    if bool(getattr(env, "_depth_disabled", False)):
        return None
    try:
        imgs = env.depth_camera.get_images()
    except Exception:
        return None
    if imgs is None or "rgb" not in imgs:
        return None
    rgb_img = imgs["rgb"]
    try:
        rgb_img = torch.as_tensor(rgb_img).detach().squeeze()
        if rgb_img.dim() == 4:
            rgb_img = rgb_img[env_id]
        if rgb_img.dim() == 3 and rgb_img.shape[0] in (3, 4):
            rgb_img = rgb_img[:3].permute(1, 2, 0)
        elif rgb_img.dim() == 3 and rgb_img.shape[-1] in (3, 4):
            rgb_img = rgb_img[..., :3]
        else:
            return None
        if rgb_img.dtype != torch.uint8:
            rgb_img = rgb_img.float()
            if float(torch.nanmax(rgb_img).item()) <= 1.0:
                rgb_img = rgb_img * 255.0
            rgb_img = torch.nan_to_num(rgb_img, nan=0.0, posinf=255.0, neginf=0.0).clamp(0.0, 255.0).byte()
        rgb_img = _rotate_image_tensor(rgb_img, image_rotate)
        rgb_img = rgb_img.cpu().contiguous()
        array = rgb_img.numpy()
        raw = zlib.compress(array.tobytes(order="C"), level=1)
        return {
            "height": int(array.shape[0]),
            "width": int(array.shape[1]),
            "encoding": "rgb8",
            "is_bigendian": 0,
            "step": int(array.shape[1] * 3),
            "zlib_base64": base64.b64encode(raw).decode("ascii"),
        }
    except Exception:
        return None


def _get_map_payload(env, env_id: int = 0) -> dict | None:
    occ = getattr(env, "_map_occ", None)
    if occ is None:
        return None
    try:
        occ_t = torch.as_tensor(occ).detach()
        if occ_t.dim() == 3:
            occ_t = occ_t[env_id]
        if occ_t.dim() != 2:
            return None
        occ_t = torch.nan_to_num(occ_t.float(), nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)
        occ_u8 = torch.round(occ_t * 100.0).to(torch.uint8).cpu().contiguous()
        array = occ_u8.numpy()
        x_min = float(getattr(env, "_map_x_min", 0.0))
        x_max = float(getattr(env, "_map_x_max", array.shape[1]))
        y_min = float(getattr(env, "_map_y_min", 0.0))
        y_max = float(getattr(env, "_map_y_max", array.shape[0]))
        dx = (x_max - x_min) / max(1, int(array.shape[1]) - 1)
        dy = (y_max - y_min) / max(1, int(array.shape[0]) - 1)
        resolution = float(max(min(dx, dy), 1e-6))
        width = int(round((x_max - x_min) / resolution)) + 1
        height = int(round((y_max - y_min) / resolution)) + 1
        if width > 0 and height > 0 and (width != int(array.shape[1]) or height != int(array.shape[0])):
            src = torch.as_tensor(array, dtype=torch.uint8)
            src_h = int(src.shape[0])
            src_w = int(src.shape[1])
            ys = torch.arange(src_h, dtype=torch.float32)
            xs = torch.arange(src_w, dtype=torch.float32)
            yy, xx = torch.meshgrid(ys, xs, indexing="ij")
            world_x = x_min + xx * float(dx)
            world_y = y_min + yy * float(dy)
            dst_x = torch.round((world_x - x_min) / resolution).long().clamp(0, width - 1)
            dst_y = torch.round((world_y - y_min) / resolution).long().clamp(0, height - 1)
            dst = torch.zeros((height, width), dtype=torch.uint8)
            vals = src.reshape(-1)
            keep = vals > 0
            if bool(keep.any().item()):
                flat_idx = (dst_y.reshape(-1)[keep] * width + dst_x.reshape(-1)[keep]).long()
                dst_flat = dst.reshape(-1)
                try:
                    dst_flat.scatter_reduce_(0, flat_idx, vals[keep], reduce="amax", include_self=True)
                except Exception:
                    for idx_i, val_i in zip(flat_idx.tolist(), vals[keep].tolist()):
                        if int(val_i) > int(dst_flat[idx_i].item()):
                            dst_flat[idx_i] = int(val_i)
            array = dst.cpu().numpy()
        raw = zlib.compress(array.tobytes(order="C"), level=1)
        return {
            "height": int(array.shape[0]),
            "width": int(array.shape[1]),
            "resolution": resolution,
            "origin": [x_min - 0.5 * resolution, y_min - 0.5 * resolution, 0.0],
            "encoding": "uint8_0_100",
            "zlib_base64": base64.b64encode(raw).decode("ascii"),
        }
    except Exception:
        return None


def _get_map_occ_debug_payload(env, env_id: int = 0, current_origin_local=None) -> dict | None:
    """Return raw internal map layers without ROS OccupancyGrid resampling."""
    occ = getattr(env, "_map_occ", None)
    if occ is None:
        return None

    def _layer_payload(layer) -> dict | None:
        if layer is None:
            return None
        layer_t = torch.as_tensor(layer).detach()
        if layer_t.dim() == 3:
            layer_t = layer_t[env_id]
        if layer_t.dim() != 2:
            return None
        layer_t = torch.nan_to_num(layer_t.float(), nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)
        layer_u8 = torch.round(layer_t * 100.0).to(torch.uint8).cpu().contiguous()
        array = layer_u8.numpy()
        raw = zlib.compress(array.tobytes(order="C"), level=1)
        return {
            "height": int(array.shape[0]),
            "width": int(array.shape[1]),
            "zlib_base64": base64.b64encode(raw).decode("ascii"),
        }

    try:
        occ_layer = _layer_payload(occ)
        if occ_layer is None:
            return None
        x_min = float(getattr(env, "_map_x_min", 0.0))
        x_max = float(getattr(env, "_map_x_max", occ_layer["width"] - 1))
        y_min = float(getattr(env, "_map_y_min", 0.0))
        y_max = float(getattr(env, "_map_y_max", occ_layer["height"] - 1))
        payload = {
            "x_min": x_min,
            "x_max": x_max,
            "y_min": y_min,
            "y_max": y_max,
            "threshold": 5,
            "layers": {"occ": occ_layer},
        }
        hit_points = getattr(env, "_map_last_hit_points_local", None)
        if hit_points is not None:
            hit_t = torch.as_tensor(hit_points).detach().float().cpu()
            if hit_t.ndim == 2 and hit_t.shape[1] >= 2:
                payload["hit_points"] = hit_t[:, :2].tolist()
        dyn_hit_points = getattr(env, "_map_last_hit_points_dyn_local", None)
        if dyn_hit_points is not None:
            dyn_hit_t = torch.as_tensor(dyn_hit_points).detach().float().cpu()
            if dyn_hit_t.ndim == 2 and dyn_hit_t.shape[1] >= 2:
                payload["dynamic_hit_points"] = dyn_hit_t[:, :2].tolist()
        if current_origin_local is not None:
            origin_t = torch.as_tensor(current_origin_local).detach().float().cpu().reshape(-1)
            if origin_t.numel() >= 2:
                z = float(origin_t[2].item()) if origin_t.numel() >= 3 else 0.0
                payload["scan_origin"] = [float(origin_t[0].item()), float(origin_t[1].item()), z]
        last_scan_origin = getattr(env, "_map_last_scan_origin_local", None)
        if last_scan_origin is not None:
            origin_t = torch.as_tensor(last_scan_origin).detach().float().cpu().reshape(-1)
            if origin_t.numel() >= 2:
                z = float(origin_t[2].item()) if origin_t.numel() >= 3 else 0.0
                payload["last_scan_origin"] = [float(origin_t[0].item()), float(origin_t[1].item()), z]
        static_layer = _layer_payload(getattr(env, "_map_static", None))
        dynamic_layer = _layer_payload(getattr(env, "_map_dynamic", None))
        if static_layer is not None:
            payload["layers"]["static"] = static_layer
        if dynamic_layer is not None:
            payload["layers"]["dynamic"] = dynamic_layer
        return payload
    except Exception:
        return None


def _get_path_payload(env, env_id: int = 0) -> dict | None:
    path_local = None
    if hasattr(env, "_maybe_refresh_overlay_path"):
        try:
            path_local = env._maybe_refresh_overlay_path(env_id, force=False)
        except Exception:
            path_local = None
    if path_local is None:
        path_local = getattr(env, "_map_last_overlay_path_local", None)
    if path_local is None and hasattr(env, "_compute_map_overlay_path_local"):
        try:
            path_local = env._compute_map_overlay_path_local(env_id)
        except Exception:
            path_local = None
    if path_local is None:
        return None
    try:
        pts = []
        for p in path_local:
            if p is None or len(p) < 2:
                continue
            z = float(p[2]) if len(p) > 2 else 0.0
            pts.append([float(p[0]), float(p[1]), z])
        if len(pts) < 2:
            return None
        return {"points": pts}
    except Exception:
        return None


def _build_telemetry_payload(
    env,
    include_depth: bool,
    include_rgb: bool,
    include_map: bool,
    image_rotate: str,
    bt_controller=None,
) -> dict:
    env_id = 0
    env.drone.get_state(env_frame=False)
    env_offset = env.envs_positions[env_id].to(env.device)
    pos_w = env.drone.pos[env_id, 0, :]
    pos_local = pos_w - env_offset
    rot_w = env.drone.rot[env_id, 0, :]
    vel_w6 = env.drone.get_velocities(True)[env_id, 0, :]
    vel_b6 = getattr(env.drone, "vel_b", None)
    goal_w = env.target_pos[env_id, 0, :] if hasattr(env, "target_pos") else None
    goal_local = goal_w - env_offset if goal_w is not None else None

    sonar = getattr(env, "_last_sonar", None)
    sonar_ranges = None
    if sonar is not None:
        try:
            sonar_ranges = torch.nan_to_num(torch.as_tensor(sonar[env_id]), nan=float(env.sonar.cfg.max_range)).cpu().tolist()
        except Exception:
            sonar_ranges = None

    obstacles = []
    try:
        if getattr(env, "obstacles", None) is not None and int(getattr(env, "num_obstacles", 0)) > 0:
            obst_pos_w, obst_rot_w = env.obstacles.get_world_poses()
            obst_pos_w = obst_pos_w.view(env.num_envs, int(env.num_obstacles), 3)[env_id]
            obst_rot_w = obst_rot_w.view(env.num_envs, int(env.num_obstacles), 4)[env_id]
            obst_pos_local = obst_pos_w - env_offset.view(1, 3)
            for pos, rot in zip(obst_pos_local, obst_rot_w):
                obstacles.append({"position": _tensor_list(pos), "orientation_wxyz": _tensor_list(rot)})
    except Exception:
        obstacles = []

    reason_rates = getattr(env, "_last_done_reason_rates", {}) or {}
    payload = {
        "stamp": time.time(),
        "frame_id": "map",
        "child_frame_id": "bluerov_base_link",
        "pose": {
            "position": _tensor_list(pos_local),
            "orientation_wxyz": _tensor_list(rot_w),
        },
        "twist": {
            "linear_world": _tensor_list(vel_w6[:3]),
            "angular_world": _tensor_list(vel_w6[3:6]),
            "body": _tensor_list(vel_b6[env_id, 0, :]) if vel_b6 is not None else None,
        },
        "goal": {"position": _tensor_list(goal_local)} if goal_local is not None else None,
        "sonar": {
            "ranges": sonar_ranges,
            "angle_min": float(-0.5 * float(env.sonar.cfg.fov_deg) * torch.pi / 180.0),
            "angle_max": float(0.5 * float(env.sonar.cfg.fov_deg) * torch.pi / 180.0),
            "range_min": float(env.sonar.cfg.min_range_filter),
            "range_max": float(env.sonar.cfg.max_range),
            "origin_offset": float(getattr(env.sonar.cfg, "origin_offset", 0.0)),
        },
        "obstacles": obstacles,
        "metrics": {
            "goal_dist": float(getattr(env, "_last_goal_dist", torch.tensor([float("nan")]))[env_id].item())
            if hasattr(env, "_last_goal_dist")
            else float("nan"),
            "min_clearance": float(getattr(env, "_last_min_clearance_mean", float("nan"))),
            "front_ttc": float(getattr(env, "_last_front_ttc_mean", float("nan"))),
            "path_available_rate": float(getattr(env, "_last_path_available_rate", float("nan"))),
        },
        "termination_reason_rates": dict(reason_rates),
    }
    if include_depth:
        depth = _get_depth_payload(env, env_id=env_id, image_rotate=image_rotate)
        if depth is not None:
            payload["depth"] = depth
    if include_rgb:
        rgb = _get_rgb_payload(env, env_id=env_id, image_rotate=image_rotate)
        if rgb is not None:
            payload["rgb"] = rgb
    if include_map:
        occ_map = _get_map_payload(env, env_id=env_id)
        if occ_map is not None:
            payload["map"] = occ_map
        occ_debug = _get_map_occ_debug_payload(env, env_id=env_id, current_origin_local=pos_local)
        if occ_debug is not None:
            payload["map_occ_debug"] = occ_debug
    path = _get_path_payload(env, env_id=env_id)
    if path is not None:
        payload["path"] = path
    if bt_controller is not None and hasattr(bt_controller, "dynamic_estimates_payload"):
        payload["dynamic_obstacle_estimates"] = bt_controller.dynamic_estimates_payload(env_id=env_id)
    return payload


def _action_for_mode(
    mode: str,
    step_i: int,
    num_envs: int,
    device,
    scale: float,
    udp_receiver: UdpActionReceiver | None = None,
) -> torch.Tensor:
    action = torch.zeros(num_envs, 4, device=device, dtype=torch.float32)
    mode = mode.strip().lower()
    if mode == "forward":
        action[:, 0] = scale
    elif mode == "strafe":
        action[:, 1] = scale
    elif mode == "yaw":
        action[:, 3] = scale
    elif mode == "circle":
        phase = 1.0 if ((step_i // 240) % 2 == 0) else -1.0
        action[:, 0] = scale
        action[:, 3] = 0.5 * scale * phase
    elif mode == "ros2_joy":
        if udp_receiver is None:
            raise ValueError("debug_launch.action_mode=ros2_joy requires a UDP action receiver")
        action = udp_receiver.action_tensor(num_envs, device)
    elif mode == "bt_auto":
        pass
    elif mode != "hold":
        raise ValueError(f"Unknown debug_launch.action_mode={mode!r}")
    return action.clamp(-1.0, 1.0)


def _obs_from_tensordict(td, env) -> torch.Tensor:
    obs = None
    for key in (("agents", "observation"), "observation", "policy"):
        try:
            obs = td.get(key)
            if obs is not None:
                break
        except Exception:
            pass
    if obs is None:
        try:
            td_now = env._compute_state_and_obs()
            obs = td_now.get(("agents", "observation"))
        except Exception as exc:
            raise RuntimeError("Could not read observation for BT controller") from exc
    obs = torch.as_tensor(obs, device=env.device, dtype=torch.float32)
    if obs.dim() == 3:
        obs = obs[:, 0, :]
    elif obs.dim() == 1:
        obs = obs.view(1, -1)
    return obs


def _bt_action(bt_controller, td, env) -> torch.Tensor:
    obs = _obs_from_tensordict(td, env)
    pos_w = env.drone.pos[:, 0, :]
    goal_w = env.target_pos[:, 0, :]
    final_goal_dist = torch.norm(goal_w - pos_w, dim=-1)
    with torch.inference_mode():
        action = bt_controller.act(obs, final_goal_dist=final_goal_dist)
    try:
        replan_env_ids = bt_controller.pop_replan_env_ids(env.device)
        if replan_env_ids.numel() > 0 and hasattr(env, "_maybe_refresh_overlay_path"):
            for env_id in replan_env_ids.tolist():
                try:
                    env._maybe_refresh_overlay_path(int(env_id), force=True)
                except Exception:
                    continue
    except Exception:
        pass
    return action.to(env.device).clamp(-1.0, 1.0)


def _maybe_reset_done(env, done: torch.Tensor, reset_on_done: bool, bt_controller, step_label: str):
    if not bool(done.any().item()) or not reset_on_done:
        return None
    reason_rates = getattr(env, "_last_done_reason_rates", {}) or {}
    done_envs = done.nonzero().flatten().tolist()
    print(
        f"[debug_open_env] reset_on_done step={step_label} "
        f"done_envs={done_envs} reasons={reason_rates}"
    )
    td = env.reset(
        TensorDict(
            {"_reset": done},
            batch_size=[env.num_envs],
            device=env.device,
        )
    )
    if bt_controller is not None:
        bt_controller.reset_internal_state(env.num_envs, env.device)
    return td


def _get_obstacle_local(env) -> torch.Tensor | None:
    if not bool(getattr(env, "enable_obstacles", False)):
        return None
    num_obstacles = int(getattr(env, "num_obstacles", 0))
    if num_obstacles <= 0:
        return None
    try:
        if getattr(env, "obstacles", None) is not None:
            pos_w, _ = env.obstacles.get_world_poses()
            pos_w = pos_w.view(env.num_envs, num_obstacles, 3)
            return pos_w - env.envs_positions.unsqueeze(1)
    except Exception:
        pass
    obst_center = getattr(env, "obst_center", None)
    if obst_center is None:
        return None
    return torch.as_tensor(obst_center, device=env.device).reshape(env.num_envs, num_obstacles, 3)


def _print_scene_summary(env) -> None:
    pos_local = env.drone.pos[:, 0, :] - env.envs_positions
    goal_local = env.target_pos[:, 0, :] - env.envs_positions
    obst_local = _get_obstacle_local(env)
    print("\n" + "=" * 78)
    print("BLUEROV HOVER_VEL ENV DEBUG")
    print("=" * 78)
    print(f"task={env.cfg.task.name} num_envs={env.num_envs} device={env.device}")
    print(f"start_local[0]={pos_local[0].detach().cpu().numpy().round(3).tolist()}")
    print(f"goal_local[0]={goal_local[0].detach().cpu().numpy().round(3).tolist()}")
    print(
        "maze: "
        f"enable={bool(getattr(env, 'enable_maze', False))} "
        f"num={int(getattr(env, 'num_maze_walls', 0))}"
    )
    print(
        "obstacles: "
        f"enable={bool(getattr(env, 'enable_obstacles', False))} "
        f"num={int(getattr(env, 'num_obstacles', 0))} "
        f"radius={float(getattr(env, 'obst_cfg', {}).get('radius', 0.0)):.3f}"
    )
    print(
        "mapping: "
        f"enable={bool(getattr(env, 'mapping_enabled', False))} "
        f"classify_dynamic_from_gt={bool(getattr(env, '_map_classify_dynamic_from_gt', False))} "
        f"dynamic_stamp_from_gt={bool(getattr(env, '_map_dyn_stamp_from_gt', False))} "
        f"maze_wall_sonar_fallback={bool(getattr(env, '_map_maze_wall_sonar_fallback', False))} "
        f"dynamic_obstacle_sonar_fallback={bool(getattr(env, '_map_dynamic_obstacle_sonar_fallback', False))} "
        f"decay={float(getattr(env, '_map_decay', 0.0)):.3f} "
        f"dynamic_decay={float(getattr(env, '_map_dyn_decay', 0.0)):.3f} "
        f"hit_stamp={float(getattr(env, '_map_static_hit_stamp_radius_m', 0.0)):.2f}/"
        f"{float(getattr(env, '_map_dynamic_hit_stamp_radius_m', 0.0)):.2f}"
    )
    if obst_local is not None and obst_local.numel() > 0:
        shown = obst_local[0, : min(6, obst_local.shape[1])].detach().cpu().numpy().round(3).tolist()
        print(f"obstacle_local[0][:6]={shown}")
    print("=" * 78 + "\n")


@hydra.main(version_base=None, config_path="../cfg", config_name="debug/open_hover_vel_env")
def main(cfg) -> None:
    OmegaConf.register_new_resolver("eval", eval)
    OmegaConf.resolve(cfg)
    OmegaConf.set_struct(cfg, False)

    from bluerov_manual.sim_app import init_app

    sim_app = init_app(cfg)

    import bluerov_manual.envs  # noqa: F401
    from marinegym.envs.isaac_env import IsaacEnv

    env_class = IsaacEnv.REGISTRY[cfg.task.name]
    env = env_class(cfg, headless=cfg.headless)
    env.set_seed(int(cfg.get("seed", 0)))

    td = env.reset()
    _print_scene_summary(env)

    launch_cfg = cfg.get("debug_launch", {}) or {}
    steps = int(launch_cfg.get("steps", 20000))
    print_every = max(1, int(launch_cfg.get("print_every", 60)))
    action_mode = str(launch_cfg.get("action_mode", "hold"))
    action_scale = float(launch_cfg.get("action_scale", 0.18))
    reset_on_done = bool(launch_cfg.get("reset_on_done", True))
    close_on_finish = bool(launch_cfg.get("close_on_finish", False))
    ros2_joy_cfg = launch_cfg.get("ros2_joy", {}) or {}
    ros2_telemetry_cfg = launch_cfg.get("ros2_telemetry", {}) or {}
    ros2_goal_cfg = launch_cfg.get("ros2_goal", {}) or {}
    udp_receiver = None
    if action_mode.strip().lower() == "ros2_joy":
        udp_host = str(ros2_joy_cfg.get("udp_host", "127.0.0.1"))
        udp_port = int(ros2_joy_cfg.get("udp_port", 15000))
        timeout_sec = float(ros2_joy_cfg.get("timeout_sec", 0.5))
        udp_receiver = UdpActionReceiver(udp_host, udp_port, timeout_sec)
        print(f"[debug_open_env] listening for ROS2 joystick UDP on {udp_host}:{udp_port}")
    telemetry_sender = None
    if bool(ros2_telemetry_cfg.get("enable", False)):
        telemetry_host = str(ros2_telemetry_cfg.get("udp_host", "127.0.0.1"))
        telemetry_port = int(ros2_telemetry_cfg.get("udp_port", 15010))
        telemetry_sender = UdpTelemetrySender(
            telemetry_host,
            telemetry_port,
            publish_every=int(ros2_telemetry_cfg.get("publish_every", 3)),
            send_depth=bool(ros2_telemetry_cfg.get("send_depth", True)),
            depth_every=int(ros2_telemetry_cfg.get("depth_every", 10)),
            send_rgb=bool(ros2_telemetry_cfg.get("send_rgb", True)),
            rgb_every=int(ros2_telemetry_cfg.get("rgb_every", 10)),
            send_map=bool(ros2_telemetry_cfg.get("send_map", True)),
            map_every=int(ros2_telemetry_cfg.get("map_every", 1)),
            image_rotate=str(ros2_telemetry_cfg.get("image_rotate", "ccw")),
        )
        print(f"[debug_open_env] sending ROS2 telemetry UDP to {telemetry_host}:{telemetry_port}")
    goal_receiver = None
    if bool(ros2_goal_cfg.get("enable", False)):
        goal_host = str(ros2_goal_cfg.get("udp_host", "127.0.0.1"))
        goal_port = int(ros2_goal_cfg.get("udp_port", 15020))
        goal_receiver = UdpGoalReceiver(goal_host, goal_port)
        print(f"[debug_open_env] listening for ROS2 goal UDP on {goal_host}:{goal_port}")
    bt_controller = None
    if action_mode.strip().lower() == "bt_auto":
        from play_bluerov_behavior_tree_maze import RobustMazeBTController

        bt_cfg = (launch_cfg.get("bt_auto", {}) or {}).get("bt", {}) or launch_cfg.get("bt", {}) or {}
        bt_controller = RobustMazeBTController(bt_cfg, env)
        bt_controller.reset_internal_state(env.num_envs, env.device)
        print("[debug_open_env] autonomous BT controller enabled")

    print(
        "[debug_open_env] "
        f"steps={steps} action_mode={action_mode} action_scale={action_scale} "
        f"reset_on_done={reset_on_done}"
    )

    t0 = time.time()
    for step_i in range(steps):
        if goal_receiver is not None and goal_receiver.poll(env) and bt_controller is not None:
            bt_controller.reset_internal_state(env.num_envs, env.device)
        if bt_controller is not None:
            action = _bt_action(bt_controller, td, env)
        else:
            action = _action_for_mode(action_mode, step_i, env.num_envs, env.device, action_scale, udp_receiver)
        td_action = TensorDict({}, batch_size=[env.num_envs], device=env.device)
        td_action.set(("agents", "action"), action.view(env.num_envs, 1, 4))
        td_step = env.step(td_action)
        td_next = td_step.get("next", td_step)
        td = td_next
        if telemetry_sender is not None:
            telemetry_sender.maybe_send(env, step_i, bt_controller=bt_controller)

        done = td_next.get(
            "done",
            torch.zeros(env.num_envs, 1, dtype=torch.bool, device=env.device),
        ).reshape(env.num_envs)

        if (step_i % print_every) == 0:
            pos_local = env.drone.pos[:, 0, :] - env.envs_positions
            goal_local = env.target_pos[:, 0, :] - env.envs_positions
            dist = torch.norm(goal_local - pos_local, dim=-1)
            min_clear = getattr(env, "_last_min_clearance_mean", float("nan"))
            front_ttc = getattr(env, "_last_front_ttc_mean", float("nan"))
            reason_rates = getattr(env, "_last_done_reason_rates", {}) or {}
            map_hit_count = int(getattr(env, "_map_last_hit_count", 0))
            map_hit_ratio = float(getattr(env, "_map_last_hit_ratio", 0.0))
            sonar_hit_path = getattr(getattr(env, "sonar", None), "last_hit_path", None)
            sonar_hit_dist = getattr(getattr(env, "sonar", None), "last_hit_distance", None)
            sonar_hit = "none"
            if sonar_hit_path is not None:
                sonar_hit = (
                    f"{sonar_hit_path}:{float(sonar_hit_dist):.2f}"
                    if sonar_hit_dist is not None
                    else str(sonar_hit_path)
                )
            joystick_status = f" {udp_receiver.status()}" if udp_receiver is not None else ""
            bt_status = ""
            if bt_controller is not None and getattr(bt_controller, "_last_debug", None):
                dbg0 = bt_controller._last_debug[0]
                bt_status = (
                    f" bt_zone={dbg0.get('zone', 'na')} "
                    f"front_min={float(dbg0.get('front_min', float('nan'))):.3f} "
                    f"final_dist={float(dbg0.get('final_goal_dist', float('nan'))):.3f} "
                    f"escape_hold={int(dbg0.get('escape_hold', 0))}"
                )
            print(
                f"[debug_open_env] step={step_i:06d} "
                f"pos0={pos_local[0].detach().cpu().numpy().round(3).tolist()} "
                f"goal_dist0={float(dist[0].item()):.3f} "
                f"min_clear={float(min_clear):.3f} front_ttc={float(front_ttc):.3f} "
                f"map_hits={map_hit_count} hit_ratio={map_hit_ratio:.2f} first_hit={sonar_hit} "
                f"done={bool(done.any().item())} reasons={reason_rates}{joystick_status}{bt_status}"
            )

        if udp_receiver is not None and udp_receiver.consume_reset_requested():
            td = env.reset()
            if bt_controller is not None:
                bt_controller.reset_internal_state(env.num_envs, env.device)

        reset_td = _maybe_reset_done(env, done, reset_on_done, bt_controller, str(step_i))
        if reset_td is not None:
            td = reset_td

    print(f"[debug_open_env] finished stepping in {time.time() - t0:.1f}s")
    if close_on_finish:
        try:
            sim_app.close()
        except Exception:
            pass
    else:
        print("[debug_open_env] leaving SimulationApp open until the process is stopped.")
        while True:
            if goal_receiver is not None and goal_receiver.poll(env) and bt_controller is not None:
                bt_controller.reset_internal_state(env.num_envs, env.device)
            if bt_controller is not None:
                action = _bt_action(bt_controller, td, env)
            else:
                action = _action_for_mode(action_mode, 0, env.num_envs, env.device, action_scale, udp_receiver)
            td_action = TensorDict({}, batch_size=[env.num_envs], device=env.device)
            td_action.set(("agents", "action"), action.view(env.num_envs, 1, 4))
            td_step = env.step(td_action)
            td = td_step.get("next", td_step)
            if telemetry_sender is not None:
                telemetry_sender.maybe_send(env, 0, bt_controller=bt_controller)
            done = td.get(
                "done",
                torch.zeros(env.num_envs, 1, dtype=torch.bool, device=env.device),
            ).reshape(env.num_envs)
            if udp_receiver is not None and udp_receiver.consume_reset_requested():
                td = env.reset()
                if bt_controller is not None:
                    bt_controller.reset_internal_state(env.num_envs, env.device)
                continue
            reset_td = _maybe_reset_done(env, done, reset_on_done, bt_controller, "open")
            if reset_td is not None:
                td = reset_td


if __name__ == "__main__":
    main()
