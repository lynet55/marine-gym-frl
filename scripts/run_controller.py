"""Run a controller in an environment, with the ROS 2 bridges.

The controller is a config group, cfg/controller/: ``controller=policy`` (a trained
policy, set controller.checkpoint) or ``controller=bt`` (the maze behavior tree). The
environment comes from the run preset, cfg/run/*.yaml. Goals arrive from RViz and
telemetry goes out over the same UDP bridges as scripts/debug_open_hover_vel_env.py,
whose bridge classes this reuses. Launch through release/scripts/03_run_controller.sh.
"""

from __future__ import annotations

import itertools
import os
import sys
import time

import hydra
import torch
from omegaconf import OmegaConf
from tensordict import TensorDict

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from controllers import make_controller, reset_finished_envs  # noqa: E402
from debug_open_hover_vel_env import UdpGoalReceiver, UdpTelemetrySender, _print_scene_summary  # noqa: E402


def _make_bridges(launch_cfg):
    goal_cfg = launch_cfg.get("ros2_goal", {}) or {}
    telemetry_cfg = launch_cfg.get("ros2_telemetry", {}) or {}
    goal_receiver = None
    if bool(goal_cfg.get("enable", False)):
        host = str(goal_cfg.get("udp_host", "127.0.0.1"))
        port = int(goal_cfg.get("udp_port", 15020))
        goal_receiver = UdpGoalReceiver(host, port)
        print(f"[run_controller] listening for ROS2 goal UDP on {host}:{port}")
    telemetry_sender = None
    if bool(telemetry_cfg.get("enable", False)):
        host = str(telemetry_cfg.get("udp_host", "127.0.0.1"))
        port = int(telemetry_cfg.get("udp_port", 15010))
        telemetry_sender = UdpTelemetrySender(
            host,
            port,
            publish_every=int(telemetry_cfg.get("publish_every", 3)),
            send_depth=bool(telemetry_cfg.get("send_depth", True)),
            depth_every=int(telemetry_cfg.get("depth_every", 10)),
            send_rgb=bool(telemetry_cfg.get("send_rgb", True)),
            rgb_every=int(telemetry_cfg.get("rgb_every", 10)),
            send_map=bool(telemetry_cfg.get("send_map", True)),
            map_every=int(telemetry_cfg.get("map_every", 1)),
            image_rotate=str(telemetry_cfg.get("image_rotate", "ccw")),
        )
        print(f"[run_controller] sending ROS2 telemetry UDP to {host}:{port}")
    return goal_receiver, telemetry_sender


def _print_status(env, step_i: int, done: torch.Tensor, controller) -> None:
    pos_local = env.drone.pos[:, 0, :] - env.envs_positions
    goal_local = env.target_pos[:, 0, :] - env.envs_positions
    dist = torch.norm(goal_local - pos_local, dim=-1)
    print(
        f"[run_controller] step={step_i:06d} "
        f"pos0={pos_local[0].detach().cpu().numpy().round(3).tolist()} "
        f"goal_dist0={float(dist[0].item()):.3f} "
        f"min_clear={float(getattr(env, '_last_min_clearance_mean', float('nan'))):.3f} "
        f"front_ttc={float(getattr(env, '_last_front_ttc_mean', float('nan'))):.3f} "
        f"done={bool(done.any().item())} reasons={getattr(env, '_last_done_reason_rates', {}) or {}} "
        f"{controller.status()}"
    )


@hydra.main(version_base=None, config_path="../cfg", config_name="run/point_a_to_b")
def main(cfg) -> None:
    # Isaac Sim exits through os._exit (fast shutdown), which drops block-buffered output
    # when stdout is a pipe or a file, so flush every line.
    sys.stdout.reconfigure(line_buffering=True)
    OmegaConf.register_new_resolver("eval", eval)
    OmegaConf.resolve(cfg)
    OmegaConf.set_struct(cfg, False)

    from bluerov_manual.sim_app import init_app

    sim_app = init_app(cfg)

    import bluerov_manual.envs  # noqa: F401
    from marinegym.envs.isaac_env import IsaacEnv

    env = IsaacEnv.REGISTRY[cfg.task.name](cfg, headless=cfg.headless)
    env.set_seed(int(cfg.get("seed", 0)))
    td = env.reset()
    _print_scene_summary(env)

    controller = make_controller(cfg.controller, env)
    launch_cfg = cfg.get("debug_launch", {}) or {}
    steps = int(launch_cfg.get("steps", 20000))
    print_every = max(1, int(launch_cfg.get("print_every", 60)))
    reset_on_done = bool(launch_cfg.get("reset_on_done", True))
    close_on_finish = bool(launch_cfg.get("close_on_finish", False))
    goal_receiver, telemetry_sender = _make_bridges(launch_cfg)
    print(f"[run_controller] controller={cfg.controller.target} steps={steps} reset_on_done={reset_on_done}")

    t0 = time.time()
    for step_i in itertools.count():
        if step_i == steps:
            print(f"[run_controller] finished stepping in {time.time() - t0:.1f}s")
            if close_on_finish:
                break
            print("[run_controller] leaving SimulationApp open until the process is stopped.")
        if goal_receiver is not None and goal_receiver.poll(env):
            controller.reset()
        action = controller.act(td)
        td_action = TensorDict({}, batch_size=[env.num_envs], device=env.device)
        td_action.set(("agents", "action"), action.view(env.num_envs, 1, 4))
        td = env.step(td_action)["next"]
        if telemetry_sender is not None:
            telemetry_sender.maybe_send(env, step_i, bt_controller=controller)

        done = td["done"].reshape(env.num_envs)
        if step_i % print_every == 0:
            _print_status(env, step_i, done, controller)
        if reset_on_done and bool(done.any().item()):
            done_ids = done.nonzero().flatten()
            print(
                f"[run_controller] reset step={step_i} done_envs={done_ids.tolist()} "
                f"reasons={getattr(env, '_last_done_reason_rates', {}) or {}}"
            )
            td = reset_finished_envs(env, td, done)
            controller.reset(done_ids)

    sim_app.close()


if __name__ == "__main__":
    main()
