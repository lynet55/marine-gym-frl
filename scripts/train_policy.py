"""Train a policy for the BlueROV env with PPO (MarineGym's PPOPolicy).

Launch through release/scripts/02_train_policy.sh; the preset is cfg/train/sonar_goal_ppo.yaml.
Checkpoints store the observation/action contract (see scripts/controllers.py), which
scripts/run_controller.py checks before it runs the policy. With wandb.enable, the
per-iteration metrics also go to Weights & Biases, together with a GIF of the Isaac Sim
viewport every wandb.render_every iterations.
"""

from __future__ import annotations

import json
import os
import sys
import time

import hydra
import torch
from hydra.utils import to_absolute_path
from omegaconf import OmegaConf

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from controllers import build_ppo_policy, check_contract, policy_contract, reset_finished_envs  # noqa: E402

# Entries PPOPolicy.train_op reads from a rollout.
ROLLOUT_KEYS = (
    ("agents", "observation"),
    ("agents", "action"),
    "sample_log_prob",
    "state_value",
    ("next", "agents", "observation"),
    ("next", "agents", "reward"),
    ("next", "terminated"),
    ("next", "truncated"),
)
# Env steps between the viewport frames of a W&B render.
RENDER_STRIDE = 2


def _init_wandb(cfg, out_dir: str):
    # Same settings as marinegym.utils.wandb.init_wandb, which calls wandb.util.generate_id,
    # removed in current wandb releases.
    import wandb

    wandb_cfg = cfg.wandb
    run_id = wandb_cfg.get("run_id")
    return wandb.init(
        project=wandb_cfg.project,
        entity=wandb_cfg.get("entity"),
        group=wandb_cfg.get("group"),
        name=f"{wandb_cfg.get('run_name') or 'run'}/{os.path.basename(out_dir)}",
        tags=list(wandb_cfg.get("tags") or []),
        mode=wandb_cfg.get("mode", "online"),
        id=run_id,
        resume="must" if run_id else None,
        config=OmegaConf.to_container(cfg, resolve=True),
        # Keep W&B's local files next to the checkpoints, and leave stdout alone (Kit captures it too).
        dir=out_dir,
        settings=wandb.Settings(console="off"),
    )


def _add_render_light() -> None:
    # A headless stage has no light (the GUI viewport brings its own), so renders come out black.
    from omni.isaac.core.utils.stage import get_current_stage
    from pxr import UsdLux

    UsdLux.DomeLight.Define(get_current_stage(), "/World/RenderLight").CreateIntensityAttr().Set(1000.0)


def _save_gif(frames: list, path: str, frame_dt: float) -> None:
    from PIL import Image

    os.makedirs(os.path.dirname(path), exist_ok=True)
    images = [Image.fromarray(frame) for frame in frames]
    images[0].save(path, save_all=True, append_images=images[1:], duration=round(1000 * frame_dt), loop=0)


def _wandb_log(run, stats: dict, frames: list, gif_path: str, frame_dt: float) -> None:
    import wandb

    log = {key: value for key, value in stats.items() if value is not None}
    if frames:
        _save_gif(frames, gif_path, frame_dt)
        log["viewport"] = wandb.Video(gif_path)
    run.log(log, step=stats["iteration"])


@hydra.main(version_base=None, config_path="../cfg", config_name="train/sonar_goal_ppo")
def main(cfg) -> None:
    # Isaac Sim exits through os._exit (fast shutdown), which drops block-buffered output
    # when stdout is a pipe or a file, so flush every line.
    sys.stdout.reconfigure(line_buffering=True)
    OmegaConf.register_new_resolver("eval", eval)
    OmegaConf.resolve(cfg)
    OmegaConf.set_struct(cfg, False)
    # A key from the config goes to wandb through the environment and is cleared, so it stays
    # out of config.yaml and the W&B run config.
    if cfg.wandb.get("api_key"):
        os.environ["WANDB_API_KEY"] = str(cfg.wandb.api_key)
        cfg.wandb.api_key = None

    from bluerov_manual.sim_app import init_app

    sim_app = init_app(cfg)

    import bluerov_manual.envs  # noqa: F401
    from marinegym.envs.isaac_env import IsaacEnv
    from torchrl.envs.utils import ExplorationType, set_exploration_type

    env = IsaacEnv.REGISTRY[cfg.task.name](cfg, headless=cfg.headless)
    env.set_seed(int(cfg.get("seed", 0)))

    algo = OmegaConf.to_container(cfg.algo, resolve=True)
    policy = build_ppo_policy(algo, env)
    contract = policy_contract(env)
    start_iter = 0
    if cfg.get("resume_checkpoint"):
        ckpt = torch.load(to_absolute_path(str(cfg.resume_checkpoint)), map_location=env.device)
        check_contract(ckpt["contract"], contract)
        policy.load_state_dict(ckpt["policy"])
        if bool(cfg.get("resume_optimizer", True)):
            policy.actor_opt.load_state_dict(ckpt["actor_opt"])
            policy.critic_opt.load_state_dict(ckpt["critic_opt"])
        start_iter = int(ckpt["iteration"]) + 1

    out_dir = to_absolute_path(str(cfg.checkpoint_dir))
    os.makedirs(out_dir, exist_ok=True)
    OmegaConf.save(cfg, os.path.join(out_dir, "config.yaml"))
    metrics_file = open(os.path.join(out_dir, "metrics.jsonl"), "a")
    print(f"[train] obs_dim={contract['obs_dim']} num_envs={env.num_envs} output={out_dir}")

    run = _init_wandb(cfg, out_dir) if bool(cfg.wandb.get("enable", False)) else None
    render_every = int(cfg.wandb.get("render_every", 0)) if run is not None else 0
    default_render = (not cfg.headless) or bool(cfg.get("enable_livestream", False))
    frame_dt = float(cfg.sim.dt) * int(cfg.sim.substeps) * RENDER_STRIDE
    if render_every > 0 and cfg.headless:
        _add_render_light()

    def save(name: str, iteration: int) -> None:
        path = os.path.join(out_dir, name)
        torch.save(
            {
                "policy": policy.state_dict(),
                "actor_opt": policy.actor_opt.state_dict(),
                "critic_opt": policy.critic_opt.state_dict(),
                "algo": algo,
                "contract": contract,
                "iteration": iteration,
            },
            path,
        )
        print(f"[train] saved {path}")

    td = env.reset().select(("agents", "observation"))
    iteration = start_iter - 1
    for iteration in range(start_iter, int(cfg.train_iters)):
        t0 = time.perf_counter()
        steps, episode_returns, frames = [], [], []
        episodes = successes = collisions = 0
        # Headless runs don't render; switch the viewport on for the iterations that are recorded.
        record = render_every > 0 and iteration % render_every == 0
        if record:
            env.enable_render(True)
        # The actor only samples under RANDOM; its default interaction type is deterministic.
        with torch.no_grad(), set_exploration_type(ExplorationType.RANDOM):
            for step in range(int(cfg.steps_per_iter)):
                td = env.step(policy(td))
                if record and step % RENDER_STRIDE == 0:
                    frames.append(env.render(mode="rgb_array").copy())
                steps.append(td.select(*ROLLOUT_KEYS))
                next_td = td["next"]
                done = next_td["done"].reshape(env.num_envs)
                if bool(done.any().item()):
                    reasons = next_td["termination_reason"]
                    episodes += int(done.sum().item())
                    successes += int(reasons["success"].sum().item())
                    collisions += int((reasons["collision"] | reasons["ttc_collision"]).sum().item())
                    episode_returns += next_td["stats", "return"][done].flatten().tolist()
                    td = reset_finished_envs(env, next_td, done)
                else:
                    td = next_td.select(("agents", "observation"))
        if record:
            env.enable_render(default_render)
        rollout = torch.stack(steps, dim=1)  # [num_envs, steps]: PPOPolicy's GAE runs along dim 1
        collect_s = time.perf_counter() - t0
        info = policy.train_op(rollout)

        stats = {
            "iteration": iteration,
            "samples_per_s": rollout.numel() / collect_s,
            "reward": float(rollout["next", "agents", "reward"].mean().item()),
            "episodes": episodes,
            "success_rate": successes / episodes if episodes else None,
            "collision_rate": collisions / episodes if episodes else None,
            "episode_return": sum(episode_returns) / len(episode_returns) if episode_returns else None,
            **info,
        }
        metrics_file.write(json.dumps(stats) + "\n")
        metrics_file.flush()
        print("[train] " + " ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}" for k, v in stats.items()))
        if run is not None:
            gif_path = os.path.join(out_dir, "renders", f"iteration_{iteration:05d}.gif")
            _wandb_log(run, stats, frames, gif_path, frame_dt)
        if (iteration + 1) % int(cfg.save_interval) == 0:
            save(f"policy_{iteration + 1:05d}.pt", iteration)

    save("policy_final.pt", iteration)
    metrics_file.close()
    if run is not None:
        run.finish()  # before sim_app.close(), which ends the process without Python's exit hooks
    sim_app.close()


if __name__ == "__main__":
    main()
