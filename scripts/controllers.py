"""Controllers that drive the BlueROV env from scripts/run_controller.py.

A controller turns the env's latest tensordict into an action of shape [num_envs, 4]
in [-1, 1] (surge, sway, heave, yaw rate). The config picks the class by import path
(``controller.target`` in cfg/controller/*.yaml), so adding one, e.g. another behavior
tree, is a Controller subclass plus a yaml file; run_controller.py stays unchanged.

This module also owns the policy checkpoint format: scripts/train_policy.py stores the
observation/action contract (CONTRACT_KEYS) in every checkpoint, and PolicyController
refuses to run a checkpoint in an env that is configured differently.
"""

from __future__ import annotations

import torch
from hydra.utils import get_class, to_absolute_path
from omegaconf import OmegaConf
from tensordict import TensorDictBase

# Config values that decide what a policy observes and what its actions mean.
CONTRACT_KEYS = (
    "task.name",
    "task.observation",
    "task.sonar.num_rays",
    "task.sonar.fov_deg",
    "task.sonar.max_range",
    "task.sonar.min_range_filter",
    "task.sonar.origin_offset",
    "task.sonar.hit_path_allow",
    "task.sonar.hit_path_deny",
    "task.sonar.filter_by_env",
    "task.sonar.ignore_empty_path",
    "task.ttc_horizon",
    "task.ttc_min_forward_speed",
    "task.controller",
    "task.mapping.max_range",
    "task.mapping.maze_wall_sonar_fallback",
    "task.mapping.dynamic_obstacle_sonar_fallback",
    "task.sim.dt",
    "task.sim.substeps",
)


class Controller:
    """What run_controller.py calls. Subclasses implement act() and, if they keep state, reset()."""

    def __init__(self, cfg, env) -> None:
        self.cfg = cfg
        self.env = env

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        """Drop per-episode state for ``env_ids``; None means all envs (also sent on a new goal)."""

    def act(self, td: TensorDictBase) -> torch.Tensor:
        """Return the action for every env: shape [num_envs, 4], values in [-1, 1]."""
        raise NotImplementedError

    def status(self) -> str:
        """Extra text for the periodic status line."""
        return ""


def make_controller(cfg, env) -> Controller:
    """Build the controller named by ``cfg.target``, e.g. controllers.PolicyController."""
    return get_class(cfg.target)(cfg, env)


class PolicyController(Controller):
    """A PPO policy from scripts/train_policy.py, acting with its mean action (no sampling)."""

    def __init__(self, cfg, env) -> None:
        super().__init__(cfg, env)
        if not cfg.get("checkpoint"):
            raise ValueError(
                "controller=policy needs controller.checkpoint=<path to .pt> "
                "(CHECKPOINT=... for release/scripts/03_run_controller.sh)"
            )
        path = to_absolute_path(str(cfg.checkpoint))
        ckpt = torch.load(path, map_location=env.device)
        check_contract(ckpt["contract"], policy_contract(env), strict=bool(cfg.get("strict", True)))
        policy = build_ppo_policy(ckpt["algo"], env)
        policy.load_state_dict(ckpt["policy"])
        self.actor = policy.actor.eval()
        print(f"[controllers] policy {path} (iteration {ckpt.get('iteration')})")

    @torch.inference_mode()
    def act(self, td: TensorDictBase) -> torch.Tensor:
        from torchrl.envs.utils import ExplorationType, set_exploration_type

        with set_exploration_type(ExplorationType.DETERMINISTIC):
            out = self.actor(td.select(("agents", "observation")))
        return out["agents", "action"].reshape(self.env.num_envs, -1).clamp(-1.0, 1.0)


class BehaviorTreeController(Controller):
    """The maze behavior tree, stepped exactly as debug_launch.action_mode=bt_auto does."""

    def __init__(self, cfg, env) -> None:
        super().__init__(cfg, env)
        # Both modules live in scripts/, next to the entry points, so they are on sys.path.
        from debug_open_hover_vel_env import _bt_action
        from play_bluerov_behavior_tree_maze import RobustMazeBTController

        self._bt_action = _bt_action
        self.bt = RobustMazeBTController(cfg.bt, env)
        self.reset()

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        if env_ids is None or env_ids.numel() == self.env.num_envs:
            self.bt.reset_internal_state(self.env.num_envs, self.env.device)
        else:
            self.bt.reset_env_state(env_ids)

    def act(self, td: TensorDictBase) -> torch.Tensor:
        return self._bt_action(self.bt, td, self.env)

    def status(self) -> str:
        debug = getattr(self.bt, "_last_debug", None)
        if not debug:
            return ""
        d = debug[0]
        return (
            f"bt_zone={d.get('zone', 'na')} "
            f"front_min={float(d.get('front_min', float('nan'))):.3f} "
            f"final_dist={float(d.get('final_goal_dist', float('nan'))):.3f} "
            f"escape_hold={int(d.get('escape_hold', 0))}"
        )

    def dynamic_estimates_payload(self, env_id: int = 0) -> list[dict]:
        # The telemetry sender looks for this: /bluerov/dynamic_obstacles_estimated in RViz.
        return self.bt.dynamic_estimates_payload(env_id=env_id)


def build_ppo_policy(algo: dict, env):
    """MarineGym's PPO actor-critic for this env, built the same way for training and running."""
    from marinegym.learning.ppo.ppo import PPOConfig, PPOPolicy

    # PPOPolicy loads checkpoint_path itself with strict=False; checkpoints are loaded by the callers.
    ppo_cfg = PPOConfig(**{**algo, "checkpoint_path": None})
    return PPOPolicy(ppo_cfg, env.observation_spec, env.action_spec, env.reward_spec, device=env.device)


def policy_contract(env) -> dict:
    """The env's values for CONTRACT_KEYS, plus its observation and action sizes."""
    contract = {}
    for key in CONTRACT_KEYS:
        value = OmegaConf.select(env.cfg, key, default=None)
        contract[key] = OmegaConf.to_container(value, resolve=True) if OmegaConf.is_config(value) else value
    contract["obs_dim"] = int(env.observation_spec["agents", "observation"].shape[-1])
    contract["action_dim"] = int(env.action_spec.shape[-1])
    return contract


def check_contract(trained: dict, current: dict, strict: bool = True) -> None:
    """Raise (or only warn, if not strict) when a checkpoint's contract differs from the env's."""
    diff = [key for key in sorted(set(trained) | set(current)) if trained.get(key) != current.get(key)]
    if not diff:
        return
    lines = "\n".join(f"  {key}: trained={trained.get(key)!r} now={current.get(key)!r}" for key in diff)
    message = f"the policy was trained with different observation/action settings:\n{lines}"
    if strict:
        raise ValueError(f"{message}\nMatch the env config (cfg/policy/), or pass controller.strict=false to run anyway.")
    print(f"[controllers] WARNING {message}")


def reset_finished_envs(env, td: TensorDictBase, done: torch.Tensor) -> TensorDictBase:
    """Reset the envs where ``done`` [num_envs] is True and return the next observation tensordict.

    The running envs' observations go in with the reset request: for envs that are not
    reset, torchrl's partial-reset merge keeps the input's values and fills keys missing
    from the input with zeros.
    """
    td_in = td.select(("agents", "observation")).clone()
    td_in.set("_reset", done.reshape(env.num_envs, 1))
    return env.reset(td_in).select(("agents", "observation"))
