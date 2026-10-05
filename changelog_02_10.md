# Changelog — 2 October 2026

Policy training, one entry point that runs either a trained policy or the behavior
tree, and Weights & Biases logging with viewport renders. The existing Python
scripts, the env code in `src/` and the `marinegym` submodule are unchanged.

## New files

| File | Purpose |
|---|---|
| `scripts/run_controller.py` | Entry point: builds the env from a run preset and steps it with the controller the config names. Reuses the UDP goal/telemetry bridges of `debug_open_hover_vel_env.py`, so RViz works as before. |
| `scripts/controllers.py` | `Controller` interface (`act`, `reset`, `status`); `PolicyController`; `BehaviorTreeController`, which steps `RobustMazeBTController` exactly as `action_mode=bt_auto` does; the checkpoint contract check; `reset_finished_envs`. |
| `scripts/train_policy.py` | PPO training with MarineGym's `PPOPolicy` and a hand-written rollout loop. Writes checkpoints, `metrics.jsonl` and `config.yaml`; logs to W&B with a viewport GIF every `wandb.render_every` iterations. |
| `cfg/policy/sonar_goal.yaml` | Observation/action contract shared by training and running: 30 observations (goal and state, 16 sonar sectors, front minimum, front TTC risk), 64 sonar rays, speed limits. |
| `cfg/controller/policy.yaml`, `cfg/controller/bt.yaml` | One file per controller; `target` is the class. `controller=bt` swaps it. |
| `cfg/run/point_a_to_b.yaml` | The environment to run in: the `bt_auto_point_a_to_b` scene plus the controller. |
| `cfg/train/sonar_goal_ppo.yaml` | Training scene (random start and goal at -0.8 m, walls and moving obstacles), PPO, W&B and viewport camera settings. |
| `release/scripts/02_train_policy.sh` | Training launcher (`NUM_ENVS`, `TRAIN_ITERS`, `WANDB_ENABLE`, `WANDB_MODE`, `WANDB_API_KEY`). |
| `release/scripts/03_run_controller.sh` | Run launcher (`CONTROLLER`, `CHECKPOINT`, `RUN_PRESET`, `RESET_ON_DONE`, plus the usual ROS 2 settings). |
| `changelog_02_10.md` | This file. |

## Edited files

| File | Change |
|---|---|
| `README.md` | New section "Training and running a policy": launchers, contract, controller group, W&B. |
| `container/sim/Dockerfile` | Installs `wandb` (a native install already has it through MarineGym's `setup.py`). |

Outside git: the local image `localhost/simenvbtuuv-sim` was rebuilt with `wandb`
(42 s; the previous image is kept untagged). Nothing is committed.

## Why `cfg/policy/sonar_goal.yaml` exists

Found while reading the configs: before these changes, training (the
`train_bluerov_isaaclab_hover_vel_distill` config) and playback (the debug
presets) disagreed on settings that change what a policy sees or does:

- Speed limits: training inherited 0.7/0.55/0.55 from `cfg/task`, playback used 0.45/0.30/0.25.
- `observation.use_path_lookahead: true` in both, which puts planner waypoints into the observation and needs the occupancy map running.
- With mapping on, the sonar fallbacks clamp no-hit rays to `mapping.max_range` (10 m) instead of the sonar's 12 m, so turning mapping off for faster training changed the sonar observation.

The contract pins all of these. Checkpoints store it, and a run refuses a checkpoint trained with different values.

## What the test runs showed

All runs were headless in the sim container (Isaac Sim 5.1, RTX 5090), with their output outside the repo.

### Training

- Throughput: about 110 samples/s with 2 envs and 270–307 with the default 16. The default 2000 iterations (2M samples) take about 2 hours.
- Finished episodes are counted, reset and logged correctly (tested with `max_episode_length=30`).
- In the 16-env run, iteration 0 already counted a success from random actions: start and goal may spawn 3 m apart and the success radius is 1 m, so early success rates include luck.
- Checkpoints are 3.4 MB and load back exactly (strict, under torch 2.7's `weights_only` default).
- Isaac Sim takes 30–40 s to start with warm caches.

### Running controllers

- Policy (from a 2-iteration training run): passed the contract check and ran 300 steps in 9.1 s with full telemetry. It only drifted, as expected this early.
- A goal sent over UDP, as the RViz goal bridge does, replaced the fixed goal (goal distance 14.4 → 7.7 m). Telemetry carried the new goal: 310 packets, 64 sonar rays, `range_max` 12.
- Behavior tree: the composed config is identical to `cfg/debug/bt_auto_point_a_to_b.yaml` apart from the `controller` block. It moved toward the goal (x 3.0 → 4.8 m in 300 steps), switched zones (move → dynamic_go), and its dynamic-obstacle estimates still reached the telemetry. 300 steps took 14.5 s (256 sonar rays plus the BT logic).
- A checkpoint run with different settings (`task.sonar.num_rays=256`) is refused with the list of differences and exit code 1; `controller.strict=false` warns and runs.

### W&B and viewport renders

What training sends to W&B when `wandb.enable` is on (the default):

- Every iteration: reward, samples/s, the number of finished episodes and the PPO statistics (losses, entropy, gradient norms, explained variance). Success rate, collision rate and episode return appear on iterations where an episode ended. The full resolved config is stored with the run.
- Every 25 iterations (`wandb.render_every`): a GIF of that iteration's rollout from the Isaac Sim viewport, under `viewport` in W&B and in `<run>/renders/`.
- Online mode is the default and needs `WANDB_API_KEY`, which the launcher passes into the container. `WANDB_MODE=offline` needs no account; upload later with `$ISAAC_PY -m wandb sync <run dir>/wandb/offline-run-*`. `WANDB_ENABLE=false` turns W&B off.

What the runs showed (offline mode):

- MarineGym's `init_wandb` (`marinegym/marinegym/utils/wandb.py`) fails with current wandb (0.30.0) because `wandb.util.generate_id` no longer exists. `train_policy.py` calls `wandb.init` directly with the same settings.
- Installing wandb into Isaac Sim's Python adds protobuf 7.36 and opentelemetry packages and upgrades click from 8.1.7 to 8.5.0. Isaac Sim, the env and torch 2.7.0 are unaffected; the Dockerfile's torch check passes.
- Headless viewport renders were black. A headless stage has no light: the GUI viewport brings its own, and MarineGym's underwater fill light only exists with `task.visual_water.enable`, never headless. Training now adds a dome light when it records; lights don't affect physics or the sonar. Frames taken from the GIFs afterwards show the whole tank, the maze walls, the moving obstacles and the ROV.
- A recorded iteration ran at about 41 samples/s instead of 105 (2 envs), about 2.5× slower. Recording only every 25 iterations keeps the overall cost small.
- A GIF at the default 64 steps has 32 frames (2.6 s, real-time speed) and is about 2.4 MB. With `render_every: 25`, a 2000-iteration run makes 80 of them, about 200 MB.
- The camera shows the env nearest the world origin. With `env_spacing` 20, the neighbouring tank shows at the frame edge.

## Not tested yet

- Online W&B upload: there was no API key, so only offline mode ran.
- ROS 2 and RViz themselves: the runs checked the UDP packets the bridges consume, not the ROS topics.
- A policy that solves the task: the training runs were a few iterations long, to check the pipeline.

## Issues in existing code (not changed)

- `debug_open_hover_vel_env.py` resets finished envs with `{"_reset": done}` only. With `num_envs > 1`, torchrl's partial-reset merge sets the observation of every env that wasn't reset to zero. Confirmed in a run with 2 envs; single-env runs are unaffected. The new scripts pass the current observations in (`reset_finished_envs`).
- MarineGym's PPO actor returns its mean action unless sampling is requested (tensordict 0.7.2 default), so training has to run under `ExplorationType.RANDOM`. Confirmed in a unit test.
- Isaac Sim exits through `os._exit` (fast shutdown), which drops block-buffered output when stdout is a pipe or a file. Seen in a run; the new scripts line-buffer stdout, but the existing ones still lose output when piped.
- An unhandled exception also starts Kit's crash reporter (py-spy dump, nvidia-smi log) after the traceback. The exit code stays 1.
- Kit logs all Python stderr as `[Error] [omni.kit.app._impl] [py stderr]`, so warnings look like errors. The ones seen are harmless:
  - torchrl's "Failed to import torchrl C++ binaries";
  - DeprecationWarnings that `env.action_spec` and `env.reward_spec` return the full spec from torchrl 0.8. `PPOPolicy` relies on the current behaviour, and torchrl is pinned to 0.7.2.
- `cfg/task/hover_vel_stable_goal.yaml` indents its reward settings (`success_distance`, `reward_alive`, … `terminate_on_ttc`) under `goal:`, where the env never reads them. Seen in the composed config. The training config sets them at task level, so the intended values apply there.
- From reading the code:
  - `HoverVelStableGoal` declares its observation spec as `(num_envs, D)` but returns `(num_envs, 1, D)`. The hand-written loops don't care; torchrl's collectors would.
  - The sonar is a Python loop over envs × rays (`src/bluerov_manual/navigation/sonar.py`), the main cost of a training step; hence 64 rays in the contract.

## Defaults worth revisiting

In `cfg/policy/sonar_goal.yaml` (changing these means retraining, and runs refuse older checkpoints):

- 64 sonar rays instead of the BT's 256, because the sonar is the slowest part of a training step.
- Speed limits 0.45/0.30/0.25 m/s, as in the BT runs, rather than the task default 0.7/0.55/0.55.

In `cfg/train/sonar_goal_ppo.yaml`:

- Reward: the env's goal-progress reward with the occupancy map off. The path-following reward needs the map and is slower.
- `wandb.render_every: 25`: lower it for more GIFs, at about 2.4 MB each and a slower recorded iteration.

## Additions, 3 October 2026

### Changes

| File | Change |
|---|---|
| `cfg/policy/sonar_goal.yaml` | Pins the sonar hit filters (`hit_path_allow`, `hit_path_deny`, `filter_by_env`, `ignore_empty_path`). Training and the BT presets set the same values, each on its own; the code defaults (`sonar.py`) drop valid colliders whose PhysX hit path is empty. |
| `scripts/controllers.py` | The four filter keys join `CONTRACT_KEYS`. Checkpoints from before this change don't store them and are refused (`controller.strict=false` runs them anyway). |
| `cfg/train/sonar_goal_ppo.yaml` | `wandb.entity`, and `wandb.api_key` (null uses `WANDB_API_KEY`). The file is tracked by git: keep a key out of commits (`git update-index --skip-worktree`). |
| `scripts/train_policy.py` | A key in `wandb.api_key` goes to wandb through the environment and is cleared from the config, so it stays out of `config.yaml` and the W&B run config. Hydra's own `.hydra/config.yaml` still holds it. |

Online W&B now tested: 30 iterations with 2 envs uploaded the per-iteration metrics and three viewport GIFs to `bbalinov-ntnu/bluerov_sonar_goal`.

### Training settings that are inherited, not used

`sonar_goal_ppo.yaml` loads `train_bluerov_isaaclab_hover_vel_distill.yaml`, so the composed config (and the W&B run config) contains `rsl`, `a2c`, `sac`, `distill_bt` and `world_model`. `train_policy.py` reads none of them: the algorithm settings are `algo` (MarineGym's `PPOConfig`). The network is fixed in `PPOPolicy` (`marinegym/learning/ppo/ppo.py`): separate actor and critic MLPs [256, 256, 256]. MarineGym has no SAC. Training starts from random weights unless `resume_checkpoint` is set.

### How the reward reaches training

`task.name: HoverVelStableGoal` (`cfg/task/hover_vel_stable_goal.yaml`) → `import bluerov_manual.envs` registers the class in `IsaacEnv.REGISTRY` → `train_policy.py` builds `IsaacEnv.REGISTRY[cfg.task.name]` → each `env.step` runs `IsaacEnv._step`, which calls `HoverVelStableGoal._compute_reward_and_done` (`src/bluerov_manual/envs/hover_vel_stable_goal.py:2123`). The reward lands in `("next", "agents", "reward")`; PPO's loss (clipped policy loss, entropy, Huber value loss) sees goal and sonar only through the reward (advantages, returns) and the observation.

Class chain: `HoverVelStableGoal` → `HoverVel` (sonar, velocity controller) → MarineGym `Hover` → `IsaacEnv`.

### The reward as trained

With d the goal distance, ψ_g − ψ the heading error to the goal, c the minimum sonar range over all 64 rays and v_x the forward speed (weights marked * are code defaults, not set in any config):

r = −0.01 + 5.0* · clip(d_{t−1} − d_t, ±2) − 0.25* · d + 0.2* · cos(ψ_g − ψ) + 0.5* · upright − 0.05* · tilt − 0.01* · effort − 1.0 · max(0, ‖v‖ − 0.7)² − 0.1 · max(0, 0.9 − c) − 0.3 · clip(1 − (c / max(v_x, 0.05)) / 3, 0, 1)

clamped to ±20, then +10* on success (d ≤ 1 m for 10* steps) and −5* on a failed termination: collision (c ≤ 0.08 m), TTC (c / v_x ≤ 0.12 s) or leaving the tank. Path following is off (no occupancy map), so d and the heading use the straight line to the goal.

### Issues found (not changed)

- The clearance c is sonar only. `_compute_reward_and_done` also calls `_obstacle_clearance` (ground-truth obstacle distance), but only `HoverTankPPO` defines it; the `AttributeError` is swallowed by `except Exception: pass` (line 2208). Moving obstacles count when the sonar sees them (`dynamic_obstacle_sonar_fallback`).
- The reward's TTC uses c over 360° (`front_min = min_clear`, line 2221), the observation's front TTC only ±30°. A wall beside or behind the robot penalizes, and can terminate, forward motion.
- `reward_effort_weight`, `reward_distance_scale` and `reward_action_smoothness_weight` are in the composed config but the env reads other names (e.g. `reward_effort_penalty_weight`); they have no effect.
- The shaping is small next to the progress term and the −5 failure penalty: avoidance is learned mostly from episodes ending.
