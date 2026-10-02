#!/usr/bin/env bash
# Train a policy with PPO: scripts/train_policy.py, preset cfg/train/sonar_goal_ppo.yaml.
#
# Checkpoints and metrics.jsonl go to experiments/policies/sonar_goal/<timestamp>/;
# run a checkpoint with release/scripts/03_run_controller.sh. Extra Hydra overrides
# apply on top of the preset, e.g.:
#   NUM_ENVS=32 bash release/scripts/02_train_policy.sh algo.lr=1e-4
#
# Weights & Biases: online mode needs WANDB_API_KEY; WANDB_MODE=offline needs no
# account (sync later with `wandb sync`), and WANDB_ENABLE=false turns it off.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
ISAAC_PY="${ISAAC_PY:-/home/eth11/isaacsim51/python.sh}"

cd "${REPO_ROOT}"

# W&B reads the key from the environment; naming it here also makes run-container.sh pass it in.
if [ -n "${WANDB_API_KEY:-}" ]; then export WANDB_API_KEY; fi

exec "${ISAAC_PY}" scripts/train_policy.py \
  --config-name="${TRAIN_PRESET:-train/sonar_goal_ppo}" \
  headless="${HEADLESS:-true}" \
  task.env.num_envs="${NUM_ENVS:-16}" \
  train_iters="${TRAIN_ITERS:-2000}" \
  wandb.enable="${WANDB_ENABLE:-true}" \
  wandb.mode="${WANDB_MODE:-online}" \
  "$@"
