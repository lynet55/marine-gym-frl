#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
ISAAC_PY="${ISAAC_PY:-/home/eth11/isaacsim51/python.sh}"

cd "${REPO_ROOT}"

exec "${ISAAC_PY}" scripts/debug_open_hover_vel_env.py \
  headless="${HEADLESS:-false}" \
  enable_livestream="${ENABLE_LIVESTREAM:-false}" \
  mode=play \
  task.env.num_envs="${NUM_ENVS:-1}" \
  debug_launch.action_mode="${ACTION_MODE:-hold}" \
  debug_launch.action_scale="${ACTION_SCALE:-0.18}" \
  debug_launch.steps="${STEPS:-20000}" \
  debug_launch.print_every="${PRINT_EVERY:-60}" \
  debug_launch.close_on_finish="${CLOSE_ON_FINISH:-false}" \
  "$@"
