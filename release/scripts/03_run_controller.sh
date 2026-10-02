#!/usr/bin/env bash
# Run a controller in an environment: scripts/run_controller.py.
#
#   CHECKPOINT=experiments/policies/sonar_goal/<timestamp>/policy_final.pt \
#   bash release/scripts/03_run_controller.sh                  # trained policy (default)
#   CONTROLLER=bt bash release/scripts/03_run_controller.sh    # behavior tree
#
# CONTROLLER picks cfg/controller/<name>.yaml and RUN_PRESET the environment,
# cfg/run/<name>.yaml. Start release/scripts/00_ros2_bridges.sh and RViz first to
# watch it, and use RViz's "2D Goal Pose" for a new goal (RESET_ON_DONE=false keeps
# that goal when the episode ends). Extra Hydra overrides apply on top.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
ISAAC_PY="${ISAAC_PY:-/home/eth11/isaacsim51/python.sh}"

cd "${REPO_ROOT}"

# Only the policy controller has a checkpoint setting.
CHECKPOINT_ARGS=()
if [ -n "${CHECKPOINT:-}" ]; then
  CHECKPOINT_ARGS+=("controller.checkpoint=${CHECKPOINT}")
fi

exec "${ISAAC_PY}" scripts/run_controller.py \
  --config-name="${RUN_PRESET:-run/point_a_to_b}" \
  controller="${CONTROLLER:-policy}" \
  headless="${HEADLESS:-false}" \
  enable_livestream="${ENABLE_LIVESTREAM:-false}" \
  mode=play \
  task.env.num_envs="${NUM_ENVS:-1}" \
  debug_launch.steps="${STEPS:-30000}" \
  debug_launch.print_every="${PRINT_EVERY:-60}" \
  debug_launch.reset_on_done="${RESET_ON_DONE:-true}" \
  debug_launch.close_on_finish="${CLOSE_ON_FINISH:-false}" \
  debug_launch.ros2_telemetry.enable="${ROS2_TELEMETRY_ENABLE:-true}" \
  debug_launch.ros2_telemetry.send_rgb="${ROS2_TELEMETRY_SEND_RGB:-true}" \
  debug_launch.ros2_telemetry.send_depth="${ROS2_TELEMETRY_SEND_DEPTH:-true}" \
  debug_launch.ros2_telemetry.publish_every="${ROS2_TELEMETRY_PUBLISH_EVERY:-1}" \
  debug_launch.ros2_telemetry.map_every="${ROS2_TELEMETRY_MAP_EVERY:-1}" \
  debug_launch.ros2_goal.enable="${ROS2_GOAL_ENABLE:-true}" \
  debug_launch.ros2_goal.udp_host="${ROS2_GOAL_UDP_HOST:-127.0.0.1}" \
  debug_launch.ros2_goal.udp_port="${ROS2_GOAL_UDP_PORT:-15020}" \
  "${CHECKPOINT_ARGS[@]}" \
  "$@"
