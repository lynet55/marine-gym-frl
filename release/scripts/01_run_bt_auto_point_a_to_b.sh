#!/usr/bin/env bash
# BT Autonomous, Point A to B — short entrypoint.
#
# All scenario tuning (start/goal, maze, obstacles, BT parameters) lives in
# cfg/debug/bt_auto_point_a_to_b.yaml — edit that file to change it.
#
# Run-mode env vars below (headless, telemetry, step count) keep their usual
# override behavior. Any extra Hydra overrides you pass on the command line
# still apply on top of the preset, e.g.:
#   bash release/scripts/01_run_bt_auto_point_a_to_b.sh task.obstacles.num=5
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
ISAAC_PY="${ISAAC_PY:-/home/eth11/isaacsim51/python.sh}"

cd "${REPO_ROOT}"

exec "${ISAAC_PY}" scripts/debug_open_hover_vel_env.py \
  --config-name=debug/bt_auto_point_a_to_b \
  headless="${HEADLESS:-false}" \
  enable_livestream="${ENABLE_LIVESTREAM:-false}" \
  mode=play \
  task.env.num_envs="${NUM_ENVS:-1}" \
  debug_launch.action_mode=bt_auto \
  debug_launch.steps="${STEPS:-30000}" \
  debug_launch.print_every="${PRINT_EVERY:-60}" \
  debug_launch.close_on_finish="${CLOSE_ON_FINISH:-false}" \
  debug_launch.ros2_telemetry.enable="${ROS2_TELEMETRY_ENABLE:-true}" \
  debug_launch.ros2_telemetry.send_rgb="${ROS2_TELEMETRY_SEND_RGB:-true}" \
  debug_launch.ros2_telemetry.send_depth="${ROS2_TELEMETRY_SEND_DEPTH:-true}" \
  debug_launch.ros2_telemetry.publish_every="${ROS2_TELEMETRY_PUBLISH_EVERY:-1}" \
  debug_launch.ros2_telemetry.map_every="${ROS2_TELEMETRY_MAP_EVERY:-1}" \
  debug_launch.ros2_goal.enable="${ROS2_GOAL_ENABLE:-true}" \
  debug_launch.ros2_goal.udp_host="${ROS2_GOAL_UDP_HOST:-127.0.0.1}" \
  debug_launch.ros2_goal.udp_port="${ROS2_GOAL_UDP_PORT:-15020}" \
  "$@"
