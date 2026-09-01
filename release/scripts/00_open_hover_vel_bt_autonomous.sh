#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export ACTION_MODE="${ACTION_MODE:-bt_auto}"
export NUM_ENVS="${NUM_ENVS:-1}"
export ROS2_TELEMETRY_ENABLE="${ROS2_TELEMETRY_ENABLE:-true}"

exec "${SCRIPT_DIR}/00_open_hover_vel_ros2_joystick.sh" \
  debug_launch.ros2_goal.enable="${ROS2_GOAL_ENABLE:-true}" \
  debug_launch.ros2_goal.udp_host="${ROS2_GOAL_UDP_HOST:-127.0.0.1}" \
  debug_launch.ros2_goal.udp_port="${ROS2_GOAL_UDP_PORT:-15020}" \
  "$@"
