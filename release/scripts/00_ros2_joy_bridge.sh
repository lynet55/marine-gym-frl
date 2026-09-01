#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

set +u
source /opt/ros/jazzy/setup.bash
set -u
mkdir -p "${REPO_ROOT}/experiments/ros2_logs"
export ROS_LOG_DIR="${ROS_LOG_DIR:-${REPO_ROOT}/experiments/ros2_logs}"

cd "${REPO_ROOT}"

exec python3 scripts/ros2_joy_to_udp.py \
  --topic "${ROS2_JOY_TOPIC:-/joy}" \
  --udp-host "${ROS2_JOY_UDP_HOST:-127.0.0.1}" \
  --udp-port "${ROS2_JOY_UDP_PORT:-15000}" \
  "$@"
