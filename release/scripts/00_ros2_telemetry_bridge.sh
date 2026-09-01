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

exec python3 scripts/ros2_udp_telemetry_publisher.py \
  --udp-host "${ROS2_TELEMETRY_UDP_HOST:-127.0.0.1}" \
  --udp-port "${ROS2_TELEMETRY_UDP_PORT:-15010}" \
  --dynamic-label-mode "${ROS2_DYNAMIC_LABEL_MODE:-compact}" \
  --dynamic-label-size "${ROS2_DYNAMIC_LABEL_SIZE:-0.16}" \
  --dynamic-label-alpha "${ROS2_DYNAMIC_LABEL_ALPHA:-0.65}" \
  --dynamic-arrow-scale "${ROS2_DYNAMIC_ARROW_SCALE:-0.70}" \
  "$@"
