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

python3 scripts/ros2_joy_to_udp.py \
  --topic "${ROS2_JOY_TOPIC:-/joy}" \
  --udp-host "${ROS2_JOY_UDP_HOST:-127.0.0.1}" \
  --udp-port "${ROS2_JOY_UDP_PORT:-15000}" &
JOY_PID=$!

python3 scripts/ros2_udp_telemetry_publisher.py \
  --udp-host "${ROS2_TELEMETRY_UDP_HOST:-127.0.0.1}" \
  --udp-port "${ROS2_TELEMETRY_UDP_PORT:-15010}" \
  --dynamic-label-mode "${ROS2_DYNAMIC_LABEL_MODE:-compact}" \
  --dynamic-label-size "${ROS2_DYNAMIC_LABEL_SIZE:-0.16}" \
  --dynamic-label-alpha "${ROS2_DYNAMIC_LABEL_ALPHA:-0.65}" \
  --dynamic-arrow-scale "${ROS2_DYNAMIC_ARROW_SCALE:-0.70}" &
TELEMETRY_PID=$!

python3 scripts/ros2_goal_to_udp.py \
  --topic "${ROS2_GOAL_TOPIC:-/goal_pose}" \
  --udp-host "${ROS2_GOAL_UDP_HOST:-127.0.0.1}" \
  --udp-port "${ROS2_GOAL_UDP_PORT:-15020}" \
  --force-z "${ROS2_GOAL_FORCE_Z:--0.8}" &
GOAL_PID=$!

cleanup() {
  kill "${JOY_PID}" "${TELEMETRY_PID}" "${GOAL_PID}" 2>/dev/null || true
  wait "${JOY_PID}" "${TELEMETRY_PID}" "${GOAL_PID}" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

echo "[ros2_bridges] joystick bridge PID=${JOY_PID}, telemetry bridge PID=${TELEMETRY_PID}, goal bridge PID=${GOAL_PID}"
echo "[ros2_bridges] publishing topics after Isaac starts sending telemetry"
wait -n "${JOY_PID}" "${TELEMETRY_PID}" "${GOAL_PID}"
