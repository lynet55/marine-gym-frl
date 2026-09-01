#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export ACTION_MODE="${ACTION_MODE:-ros2_joy}"
export NUM_ENVS="${NUM_ENVS:-1}"

exec "${SCRIPT_DIR}/00_open_hover_vel_environment.sh" \
  debug_launch.ros2_joy.udp_host="${ROS2_JOY_UDP_HOST:-127.0.0.1}" \
  debug_launch.ros2_joy.udp_port="${ROS2_JOY_UDP_PORT:-15000}" \
  debug_launch.ros2_joy.timeout_sec="${ROS2_JOY_TIMEOUT_SEC:-0.5}" \
  debug_launch.ros2_telemetry.enable="${ROS2_TELEMETRY_ENABLE:-true}" \
  debug_launch.ros2_telemetry.udp_host="${ROS2_TELEMETRY_UDP_HOST:-127.0.0.1}" \
  debug_launch.ros2_telemetry.udp_port="${ROS2_TELEMETRY_UDP_PORT:-15010}" \
  debug_launch.ros2_telemetry.publish_every="${ROS2_TELEMETRY_PUBLISH_EVERY:-3}" \
  debug_launch.ros2_telemetry.send_depth="${ROS2_TELEMETRY_SEND_DEPTH:-true}" \
  debug_launch.ros2_telemetry.depth_every="${ROS2_TELEMETRY_DEPTH_EVERY:-10}" \
  debug_launch.ros2_telemetry.send_rgb="${ROS2_TELEMETRY_SEND_RGB:-true}" \
  debug_launch.ros2_telemetry.rgb_every="${ROS2_TELEMETRY_RGB_EVERY:-3}" \
  debug_launch.ros2_telemetry.map_every="${ROS2_TELEMETRY_MAP_EVERY:-1}" \
  debug_launch.ros2_telemetry.image_rotate="${ROS2_TELEMETRY_IMAGE_ROTATE:-ccw}" \
  task.terminate_on_collision="${TERMINATE_ON_COLLISION:-true}" \
  task.collision_check_mode="${COLLISION_CHECK_MODE:-overlap}" \
  task.collision_overlap_clearance="${COLLISION_OVERLAP_CLEARANCE:-0.18}" \
  task.sonar.ignore_empty_path="${SONAR_IGNORE_EMPTY_PATH:-false}" \
  task.mapping.classify_dynamic_from_gt="${MAP_CLASSIFY_DYNAMIC_FROM_GT:-false}" \
  task.mapping.dynamic_stamp_from_gt="${MAP_DYNAMIC_STAMP_FROM_GT:-false}" \
  task.mapping.maze_stamp_from_gt="${MAP_MAZE_STAMP_FROM_GT:-false}" \
  task.mapping.bounds_min="${MAP_BOUNDS_MIN:-[-0.2,-0.2,-2.0]}" \
  task.mapping.bounds_max="${MAP_BOUNDS_MAX:-[20.2,6.2,2.0]}" \
  task.mapping.maze_wall_sonar_fallback="${MAP_MAZE_WALL_SONAR_FALLBACK:-true}" \
  task.mapping.dynamic_obstacle_sonar_fallback="${MAP_DYNAMIC_OBSTACLE_SONAR_FALLBACK:-true}" \
  task.mapping.decay="${MAP_DECAY:-0.02}" \
  task.mapping.dynamic_decay="${MAP_DYNAMIC_DECAY:-0.35}" \
  task.mapping.static_hit_stamp_radius_m="${MAP_STATIC_HIT_STAMP_RADIUS_M:-0.09}" \
  task.mapping.static_hit_stamp_inc="${MAP_STATIC_HIT_STAMP_INC:-0.35}" \
  task.mapping.dynamic_hit_stamp_radius_m="${MAP_DYNAMIC_HIT_STAMP_RADIUS_M:-0.08}" \
  task.mapping.dynamic_hit_stamp_inc="${MAP_DYNAMIC_HIT_STAMP_INC:-0.35}" \
  task.mapping.free_dec="${MAP_FREE_DEC:-0.02}" \
  task.mapping.dynamic_free_dec="${MAP_DYNAMIC_FREE_DEC:-0.08}" \
  task.mapping.path_overlay_min_clearance="${MAP_PATH_MIN_CLEARANCE:-0.35}" \
  task.mapping.path_overlay_occ_clearance="${MAP_PATH_OCC_CLEARANCE:-0.35}" \
  task.mapping.path_overlay_dynamic_predict_enable="${MAP_DYNAMIC_PREDICT_ENABLE:-false}" \
  task.planner_stack.rrt.dynamic_predict_enable="${MAP_DYNAMIC_PREDICT_ENABLE:-false}" \
  debug_launch.bt_auto.bt.dynamic_enable="${BT_DYNAMIC_ENABLE:-${BT_DYNAMIC_GT_ENABLE:-true}}" \
  debug_launch.bt_auto.bt.dynamic_from_observation_only="${BT_DYNAMIC_FROM_OBS:-true}" \
  debug_launch.bt_auto.bt.avoid_dist="${BT_AVOID_DIST:-0.42}" \
  debug_launch.bt_auto.bt.replan_dist="${BT_REPLAN_DIST:-0.50}" \
  debug_launch.bt_auto.bt.side_block_dist="${BT_SIDE_BLOCK_DIST:-0.24}" \
  debug_launch.bt_auto.bt.crowd_front_dist="${BT_CROWD_FRONT_DIST:-0.62}" \
  debug_launch.bt_auto.bt.escape_front_dist="${BT_ESCAPE_FRONT_DIST:-0.70}" \
  debug_launch.bt_auto.bt.escape_min_dist="${BT_ESCAPE_MIN_DIST:-0.35}" \
  debug_launch.bt_auto.bt.dynamic_go_enable="${BT_DYNAMIC_GO_ENABLE:-true}" \
  debug_launch.bt_auto.bt.dynamic_go_ttc="${BT_DYNAMIC_GO_TTC:-2.2}" \
  debug_launch.bt_auto.bt.dynamic_yield_ttc="${BT_DYNAMIC_YIELD_TTC:-1.0}" \
  debug_launch.bt_auto.bt.dynamic_go_front_dist="${BT_DYNAMIC_GO_FRONT_DIST:-0.55}" \
  debug_launch.bt_auto.bt.dynamic_slow_front_dist="${BT_DYNAMIC_SLOW_FRONT_DIST:-1.15}" \
  debug_launch.bt_auto.bt.dynamic_go_speed_scale="${BT_DYNAMIC_GO_SPEED_SCALE:-0.62}" \
  debug_launch.bt_auto.bt.dynamic_crossing_avoid_enable="${BT_DYNAMIC_CROSSING_AVOID_ENABLE:-true}" \
  debug_launch.bt_auto.bt.dynamic_crossing_speed_thresh="${BT_DYNAMIC_CROSSING_SPEED_THRESH:-0.04}" \
  debug_launch.bt_auto.bt.dynamic_crossing_avoid_fwd="${BT_DYNAMIC_CROSSING_AVOID_FWD:-0.04}" \
  debug_launch.bt_auto.bt.dynamic_crossing_avoid_lat="${BT_DYNAMIC_CROSSING_AVOID_LAT:-0.25}" \
  debug_launch.bt_auto.bt.dynamic_crossing_avoid_yaw="${BT_DYNAMIC_CROSSING_AVOID_YAW:-0.28}" \
  debug_launch.bt_auto.bt.dynamic_obs_est_speed_thresh="${BT_DYNAMIC_OBS_EST_SPEED_THRESH:-0.08}" \
  debug_launch.bt_auto.bt.dynamic_obs_est_min_rays="${BT_DYNAMIC_OBS_EST_MIN_RAYS:-2}" \
  debug_launch.bt_auto.bt.dynamic_obs_est_max_markers="${BT_DYNAMIC_OBS_EST_MAX_MARKERS:-8}" \
  debug_launch.bt_auto.bt.dynamic_obs_est_center_offset_radius="${BT_DYNAMIC_OBS_EST_CENTER_OFFSET_RADIUS:-0.15}" \
  debug_launch.bt_auto.bt.dynamic_obs_est_assoc_dist="${BT_DYNAMIC_OBS_EST_ASSOC_DIST:-0.75}" \
  debug_launch.bt_auto.bt.dynamic_obs_est_pos_ema="${BT_DYNAMIC_OBS_EST_POS_EMA:-0.35}" \
  debug_launch.bt_auto.bt.dynamic_obs_est_vel_ema="${BT_DYNAMIC_OBS_EST_VEL_EMA:-0.30}" \
  debug_launch.bt_auto.bt.dynamic_obs_est_ttl_steps="${BT_DYNAMIC_OBS_EST_TTL_STEPS:-12}" \
  debug_launch.bt_auto.bt.dynamic_obs_est_marker_min_speed="${BT_DYNAMIC_OBS_EST_MARKER_MIN_SPEED:-0.04}" \
  debug_launch.bt_auto.bt.dynamic_obs_est_map_threshold="${BT_DYNAMIC_OBS_EST_MAP_THRESHOLD:-0.20}" \
  debug_launch.bt_auto.bt.dynamic_obs_est_min_cells="${BT_DYNAMIC_OBS_EST_MIN_CELLS:-3}" \
  task.depth_camera.extra_data_types='[rgb]' \
  "$@"
