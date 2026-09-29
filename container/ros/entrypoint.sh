#!/usr/bin/env bash
# Runs the given command with ROS sourced, so non-interactive commands
# (e.g. ./run-container.sh ros rviz2 -d rviz/bt_autonomus.rviz) work like in a container shell.

source /opt/ros/jazzy/setup.bash

exec "$@"
