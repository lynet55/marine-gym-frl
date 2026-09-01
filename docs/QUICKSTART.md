# Quickstart

The root [README.md](../README.md) covers the main event — the BT
autonomous point-A-to-B run. This doc is everything *else* the repo can do:
opening the bare environment, driving the ROV yourself with a joystick or
keyboard, and getting RViz talking to it. If you want to swap the tank or
retune the maze/obstacles instead, that's [CUSTOM_ENVIRONMENT.md](CUSTOM_ENVIRONMENT.md).

Everything below assumes you're in the repo root with
`source /opt/ros/jazzy/setup.bash` already run.

## 1. Just open the environment, no controller

Sometimes you just want to look at the tank, the ROV, the walls, and the
obstacles without anything driving it — good for checking a scene edit
landed correctly before running the full BT task:

```bash
HEADLESS=false ACTION_MODE=hold NUM_ENVS=1 STEPS=30000 \
bash release/scripts/00_open_hover_vel_environment.sh
```

If you want to see the actual point-A-to-B scenario (fixed start/goal, the
maze, the moving obstacles) without the BT controller doing anything yet:

```bash
HEADLESS=false ACTION_MODE=hold NUM_ENVS=1 STEPS=30000 \
bash release/scripts/00_open_hover_vel_environment.sh \
  task.fixed_start_goal_enable=true \
  task.fixed_start_local=[3.0,3.0,-0.8] \
  task.fixed_goal_local=[17.0,3.0,-0.8] \
  task.maze_walls.enable=true \
  task.maze_walls.num=3 \
  task.obstacles.enable=true \
  task.obstacles.num=3 \
  task.obstacles.speed=0.40
```

## 2. The ROS2 bridges, on their own

If you just want the bridges running by themselves — say, to check topics
are flowing before you bother opening RViz:

```bash
bash release/scripts/00_ros2_bridges.sh
```

This one script starts all three bridges together: joystick, telemetry, and
goal. Once Isaac Sim is up and publishing, `ros2 topic list` should show you
the topics from the list at the bottom of the README.

## 3. Driving it yourself

If you'd rather fly the ROV around by hand than watch the BT controller do
it, here's the three-terminal setup for that.

Terminal 1 — bridges, same as always:

```bash
bash release/scripts/00_ros2_bridges.sh
```

Terminal 2 — the actual joystick node, if you have a physical one plugged
in:

```bash
ros2 run joy joy_node
```

Terminal 3 — Isaac Sim, but in joystick mode instead of BT-auto:

```bash
HEADLESS=false \
ACTION_MODE=ros2_joy \
ROS2_TELEMETRY_ENABLE=true \
ROS2_TELEMETRY_SEND_RGB=true \
ROS2_TELEMETRY_SEND_DEPTH=true \
NUM_ENVS=1 \
STEPS=30000 \
bash release/scripts/00_open_hover_vel_ros2_joystick.sh
```

Don't have a physical joystick handy? You can use your keyboard as a
stand-in instead of running `ros2 run joy joy_node` — everything else in
Terminal 3 stays the same:

```bash
bash release/scripts/00_ros2_keyboard_joy.sh
```

## 4. Watching it in RViz

```bash
rviz2 -d rviz/bt_autonomus.rviz
```

Set the fixed frame to `map` if it isn't already. The displays worth
turning on: `/bluerov/pose`, `/bluerov/trajectory`, `/bluerov/path`,
`/bluerov/sonar/scan`, `/bluerov/sonar/hits`, `/bluerov/map`,
`/bluerov/map_occ/points`, `/bluerov/map_occ/hit_points`,
`/bluerov/map_occ/scan_origin`, `/bluerov/obstacles`, `/bluerov/rgb/image`,
`/bluerov/depth/image`.

If RViz complains `Frame [map] does not exist`, it almost always means
either the bridge script isn't running, or Isaac Sim was launched without
`ROS2_TELEMETRY_ENABLE=true` — check both before digging further.

## 5. Sending a goal from inside RViz

Run the BT autonomous command from the README with the goal bridge enabled
(it is, by default, whenever `00_ros2_bridges.sh` is running), then just use
RViz's `2D Goal Pose` tool to click a new target. It publishes to
`/goal_pose`, which the bridge picks up. One thing to know: the bridge
currently forces whatever goal you click to `z = -0.8` unless you override
it with `ROS2_GOAL_FORCE_Z`.

## When something's not working

If ROS topics exist but RViz just shows nothing, this is usually the
fastest way to find out where the chain broke:

```bash
ros2 topic echo /tf_static --once
ros2 topic echo /tf --once
ros2 topic echo /bluerov/pose --once
```

And if Isaac Sim crashed and won't start cleanly on the next attempt,
there's a good chance a process from the last run is still hanging around.
Check before you relaunch:

```bash
pgrep -af "isaacsim|kit|python.sh|debug_open"
```

Just be careful here — only kill the process you actually started. Don't
sweep up unrelated desktop or GPU processes by accident.
