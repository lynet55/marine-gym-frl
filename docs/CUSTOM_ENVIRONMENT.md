# Custom Environments

If you want a different tank, a bigger maze, faster obstacles, or a
different start/goal — this is the doc for you. None of it requires
touching Python; it's all config.

Before diving into how to change things, it helps to actually understand
how the scene gets built in the first place, because "the environment"
isn't one thing — it's three separate pieces that come together at runtime,
and each one is swapped or tuned differently.

## How the scene is actually built

**1. The tank mesh.** This is a single static USD file, loaded once at the
world origin and marked collidable. This repo ships its own copy at
[`assets/usd/BlueROV/tank.usd`](../assets/usd/BlueROV/tank.usd), and
`cfg/task/hover_vel_stable_goal.yaml`'s `tank_usd_path` points at it. The
loading itself happens inside the external `marinegym` package
(`marinegym/envs/single/hover.py::Hover._design_scene`), and it's about as
simple as it sounds:

```python
tank_usd = self.cfg.task.get("tank_usd_path")
prim_utils.create_prim(
    prim_path="/World/envs/env_0/tank",
    usd_path=tank_usd,
    translation=(0.0, 0.0, 0.0),
)
# then marked static + collidable
```

Because it's just a USD file reference, any valid USD scene works here — it
doesn't have to be anything BlueROV-specific.

**2. The maze walls and moving obstacles.** These aren't part of the USD at
all — they're primitive boxes/cylinders spawned procedurally at runtime,
straight from `task.maze_walls.*` and `task.obstacles.*` in the Hydra
config. So if you're trying to change how many walls there are or how fast
obstacles move, you're not going anywhere near USD files — it's pure
config.

**3. The BlueROV robot itself.** This one's different — it's a separate
USD/YAML pair (`marinegym/robots/assets/usd/BlueROV/BlueROV.usd` +
`BlueROV.yaml`) that lives inside your `marinegym` checkout, resolved via
`task.drone_model.name: BlueROV`. Unlike the tank, you can't swap this from
a config in this repo — it needs to be a vehicle actually registered inside
`marinegym`.

Keeping these three straight matters: swapping the tank is a one-line
override, retuning the maze/obstacles is editing a couple of yaml blocks,
and swapping the robot is a `marinegym`-side change, not something this
repo controls.

## Swapping in your own tank

You've got two ways to do this, depending on whether you want it to be the
new default or just a one-off:

```bash
# one-off override via env var — checked before the bundled default
TANK_USD_PATH=/path/to/my_scene.usd \
bash release/scripts/00_open_hover_vel_environment.sh HEADLESS=false ACTION_MODE=hold

# or override directly on the Hydra command line, same effect
bash release/scripts/00_open_hover_vel_environment.sh \
  task.tank_usd_path=/path/to/my_scene.usd
```

Or, if you want it to be the new normal, drop your replacement file into
`assets/usd/BlueROV/` and point
[`cfg/task/hover_vel_stable_goal.yaml`](../cfg/task/hover_vel_stable_goal.yaml)
at it directly.

A few things your custom USD needs to actually work as a tank, not just
look like one:

- Keep it centered near the origin, and roughly the same scale as the
  default tank (this repo works in meters).
- Make sure the floor/walls are real collision geometry — a `Mesh`/`Cube`
  with a UsdPhysics collision API applied, or anything `create_prim` plus
  the default collision setup will pick up. Visual-only geometry won't stop
  the ROV, and it won't stop sonar rays either — you'll get a ROV that
  swims straight through your walls.
- Try to keep it self-contained, with no relative references to textures or
  materials outside the file itself, if you care about this staying a
  portable, drop-in-anywhere repo. I checked the bundled `tank.usd` and it
  has zero external asset references — worth matching that if you can.

## Retuning the tank bounds, walls, and obstacles

All of this lives in the Hydra configs already sitting in this repo — no
code to touch. Here's what each knob actually does and where it lives.

**Tank/world bounds.** If you swap in a differently-sized tank, this is the
piece you *must* update to match it — otherwise episodes will terminate or
plan against a boundary that doesn't match what you're actually seeing:

```yaml
# cfg/debug/open_hover_vel_env.yaml -> task.termination.tank_bounds
task:
  termination:
    tank_bounds:
      x_range: [0.0, 20.0]
      y_range: [0.0, 6.5]
      z_range: [-1.5, 0.2]
  mapping:
    bounds_min: [-0.2, -0.2, -2.0]   # occupancy-grid extent for ROS map/RViz
    bounds_max: [20.2, 7.0, 2.0]
```

**Maze walls** — remember, these are procedural, not part of any USD file:

```yaml
task:
  maze_walls:
    enable: true
    num: 3            # how many wall segments
    length: 1.0
    thickness: 0.20
    height: 1.80
    spawn_ranges:      # keep these inside your tank_bounds above
      x_range: [5.0, 15.0]
      y_range: [1.0, 5.0]
      z_range: [-0.8, -0.8]
```

**Dynamic obstacles** — also procedural:

```yaml
task:
  obstacles:
    enable: true
    num: 3
    radius: 0.25
    height: 0.40
    speed: 0.25
    motion: line       # line | ... (see obstacles.py for the full set)
    line_axis: y
    spawn_ranges:
      x_range: [5.0, 15.0]
      y_range: [1.0, 5.0]
      z_range: [-0.8, -0.8]
```

**Start and goal**, for the point-A-to-B run specifically:

```yaml
task:
  fixed_start_goal_enable: true
  fixed_start_local: [3.0, 3.0, -0.8]
  fixed_goal_local: [17.0, 3.0, -0.8]
```

You don't have to edit files for any of this, by the way — everything above
can also be overridden right on the command line for a single run, exactly
like the example in the root README:

```bash
bash release/scripts/00_open_hover_vel_bt_autonomous.sh \
  task.fixed_start_local=[2.0,2.0,-0.8] \
  task.fixed_goal_local=[10.0,4.0,-0.8] \
  task.maze_walls.num=5 \
  task.obstacles.num=6 \
  task.obstacles.speed=0.30
```

## Checking your changes before committing to a full GUI run

Isaac Sim isn't fast to boot, so I'd avoid iterating on config changes by
just launching the full GUI over and over. Two much faster checks first:

First, print the fully resolved config — including whatever overrides
you're passing — without booting Isaac Sim at all:

```bash
${ISAAC_PY:-/home/eth11/isaacsim51/python.sh} scripts/debug_open_hover_vel_env.py \
  --cfg job --resolve \
  task.tank_usd_path=/path/to/my_scene.usd \
  task.maze_walls.num=5
```

That'll immediately tell you if a value didn't resolve the way you
expected. Once that looks right, do a quick headless pass — tens of steps,
no GUI, no ROS2 — before running the full interactive workflow:

```bash
HEADLESS=true ACTION_MODE=hold NUM_ENVS=1 STEPS=50 CLOSE_ON_FINISH=true \
ROS2_TELEMETRY_ENABLE=false \
bash release/scripts/00_open_hover_vel_environment.sh
```

A clean run ends with `finished stepping in ...s` and nothing that looks
like `Traceback` or `Error` in the output. If you see that, you've caught
whatever's wrong in a few seconds instead of after a full GUI boot.
