# Container Installation and First Run

This is the setup path for running the project in containers: instead of installing Isaac Sim,
MarineGym and ROS 2 on the host, the project runs in two containers. Only a container engine and
the NVIDIA driver need to be on the host. To install everything on the host instead, follow
[INSTALL_NATIVE.md](INSTALL_NATIVE.md).

| Container | Image | Runs |
|---|---|---|
| `ros` | ROS 2 Jazzy desktop (`container/ros/`) | the ROS 2 <-> UDP bridges, the keyboard joystick, RViz |
| `sim` | Isaac Sim 5.1 + MarineGym (`container/sim/`) | the simulator and the behavior-tree controller |

The two talk over UDP on localhost (ports 15000, 15010, 15020), exactly as in a native install;
both containers share the host network.

IsaacLab is not included: nothing in the point-A-to-B runtime uses it. It is only needed for RL
training.

## Requirements

- Podman (preferred) or Docker >= 25
- An NVIDIA RTX GPU with an R580 driver (the version Isaac Sim 5.1 is tested with)
- [nvidia-container-toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) >= 1.18 with the [CDI spec generated](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/cdi-support.html). The spec pins the
  driver's library versions, so regenerate it after changing the driver:
  `sudo nvidia-ctk cdi generate --output=/var/run/cdi/nvidia.yaml`
- About 25 GB of disk for the images, plus a few GB for the Isaac Sim caches

The MarineGym dependency is a git submodule hosted on NTNU GitLab, so you need access to
`git.ntnu.no/mded/marinegym-isaacsim51`. Check it out on the host (the containers do not need
any git credentials):

```bash
git submodule update --init
```

## Running the point-A-to-B task

Three terminals, all from the repository root. No `source /opt/ros/...` is needed on the host.

**Terminal 1 — the ROS 2 bridges.** This also starts the `ros` container:

```bash
./run-container.sh ros bash release/scripts/00_ros2_bridges.sh
```

**Terminal 2 — RViz**, in the same `ros` container:

```bash
./run-container.sh ros rviz2 -d rviz/bt_autonomus.rviz
```

**Terminal 3 — the simulator with the BT controller:**

```bash
HEADLESS=false \
ROS2_TELEMETRY_ENABLE=true \
ROS2_TELEMETRY_SEND_RGB=true \
ROS2_TELEMETRY_SEND_DEPTH=true \
NUM_ENVS=1 \
STEPS=30000 \
./run-container.sh sim bash release/scripts/01_run_bt_auto_point_a_to_b.sh
```

The launcher settings (`HEADLESS`, `STEPS`, `ROS2_*`, `BT_*`, ... — every variable the scripts
in `release/scripts/` read) are passed into the container when they are set on the host, so
they work the same as with a native install. `ISAAC_PY` is the exception: the image sets it.

Config overrides work as with a native install, e.g.
`./run-container.sh sim bash release/scripts/01_run_bt_auto_point_a_to_b.sh task.obstacles.num=5`.

The first start of the `sim` container takes several minutes (image build, shader compilation);
later starts reuse the caches.

## How `run-container.sh` works

```
./run-container.sh <ros|sim> [command...]
```

- Builds the image from `container/<role>/` (cached after the first time) and starts the
  container `bluerov-bt-autonomous-<role>`, with this repository mounted at `/work`.
- Without a command, opens a shell. With a command, runs it.
- If that container is already running, opens another shell (or runs the command) in it.
- The first shell/command is the main one: when it exits, the container stops, along with
  everything else running in it.

Anything installed inside a running container is lost when it stops. Add dependencies to
`container/<role>/Dockerfile`; the image is rebuilt on the next start.

### ROS 2 discovery

The `ros` image sets `ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST`: the container shares the host
network, and this keeps its nodes from discovering (and being discovered by) ROS 2 nodes
elsewhere on the LAN.

### Isaac Sim caches

Isaac Sim's caches (shaders, Kit, assets, logs, config) are kept in `.cache/isaac/` in this
repository (ignored by git), so only the first start is slow. To use a different location, set
`ISAAC_CACHE_DIR`. To reset them, delete that directory.

### MarineGym

The `sim` image does not contain MarineGym itself: it imports it from the submodule at
`/work/marinegym`, so changes to it take effect without rebuilding the image. The image
contains its dependencies (TensorDict/TorchRL 0.7.2 etc.) on top of the PyTorch bundled with
Isaac Sim.

The warning `Failed to import torchrl C++ binaries` at startup is expected and harmless: the
compiled extensions are only used by replay buffers, which this project does not use.

## Container options

```bash
CONTAINER_ENGINE=docker ./run-container.sh ros     # force Docker instead of Podman
USE_NVIDIA=0 ./run-container.sh ros                # ros only: use /dev/dri instead of NVIDIA
PROJECT_NAME=other ./run-container.sh ros          # other image/container name prefix
```

With Docker (unlike rootless Podman), files created by the containers are owned by root on the
host.
