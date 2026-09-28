# Installation and First Run

This is the complete setup path for this repository. It installs the tested
Isaac Sim, IsaacLab, MarineGym, and ROS 2 stack, verifies each layer, and then
runs the BlueROV point-A-to-B demo.

The project was tested with the following combination:

| Component | Tested version |
|---|---|
| Operating system | Ubuntu 24.04.5 LTS |
| GPU | NVIDIA GeForce RTX 5090 (32 GB) |
| Kernel | `6.17.0-40-generic` |
| NVIDIA driver | `580.178.04-open` |
| Isaac Sim | 5.1.0 |
| IsaacLab | v2.3.2 |
| TensorDict | 0.7.2 |
| TorchRL | 0.7.2 |
| ROS 2 | Jazzy |

Isaac Sim 5.1 is an older, pinned dependency of this project. NVIDIA's
[5.1 requirements](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/installation/requirements.html)
list Ubuntu 22.04/24.04 and Linux driver 580.65.06 as the tested baseline.
The exact kernel and patch version above describe the workstation on which
this repository was verified; they are not requirements for every machine.

## 1. Install system prerequisites and verify the GPU

```bash
sudo apt update
sudo apt install -y git unzip cmake build-essential

uname -r
nvidia-smi
modinfo nvidia | grep -E '^(filename|version|license):'
dkms status
```

Use an NVIDIA RTX GPU with a working R580 driver. On the tested workstation,
the open kernel-module package was installed with:

```bash
sudo apt -s install nvidia-driver-580-open
sudo apt install nvidia-driver-580-open
sudo reboot
```

The first command is a simulation: inspect it before running the real install.
Do not run `apt autoremove` until the complete stack has been verified.

After rebooting, `nvidia-smi` must work. A `Dual MIT/GPL` value from
`modinfo nvidia` indicates that the open kernel module is loaded. If DKMS did
not build the driver for the running kernel, see [Driver and RTX crashes](#driver-and-rtx-crashes).

## 2. Install Isaac Sim 5.1

Download the Linux x86-64 archive from the official
[Isaac Sim 5.1 download page](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/installation/download.html).
The commands below install it at `~/isaacsim51`; use another location if
needed, but keep the environment variables consistent.

```bash
export ISAACSIM_PATH="$HOME/isaacsim51"
export ISAACSIM_PYTHON_EXE="$ISAACSIM_PATH/python.sh"
export ISAAC_PY="$ISAACSIM_PYTHON_EXE"

mkdir -p "$ISAACSIM_PATH"
unzip "$HOME/Downloads/<isaac-sim-5.1-archive>.zip" -d "$ISAACSIM_PATH"
cd "$ISAACSIM_PATH"
./post_install.sh
```

Replace `<isaac-sim-5.1-archive>` with the downloaded filename. Verify the
installation before installing anything into its Python environment:

```bash
cd "$ISAACSIM_PATH"
./isaac-sim.compatibility_check.sh --/app/quitAfter=10 --no-window
```

The important result is `System checking result: PASSED`. Then run a small
headless startup test:

```bash
"$ISAACSIM_PYTHON_EXE" -c "
from isaacsim import SimulationApp
app = SimulationApp({
    'headless': True,
    'multi_gpu': False,
    'extra_args': [
        '--/renderer/multiGpu/enabled=False',
        '--/physics/cudaDevice=0',
    ],
})
print('ISAAC_SIM_STARTUP_PASSED', flush=True)
for _ in range(10):
    app.update()
print('ISAAC_SIM_UPDATE_PASSED', flush=True)
app.close()
"
```

For a GUI check:

```bash
cd "$ISAACSIM_PATH"
./isaac-sim.sh
```

When upgrading from an older Isaac Sim installation, run
`./isaac-sim.sh --reset-user` once. `./clear_caches.sh` can clear old Kit and
shader caches, but it is destructive to those caches and asks for confirmation.

## 3. Install IsaacLab v2.3.2

Follow the tagged v2.3.2 workflow rather than manually installing individual
source directories. The official binary-install guide uses the same `_isaac_sim`
link and `isaaclab.sh` installer:
[IsaacLab binary installation](https://isaac-sim.github.io/IsaacLab/v2.3.0/source/setup/installation/binaries_installation.html).

```bash
cd "$HOME"
git clone --branch v2.3.2 --depth 1 \
  https://github.com/isaac-sim/IsaacLab.git
cd "$HOME/IsaacLab"

ln -sfn "$ISAACSIM_PATH" _isaac_sim
readlink -f _isaac_sim

./isaaclab.sh --install rsl_rl
```

`readlink` should print the Isaac Sim directory selected above. Installing only
`rsl_rl` is sufficient for the tested stack; `./isaaclab.sh --install` installs
all supported learning frameworks.

Verify IsaacLab:

```bash
cd "$HOME/IsaacLab"
git describe --tags --exact-match
./isaaclab.sh -p scripts/tutorials/00_sim/create_empty.py
```

The tag should be `v2.3.2`, and the second command should open an empty
simulator viewport. Exit it with `Ctrl+C`. An optional headless RSL-RL test is:

```bash
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
  --task=Isaac-Ant-v0 --headless
```

Stop it with `Ctrl+C` after training starts.

## 4. Install MarineGym into Isaac Sim's Python

This repository contains the BlueROV task and tank scene, but not the external
`marinegym` Python package or its BlueROV robot USD/YAML. Use the NTNU
[MarineGym Isaac Sim 5.1 compatibility repository](https://git.ntnu.no/mded/marinegym-isaacsim51).
It preserves the tested Isaac Sim 5.1 and TorchRL/TensorDict 0.7 port, BlueROV
assets, and calibrated hard-coded mixer that are not present in the public
[MarineGym upstream](https://github.com/Marine-RL/MarineGym).

```bash
export MARINEGYM_PATH="$HOME/marinegym-isaacsim51"

# Skip this clone if the checkout already exists.
git clone https://git.ntnu.no/mded/marinegym-isaacsim51.git \
  "$MARINEGYM_PATH"

cd "$MARINEGYM_PATH"
git checkout 55c4bf8a9668cd94604e3f3edb68eb70cc6178bc
"$ISAACSIM_PYTHON_EXE" -m pip install \
  "tensordict==0.7.2" \
  "torchrl==0.7.2"
"$ISAACSIM_PYTHON_EXE" -m pip install -e .
```

Confirm that MarineGym and its critical dependencies resolve in the same
interpreter used by this repository:

```bash
"$ISAACSIM_PYTHON_EXE" -c "
import torch, tensordict, torchrl, marinegym
print('MarineGym:', marinegym.__file__)
print('PyTorch:', torch.__version__)
print('TensorDict:', tensordict.__version__)
print('TorchRL:', torchrl.__version__)
print('CUDA available:', torch.cuda.is_available())
print('GPU:', torch.cuda.get_device_name(0))
"

"$ISAACSIM_PYTHON_EXE" -m pip check
```

On the tested Isaac Sim bundle, `pip check` also reports conflicts from optional
NVIDIA extensions (`nvidia-srl-*`, `msal`, and `fastapi`). Do not upgrade core
Isaac Sim packages blindly to silence those messages. For this repository, the
required checks are that the four imports above succeed, CUDA is available,
and the smoke test in step 7 completes.

If `import marinegym` fails, the editable install was probably run with a
different Python interpreter. Re-run it with `$ISAACSIM_PYTHON_EXE`.

If HTTPS authentication is unavailable, use the SSH clone URL
`git.ntnu.no:mded/marinegym-isaacsim51.git` instead.

The pinned commit makes the installation reproducible. To test newer companion
repository changes, return to its `main` branch explicitly; do not silently
replace the pinned revision in a verified setup.

## 5. Install ROS 2 Jazzy and RViz

ROS is optional for a headless simulator smoke test, but required for the UDP
bridges, joystick input, topics, and RViz. On Ubuntu 24.04, install ROS 2 Jazzy
using the official
[Ubuntu deb-package instructions](https://docs.ros.org/en/jazzy/Installation/Ubuntu-Install-Debs.html).
After adding the ROS repository as described there, install:

```bash
sudo apt update
sudo apt install -y ros-jazzy-desktop ros-jazzy-joy
source /opt/ros/jazzy/setup.bash

ros2 --help >/dev/null
rviz2 --help >/dev/null
```

The bridge scripts source `/opt/ros/jazzy/setup.bash` themselves. Source it in
interactive terminals too when using `ros2`, `rviz2`, or `joy_node` directly.

## 6. Configure this repository

No package installation is needed for this repository. Its Python entrypoints
load `src/bluerov_manual` directly. Clone or enter the repository, then export
the Isaac Sim interpreter path:

```bash
cd "$HOME"
git clone https://git.ntnu.no/mded/bluerov-bt-autonomous.git
cd "$HOME/bluerov-bt-autonomous"

export ISAACSIM_PATH="$HOME/isaacsim51"
export ISAACSIM_PYTHON_EXE="$ISAACSIM_PATH/python.sh"
export ISAAC_PY="$ISAACSIM_PYTHON_EXE"

test -x "$ISAAC_PY"
test -f assets/usd/BlueROV/tank.usd
```

Skip the clone command if the repository already exists. A GitHub mirror is also
available at `https://github.com/eather0056/bluerov-bt-autonomous.git`. The launch scripts
default to `/home/eth11/isaacsim51/python.sh`; setting `ISAAC_PY` makes them
portable to other usernames and install locations. Put the three exports in
your shell profile if you want them to persist across terminals.

## 7. Run the repository smoke test

Run this before starting ROS or RViz. It opens one environment, advances 50
steps without a controller, and exits:

```bash
cd "$HOME/bluerov-bt-autonomous"

HEADLESS=true \
ACTION_MODE=hold \
NUM_ENVS=1 \
STEPS=50 \
CLOSE_ON_FINISH=true \
ROS2_TELEMETRY_ENABLE=false \
bash release/scripts/00_open_hover_vel_environment.sh
```

A successful run prints the `BLUEROV HOVER_VEL ENV DEBUG` scene summary and
finishes without a traceback. This verifies Isaac Sim, MarineGym, the robot
asset, the bundled tank USD, and this repository together.

## 8. Run the autonomous point-A-to-B demo

Open three terminals. In each, enter the repository. Keep `ISAAC_PY` exported
in the simulator terminal.

Terminal 1 — ROS bridges:

```bash
cd "$HOME/bluerov-bt-autonomous"
bash release/scripts/00_ros2_bridges.sh
```

Terminal 2 — RViz:

```bash
source /opt/ros/jazzy/setup.bash
cd "$HOME/bluerov-bt-autonomous"
rviz2 -d rviz/bt_autonomus.rviz
```

Use `map` as the RViz fixed frame.

Terminal 3 — Isaac Sim and the behavior-tree controller:

```bash
cd "$HOME/bluerov-bt-autonomous"
export ISAAC_PY="$HOME/isaacsim51/python.sh"

HEADLESS=false \
ROS2_TELEMETRY_ENABLE=true \
ROS2_TELEMETRY_SEND_RGB=true \
ROS2_TELEMETRY_SEND_DEPTH=true \
NUM_ENVS=1 \
STEPS=30000 \
bash release/scripts/01_run_bt_auto_point_a_to_b.sh
```

Scenario values such as the start, goal, walls, obstacles, and controller
tuning live in `cfg/debug/bt_auto_point_a_to_b.yaml`. A one-run Hydra override
can be appended without editing the file:

```bash
bash release/scripts/01_run_bt_auto_point_a_to_b.sh task.obstacles.num=5
```

No RL checkpoint is required: this launcher uses the hand-written behavior
tree controller.

## 9. Verify ROS output

After the bridge and simulator are running:

```bash
source /opt/ros/jazzy/setup.bash
ros2 topic list
ros2 topic echo /bluerov/pose --once
ros2 topic echo /bluerov/sonar/scan --once
ros2 topic echo /bluerov/status --once
ros2 topic echo /tf_static --once
```

For manual joystick/keyboard operation and the other launch modes, continue
with [docs/QUICKSTART.md](docs/QUICKSTART.md).

## Troubleshooting

### `ModuleNotFoundError: No module named 'marinegym'`

Check which interpreter the launcher uses and reinstall MarineGym into it:

```bash
printf '%s\n' "$ISAAC_PY"
"$ISAAC_PY" -c "import marinegym; print(marinegym.__file__)"
cd "$MARINEGYM_PATH"
"$ISAAC_PY" -m pip install -e .
```

### Driver and RTX crashes

If `nvidia-smi` cannot communicate with the driver, inspect the running kernel
and DKMS state:

```bash
uname -r
modinfo nvidia | head
dkms status
lsmod | grep nvidia
```

If the driver is present only for a different kernel, build the installed
driver for the running kernel. Replace `<driver-version>` with the exact value
shown by the installed package/DKMS state:

```bash
sudo dkms install --force \
  -m nvidia \
  -v <driver-version> \
  -k "$(uname -r)"
sudo depmod -a
sudo update-initramfs -u -k "$(uname -r)"
sudo reboot
```

An `Unknown symbol in module` error usually indicates mismatched kernel modules
and user-space driver libraries; rebuild for the exact running kernel and
regenerate its initramfs.

On the tested RTX 5090 machine, newer 590/595-series packages could pass the
compatibility checker but crash during RTX initialization in
`librtx.scenedb.plugin.so`. The working resolution was the R580 open branch.
Treat this as a known tested-system workaround, and compare your installed
driver with NVIDIA's Isaac Sim 5.1 requirements before changing it.

### RViz has no `map` frame

Make sure both conditions are true:

1. `release/scripts/00_ros2_bridges.sh` is still running.
2. Isaac Sim was launched with `ROS2_TELEMETRY_ENABLE=true`.

Then check `/tf_static`, `/tf`, and `/bluerov/pose` with `ros2 topic echo`.

### UDP port already in use

The local bridge ports are 15000 (joystick), 15010 (telemetry), and 15020
(goal). Look for a previous bridge or simulator process before restarting:

```bash
pgrep -af "isaacsim|kit|python.sh|debug_open|ros2_.*_udp"
```

Terminate only processes belonging to your previous run.

### Warnings that are normally non-fatal

Telemetry cleanup messages after a timed compatibility check, AMD integrated
GPU “skipping unsupported non-NVIDIA GPU” messages, `IOMMU is enabled`,
internal `5.1.0-rc.19` metadata, `omni.isaac.version has been deprecated`, and
fast-shutdown warnings may occur on a healthy installation. The tested setup also
prints a TorchRL warning about optional C++ replay-buffer binaries because
TorchRL 0.7.2 is paired with Isaac Sim's bundled PyTorch 2.7.0; the BT demo
does not use those replay-buffer extensions. Judge success by
the compatibility result, successful `SimulationApp` updates, and the repo
smoke test completing without a traceback.
