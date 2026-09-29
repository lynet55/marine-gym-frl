#!/usr/bin/env bash
# Builds a container image and starts the container with this repository mounted at /work.
# The first argument picks the container:
#   ros  ROS 2 Jazzy: the ROS 2 <-> UDP bridges, the keyboard joystick and RViz (container/ros)
#   sim  Isaac Sim 5.1 with MarineGym: the simulator and the BT controller (container/sim)
# Without further arguments a shell is opened; with arguments they are run as the command instead
# (e.g. ./run-container.sh ros ros2 topic list).
# If the container is already running, opens another shell (or runs the command) in it instead.

# run relative to the repository root, regardless of the current directory
cd "$(dirname "$0")"

ROLE=$1
case "$ROLE" in
    ros|sim) shift ;;
    *)
        echo "Usage: $0 <ros|sim> [command...]" >&2
        exit 1
        ;;
esac

# Project name, used for the image and container names: the repository directory name in
# lowercase, with characters not allowed in image names replaced by '-'.
# Override with PROJECT_NAME=my-project ./run-container.sh <ros|sim>
PROJECT_NAME=${PROJECT_NAME:-$(basename "$(pwd)" | tr 'A-Z' 'a-z' | tr -c 'a-z0-9_.\n-' '-')}

# Container engine: podman (preferred) or docker. Override with CONTAINER_ENGINE=docker ./run-container.sh <ros|sim>
if [ -z "$CONTAINER_ENGINE" ]; then
    if command -v podman >/dev/null 2>&1; then
        CONTAINER_ENGINE=podman
    elif command -v docker >/dev/null 2>&1; then
        CONTAINER_ENGINE=docker
    else
        echo "Neither podman nor docker found" >&2
        exit 1
    fi
fi

# Container name. Override with CONTAINER_NAME=my-container ./run-container.sh <ros|sim>
CONTAINER_NAME=${CONTAINER_NAME:-$PROJECT_NAME-$ROLE}

# Pass the settings of the launchers in release/scripts (e.g. HEADLESS=true, STEPS, ROS2_*) into
# the container when they are set on the host, so they work like with a native install.
# Host paths (ISAAC_PY, ROS_LOG_DIR) are left out: they do not exist in the container.
ENV_ARGS=()
for var in $(grep -ohE '\$\{[A-Z0-9_]+:-' release/scripts/*.sh | sed 's/[${:-]//g' | sort -u); do
    case "$var" in
        ISAAC_PY|ROS_LOG_DIR) ;;
        *) [ -n "${!var+set}" ] && ENV_ARGS+=(-e "$var") ;;
    esac
done

# If the container is already running, open another shell (or run the given command) in it,
# through the entrypoint so the command gets the same environment (e.g. ROS sourced)
if $CONTAINER_ENGINE container inspect "$CONTAINER_NAME" >/dev/null 2>&1; then
    exec $CONTAINER_ENGINE exec -it "${ENV_ARGS[@]}" "$CONTAINER_NAME" /usr/local/bin/entrypoint.sh "${@:-bash}"
fi

# allow GUI apps in the container to use the host X server
if command -v xhost >/dev/null 2>&1; then
    xhost +local: >/dev/null
fi

DOCKER_IMAGE_NAME=$PROJECT_NAME-$ROLE

(
    cd "container/$ROLE"
    $CONTAINER_ENGINE build -t "$DOCKER_IMAGE_NAME" .
) || exit 1

echo "Starting container '$CONTAINER_NAME'. This is the main shell: exiting it stops the container"
echo "and all other shells. Run ./run-container.sh $ROLE again in another terminal to open more shells."

# Arguments for "run", collected in an array grouped by purpose.
# Each array element is passed as one argument, so values with spaces stay intact.
RUN_ARGS=()

# interactive terminal; remove the container on exit; fixed name so more shells can attach
RUN_ARGS+=(-it --rm --name "$CONTAINER_NAME")

# launcher settings from the host (see ENV_ARGS above)
RUN_ARGS+=("${ENV_ARGS[@]}")

# GPU: NVIDIA through nvidia-container-toolkit (CDI) if it is set up, otherwise the host
# /dev/dri devices, used by the Mesa drivers in the image (Intel, AMD).
# Override with USE_NVIDIA=0 (or 1) ./run-container.sh <ros|sim>
if [ "${USE_NVIDIA:-auto}" = auto ]; then
    if grep -qs nvidia.com/gpu /etc/cdi/*.yaml /var/run/cdi/*.yaml; then
        USE_NVIDIA=1
    else
        USE_NVIDIA=0
    fi
fi
if [ "$USE_NVIDIA" = 1 ]; then
    # expose all NVIDIA GPUs with all driver capabilities, incl. graphics (and CUDA), and
    # render on the NVIDIA GPU on hybrid-graphics laptops
    RUN_ARGS+=(
        --device nvidia.com/gpu=all
        -e NVIDIA_DRIVER_CAPABILITIES=all
        -e __NV_PRIME_RENDER_OFFLOAD=1
        -e __GLX_VENDOR_LIBRARY_NAME=nvidia
    )
elif [ -d /dev/dri ]; then
    RUN_ARGS+=(--device /dev/dri)
    # give the container user access to the GPU device groups (video, render)
    if [ "$CONTAINER_ENGINE" = podman ]; then
        RUN_ARGS+=(--group-add keep-groups)
    else
        for dev in /dev/dri/*; do
            RUN_ARGS+=(--group-add "$(stat -c %g "$dev")")
        done
    fi
else
    echo "Warning: no GPU found, GUI apps will use software rendering" >&2
fi
if [ "$ROLE" = sim ] && [ "$USE_NVIDIA" != 1 ]; then
    echo "Isaac Sim needs an NVIDIA GPU with nvidia-container-toolkit and its CDI spec set up" >&2
    exit 1
fi

# GUI: share the host X11 and Wayland display sockets
RUN_ARGS+=(
    -e DISPLAY="$DISPLAY"
    -v /tmp/.X11-unix:/tmp/.X11-unix
    -e XDG_RUNTIME_DIR=/tmp
)
if [ -n "$WAYLAND_DISPLAY" ]; then
    RUN_ARGS+=(
        -e WAYLAND_DISPLAY="$WAYLAND_DISPLAY"
        -v "$XDG_RUNTIME_DIR/$WAYLAND_DISPLAY:/tmp/$WAYLAND_DISPLAY"
    )
fi

# sound: share the host PulseAudio/PipeWire socket if there is one
if [ -S "$XDG_RUNTIME_DIR/pulse/native" ]; then
    RUN_ARGS+=(
        -e PULSE_SERVER=unix:/tmp/pulse/native
        -v "$XDG_RUNTIME_DIR/pulse/native:/tmp/pulse/native"
    )
fi

# ROS 2: share the host network and IPC namespace, so topics reach ROS nodes on the host and in
# other containers, including the shared memory transport
RUN_ARGS+=(--network host --ipc host)

# Podman only: disable SELinux labeling so the mounted host directories are accessible
if [ "$CONTAINER_ENGINE" = podman ]; then
    RUN_ARGS+=(--security-opt label=disable)
fi

# mount this repository at /work (sources and output live on the host)
RUN_ARGS+=(-v "$(realpath .):/work")

# Isaac Sim: run as root, so files it writes belong to the host user under rootless Podman, and
# keep its caches (shaders, assets, Kit) in .cache/isaac in this repository, so only the first
# start is slow. Override the location with ISAAC_CACHE_DIR=/some/dir ./run-container.sh sim
if [ "$ROLE" = sim ]; then
    ISAAC_CACHE_DIR=$(realpath -m "${ISAAC_CACHE_DIR:-.cache/isaac}")
    RUN_ARGS+=(--user root)
    for mount in \
        kit:/isaac-sim/kit/cache \
        cache:/isaac-sim/.cache \
        computecache:/isaac-sim/.nv/ComputeCache \
        logs:/isaac-sim/.nvidia-omniverse/logs \
        config:/isaac-sim/.nvidia-omniverse/config \
        data:/isaac-sim/.local/share/ov/data
    do
        mkdir -p "$ISAAC_CACHE_DIR/${mount%%:*}"
        RUN_ARGS+=(-v "$ISAAC_CACHE_DIR/${mount%%:*}:${mount#*:}")
    done
fi

"$CONTAINER_ENGINE" run "${RUN_ARGS[@]}" "$DOCKER_IMAGE_NAME" "$@"
