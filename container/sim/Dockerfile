# Simulator side of the project: Isaac Sim 5.1 with MarineGym, running the environment and the
# behavior-tree controller (release/scripts/01_run_bt_auto_point_a_to_b.sh and friends).
# The source code is not copied into the image; run-container.sh mounts the repository at /work,
# including the MarineGym submodule at /work/marinegym.
# Add every dependency here: anything installed inside a running container is lost when it stops.

# NVIDIA's Isaac Sim 5.1 image (Ubuntu 24.04, bundled Python 3.11 with PyTorch 2.7.0)
FROM nvcr.io/nvidia/isaac-sim:5.1.0

# accept the Isaac Sim EULA and privacy terms non-interactively
ENV ACCEPT_EULA=Y PRIVACY_CONSENT=Y

# the launchers in release/scripts run Isaac Sim's Python through $ISAAC_PY; MarineGym finds the
# Isaac Sim app (.kit) files through $ISAACSIM_PATH
ENV ISAAC_PY=/isaac-sim/python.sh ISAACSIM_PATH=/isaac-sim

# run-container.sh runs this image as root (so files written to /work and the caches belong to
# the host user under rootless Podman). Keep the image's home, so Isaac Sim's caches stay at the
# paths run-container.sh mounts from the host.
USER root
ENV HOME=/isaac-sim

# Python packages for Isaac Sim's Python: the tested TensorDict/TorchRL versions and MarineGym's
# runtime dependencies (see marinegym/setup.py; the unused ones are left out). tqdm is not in
# setup.py, but marinegym imports it (a native install gets it through moviepy). wandb is for
# the Weights & Biases logging of scripts/train_policy.py.
# PyTorch itself is bundled with Isaac Sim and must not be replaced.
RUN /isaac-sim/python.sh -m pip install --no-cache-dir \
    tensordict==0.7.2 \
    torchrl==0.7.2 \
    hydra-core \
    omegaconf \
    einops \
    tqdm \
    wandb \
    && /isaac-sim/python.sh -c "import torch; assert torch.__version__.startswith('2.7.0'), torch.__version__"

# MarineGym: make the mounted submodule importable (like an editable install, without writing
# build metadata into the submodule)
RUN echo /work/marinegym > "$(/isaac-sim/python.sh -c 'import site; print(site.getsitepackages()[0])')/marinegym.pth"

COPY entrypoint.sh /usr/local/bin/entrypoint.sh

# start all shells (including exec) in the mounted repository
WORKDIR /work

ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]

# open a shell by default
CMD ["bash"]
