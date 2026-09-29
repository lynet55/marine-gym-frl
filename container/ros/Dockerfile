# ROS 2 side of the project: the ROS 2 <-> UDP bridges (scripts/ros2_*.py), the keyboard
# joystick and RViz. The source code is not copied into the image; run-container.sh mounts the
# repository at /work.
# Add every dependency here: anything installed inside a running container is lost when it stops.

# ROS 2 Jazzy (the tested distro) with RViz, rqt and the standard message packages
FROM docker.io/osrf/ros:jazzy-desktop

# system packages: GUI and OpenGL/Vulkan runtime libraries (the NVIDIA driver is injected by the
# container toolkit, Mesa covers Intel/AMD GPUs).
# Add your own packages to this list (e.g. ros-jazzy-<package>).
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libegl1 libglx-mesa0 libvulkan1 mesa-vulkan-drivers \
    && rm -rf /var/lib/apt/lists/*

# keep ROS 2 discovery on this machine: the container shares the host network, so without this
# the nodes would also discover (and be discovered by) ROS 2 nodes elsewhere on the LAN
ENV ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST

# source ROS in every interactive shell
RUN echo 'source /opt/ros/jazzy/setup.bash' >> /root/.bashrc

COPY entrypoint.sh /usr/local/bin/entrypoint.sh

# start all shells (including exec) in the mounted repository
WORKDIR /work

ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]

# open a shell by default
CMD ["bash"]
