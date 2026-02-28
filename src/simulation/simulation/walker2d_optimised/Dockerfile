# Use ROS2 Humble as base (Ubuntu 22.04 Jammy)
FROM osrf/ros:humble-desktop

# Set environment variables to avoid interactive prompts during build
ENV DEBIAN_FRONTEND=noninteractive
ENV ROS_DISTRO=humble

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-colcon-common-extensions \
    python3-rosdep \
    build-essential \
    git \
    curl \
    libglew-dev \
    libosmesa6-dev \
    libgl1-mesa-glx \
    libglfw3 \
    patchelf \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN python3 -m pip install --upgrade pip

# Using CPU here to keep it simple, but for heavy RL you'll want the nvidia-docker base
RUN apt-get remove -y python3-sympy && \
    apt-get autoremove -y

# Install PyTorch (CPU version for sim, change to CUDA if using NVIDIA runtime)
RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Install RL, MuJoCo, and ROS2 bridging packages
RUN pip3 install \
    stable-baselines3[extra] \
    gymnasium[mujoco] \
    mujoco \
    tensorboard \
    setuptools==58.2.0 # Often needed to avoid colcon build warnings in Humble

# Create ROS2 workspace
RUN mkdir -p /ros2_ws/src
WORKDIR /ros2_ws

# Setup entrypoint to automatically source ROS2
RUN echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc
RUN echo "if [ -f /ros2_ws/install/setup.bash ]; then source /ros2_ws/install/setup.bash; fi" >> ~/.bashrc

# Copy your source code into the container (assuming your code is in a folder named 'src' on your host)
COPY ./src ./src

# Default command
CMD ["python3", "src/test.py"]