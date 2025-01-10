#!/bin/bash

# Print release information
cat /etc/os-release

# Detect system architecture
ARCH=$(uname -m)

# Install Vulkan and other dependencies
dnf install -y https://dl.fedoraproject.org/pub/epel/epel-release-latest-8.noarch.rpm
dnf update -y
dnf install -y vulkan-devel libuuid-devel libXinerama-devel libXcursor-devel libXi-devel libXrandr-devel libxkbcommon-devel wayland-devel patchelf

# Install Rust
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y

# Install CUDA (only for x86_64, since CUDA is not officially supported on aarch64)
if [[ "$ARCH" == "x86_64" ]]; then
    dnf config-manager --add-repo https://developer.download.nvidia.com/compute/cuda/repos/rhel8/x86_64/cuda-rhel8.repo
    dnf clean expire-cache
    dnf install -y cuda-toolkit-12-8
elif [[ "$ARCH" == "aarch64" ]]; then
    dnf config-manager --add-repo https://developer.download.nvidia.com/compute/cuda/repos/rhel8/sbsa/cuda-rhel8.repo
    dnf clean expire-cache
    dnf install -y cuda-toolkit-12-8
fi