#!/bin/bash

# print release information
cat /etc/os-release

# install vulkan and other dependencies
dnf install -y vulkan-devel libuuid-devel libXinerama-devel libXcursor-devel libXi-devel libXrandr-devel

# install rust
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y

# install cuda
dnf install -y https://dl.fedoraproject.org/pub/epel/epel-release-latest-8.noarch.rpm
dnf config-manager --add-repo https://developer.download.nvidia.com/compute/cuda/repos/rhel8/x86_64/cuda-rhel8.repo
dnf clean expire-cache
dnf install -y cuda
