#!/bin/bash

uname -a

curl https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.0-1_all.deb -O
dpkg -i cuda-keyring_1.0-1_all.deb
apt-get update
apt-get -y install cuda libvulkan-dev cmake uuid-dev libglfw3-dev libxinerama-dev libxcursor-dev libxi-dev
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y