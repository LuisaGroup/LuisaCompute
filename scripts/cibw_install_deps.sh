#!/bin/bash

# print release information
cat /etc/os-release

# install gcc-11 and g++-11
yum install -y centos-release-scl
yum install -y devtoolset-11-gcc devtoolset-11-gcc-c++

# install python 3.10 and 3.11 with development libraries
yum install -y python310 python311 python310-devel python311-devel

# install cuda
yum install -y https://dl.fedoraproject.org/pub/epel/epel-release-latest-7.noarch.rpm
yum-config-manager --add-repo https://developer.download.nvidia.com/compute/cuda/repos/rhel7/x86_64/cuda-rhel7.repo
yum install -y cuda

# install rust
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
