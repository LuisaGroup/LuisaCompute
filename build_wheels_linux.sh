#!/bin/bash

export CIBW_ARCHS=auto64
export CIBW_BUILD="*manylinux*"
export CIBW_MANYLINUX_X86_64_IMAGE=manylinux_2_28
export CIBW_BEFORE_ALL="./scripts/cibw_install_deps.sh"
export CIBW_BUILD_VERBOSITY=2
export CIBW_REPAIR_WHEEL_COMMAND="auditwheel show {wheel} && auditwheel repair -w {dest_dir} {wheel} --exclude libcuda.so.1 --exclude libvulkan.so.1"

cibuildwheel --output-dir wheelhouse --platform linux
