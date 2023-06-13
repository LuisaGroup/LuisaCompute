#!/bin/bash

export CIBW_MANYLINUX_X86_64_IMAGE=manylinux_2_28
export CIBW_BEFORE_ALL="./scripts/cibw_install_deps.sh"


