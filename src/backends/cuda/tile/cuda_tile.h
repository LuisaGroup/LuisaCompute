#pragma once

// CUDA native-Tile factory entry point. The implementation lives in
// cuda_tile.cpp and defines CUDADevice::create_tile_kernel; this header gives
// the tile sources a stable include point and documents the boundary that the
// backend owns (lowering + PTX compilation), mirroring
// src/backends/metal/tile/metal_tile_codegen.h.
namespace luisa::compute::cuda {
class CUDADevice;
}// namespace luisa::compute::cuda
