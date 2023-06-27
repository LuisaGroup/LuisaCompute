#pragma once

#include <luisa/core/basic_types.h>
#include <luisa/core/stl/variant.h>
#include <luisa/core/stl/vector.h>
namespace luisa::compute {
class DeviceInterface;
struct SparseTextureMapOperation {
    uint3 start_tile;
    uint3 tile_count;
    uint64_t allocated_heap;
    uint mip_level;
};

struct SparseTextureUnMapOperation {
    uint3 start_tile;
    uint3 tile_count;
    uint mip_level;
};

struct SparseBufferMapOperation {
    uint64_t allocated_heap;
    uint start_tile;
    uint tile_count;
};

struct SparseBufferUnMapOperation {
    uint start_tile;
    uint tile_count;
};

using SparseOperation = luisa::variant<
    SparseTextureMapOperation,
    SparseTextureUnMapOperation,
    SparseBufferMapOperation,
    SparseBufferUnMapOperation>;

struct SparseUpdateTile {
    uint64_t handle;
    SparseOperation operations;
};
}// namespace luisa::compute
