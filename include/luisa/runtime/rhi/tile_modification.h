#pragma once

#include <luisa/core/basic_types.h>
#include <luisa/core/stl/variant.h>
#include <luisa/core/stl/vector.h>
namespace luisa::compute {
class DeviceInterface;
struct SparseTextureMapOperation {
    uint3 start_tile;
    uint3 tile_count;
    uint mip_level;
};

struct SparseTextureUnMapOperation {
    uint3 start_tile;
    uint mip_level;
};

struct SparseBufferMapOperation {
    uint start_tile;
    uint tile_count;
};

struct SparseBufferUnMapOperation {
    uint start_tile;
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
