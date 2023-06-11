#pragma once

#include <luisa/core/basic_types.h>
#include <luisa/core/stl/variant.h>
namespace luisa::compute {

struct SparseTextureMapOperation {
    uint3 start_tile;
    uint3 tile_count;
    uint mip_level;
};

struct SparseTextureUnMapOperation {
    uint3 start_tile;
    uint mip_level;
};

using SparseTextureOperation = luisa::variant<SparseTextureMapOperation, SparseTextureUnMapOperation>;

struct SparseBufferMapOperation {
    uint start_tile;
    uint tile_count;
};

struct SparseBufferUnMapOperation {
    uint start_tile;
};

using SparseBufferOperation = luisa::variant<SparseBufferMapOperation, SparseBufferUnMapOperation>;
}// namespace luisa::compute
