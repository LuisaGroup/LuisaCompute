#pragma once

#include <core/basic_types.h>

namespace luisa::compute {
enum struct TileOperation : uint {
    Map,
    UnMap
};
struct SparseTexModification {
    using Operation = TileOperation;
    uint3 start_tile;
    uint3 tile_size;
    uint mip_level;
    Operation operation;
};

struct SparseBufferModification {
    using Operation = TileOperation;
    // In tile
    size_t offset;
    // In tile
    size_t size;
    Operation operation;
};
}// namespace luisa::compute