#pragma once

#include <core/basic_types.h>

namespace luisa::compute {
struct TileModification {
    enum struct Operation : uint {
        Map,
        UnMap
    };
    uint3 start_coord;
    uint3 size;
    uint mip_level;
    Operation operation;
};
}// namespace luisa::compute