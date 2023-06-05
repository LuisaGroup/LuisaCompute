#pragma once

#include <core/basic_types.h>

namespace luisa::compute {
enum class UpdateCommand : uint {
    Map,
    UnMap
};
struct UpdateTile {
    uint3 start_coord;
    uint3 size;
    uint mip_level;
    UpdateCommand update_cmd;
};
}// namespace luisa::compute