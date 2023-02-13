#pragma once

#include <dsl/struct.h>

namespace luisa::compute {

struct AABB {
    std::array<float, 3> packed_min;
    std::array<float, 3> packed_max;
};

}// namespace luisa::compute

LUISA_STRUCT(luisa::compute::AABB, packed_min, packed_max)
