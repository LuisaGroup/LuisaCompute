#pragma once

#include <array>

namespace luisa::compute {

struct AABB {
    std::array<float, 3> packed_min;
    std::array<float, 3> packed_max;
};

}// namespace luisa::compute
