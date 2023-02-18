//
// Created by Mike Smith on 2021/6/24.
//
#pragma once

#include <array>

namespace luisa::compute {

struct alignas(16) Ray {
    std::array<float, 3> compressed_origin;
    float compressed_t_min;
    std::array<float, 3> compressed_direction;
    float compressed_t_max;
};

}// namespace luisa::compute
