//
// Created by Mike Smith on 2021/6/24.
//
#pragma once

#include <dsl/struct.h>
#include <core/mathematics.h>
#include <core/stl/format.h>

namespace luisa::compute {

struct alignas(16) Ray {
    std::array<float, 3> compressed_origin;
    float compressed_t_min;
    std::array<float, 3> compressed_direction;
    float compressed_t_max;
};

}// namespace luisa::compute

LUISA_STRUCT(
    luisa::compute::Ray,
    compressed_origin,
    compressed_t_min,
    compressed_direction,
    compressed_t_max)