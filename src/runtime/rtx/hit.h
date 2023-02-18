//
// Created by Mike Smith on 2021/6/24.
//

#pragma once

#include <core/basic_types.h>

namespace luisa::compute {

enum class HitType : uint8_t {
    Miss = 0,
    Triangle = 1,
    Procedural = 2
};

struct Hit {
    uint inst;
    uint prim;
    float2 bary;
    uint hit_type;
    float committed_ray_t;
};

}// namespace luisa::compute
