//
// Created by Mike Smith on 2021/6/24.
//

#pragma once

#include <dsl/struct.h>
#include <core/mathematics.h>
#include <core/stl/format.h>

namespace luisa::compute {

enum class HitType : uint8_t {
    Miss = 0,
    Triangle = 1,
    Procedural = 2
};

struct Hit {
    uint inst{0u};
    uint prim{0u};
    float2 bary;
    uint hit_type;
    float committed_ray_t;
};

}// namespace luisa::compute

LUISA_STRUCT(luisa::compute::Hit, inst, prim, bary, hit_type, committed_ray_t)
