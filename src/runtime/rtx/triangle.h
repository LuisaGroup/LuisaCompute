//
// Created by Mike Smith on 2023/2/13.
//

#pragma once

#include <dsl/struct.h>

namespace luisa::compute {

struct Triangle {
    uint i0;
    uint i1;
    uint i2;
};

}// namespace luisa::compute

LUISA_STRUCT(luisa::compute::Triangle, i0, i1, i2)
