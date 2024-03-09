#pragma once

#include <luisa/core/mathematics.h>

namespace luisa::compute {

struct Viewport {
    uint2 start;
    uint2 size;
    Viewport(uint2 start, uint2 size) : start(start), size(size) {}
    Viewport(uint start_x, uint start_y, uint size_x, uint size_y) : start(uint2(start_x, start_y)), size(uint2(size_x, size_y)) {}
};

}// namespace luisa::compute
