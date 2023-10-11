#pragma once

#include <luisa/core/mathematics.h>

namespace luisa::compute {

struct Viewport {
    float2 start;
    float2 size;
    Viewport(float2 start, float2 size) : start(start), size(size) {}
    Viewport(float start_x, float start_y, float size_x, float size_y) : start(float2(start_x, start_y)), size(float2(size_x, size_y)) {}
};

}// namespace luisa::compute
