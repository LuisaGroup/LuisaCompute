#pragma once

#include <cstdint>

namespace luisa::compute {

enum class CurveBasis : uint32_t {
    PIECEWISE_LINEAR,
    QUADRATIC_BSPLINE,
    CUBIC_BSPLINE,
    CATMULL_ROM,
    BEZIER
};

}// namespace luisa::compute
