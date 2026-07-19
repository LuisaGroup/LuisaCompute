//
// Conservative host-side bounds for HIP curve primitives.
//

#include <algorithm>
#include <cmath>
#include <limits>

#include <luisa/core/logging.h>

#include "hip_curve_bounds.h"

namespace luisa::compute::hip {

namespace {

struct Double4 {
    double x;
    double y;
    double z;
    double w;
};

[[nodiscard]] Double4 to_double4(float4 v) noexcept {
    return {v.x, v.y, v.z, v.w};
}

[[nodiscard]] Double4 add(Double4 a, Double4 b) noexcept {
    return {a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w};
}

[[nodiscard]] Double4 mul(Double4 v, double s) noexcept {
    return {v.x * s, v.y * s, v.z * s, v.w * s};
}

[[nodiscard]] float outward_float(double value, bool lower) noexcept {
    constexpr auto float_max = static_cast<double>(std::numeric_limits<float>::max());
    LUISA_ASSERT(std::isfinite(value) && value >= -float_max && value <= float_max,
                 "HIP curve bound {} is not finite and representable as float.", value);
    auto result = static_cast<float>(value);
    result = std::nextafter(
        result, lower ? -std::numeric_limits<float>::infinity() :
                        std::numeric_limits<float>::infinity());
    LUISA_ASSERT(std::isfinite(result),
                 "HIP curve bound {} overflows after outward rounding.", value);
    return result;
}

}// namespace

HIPCurveAABB compute_hip_curve_aabb(
    CurveBasis basis, const float4 *control_points) noexcept {
    LUISA_ASSERT(control_points != nullptr, "HIP curve control points must not be null.");
    auto cp_count = segment_control_point_count(basis);
    LUISA_ASSERT(cp_count != 0u, "Invalid curve basis 0x{:x}.", luisa::to_underlying(basis));

    Double4 p[4]{};
    auto scale = 1.0e-7;
    for (auto i = 0u; i < cp_count; i++) {
        auto v = control_points[i];
        LUISA_ASSERT(std::isfinite(v.x) && std::isfinite(v.y) &&
                         std::isfinite(v.z) && std::isfinite(v.w),
                     "HIP curve control point {} contains a non-finite component.", i);
        p[i] = to_double4(v);
        scale = std::max({scale, std::abs(p[i].x), std::abs(p[i].y),
                          std::abs(p[i].z), std::abs(p[i].w)});
    }

    // Convert the segment to Bézier form. Bernstein basis functions are
    // non-negative and sum to one on [0, 1], so their control-point hull
    // encloses the complete center curve and the absolute radius is bounded
    // by the largest absolute radius control point.
    Double4 b[4]{};
    auto bezier_count = 4u;
    switch (basis) {
        case CurveBasis::PIECEWISE_LINEAR:
            b[0] = p[0];
            b[1] = p[1];
            bezier_count = 2u;
            break;
        case CurveBasis::CUBIC_BSPLINE:
            b[0] = mul(add(add(p[0], mul(p[1], 4.0)), p[2]), 1.0 / 6.0);
            b[1] = mul(add(mul(p[1], 2.0), p[2]), 1.0 / 3.0);
            b[2] = mul(add(p[1], mul(p[2], 2.0)), 1.0 / 3.0);
            b[3] = mul(add(add(p[1], mul(p[2], 4.0)), p[3]), 1.0 / 6.0);
            break;
        case CurveBasis::CATMULL_ROM:
            b[0] = p[1];
            b[1] = add(p[1], mul(add(p[2], mul(p[0], -1.0)), 1.0 / 6.0));
            b[2] = add(p[2], mul(add(p[3], mul(p[1], -1.0)), -1.0 / 6.0));
            b[3] = p[2];
            break;
        case CurveBasis::BEZIER:
            for (auto i = 0u; i < 4u; i++) { b[i] = p[i]; }
            break;
    }

    auto lo_x = std::numeric_limits<double>::infinity();
    auto lo_y = std::numeric_limits<double>::infinity();
    auto lo_z = std::numeric_limits<double>::infinity();
    auto hi_x = -std::numeric_limits<double>::infinity();
    auto hi_y = -std::numeric_limits<double>::infinity();
    auto hi_z = -std::numeric_limits<double>::infinity();
    auto max_radius = 1.0e-7;
    for (auto i = 0u; i < bezier_count; i++) {
        lo_x = std::min(lo_x, b[i].x);
        lo_y = std::min(lo_y, b[i].y);
        lo_z = std::min(lo_z, b[i].z);
        hi_x = std::max(hi_x, b[i].x);
        hi_y = std::max(hi_y, b[i].y);
        hi_z = std::max(hi_z, b[i].z);
        max_radius = std::max(max_radius, std::abs(b[i].w));
        scale = std::max({scale, std::abs(b[i].x), std::abs(b[i].y),
                          std::abs(b[i].z), std::abs(b[i].w)});
    }

    // The device evaluator uses float power-basis arithmetic for B-spline and
    // Catmull-Rom curves. Cover its rounding error, as well as the subsequent
    // capsule-radius arithmetic, before rounding every bound outward again.
    constexpr auto arithmetic_guard =
        1024.0 * static_cast<double>(std::numeric_limits<float>::epsilon());
    auto guard = scale * arithmetic_guard;
    HIPCurveAABB result{};
    result.min[0] = outward_float(lo_x - max_radius - guard, true);
    result.min[1] = outward_float(lo_y - max_radius - guard, true);
    result.min[2] = outward_float(lo_z - max_radius - guard, true);
    result.max[0] = outward_float(hi_x + max_radius + guard, false);
    result.max[1] = outward_float(hi_y + max_radius + guard, false);
    result.max[2] = outward_float(hi_z + max_radius + guard, false);
    return result;
}

}// namespace luisa::compute::hip
