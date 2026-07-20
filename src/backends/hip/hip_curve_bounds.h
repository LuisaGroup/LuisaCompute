//
// Conservative host-side bounds for HIP curve primitives.
//

#pragma once

#include <luisa/core/basic_types.h>
#include <luisa/runtime/rhi/curve_basis.h>

namespace luisa::compute::hip {

struct HIPCurveAABB {
    float min[3];
    float max[3];
};

static_assert(sizeof(HIPCurveAABB) == sizeof(float) * 6u);

[[nodiscard]] HIPCurveAABB compute_hip_curve_aabb(
    CurveBasis basis, const float4 *control_points) noexcept;

}// namespace luisa::compute::hip
