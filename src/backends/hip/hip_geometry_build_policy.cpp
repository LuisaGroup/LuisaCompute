#include "hip_geometry_build_policy.h"

#include "hip_check.h"
#include "hip_geometry.h"
#include "hip_geometry_memory_policy.h"

namespace luisa::compute::hip {

HIPGeometryBuildPolicy select_hiprt_geometry_build_policy(
    hiprtContext context,
    const hiprtGeometryBuildInput &input,
    const AccelOption &option) noexcept {
    HIPGeometryBuildPolicy result;
    result.options.buildFlags = make_hiprt_build_flags(option);
    LUISA_CHECK_HIPRT(hiprtGetGeometryBuildTemporaryBufferSize(
        context,
        input,
        result.options,
        result.temporary_buffer_size));
    if (option.hint != AccelOption::UsageHint::FAST_TRACE) {
        return result;
    }

    result.high_quality_temporary_buffer_size =
        result.temporary_buffer_size;
    size_t total_memory_size{};
    LUISA_CHECK_HIP(hipMemGetInfo(
        &result.free_memory_size,
        &total_memory_size));
    static_cast<void>(total_memory_size);
    result.memory_constrained =
        hiprt_high_quality_scratch_exceeds_budget(
            result.high_quality_temporary_buffer_size,
            result.free_memory_size);
    if (!result.memory_constrained) {
        return result;
    }

    result.options.buildFlags = make_hiprt_build_flags(
        option, hiprtBuildFlagBitPreferBalancedBuild);
    LUISA_CHECK_HIPRT(hiprtGetGeometryBuildTemporaryBufferSize(
        context,
        input,
        result.options,
        result.temporary_buffer_size));
    LUISA_INFO(
        "HIPRT geometry build selected BalancedBuild under memory pressure "
        "(free = {} bytes, high-quality scratch = {} bytes, selected scratch "
        "= {} bytes).",
        result.free_memory_size,
        result.high_quality_temporary_buffer_size,
        result.temporary_buffer_size);
    return result;
}

}// namespace luisa::compute::hip
