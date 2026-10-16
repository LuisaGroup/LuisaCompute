#pragma once

#include <cstddef>

#include <hiprt/hiprt.h>

#include <luisa/runtime/rhi/resource.h>

namespace luisa::compute::hip {

struct HIPGeometryBuildPolicy {
    hiprtBuildOptions options{};
    size_t temporary_buffer_size{};
    size_t high_quality_temporary_buffer_size{};
    size_t free_memory_size{};
    bool memory_constrained{};
};

[[nodiscard]] HIPGeometryBuildPolicy select_hiprt_geometry_build_policy(
    hiprtContext context,
    const hiprtGeometryBuildInput &input,
    const AccelOption &option) noexcept;

}// namespace luisa::compute::hip
