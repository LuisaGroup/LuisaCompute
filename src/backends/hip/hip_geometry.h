//
// Created by mike on 4/8/26.
//

#pragma once

#include <hiprt/hiprt.h>

#include <luisa/runtime/rhi/resource.h>

namespace luisa::compute::hip {

class HIPCommandEncoder;

[[nodiscard]] inline hiprtBuildFlags make_hiprt_build_flags(
    const AccelOption &option) noexcept {
    hiprtBuildFlags flags = option.hint == AccelOption::UsageHint::FAST_BUILD ?
                                hiprtBuildFlagBitPreferFastBuild :
                                hiprtBuildFlagBitPreferHighQualityBuild;
    if (option.allow_update) {
        // HIPRT refits require a topology-preserving builder configuration.
        flags |= hiprtBuildFlagBitDisableSpatialSplits |
                 hiprtBuildFlagBitDisableOrientedBoundingBoxes;
    }
    return flags;
}

/// Common binding interface for HIP geometries and nested motion scenes.
class HIPPrimitive {
public:
    enum class Kind : uint32_t {
        TRIANGLE = 0u,
        PROCEDURAL = 1u,
        CURVE = 2u,
        MOTION_TRIANGLE = 3u,
    };

    struct Binding {
        hiprtInstance instance{};
        Kind kind{};
        uint64_t codegen_handle{};
        uint64_t motion_data{};
    };

    virtual ~HIPPrimitive() noexcept = default;
    [[nodiscard]] virtual Binding binding() const noexcept = 0;
    virtual void prepare_for_tlas_build(HIPCommandEncoder &) const noexcept {}
};

/// Common base for concrete HIP geometry resources.
class HIPGeometry : public HIPPrimitive {
public:
    ~HIPGeometry() noexcept override = default;
    [[nodiscard]] virtual hiprtGeometry handle() const noexcept = 0;
    [[nodiscard]] virtual Kind kind() const noexcept = 0;
    /// Device-visible geometry data stored in HIPAccel::CodegenInstance.
    /// Static triangle/procedural geometry does not currently consume this field
    /// and therefore uses its HIPRT geometry handle. Curves and deforming meshes
    /// override it with pointers to their device-side geometry descriptors.
    [[nodiscard]] virtual uint64_t codegen_handle() const noexcept {
        return reinterpret_cast<uint64_t>(handle());
    }

    [[nodiscard]] Binding binding() const noexcept final {
        hiprtInstance instance{};
        instance.type = hiprtInstanceTypeGeometry;
        instance.geometry = handle();
        return Binding{
            .instance = instance,
            .kind = kind(),
            .codegen_handle = codegen_handle(),
            .motion_data = 0u};
    }
};

}// namespace luisa::compute::hip
