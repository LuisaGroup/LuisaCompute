//
// Created by mike on 4/8/26.
//

#pragma once

#include <hip/hip_runtime.h>
#include <hiprt/hiprt.h>

#include <luisa/core/spin_mutex.h>
#include <luisa/runtime/rtx/procedural_primitive.h>

#include "hip_geometry.h"

namespace luisa::compute::hip {

class HIPCommandEncoder;

class HIPProceduralPrimitive : public HIPGeometry {

private:
    AccelOption _option;
    hiprtContext _hiprt_ctx{nullptr};
    hiprtGeometry _geometry{nullptr};
    hiprtBuildFlags _build_flags{};
    hipDeviceptr_t _aabb_buffer{};
    size_t _aabb_buffer_size{};
    mutable spin_mutex _mutex;

public:
    explicit HIPProceduralPrimitive(hiprtContext ctx, const AccelOption &option) noexcept;
    ~HIPProceduralPrimitive() noexcept;
    void build(HIPCommandEncoder &encoder, ProceduralPrimitiveBuildCommand *command) noexcept;
    [[nodiscard]] hiprtGeometry handle() const noexcept override {
        std::scoped_lock lock{_mutex};
        return _geometry;
    }
    [[nodiscard]] Kind kind() const noexcept override { return Kind::PROCEDURAL; }
    [[nodiscard]] auto option() const noexcept { return _option; }
};

}// namespace luisa::compute::hip
