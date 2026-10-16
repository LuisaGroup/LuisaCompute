#pragma once

#include <luisa/runtime/rtx/mesh.h>

#include "simd_primitive.h"

namespace luisa::compute::simd {

class SIMDMesh final : public SIMDPrimitive {

private:
    RTCScene _scene{nullptr};
    RTCGeometry _geometry{nullptr};
    AccelMotionOption _motion{};

public:
    SIMDMesh(RTCDevice device, const AccelOption &option) noexcept;
    ~SIMDMesh() noexcept;

    void build(const MeshBuildCommand &command) noexcept;
    [[nodiscard]] RTCScene handle() const noexcept override { return _scene; }
};

}// namespace luisa::compute::simd
