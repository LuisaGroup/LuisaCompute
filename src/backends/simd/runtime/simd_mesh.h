#pragma once

#include <luisa/runtime/rtx/mesh.h>

#include "simd_embree.h"

namespace luisa::compute::simd {

class SIMDMesh {

private:
    RTCScene _scene{nullptr};
    RTCGeometry _geometry{nullptr};
    AccelMotionOption _motion{};

public:
    SIMDMesh(RTCDevice device, const AccelOption &option) noexcept;
    ~SIMDMesh() noexcept;

    void build(const MeshBuildCommand &command) noexcept;
    [[nodiscard]] auto handle() const noexcept { return _scene; }
};

}// namespace luisa::compute::simd
