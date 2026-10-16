#pragma once

#include <luisa/runtime/rtx/procedural_primitive.h>

#include "simd_primitive.h"

namespace luisa::compute::simd {

// Embree invokes these callbacks for both direct traversal and ray queries.
// Direct traversal rejects user geometry; an active SIMD query scan records a
// stable procedural candidate and resumes the ordinary generated CFG handler.
void simd_procedural_intersect(
    const RTCIntersectFunctionNArguments *arguments) noexcept;
void simd_procedural_occluded(
    const RTCOccludedFunctionNArguments *arguments) noexcept;

class SIMDProceduralPrimitive final : public SIMDPrimitive {

private:
    RTCScene _scene{nullptr};
    RTCGeometry _geometry{nullptr};
    AccelMotionOption _motion{};

public:
    SIMDProceduralPrimitive(
        RTCDevice device, const AccelOption &option) noexcept;
    ~SIMDProceduralPrimitive() noexcept;

    void build(const ProceduralPrimitiveBuildCommand &command) noexcept;
    [[nodiscard]] RTCScene handle() const noexcept override { return _scene; }
};

}// namespace luisa::compute::simd
