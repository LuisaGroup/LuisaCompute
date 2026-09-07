#pragma once

#include <luisa/runtime/rtx/curve.h>

#include "simd_primitive.h"

namespace luisa::compute::simd {

class SIMDCurve final : public SIMDPrimitive {

private:
    RTCScene _scene{nullptr};
    RTCGeometry _geometry{nullptr};
    AccelMotionOption _motion{};
    CurveBasis _basis{};

public:
    SIMDCurve(RTCDevice device, const AccelOption &option) noexcept;
    ~SIMDCurve() noexcept;

    void build(const CurveBuildCommand &command) noexcept;
    [[nodiscard]] RTCScene handle() const noexcept override { return _scene; }
};

}// namespace luisa::compute::simd
