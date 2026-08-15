#pragma once

#include <cstdint>

#include <luisa/core/stl/vector.h>
#include <luisa/runtime/rtx/motion_instance.h>

#include "simd_primitive.h"

namespace luisa::compute::simd {

class SIMDMotionInstance final : public SIMDPrimitive {

private:
    AccelMotionOption _option{};
    SIMDPrimitive *_child{nullptr};
    luisa::vector<MotionInstanceTransform> _keyframes;
    uint64_t _build_version{0u};

public:
    explicit SIMDMotionInstance(const AccelMotionOption &option) noexcept;
    void build(const MotionInstanceBuildCommand &command) noexcept;
    [[nodiscard]] RTCScene handle() const noexcept override;
    [[nodiscard]] auto option() const noexcept { return _option; }
    [[nodiscard]] auto child() const noexcept { return _child; }
    [[nodiscard]] auto build_version() const noexcept {
        return _build_version;
    }
    [[nodiscard]] auto keyframes() const noexcept {
        return luisa::span{_keyframes};
    }
};

}// namespace luisa::compute::simd
