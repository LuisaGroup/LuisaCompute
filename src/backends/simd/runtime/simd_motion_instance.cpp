#include "simd_motion_instance.h"

#include <cmath>

#include <luisa/core/logging.h>

namespace luisa::compute::simd {

SIMDMotionInstance::SIMDMotionInstance(
    const AccelMotionOption &option) noexcept
    : SIMDPrimitive{Kind::motion_instance}, _option{option} {
    LUISA_ASSERT(
        option.mode == AccelMotionMode::MATRIX ||
            option.mode == AccelMotionMode::SRT,
        "SIMD motion instances only support MATRIX and SRT transforms.");
    LUISA_ASSERT(
        option.keyframe_count >= 2u &&
            option.keyframe_count <= RTC_MAX_TIME_STEP_COUNT,
        "SIMD motion instance keyframe count must be in [2, {}] (got {}).",
        RTC_MAX_TIME_STEP_COUNT, option.keyframe_count);
    LUISA_ASSERT(
        std::isfinite(option.time_start) &&
            std::isfinite(option.time_end) &&
            option.time_start < option.time_end,
        "SIMD motion instance time range must be finite and strictly "
        "increasing (got [{}, {}]).",
        option.time_start, option.time_end);
    LUISA_ASSERT(
        option.time_start <= 0.0f || option.should_vanish_start,
        "SIMD motion instances require should_vanish_start when the "
        "first keyframe is inside the camera shutter interval.");
    LUISA_ASSERT(
        option.time_end >= 1.0f || option.should_vanish_end,
        "SIMD motion instances require should_vanish_end when the "
        "last keyframe is inside the camera shutter interval.");
}

void SIMDMotionInstance::build(
    const MotionInstanceBuildCommand &command) noexcept {
    LUISA_ASSERT(
        command.keyframes().size() == _option.keyframe_count,
        "SIMD motion instance keyframe count mismatch (expected {}, got {}).",
        _option.keyframe_count, command.keyframes().size());
    auto *child = reinterpret_cast<SIMDPrimitive *>(command.child());
    LUISA_ASSERT(
        child != nullptr &&
            (child->kind() == Kind::mesh ||
             child->kind() == Kind::curve ||
             child->kind() == Kind::procedural),
        "SIMD motion instances require a mesh, curve, or procedural child.");
    _child = child;
    _keyframes.assign(
        command.keyframes().begin(), command.keyframes().end());
    _build_version++;
}

RTCScene SIMDMotionInstance::handle() const noexcept {
    return _child == nullptr ? nullptr : _child->handle();
}

}// namespace luisa::compute::simd
