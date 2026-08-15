#pragma once

#include <array>

#include <luisa/core/stl.h>
#include <luisa/runtime/rtx/motion_instance.h>

#include "simd_embree.h"

namespace luisa::compute::simd {

// Embree builds configured with a one-entry instance-ID stack cannot express
// the public outer-affine -> quaternion-SRT -> BLAS hierarchy as two native
// instance geometries. This helper implements the outer level as a user
// geometry: its callbacks inverse-transform one complete ray packet and enter
// the child BLAS through Embree 4's forwarding ABI or Embree 3's supported
// recursive packet traversal, while consuming only the one public instance-ID
// level.
class SIMDSRTMotionForwarder final {

private:
    RTCScene _transform_scene{nullptr};
    RTCGeometry _transform_geometry{nullptr};
    RTCScene _child_scene{nullptr};
    std::array<float, 12u> _outer{};
    std::array<RTCBounds, 2u> _bounds{};
    float _time_start{0.0f};
    float _time_end{1.0f};
    float _inverse_time_extent{1.0f};

private:
    static void _bounds_callback(
        const RTCBoundsFunctionArguments *arguments) noexcept;
    static void _intersect_callback(
        const RTCIntersectFunctionNArguments *arguments) noexcept;
    static void _occluded_callback(
        const RTCOccludedFunctionNArguments *arguments) noexcept;

public:
    SIMDSRTMotionForwarder(
        RTCDevice device, RTCScene child_scene,
        const AccelMotionOption &option,
        luisa::span<const MotionInstanceTransform> keyframes,
        const float outer[12]) noexcept;
    ~SIMDSRTMotionForwarder() noexcept;

    SIMDSRTMotionForwarder(
        const SIMDSRTMotionForwarder &) = delete;
    SIMDSRTMotionForwarder(
        SIMDSRTMotionForwarder &&) = delete;
    SIMDSRTMotionForwarder &operator=(
        const SIMDSRTMotionForwarder &) = delete;
    SIMDSRTMotionForwarder &operator=(
        SIMDSRTMotionForwarder &&) = delete;

    [[nodiscard]] auto transform_geometry() const noexcept {
        return _transform_geometry;
    }
    [[nodiscard]] auto child_scene() const noexcept {
        return _child_scene;
    }
    [[nodiscard]] auto outer() const noexcept {
        return _outer.data();
    }
    [[nodiscard]] auto time_start() const noexcept {
        return _time_start;
    }
    [[nodiscard]] auto inverse_time_extent() const noexcept {
        return _inverse_time_extent;
    }

    void configure_geometry(RTCGeometry geometry) noexcept;
};

}// namespace luisa::compute::simd
