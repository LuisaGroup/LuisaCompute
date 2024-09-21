//
// Created by Mike on 2024/9/21.
//

#pragma once

#include <luisa/runtime/device.h>

namespace luisa::compute {

class Curve;
class Mesh;
class ProceduralPrimitive;

class LC_RUNTIME_API MotionInstance : public Resource {

    friend class Device;

private:
    uint64_t _child_handle;
    AccelMotionMode _mode;
    luisa::vector<MotionInstanceTransform> _transform_keyframes;

private:
    // for internal use only; does not have linkages
    MotionInstance(DeviceInterface *device,
                   const Resource &resource,
                   const AccelMotionOption &option) noexcept;

private:
    MotionInstance(DeviceInterface *device,
                   const Mesh &mesh,
                   const AccelMotionOption &option) noexcept;

    MotionInstance(DeviceInterface *device,
                   const Curve &curve,
                   const AccelMotionOption &option) noexcept;

    MotionInstance(DeviceInterface *device,
                   const ProceduralPrimitive &primitive,
                   const AccelMotionOption &option) noexcept;

public:
    void set_keyframe(size_t index, const MotionInstanceTransform &transform) noexcept;
    void set_keyframe(size_t index, const MotionInstanceTransformMatrix &transform) noexcept;
    void set_keyframe(size_t index, const MotionInstanceTransformSRT &transform) noexcept;
    void set_keyframes(luisa::span<const MotionInstanceTransform> transforms) noexcept;
    void set_keyframes(luisa::span<const MotionInstanceTransformMatrix> transforms) noexcept;
    void set_keyframes(luisa::span<const MotionInstanceTransformSRT> transforms) noexcept;

    [[nodiscard]] const MotionInstanceTransform &keyframe(size_t index) const noexcept;
    [[nodiscard]] const MotionInstanceTransformMatrix &keyframe_matrix(size_t index) const noexcept;
    [[nodiscard]] const MotionInstanceTransformSRT &keyframe_srt(size_t index) const noexcept;

    [[nodiscard]] luisa::span<const MotionInstanceTransform> keyframes() const noexcept;
    [[nodiscard]] luisa::span<const MotionInstanceTransformMatrix> keyframes_matrix() const noexcept;
    [[nodiscard]] luisa::span<const MotionInstanceTransformSRT> keyframes_srt() const noexcept;

    [[nodiscard]] auto keyframe_count() const noexcept { return _transform_keyframes.size(); }
    [[nodiscard]] auto mode() const noexcept { return _mode; }

    [[nodiscard]] luisa::unique_ptr<Command> build() noexcept;
};

}// namespace luisa::compute
