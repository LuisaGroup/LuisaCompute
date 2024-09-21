//
// Created by Mike on 2024/9/21.
//

#include <luisa/core/logging.h>
#include <luisa/runtime/rtx/mesh.h>
#include <luisa/runtime/rtx/curve.h>
#include <luisa/runtime/rtx/procedural_primitive.h>
#include <luisa/runtime/rtx/motion_instance.h>

namespace luisa::compute {

MotionInstance::MotionInstance(DeviceInterface *device,
                               const Resource &resource,
                               const AccelMotionOption &option) noexcept
    : Resource{device, Tag::MOTION_INSTANCE, {}},
      _mode{option.mode} {
    LUISA_ASSERT(resource &&
                     (resource.tag() == Tag::MESH ||
                      resource.tag() == Tag::CURVE ||
                      resource.tag() == Tag::PROCEDURAL_PRIMITIVE),
                 "Invalid resource type for motion instance.");
    switch (_mode) {
        case AccelOption::MotionMode::STATIC: {
            LUISA_ASSERT(option.keyframe_count == 1u,
                         "Only one keyframe is allowed for static motion (got {}).",
                         option.keyframe_count);
            break;
        }
        case AccelOption::MotionMode::MATRIX: [[fallthrough]];
        case AccelOption::MotionMode::SRT: {
            LUISA_ASSERT(option.keyframe_count >= 2u,
                         "At least two keyframes are required for non-static motion (got {}).",
                         option.keyframe_count);
            _transform_keyframes.resize(option.keyframe_count);
            break;
        }
    }
}

MotionInstance::MotionInstance(DeviceInterface *device,
                               const Mesh &mesh,
                               const AccelMotionOption &option) noexcept
    : MotionInstance{device, static_cast<const Resource &>(mesh), option} {}

MotionInstance::MotionInstance(DeviceInterface *device,
                               const Curve &curve,
                               const AccelMotionOption &option) noexcept
    : MotionInstance{device, static_cast<const Resource &>(curve), option} {}

MotionInstance::MotionInstance(DeviceInterface *device,
                               const ProceduralPrimitive &primitive,
                               const AccelMotionOption &option) noexcept
    : MotionInstance{device, static_cast<const Resource &>(primitive), option} {}

void MotionInstance::set_keyframe(size_t index, const MotionInstanceTransform &transform) noexcept {
    LUISA_ASSERT(index < _transform_keyframes.size(),
                 "Keyframe index out of range.");
    _transform_keyframes[index] = transform;
}

void MotionInstance::set_keyframe(size_t index, const MotionInstanceTransformMatrix &transform) noexcept {
    LUISA_ASSERT(_mode == AccelOption::MotionMode::MATRIX,
                 "Invalid motion mode for matrix transform.");
    set_keyframe(index, MotionInstanceTransform{transform});
}

void MotionInstance::set_keyframe(size_t index, const MotionInstanceTransformSRT &transform) noexcept {
    LUISA_ASSERT(_mode == AccelOption::MotionMode::SRT,
                 "Invalid motion mode for SRT transform.");
    set_keyframe(index, MotionInstanceTransform{transform});
}

void MotionInstance::set_keyframes(luisa::span<const MotionInstanceTransform> transforms) noexcept {
    LUISA_ASSERT(transforms.size() == _transform_keyframes.size(), "Keyframe count mismatch.");
    std::memmove(_transform_keyframes.data(), transforms.data(), transforms.size_bytes());
}

void MotionInstance::set_keyframes(luisa::span<const MotionInstanceTransformMatrix> transforms) noexcept {
    LUISA_ASSERT(_mode == AccelOption::MotionMode::MATRIX, "Invalid motion mode for matrix transform.");
    set_keyframes(luisa::span{reinterpret_cast<const MotionInstanceTransform *>(transforms.data()), transforms.size()});
}

void MotionInstance::set_keyframes(luisa::span<const MotionInstanceTransformSRT> transforms) noexcept {
    LUISA_ASSERT(_mode == AccelOption::MotionMode::SRT, "Invalid motion mode for SRT transform.");
    set_keyframes(luisa::span{reinterpret_cast<const MotionInstanceTransform *>(transforms.data()), transforms.size()});
}

const MotionInstanceTransform &MotionInstance::keyframe(size_t index) const noexcept {
    LUISA_ASSERT(index < _transform_keyframes.size(),
                 "Keyframe index out of range.");
    return _transform_keyframes[index];
}

const MotionInstanceTransformMatrix &MotionInstance::keyframe_matrix(size_t index) const noexcept {
    LUISA_ASSERT(_mode == AccelOption::MotionMode::MATRIX,
                 "Invalid motion mode for matrix transform.");
    return keyframe(index).as_matrix();
}

const MotionInstanceTransformSRT &MotionInstance::keyframe_srt(size_t index) const noexcept {
    LUISA_ASSERT(_mode == AccelOption::MotionMode::SRT,
                 "Invalid motion mode for SRT transform.");
    return keyframe(index).as_srt();
}

luisa::span<const MotionInstanceTransform> MotionInstance::keyframes() const noexcept {
    return _transform_keyframes;
}

luisa::span<const MotionInstanceTransformMatrix> MotionInstance::keyframes_matrix() const noexcept {
    LUISA_ASSERT(_mode == AccelOption::MotionMode::MATRIX,
                 "Invalid motion mode for matrix transform.");
    return luisa::span{reinterpret_cast<const MotionInstanceTransformMatrix *>(
                           _transform_keyframes.data()),
                       _transform_keyframes.size()};
}

luisa::span<const MotionInstanceTransformSRT> MotionInstance::keyframes_srt() const noexcept {
    LUISA_ASSERT(_mode == AccelOption::MotionMode::SRT,
                 "Invalid motion mode for SRT transform.");
    return luisa::span{reinterpret_cast<const MotionInstanceTransformSRT *>(
                           _transform_keyframes.data()),
                       _transform_keyframes.size()};
}

luisa::unique_ptr<Command> MotionInstance::build() noexcept {
    return luisa::make_unique<MotionInstanceBuildCommand>(
        handle(), _transform_keyframes /* note this needs to be copied */);
}

}// namespace luisa::compute
