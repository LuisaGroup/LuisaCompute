#include <cmath>

#include <luisa/core/logging.h>
#include "metal_motion_instance.h"

namespace luisa::compute::metal {

namespace {

void validate_finite(luisa::span<const float> values,
                     size_t keyframe,
                     luisa::string_view field) noexcept {
    for (auto i = size_t{0u}; i < values.size(); i++) {
        LUISA_ASSERT(std::isfinite(values[i]),
                     "Metal4 motion keyframe {} contains a non-finite {} "
                     "component at index {}.",
                     keyframe, field, i);
    }
}

void validate_matrix_keyframe(const MotionInstanceTransformMatrix &matrix,
                              size_t keyframe) noexcept {
    validate_finite({&matrix[0u][0u], 16u}, keyframe, "matrix");
    LUISA_ASSERT(matrix[0u][3u] == 0.0f &&
                     matrix[1u][3u] == 0.0f &&
                     matrix[2u][3u] == 0.0f &&
                     matrix[3u][3u] == 1.0f,
                 "Metal4 matrix motion keyframe {} is not affine.",
                 keyframe);
}

void validate_srt_keyframe(const MotionInstanceTransformSRT &srt,
                           size_t keyframe) noexcept {
    validate_finite({srt.pivot, 3u}, keyframe, "pivot");
    validate_finite({srt.quaternion, 4u}, keyframe, "quaternion");
    validate_finite({srt.scale, 3u}, keyframe, "scale");
    validate_finite({srt.shear, 3u}, keyframe, "shear");
    validate_finite({srt.translation, 3u}, keyframe, "translation");
    auto norm_squared = srt.quaternion[0u] * srt.quaternion[0u] +
                        srt.quaternion[1u] * srt.quaternion[1u] +
                        srt.quaternion[2u] * srt.quaternion[2u] +
                        srt.quaternion[3u] * srt.quaternion[3u];
    LUISA_ASSERT(norm_squared > 0.0f,
                 "Metal4 SRT motion keyframe {} has a zero quaternion.",
                 keyframe);
}

}// namespace

MetalMotionInstance::MetalMotionInstance(
    const AccelMotionOption &option) noexcept
    : MetalPrimitiveBase{Kind::MOTION_INSTANCE},
      _option{option} {
    LUISA_ASSERT(option.keyframe_count >= 2u,
                 "Metal4 motion instances require at least two keyframes.");
    LUISA_ASSERT(std::isfinite(option.time_start) &&
                     std::isfinite(option.time_end) &&
                     option.time_start < option.time_end,
                 "Metal4 motion instances require a finite, strictly "
                 "increasing time range (got [{}, {}]).",
                 option.time_start, option.time_end);
    _keyframes.resize(option.keyframe_count);
}

void MetalMotionInstance::build(
    MotionInstanceBuildCommand *command) noexcept {
    auto child = reinterpret_cast<MetalPrimitive *>(command->child());
    auto keyframes = command->steal_keyframes();
    LUISA_ASSERT(child != nullptr && child->handle() != nullptr,
                 "Metal4 motion instance must reference a built mesh, curve, "
                 "or procedural primitive.");
    LUISA_ASSERT(keyframes.size() == _option.keyframe_count,
                 "Metal4 motion instance keyframe count mismatch: expected "
                 "{}, got {}.",
                 _option.keyframe_count, keyframes.size());
    for (auto i = size_t{0u}; i < keyframes.size(); i++) {
        switch (_option.mode) {
            case AccelMotionMode::MATRIX:
                validate_matrix_keyframe(keyframes[i].as_matrix(), i);
                break;
            case AccelMotionMode::SRT:
                validate_srt_keyframe(keyframes[i].as_srt(), i);
                break;
            default:
                LUISA_ERROR_WITH_LOCATION(
                    "Invalid Metal4 motion instance transform mode.");
        }
    }
    std::scoped_lock lock{_mutex};
    _child = child;
    _keyframes = std::move(keyframes);
    _build_version++;
}

MetalMotionInstance::Snapshot MetalMotionInstance::snapshot() const noexcept {
    std::scoped_lock lock{_mutex};
    return Snapshot{_option, _child, _keyframes, _build_version};
}

MTL::AccelerationStructure *MetalMotionInstance::handle() const noexcept {
    std::scoped_lock lock{_mutex};
    return _child == nullptr ? nullptr : _child->handle();
}

void MetalMotionInstance::set_name(luisa::string_view name) noexcept {
    std::scoped_lock lock{_mutex};
    _name = name;
}

void MetalMotionInstance::add_resources(
    luisa::vector<MTL::Resource *> &resources) noexcept {
    std::scoped_lock lock{_mutex};
    LUISA_ASSERT(_child != nullptr && _child->handle() != nullptr,
                 "Metal4 motion instance has not been built.");
    _child->add_resources(resources);
}

}// namespace luisa::compute::metal
