#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>

#include <luisa/core/basic_types.h>
#include <luisa/core/logging.h>
#include "metal_command_encoder.h"
#include "metal_motion_instance.h"
#include "metal_stream.h"
#include "metal_accel.h"

namespace luisa::compute::metal {

namespace {

struct Vec3 {
    float x;
    float y;
    float z;
};

[[nodiscard]] Vec3 sub(Vec3 a, Vec3 b) noexcept {
    return {a.x - b.x, a.y - b.y, a.z - b.z};
}

[[nodiscard]] Vec3 mul(Vec3 v, float s) noexcept {
    return {v.x * s, v.y * s, v.z * s};
}

[[nodiscard]] float dot(Vec3 a, Vec3 b) noexcept {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

[[nodiscard]] Vec3 cross(Vec3 a, Vec3 b) noexcept {
    return {a.y * b.z - a.z * b.y,
            a.z * b.x - a.x * b.z,
            a.x * b.y - a.y * b.x};
}

[[nodiscard]] float length(Vec3 v) noexcept {
    return std::sqrt(dot(v, v));
}

[[nodiscard]] bool finite(Vec3 v) noexcept {
    return std::isfinite(v.x) && std::isfinite(v.y) &&
           std::isfinite(v.z);
}

[[nodiscard]] MTL::PackedFloat3 packed(Vec3 v) noexcept {
    return MTL::PackedFloat3{v.x, v.y, v.z};
}

[[nodiscard]] luisa::float4x4 unpack_matrix(
    const MTL::PackedFloat4x3 &matrix) noexcept {
    return luisa::make_float4x4(
        luisa::make_float4(matrix[0u].x, matrix[0u].y, matrix[0u].z, 0.0f),
        luisa::make_float4(matrix[1u].x, matrix[1u].y, matrix[1u].z, 0.0f),
        luisa::make_float4(matrix[2u].x, matrix[2u].y, matrix[2u].z, 0.0f),
        luisa::make_float4(matrix[3u].x, matrix[3u].y, matrix[3u].z, 1.0f));
}

[[nodiscard]] MTL::PackedFloat4x3 pack_matrix(
    const luisa::float4x4 &matrix) noexcept {
    return MTL::PackedFloat4x3{
        MTL::PackedFloat3{matrix[0u].x, matrix[0u].y, matrix[0u].z},
        MTL::PackedFloat3{matrix[1u].x, matrix[1u].y, matrix[1u].z},
        MTL::PackedFloat3{matrix[2u].x, matrix[2u].y, matrix[2u].z},
        MTL::PackedFloat3{matrix[3u].x, matrix[3u].y, matrix[3u].z}};
}

[[nodiscard]] bool affine_linear_is_identity(
    const MTL::PackedFloat4x3 &matrix) noexcept {
    return matrix[0u].x == 1.0f && matrix[0u].y == 0.0f &&
           matrix[0u].z == 0.0f && matrix[1u].x == 0.0f &&
           matrix[1u].y == 1.0f && matrix[1u].z == 0.0f &&
           matrix[2u].x == 0.0f && matrix[2u].y == 0.0f &&
           matrix[2u].z == 1.0f;
}

[[nodiscard]] MTL::PackedFloatQuaternion quaternion_from_columns(
    Vec3 c0, Vec3 c1, Vec3 c2) noexcept {
    auto m00 = c0.x;
    auto m01 = c1.x;
    auto m02 = c2.x;
    auto m10 = c0.y;
    auto m11 = c1.y;
    auto m12 = c2.y;
    auto m20 = c0.z;
    auto m21 = c1.z;
    auto m22 = c2.z;
    auto x = 0.0f;
    auto y = 0.0f;
    auto z = 0.0f;
    auto w = 1.0f;
    auto trace = m00 + m11 + m22;
    if (trace > 0.0f) {
        auto s = std::sqrt(trace + 1.0f) * 2.0f;
        w = 0.25f * s;
        x = (m21 - m12) / s;
        y = (m02 - m20) / s;
        z = (m10 - m01) / s;
    } else if (m00 > m11 && m00 > m22) {
        auto s = std::sqrt(1.0f + m00 - m11 - m22) * 2.0f;
        w = (m21 - m12) / s;
        x = 0.25f * s;
        y = (m01 + m10) / s;
        z = (m02 + m20) / s;
    } else if (m11 > m22) {
        auto s = std::sqrt(1.0f + m11 - m00 - m22) * 2.0f;
        w = (m02 - m20) / s;
        x = (m01 + m10) / s;
        y = 0.25f * s;
        z = (m12 + m21) / s;
    } else {
        auto s = std::sqrt(1.0f + m22 - m00 - m11) * 2.0f;
        w = (m10 - m01) / s;
        x = (m02 + m20) / s;
        y = (m12 + m21) / s;
        z = 0.25f * s;
    }
    auto inverse_norm =
        1.0f / std::sqrt(x * x + y * y + z * z + w * w);
    return MTL::PackedFloatQuaternion{
        x * inverse_norm, y * inverse_norm,
        z * inverse_norm, w * inverse_norm};
}

[[nodiscard]] MTL::ComponentTransform decompose_affine(
    const MTL::PackedFloat4x3 &matrix,
    size_t instance_index) noexcept {
    auto c0 = Vec3{matrix[0u].x, matrix[0u].y, matrix[0u].z};
    auto c1 = Vec3{matrix[1u].x, matrix[1u].y, matrix[1u].z};
    auto c2 = Vec3{matrix[2u].x, matrix[2u].y, matrix[2u].z};
    auto translation =
        Vec3{matrix[3u].x, matrix[3u].y, matrix[3u].z};
    LUISA_ASSERT(finite(c0) && finite(c1) && finite(c2) &&
                     finite(translation),
                 "Metal4 acceleration-structure instance {} has a "
                 "non-finite transform.",
                 instance_index);
    constexpr auto epsilon = 1.0e-20f;
    auto scale_x = length(c0);
    LUISA_ASSERT(scale_x > epsilon,
                 "Metal4 component-motion TLAS instance {} has a singular "
                 "transform.",
                 instance_index);
    auto q0 = mul(c0, 1.0f / scale_x);
    auto shear_xy = dot(q0, c1);
    auto q1 = sub(c1, mul(q0, shear_xy));
    auto scale_y = length(q1);
    LUISA_ASSERT(scale_y > epsilon,
                 "Metal4 component-motion TLAS instance {} has a singular "
                 "transform.",
                 instance_index);
    q1 = mul(q1, 1.0f / scale_y);
    auto shear_xz = dot(q0, c2);
    auto q2 = sub(c2, mul(q0, shear_xz));
    auto shear_yz = dot(q1, q2);
    q2 = sub(q2, mul(q1, shear_yz));
    auto scale_z = length(q2);
    LUISA_ASSERT(scale_z > epsilon,
                 "Metal4 component-motion TLAS instance {} has a singular "
                 "transform.",
                 instance_index);
    q2 = mul(q2, 1.0f / scale_z);
    if (dot(q0, cross(q1, q2)) < 0.0f) {
        q0 = mul(q0, -1.0f);
        scale_x = -scale_x;
        shear_xy = -shear_xy;
        shear_xz = -shear_xz;
    }
    MTL::ComponentTransform result{};
    result.scale = packed({scale_x, scale_y, scale_z});
    result.shear = packed({shear_xy, shear_xz, shear_yz});
    result.pivot = packed({0.0f, 0.0f, 0.0f});
    result.rotation = quaternion_from_columns(q0, q1, q2);
    result.translation = packed(translation);
    return result;
}

[[nodiscard]] MTL::ComponentTransform component_transform(
    const MotionInstanceTransformSRT &srt,
    const MTL::PackedFloat4x3 &outer,
    size_t instance_index) noexcept {
    LUISA_ASSERT(
        affine_linear_is_identity(outer),
        "Metal4 cannot preserve SRT component interpolation for motion "
        "instance {} under a non-translation outer Accel transform. Use an "
        "identity/translation outer transform or MATRIX motion.",
        instance_index);
    MTL::ComponentTransform result{};
    result.scale = MTL::PackedFloat3{
        srt.scale[0u], srt.scale[1u], srt.scale[2u]};
    result.shear = MTL::PackedFloat3{
        srt.shear[0u], srt.shear[1u], srt.shear[2u]};
    result.pivot = MTL::PackedFloat3{
        srt.pivot[0u], srt.pivot[1u], srt.pivot[2u]};
    result.rotation = MTL::PackedFloatQuaternion{
        srt.quaternion[0u], srt.quaternion[1u],
        srt.quaternion[2u], srt.quaternion[3u]};
    auto inverse_rotation_norm =
        1.0f / std::sqrt(
                   result.rotation.x * result.rotation.x +
                   result.rotation.y * result.rotation.y +
                   result.rotation.z * result.rotation.z +
                   result.rotation.w * result.rotation.w);
    result.rotation.x *= inverse_rotation_norm;
    result.rotation.y *= inverse_rotation_norm;
    result.rotation.z *= inverse_rotation_norm;
    result.rotation.w *= inverse_rotation_norm;
    result.translation = MTL::PackedFloat3{
        srt.translation[0u] + outer[3u].x,
        srt.translation[1u] + outer[3u].y,
        srt.translation[2u] + outer[3u].z};
    return result;
}

}// namespace

void MetalAccel::_prepare_motion_data(
    MetalCommandEncoder &encoder) noexcept {
    auto desired_mode = MotionMode::NONE;
    auto transform_count = size_t{0u};
    for (auto primitive : _primitives) {
        if (primitive->is_motion_instance()) {
            auto motion = static_cast<MetalMotionInstance *>(primitive);
            auto mode = motion->option().mode == AccelMotionMode::MATRIX ?
                            MotionMode::MATRIX :
                            MotionMode::COMPONENT;
            LUISA_ASSERT(desired_mode == MotionMode::NONE ||
                             desired_mode == mode,
                         "Metal4 cannot place MATRIX and SRT MotionInstance "
                         "resources in the same acceleration structure because "
                         "Metal exposes one motion-transform type per TLAS.");
            desired_mode = mode;
            LUISA_ASSERT(
                motion->option().keyframe_count <=
                    std::numeric_limits<uint32_t>::max() - transform_count,
                "Metal4 motion TLAS transform-index range overflow.");
            transform_count += motion->option().keyframe_count;
        } else {
            LUISA_ASSERT(
                transform_count < std::numeric_limits<uint32_t>::max(),
                "Metal4 motion TLAS transform-index range overflow.");
            transform_count++;
        }
    }
    if (desired_mode == MotionMode::NONE) {
        if (_motion_mode != MotionMode::NONE) { _requires_rebuild = true; }
        _motion_mode = MotionMode::NONE;
        _motion_transform_count = 0u;
        return;
    }
    if (desired_mode == MotionMode::COMPONENT) {
        LUISA_ASSERT(
            encoder.stream()->supports_address_driven_acceleration_structures(),
            "Metal4 SRT/component motion requires Apple9 or newer.");
    }
    if (_motion_mode != desired_mode) { _requires_rebuild = true; }
    _motion_mode = desired_mode;
    _motion_transform_count = transform_count;

    auto instance_buffer_size = _primitives.size() * sizeof(MotionInstance);
    auto transform_stride = desired_mode == MotionMode::MATRIX ?
                                sizeof(MTL::PackedFloat4x3) :
                                sizeof(MTL::ComponentTransform);
    LUISA_ASSERT(
        transform_count <=
            std::numeric_limits<size_t>::max() / transform_stride,
        "Metal4 motion TLAS transform-buffer size overflow.");
    auto transform_buffer_size = transform_count * transform_stride;
    auto ensure_buffer = [&](MTL::Buffer *&buffer, size_t size) noexcept {
        if (buffer != nullptr && buffer->length() >= size) { return; }
        auto old_buffer = buffer;
        buffer = encoder.device()->newBuffer(
            size, MTL::ResourceStorageModePrivate |
                      MTL::ResourceHazardTrackingModeTracked);
        LUISA_ASSERT(buffer != nullptr,
                     "Failed to allocate Metal4 motion acceleration-structure "
                     "buffer ({} bytes).",
                     size);
        if (old_buffer != nullptr) {
            encoder.add_callback(FunctionCallbackContext::create(
                [old_buffer]() noexcept { old_buffer->release(); }));
        }
        _requires_rebuild = true;
    };
    ensure_buffer(_motion_instance_buffer, instance_buffer_size);
    ensure_buffer(_motion_transform_buffer, transform_buffer_size);

    luisa::vector<MotionInstance> motion_instances(_primitives.size());
    luisa::vector<std::byte> transforms(transform_buffer_size);
    auto transform_index = size_t{0u};
    for (auto i = size_t{0u}; i < _primitives.size(); i++) {
        auto primitive = _primitives[i];
        auto handle = primitive->handle();
        LUISA_ASSERT(handle != nullptr,
                     "Metal4 acceleration-structure instance {} has no built "
                     "primitive.",
                     i);
        auto &source = _instances[i];
        auto &destination = motion_instances[i];
        destination.options = source.options;
        destination.mask = source.mask;
        destination.intersectionFunctionTableOffset = source.user_id;
        destination.userID = source.mesh_index;
        destination.accelerationStructureID = handle->gpuResourceID();
        destination.motionTransformsStartIndex =
            static_cast<uint32_t>(transform_index);
        if (primitive->is_motion_instance()) {
            auto motion = static_cast<MetalMotionInstance *>(primitive);
            auto snapshot = motion->snapshot();
            LUISA_ASSERT(snapshot.child != nullptr &&
                             snapshot.child->handle() != nullptr &&
                             snapshot.keyframes.size() ==
                                 snapshot.option.keyframe_count,
                         "Metal4 motion instance {} must be built before its "
                         "containing acceleration structure.",
                         i);
            destination.motionTransformsCount =
                snapshot.option.keyframe_count;
            destination.motionStartBorderMode =
                snapshot.option.should_vanish_start ?
                    MTL::MotionBorderModeVanish :
                    MTL::MotionBorderModeClamp;
            destination.motionEndBorderMode =
                snapshot.option.should_vanish_end ?
                    MTL::MotionBorderModeVanish :
                    MTL::MotionBorderModeClamp;
            destination.motionStartTime = snapshot.option.time_start;
            destination.motionEndTime = snapshot.option.time_end;
            for (auto key = size_t{0u};
                 key < snapshot.keyframes.size(); key++) {
                if (desired_mode == MotionMode::MATRIX) {
                    auto matrix = unpack_matrix(source.transformation) *
                                  snapshot.keyframes[key].as_matrix();
                    auto packed_matrix = pack_matrix(matrix);
                    std::memcpy(
                        transforms.data() +
                            transform_index * transform_stride,
                        &packed_matrix, sizeof(packed_matrix));
                } else {
                    auto component = component_transform(
                        snapshot.keyframes[key].as_srt(),
                        source.transformation, i);
                    std::memcpy(
                        transforms.data() +
                            transform_index * transform_stride,
                        &component, sizeof(component));
                }
                transform_index++;
            }
        } else {
            destination.motionTransformsCount = 1u;
            destination.motionStartBorderMode = MTL::MotionBorderModeClamp;
            destination.motionEndBorderMode = MTL::MotionBorderModeClamp;
            destination.motionStartTime = 0.0f;
            destination.motionEndTime = 1.0f;
            if (desired_mode == MotionMode::MATRIX) {
                std::memcpy(
                    transforms.data() + transform_index * transform_stride,
                    &source.transformation, sizeof(source.transformation));
            } else {
                auto component = decompose_affine(source.transformation, i);
                std::memcpy(
                    transforms.data() + transform_index * transform_stride,
                    &component, sizeof(component));
            }
            transform_index++;
        }
    }
    LUISA_ASSERT(transform_index == transform_count,
                 "Metal4 motion transform count mismatch.");

    auto upload = [&](MTL::Buffer *buffer, const void *data,
                      size_t size) noexcept {
        encoder.with_upload_buffer(size, [&](auto staging) noexcept {
            std::memcpy(staging->data(), data, size);
            auto copy_encoder = encoder.compute_encoder();
            copy_encoder->copyFromBuffer(
                staging->buffer(), staging->offset(), buffer, 0u, size);
            encoder.use_resource(staging->buffer());
            encoder.use_resource(buffer);
            copy_encoder->endEncoding();
        });
    };
    upload(_motion_instance_buffer, motion_instances.data(),
           instance_buffer_size);
    upload(_motion_transform_buffer, transforms.data(),
           transform_buffer_size);
}

}// namespace luisa::compute::metal
