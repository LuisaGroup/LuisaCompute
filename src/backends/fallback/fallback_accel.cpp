#include <luisa/core/stl.h>
#include <luisa/core/logging.h>

#include "fallback_device_api.h"
#include "fallback_prim.h"
#include "fallback_motion_instance.h"
#include "fallback_accel.h"
#include "fallback_command_queue.h"

namespace luisa::compute::fallback {

FallbackAccel::FallbackAccel(RTCDevice device, const AccelOption &option) noexcept
    : _handle{rtcNewScene(device)} {
    luisa_fallback_accel_set_flags(_handle, option);
}

FallbackAccel::~FallbackAccel() noexcept { rtcReleaseScene(_handle); }

[[nodiscard]] static auto luisa_fallback_affine_transform_is_identity(const float a[12]) noexcept {
    return a[0] == 1.f && a[1] == 0.f && a[2] == 0.f && a[3] == 0.f &&
           a[4] == 0.f && a[5] == 1.f && a[6] == 0.f && a[7] == 0.f &&
           a[8] == 0.f && a[9] == 0.f && a[10] == 1.f && a[11] == 0.f;
}

[[nodiscard]] static auto luisa_fallback_srt_to_embree_quaternion(const MotionInstanceTransformSRT &srt) noexcept {
    return RTCQuaternionDecomposition{
        .scale_x = srt.scale[0],
        .scale_y = srt.scale[1],
        .scale_z = srt.scale[2],
        .skew_xy = srt.shear[0],
        .skew_xz = srt.shear[1],
        .skew_yz = srt.shear[2],
        .shift_x = srt.pivot[0],
        .shift_y = srt.pivot[1],
        .shift_z = srt.pivot[2],
        .quaternion_r = srt.quaternion[3],
        .quaternion_i = srt.quaternion[0],
        .quaternion_j = srt.quaternion[1],
        .quaternion_k = srt.quaternion[2],
        .translation_x = srt.translation[0],
        .translation_y = srt.translation[1],
        .translation_z = srt.translation[2]};
}

[[nodiscard]] static auto luisa_fallback_affine_to_matrix(const float a[12]) noexcept {
    return luisa::make_float4x4(a[0], a[4], a[8], 0.f,
                                a[1], a[5], a[9], 0.f,
                                a[2], a[6], a[10], 0.f,
                                a[3], a[7], a[11], 1.f);
}

void FallbackAccel::build(luisa::unique_ptr<AccelBuildCommand> cmd) noexcept {
    LUISA_ASSERT(!cmd->update_instance_buffer_only(),
                 "FallbackAccel does not support update_instance_buffer_only.");
    if (auto n = cmd->instance_count(); n < _instances.size()) {
        // remove redundant geometries
        for (auto i = n; i < _instances.size(); i++) {
            rtcDetachGeometry(_handle, i);
            if (auto m = _instances[i].motion) {
                luisa::deallocate_with_allocator(m);
            }
        }
        _instances.resize(n);
        _geometries.resize(n);
    } else {
        // create new geometries
        auto device = rtcGetSceneDevice(_handle);
        _instances.reserve(next_pow2(n));
        _geometries.reserve(next_pow2(n));
        for (auto i = _instances.size(); i < n; i++) {
            auto geometry = rtcNewGeometry(device, RTC_GEOMETRY_TYPE_INSTANCE);
            rtcSetGeometryBuildQuality(geometry, RTC_BUILD_QUALITY_HIGH);
            rtcAttachGeometryByID(_handle, geometry, i);
            rtcReleaseGeometry(geometry);// already moved into the scene
            _instances.emplace_back(Instance{});
            _geometries.emplace_back(geometry);
        }
    }
    // Views captured by already-submitted shader commands point to this
    // stable descriptor field. Publish the vector's current storage only
    // after all operations which can reallocate it.
    _instance_data = _instances.data();
    for (auto m : cmd->modifications()) {
        using Mod = AccelBuildCommand::Modification;
        auto &instance = _instances[m.index];
        if (m.flags & Mod::flag_primitive) {
            auto geometry = _geometries[m.index];
            if (auto prim_base = reinterpret_cast<const FallbackPrimBase *>(m.primitive); prim_base->is_motion_instance()) {
                auto motion_instance = static_cast<const FallbackMotionInstance *>(prim_base);
                auto option = motion_instance->option();
                auto child = motion_instance->child();
                rtcSetGeometryInstancedScene(geometry, child->handle());
                rtcSetGeometryTimeStepCount(geometry, option.keyframe_count);
                rtcSetGeometryTimeRange(geometry, option.time_start, option.time_end);
                instance.is_curve = child->is_curve();
                instance.is_motion = true;
                instance.is_matrix = option.mode == AccelMotionMode::MATRIX;
                instance.is_srt = option.mode == AccelMotionMode::SRT;
                if (instance.motion_steps != option.keyframe_count) {
                    if (instance.motion) { luisa::deallocate_with_allocator(instance.motion); }
                    instance.motion_steps = option.keyframe_count;
                    instance.motion = luisa::allocate_with_allocator<MotionInstanceTransform>(option.keyframe_count);
                }
                auto transforms = motion_instance->transforms();
                std::memcpy(instance.motion, transforms.data(), transforms.size_bytes());
            } else {
                auto prim = static_cast<const FallbackPrim *>(prim_base);
                rtcSetGeometryInstancedScene(geometry, prim->handle());
                rtcSetGeometryTimeStepCount(geometry, 1u);
                rtcSetGeometryTimeRange(geometry, 0.f, 1.f);
                instance.is_curve = prim->is_curve();
                instance.is_motion = false;
                instance.is_matrix = false;
                instance.is_srt = false;
                instance.motion_steps = 0u;
                if (instance.motion) {
                    luisa::deallocate_with_allocator(instance.motion);
                    instance.motion = nullptr;
                }
            }
        }
        if (m.flags & Mod::flag_transform) {
            std::memcpy(instance.affine, m.affine, sizeof(m.affine));
        }
        if (m.flags & Mod::flag_visibility) {
            instance.mask = m.vis_mask;
        }
        if (m.flags & Mod::flag_user_id) {
            instance.user_id = m.user_id;
        }
        if (m.flags & Mod::flag_opaque) {
            // AccelInstance::opaque is a one-bit field. Assign a normalized
            // boolean instead of the flag mask itself, whose low bit may be
            // zero and would therefore be truncated to false.
            instance.opaque =
                (m.flags & Mod::flag_opaque_on) != 0u;
        }
        instance.dirty = true;
    }
    for (auto i = 0u; i < _instances.size(); i++) {
        if (auto &instance = _instances[i]; instance.dirty) {
            auto geometry = _geometries[i];
            if (instance.is_motion) {
                LUISA_DEBUG_ASSERT(instance.is_matrix || instance.is_srt, "Invalid motion mode.");
                if (instance.is_matrix) {
                    if (auto a = instance.affine; luisa_fallback_affine_transform_is_identity(a)) {
                        for (auto k = 0u; k < instance.motion_steps; k++) {
                            auto &t = instance.motion[k].as_matrix();
                            rtcSetGeometryTransform(geometry, k, RTC_FORMAT_FLOAT4X4_COLUMN_MAJOR, &t);
                        }
                    } else {
                        auto m = luisa_fallback_affine_to_matrix(a);
                        for (auto k = 0u; k < instance.motion_steps; k++) {
                            auto t = m * instance.motion[k].as_matrix();
                            rtcSetGeometryTransform(geometry, k, RTC_FORMAT_FLOAT4X4_COLUMN_MAJOR, &t);
                        }
                    }
                } else {// SRT quaternion
                    if (auto a = instance.affine; luisa_fallback_affine_transform_is_identity(a)) {
                        for (auto k = 0u; k < instance.motion_steps; k++) {
                            auto &t = instance.motion[k].as_srt();
                            auto q = luisa_fallback_srt_to_embree_quaternion(t);
                            rtcSetGeometryTransformQuaternion(geometry, k, &q);
                        }
                    } else {
                        // FIXME: seems incorrect
                        auto m = luisa_fallback_affine_to_matrix(a);
                        LUISA_NOT_IMPLEMENTED();
                    }
                }
            } else {
                rtcSetGeometryTransform(geometry, 0u, RTC_FORMAT_FLOAT3X4_ROW_MAJOR, instance.affine);
            }
            rtcSetGeometryMask(geometry, instance.mask);
            auto flags = (instance.opaque ? api::luisa_fallback_embree_accel_user_data_flags_opaque : 0u) |
                         (instance.is_curve ? api::luisa_fallback_embree_accel_user_data_flags_curve : 0u);
            rtcSetGeometryUserData(geometry, reinterpret_cast<void *>(flags));
            rtcCommitGeometry(geometry);
            instance.dirty = false;
        }
    }
    // TODO: support update?
    rtcCommitScene(_handle);
}

}// namespace luisa::compute::fallback
