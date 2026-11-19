#include <luisa/core/stl.h>
#include <luisa/core/logging.h>

#include "fallback_device_api.h"
#include "fallback_prim.h"
#include "fallback_mesh.h"
#include "fallback_motion_instance.h"
#include "fallback_accel.h"
#include "fallback_command_queue.h"

namespace luisa::compute::fallback {

FallbackAccel::FallbackAccel(RTCDevice device, const AccelOption &option) noexcept
    : _handle{rtcNewScene(device)} {
    luisa_fallback_accel_set_flags(_handle, option);
}

FallbackAccel::~FallbackAccel() noexcept {
    for (auto &inst : _instances) {
        if (inst.motion) {
            luisa::deallocate_with_allocator(inst.motion);
            inst.motion = nullptr;
        }
    }
    rtcReleaseScene(_handle);
}

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
    const auto n = cmd->instance_count();
    auto device = rtcGetSceneDevice(_handle);
    if (n < _instances.size()) {
        for (auto i = n; i < _instances.size(); i++) {
            if (_geometries[i] != nullptr) {
                rtcDetachGeometry(_handle, i);
            }
            if (auto m = _instances[i].motion) {
                luisa::deallocate_with_allocator(m);
            }
        }
        _instances.resize(n);
        _instance_flags.resize(n);
        _geometries.resize(n);
        _primitives.resize(n);
        _direct_meshes.resize(n);
        _mesh_revisions.resize(n);
    } else if (n > _instances.size()) {
        const auto cap = next_pow2(n);
        _instances.reserve(cap);
        _instance_flags.reserve(cap);
        _geometries.reserve(cap);
        _primitives.reserve(cap);
        _direct_meshes.reserve(cap);
        _mesh_revisions.reserve(cap);
        _instances.resize(n, Instance{});
        _instance_flags.resize(n, 0u);
        _geometries.resize(n, nullptr);
        _primitives.resize(n, nullptr);
        _direct_meshes.resize(n, nullptr);
        _mesh_revisions.resize(n, 0u);
    }
    // Views captured by already-submitted shader commands point to this
    // stable descriptor field. Publish the vectors' current storage only
    // after all operations which can reallocate them.
    _header.instances = _instances.data();
    _header.instance_flags = _instance_flags.data();

    for (auto m : cmd->modifications()) {
        using Mod = AccelBuildCommand::Modification;
        auto &instance = _instances[m.index];
        if (m.flags & Mod::flag_primitive) {
            auto prim_base = reinterpret_cast<const FallbackPrimBase *>(m.primitive);
            _primitives[m.index] = prim_base;
            if (prim_base->is_motion_instance()) {
                auto motion_instance = static_cast<const FallbackMotionInstance *>(prim_base);
                auto option = motion_instance->option();
                auto child = motion_instance->child();
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
            instance.opaque = (m.flags & Mod::flag_opaque_on) != 0u;
        }
        instance.dirty = true;
    }

    // Mirror Cycles BVHEmbree::build (!ob->get_geometry()->is_instanced()):
    // attach static single-reference identity-transform triangle meshes directly
    // to the top-level RTCScene via rtcSetSharedGeometryBuffer (zero vertex or
    // index buffer duplication, zero unused BLAS build, exact world==object ray).
    // Repeated/shared meshes (R > 1), non-identity transforms, dynamic/motion
    // meshes, curves, and procedural primitives always use RTC_GEOMETRY_TYPE_INSTANCE.
    const bool direct_enabled = luisa_fallback_direct_static_mesh_enabled();
    luisa::unordered_map<const FallbackMesh *, uint32_t> mesh_ref_counts;
    if (direct_enabled && n > 0u) {
        mesh_ref_counts.reserve(n);
        for (size_t i = 0u; i < n; i++) {
            const auto *prim_base = _primitives[i];
            if (prim_base != nullptr && !prim_base->is_motion_instance()) {
                const auto *prim = static_cast<const FallbackPrim *>(prim_base);
                if (prim->is_mesh()) {
                    ++mesh_ref_counts[static_cast<const FallbackMesh *>(prim)];
                }
            }
        }
    }

    for (auto i = 0u; i < n; i++) {
        auto &instance = _instances[i];
        const auto flags = static_cast<uint8_t>(
            (instance.opaque ? api::luisa_fallback_embree_accel_user_data_flags_opaque : 0u) |
            (instance.is_curve ? api::luisa_fallback_embree_accel_user_data_flags_curve : 0u));
        _instance_flags[i] = flags;

        const auto *prim_base = _primitives[i];
        if (prim_base == nullptr) { continue; }

        const FallbackMesh *want_direct_mesh = nullptr;
        if (direct_enabled && !instance.is_motion && !prim_base->is_motion_instance()) {
            const auto *prim = static_cast<const FallbackPrim *>(prim_base);
            if (prim->is_mesh() && !prim->motion() && !prim->option().allow_update &&
                luisa_fallback_affine_transform_is_identity(instance.affine)) {
                const auto *mesh = static_cast<const FallbackMesh *>(prim);
                if (mesh->has_shared_buffers() && mesh_ref_counts[mesh] == 1u) {
                    want_direct_mesh = mesh;
                }
            }
        }

        if (want_direct_mesh != nullptr) {
            const uint64_t rev = want_direct_mesh->build_revision();
            const bool need_recreate =
                (_geometries[i] == nullptr) ||
                (_direct_meshes[i] != want_direct_mesh) ||
                (_mesh_revisions[i] != rev);
            if (need_recreate) {
                if (_geometries[i] != nullptr) {
                    rtcDetachGeometry(_handle, i);
                }
                auto geom = rtcNewGeometry(device, RTC_GEOMETRY_TYPE_TRIANGLE);
                rtcSetGeometryBuildQuality(
                    geom,
                    luisa_fallback_embree_geometry_quality(
                        RTC_GEOMETRY_TYPE_TRIANGLE, want_direct_mesh->option()));
#if LUISA_COMPUTE_FALLBACK_EMBREE_VERSION >= 4
                rtcSetGeometryEnableFilterFunctionFromArguments(geom, true);
#endif
                want_direct_mesh->bind_shared_buffers(geom);
                rtcSetGeometryMask(geom, instance.mask);
                rtcSetGeometryUserData(geom, reinterpret_cast<void *>(static_cast<uint64_t>(flags)));
                rtcCommitGeometry(geom);
                rtcAttachGeometryByID(_handle, geom, i);
                rtcReleaseGeometry(geom);
                _geometries[i] = geom;
                _direct_meshes[i] = want_direct_mesh;
                _mesh_revisions[i] = rev;
                instance.dirty = false;
            } else if (instance.dirty) {
                auto geom = _geometries[i];
                rtcSetGeometryMask(geom, instance.mask);
                rtcSetGeometryUserData(geom, reinterpret_cast<void *>(static_cast<uint64_t>(flags)));
                rtcCommitGeometry(geom);
                instance.dirty = false;
            }
            continue;
        }

        // Instanced path (RTC_GEOMETRY_TYPE_INSTANCE)
        if (_direct_meshes[i] != nullptr && _geometries[i] != nullptr) {
            rtcDetachGeometry(_handle, i);
            _geometries[i] = nullptr;
            _direct_meshes[i] = nullptr;
            _mesh_revisions[i] = 0u;
            instance.dirty = true;
        }
        if (_geometries[i] == nullptr) {
            auto geom = rtcNewGeometry(device, RTC_GEOMETRY_TYPE_INSTANCE);
            rtcSetGeometryBuildQuality(geom, RTC_BUILD_QUALITY_HIGH);
#if LUISA_COMPUTE_FALLBACK_EMBREE_VERSION >= 4
            rtcSetGeometryEnableFilterFunctionFromArguments(geom, true);
#endif
            rtcAttachGeometryByID(_handle, geom, i);
            rtcReleaseGeometry(geom);
            _geometries[i] = geom;
            instance.dirty = true;
        }

        const FallbackPrim *target_prim = prim_base->is_motion_instance()
            ? static_cast<const FallbackMotionInstance *>(prim_base)->child()
            : static_cast<const FallbackPrim *>(prim_base);
        const uint64_t prim_rev = target_prim->build_revision();
        if (_mesh_revisions[i] != prim_rev) {
            _mesh_revisions[i] = prim_rev;
            instance.dirty = true;
        }

        if (instance.dirty) {
            auto geometry = _geometries[i];
            RTCScene instanced_scene = target_prim->committed_handle();
            rtcSetGeometryInstancedScene(geometry, instanced_scene);
            if (instance.is_motion) {
                auto motion_instance = static_cast<const FallbackMotionInstance *>(prim_base);
                auto option = motion_instance->option();
                rtcSetGeometryTimeStepCount(geometry, option.keyframe_count);
                rtcSetGeometryTimeRange(geometry, option.time_start, option.time_end);
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
                } else {
                    if (auto a = instance.affine; luisa_fallback_affine_transform_is_identity(a)) {
                        for (auto k = 0u; k < instance.motion_steps; k++) {
                            auto &t = instance.motion[k].as_srt();
                            auto q = luisa_fallback_srt_to_embree_quaternion(t);
                            rtcSetGeometryTransformQuaternion(geometry, k, &q);
                        }
                    } else {
                        auto m = luisa_fallback_affine_to_matrix(a);
                        (void)m;
                        LUISA_NOT_IMPLEMENTED();
                    }
                }
            } else {
                rtcSetGeometryTimeStepCount(geometry, 1u);
                rtcSetGeometryTimeRange(geometry, 0.f, 1.f);
                rtcSetGeometryTransform(geometry, 0u, RTC_FORMAT_FLOAT3X4_ROW_MAJOR, instance.affine);
            }
            rtcSetGeometryMask(geometry, instance.mask);
            rtcSetGeometryUserData(geometry, reinterpret_cast<void *>(static_cast<uint64_t>(flags)));
            rtcCommitGeometry(geometry);
            instance.dirty = false;
        }
    }
    const auto tlas_t0 = luisa_fallback_bvh_stats_enabled()
        ? std::chrono::steady_clock::now()
        : std::chrono::steady_clock::time_point{};
    rtcCommitScene(_handle);
    if (luisa_fallback_bvh_stats_enabled()) {
        const auto ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now() - tlas_t0).count();
        uint64_t direct_cnt = 0u;
        uint64_t inst_cnt = 0u;
        for (size_t i = 0u; i < n; i++) {
            if (_direct_meshes[i] != nullptr) {
                ++direct_cnt;
            } else if (_geometries[i] != nullptr) {
                ++inst_cnt;
            }
        }
        auto &s = luisa_fallback_bvh_stats();
        s.tlas_commits.fetch_add(1u, std::memory_order_relaxed);
        s.tlas_commit_ns.fetch_add(static_cast<uint64_t>(ns), std::memory_order_relaxed);
        s.last_direct_meshes.store(direct_cnt, std::memory_order_relaxed);
        s.last_instanced_slots.store(inst_cnt, std::memory_order_relaxed);
        s.retained_after_accel_bytes.store(s.live_bytes.load(std::memory_order_relaxed), std::memory_order_relaxed);
    }
}

}// namespace luisa::compute::fallback
