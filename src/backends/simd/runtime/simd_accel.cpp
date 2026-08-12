#include "simd_accel.h"

#include <array>
#include <cstring>

#include <luisa/core/logging.h>

#include "simd_mesh.h"

namespace luisa::compute::simd {

namespace {

[[nodiscard]] bool lane_active(
    uint64_t bits, uint32_t lane, uint32_t lane_count) noexcept {
    return lane < lane_count && ((bits >> lane) & 1u) != 0u;
}

template<size_t packet_width>
[[nodiscard]] bool packet_fully_active(
    uint32_t lane_count, uint64_t active_mask_bits) noexcept {
    static_assert(packet_width < 64u);
    constexpr auto packet_mask =
        (uint64_t{1u} << packet_width) - 1u;
    return lane_count == packet_width &&
           (active_mask_bits & packet_mask) == packet_mask;
}

template<size_t packet_width, typename RayPacket>
void initialize_ray_packet(
    RayPacket &ray, std::array<int, packet_width> &valid,
    uint32_t lane_count, uint64_t active_mask_bits,
    const float *components,
    const uint32_t *visibility_masks,
    const float *times) noexcept {
    if (lane_count == packet_width) {
        constexpr auto component_bytes =
            sizeof(float) * packet_width;
        if (packet_fully_active<packet_width>(
                lane_count, active_mask_bits)) {
            valid.fill(-1);
        } else {
            for (auto lane = uint32_t{0u};
                 lane < packet_width; lane++) {
                valid[lane] =
                    ((active_mask_bits >> lane) & 1u) != 0u ? -1 : 0;
            }
        }
        // The LLVM callback boundary initializes and sanitizes every native
        // packet lane before entering the runtime, including inactive lanes.
        // The complete component vectors can therefore be copied without
        // inspecting inactive values; Embree's valid array remains the sole
        // traversal predicate. W2 is padded to W4 and retains the guarded
        // path below because the source arrays contain only two lanes.
        std::memcpy(ray.org_x, components, component_bytes);
        std::memcpy(
            ray.org_y, components + packet_width,
            component_bytes);
        std::memcpy(
            ray.org_z, components + 2u * packet_width,
            component_bytes);
        std::memcpy(
            ray.tnear, components + 3u * packet_width,
            component_bytes);
        std::memcpy(
            ray.dir_x, components + 4u * packet_width,
            component_bytes);
        std::memcpy(
            ray.dir_y, components + 5u * packet_width,
            component_bytes);
        std::memcpy(
            ray.dir_z, components + 6u * packet_width,
            component_bytes);
        if (times == nullptr) {
            std::memset(ray.time, 0, component_bytes);
        } else {
            std::memcpy(ray.time, times, component_bytes);
        }
        std::memcpy(
            ray.tfar, components + 7u * packet_width,
            component_bytes);
        std::memcpy(
            ray.mask, visibility_masks,
            sizeof(uint32_t) * packet_width);
        static constexpr auto lane_ids = []() noexcept {
            std::array<uint32_t, packet_width> result{};
            for (auto lane = uint32_t{0u}; lane < packet_width; lane++) {
                result[lane] = lane;
            }
            return result;
        }();
        std::memcpy(
            ray.id, lane_ids.data(),
            sizeof(uint32_t) * packet_width);
        std::memset(
            ray.flags, 0, sizeof(uint32_t) * packet_width);
        return;
    }
    for (auto lane = uint32_t{0u}; lane < packet_width; lane++) {
        auto active = lane_active(
            active_mask_bits, lane, lane_count);
        valid[lane] = active ? -1 : 0;
        auto component = [&](uint32_t index, float fallback = 0.0f) {
            return active ? components[index * lane_count + lane] :
                            fallback;
        };
        ray.org_x[lane] = component(0u);
        ray.org_y[lane] = component(1u);
        ray.org_z[lane] = component(2u);
        ray.tnear[lane] = component(3u);
        ray.dir_x[lane] = component(4u);
        ray.dir_y[lane] = component(5u);
        ray.dir_z[lane] = component(6u, 1.0f);
        ray.time[lane] = active && times != nullptr ? times[lane] : 0.0f;
        ray.tfar[lane] = component(7u);
        ray.mask[lane] = active ? visibility_masks[lane] : 0u;
        ray.id[lane] = lane;
        ray.flags[lane] = 0u;
    }
}

template<size_t packet_width, typename HitPacket>
void initialize_hit_packet(HitPacket &hit) noexcept {
    for (auto lane = uint32_t{0u}; lane < packet_width; lane++) {
        hit.Ng_x[lane] = 0.0f;
        hit.Ng_y[lane] = 0.0f;
        hit.Ng_z[lane] = 0.0f;
        hit.u[lane] = 0.0f;
        hit.v[lane] = 0.0f;
        hit.primID[lane] = RTC_INVALID_GEOMETRY_ID;
        hit.geomID[lane] = RTC_INVALID_GEOMETRY_ID;
        for (auto level = 0u;
             level < RTC_MAX_INSTANCE_LEVEL_COUNT; level++) {
            hit.instID[level][lane] = RTC_INVALID_GEOMETRY_ID;
        }
    }
}

void initialize_scalar_ray(
    RTCRay &ray, uint32_t lane_count,
    uint64_t active_mask_bits, const float *components,
    const uint32_t *visibility_masks,
    const float *times) noexcept {
    auto active = lane_active(active_mask_bits, 0u, lane_count);
    auto component = [&](uint32_t index, float fallback = 0.0f) {
        return active ? components[index * lane_count] : fallback;
    };
    ray.org_x = component(0u);
    ray.org_y = component(1u);
    ray.org_z = component(2u);
    ray.tnear = component(3u);
    ray.dir_x = component(4u);
    ray.dir_y = component(5u);
    ray.dir_z = component(6u, 1.0f);
    ray.time = active && times != nullptr ? times[0u] : 0.0f;
    ray.tfar = component(7u);
    ray.mask = active ? visibility_masks[0u] : 0u;
    ray.id = 0u;
    ray.flags = 0u;
}

void initialize_scalar_hit(RTCHit &hit) noexcept {
    hit.Ng_x = 0.0f;
    hit.Ng_y = 0.0f;
    hit.Ng_z = 0.0f;
    hit.u = 0.0f;
    hit.v = 0.0f;
    hit.primID = RTC_INVALID_GEOMETRY_ID;
    hit.geomID = RTC_INVALID_GEOMETRY_ID;
    for (auto &instance_id : hit.instID) {
        instance_id = RTC_INVALID_GEOMETRY_ID;
    }
}

void intersect_scalar(RTCScene scene, RTCRayHit &ray_hit) noexcept {
#if LUISA_COMPUTE_SIMD_EMBREE_VERSION == 3
    RTCIntersectContext context{};
    rtcInitIntersectContext(&context);
    rtcIntersect1(scene, &context, &ray_hit);
#else
    RTCIntersectArguments arguments{};
    rtcInitIntersectArguments(&arguments);
    rtcIntersect1(scene, &ray_hit, &arguments);
#endif
}

void occlude_scalar(RTCScene scene, RTCRay &ray) noexcept {
#if LUISA_COMPUTE_SIMD_EMBREE_VERSION == 3
    RTCIntersectContext context{};
    rtcInitIntersectContext(&context);
    rtcOccluded1(scene, &context, &ray);
#else
    RTCOccludedArguments arguments{};
    rtcInitOccludedArguments(&arguments);
    rtcOccluded1(scene, &ray, &arguments);
#endif
}

template<size_t packet_width, typename RayHitPacket, typename Invoke>
void trace_closest_packet(
    RTCScene scene, uint32_t lane_count,
    uint64_t active_mask_bits, const float *components,
    const uint32_t *visibility_masks,
    const float *times, uint32_t *hit_ids, float *hit_values,
    Invoke &&invoke) noexcept {
    alignas(64) RayHitPacket packet{};
    alignas(64) std::array<int, packet_width> valid{};
    initialize_ray_packet<packet_width>(
        packet.ray, valid, lane_count, active_mask_bits,
        components, visibility_masks, times);
    initialize_hit_packet<packet_width>(packet.hit);
    invoke(valid.data(), scene, &packet);
    if (lane_count == packet_width) {
        constexpr auto component_bytes =
            sizeof(uint32_t) * packet_width;
        // Hit scratch is initialized before traversal, so copying inactive
        // lanes back only publishes benign miss values into already masked
        // LLVM scratch. It cannot expose an uninitialized Embree field.
        std::memcpy(
            hit_ids, packet.hit.instID[0u], component_bytes);
        std::memcpy(
            hit_ids + packet_width, packet.hit.primID,
            component_bytes);
        std::memcpy(
            hit_values, packet.hit.u, component_bytes);
        std::memcpy(
            hit_values + packet_width, packet.hit.v,
            component_bytes);
        std::memcpy(
            hit_values + 2u * packet_width, packet.ray.tfar,
            component_bytes);
        return;
    }
    for (auto lane = uint32_t{0u}; lane < lane_count; lane++) {
        if (!lane_active(active_mask_bits, lane, lane_count)) { continue; }
        hit_ids[lane] = packet.hit.instID[0u][lane];
        hit_ids[lane_count + lane] = packet.hit.primID[lane];
        hit_values[lane] = packet.hit.u[lane];
        hit_values[lane_count + lane] = packet.hit.v[lane];
        hit_values[2u * lane_count + lane] = packet.ray.tfar[lane];
    }
}

template<size_t packet_width, typename RayPacket, typename Invoke>
void trace_any_packet(
    RTCScene scene, uint32_t lane_count,
    uint64_t active_mask_bits, const float *components,
    const uint32_t *visibility_masks, const float *times,
    uint32_t *occluded,
    Invoke &&invoke) noexcept {
    alignas(64) RayPacket packet{};
    alignas(64) std::array<int, packet_width> valid{};
    initialize_ray_packet<packet_width>(
        packet, valid, lane_count, active_mask_bits,
        components, visibility_masks, times);
    invoke(valid.data(), scene, &packet);
    if (lane_count == packet_width) {
        for (auto lane = uint32_t{0u}; lane < packet_width; lane++) {
            occluded[lane] = packet.tfar[lane] < 0.0f ? 1u : 0u;
        }
        return;
    }
    for (auto lane = uint32_t{0u}; lane < lane_count; lane++) {
        if (!lane_active(active_mask_bits, lane, lane_count)) { continue; }
        occluded[lane] = packet.tfar[lane] < 0.0f ? 1u : 0u;
    }
}

}// namespace

SIMDAccel::SIMDAccel(
    RTCDevice device, const AccelOption &option) noexcept
    : _scene{rtcNewScene(device)} {
    simd_accel_set_flags(_scene, option);
}

SIMDAccel::~SIMDAccel() noexcept { rtcReleaseScene(_scene); }

void SIMDAccel::build(const AccelBuildCommand &command) noexcept {
    LUISA_ASSERT(
        !command.update_instance_buffer_only(),
        "SIMD acceleration structures do not yet support "
        "update_instance_buffer_only.");
    auto instance_count = command.instance_count();
    if (instance_count < _instances.size()) {
        for (auto i = instance_count; i < _instances.size(); i++) {
            rtcDetachGeometry(_scene, static_cast<unsigned>(i));
        }
        _instances.resize(instance_count);
        _geometries.resize(instance_count);
    } else {
        auto device = rtcGetSceneDevice(_scene);
        _instances.reserve(instance_count);
        _geometries.reserve(instance_count);
        for (auto i = _instances.size(); i < instance_count; i++) {
            auto geometry = rtcNewGeometry(
                device, RTC_GEOMETRY_TYPE_INSTANCE);
            rtcSetGeometryBuildQuality(
                geometry, RTC_BUILD_QUALITY_HIGH);
            rtcAttachGeometryByID(
                _scene, geometry, static_cast<unsigned>(i));
            rtcReleaseGeometry(geometry);
            _instances.emplace_back();
            _geometries.emplace_back(geometry);
        }
    }
    // Shader arguments may retain the table address across a later build.
    // Publish the current vector storage only after every possible reallocation.
    _instance_table.data = _instances.data();
    _instance_table.size = _instances.size();

    using Modification = AccelBuildCommand::Modification;
    for (auto modification : command.modifications()) {
        LUISA_ASSERT(
            modification.index < _instances.size(),
            "SIMD accel modification index is out of range.");
        auto &instance = _instances[modification.index];
        auto geometry = _geometries[modification.index];
        if ((modification.flags & Modification::flag_primitive) != 0u) {
            auto *mesh = reinterpret_cast<SIMDMesh *>(
                modification.primitive);
            LUISA_ASSERT(
                mesh != nullptr,
                "SIMD accel instance has a null mesh.");
            rtcSetGeometryInstancedScene(geometry, mesh->handle());
            rtcSetGeometryTimeStepCount(geometry, 1u);
            rtcSetGeometryTimeRange(geometry, 0.0f, 1.0f);
        }
        if ((modification.flags & Modification::flag_transform) != 0u) {
            std::memcpy(
                instance.affine, modification.affine,
                sizeof(instance.affine));
        }
        if ((modification.flags & Modification::flag_visibility) != 0u) {
            instance.mask = modification.vis_mask;
        }
        if ((modification.flags & Modification::flag_user_id) != 0u) {
            instance.user_id = modification.user_id;
        }
        if ((modification.flags & Modification::flag_opaque) != 0u) {
            instance.opaque =
                (modification.flags & Modification::flag_opaque_on) != 0u ?
                    1u :
                    0u;
        }
        instance.dirty = 1u;
    }
    for (auto i = size_t{0u}; i < _instances.size(); i++) {
        auto &instance = _instances[i];
        if (!instance.dirty) { continue; }
        auto geometry = _geometries[i];
        rtcSetGeometryTransform(
            geometry, 0u, RTC_FORMAT_FLOAT3X4_ROW_MAJOR,
            instance.affine);
        rtcSetGeometryMask(geometry, instance.mask);
        rtcCommitGeometry(geometry);
        instance.dirty = 0u;
    }
    rtcCommitScene(_scene);
}

void SIMDAccel::_trace_closest(
    void *accel, uint32_t lane_count,
    uint64_t active_mask_bits, const float *ray_components,
    const uint32_t *visibility_masks,
    const float *times, uint32_t *hit_ids,
    float *hit_values) noexcept {
    auto *self = static_cast<SIMDAccel *>(accel);
    LUISA_ASSERT(
        self != nullptr && ray_components != nullptr &&
            visibility_masks != nullptr && hit_ids != nullptr &&
            hit_values != nullptr,
        "Invalid SIMD closest-hit packet arguments.");
    if (active_mask_bits == 0u) { return; }
    switch (lane_count) {
        case 1u: {
            RTCRayHit ray_hit{};
            initialize_scalar_ray(
                ray_hit.ray, lane_count, active_mask_bits,
                ray_components, visibility_masks, times);
            initialize_scalar_hit(ray_hit.hit);
            intersect_scalar(self->_scene, ray_hit);
            hit_ids[0u] = ray_hit.hit.instID[0u];
            hit_ids[1u] = ray_hit.hit.primID;
            hit_values[0u] = ray_hit.hit.u;
            hit_values[1u] = ray_hit.hit.v;
            hit_values[2u] = ray_hit.ray.tfar;
            break;
        }
        case 2u:
        case 4u:
            trace_closest_packet<4u, RTCRayHit4>(
                self->_scene, lane_count, active_mask_bits,
                ray_components, visibility_masks, times,
                hit_ids, hit_values,
                [](const int *valid, RTCScene scene,
                   RTCRayHit4 *packet) noexcept {
#if LUISA_COMPUTE_SIMD_EMBREE_VERSION == 3
                    RTCIntersectContext context{};
                    rtcInitIntersectContext(&context);
                    rtcIntersect4(valid, scene, &context, packet);
#else
                    RTCIntersectArguments arguments{};
                    rtcInitIntersectArguments(&arguments);
                    rtcIntersect4(valid, scene, packet, &arguments);
#endif
                });
            break;
        case 8u:
            trace_closest_packet<8u, RTCRayHit8>(
                self->_scene, lane_count, active_mask_bits,
                ray_components, visibility_masks, times,
                hit_ids, hit_values,
                [](const int *valid, RTCScene scene,
                   RTCRayHit8 *packet) noexcept {
#if LUISA_COMPUTE_SIMD_EMBREE_VERSION == 3
                    RTCIntersectContext context{};
                    rtcInitIntersectContext(&context);
                    rtcIntersect8(valid, scene, &context, packet);
#else
                    RTCIntersectArguments arguments{};
                    rtcInitIntersectArguments(&arguments);
                    rtcIntersect8(valid, scene, packet, &arguments);
#endif
                });
            break;
        case 16u:
            trace_closest_packet<16u, RTCRayHit16>(
                self->_scene, lane_count, active_mask_bits,
                ray_components, visibility_masks, times,
                hit_ids, hit_values,
                [](const int *valid, RTCScene scene,
                   RTCRayHit16 *packet) noexcept {
#if LUISA_COMPUTE_SIMD_EMBREE_VERSION == 3
                    RTCIntersectContext context{};
                    rtcInitIntersectContext(&context);
                    rtcIntersect16(valid, scene, &context, packet);
#else
                    RTCIntersectArguments arguments{};
                    rtcInitIntersectArguments(&arguments);
                    rtcIntersect16(valid, scene, packet, &arguments);
#endif
                });
            break;
        default:
            LUISA_ERROR_WITH_LOCATION(
                "Unsupported SIMD Embree packet width {}.", lane_count);
    }
}

void SIMDAccel::_trace_any(
    void *accel, uint32_t lane_count,
    uint64_t active_mask_bits, const float *ray_components,
    const uint32_t *visibility_masks,
    const float *times, uint32_t *occluded) noexcept {
    auto *self = static_cast<SIMDAccel *>(accel);
    LUISA_ASSERT(
        self != nullptr && ray_components != nullptr &&
            visibility_masks != nullptr && occluded != nullptr,
        "Invalid SIMD any-hit packet arguments.");
    if (active_mask_bits == 0u) { return; }
    switch (lane_count) {
        case 1u: {
            RTCRay ray{};
            initialize_scalar_ray(
                ray, lane_count, active_mask_bits,
                ray_components, visibility_masks, times);
            occlude_scalar(self->_scene, ray);
            occluded[0u] = ray.tfar < 0.0f ? 1u : 0u;
            break;
        }
        case 2u:
        case 4u:
            trace_any_packet<4u, RTCRay4>(
                self->_scene, lane_count, active_mask_bits,
                ray_components, visibility_masks, times, occluded,
                [](const int *valid, RTCScene scene,
                   RTCRay4 *packet) noexcept {
#if LUISA_COMPUTE_SIMD_EMBREE_VERSION == 3
                    RTCIntersectContext context{};
                    rtcInitIntersectContext(&context);
                    rtcOccluded4(valid, scene, &context, packet);
#else
                    RTCOccludedArguments arguments{};
                    rtcInitOccludedArguments(&arguments);
                    rtcOccluded4(valid, scene, packet, &arguments);
#endif
                });
            break;
        case 8u:
            trace_any_packet<8u, RTCRay8>(
                self->_scene, lane_count, active_mask_bits,
                ray_components, visibility_masks, times, occluded,
                [](const int *valid, RTCScene scene,
                   RTCRay8 *packet) noexcept {
#if LUISA_COMPUTE_SIMD_EMBREE_VERSION == 3
                    RTCIntersectContext context{};
                    rtcInitIntersectContext(&context);
                    rtcOccluded8(valid, scene, &context, packet);
#else
                    RTCOccludedArguments arguments{};
                    rtcInitOccludedArguments(&arguments);
                    rtcOccluded8(valid, scene, packet, &arguments);
#endif
                });
            break;
        case 16u:
            trace_any_packet<16u, RTCRay16>(
                self->_scene, lane_count, active_mask_bits,
                ray_components, visibility_masks, times, occluded,
                [](const int *valid, RTCScene scene,
                   RTCRay16 *packet) noexcept {
#if LUISA_COMPUTE_SIMD_EMBREE_VERSION == 3
                    RTCIntersectContext context{};
                    rtcInitIntersectContext(&context);
                    rtcOccluded16(valid, scene, &context, packet);
#else
                    RTCOccludedArguments arguments{};
                    rtcInitOccludedArguments(&arguments);
                    rtcOccluded16(valid, scene, packet, &arguments);
#endif
                });
            break;
        default:
            LUISA_ERROR_WITH_LOCATION(
                "Unsupported SIMD Embree packet width {}.", lane_count);
    }
}

}// namespace luisa::compute::simd
