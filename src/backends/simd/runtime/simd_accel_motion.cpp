#include "simd_accel_motion.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <limits>

#include <luisa/core/logging.h>

namespace luisa::compute::simd {
namespace {

void validate_finite(
    const float *values, size_t count,
    size_t keyframe, const char *field) noexcept {
    for (auto i = size_t{0u}; i < count; i++) {
        LUISA_ASSERT(
            std::isfinite(values[i]),
            "SIMD motion keyframe {} contains a non-finite {} component "
            "at index {}.",
            keyframe, field, i);
    }
}

[[nodiscard]] RTCQuaternionDecomposition quaternion_keyframe(
    const MotionInstanceTransformSRT &keyframe,
    size_t keyframe_index) noexcept {
    validate_finite(
        keyframe.pivot, 3u, keyframe_index, "pivot");
    validate_finite(
        keyframe.quaternion, 4u, keyframe_index, "quaternion");
    validate_finite(
        keyframe.scale, 3u, keyframe_index, "scale");
    validate_finite(
        keyframe.shear, 3u, keyframe_index, "shear");
    validate_finite(
        keyframe.translation, 3u, keyframe_index, "translation");
    auto norm_squared =
        keyframe.quaternion[0u] * keyframe.quaternion[0u] +
        keyframe.quaternion[1u] * keyframe.quaternion[1u] +
        keyframe.quaternion[2u] * keyframe.quaternion[2u] +
        keyframe.quaternion[3u] * keyframe.quaternion[3u];
    LUISA_ASSERT(
        norm_squared > 0.0f,
        "SIMD SRT motion keyframe {} has a zero quaternion.",
        keyframe_index);
    return RTCQuaternionDecomposition{
        .scale_x = keyframe.scale[0u],
        .scale_y = keyframe.scale[1u],
        .scale_z = keyframe.scale[2u],
        .skew_xy = keyframe.shear[0u],
        .skew_xz = keyframe.shear[1u],
        .skew_yz = keyframe.shear[2u],
        .shift_x = keyframe.pivot[0u],
        .shift_y = keyframe.pivot[1u],
        .shift_z = keyframe.pivot[2u],
        .quaternion_r = keyframe.quaternion[3u],
        .quaternion_i = keyframe.quaternion[0u],
        .quaternion_j = keyframe.quaternion[1u],
        .quaternion_k = keyframe.quaternion[2u],
        .translation_x = keyframe.translation[0u],
        .translation_y = keyframe.translation[1u],
        .translation_z = keyframe.translation[2u],
    };
}

[[nodiscard]] RTCBounds transform_bounds(
    const RTCBounds &bounds, const float affine[12]) noexcept {
    if (bounds.lower_x > bounds.upper_x ||
        bounds.lower_y > bounds.upper_y ||
        bounds.lower_z > bounds.upper_z) {
        constexpr auto infinity =
            std::numeric_limits<float>::infinity();
        return RTCBounds{
            .lower_x = infinity,
            .lower_y = infinity,
            .lower_z = infinity,
            .align0 = 0.0f,
            .upper_x = -infinity,
            .upper_y = -infinity,
            .upper_z = -infinity,
            .align1 = 0.0f,
        };
    }
    auto center = std::array{
        0.5f * (bounds.lower_x + bounds.upper_x),
        0.5f * (bounds.lower_y + bounds.upper_y),
        0.5f * (bounds.lower_z + bounds.upper_z)};
    auto extent = std::array{
        0.5f * (bounds.upper_x - bounds.lower_x),
        0.5f * (bounds.upper_y - bounds.lower_y),
        0.5f * (bounds.upper_z - bounds.lower_z)};
    auto transformed_center = std::array<float, 3u>{};
    auto transformed_extent = std::array<float, 3u>{};
    for (auto row = 0u; row < 3u; row++) {
        transformed_center[row] =
            affine[row * 4u + 0u] * center[0u] +
            affine[row * 4u + 1u] * center[1u] +
            affine[row * 4u + 2u] * center[2u] +
            affine[row * 4u + 3u];
        transformed_extent[row] =
            std::abs(affine[row * 4u + 0u]) * extent[0u] +
            std::abs(affine[row * 4u + 1u]) * extent[1u] +
            std::abs(affine[row * 4u + 2u]) * extent[2u];
    }
    return RTCBounds{
        .lower_x = transformed_center[0u] - transformed_extent[0u],
        .lower_y = transformed_center[1u] - transformed_extent[1u],
        .lower_z = transformed_center[2u] - transformed_extent[2u],
        .align0 = 0.0f,
        .upper_x = transformed_center[0u] + transformed_extent[0u],
        .upper_y = transformed_center[1u] + transformed_extent[1u],
        .upper_z = transformed_center[2u] + transformed_extent[2u],
        .align1 = 0.0f,
    };
}

void compose_affine(
    float result[12], const float outer[12],
    const float inner[12]) noexcept {
    for (auto row = 0u; row < 3u; row++) {
        for (auto column = 0u; column < 4u; column++) {
            result[row * 4u + column] =
                outer[row * 4u + 0u] * inner[0u * 4u + column] +
                outer[row * 4u + 1u] * inner[1u * 4u + column] +
                outer[row * 4u + 2u] * inner[2u * 4u + column] +
                (column == 3u ? outer[row * 4u + 3u] : 0.0f);
        }
    }
}

[[nodiscard]] bool invert_affine(
    float inverse[12], const float affine[12]) noexcept {
    auto a = affine[0u];
    auto b = affine[1u];
    auto c = affine[2u];
    auto d = affine[4u];
    auto e = affine[5u];
    auto f = affine[6u];
    auto g = affine[8u];
    auto h = affine[9u];
    auto i = affine[10u];
    auto determinant =
        a * (e * i - f * h) -
        b * (d * i - f * g) +
        c * (d * h - e * g);
    if (!std::isfinite(determinant) || determinant == 0.0f) {
        return false;
    }
    auto reciprocal = 1.0f / determinant;
    inverse[0u] = (e * i - f * h) * reciprocal;
    inverse[1u] = (c * h - b * i) * reciprocal;
    inverse[2u] = (b * f - c * e) * reciprocal;
    inverse[4u] = (f * g - d * i) * reciprocal;
    inverse[5u] = (a * i - c * g) * reciprocal;
    inverse[6u] = (c * d - a * f) * reciprocal;
    inverse[8u] = (d * h - e * g) * reciprocal;
    inverse[9u] = (b * g - a * h) * reciprocal;
    inverse[10u] = (a * e - b * d) * reciprocal;
    for (auto row = 0u; row < 3u; row++) {
        inverse[row * 4u + 3u] =
            -(inverse[row * 4u + 0u] * affine[3u] +
              inverse[row * 4u + 1u] * affine[7u] +
              inverse[row * 4u + 2u] * affine[11u]);
    }
    return std::all_of(
        inverse, inverse + 12u,
        [](float value) noexcept { return std::isfinite(value); });
}

[[nodiscard]] std::array<float, 3u> transform_point(
    const float affine[12], float x, float y, float z) noexcept {
    return {
        affine[0u] * x + affine[1u] * y + affine[2u] * z + affine[3u],
        affine[4u] * x + affine[5u] * y + affine[6u] * z + affine[7u],
        affine[8u] * x + affine[9u] * y + affine[10u] * z + affine[11u]};
}

[[nodiscard]] std::array<float, 3u> transform_vector(
    const float affine[12], float x, float y, float z) noexcept {
    return {
        affine[0u] * x + affine[1u] * y + affine[2u] * z,
        affine[4u] * x + affine[5u] * y + affine[6u] * z,
        affine[8u] * x + affine[9u] * y + affine[10u] * z};
}

[[nodiscard]] std::array<float, 3u> transform_normal(
    const float inverse[12], float x, float y, float z) noexcept {
    return {
        inverse[0u] * x + inverse[4u] * y + inverse[8u] * z,
        inverse[1u] * x + inverse[5u] * y + inverse[9u] * z,
        inverse[2u] * x + inverse[6u] * y + inverse[10u] * z};
}

}// namespace

SIMDSRTMotionForwarder::SIMDSRTMotionForwarder(
    RTCDevice device, RTCScene child_scene,
    const AccelMotionOption &option,
    luisa::span<const MotionInstanceTransform> keyframes,
    const float outer[12]) noexcept
    : _transform_scene{rtcNewScene(device)},
      _child_scene{child_scene},
      _time_start{option.time_start},
      _time_end{option.time_end},
      _inverse_time_extent{1.0f / (option.time_end - option.time_start)} {
    LUISA_ASSERT(
        device != nullptr && child_scene != nullptr &&
            keyframes.size() == option.keyframe_count &&
            option.mode == AccelMotionMode::SRT,
        "Invalid SIMD SRT forwarding state.");
    std::memcpy(_outer.data(), outer, sizeof(float) * _outer.size());
    validate_finite(_outer.data(), _outer.size(), 0u, "outer affine");

    _transform_geometry =
        rtcNewGeometry(device, RTC_GEOMETRY_TYPE_INSTANCE);
    rtcSetGeometryInstancedScene(_transform_geometry, child_scene);
    rtcSetGeometryTimeStepCount(
        _transform_geometry, static_cast<unsigned>(keyframes.size()));
    // Normalize the helper geometry to [0, 1]. The public user geometry owns
    // the actual vanish interval; this makes transform lookup and linear
    // bounds independent of that interval's location.
    rtcSetGeometryTimeRange(_transform_geometry, 0.0f, 1.0f);
    for (auto key = size_t{0u}; key < keyframes.size(); key++) {
        auto quaternion = quaternion_keyframe(
            keyframes[key].as_srt(), key);
        rtcSetGeometryTransformQuaternion(
            _transform_geometry, static_cast<unsigned>(key), &quaternion);
    }
    rtcCommitGeometry(_transform_geometry);
    rtcAttachGeometryByID(_transform_scene, _transform_geometry, 0u);
    rtcCommitScene(_transform_scene);

    auto linear_bounds = RTCLinearBounds{};
    rtcGetSceneLinearBounds(_transform_scene, &linear_bounds);
    _bounds[0u] = transform_bounds(linear_bounds.bounds0, _outer.data());
    _bounds[1u] = transform_bounds(linear_bounds.bounds1, _outer.data());
}

SIMDSRTMotionForwarder::~SIMDSRTMotionForwarder() noexcept {
    if (_transform_scene != nullptr) {
        rtcReleaseScene(_transform_scene);
    }
    if (_transform_geometry != nullptr) {
        rtcReleaseGeometry(_transform_geometry);
    }
}

void SIMDSRTMotionForwarder::configure_geometry(
    RTCGeometry geometry) noexcept {
    LUISA_ASSERT(geometry != nullptr, "Invalid SIMD SRT user geometry.");
    rtcSetGeometryUserPrimitiveCount(geometry, 1u);
    rtcSetGeometryTimeStepCount(geometry, 2u);
    rtcSetGeometryTimeRange(
        geometry, _time_start, _time_end);
    rtcSetGeometryUserData(geometry, this);
    rtcSetGeometryBoundsFunction(
        geometry, _bounds_callback, nullptr);
    rtcSetGeometryIntersectFunction(
        geometry, _intersect_callback);
    rtcSetGeometryOccludedFunction(
        geometry, _occluded_callback);
}

void SIMDSRTMotionForwarder::_bounds_callback(
    const RTCBoundsFunctionArguments *arguments) noexcept {
    auto *self = static_cast<const SIMDSRTMotionForwarder *>(
        arguments == nullptr ? nullptr : arguments->geometryUserPtr);
    LUISA_ASSERT(
        self != nullptr && arguments->bounds_o != nullptr &&
            arguments->primID == 0u && arguments->timeStep < 2u,
        "Invalid SIMD SRT bounds callback invocation.");
    *arguments->bounds_o = self->_bounds[arguments->timeStep];
}

namespace {

[[nodiscard]] bool inverse_at_time(
    const SIMDSRTMotionForwarder &forwarder,
    float time, float inverse[12]) noexcept {
    if (!std::isfinite(time)) { return false; }
    auto normalized = std::clamp(
        (time - forwarder.time_start()) *
            forwarder.inverse_time_extent(),
        0.0f, 1.0f);
    auto inner = std::array<float, 12u>{};
    rtcGetGeometryTransform(
        forwarder.transform_geometry(), normalized,
        RTC_FORMAT_FLOAT3X4_ROW_MAJOR, inner.data());
    auto composed = std::array<float, 12u>{};
    compose_affine(
        composed.data(), forwarder.outer(), inner.data());
    return invert_affine(inverse, composed.data());
}

template<size_t Width, typename RayPacket>
void initialize_forward_packet(
    const SIMDSRTMotionForwarder &forwarder,
    RTCRayN *source, const int *valid,
    RayPacket &packet,
    std::array<std::array<float, 12u>, Width> &inverses,
    std::array<bool, Width> &invertible) noexcept {
    for (auto lane = 0u; lane < Width; lane++) {
        if (valid[lane] == 0) { continue; }
        packet.tnear[lane] = RTCRayN_tnear(source, Width, lane);
        packet.time[lane] = RTCRayN_time(source, Width, lane);
        packet.tfar[lane] = RTCRayN_tfar(source, Width, lane);
        packet.mask[lane] = RTCRayN_mask(source, Width, lane);
        packet.id[lane] = RTCRayN_id(source, Width, lane);
        packet.flags[lane] = RTCRayN_flags(source, Width, lane);
        invertible[lane] = inverse_at_time(
            forwarder,
            packet.time[lane],
            inverses[lane].data());
        if (!invertible[lane]) { continue; }
        auto origin = transform_point(
            inverses[lane].data(),
            RTCRayN_org_x(source, Width, lane),
            RTCRayN_org_y(source, Width, lane),
            RTCRayN_org_z(source, Width, lane));
        auto direction = transform_vector(
            inverses[lane].data(),
            RTCRayN_dir_x(source, Width, lane),
            RTCRayN_dir_y(source, Width, lane),
            RTCRayN_dir_z(source, Width, lane));
        packet.org_x[lane] = origin[0u];
        packet.org_y[lane] = origin[1u];
        packet.org_z[lane] = origin[2u];
        packet.dir_x[lane] = direction[0u];
        packet.dir_y[lane] = direction[1u];
        packet.dir_z[lane] = direction[2u];
    }
}

#if LUISA_COMPUTE_SIMD_EMBREE_VERSION == 3

void push_instance_id(
    RTCIntersectContext *context, uint32_t instance) noexcept {
    LUISA_ASSERT(
        context != nullptr,
        "Embree 3 SRT forwarding requires an intersection context.");
#if RTC_MAX_INSTANCE_LEVEL_COUNT > 1
    LUISA_ASSERT(
        context->instStackSize < RTC_MAX_INSTANCE_LEVEL_COUNT,
        "Embree 3 SRT forwarding exhausted the instance-ID stack.");
    context->instID[context->instStackSize++] = instance;
#else
    LUISA_ASSERT(
        context->instID[0u] == RTC_INVALID_GEOMETRY_ID,
        "Embree 3 SRT forwarding cannot overwrite an active instance ID.");
    context->instID[0u] = instance;
#endif
}

void pop_instance_id(RTCIntersectContext *context) noexcept {
#if RTC_MAX_INSTANCE_LEVEL_COUNT > 1
    LUISA_ASSERT(
        context != nullptr && context->instStackSize != 0u,
        "Embree 3 SRT forwarding instance-ID stack underflow.");
    context->instID[--context->instStackSize] =
        RTC_INVALID_GEOMETRY_ID;
#else
    context->instID[0u] = RTC_INVALID_GEOMETRY_ID;
#endif
}

template<size_t Width, typename HitPacket>
void initialize_recursive_hit(HitPacket &hit) noexcept {
    for (auto lane = 0u; lane < Width; lane++) {
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

template<size_t Width, typename RayHitPacket>
void copy_recursive_hits(
    RayHitPacket &destination, const RayHitPacket &source,
    const int *valid) noexcept {
    for (auto lane = 0u; lane < Width; lane++) {
        if (valid[lane] == 0 ||
            source.hit.geomID[lane] == RTC_INVALID_GEOMETRY_ID) {
            continue;
        }
        destination.ray.tfar[lane] = source.ray.tfar[lane];
        destination.hit.Ng_x[lane] = source.hit.Ng_x[lane];
        destination.hit.Ng_y[lane] = source.hit.Ng_y[lane];
        destination.hit.Ng_z[lane] = source.hit.Ng_z[lane];
        destination.hit.u[lane] = source.hit.u[lane];
        destination.hit.v[lane] = source.hit.v[lane];
        destination.hit.primID[lane] = source.hit.primID[lane];
        destination.hit.geomID[lane] = source.hit.geomID[lane];
        for (auto level = 0u;
             level < RTC_MAX_INSTANCE_LEVEL_COUNT; level++) {
            destination.hit.instID[level][lane] =
                source.hit.instID[level][lane];
        }
    }
}

#endif

template<size_t Width, typename HitPacket>
void transform_hit_normals(
    const RTCIntersectFunctionNArguments *arguments,
    HitPacket &hit,
    const std::array<std::array<float, 12u>, Width> &inverses,
    const std::array<bool, Width> &invertible) noexcept {
    for (auto lane = 0u; lane < Width; lane++) {
        if (arguments->valid[lane] == 0 || !invertible[lane] ||
            hit.instID[0u][lane] != arguments->geomID) {
            continue;
        }
        auto normal = transform_normal(
            inverses[lane].data(),
            hit.Ng_x[lane], hit.Ng_y[lane], hit.Ng_z[lane]);
        hit.Ng_x[lane] = normal[0u];
        hit.Ng_y[lane] = normal[1u];
        hit.Ng_z[lane] = normal[2u];
    }
}

template<size_t Width, typename RayHitPacket, typename RayPacket>
void forward_intersect_packet(
    const SIMDSRTMotionForwarder &forwarder,
    const RTCIntersectFunctionNArguments *arguments) noexcept {
    auto *source = RTCRayHitN_RayN(arguments->rayhit, Width);
    auto *hit = reinterpret_cast<RayHitPacket *>(arguments->rayhit);
    auto inverses =
        std::array<std::array<float, 12u>, Width>{};
    auto invertible = std::array<bool, Width>{};
#if LUISA_COMPUTE_SIMD_EMBREE_VERSION == 3
    auto packet = RayHitPacket{};
    initialize_recursive_hit<Width>(packet.hit);
    auto &ray = packet.ray;
#else
    auto ray = RayPacket{};
#endif
    initialize_forward_packet(
        forwarder, source, arguments->valid,
        ray, inverses, invertible);
    auto forwarded_valid = std::array<int, Width>{};
    for (auto lane = 0u; lane < Width; lane++) {
        forwarded_valid[lane] =
            invertible[lane] ? arguments->valid[lane] : 0;
    }
#if LUISA_COMPUTE_SIMD_EMBREE_VERSION == 3
    push_instance_id(arguments->context, arguments->geomID);
    if constexpr (Width == 4u) {
        rtcIntersect4(
            forwarded_valid.data(), forwarder.child_scene(),
            arguments->context, &packet);
    } else if constexpr (Width == 8u) {
        rtcIntersect8(
            forwarded_valid.data(), forwarder.child_scene(),
            arguments->context, &packet);
    } else {
        static_assert(Width == 16u);
        rtcIntersect16(
            forwarded_valid.data(), forwarder.child_scene(),
            arguments->context, &packet);
    }
    pop_instance_id(arguments->context);
    copy_recursive_hits<Width>(*hit, packet, forwarded_valid.data());
#else
    if constexpr (Width == 4u) {
        rtcForwardIntersect4(
            forwarded_valid.data(), arguments,
            forwarder.child_scene(), &ray, arguments->geomID);
    } else if constexpr (Width == 8u) {
        rtcForwardIntersect8(
            forwarded_valid.data(), arguments,
            forwarder.child_scene(), &ray, arguments->geomID);
    } else {
        static_assert(Width == 16u);
        rtcForwardIntersect16(
            forwarded_valid.data(), arguments,
            forwarder.child_scene(), &ray, arguments->geomID);
    }
#endif
    transform_hit_normals(
        arguments, hit->hit, inverses, invertible);
}

template<size_t Width, typename RayPacket>
void forward_occluded_packet(
    const SIMDSRTMotionForwarder &forwarder,
    const RTCOccludedFunctionNArguments *arguments) noexcept {
    auto packet = RayPacket{};
    auto inverses =
        std::array<std::array<float, 12u>, Width>{};
    auto invertible = std::array<bool, Width>{};
    initialize_forward_packet(
        forwarder, arguments->ray, arguments->valid,
        packet, inverses, invertible);
    auto forwarded_valid = std::array<int, Width>{};
    for (auto lane = 0u; lane < Width; lane++) {
        forwarded_valid[lane] =
            invertible[lane] ? arguments->valid[lane] : 0;
    }
#if LUISA_COMPUTE_SIMD_EMBREE_VERSION == 3
    push_instance_id(arguments->context, arguments->geomID);
    if constexpr (Width == 4u) {
        rtcOccluded4(
            forwarded_valid.data(), forwarder.child_scene(),
            arguments->context, &packet);
    } else if constexpr (Width == 8u) {
        rtcOccluded8(
            forwarded_valid.data(), forwarder.child_scene(),
            arguments->context, &packet);
    } else {
        static_assert(Width == 16u);
        rtcOccluded16(
            forwarded_valid.data(), forwarder.child_scene(),
            arguments->context, &packet);
    }
    pop_instance_id(arguments->context);
    auto *source = reinterpret_cast<RayPacket *>(arguments->ray);
    for (auto lane = 0u; lane < Width; lane++) {
        if (forwarded_valid[lane] != 0) {
            source->tfar[lane] = packet.tfar[lane];
        }
    }
#else
    if constexpr (Width == 4u) {
        rtcForwardOccluded4(
            forwarded_valid.data(), arguments,
            forwarder.child_scene(), &packet, arguments->geomID);
    } else if constexpr (Width == 8u) {
        rtcForwardOccluded8(
            forwarded_valid.data(), arguments,
            forwarder.child_scene(), &packet, arguments->geomID);
    } else {
        static_assert(Width == 16u);
        rtcForwardOccluded16(
            forwarded_valid.data(), arguments,
            forwarder.child_scene(), &packet, arguments->geomID);
    }
#endif
}

}// namespace

void SIMDSRTMotionForwarder::_intersect_callback(
    const RTCIntersectFunctionNArguments *arguments) noexcept {
    auto *self = static_cast<const SIMDSRTMotionForwarder *>(
        arguments == nullptr ? nullptr : arguments->geometryUserPtr);
    LUISA_ASSERT(
        self != nullptr && arguments->valid != nullptr &&
            arguments->rayhit != nullptr,
        "Invalid SIMD SRT intersect callback invocation.");
    switch (arguments->N) {
        case 1u: {
            auto *ray_hit = reinterpret_cast<RTCRayHit *>(arguments->rayhit);
            auto inverse = std::array<float, 12u>{};
            if (arguments->valid[0u] == 0 ||
                !inverse_at_time(*self, ray_hit->ray.time, inverse.data())) {
                return;
            }
            auto forwarded_ray = ray_hit->ray;
            auto origin = transform_point(
                inverse.data(), ray_hit->ray.org_x,
                ray_hit->ray.org_y, ray_hit->ray.org_z);
            auto direction = transform_vector(
                inverse.data(), ray_hit->ray.dir_x,
                ray_hit->ray.dir_y, ray_hit->ray.dir_z);
            forwarded_ray.org_x = origin[0u];
            forwarded_ray.org_y = origin[1u];
            forwarded_ray.org_z = origin[2u];
            forwarded_ray.dir_x = direction[0u];
            forwarded_ray.dir_y = direction[1u];
            forwarded_ray.dir_z = direction[2u];
#if LUISA_COMPUTE_SIMD_EMBREE_VERSION == 3
            auto forwarded_hit = RTCRayHit{.ray = forwarded_ray};
            forwarded_hit.hit.Ng_x = 0.0f;
            forwarded_hit.hit.Ng_y = 0.0f;
            forwarded_hit.hit.Ng_z = 0.0f;
            forwarded_hit.hit.u = 0.0f;
            forwarded_hit.hit.v = 0.0f;
            forwarded_hit.hit.primID = RTC_INVALID_GEOMETRY_ID;
            forwarded_hit.hit.geomID = RTC_INVALID_GEOMETRY_ID;
            for (auto &instance : forwarded_hit.hit.instID) {
                instance = RTC_INVALID_GEOMETRY_ID;
            }
            push_instance_id(arguments->context, arguments->geomID);
            rtcIntersect1(
                self->_child_scene, arguments->context,
                &forwarded_hit);
            pop_instance_id(arguments->context);
            if (forwarded_hit.hit.geomID != RTC_INVALID_GEOMETRY_ID) {
                ray_hit->ray.tfar = forwarded_hit.ray.tfar;
                ray_hit->hit = forwarded_hit.hit;
            }
#else
            rtcForwardIntersect1(
                arguments, self->_child_scene,
                &forwarded_ray, arguments->geomID);
#endif
            if (arguments->valid[0u] != 0 &&
                ray_hit->hit.instID[0u] == arguments->geomID) {
                auto normal = transform_normal(
                    inverse.data(), ray_hit->hit.Ng_x,
                    ray_hit->hit.Ng_y, ray_hit->hit.Ng_z);
                ray_hit->hit.Ng_x = normal[0u];
                ray_hit->hit.Ng_y = normal[1u];
                ray_hit->hit.Ng_z = normal[2u];
            }
            break;
        }
        case 4u:
            forward_intersect_packet<4u, RTCRayHit4, RTCRay4>(
                *self, arguments);
            break;
        case 8u:
            forward_intersect_packet<8u, RTCRayHit8, RTCRay8>(
                *self, arguments);
            break;
        case 16u:
            forward_intersect_packet<16u, RTCRayHit16, RTCRay16>(
                *self, arguments);
            break;
        default:
            LUISA_ERROR_WITH_LOCATION(
                "Unsupported Embree SRT forwarding width {}.",
                arguments->N);
    }
}

void SIMDSRTMotionForwarder::_occluded_callback(
    const RTCOccludedFunctionNArguments *arguments) noexcept {
    auto *self = static_cast<const SIMDSRTMotionForwarder *>(
        arguments == nullptr ? nullptr : arguments->geometryUserPtr);
    LUISA_ASSERT(
        self != nullptr && arguments->valid != nullptr &&
            arguments->ray != nullptr,
        "Invalid SIMD SRT occluded callback invocation.");
    switch (arguments->N) {
        case 1u: {
            auto *ray = reinterpret_cast<RTCRay *>(arguments->ray);
            auto inverse = std::array<float, 12u>{};
            if (arguments->valid[0u] == 0 ||
                !inverse_at_time(*self, ray->time, inverse.data())) {
                return;
            }
            auto forwarded_ray = *ray;
            auto origin = transform_point(
                inverse.data(), ray->org_x, ray->org_y, ray->org_z);
            auto direction = transform_vector(
                inverse.data(), ray->dir_x, ray->dir_y, ray->dir_z);
            forwarded_ray.org_x = origin[0u];
            forwarded_ray.org_y = origin[1u];
            forwarded_ray.org_z = origin[2u];
            forwarded_ray.dir_x = direction[0u];
            forwarded_ray.dir_y = direction[1u];
            forwarded_ray.dir_z = direction[2u];
#if LUISA_COMPUTE_SIMD_EMBREE_VERSION == 3
            push_instance_id(arguments->context, arguments->geomID);
            rtcOccluded1(
                self->_child_scene, arguments->context,
                &forwarded_ray);
            pop_instance_id(arguments->context);
            ray->tfar = forwarded_ray.tfar;
#else
            rtcForwardOccluded1(
                arguments, self->_child_scene,
                &forwarded_ray, arguments->geomID);
#endif
            break;
        }
        case 4u:
            forward_occluded_packet<4u, RTCRay4>(
                *self, arguments);
            break;
        case 8u:
            forward_occluded_packet<8u, RTCRay8>(
                *self, arguments);
            break;
        case 16u:
            forward_occluded_packet<16u, RTCRay16>(
                *self, arguments);
            break;
        default:
            LUISA_ERROR_WITH_LOCATION(
                "Unsupported Embree SRT forwarding width {}.",
                arguments->N);
    }
}

}// namespace luisa::compute::simd
