#include "simd_accel.h"
#include "simd_accel_ray_query.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstring>
#include <limits>
#include <type_traits>

#include <luisa/core/logging.h>

#include "../../common/env_flag.h"
#include "simd_motion_instance.h"
#include "simd_procedural_primitive.h"

namespace luisa::compute::simd {

static_assert(
    static_cast<uint32_t>(AccelMotionMode::MATRIX) ==
    static_cast<uint32_t>(SIMDHostAccelMotionMode::matrix));
static_assert(
    static_cast<uint32_t>(AccelMotionMode::SRT) ==
    static_cast<uint32_t>(SIMDHostAccelMotionMode::srt));
static_assert(
    sizeof(MotionInstanceTransform) ==
    simd_host_accel_motion_frame_size);

#if defined(RTC_GEOMETRY_INSTANCE_ARRAY)
inline constexpr auto embree_hit_packet_field_count =
    7u + 2u * RTC_MAX_INSTANCE_LEVEL_COUNT;
#else
inline constexpr auto embree_hit_packet_field_count =
    7u + RTC_MAX_INSTANCE_LEVEL_COUNT;
#endif
inline constexpr auto embree_ray_packet_field_count = 12u;

static_assert(sizeof(int) == sizeof(uint32_t));
static_assert(std::bit_cast<uint32_t>(-1) == ~uint32_t{0u});
static_assert(sizeof(RTCRay) ==
              embree_ray_packet_field_count * sizeof(uint32_t));
static_assert(alignof(RTCRay) == 16u);
static_assert(offsetof(RTCRay, tfar) ==
              simd_host_accel_ray_tfar_field * sizeof(uint32_t));
static_assert(offsetof(RTCRay, id) ==
              simd_host_accel_ray_id_field * sizeof(uint32_t));
static_assert(offsetof(RTCHit, u) == 3u * sizeof(uint32_t));
static_assert(offsetof(RTCHit, v) == 4u * sizeof(uint32_t));
static_assert(offsetof(RTCHit, primID) == 5u * sizeof(uint32_t));
static_assert(offsetof(RTCHit, geomID) == 6u * sizeof(uint32_t));
static_assert(offsetof(RTCHit, instID) == 7u * sizeof(uint32_t));
static_assert(sizeof(RTCHit) >=
              embree_hit_packet_field_count * sizeof(uint32_t));
static_assert(sizeof(RTCHit) % alignof(RTCHit) == 0u);
static_assert(offsetof(RTCRayHit, hit) == sizeof(RTCRay));
static_assert(sizeof(RTCRayHit) >=
              sizeof(RTCRay) + sizeof(RTCHit));

#define LUISA_SIMD_CHECK_EMBREE_PACKET_LAYOUT(width)                   \
    static_assert(sizeof(RTCRay##width) ==                             \
                  embree_ray_packet_field_count * width *              \
                      sizeof(uint32_t));                               \
    static_assert(alignof(RTCRay##width) == width * sizeof(uint32_t)); \
    static_assert(offsetof(RTCRay##width, tfar) ==                     \
                  simd_host_accel_ray_tfar_field * width *             \
                      sizeof(uint32_t));                               \
    static_assert(offsetof(RTCRay##width, id) ==                       \
                  simd_host_accel_ray_id_field * width *               \
                      sizeof(uint32_t));                               \
    static_assert(offsetof(RTCHit##width, u) ==                        \
                  3u * width * sizeof(uint32_t));                      \
    static_assert(offsetof(RTCHit##width, v) ==                        \
                  4u * width * sizeof(uint32_t));                      \
    static_assert(offsetof(RTCHit##width, primID) ==                   \
                  5u * width * sizeof(uint32_t));                      \
    static_assert(offsetof(RTCHit##width, geomID) ==                   \
                  6u * width * sizeof(uint32_t));                      \
    static_assert(offsetof(RTCHit##width, instID) ==                   \
                  7u * width * sizeof(uint32_t));                      \
    static_assert(sizeof(RTCHit##width) ==                             \
                  embree_hit_packet_field_count * width *              \
                      sizeof(uint32_t));                               \
    static_assert(offsetof(RTCRayHit##width, hit) ==                   \
                  sizeof(RTCRay##width))

LUISA_SIMD_CHECK_EMBREE_PACKET_LAYOUT(4);
LUISA_SIMD_CHECK_EMBREE_PACKET_LAYOUT(8);
LUISA_SIMD_CHECK_EMBREE_PACKET_LAYOUT(16);
#undef LUISA_SIMD_CHECK_EMBREE_PACKET_LAYOUT

namespace {

[[nodiscard]] bool affine_is_identity(const float affine[12]) noexcept {
    return affine[0u] == 1.0f && affine[1u] == 0.0f &&
           affine[2u] == 0.0f && affine[3u] == 0.0f &&
           affine[4u] == 0.0f && affine[5u] == 1.0f &&
           affine[6u] == 0.0f && affine[7u] == 0.0f &&
           affine[8u] == 0.0f && affine[9u] == 0.0f &&
           affine[10u] == 1.0f && affine[11u] == 0.0f;
}

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

void compose_matrix_keyframe(
    float result[12], const float outer[12],
    const MotionInstanceTransformMatrix &keyframe,
    size_t keyframe_index) noexcept {
    validate_finite(
        &keyframe[0u][0u], 16u, keyframe_index, "matrix");
    for (auto row = 0u; row < 3u; row++) {
        for (auto column = 0u; column < 4u; column++) {
            result[row * 4u + column] =
                outer[row * 4u + 0u] * keyframe[column][0u] +
                outer[row * 4u + 1u] * keyframe[column][1u] +
                outer[row * 4u + 2u] * keyframe[column][2u] +
                outer[row * 4u + 3u] * keyframe[column][3u];
        }
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

template<typename Packet, typename Invoke>
void trace_packet_in_place(
    RTCScene scene, const int *valid,
    void *packet_storage, Invoke &&invoke) noexcept {
    invoke(
        valid, scene,
        static_cast<Packet *>(packet_storage));
}

template<typename Packet, typename Invoke>
void trace_w2_padded(
    RTCScene scene, const int *source_valid,
    void *packet_storage,
    Invoke &&invoke) noexcept {
    static_assert(
        std::is_same_v<Packet, RTCRay4> ||
        std::is_same_v<Packet, RTCRayHit4>);
    constexpr auto source_width = 2u;
    constexpr auto packet_width = 4u;
    constexpr auto field_count =
        sizeof(Packet) / (packet_width * sizeof(uint32_t));
    alignas(64) Packet padded{};
    if constexpr (std::is_same_v<Packet, RTCRayHit4>) {
        initialize_hit_packet<packet_width>(padded.hit);
        padded.ray.dir_z[2u] = 1.0f;
        padded.ray.dir_z[3u] = 1.0f;
    } else {
        padded.dir_z[2u] = 1.0f;
        padded.dir_z[3u] = 1.0f;
    }
    auto *source = static_cast<std::byte *>(packet_storage);
    auto *destination = reinterpret_cast<std::byte *>(&padded);
    for (auto field = uint32_t{0u}; field < field_count; field++) {
        std::memcpy(
            destination + field * packet_width * sizeof(uint32_t),
            source + field * source_width * sizeof(uint32_t),
            source_width * sizeof(uint32_t));
    }
    std::array<int, packet_width> valid{};
    valid[0u] = source_valid[0u];
    valid[1u] = source_valid[1u];
    invoke(valid.data(), scene, &padded);
    // Copy the complete public packet ABI back. Inactive fields were already
    // sanitized by the JIT, while the two padded lanes remain unreachable.
    // This also transports tfar and closest-hit fields without a second ABI.
    for (auto field = uint32_t{0u}; field < field_count; field++) {
        std::memcpy(
            source + field * source_width * sizeof(uint32_t),
            destination + field * packet_width * sizeof(uint32_t),
            source_width * sizeof(uint32_t));
    }
}

void mark_curve_surface_hits(
    const SIMDHostAccelInstanceTable &instances,
    uint32_t lane_count, const int *valid,
    const uint32_t *hit_instances, float *hit_v) noexcept {
    for (auto lane = 0u; lane < lane_count; lane++) {
        if (valid[lane] == 0) { continue; }
        auto inst = hit_instances[lane];
        if (inst == RTC_INVALID_GEOMETRY_ID) { continue; }
        LUISA_ASSERT(
            instances.data != nullptr && inst < instances.size,
            "SIMD curve trace returned an invalid instance ID {}.", inst);
        if (instances.data[inst].geometry_kind ==
            static_cast<uint8_t>(SIMDHostAccelGeometryKind::curve)) {
            hit_v[lane] = -1.0f;
        }
    }
}

using RayQueryBatchBuildState = detail::RayQueryBatchBuildState;
using RayQueryRTCContext = detail::RayQueryRTCContext;

struct RayQueryScanContext final : detail::RayQueryScanContext {};
static_assert(std::is_standard_layout_v<detail::RayQueryScanContext>);
static_assert(std::is_standard_layout_v<RayQueryScanContext>);
static_assert(std::is_pointer_interconvertible_base_of_v<
              detail::RayQueryScanContext, RayQueryScanContext>);
static_assert(sizeof(RayQueryScanContext) ==
              sizeof(detail::RayQueryScanContext));
static_assert(alignof(RayQueryScanContext) ==
              alignof(detail::RayQueryScanContext));

thread_local RayQueryScanContext *active_ray_query_scan_context{nullptr};

class ScopedRayQueryScanContext {

private:
    RayQueryScanContext *_previous;

public:
    explicit ScopedRayQueryScanContext(
        RayQueryScanContext &context) noexcept
        : _previous{active_ray_query_scan_context} {
        active_ray_query_scan_context = &context;
    }
    ~ScopedRayQueryScanContext() noexcept {
        active_ray_query_scan_context = _previous;
    }
};

[[nodiscard]] constexpr float ray_query_embree_tnear(float tnear) noexcept {
    static_assert(
        sizeof(float) == sizeof(uint32_t) &&
        std::numeric_limits<float>::is_iec559);
    constexpr auto sign_bit = 0x80000000u;
    constexpr auto magnitude_mask = 0x7fffffffu;
    constexpr auto minimum_normal_bits = 0x00800000u;
    constexpr auto infinity_bits = 0x7f800000u;
    auto bits = std::bit_cast<uint32_t>(tnear);
    auto magnitude = bits & magnitude_mask;
    if (magnitude >= infinity_bits) { return tnear; }
    if (magnitude == 0u) {
        return std::bit_cast<float>(sign_bit | minimum_normal_bits);
    }
    bits = (bits & sign_bit) != 0u ? bits + 1u : bits - 1u;
    if ((bits & magnitude_mask) < minimum_normal_bits) {
        bits = sign_bit | minimum_normal_bits;
    }
    return std::bit_cast<float>(bits);
}

static_assert(
    std::bit_cast<uint32_t>(ray_query_embree_tnear(1.0f)) ==
    std::bit_cast<uint32_t>(1.0f) - 1u);
static_assert(
    std::bit_cast<uint32_t>(ray_query_embree_tnear(0.0f)) ==
    0x80800000u);
static_assert(
    std::bit_cast<uint32_t>(ray_query_embree_tnear(-0.0f)) ==
    0x80800000u);
static_assert(
    std::bit_cast<uint32_t>(ray_query_embree_tnear(
        std::bit_cast<float>(uint32_t{0x7fc12345u}))) ==
    0x7fc12345u);

[[nodiscard]] LUISA_FORCE_INLINE bool ray_query_key_after_cursor(
    const SIMDHostRayQueryState &state,
    float t, uint32_t inst, uint32_t prim,
    const SIMDHostAccelInstanceTable &instances) noexcept {
    if (state.cursor_valid == 0u) { return true; }
    // Embree's round-curve intersectors may report both the front and back
    // surface of one curve primitive after the first hit is rejected by the
    // query filter. Luisa exposes one closest surface candidate per primitive,
    // so a curve primitive already published by proceed() must not reappear on
    // a continuation scan.
    if (inst == state.cursor_inst && prim == state.cursor_prim &&
        instances.data != nullptr && inst < instances.size &&
        instances.data[inst].geometry_kind ==
            static_cast<uint8_t>(SIMDHostAccelGeometryKind::curve)) {
        return false;
    }
    if (t != state.cursor_t) { return t > state.cursor_t; }
    if (inst != state.cursor_inst) { return inst > state.cursor_inst; }
    return prim > state.cursor_prim;
}

[[nodiscard]] bool ray_query_key_before(
    float t, uint32_t inst, uint32_t prim,
    const SIMDHostRayQuerySurfaceHit &candidate) noexcept {
    if (t != candidate.t) { return t < candidate.t; }
    if (inst != candidate.inst) { return inst < candidate.inst; }
    return prim < candidate.prim;
}

[[nodiscard]] bool ray_query_candidate_before(
    const SIMDHostRayQuerySurfaceHit &lhs,
    const SIMDHostRayQuerySurfaceHit &rhs) noexcept {
    return ray_query_key_before(lhs.t, lhs.inst, lhs.prim, rhs);
}

void ray_query_insert_candidate(
    SIMDHostRayQueryState &state,
    RayQueryBatchBuildState &build,
    SIMDHostRayQuerySurfaceHit candidate,
    bool deduplicate_primitive) noexcept {
    constexpr auto capacity =
        simd_host_ray_query_candidate_batch_capacity;
    if (deduplicate_primitive) {
        for (auto i = 0u; i < state.candidate_batch_count; i++) {
            auto &existing = state.candidate_batch[i];
            if (candidate.inst != existing.inst ||
                candidate.prim != existing.prim) {
                continue;
            }
            if (ray_query_candidate_before(candidate, existing)) {
                existing = candidate;
                if (build.heapified) {
                    auto begin = std::begin(state.candidate_batch);
                    std::make_heap(
                        begin, begin + state.candidate_batch_count,
                        ray_query_candidate_before);
                } else {
                    build.ascending = false;
                    build.descending = false;
                }
            }
            return;
        }
    }
    if (state.candidate_batch_count < capacity) {
        if (state.candidate_batch_count != 0u) {
            auto &&previous = state.candidate_batch[state.candidate_batch_count - 1u];
            build.ascending &=
                !ray_query_candidate_before(candidate, previous);
            build.descending &=
                !ray_query_candidate_before(previous, candidate);
        }
        state.candidate_batch[state.candidate_batch_count++] = candidate;
        return;
    }
    state.candidate_batch_has_more = 1u;
    auto begin = std::begin(state.candidate_batch);
    auto end = begin + state.candidate_batch_count;
    if (!build.heapified) {
        std::make_heap(begin, end, ray_query_candidate_before);
        build.heapified = true;
        build.ascending = false;
        build.descending = false;
    }
    if (!ray_query_candidate_before(candidate, state.candidate_batch[0u])) {
        return;
    }
    std::pop_heap(begin, end, ray_query_candidate_before);
    state.candidate_batch[state.candidate_batch_count - 1u] = candidate;
    std::push_heap(begin, end, ray_query_candidate_before);
}

struct RayQueryProceduralBefore {
    [[nodiscard]] LUISA_FORCE_INLINE bool operator()(
        const SIMDHostRayQueryProceduralHit &lhs,
        const SIMDHostRayQueryProceduralHit &rhs) const noexcept {
        if (lhs.inst != rhs.inst) { return lhs.inst < rhs.inst; }
        return lhs.prim < rhs.prim;
    }
};
inline constexpr RayQueryProceduralBefore
    ray_query_procedural_before{};

void ray_query_insert_procedural_candidate(
    SIMDHostRayQueryState &state,
    RayQueryBatchBuildState &build,
    SIMDHostRayQueryProceduralHit candidate) noexcept {
    constexpr auto capacity =
        simd_host_ray_query_candidate_batch_capacity;
    if (state.procedural_batch_count < capacity) {
        if (state.procedural_batch_count != 0u) {
            auto &&previous = state.procedural_batch[state.procedural_batch_count - 1u];
            build.ascending &=
                !ray_query_procedural_before(candidate, previous);
            build.descending &=
                !ray_query_procedural_before(previous, candidate);
        }
        state.procedural_batch[state.procedural_batch_count++] = candidate;
        return;
    }
    state.procedural_batch_has_more = 1u;
    auto begin = std::begin(state.procedural_batch);
    auto end = begin + state.procedural_batch_count;
    if (!build.heapified) {
        std::make_heap(begin, end, ray_query_procedural_before);
        build.heapified = true;
        build.ascending = false;
        build.descending = false;
    }
    if (!ray_query_procedural_before(
            candidate, state.procedural_batch[0u])) {
        return;
    }
    std::pop_heap(begin, end, ray_query_procedural_before);
    state.procedural_batch[state.procedural_batch_count - 1u] = candidate;
    std::push_heap(begin, end, ray_query_procedural_before);
}

void ray_query_filter(
    const RTCFilterFunctionNArguments *arguments) noexcept {
    auto *context = reinterpret_cast<RayQueryScanContext *>(
        arguments->context);
    LUISA_ASSERT(
        context != nullptr && arguments->valid != nullptr &&
            arguments->ray != nullptr && arguments->hit != nullptr &&
            arguments->N != 0u && arguments->N <= 16u &&
            context->instances != nullptr,
        "Invalid SIMD ray-query filter invocation.");
    for (auto packet_lane = 0u; packet_lane < arguments->N;
         packet_lane++) {
        if (arguments->valid[packet_lane] != -1) { continue; }
        // Every candidate is rejected from Embree itself. The closest fixed
        // batch of lexicographically subsequent candidates is retained in
        // lane-private state and exposed to the ordinary SIMD CFG by
        // proceed().
        arguments->valid[packet_lane] = 0;
        auto lane = RTCRayN_id(
            arguments->ray, arguments->N, packet_lane);
        if (lane >= context->lane_count) { continue; }
        auto *state = context->states[lane];
        if (state == nullptr || state->terminated != 0u) { continue; }
        auto t = RTCRayN_tfar(
            arguments->ray, arguments->N, packet_lane);
        auto inst = RTCHitN_instID(
            arguments->hit, arguments->N, packet_lane, 0u);
        auto prim = RTCHitN_primID(
            arguments->hit, arguments->N, packet_lane);
        if (!(t >= state->world_ray[3u] &&
              t <= state->world_ray[7u]) ||
            inst == RTC_INVALID_GEOMETRY_ID ||
            prim == RTC_INVALID_GEOMETRY_ID ||
            !ray_query_key_after_cursor(
                *state, t, inst, prim, *context->instances)) {
            continue;
        }
        auto v = RTCHitN_v(
            arguments->hit, arguments->N, packet_lane);
        auto curve = context->instances->data != nullptr &&
                     inst < context->instances->size &&
                     context->instances->data[inst].geometry_kind ==
                         static_cast<uint8_t>(
                             SIMDHostAccelGeometryKind::curve);
        if (curve) {
            v = -1.0f;
        }
        ray_query_insert_candidate(
            *state, context->batch_build[lane],
            SIMDHostRayQuerySurfaceHit{
                .inst = inst,
                .prim = prim,
                .bary = {
                    RTCHitN_u(
                        arguments->hit, arguments->N, packet_lane),
                    v},
                .t = t,
            },
            curve);
    }
}

[[nodiscard]] bool procedural_key_after_cursor(
    const SIMDHostRayQueryState &state,
    uint32_t inst, uint32_t prim) noexcept {
    if (state.procedural_cursor_valid == 0u) { return true; }
    if (inst != state.procedural_cursor_inst) {
        return inst > state.procedural_cursor_inst;
    }
    return prim > state.procedural_cursor_prim;
}

void collect_procedural_candidates(
    int *valid, RTCRayN *rays, uint32_t packet_width,
    uint32_t primitive, RayQueryRTCContext *rtc_context) noexcept {
    if (valid == nullptr || rays == nullptr || packet_width == 0u ||
        packet_width > 16u) {
        return;
    }
    auto *context = active_ray_query_scan_context;
    auto query_scan = context != nullptr &&
                      rtc_context == &context->rtc &&
                      context->instances != nullptr;
    auto instance = query_scan ? rtc_context->instID[0u] :
                                 RTC_INVALID_GEOMETRY_ID;
    for (auto packet_lane = 0u; packet_lane < packet_width;
         packet_lane++) {
        if (valid[packet_lane] != -1) { continue; }
        // Procedural AABBs are never physical Embree hits. Direct traversal
        // therefore reports a miss, while a query scan records one candidate
        // and executes its DSL handler after returning to the SIMD scheduler.
        valid[packet_lane] = 0;
        if (!query_scan || instance == RTC_INVALID_GEOMETRY_ID) { continue; }
        auto lane = RTCRayN_id(rays, packet_width, packet_lane);
        if (lane >= context->lane_count) { continue; }
        auto *state = context->states[lane];
        if (state == nullptr || state->terminated != 0u ||
            !procedural_key_after_cursor(*state, instance, primitive)) {
            continue;
        }
        ray_query_insert_procedural_candidate(
            *state, context->procedural_batch_build[lane],
            SIMDHostRayQueryProceduralHit{
                .inst = instance,
                .prim = primitive,
            });
    }
}

void initialize_ray_query_context(
    RayQueryScanContext &context, uint32_t lane_count,
    uint64_t active_mask_bits,
    SIMDHostRayQueryState *const *states,
    const SIMDHostAccelInstanceTable &instances) noexcept {
    context.lane_count = lane_count;
    context.instances = &instances;
    for (auto lane = 0u; lane < lane_count; lane++) {
        context.states[lane] = states[lane];
        if (!lane_active(active_mask_bits, lane, lane_count)) { continue; }
        auto *state = states[lane];
        LUISA_ASSERT(
            state != nullptr,
            "SIMD ray-query scan contains a null active state.");
        state->candidate_batch_count = 0u;
        state->candidate_batch_index = 0u;
        state->candidate_batch_has_more = 0u;
        state->candidate_batch_initialized = 0u;
        state->procedural_batch_count = 0u;
        state->procedural_batch_index = 0u;
        state->procedural_batch_has_more = 0u;
        state->procedural_batch_initialized = 0u;
    }
#if LUISA_COMPUTE_SIMD_EMBREE_VERSION == 3
    rtcInitIntersectContext(&context.rtc);
    context.rtc.filter = ray_query_filter;
#else
    rtcInitRayQueryContext(&context.rtc);
#endif
}

void initialize_ray_query_context_wide(
    RayQueryScanContext &context, uint32_t lane_count,
    uint64_t active_mask_bits,
    SIMDHostRayQueryState *const *states,
    const SIMDHostAccelInstanceTable &instances) noexcept {
    context.lane_count = lane_count;
    context.instances = &instances;
    while (active_mask_bits != 0u) {
        auto lane = static_cast<uint32_t>(
            std::countr_zero(active_mask_bits));
        active_mask_bits &= active_mask_bits - 1u;
        context.states[lane] = states[lane];
        auto *state = states[lane];
        LUISA_ASSERT(
            state != nullptr,
            "SIMD ray-query scan contains a null active state.");
        state->candidate_batch_count = 0u;
        state->candidate_batch_index = 0u;
        state->candidate_batch_has_more = 0u;
        state->candidate_batch_initialized = 0u;
        state->procedural_batch_count = 0u;
        state->procedural_batch_index = 0u;
        state->procedural_batch_has_more = 0u;
        state->procedural_batch_initialized = 0u;
    }
#if LUISA_COMPUTE_SIMD_EMBREE_VERSION == 3
    rtcInitIntersectContext(&context.rtc);
    context.rtc.filter = detail::ray_query_filter_wide;
#else
    rtcInitRayQueryContext(&context.rtc);
#endif
}

LUISA_FORCE_INLINE void initialize_ray_query_inputs(
    uint32_t lane_count, uint64_t active_mask_bits,
    SIMDHostRayQueryState *const *states,
    float *components, uint32_t *visibility_masks,
    float *times) noexcept {
    for (auto lane = 0u; lane < lane_count; lane++) {
        if (!lane_active(active_mask_bits, lane, lane_count)) { continue; }
        auto *state = states[lane];
        LUISA_ASSERT(
            state != nullptr,
            "SIMD ray-query packet contains a null active state.");
        for (auto component = 0u; component < 8u; component++) {
            components[component * lane_count + lane] =
                state->world_ray[component];
        }
        components[3u * lane_count + lane] =
            ray_query_embree_tnear(state->world_ray[3u]);
        visibility_masks[lane] = state->visibility_mask;
        times[lane] = state->time;
    }
}

void initialize_ray_query_inputs_wide(
    uint32_t lane_count, uint64_t active_mask_bits,
    SIMDHostRayQueryState *const *states,
    float *components, uint32_t *visibility_masks,
    float *times) noexcept {
    while (active_mask_bits != 0u) {
        auto lane = static_cast<uint32_t>(
            std::countr_zero(active_mask_bits));
        active_mask_bits &= active_mask_bits - 1u;
        auto *state = states[lane];
        LUISA_ASSERT(
            state != nullptr,
            "SIMD ray-query packet contains a null active state.");
        for (auto component = 0u; component < 8u; component++) {
            components[component * lane_count + lane] =
                state->world_ray[component];
        }
        components[3u * lane_count + lane] =
            ray_query_embree_tnear(state->world_ray[3u]);
        visibility_masks[lane] = state->visibility_mask;
        times[lane] = state->time;
    }
}

enum struct RayQueryCandidateAdvance : uint8_t {
    published,
    needs_scan,
    terminated,
};

[[nodiscard]] LUISA_FORCE_INLINE RayQueryCandidateAdvance
advance_ray_query_candidate(
    SIMDHostRayQueryState &state,
    const SIMDHostAccelInstanceTable &instances) noexcept {
    if (state.terminated != 0u) {
        return RayQueryCandidateAdvance::terminated;
    }
    if (state.candidate_batch_initialized == 0u ||
        state.procedural_batch_initialized == 0u) {
        return RayQueryCandidateAdvance::needs_scan;
    }
    constexpr auto capacity =
        simd_host_ray_query_candidate_batch_capacity;
    LUISA_ASSERT(
        state.candidate_batch_count <= capacity &&
            state.candidate_batch_index <= state.candidate_batch_count &&
            state.procedural_batch_count <= capacity &&
            state.procedural_batch_index <= state.procedural_batch_count,
        "SIMD ray-query candidate batch metadata is invalid.");
    while (state.procedural_batch_index <
           state.procedural_batch_count) {
        auto candidate = state.procedural_batch[state.procedural_batch_index++];
        if (!procedural_key_after_cursor(
                state, candidate.inst, candidate.prim)) {
            continue;
        }
        state.procedural_cursor_valid = 1u;
        state.procedural_cursor_inst = candidate.inst;
        state.procedural_cursor_prim = candidate.prim;
        state.candidate = SIMDHostRayQuerySurfaceHit{
            .inst = candidate.inst,
            .prim = candidate.prim,
            .bary = {-1.0f, -1.0f},
            .t = 0.0f,
        };
        state.candidate_committed = 0u;
        state.candidate_kind = static_cast<uint32_t>(
            SIMDHostRayQueryCandidateKind::procedural);
        return RayQueryCandidateAdvance::published;
    }
    while (state.candidate_batch_index < state.candidate_batch_count) {
        auto candidate = state.candidate_batch[state.candidate_batch_index++];
        if (!ray_query_key_after_cursor(
                state, candidate.t, candidate.inst, candidate.prim,
                instances)) {
            continue;
        }
        if (candidate.t < state.world_ray[3u]) { continue; }
        if (candidate.t > state.world_ray[7u]) {
            state.candidate_batch_index = state.candidate_batch_count;
            state.candidate_batch_has_more = 0u;
            break;
        }
        state.cursor_valid = 1u;
        state.cursor_inst = candidate.inst;
        state.cursor_prim = candidate.prim;
        state.cursor_t = candidate.t;
        state.candidate = candidate;
        state.candidate_committed = 0u;
        LUISA_ASSERT(
            instances.data != nullptr &&
                candidate.inst < instances.size,
            "SIMD ray query returned an invalid instance ID {}.",
            candidate.inst);
        if (instances.data[candidate.inst].opaque != 0u) {
            state.committed = SIMDHostRayQueryCommittedHit{
                .inst = candidate.inst,
                .prim = candidate.prim,
                .bary = {candidate.bary[0u], candidate.bary[1u]},
                .kind = static_cast<uint32_t>(
                    SIMDHostRayQueryCandidateKind::surface),
                .t = candidate.t,
            };
            state.world_ray[7u] = candidate.t;
            state.candidate_kind = static_cast<uint32_t>(
                SIMDHostRayQueryCandidateKind::none);
            state.terminated = 1u;
            return RayQueryCandidateAdvance::terminated;
        }
        state.candidate_kind = static_cast<uint32_t>(
            SIMDHostRayQueryCandidateKind::surface);
        return RayQueryCandidateAdvance::published;
    }
    if (state.candidate_batch_has_more != 0u ||
        state.procedural_batch_has_more != 0u) {
        return RayQueryCandidateAdvance::needs_scan;
    }
    state.candidate_kind = static_cast<uint32_t>(
        SIMDHostRayQueryCandidateKind::none);
    state.terminated = 1u;
    return RayQueryCandidateAdvance::terminated;
}

void install_ray_query_candidate_batches(
    RayQueryScanContext &context, uint64_t active_mask_bits,
    const SIMDHostAccelInstanceTable &instances) noexcept {
    for (auto lane = 0u; lane < context.lane_count; lane++) {
        if (!lane_active(
                active_mask_bits, lane, context.lane_count)) {
            continue;
        }
        auto *state = context.states[lane];
        auto &build = context.batch_build[lane];
        auto begin = std::begin(state->candidate_batch);
        auto end = begin + state->candidate_batch_count;
        if (build.heapified) {
            std::sort_heap(begin, end, ray_query_candidate_before);
        } else if (build.descending && !build.ascending) {
            std::reverse(begin, end);
        } else if (!build.ascending) {
            std::sort(begin, end, ray_query_candidate_before);
        }
        auto &procedural_build =
            context.procedural_batch_build[lane];
        auto procedural_begin = std::begin(state->procedural_batch);
        auto procedural_end =
            procedural_begin + state->procedural_batch_count;
        if (procedural_build.heapified) {
            std::sort_heap(
                procedural_begin, procedural_end,
                ray_query_procedural_before);
        } else if (procedural_build.descending &&
                   !procedural_build.ascending) {
            std::reverse(procedural_begin, procedural_end);
        } else if (!procedural_build.ascending) {
            std::sort(
                procedural_begin, procedural_end,
                ray_query_procedural_before);
        }
        state->candidate_batch_index = 0u;
        state->candidate_batch_initialized = 1u;
        state->procedural_batch_index = 0u;
        state->procedural_batch_initialized = 1u;
        auto advanced = advance_ray_query_candidate(*state, instances);
        LUISA_ASSERT(
            advanced != RayQueryCandidateAdvance::needs_scan,
            "A newly scanned SIMD ray-query batch made no progress.");
    }
}

void install_ray_query_candidate_batches_wide(
    RayQueryScanContext &context, uint64_t active_mask_bits,
    const SIMDHostAccelInstanceTable &instances) noexcept {
    while (active_mask_bits != 0u) {
        auto lane = static_cast<uint32_t>(
            std::countr_zero(active_mask_bits));
        active_mask_bits &= active_mask_bits - 1u;
        auto *state = context.states[lane];
        auto &build = context.batch_build[lane];
        auto begin = std::begin(state->candidate_batch);
        auto end = begin + state->candidate_batch_count;
        if (build.heapified) {
            std::sort_heap(begin, end, ray_query_candidate_before);
        } else if (build.descending && !build.ascending) {
            std::reverse(begin, end);
        } else if (!build.ascending) {
            std::sort(begin, end, ray_query_candidate_before);
        }
        auto &procedural_build =
            context.procedural_batch_build[lane];
        auto procedural_begin = std::begin(state->procedural_batch);
        auto procedural_end =
            procedural_begin + state->procedural_batch_count;
        if (procedural_build.heapified) {
            std::sort_heap(
                procedural_begin, procedural_end,
                ray_query_procedural_before);
        } else if (procedural_build.descending &&
                   !procedural_build.ascending) {
            std::reverse(procedural_begin, procedural_end);
        } else if (!procedural_build.ascending) {
            std::sort(
                procedural_begin, procedural_end,
                ray_query_procedural_before);
        }
        state->candidate_batch_index = 0u;
        state->candidate_batch_initialized = 1u;
        state->procedural_batch_index = 0u;
        state->procedural_batch_initialized = 1u;
        auto advanced = advance_ray_query_candidate(*state, instances);
        LUISA_ASSERT(
            advanced != RayQueryCandidateAdvance::needs_scan,
            "A newly scanned SIMD ray-query batch made no progress.");
    }
}

template<size_t packet_width, typename RayHitPacket, typename RayPacket>
void scan_ray_query_packet(
    RTCScene scene, const SIMDHostAccelInstanceTable &instances,
    uint32_t lane_count, uint64_t active_mask_bits,
    SIMDHostRayQueryState *const *states,
    bool terminate_on_first) noexcept {
    alignas(64) std::array<float, 8u * 16u> components{};
    alignas(64) std::array<uint32_t, 16u> visibility_masks{};
    alignas(64) std::array<float, 16u> times{};
    initialize_ray_query_inputs(
        lane_count, active_mask_bits, states,
        components.data(), visibility_masks.data(), times.data());
    RayQueryScanContext context{};
    initialize_ray_query_context(
        context, lane_count, active_mask_bits, states, instances);
    alignas(64) std::array<int, packet_width> valid{};
    ScopedRayQueryScanContext active_context{context};
    if (terminate_on_first) {
        alignas(64) RayPacket packet{};
        initialize_ray_packet<packet_width>(
            packet, valid, lane_count, active_mask_bits,
            components.data(), visibility_masks.data(), times.data());
        for (auto lane = 0u; lane < packet_width; lane++) {
            packet.id[lane] = lane;
        }
#if LUISA_COMPUTE_SIMD_EMBREE_VERSION == 3
        if constexpr (packet_width == 4u) {
            rtcOccluded4(valid.data(), scene, &context.rtc, &packet);
        } else if constexpr (packet_width == 8u) {
            rtcOccluded8(valid.data(), scene, &context.rtc, &packet);
        } else {
            rtcOccluded16(valid.data(), scene, &context.rtc, &packet);
        }
#else
        RTCOccludedArguments arguments{};
        rtcInitOccludedArguments(&arguments);
        arguments.context = &context.rtc;
        arguments.flags = static_cast<RTCRayQueryFlags>(
            arguments.flags |
            RTC_RAY_QUERY_FLAG_INVOKE_ARGUMENT_FILTER);
        arguments.filter = ray_query_filter;
        if constexpr (packet_width == 4u) {
            rtcOccluded4(valid.data(), scene, &packet, &arguments);
        } else if constexpr (packet_width == 8u) {
            rtcOccluded8(valid.data(), scene, &packet, &arguments);
        } else {
            rtcOccluded16(valid.data(), scene, &packet, &arguments);
        }
#endif
    } else {
        alignas(64) RayHitPacket packet{};
        initialize_ray_packet<packet_width>(
            packet.ray, valid, lane_count, active_mask_bits,
            components.data(), visibility_masks.data(), times.data());
        initialize_hit_packet<packet_width>(packet.hit);
        for (auto lane = 0u; lane < packet_width; lane++) {
            packet.ray.id[lane] = lane;
        }
#if LUISA_COMPUTE_SIMD_EMBREE_VERSION == 3
        if constexpr (packet_width == 4u) {
            rtcIntersect4(valid.data(), scene, &context.rtc, &packet);
        } else if constexpr (packet_width == 8u) {
            rtcIntersect8(valid.data(), scene, &context.rtc, &packet);
        } else {
            rtcIntersect16(valid.data(), scene, &context.rtc, &packet);
        }
#else
        RTCIntersectArguments arguments{};
        rtcInitIntersectArguments(&arguments);
        arguments.context = &context.rtc;
        arguments.flags = static_cast<RTCRayQueryFlags>(
            arguments.flags |
            RTC_RAY_QUERY_FLAG_INVOKE_ARGUMENT_FILTER);
        arguments.filter = ray_query_filter;
        if constexpr (packet_width == 4u) {
            rtcIntersect4(valid.data(), scene, &packet, &arguments);
        } else if constexpr (packet_width == 8u) {
            rtcIntersect8(valid.data(), scene, &packet, &arguments);
        } else {
            rtcIntersect16(valid.data(), scene, &packet, &arguments);
        }
#endif
    }
    install_ray_query_candidate_batches(
        context, active_mask_bits, instances);
}

template<size_t packet_width, typename RayHitPacket, typename RayPacket>
void scan_ray_query_packet_wide(
    RTCScene scene, const SIMDHostAccelInstanceTable &instances,
    uint32_t lane_count, uint64_t active_mask_bits,
    SIMDHostRayQueryState *const *states,
    bool terminate_on_first) noexcept {
    static_assert(packet_width == 8u || packet_width == 16u);
    alignas(64) std::array<float, 8u * 16u> components{};
    alignas(64) std::array<uint32_t, 16u> visibility_masks{};
    alignas(64) std::array<float, 16u> times{};
    auto fully_active = packet_fully_active<packet_width>(
        lane_count, active_mask_bits);
    if (fully_active) {
        initialize_ray_query_inputs(
            lane_count, active_mask_bits, states,
            components.data(), visibility_masks.data(), times.data());
    } else {
        initialize_ray_query_inputs_wide(
            lane_count, active_mask_bits, states,
            components.data(), visibility_masks.data(), times.data());
    }
    RayQueryScanContext context{};
    if (fully_active) {
        initialize_ray_query_context(
            context, lane_count, active_mask_bits, states, instances);
    } else {
        initialize_ray_query_context_wide(
            context, lane_count, active_mask_bits, states, instances);
    }
#if LUISA_COMPUTE_SIMD_EMBREE_VERSION == 3
    // Embree 3 stores the filter on the intersect context rather than the
    // per-call arguments used by Embree 4. Keep the W8/W16 callback contract
    // identical across both APIs, including for an initially full cohort:
    // traversal may still present a sparse valid mask to the filter.
    context.rtc.filter = detail::ray_query_filter_wide;
#endif
    alignas(64) std::array<int, packet_width> valid{};
    ScopedRayQueryScanContext active_context{context};
    if (terminate_on_first) {
        alignas(64) RayPacket packet{};
        initialize_ray_packet<packet_width>(
            packet, valid, lane_count, active_mask_bits,
            components.data(), visibility_masks.data(), times.data());
        for (auto lane = 0u; lane < packet_width; lane++) {
            packet.id[lane] = lane;
        }
#if LUISA_COMPUTE_SIMD_EMBREE_VERSION == 3
        if constexpr (packet_width == 8u) {
            rtcOccluded8(valid.data(), scene, &context.rtc, &packet);
        } else {
            rtcOccluded16(valid.data(), scene, &context.rtc, &packet);
        }
#else
        RTCOccludedArguments arguments{};
        rtcInitOccludedArguments(&arguments);
        arguments.context = &context.rtc;
        arguments.flags = static_cast<RTCRayQueryFlags>(
            arguments.flags |
            RTC_RAY_QUERY_FLAG_INVOKE_ARGUMENT_FILTER);
        arguments.filter = detail::ray_query_filter_wide;
        if constexpr (packet_width == 8u) {
            rtcOccluded8(valid.data(), scene, &packet, &arguments);
        } else {
            rtcOccluded16(valid.data(), scene, &packet, &arguments);
        }
#endif
    } else {
        alignas(64) RayHitPacket packet{};
        initialize_ray_packet<packet_width>(
            packet.ray, valid, lane_count, active_mask_bits,
            components.data(), visibility_masks.data(), times.data());
        initialize_hit_packet<packet_width>(packet.hit);
        for (auto lane = 0u; lane < packet_width; lane++) {
            packet.ray.id[lane] = lane;
        }
#if LUISA_COMPUTE_SIMD_EMBREE_VERSION == 3
        if constexpr (packet_width == 8u) {
            rtcIntersect8(valid.data(), scene, &context.rtc, &packet);
        } else {
            rtcIntersect16(valid.data(), scene, &context.rtc, &packet);
        }
#else
        RTCIntersectArguments arguments{};
        rtcInitIntersectArguments(&arguments);
        arguments.context = &context.rtc;
        arguments.flags = static_cast<RTCRayQueryFlags>(
            arguments.flags |
            RTC_RAY_QUERY_FLAG_INVOKE_ARGUMENT_FILTER);
        arguments.filter = detail::ray_query_filter_wide;
        if constexpr (packet_width == 8u) {
            rtcIntersect8(valid.data(), scene, &packet, &arguments);
        } else {
            rtcIntersect16(valid.data(), scene, &packet, &arguments);
        }
#endif
    }
    if (fully_active) {
        install_ray_query_candidate_batches(
            context, active_mask_bits, instances);
    } else {
        install_ray_query_candidate_batches_wide(
            context, active_mask_bits, instances);
    }
}

void scan_ray_query_scalar(
    RTCScene scene, const SIMDHostAccelInstanceTable &instances,
    uint64_t active_mask_bits,
    SIMDHostRayQueryState *const *states,
    bool terminate_on_first) noexcept {
    alignas(64) std::array<float, 8u> components{};
    alignas(64) std::array<uint32_t, 1u> visibility_masks{};
    alignas(64) std::array<float, 1u> times{};
    initialize_ray_query_inputs(
        1u, active_mask_bits, states,
        components.data(), visibility_masks.data(), times.data());
    RayQueryScanContext context{};
    initialize_ray_query_context(
        context, 1u, active_mask_bits, states, instances);
    ScopedRayQueryScanContext active_context{context};
    if (terminate_on_first) {
        RTCRay ray{};
        initialize_scalar_ray(
            ray, 1u, active_mask_bits, components.data(),
            visibility_masks.data(), times.data());
#if LUISA_COMPUTE_SIMD_EMBREE_VERSION == 3
        rtcOccluded1(scene, &context.rtc, &ray);
#else
        RTCOccludedArguments arguments{};
        rtcInitOccludedArguments(&arguments);
        arguments.context = &context.rtc;
        arguments.flags = static_cast<RTCRayQueryFlags>(
            arguments.flags |
            RTC_RAY_QUERY_FLAG_INVOKE_ARGUMENT_FILTER);
        arguments.filter = ray_query_filter;
        rtcOccluded1(scene, &ray, &arguments);
#endif
    } else {
        RTCRayHit ray_hit{};
        initialize_scalar_ray(
            ray_hit.ray, 1u, active_mask_bits,
            components.data(), visibility_masks.data(), times.data());
        initialize_scalar_hit(ray_hit.hit);
#if LUISA_COMPUTE_SIMD_EMBREE_VERSION == 3
        rtcIntersect1(scene, &context.rtc, &ray_hit);
#else
        RTCIntersectArguments arguments{};
        rtcInitIntersectArguments(&arguments);
        arguments.context = &context.rtc;
        arguments.flags = static_cast<RTCRayQueryFlags>(
            arguments.flags |
            RTC_RAY_QUERY_FLAG_INVOKE_ARGUMENT_FILTER);
        arguments.filter = ray_query_filter;
        rtcIntersect1(scene, &ray_hit, &arguments);
#endif
    }
    install_ray_query_candidate_batches(
        context, active_mask_bits, instances);
}

}// namespace

void simd_procedural_intersect(
    const RTCIntersectFunctionNArguments *arguments) noexcept {
    if (arguments == nullptr || arguments->rayhit == nullptr) { return; }
    auto *rays = RTCRayHitN_RayN(
        arguments->rayhit, arguments->N);
    collect_procedural_candidates(
        arguments->valid, rays, arguments->N,
        arguments->primID, arguments->context);
}

void simd_procedural_occluded(
    const RTCOccludedFunctionNArguments *arguments) noexcept {
    if (arguments == nullptr || arguments->ray == nullptr) { return; }
    collect_procedural_candidates(
        arguments->valid, arguments->ray, arguments->N,
        arguments->primID, arguments->context);
}

SIMDAccel::SIMDAccel(
    RTCDevice device, const AccelOption &option,
    uint32_t warp_width) noexcept
    : _scene{rtcNewScene(device)},
      _warp_width{warp_width},
      _enable_triangle_only_ray_query{
          triangle_ray_query::triangle_only_ray_query_enabled()},
      _enable_procedural_dense_status{
          !luisa::compute::detail::env_flag(
              "LUISA_SIMD_DISABLE_PROCEDURAL_WIDE_STATUS_PACK")} {
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
        _motion_states.resize(instance_count);
    } else {
        auto device = rtcGetSceneDevice(_scene);
        _instances.reserve(instance_count);
        _geometries.reserve(instance_count);
        _motion_states.reserve(instance_count);
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
            _motion_states.emplace_back();
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
            auto *primitive = reinterpret_cast<SIMDPrimitive *>(
                modification.primitive);
            LUISA_ASSERT(
                primitive != nullptr,
                "SIMD accel instance has a null primitive.");
            if (primitive->kind() == SIMDPrimitive::Kind::motion_instance) {
                auto *motion = static_cast<SIMDMotionInstance *>(primitive);
                LUISA_ASSERT(
                    motion->child() != nullptr &&
                        motion->keyframes().size() ==
                            motion->option().keyframe_count,
                    "SIMD motion instance must be built before its accel.");
                auto state = luisa::make_unique<MotionState>();
                state->option = motion->option();
                state->keyframes.assign(
                    motion->keyframes().begin(), motion->keyframes().end());
                instance.motion_frames = state->keyframes.data();
                instance.motion_keyframe_count =
                    static_cast<uint32_t>(state->keyframes.size());
                instance.motion_mode =
                    static_cast<uint32_t>(state->option.mode);
                rtcSetGeometryInstancedScene(
                    geometry, motion->child()->handle());
                rtcSetGeometryTimeStepCount(
                    geometry, state->option.keyframe_count);
                rtcSetGeometryTimeRange(
                    geometry, state->option.time_start,
                    state->option.time_end);
                instance.geometry_kind = static_cast<uint8_t>(
                    motion->child()->kind() == SIMDPrimitive::Kind::curve ?
                        SIMDHostAccelGeometryKind::curve :
                    motion->child()->kind() ==
                            SIMDPrimitive::Kind::procedural ?
                        SIMDHostAccelGeometryKind::procedural :
                        SIMDHostAccelGeometryKind::triangle);
                _motion_states[modification.index] = std::move(state);
            } else {
                LUISA_ASSERT(
                    primitive->kind() == SIMDPrimitive::Kind::mesh ||
                        primitive->kind() == SIMDPrimitive::Kind::curve ||
                        primitive->kind() == SIMDPrimitive::Kind::procedural,
                    "SIMD accel instances require a mesh, curve, procedural, "
                    "or motion-instance primitive.");
                rtcSetGeometryInstancedScene(geometry, primitive->handle());
                rtcSetGeometryTimeStepCount(geometry, 1u);
                rtcSetGeometryTimeRange(geometry, 0.0f, 1.0f);
                instance.geometry_kind = static_cast<uint8_t>(
                    primitive->kind() == SIMDPrimitive::Kind::curve ?
                        SIMDHostAccelGeometryKind::curve :
                    primitive->kind() == SIMDPrimitive::Kind::procedural ?
                        SIMDHostAccelGeometryKind::procedural :
                        SIMDHostAccelGeometryKind::triangle);
                instance.motion_frames = nullptr;
                instance.motion_keyframe_count = 0u;
                instance.motion_mode = 0u;
                _motion_states[modification.index].reset();
            }
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
        auto geometry = _geometries[i];
        if (instance.dirty != 0u) {
            if (auto &motion = _motion_states[i]; motion != nullptr) {
                LUISA_ASSERT(
                    instance.motion_frames == motion->keyframes.data() &&
                        instance.motion_keyframe_count ==
                            motion->keyframes.size() &&
                        instance.motion_mode ==
                            static_cast<uint32_t>(motion->option.mode),
                    "SIMD motion-instance metadata is inconsistent.");
                if (motion->option.mode == AccelMotionMode::MATRIX) {
                    std::array<float, 12u> composed{};
                    for (auto key = size_t{0u};
                         key < motion->keyframes.size(); key++) {
                        compose_matrix_keyframe(
                            composed.data(), instance.affine,
                            motion->keyframes[key].as_matrix(), key);
                        rtcSetGeometryTransform(
                            geometry, static_cast<unsigned>(key),
                            RTC_FORMAT_FLOAT3X4_ROW_MAJOR,
                            composed.data());
                    }
                } else {
                    LUISA_ASSERT(
                        affine_is_identity(instance.affine),
                        "SIMD SRT motion instances currently require an "
                        "identity outer affine transform.");
                    for (auto key = size_t{0u};
                         key < motion->keyframes.size(); key++) {
                        auto quaternion = quaternion_keyframe(
                            motion->keyframes[key].as_srt(), key);
                        rtcSetGeometryTransformQuaternion(
                            geometry, static_cast<unsigned>(key),
                            &quaternion);
                    }
                }
            } else {
                rtcSetGeometryTransform(
                    geometry, 0u, RTC_FORMAT_FLOAT3X4_ROW_MAJOR,
                    instance.affine);
            }
            rtcSetGeometryMask(geometry, instance.mask);
        }
        // A BLAS can be rebuilt without changing the Accel instance record.
        // Recommitting every instance geometry refreshes Embree's cached child
        // scene bounds before the TLAS commit; transform/mask setup remains
        // dirty-only.
        rtcCommitGeometry(geometry);
        instance.dirty = 0u;
    }
    _has_curve_instances = std::any_of(
        _instances.cbegin(), _instances.cend(),
        [](const Instance &instance) noexcept {
            return instance.geometry_kind ==
                   static_cast<uint8_t>(
                       SIMDHostAccelGeometryKind::curve);
        });
    _has_procedural_instances = std::any_of(
        _instances.cbegin(), _instances.cend(),
        [](const Instance &instance) noexcept {
            return instance.geometry_kind ==
                   static_cast<uint8_t>(
                       SIMDHostAccelGeometryKind::procedural);
        });
    _instance_table.ray_query_proceed_status =
        simd_host_ray_query_proceed_status;
    _instance_table.ray_query_proceed_wide_status =
        simd_host_ray_query_use_procedural_wide_status(
            _warp_width,
            _has_procedural_instances,
            _enable_procedural_dense_status) ?
            simd_host_ray_query_proceed_wide_procedural_status :
            simd_host_ray_query_proceed_status;
    rtcCommitScene(_scene);
}

void SIMDAccel::_trace_closest(
    void *accel, uint32_t lane_count,
    void *ray_hit_packet) noexcept {
    auto *self = static_cast<SIMDAccel *>(accel);
    LUISA_ASSERT(
        self != nullptr && ray_hit_packet != nullptr &&
            (lane_count == 1u || lane_count == 2u ||
             lane_count == 4u || lane_count == 8u ||
             lane_count == 16u),
        "Invalid SIMD closest-hit packet arguments.");
    auto *valid = reinterpret_cast<const int *>(
        static_cast<uint32_t *>(ray_hit_packet) +
        simd_host_accel_ray_id_field * lane_count);
    switch (lane_count) {
        case 1u: {
            if (valid[0u] != 0) {
                intersect_scalar(
                    self->_scene,
                    *static_cast<RTCRayHit *>(ray_hit_packet));
            }
            break;
        }
        case 2u:
            trace_w2_padded<RTCRayHit4>(
                self->_scene, valid, ray_hit_packet,
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
        case 4u:
            trace_packet_in_place<RTCRayHit4>(
                self->_scene, valid, ray_hit_packet,
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
            trace_packet_in_place<RTCRayHit8>(
                self->_scene, valid, ray_hit_packet,
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
            trace_packet_in_place<RTCRayHit16>(
                self->_scene, valid, ray_hit_packet,
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
    if (self->_has_curve_instances) {
        auto *packet_words =
            static_cast<uint32_t *>(ray_hit_packet);
        mark_curve_surface_hits(
            self->_instance_table, lane_count, valid,
            packet_words +
                simd_host_accel_hit_inst_field * lane_count,
            reinterpret_cast<float *>(
                packet_words +
                simd_host_accel_hit_v_field * lane_count));
    }
}

void SIMDAccel::_trace_any(
    void *accel, uint32_t lane_count,
    void *ray_packet) noexcept {
    auto *self = static_cast<SIMDAccel *>(accel);
    LUISA_ASSERT(
        self != nullptr && ray_packet != nullptr &&
            (lane_count == 1u || lane_count == 2u ||
             lane_count == 4u || lane_count == 8u ||
             lane_count == 16u),
        "Invalid SIMD any-hit packet arguments.");
    auto *valid = reinterpret_cast<const int *>(
        static_cast<uint32_t *>(ray_packet) +
        simd_host_accel_ray_id_field * lane_count);
    switch (lane_count) {
        case 1u: {
            if (valid[0u] != 0) {
                occlude_scalar(
                    self->_scene,
                    *static_cast<RTCRay *>(ray_packet));
            }
            break;
        }
        case 2u:
            trace_w2_padded<RTCRay4>(
                self->_scene, valid, ray_packet,
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
        case 4u:
            trace_packet_in_place<RTCRay4>(
                self->_scene, valid, ray_packet,
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
            trace_packet_in_place<RTCRay8>(
                self->_scene, valid, ray_packet,
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
            trace_packet_in_place<RTCRay16>(
                self->_scene, valid, ray_packet,
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

void SIMDAccel::_ray_query_proceed(
    uint32_t lane_count, uint64_t active_mask_bits,
    SIMDHostRayQueryState *const *states) noexcept {
    LUISA_ASSERT(
        states != nullptr &&
            (lane_count == 1u || lane_count == 2u ||
             lane_count == 4u || lane_count == 8u ||
             lane_count == 16u),
        "Invalid SIMD ray-query packet width {}.", lane_count);
    auto lane_mask = lane_count == 64u ? ~uint64_t{0u} :
                                         (uint64_t{1u} << lane_count) - 1u;
    active_mask_bits &= lane_mask;
    if (active_mask_bits == 0u) { return; }

    auto pending = uint64_t{0u};
    for (auto lane = 0u; lane < lane_count; lane++) {
        auto bit = uint64_t{1u} << lane;
        if ((active_mask_bits & bit) == 0u) { continue; }
        auto *state = states[lane];
        LUISA_ASSERT(
            state != nullptr && state->accel != nullptr &&
                state->proceed == _ray_query_proceed,
            "Invalid active SIMD ray-query state in lane {}.", lane);
        if (state->candidate_committed != 0u) {
            auto kind = static_cast<SIMDHostRayQueryCandidateKind>(
                state->candidate_kind);
            LUISA_ASSERT(
                kind == SIMDHostRayQueryCandidateKind::surface ||
                    kind == SIMDHostRayQueryCandidateKind::procedural,
                "SIMD ray query committed an invalid candidate kind.");
            state->committed = SIMDHostRayQueryCommittedHit{
                .inst = state->candidate.inst,
                .prim = state->candidate.prim,
                .bary = {
                    state->candidate.bary[0u],
                    state->candidate.bary[1u]},
                .kind = static_cast<uint32_t>(kind),
                .t = state->candidate.t,
            };
            // The procedural batch was collected against the pre-commit ray
            // interval. A successful surface or procedural commit may shrink
            // tmax, so discard every unexposed speculative AABB candidate and
            // request a continuation scan after cached exact surface hits
            // only when such unexposed candidates may still exist.
            auto procedural_candidates_remain =
                state->procedural_batch_index <
                    state->procedural_batch_count ||
                state->procedural_batch_has_more != 0u;
            state->procedural_batch_count = 0u;
            state->procedural_batch_index = 0u;
            state->procedural_batch_has_more =
                procedural_candidates_remain ? 1u : 0u;
            state->procedural_batch_initialized = 1u;
            state->candidate_committed = 0u;
            if (state->terminate_on_first != 0u) {
                state->terminated = 1u;
            }
        }
        if (state->terminated == 0u) {
            auto *self = static_cast<SIMDAccel *>(state->accel);
            auto advanced = advance_ray_query_candidate(
                *state, self->_instance_table);
            if (advanced == RayQueryCandidateAdvance::needs_scan) {
                pending |= bit;
            }
        }
    }

    while (pending != 0u) {
        auto first_lane = 0u;
        while (((pending >> first_lane) & 1u) == 0u) {
            first_lane++;
        }
        auto *first_state = states[first_lane];
        auto *self = static_cast<SIMDAccel *>(first_state->accel);
        auto terminate_on_first =
            first_state->terminate_on_first != 0u;
        auto group = uint64_t{0u};
        for (auto lane = 0u; lane < lane_count; lane++) {
            auto bit = uint64_t{1u} << lane;
            if ((pending & bit) == 0u) { continue; }
            auto *state = states[lane];
            if (state->accel == self &&
                (state->terminate_on_first != 0u) ==
                    terminate_on_first) {
                group |= bit;
            }
        }
        LUISA_ASSERT(group != 0u, "Empty SIMD ray-query packet group.");
        switch (lane_count) {
            case 1u:
                scan_ray_query_scalar(
                    self->_scene, self->_instance_table,
                    group, states, terminate_on_first);
                break;
            case 2u:
            case 4u:
                scan_ray_query_packet<
                    4u, RTCRayHit4, RTCRay4>(
                    self->_scene, self->_instance_table,
                    lane_count, group, states,
                    terminate_on_first);
                break;
            case 8u:
                scan_ray_query_packet<
                    8u, RTCRayHit8, RTCRay8>(
                    self->_scene, self->_instance_table,
                    lane_count, group, states,
                    terminate_on_first);
                break;
            case 16u:
                scan_ray_query_packet<
                    16u, RTCRayHit16, RTCRay16>(
                    self->_scene, self->_instance_table,
                    lane_count, group, states,
                    terminate_on_first);
                break;
            default: break;
        }
        pending &= ~group;
    }
}

LUISA_FORCE_INLINE void SIMDAccel::_ray_query_proceed_wide_lane(
    SIMDHostRayQueryState *state, uint32_t lane,
    uint64_t bit, uint64_t &pending) noexcept {
    LUISA_ASSERT(
        state != nullptr && state->accel != nullptr &&
            state->proceed == _ray_query_proceed_wide,
        "Invalid active wide SIMD ray-query state in lane {}.", lane);
    if (state->candidate_committed != 0u) {
        auto kind = static_cast<SIMDHostRayQueryCandidateKind>(
            state->candidate_kind);
        LUISA_ASSERT(
            kind == SIMDHostRayQueryCandidateKind::surface ||
                kind == SIMDHostRayQueryCandidateKind::procedural,
            "SIMD ray query committed an invalid candidate kind.");
        state->committed = SIMDHostRayQueryCommittedHit{
            .inst = state->candidate.inst,
            .prim = state->candidate.prim,
            .bary = {
                state->candidate.bary[0u],
                state->candidate.bary[1u]},
            .kind = static_cast<uint32_t>(kind),
            .t = state->candidate.t,
        };
        auto procedural_candidates_remain =
            state->procedural_batch_index <
                state->procedural_batch_count ||
            state->procedural_batch_has_more != 0u;
        state->procedural_batch_count = 0u;
        state->procedural_batch_index = 0u;
        state->procedural_batch_has_more =
            procedural_candidates_remain ? 1u : 0u;
        state->procedural_batch_initialized = 1u;
        state->candidate_committed = 0u;
        if (state->terminate_on_first != 0u) {
            state->terminated = 1u;
        }
    }
    if (state->terminated == 0u) {
        auto *self = static_cast<SIMDAccel *>(state->accel);
        auto advanced = advance_ray_query_candidate(
            *state, self->_instance_table);
        if (advanced == RayQueryCandidateAdvance::needs_scan) {
            pending |= bit;
        }
    }
}

void SIMDAccel::_ray_query_proceed_wide(
    uint32_t lane_count, uint64_t active_mask_bits,
    SIMDHostRayQueryState *const *states) noexcept {
    LUISA_ASSERT(
        states != nullptr &&
            (lane_count == 8u || lane_count == 16u),
        "Invalid wide SIMD ray-query packet width {}.", lane_count);
    auto lane_mask = (uint64_t{1u} << lane_count) - 1u;
    active_mask_bits &= lane_mask;
    if (active_mask_bits == 0u) { return; }
    auto fully_active = active_mask_bits == lane_mask;

    auto pending = uint64_t{0u};
    if (fully_active) [[likely]] {
        for (auto lane = 0u; lane < lane_count; lane++) {
            _ray_query_proceed_wide_lane(
                states[lane], lane,
                uint64_t{1u} << lane, pending);
        }
    } else {
        auto remaining_active = active_mask_bits;
        while (remaining_active != 0u) {
            auto lane = static_cast<uint32_t>(
                std::countr_zero(remaining_active));
            auto bit = uint64_t{1u} << lane;
            remaining_active &= remaining_active - 1u;
            _ray_query_proceed_wide_lane(
                states[lane], lane, bit, pending);
        }
    }

    while (pending != 0u) {
        auto first_lane = uint32_t{0u};
        if (fully_active) [[likely]] {
            while (((pending >> first_lane) & 1u) == 0u) {
                first_lane++;
            }
        } else {
            first_lane = static_cast<uint32_t>(
                std::countr_zero(pending));
        }
        auto *first_state = states[first_lane];
        auto *self = static_cast<SIMDAccel *>(first_state->accel);
        auto terminate_on_first =
            first_state->terminate_on_first != 0u;
        auto group = uint64_t{0u};
        if (fully_active) [[likely]] {
            for (auto lane = 0u; lane < lane_count; lane++) {
                auto bit = uint64_t{1u} << lane;
                if ((pending & bit) == 0u) { continue; }
                auto *state = states[lane];
                if (state->accel == self &&
                    (state->terminate_on_first != 0u) ==
                        terminate_on_first) {
                    group |= bit;
                }
            }
        } else {
            auto remaining_pending = pending;
            while (remaining_pending != 0u) {
                auto lane = static_cast<uint32_t>(
                    std::countr_zero(remaining_pending));
                auto bit = uint64_t{1u} << lane;
                remaining_pending &= remaining_pending - 1u;
                auto *state = states[lane];
                if (state->accel == self &&
                    (state->terminate_on_first != 0u) ==
                        terminate_on_first) {
                    group |= bit;
                }
            }
        }
        LUISA_ASSERT(group != 0u, "Empty wide SIMD ray-query packet group.");
        if (lane_count == 8u) {
            scan_ray_query_packet_wide<
                8u, RTCRayHit8, RTCRay8>(
                self->_scene, self->_instance_table,
                lane_count, group, states,
                terminate_on_first);
        } else {
            scan_ray_query_packet_wide<
                16u, RTCRayHit16, RTCRay16>(
                self->_scene, self->_instance_table,
                lane_count, group, states,
                terminate_on_first);
        }
        pending &= ~group;
    }
}

}// namespace luisa::compute::simd
