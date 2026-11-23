#include "simd_accel.h"
#include "simd_accel_motion.h"
#include "simd_accel_ray_query.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstring>
#include <limits>
#include <optional>
#include <type_traits>

#include <luisa/core/logging.h>
#include <luisa/core/stl/algorithm.h>
#include <luisa/core/stl/memory.h>
#include <luisa/core/stl/unordered_map.h>

#include "../../common/env_flag.h"
#include "simd_mesh.h"
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
inline constexpr auto embree_ray_packet_field_count =
    simd_host_accel_ray_packet_field_count;

static_assert(sizeof(int) == sizeof(uint32_t));
static_assert(luisa::bit_cast<uint32_t>(-1) == ~uint32_t{0u});
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

[[nodiscard]] bool direct_trace_w16_packet_is_full(
    const int *valid) noexcept {
    return std::all_of(
        valid, valid + 16u,
        [](int lane) noexcept { return lane != 0; });
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
            instances.committed_instances != nullptr &&
                inst < instances.committed_size,
            "SIMD curve trace returned an invalid instance ID {}.", inst);
        if (instances.committed_instances[inst].geometry_kind ==
            static_cast<uint8_t>(SIMDHostAccelGeometryKind::curve)) {
            hit_v[lane] = -1.0f;
        }
    }
}

using RayQueryBatchBuildState = detail::RayQueryBatchBuildState;
using RayQueryRTCContext = detail::RayQueryRTCContext;

// Use the callback ABI type directly. The previous empty derived wrapper
// needed std::is_pointer_interconvertible_base_of_v to prove that Embree's
// pointer to the leading RTC context could be recovered as either type, but
// that C++20 library trait is not provided by Apple libc++. An alias makes the
// identity stronger and keeps the existing standard-layout/offset proof in
// simd_accel_ray_query.h valid on every standard library.
using RayQueryScanContext = detail::RayQueryScanContext;
static_assert(std::is_same_v<RayQueryScanContext,
                             detail::RayQueryScanContext>);
static_assert(std::is_standard_layout_v<RayQueryScanContext>);

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

struct LiveRayQueryContext {
    RayQueryRTCContext rtc{};
    const SIMDHostAccelInstanceTable *instances{nullptr};
    const uint8_t *instance_flags{nullptr};
    size_t instance_flags_size{0u};
    SIMDHostRayQueryState *const *states{nullptr};
    SIMDHostRayQueryState *state{nullptr};
    const void *capture{nullptr};
    const SIMDPacketLaunchConfig *launch_config{nullptr};
    SIMDHostRayQueryLiveCandidateHandler *on_candidate{nullptr};
    SIMDHostRayQueryW1LiveCandidateHandler *w1_on_candidate{nullptr};
    uint32_t lane_count{0u};
    uint32_t lane{0u};
    uint32_t excl_inst0{~uint32_t{0u}};
    uint32_t excl_prim0{~uint32_t{0u}};
    uint32_t excl_inst1{~uint32_t{0u}};
    uint32_t excl_prim1{~uint32_t{0u}};
    float t_min{0.0f};
    float t_max{0.0f};
    bool terminate_on_commit{false};
    bool has_curves{false};
};
static_assert(offsetof(LiveRayQueryContext, rtc) == 0u);

thread_local LiveRayQueryContext *active_live_ray_query_context{nullptr};

class ScopedLiveRayQueryContext {
private:
    LiveRayQueryContext *_previous;

public:
    explicit ScopedLiveRayQueryContext(LiveRayQueryContext &ctx) noexcept
        : _previous{active_live_ray_query_context} {
        active_live_ray_query_context = &ctx;
    }
    ~ScopedLiveRayQueryContext() noexcept {
        active_live_ray_query_context = _previous;
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
    auto bits = luisa::bit_cast<uint32_t>(tnear);
    auto magnitude = bits & magnitude_mask;
    if (magnitude >= infinity_bits) { return tnear; }
    if (magnitude == 0u) {
        return luisa::bit_cast<float>(sign_bit | minimum_normal_bits);
    }
    bits = (bits & sign_bit) != 0u ? bits + 1u : bits - 1u;
    if ((bits & magnitude_mask) < minimum_normal_bits) {
        bits = sign_bit | minimum_normal_bits;
    }
    return luisa::bit_cast<float>(bits);
}

static_assert(
    luisa::bit_cast<uint32_t>(ray_query_embree_tnear(1.0f)) ==
    luisa::bit_cast<uint32_t>(1.0f) - 1u);
static_assert(
    luisa::bit_cast<uint32_t>(ray_query_embree_tnear(0.0f)) ==
    0x80800000u);
static_assert(
    luisa::bit_cast<uint32_t>(ray_query_embree_tnear(-0.0f)) ==
    0x80800000u);
static_assert(
    luisa::bit_cast<uint32_t>(ray_query_embree_tnear(
        luisa::bit_cast<float>(uint32_t{0x7fc12345u}))) ==
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
        instances.committed_instances != nullptr &&
        inst < instances.committed_size &&
        instances.committed_instances[inst].geometry_kind ==
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
        auto raw_inst = RTCHitN_instID(
            arguments->hit, arguments->N, packet_lane, 0u);
        auto inst = raw_inst == RTC_INVALID_GEOMETRY_ID ?
                        RTCHitN_geomID(
                            arguments->hit, arguments->N, packet_lane) :
                        raw_inst;
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
        auto curve = context->instances->committed_instances != nullptr &&
                     inst < context->instances->committed_size &&
                     context->instances->committed_instances[inst]
                             .geometry_kind ==
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
            instances.committed_instances != nullptr &&
                candidate.inst < instances.committed_size,
            "SIMD ray query returned an invalid instance ID {}.",
            candidate.inst);
        auto opaque = instances.data != nullptr &&
                              candidate.inst < instances.size ?
                          instances.data[candidate.inst].opaque :
                          instances.committed_instances[candidate.inst].opaque;
        if (opaque != 0u) {
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
            luisa::sort(begin, end, ray_query_candidate_before);
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
            luisa::sort(
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
            luisa::sort(begin, end, ray_query_candidate_before);
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
            luisa::sort(
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

[[nodiscard]] LUISA_FORCE_INLINE uint64_t ray_query_lane_status(
    const SIMDHostRayQueryState &state, uint64_t bit) noexcept {
    auto status = state.terminated != 0u ? bit : 0u;
    if (state.candidate_kind == static_cast<uint32_t>(
                                    SIMDHostRayQueryCandidateKind::surface)) {
        status |= bit << simd_host_ray_query_surface_status_shift;
    } else if (state.candidate_kind == static_cast<uint32_t>(
                                           SIMDHostRayQueryCandidateKind::procedural)) {
        status |= bit << simd_host_ray_query_procedural_status_shift;
    }
    return status;
}

[[nodiscard]] LUISA_NEVER_INLINE uint64_t
install_ray_query_candidate_batches_status(
    RayQueryScanContext &context, uint64_t active_mask_bits,
    const SIMDHostAccelInstanceTable &instances) noexcept {
    // Batch construction maintains heapified => !ascending. Keep the common
    // already-ascending case on one branch; heap repair and both reorderings
    // are cold for coherent W16 procedural packets.
    auto status = uint64_t{0u};
    for (auto lane = 0u; lane < context.lane_count; lane++) {
        auto bit = uint64_t{1u} << lane;
        if ((active_mask_bits & bit) == 0u) { continue; }
        auto *state = context.states[lane];
        auto &build = context.batch_build[lane];
        auto begin = std::begin(state->candidate_batch);
        auto end = begin + state->candidate_batch_count;
        if (!build.ascending) [[unlikely]] {
            if (build.heapified) {
                std::sort_heap(begin, end, ray_query_candidate_before);
            } else if (build.descending) {
                std::reverse(begin, end);
            } else {
                luisa::sort(begin, end, ray_query_candidate_before);
            }
        }
        auto &procedural_build =
            context.procedural_batch_build[lane];
        auto procedural_begin = std::begin(state->procedural_batch);
        auto procedural_end =
            procedural_begin + state->procedural_batch_count;
        if (!procedural_build.ascending) [[unlikely]] {
            if (procedural_build.heapified) {
                std::sort_heap(
                    procedural_begin, procedural_end,
                    ray_query_procedural_before);
            } else if (procedural_build.descending) {
                std::reverse(procedural_begin, procedural_end);
            } else {
                luisa::sort(
                    procedural_begin, procedural_end,
                    ray_query_procedural_before);
            }
        }
        state->candidate_batch_index = 0u;
        state->candidate_batch_initialized = 1u;
        state->procedural_batch_index = 0u;
        state->procedural_batch_initialized = 1u;
        auto advanced = advance_ray_query_candidate(*state, instances);
        LUISA_ASSERT(
            advanced != RayQueryCandidateAdvance::needs_scan,
            "A newly scanned SIMD ray-query batch made no progress.");
        status |= ray_query_lane_status(*state, bit);
    }
    return status;
}

[[nodiscard]] LUISA_NEVER_INLINE uint64_t
install_ray_query_candidate_batches_wide_status(
    RayQueryScanContext &context, uint64_t active_mask_bits,
    const SIMDHostAccelInstanceTable &instances) noexcept {
    auto status = uint64_t{0u};
    while (active_mask_bits != 0u) {
        auto lane = static_cast<uint32_t>(
            std::countr_zero(active_mask_bits));
        auto bit = uint64_t{1u} << lane;
        active_mask_bits &= active_mask_bits - 1u;
        auto *state = context.states[lane];
        auto &build = context.batch_build[lane];
        auto begin = std::begin(state->candidate_batch);
        auto end = begin + state->candidate_batch_count;
        if (!build.ascending) [[unlikely]] {
            if (build.heapified) {
                std::sort_heap(begin, end, ray_query_candidate_before);
            } else if (build.descending) {
                std::reverse(begin, end);
            } else {
                luisa::sort(begin, end, ray_query_candidate_before);
            }
        }
        auto &procedural_build =
            context.procedural_batch_build[lane];
        auto procedural_begin = std::begin(state->procedural_batch);
        auto procedural_end =
            procedural_begin + state->procedural_batch_count;
        if (!procedural_build.ascending) [[unlikely]] {
            if (procedural_build.heapified) {
                std::sort_heap(
                    procedural_begin, procedural_end,
                    ray_query_procedural_before);
            } else if (procedural_build.descending) {
                std::reverse(procedural_begin, procedural_end);
            } else {
                luisa::sort(
                    procedural_begin, procedural_end,
                    ray_query_procedural_before);
            }
        }
        state->candidate_batch_index = 0u;
        state->candidate_batch_initialized = 1u;
        state->procedural_batch_index = 0u;
        state->procedural_batch_initialized = 1u;
        auto advanced = advance_ray_query_candidate(*state, instances);
        LUISA_ASSERT(
            advanced != RayQueryCandidateAdvance::needs_scan,
            "A newly scanned SIMD ray-query batch made no progress.");
        status |= ray_query_lane_status(*state, bit);
    }
    return status;
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

template<size_t packet_width, typename RayHitPacket, typename RayPacket>
[[nodiscard]] LUISA_NEVER_INLINE uint64_t
scan_ray_query_packet_wide_status(
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
        return install_ray_query_candidate_batches_status(
            context, active_mask_bits, instances);
    }
    return install_ray_query_candidate_batches_wide_status(
        context, active_mask_bits, instances);
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
    if (auto *live = active_live_ray_query_context;
        live != nullptr &&
        reinterpret_cast<const void *>(arguments->context) ==
            reinterpret_cast<const void *>(&live->rtc)) {
        if (arguments->valid == nullptr || arguments->N == 0u) { return; }
        auto instance = live->rtc.instID[0u];
        auto primitive = arguments->primID;
        for (auto i = 0u; i < arguments->N; ++i) {
            if (arguments->valid[i] != -1) { continue; }
            arguments->valid[i] = 0;
            auto *state = live->state;
            if (state == nullptr || instance == RTC_INVALID_GEOMETRY_ID) {
                continue;
            }
            if (state->terminated != 0u) {
                RTCRayN_tfar(rays, arguments->N, i) =
                    -std::numeric_limits<float>::infinity();
                continue;
            }
            state->candidate = SIMDHostRayQuerySurfaceHit{
                .inst = instance,
                .prim = primitive,
                .bary = {-1.0f, -1.0f},
                .t = 0.0f,
            };
            state->candidate_kind = static_cast<uint32_t>(
                SIMDHostRayQueryCandidateKind::procedural);
            state->candidate_committed = 0u;
            if (live->w1_on_candidate != nullptr) {
                live->w1_on_candidate(
                    state, live->capture, live->launch_config,
                    static_cast<uint32_t>(
                        SIMDHostRayQueryCandidateKind::procedural));
            } else {
                live->on_candidate(
                    live->lane_count, uint64_t{1u} << live->lane,
                    live->states, live->capture, live->launch_config,
                    static_cast<uint32_t>(
                        SIMDHostRayQueryCandidateKind::procedural));
            }
            if (state->candidate_committed != 0u) {
                auto t = state->candidate.t;
                state->committed = SIMDHostRayQueryCommittedHit{
                    .inst = instance,
                    .prim = primitive,
                    .bary = {0.0f, 0.0f},
                    .kind = static_cast<uint32_t>(
                        SIMDHostRayQueryCandidateKind::procedural),
                    .t = t,
                };
                state->world_ray[7u] = t;
                RTCRayN_tfar(rays, arguments->N, i) = t;
                if (state->terminate_on_first != 0u) {
                    state->terminated = 1u;
                }
            }
            state->candidate_kind = static_cast<uint32_t>(
                SIMDHostRayQueryCandidateKind::none);
            state->candidate_committed = 0u;
            if (state->terminated != 0u) {
                RTCRayN_tfar(rays, arguments->N, i) =
                    -std::numeric_limits<float>::infinity();
            }
        }
        return;
    }
    collect_procedural_candidates(
        arguments->valid, rays, arguments->N,
        arguments->primID, arguments->context);
}

void simd_procedural_occluded(
    const RTCOccludedFunctionNArguments *arguments) noexcept {
    if (arguments == nullptr || arguments->ray == nullptr) { return; }
    if (auto *live = active_live_ray_query_context;
        live != nullptr &&
        reinterpret_cast<const void *>(arguments->context) ==
            reinterpret_cast<const void *>(&live->rtc)) {
        if (arguments->valid == nullptr || arguments->N == 0u) { return; }
        auto instance = live->rtc.instID[0u];
        auto primitive = arguments->primID;
        for (auto i = 0u; i < arguments->N; ++i) {
            if (arguments->valid[i] != -1) { continue; }
            arguments->valid[i] = 0;
            auto *state = live->state;
            if (state == nullptr || instance == RTC_INVALID_GEOMETRY_ID) {
                continue;
            }
            if (state->terminated != 0u) {
                RTCRayN_tfar(arguments->ray, arguments->N, i) =
                    -std::numeric_limits<float>::infinity();
                continue;
            }
            state->candidate = SIMDHostRayQuerySurfaceHit{
                .inst = instance,
                .prim = primitive,
                .bary = {-1.0f, -1.0f},
                .t = 0.0f,
            };
            state->candidate_kind = static_cast<uint32_t>(
                SIMDHostRayQueryCandidateKind::procedural);
            state->candidate_committed = 0u;
            if (live->w1_on_candidate != nullptr) {
                live->w1_on_candidate(
                    state, live->capture, live->launch_config,
                    static_cast<uint32_t>(
                        SIMDHostRayQueryCandidateKind::procedural));
            } else {
                live->on_candidate(
                    live->lane_count, uint64_t{1u} << live->lane,
                    live->states, live->capture, live->launch_config,
                    static_cast<uint32_t>(
                        SIMDHostRayQueryCandidateKind::procedural));
            }
            if (state->candidate_committed != 0u) {
                auto t = state->candidate.t;
                state->committed = SIMDHostRayQueryCommittedHit{
                    .inst = instance,
                    .prim = primitive,
                    .bary = {0.0f, 0.0f},
                    .kind = static_cast<uint32_t>(
                        SIMDHostRayQueryCandidateKind::procedural),
                    .t = t,
                };
                state->world_ray[7u] = t;
                state->terminated = 1u;
            }
            state->candidate_kind = static_cast<uint32_t>(
                SIMDHostRayQueryCandidateKind::none);
            state->candidate_committed = 0u;
            if (state->terminated != 0u) {
                RTCRayN_tfar(arguments->ray, arguments->N, i) =
                    -std::numeric_limits<float>::infinity();
            }
        }
        return;
    }
    collect_procedural_candidates(
        arguments->valid, arguments->ray, arguments->N,
        arguments->primID, arguments->context);
}

SIMDAccel::SIMDAccel(
    RTCDevice device, const AccelOption &option,
    uint32_t warp_width) noexcept
    : _scene{rtcNewScene(device)},
      _warp_width{warp_width},
      _enable_coherent_w16_direct_trace{
          !luisa::compute::detail::env_flag(
              "LUISA_SIMD_DISABLE_COHERENT_W16_DIRECT_TRACE")},
      _enable_triangle_only_ray_query{
          triangle_ray_query::triangle_only_ray_query_enabled()},
      _enable_surface_filter_pipeline{
          !luisa::compute::detail::env_flag(
              "LUISA_SIMD_DISABLE_IN_FILTER_RAY_QUERY_PIPELINE")},
      _enable_surface_filter_ray_packet{
          !luisa::compute::detail::env_flag(
              "LUISA_SIMD_DISABLE_IN_FILTER_RAY_PACKET_INPUT")},
      _embree_native_ray_packet{
          simd_embree_native_ray_packet_support(device)
              .supports(warp_width)},
      _enable_direct_surface_filter_candidate{
          _embree_native_ray_packet &&
          !luisa::compute::detail::env_flag(
              "LUISA_SIMD_DISABLE_DIRECT_SURFACE_FILTER_CANDIDATE")},
      _enable_output_only_empty_surface_filter{
          !luisa::compute::detail::env_flag(
              "LUISA_SIMD_DISABLE_OUTPUT_ONLY_EMPTY_SURFACE_FILTER")},
      _enable_direct_output_surface_filter{
          !luisa::compute::detail::env_flag(
              "LUISA_SIMD_DISABLE_DIRECT_OUTPUT_SURFACE_FILTER")},
      _enable_narrow_shared_status{
          !luisa::compute::detail::env_flag(
              "LUISA_SIMD_DISABLE_NARROW_SHARED_STATUS")},
      _enable_w8_wide_shared_status{
          !luisa::compute::detail::env_flag(
              "LUISA_SIMD_DISABLE_W8_WIDE_SHARED_STATUS")},
      _enable_procedural_dense_status{
          !luisa::compute::detail::env_flag(
              "LUISA_SIMD_DISABLE_PROCEDURAL_WIDE_STATUS_PACK")},
      _enable_procedural_fused_status{
          !luisa::compute::detail::env_flag(
              "LUISA_SIMD_DISABLE_PROCEDURAL_WIDE_FUSED_STATUS")},
      _enable_live_ray_query_pipeline{
          !luisa::compute::detail::env_flag(
              "LUISA_SIMD_DISABLE_LIVE_RAY_QUERY_PIPELINE")} {
    _instance_table.ray_query_pipeline_w1 =
        _ray_query_pipeline_w1;
    simd_accel_set_flags(_scene, option);
}

SIMDAccel::~SIMDAccel() noexcept { rtcReleaseScene(_scene); }

void SIMDAccel::build(const AccelBuildCommand &command) noexcept {
    auto instance_count = command.instance_count();
    _instance_summary_dirty |=
        instance_count != _instances.size();
    if (instance_count < _instances.size()) {
        _instances.resize(instance_count);
        _primitives.resize(instance_count);
        _motion_states.resize(instance_count);
    } else {
        _instances.reserve(instance_count);
        _primitives.reserve(instance_count);
        _motion_states.reserve(instance_count);
        for (auto i = _instances.size(); i < instance_count; i++) {
            _instances.emplace_back();
            _primitives.emplace_back(nullptr);
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
        if ((modification.flags & Modification::flag_primitive) != 0u) {
            _instance_summary_dirty = true;
            auto *primitive = reinterpret_cast<SIMDPrimitive *>(
                modification.primitive);
            LUISA_ASSERT(
                primitive != nullptr,
                "SIMD accel instance has a null primitive.");
            _primitives[modification.index] = primitive;
            if (primitive->kind() == SIMDPrimitive::Kind::motion_instance) {
                auto *motion = static_cast<SIMDMotionInstance *>(primitive);
                LUISA_ASSERT(
                    motion->child() != nullptr &&
                        motion->keyframes().size() ==
                            motion->option().keyframe_count,
                    "SIMD motion instance must be built before its accel.");
                auto state = std::make_unique<MotionState>();
                state->option = motion->option();
                state->keyframes.assign(
                    motion->keyframes().begin(), motion->keyframes().end());
                state->source_build_version = motion->build_version();
                instance.motion_frames = state->keyframes.data();
                instance.motion_keyframe_count =
                    static_cast<uint32_t>(state->keyframes.size());
                instance.motion_mode =
                    static_cast<uint32_t>(state->option.mode);
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

    // A MotionInstance build is a resource update rather than an Accel
    // modification. Import a new host keyframe generation here, while leaving
    // device-authored keyframe stores intact when the source generation did
    // not change.
    for (auto i = size_t{0u}; i < _instances.size(); i++) {
        auto *primitive = _primitives[i];
        auto &state = _motion_states[i];
        if (primitive == nullptr || state == nullptr ||
            primitive->kind() != SIMDPrimitive::Kind::motion_instance) {
            continue;
        }
        auto *motion = static_cast<SIMDMotionInstance *>(primitive);
        if (state->source_build_version == motion->build_version()) {
            continue;
        }
        LUISA_ASSERT(
            motion->child() != nullptr &&
                motion->keyframes().size() ==
                    motion->option().keyframe_count,
            "SIMD motion instance must be built before its accel.");
        state->option = motion->option();
        state->keyframes.assign(
            motion->keyframes().begin(), motion->keyframes().end());
        state->source_build_version = motion->build_version();
        auto &instance = _instances[i];
        instance.motion_frames = state->keyframes.data();
        instance.motion_keyframe_count =
            static_cast<uint32_t>(state->keyframes.size());
        instance.motion_mode =
            static_cast<uint32_t>(state->option.mode);
        instance.dirty = 1u;
    }

    auto refresh_instance_summary = [&]() noexcept {
        _has_curve_instances = std::any_of(
            _committed_instances.cbegin(), _committed_instances.cend(),
            [](const SIMDHostAccelCommittedInstance &instance) noexcept {
                return instance.geometry_kind ==
                       static_cast<uint8_t>(
                           SIMDHostAccelGeometryKind::curve);
            });
        _has_procedural_instances = std::any_of(
            _committed_instances.cbegin(), _committed_instances.cend(),
            [](const SIMDHostAccelCommittedInstance &instance) noexcept {
                return instance.geometry_kind ==
                       static_cast<uint8_t>(
                           SIMDHostAccelGeometryKind::procedural);
            });
        auto use_triangle_only_provider =
            !_has_procedural_instances && !_has_curve_instances &&
            _enable_triangle_only_ray_query;
        if (use_triangle_only_provider &&
            _enable_surface_filter_pipeline) {
            if (_warp_width >= 4u) {
                _instance_table.ray_query_surface_filter_packet_pipeline =
                    _enable_surface_filter_ray_packet ?
                        triangle_ray_query::
                            ray_query_surface_filter_packet_pipeline_triangle_only :
                        triangle_ray_query::
                            ray_query_surface_filter_packet_pipeline_triangle_only_state_oracle;
            } else {
                _instance_table.ray_query_surface_filter_pipeline =
                    triangle_ray_query::
                        ray_query_surface_filter_pipeline_triangle_only;
            }
        } else {
            _instance_table.ray_query_surface_filter_pipeline = nullptr;
        }
        _instance_table.ray_query_empty_surface_filter_packet_pipeline =
            use_triangle_only_provider &&
                    _enable_surface_filter_pipeline &&
                    _enable_surface_filter_ray_packet &&
                    _embree_native_ray_packet &&
                    _enable_output_only_empty_surface_filter &&
                    _warp_width >= 2u ?
                triangle_ray_query::
                    ray_query_empty_surface_filter_packet_pipeline_triangle_only :
                nullptr;
        _instance_table.ray_query_direct_output_surface_filter_packet_pipeline =
            use_triangle_only_provider &&
                    _enable_surface_filter_pipeline &&
                    _enable_surface_filter_ray_packet &&
                    _embree_native_ray_packet &&
                    _enable_direct_surface_filter_candidate &&
                    _enable_direct_output_surface_filter &&
                    _warp_width >= 2u ?
                triangle_ray_query::
                    ray_query_direct_output_surface_filter_packet_pipeline_triangle_only :
                nullptr;
        auto use_narrow_shared_status =
            !use_triangle_only_provider &&
            (_warp_width == 2u || _warp_width == 4u) &&
            _enable_narrow_shared_status;
        _instance_table.ray_query_proceed_status =
            use_narrow_shared_status ?
                _ray_query_proceed_status :
                simd_host_ray_query_proceed_status;
        auto use_w16_procedural_status =
            simd_host_ray_query_use_procedural_wide_status(
                _warp_width,
                _has_procedural_instances,
                _enable_procedural_dense_status);
        auto use_w8_shared_status =
            !use_triangle_only_provider &&
            _warp_width == 8u && _has_procedural_instances &&
            _enable_w8_wide_shared_status;
        _instance_table.ray_query_proceed_wide_status =
            use_w16_procedural_status ?
                (_enable_procedural_fused_status ?
                     simd_host_ray_query_proceed_wide_procedural_fused_status :
                     simd_host_ray_query_proceed_wide_procedural_status) :
            use_w8_shared_status ?
                _ray_query_proceed_wide_status :
                simd_host_ray_query_proceed_status;
    };
    if (command.update_instance_buffer_only()) {
        // The public table is now current, but Embree deliberately remains at
        // its last committed state. Keep every dirty bit and desired primitive
        // binding so a later ordinary build can catch up even though this
        // command has consumed the runtime modification list.
        _instance_flags_valid = false;
        return;
    }

    auto device = rtcGetSceneDevice(_scene);
    if (instance_count < _geometries.size()) {
        for (auto i = instance_count; i < _geometries.size(); i++) {
            if (_geometries[i] != nullptr) {
                rtcDetachGeometry(_scene, static_cast<unsigned>(i));
            }
        }
        _geometries.resize(instance_count);
        _geometry_routes.resize(instance_count);
        _direct_meshes.resize(instance_count);
        _mesh_revisions.resize(instance_count);
        _motion_forwarders.resize(instance_count);
    } else if (instance_count > _geometries.size()) {
        _geometries.reserve(instance_count);
        _geometry_routes.reserve(instance_count);
        _direct_meshes.reserve(instance_count);
        _mesh_revisions.reserve(instance_count);
        _motion_forwarders.reserve(instance_count);
        for (auto i = _geometries.size(); i < instance_count; i++) {
            _geometries.emplace_back(nullptr);
            _geometry_routes.emplace_back(GeometryRoute::instance);
            _direct_meshes.emplace_back(nullptr);
            _mesh_revisions.emplace_back(0u);
            _motion_forwarders.emplace_back();
            _instances[i].dirty = 1u;
        }
    }
    const auto direct_enabled = simd_direct_static_mesh_enabled();
    luisa::unordered_map<const SIMDMesh *, uint32_t> mesh_ref_counts;
    if (direct_enabled && instance_count != 0u) {
        mesh_ref_counts.reserve(instance_count);
        for (auto i = size_t{0u}; i < instance_count; i++) {
            auto *primitive = _primitives[i];
            if (primitive != nullptr &&
                primitive->kind() == SIMDPrimitive::Kind::mesh) {
                ++mesh_ref_counts[static_cast<const SIMDMesh *>(primitive)];
            }
        }
    }
    auto ensure_geometry_route = [&](
                                     size_t index,
                                     GeometryRoute route,
                                     const SIMDMesh *direct_mesh) noexcept {
        if (_geometries[index] != nullptr &&
            _geometry_routes[index] == route &&
            (route != GeometryRoute::direct_triangle ||
             _direct_meshes[index] == direct_mesh)) {
            return false;
        }
        if (_geometries[index] != nullptr) {
            rtcDetachGeometry(
                _scene, static_cast<unsigned>(index));
        }
        _motion_forwarders[index].reset();
        auto geometry_type =
            route == GeometryRoute::forwarded_srt ?
                RTC_GEOMETRY_TYPE_USER :
            route == GeometryRoute::direct_triangle ?
                RTC_GEOMETRY_TYPE_TRIANGLE :
                RTC_GEOMETRY_TYPE_INSTANCE;
        auto geometry = rtcNewGeometry(device, geometry_type);
        rtcSetGeometryBuildQuality(
            geometry,
            route == GeometryRoute::direct_triangle && direct_mesh != nullptr ?
                simd_triangle_geometry_build_quality(direct_mesh->option()) :
                RTC_BUILD_QUALITY_HIGH);
        simd_geometry_enable_argument_filter(geometry);
        rtcAttachGeometryByID(
            _scene, geometry, static_cast<unsigned>(index));
        rtcReleaseGeometry(geometry);
        _geometries[index] = geometry;
        _geometry_routes[index] = route;
        _direct_meshes[index] =
            route == GeometryRoute::direct_triangle ? direct_mesh : nullptr;
        _mesh_revisions[index] = 0u;
        return true;
    };
    _committed_instances.resize(instance_count);
    _has_direct_triangle_instances = false;
    for (auto i = size_t{0u}; i < _instances.size(); i++) {
        auto &instance = _instances[i];
        auto *primitive = _primitives[i];
        LUISA_ASSERT(
            primitive != nullptr,
            "SIMD accel instance {} has no primitive binding.", i);
        auto &motion = _motion_states[i];
        auto forwarded_srt =
            motion != nullptr &&
            motion->option.mode == AccelMotionMode::SRT &&
            !affine_is_identity(instance.affine);
        const SIMDMesh *want_direct_mesh = nullptr;
        if (direct_enabled && motion == nullptr &&
            primitive->kind() == SIMDPrimitive::Kind::mesh) {
            const auto *mesh = static_cast<const SIMDMesh *>(primitive);
            if (!mesh->option().motion &&
                !mesh->option().allow_update &&
                mesh->has_shared_buffers() &&
                affine_is_identity(instance.affine) &&
                mesh_ref_counts[mesh] == 1u) {
                want_direct_mesh = mesh;
            }
        }
        if (want_direct_mesh != nullptr) {
            _has_direct_triangle_instances = true;
            const auto rev = want_direct_mesh->build_revision();
            const auto recreated = ensure_geometry_route(
                i, GeometryRoute::direct_triangle, want_direct_mesh);
            auto geometry = _geometries[i];
            const auto mesh_rebuilt = _mesh_revisions[i] != rev;
            if (recreated || mesh_rebuilt) {
                want_direct_mesh->bind_shared_buffers(geometry);
                _mesh_revisions[i] = rev;
            }
            if (recreated || instance.dirty != 0u) {
                rtcSetGeometryMask(geometry, instance.mask);
                _committed_instances[i] = {
                    .geometry_kind = instance.geometry_kind,
                    .opaque = instance.opaque};
            }
            if (recreated || mesh_rebuilt || instance.dirty != 0u) {
                rtcCommitGeometry(geometry);
            }
            instance.dirty = 0u;
            continue;
        }
        if (instance.dirty != 0u ||
            _geometries[i] == nullptr ||
            _geometry_routes[i] == GeometryRoute::direct_triangle) {
            if (ensure_geometry_route(
                    i,
                    forwarded_srt ?
                        GeometryRoute::forwarded_srt :
                        GeometryRoute::instance,
                    nullptr)) {
                instance.dirty = 1u;
            }
        }
        auto geometry = _geometries[i];
        if (motion != nullptr) {
            LUISA_ASSERT(
                primitive->kind() == SIMDPrimitive::Kind::motion_instance,
                "SIMD motion-instance state has a non-motion primitive.");
            auto *motion_primitive =
                static_cast<SIMDMotionInstance *>(primitive);
            LUISA_ASSERT(
                instance.motion_frames == motion->keyframes.data() &&
                    instance.motion_keyframe_count ==
                        motion->keyframes.size() &&
                    instance.motion_mode ==
                        static_cast<uint32_t>(motion->option.mode),
                "SIMD motion-instance metadata is inconsistent.");
            if (forwarded_srt) {
                auto forwarder =
                    std::make_unique<SIMDSRTMotionForwarder>(
                        device, motion_primitive->child()->handle(),
                        motion->option, motion->keyframes,
                        instance.affine);
                forwarder->configure_geometry(geometry);
                _motion_forwarders[i] = std::move(forwarder);
            } else if (instance.dirty != 0u) {
                _motion_forwarders[i].reset();
                rtcSetGeometryInstancedScene(
                    geometry, motion_primitive->child()->handle());
                rtcSetGeometryTimeStepCount(
                    geometry, motion->option.keyframe_count);
                rtcSetGeometryTimeRange(
                    geometry, motion->option.time_start,
                    motion->option.time_end);
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
                    for (auto key = size_t{0u};
                         key < motion->keyframes.size(); key++) {
                        auto quaternion = quaternion_keyframe(
                            motion->keyframes[key].as_srt(), key);
                        rtcSetGeometryTransformQuaternion(
                            geometry, static_cast<unsigned>(key),
                            &quaternion);
                    }
                }
            }
        } else if (instance.dirty != 0u) {
            _motion_forwarders[i].reset();
            LUISA_ASSERT(
                primitive->kind() == SIMDPrimitive::Kind::mesh ||
                    primitive->kind() == SIMDPrimitive::Kind::curve ||
                    primitive->kind() == SIMDPrimitive::Kind::procedural,
                "SIMD static instance has an invalid primitive binding.");
            rtcSetGeometryInstancedScene(
                geometry, primitive->handle());
            rtcSetGeometryTimeStepCount(geometry, 1u);
            rtcSetGeometryTimeRange(geometry, 0.0f, 1.0f);
            rtcSetGeometryTransform(
                geometry, 0u, RTC_FORMAT_FLOAT3X4_ROW_MAJOR,
                instance.affine);
        }
        if (instance.dirty != 0u) {
            rtcSetGeometryMask(geometry, instance.mask);
            _committed_instances[i] = {
                .geometry_kind = instance.geometry_kind,
                .opaque = instance.opaque};
        }
        // A BLAS can be rebuilt without changing the Accel instance record.
        // Recommitting every instance geometry refreshes Embree's cached child
        // scene bounds before the TLAS commit; transform/mask setup remains
        // dirty-only.
        rtcCommitGeometry(geometry);
        instance.dirty = 0u;
    }
    _instance_table.committed_instances = _committed_instances.data();
    _instance_table.committed_size = _committed_instances.size();
    _instance_flags.resize(_committed_instances.size());
    for (size_t i = 0u; i < _committed_instances.size(); ++i) {
        const auto &ci = _committed_instances[i];
        _instance_flags[i] =
            (ci.opaque != 0u ? uint8_t{1u} : uint8_t{0u}) |
            (ci.geometry_kind ==
                     static_cast<uint8_t>(SIMDHostAccelGeometryKind::curve) ?
                 uint8_t{2u} :
                 uint8_t{0u});
    }
    _instance_flags_size = _instance_flags.size();
    _instance_flags_valid = true;
    if (_instance_summary_dirty) {
        refresh_instance_summary();
        _instance_summary_dirty = false;
    }
    rtcCommitScene(_scene);
    rtcReleaseDevice(device);
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
        case 8u: {
            static const uint32_t direct_scalar_max_lanes = []() noexcept -> uint32_t {
                const char *raw = std::getenv("LUISA_SIMD_DIRECT_SCALAR_MAX_LANES");
                if (raw == nullptr || *raw == '\0') { return 2u; }
                return static_cast<uint32_t>(std::strtoul(raw, nullptr, 10));
            }();
            uint32_t valid_count = 0u;
            for (uint32_t l = 0u; l < 8u; ++l) {
                valid_count += (valid[l] != 0) ? 1u : 0u;
            }
            if (valid_count == 0u) { break; }
            if (valid_count <= direct_scalar_max_lanes) {
                auto *pkt = static_cast<RTCRayHit8 *>(ray_hit_packet);
                for (uint32_t l = 0u; l < 8u; ++l) {
                    if (valid[l] == 0) { continue; }
                    RTCRayHit scalar{};
                    scalar.ray.org_x = pkt->ray.org_x[l];
                    scalar.ray.org_y = pkt->ray.org_y[l];
                    scalar.ray.org_z = pkt->ray.org_z[l];
                    scalar.ray.tnear = pkt->ray.tnear[l];
                    scalar.ray.dir_x = pkt->ray.dir_x[l];
                    scalar.ray.dir_y = pkt->ray.dir_y[l];
                    scalar.ray.dir_z = pkt->ray.dir_z[l];
                    scalar.ray.time = pkt->ray.time[l];
                    scalar.ray.tfar = pkt->ray.tfar[l];
                    scalar.ray.mask = pkt->ray.mask[l];
                    scalar.ray.id = pkt->ray.id[l];
                    scalar.ray.flags = pkt->ray.flags[l];
                    scalar.hit.Ng_x = pkt->hit.Ng_x[l];
                    scalar.hit.Ng_y = pkt->hit.Ng_y[l];
                    scalar.hit.Ng_z = pkt->hit.Ng_z[l];
                    scalar.hit.u = pkt->hit.u[l];
                    scalar.hit.v = pkt->hit.v[l];
                    scalar.hit.primID = pkt->hit.primID[l];
                    scalar.hit.geomID = pkt->hit.geomID[l];
                    for (uint32_t lvl = 0u; lvl < RTC_MAX_INSTANCE_LEVEL_COUNT; ++lvl) {
                        scalar.hit.instID[lvl] = pkt->hit.instID[lvl][l];
                    }
                    intersect_scalar(self->_scene, scalar);
                    pkt->ray.tfar[l] = scalar.ray.tfar;
                    pkt->hit.Ng_x[l] = scalar.hit.Ng_x;
                    pkt->hit.Ng_y[l] = scalar.hit.Ng_y;
                    pkt->hit.Ng_z[l] = scalar.hit.Ng_z;
                    pkt->hit.u[l] = scalar.hit.u;
                    pkt->hit.v[l] = scalar.hit.v;
                    pkt->hit.primID[l] = scalar.hit.primID;
                    pkt->hit.geomID[l] = scalar.hit.geomID;
                    for (uint32_t lvl = 0u; lvl < RTC_MAX_INSTANCE_LEVEL_COUNT; ++lvl) {
                        pkt->hit.instID[lvl][l] = scalar.hit.instID[lvl];
                    }
                }
                break;
            }
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
        }
        case 16u: {
            auto coherent =
                self->_enable_coherent_w16_direct_trace &&
                direct_trace_w16_packet_is_full(valid);
            trace_packet_in_place<RTCRayHit16>(
                self->_scene, valid, ray_hit_packet,
                [coherent](const int *valid, RTCScene scene,
                           RTCRayHit16 *packet) noexcept {
#if LUISA_COMPUTE_SIMD_EMBREE_VERSION == 3
                    RTCIntersectContext context{};
                    rtcInitIntersectContext(&context);
                    if (coherent) {
                        context.flags =
                            RTC_INTERSECT_CONTEXT_FLAG_COHERENT;
                    }
                    rtcIntersect16(valid, scene, &context, packet);
#else
                    RTCIntersectArguments arguments{};
                    rtcInitIntersectArguments(&arguments);
                    if (coherent) {
                        arguments.flags = RTC_RAY_QUERY_FLAG_COHERENT;
                    }
                    rtcIntersect16(valid, scene, packet, &arguments);
#endif
                });
            break;
        }
        default:
            LUISA_ERROR_WITH_LOCATION(
                "Unsupported SIMD Embree packet width {}.", lane_count);
    }
    if (self->_has_direct_triangle_instances) {
        auto *packet_words = static_cast<uint32_t *>(ray_hit_packet);
        const auto *geom_words =
            packet_words + simd_host_accel_hit_geom_field * lane_count;
        auto *inst_words =
            packet_words + simd_host_accel_hit_inst_field * lane_count;
        for (auto lane = 0u; lane < lane_count; lane++) {
            if (valid[lane] != 0 &&
                inst_words[lane] == RTC_INVALID_GEOMETRY_ID) {
                inst_words[lane] = geom_words[lane];
            }
        }
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
        case 8u: {
            static const uint32_t direct_scalar_max_lanes = []() noexcept -> uint32_t {
                const char *raw = std::getenv("LUISA_SIMD_DIRECT_SCALAR_MAX_LANES");
                if (raw == nullptr || *raw == '\0') { return 2u; }
                return static_cast<uint32_t>(std::strtoul(raw, nullptr, 10));
            }();
            uint32_t valid_count = 0u;
            for (uint32_t l = 0u; l < 8u; ++l) {
                valid_count += (valid[l] != 0) ? 1u : 0u;
            }
            if (valid_count == 0u) { break; }
            if (valid_count <= direct_scalar_max_lanes) {
                auto *pkt = static_cast<RTCRay8 *>(ray_packet);
                for (uint32_t l = 0u; l < 8u; ++l) {
                    if (valid[l] == 0) { continue; }
                    RTCRay scalar{};
                    scalar.org_x = pkt->org_x[l];
                    scalar.org_y = pkt->org_y[l];
                    scalar.org_z = pkt->org_z[l];
                    scalar.tnear = pkt->tnear[l];
                    scalar.dir_x = pkt->dir_x[l];
                    scalar.dir_y = pkt->dir_y[l];
                    scalar.dir_z = pkt->dir_z[l];
                    scalar.time = pkt->time[l];
                    scalar.tfar = pkt->tfar[l];
                    scalar.mask = pkt->mask[l];
                    scalar.id = pkt->id[l];
                    scalar.flags = pkt->flags[l];
                    occlude_scalar(self->_scene, scalar);
                    pkt->tfar[l] = scalar.tfar;
                }
                break;
            }
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
        }
        case 16u: {
            auto coherent =
                self->_enable_coherent_w16_direct_trace &&
                direct_trace_w16_packet_is_full(valid);
            trace_packet_in_place<RTCRay16>(
                self->_scene, valid, ray_packet,
                [coherent](const int *valid, RTCScene scene,
                           RTCRay16 *packet) noexcept {
#if LUISA_COMPUTE_SIMD_EMBREE_VERSION == 3
                    RTCIntersectContext context{};
                    rtcInitIntersectContext(&context);
                    if (coherent) {
                        context.flags =
                            RTC_INTERSECT_CONTEXT_FLAG_COHERENT;
                    }
                    rtcOccluded16(valid, scene, &context, packet);
#else
                    RTCOccludedArguments arguments{};
                    rtcInitOccludedArguments(&arguments);
                    if (coherent) {
                        arguments.flags = RTC_RAY_QUERY_FLAG_COHERENT;
                    }
                    rtcOccluded16(valid, scene, packet, &arguments);
#endif
                });
            break;
        }
        default:
            LUISA_ERROR_WITH_LOCATION(
                "Unsupported SIMD Embree packet width {}.", lane_count);
    }
}

void SIMDAccel::_ray_query_pipeline_w1(
    SIMDHostRayQueryState *state, const void *capture,
    const SIMDPacketLaunchConfig *launch_config,
    SIMDHostRayQueryPipelineHandlerW1 *on_candidate) noexcept {
    LUISA_ASSERT(
        state != nullptr && state->accel != nullptr &&
            launch_config != nullptr && on_candidate != nullptr,
        "Invalid SIMD W1 resident ray-query pipeline invocation.");
    auto *self = static_cast<SIMDAccel *>(state->accel);
    LUISA_ASSERT(
        self->_instance_table.ray_query_pipeline_w1 ==
            _ray_query_pipeline_w1,
        "SIMD W1 ray-query state selected a mismatched pipeline provider.");
    std::array states{state};
    auto invoke_handler = [&]() noexcept {
        auto kind = static_cast<SIMDHostRayQueryCandidateKind>(
            state->candidate_kind);
        if (kind == SIMDHostRayQueryCandidateKind::surface) {
            on_candidate(
                state, capture, launch_config,
                static_cast<uint32_t>(kind));
        } else if (
            kind == SIMDHostRayQueryCandidateKind::procedural) {
            on_candidate(
                state, capture, launch_config,
                static_cast<uint32_t>(kind));
        } else {
            LUISA_ERROR_WITH_LOCATION(
                "SIMD W1 resident ray query published an invalid candidate kind.");
        }
    };
    for (;;) {
        // Match the ordinary proceed provider exactly, but keep its state
        // transition and the selected JIT handler resident inside one host
        // call. Candidate batching, deterministic ordering, continuation
        // scans, and commit/terminate semantics therefore remain unchanged.
        if (state->candidate_committed != 0u) {
            auto kind = static_cast<SIMDHostRayQueryCandidateKind>(
                state->candidate_kind);
            LUISA_ASSERT(
                kind == SIMDHostRayQueryCandidateKind::surface ||
                    kind == SIMDHostRayQueryCandidateKind::procedural,
                "SIMD W1 resident ray query committed an invalid candidate kind.");
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
        if (state->terminated != 0u) { break; }
        auto advanced = advance_ray_query_candidate(
            *state, self->_instance_table);
        switch (advanced) {
            case RayQueryCandidateAdvance::published:
                invoke_handler();
                break;
            case RayQueryCandidateAdvance::needs_scan:
                scan_ray_query_scalar(
                    self->_scene, self->_instance_table, 1u,
                    states.data(),
                    state->terminate_on_first != 0u);
                // Batch installation publishes the first candidate before
                // returning, just like the ordinary proceed provider. Handle
                // it here instead of advancing past the batch head.
                if (state->terminated == 0u) {
                    invoke_handler();
                }
                break;
            case RayQueryCandidateAdvance::terminated:
                break;
        }
    }
    state->candidate_kind = static_cast<uint32_t>(
        SIMDHostRayQueryCandidateKind::none);
    state->candidate_committed = 0u;
    state->terminated = 1u;
}

LUISA_NEVER_INLINE uint64_t SIMDAccel::_ray_query_proceed_narrow_shared(
    uint32_t lane_count, uint64_t active_mask_bits,
    SIMDHostRayQueryState *const *states,
    bool publish_status) noexcept {
    LUISA_ASSERT(
        states != nullptr &&
            (lane_count == 1u || lane_count == 2u ||
             lane_count == 4u || lane_count == 8u ||
             lane_count == 16u),
        "Invalid SIMD ray-query packet width {}.", lane_count);
    auto lane_mask = lane_count == 64u ? ~uint64_t{0u} :
                                         (uint64_t{1u} << lane_count) - 1u;
    active_mask_bits &= lane_mask;
    if (active_mask_bits == 0u) { return 0u; }

    auto pending = uint64_t{0u};
    auto status = uint64_t{0u};
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
        auto advanced = RayQueryCandidateAdvance::terminated;
        if (state->terminated == 0u) {
            auto *self = static_cast<SIMDAccel *>(state->accel);
            advanced = advance_ray_query_candidate(
                *state, self->_instance_table);
            if (advanced == RayQueryCandidateAdvance::needs_scan) {
                pending |= bit;
            }
        }
        if (publish_status &&
            advanced != RayQueryCandidateAdvance::needs_scan) {
            status |= ray_query_lane_status(*state, bit);
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
        if (publish_status) {
            status |= simd_host_ray_query_pack_status(
                lane_count, group, states);
        }
        pending &= ~group;
    }
    return status;
}

void SIMDAccel::_ray_query_proceed(
    uint32_t lane_count, uint64_t active_mask_bits,
    SIMDHostRayQueryState *const *states) noexcept {
    static_cast<void>(_ray_query_proceed_narrow_shared(
        lane_count, active_mask_bits, states, false));
}

uint64_t SIMDAccel::_ray_query_proceed_status(
    uint32_t lane_count, uint64_t active_mask_bits,
    SIMDHostRayQueryState *const *states) noexcept {
    return _ray_query_proceed_narrow_shared(
        lane_count, active_mask_bits, states, true);
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

LUISA_NEVER_INLINE uint64_t SIMDAccel::_ray_query_proceed_wide_shared(
    uint32_t lane_count, uint64_t active_mask_bits,
    SIMDHostRayQueryState *const *states,
    bool publish_status) noexcept {
    LUISA_ASSERT(
        states != nullptr &&
            (lane_count == 8u || lane_count == 16u),
        "Invalid wide SIMD ray-query packet width {}.", lane_count);
    auto lane_mask = (uint64_t{1u} << lane_count) - 1u;
    active_mask_bits &= lane_mask;
    if (active_mask_bits == 0u) { return 0u; }
    auto fully_active = active_mask_bits == lane_mask;

    auto pending = uint64_t{0u};
    auto status = uint64_t{0u};
    if (fully_active) [[likely]] {
        for (auto lane = 0u; lane < lane_count; lane++) {
            auto bit = uint64_t{1u} << lane;
            _ray_query_proceed_wide_lane(
                states[lane], lane, bit, pending);
            if (publish_status && (pending & bit) == 0u) {
                status |= ray_query_lane_status(*states[lane], bit);
            }
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
            if (publish_status && (pending & bit) == 0u) {
                status |= ray_query_lane_status(*states[lane], bit);
            }
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
            if (publish_status) {
                status |= scan_ray_query_packet_wide_status<
                    8u, RTCRayHit8, RTCRay8>(
                    self->_scene, self->_instance_table,
                    lane_count, group, states,
                    terminate_on_first);
            } else {
                scan_ray_query_packet_wide<
                    8u, RTCRayHit8, RTCRay8>(
                    self->_scene, self->_instance_table,
                    lane_count, group, states,
                    terminate_on_first);
            }
        } else {
            if (publish_status) {
                status |= scan_ray_query_packet_wide_status<
                    16u, RTCRayHit16, RTCRay16>(
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
        }
        pending &= ~group;
    }
    return status;
}

void SIMDAccel::_ray_query_proceed_wide(
    uint32_t lane_count, uint64_t active_mask_bits,
    SIMDHostRayQueryState *const *states) noexcept {
    static_cast<void>(_ray_query_proceed_wide_shared(
        lane_count, active_mask_bits, states, false));
}

void SIMDAccel::_ray_query_candidate_object_ray(
    uint32_t lane_count, uint64_t active_mask_bits,
    SIMDHostRayQueryState *const *states) noexcept {
    LUISA_ASSERT(
        states != nullptr && lane_count >= 1u && lane_count <= 16u,
        "Invalid SIMD candidate object-ray packet arguments.");
    auto remaining = active_mask_bits &
                     ((uint64_t{1u} << lane_count) - 1u);
    while (remaining != 0u) {
        auto lane = static_cast<uint32_t>(std::countr_zero(remaining));
        remaining &= remaining - 1u;
        auto *state = states[lane];
        LUISA_ASSERT(
            state != nullptr && state->accel != nullptr &&
                state->candidate_kind != static_cast<uint32_t>(
                                             SIMDHostRayQueryCandidateKind::none),
            "Invalid SIMD candidate object-ray lane {}.", lane);
        auto *self = static_cast<SIMDAccel *>(state->accel);
        auto instance = state->candidate.inst;
        LUISA_ASSERT(
            instance < self->_geometries.size() &&
                instance < self->_geometry_routes.size(),
            "SIMD candidate object ray references invalid instance {}.",
            instance);
        auto transformed = false;
        if (self->_geometry_routes[instance] ==
            GeometryRoute::direct_triangle) {
            std::memcpy(
                state->object_ray, state->world_ray,
                sizeof(state->object_ray));
            transformed = true;
        } else if (self->_geometry_routes[instance] ==
                   GeometryRoute::forwarded_srt) {
            transformed =
                self->_motion_forwarders[instance]->transform_ray_to_object(
                    state->time, state->world_ray, state->object_ray);
        } else {
            transformed = simd_transform_ray_to_object(
                self->_geometries[instance], state->time,
                state->world_ray, state->object_ray);
        }
        LUISA_ASSERT(
            transformed,
            "Failed to materialize SIMD candidate object ray for instance {}.",
            instance);
    }
}

uint64_t SIMDAccel::_ray_query_proceed_wide_status(
    uint32_t lane_count, uint64_t active_mask_bits,
    SIMDHostRayQueryState *const *states) noexcept {
    return _ray_query_proceed_wide_shared(
        lane_count, active_mask_bits, states, true);
}

const SIMDHostAccelRayQueryLiveDispatch SIMDAccel::_live_dispatch{
    .surface_exclusion = SIMDAccel::_ray_query_surface_exclusion,
    .live_pipeline = SIMDAccel::_ray_query_live_pipeline,
    .surface_exclusion_soa = SIMDAccel::_ray_query_surface_exclusion_soa,
    .surface_exclusion_w1 = SIMDAccel::_ray_query_surface_exclusion_w1,
    .live_pipeline_w1 = SIMDAccel::_ray_query_live_pipeline_w1,
};

namespace {

[[nodiscard]] LUISA_FORCE_INLINE bool live_instance_is_opaque(
    const SIMDHostAccelInstanceTable &instances,
    uint32_t inst) noexcept {
    return instances.data != nullptr && inst < instances.size ?
               instances.data[inst].opaque != 0u :
               instances.committed_instances[inst].opaque != 0u;
}

[[nodiscard]] LUISA_FORCE_INLINE bool live_instance_is_curve(
    const SIMDHostAccelInstanceTable &instances,
    uint32_t inst) noexcept {
    return instances.committed_instances != nullptr &&
           inst < instances.committed_size &&
           instances.committed_instances[inst].geometry_kind ==
               static_cast<uint8_t>(SIMDHostAccelGeometryKind::curve);
}

[[nodiscard]] LUISA_FORCE_INLINE uint8_t live_instance_flags_fast(
    const uint8_t *flags, size_t flags_size,
    const SIMDHostAccelInstanceTable &instances,
    uint32_t inst) noexcept {
    if (flags != nullptr && inst < flags_size) [[likely]] {
        return flags[inst];
    }
    return static_cast<uint8_t>(
        (live_instance_is_opaque(instances, inst) ? 1u : 0u) |
        (live_instance_is_curve(instances, inst) ? 2u : 0u));
}

[[nodiscard]] LUISA_FORCE_INLINE bool live_instance_is_opaque_fast(
    const uint8_t *flags, size_t flags_size,
    const SIMDHostAccelInstanceTable &instances,
    uint32_t inst) noexcept {
    if (flags != nullptr && inst < flags_size) [[likely]] {
        return (flags[inst] & 1u) != 0u;
    }
    return live_instance_is_opaque(instances, inst);
}

[[nodiscard]] LUISA_FORCE_INLINE bool live_instance_is_curve_fast(
    const uint8_t *flags, size_t flags_size,
    const SIMDHostAccelInstanceTable &instances,
    uint32_t inst) noexcept {
    if (flags != nullptr && inst < flags_size) [[likely]] {
        return (flags[inst] & 2u) != 0u;
    }
    return live_instance_is_curve(instances, inst);
}

void live_surface_exclusion_filter(
    const RTCFilterFunctionNArguments *args) noexcept {
    auto *ctx = reinterpret_cast<LiveRayQueryContext *>(args->context);
    if (ctx == nullptr || ctx->state == nullptr ||
        ctx->instances == nullptr || args->valid == nullptr ||
        args->ray == nullptr || args->hit == nullptr ||
        args->N != 1u || args->valid[0u] != -1) [[unlikely]] {
        return;
    }
    auto *state = ctx->state;
    auto *ray = reinterpret_cast<RTCRay *>(args->ray);
    auto *hit = reinterpret_cast<RTCHit *>(args->hit);
    auto t = ray->tfar;
    auto raw_inst = hit->instID[0u];
    auto inst = raw_inst == RTC_INVALID_GEOMETRY_ID ? hit->geomID : raw_inst;
    auto prim = hit->primID;
    if (state->terminated != 0u ||
        !(t >= state->world_ray[3u] && t <= state->world_ray[7u]) ||
        inst == RTC_INVALID_GEOMETRY_ID ||
        prim == RTC_INVALID_GEOMETRY_ID) [[unlikely]] {
        args->valid[0u] = 0;
        return;
    }
    const bool opaque = live_instance_is_opaque_fast(
        ctx->instance_flags, ctx->instance_flags_size, *ctx->instances, inst);
    if (!opaque &&
        ((inst == ctx->excl_inst0 && prim == ctx->excl_prim0) ||
         (inst == ctx->excl_inst1 && prim == ctx->excl_prim1))) {
        args->valid[0u] = 0;
        return;
    }
    float v = (ctx->has_curves &&
               live_instance_is_curve_fast(
                   ctx->instance_flags, ctx->instance_flags_size,
                   *ctx->instances, inst)) ?
                  -1.0f :
                  hit->v;
    state->committed = SIMDHostRayQueryCommittedHit{
        .inst = inst,
        .prim = prim,
        .bary = {hit->u, v},
        .kind = static_cast<uint32_t>(SIMDHostRayQueryCandidateKind::surface),
        .t = t,
    };
    state->world_ray[7u] = t;
    if (state->terminate_on_first != 0u ||
        (!opaque && ctx->terminate_on_commit)) {
        state->terminated = 1u;
        if (state->terminate_on_first == 0u) {
            args->valid[0u] = 0;
            ray->tfar = -std::numeric_limits<float>::infinity();
        }
    }
}

void live_ray_query_filter(
    const RTCFilterFunctionNArguments *args) noexcept {
    auto *ctx = reinterpret_cast<LiveRayQueryContext *>(args->context);
    LUISA_DEBUG_ASSERT(
        ctx != nullptr && ctx->state != nullptr &&
            ctx->instances != nullptr &&
            (ctx->on_candidate != nullptr || ctx->w1_on_candidate != nullptr) &&
            args->valid != nullptr && args->ray != nullptr &&
            args->hit != nullptr && args->N == 1u && args->valid[0u] == -1,
        "Invalid SIMD live ray-query filter arguments.");
    auto *state = ctx->state;
    auto *ray = reinterpret_cast<RTCRay *>(args->ray);
    auto *hit = reinterpret_cast<RTCHit *>(args->hit);
    if (state->terminated != 0u) {
        args->valid[0u] = 0;
        ray->tfar = -std::numeric_limits<float>::infinity();
        return;
    }
    auto t = ray->tfar;
    auto raw_inst = hit->instID[0u];
    auto inst = raw_inst == RTC_INVALID_GEOMETRY_ID ? hit->geomID : raw_inst;
    auto prim = hit->primID;
    if (!(t >= state->world_ray[3u] && t <= state->world_ray[7u]) ||
        inst == RTC_INVALID_GEOMETRY_ID ||
        prim == RTC_INVALID_GEOMETRY_ID) {
        args->valid[0u] = 0;
        return;
    }
    float v = (ctx->has_curves &&
               live_instance_is_curve_fast(
                   ctx->instance_flags, ctx->instance_flags_size,
                   *ctx->instances, inst)) ?
                  -1.0f :
                  hit->v;
    if (live_instance_is_opaque_fast(
            ctx->instance_flags, ctx->instance_flags_size,
            *ctx->instances, inst)) {
        state->committed = SIMDHostRayQueryCommittedHit{
            .inst = inst,
            .prim = prim,
            .bary = {hit->u, v},
            .kind = static_cast<uint32_t>(
                SIMDHostRayQueryCandidateKind::surface),
            .t = t,
        };
        state->world_ray[7u] = t;
        if (state->terminate_on_first != 0u) {
            state->terminated = 1u;
        }
        return;
    }
    state->candidate = SIMDHostRayQuerySurfaceHit{
        .inst = inst,
        .prim = prim,
        .bary = {hit->u, v},
        .t = t,
    };
    state->candidate_kind = static_cast<uint32_t>(
        SIMDHostRayQueryCandidateKind::surface);
    state->candidate_committed = 0u;
    if (ctx->w1_on_candidate != nullptr) {
        ctx->w1_on_candidate(
            state, ctx->capture, ctx->launch_config,
            static_cast<uint32_t>(SIMDHostRayQueryCandidateKind::surface));
    } else {
        ctx->on_candidate(
            ctx->lane_count, uint64_t{1u} << ctx->lane,
            ctx->states, ctx->capture, ctx->launch_config,
            static_cast<uint32_t>(SIMDHostRayQueryCandidateKind::surface));
    }
    const bool committed = state->candidate_committed != 0u;
    state->candidate_kind = static_cast<uint32_t>(
        SIMDHostRayQueryCandidateKind::none);
    state->candidate_committed = 0u;
    if (committed) {
        state->committed = SIMDHostRayQueryCommittedHit{
            .inst = inst,
            .prim = prim,
            .bary = {hit->u, v},
            .kind = static_cast<uint32_t>(
                SIMDHostRayQueryCandidateKind::surface),
            .t = t,
        };
        state->world_ray[7u] = t;
        if (state->terminate_on_first != 0u) {
            state->terminated = 1u;
            return;
        }
        if (state->terminated != 0u) {
            args->valid[0u] = 0;
            ray->tfar = -std::numeric_limits<float>::infinity();
            return;
        }
        return;
    }
    if (state->terminated != 0u) {
        if (state->terminate_on_first != 0u) {
            // Leave valid[0] == -1 so rtcOccluded1 halts immediately without
            // overwriting state->committed.
            return;
        }
        args->valid[0u] = 0;
        ray->tfar = -std::numeric_limits<float>::infinity();
        return;
    }
    args->valid[0u] = 0;
}

void live_ray_query_occluded_filter(
    const RTCFilterFunctionNArguments *args) noexcept {
    auto *ctx = reinterpret_cast<LiveRayQueryContext *>(args->context);
    LUISA_DEBUG_ASSERT(
        ctx != nullptr && ctx->state != nullptr &&
            ctx->instances != nullptr &&
            (ctx->on_candidate != nullptr || ctx->w1_on_candidate != nullptr) &&
            args->valid != nullptr && args->ray != nullptr &&
            args->hit != nullptr && args->N == 1u && args->valid[0u] == -1,
        "Invalid SIMD live occluded ray-query filter arguments.");
    auto *state = ctx->state;
    auto *ray = reinterpret_cast<RTCRay *>(args->ray);
    auto *hit = reinterpret_cast<RTCHit *>(args->hit);
    auto t = ray->tfar;
    auto raw_inst = hit->instID[0u];
    auto inst = raw_inst == RTC_INVALID_GEOMETRY_ID ? hit->geomID : raw_inst;
    auto prim = hit->primID;
    if (!(t >= ctx->t_min && t <= ctx->t_max) ||
        inst == RTC_INVALID_GEOMETRY_ID ||
        prim == RTC_INVALID_GEOMETRY_ID) [[unlikely]] {
        args->valid[0u] = 0;
        return;
    }
    float v = (ctx->has_curves &&
               live_instance_is_curve_fast(
                   ctx->instance_flags, ctx->instance_flags_size,
                   *ctx->instances, inst)) ?
                  -1.0f :
                  hit->v;
    if (live_instance_is_opaque_fast(
            ctx->instance_flags, ctx->instance_flags_size,
            *ctx->instances, inst)) {
        state->committed = SIMDHostRayQueryCommittedHit{
            .inst = inst,
            .prim = prim,
            .bary = {hit->u, v},
            .kind = static_cast<uint32_t>(
                SIMDHostRayQueryCandidateKind::surface),
            .t = t,
        };
        state->world_ray[7u] = t;
        state->terminated = 1u;
        return;
    }
    state->candidate = SIMDHostRayQuerySurfaceHit{
        .inst = inst,
        .prim = prim,
        .bary = {hit->u, v},
        .t = t,
    };
    state->candidate_kind = static_cast<uint32_t>(
        SIMDHostRayQueryCandidateKind::surface);
    state->candidate_committed = 0u;
    if (ctx->w1_on_candidate != nullptr) {
        ctx->w1_on_candidate(
            state, ctx->capture, ctx->launch_config,
            static_cast<uint32_t>(SIMDHostRayQueryCandidateKind::surface));
    } else {
        ctx->on_candidate(
            ctx->lane_count, uint64_t{1u} << ctx->lane,
            ctx->states, ctx->capture, ctx->launch_config,
            static_cast<uint32_t>(SIMDHostRayQueryCandidateKind::surface));
    }
    if (state->candidate_committed != 0u) {
        state->committed = SIMDHostRayQueryCommittedHit{
            .inst = inst,
            .prim = prim,
            .bary = {hit->u, v},
            .kind = static_cast<uint32_t>(
                SIMDHostRayQueryCandidateKind::surface),
            .t = t,
        };
        state->world_ray[7u] = t;
        state->terminated = 1u;
        state->candidate_committed = 0u;
        return;
    }
    if (state->terminated != 0u) {
        // Leave valid[0] == -1 so rtcOccluded1 halts immediately without
        // overwriting state->committed.
        return;
    }
    args->valid[0u] = 0;
}

struct LiveSurfaceExclusionSoAContext {
#if LUISA_COMPUTE_SIMD_EMBREE_VERSION == 3
    RTCIntersectContext rtc{};
#else
    RTCRayQueryContext rtc{};
#endif
    const SIMDHostAccelInstanceTable *instances{nullptr};
    const uint8_t *instance_flags{nullptr};
    size_t instance_flags_size{0u};
    SIMDHostRayQueryOutputPacket *outputs{nullptr};
    uint32_t lane{0u};
    uint32_t excl_inst0{~0u};
    uint32_t excl_prim0{~0u};
    uint32_t excl_inst1{~0u};
    uint32_t excl_prim1{~0u};
    float t_min{0.0f};
    float t_max_current{0.0f};
    bool query_any{false};
    bool terminate_on_commit{false};
    bool has_curves{false};
    bool defer_output_write{false};
    bool terminated{false};
};

void live_surface_exclusion_soa_filter(
    const RTCFilterFunctionNArguments *args) noexcept {
    auto *ctx =
        reinterpret_cast<LiveSurfaceExclusionSoAContext *>(args->context);
    LUISA_DEBUG_ASSERT(
        ctx != nullptr && ctx->outputs != nullptr &&
            ctx->instances != nullptr && args->valid != nullptr &&
            args->ray != nullptr && args->hit != nullptr &&
            args->N == 1u && args->valid[0u] == -1,
        "Invalid SIMD surface-exclusion SoA filter arguments.");
    auto *ray = reinterpret_cast<RTCRay *>(args->ray);
    auto *hit = reinterpret_cast<RTCHit *>(args->hit);
    auto t = ray->tfar;
    auto raw_inst = hit->instID[0u];
    auto inst = raw_inst == RTC_INVALID_GEOMETRY_ID ? hit->geomID : raw_inst;
    auto prim = hit->primID;
    if (ctx->terminated ||
        !(t >= ctx->t_min && t <= ctx->t_max_current) ||
        inst == RTC_INVALID_GEOMETRY_ID ||
        prim == RTC_INVALID_GEOMETRY_ID) [[unlikely]] {
        args->valid[0u] = 0;
        return;
    }
    const bool opaque = live_instance_is_opaque_fast(
        ctx->instance_flags, ctx->instance_flags_size, *ctx->instances, inst);
    if (!opaque &&
        ((inst == ctx->excl_inst0 && prim == ctx->excl_prim0) ||
         (inst == ctx->excl_inst1 && prim == ctx->excl_prim1))) {
        args->valid[0u] = 0;
        return;
    }
    if (ctx->defer_output_write) {
        ctx->t_max_current = t;
        return;
    }
    float v = (ctx->has_curves &&
               live_instance_is_curve_fast(
                   ctx->instance_flags, ctx->instance_flags_size,
                   *ctx->instances, inst)) ?
                  -1.0f :
                  hit->v;
    const auto l = ctx->lane;
    ctx->outputs->committed_inst[l] = inst;
    ctx->outputs->committed_prim[l] = prim;
    ctx->outputs->committed_bary_x[l] = hit->u;
    ctx->outputs->committed_bary_y[l] = v;
    ctx->outputs->committed_kind[l] =
        static_cast<uint32_t>(SIMDHostRayQueryCandidateKind::surface);
    ctx->outputs->committed_t[l] = t;
    ctx->t_max_current = t;
    if (ctx->query_any || (!opaque && ctx->terminate_on_commit)) {
        ctx->terminated = true;
        if (!ctx->query_any) {
            args->valid[0u] = 0;
            ray->tfar = -std::numeric_limits<float>::infinity();
        }
    }
}

#if LUISA_COMPUTE_SIMD_EMBREE_VERSION != 3
struct LiveSurfaceExclusionSoAPacketContext {
    RTCRayQueryContext rtc{};
    const SIMDHostAccelInstanceTable *instances{nullptr};
    const uint8_t *instance_flags{nullptr};
    size_t instance_flags_size{0u};
    SIMDHostRayQueryOutputPacket *outputs{nullptr};
    const uint32_t *excl_inst0{nullptr};
    const uint32_t *excl_prim0{nullptr};
    const uint32_t *excl_inst1{nullptr};
    const uint32_t *excl_prim1{nullptr};
    float t_min[8]{};
    float t_max_current[8]{};
    bool query_any{false};
    bool terminate_on_commit{false};
    bool has_curves{false};
    bool terminated[8]{};
};

void live_surface_exclusion_soa_filter_n(
    const RTCFilterFunctionNArguments *args) noexcept {
    auto *ctx = reinterpret_cast<LiveSurfaceExclusionSoAPacketContext *>(
        args->context);
    if (ctx == nullptr || ctx->outputs == nullptr ||
        ctx->instances == nullptr || args->valid == nullptr ||
        args->ray == nullptr || args->hit == nullptr) [[unlikely]] {
        return;
    }
    const unsigned N = args->N;
    for (unsigned i = 0u; i < N; ++i) {
        if (args->valid[i] != -1) { continue; }
        const uint32_t l = RTCRayN_id(args->ray, N, i);
        if (l >= 8u || ctx->terminated[l]) [[unlikely]] {
            args->valid[i] = 0;
            continue;
        }
        const float t = RTCRayN_tfar(args->ray, N, i);
        const uint32_t raw_inst = RTCHitN_instID(args->hit, N, i, 0u);
        const uint32_t inst = raw_inst == RTC_INVALID_GEOMETRY_ID ?
                                  RTCHitN_geomID(args->hit, N, i) :
                                  raw_inst;
        const uint32_t prim = RTCHitN_primID(args->hit, N, i);
        if (!(t >= ctx->t_min[l] && t <= ctx->t_max_current[l]) ||
            inst == RTC_INVALID_GEOMETRY_ID ||
            prim == RTC_INVALID_GEOMETRY_ID) [[unlikely]] {
            args->valid[i] = 0;
            continue;
        }
        const bool opaque = live_instance_is_opaque_fast(
            ctx->instance_flags, ctx->instance_flags_size,
            *ctx->instances, inst);
        if (!opaque &&
            ((inst == ctx->excl_inst0[l] && prim == ctx->excl_prim0[l]) ||
             (inst == ctx->excl_inst1[l] && prim == ctx->excl_prim1[l]))) {
            args->valid[i] = 0;
            continue;
        }
        const float v = (ctx->has_curves &&
                         live_instance_is_curve_fast(
                             ctx->instance_flags, ctx->instance_flags_size,
                             *ctx->instances, inst)) ?
                            -1.0f :
                            RTCHitN_v(args->hit, N, i);
        ctx->outputs->committed_inst[l] = inst;
        ctx->outputs->committed_prim[l] = prim;
        ctx->outputs->committed_bary_x[l] = RTCHitN_u(args->hit, N, i);
        ctx->outputs->committed_bary_y[l] = v;
        ctx->outputs->committed_kind[l] =
            static_cast<uint32_t>(SIMDHostRayQueryCandidateKind::surface);
        ctx->outputs->committed_t[l] = t;
        ctx->t_max_current[l] = t;
        if (ctx->query_any || (!opaque && ctx->terminate_on_commit)) {
            ctx->terminated[l] = true;
            if (!ctx->query_any) {
                args->valid[i] = 0;
                RTCRayN_tfar(args->ray, N, i) =
                    -std::numeric_limits<float>::infinity();
            }
        }
    }
}

template<uint32_t PacketWidth, typename RayHitPacket, size_t PacketAlign>
void trace_unexcluded_surface_soa_packet(
    RTCScene scene,
    const SIMDHostAccelInstanceTable &instances,
    bool has_curves,
    const int *valid,
    bool coherent,
    const SIMDHostRayQuerySurfaceExclusionInputPacket *inputs,
    SIMDHostRayQueryOutputPacket *outputs) noexcept {
    alignas(PacketAlign) RayHitPacket packet{};
    std::memcpy(packet.ray.org_x, inputs->org_x.data(), PacketWidth * sizeof(float));
    std::memcpy(packet.ray.org_y, inputs->org_y.data(), PacketWidth * sizeof(float));
    std::memcpy(packet.ray.org_z, inputs->org_z.data(), PacketWidth * sizeof(float));
    std::memcpy(packet.ray.dir_x, inputs->dir_x.data(), PacketWidth * sizeof(float));
    std::memcpy(packet.ray.dir_y, inputs->dir_y.data(), PacketWidth * sizeof(float));
    std::memcpy(packet.ray.dir_z, inputs->dir_z.data(), PacketWidth * sizeof(float));
    std::memcpy(packet.ray.time, inputs->time.data(), PacketWidth * sizeof(float));
    std::memcpy(packet.ray.tfar, outputs->t_max.data(), PacketWidth * sizeof(float));
    std::memcpy(packet.ray.mask, inputs->visibility_mask.data(), PacketWidth * sizeof(uint32_t));
    for (uint32_t l = 0u; l < PacketWidth; ++l) {
        packet.ray.id[l] = l;
        packet.hit.geomID[l] = RTC_INVALID_GEOMETRY_ID;
        packet.hit.primID[l] = RTC_INVALID_GEOMETRY_ID;
        for (uint32_t lv = 0u; lv < RTC_MAX_INSTANCE_LEVEL_COUNT; ++lv) {
            packet.hit.instID[lv][l] = RTC_INVALID_GEOMETRY_ID;
        }
    }
    RTCIntersectArguments arguments{};
    rtcInitIntersectArguments(&arguments);
    arguments.flags = coherent ? RTC_RAY_QUERY_FLAG_COHERENT :
                                 RTC_RAY_QUERY_FLAG_INCOHERENT;
    if constexpr (PacketWidth == 4u) {
        rtcIntersect4(valid, scene, &packet, &arguments);
    } else {
        rtcIntersect8(valid, scene, &packet, &arguments);
    }
    for (uint32_t l = 0u; l < PacketWidth; ++l) {
        if (valid[l] != 0 &&
            packet.hit.geomID[l] != RTC_INVALID_GEOMETRY_ID) {
            auto raw_inst = packet.hit.instID[0u][l];
            auto inst = raw_inst == RTC_INVALID_GEOMETRY_ID ?
                            packet.hit.geomID[l] :
                            raw_inst;
            float v = (has_curves && live_instance_is_curve(instances, inst)) ?
                          -1.0f :
                          packet.hit.v[l];
            outputs->committed_inst[l] = inst;
            outputs->committed_prim[l] = packet.hit.primID[l];
            outputs->committed_bary_x[l] = packet.hit.u[l];
            outputs->committed_bary_y[l] = v;
            outputs->committed_kind[l] = static_cast<uint32_t>(
                SIMDHostRayQueryCandidateKind::surface);
            outputs->committed_t[l] = packet.ray.tfar[l];
        }
    }
}

template<uint32_t PacketWidth, typename RayHitPacket, typename RayPacket, size_t PacketAlign>
void trace_filtered_surface_soa_packet(
    RTCScene scene,
    const SIMDHostAccelInstanceTable &instances,
    const uint8_t *instance_flags,
    size_t instance_flags_size,
    bool has_curves,
    const int *valid,
    bool query_any,
    bool terminate_on_commit,
    bool coherent,
    const SIMDHostRayQuerySurfaceExclusionInputPacket *inputs,
    SIMDHostRayQueryOutputPacket *outputs,
    const uint32_t *excl_inst0,
    const uint32_t *excl_prim0,
    const uint32_t *excl_inst1,
    const uint32_t *excl_prim1) noexcept {
    LiveSurfaceExclusionSoAPacketContext pctx{};
    rtcInitRayQueryContext(&pctx.rtc);
    pctx.instances = &instances;
    pctx.instance_flags = instance_flags;
    pctx.instance_flags_size = instance_flags_size;
    pctx.outputs = outputs;
    pctx.excl_inst0 = excl_inst0;
    pctx.excl_prim0 = excl_prim0;
    pctx.excl_inst1 = excl_inst1;
    pctx.excl_prim1 = excl_prim1;
    pctx.query_any = query_any;
    pctx.terminate_on_commit = terminate_on_commit;
    pctx.has_curves = has_curves;
    for (uint32_t l = 0u; l < PacketWidth; ++l) {
        pctx.t_min[l] = outputs->t_min[l];
        pctx.t_max_current[l] = outputs->t_max[l];
        pctx.terminated[l] = false;
    }
    if (query_any) {
        alignas(PacketAlign) RayPacket ray_packet{};
        std::memcpy(ray_packet.org_x, inputs->org_x.data(), PacketWidth * sizeof(float));
        std::memcpy(ray_packet.org_y, inputs->org_y.data(), PacketWidth * sizeof(float));
        std::memcpy(ray_packet.org_z, inputs->org_z.data(), PacketWidth * sizeof(float));
        std::memcpy(ray_packet.dir_x, inputs->dir_x.data(), PacketWidth * sizeof(float));
        std::memcpy(ray_packet.dir_y, inputs->dir_y.data(), PacketWidth * sizeof(float));
        std::memcpy(ray_packet.dir_z, inputs->dir_z.data(), PacketWidth * sizeof(float));
        std::memcpy(ray_packet.time, inputs->time.data(), PacketWidth * sizeof(float));
        std::memcpy(ray_packet.tfar, outputs->t_max.data(), PacketWidth * sizeof(float));
        std::memcpy(ray_packet.mask, inputs->visibility_mask.data(), PacketWidth * sizeof(uint32_t));
        for (uint32_t l = 0u; l < PacketWidth; ++l) {
            ray_packet.tnear[l] = ray_query_embree_tnear(outputs->t_min[l]);
            ray_packet.id[l] = l;
            ray_packet.flags[l] = 0u;
        }
        RTCOccludedArguments args{};
        rtcInitOccludedArguments(&args);
        args.context = &pctx.rtc;
        args.flags = static_cast<RTCRayQueryFlags>(
            (coherent ? RTC_RAY_QUERY_FLAG_COHERENT :
                        RTC_RAY_QUERY_FLAG_INCOHERENT) |
            RTC_RAY_QUERY_FLAG_INVOKE_ARGUMENT_FILTER);
        args.filter = live_surface_exclusion_soa_filter_n;
        if constexpr (PacketWidth == 4u) {
            rtcOccluded4(valid, scene, &ray_packet, &args);
        } else {
            rtcOccluded8(valid, scene, &ray_packet, &args);
        }
    } else {
        alignas(PacketAlign) RayHitPacket packet{};
        std::memcpy(packet.ray.org_x, inputs->org_x.data(), PacketWidth * sizeof(float));
        std::memcpy(packet.ray.org_y, inputs->org_y.data(), PacketWidth * sizeof(float));
        std::memcpy(packet.ray.org_z, inputs->org_z.data(), PacketWidth * sizeof(float));
        std::memcpy(packet.ray.dir_x, inputs->dir_x.data(), PacketWidth * sizeof(float));
        std::memcpy(packet.ray.dir_y, inputs->dir_y.data(), PacketWidth * sizeof(float));
        std::memcpy(packet.ray.dir_z, inputs->dir_z.data(), PacketWidth * sizeof(float));
        std::memcpy(packet.ray.time, inputs->time.data(), PacketWidth * sizeof(float));
        std::memcpy(packet.ray.tfar, outputs->t_max.data(), PacketWidth * sizeof(float));
        std::memcpy(packet.ray.mask, inputs->visibility_mask.data(), PacketWidth * sizeof(uint32_t));
        for (uint32_t l = 0u; l < PacketWidth; ++l) {
            packet.ray.tnear[l] = ray_query_embree_tnear(outputs->t_min[l]);
            packet.ray.id[l] = l;
            packet.ray.flags[l] = 0u;
            packet.hit.geomID[l] = RTC_INVALID_GEOMETRY_ID;
            packet.hit.primID[l] = RTC_INVALID_GEOMETRY_ID;
            for (uint32_t lv = 0u; lv < RTC_MAX_INSTANCE_LEVEL_COUNT; ++lv) {
                packet.hit.instID[lv][l] = RTC_INVALID_GEOMETRY_ID;
            }
        }
        RTCIntersectArguments args{};
        rtcInitIntersectArguments(&args);
        args.context = &pctx.rtc;
        args.flags = static_cast<RTCRayQueryFlags>(
            (coherent ? RTC_RAY_QUERY_FLAG_COHERENT :
                        RTC_RAY_QUERY_FLAG_INCOHERENT) |
            RTC_RAY_QUERY_FLAG_INVOKE_ARGUMENT_FILTER);
        args.filter = live_surface_exclusion_soa_filter_n;
        if constexpr (PacketWidth == 4u) {
            rtcIntersect4(valid, scene, &packet, &args);
        } else {
            rtcIntersect8(valid, scene, &packet, &args);
        }
    }
}
#endif

struct LiveSurfaceExclusionW1Context {
    RayQueryRTCContext rtc{};
    const SIMDHostAccelInstanceTable *instances{nullptr};
    const uint8_t *instance_flags{nullptr};
    size_t instance_flags_size{0u};
    SIMDHostRayQueryW1SurfaceExclusionIO *io{nullptr};
    uint32_t excl_inst0{~0u};
    uint32_t excl_prim0{~0u};
    uint32_t excl_inst1{~0u};
    uint32_t excl_prim1{~0u};
    float t_min{0.0f};
    float t_max_current{0.0f};
    bool terminate_on_commit{false};
    bool has_curves{false};
    bool terminated{false};
    bool hit_committed{false};
};
static_assert(offsetof(LiveSurfaceExclusionW1Context, rtc) == 0u);

void live_surface_exclusion_w1_intersect_filter(
    const RTCFilterFunctionNArguments *args) noexcept {
    auto *ctx =
        reinterpret_cast<LiveSurfaceExclusionW1Context *>(args->context);
    auto *ray = reinterpret_cast<RTCRay *>(args->ray);
    auto *hit = reinterpret_cast<RTCHit *>(args->hit);
    const float t = ray->tfar;
    const uint32_t raw_inst = hit->instID[0u];
    const uint32_t inst =
        raw_inst == RTC_INVALID_GEOMETRY_ID ? hit->geomID : raw_inst;
    const uint32_t prim = hit->primID;
    if (ctx->terminated ||
        !(t >= ctx->t_min && t <= ctx->t_max_current) ||
        inst == RTC_INVALID_GEOMETRY_ID ||
        prim == RTC_INVALID_GEOMETRY_ID) [[unlikely]] {
        args->valid[0u] = 0;
        return;
    }
    const uint8_t inst_flags = live_instance_flags_fast(
        ctx->instance_flags, ctx->instance_flags_size, *ctx->instances, inst);
    const bool opaque = (inst_flags & 1u) != 0u;
    if (!opaque &&
        ((inst == ctx->excl_inst0 && prim == ctx->excl_prim0) ||
         (inst == ctx->excl_inst1 && prim == ctx->excl_prim1))) {
        args->valid[0u] = 0;
        return;
    }
    ctx->t_max_current = t;
    if (!ctx->terminate_on_commit) [[likely]] {
        return;
    }
    const float v =
        (ctx->has_curves && (inst_flags & 2u) != 0u) ? -1.0f : hit->v;
    auto *io = ctx->io;
    io->committed_inst = inst;
    io->committed_prim = prim;
    io->committed_bary_x = hit->u;
    io->committed_bary_y = v;
    io->committed_kind =
        static_cast<uint32_t>(SIMDHostRayQueryCandidateKind::surface);
    io->committed_t = t;
    ctx->hit_committed = true;
    if (!opaque) {
        ctx->terminated = true;
        args->valid[0u] = 0;
        ray->tfar = -std::numeric_limits<float>::infinity();
    }
}

void live_surface_exclusion_w1_occluded_filter(
    const RTCFilterFunctionNArguments *args) noexcept {
    auto *ctx =
        reinterpret_cast<LiveSurfaceExclusionW1Context *>(args->context);
    auto *ray = reinterpret_cast<RTCRay *>(args->ray);
    auto *hit = reinterpret_cast<RTCHit *>(args->hit);
    const float t = ray->tfar;
    const uint32_t raw_inst = hit->instID[0u];
    const uint32_t inst =
        raw_inst == RTC_INVALID_GEOMETRY_ID ? hit->geomID : raw_inst;
    const uint32_t prim = hit->primID;
    if (!(t >= ctx->t_min && t <= ctx->t_max_current) ||
        inst == RTC_INVALID_GEOMETRY_ID ||
        prim == RTC_INVALID_GEOMETRY_ID) [[unlikely]] {
        args->valid[0u] = 0;
        return;
    }
    const uint8_t inst_flags = live_instance_flags_fast(
        ctx->instance_flags, ctx->instance_flags_size, *ctx->instances, inst);
    const bool opaque = (inst_flags & 1u) != 0u;
    if (!opaque &&
        ((inst == ctx->excl_inst0 && prim == ctx->excl_prim0) ||
         (inst == ctx->excl_inst1 && prim == ctx->excl_prim1))) {
        args->valid[0u] = 0;
        return;
    }
    const float v =
        (ctx->has_curves && (inst_flags & 2u) != 0u) ? -1.0f : hit->v;
    auto *io = ctx->io;
    io->committed_inst = inst;
    io->committed_prim = prim;
    io->committed_bary_x = hit->u;
    io->committed_bary_y = v;
    io->committed_kind =
        static_cast<uint32_t>(SIMDHostRayQueryCandidateKind::surface);
    io->committed_t = t;
    ctx->hit_committed = true;
}

struct LiveRayQueryW1Context {
    RayQueryRTCContext rtc{};
    const SIMDHostAccelInstanceTable *instances{nullptr};
    const uint8_t *instance_flags{nullptr};
    size_t instance_flags_size{0u};
    SIMDHostRayQueryState *state{nullptr};
    const void *capture{nullptr};
    const SIMDPacketLaunchConfig *launch_config{nullptr};
    SIMDHostRayQueryW1LiveCandidateHandler *on_candidate{nullptr};
    float t_min{0.0f};
    float t_max{0.0f};
    bool has_curves{false};
};
static_assert(offsetof(LiveRayQueryW1Context, rtc) == 0u);

void live_ray_query_intersect_filter_w1(
    const RTCFilterFunctionNArguments *args) noexcept {
    auto *ctx = reinterpret_cast<LiveRayQueryW1Context *>(args->context);
    auto *state = ctx->state;
    auto *ray = reinterpret_cast<RTCRay *>(args->ray);
    auto *hit = reinterpret_cast<RTCHit *>(args->hit);
    if (state->terminated != 0u) {
        args->valid[0u] = 0;
        ray->tfar = -std::numeric_limits<float>::infinity();
        return;
    }
    const float t = ray->tfar;
    const uint32_t raw_inst = hit->instID[0u];
    const uint32_t inst =
        raw_inst == RTC_INVALID_GEOMETRY_ID ? hit->geomID : raw_inst;
    const uint32_t prim = hit->primID;
    if (!(t >= state->world_ray[3u] && t <= state->world_ray[7u]) ||
        inst == RTC_INVALID_GEOMETRY_ID ||
        prim == RTC_INVALID_GEOMETRY_ID) [[unlikely]] {
        args->valid[0u] = 0;
        return;
    }
    const uint8_t inst_flags = live_instance_flags_fast(
        ctx->instance_flags, ctx->instance_flags_size, *ctx->instances, inst);
    const float v =
        (ctx->has_curves && (inst_flags & 2u) != 0u) ? -1.0f : hit->v;
    if ((inst_flags & 1u) != 0u) {
        state->committed = SIMDHostRayQueryCommittedHit{
            .inst = inst,
            .prim = prim,
            .bary = {hit->u, v},
            .kind = static_cast<uint32_t>(
                SIMDHostRayQueryCandidateKind::surface),
            .t = t,
        };
        state->world_ray[7u] = t;
        if (state->terminate_on_first != 0u) {
            state->terminated = 1u;
        }
        return;
    }
    state->candidate = SIMDHostRayQuerySurfaceHit{
        .inst = inst,
        .prim = prim,
        .bary = {hit->u, v},
        .t = t,
    };
    state->candidate_kind = static_cast<uint32_t>(
        SIMDHostRayQueryCandidateKind::surface);
    state->candidate_committed = 0u;
    ctx->on_candidate(
        state, ctx->capture, ctx->launch_config,
        static_cast<uint32_t>(SIMDHostRayQueryCandidateKind::surface));
    const bool committed = state->candidate_committed != 0u;
    state->candidate_kind = static_cast<uint32_t>(
        SIMDHostRayQueryCandidateKind::none);
    state->candidate_committed = 0u;
    if (committed) {
        state->committed = SIMDHostRayQueryCommittedHit{
            .inst = inst,
            .prim = prim,
            .bary = {hit->u, v},
            .kind = static_cast<uint32_t>(
                SIMDHostRayQueryCandidateKind::surface),
            .t = t,
        };
        state->world_ray[7u] = t;
        if (state->terminate_on_first != 0u) {
            state->terminated = 1u;
            return;
        }
        if (state->terminated != 0u) {
            args->valid[0u] = 0;
            ray->tfar = -std::numeric_limits<float>::infinity();
            return;
        }
        return;
    }
    if (state->terminated != 0u) {
        if (state->terminate_on_first != 0u) {
            return;
        }
        args->valid[0u] = 0;
        ray->tfar = -std::numeric_limits<float>::infinity();
        return;
    }
    args->valid[0u] = 0;
}

void live_ray_query_occluded_filter_w1(
    const RTCFilterFunctionNArguments *args) noexcept {
    auto *ctx = reinterpret_cast<LiveRayQueryW1Context *>(args->context);
    auto *state = ctx->state;
    auto *ray = reinterpret_cast<RTCRay *>(args->ray);
    auto *hit = reinterpret_cast<RTCHit *>(args->hit);
    const float t = ray->tfar;
    const uint32_t raw_inst = hit->instID[0u];
    const uint32_t inst =
        raw_inst == RTC_INVALID_GEOMETRY_ID ? hit->geomID : raw_inst;
    const uint32_t prim = hit->primID;
    if (!(t >= ctx->t_min && t <= ctx->t_max) ||
        inst == RTC_INVALID_GEOMETRY_ID ||
        prim == RTC_INVALID_GEOMETRY_ID) [[unlikely]] {
        args->valid[0u] = 0;
        return;
    }
    const uint8_t inst_flags = live_instance_flags_fast(
        ctx->instance_flags, ctx->instance_flags_size, *ctx->instances, inst);
    const float v =
        (ctx->has_curves && (inst_flags & 2u) != 0u) ? -1.0f : hit->v;
    if ((inst_flags & 1u) != 0u) {
        state->committed = SIMDHostRayQueryCommittedHit{
            .inst = inst,
            .prim = prim,
            .bary = {hit->u, v},
            .kind = static_cast<uint32_t>(
                SIMDHostRayQueryCandidateKind::surface),
            .t = t,
        };
        state->world_ray[7u] = t;
        state->terminated = 1u;
        return;
    }
    state->candidate = SIMDHostRayQuerySurfaceHit{
        .inst = inst,
        .prim = prim,
        .bary = {hit->u, v},
        .t = t,
    };
    state->candidate_kind = static_cast<uint32_t>(
        SIMDHostRayQueryCandidateKind::surface);
    state->candidate_committed = 0u;
    ctx->on_candidate(
        state, ctx->capture, ctx->launch_config,
        static_cast<uint32_t>(SIMDHostRayQueryCandidateKind::surface));
    if (state->candidate_committed != 0u) {
        state->committed = SIMDHostRayQueryCommittedHit{
            .inst = inst,
            .prim = prim,
            .bary = {hit->u, v},
            .kind = static_cast<uint32_t>(
                SIMDHostRayQueryCandidateKind::surface),
            .t = t,
        };
        state->world_ray[7u] = t;
        state->terminated = 1u;
        state->candidate_committed = 0u;
        return;
    }
    if (state->terminated != 0u) {
        return;
    }
    args->valid[0u] = 0;
}

}// namespace

void SIMDAccel::_ray_query_surface_exclusion_soa(
    uint32_t lane_count, uint64_t active_mask_bits,
    void *accel,
    const SIMDHostRayQuerySurfaceExclusionInputPacket *inputs,
    SIMDHostRayQueryOutputPacket *outputs,
    const uint32_t *excl_inst0, const uint32_t *excl_prim0,
    const uint32_t *excl_inst1, const uint32_t *excl_prim1,
    uint32_t flags) noexcept {
    if (accel == nullptr || inputs == nullptr || outputs == nullptr ||
        lane_count == 0u || lane_count > 16u) {
        return;
    }
    auto lane_mask = (uint64_t{1u} << lane_count) - 1u;
    auto remaining = active_mask_bits & lane_mask;
    if (remaining == 0u) { return; }
    auto *self = static_cast<SIMDAccel *>(accel);
    const bool terminate_on_commit =
        (flags & simd_host_ray_query_live_flag_terminate_on_commit) != 0u;
    const bool query_any =
        (flags & simd_host_ray_query_live_flag_query_any) != 0u;

#if LUISA_COMPUTE_SIMD_EMBREE_VERSION != 3
    if ((lane_count == 8u || lane_count == 4u) &&
        !self->_has_procedural_instances &&
        self->_embree_native_ray_packet) {
        const uint64_t full_mask = lane_count == 8u ? uint64_t{0xffu} : uint64_t{0xfu};
        if (remaining == full_mask && !query_any && !terminate_on_commit) {
            bool all_unexcluded = true;
            for (uint32_t l = 0u; l < lane_count; ++l) {
                if (excl_inst0[l] != RTC_INVALID_GEOMETRY_ID ||
                    excl_inst1[l] != RTC_INVALID_GEOMETRY_ID ||
                    outputs->t_min[l] != 0.0f ||
                    !(outputs->t_max[l] >= 0.0f)) {
                    all_unexcluded = false;
                    break;
                }
            }
            if (all_unexcluded) {
                static const bool coherent_primary =
                    !luisa::compute::detail::env_flag("LUISA_SIMD_INCOHERENT_PRIMARY");
                if (lane_count == 4u) {
                    alignas(16) int valid[4] = {-1, -1, -1, -1};
                    trace_unexcluded_surface_soa_packet<4u, RTCRayHit4, 16u>(
                        self->_scene, self->_instance_table,
                        self->_has_curve_instances, valid, coherent_primary,
                        inputs, outputs);
                } else {
                    alignas(32) int valid[8] = {-1, -1, -1, -1, -1, -1, -1, -1};
                    trace_unexcluded_surface_soa_packet<8u, RTCRayHit8, 32u>(
                        self->_scene, self->_instance_table,
                        self->_has_curve_instances, valid, coherent_primary,
                        inputs, outputs);
                }
                return;
            }
        }

        static const uint32_t shadow_packet_min_lanes = []() noexcept -> uint32_t {
            const char *raw = std::getenv("LUISA_SIMD_SHADOW_PACKET_MIN_LANES");
            if (raw == nullptr || *raw == '\0') { return 0u; }
            return static_cast<uint32_t>(std::strtoul(raw, nullptr, 10));
        }();
        static const uint32_t secondary_packet_min_lanes = []() noexcept -> uint32_t {
            const char *raw = std::getenv("LUISA_SIMD_SECONDARY_PACKET_MIN_LANES");
            if (raw == nullptr || *raw == '\0') { return 0u; }
            return static_cast<uint32_t>(std::strtoul(raw, nullptr, 10));
        }();
        static const bool coherent_shadow =
            luisa::compute::detail::env_flag("LUISA_SIMD_COHERENT_SHADOW_PACKET");
        static const bool coherent_secondary =
            luisa::compute::detail::env_flag("LUISA_SIMD_COHERENT_SECONDARY_PACKET");
        const uint32_t min_lanes =
            query_any ? shadow_packet_min_lanes : secondary_packet_min_lanes;
        const uint32_t active_count =
            static_cast<uint32_t>(std::popcount(remaining));
        if (min_lanes > 0u && min_lanes <= lane_count && active_count >= min_lanes) {
            alignas(32) int valid[8]{};
            uint32_t valid_count = 0u;
            bool all_active_unexcluded = !query_any && !terminate_on_commit;
            for (uint32_t l = 0u; l < lane_count; ++l) {
                if ((remaining & (uint64_t{1u} << l)) != 0u &&
                    outputs->t_max[l] >= outputs->t_min[l]) {
                    valid[l] = -1;
                    ++valid_count;
                    if (all_active_unexcluded &&
                        (excl_inst0[l] != RTC_INVALID_GEOMETRY_ID ||
                         excl_inst1[l] != RTC_INVALID_GEOMETRY_ID ||
                         outputs->t_min[l] != 0.0f)) {
                        all_active_unexcluded = false;
                    }
                } else {
                    valid[l] = 0;
                }
            }
            if (valid_count == 0u) { return; }
            if (valid_count >= min_lanes) {
                if (all_active_unexcluded) {
                    if (lane_count == 4u) {
                        trace_unexcluded_surface_soa_packet<4u, RTCRayHit4, 16u>(
                            self->_scene, self->_instance_table,
                            self->_has_curve_instances, valid, coherent_secondary,
                            inputs, outputs);
                    } else {
                        trace_unexcluded_surface_soa_packet<8u, RTCRayHit8, 32u>(
                            self->_scene, self->_instance_table,
                            self->_has_curve_instances, valid, coherent_secondary,
                            inputs, outputs);
                    }
                    return;
                }
                const bool coherent =
                    query_any ? coherent_shadow : coherent_secondary;
                const uint8_t *inst_flags =
                    self->_instance_flags_valid ? self->_instance_flags.data() : nullptr;
                const size_t inst_flags_size =
                    self->_instance_flags_valid ? self->_instance_flags_size : 0u;
                if (lane_count == 4u) {
                    trace_filtered_surface_soa_packet<4u, RTCRayHit4, RTCRay4, 16u>(
                        self->_scene, self->_instance_table,
                        inst_flags, inst_flags_size,
                        self->_has_curve_instances, valid, query_any,
                        terminate_on_commit, coherent, inputs, outputs,
                        excl_inst0, excl_prim0, excl_inst1, excl_prim1);
                } else {
                    trace_filtered_surface_soa_packet<8u, RTCRayHit8, RTCRay8, 32u>(
                        self->_scene, self->_instance_table,
                        inst_flags, inst_flags_size,
                        self->_has_curve_instances, valid, query_any,
                        terminate_on_commit, coherent, inputs, outputs,
                        excl_inst0, excl_prim0, excl_inst1, excl_prim1);
                }
                return;
            }
        }
    }
#endif

    LiveSurfaceExclusionSoAContext ctx{};
#if LUISA_COMPUTE_SIMD_EMBREE_VERSION == 3
    rtcInitIntersectContext(&ctx.rtc);
    ctx.rtc.filter = live_surface_exclusion_soa_filter;
#else
    rtcInitRayQueryContext(&ctx.rtc);
#endif
    ctx.instances = &self->_instance_table;
    ctx.instance_flags =
        self->_instance_flags_valid ? self->_instance_flags.data() : nullptr;
    ctx.instance_flags_size =
        self->_instance_flags_valid ? self->_instance_flags_size : 0u;
    ctx.outputs = outputs;
    ctx.query_any = query_any;
    ctx.terminate_on_commit = terminate_on_commit;
    ctx.has_curves = self->_has_curve_instances;
    ctx.defer_output_write = !query_any && !terminate_on_commit;

    while (remaining != 0u) {
        auto lane = static_cast<uint32_t>(std::countr_zero(remaining));
        remaining &= remaining - 1u;
        const float t_min = outputs->t_min[lane];
        const float t_max = outputs->t_max[lane];
        if (!(t_max >= t_min)) { continue; }
        const uint32_t e_i0 = excl_inst0[lane];
        const uint32_t e_p0 = excl_prim0[lane];
        const uint32_t e_i1 = excl_inst1[lane];
        const uint32_t e_p1 = excl_prim1[lane];

#if LUISA_COMPUTE_SIMD_EMBREE_VERSION != 3
        if (!query_any && !terminate_on_commit &&
            !self->_has_procedural_instances &&
            e_i0 == RTC_INVALID_GEOMETRY_ID &&
            e_i1 == RTC_INVALID_GEOMETRY_ID &&
            t_min == 0.0f) {
            RTCRayHit ray_hit{};
            ray_hit.ray.org_x = inputs->org_x[lane];
            ray_hit.ray.org_y = inputs->org_y[lane];
            ray_hit.ray.org_z = inputs->org_z[lane];
            ray_hit.ray.tnear = 0.0f;
            ray_hit.ray.dir_x = inputs->dir_x[lane];
            ray_hit.ray.dir_y = inputs->dir_y[lane];
            ray_hit.ray.dir_z = inputs->dir_z[lane];
            ray_hit.ray.time = inputs->time[lane];
            ray_hit.ray.tfar = t_max;
            ray_hit.ray.mask = inputs->visibility_mask[lane];
            ray_hit.ray.id = lane;
            ray_hit.ray.flags = 0u;
            ray_hit.hit.geomID = RTC_INVALID_GEOMETRY_ID;
            ray_hit.hit.primID = RTC_INVALID_GEOMETRY_ID;
            for (auto lv = 0u; lv < RTC_MAX_INSTANCE_LEVEL_COUNT; ++lv) {
                ray_hit.hit.instID[lv] = RTC_INVALID_GEOMETRY_ID;
            }
            rtcIntersect1(self->_scene, &ray_hit);
            if (ray_hit.hit.geomID != RTC_INVALID_GEOMETRY_ID) {
                auto raw_inst = ray_hit.hit.instID[0u];
                auto inst = raw_inst == RTC_INVALID_GEOMETRY_ID ?
                                ray_hit.hit.geomID :
                                raw_inst;
                float v = (self->_has_curve_instances &&
                           live_instance_is_curve(self->_instance_table, inst)) ?
                              -1.0f :
                              ray_hit.hit.v;
                outputs->committed_inst[lane] = inst;
                outputs->committed_prim[lane] = ray_hit.hit.primID;
                outputs->committed_bary_x[lane] = ray_hit.hit.u;
                outputs->committed_bary_y[lane] = v;
                outputs->committed_kind[lane] = static_cast<uint32_t>(
                    SIMDHostRayQueryCandidateKind::surface);
                outputs->committed_t[lane] = ray_hit.ray.tfar;
            }
            continue;
        }
#endif

        ctx.lane = lane;
        ctx.excl_inst0 = e_i0;
        ctx.excl_prim0 = e_p0;
        ctx.excl_inst1 = e_i1;
        ctx.excl_prim1 = e_p1;
        ctx.t_min = t_min;
        ctx.t_max_current = t_max;
        ctx.terminated = false;

        if (query_any) {
            RTCRay ray{};
            ray.org_x = inputs->org_x[lane];
            ray.org_y = inputs->org_y[lane];
            ray.org_z = inputs->org_z[lane];
            ray.tnear = ray_query_embree_tnear(t_min);
            ray.dir_x = inputs->dir_x[lane];
            ray.dir_y = inputs->dir_y[lane];
            ray.dir_z = inputs->dir_z[lane];
            ray.time = inputs->time[lane];
            ray.tfar = t_max;
            ray.mask = inputs->visibility_mask[lane];
            ray.id = lane;
            ray.flags = 0u;
#if LUISA_COMPUTE_SIMD_EMBREE_VERSION == 3
            rtcOccluded1(self->_scene, &ctx.rtc, &ray);
#else
            RTCOccludedArguments args{};
            rtcInitOccludedArguments(&args);
            args.context = &ctx.rtc;
            args.flags = RTC_RAY_QUERY_FLAG_INVOKE_ARGUMENT_FILTER;
            args.filter = live_surface_exclusion_soa_filter;
            rtcOccluded1(self->_scene, &ray, &args);
#endif
        } else {
            RTCRayHit ray_hit{};
            ray_hit.ray.org_x = inputs->org_x[lane];
            ray_hit.ray.org_y = inputs->org_y[lane];
            ray_hit.ray.org_z = inputs->org_z[lane];
            ray_hit.ray.tnear = ray_query_embree_tnear(t_min);
            ray_hit.ray.dir_x = inputs->dir_x[lane];
            ray_hit.ray.dir_y = inputs->dir_y[lane];
            ray_hit.ray.dir_z = inputs->dir_z[lane];
            ray_hit.ray.time = inputs->time[lane];
            ray_hit.ray.tfar = t_max;
            ray_hit.ray.mask = inputs->visibility_mask[lane];
            ray_hit.ray.id = lane;
            ray_hit.ray.flags = 0u;
            ray_hit.hit.geomID = RTC_INVALID_GEOMETRY_ID;
            ray_hit.hit.primID = RTC_INVALID_GEOMETRY_ID;
            for (auto lv = 0u; lv < RTC_MAX_INSTANCE_LEVEL_COUNT; ++lv) {
                ray_hit.hit.instID[lv] = RTC_INVALID_GEOMETRY_ID;
            }
#if LUISA_COMPUTE_SIMD_EMBREE_VERSION == 3
            rtcIntersect1(self->_scene, &ctx.rtc, &ray_hit);
#else
            RTCIntersectArguments args{};
            rtcInitIntersectArguments(&args);
            args.context = &ctx.rtc;
            args.flags = RTC_RAY_QUERY_FLAG_INVOKE_ARGUMENT_FILTER;
            args.filter = live_surface_exclusion_soa_filter;
            rtcIntersect1(self->_scene, &ray_hit, &args);
#endif
            if (ctx.defer_output_write &&
                ray_hit.hit.geomID != RTC_INVALID_GEOMETRY_ID) {
                auto raw_inst = ray_hit.hit.instID[0u];
                auto inst = raw_inst == RTC_INVALID_GEOMETRY_ID ?
                                ray_hit.hit.geomID :
                                raw_inst;
                float v = (self->_has_curve_instances &&
                           live_instance_is_curve(self->_instance_table, inst)) ?
                              -1.0f :
                              ray_hit.hit.v;
                outputs->committed_inst[lane] = inst;
                outputs->committed_prim[lane] = ray_hit.hit.primID;
                outputs->committed_bary_x[lane] = ray_hit.hit.u;
                outputs->committed_bary_y[lane] = v;
                outputs->committed_kind[lane] = static_cast<uint32_t>(
                    SIMDHostRayQueryCandidateKind::surface);
                outputs->committed_t[lane] = ray_hit.ray.tfar;
            }
        }
    }
}

void SIMDAccel::_ray_query_surface_exclusion(
    uint32_t lane_count, uint64_t active_mask_bits,
    SIMDHostRayQueryState *const *states,
    const uint32_t *excl_inst0, const uint32_t *excl_prim0,
    const uint32_t *excl_inst1, const uint32_t *excl_prim1,
    uint32_t flags) noexcept {
    if (states == nullptr || lane_count == 0u || lane_count > 64u) { return; }
    auto lane_mask = lane_count == 64u ? ~uint64_t{0u} :
                                         ((uint64_t{1u} << lane_count) - 1u);
    auto remaining = active_mask_bits & lane_mask;
    const bool terminate_on_commit =
        (flags & simd_host_ray_query_live_flag_terminate_on_commit) != 0u;

    LiveRayQueryContext ctx{};
#if LUISA_COMPUTE_SIMD_EMBREE_VERSION == 3
    rtcInitIntersectContext(&ctx.rtc);
    ctx.rtc.filter = live_surface_exclusion_filter;
#else
    rtcInitRayQueryContext(&ctx.rtc);
#endif
    ctx.states = states;
    ctx.lane_count = lane_count;
    ctx.terminate_on_commit = terminate_on_commit;

    while (remaining != 0u) {
        auto lane = static_cast<uint32_t>(std::countr_zero(remaining));
        remaining &= remaining - 1u;
        auto *state = states[lane];
        if (state == nullptr || state->accel == nullptr) { continue; }
        auto *self = static_cast<SIMDAccel *>(state->accel);
        const float t_min = state->world_ray[3u];
        const float t_max = state->world_ray[7u];
        if (!(t_max >= t_min)) {
            state->terminated = 1u;
            continue;
        }
        const uint32_t e_i0 = excl_inst0[lane];
        const uint32_t e_p0 = excl_prim0[lane];
        const uint32_t e_i1 = excl_inst1[lane];
        const uint32_t e_p1 = excl_prim1[lane];
        const bool query_any = state->terminate_on_first != 0u;

#if LUISA_COMPUTE_SIMD_EMBREE_VERSION != 3
        if (!query_any && !terminate_on_commit &&
            !self->_has_procedural_instances &&
            e_i0 == RTC_INVALID_GEOMETRY_ID &&
            e_i1 == RTC_INVALID_GEOMETRY_ID &&
            t_min == 0.0f) {
            RTCRayHit ray_hit{};
            ray_hit.ray.org_x = state->world_ray[0u];
            ray_hit.ray.org_y = state->world_ray[1u];
            ray_hit.ray.org_z = state->world_ray[2u];
            ray_hit.ray.tnear = 0.0f;
            ray_hit.ray.dir_x = state->world_ray[4u];
            ray_hit.ray.dir_y = state->world_ray[5u];
            ray_hit.ray.dir_z = state->world_ray[6u];
            ray_hit.ray.time = state->time;
            ray_hit.ray.tfar = t_max;
            ray_hit.ray.mask = state->visibility_mask;
            ray_hit.ray.id = lane;
            ray_hit.ray.flags = 0u;
            ray_hit.hit.geomID = RTC_INVALID_GEOMETRY_ID;
            ray_hit.hit.primID = RTC_INVALID_GEOMETRY_ID;
            for (auto l = 0u; l < RTC_MAX_INSTANCE_LEVEL_COUNT; ++l) {
                ray_hit.hit.instID[l] = RTC_INVALID_GEOMETRY_ID;
            }
            rtcIntersect1(self->_scene, &ray_hit);
            if (ray_hit.hit.geomID != RTC_INVALID_GEOMETRY_ID) {
                auto raw_inst = ray_hit.hit.instID[0u];
                auto inst = raw_inst == RTC_INVALID_GEOMETRY_ID ?
                                ray_hit.hit.geomID :
                                raw_inst;
                float v = (self->_has_curve_instances &&
                           live_instance_is_curve(self->_instance_table, inst)) ?
                              -1.0f :
                              ray_hit.hit.v;
                state->committed = SIMDHostRayQueryCommittedHit{
                    .inst = inst,
                    .prim = ray_hit.hit.primID,
                    .bary = {ray_hit.hit.u, v},
                    .kind = static_cast<uint32_t>(
                        SIMDHostRayQueryCandidateKind::surface),
                    .t = ray_hit.ray.tfar,
                };
                state->world_ray[7u] = ray_hit.ray.tfar;
            }
            state->terminated = 1u;
            continue;
        }
#endif

        ctx.instances = &self->_instance_table;
        ctx.instance_flags =
            self->_instance_flags_valid ? self->_instance_flags.data() : nullptr;
        ctx.instance_flags_size =
            self->_instance_flags_valid ? self->_instance_flags_size : 0u;
        ctx.state = state;
        ctx.lane = lane;
        ctx.excl_inst0 = e_i0;
        ctx.excl_prim0 = e_p0;
        ctx.excl_inst1 = e_i1;
        ctx.excl_prim1 = e_p1;
        ctx.has_curves = self->_has_curve_instances;

        if (query_any) {
            RTCRay ray{};
            ray.org_x = state->world_ray[0u];
            ray.org_y = state->world_ray[1u];
            ray.org_z = state->world_ray[2u];
            ray.tnear = ray_query_embree_tnear(t_min);
            ray.dir_x = state->world_ray[4u];
            ray.dir_y = state->world_ray[5u];
            ray.dir_z = state->world_ray[6u];
            ray.time = state->time;
            ray.tfar = t_max;
            ray.mask = state->visibility_mask;
            ray.id = lane;
            ray.flags = 0u;
#if LUISA_COMPUTE_SIMD_EMBREE_VERSION == 3
            rtcOccluded1(self->_scene, &ctx.rtc, &ray);
#else
            RTCOccludedArguments args{};
            rtcInitOccludedArguments(&args);
            args.context = &ctx.rtc;
            args.flags = RTC_RAY_QUERY_FLAG_INVOKE_ARGUMENT_FILTER;
            args.filter = live_surface_exclusion_filter;
            rtcOccluded1(self->_scene, &ray, &args);
#endif
        } else {
            RTCRayHit ray_hit{};
            ray_hit.ray.org_x = state->world_ray[0u];
            ray_hit.ray.org_y = state->world_ray[1u];
            ray_hit.ray.org_z = state->world_ray[2u];
            ray_hit.ray.tnear = ray_query_embree_tnear(t_min);
            ray_hit.ray.dir_x = state->world_ray[4u];
            ray_hit.ray.dir_y = state->world_ray[5u];
            ray_hit.ray.dir_z = state->world_ray[6u];
            ray_hit.ray.time = state->time;
            ray_hit.ray.tfar = t_max;
            ray_hit.ray.mask = state->visibility_mask;
            ray_hit.ray.id = lane;
            ray_hit.ray.flags = 0u;
            ray_hit.hit.geomID = RTC_INVALID_GEOMETRY_ID;
            ray_hit.hit.primID = RTC_INVALID_GEOMETRY_ID;
            for (auto l = 0u; l < RTC_MAX_INSTANCE_LEVEL_COUNT; ++l) {
                ray_hit.hit.instID[l] = RTC_INVALID_GEOMETRY_ID;
            }
#if LUISA_COMPUTE_SIMD_EMBREE_VERSION == 3
            rtcIntersect1(self->_scene, &ctx.rtc, &ray_hit);
#else
            RTCIntersectArguments args{};
            rtcInitIntersectArguments(&args);
            args.context = &ctx.rtc;
            args.flags = RTC_RAY_QUERY_FLAG_INVOKE_ARGUMENT_FILTER;
            args.filter = live_surface_exclusion_filter;
            rtcIntersect1(self->_scene, &ray_hit, &args);
#endif
        }
        state->terminated = 1u;
    }
}

void SIMDAccel::_ray_query_live_pipeline(
    uint32_t lane_count, uint64_t active_mask_bits,
    SIMDHostRayQueryState *const *states, const void *capture,
    const SIMDPacketLaunchConfig *launch_config,
    SIMDHostRayQueryLiveCandidateHandler *on_candidate,
    uint32_t flags) noexcept {
    if (states == nullptr || on_candidate == nullptr ||
        lane_count == 0u || lane_count > 64u) {
        return;
    }
    static_cast<void>(flags);

    auto trace_single_live_ray = [&](SIMDAccel *self,
                                     LiveRayQueryContext &ctx,
                                     SIMDHostRayQueryState *state,
                                     uint32_t lane,
                                     float t_min,
                                     float t_max) noexcept {
        ctx.t_min = t_min;
        ctx.t_max = t_max;
        if (state->terminate_on_first != 0u) {
            state->candidate_kind = static_cast<uint32_t>(
                SIMDHostRayQueryCandidateKind::surface);
            RTCRay ray{};
            ray.org_x = state->world_ray[0u];
            ray.org_y = state->world_ray[1u];
            ray.org_z = state->world_ray[2u];
            ray.tnear = ray_query_embree_tnear(t_min);
            ray.dir_x = state->world_ray[4u];
            ray.dir_y = state->world_ray[5u];
            ray.dir_z = state->world_ray[6u];
            ray.time = state->time;
            ray.tfar = t_max;
            ray.mask = state->visibility_mask;
            ray.id = lane;
            ray.flags = 0u;
#if LUISA_COMPUTE_SIMD_EMBREE_VERSION == 3
            ctx.rtc.filter = live_ray_query_occluded_filter;
            rtcOccluded1(self->_scene, &ctx.rtc, &ray);
#else
            RTCOccludedArguments args{};
            rtcInitOccludedArguments(&args);
            args.context = &ctx.rtc;
            args.flags = RTC_RAY_QUERY_FLAG_INVOKE_ARGUMENT_FILTER;
            args.filter = live_ray_query_occluded_filter;
            rtcOccluded1(self->_scene, &ray, &args);
#endif
            state->candidate_kind = static_cast<uint32_t>(
                SIMDHostRayQueryCandidateKind::none);
            state->candidate_committed = 0u;
        } else {
            RTCRayHit ray_hit{};
            ray_hit.ray.org_x = state->world_ray[0u];
            ray_hit.ray.org_y = state->world_ray[1u];
            ray_hit.ray.org_z = state->world_ray[2u];
            ray_hit.ray.tnear = ray_query_embree_tnear(t_min);
            ray_hit.ray.dir_x = state->world_ray[4u];
            ray_hit.ray.dir_y = state->world_ray[5u];
            ray_hit.ray.dir_z = state->world_ray[6u];
            ray_hit.ray.time = state->time;
            ray_hit.ray.tfar = t_max;
            ray_hit.ray.mask = state->visibility_mask;
            ray_hit.ray.id = lane;
            ray_hit.ray.flags = 0u;
            ray_hit.hit.geomID = RTC_INVALID_GEOMETRY_ID;
            ray_hit.hit.primID = RTC_INVALID_GEOMETRY_ID;
            for (auto l = 0u; l < RTC_MAX_INSTANCE_LEVEL_COUNT; ++l) {
                ray_hit.hit.instID[l] = RTC_INVALID_GEOMETRY_ID;
            }
#if LUISA_COMPUTE_SIMD_EMBREE_VERSION == 3
            ctx.rtc.filter = live_ray_query_filter;
            rtcIntersect1(self->_scene, &ctx.rtc, &ray_hit);
#else
            RTCIntersectArguments args{};
            rtcInitIntersectArguments(&args);
            args.context = &ctx.rtc;
            args.flags = RTC_RAY_QUERY_FLAG_INVOKE_ARGUMENT_FILTER;
            args.filter = live_ray_query_filter;
            rtcIntersect1(self->_scene, &ray_hit, &args);
#endif
        }
        state->terminated = 1u;
    };

    if (lane_count == 1u) {
        if ((active_mask_bits & 1u) == 0u) { return; }
        auto *state = states[0u];
        if (state == nullptr || state->accel == nullptr) { return; }
        const float t_min = state->world_ray[3u];
        const float t_max = state->world_ray[7u];
        if (!(t_max >= t_min)) {
            state->terminated = 1u;
            return;
        }
        auto *self = static_cast<SIMDAccel *>(state->accel);
        LiveRayQueryContext ctx;
#if LUISA_COMPUTE_SIMD_EMBREE_VERSION == 3
        rtcInitIntersectContext(&ctx.rtc);
        ctx.rtc.filter = live_ray_query_filter;
#else
        rtcInitRayQueryContext(&ctx.rtc);
#endif
        ctx.instances = &self->_instance_table;
        ctx.instance_flags =
            self->_instance_flags_valid ? self->_instance_flags.data() : nullptr;
        ctx.instance_flags_size =
            self->_instance_flags_valid ? self->_instance_flags_size : 0u;
        ctx.states = states;
        ctx.state = state;
        ctx.capture = capture;
        ctx.launch_config = launch_config;
        ctx.on_candidate = on_candidate;
        ctx.lane_count = 1u;
        ctx.lane = 0u;
        ctx.has_curves = self->_has_curve_instances;
        if (self->_has_procedural_instances) [[unlikely]] {
            ScopedLiveRayQueryContext active_live{ctx};
            trace_single_live_ray(self, ctx, state, 0u, t_min, t_max);
        } else {
            trace_single_live_ray(self, ctx, state, 0u, t_min, t_max);
        }
        return;
    }

    auto lane_mask = lane_count == 64u ? ~uint64_t{0u} :
                                         ((uint64_t{1u} << lane_count) - 1u);
    auto remaining = active_mask_bits & lane_mask;

    LiveRayQueryContext ctx{};
#if LUISA_COMPUTE_SIMD_EMBREE_VERSION == 3
    rtcInitIntersectContext(&ctx.rtc);
    ctx.rtc.filter = live_ray_query_filter;
#else
    rtcInitRayQueryContext(&ctx.rtc);
#endif
    ctx.states = states;
    ctx.capture = capture;
    ctx.launch_config = launch_config;
    ctx.on_candidate = on_candidate;
    ctx.lane_count = lane_count;
    std::optional<ScopedLiveRayQueryContext> active_live;

    while (remaining != 0u) {
        auto lane = static_cast<uint32_t>(std::countr_zero(remaining));
        remaining &= remaining - 1u;
        auto *state = states[lane];
        if (state == nullptr || state->accel == nullptr) { continue; }
        auto *self = static_cast<SIMDAccel *>(state->accel);
        const float t_min = state->world_ray[3u];
        const float t_max = state->world_ray[7u];
        if (!(t_max >= t_min)) {
            state->terminated = 1u;
            continue;
        }
        ctx.instances = &self->_instance_table;
        ctx.instance_flags =
            self->_instance_flags_valid ? self->_instance_flags.data() : nullptr;
        ctx.instance_flags_size =
            self->_instance_flags_valid ? self->_instance_flags_size : 0u;
        ctx.state = state;
        ctx.lane = lane;
        ctx.has_curves = self->_has_curve_instances;
        if (self->_has_procedural_instances && !active_live.has_value()) {
            active_live.emplace(ctx);
        }
        trace_single_live_ray(self, ctx, state, lane, t_min, t_max);
    }
}

void SIMDAccel::_ray_query_surface_exclusion_w1(
    void *accel,
    SIMDHostRayQueryW1SurfaceExclusionIO *io,
    uint32_t excl_inst0, uint32_t excl_prim0,
    uint32_t excl_inst1, uint32_t excl_prim1,
    uint32_t flags) noexcept {
    if (accel == nullptr || io == nullptr) { return; }
    auto write_miss = [](SIMDHostRayQueryW1SurfaceExclusionIO *out) noexcept {
        out->committed_inst = ~uint32_t{0u};
        out->committed_prim = ~uint32_t{0u};
        out->committed_bary_x = 0.0f;
        out->committed_bary_y = 0.0f;
        out->committed_kind =
            static_cast<uint32_t>(SIMDHostRayQueryCandidateKind::none);
        out->committed_t = 0.0f;
    };
    const float t_min = io->t_min;
    const float t_max = io->t_max;
    if (!(t_max >= t_min)) {
        write_miss(io);
        return;
    }
    auto *self = static_cast<SIMDAccel *>(accel);
    const bool terminate_on_commit =
        (flags & simd_host_ray_query_live_flag_terminate_on_commit) != 0u;
    const bool query_any =
        (flags & simd_host_ray_query_live_flag_query_any) != 0u;

#if LUISA_COMPUTE_SIMD_EMBREE_VERSION != 3
    if (!query_any && !terminate_on_commit &&
        !self->_has_procedural_instances &&
        excl_inst0 == RTC_INVALID_GEOMETRY_ID &&
        excl_inst1 == RTC_INVALID_GEOMETRY_ID &&
        t_min == 0.0f) {
        RTCRayHit ray_hit;
        ray_hit.ray.org_x = io->org_x;
        ray_hit.ray.org_y = io->org_y;
        ray_hit.ray.org_z = io->org_z;
        ray_hit.ray.tnear = 0.0f;
        ray_hit.ray.dir_x = io->dir_x;
        ray_hit.ray.dir_y = io->dir_y;
        ray_hit.ray.dir_z = io->dir_z;
        ray_hit.ray.time = io->time;
        ray_hit.ray.tfar = t_max;
        ray_hit.ray.mask = io->visibility_mask;
        ray_hit.ray.id = 0u;
        ray_hit.ray.flags = 0u;
        ray_hit.hit.geomID = RTC_INVALID_GEOMETRY_ID;
        ray_hit.hit.primID = RTC_INVALID_GEOMETRY_ID;
        for (auto lv = 0u; lv < RTC_MAX_INSTANCE_LEVEL_COUNT; ++lv) {
            ray_hit.hit.instID[lv] = RTC_INVALID_GEOMETRY_ID;
        }
        rtcIntersect1(self->_scene, &ray_hit);
        if (ray_hit.hit.geomID != RTC_INVALID_GEOMETRY_ID) {
            const uint32_t raw_inst = ray_hit.hit.instID[0u];
            const uint32_t inst =
                raw_inst == RTC_INVALID_GEOMETRY_ID ? ray_hit.hit.geomID :
                                                      raw_inst;
            const float v =
                (self->_has_curve_instances &&
                 live_instance_is_curve_fast(
                     self->_instance_flags_valid ? self->_instance_flags.data() :
                                                   nullptr,
                     self->_instance_flags_valid ? self->_instance_flags_size :
                                                   0u,
                     self->_instance_table, inst)) ?
                    -1.0f :
                    ray_hit.hit.v;
            io->committed_inst = inst;
            io->committed_prim = ray_hit.hit.primID;
            io->committed_bary_x = ray_hit.hit.u;
            io->committed_bary_y = v;
            io->committed_kind =
                static_cast<uint32_t>(SIMDHostRayQueryCandidateKind::surface);
            io->committed_t = ray_hit.ray.tfar;
        } else {
            write_miss(io);
        }
        return;
    }
#endif

    LiveSurfaceExclusionW1Context ctx{};
#if LUISA_COMPUTE_SIMD_EMBREE_VERSION == 3
    rtcInitIntersectContext(&ctx.rtc);
#else
    rtcInitRayQueryContext(&ctx.rtc);
#endif
    ctx.instances = &self->_instance_table;
    ctx.instance_flags =
        self->_instance_flags_valid ? self->_instance_flags.data() : nullptr;
    ctx.instance_flags_size =
        self->_instance_flags_valid ? self->_instance_flags_size : 0u;
    ctx.io = io;
    ctx.excl_inst0 = excl_inst0;
    ctx.excl_prim0 = excl_prim0;
    ctx.excl_inst1 = excl_inst1;
    ctx.excl_prim1 = excl_prim1;
    ctx.t_min = t_min;
    ctx.t_max_current = t_max;
    ctx.terminate_on_commit = terminate_on_commit;
    ctx.has_curves = self->_has_curve_instances;

    if (query_any) {
        RTCRay ray;
        ray.org_x = io->org_x;
        ray.org_y = io->org_y;
        ray.org_z = io->org_z;
        ray.tnear = ray_query_embree_tnear(t_min);
        ray.dir_x = io->dir_x;
        ray.dir_y = io->dir_y;
        ray.dir_z = io->dir_z;
        ray.time = io->time;
        ray.tfar = t_max;
        ray.mask = io->visibility_mask;
        ray.id = 0u;
        ray.flags = 0u;
#if LUISA_COMPUTE_SIMD_EMBREE_VERSION == 3
        ctx.rtc.filter = live_surface_exclusion_w1_occluded_filter;
        rtcOccluded1(self->_scene, &ctx.rtc, &ray);
#else
        RTCOccludedArguments args{};
        rtcInitOccludedArguments(&args);
        args.context = &ctx.rtc;
        args.flags = RTC_RAY_QUERY_FLAG_INVOKE_ARGUMENT_FILTER;
        args.filter = live_surface_exclusion_w1_occluded_filter;
        rtcOccluded1(self->_scene, &ray, &args);
#endif
        if (!ctx.hit_committed) {
            write_miss(io);
        }
        return;
    }

    RTCRayHit ray_hit;
    ray_hit.ray.org_x = io->org_x;
    ray_hit.ray.org_y = io->org_y;
    ray_hit.ray.org_z = io->org_z;
    ray_hit.ray.tnear = ray_query_embree_tnear(t_min);
    ray_hit.ray.dir_x = io->dir_x;
    ray_hit.ray.dir_y = io->dir_y;
    ray_hit.ray.dir_z = io->dir_z;
    ray_hit.ray.time = io->time;
    ray_hit.ray.tfar = t_max;
    ray_hit.ray.mask = io->visibility_mask;
    ray_hit.ray.id = 0u;
    ray_hit.ray.flags = 0u;
    ray_hit.hit.geomID = RTC_INVALID_GEOMETRY_ID;
    ray_hit.hit.primID = RTC_INVALID_GEOMETRY_ID;
    for (auto lv = 0u; lv < RTC_MAX_INSTANCE_LEVEL_COUNT; ++lv) {
        ray_hit.hit.instID[lv] = RTC_INVALID_GEOMETRY_ID;
    }
#if LUISA_COMPUTE_SIMD_EMBREE_VERSION == 3
    ctx.rtc.filter = live_surface_exclusion_w1_intersect_filter;
    rtcIntersect1(self->_scene, &ctx.rtc, &ray_hit);
#else
    RTCIntersectArguments args{};
    rtcInitIntersectArguments(&args);
    args.context = &ctx.rtc;
    args.flags = RTC_RAY_QUERY_FLAG_INVOKE_ARGUMENT_FILTER;
    args.filter = live_surface_exclusion_w1_intersect_filter;
    rtcIntersect1(self->_scene, &ray_hit, &args);
#endif
    if (!terminate_on_commit) [[likely]] {
        if (ray_hit.hit.geomID != RTC_INVALID_GEOMETRY_ID) {
            const uint32_t raw_inst = ray_hit.hit.instID[0u];
            const uint32_t inst =
                raw_inst == RTC_INVALID_GEOMETRY_ID ? ray_hit.hit.geomID :
                                                      raw_inst;
            const float v =
                (self->_has_curve_instances &&
                 live_instance_is_curve_fast(
                     ctx.instance_flags, ctx.instance_flags_size,
                     self->_instance_table, inst)) ?
                    -1.0f :
                    ray_hit.hit.v;
            io->committed_inst = inst;
            io->committed_prim = ray_hit.hit.primID;
            io->committed_bary_x = ray_hit.hit.u;
            io->committed_bary_y = v;
            io->committed_kind =
                static_cast<uint32_t>(SIMDHostRayQueryCandidateKind::surface);
            io->committed_t = ray_hit.ray.tfar;
        } else {
            write_miss(io);
        }
    } else if (!ctx.hit_committed) {
        write_miss(io);
    }
}

void SIMDAccel::_ray_query_live_pipeline_w1(
    SIMDHostRayQueryState *state, const void *capture,
    const SIMDPacketLaunchConfig *launch_config,
    SIMDHostRayQueryW1LiveCandidateHandler *on_candidate,
    uint32_t flags) noexcept {
    if (state == nullptr || state->accel == nullptr ||
        on_candidate == nullptr) {
        return;
    }
    static_cast<void>(flags);
    const float t_min = state->world_ray[3u];
    const float t_max = state->world_ray[7u];
    if (!(t_max >= t_min)) {
        state->terminated = 1u;
        return;
    }
    auto *self = static_cast<SIMDAccel *>(state->accel);
    const uint8_t *inst_flags =
        self->_instance_flags_valid ? self->_instance_flags.data() : nullptr;
    const size_t inst_flags_size =
        self->_instance_flags_valid ? self->_instance_flags_size : 0u;

    if (self->_has_procedural_instances) [[unlikely]] {
        LiveRayQueryContext ctx{};
#if LUISA_COMPUTE_SIMD_EMBREE_VERSION == 3
        rtcInitIntersectContext(&ctx.rtc);
#else
        rtcInitRayQueryContext(&ctx.rtc);
#endif
        ctx.instances = &self->_instance_table;
        ctx.instance_flags = inst_flags;
        ctx.instance_flags_size = inst_flags_size;
        ctx.state = state;
        ctx.capture = capture;
        ctx.launch_config = launch_config;
        ctx.w1_on_candidate = on_candidate;
        ctx.lane_count = 1u;
        ctx.lane = 0u;
        ctx.t_min = t_min;
        ctx.t_max = t_max;
        ctx.has_curves = self->_has_curve_instances;
        ScopedLiveRayQueryContext active_live{ctx};
        if (state->terminate_on_first != 0u) {
            state->candidate_kind = static_cast<uint32_t>(
                SIMDHostRayQueryCandidateKind::surface);
            RTCRay ray;
            ray.org_x = state->world_ray[0u];
            ray.org_y = state->world_ray[1u];
            ray.org_z = state->world_ray[2u];
            ray.tnear = ray_query_embree_tnear(t_min);
            ray.dir_x = state->world_ray[4u];
            ray.dir_y = state->world_ray[5u];
            ray.dir_z = state->world_ray[6u];
            ray.time = state->time;
            ray.tfar = t_max;
            ray.mask = state->visibility_mask;
            ray.id = 0u;
            ray.flags = 0u;
#if LUISA_COMPUTE_SIMD_EMBREE_VERSION == 3
            ctx.rtc.filter = live_ray_query_occluded_filter;
            rtcOccluded1(self->_scene, &ctx.rtc, &ray);
#else
            RTCOccludedArguments args{};
            rtcInitOccludedArguments(&args);
            args.context = &ctx.rtc;
            args.flags = RTC_RAY_QUERY_FLAG_INVOKE_ARGUMENT_FILTER;
            args.filter = live_ray_query_occluded_filter;
            rtcOccluded1(self->_scene, &ray, &args);
#endif
            state->candidate_kind = static_cast<uint32_t>(
                SIMDHostRayQueryCandidateKind::none);
            state->candidate_committed = 0u;
        } else {
            RTCRayHit ray_hit;
            ray_hit.ray.org_x = state->world_ray[0u];
            ray_hit.ray.org_y = state->world_ray[1u];
            ray_hit.ray.org_z = state->world_ray[2u];
            ray_hit.ray.tnear = ray_query_embree_tnear(t_min);
            ray_hit.ray.dir_x = state->world_ray[4u];
            ray_hit.ray.dir_y = state->world_ray[5u];
            ray_hit.ray.dir_z = state->world_ray[6u];
            ray_hit.ray.time = state->time;
            ray_hit.ray.tfar = t_max;
            ray_hit.ray.mask = state->visibility_mask;
            ray_hit.ray.id = 0u;
            ray_hit.ray.flags = 0u;
            ray_hit.hit.geomID = RTC_INVALID_GEOMETRY_ID;
            ray_hit.hit.primID = RTC_INVALID_GEOMETRY_ID;
            for (auto l = 0u; l < RTC_MAX_INSTANCE_LEVEL_COUNT; ++l) {
                ray_hit.hit.instID[l] = RTC_INVALID_GEOMETRY_ID;
            }
#if LUISA_COMPUTE_SIMD_EMBREE_VERSION == 3
            ctx.rtc.filter = live_ray_query_filter;
            rtcIntersect1(self->_scene, &ctx.rtc, &ray_hit);
#else
            RTCIntersectArguments args{};
            rtcInitIntersectArguments(&args);
            args.context = &ctx.rtc;
            args.flags = RTC_RAY_QUERY_FLAG_INVOKE_ARGUMENT_FILTER;
            args.filter = live_ray_query_filter;
            rtcIntersect1(self->_scene, &ray_hit, &args);
#endif
        }
        state->terminated = 1u;
        return;
    }

    LiveRayQueryW1Context ctx{};
#if LUISA_COMPUTE_SIMD_EMBREE_VERSION == 3
    rtcInitIntersectContext(&ctx.rtc);
#else
    rtcInitRayQueryContext(&ctx.rtc);
#endif
    ctx.instances = &self->_instance_table;
    ctx.instance_flags = inst_flags;
    ctx.instance_flags_size = inst_flags_size;
    ctx.state = state;
    ctx.capture = capture;
    ctx.launch_config = launch_config;
    ctx.on_candidate = on_candidate;
    ctx.t_min = t_min;
    ctx.t_max = t_max;
    ctx.has_curves = self->_has_curve_instances;

    if (state->terminate_on_first != 0u) {
        state->candidate_kind = static_cast<uint32_t>(
            SIMDHostRayQueryCandidateKind::surface);
        RTCRay ray;
        ray.org_x = state->world_ray[0u];
        ray.org_y = state->world_ray[1u];
        ray.org_z = state->world_ray[2u];
        ray.tnear = ray_query_embree_tnear(t_min);
        ray.dir_x = state->world_ray[4u];
        ray.dir_y = state->world_ray[5u];
        ray.dir_z = state->world_ray[6u];
        ray.time = state->time;
        ray.tfar = t_max;
        ray.mask = state->visibility_mask;
        ray.id = 0u;
        ray.flags = 0u;
#if LUISA_COMPUTE_SIMD_EMBREE_VERSION == 3
        ctx.rtc.filter = live_ray_query_occluded_filter_w1;
        rtcOccluded1(self->_scene, &ctx.rtc, &ray);
#else
        RTCOccludedArguments args{};
        rtcInitOccludedArguments(&args);
        args.context = &ctx.rtc;
        args.flags = RTC_RAY_QUERY_FLAG_INVOKE_ARGUMENT_FILTER;
        args.filter = live_ray_query_occluded_filter_w1;
        rtcOccluded1(self->_scene, &ray, &args);
#endif
        state->candidate_kind = static_cast<uint32_t>(
            SIMDHostRayQueryCandidateKind::none);
        state->candidate_committed = 0u;
    } else {
        RTCRayHit ray_hit;
        ray_hit.ray.org_x = state->world_ray[0u];
        ray_hit.ray.org_y = state->world_ray[1u];
        ray_hit.ray.org_z = state->world_ray[2u];
        ray_hit.ray.tnear = ray_query_embree_tnear(t_min);
        ray_hit.ray.dir_x = state->world_ray[4u];
        ray_hit.ray.dir_y = state->world_ray[5u];
        ray_hit.ray.dir_z = state->world_ray[6u];
        ray_hit.ray.time = state->time;
        ray_hit.ray.tfar = t_max;
        ray_hit.ray.mask = state->visibility_mask;
        ray_hit.ray.id = 0u;
        ray_hit.ray.flags = 0u;
        ray_hit.hit.geomID = RTC_INVALID_GEOMETRY_ID;
        ray_hit.hit.primID = RTC_INVALID_GEOMETRY_ID;
        for (auto l = 0u; l < RTC_MAX_INSTANCE_LEVEL_COUNT; ++l) {
            ray_hit.hit.instID[l] = RTC_INVALID_GEOMETRY_ID;
        }
#if LUISA_COMPUTE_SIMD_EMBREE_VERSION == 3
        ctx.rtc.filter = live_ray_query_intersect_filter_w1;
        rtcIntersect1(self->_scene, &ctx.rtc, &ray_hit);
#else
        RTCIntersectArguments args{};
        rtcInitIntersectArguments(&args);
        args.context = &ctx.rtc;
        args.flags = RTC_RAY_QUERY_FLAG_INVOKE_ARGUMENT_FILTER;
        args.filter = live_ray_query_intersect_filter_w1;
        rtcIntersect1(self->_scene, &ray_hit, &args);
#endif
    }
    state->terminated = 1u;
}

uint64_t simd_host_ray_query_proceed_wide_procedural_fused_status(
    uint32_t lane_count, uint64_t active_mask_bits,
    SIMDHostRayQueryState *const *states) noexcept {
    LUISA_ASSERT(
        states != nullptr && lane_count == 16u,
        "Invalid fused procedural SIMD ray-query packet width {}.",
        lane_count);
    auto lane_mask = (uint64_t{1u} << lane_count) - 1u;
    active_mask_bits &= lane_mask;
    if (active_mask_bits == 0u) { return 0u; }
    auto fully_active = active_mask_bits == lane_mask;
    auto pending = uint64_t{0u};
    auto status = uint64_t{0u};
    if (fully_active) [[likely]] {
        for (auto lane = 0u; lane < lane_count; lane++) {
            auto bit = uint64_t{1u} << lane;
            SIMDAccel::_ray_query_proceed_wide_lane(
                states[lane], lane, bit, pending);
            if ((pending & bit) == 0u) {
                status |= ray_query_lane_status(*states[lane], bit);
            }
        }
    } else {
        auto remaining_active = active_mask_bits;
        while (remaining_active != 0u) {
            auto lane = static_cast<uint32_t>(
                std::countr_zero(remaining_active));
            auto bit = uint64_t{1u} << lane;
            remaining_active &= remaining_active - 1u;
            SIMDAccel::_ray_query_proceed_wide_lane(
                states[lane], lane, bit, pending);
            if ((pending & bit) == 0u) {
                status |= ray_query_lane_status(*states[lane], bit);
            }
        }
    }
    while (pending != 0u) {
        auto first_lane = fully_active ? uint32_t{0u} :
                                         static_cast<uint32_t>(
                                             std::countr_zero(pending));
        if (fully_active) [[likely]] {
            while (((pending >> first_lane) & 1u) == 0u) {
                first_lane++;
            }
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
        LUISA_ASSERT(
            group != 0u,
            "Empty fused wide SIMD ray-query packet group.");
        status |= scan_ray_query_packet_wide_status<
            16u, RTCRayHit16, RTCRay16>(
            self->_scene, self->_instance_table,
            lane_count, group, states,
            terminate_on_first);
        pending &= ~group;
    }
    return status;
}

}// namespace luisa::compute::simd
