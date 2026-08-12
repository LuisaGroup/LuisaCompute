#include "simd_accel.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <limits>

#include <luisa/core/logging.h>

#include "simd_motion_instance.h"

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

void mark_curve_surface_hits(
    const SIMDHostAccelInstanceTable &instances,
    uint32_t lane_count, uint64_t active_mask_bits,
    const uint32_t *hit_ids, float *hit_values) noexcept {
    for (auto lane = 0u; lane < lane_count; lane++) {
        if (!lane_active(active_mask_bits, lane, lane_count)) { continue; }
        auto inst = hit_ids[lane];
        if (inst == RTC_INVALID_GEOMETRY_ID) { continue; }
        LUISA_ASSERT(
            instances.data != nullptr && inst < instances.size,
            "SIMD curve trace returned an invalid instance ID {}.", inst);
        if (instances.data[inst].curve != 0u) {
            hit_values[lane_count + lane] = -1.0f;
        }
    }
}

struct RayQueryBatchBuildState {
    bool heapified{false};
    bool ascending{true};
    bool descending{true};
};

#if LUISA_COMPUTE_SIMD_EMBREE_VERSION == 3
using RayQueryRTCContext = RTCIntersectContext;
#else
using RayQueryRTCContext = RTCRayQueryContext;
#endif

struct RayQueryScanContext {
    RayQueryRTCContext rtc{};
    uint32_t lane_count{0u};
    const SIMDHostAccelInstanceTable *instances{nullptr};
    std::array<SIMDHostRayQueryState *, 16u> states{};
    std::array<RayQueryBatchBuildState, 16u> batch_build{};
};
static_assert(offsetof(RayQueryScanContext, rtc) == 0u);

[[nodiscard]] float ray_query_embree_tnear(float tnear) noexcept {
    if (!std::isfinite(tnear)) { return tnear; }
    auto widened = std::nextafter(
        tnear, -std::numeric_limits<float>::infinity());
    if (std::abs(widened) < std::numeric_limits<float>::min()) {
        widened = -std::numeric_limits<float>::min();
    }
    return widened;
}

[[nodiscard]] bool ray_query_key_after_cursor(
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
        instances.data[inst].curve != 0u) {
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
                     context->instances->data[inst].curve != 0u;
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
    }
#if LUISA_COMPUTE_SIMD_EMBREE_VERSION == 3
    rtcInitIntersectContext(&context.rtc);
    context.rtc.filter = ray_query_filter;
#else
    rtcInitRayQueryContext(&context.rtc);
#endif
}

void initialize_ray_query_inputs(
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
    if (state.candidate_batch_initialized == 0u) {
        return RayQueryCandidateAdvance::needs_scan;
    }
    constexpr auto capacity =
        simd_host_ray_query_candidate_batch_capacity;
    LUISA_ASSERT(
        state.candidate_batch_count <= capacity &&
            state.candidate_batch_index <= state.candidate_batch_count,
        "SIMD ray-query candidate batch metadata is invalid.");
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
    if (state.candidate_batch_has_more != 0u) {
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
        state->candidate_batch_index = 0u;
        state->candidate_batch_initialized = 1u;
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
                instance.curve =
                    motion->child()->kind() == SIMDPrimitive::Kind::curve ?
                        1u :
                        0u;
                _motion_states[modification.index] = std::move(state);
            } else {
                LUISA_ASSERT(
                    primitive->kind() == SIMDPrimitive::Kind::mesh ||
                        primitive->kind() == SIMDPrimitive::Kind::curve,
                    "SIMD accel instances currently require a mesh, curve, "
                    "or motion-instance primitive.");
                rtcSetGeometryInstancedScene(geometry, primitive->handle());
                rtcSetGeometryTimeStepCount(geometry, 1u);
                rtcSetGeometryTimeRange(geometry, 0.0f, 1.0f);
                instance.curve =
                    primitive->kind() == SIMDPrimitive::Kind::curve ?
                        1u :
                        0u;
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
        if (!instance.dirty) { continue; }
        auto geometry = _geometries[i];
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
                    "SIMD SRT motion instances currently require an identity "
                    "outer affine transform.");
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
    mark_curve_surface_hits(
        self->_instance_table, lane_count, active_mask_bits,
        hit_ids, hit_values);
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

}// namespace luisa::compute::simd
