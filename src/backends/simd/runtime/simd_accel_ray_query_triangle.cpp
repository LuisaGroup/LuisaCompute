#include "simd_accel.h"
#include "simd_accel_ray_query.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cstring>
#include <cstdlib>
#include <limits>

#include <luisa/core/logging.h>

#include "../../common/env_flag.h"

namespace luisa::compute::simd::triangle_ray_query {

using RayQueryRTCContext = detail::RayQueryRTCContext;

struct RayQueryBatchBuildState {
    bool heapified;
    bool ascending;
    bool descending;
};

struct RayQueryScanContext {
    RayQueryRTCContext rtc;
    uint32_t lane_count;
    std::array<SIMDHostRayQueryState *, 16u> states;
    std::array<RayQueryBatchBuildState, 16u> batch_build;
};
static_assert(offsetof(RayQueryScanContext, rtc) == 0u);

struct SIMDAccelAccess {
    [[nodiscard]] static RTCScene scene(SIMDAccel &accel) noexcept {
        return accel._scene;
    }
    [[nodiscard]] static const SIMDHostAccelInstanceTable &instances(
        SIMDAccel &accel) noexcept {
        return accel._instance_table;
    }
};
namespace {

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

LUISA_NEVER_INLINE void ray_query_insert_triangle_candidate_overflow(
    SIMDHostRayQueryState &state,
    RayQueryBatchBuildState &build,
    SIMDHostRayQuerySurfaceHit candidate) noexcept {
    constexpr auto capacity =
        simd_host_ray_query_candidate_batch_capacity;
    LUISA_ASSERT(
        state.candidate_batch_count == capacity,
        "Triangle-only SIMD ray-query overflow helper received a partial batch.");
    state.candidate_batch_has_more = 1u;
    auto begin = std::begin(state.candidate_batch);
    auto end = begin + state.candidate_batch_count;
    if (!build.heapified) {
        std::make_heap(begin, end, ray_query_candidate_before);
        build.heapified = true;
        build.ascending = false;
        build.descending = false;
    }
    if (!ray_query_candidate_before(
            candidate, state.candidate_batch[0u])) {
        return;
    }
    std::pop_heap(begin, end, ray_query_candidate_before);
    state.candidate_batch[state.candidate_batch_count - 1u] = candidate;
    std::push_heap(begin, end, ray_query_candidate_before);
}

LUISA_FORCE_INLINE void ray_query_insert_triangle_candidate(
    SIMDHostRayQueryState &state,
    RayQueryBatchBuildState &build,
    SIMDHostRayQuerySurfaceHit candidate) noexcept {
    constexpr auto capacity =
        simd_host_ray_query_candidate_batch_capacity;
    if (state.candidate_batch_count < capacity) [[likely]] {
        if (state.candidate_batch_count != 0u) {
            auto &&previous =
                state.candidate_batch[state.candidate_batch_count - 1u];
            build.ascending &=
                !ray_query_candidate_before(candidate, previous);
            build.descending &=
                !ray_query_candidate_before(previous, candidate);
        }
        state.candidate_batch[state.candidate_batch_count++] = candidate;
        return;
    }
    ray_query_insert_triangle_candidate_overflow(
        state, build, candidate);
}

template<size_t lane_count>
[[nodiscard]] LUISA_FORCE_INLINE uint32_t ray_query_valid_mask(
    const int *valid) noexcept {
    static_assert(lane_count >= 1u && lane_count <= 16u);
    auto mask = uint32_t{0u};
    for (auto lane = 0u; lane < lane_count; lane++) {
        mask |= static_cast<uint32_t>(valid[lane] == -1) << lane;
    }
    return mask;
}

[[nodiscard]] LUISA_FORCE_INLINE bool ray_query_key_after_cursor(
    const SIMDHostRayQueryState &state,
    float t, uint32_t inst, uint32_t prim) noexcept {
    if (state.cursor_valid == 0u) { return true; }
    if (t != state.cursor_t) { return t > state.cursor_t; }
    if (inst != state.cursor_inst) { return inst > state.cursor_inst; }
    return prim > state.cursor_prim;
}

}// namespace

namespace {

template<size_t packet_width>
void ray_query_filter_wide_triangle_only_specialized(
    const RTCFilterFunctionNArguments *arguments) noexcept {
    auto *context = reinterpret_cast<RayQueryScanContext *>(
        arguments->context);
    auto valid_mask =
        ray_query_valid_mask<packet_width>(arguments->valid);
    while (valid_mask != 0u) {
        auto packet_lane = static_cast<uint32_t>(
            std::countr_zero(valid_mask));
        valid_mask &= valid_mask - 1u;
        arguments->valid[packet_lane] = 0;
        auto lane = RTCRayN_id(
            arguments->ray, packet_width, packet_lane);
        if (lane >= context->lane_count) { continue; }
        auto *state = context->states[lane];
        if (state == nullptr || state->terminated != 0u) { continue; }
        auto t = RTCRayN_tfar(
            arguments->ray, packet_width, packet_lane);
        auto inst = RTCHitN_instID(
            arguments->hit, packet_width, packet_lane, 0u);
        auto prim = RTCHitN_primID(
            arguments->hit, packet_width, packet_lane);
        if (!(t >= state->world_ray[3u] &&
              t <= state->world_ray[7u]) ||
            inst == RTC_INVALID_GEOMETRY_ID ||
            prim == RTC_INVALID_GEOMETRY_ID ||
            !ray_query_key_after_cursor(*state, t, inst, prim)) {
            continue;
        }
        ray_query_insert_triangle_candidate(
            *state, context->batch_build[lane],
            SIMDHostRayQuerySurfaceHit{
                .inst = inst,
                .prim = prim,
                .bary = {
                    RTCHitN_u(
                        arguments->hit, packet_width, packet_lane),
                    RTCHitN_v(
                        arguments->hit, packet_width, packet_lane)},
                .t = t,
            });
    }
}

LUISA_NEVER_INLINE void ray_query_filter_wide_triangle_only_generic(
    const RTCFilterFunctionNArguments *arguments) noexcept {
    auto *context = reinterpret_cast<RayQueryScanContext *>(
        arguments->context);
    auto valid_mask = uint32_t{0u};
    for (auto lane = 0u; lane < arguments->N; lane++) {
        valid_mask |= static_cast<uint32_t>(
                          arguments->valid[lane] == -1)
                      << lane;
    }
    while (valid_mask != 0u) {
        auto packet_lane = static_cast<uint32_t>(
            std::countr_zero(valid_mask));
        valid_mask &= valid_mask - 1u;
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
            !ray_query_key_after_cursor(*state, t, inst, prim)) {
            continue;
        }
        ray_query_insert_triangle_candidate(
            *state, context->batch_build[lane],
            SIMDHostRayQuerySurfaceHit{
                .inst = inst,
                .prim = prim,
                .bary = {
                    RTCHitN_u(
                        arguments->hit, arguments->N, packet_lane),
                    RTCHitN_v(
                        arguments->hit, arguments->N, packet_lane)},
                .t = t,
            });
    }
}

[[nodiscard]] bool specialized_triangle_filter_enabled() noexcept {
    static const auto enabled =
        !luisa::compute::detail::env_flag(
            "LUISA_SIMD_DISABLE_SPECIALIZED_TRIANGLE_FILTER");
    return enabled;
}

}// namespace

void ray_query_filter_wide_triangle_only(
    const RTCFilterFunctionNArguments *arguments) noexcept {
    if (arguments == nullptr || arguments->context == nullptr ||
        arguments->valid == nullptr || arguments->ray == nullptr ||
        arguments->hit == nullptr || arguments->N == 0u ||
        arguments->N > 16u) [[unlikely]] {
        std::abort();
    }
    if (!specialized_triangle_filter_enabled()) [[unlikely]] {
        ray_query_filter_wide_triangle_only_generic(arguments);
        return;
    }
    switch (arguments->N) {
        case 1u:
            ray_query_filter_wide_triangle_only_specialized<1u>(arguments);
            break;
        case 4u:
            ray_query_filter_wide_triangle_only_specialized<4u>(arguments);
            break;
        case 8u:
            ray_query_filter_wide_triangle_only_specialized<8u>(arguments);
            break;
        case 16u:
            ray_query_filter_wide_triangle_only_specialized<16u>(arguments);
            break;
        default:
            ray_query_filter_wide_triangle_only_generic(arguments);
            break;
    }
}

void ray_query_filter_triangle_only(
    const RTCFilterFunctionNArguments *arguments) noexcept {
    auto *context = reinterpret_cast<RayQueryScanContext *>(
        arguments->context);
    if (context == nullptr || arguments->valid == nullptr ||
        arguments->ray == nullptr || arguments->hit == nullptr ||
        arguments->N == 0u || arguments->N > 16u) [[unlikely]] {
        std::abort();
    }
    for (auto packet_lane = 0u; packet_lane < arguments->N;
         packet_lane++) {
        if (arguments->valid[packet_lane] != -1) { continue; }
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
            !ray_query_key_after_cursor(*state, t, inst, prim)) {
            continue;
        }
        ray_query_insert_triangle_candidate(
            *state, context->batch_build[lane],
            SIMDHostRayQuerySurfaceHit{
                .inst = inst,
                .prim = prim,
                .bary = {
                    RTCHitN_u(
                        arguments->hit, arguments->N, packet_lane),
                    RTCHitN_v(
                        arguments->hit, arguments->N, packet_lane)},
                .t = t,
            });
    }
}

[[nodiscard]] bool triangle_only_ray_query_enabled() noexcept {
    return !luisa::compute::detail::env_flag(
        "LUISA_SIMD_DISABLE_TRIANGLE_ONLY_RAY_QUERY");
}

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
        std::memcpy(ray.org_x, components, component_bytes);
        std::memcpy(
            ray.org_y, components + packet_width, component_bytes);
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
        std::memcpy(ray.time, times, component_bytes);
        std::memcpy(
            ray.tfar, components + 7u * packet_width,
            component_bytes);
        std::memcpy(
            ray.mask, visibility_masks,
            sizeof(uint32_t) * packet_width);
        static constexpr auto lane_ids = []() noexcept {
            std::array<uint32_t, packet_width> result{};
            for (auto lane = uint32_t{0u};
                 lane < packet_width; lane++) {
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
        ray.time[lane] = active ? times[lane] : 0.0f;
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
    RTCRay &ray, const SIMDHostRayQueryState &state) noexcept {
    ray.org_x = state.world_ray[0u];
    ray.org_y = state.world_ray[1u];
    ray.org_z = state.world_ray[2u];
    ray.tnear = ray_query_embree_tnear(state.world_ray[3u]);
    ray.dir_x = state.world_ray[4u];
    ray.dir_y = state.world_ray[5u];
    ray.dir_z = state.world_ray[6u];
    ray.time = state.time;
    ray.tfar = state.world_ray[7u];
    ray.mask = state.visibility_mask;
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

void initialize_ray_query_inputs_dense(
    uint32_t lane_count, uint64_t active_mask_bits,
    SIMDHostRayQueryState *const *states,
    float *components, uint32_t *visibility_masks,
    float *times) noexcept {
    for (auto lane = 0u; lane < lane_count; lane++) {
        if (!lane_active(active_mask_bits, lane, lane_count)) { continue; }
        auto *state = states[lane];
        LUISA_ASSERT(
            state != nullptr,
            "Triangle-only SIMD ray-query packet contains a null state.");
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

void initialize_ray_query_inputs_sparse(
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
            lane < lane_count && state != nullptr,
            "Triangle-only sparse ray-query packet contains a null state.");
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

template<bool wide, bool sparse>
void initialize_ray_query_context(
    RayQueryScanContext &context, uint32_t lane_count,
    uint64_t active_mask_bits,
    SIMDHostRayQueryState *const *states) noexcept {
    context.lane_count = lane_count;
    auto initialize_lane = [&](uint32_t lane) noexcept {
        context.states[lane] = states[lane];
        context.batch_build[lane] = {
            .heapified = false,
            .ascending = true,
            .descending = true,
        };
        auto *state = states[lane];
        LUISA_ASSERT(
            state != nullptr,
            "Triangle-only SIMD ray-query scan contains a null state.");
        state->candidate_batch_count = 0u;
        state->candidate_batch_index = 0u;
        state->candidate_batch_has_more = 0u;
        state->candidate_batch_initialized = 0u;
    };
    if constexpr (sparse) {
        while (active_mask_bits != 0u) {
            auto lane = static_cast<uint32_t>(
                std::countr_zero(active_mask_bits));
            active_mask_bits &= active_mask_bits - 1u;
            initialize_lane(lane);
        }
    } else {
        for (auto lane = 0u; lane < lane_count; lane++) {
            if (lane_active(active_mask_bits, lane, lane_count)) {
                initialize_lane(lane);
            }
        }
    }
#if LUISA_COMPUTE_SIMD_EMBREE_VERSION == 3
    rtcInitIntersectContext(&context.rtc);
    context.rtc.filter = wide ?
                             ray_query_filter_wide_triangle_only :
                             ray_query_filter_triangle_only;
#else
    rtcInitRayQueryContext(&context.rtc);
#endif
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
        "Triangle-only SIMD ray-query batch metadata is invalid.");
    while (state.candidate_batch_index < state.candidate_batch_count) {
        auto candidate =
            state.candidate_batch[state.candidate_batch_index++];
        if (!ray_query_key_after_cursor(
                state, candidate.t, candidate.inst, candidate.prim)) {
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
            "Triangle-only SIMD ray query returned invalid instance {}.",
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
    if (state.candidate_batch_has_more != 0u) {
        return RayQueryCandidateAdvance::needs_scan;
    }
    state.candidate_kind = static_cast<uint32_t>(
        SIMDHostRayQueryCandidateKind::none);
    state.terminated = 1u;
    return RayQueryCandidateAdvance::terminated;
}

void install_ray_query_candidate_batch(
    RayQueryScanContext &context, uint32_t lane,
    const SIMDHostAccelInstanceTable &instances) noexcept {
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
        "A new triangle-only SIMD ray-query batch made no progress.");
}

template<bool sparse>
void install_ray_query_candidate_batches(
    RayQueryScanContext &context, uint64_t active_mask_bits,
    const SIMDHostAccelInstanceTable &instances) noexcept {
    if constexpr (sparse) {
        while (active_mask_bits != 0u) {
            auto lane = static_cast<uint32_t>(
                std::countr_zero(active_mask_bits));
            active_mask_bits &= active_mask_bits - 1u;
            install_ray_query_candidate_batch(context, lane, instances);
        }
    } else {
        for (auto lane = 0u; lane < context.lane_count; lane++) {
            if (lane_active(
                    active_mask_bits, lane, context.lane_count)) {
                install_ray_query_candidate_batch(
                    context, lane, instances);
            }
        }
    }
}

template<
    size_t packet_width, typename RayHitPacket, typename RayPacket,
    bool wide>
void scan_ray_query_packet(
    RTCScene scene, const SIMDHostAccelInstanceTable &instances,
    uint32_t lane_count, uint64_t active_mask_bits,
    SIMDHostRayQueryState *const *states,
    bool terminate_on_first) noexcept {
    alignas(64) std::array<float, 8u * 16u> components{};
    alignas(64) std::array<uint32_t, 16u> visibility_masks{};
    alignas(64) std::array<float, 16u> times{};
    auto sparse = wide &&
                  !packet_fully_active<packet_width>(
                      lane_count, active_mask_bits);
    if (sparse) {
        initialize_ray_query_inputs_sparse(
            lane_count, active_mask_bits, states,
            components.data(), visibility_masks.data(), times.data());
    } else {
        initialize_ray_query_inputs_dense(
            lane_count, active_mask_bits, states,
            components.data(), visibility_masks.data(), times.data());
    }
    RayQueryScanContext context;
    if (sparse) {
        initialize_ray_query_context<wide, true>(
            context, lane_count, active_mask_bits, states);
    } else {
        initialize_ray_query_context<wide, false>(
            context, lane_count, active_mask_bits, states);
    }
    alignas(64) std::array<int, packet_width> valid{};
    if (terminate_on_first) {
        alignas(64) RayPacket packet{};
        initialize_ray_packet<packet_width>(
            packet, valid, lane_count, active_mask_bits,
            components.data(), visibility_masks.data(), times.data());
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
        arguments.filter = wide ?
                               ray_query_filter_wide_triangle_only :
                               ray_query_filter_triangle_only;
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
        arguments.filter = wide ?
                               ray_query_filter_wide_triangle_only :
                               ray_query_filter_triangle_only;
        if constexpr (packet_width == 4u) {
            rtcIntersect4(valid.data(), scene, &packet, &arguments);
        } else if constexpr (packet_width == 8u) {
            rtcIntersect8(valid.data(), scene, &packet, &arguments);
        } else {
            rtcIntersect16(valid.data(), scene, &packet, &arguments);
        }
#endif
    }
    if (sparse) {
        install_ray_query_candidate_batches<true>(
            context, active_mask_bits, instances);
    } else {
        install_ray_query_candidate_batches<false>(
            context, active_mask_bits, instances);
    }
}

void scan_ray_query_scalar(
    RTCScene scene, const SIMDHostAccelInstanceTable &instances,
    SIMDHostRayQueryState *const *states,
    bool terminate_on_first) noexcept {
    auto *state = states[0u];
    LUISA_ASSERT(
        state != nullptr,
        "Triangle-only scalar ray-query scan contains a null state.");
    RayQueryScanContext context;
    initialize_ray_query_context<false, false>(
        context, 1u, 1u, states);
    if (terminate_on_first) {
        RTCRay ray{};
        initialize_scalar_ray(ray, *state);
#if LUISA_COMPUTE_SIMD_EMBREE_VERSION == 3
        rtcOccluded1(scene, &context.rtc, &ray);
#else
        RTCOccludedArguments arguments{};
        rtcInitOccludedArguments(&arguments);
        arguments.context = &context.rtc;
        arguments.flags = static_cast<RTCRayQueryFlags>(
            arguments.flags |
            RTC_RAY_QUERY_FLAG_INVOKE_ARGUMENT_FILTER);
        arguments.filter = ray_query_filter_triangle_only;
        rtcOccluded1(scene, &ray, &arguments);
#endif
    } else {
        RTCRayHit ray_hit{};
        initialize_scalar_ray(ray_hit.ray, *state);
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
        arguments.filter = ray_query_filter_triangle_only;
        rtcIntersect1(scene, &ray_hit, &arguments);
#endif
    }
    install_ray_query_candidate_batches<false>(
        context, 1u, instances);
}

void scan_ray_query_group(
    SIMDAccel &accel, uint32_t lane_count,
    uint64_t group, SIMDHostRayQueryState *const *states,
    bool terminate_on_first, bool wide) noexcept {
    switch (lane_count) {
        case 1u:
            scan_ray_query_scalar(
                SIMDAccelAccess::scene(accel),
                SIMDAccelAccess::instances(accel),
                states, terminate_on_first);
            break;
        case 2u:
        case 4u:
            scan_ray_query_packet<
                4u, RTCRayHit4, RTCRay4, false>(
                SIMDAccelAccess::scene(accel),
                SIMDAccelAccess::instances(accel),
                lane_count, group, states, terminate_on_first);
            break;
        case 8u:
            if (wide) {
                scan_ray_query_packet<
                    8u, RTCRayHit8, RTCRay8, true>(
                    SIMDAccelAccess::scene(accel),
                    SIMDAccelAccess::instances(accel),
                    lane_count, group, states, terminate_on_first);
            } else {
                scan_ray_query_packet<
                    8u, RTCRayHit8, RTCRay8, false>(
                    SIMDAccelAccess::scene(accel),
                    SIMDAccelAccess::instances(accel),
                    lane_count, group, states, terminate_on_first);
            }
            break;
        case 16u:
            if (wide) {
                scan_ray_query_packet<
                    16u, RTCRayHit16, RTCRay16, true>(
                    SIMDAccelAccess::scene(accel),
                    SIMDAccelAccess::instances(accel),
                    lane_count, group, states, terminate_on_first);
            } else {
                scan_ray_query_packet<
                    16u, RTCRayHit16, RTCRay16, false>(
                    SIMDAccelAccess::scene(accel),
                    SIMDAccelAccess::instances(accel),
                    lane_count, group, states, terminate_on_first);
            }
            break;
        default: break;
    }
}

LUISA_FORCE_INLINE void commit_and_advance_lane(
    SIMDHostRayQueryState *state, uint32_t lane,
    SIMDHostAccelRayQueryProceed *expected_provider,
    uint64_t bit, uint64_t &pending) noexcept {
    LUISA_ASSERT(
        state != nullptr && state->accel != nullptr &&
            state->proceed == expected_provider,
        "Invalid active triangle-only SIMD ray-query state in lane {}.",
        lane);
    if (state->candidate_committed != 0u) {
        LUISA_ASSERT(
            state->candidate_kind == static_cast<uint32_t>(
                                         SIMDHostRayQueryCandidateKind::surface),
            "Triangle-only SIMD ray query committed a non-surface hit.");
        state->committed = SIMDHostRayQueryCommittedHit{
            .inst = state->candidate.inst,
            .prim = state->candidate.prim,
            .bary = {
                state->candidate.bary[0u],
                state->candidate.bary[1u]},
            .kind = static_cast<uint32_t>(SIMDHostRayQueryCandidateKind::surface),
            .t = state->candidate.t,
        };
        state->candidate_committed = 0u;
        if (state->terminate_on_first != 0u) {
            state->terminated = 1u;
        }
    }
    if (state->terminated == 0u) {
        auto *accel = static_cast<SIMDAccel *>(state->accel);
        auto advanced = advance_ray_query_candidate(
            *state, SIMDAccelAccess::instances(*accel));
        if (advanced == RayQueryCandidateAdvance::needs_scan) {
            pending |= bit;
        }
    }
}

template<bool wide>
void ray_query_proceed_triangle_only_impl(
    uint32_t lane_count, uint64_t active_mask_bits,
    SIMDHostRayQueryState *const *states,
    SIMDHostAccelRayQueryProceed *expected_provider) noexcept {
    LUISA_ASSERT(
        states != nullptr &&
            (wide ? (lane_count == 8u || lane_count == 16u) :
                    (lane_count == 1u || lane_count == 2u ||
                     lane_count == 4u || lane_count == 8u ||
                     lane_count == 16u)),
        "Invalid triangle-only SIMD ray-query width {}.", lane_count);
    auto lane_mask = (uint64_t{1u} << lane_count) - 1u;
    active_mask_bits &= lane_mask;
    if (active_mask_bits == 0u) { return; }
    auto fully_active = active_mask_bits == lane_mask;

    auto pending = uint64_t{0u};
    if constexpr (wide) {
        if (fully_active) [[likely]] {
            for (auto lane = 0u; lane < lane_count; lane++) {
                commit_and_advance_lane(
                    states[lane], lane, expected_provider,
                    uint64_t{1u} << lane, pending);
            }
        } else {
            auto remaining = active_mask_bits;
            while (remaining != 0u) {
                auto lane = static_cast<uint32_t>(
                    std::countr_zero(remaining));
                auto bit = uint64_t{1u} << lane;
                remaining &= remaining - 1u;
                commit_and_advance_lane(
                    states[lane], lane, expected_provider, bit, pending);
            }
        }
    } else {
        for (auto lane = 0u; lane < lane_count; lane++) {
            auto bit = uint64_t{1u} << lane;
            if ((active_mask_bits & bit) == 0u) { continue; }
            commit_and_advance_lane(
                states[lane], lane, expected_provider, bit, pending);
        }
    }

    while (pending != 0u) {
        auto first_lane = static_cast<uint32_t>(
            std::countr_zero(pending));
        auto *first_state = states[first_lane];
        auto *accel = static_cast<SIMDAccel *>(first_state->accel);
        auto terminate_on_first =
            first_state->terminate_on_first != 0u;
        auto group = uint64_t{0u};
        if constexpr (wide) {
            auto remaining = pending;
            while (remaining != 0u) {
                auto lane = static_cast<uint32_t>(
                    std::countr_zero(remaining));
                auto bit = uint64_t{1u} << lane;
                remaining &= remaining - 1u;
                auto *state = states[lane];
                if (state->accel == accel &&
                    (state->terminate_on_first != 0u) ==
                        terminate_on_first) {
                    group |= bit;
                }
            }
        } else {
            for (auto lane = 0u; lane < lane_count; lane++) {
                auto bit = uint64_t{1u} << lane;
                if ((pending & bit) == 0u) { continue; }
                auto *state = states[lane];
                if (state->accel == accel &&
                    (state->terminate_on_first != 0u) ==
                        terminate_on_first) {
                    group |= bit;
                }
            }
        }
        LUISA_ASSERT(
            group != 0u,
            "Empty triangle-only SIMD ray-query packet group.");
        scan_ray_query_group(
            *accel, lane_count, group, states,
            terminate_on_first, wide);
        pending &= ~group;
    }
}

}// namespace

void ray_query_proceed_triangle_only(
    uint32_t lane_count, uint64_t active_mask_bits,
    SIMDHostRayQueryState *const *states) noexcept {
    ray_query_proceed_triangle_only_impl<false>(
        lane_count, active_mask_bits, states,
        ray_query_proceed_triangle_only);
}

void ray_query_proceed_wide_triangle_only(
    uint32_t lane_count, uint64_t active_mask_bits,
    SIMDHostRayQueryState *const *states) noexcept {
    ray_query_proceed_triangle_only_impl<true>(
        lane_count, active_mask_bits, states,
        ray_query_proceed_wide_triangle_only);
}

}// namespace luisa::compute::simd::triangle_ray_query
