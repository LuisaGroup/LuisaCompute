#include "simd_accel_ray_query.h"

#include <algorithm>
#include <bit>
#include <cstdlib>

namespace luisa::compute::simd::detail {
namespace {

// This file intentionally carries a private copy of the candidate insertion
// logic used by the established dense filter. Sharing the implementation lets
// GCC outline it from simd_accel.cpp and measurably perturbs W1/W2/W4. The
// layout assertions in simd_accel.cpp keep the callback context ABI common;
// integration tests keep both insertion paths semantically identical.

[[nodiscard]] LUISA_FORCE_INLINE bool ray_query_key_after_cursor(
    const SIMDHostRayQueryState &state,
    float t, uint32_t inst, uint32_t prim,
    const SIMDHostAccelInstanceTable &instances) noexcept {
    if (state.cursor_valid == 0u) { return true; }
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

LUISA_FORCE_INLINE void ray_query_insert_candidate(
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

}// namespace

void ray_query_filter_wide(
    const RTCFilterFunctionNArguments *arguments) noexcept {
    auto *context = reinterpret_cast<RayQueryScanContext *>(
        arguments->context);
    if (context == nullptr || arguments->valid == nullptr ||
        arguments->ray == nullptr || arguments->hit == nullptr ||
        arguments->N == 0u || arguments->N > 16u ||
        context->instances == nullptr) [[unlikely]] {
        // Avoid pulling fmt/spdlog static initialization into this isolated
        // translation unit. These are internal Embree callback invariants.
        std::abort();
    }
    auto valid_mask = [&]() noexcept {
        switch (arguments->N) {
            case 1u: return ray_query_valid_mask<1u>(arguments->valid);
            case 4u: return ray_query_valid_mask<4u>(arguments->valid);
            case 8u: return ray_query_valid_mask<8u>(arguments->valid);
            case 16u: return ray_query_valid_mask<16u>(arguments->valid);
            default: {
                auto mask = uint32_t{0u};
                for (auto lane = 0u; lane < arguments->N; lane++) {
                    mask |= static_cast<uint32_t>(
                                arguments->valid[lane] == -1)
                            << lane;
                }
                return mask;
            }
        }
    }();
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
        if (curve) { v = -1.0f; }
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

}// namespace luisa::compute::simd::detail
