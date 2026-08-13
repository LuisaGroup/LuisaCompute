#include <bit>

#include <luisa/core/logging.h>

#include "../llvm/llvm_schedule_codegen.h"

namespace luisa::compute::simd {

namespace {

[[noreturn]] LUISA_NEVER_INLINE void invalid_status_packet(
    uint32_t lane_count) noexcept {
    LUISA_ERROR_WITH_LOCATION(
        "Invalid status-aware SIMD ray-query packet width {}.", lane_count);
}

[[noreturn]] LUISA_NEVER_INLINE void missing_plain_provider() noexcept {
    LUISA_ERROR_WITH_LOCATION(
        "Status-aware SIMD ray query has no plain provider.");
}

}// namespace

uint64_t simd_host_ray_query_proceed_status(
    uint32_t lane_count, uint64_t active_mask_bits,
    SIMDHostRayQueryState *const *states) noexcept {
    if (states == nullptr || lane_count == 0u || lane_count > 16u) [[unlikely]] {
        invalid_status_packet(lane_count);
    }
    auto lane_mask = (uint64_t{1u} << lane_count) - 1u;
    auto active = active_mask_bits & lane_mask;
    if (active == 0u) { return 0u; }
    auto first_lane = static_cast<uint32_t>(std::countr_zero(active));
    auto *first_state = states[first_lane];
    if (first_state == nullptr || first_state->proceed == nullptr) [[unlikely]] {
        missing_plain_provider();
    }
    auto *proceed = first_state->proceed;
    proceed(lane_count, active, states);
    return simd_host_ray_query_pack_status(
        lane_count, active, states);
}

uint64_t simd_host_ray_query_proceed_wide_procedural_status(
    uint32_t lane_count, uint64_t active_mask_bits,
    SIMDHostRayQueryState *const *states) noexcept {
    if (states == nullptr || lane_count != 16u) [[unlikely]] {
        invalid_status_packet(lane_count);
    }
    auto lane_mask = (uint64_t{1u} << lane_count) - 1u;
    auto active = active_mask_bits & lane_mask;
    if (active == 0u) { return 0u; }
    auto first_lane = static_cast<uint32_t>(std::countr_zero(active));
    auto *first_state = states[first_lane];
    if (first_state == nullptr || first_state->proceed == nullptr) [[unlikely]] {
        missing_plain_provider();
    }
    auto *proceed = first_state->proceed;
    proceed(lane_count, active, states);
    return simd_host_ray_query_pack_procedural_wide_status(
        lane_count, active, states);
}

}// namespace luisa::compute::simd
