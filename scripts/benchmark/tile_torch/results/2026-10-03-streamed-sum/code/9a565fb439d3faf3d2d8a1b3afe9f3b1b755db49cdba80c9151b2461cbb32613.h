#pragma once

#include <luisa/tile/collective_partition.h>

namespace luisa::compute::cuda::native_tile {

// Private prototype. Logical IR quantities only; no occupancy or traffic claim.
struct StreamedSumIRFacts {
    uint64_t discarded_pure_program_operations{};
    uint64_t programs{};
    uint64_t serial_iterations{};
    uint64_t collective_invocations_per_program{};
    // Inputs to actual REDUCE operations; excludes the separately counted
    // elementwise SERIAL carry additions. V4 has one chunk-wide REDUCE.
    uint64_t contribution_elements_per_program{};
    uint64_t elementwise_elements_per_program{};
    uint64_t nominal_read_bytes_per_program{};
    uint64_t nominal_write_bytes_per_program{};
    uint64_t largest_materialized_tile_elements{};
    // Sum of all explicit Tile definitions AND block arguments, including
    // old/new loop carries. A conservative storage upper bound, not exact
    // liveness, physical allocation, or an execution-weighted byte total.
    uint64_t explicit_tile_storage_upper_bound_per_program{};
};

struct StreamedSumIR {
    luisa::string error;
    luisa::unique_ptr<tile::Module> module;
    tile::Function *function{};
    tile::DisjointRequirement disjoint;
    StreamedSumIRFacts facts;
    [[nodiscard]] bool ok() const noexcept { return error.empty() && function != nullptr; }
};

// Owned clone of a strict BR1 load/cast/SUM/row-only-pure-epilogue/store slice.
// V4 carries an FP32 contribution Tile through SERIAL, then reduces once.
// No environment/default integration. The caller must keep the incumbent and
// require actual final-pointer whole-range disjointness before invocation.
[[nodiscard]] StreamedSumIR build_streamed_sum_ir(
    const tile::Function &original, uint32_t chunk_extent,
    bool enable_fast_math = false) noexcept;

}// namespace luisa::compute::cuda::native_tile
