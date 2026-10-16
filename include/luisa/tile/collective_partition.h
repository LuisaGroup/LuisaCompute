#pragma once

#include <luisa/tile/collective_plan.h>

namespace luisa::compute::tile {

struct IndependentPartitionRequest {
    // Unspecified identities infer the unique admitted collective/axis.
    uint64_t collective_operation_id{~uint64_t{0u}};
    Dim independent_axis;
    uint64_t target_extent_per_program{1u};
};

struct RootViewInterval {
    uint32_t argument_index{0u};
    uint64_t byte_offset{0u};
    uint64_t byte_count{0u};
};

struct DisjointRequirement {
    RootViewInterval input;
    RootViewInterval output;
};

struct CollectiveCandidateGeometry {
    uint64_t programs{0u};
    uint64_t independent_extent_per_program{0u};
    uint64_t full_programs{0u};
    // Zero means there is no partial program.
    uint64_t tail_valid_extent{0u};
};

// A conditional realization recipe for a closed, single-axis FP32 SUM/MAXIMUM.
// Dimension/operation identities borrow the unmodified function. This is not
// a device schedule, a cost score, a noalias declaration or an SSA rewrite.
// The input/output ranges MUST be disjoint at invocation after binding offsets
// are applied; otherwise use the original realization and original geometry.
struct IndependentCollectivePlan {
    luisa::string error;
    const Function *function{nullptr};
    uint64_t parallel_operation_id{0u};
    uint64_t collective_operation_id{0u};
    uint64_t load_operation_id{0u};
    uint64_t store_operation_id{0u};
    CollectiveKind kind{CollectiveKind::SUM};
    ScalarType input_storage{ScalarType::INVALID};
    ScalarType output_storage{ScalarType::INVALID};
    Dim independent_axis;
    Dim contribution_axis;
    uint32_t input_independent_axis{0u};
    uint32_t input_contribution_axis{0u};
    uint32_t output_independent_axis{0u};
    uint32_t output_rank{0u};
    uint64_t logical_independent_extent{0u};
    uint64_t logical_contribution_extent{0u};
    uint64_t tile_contribution_extent{0u};
    // Optional exact coordinate < logical_contribution_extent selection of
    // the FP32 source versus the reducer identity. Preserve it if present;
    // unmasked zero-padded MAXIMUM is intentionally not changed to -infinity.
    bool contribution_identity_mask{false};
    CollectiveCandidateGeometry original;
    CollectiveCandidateGeometry candidate;
    DisjointRequirement disjoint;
    [[nodiscard]] bool ok() const noexcept { return error.empty() && function != nullptr; }
};

// Reexamines the actual verified IR, not a supplied logical-work receipt.
// Semantic partitioning accepts any positive proper divisor of the original
// independent extent. Device-specific extent/grid/layout limits are separate.
[[nodiscard]] LUISA_TILE_API IndependentCollectivePlan plan_independent_collective(
    const Function &function, IndependentPartitionRequest request = {}) noexcept;

enum class IndependentCollectiveGeometryKind : uint8_t { ORIGINAL,
                                                         PARTITIONED };

inline constexpr luisa::string_view kIndependentCollectiveWorkSchema = "partition-work-v1";

// Exact semantic recipe quantities, not physical allocation, transactions or
// occupancy. Original-geometry facts do not replace original IR SSA liveness.
// Component byte counts may describe the same/coalesced value and MUST NOT be
// added or scaled from original analysis to manufacture a candidate peak.
// No peak field is supplied: candidate liveness is unknown without an actual
// candidate IR or explicit recipe SSA analysis. Cost facts grant no legality.
struct IndependentCollectiveWorkFacts {
    luisa::string error;
    IndependentCollectiveGeometryKind geometry_kind{IndependentCollectiveGeometryKind::PARTITIONED};
    CollectiveKind kind{CollectiveKind::SUM};
    ScalarType input_storage{ScalarType::INVALID};
    ScalarType output_storage{ScalarType::INVALID};
    Dim independent_axis;
    Dim contribution_axis;
    uint64_t collective_operation_id{0u};
    CollectiveCandidateGeometry geometry;
    uint64_t logical_independent_extent{0u};
    uint64_t logical_contribution_extent{0u};
    uint64_t tile_contribution_extent{0u};
    uint64_t padded_independent_elements{0u};
    uint64_t collective_input_elements_per_program{0u};
    uint64_t collective_input_elements_total{0u};
    uint64_t valid_input_elements{0u};
    uint64_t valid_output_elements{0u};
    uint64_t valid_input_bytes{0u};
    uint64_t valid_output_bytes{0u};
    uint64_t input_snapshot_bytes_per_program{0u};
    uint64_t fp32_source_bytes_per_program{0u};
    uint64_t fp32_result_bytes_per_program{0u};
    uint64_t output_value_bytes_per_program{0u};
    bool independent_bounds_elidable{false};
    bool contribution_bounds_elidable{false};
    bool contribution_identity_mask{false};
    [[nodiscard]] bool ok() const noexcept { return error.empty() && geometry.programs != 0u; }
};

// Derive original or candidate geometry from a successful unmodified-function
// plan. Recheck all arithmetic/geometry and return a diagnostic on overflow.
// Backends must still validate actual IR and invocation disjointness before
// selecting a realization; this read-only cost record is not a proof token.
[[nodiscard]] LUISA_TILE_API IndependentCollectiveWorkFacts analyze_independent_collective_candidate(
    const IndependentCollectivePlan &plan,
    IndependentCollectiveGeometryKind geometry = IndependentCollectiveGeometryKind::PARTITIONED) noexcept;

}// namespace luisa::compute::tile
