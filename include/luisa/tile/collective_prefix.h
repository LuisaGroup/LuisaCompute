#pragma once

#include <luisa/tile/collective_partition.h>

namespace luisa::compute::tile {

// A read-only proof of the existing closed unordered inclusive FP32 SUM
// dataflow. This is neither a new IR operation nor an execution/schedule node.
// IDs/Dim values borrow the unchanged Function; reanalyze after any mutation.
// First scope deliberately matches the existing rank-two, final-axis matcher.
struct ClosedPrefixAnalysis {
    luisa::string error;
    const Function *function{nullptr};
    uint64_t parallel_operation_id{0u};
    uint64_t load_operation_id{0u};
    uint64_t map_operation_id{0u};
    uint64_t store_operation_id{0u};
    // Reuse the unique admitted work record: INCLUSIVE_SUM / FLOAT32,
    // operation identity, contribution width and independent/input volumes.
    CollectiveWork collective;
    ScalarType storage{ScalarType::INVALID};
    Dim independent_axis;
    Dim contribution_axis;
    uint64_t logical_independent_extent{0u};
    uint64_t logical_contribution_extent{0u};
    CollectiveCandidateGeometry original;
    // These are root view byte intervals, NOT a static noalias declaration.
    // Every realization which changes the snapshot/schedule must check the
    // complete intervals after actual binding offsets, else retain original.
    DisjointRequirement disjoint;
    [[nodiscard]] bool ok() const noexcept {
        return error.empty() && function != nullptr;
    }
};

// Runs fresh analyze_collective_work on the actual attached, verified IR,
// then establishes the memory/cast/origin/use closure. No supplied analysis
// receipt, backend artifact, chunk size, threads, alignment or device input.
[[nodiscard]] LUISA_TILE_API ClosedPrefixAnalysis analyze_closed_prefix(
    const Function &function) noexcept;

}// namespace luisa::compute::tile
