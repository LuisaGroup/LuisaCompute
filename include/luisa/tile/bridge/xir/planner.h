#pragma once

#include <luisa/tile/bridge/xir/lower.h>

namespace luisa::compute::tile::bridge::xir {

struct ExecutionTarget {
    uint32_t packet_width{8u};
    uint32_t worker_count{1u};
};

// Relative packet-work coefficients, not nanoseconds or physical instruction
// counts. The current model is an uncalibrated prior, independently replaceable
// without changing candidate legality or the bounded exhaustive solver.
struct ExecutionCostModel {
    double arithmetic{1.0};
    double broadcast_load{1.0};
    double contiguous_memory{2.0};
    double gathered_lane{2.0};
    double block_dispatch{128.0};
};

struct PlannerOptions {
    uint32_t block_size{0u};
    luisa::vector<uint32_t> root_axis_order;
    uint32_t max_candidates{1024u};
    ExecutionCostModel cost;
};

struct ExecutionCost {
    double arithmetic_work{0.0};
    double memory_work{0.0};
    double dispatch_work{0.0};
    double imbalance_work{0.0};
    double score{0.0};
};

struct ExecutionPlan {
    uint32_t block_size{64u};
    // Outer-to-inner order of root parallel axes. This is an execution map,
    // not a change to any buffer's physical layout or to Tile value semantics.
    luisa::vector<uint32_t> root_axis_order;
    uint32_t dispatch_size{0u};
    ExecutionCost cost;
};

struct PlanningResult {
    ExecutionPlan selected;
    luisa::vector<ExecutionPlan> candidates;
    luisa::string error;
    [[nodiscard]] bool ok() const noexcept { return error.empty() && !candidates.empty(); }
    [[nodiscard]] explicit operator bool() const noexcept { return ok(); }
};

// Exact minimum over all legal axis permutations and block widths in the
// declared finite candidate space; exceeding the search budget is an error.
// This first solver preserves each logical worker's entire Tile program.
// Splitting a Tile across workers requires additional dependence/alias and
// collective realizations, and is deliberately not inferred from its shape.
[[nodiscard]] LUISA_TILE_XIR_BRIDGE_API PlanningResult plan(
    const Function &function, ExecutionTarget target, const PlannerOptions &options = {}) noexcept;

}// namespace luisa::compute::tile::bridge::xir
