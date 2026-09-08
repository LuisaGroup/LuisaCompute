#pragma once

#include <luisa/tile/bridge/xir/lower.h>

namespace luisa::compute::tile::bridge::xir {

class ExecutionCostPolicy;

struct ExecutionTarget {
    uint32_t packet_width{8u};
    uint32_t worker_count{1u};
    uint32_t task_chunks_per_worker{32u};
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
    double task_dispatch{0.0};
    double worker_activation{0.0};
};

struct PlannerOptions {
    uint32_t block_size{0u};
    luisa::vector<uint32_t> root_axis_order;
    uint32_t max_candidates{1024u};
    ExecutionCostModel cost;
    // Representation/code-size constraint, shared by cost extraction and
    // lowering. Zero retains the expanded diagnostic baseline.
    uint32_t max_unrolled_tile_elements{64u};
    uint32_t reduction_partitions{4u};
    // One preserves the default complete-program mapping. Zero opts into the
    // experimental joint search; packet_width forces a legal local-axis map.
    // The relative prior does not yet price tail CFG and worker activation
    // accurately enough to make joint search the default.
    uint32_t local_lanes{1u};
    // Fixed experimental realization, not an automatically selected winner.
    // The work prior does not yet model its masked-memory/CFG interactions.
    bool enable_load_reduction_fusion{false};
    // CPU task grain is independent of the physical block/packet mapping.
    // Zero retains the target's historical chunks-per-worker heuristic.
    uint32_t blocks_per_task{0u};
    bool search_task_grain{false};
    // Borrowed only during plan(). A policy may change costs, never legality.
    const ExecutionCostPolicy *cost_policy{nullptr};
};

struct ExecutionCost {
    double arithmetic_work{0.0};
    double memory_work{0.0};
    double dispatch_work{0.0};
    double imbalance_work{0.0};
    double score{0.0};
    double task_dispatch_work{0.0};
    double activation_work{0.0};
};

struct ExecutionPlan {
    uint32_t block_size{64u};
    // Outer-to-inner order of root parallel axes. This is an execution map,
    // not a change to any buffer's physical layout or to Tile value semantics.
    luisa::vector<uint32_t> root_axis_order;
    uint32_t dispatch_size{0u};
    ExecutionCost cost;
    uint32_t local_lanes{1u};
    uint32_t blocks_per_task{0u};
};

// The bridge extracts work and the exact static home-chunk assignment. The
// latter is a cost prior for the work-stealing Runtime, not a timing bound on
// heterogeneous CPU cores. Packet counts include a possibly masked tail.
struct ExecutionWork {
    double arithmetic_per_packet{0.0};
    double memory_per_packet{0.0};
    uint64_t packet_count{0u};
    uint64_t block_count{0u};
    uint64_t task_count{0u};
    uint32_t active_workers{0u};
    uint32_t blocks_per_task{0u};
    uint64_t critical_packets{0u};
    uint64_t critical_blocks{0u};
    uint64_t critical_tasks{0u};
};

class LUISA_TILE_XIR_BRIDGE_API ExecutionCostPolicy {
public:
    virtual ~ExecutionCostPolicy() noexcept = default;
    [[nodiscard]] virtual ExecutionCostModel coefficients(
        ExecutionTarget, const ExecutionCostModel &prior) const noexcept { return prior; }
    [[nodiscard]] virtual ExecutionCost evaluate(
        ExecutionTarget target, const ExecutionPlan &candidate,
        const ExecutionWork &work, const ExecutionCostModel &model) const noexcept = 0;
};

class LUISA_TILE_XIR_BRIDGE_API AnalyticExecutionCostPolicy : public ExecutionCostPolicy {
public:
    [[nodiscard]] ExecutionCost evaluate(
        ExecutionTarget target, const ExecutionPlan &candidate,
        const ExecutionWork &work, const ExecutionCostModel &model) const noexcept override;
};

struct PlanningResult {
    ExecutionPlan selected;
    luisa::vector<ExecutionPlan> candidates;
    luisa::string error;
    [[nodiscard]] bool ok() const noexcept { return error.empty() && !candidates.empty(); }
    [[nodiscard]] explicit operator bool() const noexcept { return ok(); }
};

// Exact minimum over legal axis permutations, block widths and whole-program
// versus packet-local distribution, and optionally power-of-two CPU task grains
// (plus the legacy grain and the whole launch), in the
// declared finite candidate space; exceeding the search budget is an error.
// Local distribution requires a common axis with owner-preserving extracts
// and closed unordered reductions. Other programs retain complete-program
// lanes, not because parallel needs an extra conflict proof, but because their
// redistribution/carry realizations are not implemented here yet.
[[nodiscard]] LUISA_TILE_XIR_BRIDGE_API PlanningResult plan(
    const Function &function, ExecutionTarget target, const PlannerOptions &options = {}) noexcept;

}// namespace luisa::compute::tile::bridge::xir
