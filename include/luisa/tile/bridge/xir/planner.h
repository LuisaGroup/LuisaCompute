#pragma once

#include <luisa/tile/bridge/xir/lower.h>
#include <type_traits>

namespace luisa::compute::tile::bridge::xir {

class ExecutionCostPolicy;

struct ExecutionTarget {
    // The physical width used by warp_lane_id and warp collectives. It is an
    // ABI requirement, not an upper bound on an arbitrary logical subgroup.
    uint32_t packet_width{8u};
    // Scheduling parameters for the thread-pool target info. Other targets
    // define their own scheduling model; these are not GPU core counts.
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
    // Fixed guarded streaming candidate. The current prior still estimates
    // the original snapshot path, not an assumed noalias probability.
    bool enable_pointwise_fusion{false};
    // Same first-consumer admission as lowering. Prices saved snapshot
    // accesses, not an assumed native speedup or automatically selected winner.
    bool enable_expression_reduction_fusion{false};
    // Shares pure map admission with lower(); work is charged at reads,
    // including repeated/broadcast reads, rather than at an elided snapshot.
    bool enable_map_fusion{false};
    // Fixed opt-in root traversal, not searched or credited with cache reuse.
    // Factors are in original-axis order and must divide the static extents.
    luisa::vector<uint32_t> root_axis_tiles;
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

struct ExecutionResourceLimits {
    // Compiler-owned static snapshots only. This does not bound native
    // registers, peak liveness, aligned workspace, or hardware occupancy.
    uint64_t max_snapshot_bytes_per_worker{LowerOptions{}.max_local_bytes};
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
    luisa::vector<uint32_t> root_axis_tiles;
    ExecutionResources resources;
    ExecutionResourceLimits resource_limits;
};

// The bridge extracts work and packet/block counts (including masked tails).
// Target info supplies scheduling fields. The thread-pool implementation uses
// static home chunks as a cost prior, not a timing bound on heterogeneous CPUs.
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

// Backend-provided execution capabilities and scheduling model. The bridge
// owns semantic admission and mapping invariants; accepts() may only narrow
// that legal set. Cost policies rank admitted candidates, never legalize them.
// Objects and their returned cost policy are borrowed only during plan().
class LUISA_TILE_XIR_BRIDGE_API ExecutionTargetInfo {
public:
    virtual ~ExecutionTargetInfo() noexcept = default;
    [[nodiscard]] virtual ExecutionTarget target() const noexcept = 0;
    [[nodiscard]] virtual luisa::vector<uint32_t> block_sizes() const noexcept = 0;
    [[nodiscard]] virtual bool supports_local_distribution() const noexcept { return true; }
    [[nodiscard]] virtual bool supports_task_grain() const noexcept { return false; }
    [[nodiscard]] virtual bool accepts(const ExecutionPlan &candidate) const noexcept = 0;
    // Called after geometry admission, with the candidate's shared static
    // resource analysis attached. The returned budget is retained in the plan
    // and must also be used by lowering. Zero permits allocation-free plans.
    [[nodiscard]] virtual ExecutionResourceLimits resource_limits(const ExecutionPlan &candidate) const noexcept { return {}; }
    // The input contains target-independent work/packet/block counts; the
    // target fills scheduling quantities. A GPU must not inherit CPU home
    // chunks, work stealing, or caller-thread activation costs by accident.
    [[nodiscard]] virtual ExecutionWork schedule(const ExecutionPlan &candidate, ExecutionWork work) const noexcept = 0;
    [[nodiscard]] virtual const ExecutionCostPolicy &cost_policy() const noexcept = 0;
};

// Reusable CPU thread-pool realization. Kept explicit so new backend policies
// do not inherit CPU scheduling by default. Native SIMD additionally checks
// the device's supported physical widths and workspace ABI.
class LUISA_TILE_XIR_BRIDGE_API ThreadPoolExecutionTargetInfo : public ExecutionTargetInfo {
private:
    ExecutionTarget _target;

public:
    explicit ThreadPoolExecutionTargetInfo(ExecutionTarget target) noexcept : _target{target} {}
    [[nodiscard]] ExecutionTarget target() const noexcept override { return _target; }
    [[nodiscard]] luisa::vector<uint32_t> block_sizes() const noexcept override;
    [[nodiscard]] bool supports_task_grain() const noexcept override { return true; }
    [[nodiscard]] bool accepts(const ExecutionPlan &candidate) const noexcept override;
    [[nodiscard]] ExecutionWork schedule(const ExecutionPlan &candidate, ExecutionWork work) const noexcept override;
    [[nodiscard]] const ExecutionCostPolicy &cost_policy() const noexcept override;
};

struct ExecutionRejection {
    ExecutionPlan candidate;
    luisa::string reason;
};

struct PlanningResult {
    ExecutionPlan selected;
    luisa::vector<ExecutionPlan> candidates;
    luisa::string error;
    luisa::vector<ExecutionRejection> rejected;
    [[nodiscard]] bool ok() const noexcept { return error.empty() && !candidates.empty(); }
    [[nodiscard]] explicit operator bool() const noexcept { return ok(); }
};

// Compatibility entry point using ThreadPoolExecutionTargetInfo.
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

// Backend entry point. Candidate block widths, additional legality, scheduling
// and the default cost policy come from info. options.cost_policy, when set,
// replaces only the objective, not the backend's capabilities or schedule.
[[nodiscard]] LUISA_TILE_XIR_BRIDGE_API PlanningResult plan_with_target_info(
    const Function &function, const ExecutionTargetInfo &info, const PlannerOptions &options = {}) noexcept;

// Deduction keeps the existing plan(function, {}) spelling unambiguous: an
// empty initializer cannot deduce Info and still selects ExecutionTarget.
template<typename Info>
    requires std::is_base_of_v<ExecutionTargetInfo, Info>
[[nodiscard]] PlanningResult plan(const Function &function, const Info &info, const PlannerOptions &options = {}) noexcept {
    return plan_with_target_info(function, info, options);
}

}// namespace luisa::compute::tile::bridge::xir
