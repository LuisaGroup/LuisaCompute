#pragma once

#include <cstdint>

#include <luisa/core/dll_export.h>
#include <luisa/core/stl/string.h>
#include <luisa/core/stl/vector.h>

namespace luisa::compute::tile::bridge::tirx {

class ExecutionCostPolicy;

enum class MatrixCostBasis : uint8_t {
    SIMDGROUP_REFERENCE,
    METAL_MPP_MEMORY,
};

// Relative issue-work coefficients, NOT measured nanoseconds or a promise of
// occupancy. Device calibration can replace these priors without changing
// candidate legality or the solver. Unknown register allocation is represented
// by live fragment scalars, never reported as a measured hardware register count.
struct ExecutionCostModel {
    double matrix_issue{1.0};
    double shared_fragment_transfer{2.0};
    double independent_element{0.0078125};
    double subgroup_setup{8.0};
    // Metal reduction planner v1 scores discrete SIMD-group
    // realizations in abstract issue rounds. The first term prices one scalar
    // stripe round, the second prices each reduction collective per
    // participating SIMD-group, and the third prices a cooperative
    // threadgroup. One-group programs amortize the latter when several
    // independent logical programs share a threadgroup.
    double subgroup_reduction_scalar_round{1.0};
    double subgroup_reduction_collective{2.0};
    double subgroup_reduction_group_setup{16.0};
    // MPP reads A/B from memory inside each tensor operation. These terms are
    // expressed in logical 8x8 fragments. The footprint terms distinguish the
    // two row-major operands without pretending to be byte-accurate cache
    // traffic; the aspect terms are versioned target priors for MPP descriptor
    // and subgroup-grid shape. Correctness-checked JIT data may replace them.
    double metal_mpp_memory_fragment{1.0};
    double metal_mpp_lhs_footprint{0.125};
    double metal_mpp_rhs_footprint{0.375};
    double metal_mpp_output_fragment{1.0};
    double metal_mpp_tensor_operation{8.0};
    double metal_mpp_accumulator_init{1.0};
    double metal_mpp_tile_aspect{0.1875};
    double metal_mpp_local_row_aspect{0.25};
    double metal_mpp_local_column_aspect{0.125};
    // Non-matrix element work is divided across the participating subgroups.
    // One logical unit therefore represents one scalar per subgroup lane.
    double metal_mpp_independent_element{0.03125};
    // Fixed critical-path prior for entering one cooperative MPP program.
    double metal_mpp_group_setup{8.0};
    uint32_t preferred_subgroups{4u};
    uint32_t preferred_fragment_scalars_per_lane{32u};
    // Reference-realization latency-wave prior for independent programs.
    uint32_t preferred_concurrent_programs{64u};
    // MPP v2 uses subgroup demand rather than pretending that 64-, 128-, and
    // 256-thread programs consume the same machine capacity. This M1-class
    // prior is deliberately replaceable by a calibrated target profile; it is
    // not a queried occupancy limit or a portable hardware fact.
    uint32_t metal_mpp_concurrent_subgroups{512u};
};

struct PlannerOptions {
    // Automatic CPU roots below this many independent tasks stay serial to
    // avoid paying the TVM thread-pool launch cost. Explicit worker bindings
    // remain hard constraints. One preserves the always-parallel policy.
    uint32_t min_cpu_parallel_tasks{64u};
    // Opt-in LLVM storage realization. At most this many bytes of compact,
    // nonescaping compiler temporaries become stack allocations per PrimFunc
    // (including alignment padding). Valid range is [0,65536]. Zero, or a
    // disabled planner, retains TVM workspace allocation.
    // This is not a bound on LLVM spills or user/TVM-owned stack objects.
    uint32_t max_cpu_stack_bytes{0u};
    // Logical SIMD-pack budget, not a hardware vector width/register count.
    // Sixteen retains single-row packing. Larger powers of two (up to 128)
    // may combine adjacent independent rows while preserving each element's
    // serial recurrence. Requires CPU auto-vectorization when non-default.
    uint32_t max_cpu_vector_lanes{16u};
    bool enabled{true};
    // Fuse an automatic root program with its sole independent element map.
    // Requires immutable input forwarding and disjoint output effects. This
    // is a physical mapping choice; explicit worker/group bindings stay exact.
    bool fuse_gpu_elementwise{true};
    // Opt-in Metal realization for proved FP32 add/max/min reductions.
    // Independent short programs may share one threadgroup; wider programs
    // search every whole-SIMD-group width within the target limit and the
    // two-level collective's capacity (at most 32 subgroups). Element domains
    // are striped over workers and reducers use native collectives plus proved
    // shared partials when necessary. Floating-point addition therefore uses
    // a tree order rather than the reference left fold. This policy is both
    // the numerical permission and the planner candidate switch; it is never
    // inferred from a target name or a coincidental loop annotation.
    bool metal_subgroup_reductions{false};
    // Zero retains automatic packing. Nonzero fixes independent logical
    // programs per group, separately from threads_per_group. More than one
    // program currently requires one subgroup per program (no group fences).
    uint32_t reduction_programs_per_group{0u};
    // Partial unrolling of each worker's ordered stripe. Does not introduce
    // independent accumulators or change its floating-point recurrence order.
    // A bounded code-size choice, not a hardware vector-width promise.
    uint32_t reduction_unroll_factor{1u};
    // Consecutive logical elements owned by one worker before advancing to
    // the next worker (1, 2, 4 or 8). An ownership layout, not a vector ISA
    // guarantee. Non-default widths require the reduction-tree permission.
    uint32_t reduction_lane_elements{1u};
    bool retain_accumulators{true};
    // Elide the initial/final shared accumulator only when its literal fill
    // and sole, fully in-bounds global store have been proved by analysis.
    bool direct_accumulator_store{true};
    // Merge compiler-inserted group barriers only across independent effects.
    // Retain loop-boundary fences unless the independent-subgroup option below
    // is also selected. Explicit/unknown synchronization is never removed.
    bool coalesce_group_barriers{true};
    // A profitability choice, not permission to assume independence. Requires
    // a whole-group proof of private subgroup programs and one terminal store.
    // Default off: even redundant fences can improve cache/issue scheduling.
    bool elide_independent_subgroup_barriers{false};
    // Zero lets the solver choose. A nonzero value is an exact tuning
    // constraint, checked against the target before any code is generated.
    uint32_t threads_per_group{0u};
    // A compiler search/code-size budget, not a claimed hardware register limit.
    uint32_t max_fragment_scalars_per_lane{64u};
    // Bound compiler-created worker-private stripes for one logical row
    // program. This is a software-state budget, not a hardware register count:
    // the final backend may keep, spill, or scalarize these values. Candidates
    // above the limit are rejected before code generation so wide reused Tiles
    // cannot silently create pathological per-thread arrays.
    uint32_t max_reduction_striped_scalars_per_worker{64u};
    // Bound compilation work independently of target capacity. If an automatic
    // search would exceed this many widths, require a larger budget or an exact
    // thread count instead of silently truncating the supposedly exact search.
    uint32_t max_thread_candidates{32u};
    // Maximum independent values loaded before their stores in a cooperative
    // copy. One retains the scalar load/store sequence; no async engine or
    // vector-alignment promise is implied. The emitter checks each domain.
    uint32_t max_copy_batch{1u};
    // Bound worker-private software-prefetch storage, not hardware registers.
    // Zero disables late prefetching; pipeline window one is always ordered.
    uint32_t max_pipeline_prefetch_scalars_per_lane{32u};
    ExecutionCostModel cost;
    // Borrowed for synchronous planning/compilation only; never retained in a
    // shader or serialized. The backend owns device calibration and policy.
    const ExecutionCostPolicy *cost_policy{nullptr};
};

struct ExecutionLimits {
    uint32_t max_threads{256u};
    uint32_t subgroup_size{32u};
    uint64_t shared_memory_bytes{0u};
};

struct ReductionCandidate {
    uint64_t programs{0u};
    uint32_t threads{0u};
    uint32_t subgroups_per_program{0u};
    uint32_t programs_per_group{0u};
    uint64_t shared_memory_bytes{0u};
    uint64_t striped_scalars_per_worker{0u};
    uint64_t reductions{0u};
    double scalar_rounds{0.0};
    uint32_t unroll_factor{1u};
    uint32_t lane_elements{1u};
    // Exact launch/ownership features, not measured occupancy. Packed tails
    // may leave some programs inactive in the last physical threadgroup.
    uint64_t threadgroups{0u};
    // Sum over separately distributed domains (including reduction inputs).
    // Dividing by scalar_rounds * cooperating workers gives useful lane work
    // relative to the maximum per-worker scalar recurrence length.
    double scalar_elements{0.0};
    double lane_utilization{0.0};
};

// Bridge-owned proofs and candidate generation precede these read-only
// profitability hooks. Hard device limits and numerical permissions are not
// overridable by a score. No TVM types, RTTI or cross-module ownership.
class LUISA_TILE_TIRX_BRIDGE_API ExecutionCostPolicy {
public:
    virtual ~ExecutionCostPolicy() noexcept = default;
    [[nodiscard]] virtual ExecutionCostModel coefficients(
        const ExecutionLimits &limits, MatrixCostBasis basis,
        const ExecutionCostModel &prior) const noexcept = 0;
    [[nodiscard]] virtual double reduction_score(
        const ReductionCandidate &candidate, const ExecutionCostModel &model) const noexcept = 0;
};

// Existing deterministic prior. Backends can inherit and override either
// calibration or the complete row-program objective independently of search.
class LUISA_TILE_TIRX_BRIDGE_API AnalyticExecutionCostPolicy : public ExecutionCostPolicy {
public:
    [[nodiscard]] ExecutionCostModel coefficients(
        const ExecutionLimits &, MatrixCostBasis, const ExecutionCostModel &prior) const noexcept override { return prior; }
    [[nodiscard]] double reduction_score(
        const ReductionCandidate &candidate, const ExecutionCostModel &model) const noexcept override;
};

struct MatrixWorkload {
    uint64_t rows{0u};
    uint64_t columns{0u};
    uint64_t contraction{0u};
    uint64_t executions{1u};
    // A proved closed recurrence: C -> MMA -> D -> yield C, with no other
    // observation of C/D inside this many loop iterations. Zero disables it.
    uint64_t accumulator_iterations{0u};
    bool has_direct_output{false};
    // A proved one-iteration, positive-zero recurrence that may use MPP's
    // D=A*B mode. This is a semantic proof result, not a profitability hint.
    bool overwrites_accumulator{false};
};

struct GroupWorkload {
    uint64_t programs{0u};
    uint64_t independent_elements{0u};
    uint64_t max_independent_elements{1u};
    uint64_t shared_memory_bytes{0u};
    luisa::vector<MatrixWorkload> matrices;
};

struct MatrixDistribution {
    // A zero subgroup grid selects the existing one-atom-at-a-time reference
    // realization, including uniform subgroup tails. Otherwise the exact map is
    // (sg_m * atom_rows + local_m, sg_n * atom_columns + local_n).
    uint32_t subgroups_m{0u};
    uint32_t subgroups_n{0u};
    uint64_t atom_rows{1u};
    uint64_t atom_columns{1u};
    bool persistent_accumulator{false};
    bool direct_accumulator_store{false};

    [[nodiscard]] bool rectangular() const noexcept { return subgroups_m != 0u && subgroups_n != 0u; }
};

struct PlanCost {
    double matrix_issues{0.0};
    double shared_fragment_transfers{0.0};
    double direct_fragment_stores{0.0};
    double metal_mpp_operations{0.0};
    double memory_fragment_reads{0.0};
    double lhs_footprint_fragments{0.0};
    double rhs_footprint_fragments{0.0};
    double accumulator_initializations{0.0};
    double tile_aspect_fragments{0.0};
    double local_row_aspect_issues{0.0};
    double local_column_aspect_issues{0.0};
    double independent_elements{0.0};
    uint64_t fragment_scalars_per_lane{0u};
    // score is one logical program's relative work and drives the inner
    // solver. kernel_score applies the outer program-wave prior so separately
    // staged/JIT-compiled execution shapes can be compared on the same basis.
    double score{0.0};
    double concurrent_waves{1.0};
    double kernel_score{0.0};
};

struct GroupPlan {
    luisa::string name;
    uint64_t programs{0u};
    uint32_t threads{1u};
    uint64_t shared_memory_bytes{0u};
    uint64_t candidates_considered{0u};
    uint64_t candidates_rejected{0u};
    uint32_t max_copy_batch{1u};
    uint64_t batched_copy_operations{0u};
    uint64_t prefetched_pipeline_loops{0u};
    uint64_t prefetch_storage_scalars_per_lane{0u};
    // Reduction-realization facts. Zero subgroups means this is not a native
    // subgroup-reduction plan. Striped storage is compiler-local state after
    // logical Tile materializations are compacted to each worker's ownership.
    uint32_t reduction_subgroups_per_program{0u};
    uint32_t reduction_programs_per_group{0u};
    uint64_t striped_storage_scalars_per_worker{0u};
    uint64_t reduction_operations{0u};
    uint64_t reduction_elements{0u};
    uint64_t elementwise_elements_per_program{0u};
    // Shared pure Tile SSA values scalarized to one definition per worker.
    uint32_t elementwise_scalar_temporaries{0u};
    uint32_t reduction_unroll_factor{1u};
    uint32_t reduction_lane_elements{1u};
    uint64_t reduction_threadgroups{0u};
    double reduction_scalar_rounds{0.0};
    double reduction_lane_utilization{0.0};
    // Static emitted synchronization sites, not dynamic barrier executions.
    // Filled by realization; the bootstrap ranking does not yet price them.
    uint64_t group_barrier_sites_before{0u};
    uint64_t group_barrier_sites_after{0u};
    // Realization proof, independent of the selected fence-elision policy.
    bool independent_subgroups{false};
    luisa::vector<MatrixDistribution> matrices;
    MatrixCostBasis cost_basis{MatrixCostBasis::SIMDGROUP_REFERENCE};
    // Explicit realization selected by the target bridge. cost_basis records
    // which separately versioned feature model ranked it.
    bool metal_mpp{false};
    PlanCost cost;
    bool optimized{false};
};

struct PlanningResult {
    GroupPlan plan;
    luisa::string error;

    [[nodiscard]] bool ok() const noexcept { return error.empty(); }
    [[nodiscard]] explicit operator bool() const noexcept { return ok(); }
};

// Exhaustive discrete solver for the currently implemented group realization
// family: thread count x rectangular subgroup factorizations x resident atom
// blocks. Covers every exact rectangular factorization in the supplied bounds,
// not all possible GPU programs. No compiler IR, JIT, measurements, or mutable
// global state is needed to rank candidates. The SIMD-group basis includes the
// reference realization; an
// explicit MPP basis has no scalar fallback and fails if no legal descriptor
// exists. MatrixWorkloads must come from proved semantic MMA contracts.
[[nodiscard]] LUISA_TILE_TIRX_BRIDGE_API PlanningResult plan_group(
    const GroupWorkload &workload, const ExecutionLimits &limits,
    const PlannerOptions &options = {},
    MatrixCostBasis cost_basis = MatrixCostBasis::SIMDGROUP_REFERENCE) noexcept;

// Independent, integer-only coverage/resource check, also applied by codegen.
// Cost-model coefficients cannot grant legality to a candidate.
[[nodiscard]] LUISA_TILE_TIRX_BRIDGE_API bool verify_matrix_distribution(
    const MatrixWorkload &workload, const MatrixDistribution &distribution,
    uint32_t threads, uint32_t subgroup_size) noexcept;

}// namespace luisa::compute::tile::bridge::tirx
