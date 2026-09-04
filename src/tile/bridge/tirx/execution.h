#pragma once

#include <array>
#include <functional>
#include <optional>
#include <tvm/tirx/buffer.h>
#include <tvm/tirx/stmt.h>
#include <tvm/tirx/function.h>

#include <luisa/tile/bridge/tirx/planner.h>
#include <luisa/core/stl/memory.h>

namespace luisa::compute::tile::bridge::tirx::detail {

// The barrier identity belongs to this emission, not an intrinsic-name match.
// Only the supplied fresh shared allocations are known disjoint. Unknown
// effects and the last barrier in each sequential region are hard boundaries,
// unless the whole group is proved to have independent subgroup execution.
// The optional statement identities are emission-local facts: private ops
// touch only nonescaping subgroup state and immutable inputs; output stores
// are synchronous, in bounds, and partitioned across the same subgroups.
[[nodiscard]] tvm::tirx::Stmt coalesce_group_barriers(
    tvm::tirx::Stmt body, const tvm::tirx::Stmt &compiler_barrier,
    luisa::span<const tvm::tirx::BufferVar> shared_allocations,
    bool enabled, bool elide_independent_subgroups, GroupPlan &plan,
    luisa::span<const tvm::tirx::Stmt> subgroup_private_operations,
    luisa::span<const tvm::tirx::Stmt> subgroup_output_stores);

// The structural bridge preserves logical domains and hard scope constraints.
// The target mapper consumes these only after resolving a legal realization.
inline constexpr auto logical_parallel_annotation = "luisa.tile.logical_parallel";
inline constexpr auto execution_scope_annotation = "luisa.tile.execution_scope";
// Positive rank of a perfect, rectangular serial element-loop nest. Its
// instances are independent by semantics, not a dependence-analysis hint.
// Keep axes intact until target binding chooses worker/SIMD partitioning.
inline constexpr auto independent_elements_annotation = "luisa.tile.independent_elements";
// Marks a reference MMA expansion and carries its reassociation permission.
// Target tensorization must still match and prove the current statement body;
// a stale annotation alone is never authority to replace arbitrary effects.
inline constexpr auto mma_annotation = "luisa.tile.mma";
// A versioned semantic contract emitted only after matching the complete
// TileIR dataflow. Target realization must revalidate its typed ABI before
// consuming it; these attributes are not inferred from names.
inline constexpr auto whole_gemm_contract_annotation = "luisa.tile.contract.whole_gemm";
inline constexpr auto whole_gemm_m_annotation = "luisa.tile.contract.whole_gemm.m";
inline constexpr auto whole_gemm_n_annotation = "luisa.tile.contract.whole_gemm.n";
inline constexpr auto whole_gemm_k_annotation = "luisa.tile.contract.whole_gemm.k";
inline constexpr auto cpu_matrix_realization_annotation = "luisa.tile.realization.cpu_matrix";
// Versioned provenance for a compiler-owned shared exp Tile. It licenses only
// the checked synchronous array-map realization; ordinary targets erase it.
inline constexpr auto materialized_exp_annotation = "luisa.tile.contract.materialized_exp";
inline constexpr auto reduction_contract_annotation = "luisa.tile.contract.reduction";
inline constexpr int64_t reduction_add_contract = 1;
inline constexpr int64_t reduction_max_contract = 2;
inline constexpr int64_t reduction_min_contract = 3;
inline constexpr auto cpu_math_realization_annotation = "luisa.tile.realization.cpu_math";
// Hard resource constraints survive structural export until target binding.
inline constexpr auto memory_resource_annotation = "luisa.tile.memory_resource";
// A user materialization remains explicit even without a placement constraint.
inline constexpr auto manual_memory_annotation = "luisa.tile.manual_memory";
inline constexpr auto logical_pipeline_annotation = "luisa.tile.pipeline";
inline constexpr auto pipeline_window_annotation = "luisa.tile.pipeline_window";
inline constexpr auto pipeline_interval_annotation = "luisa.tile.pipeline_interval";
inline constexpr auto pipeline_stage_annotation = "luisa.tile.pipeline_stage";
// A scheduling opportunity, never a dependence proof. Preserve an ordered
// pipeline for a second attempt after target resource/accumulator realization.
inline constexpr auto deferred_pipeline_annotation = "luisa.tile.deferred_pipeline";

// A proof query using native TIRx simplification, not a change to the program.
[[nodiscard]] bool prove_in_loop_domain(
    tvm::PrimExpr predicate, luisa::span<const tvm::tirx::ForNode *const> domain);

struct ReadonlyViews {
    tvm::tirx::Stmt body;
    // Only these explicitly proved, noalias input parameters may supplement
    // compiler-owned shared allocations in the matrix contract matcher.
    luisa::vector<tvm::tirx::BufferVar> inputs;
};

// CPU scalar/SIMD consumers can retain the original guarded read expression.
// Cooperative memory-input atoms currently require an unconditionally valid
// address, so their default remains the stricter fully-in-bounds view policy.
[[nodiscard]] ReadonlyViews forward_readonly_tile_loads(const tvm::tirx::PrimFunc &function, bool noalias, bool preserve_guards = false);

// Preserve stage cuts until dependence and storage planning. The current
// planner uses TVMx's native software-pipeline pass for safe two-phase
// prefetching, and leaves other pipelines ordered.
[[nodiscard]] tvm::tirx::Stmt schedule_pipelines(
    tvm::tirx::Stmt body, bool noalias, uint64_t shared_memory_limit, bool defer_prefetch = false);

// Split proved read-only global-to-shared copies around a closed matrix
// recurrence. Reuse shared storage and carry the next iteration in private
// scalars. Undefined means keep the existing ordered loop unchanged.
[[nodiscard]] tvm::tirx::Stmt try_prefetch_matrix_pipeline(
    const tvm::tirx::For &loop, const tvm::tirx::Stmt &compiler_barrier,
    uint32_t scalar_budget, GroupPlan &plan);

// Give every logical vector lane its own compiler-local storage before TIRx
// vectorization. TIRx currently does not privatize AllocBuffer itself.
[[nodiscard]] tvm::tirx::Stmt privatize_vector_storage(const tvm::tirx::For &loop);

// Pack a CPU independent-element domain into SIMD lanes, retaining inner
// serial/reduction order. Undefined means retain the ordinary reference loop.
[[nodiscard]] tvm::tirx::Stmt vectorize_independent_elements(const tvm::tirx::For &loop, uint32_t max_lanes = 16u);

// Run after flattening/vectorization/unrolling, immediately before host
// builtin lowering. Also consumes the retained manual-memory marker.
[[nodiscard]] tvm::tirx::Stmt plan_cpu_storage(tvm::tirx::Stmt body, uint32_t stack_budget);

// Replace versioned compiler-owned FP32 exp materializations with an Apple
// array-math call. The matcher revalidates the perfect compact map and refuses
// dependent/in-place bodies; unsupported hosts fail closed when requested.
[[nodiscard]] tvm::tirx::Stmt realize_cpu_vector_math(tvm::tirx::Stmt body);

// Replace the reference body only when the versioned whole-GEMM contract,
// parameter ABI, noalias promise, and registered provider all agree.
// Throws on any mismatch; an explicit library request never silently falls
// back to a different numerical/performance contract.
[[nodiscard]] tvm::tirx::PrimFunc realize_cpu_whole_gemm(
    tvm::tirx::PrimFunc function, bool noalias);

// Realize one logical group per Metal threadgroup. Independent element
// domains and child workers share group-owned compiler temporaries.
[[nodiscard]] tvm::tirx::Stmt map_metal_cooperative_group(
    const tvm::tirx::For &loop, uint32_t max_threads, uint64_t shared_memory_limit,
    bool cooperative_matrix, bool metal_mpp, const PlannerOptions &options, luisa::vector<GroupPlan> &plans,
    luisa::span<const tvm::tirx::BufferVar> readonly_inputs);

// The planner and emitter use the same semantic contract matcher. A diagnostic
// annotation or coincidentally named buffer alone cannot authorize an MMA.
[[nodiscard]] std::optional<MatrixWorkload> metal_matrix_workload(
    const tvm::tirx::For &loop,
    const std::function<tvm::tirx::BufferVar(tvm::tirx::BufferVar)> &map_buffer);

struct MatrixCarry {
    tvm::tirx::BufferVar initial;
    tvm::tirx::BufferVar result;
    uint64_t rows, columns;
};

[[nodiscard]] std::optional<MatrixCarry> metal_matrix_carry(
    const tvm::tirx::For &loop,
    const std::function<tvm::tirx::BufferVar(tvm::tirx::BufferVar)> &map_buffer);

struct MatrixLoopEmission {
    tvm::tirx::Stmt before;
    tvm::tirx::Stmt after;
    tvm::PrimExpr initial;
    // The recurrence proof may authorize an overwriting multiply only when
    // this is a single-iteration, positive-zero-initialized accumulator.
    bool overwrite_accumulator{false};
    // Present only for a synchronous subgroup-private MPP recurrence with a
    // literal initializer and a verified direct output. These are the entire
    // memory inputs of step; the caller must prove their immutability before
    // granting subgroup-isolation facts to the synchronization pass.
    std::optional<std::array<tvm::tirx::BufferVar, 2u>> subgroup_inputs;
    tvm::tirx::Stmt subgroup_step;
    struct Output {
        tvm::tirx::BufferVar buffer;
        tvm::ffi::Array<tvm::PrimExpr> indices;
        tvm::tirx::PrimVar row, column;
        uint64_t stride;
        bool transpose;
    };
    std::optional<Output> output;
};

// Match a canonical C-tile copy and prove every destination element/guard
// in bounds under the enclosing loop domains. Unknown is not permission to
// issue an unguarded cooperative store.
[[nodiscard]] std::optional<MatrixLoopEmission::Output> metal_matrix_output(
    const tvm::tirx::For &loop, const MatrixCarry &carry,
    luisa::span<const tvm::tirx::ForNode *const> ancestors);

// Select a native 8x8 FP32 matrix atom only for a proved reference MMA body.
// Undefined means the ordinary independent-element realization must be used.
[[nodiscard]] tvm::tirx::Stmt try_metal_matrix(
    const tvm::tirx::For &loop, const tvm::tirx::PrimVar &thread, uint64_t threads,
    const std::function<tvm::tirx::BufferVar(tvm::tirx::BufferVar)> &map_buffer,
    const MatrixDistribution &distribution = {}, MatrixLoopEmission *loop_emission = nullptr,
    bool metal_mpp = false);

}// namespace luisa::compute::tile::bridge::tirx::detail
