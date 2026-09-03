#pragma once

#include <functional>
#include <optional>
#include <tvm/tirx/buffer.h>
#include <tvm/tirx/stmt.h>

#include <luisa/tile/bridge/tirx/planner.h>
#include <luisa/core/stl/memory.h>

namespace luisa::compute::tile::bridge::tirx::detail {

// The barrier identity belongs to this emission, not an intrinsic-name match.
// Only the supplied fresh shared allocations are known disjoint. Unknown
// effects and the last barrier in each sequential region are hard boundaries.
[[nodiscard]] tvm::tirx::Stmt coalesce_group_barriers(
    tvm::tirx::Stmt body, const tvm::tirx::Stmt &compiler_barrier,
    luisa::span<const tvm::tirx::BufferVar> shared_allocations,
    bool enabled, GroupPlan &plan);

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
// Hard resource constraints survive structural export until target binding.
inline constexpr auto memory_resource_annotation = "luisa.tile.memory_resource";
inline constexpr auto logical_pipeline_annotation = "luisa.tile.pipeline";
inline constexpr auto pipeline_window_annotation = "luisa.tile.pipeline_window";
inline constexpr auto pipeline_interval_annotation = "luisa.tile.pipeline_interval";
inline constexpr auto pipeline_stage_annotation = "luisa.tile.pipeline_stage";

// Preserve stage cuts until dependence and storage planning. The current
// planner uses TVMx's native software-pipeline pass for safe two-phase
// prefetching, and leaves other pipelines ordered.
[[nodiscard]] tvm::tirx::Stmt schedule_pipelines(
    tvm::tirx::Stmt body, bool noalias, uint64_t shared_memory_limit);

// Give every logical vector lane its own compiler-local storage before TIRx
// vectorization. TIRx currently does not privatize AllocBuffer itself.
[[nodiscard]] tvm::tirx::Stmt privatize_vector_storage(const tvm::tirx::For &loop);

// Pack a CPU independent-element domain into SIMD lanes, retaining inner
// serial/reduction order. Undefined means retain the ordinary reference loop.
[[nodiscard]] tvm::tirx::Stmt vectorize_independent_elements(const tvm::tirx::For &loop);

// Realize one logical group per Metal threadgroup. Independent element
// domains and child workers share group-owned compiler temporaries.
[[nodiscard]] tvm::tirx::Stmt map_metal_cooperative_group(
    const tvm::tirx::For &loop, uint32_t max_threads, uint64_t shared_memory_limit,
    bool cooperative_matrix, const PlannerOptions &options, luisa::vector<GroupPlan> &plans);

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
    const MatrixDistribution &distribution = {}, MatrixLoopEmission *loop_emission = nullptr);

}// namespace luisa::compute::tile::bridge::tirx::detail
