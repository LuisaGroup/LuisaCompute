#pragma once

// tile_to_kernel.h — tile IR (TensorStmt / TileFunctionBuilder) → regular Luisa
// GPU kernel (FunctionBuilder) lowering entry point.
//
// A tile function (traced by <luisa/dsl/tensor.h> into a
// luisa::compute::detail::TileFunctionBuilder, see <luisa/ast/tensor.h> and
// <luisa/ast/tile_function_builder.h>) describes a *tile-style* program: whole
// tiles, T.copy / T.gemm / T.reduce_sum / ... , with no threads, no block size
// and no dispatch.  `tile_to_kernel` performs the SIMD→SIMT step (see the
// lowering plan in src/ast/tile_to_kernel.cpp) and emits a REGULAR Luisa
// kernel (a FunctionBuilder of Tag::KERNEL) that implements the same tile
// program on every backend:
//
//   TileCompileResult r = tile_to_kernel(tile_kernel.function());
//   Kernel2D k{r.function};                        // a normal Luisa kernel
//   stream << k(buffer_a, buffer_b, buffer_c).dispatch(r.dispatch_size.x, r.dispatch_size.y);
//
//   // Batched form (TileToKernelConfig min/max batching size != (1,1)): each
//   // thread group computes several batch items at once (one per z-thread) and
//   // the runtime batch count rides on the z dispatch axis, so the kernel must
//   // be wrapped as a 3D kernel and issued as ONE dispatch per batch set:
//   //   Kernel3D<...> k{r.function};
//   //   stream << k(buffers...).dispatch(r.dispatch_size.x, r.dispatch_size.y, batch_count);
//   // When a loop needs several tile kernels / uploads per iteration, batch
//   // them into a single CommandList::create() -> commit() instead of N stream
//   // submissions so host-side submission overhead does not defeat batching.
//
// The generated kernel takes one Buffer<T> argument per Global tensor of the
// tile function (in AllocStmt order), declares Shared<T> for Shared tensors
// and per-thread local arrays for Fragment tensors, and lowers every tile op
// into partitioned (Global/Shared) or replicated (Fragment) element loops.

#include <luisa/core/basic_types.h>
#include <luisa/core/stl/memory.h>
#include <luisa/ast/function_builder.h>
#include <luisa/ast/tile_function_builder.h>

namespace luisa::compute {

/// Result of translating a traced tile function into a regular kernel.
struct LUISA_AST_API TileCompileResult {
    /// The generated kernel (FunctionBuilder with Tag::KERNEL).  It declares
    /// one Buffer argument per Global tensor of the tile function (in
    /// AllocStmt order) and carries the launch block size via
    /// FunctionBuilder::set_block_size.
    luisa::shared_ptr<detail::FunctionBuilder> function;
    /// Total 2D dispatch size to use at dispatch time.  x/y are the
    /// compile-time tile grid as today: .dispatch(x, y) launches
    /// ceildiv(dispatch, block_size) blocks, so the T.Kernel grid
    /// (gx blocks x gy blocks of `threads` threads) maps to
    /// (gx * threads, gy) for KERNEL_2D and (gx * threads, 1) for
    /// KERNEL_1D.  z is reserved for the RUNTIME batch count and is never
    /// part of the compile-time dispatch.
    ///
    /// Non-batched callers dispatch the 2D grid directly:
    ///   stream << sh(buffers...).dispatch(r.dispatch_size.x, r.dispatch_size.y);
    ///
    /// Batched callers (TileToKernelConfig min/max batching size != (1,1))
    /// additionally pass the runtime batch count on the z axis:
    ///   stream << sh(buffers...).dispatch(r.dispatch_size.x, r.dispatch_size.y, batch_count);
    /// The kernel must be wrapped as `to_kernel<3>()`; the kernel derives
    /// `batch_index` from `block_id().z` / `thread_id().z` and adds
    /// `batch_index * tensor_volume` to every Global access automatically.
    uint2 dispatch_size;
};
struct TileToKernelConfig {
    bool use_cooperative : 1 {false};
    // Tensor-op fast path: when enabled, eligible whole-tensor tile programs
    // (Global-only operands with F16/F32/I32 dtypes, no batching, no
    // shared/fragment storage) lower each op to a single side-effecting
    // TENSOR_* CallOp (see include/luisa/ast/op.h) instead of per-element
    // partition loops.  Only the CUDA AST codegen implements TENSOR_* ops, so
    // this defaults to false; ineligible ops fall back per op to the
    // partition path.
    bool use_tensor : 1 {false};
    // Dynamic batching: when enabled (min != 1 || max != 1), each thread
    // group computes `block_size().z` batch items at once — one per z-thread —
    // and the z axis of the dispatch carries the runtime batch count
    // (batch_count must lie in [min_batching_size, max_batching_size]).
    //   * min_batching_size / max_batching_size are >= 1 with min <= max;
    //   * (1, 1) disables batching and adds zero lowering overhead;
    //   * when enabled the z block size is chosen by the lowering heuristic as
    //     clamp(ceil(target_threads / threads), 1,
    //           min(min_batching_size, 64, max(1, 1024 / threads))), so
    //     B_z <= min_batching_size and B_z <= 64 always hold.
    uint32_t min_batching_size{1};
    uint32_t max_batching_size{1};
    // Multi-buffered software pipelining (plan 2.16 / CUTASS "Pipelining"):
    // when a T.Pipelined(count, stages) loop has stages >= 2 and its body is
    // the GEMM pattern (global->shared rank-2 copies feeding shared-reading
    // consumers, chunkable into contiguous row chunks), the lowering
    // allocates `stages` copies of each pipelined shared tile and overlaps the
    // next iterations' global->shared copies with the current compute
    // (prologue prefetch / mainloop issue+wait / epilogue drain, per CUTASS).
    // Two lowering strategies are available:
    //   * pipeline_use_async_copy = true  (default): emit the CUDA cp.async
    //     pipeline builtins (ASYNC_COPY / PIPELINE_COMMIT / PIPELINE_WAIT_PRIOR
    //     + BUFFER_ADDRESS).  CUDA only — other backends reject these ops.
    //   * pipeline_use_async_copy = false: emit the portable manual-copy
    //     pipeline (BUFFER_READ + shared stores, no BUFFER_ADDRESS /
    //     ASYNC_COPY / PIPELINE_*), used by DX/VK.  When Tier-2 gates pass
    //     (see src/ast/tile_to_kernel.cpp) the leading copy warps overlap the
    //     next stage's copy with the compute warps' current-stage compute;
    //     otherwise the fallback is the synchronous manual pipeline (every
    //     thread copies, then every thread computes), which is always correct.
    // On any gate failure (bad chunk shape, batching, non-matching body) the
    // synchronous lowering is kept, so enabling it is always safe.
    bool use_pipeline : 1 {false};
    // true  -> emit the CUDA cp.async builtins (ASYNC_COPY / PIPELINE_COMMIT /
    // PIPELINE_WAIT_PRIOR + BUFFER_ADDRESS); CUDA only.
    // false -> emit the portable manual-copy pipeline (no BUFFER_ADDRESS /
    // ASYNC_COPY); used by dx/vk.
    bool pipeline_use_async_copy : 1 {true};
    // Tier-2 warp specialization: number of leading warps dedicated to the
    // manual copy stage.  0 = auto (see _manual_copy_threads()).
    uint32_t pipeline_copy_warps{1u};
    // Host wavefront width used to size the copy stage (default 32; set to the
    // device's wave size, e.g. 64 on AMD, when known).
    uint32_t pipeline_warp_size{32u};
};
/// Translate a compiled tile kernel (a traced TileFunctionBuilder) into a
/// regular Luisa kernel (FunctionBuilder).  The traced builder is only read;
/// a const shared_ptr (as returned by tile::Kernel::function()) is accepted.
[[nodiscard]] LUISA_AST_API TileCompileResult tile_to_kernel(
    luisa::shared_ptr<const detail::TileFunctionBuilder> const &tile_function,
    TileToKernelConfig const &config = {});

}// namespace luisa::compute
