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
    /// Total dispatch size to use at dispatch time: .dispatch(x, y) launches
    /// ceildiv(dispatch, block_size) blocks, so the T.Kernel grid
    /// (gx blocks x gy blocks of `threads` threads) maps to
    /// (gx * threads, gy, 1) for KERNEL_2D and (gx * threads, 1, 1) for
    /// KERNEL_1D.  Dispatch with
    /// stream << sh(buffers...).dispatch(result.dispatch_size.x, result.dispatch_size.y).
    uint3 dispatch_size;
};
struct TileToKernelConfig {
    bool use_cooperative : 1 {false};
    // TODO: dynamic batching size support
    uint32_t min_batching_size{1};
    uint32_t max_batching_size{1};
};
/// Translate a compiled tile kernel (a traced TileFunctionBuilder) into a
/// regular Luisa kernel (FunctionBuilder).  The traced builder is only read;
/// a const shared_ptr (as returned by tile::Kernel::function()) is accepted.
[[nodiscard]] LUISA_AST_API TileCompileResult tile_to_kernel(
    luisa::shared_ptr<const detail::TileFunctionBuilder> const &tile_function,
    TileToKernelConfig const &config = {});

}// namespace luisa::compute
