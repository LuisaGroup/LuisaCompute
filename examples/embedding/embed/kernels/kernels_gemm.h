// bge-m3 embedding engine -- GEMM with fused epilogues (plan §4 kernel #3) and
// the standalone residual-add kernel (plan §4 kernel #7).
//
// gemm: C[M,N] = epilogue(A[M,K] @ B[K,N]). A is row-major [M,K], B is row-major
// [K,N] (the packer pre-transposes every weight so reads coalesce along N --
// plan §3.3), C is row-major [M,N]. It replaces the ONNX `MatMul` x192 plus the
// bias `Add` and the elementwise `Erf` / `Relu` chains that follow it in the
// bge-m3 encoder (QKV projection, attention output projection, FFN x2 per layer,
// and the retrieval heads -- plan §0/§4 table row #3).
//
// Epilogues are compile-time: the caller passes a GemmEpilogue and the factory
// returns the matching pre-built Kernel2D. The selection is a host-side switch
// over template instantiations; the unused epilogue math is erased by
// `if constexpr` while the DSL kernel is being recorded.
//
// Launch geometry (plan §4 #3): one 16x16 output tile per thread block,
// set_block_size(256) (16*16 = 256 threads, lc_optimize-measured best on
// vk/dx). K is tiled in chunks of kGemmTile through shared memory
// (2 * 16 * 16 * 4 B = 2 KiB per block, well under the 32 KiB DX floor).
// Ragged M/N/K are handled by predicated (zero-filling) tile loads and a guarded
// store, so arbitrary sizes are correct, not just multiples of 16.
//
// Device-free Kernel2D factory per CONVENTIONS.md: the caller (Phase B executor
// / tests) compiles the result with device.compile(...) and caches it as a
// Shader. All dimensions are runtime kernel args, so one compilation serves
// every shape.
#pragma once
#include <luisa/dsl/syntax.h>

namespace luisa::embed {

// Output tile side (plan §4 #3: 16x16 tile).
inline constexpr uint kGemmTile = 16u;
// Threads per block: kGemmTile * kGemmTile = 256 (plan §4 #3).
inline constexpr uint kGemmBlock = kGemmTile * kGemmTile;
static_assert(kGemmBlock == 256u, "gemm block must be 256 threads (plan §4 #3)");
// Thread block width of residual_add (plan §4 #7: 1D, block 256).
inline constexpr uint kResidualAddBlock = 256u;

// Compile-time epilogue flags (plan §4 #3: NONE / BIAS / BIAS_GELU(erf) /
// BIAS_RELU / RESIDUAL_ADD).
enum class GemmEpilogue : uint32_t {
    None,       ///< C = A @ B
    Bias,       ///< C = A @ B + bias[N]
    BiasGeluErf,///< C = gelu_erf(A @ B + bias[N]) -- the ONNX `Erf` GELU variant
    BiasRelu,   ///< C = relu(A @ B + bias[N])
    ResidualAdd,///< C = A @ B + R[M,N] (the aux slot carries the residual row)
};

// Log/debug name of an epilogue flag (also used as the compiled kernel name).
[[nodiscard]] const char *gemm_epilogue_name(GemmEpilogue ep) noexcept;

// C[M,N] = epilogue(A[M,K] @ B[K,N]).
// Buffers: A [M*K], B [K*N] (row-major, coalesced along N), C [M*N],
//          aux = bias[N] for Bias/BiasGeluErf/BiasRelu, residual R[M*N] for
//          ResidualAdd, unused for None (bind a dummy 1-element buffer view).
// Scalars: M, N, K (runtime; arbitrary ragged values allowed).
using GemmKernel = compute::Kernel2D<void(compute::Buffer<float> /*A*/,
                                          compute::Buffer<float> /*B*/,
                                          compute::Buffer<float> /*C*/,
                                          compute::Buffer<float> /*aux*/,
                                          uint32_t /*M*/,
                                          uint32_t /*N*/,
                                          uint32_t /*K*/)>;

// Compile-time epilogue dispatch: returns the Kernel2D for the requested flag
// (host-side switch over template instantiations; see kernels_gemm.cpp).
[[nodiscard]] GemmKernel make_gemm_kernel(GemmEpilogue ep) noexcept;

// Total-thread dispatch size for a [M,N] gemm output: the x dimension carries
// ceil(N/16) blocks of 256 threads (tiles along N), the y dimension carries
// ceil(M/16) one-thread-tall blocks (tiles along M). Both grid dimensions stay
// far below the DX 65535-group cap for every bge-m3 shape (lc_optimize trap #2).
[[nodiscard]] constexpr uint2 gemm_dispatch_size(uint32_t m, uint32_t n) noexcept {
    const auto tiles_n = (n + kGemmTile - 1u) / kGemmTile;
    const auto tiles_m = (m + kGemmTile - 1u) / kGemmTile;
    return {tiles_n * kGemmBlock, tiles_m};
}

// residual_add (plan §4 #7): out[i] = a[i] + b[i] over `count` elements -- used
// for the residual `Add` edges the GEMM epilogue does not already fuse
// (ONNX `Add` x341 subset, plan §4 table row #7).
// Launch: block kResidualAddBlock, dispatch(count) = TOTAL THREADS
// (lc_optimize trap #1); the tail thread is guarded. The kernel implements the
// flattened 2D-grid idiom (see kernels_embed.cpp), so the host MAY dodge the
// DX 65535-group cap with dispatch(x = bs*cols, y = blocks/cols) when
// count/kResidualAddBlock would exceed it.
using ResidualAddKernel = compute::Kernel1D<void(compute::Buffer<float> /*a*/,
                                                 compute::Buffer<float> /*b*/,
                                                 compute::Buffer<float> /*out*/,
                                                 uint32_t /*count*/)>;

[[nodiscard]] ResidualAddKernel make_residual_add_kernel() noexcept;

}// namespace luisa::embed
