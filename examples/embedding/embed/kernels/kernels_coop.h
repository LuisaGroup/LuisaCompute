// bge-m3 embedding engine -- cooperative-vector (coop) matvec projection
// kernels (optimization plan tensor_optimize.md §5: C0 probe + C2
// K-slab x N-tile matvec). These kernels are the experimental alternative to
// the classic tiled GEMM (kernels_gemm.h) for the projection hot spots
// (QKV / FFN up-down / output projection -- plan §3): one thread per token
// computes a full output row y[N] = W[K,N] @ x[K] + bias[N] with the
// device cooperative-vector matrix-vector ops.
//
// See tensor_optimize.md §5.3 (C2) for the algorithm:
//   acc[Ntile] = 0
//   for each Kslab:
//       partial = cooperative_mat_mul(W_slab, x_slab)
//       for n in [0, Ntile): acc[n] += partial[n]  // local, fixed order
//   acc += bias  (fused via cooperative_mat_mul_add on the LAST slab)
//   apply epilogue; store
// The K-slab accumulator is a per-thread CoopVector<float> register family
// accumulated in a fixed, deterministic element order. NO cooperative element-
// wise CallOps are emitted (they are "future" ops rejected by the backends --
// plan §1.2/§1.3 and coop_api.md §3) and NO COOPERATIVE_VECTOR_ACCUMULATE is
// used to join slabs (global RMW ordering is forbidden -- plan §1.3/§5.3).
//
// HOST CONTRACT (checked with LUISA_ASSERT where possible, otherwise a caller
// precondition -- both documented here and enforced in kernels_coop.cpp):
// - fp32 weights only. The buffers must hold plain fp32; packed half/bf16
//   storage MUST NOT enter the coop path (plan §1.3: no BF16 coop; §5.5:
//   fp16/bf16 are follow-up work gated on fp32 correctness). The kernel
//   itself takes opaque ByteBuffers and cannot inspect the element format, so
//   every coop entry point asserts the format at the boundary it can see
//   (GemmWeightFormat == Fp32, or -- for the untyped probe -- the caller
//   promise below).
// - K % KSLAB == 0 and N % NTILE == 0 are GUARANTEED BY THE CALLER. Ragged
//   K/N are NOT supported by the coop kernel (no tail predicates are
//   emitted); the host must guard and fall back to the regular GEMM for
//   shapes that are not exact multiples (plan §5.3: "ragged K/N -> fallback").
// - x/y/bias/weight are device buffers holding fp32:
//     x     : [M*K] fp32 (token-major, K contiguous -- the [M,K] activation)
//     weight: [K*N] fp32 row-major canonical layout (K rows of N contiguous
//             fp32). Whether this canonical layout is what the backend
//             cooperative mul (MUL_OPTIMAL / InferencingOptimal, stride 0)
//             expects IS THE EXPERIMENT; the C0 probe validates it (see
//             make_coop_probe_kernel). coop_api.md §8.
//     bias  : [N] fp32
//     y     : [M*N] fp32 (row-major output)
//   The host binds these as ByteBufferViews (zero-copy alias of the typed
//   Buffer<float>, coop_api.md §4: `ByteBufferView byte_view{typed_buffer}`);
//   the DSL kernel parameters are ByteBuffers because the coop ops address
//   their operands by BYTE offset.
// - All byte offsets passed to set_byte_offset are 32-bit (the DSL ref API
//   takes uint32). The kernels compute the offsets in 64-bit DSL arithmetic
//   and truncate, which is exact for the bge-m3 envelope (largest byte
//   offset < 64 MiB << 2^32; plan §5.3 "offset computed 64-bit safe").
//
// The FunctionBuilder flips Function::use_cooperative_operations() whenever a
// coop local is created (coop_api.md §5), so every kernel defined here carries
// the flag automatically; the probe harness may assert it on the built
// Function.
#pragma once
#include <luisa/core/stl/string.h>// also exports luisa::string_view
#include <luisa/dsl/syntax.h>

#include "embedding/embed/kernels/kernels_gemm.h"// GemmEpilogue

namespace luisa::embed {

// Fixed thread-block width of the coop kernels (256 tokens per block when
// tokens_per_block == 1; block == 256 is asserted against the table below).
inline constexpr uint kCoopGemmBlock = 256u;

// C0 probe geometry (plan §5.1): fixed tiny K x N so the probe compiles and
// dispatches identically on every coop-capable path.
inline constexpr uint kCoopProbeK = 16u;
inline constexpr uint kCoopProbeN = 8u;

// One entry of the coop tuning table (plan §5.3: Kslab/Ntile are
// emission-time constants; tpb tokens are C++-unrolled per thread).
struct CoopGemmConfig {
    const char *name;    ///< variant key, e.g. "coop_k64_n32_t2"
    uint32_t kslab;      ///< K slab per cooperative_mat_mul (CoopMatrixRef n)
    uint32_t ntile;      ///< N tile per accumulator  (CoopMatrixRef m)
    uint32_t tokens_per_block;// tokens computed per thread (C++-unrolled)
};

// Compiled-in variant table (host side; the sweep harness iterates it).
[[nodiscard]] uint32_t coop_gemm_variant_count() noexcept;
[[nodiscard]] const CoopGemmConfig *coop_gemm_variant(uint32_t i) noexcept;
// nullptr when the name is unknown.
[[nodiscard]] const CoopGemmConfig *coop_gemm_variant_by_name(
    luisa::string_view name) noexcept;

// C0 layout/numerics probe (plan §5.1): ONE block, ONE thread computes
//   y[8] = M[16x8] @ x[16] + bias[8]
// with a single cooperative_mat_mul_add over fixed K=16, N=8 -- canonical
// row-major [K,N] fp32 weights aliased as a ByteBuffer are the layout under
// test. All four byte offsets are 0. The kernel asserts the host contract
// pieces that are visible to it (fp32 weights, block == 1). Buffers:
//   x = [16] fp32 input vector, weight = [16*8] fp32 row-major matrix,
//   bias = [8] fp32, y = [8] fp32 output.
// Host: dispatch(32) -- one warp; thread 0 performs the coop matvec.
using CoopProbeKernel = compute::Kernel1D<void(compute::ByteBuffer /*x*/,
                                                compute::ByteBuffer /*weight*/,
                                                compute::ByteBuffer /*bias*/,
                                                compute::ByteBuffer /*y*/)>;

[[nodiscard]] CoopProbeKernel make_coop_probe_kernel() noexcept;

// C2 coop matvec projection (plan §5.3):
//   y[M,N] = W[K,N] @ x[M,K]^T + bias[N], fp32, one thread per token group
// (TPB = cfg.tokens_per_block tokens per thread, C++-unrolled).
// Args: x [M*K], weight [K*N] canonical row-major fp32, bias [N], y [M*N],
//       then the runtime scalars M, N, K.
// Launch: block 256 threads (one block covers 256*TPB tokens);
//         dispatch.x = ceil(M / (256*TPB)) * 256 threads (lc_optimize trap #1:
//         dispatch counts TOTAL THREADS; use coop_gemm_dispatch_size below).
// KSLAB/NTILE/TPB are emission-time template constants baked by the factory
// from the table entry `cfg` (the factory LUISA_ASSERTs that `cfg` is one of
// the table entries and that ep is Bias or BiasGeluErf -- plan §5.3/§1.3;
// None/BiasRelu/ResidualAdd are not emitted by the coop v1 path).
using CoopGemmKernel = compute::Kernel1D<void(compute::ByteBuffer /*x*/,
                                               compute::ByteBuffer /*weight*/,
                                               compute::ByteBuffer /*bias*/,
                                               compute::ByteBuffer /*y*/,
                                               uint32_t /*M*/,
                                               uint32_t /*N*/,
                                               uint32_t /*K*/)>;

[[nodiscard]] CoopGemmKernel make_coop_gemm_kernel(
    GemmEpilogue ep, const CoopGemmConfig &cfg, const char *vname) noexcept;

// Overload that checks the visible format contract: only
// GemmWeightFormat::Fp32 may reach the coop path (plan §1.3: packed
// half/bf16 must NOT enter coop; §5.5 gates half formats behind fp32
// correctness).
[[nodiscard]] CoopGemmKernel make_coop_gemm_kernel(
    GemmEpilogue ep, const CoopGemmConfig &cfg, const char *vname,
    GemmWeightFormat wf) noexcept;

// Dispatch size of a coop matvec launch: x = TOTAL THREADS covering M tokens
// (1D launch; the block is fixed at kCoopGemmBlock).
[[nodiscard]] uint2 coop_gemm_dispatch_size_config(const CoopGemmConfig &cfg,
                                                   uint32_t m) noexcept;

}// namespace luisa::embed
