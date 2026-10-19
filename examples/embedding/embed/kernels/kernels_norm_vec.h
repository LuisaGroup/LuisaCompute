// bge-m3 embedding engine -- vec4 (float4) variants of the normalization
// kernels (tensor_optimize.md §4.1 V0 vec4 infrastructure + §4.2 V1 bandwidth
// kernels). These are the load-width-4 twins of kernels_norm.{h,cpp}: identical
// semantics, identical block reduction (shared-memory halving tree), identical
// flattened-block-id geometry and identical per-row index math -- the ONLY
// change is that each thread reads/writes one float4 (16 B) per loop step
// instead of one float, widening the coalesced access to the bus width.
//
// HOST CONTRACT (the caller MUST guarantee both -- these kernels do not
// self-guard, so a violation reads/writes out of range or misaligns):
//   1. 16-byte alignment. x/y/gamma/beta/hidden/out/a/b are bound as
//      Buffer<float4> (or a BufferView<float>::as<float4>() reinterpretation);
//      the device address of every element-0 of every bound buffer must be a
//      multiple of 16 B. LuisaCompute float4 buffers created with
//      device.create_buffer<float4> satisfy this; a BufferView<float> rebind
//      must start on a 16-B-aligned element offset.
//   2. Length divisible by 4. For layernorm/l2_normalize/dense_pool the row
//      length H must obey H % 4 == 0 (so H4 = H/4 vec4 lanes cover it exactly);
//      for residual_add the element count must obey count % 4 == 0 (the caller
//      passes count4 = count/4). Ragged shapes (any dimension not a multiple of
//      4) stay on the legacy scalar kernels -- a tail fast/slow split is NOT
//      implemented here (tensor_optimize.md §4.1: the engine shapes H=1024,
//      head_dim=64 are all multiples of 4 and take the no-tail path).
//   Recommended host-side enforcement:
//       LUISA_ASSERT(H % 4u == 0u, "layernorm_vec4: H must be a multiple of 4");
//       LUISA_ASSERT((ptr & 15u) == 0u, "vec4 buffers must be 16B-aligned");
//
// Reduction-order note (kernels.md §7 / tensor_optimize.md §4.2): widening the
// load to float4 changes the per-thread partial SET (thread tx now owns vec4
// lanes {tx, tx+bs4, ...}, each summed lane-wise with the PINNED horizontal
// order (acc4.x + acc4.y) + (acc4.z + acc4.w)) and therefore the summation
// order versus the scalar legacy kernels. The block tree itself is unchanged
// (same kNormVecBlock-wide halving strides, same barrier discipline). These
// vec4 kernels are validated against the existing numeric tolerance gate
// (<=1e-4 / <=2e-4); the legacy scalar kernels remain the bit-identity
// regression anchor. dot() is deliberately NOT used here -- the lane-wise
// accumulation keeps this the vec4_io-flavored variant.
//
// Device-free Kernel1D factories per CONVENTIONS.md / kernels.md §0: the caller
// compiles each with device.compile(...) and caches the Shader; dimensions are
// runtime kernel args so one compilation serves every shape. dispatch() counts
// TOTAL THREADS (lc_optimize trap #1); rows/blocks come from a flattened block
// index so a host that must dodge the DX 65535-group cap can dispatch a 2D grid
// without changing the kernels (lc_optimize trap #2, same convention as
// kernels_norm.cpp / kernels_embed.cpp).
#pragma once
#include <luisa/dsl/syntax.h>

namespace luisa::embed {

// Thread block width of the vec4 normalization kernels (mirrors kNormBlock).
inline constexpr uint kNormVecBlock = 256u;

// layernorm (vec4): in-place twin of make_layernorm_kernel(). Reads row `row`
// of x [rows*H] as float4, two-pass fp32 mean / biased variance over the row,
// then writes back into x with y = ((x - mean) * (1/sqrt(var + eps))) * gamma +
// beta, all elementwise float4 math. NOTE the vec4 API binds x (in: read; out:
// written back), gamma [H4], beta [H4] -- there is no separate y buffer; the
// normalization is written into x. H is the element row length (H % 4 == 0);
// H4 = H/4 vec4 lanes. eps is the runtime layer_norm_eps (1e-5 for bge-m3).
using LayerNormVec4Kernel = compute::Kernel1D<void(compute::Buffer<float4> /*x*/,
                                                   compute::Buffer<float4> /*gamma*/,
                                                   compute::Buffer<float4> /*beta*/,
                                                   uint32_t /*H (elements; H%4==0)*/,
                                                   float /*eps*/)>;

[[nodiscard]] LayerNormVec4Kernel make_layernorm_vec4_kernel() noexcept;

// l2_normalize (vec4): y[row] = x[row] / max(||x[row]||_2, 1e-12). One block per
// row. Buffers x [rows*H], y [rows*H] as float4. H is the element row length
// (H % 4 == 0); H4 = H/4. The 1e-12 clamp (on the norm, after sqrt) is kept.
using L2NormalizeVec4Kernel = compute::Kernel1D<void(compute::Buffer<float4> /*x*/,
                                                     compute::Buffer<float4> /*y*/,
                                                     uint32_t /*H (elements; H%4==0)*/)>;

[[nodiscard]] L2NormalizeVec4Kernel make_l2_normalize_vec4_kernel() noexcept;

// dense_pool (vec4): fused CLS-slice + L2-normalize twin of make_dense_pool_kernel().
// One block per batch row b: L2-normalizes hidden[b*S .. b*S+1) (the CLS row of
// the [B*S, H] final hidden state, indexed in float4 units) into out[b]. Row
// stride of hidden is S*H elements = S*H4 float4. Buffers hidden [B*S*H], out
// [B*H] as float4. Scalars S (seq len), H (element row length; H % 4 == 0).
// Same 1e-12 norm clamp as l2_normalize.
using DensePoolVec4Kernel = compute::Kernel1D<void(compute::Buffer<float4> /*hidden*/,
                                                   compute::Buffer<float4> /*out*/,
                                                   uint32_t /*S*/,
                                                   uint32_t /*H (elements; H%4==0)*/)>;

[[nodiscard]] DensePoolVec4Kernel make_dense_pool_vec4_kernel() noexcept;

// residual_add (vec4): elementwise out[i] = a[i] + b[i] over count4 float4
// elements (twin of make_residual_add_kernel() in kernels_gemm.cpp, but widened).
// Buffers a, b, out [count4 float4]. Scalar count4 = count/4 (the vec4 element
// count; the caller guarantees the logical count is a multiple of 4). Block
// kNormVecBlock, dispatch(count4) = TOTAL THREADS; tail guard against the
// possibly-over-dispatched last block.
using ResidualAddVec4Kernel = compute::Kernel1D<void(compute::Buffer<float4> /*a*/,
                                                     compute::Buffer<float4> /*b*/,
                                                     compute::Buffer<float4> /*out*/,
                                                     uint32_t /*count4 (vec4 count)*/)>;

[[nodiscard]] ResidualAddVec4Kernel make_residual_add_vec4_kernel() noexcept;

}// namespace luisa::embed
