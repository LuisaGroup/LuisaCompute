#pragma once
// OCR engine -- GEMM / GEMV operators (Phase 2). Device-free DSL factories.
//
// C[M,N] = epilogue(A[M,K] @ B[K,N]). A row-major [M,K] FP32 activations,
// B row-major [K,N] (packer pre-transposes weights so reads coalesce along N),
// C row-major [M,N] FP32. Accumulation is FP32 in every variant. Arbitrary
// ragged M/N/K are handled by tail guards; nothing is clamped or dropped.
//
// Weight storage formats (B operand only; A/C/aux stay FP32):
//   fp32        B is Buffer<float> [K*N].
//   fp16/bf16   B is Buffer<uint> [ceil(K*N/2)], two halves per word (low first).
//   int4_sym/asym B is Buffer<uint> [ceil(K*N/8)], eight nibbles per word (low
//               first, Phase 1 convention) + scales Buffer<float>[groups] +
//               zero_points Buffer<uint>[groups] (asym only; sym zp=8).
//   int8_sym    B is Buffer<uint> [ceil(K*N/4)] + scales Buffer<float>[groups].
// Groups are contiguous along the flattened row-major K*N order; group size
// 64 or 128; the last group may be partial. Dequantization is storage-only:
// every FMA is FP32.
//
// W4A16 is an INDEPENDENT, explicitly named variant (make_gemm_w4a16_kernel).
// Phase 2 does NOT implement FP16 activation arithmetic, so that factory
// returns an unsupported Status -- it is never silently substituted by W4A32.
//
// Epilogues are compile-time (GemmEpilogue). aux carries bias[N] for the bias
// family, residual[M,N] for residual/bias_residual, and is unused (bind a
// 1-element dummy view) otherwise. SwiGLU/GeGLU are separate gated-activation
// kernels over the two halves of a fused projection (see make_gated_act_kernel).
//
// Launch: one thread per output element. plan_gemm_dispatch() returns the
// matching TOTAL-thread geometry (flattened 2D when >65535 groups). The decode
// GEMV/small-M path (make_gemv_kernel) parallelizes over N with one thread per
// column and loops K/M in-thread -- weight-bandwidth friendly for M=1.
#include <cstdint>

#include <luisa/dsl/syntax.h>

#include "ocr/runtime/dispatch.h"
#include "ocr/status.h"

namespace luisa::ocr {

enum class GemmEpilogue : uint32_t {
    none = 0,
    bias = 1,
    bias_gelu_erf = 2,
    bias_gelu_tanh = 3,
    bias_relu = 4,
    bias_relu6 = 5,
    bias_silu = 6,
    bias_hard_swish = 7,
    residual = 8,
    bias_residual = 9,
};

[[nodiscard]] luisa::string_view gemm_epilogue_name(GemmEpilogue epilogue) noexcept;

enum class GemmWeightFormat : uint32_t {
    fp32 = 0,
    fp16 = 1,
    bf16 = 2,
    int4_sym = 3,
    int4_asym = 4,
    int8_sym = 5,
};

[[nodiscard]] luisa::string_view gemm_weight_format_name(GemmWeightFormat format) noexcept;
[[nodiscard]] bool gemm_format_is_quantized(GemmWeightFormat format) noexcept;

// Dense FP32-B GEMM. aux layout: bias family = bias[N]; residual =
// residual[M,N]; bias_residual = [bias N | residual M*N] in ONE buffer;
// none = unused (bind a 1-element dummy view). padded_cols is the plan's
// in-kernel column stride (LaunchPlan2D::padded_cols) for flattened-id
// (row, col) recovery. Kernel args: A, B, C, aux, M, N, K, padded_cols.
using GemmKernel = compute::Kernel2D<void(
    compute::Buffer<float> /*A*/,
    compute::Buffer<float> /*B fp32 [K*N]*/,
    compute::Buffer<float> /*C*/,
    compute::Buffer<float> /*aux*/,
    uint32_t /*M*/, uint32_t /*N*/, uint32_t /*K*/, uint32_t /*padded_cols*/)>;

// Packed-B GEMM (fp16/bf16/int4/int8). B is uint code words; scales is
// Buffer<float>[groups] for quantized formats (dummy for fp16/bf16);
// zero_points is Buffer<uint>[groups] for int4_asym (dummy otherwise).
// Kernel args: A, B, C, aux, scales, zero_points, M, N, K, group_size.
using GemmPackedKernel = compute::Kernel2D<void(
    compute::Buffer<float> /*A*/,
    compute::Buffer<uint32_t> /*B packed words*/,
    compute::Buffer<float> /*C*/,
    compute::Buffer<float> /*aux*/,
    compute::Buffer<float> /*scales*/,
    compute::Buffer<uint32_t> /*zero_points*/,
    uint32_t /*M*/, uint32_t /*N*/, uint32_t /*K*/, uint32_t /*group_size*/,
    uint32_t /*padded_cols*/)>;

// FP32 B factory.
[[nodiscard]] GemmKernel make_gemm_kernel(GemmEpilogue epilogue) noexcept;
// Packed B factory (fp16/bf16/int4/int8). `format` must not be fp32.
[[nodiscard]] GemmPackedKernel make_gemm_packed_kernel(GemmEpilogue epilogue, GemmWeightFormat format) noexcept;

// W4A16: INDEPENDENT variant. Phase 2 has no FP16 activation arithmetic, so
// this returns unsupported. It is NEVER an alias for the W4A32 path.
[[nodiscard]] Result<GemmPackedKernel> make_gemm_w4a16_kernel(GemmEpilogue epilogue) noexcept;

// Decode GEMV / small-M: one thread per output column, loops K (and M) in
// thread. Same epilogue/format support as the prefill GEMM. M is a runtime
// arg; intended for M <= kGemvSmallM (host picks the path).
inline constexpr uint32_t kGemvSmallM = 4u;
using GemvKernel = compute::Kernel1D<void(
    compute::Buffer<float> /*A*/,
    compute::Buffer<float> /*B fp32*/,
    compute::Buffer<float> /*C*/,
    compute::Buffer<float> /*aux*/,
    uint32_t /*M*/, uint32_t /*N*/, uint32_t /*K*/)>;
using GemvPackedKernel = compute::Kernel1D<void(
    compute::Buffer<float> /*A*/,
    compute::Buffer<uint32_t> /*B packed*/,
    compute::Buffer<float> /*C*/,
    compute::Buffer<float> /*aux*/,
    compute::Buffer<float> /*scales*/,
    compute::Buffer<uint32_t> /*zero_points*/,
    uint32_t /*M*/, uint32_t /*N*/, uint32_t /*K*/, uint32_t /*group_size*/)>;

[[nodiscard]] GemvKernel make_gemv_kernel(GemmEpilogue epilogue) noexcept;
[[nodiscard]] GemvPackedKernel make_gemv_packed_kernel(GemmEpilogue epilogue, GemmWeightFormat format) noexcept;

// Gated activation: C[i] = act(gate[i]) * up[i] over count elements.
// act: 0=silu(SwiGLU), 1=gelu_erf, 2=gelu_tanh(GeGLU), 3=relu, 4=identity.
using GatedActKernel = compute::Kernel1D<void(
    compute::Buffer<float> /*gate*/,
    compute::Buffer<float> /*up*/,
    compute::Buffer<float> /*C*/,
    uint32_t /*count*/)>;
[[nodiscard]] GatedActKernel make_gated_act_kernel(uint32_t act) noexcept;

// Dispatch helpers (TOTAL threads; every factory has one). The GEMM/GEMV
// plans fail closed when ANY logical buffer element count (M*K, K*N, M*N)
// exceeds the uint32 kernel index envelope -- the kernels index with uint32,
// so a wrapping product would silently corrupt memory. Chunk M (rows) instead.
[[nodiscard]] Result<LaunchPlan2D> plan_gemm_dispatch(uint64_t m, uint64_t n, uint64_t k);
// Epilogue-aware variant: also fails closed when the aux element count
// (bias_residual = N + M*N) exceeds the uint32 index envelope.
[[nodiscard]] Result<LaunchPlan2D> plan_gemm_dispatch_epilogue(uint64_t m, uint64_t n, uint64_t k,
                                                               GemmEpilogue epilogue);
[[nodiscard]] Result<LaunchPlan1D> plan_gemv_dispatch(uint64_t m, uint64_t n, uint64_t k);
[[nodiscard]] Result<LaunchPlan1D> plan_gated_act_dispatch(uint64_t count);

// Checked aux-buffer element count for an epilogue: bias_residual carries
// [bias N | residual M*N] in ONE aux buffer (S6); residual carries M*N; the
// bias family carries N; none carries 0 (bind a dummy).
[[nodiscard]] Result<uint64_t> gemm_aux_elements(GemmEpilogue epilogue, uint64_t m, uint64_t n);

// Stable shader variant key for the cache.
[[nodiscard]] luisa::string gemm_variant_key(GemmEpilogue epilogue, GemmWeightFormat format, bool gemv);

}// namespace luisa::ocr