#pragma once
// OCR engine -- Gated Delta Net (GDN) operators (Phase 9).
// Device-free DSL factories + INDEPENDENT host CPU oracles. FP32 activations.
//
// ---------------------------------------------------------------------------
// DECLARED SYNTHETIC GDN CONTRACT (exact equations; no official checkpoint is
// pinned or validated -- see MODELS.md). One GDN layer over tokens t=0..T-1,
// H heads, key/value head dim d (key dim == value dim == d), hidden size Dh:
//
//   Projections (row-major packed [K,N] weights, FP32):
//     q = x @ Wq, k = x @ Wk, v = x @ Wv          [T, H*d] each
//     z = x @ Wz   (output gate)                  [T, H*d]
//     a = x @ Wa   (decay input)                  [T, H]
//     b = x @ Wb   (beta input)                   [T, H]
//
//   Causal depthwise Conv1D + SiLU, applied SEPARATELY to q, k, v (channel
//   groups of one logical [T, 3*H*d] stream; kernel size K, left zero padding
//   K-1, stride 1, dilation 1, groups == channels):
//     u_t[c] = bias[c] + sum_{j=0..K-1} w[c,j] * x_{t-(K-1)+j}[c]
//              (x at negative absolute positions == 0)
//     y_t[c] = silu(u_t[c]) = u_t[c] / (1 + exp(-u_t[c]))
//   RING STATE: per request/layer/group the last K-1 PRE-activation projected
//   values x are kept in ring[c, s], s = absolute_position mod (K-1). After
//   consuming absolute position p the kernel writes x_p[c] into slot
//   p mod (K-1). Reads of positions < chunk start come from the ring; reads
//   >= chunk start come from the source rows. The ring is updated in place.
//
//   L2 / QK normalization (REQUIRED, per head over d, floor eps):
//     q <- q / max(||q||_2, eps),  k <- k / max(||k||_2, eps)
//   (the shared ops_norm L2 kernel / ref_l2_norm; applied to [T*H, d] rows).
//
//   Gates (per token, per head):
//     beta_t = sigmoid(b_t) = 1 / (1 + exp(-b_t))
//     sp(x)  = x > 20 ? x : log(1 + exp(x))      (softplus, threshold 20)
//     g_t    = -exp(A_log[h]) * sp(a_t + dt_bias[h])   (log decay, <= 0)
//     alpha_t = exp(g_t)
//
//   Recurrent state S[h] is a [d, d] matrix, layout S[k_index, v_index]
//   row-major (state buffer [H, d, d] per layer/request). DELTA RULE:
//     S <- alpha_t * S
//     err_v = v_t[v] - sum_k S[k, v] * k_t[k]
//     S[k, v] <- S[k, v] + beta_t * k_t[k] * err_v
//     o_t[v] = sum_k S[k, v] * q_t[k]
//   The straight recurrent path applies this EXACTLY per token (prefill loops
//   tokens in order; decode is T=1). NO sequence x sequence attention matrix
//   is ever materialized on the recurrent path.
//
//   CHUNKED path (WY / UT-transform chunk algebra; correctness-first SERIAL
//   forward substitution inside a chunk -- explicitly NOT a parallel scan and
//   NOT a relabeled sequential loop; it is a different arithmetic path that
//   must match the recurrent path within tolerance). For a chunk of T tokens
//   with cumulative log decay c_i = sum_{j<=i} g_j (1-based i in the proof):
//     L[i,j] = beta_i * <k_i, k_j> * exp(c_i - c_j)   for j < i (else 0)
//     w_i = beta_i * exp(c_i) * k_i - sum_{j<i} L[i,j] * w_j
//     u_i = beta_i * v_i           - sum_{j<i} L[i,j] * u_j
//     ue_i = u_i - S_prev^T w_i                       (pseudo values)
//     S_new = exp(c_T) * S_prev + sum_i exp(c_T - c_i) * k_i ue_i^T
//     o_i   = exp(c_i) * (S_prev^T q_i)
//             + sum_{j<=i} <q_i, k_j> * exp(c_i - c_j) * ue_j
//   All exponentials have non-positive arguments (g <= 0), so the chunk form
//   is numerically safe in FP32 (underflow to 0 is the correct limit).
//   Scratch: w/u [H, T, d] each, c [H, T] (caller workspace; never retained).
//
//   Gated output normalization (per token, per head, over d):
//     y_t[h,:] = (o_t[h,:] / sqrt(mean(o_t[h,:]^2) + eps)) * norm_w[:]
//                * silu(z_t[h,:])
//   Output / residual path (executor): out = concat_heads(y) @ Wo + x.
//
// ---------------------------------------------------------------------------
// Layouts (all row-major FP32 unless noted):
//   q/k/v/z/out : [T, H, d] flattened (token-major)
//   a/b/beta/g  : [T, H]
//   conv weight : [C, K] (C = H*d per group), bias [C], ring [C, K-1]
//   A_log/dt_bias/norm_w : [H] / [H] / [d]
//   state       : [H, d, d] (S[k, v])
//
// Host validators reject malformed geometry BEFORE launch; kernels guard
// their tails and never read/write out of bounds. NaN inputs propagate as
// NaN (documented, tested) -- they never cause OOB access.
#include <cstdint>

#include <luisa/core/stl/vector.h>
#include <luisa/dsl/syntax.h>

#include "ocr/runtime/dispatch.h"
#include "ocr/runtime/hybrid_gdn_config.h"// GdnLayerConfig (host-safe shared geometry)
#include "ocr/status.h"

namespace luisa::ocr {

// GdnLayerConfig (heads / head_dim / conv kernel / epsilons) is declared in the
// host-safe runtime/hybrid_gdn_config.h (included above) so ops, cache, executor,
// and host model modules share ONE geometry contract.

// ---------------------------------------------------------------------------
// Device kernels (device-free factories).
// ---------------------------------------------------------------------------

// Causal depthwise Conv1D + SiLU with in-place ring-state update over T tokens
// starting at ABSOLUTE position pos0. One thread per channel; tokens are
// processed in order inside the thread (causal ring reads are safe: a slot is
// overwritten only AFTER the output that reads it has been computed).
// src/out [T, C], weight [C, K], bias [C], ring [C, K-1] (in/out).
using GdnConvSiluKernel = compute::Kernel1D<void(
    compute::Buffer<float> /*src [T,C] pre-activation projected values*/,
    compute::Buffer<float> /*weight [C,K]*/,
    compute::Buffer<float> /*bias [C]*/,
    compute::Buffer<float> /*ring [C,K-1] in/out*/,
    compute::Buffer<float> /*out [T,C]*/,
    uint32_t /*tokens*/, uint32_t /*channels*/, uint32_t /*kernel_size*/,
    uint32_t /*pos0*/)>;
[[nodiscard]] GdnConvSiluKernel make_gdn_conv_silu_kernel() noexcept;

// Beta/gate computation: beta = sigmoid(b), g = -exp(A_log)*softplus(a+dt_bias).
// a/b [T,H], A_log/dt_bias [H], beta/g out [T,H]. One thread per (token,head).
using GdnGatesKernel = compute::Kernel1D<void(
    compute::Buffer<float> /*a [T,H]*/,
    compute::Buffer<float> /*b [T,H]*/,
    compute::Buffer<float> /*A_log [H]*/,
    compute::Buffer<float> /*dt_bias [H]*/,
    compute::Buffer<float> /*beta out [T,H]*/,
    compute::Buffer<float> /*g out [T,H]*/,
    uint32_t /*tokens*/, uint32_t /*heads*/, float /*softplus_threshold*/)>;
[[nodiscard]] GdnGatesKernel make_gdn_gates_kernel() noexcept;

// Straight recurrent delta-rule update over T tokens (prefill or T=1 decode).
// q/k are ALREADY L2-normalized. One thread per head; state [H,d,d] updated
// in place; out [T,H,d]. No T x T matrix is ever formed.
using GdnRecurrentKernel = compute::Kernel1D<void(
    compute::Buffer<float> /*q [T,H,d] L2-normalized*/,
    compute::Buffer<float> /*k [T,H,d] L2-normalized*/,
    compute::Buffer<float> /*v [T,H,d]*/,
    compute::Buffer<float> /*beta [T,H]*/,
    compute::Buffer<float> /*g [T,H] log decay*/,
    compute::Buffer<float> /*state [H,d,d] in/out*/,
    compute::Buffer<float> /*out [T,H,d]*/,
    uint32_t /*tokens*/, uint32_t /*heads*/, uint32_t /*head_dim*/)>;
[[nodiscard]] GdnRecurrentKernel make_gdn_recurrent_kernel() noexcept;

// Chunked WY/UT-transform prefill over ONE chunk of T tokens (see the header
// equations). One thread per head; serial forward substitution inside the
// chunk (correctness-first; NOT a parallel scan). state [H,d,d] is the carried
// chunk-boundary state (in/out). Scratch w/u [H,T,d], c [H,T] are caller
// workspace, fully overwritten before use.
using GdnChunkKernel = compute::Kernel1D<void(
    compute::Buffer<float> /*q [T,H,d] L2-normalized*/,
    compute::Buffer<float> /*k [T,H,d] L2-normalized*/,
    compute::Buffer<float> /*v [T,H,d]*/,
    compute::Buffer<float> /*beta [T,H]*/,
    compute::Buffer<float> /*g [T,H]*/,
    compute::Buffer<float> /*state [H,d,d] in/out*/,
    compute::Buffer<float> /*out [T,H,d]*/,
    compute::Buffer<float> /*w scratch [H,T,d]*/,
    compute::Buffer<float> /*u scratch [H,T,d]*/,
    compute::Buffer<float> /*c scratch [H,T]*/,
    uint32_t /*tokens*/, uint32_t /*heads*/, uint32_t /*head_dim*/)>;
[[nodiscard]] GdnChunkKernel make_gdn_chunk_kernel() noexcept;

// Gated output RMSNorm: y = o/sqrt(mean(o^2)+eps) * norm_w * silu(z), per
// (token, head) over d. o/z/out [T,H,d], norm_w [d].
using GdnGatedNormKernel = compute::Kernel1D<void(
    compute::Buffer<float> /*o [T,H,d]*/,
    compute::Buffer<float> /*z [T,H,d]*/,
    compute::Buffer<float> /*norm_w [d]*/,
    compute::Buffer<float> /*out [T,H,d]*/,
    uint32_t /*tokens*/, uint32_t /*heads*/, uint32_t /*head_dim*/, float /*eps*/)>;
[[nodiscard]] GdnGatedNormKernel make_gdn_gated_norm_kernel() noexcept;

// Shared elementwise utilities (used by the GDN executor/cache and the Surya
// detector head): zero-fill and sigmoid.
using ZeroFillKernel = compute::Kernel1D<void(
    compute::Buffer<float> /*out*/, uint32_t /*count*/)>;
[[nodiscard]] ZeroFillKernel make_zero_fill_kernel() noexcept;

using ElementwiseSigmoidKernel = compute::Kernel1D<void(
    compute::Buffer<float> /*x*/, compute::Buffer<float> /*out*/, uint32_t /*count*/)>;
[[nodiscard]] ElementwiseSigmoidKernel make_elementwise_sigmoid_kernel() noexcept;

// Dispatch helpers (TOTAL threads; every factory has one).
[[nodiscard]] Result<LaunchPlan1D> plan_gdn_conv_dispatch(uint64_t channels);
[[nodiscard]] Result<LaunchPlan1D> plan_gdn_gates_dispatch(uint64_t tokens, uint64_t heads);
[[nodiscard]] Result<LaunchPlan1D> plan_gdn_heads_dispatch(uint64_t heads);
[[nodiscard]] Result<LaunchPlan1D> plan_gdn_gated_norm_dispatch(uint64_t tokens, uint64_t heads);
[[nodiscard]] Result<LaunchPlan1D> plan_gdn_elementwise_dispatch(uint64_t count);

// Host-side geometry validation before launch (recoverable).
[[nodiscard]] Status validate_gdn_conv_inputs(uint64_t tokens, uint64_t channels,
                                              uint64_t kernel_size, uint64_t pos0);
[[nodiscard]] Status validate_gdn_state_inputs(uint64_t tokens, uint64_t heads, uint64_t head_dim);

[[nodiscard]] luisa::string gdn_variant_key(luisa::string_view op);

// ---------------------------------------------------------------------------
// INDEPENDENT host CPU oracles (double accumulation where it matters; defined
// in ops_gdn.cpp; NO device/DSL calls). They validate sizes and never read
// OOB. The chunk oracle builds the FULL T x T L matrix and solves it by
// generic forward substitution -- deliberately a different code path from the
// device's fused per-head loops.
// ---------------------------------------------------------------------------

// Conv+SiLU with ring semantics. ring is [C, K-1] in/out; pos0 is the absolute
// start position. out [T, C]. Returns the updated ring in `ring`.
[[nodiscard]] Status ref_gdn_conv_silu(luisa::span<const float> src, luisa::span<const float> weight,
                                       luisa::span<const float> bias, luisa::span<float> ring,
                                       uint32_t tokens, uint32_t channels, uint32_t kernel_size,
                                       uint32_t pos0, luisa::vector<float> &out);

// Gates. a/b [T,H]; A_log/dt_bias [H]; beta/g out [T,H].
[[nodiscard]] Status ref_gdn_gates(luisa::span<const float> a, luisa::span<const float> b,
                                   luisa::span<const float> a_log, luisa::span<const float> dt_bias,
                                   uint32_t tokens, uint32_t heads, float softplus_threshold,
                                   luisa::vector<float> &beta, luisa::vector<float> &g);

// Straight recurrent delta rule. q/k L2-normalized [T,H,d]; state [H,d,d]
// in/out; out [T,H,d]. When `state_snapshots` is non-null it receives the
// state AFTER EVERY token ((T+1) * H*d*d entries, snapshot 0 = input state).
[[nodiscard]] Status ref_gdn_recurrent(luisa::span<const float> q, luisa::span<const float> k,
                                       luisa::span<const float> v, luisa::span<const float> beta,
                                       luisa::span<const float> g, luisa::span<float> state,
                                       uint32_t tokens, uint32_t heads, uint32_t head_dim,
                                       luisa::vector<float> &out,
                                       luisa::vector<float> *state_snapshots = nullptr);

// Chunked WY/UT oracle over ONE chunk (full L matrix + forward substitution).
// Same layouts as ref_gdn_recurrent; state in/out is the chunk-boundary state.
[[nodiscard]] Status ref_gdn_chunk(luisa::span<const float> q, luisa::span<const float> k,
                                   luisa::span<const float> v, luisa::span<const float> beta,
                                   luisa::span<const float> g, luisa::span<float> state,
                                   uint32_t tokens, uint32_t heads, uint32_t head_dim,
                                   luisa::vector<float> &out);

// Gated output RMSNorm. o/z [T,H,d], norm_w [d]; out [T,H,d].
[[nodiscard]] Status ref_gdn_gated_norm(luisa::span<const float> o, luisa::span<const float> z,
                                        luisa::span<const float> norm_w, uint32_t tokens,
                                        uint32_t heads, uint32_t head_dim, float eps,
                                        luisa::vector<float> &out);

}// namespace luisa::ocr
