#pragma once
// OCR engine -- independent naive host oracles (Phase 2). Host-only.
//
// Every device operator has a matching reference here. These are deliberately
// INDEPENDENT naive implementations (double accumulation where it matters),
// NOT calls into the device helpers, so a test compares two separate code
// paths. Tolerances are defined per family in the tests; the references
// themselves are exact-to-double then rounded to float.
//
// All references take luisa::span inputs and write into caller-provided
// luisa::vector outputs. They validate sizes and return Status on malformed
// input (never read OOB). No device/DSL headers are included.
#include <cstdint>

#include <luisa/core/stl/memory.h>
#include <luisa/core/stl/vector.h>

#include "ocr/quantization.h"
#include "ocr/status.h"

namespace luisa::ocr {

// ---------------------------------------------------------------------------
// GEMM / GEMV
// ---------------------------------------------------------------------------

// Epilogue tags mirrored from ops_gemm.h (kept here as plain ints so the host
// oracle header does not include DSL headers).
enum class RefGemmEpilogue : uint32_t {
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

// C[M,N] = epilogue(A[M,K] @ B[K,N]). A/B row-major. aux layout by epilogue:
// bias family = bias[N]; residual = residual[M*N]; bias_residual =
// [bias N | residual M*N] concatenated in ONE buffer; none = empty.
// Double accumulate.
[[nodiscard]] Status ref_gemm(luisa::span<const float> a, luisa::span<const float> b,
                              luisa::span<const float> aux, uint32_t m, uint32_t n, uint32_t k,
                              RefGemmEpilogue epilogue, luisa::vector<float> &out);

// Gated epilogues: C[M,N] = act(gate[M,N]) * up[M,N] where gate/up are two
// [M,N] halves (e.g. the two halves of a SwiGLU projection). act: 0=silu,
// 1=gelu_erf, 2=gelu_tanh, 3=relu, 4=identity(GeGLU uses gelu). Provided as a
// separate reference so the device SwiGLU/GeGLU epilogue is checked against an
// independent elementwise oracle.
[[nodiscard]] Status ref_gated_act(luisa::span<const float> gate, luisa::span<const float> up,
                                   uint32_t act, luisa::vector<float> &out);

// Dequantize grouped INT4 (Phase 1 semantics) to FP32 for the oracle arm.
[[nodiscard]] Status ref_dequant_int4(luisa::span<const uint8_t> packed, uint64_t count,
                                      luisa::span<const float> scales, luisa::span<const uint8_t> zero_points,
                                      uint32_t group_size, bool symmetric, luisa::vector<float> &out);
[[nodiscard]] Status ref_dequant_int8(luisa::span<const int8_t> codes, uint64_t count,
                                      luisa::span<const float> scales, uint32_t group_size,
                                      luisa::vector<float> &out);
// Round FP32 -> fp16/bf16 -> FP32 (storage rounding) for the packed-B oracle.
[[nodiscard]] Status ref_round_half(luisa::span<const float> values, bool bf16, luisa::vector<float> &out);

// ---------------------------------------------------------------------------
// Normalization / reductions
// ---------------------------------------------------------------------------

// LayerNorm over the last axis: y = (x-mean)/sqrt(var+eps) * gamma + beta.
// gamma/beta empty = parameterless. rows x H. Double mean/var.
[[nodiscard]] Status ref_layer_norm(luisa::span<const float> x, luisa::span<const float> gamma,
                                    luisa::span<const float> beta, uint32_t rows, uint32_t h,
                                    float eps, luisa::vector<float> &out);
// RMSNorm: y = x / sqrt(mean(x^2)+eps) * gamma. gamma empty = parameterless.
[[nodiscard]] Status ref_rms_norm(luisa::span<const float> x, luisa::span<const float> gamma,
                                  uint32_t rows, uint32_t h, float eps, luisa::vector<float> &out);
// L2 normalize rows: y = x / max(||x||, floor).
[[nodiscard]] Status ref_l2_norm(luisa::span<const float> x, uint32_t rows, uint32_t h,
                                 float floor, luisa::vector<float> &out);
// Row sums (FP32 reduction oracle, double accumulate).
[[nodiscard]] Status ref_row_sum(luisa::span<const float> x, uint32_t rows, uint32_t h,
                                 luisa::vector<float> &out);
// Residual add: out = a + b elementwise.
[[nodiscard]] Status ref_residual_add(luisa::span<const float> a, luisa::span<const float> b,
                                      luisa::vector<float> &out);

// ---------------------------------------------------------------------------
// RoPE
// ---------------------------------------------------------------------------

// 1D RoPE over [tokens, heads, head_dim]. positions[t] gives the position.
// inv_freq[d] = theta^(-2d/head_dim) for d in [0, head_dim/2). Pairs (2d,2d+1).
[[nodiscard]] Status ref_rope_1d(luisa::span<const float> x, luisa::span<const int32_t> positions,
                                 uint32_t tokens, uint32_t heads, uint32_t head_dim, float theta,
                                 luisa::vector<float> &out);
// Rotate-half 1D RoPE (Qwen2.5 HF text convention): pairs (d, d+head_dim/2)
// instead of (2d,2d+1), one position axis. Same inv_freq as ref_rope_1d.
[[nodiscard]] Status ref_rope_rotate_half(luisa::span<const float> x, luisa::span<const int32_t> positions,
                                          uint32_t tokens, uint32_t heads, uint32_t head_dim, float theta,
                                          luisa::vector<float> &out);
// Generic axial RoPE: each pair d uses axis axis_of[d] and position
// positions[t*axes + axis]. head_dim must be even; axis count arbitrary.
[[nodiscard]] Status ref_rope_axial(luisa::span<const float> x, luisa::span<const int32_t> positions,
                                    luisa::span<const uint32_t> axis_of_pair, uint32_t tokens,
                                    uint32_t heads, uint32_t head_dim, uint32_t axes, float theta,
                                    luisa::vector<float> &out);
// Interleaved RoPE: pairs are (d, d + head_dim/2) instead of (2d, 2d+1).
[[nodiscard]] Status ref_rope_interleaved(luisa::span<const float> x, luisa::span<const int32_t> positions,
                                          luisa::span<const uint32_t> axis_of_pair, uint32_t tokens,
                                          uint32_t heads, uint32_t head_dim, uint32_t axes, float theta,
                                          luisa::vector<float> &out);
// Additive Phase 7 partial rotary (GLM style): pairs over the FIRST rotary_dim
// elements of each head; inv_freq[d] = theta^(-2d/rotary_dim) uses the ROTATED
// dimension; the tail [rotary_dim, head_dim) is copied unchanged. axis_of_pair
// has rotary_dim/2 entries. rotary_dim must be even and <= head_dim.
// adjacent: pairs (2d, 2d+1); rotate_half: pairs (d, d+rotary_dim/2) within
// the rotated prefix.
[[nodiscard]] Status ref_rope_partial_adjacent(luisa::span<const float> x, luisa::span<const int32_t> positions,
                                               luisa::span<const uint32_t> axis_of_pair, uint32_t tokens,
                                               uint32_t heads, uint32_t head_dim, uint32_t rotary_dim,
                                               uint32_t axes, float theta, luisa::vector<float> &out);
[[nodiscard]] Status ref_rope_partial_rotate_half(luisa::span<const float> x, luisa::span<const int32_t> positions,
                                                  luisa::span<const uint32_t> axis_of_pair, uint32_t tokens,
                                                  uint32_t heads, uint32_t head_dim, uint32_t rotary_dim,
                                                  uint32_t axes, float theta, luisa::vector<float> &out);
// FP32-ANGLE 1D RoPE reference (S8): mirrors the device evaluation order in
// float -- inv_freq = exp2f(log2f(theta) * (-2d/head_dim)), angle =
// float(pos) * inv_freq, cosf/sinf -- instead of double-precision angles.
// For large positions (~1e6) the double-angle oracle is an impossible gate:
// the device legitimately computes a different (fp32-rounded) angle. Use
// THIS reference for long-context tests with a tight tolerance.
[[nodiscard]] Status ref_rope_1d_fp32_angle(luisa::span<const float> x, luisa::span<const int32_t> positions,
                                            uint32_t tokens, uint32_t heads, uint32_t head_dim, float theta,
                                            luisa::vector<float> &out);

// Sinusoidal position table [len, dim] (dim even).
[[nodiscard]] Status ref_sinusoidal_table(uint32_t len, uint32_t dim, float theta,
                                          luisa::vector<float> &out);

// ---------------------------------------------------------------------------
// Attention
// ---------------------------------------------------------------------------

// Mask kinds mirrored from ops_attention.h.
enum class RefMaskKind : uint32_t {
    bidirectional = 0,
    causal = 1,
    windowed = 2,
    prefix_causal = 3,
    custom_block = 4,
};

struct RefAttentionMask {
    RefMaskKind kind{RefMaskKind::bidirectional};
    uint32_t window{0};        // windowed: attend [q-window+1, q]
    uint32_t prefix_len{0};    // prefix_causal: keys [0,prefix_len) fully visible
    uint32_t block_size{0};    // custom_block: block granularity
    luisa::vector<uint8_t> block_mask;// custom_block: [num_q_blocks*num_k_blocks] 0/1
    // Optional additive relative bias [num_q_heads, q_len, k_len] (empty=none).
    luisa::vector<float> rel_bias;
};

// Reference packed GQA/MHA attention. q [q_len, q_heads, head_dim],
// k/v [k_len, kv_heads, head_dim]. scale multiplies QK^T; softcap>0 applies
// tanh softcap. FP32 softmax (double accumulate). out [q_len, q_heads, head_dim].
[[nodiscard]] Status ref_attention(luisa::span<const float> q, luisa::span<const float> k,
                                   luisa::span<const float> v, uint32_t q_len, uint32_t k_len,
                                   uint32_t q_heads, uint32_t kv_heads, uint32_t head_dim,
                                   float scale, float softcap, const RefAttentionMask &mask,
                                   luisa::vector<float> &out);
// Decode attention: single query per batch over contiguous KV. q [batch,
// q_heads, head_dim], k/v [batch, kv_capacity, kv_heads, head_dim], kv_len
// per batch. positions[batch] = absolute query position for causal masking.
[[nodiscard]] Status ref_attention_decode(luisa::span<const float> q, luisa::span<const float> k,
                                          luisa::span<const float> v, luisa::span<const uint32_t> kv_len,
                                          luisa::span<const int32_t> positions, uint32_t batch,
                                          uint32_t kv_capacity, uint32_t q_heads, uint32_t kv_heads,
                                          uint32_t head_dim, float scale, float softcap,
                                          const RefAttentionMask &mask, luisa::vector<float> &out);

// ---------------------------------------------------------------------------
// Image / conv / vision / decode
// ---------------------------------------------------------------------------

// Bilinear resize + normalize matching the Phase 1 preprocess convention
// (half-pixel centers, clamp edges). src HWC uint8, dst CHW float.
[[nodiscard]] Status ref_image_resize_normalize(luisa::span<const uint8_t> src, uint32_t sw, uint32_t sh,
                                                uint32_t dw, uint32_t dh, bool bgr_to_rgb, bool bilinear,
                                                luisa::span<const float> mean, luisa::span<const float> std_dev,
                                                luisa::vector<float> &out);
// Conv2D NCHW. weight [out_c, in_c/groups, kh, kw], bias [out_c] (empty=none).
[[nodiscard]] Status ref_conv2d(luisa::span<const float> x, luisa::span<const float> w,
                                luisa::span<const float> bias, uint32_t n, uint32_t in_c, uint32_t h,
                                uint32_t w_in, uint32_t out_c, uint32_t kh, uint32_t kw,
                                uint32_t stride_h, uint32_t stride_w, uint32_t pad_h, uint32_t pad_w,
                                uint32_t dil_h, uint32_t dil_w, uint32_t groups,
                                luisa::vector<float> &out, uint32_t &out_h, uint32_t &out_w);
// Conv1D NCL (causal depthwise when causal=true pads left only).
[[nodiscard]] Status ref_conv1d(luisa::span<const float> x, luisa::span<const float> w,
                                luisa::span<const float> bias, uint32_t n, uint32_t c, uint32_t len,
                                uint32_t out_c, uint32_t k, uint32_t stride, uint32_t pad,
                                uint32_t dilation, uint32_t groups, bool causal,
                                luisa::vector<float> &out, uint32_t &out_len);
// BatchNorm inference: y = (x-mean)/sqrt(var+eps)*gamma+beta per channel (NCHW).
[[nodiscard]] Status ref_batch_norm(luisa::span<const float> x, luisa::span<const float> mean,
                                    luisa::span<const float> var, luisa::span<const float> gamma,
                                    luisa::span<const float> beta, uint32_t n, uint32_t c,
                                    uint32_t spatial, float eps, luisa::vector<float> &out);
// Average pool NCHW with kernel/stride/pad (count_include_pad=false).
[[nodiscard]] Status ref_avg_pool2d(luisa::span<const float> x, uint32_t n, uint32_t c, uint32_t h,
                                    uint32_t w, uint32_t kh, uint32_t kw, uint32_t sh, uint32_t sw,
                                    uint32_t ph, uint32_t pw, luisa::vector<float> &out,
                                    uint32_t &out_h, uint32_t &out_w);
// Adaptive average pool NCHW to [out_h, out_w].
[[nodiscard]] Status ref_adaptive_avg_pool2d(luisa::span<const float> x, uint32_t n, uint32_t c,
                                             uint32_t h, uint32_t w, uint32_t out_h, uint32_t out_w,
                                             luisa::vector<float> &out);
// Pixel unshuffle NCHW: [N,C,H,W] -> [N,C*r*r,H/r,W/r].
[[nodiscard]] Status ref_pixel_unshuffle(luisa::span<const float> x, uint32_t n, uint32_t c,
                                         uint32_t h, uint32_t w, uint32_t r, luisa::vector<float> &out);
// Concat along channel axis (NCHW). a/b same N,H,W.
[[nodiscard]] Status ref_concat_channels(luisa::span<const float> a, luisa::span<const float> b,
                                         uint32_t n, uint32_t ca, uint32_t cb, uint32_t spatial,
                                         luisa::vector<float> &out);
// Channel shuffle NCHW: reshape C into [groups, C/groups] and transpose.
[[nodiscard]] Status ref_channel_shuffle(luisa::span<const float> x, uint32_t n, uint32_t c,
                                         uint32_t spatial, uint32_t groups, luisa::vector<float> &out);
// Embedding lookup: out[t,:] = table[ids[t],:]. table [vocab, hidden].
[[nodiscard]] Status ref_embedding_lookup(luisa::span<const float> table, luisa::span<const int32_t> ids,
                                          uint32_t vocab, uint32_t hidden, luisa::vector<float> &out);
// Multimodal scatter: replace rows [start, start+count) of embeddings with
// vision features. embeddings [seq, hidden], features [count, hidden].
[[nodiscard]] Status ref_mm_scatter(luisa::span<const float> embeddings, luisa::span<const float> features,
                                    uint32_t seq, uint32_t hidden, uint32_t start,
                                    luisa::vector<float> &out);
// Greedy argmax per row: out[row] = argmax(logits[row,:]).
[[nodiscard]] Status ref_argmax(luisa::span<const float> logits, uint32_t rows, uint32_t cols,
                                luisa::vector<int32_t> &out);
// Contiguous-KV append (Phase 3): copy `count` packed [count, kv_dim] rows
// into a [capacity, kv_dim] layer at token offset `start`. dst_k/dst_v are
// the FULL layer storage (capacity*kv_dim); only rows [start,start+count)
// are written. Independent of the device kernel.
[[nodiscard]] Status ref_kv_append(luisa::span<const float> src_k, luisa::span<const float> src_v,
                                   luisa::span<float> dst_k, luisa::span<float> dst_v,
                                   uint32_t count, uint32_t kv_dim, uint32_t start, uint32_t capacity);
// Softmax rows (double accumulate) for sampling building blocks.
[[nodiscard]] Status ref_softmax(luisa::span<const float> logits, uint32_t rows, uint32_t cols,
                                 float temperature, luisa::vector<float> &out);
// Add absolute position embeddings: out[t] = x[t] + pos[pos_ids[t]].
[[nodiscard]] Status ref_add_position(luisa::span<const float> x, luisa::span<const float> pos,
                                      luisa::span<const int32_t> pos_ids, uint32_t tokens,
                                      uint32_t hidden, luisa::vector<float> &out);
// DeepStack injection: out[t] = x[t] + proj[t] for tokens in [start,start+count).
[[nodiscard]] Status ref_deepstack_inject(luisa::span<const float> x, luisa::span<const float> proj,
                                          uint32_t tokens, uint32_t hidden, uint32_t start,
                                          luisa::vector<float> &out);

}// namespace luisa::ocr