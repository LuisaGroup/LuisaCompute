#pragma once
// OCR engine -- vision connector / composition operators (Phase 2).
// Device-free DSL factories; FP32 activations.
//
// Reusable primitives for vision towers and connectors (NOT a model graph):
//   pixel_unshuffle   : NCHW [N,C,H,W] -> [N,C*r*r,H/r,W/r] (MinerU 2x2 etc.).
//   spatial_merge     : token packing [T, H] with merge_size m -> [T/(m*m),
//                       H*m*m] by grouping m*m spatially-adjacent tokens
//                       (row-major within each m x m block). Token downsampling.
//   add_position      : absolute/learned position addition out[t]=x[t]+pos[id[t]].
//   deepstack_project : per-layer projection primitive (a GEMM wrapper is the
//                       caller's job; here we provide the injection).
//   deepstack_inject  : residual injection out[t]=x[t]+proj[t] for tokens in
//                       [start,start+count). The selected decoder layer indices
//                       are CALLER DATA (start/count per layer), never hardcoded
//                       Qwen2 behavior.
//   mlp_connector     : two-layer MLP with GELU (hidden->inter->hidden) composed
//                       from ops_gemm + gated act; provided as a host helper.
// Adaptive pooling lives in ops_conv; patch projection is ops_gemm over
// patchified tokens (ops_image patchify). Full/window block composition is a
// host-side command-recording concern (later executor phase), built from these
// primitives + ops_attention/ops_norm/ops_gemm.
#include <cstdint>

#include <luisa/dsl/syntax.h>

#include "ocr/runtime/dispatch.h"

namespace luisa::ocr {

// Pixel unshuffle NCHW: [n,c,h,w] -> [n,c*r*r,h/r,w/r]. h,w divisible by r.
using PixelUnshuffleKernel = compute::Kernel1D<void(
    compute::Buffer<float> /*x*/,
    compute::Buffer<float> /*out*/,
    uint32_t /*n*/, uint32_t /*c*/, uint32_t /*h*/, uint32_t /*w*/, uint32_t /*r*/)>;
[[nodiscard]] PixelUnshuffleKernel make_pixel_unshuffle_kernel() noexcept;

// Spatial merge / token packing: [T, H] -> [T/(m*m), H*m*m]. Tokens are in
// row-major spatial order with `tokens_w` tokens per row; each output token
// concatenates the m*m tokens of an m x m block in (by, bx) order. T must be
// divisible by m*m and tokens_w by m.
using SpatialMergeKernel = compute::Kernel1D<void(
    compute::Buffer<float> /*x [T,H]*/,
    compute::Buffer<float> /*out [T/(m*m), H*m*m]*/,
    uint32_t /*T*/, uint32_t /*H*/, uint32_t /*tokens_w*/, uint32_t /*m*/)>;
[[nodiscard]] SpatialMergeKernel make_spatial_merge_kernel() noexcept;

// Absolute/learned position addition: out[t] = x[t] + pos[pos_ids[t]].
using AddPositionKernel = compute::Kernel1D<void(
    compute::Buffer<float> /*x [tokens,hidden]*/,
    compute::Buffer<float> /*pos [pos_rows,hidden]*/,
    compute::Buffer<int32_t> /*pos_ids [tokens]*/,
    compute::Buffer<float> /*out*/,
    uint32_t /*tokens*/, uint32_t /*hidden*/)>;
[[nodiscard]] AddPositionKernel make_add_position_kernel() noexcept;

// Additive Phase 7: 2D transpose [rows, cols] -> [cols, rows] (row-major both).
// Used by the MinerU pixel-unshuffle connector to turn the channel-major
// unshuffle output [C*r*r, tokens] into token rows [tokens, C*r*r].
using Transpose2dKernel = compute::Kernel1D<void(
    compute::Buffer<float> /*x [rows,cols]*/,
    compute::Buffer<float> /*out [cols,rows]*/,
    uint32_t /*rows*/, uint32_t /*cols*/)>;
[[nodiscard]] Transpose2dKernel make_transpose_2d_kernel() noexcept;

// DeepStack residual injection: out[t] = x[t] + proj[t-start] for t in
// [start, start+count); other tokens copied. proj is [count, hidden].
using DeepStackInjectKernel = compute::Kernel1D<void(
    compute::Buffer<float> /*x [tokens,hidden]*/,
    compute::Buffer<float> /*proj [count,hidden]*/,
    compute::Buffer<float> /*out*/,
    uint32_t /*tokens*/, uint32_t /*hidden*/, uint32_t /*start*/, uint32_t /*count*/)>;
[[nodiscard]] DeepStackInjectKernel make_deepstack_inject_kernel() noexcept;

// Dispatch helpers.
[[nodiscard]] Result<LaunchPlan1D> plan_pixel_unshuffle_dispatch(uint64_t n, uint64_t c, uint64_t h, uint64_t w);
[[nodiscard]] Result<LaunchPlan1D> plan_spatial_merge_dispatch(uint64_t out_tokens, uint64_t out_hidden);
[[nodiscard]] Result<LaunchPlan1D> plan_add_position_dispatch(uint64_t tokens, uint64_t hidden);
[[nodiscard]] Result<LaunchPlan1D> plan_transpose_2d_dispatch(uint64_t rows, uint64_t cols);
[[nodiscard]] Result<LaunchPlan1D> plan_deepstack_inject_dispatch(uint64_t tokens, uint64_t hidden);

[[nodiscard]] luisa::string vision_variant_key(luisa::string_view op);

}// namespace luisa::ocr