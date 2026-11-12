#pragma once
// OCR engine -- CTC frame operators (Phase 5). Device-free DSL factories;
// FP32 activations and accumulation.
//
// Real device kernels for the CTC recognizer tail:
//   ctc_log_softmax : row-wise numerically-stable log-softmax over frame
//                     logits [frames, classes] with an explicit temperature.
//                     out[t,c] = x[t,c]/T - max_c(x[t,:]/T)
//                                - log(sum_c exp(x[t,c]/T - max)).
//                     Max-subtracted, so |x| up to the FP32 range never
//                     overflows exp. Temperature must be finite and > 0
//                     (host validator; the kernel defensively writes 0 for
//                     an invalid temperature so a skipped validation can
//                     never produce NaN-driven garbage).
//   ctc_argmax_probs: per-frame argmax id + per-frame probability of the
//                     selected id, computed from the LOG-probabilities
//                     (exp of the selected row entry; no second softmax).
//                     Deterministic lowest-index tie-break, identical to
//                     ops_decode argmax.
//   sigmoid         : elementwise logistic sigmoid (DB probability head).
//
// Greedy frame ids/probabilities feed output/ctc_decode.h (host), which
// implements the DECLARED PP-OCR reference collapse/confidence semantics.
// There is no beam search, NRTR attention decoder, dropout, or loss here.
#include <cstdint>

#include <luisa/dsl/syntax.h>

#include "ocr/runtime/dispatch.h"

namespace luisa::ocr {

// Row log-softmax with temperature. logits/out are [rows, cols] FP32.
using CtcLogSoftmaxKernel = compute::Kernel1D<void(
    compute::Buffer<float> /*logits [rows,cols]*/,
    compute::Buffer<float> /*out [rows,cols]*/,
    uint32_t /*rows*/, uint32_t /*cols*/, float /*temperature*/)>;
[[nodiscard]] CtcLogSoftmaxKernel make_ctc_log_softmax_kernel() noexcept;

// Per-frame greedy id + probability of the selected class. `log_probs` is
// [rows, cols] (the output of make_ctc_log_softmax_kernel). ids is int32
// [rows], probs is float [rows] = exp(log_probs[row, id]).
using CtcArgmaxProbsKernel = compute::Kernel1D<void(
    compute::Buffer<float> /*log_probs [rows,cols]*/,
    compute::Buffer<int32_t> /*ids [rows]*/,
    compute::Buffer<float> /*probs [rows]*/,
    uint32_t /*rows*/, uint32_t /*cols*/)>;
[[nodiscard]] CtcArgmaxProbsKernel make_ctc_argmax_probs_kernel() noexcept;

// Elementwise sigmoid: out[i] = 1/(1+exp(-x[i])) over count elements.
using SigmoidKernel = compute::Kernel1D<void(
    compute::Buffer<float> /*x*/,
    compute::Buffer<float> /*out*/,
    uint32_t /*count*/)>;
[[nodiscard]] SigmoidKernel make_sigmoid_kernel() noexcept;

// LightSVTR height-collapse reshape (pixel-unshuffle along HEIGHT only, then
// transpose to a per-frame sequence). Input NCHW feature [n, c, h, w] with h
// divisible by r; the height axis is folded into channels producing
// [n, c*r, h/r, w], then each column x becomes one frame:
//     out[(ni*w + x), ry*c + ch] = x[ni, ch, by*r + ry, x]
// where by = (h/r - 1) is the SINGLE remaining height block when h == r, and
// the general form iterates by. For the recognizer h == r so by == 0 and the
// output is [n*w, c*r]. One thread per output element.
using HeightCollapseKernel = compute::Kernel1D<void(
    compute::Buffer<float> /*x [n,c,h,w]*/,
    compute::Buffer<float> /*out [n*w*(h/r), c*r]*/,
    uint32_t /*n*/, uint32_t /*c*/, uint32_t /*h*/, uint32_t /*w*/, uint32_t /*r*/)>;
[[nodiscard]] HeightCollapseKernel make_height_collapse_kernel() noexcept;

// Elementwise activation (compile-time `act`): 0=identity, 1=relu, 2=relu6,
// 3=hard_swish. out[i] = act(x[i]) over count elements. Used after folded
// Conv+BN stages (the conv kernel itself has no activation epilogue).
using CnnActKernel = compute::Kernel1D<void(
    compute::Buffer<float> /*x*/,
    compute::Buffer<float> /*out*/,
    uint32_t /*count*/)>;
[[nodiscard]] CnnActKernel make_cnn_act_kernel(uint32_t act) noexcept;

// FPN bilinear upsample NCHW (align_corners=false, half-pixel centers, clamp
// edges). x [n,c,h,w] -> out [n,c,out_h,out_w]. One thread per output element.
using FpnResizeKernel = compute::Kernel1D<void(
    compute::Buffer<float> /*x*/,
    compute::Buffer<float> /*out*/,
    uint32_t /*n*/, uint32_t /*c*/, uint32_t /*h*/, uint32_t /*w*/,
    uint32_t /*out_h*/, uint32_t /*out_w*/)>;
[[nodiscard]] FpnResizeKernel make_fpn_resize_kernel() noexcept;

// Polygon line-crop with rotation: sample a rotated rectangle from a source
// CHW float image into a dense [crop_h, crop_w] CHW crop. The rectangle is
// defined by its center (cx,cy), width/height (rw,rh) and angle (radians,
// counter-clockwise in image coordinates). Bilinear sampling, zero outside
// the source. One thread per (crop pixel, channel).
using PolygonCropKernel = compute::Kernel1D<void(
    compute::Buffer<float> /*src CHW [3,sh,sw]*/,
    compute::Buffer<float> /*dst CHW [3,crop_h,crop_w]*/,
    uint32_t /*sw*/, uint32_t /*sh*/, uint32_t /*crop_w*/, uint32_t /*crop_h*/,
    float /*cx*/, float /*cy*/, float /*rw*/, float /*rh*/, float /*angle*/)>;
[[nodiscard]] PolygonCropKernel make_polygon_crop_kernel() noexcept;

// Host validators: recoverable invalid inputs return invalid_argument BEFORE
// launch. `temperature` must be finite and strictly positive.
[[nodiscard]] Status validate_ctc_log_softmax_inputs(uint64_t rows, uint64_t cols, float temperature);
[[nodiscard]] Status validate_ctc_frame_inputs(uint64_t rows, uint64_t cols);

// Dispatch helpers (TOTAL threads).
[[nodiscard]] Result<LaunchPlan1D> plan_ctc_log_softmax_dispatch(uint64_t rows);
[[nodiscard]] Result<LaunchPlan1D> plan_ctc_argmax_dispatch(uint64_t rows);
[[nodiscard]] Result<LaunchPlan1D> plan_sigmoid_dispatch(uint64_t count);
[[nodiscard]] Result<LaunchPlan1D> plan_height_collapse_dispatch(uint64_t frames, uint64_t channels);
[[nodiscard]] Result<LaunchPlan1D> plan_cnn_act_dispatch(uint64_t count);
[[nodiscard]] Result<LaunchPlan1D> plan_fpn_resize_dispatch(uint64_t n, uint64_t c, uint64_t out_h, uint64_t out_w);
[[nodiscard]] Result<LaunchPlan1D> plan_polygon_crop_dispatch(uint64_t crop_w, uint64_t crop_h);

[[nodiscard]] luisa::string ctc_variant_key(luisa::string_view op);

// ---------------------------------------------------------------------------
// Independent host oracles (double accumulation where it matters). These are
// deliberately separate naive implementations, NOT calls into the device
// helpers, so tests compare two code paths.
// ---------------------------------------------------------------------------

// Row log-softmax with temperature (double max/sum/exp/log, rounded to float).
[[nodiscard]] Status ref_ctc_log_softmax(luisa::span<const float> logits, uint32_t rows, uint32_t cols,
                                         float temperature, luisa::vector<float> &out);
// Per-frame greedy id (lowest-index ties) + exp of the selected log-prob.
// The probability is computed in double from the INPUT log-probabilities.
[[nodiscard]] Status ref_ctc_argmax_probs(luisa::span<const float> log_probs, uint32_t rows, uint32_t cols,
                                          luisa::vector<int32_t> &ids, luisa::vector<float> &probs);
// Elementwise sigmoid (double exp).
[[nodiscard]] Status ref_sigmoid(luisa::span<const float> x, luisa::vector<float> &out);
// Height-collapse reshape oracle (see HeightCollapseKernel). h must be
// divisible by r. out is [n*w*(h/r), c*r].
[[nodiscard]] Status ref_height_collapse(luisa::span<const float> x, uint32_t n, uint32_t c,
                                         uint32_t h, uint32_t w, uint32_t r, luisa::vector<float> &out);
// Elementwise activation oracle (act tags as CnnActKernel).
[[nodiscard]] Status ref_cnn_act(luisa::span<const float> x, uint32_t act, luisa::vector<float> &out);
// FPN bilinear resize oracle (double sample math, half-pixel centers).
[[nodiscard]] Status ref_fpn_resize(luisa::span<const float> x, uint32_t n, uint32_t c, uint32_t h,
                                    uint32_t w, uint32_t out_h, uint32_t out_w, luisa::vector<float> &out);
// Polygon crop oracle (double bilinear sampling of the rotated rectangle).
[[nodiscard]] Status ref_polygon_crop(luisa::span<const float> src, uint32_t sw, uint32_t sh,
                                      uint32_t crop_w, uint32_t crop_h, float cx, float cy,
                                      float rw, float rh, float angle, luisa::vector<float> &out);

}// namespace luisa::ocr