#pragma once
// OCR engine -- convolution / pooling / tensor-shape operators (Phase 2).
// Device-free DSL factories; FP32 activations and accumulation. Naive and
// correct first: one thread per output element, loops kernel in thread.
//
// Conv2D NCHW: general kernel dims (1x1/3x3/5x5/7x7 and arbitrary), stride
// incl. asymmetric (2,1), padding, dilation, groups (depthwise = groups==C,
// separable = depthwise + 1x1 pointwise composed host-side). Conv1D NCL incl.
// causal depthwise (left-only padding). BatchNorm inference, residual add,
// channel concat/split/shuffle, global/average/adaptive pooling are reusable
// ops here. CPU contours/postprocessing stay in output/, NOT here.
#include <cstdint>

#include <luisa/dsl/syntax.h>

#include "ocr/runtime/dispatch.h"

namespace luisa::ocr {

// Conv2D NCHW. weight [out_c, in_c/groups, kh, kw], bias [out_c] (dummy +
// has_bias=0 to skip). One thread per output element (n, oc, oy, ox).
using Conv2dKernel = compute::Kernel1D<void(
    compute::Buffer<float> /*x [n,in_c,h,w]*/,
    compute::Buffer<float> /*weight*/,
    compute::Buffer<float> /*bias*/,
    compute::Buffer<float> /*out [n,out_c,oh,ow]*/,
    uint32_t /*n*/, uint32_t /*in_c*/, uint32_t /*h*/, uint32_t /*w*/,
    uint32_t /*out_c*/, uint32_t /*kh*/, uint32_t /*kw*/,
    uint32_t /*stride_h*/, uint32_t /*stride_w*/, uint32_t /*pad_h*/, uint32_t /*pad_w*/,
    uint32_t /*dil_h*/, uint32_t /*dil_w*/, uint32_t /*groups*/,
    uint32_t /*out_h*/, uint32_t /*out_w*/, uint32_t /*has_bias*/)>;
[[nodiscard]] Conv2dKernel make_conv2d_kernel() noexcept;

// Conv1D NCL. weight [out_c, c/groups, k]. causal=1 pads left only (eff_k-1).
using Conv1dKernel = compute::Kernel1D<void(
    compute::Buffer<float> /*x [n,c,len]*/,
    compute::Buffer<float> /*weight*/,
    compute::Buffer<float> /*bias*/,
    compute::Buffer<float> /*out [n,out_c,out_len]*/,
    uint32_t /*n*/, uint32_t /*c*/, uint32_t /*len*/, uint32_t /*out_c*/, uint32_t /*k*/,
    uint32_t /*stride*/, uint32_t /*pad*/, uint32_t /*dilation*/, uint32_t /*groups*/,
    uint32_t /*causal*/, uint32_t /*out_len*/, uint32_t /*has_bias*/)>;
[[nodiscard]] Conv1dKernel make_conv1d_kernel() noexcept;

// BatchNorm inference NCHW: y=(x-mean)/sqrt(var+eps)*gamma+beta per channel.
using BatchNormKernel = compute::Kernel1D<void(
    compute::Buffer<float> /*x*/,
    compute::Buffer<float> /*mean*/,
    compute::Buffer<float> /*var*/,
    compute::Buffer<float> /*gamma*/,
    compute::Buffer<float> /*beta*/,
    compute::Buffer<float> /*out*/,
    uint32_t /*n*/, uint32_t /*c*/, uint32_t /*spatial*/, float /*eps*/)>;
[[nodiscard]] BatchNormKernel make_batch_norm_kernel() noexcept;

// Average pool2D NCHW (count_include_pad=false). One thread per output element.
using AvgPool2dKernel = compute::Kernel1D<void(
    compute::Buffer<float> /*x*/,
    compute::Buffer<float> /*out*/,
    uint32_t /*n*/, uint32_t /*c*/, uint32_t /*h*/, uint32_t /*w*/,
    uint32_t /*kh*/, uint32_t /*kw*/, uint32_t /*sh*/, uint32_t /*sw*/,
    uint32_t /*ph*/, uint32_t /*pw*/, uint32_t /*out_h*/, uint32_t /*out_w*/)>;
[[nodiscard]] AvgPool2dKernel make_avg_pool2d_kernel() noexcept;

// Adaptive average pool2D NCHW to [out_h,out_w] (floor-div bin edges).
using AdaptiveAvgPool2dKernel = compute::Kernel1D<void(
    compute::Buffer<float> /*x*/,
    compute::Buffer<float> /*out*/,
    uint32_t /*n*/, uint32_t /*c*/, uint32_t /*h*/, uint32_t /*w*/,
    uint32_t /*out_h*/, uint32_t /*out_w*/)>;
[[nodiscard]] AdaptiveAvgPool2dKernel make_adaptive_avg_pool2d_kernel() noexcept;

// Global average pool2D NCHW -> [n,c,1,1].
using GlobalAvgPool2dKernel = compute::Kernel1D<void(
    compute::Buffer<float> /*x*/,
    compute::Buffer<float> /*out*/,
    uint32_t /*n*/, uint32_t /*c*/, uint32_t /*spatial*/)>;
[[nodiscard]] GlobalAvgPool2dKernel make_global_avg_pool2d_kernel() noexcept;

// Channel concat NCHW: out[n, ca+cb, spatial] = [a, b] along channels.
using ConcatChannelsKernel = compute::Kernel1D<void(
    compute::Buffer<float> /*a*/,
    compute::Buffer<float> /*b*/,
    compute::Buffer<float> /*out*/,
    uint32_t /*n*/, uint32_t /*ca*/, uint32_t /*cb*/, uint32_t /*spatial*/)>;
[[nodiscard]] ConcatChannelsKernel make_concat_channels_kernel() noexcept;

// Channel split NCHW: copy channels [c0, c0+count) to out (a reusable slice).
using SplitChannelsKernel = compute::Kernel1D<void(
    compute::Buffer<float> /*x*/,
    compute::Buffer<float> /*out*/,
    uint32_t /*n*/, uint32_t /*c*/, uint32_t /*c0*/, uint32_t /*count*/, uint32_t /*spatial*/)>;
[[nodiscard]] SplitChannelsKernel make_split_channels_kernel() noexcept;

// Channel shuffle NCHW: reshape C into [groups, C/groups] and transpose.
using ChannelShuffleKernel = compute::Kernel1D<void(
    compute::Buffer<float> /*x*/,
    compute::Buffer<float> /*out*/,
    uint32_t /*n*/, uint32_t /*c*/, uint32_t /*spatial*/, uint32_t /*groups*/)>;
[[nodiscard]] ChannelShuffleKernel make_channel_shuffle_kernel() noexcept;

// Dispatch helpers (one thread per output element; TOTAL threads).
[[nodiscard]] Result<LaunchPlan1D> plan_conv2d_dispatch(uint64_t n, uint64_t out_c, uint64_t out_h, uint64_t out_w);
[[nodiscard]] Result<LaunchPlan1D> plan_conv1d_dispatch(uint64_t n, uint64_t out_c, uint64_t out_len);
[[nodiscard]] Result<LaunchPlan1D> plan_elementwise_dispatch(uint64_t count);

// Host validators (S5): invalid groups/channel divisibility returns
// invalid_argument BEFORE launch (never silently drops channels). The
// kernels also guard in-device: an invalid configuration writes zeros.
[[nodiscard]] Status validate_conv2d_inputs(uint32_t n, uint32_t in_c, uint32_t h, uint32_t w,
                                            uint32_t out_c, uint32_t kh, uint32_t kw,
                                            uint32_t stride_h, uint32_t stride_w,
                                            uint32_t dil_h, uint32_t dil_w, uint32_t groups);
[[nodiscard]] Status validate_conv1d_inputs(uint32_t n, uint32_t c, uint32_t len, uint32_t out_c,
                                            uint32_t k, uint32_t stride, uint32_t dilation, uint32_t groups);

// Checked output-dimension helpers (mirror the host oracle formulas).
[[nodiscard]] Result<uint32_t> conv2d_out_dim(uint32_t in, uint32_t k, uint32_t stride, uint32_t pad, uint32_t dil);
[[nodiscard]] Result<uint32_t> conv1d_out_len(uint32_t len, uint32_t k, uint32_t stride, uint32_t pad,
                                              uint32_t dil, bool causal);

[[nodiscard]] luisa::string conv_variant_key(luisa::string_view op);

}// namespace luisa::ocr