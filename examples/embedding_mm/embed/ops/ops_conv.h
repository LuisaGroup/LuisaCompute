#pragma once
// embedding_mm engine -- Audio 2D Subsampling & Causal Depthwise 1D Convolution operators.
//
// 1. AudioSubsampleConv0Kernel: Conv2D(1 -> 128, 3x3, stride 2, pad 1) + LayerNorm(128, eps) + ReLU
// 2. AudioSubsampleConv1Kernel: Conv2D(128 -> 32, 3x3, stride 2, pad 1) + LayerNorm(32, eps) + ReLU -> flat [H_out, 1024]
// 3. AudioDepthwiseConv1DKernel: Causal Depthwise Conv1D (kernel 5, left pad 4) for LightConv1D

#include <cstdint>
#include <luisa/dsl/sugar.h>
#include <luisa/runtime/buffer.h>
#include <luisa/runtime/shader.h>

namespace luisa::embed_mm {

using AudioSubsampleConv0Kernel = compute::Kernel3D<void(
    compute::Buffer<float> /*in_features*/,
    compute::Buffer<float> /*weight*/,
    compute::Buffer<float> /*norm_weight*/,
    compute::Buffer<float> /*out_h0*/,
    uint32_t /*H_in*/,
    uint32_t /*W_in*/,
    uint32_t /*H_out*/,
    uint32_t /*W_out*/,
    float /*eps*/)>;

using AudioSubsampleConv1Kernel = compute::Kernel3D<void(
    compute::Buffer<float> /*in_h0*/,
    compute::Buffer<float> /*weight*/,
    compute::Buffer<float> /*norm_weight*/,
    compute::Buffer<float> /*out_flat*/,
    uint32_t /*H_in*/,
    uint32_t /*W_in*/,
    uint32_t /*H_out*/,
    uint32_t /*W_out*/,
    float /*eps*/)>;

using AudioDepthwiseConv1DKernel = compute::Kernel2D<void(
    compute::Buffer<float> /*in_x*/,
    compute::Buffer<float> /*weight*/,
    compute::Buffer<float> /*out_y*/,
    uint32_t /*T*/,
    uint32_t /*channels*/)>;

[[nodiscard]] AudioSubsampleConv0Kernel make_audio_subsample_conv0_kernel() noexcept;
[[nodiscard]] AudioSubsampleConv1Kernel make_audio_subsample_conv1_kernel() noexcept;
[[nodiscard]] AudioDepthwiseConv1DKernel make_audio_depthwise_conv1d_kernel() noexcept;

}// namespace luisa::embed_mm
