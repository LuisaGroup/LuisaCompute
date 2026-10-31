#pragma once
// embedding_mm engine -- GEMM operators (matrix multiplication C = A @ B and C = A @ B + bias).
//
// A is [M, K], row-major.
// B is [K, N], row-major (pre-transposed offline so reads coalesce along N).
// C is [M, N], row-major.
// Tiled 16x16 shared-memory kernels supporting arbitrary M, N, K with zero-padding.

#include <cstdint>
#include <luisa/core/basic_types.h>
#include <luisa/dsl/syntax.h>

namespace luisa::embed_mm {

inline constexpr uint32_t kGemmTileSize = 16u;

// Standard GEMM: C = A @ B
using GemmKernel = compute::Kernel2D<void(
    compute::Buffer<float> /*A*/,
    compute::Buffer<float> /*B*/,
    compute::Buffer<float> /*C*/,
    uint32_t /*M*/,
    uint32_t /*N*/,
    uint32_t /*K*/)>;

[[nodiscard]] GemmKernel make_gemm_kernel() noexcept;

// GEMM with 1D row bias: C = A @ B + bias
using GemmBiasKernel = compute::Kernel2D<void(
    compute::Buffer<float> /*A*/,
    compute::Buffer<float> /*B*/,
    compute::Buffer<float> /*bias*/,
    compute::Buffer<float> /*C*/,
    uint32_t /*M*/,
    uint32_t /*N*/,
    uint32_t /*K*/)>;

[[nodiscard]] GemmBiasKernel make_gemm_bias_kernel() noexcept;

[[nodiscard]] inline compute::uint2 gemm_dispatch_size(uint32_t M, uint32_t N) noexcept {
    auto ceil_div = [](uint32_t a, uint32_t b) { return (a + b - 1u) / b; };
    return compute::make_uint2(
        ceil_div(N, kGemmTileSize) * kGemmTileSize,
        ceil_div(M, kGemmTileSize) * kGemmTileSize);
}

}// namespace luisa::embed_mm
