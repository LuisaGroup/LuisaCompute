// =============================================================================
// kernel_rms_norm.cpp — KERNEL_1D / COPY / BINARY / STORE / REDUCE_SUM / RSQRT
// =============================================================================
// RMSNorm: B[r][c] = A[r][c] * rsqrt(sum_c A[r][c]^2 / N + 1e-12)

#include "kernels.h"

Tensor<tile_f32, 2> rms_norm(Tensor<tile_f32, 2> A) {
    constexpr tile_i32 M = 64, N = 64;
    constexpr tile_i32 blk_m = 8;
    constexpr tile_i32 threads = 64;

    Tensor<tile_f32, 2> B = LuisaTensor.empty(LuisaTensor.shape(M, N), tile_f32{});

    for (auto bx : LuisaTensor.Kernel(LuisaTensor.ceildiv(M, blk_m), threads)) {
        auto A_local = LuisaTensor.alloc_fragment(LuisaTensor.shape(blk_m, N), tile_f32{});
        auto A_pow_local = LuisaTensor.alloc_fragment(LuisaTensor.shape(blk_m, N), tile_f32{});
        auto A_powsum = LuisaTensor.alloc_fragment(LuisaTensor.shape(blk_m), tile_f32{});

        LuisaTensor.copy(A(LuisaTensor.range(bx * blk_m, (bx + 1) * blk_m), LuisaTensor.all()),
                         A_local(blk_m, N));

        A_pow_local(blk_m, N) = A_local(blk_m, N) * A_local(blk_m, N);

        LuisaTensor.reduce_sum(A_pow_local(blk_m, N), A_powsum(blk_m), /*dim=*/1);

        A_powsum(blk_m) = LuisaTensor.rsqrt(A_powsum(blk_m) / tile_f32(N) + 1e-12f);

        A_local(blk_m, N) *= A_powsum(blk_m);

        LuisaTensor.copy(A_local(blk_m, N),
                         B(LuisaTensor.range(bx * blk_m, (bx + 1) * blk_m), LuisaTensor.all()));
    }
    return B;
}
