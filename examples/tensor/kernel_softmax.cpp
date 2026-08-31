// =============================================================================
// kernel_softmax.cpp — SOFTMAX
// =============================================================================
// Row-wise softmax: exp(x)/sum(exp(x))

#include "kernels.h"

Tensor<tile_f32, 2> softmax_kernel(Tensor<tile_f32, 2> A) {
    constexpr tile_i32 M = 64, N = 64;
    constexpr tile_i32 blk_m = 8;
    constexpr tile_i32 threads = 64;

    Tensor<tile_f32, 2> B = LuisaTensor.empty(LuisaTensor.shape(M, N), tile_f32{});

    for (auto bx : LuisaTensor.Kernel(LuisaTensor.ceildiv(M, blk_m), threads)) {
        auto A_local = LuisaTensor.alloc_fragment(LuisaTensor.shape(blk_m, N), tile_f32{});
        auto row_sum = LuisaTensor.alloc_fragment(LuisaTensor.shape(blk_m), tile_f32{});

        LuisaTensor.copy(A(LuisaTensor.range(bx * blk_m, (bx + 1) * blk_m), LuisaTensor.all()),
                         A_local(blk_m, N));

        A_local(blk_m, N) = LuisaTensor.exp(A_local(blk_m, N));
        LuisaTensor.reduce_sum(A_local(blk_m, N), row_sum(blk_m), /*dim=*/1);
        A_local(blk_m, N) *= LuisaTensor.rsqrt(row_sum(blk_m));
        A_local(blk_m, N) *= LuisaTensor.rsqrt(row_sum(blk_m));

        LuisaTensor.copy(A_local(blk_m, N),
                         B(LuisaTensor.range(bx * blk_m, (bx + 1) * blk_m), LuisaTensor.all()));
    }
    return B;
}
