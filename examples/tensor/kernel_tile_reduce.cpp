// =============================================================================
// kernel_tile_reduce.cpp — REDUCE
// =============================================================================
// Row-wise reductions: max / min / abssum / absmax

#include "kernels.h"

Tensor<tile_f32, 1> tile_reduce_kernel(Tensor<tile_f32, 2> A,
                                       Tensor<tile_f32, 1> Bmax,
                                       Tensor<tile_f32, 1> Bmin,
                                       Tensor<tile_f32, 1> Babssum) {
    constexpr tile_i32 M = 64, N = 64;
    constexpr tile_i32 blk_m = 8;
    constexpr tile_i32 threads = 64;

    Tensor<tile_f32, 1> Babsmax = LuisaTensor.empty(LuisaTensor.shape(M), tile_f32{});

    for (auto bx : LuisaTensor.Kernel(LuisaTensor.ceildiv(M, blk_m), threads)) {
        auto A_local = LuisaTensor.alloc_fragment(LuisaTensor.shape(blk_m, N), tile_f32{});
        auto v_max = LuisaTensor.alloc_fragment(LuisaTensor.shape(blk_m), tile_f32{});
        auto v_min = LuisaTensor.alloc_fragment(LuisaTensor.shape(blk_m), tile_f32{});
        auto v_abssum = LuisaTensor.alloc_fragment(LuisaTensor.shape(blk_m), tile_f32{});
        auto v_absmax = LuisaTensor.alloc_fragment(LuisaTensor.shape(blk_m), tile_f32{});

        LuisaTensor.copy(A(LuisaTensor.range(bx * blk_m, (bx + 1) * blk_m), LuisaTensor.all()),
                         A_local(blk_m, N));

        LuisaTensor.reduce_max(A_local(blk_m, N), v_max(blk_m), /*dim=*/1);
        LuisaTensor.reduce_min(A_local(blk_m, N), v_min(blk_m), /*dim=*/1);
        LuisaTensor.reduce_abssum(A_local(blk_m, N), v_abssum(blk_m), /*dim=*/1);
        LuisaTensor.reduce_absmax(A_local(blk_m, N), v_absmax(blk_m), /*dim=*/1);

        LuisaTensor.copy(v_max(blk_m), Bmax(bx * blk_m));
        LuisaTensor.copy(v_min(blk_m), Bmin(bx * blk_m));
        LuisaTensor.copy(v_abssum(blk_m), Babssum(bx * blk_m));
        LuisaTensor.copy(v_absmax(blk_m), Babsmax(bx * blk_m));
    }
    return Babsmax;
}
