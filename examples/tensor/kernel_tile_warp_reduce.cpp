// =============================================================================
// kernel_tile_warp_reduce.cpp — WARP_REDUCE
// =============================================================================
// Register-level warp reduction (sum / max)

#include "kernels.h"

Tensor<tile_f32, 1> tile_warp_reduce_kernel() {
    constexpr tile_i32 N = 1;
    constexpr tile_i32 threads = 32;

    Tensor<tile_f32, 1> W = LuisaTensor.empty(LuisaTensor.shape(N), tile_f32{});

    for (auto bx : LuisaTensor.Kernel(1, threads)) {
        auto v = LuisaTensor.alloc_fragment(LuisaTensor.shape(N), tile_f32{});
        LuisaTensor.fill(v(N), 7.0f);
        LuisaTensor.warp_reduce_sum(v(N));
        LuisaTensor.warp_reduce_max(v(N));
        LuisaTensor.copy(v(N), W(0));
    }
    return W;
}
