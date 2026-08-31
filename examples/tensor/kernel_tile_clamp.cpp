// =============================================================================
// kernel_tile_clamp.cpp — CLAMP
// =============================================================================
// In-place elementwise clamp into [lo, hi]

#include "kernels.h"

Tensor<tile_f32, 2> tile_clamp_kernel(Tensor<tile_f32, 2> A) {
    constexpr tile_i32 BM = 8, BN = 8;
    constexpr tile_i32 threads = 32;

    Tensor<tile_f32, 2> C = LuisaTensor.empty(LuisaTensor.shape(BM, BN), tile_f32{});

    for (auto [bx, by] : LuisaTensor.Kernel(1, 1, threads)) {
        auto C_local = LuisaTensor.alloc_fragment(LuisaTensor.shape(BM, BN), tile_f32{});

        LuisaTensor.copy(A(by * BM, bx * BN), C_local(BM, BN));
        LuisaTensor.clamp(C_local(BM, BN), 0.1f, 0.9f);
        LuisaTensor.copy(C_local(BM, BN), C(by * BM, bx * BN));
    }
    return C;
}
