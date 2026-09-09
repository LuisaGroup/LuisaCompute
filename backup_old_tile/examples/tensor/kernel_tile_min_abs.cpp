// =============================================================================
// kernel_tile_min_abs.cpp — MIN / ABS
// =============================================================================
// Whole-tile elementwise min and abs

#include "kernels.h"

Tensor<tile_f32, 2> tile_min_abs_kernel(Tensor<tile_f32, 2> A, Tensor<tile_f32, 2> B) {
    constexpr tile_i32 BM = 8, BN = 8;
    constexpr tile_i32 threads = 32;

    Tensor<tile_f32, 2> C = LuisaTensor.empty(LuisaTensor.shape(BM, BN), tile_f32{});

    for (auto [bx, by] : LuisaTensor.Kernel(1, 1, threads)) {
        auto A_local = LuisaTensor.alloc_fragment(LuisaTensor.shape(BM, BN), tile_f32{});
        auto M_local = LuisaTensor.alloc_fragment(LuisaTensor.shape(BM, BN), tile_f32{});
        auto Abs_local = LuisaTensor.alloc_fragment(LuisaTensor.shape(BM, BN), tile_f32{});

        LuisaTensor.copy(A(by * BM, bx * BN), A_local(BM, BN));
        M_local(BM, BN) = LuisaTensor.min(A_local(BM, BN), 0.5f);
        Abs_local(BM, BN) = LuisaTensor.abs(A_local(BM, BN));
        LuisaTensor.copy(M_local(BM, BN), B(by * BM, bx * BN));
        LuisaTensor.copy(Abs_local(BM, BN), C(by * BM, bx * BN));
    }
    return C;
}
