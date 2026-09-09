// =============================================================================
// kernel_tile_transpose.cpp — TRANSPOSE
// =============================================================================
// Shared-memory 2D transpose: B[i][j] = A[j][i]

#include "kernels.h"

Tensor<tile_f32, 2> tile_transpose_kernel(Tensor<tile_f32, 2> A) {
    constexpr tile_i32 BM = 8, BN = 8;
    constexpr tile_i32 threads = 32;

    Tensor<tile_f32, 2> B = LuisaTensor.empty(LuisaTensor.shape(BM, BN), tile_f32{});

    for (auto [bx, by] : LuisaTensor.Kernel(1, 1, threads)) {
        auto A_shared = LuisaTensor.alloc_shared(LuisaTensor.shape(BM, BN), tile_f32{});
        auto T_shared = LuisaTensor.alloc_shared(LuisaTensor.shape(BN, BM), tile_f32{});

        LuisaTensor.copy(A(by * BM, bx * BN), A_shared(BM, BN));
        LuisaTensor.transpose(A_shared(BM, BN), T_shared(BN, BM));
        LuisaTensor.sync_threads();
        LuisaTensor.copy(T_shared(BN, BM), B(by * BM, bx * BN));
    }
    return B;
}
