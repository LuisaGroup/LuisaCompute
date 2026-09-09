// =============================================================================
// kernel_tile_sync.cpp — SYNC
// =============================================================================
// Explicit block barrier between shared accesses

#include "kernels.h"

Tensor<tile_f32, 2> tile_sync_kernel(Tensor<tile_f32, 2> A) {
    constexpr tile_i32 BM = 8, BN = 8;
    constexpr tile_i32 threads = 32;

    Tensor<tile_f32, 2> C = LuisaTensor.empty(LuisaTensor.shape(BM, BN), tile_f32{});

    for (auto [bx, by] : LuisaTensor.Kernel(1, 1, threads)) {
        auto A_shared = LuisaTensor.alloc_shared(LuisaTensor.shape(BM, BN), tile_f32{});

        LuisaTensor.copy(A(by * BM, bx * BN), A_shared(BM, BN));
        LuisaTensor.sync_threads();
        LuisaTensor.copy(A_shared(BM, BN), C(by * BM, bx * BN));
    }
    return C;
}
