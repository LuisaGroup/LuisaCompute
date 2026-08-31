// =============================================================================
// kernel_tile_scan.cpp — CUMSUM / CUMMAX
// =============================================================================
// Inclusive prefix scan (sum / max)

#include "kernels.h"

Tensor<tile_f32, 1> tile_scan_kernel(Tensor<tile_f32, 1> A, Tensor<tile_f32, 1> S) {
    constexpr tile_i32 N = 64;
    constexpr tile_i32 threads = 32;

    Tensor<tile_f32, 1> Mx = LuisaTensor.empty(LuisaTensor.shape(N), tile_f32{});

    for (auto bx : LuisaTensor.Kernel(1, threads)) {
        auto A_local = LuisaTensor.alloc_fragment(LuisaTensor.shape(N), tile_f32{});
        auto S_local = LuisaTensor.alloc_fragment(LuisaTensor.shape(N), tile_f32{});
        auto M_local = LuisaTensor.alloc_fragment(LuisaTensor.shape(N), tile_f32{});

        LuisaTensor.copy(A(0), A_local(N));
        LuisaTensor.cumsum(A_local(N), S_local(N), /*dim=*/0);
        LuisaTensor.cummax(A_local(N), M_local(N), /*dim=*/0);
        LuisaTensor.copy(S_local(N), S(0));
        LuisaTensor.copy(M_local(N), Mx(0));
    }
    return Mx;
}
