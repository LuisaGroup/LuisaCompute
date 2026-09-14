// =============================================================================
// kernel_tile_fill.cpp — FILL
// =============================================================================
// Fill a per-thread fragment tile with a constant and copy it out.

#include "kernels.h"

Tensor<tile_f32, 1> tile_fill_kernel() {
    constexpr tile_i32 N = 64;
    constexpr tile_i32 threads = 32;

    Tensor<tile_f32, 1> C = LuisaTensor.empty(LuisaTensor.shape(N), tile_f32{});

    for (auto bx : LuisaTensor.Kernel(1, threads)) {
        auto F = LuisaTensor.alloc_fragment(LuisaTensor.shape(N), tile_f32{});
        LuisaTensor.fill(F(N), 3.5f);
        LuisaTensor.copy(F(N), C(0));
    }
    return C;
}
