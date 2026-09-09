// =============================================================================
// kernel_tile_atomic.cpp — ATOMIC
// =============================================================================
// Atomic load / store / add / max / min / or on an int tile

#include "kernels.h"

Tensor<tile_i32, 1> tile_atomic_kernel() {
    constexpr tile_i32 N = 32;
    constexpr tile_i32 threads = 32;

    Tensor<tile_i32, 1> D = LuisaTensor.empty(LuisaTensor.shape(N), tile_i32{});

    for (auto bx : LuisaTensor.Kernel(1, threads)) {
        LuisaTensor.atomic_load(D);
        LuisaTensor.atomic_store(D, 5);
        LuisaTensor.atomic_add(D, 2);
        LuisaTensor.atomic_max(D, 3);
        LuisaTensor.atomic_min(D, 4);
        LuisaTensor.atomic_or(D, 8);
    }
    return D;
}
