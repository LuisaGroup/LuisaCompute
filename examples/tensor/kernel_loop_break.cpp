// =============================================================================
// kernel_loop_break.cpp — LOOP_BREAK (traced only)
// =============================================================================
// A top-level break is representable in the tile IR but not yet lowerable.

#include "kernels.h"

Tensor<tile_f32, 2> loop_break_kernel(Tensor<tile_f32, 2> A) {
    Tensor<tile_f32, 2> B = LuisaTensor.empty(LuisaTensor.shape(8, 8), tile_f32{});
    for (auto [bx, by] : LuisaTensor.Kernel(1, 1, 32)) {
        LuisaTensor.copy(A(0, 0), B(0, 0));
        LuisaTensor.loop_break();
    }
    return B;
}
