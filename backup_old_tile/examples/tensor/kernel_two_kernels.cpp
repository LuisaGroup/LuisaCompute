// =============================================================================
// kernel_two_kernels.cpp — Multiple-T.Kernel guard (INVALID)
// =============================================================================
// A tile function with two T.Kernel blocks — aborts when compiled.

#include "kernels.h"

Tensor<tile_f32, 2> two_kernels(Tensor<tile_f32, 2> A, Tensor<tile_f32, 2> B) {
    for (auto [bx, by] : LuisaTensor.Kernel(4, 4, 32)) {
        LuisaTensor.copy(A(0, 0), B(0, 0));
    }
    for (auto [bx, by] : LuisaTensor.Kernel(8, 8, 64)) {
        LuisaTensor.copy(A(0, 0), B(0, 0));
    }
    return B;
}
