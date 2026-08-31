// =============================================================================
// kernel_cast.cpp — CAST (int32 -> float32)
// =============================================================================
// Element-wise cast

#include "kernels.h"

Tensor<tile_f32, 1> cast_kernel(Tensor<tile_i32, 1> A) {
    constexpr tile_i32 N = 64;
    constexpr tile_i32 threads = 32;

    Tensor<tile_f32, 1> B = LuisaTensor.empty(LuisaTensor.shape(N), tile_f32{});

    for (auto bx : LuisaTensor.Kernel(1, threads)) {
        auto A_local = LuisaTensor.alloc_fragment(LuisaTensor.shape(N), tile_i32{});
        auto B_local = LuisaTensor.alloc_fragment(LuisaTensor.shape(N), tile_f32{});

        LuisaTensor.copy(A(0), A_local(N));
        B_local(N) = LuisaTensor.cast<tile_f32>(A_local(N));
        LuisaTensor.copy(B_local(N), B(0));
    }
    return B;
}
