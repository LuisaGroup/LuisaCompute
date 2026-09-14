// =============================================================================
// kernel_pow.cpp — POW
// =============================================================================
// Element-wise power: pow(a, b)

#include "kernels.h"

Tensor<tile_f32, 1> pow_kernel(Tensor<tile_f32, 1> A, Tensor<tile_f32, 1> B) {
    constexpr tile_i32 N = 64;
    constexpr tile_i32 threads = 32;

    Tensor<tile_f32, 1> C = LuisaTensor.empty(LuisaTensor.shape(N), tile_f32{});

    for (auto bx : LuisaTensor.Kernel(1, threads)) {
        auto A_local = LuisaTensor.alloc_fragment(LuisaTensor.shape(N), tile_f32{});
        auto B_local = LuisaTensor.alloc_fragment(LuisaTensor.shape(N), tile_f32{});
        auto C_local = LuisaTensor.alloc_fragment(LuisaTensor.shape(N), tile_f32{});

        LuisaTensor.copy(A(0), A_local(N));
        LuisaTensor.copy(B(0), B_local(N));
        C_local(N) = LuisaTensor.pow(A_local(N), B_local(N));
        LuisaTensor.copy(C_local(N), C(0));
    }
    return C;
}
