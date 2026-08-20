// =============================================================================
// kernel_sigmoid.cpp — SIGMOID
// =============================================================================
// Sigmoid: 1/(1+exp(-x))

#include "kernels.h"

Tensor<tile_f32, 1> sigmoid_kernel(Tensor<tile_f32, 1> A) {
    constexpr tile_i32 N = 64;
    constexpr tile_i32 threads = 32;

    Tensor<tile_f32, 1> B = LuisaTensor.empty(LuisaTensor.shape(N), tile_f32{});

    for (auto bx : LuisaTensor.Kernel(1, threads)) {
        auto A_local = LuisaTensor.alloc_fragment(LuisaTensor.shape(N), tile_f32{});
        auto B_local = LuisaTensor.alloc_fragment(LuisaTensor.shape(N), tile_f32{});

        LuisaTensor.copy(A(0), A_local(N));
        auto neg_x = A_local(N) / (-1.0f);
        auto denom = LuisaTensor.exp(neg_x) + 1.0f;
        B_local(N) = LuisaTensor.rsqrt(denom) * LuisaTensor.rsqrt(denom);
        LuisaTensor.copy(B_local(N), B(0));
    }
    return B;
}
