// =============================================================================
// kernel_leaky_relu.cpp — LEAKY_RELU
// =============================================================================
// Leaky ReLU: max(alpha*x, x)

#include "kernels.h"

Tensor<tile_f32, 1> leaky_relu_kernel(Tensor<tile_f32, 1> A) {
    constexpr tile_i32 N = 64;
    constexpr tile_i32 threads = 32;
    constexpr float alpha = 0.01f;

    Tensor<tile_f32, 1> B = LuisaTensor.empty(LuisaTensor.shape(N), tile_f32{});

    for (auto bx : LuisaTensor.Kernel(1, threads)) {
        auto A_local = LuisaTensor.alloc_fragment(LuisaTensor.shape(N), tile_f32{});
        auto B_local = LuisaTensor.alloc_fragment(LuisaTensor.shape(N), tile_f32{});

        LuisaTensor.copy(A(0), A_local(N));
        auto neg_part = LuisaTensor.min(A_local(N), 0.0f);
        B_local(N) = A_local(N) + neg_part / (1.0f / (alpha - 1.0f));
        LuisaTensor.copy(B_local(N), B(0));
    }
    return B;
}
