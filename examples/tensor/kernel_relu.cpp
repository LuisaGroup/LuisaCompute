// =============================================================================
// kernel_relu.cpp — RELU
// =============================================================================
// ReLU: B[i] = max(A[i], 0)

#include "kernels.h"

Tensor<tile_f32, 1> relu_kernel(Tensor<tile_f32, 1> A) {
    constexpr tile_i32 N = 64;
    constexpr tile_i32 threads = 32;

    Tensor<tile_f32, 1> B = LuisaTensor.empty(LuisaTensor.shape(N), tile_f32{});

    for (auto bx : LuisaTensor.Kernel(1, threads)) {
        auto A_local = LuisaTensor.alloc_fragment(LuisaTensor.shape(N), tile_f32{});
        auto B_local = LuisaTensor.alloc_fragment(LuisaTensor.shape(N), tile_f32{});

        LuisaTensor.copy(A(0), A_local(N));
        B_local(N) = LuisaTensor.max(A_local(N), 0.0f);
        LuisaTensor.copy(B_local(N), B(0));
    }
    return B;
}
