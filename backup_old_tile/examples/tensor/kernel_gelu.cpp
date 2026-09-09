// =============================================================================
// kernel_gelu.cpp — GELU
// =============================================================================
// GELU: 0.5*x*(1+erf(x/sqrt(2)))

#include "kernels.h"

Tensor<tile_f32, 1> gelu_kernel(Tensor<tile_f32, 1> A) {
    constexpr tile_i32 N = 64;
    constexpr tile_i32 threads = 32;
    constexpr float inv_sqrt2 = 0.7071067811865475f;

    Tensor<tile_f32, 1> B = LuisaTensor.empty(LuisaTensor.shape(N), tile_f32{});

    for (auto bx : LuisaTensor.Kernel(1, threads)) {
        auto A_local = LuisaTensor.alloc_fragment(LuisaTensor.shape(N), tile_f32{});
        auto B_local = LuisaTensor.alloc_fragment(LuisaTensor.shape(N), tile_f32{});

        LuisaTensor.copy(A(0), A_local(N));
        auto scaled = A_local(N) / 1.4142135f;
        auto erf_part = LuisaTensor.erf(scaled);
        auto half = A_local(N) / 2.0f;
        B_local(N) = half * (erf_part + 1.0f);
        LuisaTensor.copy(B_local(N), B(0));
    }
    return B;
}
