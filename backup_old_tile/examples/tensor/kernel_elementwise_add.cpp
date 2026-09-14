// =============================================================================
// kernel_elementwise_add.cpp — ALLOC / CEILDIV / KERNEL_2D / COPY / BINARY / STORE
// =============================================================================
// Elementwise add: C = A + B

#include "kernels.h"

Tensor<tile_f32, 2> elementwise_add(Tensor<tile_f32, 2> A, Tensor<tile_f32, 2> B) {
    constexpr tile_i32 M = 64, N = 64;
    constexpr tile_i32 block_M = 16, block_N = 16;
    constexpr tile_i32 threads = 32;

    Tensor<tile_f32, 2> C = LuisaTensor.empty(LuisaTensor.shape(M, N), tile_f32{});

    for (auto [bx, by] : LuisaTensor.Kernel(LuisaTensor.ceildiv(N, block_N), LuisaTensor.ceildiv(M, block_M), threads)) {
        auto A_shared = LuisaTensor.alloc_shared(LuisaTensor.shape(block_M, block_N), tile_f32{});
        auto B_shared = LuisaTensor.alloc_shared(LuisaTensor.shape(block_M, block_N), tile_f32{});
        auto C_local = LuisaTensor.alloc_fragment(LuisaTensor.shape(block_M, block_N), tile_f32{});

        LuisaTensor.copy(A(by * block_M, bx * block_N), A_shared(block_M, block_N));
        LuisaTensor.copy(B(by * block_M, bx * block_N), B_shared(block_M, block_N));

        C_local(block_M, block_N) = A_shared(block_M, block_N) + B_shared(block_M, block_N);

        LuisaTensor.copy(C_local(block_M, block_N), C(by * block_M, bx * block_N));
    }
    return C;
}
