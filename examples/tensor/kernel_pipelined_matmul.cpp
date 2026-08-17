// =============================================================================
// kernel_pipelined_matmul.cpp — CLEAR / PIPELINED / GEMM / MAX / PRINT / STORE
// =============================================================================
// Tiled GEMM with a software pipeline: C = max(A @ B, 0)

#include "kernels.h"

Tensor<tile_f16, 2> pipelined_matmul(Tensor<tile_f16, 2> A, Tensor<tile_f16, 2> B) {
    constexpr tile_i32 M = 64, N = 64, K = 64;
    constexpr tile_i32 block_M = 16, block_N = 16, block_K = 8;
    constexpr tile_i32 threads = 32;
    constexpr tile_i32 num_stages = 2;

    Tensor<tile_f16, 2> C = LuisaTensor.empty(LuisaTensor.shape(M, N), tile_f16{});

    for (auto [bx, by] : LuisaTensor.Kernel(LuisaTensor.ceildiv(N, block_N), LuisaTensor.ceildiv(M, block_M), threads)) {
        auto A_shared = LuisaTensor.alloc_shared(LuisaTensor.shape(block_M, block_K), tile_f16{});
        auto B_shared = LuisaTensor.alloc_shared(LuisaTensor.shape(block_K, block_N), tile_f16{});
        auto C_local = LuisaTensor.alloc_fragment(LuisaTensor.shape(block_M, block_N), tile_f32{});

        LuisaTensor.clear(C_local);

        for (auto ko : LuisaTensor.Pipelined(LuisaTensor.ceildiv(K, block_K), num_stages)) {
            LuisaTensor.copy(A(by * block_M, ko * block_K), A_shared(block_M, block_K));
            LuisaTensor.copy(B(ko * block_K, bx * block_N), B_shared(block_K, block_N));
            LuisaTensor.gemm(A_shared(block_M, block_K), B_shared(block_K, block_N), C_local(block_M, block_N));
        }

        C_local(block_M, block_N) = LuisaTensor.max(C_local(block_M, block_N), 0.0f);

        LuisaTensor.print(C_local(block_M, block_N), "C tile:");

        LuisaTensor.copy(C_local(block_M, block_N), C(by * block_M, bx * block_N));
    }
    return C;
}
