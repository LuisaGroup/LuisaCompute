// =============================================================================
// kernel_pipelined_matmul.cpp — pipelined GEMM + ReLU: C = max(A @ B, 0)
// =============================================================================
// Ported op  : tiled f16 GEMM, f32 accumulator, elementwise ReLU on the
//              accumulator, f16 store. M = N = K = 64, 16x16x8 tiles, 8
//              software-pipelined k-steps (old kernel_pipelined_matmul.cpp:
//              2-stage Pipelined loop over ceildiv(K, block_K)).
// Provenance : backup_old_tile/examples/tensor/kernel_pipelined_matmul.cpp
// Semantics  : host inputs are f16-exact ((i%8)*0.25, (i%4)*0.5) so the f32
//              host reference stays meaningful; tolerance 1e-2 follows the old
//              main.cpp device pass. The old LuisaTensor.print(C_local, ...)
//              debug dump has no counterpart in the new Tile runtime — it is
//              omitted. Storage/accumulation precision split follows the gemm
//              pattern in examples/compute/tile/kernels.h.
// =============================================================================

#include "tensor_kernels.h"

namespace tensor_example {

[[nodiscard]] tile::Kernel make_pipelined_matmul_kernel() {
    constexpr int64_t M = 64, N = 64, K = 64;
    constexpr int64_t block_m = 16, block_n = 16, block_k = 8;
    auto definition = tile::tile_kernel("pipelined_matmul", [=](
                                            tile::TensorView<const luisa::half, 2> A,
                                            tile::TensorView<const luisa::half, 2> B,
                                            tile::TensorView<luisa::half, 2> C) {
        using namespace tile;
        auto gm = axis("gm", luisa::ceil_div(M, block_m));
        auto gn = axis("gn", luisa::ceil_div(N, block_n));
        auto m = axis("m", block_m), n = axis("n", block_n), k = axis("k", block_k);
        for (auto &nest : parallel(shape(gm, gn))) {
            auto row = nest.index(gm) * block_m, column = nest.index(gn) * block_n;
            auto acc = zeros<float>(shape(m, n));
            for (auto &step : nest.pipeline(shape(luisa::ceil_div(K, block_k)), {.window = 1u, .interval = 1u})) {
                step.stage("load");
                auto a = A.tile(coord(row, step.index() * block_k), shape(m, k)).load();
                auto b = B.tile(coord(step.index() * block_k, column), shape(k, n)).load();
                step.stage("compute");
                acc = mma(a, b, acc);
            }
            C(coord(row, column), shape(m, n)).store(cast<luisa::half>(max(acc, 0.0f)));
        }
    });
    return definition.capture(tile::tensor_shape(M, K), tile::tensor_shape(K, N), tile::tensor_shape(M, N));
}

void run_pipelined_matmul(lc::Device &device, lc::Stream &stream) {
    constexpr uint32_t M = 64u, N = 64u, K = 64u;
    auto kernel = make_pipelined_matmul_kernel();
    auto shader = compile_tile(device, kernel, "pipelined_matmul");
    if (!shader) {
        record("pipelined_matmul", false, shader.metadata().error);
        return;
    }
    auto bufA = device.create_buffer<luisa::half>(M * K);
    auto bufB = device.create_buffer<luisa::half>(K * N);
    auto bufC = device.create_buffer<luisa::half>(M * N);
    luisa::vector<luisa::half> hA(M * K), hB(K * N), hC(M * N);
    // f16-exact inputs so the f32 host reference is meaningful.
    for (auto i = 0u; i < M * K; ++i) { hA[i] = luisa::half{static_cast<float>(i % 8u) * 0.25f}; }
    for (auto i = 0u; i < K * N; ++i) { hB[i] = luisa::half{static_cast<float>(i % 4u) * 0.5f}; }
    stream << bufA.copy_from(luisa::span{hA}) << bufB.copy_from(luisa::span{hB}) << lc::synchronize();
    stream << shader(bufA, bufB, bufC).dispatch() << bufC.copy_to(luisa::span{hC}) << lc::synchronize();
    auto err = 0.0;
    for (auto r = 0u; r < M; ++r) {
        for (auto c = 0u; c < N; ++c) {
            auto s = 0.0;
            for (auto k = 0u; k < K; ++k) {
                s += static_cast<float>(hA[r * K + k]) * static_cast<float>(hB[k * N + c]);
            }
            auto ref = luisa::max(s, 0.0);
            err = luisa::max(err, luisa::abs(static_cast<double>(static_cast<float>(hC[r * N + c])) - ref));
        }
    }
    check("pipelined_matmul", err, 1e-2);
}

}// namespace tensor_example
