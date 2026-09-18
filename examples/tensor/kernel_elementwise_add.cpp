// =============================================================================
// kernel_elementwise_add.cpp — elementwise add: C = A + B
// =============================================================================
// Ported op  : C[M,N] = A[M,N] + B[M,N] with M = N = 64, 16x16 tiles (old
//              kernel_elementwise_add.cpp: shared-staged 2-D tiled add).
// Provenance : backup_old_tile/examples/tensor/kernel_elementwise_add.cpp
// Semantics  : host inputs and the 1e-3 tolerance follow the old main.cpp
//              device pass (hA[i] = i*0.5, hB[i] = i*1.5+1). The matrix sizes
//              are exact multiples of the tile size, so plain in-bounds loads
//              suffice; no zero-fill bounds mode is required.
// =============================================================================

#include "tensor_kernels.h"

namespace tensor_example {

[[nodiscard]] tile::Kernel make_elementwise_add_kernel() {
    constexpr int64_t M = 64, N = 64;
    constexpr int64_t block_m = 16, block_n = 16;
    auto definition = tile::tile_kernel("elementwise_add", [=](
                                            tile::TensorView<const float, 2> A,
                                            tile::TensorView<const float, 2> B,
                                            tile::TensorView<float, 2> C) {
        using namespace tile;
        auto gm = axis("gm", luisa::ceil_div(M, block_m));
        auto gn = axis("gn", luisa::ceil_div(N, block_n));
        auto m = axis("m", block_m), n = axis("n", block_n);
        for (auto &nest : parallel(shape(gm, gn))) {
            auto origin = coord(nest.index(gm) * block_m, nest.index(gn) * block_n);
            auto a = A.tile(origin, shape(m, n)).load();
            auto b = B.tile(origin, shape(m, n)).load();
            C(origin, shape(m, n)).store(a + b);
        }
    });
    return definition.capture(tile::tensor_shape(M, N), tile::tensor_shape(M, N), tile::tensor_shape(M, N));
}

void run_elementwise_add(lc::Device &device, lc::Stream &stream) {
    constexpr uint32_t M = 64u, N = 64u;
    auto kernel = make_elementwise_add_kernel();
    auto shader = compile_tile(device, kernel, "elementwise_add");
    if (!shader) {
        record("elementwise_add", false, shader.metadata().error);
        return;
    }
    auto bufA = device.create_buffer<float>(M * N);
    auto bufB = device.create_buffer<float>(M * N);
    auto bufC = device.create_buffer<float>(M * N);
    luisa::vector<float> hA(M * N), hB(M * N), hC(M * N), hRef(M * N);
    for (auto i = 0u; i < M * N; ++i) {
        hA[i] = static_cast<float>(i) * 0.5f;
        hB[i] = static_cast<float>(i) * 1.5f + 1.0f;
        hRef[i] = hA[i] + hB[i];
    }
    stream << bufA.copy_from(luisa::span{hA}) << bufB.copy_from(luisa::span{hB}) << lc::synchronize();
    stream << shader(bufA, bufB, bufC).dispatch() << bufC.copy_to(luisa::span{hC}) << lc::synchronize();
    auto err = 0.0;
    for (auto i = 0u; i < M * N; ++i) { err = luisa::max(err, static_cast<double>(luisa::abs(hC[i] - hRef[i]))); }
    check("elementwise_add", err, 1e-3);
}

}// namespace tensor_example
