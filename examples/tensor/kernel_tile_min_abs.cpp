// =============================================================================
// kernel_tile_min_abs.cpp — C = min(|A|, |B|)
// =============================================================================
// Ported op  : whole-tile elementwise min/abs on an 8x8 tensor (old
//              kernel_tile_min_abs.cpp: B = min(A, 0.5) and C = abs(A)).
// Provenance : backup_old_tile/examples/tensor/kernel_tile_min_abs.cpp
// Semantics  : the new contract (tensor_kernels.h) is a single min-abs kernel
//              of two inputs: C[i] = min(|A[i]|, |B[i]|). This ports the old
//              elementwise min+abs semantics; because the new kernel has two
//              real input tensors, a second input generator was substituted
//              for the old constant 0.5: hB[i] = (i%11-5)*0.25 (documented
//              substitution — the old main.cpp only exercised the constant
//              form). A's inputs hA[i] = (i%13-6)*0.25 and the 1e-5
//              tolerance follow the old main.cpp device pass.
// =============================================================================

#include "tensor_kernels.h"

namespace tensor_example {

[[nodiscard]] tile::Kernel make_tile_min_abs_kernel() {
    constexpr int64_t BM = 8, BN = 8;
    auto definition = tile::tile_kernel("tile_min_abs", [=](
                                            tile::TensorView<const float, 2> A,
                                            tile::TensorView<const float, 2> B,
                                            tile::TensorView<float, 2> C) {
        using namespace tile;
        auto g = axis("g", 1);
        auto m = axis("m", BM), n = axis("n", BN);
        for (auto &nest : parallel(shape(g))) {
            auto a = A.tile(coord(0, 0), shape(m, n)).load();
            auto b = B.tile(coord(0, 0), shape(m, n)).load();
            C(coord(0, 0), shape(m, n)).store(min(abs(a), abs(b)));
        }
    });
    return definition.capture(tile::tensor_shape(BM, BN), tile::tensor_shape(BM, BN), tile::tensor_shape(BM, BN));
}

void run_tile_min_abs(lc::Device &device, lc::Stream &stream) {
    constexpr uint32_t BM = 8u, BN = 8u;
    auto kernel = make_tile_min_abs_kernel();
    auto shader = compile_tile(device, kernel, "tile_min_abs");
    if (!shader) {
        record("tile_min_abs", false, shader.metadata().error);
        return;
    }
    auto bufA = device.create_buffer<float>(BM * BN);
    auto bufB = device.create_buffer<float>(BM * BN);
    auto bufC = device.create_buffer<float>(BM * BN);
    luisa::vector<float> hA(BM * BN), hB(BM * BN), hC(BM * BN);
    for (auto i = 0u; i < BM * BN; ++i) {
        hA[i] = static_cast<float>(static_cast<int>(i % 13u) - 6) * 0.25f;
        hB[i] = static_cast<float>(static_cast<int>(i % 11u) - 5) * 0.25f;
    }
    stream << bufA.copy_from(luisa::span{hA}) << bufB.copy_from(luisa::span{hB}) << lc::synchronize();
    stream << shader(bufA, bufB, bufC).dispatch() << bufC.copy_to(luisa::span{hC}) << lc::synchronize();
    auto err = 0.0;
    for (auto i = 0u; i < BM * BN; ++i) {
        auto ref = luisa::min(luisa::abs(hA[i]), luisa::abs(hB[i]));
        err = luisa::max(err, static_cast<double>(luisa::abs(hC[i] - ref)));
    }
    check("tile_min_abs", err, 1e-5);
}

}// namespace tensor_example
