// =============================================================================
// kernel_tile_clamp.cpp — clamp: C[i] = clamp(A[i], 0.1, 0.9)
// =============================================================================
// Ported op  : elementwise clamp into [lo, hi] on an 8x8 tensor (old
//              kernel_tile_clamp.cpp: in-place clamp of a register fragment).
// Provenance : backup_old_tile/examples/tensor/kernel_tile_clamp.cpp
// Semantics  : old kernel copied A into a fragment, clamped it, and copied it
//              to C; the new TileIR composes min(max(a, lo), hi) directly on
//              the loaded Tile. Host inputs hA[i] = (i%16)*0.1 and the 1e-5
//              tolerance follow the old main.cpp device pass.
// =============================================================================

#include "tensor_kernels.h"

namespace tensor_example {

[[nodiscard]] tile::Kernel make_tile_clamp_kernel() {
    constexpr int64_t BM = 8, BN = 8;
    auto definition = tile::tile_kernel("tile_clamp", [=](
                                            tile::TensorView<const float, 2> A,
                                            tile::TensorView<float, 2> C) {
        using namespace tile;
        auto g = axis("g", 1);
        auto m = axis("m", BM), n = axis("n", BN);
        for (auto &nest : parallel(shape(g))) {
            auto a = A.tile(coord(0, 0), shape(m, n)).load();
            C(coord(0, 0), shape(m, n)).store(min(max(a, 0.1f), 0.9f));
        }
    });
    return definition.capture(tile::tensor_shape(BM, BN), tile::tensor_shape(BM, BN));
}

void run_tile_clamp(lc::Device &device, lc::Stream &stream) {
    constexpr uint32_t BM = 8u, BN = 8u;
    auto kernel = make_tile_clamp_kernel();
    auto shader = compile_tile(device, kernel, "tile_clamp");
    if (!shader) {
        record("tile_clamp", false, shader.metadata().error);
        return;
    }
    auto bufA = device.create_buffer<float>(BM * BN);
    auto bufC = device.create_buffer<float>(BM * BN);
    luisa::vector<float> hA(BM * BN), hC(BM * BN);
    for (auto i = 0u; i < BM * BN; ++i) { hA[i] = static_cast<float>(i % 16u) * 0.1f; }
    stream << bufA.copy_from(luisa::span{hA}) << lc::synchronize();
    stream << shader(bufA, bufC).dispatch() << bufC.copy_to(luisa::span{hC}) << lc::synchronize();
    auto err = 0.0;
    for (auto i = 0u; i < BM * BN; ++i) {
        auto ref = luisa::clamp(hA[i], 0.1f, 0.9f);
        err = luisa::max(err, static_cast<double>(luisa::abs(hC[i] - ref)));
    }
    check("tile_clamp", err, 1e-5);
}

}// namespace tensor_example
