// =============================================================================
// kernel_tile_fill.cpp — fill: C[i] = 3.5
// =============================================================================
// Ported op  : fill a 64-element 1-D tensor with a constant (old
//              kernel_tile_fill.cpp: per-thread fragment fill + copy out).
// Provenance : backup_old_tile/examples/tensor/kernel_tile_fill.cpp
// Semantics  : old kernel staged through a register fragment per block; the
//              new TileIR expresses the same dataflow as a constant Tile SSA
//              value stored once. One parallel instance covers the whole
//              vector (old: one Kernel(1, threads) block). Tolerance 1e-5
//              follows the old main.cpp device pass.
// =============================================================================

#include "tensor_kernels.h"

namespace tensor_example {

[[nodiscard]] tile::Kernel make_tile_fill_kernel() {
    constexpr int64_t N = 64;
    auto definition = tile::tile_kernel("tile_fill", [=](tile::TensorView<float, 1> C) {
        using namespace tile;
        auto g = axis("g", 1), n = axis("n", N);
        for (auto &nest : parallel(shape(g))) {
            C(coord(0), shape(n)).store(full<float>(shape(n), 3.5f));
        }
    });
    return definition.capture(tile::tensor_shape(N));
}

void run_tile_fill(lc::Device &device, lc::Stream &stream) {
    constexpr uint32_t N = 64u;
    auto kernel = make_tile_fill_kernel();
    auto shader = compile_tile(device, kernel, "tile_fill");
    if (!shader) {
        record("tile_fill", false, shader.metadata().error);
        return;
    }
    auto bufC = device.create_buffer<float>(N);
    luisa::vector<float> hC(N);
    stream << shader(bufC).dispatch() << bufC.copy_to(luisa::span{hC}) << lc::synchronize();
    auto err = 0.0;
    for (auto i = 0u; i < N; ++i) { err = luisa::max(err, static_cast<double>(luisa::abs(hC[i] - 3.5f))); }
    check("tile_fill", err, 1e-5);
}

}// namespace tensor_example
