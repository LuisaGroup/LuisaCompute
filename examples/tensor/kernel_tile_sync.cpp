// =============================================================================
// kernel_tile_sync.cpp — copy with an explicit producer/consumer dependency
// =============================================================================
// Ported op  : C = A (8x8) (old kernel_tile_sync.cpp: global -> shared copy,
//              T.sync_threads(), shared -> global copy).
// Provenance : backup_old_tile/examples/tensor/kernel_tile_sync.cpp
// Semantics  : the new Tile DSL has NO explicit barrier primitive: nests
//              communicate through Tile SSA values and the planner inserts
//              whatever synchronization the chosen lowering needs (the old
//              shared-memory staging and sync_threads are a lowering detail,
//              not part of the program's semantics). The same dataflow —
//              load the tile, store the tile — is expressed directly.
//              Host inputs hA[i] = i*0.25 and the 1e-5 tolerance follow the
//              old main.cpp device pass.
// =============================================================================

#include "tensor_kernels.h"

namespace tensor_example {

[[nodiscard]] tile::Kernel make_tile_sync_kernel() {
    constexpr int64_t BM = 8, BN = 8;
    auto definition = tile::tile_kernel("tile_sync", [=](
                                            tile::TensorView<const float, 2> A,
                                            tile::TensorView<float, 2> C) {
        using namespace tile;
        auto g = axis("g", 1);
        auto m = axis("m", BM), n = axis("n", BN);
        for (auto &nest : parallel(shape(g))) {
            auto a = A.tile(coord(0, 0), shape(m, n)).load();
            C(coord(0, 0), shape(m, n)).store(a);
        }
    });
    return definition.capture(tile::tensor_shape(BM, BN), tile::tensor_shape(BM, BN));
}

void run_tile_sync(lc::Device &device, lc::Stream &stream) {
    constexpr uint32_t BM = 8u, BN = 8u;
    auto kernel = make_tile_sync_kernel();
    auto shader = compile_tile(device, kernel, "tile_sync");
    if (!shader) {
        record("tile_sync", false, shader.metadata().error);
        return;
    }
    auto bufA = device.create_buffer<float>(BM * BN);
    auto bufC = device.create_buffer<float>(BM * BN);
    luisa::vector<float> hA(BM * BN), hC(BM * BN);
    for (auto i = 0u; i < BM * BN; ++i) { hA[i] = static_cast<float>(i) * 0.25f; }
    stream << bufA.copy_from(luisa::span{hA}) << lc::synchronize();
    stream << shader(bufA, bufC).dispatch() << bufC.copy_to(luisa::span{hC}) << lc::synchronize();
    auto err = 0.0;
    for (auto i = 0u; i < BM * BN; ++i) {
        err = luisa::max(err, static_cast<double>(luisa::abs(hC[i] - hA[i])));
    }
    check("tile_sync", err, 1e-5);
}

}// namespace tensor_example
