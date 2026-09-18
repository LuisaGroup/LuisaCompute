// =============================================================================
// kernel_tile_transpose.cpp — transpose: B[i][j] = A[j][i]
// =============================================================================
// Ported op  : 8x8 transpose (old kernel_tile_transpose.cpp: shared-memory
//              staged transpose + sync_threads). Single 1x1 tile grid.
// Provenance : backup_old_tile/examples/tensor/kernel_tile_transpose.cpp
// Semantics  : uses the reindex transpose pattern from
//              examples/compute/tile/kernels.h. The old kernel staged the
//              tile through shared memory with an explicit barrier; the new
//              TileIR expresses the transpose as a pure Tile reindex and the
//              planner owns any staging/synchronization. Old host reference
//              (hB[i*BN+j] = hA[j*BM+i], hA[i] = i) and 1e-5 tolerance follow
//              the old main.cpp device pass.
// =============================================================================

#include "tensor_kernels.h"

namespace tensor_example {

[[nodiscard]] tile::Kernel make_tile_transpose_kernel() {
    constexpr int64_t BM = 8, BN = 8;
    auto definition = tile::tile_kernel("tile_transpose", [=](
                                            tile::TensorView<const float, 2> A,
                                            tile::TensorView<float, 2> B) {
        using namespace tile;
        auto gm = axis("gm", luisa::ceil_div(BM, BM));
        auto gn = axis("gn", luisa::ceil_div(BN, BN));
        auto m = axis("m", BM), n = axis("n", BN);
        for (auto &nest : parallel(shape(gm, gn))) {
            auto row = nest.index(gm) * BM, column = nest.index(gn) * BN;
            auto a = A.tile(coord(row, column), shape(m, n)).load();
            auto transposed = reindex(a, shape(n, m), [&](const tile::Nest &element) {
                return coord(element.index(m), element.index(n));
            });
            B(coord(column, row), shape(n, m)).store(transposed);
        }
    });
    return definition.capture(tile::tensor_shape(BM, BN), tile::tensor_shape(BM, BN));
}

void run_tile_transpose(lc::Device &device, lc::Stream &stream) {
    constexpr uint32_t BM = 8u, BN = 8u;
    auto kernel = make_tile_transpose_kernel();
    auto shader = compile_tile(device, kernel, "tile_transpose");
    if (!shader) {
        record("tile_transpose", false, shader.metadata().error);
        return;
    }
    auto bufA = device.create_buffer<float>(BM * BN);
    auto bufB = device.create_buffer<float>(BM * BN);
    luisa::vector<float> hA(BM * BN), hB(BM * BN);
    for (auto i = 0u; i < BM * BN; ++i) { hA[i] = static_cast<float>(i); }
    stream << bufA.copy_from(luisa::span{hA}) << lc::synchronize();
    stream << shader(bufA, bufB).dispatch() << bufB.copy_to(luisa::span{hB}) << lc::synchronize();
    auto err = 0.0;
    for (auto i = 0u; i < BM; ++i) {
        for (auto j = 0u; j < BN; ++j) {
            err = luisa::max(err, static_cast<double>(luisa::abs(hB[i * BN + j] - hA[j * BM + i])));
        }
    }
    check("tile_transpose", err, 1e-5);
}

}// namespace tensor_example
