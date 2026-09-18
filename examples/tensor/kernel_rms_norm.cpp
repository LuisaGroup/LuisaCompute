// =============================================================================
// kernel_rms_norm.cpp — RMSNorm: B[r][c] = A[r][c] * rsqrt(mean_c A^2 + 1e-12)
// =============================================================================
// Ported op  : unweighted RMSNorm over rows, M = N = 64, 8 rows per block
//              (old kernel_rms_norm.cpp: reduce_sum over dim 1 of A^2).
// Provenance : backup_old_tile/examples/tensor/kernel_rms_norm.cpp
// Semantics  : mirrors examples/compute/tile/kernels.h rms_norm: 1/sqrt(x)
//              composition (the rsqrt opcode does not exist in the new TileIR
//              elementwise set). Host inputs hA[i] = i*0.5 and the 1e-3
//              tolerance follow the old main.cpp device pass.
// =============================================================================

#include "tensor_kernels.h"

namespace tensor_example {

[[nodiscard]] tile::Kernel make_rms_norm_kernel() {
    constexpr int64_t M = 64, N = 64;
    constexpr int64_t block_rows = 8;
    auto definition = tile::tile_kernel("rms_norm", [=](
                                            tile::TensorView<const float, 2> A,
                                            tile::TensorView<float, 2> B) {
        using namespace tile;
        auto m = axis("m", block_rows), n = axis("n", N);
        for (auto &nest : parallel(shape(luisa::ceil_div(M, block_rows)))) {
            auto origin = coord(nest.index() * block_rows, 0);
            auto a = A.tile(origin, shape(m, n)).load();
            auto mean_square = reduce(a * a, n, add) / static_cast<float>(N);
            B(origin, shape(m, n)).store(a * (1.0f / sqrt(mean_square + 1e-12f)));
        }
    });
    return definition.capture(tile::tensor_shape(M, N), tile::tensor_shape(M, N));
}

void run_rms_norm(lc::Device &device, lc::Stream &stream) {
    constexpr uint32_t M = 64u, N = 64u;
    auto kernel = make_rms_norm_kernel();
    auto shader = compile_tile(device, kernel, "rms_norm");
    if (!shader) {
        record("rms_norm", false, shader.metadata().error);
        return;
    }
    auto bufA = device.create_buffer<float>(M * N);
    auto bufB = device.create_buffer<float>(M * N);
    luisa::vector<float> hA(M * N), hB(M * N);
    for (auto i = 0u; i < M * N; ++i) { hA[i] = static_cast<float>(i) * 0.5f; }
    stream << bufA.copy_from(luisa::span{hA}) << lc::synchronize();
    stream << shader(bufA, bufB).dispatch() << bufB.copy_to(luisa::span{hB}) << lc::synchronize();
    auto err = 0.0;
    for (auto r = 0u; r < M; ++r) {
        auto s = 0.0;
        for (auto c = 0u; c < N; ++c) { s += static_cast<double>(hA[r * N + c]) * hA[r * N + c]; }
        auto scale = 1.0 / std::sqrt(s / static_cast<double>(N) + 1e-12);
        for (auto c = 0u; c < N; ++c) {
            err = luisa::max(err, luisa::abs(static_cast<double>(hB[r * N + c]) - static_cast<double>(hA[r * N + c]) * scale));
        }
    }
    check("rms_norm", err, 1e-3);
}

}// namespace tensor_example
