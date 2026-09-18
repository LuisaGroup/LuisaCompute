// =============================================================================
// kernel_softmax.cpp — row-wise softmax
// =============================================================================
// Ported op  : row-wise softmax over a 64x64 matrix, 8 rows per block (old
//              kernel_softmax.cpp: exp + reduce_sum + rsqrt(row_sum) applied
//              twice = divide by the row sum).
// Provenance : backup_old_tile/examples/tensor/kernel_softmax.cpp
// Semantics  : the old kernel divided by sum via rsqrt(row_sum)^2. The new
//              TileIR has no rsqrt opcode, and the authoritative old host
//              reference (backup_old_tile main.cpp) computes the numerically
//              stabilized form exp(x - row_max) / sum(exp(x - row_max)); this
//              port mirrors THAT: row max is subtracted before exp and the
//              exponential tile is divided by the broadcast row sum. Named
//              dimensions give the (m) -> (m,n) broadcast for free.
//              Host inputs hA[i] = (i%17-8)*0.5 and the 1e-3 tolerance follow
//              the old main.cpp device pass.
// =============================================================================

#include "tensor_kernels.h"

namespace tensor_example {

[[nodiscard]] tile::Kernel make_softmax_kernel() {
    constexpr int64_t M = 64, N = 64;
    constexpr int64_t block_rows = 8;
    auto definition = tile::tile_kernel("softmax_kernel", [=](
                                            tile::TensorView<const float, 2> A,
                                            tile::TensorView<float, 2> B) {
        using namespace tile;
        auto m = axis("m", block_rows), n = axis("n", N);
        for (auto &nest : parallel(shape(luisa::ceil_div(M, block_rows)))) {
            auto origin = coord(nest.index() * block_rows, 0);
            auto a = A.tile(origin, shape(m, n)).load();
            auto row_max = reduce(a, n, maximum);
            auto e = exp(a - row_max);
            auto denom = reduce(e, n, add);
            B(origin, shape(m, n)).store(e / denom);
        }
    });
    return definition.capture(tile::tensor_shape(M, N), tile::tensor_shape(M, N));
}

void run_softmax(lc::Device &device, lc::Stream &stream) {
    constexpr uint32_t M = 64u, N = 64u;
    auto kernel = make_softmax_kernel();
    auto shader = compile_tile(device, kernel, "softmax_kernel");
    if (!shader) {
        record("softmax", false, shader.metadata().error);
        return;
    }
    auto bufA = device.create_buffer<float>(M * N);
    auto bufB = device.create_buffer<float>(M * N);
    luisa::vector<float> hA(M * N), hB(M * N), hRef(M * N);
    for (auto i = 0u; i < M * N; ++i) {
        hA[i] = static_cast<float>(static_cast<int>(i % 17u) - 8) * 0.5f;
    }
    stream << bufA.copy_from(luisa::span{hA}) << lc::synchronize();
    stream << shader(bufA, bufB).dispatch() << bufB.copy_to(luisa::span{hB}) << lc::synchronize();
    // Old main.cpp host reference (authoritative): stabilized softmax.
    auto err = 0.0;
    for (auto r = 0u; r < M; ++r) {
        auto row_max = -1e30;
        for (auto c = 0u; c < N; ++c) { row_max = luisa::max(row_max, static_cast<double>(hA[r * N + c])); }
        auto sum = 0.0;
        for (auto c = 0u; c < N; ++c) { sum += std::exp(static_cast<double>(hA[r * N + c]) - row_max); }
        for (auto c = 0u; c < N; ++c) {
            hRef[r * N + c] = static_cast<float>(std::exp(static_cast<double>(hA[r * N + c]) - row_max) / sum);
            err = luisa::max(err, static_cast<double>(luisa::abs(hB[r * N + c] - hRef[r * N + c])));
        }
    }
    check("softmax", err, 1e-3);
}

}// namespace tensor_example
