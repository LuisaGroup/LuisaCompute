// =============================================================================
// kernel_tile_scan.cpp — inclusive cumsum / cummax of a 64-vector
// =============================================================================
// Ported op  : inclusive prefix sum and prefix max of a 64-element 1-D float
//              vector (old kernel_tile_scan.cpp: fragment cumsum/cummax).
// Provenance : backup_old_tile/examples/tensor/kernel_tile_scan.cpp
// Semantics  : composed like examples/compute/tile/kernels.h row_scan:
//              Hillis-Steele within the chunk (log2(64) = 6 gather steps)
//              with out-of-range gather lanes contributing the reduction
//              identity. N = 64 fits a single chunk, so no serial carry nest
//              is needed. The old kernel's cumsum lowered to a tree; the
//              floating sum here is likewise reassociated, so cumsum is
//              checked at 1e-3 while cummax is order-insensitive at 1e-5.
//              Host inputs hA[i] = (i%9-4)*0.5 follow the old main.cpp pass.
// =============================================================================

#include "tensor_kernels.h"

namespace tensor_example {

[[nodiscard]] tile::Kernel make_tile_scan_kernel() {
    constexpr int64_t N = 64;
    auto definition = tile::tile_kernel("tile_scan", [=](
                                            tile::TensorView<const float, 1> A,
                                            tile::TensorView<float, 1> S,
                                            tile::TensorView<float, 1> Mx) {
        using namespace tile;
        auto g = axis("g", 1), n = axis("n", N);
        constexpr auto neg_inf = -std::numeric_limits<float>::infinity();
        for (auto &nest : parallel(shape(g))) {
            auto value = A.tile(coord(0), shape(n)).load(0.0f);
            // Inclusive Hillis-Steele cumsum: out-of-range gathers see 0.
            // The gather index is clamped into [0, N) and the identity is
            // selected explicitly: a negative literal index would otherwise
            // materialize as an out-of-bounds constant array subscript in the
            // fully-expanded HLSL realization (DXC rejects it).
            auto sum = value;
            for (auto offset = int64_t{1}; offset < N; offset *= 2) {
                auto index = iota(n) - offset;
                auto shifted = gather(sum, max(index, Scalar<int64_t>{0}), n, 0.0f);
                sum = sum + ite(index >= 0, shifted, 0.0f);
            }
            S(coord(0), shape(n)).store(sum);
            // Inclusive Hillis-Steele cummax: out-of-range gathers see -inf.
            auto mx = value;
            for (auto offset = int64_t{1}; offset < N; offset *= 2) {
                auto index = iota(n) - offset;
                auto shifted = gather(mx, max(index, Scalar<int64_t>{0}), n, neg_inf);
                mx = max(mx, ite(index >= 0, shifted, neg_inf));
            }
            Mx(coord(0), shape(n)).store(mx);
        }
    });
    return definition.capture(tile::tensor_shape(N), tile::tensor_shape(N), tile::tensor_shape(N));
}

void run_tile_scan(lc::Device &device, lc::Stream &stream) {
    constexpr uint32_t N = 64u;
    auto kernel = make_tile_scan_kernel();
    auto shader = compile_tile(device, kernel, "tile_scan");
    if (!shader) {
        record("tile_scan", false, shader.metadata().error);
        return;
    }
    auto bufA = device.create_buffer<float>(N);
    auto bufS = device.create_buffer<float>(N);
    auto bufMx = device.create_buffer<float>(N);
    luisa::vector<float> hA(N), hS(N), hMx(N);
    for (auto i = 0u; i < N; ++i) { hA[i] = static_cast<float>(static_cast<int>(i % 9u) - 4) * 0.5f; }
    stream << bufA.copy_from(luisa::span{hA}) << lc::synchronize();
    stream << shader(bufA, bufS, bufMx).dispatch()
           << bufS.copy_to(luisa::span{hS}) << bufMx.copy_to(luisa::span{hMx}) << lc::synchronize();
    auto err_sum = 0.0, err_max = 0.0;
    auto run_sum = 0.0, run_max = -1e30;
    for (auto i = 0u; i < N; ++i) {
        run_sum += static_cast<double>(hA[i]);
        run_max = luisa::max(run_max, static_cast<double>(hA[i]));
        err_sum = luisa::max(err_sum, luisa::abs(static_cast<double>(hS[i]) - run_sum));
        err_max = luisa::max(err_max, luisa::abs(static_cast<double>(hMx[i]) - run_max));
    }
    check("tile_scan.cumsum", err_sum, 1e-3);
    check("tile_scan.cummax", err_max, 1e-5);
}

}// namespace tensor_example
