// =============================================================================
// kernel_tile_reduce.cpp — row-wise max / min / abssum / absmax
// =============================================================================
// Ported op  : four row-wise reductions of a 64x64 matrix, 8 rows per block
//              (old kernel_tile_reduce.cpp: reduce_max/min/abssum/absmax over
//              dim 1). Old signature returned Babsmax and took three output
//              vectors; the new ABI has no return-channel tensor, so the
//              kernel takes all four outputs as arguments (5 total).
// Provenance : backup_old_tile/examples/tensor/kernel_tile_reduce.cpp
// Semantics  : mixed-sign host inputs hA[i] = (i%17-8)*0.25 follow the old
//              main.cpp device pass. The old check applied one tolerance
//              (1e-3) to all four; here each reduction is recorded separately:
//              max/min/absmax are order-insensitive (1e-5), abssum is an
//              unordered-tree reassociation (1e-2). absmax is composed as
//              reduce(abs(a), maximum), matching the old reduce_absmax.
// =============================================================================

#include "tensor_kernels.h"

namespace tensor_example {

[[nodiscard]] tile::Kernel make_tile_reduce_kernel() {
    constexpr int64_t M = 64, N = 64;
    constexpr int64_t block_rows = 8;
    auto definition = tile::tile_kernel("tile_reduce", [=](
                                            tile::TensorView<const float, 2> A,
                                            tile::TensorView<float, 1> Bmax,
                                            tile::TensorView<float, 1> Bmin,
                                            tile::TensorView<float, 1> Babssum,
                                            tile::TensorView<float, 1> Babsmax) {
        using namespace tile;
        auto m = axis("m", block_rows), n = axis("n", N);
        for (auto &nest : parallel(shape(luisa::ceil_div(M, block_rows)))) {
            auto row = nest.index() * block_rows;
            auto a = A.tile(coord(row, 0), shape(m, n)).load();
            Bmax(coord(row), shape(m)).store(reduce(a, n, maximum));
            Bmin(coord(row), shape(m)).store(reduce(a, n, minimum));
            Babssum(coord(row), shape(m)).store(reduce(abs(a), n, add));
            Babsmax(coord(row), shape(m)).store(reduce(abs(a), n, maximum));
        }
    });
    return definition.capture(tile::tensor_shape(M, N), tile::tensor_shape(M), tile::tensor_shape(M),
                              tile::tensor_shape(M), tile::tensor_shape(M));
}

void run_tile_reduce(lc::Device &device, lc::Stream &stream) {
    constexpr uint32_t M = 64u, N = 64u;
    auto kernel = make_tile_reduce_kernel();
    auto shader = compile_tile(device, kernel, "tile_reduce");
    if (!shader) {
        record("tile_reduce", false, shader.metadata().error);
        return;
    }
    auto bufA = device.create_buffer<float>(M * N);
    auto bufMax = device.create_buffer<float>(M);
    auto bufMin = device.create_buffer<float>(M);
    auto bufAbsSum = device.create_buffer<float>(M);
    auto bufAbsMax = device.create_buffer<float>(M);
    luisa::vector<float> hA(M * N), hMax(M), hMin(M), hAbsSum(M), hAbsMax(M);
    // mixed-sign inputs: negatives exercise min/abs, the spread separates max
    // from absmax (old main.cpp input generator).
    for (auto i = 0u; i < M * N; ++i) {
        hA[i] = static_cast<float>(static_cast<int>(i % 17u) - 8) * 0.25f;
    }
    stream << bufA.copy_from(luisa::span{hA}) << lc::synchronize();
    stream << shader(bufA, bufMax, bufMin, bufAbsSum, bufAbsMax).dispatch()
           << bufMax.copy_to(luisa::span{hMax}) << bufMin.copy_to(luisa::span{hMin})
           << bufAbsSum.copy_to(luisa::span{hAbsSum}) << bufAbsMax.copy_to(luisa::span{hAbsMax})
           << lc::synchronize();
    auto err_max = 0.0, err_min = 0.0, err_abssum = 0.0, err_absmax = 0.0;
    for (auto r = 0u; r < M; ++r) {
        auto ref_max = -1e30, ref_min = 1e30, ref_abssum = 0.0, ref_absmax = 0.0;
        for (auto c = 0u; c < N; ++c) {
            auto v = static_cast<double>(hA[r * N + c]);
            ref_max = luisa::max(ref_max, v);
            ref_min = luisa::min(ref_min, v);
            ref_abssum += luisa::abs(v);
            ref_absmax = luisa::max(ref_absmax, luisa::abs(v));
        }
        err_max = luisa::max(err_max, luisa::abs(static_cast<double>(hMax[r]) - ref_max));
        err_min = luisa::max(err_min, luisa::abs(static_cast<double>(hMin[r]) - ref_min));
        err_abssum = luisa::max(err_abssum, luisa::abs(static_cast<double>(hAbsSum[r]) - ref_abssum));
        err_absmax = luisa::max(err_absmax, luisa::abs(static_cast<double>(hAbsMax[r]) - ref_absmax));
    }
    check("tile_reduce.max", err_max, 1e-5);
    check("tile_reduce.min", err_min, 1e-5);
    check("tile_reduce.abssum", err_abssum, 1e-2);
    check("tile_reduce.absmax", err_absmax, 1e-5);
}

}// namespace tensor_example
