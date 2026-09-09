#pragma once

// Execution-first counterparts of backup_old_tile/examples/compute/tile_bench.cpp.
// These examples contain no legacy frontend, implicit memory assignment, or
// backend-specific temporary storage. The benchmark and regression tests use
// these same captures.
#include <luisa/core/mathematics.h>
#include <luisa/tile/algorithms.h>

namespace luisa::example::tile {

using namespace compute::tile;

enum class Pointwise { COPY,
                       ADD,
                       SAXPY,
                       CLAMP,
                       EXP };
enum class RowReduction { SUM,
                          MAX,
                          MIN,
                          ABS_SUM,
                          ABS_MAX };
enum class Scan { SUM,
                  MAX };

struct Block {
    int64_t m{16}, n{16}, k{32};
};

[[nodiscard]] inline Kernel pointwise(Pointwise op, int64_t rows, int64_t columns,
                                      Block block = {}) {
    auto definition = tile_kernel("tile_pointwise", [=](TensorView<const float, 2> A,
                                                        TensorView<const float, 2> B,
                                                        TensorView<float, 2> C) {
        auto gm = axis("gm", ceil_div(rows, block.m));
        auto gn = axis("gn", ceil_div(columns, block.n));
        auto m = axis("m", block.m), n = axis("n", block.n);
        for (auto &nest : parallel(shape(gm, gn))) {
            auto origin = coord(nest.index(gm) * block.m, nest.index(gn) * block.n);
            auto a = A.tile(origin, shape(m, n), bounds::zero).load();
            auto result = a;
            switch (op) {
                case Pointwise::COPY: break;
                case Pointwise::ADD: result = a + B.tile(origin, shape(m, n)).load(); break;
                // Preserve the old expression, including its FP32 divisor.
                case Pointwise::SAXPY: result = a / 0.4f + B.tile(origin, shape(m, n)).load(); break;
                case Pointwise::CLAMP: result = min(max(a, -0.5f), 0.5f); break;
                case Pointwise::EXP: result = exp(a); break;
            }
            C(origin, shape(m, n)).store(result);
        }
    });
    return definition.capture(tensor_shape(rows, columns), tensor_shape(rows, columns),
                              tensor_shape(rows, columns));
}

// Deliberately unweighted, eps=1e-12: this is the legacy benchmark's RMSNorm,
// not the gamma-weighted eps=1e-5 LLM benchmark.
[[nodiscard]] inline Kernel rms_norm(int64_t rows, int64_t columns, int64_t block_rows = 4) {
    auto definition = tile_kernel("tile_unweighted_rmsnorm", [=](TensorView<const float, 2> A,
                                                                 TensorView<float, 2> B) {
        auto m = axis("m", block_rows), n = axis("n", columns);
        for (auto &nest : parallel(shape(ceil_div(rows, block_rows)))) {
            auto origin = coord(nest.index() * block_rows, 0);
            auto a = A.tile(origin, shape(m, n)).load();
            auto mean_square = reduce(a * a, n, add) / static_cast<float>(columns);
            B(origin, shape(m, n)).store(a * (1.0f / sqrt(mean_square + 1e-12f)));
        }
    });
    return definition.capture(tensor_shape(rows, columns), tensor_shape(rows, columns));
}

[[nodiscard]] inline Kernel row_reduce(RowReduction op, int64_t rows, int64_t columns,
                                       int64_t block_rows = 4) {
    auto definition = tile_kernel("tile_row_reduce", [=](TensorView<const float, 2> A,
                                                         TensorView<float, 1> R) {
        auto m = axis("m", block_rows), n = axis("n", columns);
        for (auto &nest : parallel(shape(ceil_div(rows, block_rows)))) {
            auto row = nest.index() * block_rows;
            auto a = A.tile(coord(row, 0), shape(m, n)).load();
            if (op == RowReduction::ABS_SUM || op == RowReduction::ABS_MAX) { a = abs(a); }
            auto result = op == RowReduction::MAX || op == RowReduction::ABS_MAX ? reduce(a, n, maximum) :
                          op == RowReduction::MIN                                ? reduce(a, n, minimum) :
                                                                                   reduce(a, n, add);
            R(coord(row), shape(m)).store(result);
        }
    });
    return definition.capture(tensor_shape(rows, columns), tensor_shape(rows));
}

// Blocked inclusive Hillis-Steele scan, composed from existing operations.
// A serial nest carries one value per row between bounded column chunks;
// a host loop stages O(log(chunk_columns)) SSA steps within each chunk.
// This is a portable example, NOT a claim of warp-prefix intrinsic lowering.
// Floating sum uses a reassociated tree; it does not promise fold-left bits.
[[nodiscard]] inline Kernel row_scan(Scan op, int64_t rows, int64_t columns,
                                     int64_t block_rows = 4, int64_t chunk_columns = 32) {
    auto definition = tile_kernel("tile_inclusive_scan", [=](TensorView<const float, 2> A,
                                                             TensorView<float, 2> B) {
        auto width = std::min(columns, chunk_columns);
        auto m = axis("m", block_rows), n = axis("n", width);
        auto identity = op == Scan::SUM ? 0.0f : -std::numeric_limits<float>::infinity();
        for (auto &nest : parallel(shape(ceil_div(rows, block_rows)))) {
            auto row = nest.index() * block_rows;
            auto carry = full<float>(shape(m), identity);
            for (auto &chunk : nest.serial(shape(ceil_div(columns, width)))) {
                auto origin = coord(row, chunk.index() * width);
                auto value = A.tile(origin, shape(m, n)).load(identity);
                for (auto offset = int64_t{1}; offset < width; offset *= 2) {
                    auto previous = gather(value, iota(n) - offset, n, identity);
                    value = op == Scan::SUM ? value + previous : max(value, previous);
                }
                value = op == Scan::SUM ? value + carry : max(value, carry);
                B(origin, shape(m, n)).store(value);
                carry = map<float>(shape(m), [&](const Nest &element) {
                    return value.at(coord(element.index(m), width - 1));
                });
            }
        }
    });
    return definition.capture(tensor_shape(rows, columns), tensor_shape(rows, columns));
}

[[nodiscard]] inline Kernel transpose(int64_t rows, int64_t columns, Block block = {}) {
    auto definition = tile_kernel("tile_transpose", [=](TensorView<const float, 2> A,
                                                        TensorView<float, 2> B) {
        auto gm = axis("gm", ceil_div(rows, block.m));
        auto gn = axis("gn", ceil_div(columns, block.n));
        auto m = axis("m", block.m), n = axis("n", block.n);
        for (auto &nest : parallel(shape(gm, gn))) {
            auto row = nest.index(gm) * block.m, column = nest.index(gn) * block.n;
            auto a = A.tile(coord(row, column), shape(m, n)).load();
            auto transposed = reindex(a, shape(n, m), [&](const Nest &element) {
                return coord(element.index(m), element.index(n));
            });
            B(coord(column, row), shape(n, m)).store(transposed);
        }
    });
    return definition.capture(tensor_shape(rows, columns), tensor_shape(columns, rows));
}

// Storage and accumulation precision are independent. float matches the large
// legacy GEMM; half matches the old 512^3 example with FP32 accumulation.
// Types and Tile sizes are ordinary C++ staging values.
template<scalar_cpp_type T = float, scalar_cpp_type Acc = float>
[[nodiscard]] inline Kernel gemm(int64_t rows, int64_t columns, int64_t depth, Block block = {}) {
    auto definition = tile_kernel("tile_gemm", [=](TensorView<const T, 2> A,
                                                   TensorView<const T, 2> B,
                                                   TensorView<T, 2> C) {
        auto gm = axis("gm", ceil_div(rows, block.m));
        auto gn = axis("gn", ceil_div(columns, block.n));
        auto m = axis("m", block.m), n = axis("n", block.n), k = axis("k", block.k);
        for (auto &nest : parallel(shape(gm, gn))) {
            auto row = nest.index(gm) * block.m, column = nest.index(gn) * block.n;
            auto acc = zeros<Acc>(shape(m, n));
            for (auto &step : nest.pipeline(shape(ceil_div(depth, block.k)), {.stages = 1u, .initiation_interval = 1u})) {
                step.stage("load");
                auto a = A.tile(coord(row, step.index() * block.k), shape(m, k)).load();
                auto b = B.tile(coord(step.index() * block.k, column), shape(k, n)).load();
                step.stage("compute");
                acc = mma(a, b, acc);
            }
            C(coord(row, column), shape(m, n)).store(cast<T>(acc));
        }
    });
    return definition.capture(tensor_shape(rows, depth), tensor_shape(depth, columns), tensor_shape(rows, columns));
}

}// namespace luisa::example::tile
