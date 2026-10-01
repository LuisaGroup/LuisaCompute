#pragma once

// Actual Tile DSL fixtures shared by the CUDA/SIMD workload benchmark.
// Exported inputs, rather than an operation-name-dependent device substitute,
// define the independent Python comparison.
#include "tile_llm_test_utils.h"
#include "tile_rank_test_utils.h"
#include "tile_selection_test_utils.h"
#include "tile_argmax_test_utils.h"
#include "tile_embedding_test_utils.h"
#include "tile_sort_pipeline_test_utils.h"
#include <luisa/tile/algorithms.h>
#include <luisa/core/stl/optional.h>
#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <numeric>

namespace luisa::test::tile_workloads {

struct Options {
    string backend, lowering, operation, precision, pattern;
    string ranking_algorithm{"full_sort_prefix"};
    vector<int64_t> dimensions;
    std::array<int64_t, 3u> tile{};
    bool fast_math{false};
    uint64_t seed{0u};
    uint32_t samples{7u}, sample_ms{100u}, warmup_ms{500u}, graph_batch{0u};
};

struct Fixture {
    optional<compute::tile::Kernel> kernel;
    // First stage stays in kernel; the following stages execute in order.
    vector<compute::tile::Kernel> continuation_kernels;
    vector<int64_t> pipeline_widths;
    size_t scratch_elements{0u};
    int64_t pipeline_chunk{0};
    std::array<vector<int64_t>, 3u> input_shapes;
    std::array<vector<float>, 3u> inputs;
    // Embedding alone has a second, genuinely INT64 input. Other operators'
    // three homogeneous input descriptors and ABI remain unchanged.
    optional<vector<int64_t>> input_ids;
    vector<int64_t> output_shape;
    vector<double> expected, bound;
    vector<double> strict_bound, probability_rounding_bound;
    vector<int64_t> expected_indices;
    string error, algorithm;
    bool ranking{false};
};

[[nodiscard]] inline uint64_t random_bits(uint64_t &state) noexcept {
    auto x = (state += 0x9e3779b97f4a7c15ull);
    x = (x ^ (x >> 30u)) * 0xbf58476d1ce4e5b9ull;
    x = (x ^ (x >> 27u)) * 0x94d049bb133111ebull;
    return x ^ (x >> 31u);
}

[[nodiscard]] inline float random_value(uint64_t &state) noexcept {
    // Exact 24-bit mantissa construction; no platform-dependent libm RNG.
    return static_cast<float>(random_bits(state) >> 40u) * 0x1p-23f - 1.0f;
}

[[nodiscard]] inline bool product_bounded(span<const int64_t> dimensions, uint64_t limit) noexcept {
    auto product = uint64_t{1u};
    for (auto d : dimensions) {
        if (d <= 0 || static_cast<uint64_t>(d) > limit / product) { return false; }
        product *= static_cast<uint64_t>(d);
    }
    return true;
}

[[nodiscard]] inline size_t volume(span<const int64_t> dimensions) noexcept {
    size_t n = 1u;
    for (auto d : dimensions) { n *= static_cast<size_t>(d); }
    return n;
}

[[nodiscard]] inline double sum_bound(uint64_t count, double absolute_sum) noexcept {
    auto q = static_cast<double>(2u * count + 2u);
    auto u = 0x1p-24;
    return (q * u / (1.0 - q * u) + 8.0 * count * 0x1p-53) * absolute_sum + q * 0x1p-149;
}

[[nodiscard]] inline bool is_row(string_view op) noexcept {
    return op == "rmsnorm" || op == "layernorm" || op == "softmax" || op == "masked_softmax" ||
           op == "rope" || op == "swiglu" || op == "gelu_residual" || op == "reduce_sum" ||
           op == "reduce_max" || op == "scan" || op == "scan_ordered";
}

template<typename T>
inline void build_rows(Fixture &f, const Options &o) {
    using namespace compute::tile;
    auto rows = o.dimensions[0], width = o.dimensions[1], tile_width = o.tile[1], block_rows = o.tile[0];
    auto op = o.operation;
    auto norm = op == "rmsnorm" || op == "layernorm";
    auto reduction = op == "reduce_sum" || op == "reduce_max";
    f.input_shapes = {vector<int64_t>{rows, width}, {norm ? 1 : rows, op == "rope" ? width / 2 : width}, {norm ? 1 : rows, op == "rope" ? width / 2 : width}};
    f.output_shape = {rows, reduction ? 1 : width};
    f.algorithm = op == "scan" ? "inclusive_sum_unordered_tree" : op == "scan_ordered" ? "quadratic_reference_ordered_inclusive_scan" :
                                                                                         "whole_row_tile_fp32_compute";
    auto pointwise = op == "swiglu" || op == "gelu_residual" || op == "rope";
    auto logical_width = op == "rope" ? width / 2 : width;
    if (pointwise && tile_width < logical_width) {
        f.algorithm = "feature_tiled_pointwise_fp32_compute";
        auto blocks = ceil_div(logical_width, tile_width);
        if (op == "rope") {
            // Keep the physical pair axis: a feature tail cannot overwrite
            // the other half of the logical RoPE row.
            auto definition = tile_kernel("workload_rope_feature_tiles", [=](TensorView<const T, 3> X,
                                                                               TensorView<const T, 3> C,
                                                                               TensorView<const T, 3> S,
                                                                               TensorView<T, 3> Y) {
                auto program_row = axis("program_row", rows), block = axis("feature_block", blocks);
                auto m = axis("m", 1), pair = axis("pair", 1), n = axis("n", tile_width);
                for (auto &nest : parallel(shape(program_row, block))) {
                    auto r = nest.index(program_row), start = nest.index(block) * tile_width;
                    auto x = cast<float>(X.tile(coord(r, 0, start), shape(m, pair, n)).load());
                    auto y = cast<float>(X.tile(coord(r, 1, start), shape(m, pair, n)).load());
                    auto c = cast<float>(C.tile(coord(r, 0, start), shape(m, pair, n)).load());
                    auto s = cast<float>(S.tile(coord(r, 0, start), shape(m, pair, n)).load());
                    Y(coord(r, 0, start), shape(m, pair, n)).store(cast<T>(x * c - y * s));
                    Y(coord(r, 1, start), shape(m, pair, n)).store(cast<T>(x * s + y * c));
                }
            });
            f.kernel = definition.capture(tensor_shape(rows, 2, width / 2), tensor_shape(rows, 1, width / 2),
                                          tensor_shape(rows, 1, width / 2), tensor_shape(rows, 2, width / 2));
        } else {
            auto definition = tile_kernel("workload_pointwise_feature_tiles", [=](TensorView<const T, 2> X,
                                                                                  TensorView<const T, 2> U,
                                                                                  TensorView<const T, 2> V,
                                                                                  TensorView<T, 2> Y) {
                auto program_row = axis("program_row", rows), block = axis("feature_block", blocks);
                auto m = axis("m", 1), n = axis("n", tile_width);
                for (auto &nest : parallel(shape(program_row, block))) {
                    auto row = nest.index(program_row), start = nest.index(block) * tile_width;
                    auto x = cast<float>(X.tile(coord(row, start), shape(m, n)).load());
                    auto result = op == "swiglu" ?
                        x / (1.0f + exp(-x)) * cast<float>(U.tile(coord(row, start), shape(m, n)).load()) :
                        0.5f * x * (1.0f + tanh(0.7978845608f * (x + 0.044715f * x * x * x))) + cast<float>(U.tile(coord(row, start), shape(m, n)).load());
                    Y(coord(row, start), shape(m, n)).store(cast<T>(result));
                }
            });
            f.kernel = definition.capture(tensor_shape(rows, width), tensor_shape(rows, width),
                                          tensor_shape(rows, width), tensor_shape(rows, width));
        }
        return;
    }
    if (op == "rope") {
        // A physical pair axis makes a padded half-width Tile safe: stores
        // cannot overlap the other half when the logical half has a tail.
        auto definition = tile_kernel("workload_rope", [=](TensorView<const T, 3> X,
                                                           TensorView<const T, 3> C,
                                                           TensorView<const T, 3> S,
                                                           TensorView<T, 3> Y) {
            auto m = axis("m", 1), pair = axis("pair", 1), n = axis("n", tile_width);
            for (auto &nest : parallel(shape(rows))) {
                auto r = nest.index();
                auto x = cast<float>(X.tile(coord(r, 0, 0), shape(m, pair, n)).load());
                auto y = cast<float>(X.tile(coord(r, 1, 0), shape(m, pair, n)).load());
                auto c = cast<float>(C.tile(coord(r, 0, 0), shape(m, pair, n)).load());
                auto s = cast<float>(S.tile(coord(r, 0, 0), shape(m, pair, n)).load());
                Y(coord(r, 0, 0), shape(m, pair, n)).store(cast<T>(x * c - y * s));
                Y(coord(r, 1, 0), shape(m, pair, n)).store(cast<T>(x * s + y * c));
            }
        });
        f.kernel = definition.capture(tensor_shape(rows, 2, width / 2), tensor_shape(rows, 1, width / 2),
                                      tensor_shape(rows, 1, width / 2), tensor_shape(rows, 2, width / 2));
        return;
    }
    auto definition = tile_kernel("workload_rows", [=](TensorView<const T, 2> X,
                                                       TensorView<const T, 2> U,
                                                       TensorView<const T, 2> V,
                                                       TensorView<T, 2> Y) {
        auto m = axis("m", block_rows), n = axis("n", tile_width);
        for (auto &nest : parallel(shape(ceil_div(rows, block_rows)))) {
            // Preserve the BR1 capture; explicit blocked schedules only change
            // independent row coordinates, never the scan/reduction axis.
            auto row = block_rows == 1 ? nest.index() : nest.index() * block_rows;
            auto x = cast<float>(X.tile(coord(row, 0), shape(m, n)).load());
            auto valid = iota(n) < width;
            if (op == "reduce_sum" || op == "reduce_max") {
                auto result = op == "reduce_sum" ? reduce(x, n, add) : reduce(ite(valid, x, -std::numeric_limits<float>::infinity()), n, maximum);
                Y(coord(row, 0), shape(m, axis("out", 1))).store(cast<T>(result));
            } else if (op == "scan") {
                auto result = inclusive_sum(x, n, reduction::unordered_tree);
                Y(coord(row, 0), shape(m, n)).store(cast<T>(result));
            } else if (op == "scan_ordered") {
                auto k = axis("prefix", tile_width);
                auto result = map<float>(shape(m, n), [&](const Nest &item) {
                    auto sum = Scalar<float>{0.0f};
                    for (auto &part : item.reduce(shape(k), reduction::fold_left)) {
                        auto index = part.index(k);
                        sum += ite(index <= item.index(n), x.at(coord(0, index)), 0.0f);
                    }
                    return sum;
                });
                Y(coord(row, 0), shape(m, n)).store(cast<T>(result));
            } else {
                auto result = x;
                if (op == "rmsnorm" || op == "layernorm") {
                    auto centered = op == "layernorm" ? x - reduce(x, n, add) / static_cast<float>(width) : x;
                    auto square = ite(valid, centered * centered, 0.0f);
                    auto variance = reduce(square, n, add) / static_cast<float>(width);
                    result = centered / sqrt(variance + 1e-5f) * cast<float>(U.tile(coord(0, 0), shape(m, n)).load());
                    if (op == "layernorm") { result += cast<float>(V.tile(coord(0, 0), shape(m, n)).load()); }
                } else if (op == "swiglu") {
                    result = x / (1.0f + exp(-x)) * cast<float>(U.tile(coord(row, 0), shape(m, n)).load());
                } else if (op == "gelu_residual") {
                    result = 0.5f * x * (1.0f + tanh(0.7978845608f * (x + 0.044715f * x * x * x))) + cast<float>(U.tile(coord(row, 0), shape(m, n)).load());
                } else {
                    auto mask = op == "masked_softmax" ? valid && (iota(n) <= row % width) : valid;
                    auto score = ite(mask, x, -1e30f);
                    auto exponential = ite(mask, exp(score - reduce(score, n, maximum)), 0.0f);
                    result = exponential / reduce(exponential, n, add);
                }
                Y(coord(row, 0), shape(m, n)).store(cast<T>(result));
            }
        }
    });
    f.kernel = definition.capture(tensor_shape(rows, width), tensor_shape(norm ? 1 : rows, width),
                                  tensor_shape(norm ? 1 : rows, width), tensor_shape(rows, reduction ? 1 : width));
}

inline void row_oracle(Fixture &f, const Options &o) {
    auto rows = o.dimensions[0], width = o.dimensions[1];
    auto &x = f.inputs[0];
    auto &u = f.inputs[1];
    auto &v = f.inputs[2];
    auto reduction = o.operation == "reduce_sum" || o.operation == "reduce_max";
    for (int64_t row = 0; row < rows; row++) {
        auto base = row * width;
        double sum = 0.0, absolute = 0.0, peak = -std::numeric_limits<double>::infinity();
        auto valid_count = o.operation == "masked_softmax" ? row % width + 1 : width;
        for (int64_t col = 0; col < width; col++) {
            sum += x[base + col];
            absolute += std::abs(static_cast<double>(x[base + col]));
            if (col < valid_count) { peak = std::max(peak, static_cast<double>(x[base + col])); }
        }
        if (reduction) {
            f.expected[row] = o.operation == "reduce_sum" ? sum : peak;
            f.bound[row] = o.operation == "reduce_sum" ? sum_bound(width, absolute) : 0.0;
            continue;
        }
        auto mean = o.operation == "layernorm" ? sum / width : 0.0;
        double variance = 0.0, denominator = 0.0, prefix = 0.0, prefix_absolute = 0.0;
        for (int64_t col = 0; col < width; col++) {
            auto centered = static_cast<double>(x[base + col]) - mean;
            variance += centered * centered / width;
            if (col < valid_count) { denominator += std::exp(static_cast<double>(x[base + col]) - peak); }
        }
        for (int64_t col = 0; col < width; col++) {
            auto i = base + col;
            auto value = static_cast<double>(x[i]);
            if (o.operation == "rmsnorm" || o.operation == "layernorm") {
                value = (value - mean) / std::sqrt(variance + static_cast<double>(1e-5f)) * u[col];
                if (o.operation == "layernorm") { value += v[col]; }
            } else if (o.operation == "swiglu") {
                value = value / (1.0 + std::exp(-value)) * u[i];
            } else if (o.operation == "gelu_residual") {
                value = .5 * value * (1.0 + std::tanh(static_cast<double>(0.7978845608f) * (value + static_cast<double>(0.044715f) * value * value * value))) + u[i];
            } else if (o.operation == "rope") {
                auto j = row * (width / 2) + col % (width / 2);
                auto left = static_cast<double>(x[base + col % (width / 2)]);
                auto right = static_cast<double>(x[base + col % (width / 2) + width / 2]);
                value = col < width / 2 ? left * u[j] - right * v[j] : left * v[j] + right * u[j];
            } else if (o.operation == "scan" || o.operation == "scan_ordered") {
                prefix += value;
                prefix_absolute += std::abs(value);
                f.expected[i] = prefix;
                f.bound[i] = sum_bound(col + 1, prefix_absolute);
                continue;
            } else {
                value = col < valid_count ? std::exp(value - peak) / denominator : 0.0;
            }
            f.expected[i] = value;
            f.bound[i] = 5e-5 + 5e-5 * std::abs(value);
        }
    }
}

template<typename T = float>
[[nodiscard]] inline Fixture make_fixture(const Options &o) {
    using namespace compute::tile;
    Fixture f;
    auto op = string_view{o.operation};
    auto rows = is_row(op);
    auto gemm = op == "gemm" || op == "gemv";
    auto bmm = op == "bmm";
    auto first_max = op == "argmax";
    auto embedding = op == "embedding";
    auto rank = op == "sort" || op == "topk" || first_max;
    auto tensorcore_attention = op == "attention_tensorcore";
    auto attention = op == "attention" || tensorcore_attention;
    auto chunked = o.ranking_algorithm == "chunked_bitonic_c256" || o.ranking_algorithm == "chunked_bitonic_c512";
    if ((o.ranking_algorithm != "full_sort_prefix" && o.ranking_algorithm != "packed_fp32" && o.ranking_algorithm != "repeated_extrema" && !chunked) ||
        (o.ranking_algorithm == "packed_fp32" && op != "sort" && op != "topk") ||
        (o.ranking_algorithm == "repeated_extrema" && op != "topk") || (chunked && op != "sort")) {
        f.error = "ranking_algorithm must be full_sort_prefix/packed_fp32, repeated_extrema for topk, or chunked_bitonic_c256/c512 for sort";
        return f;
    }
    if constexpr (std::is_same_v<T, float>) {
        if (tensorcore_attention) {
            f.error = "attention_tensorcore requires FP16/BF16 storage";
            return f;
        }
    }
    if (!rows && !gemm && !bmm && !rank && !attention && !embedding) {
        f.error = "operation has no benchmark fixture";
        return f;
    }
    auto count = rows || first_max ? 2u : attention ? 7u : bmm ? 4u : 3u;
    // The complete matrix FP64 oracle visits one product per B*M*N*K term.
    // This work limit is separate from each tensor's unchanged allocation cap.
    auto work_limit = embedding ? (1ull << 48u) : gemm || bmm ? (1ull << 34u) : attention ? (1ull << 31u) : (1ull << 28u);
    if (o.dimensions.size() != count || !product_bounded(o.dimensions, work_limit)) {
        f.error = "invalid dimensions or fixture work bound exceeded";
        return f;
    }
    if (gemm || bmm) {
        auto base = bmm ? 1u : 0u;
        auto batches = bmm ? o.dimensions[0] : int64_t{1};
        auto m = o.dimensions[base], n = o.dimensions[base + 1u], k = o.dimensions[base + 2u];
        auto bm = o.tile[0], bn = o.tile[1], bk = o.tile[2];
        if (std::any_of(o.dimensions.begin(), o.dimensions.end(), [](int64_t value) noexcept { return value > 65536; }) ||
            bm <= 0 || bn <= 0 || bk <= 0 || bm > 128 || bn > 128 || bk > (op == "gemv" ? 1024 : 256) ||
            (bmm && (!std::has_single_bit(static_cast<uint64_t>(bm)) || !std::has_single_bit(static_cast<uint64_t>(bn)) ||
                     !std::has_single_bit(static_cast<uint64_t>(bk)))) || (op == "gemv" && n != 1)) {
            f.error = "matrix fixtures require dimensions<=65536, bounded tile extents, BMM power-of-two tiles and GEMV N=1";
            return f;
        }
        if (!product_bounded(std::array{batches, m, k}, 1ull << 24u) ||
            !product_bounded(std::array{batches, k, n}, 1ull << 24u) ||
            !product_bounded(std::array{batches, m, n}, 1ull << 24u) ||
            (bmm && ceil_div(m, bm) > 65535) || ((bmm || o.backend == "cuda") && ceil_div(n, bn) > 65535)) {
            f.error = "matrix input/output allocation exceeds 2^24 elements or launch grid exceeds CUDA limits";
            return f;
        }
    }
    if (rows) {
        auto width = o.dimensions[1];
        auto logical_width = op == "rope" ? width / 2 : width;
        auto independent_rows = op == "scan" || op == "reduce_sum" || op == "reduce_max";
        auto pointwise = op == "swiglu" || op == "gelu_residual" || op == "rope";
        auto valid_row_block = o.tile[0] == 1 || (independent_rows && (o.tile[0] == 4 || o.tile[0] == 8));
        if (!valid_row_block || o.tile[2] != 1 || o.tile[1] <= 0 || (!pointwise && o.tile[1] < logical_width) || o.tile[1] > 16384 ||
            (op == "rope" && width % 2 != 0) || (op == "scan_ordered" && o.tile[1] > 1024)) {
            f.error = "row schedule requires BR1 (or BR4/8 for scan/sum/max), positive BD (full width except swiglu/gelu_residual/rope) and K=1; ordered scan width <=1024 and RoPE width even";
            return f;
        }
        if (o.backend == "cuda" && pointwise && ceil_div(logical_width, o.tile[1]) > 65535) {
            f.error = "pointwise feature launch grid exceeds CUDA limits";
            return f;
        }
        build_rows<T>(f, o);
    } else if (bmm) {
        auto batches = o.dimensions[0], m = o.dimensions[1], n = o.dimensions[2], k = o.dimensions[3];
        auto bm = o.tile[0], bn = o.tile[1], bk = o.tile[2];
        auto definition = tile_kernel("workload_bmm", [=](TensorView<const T, 3> A, TensorView<const T, 3> B,
                                                       TensorView<const T, 1> Unused, TensorView<T, 3> C) {
            static_cast<void>(Unused);
            auto batch = axis("batch", 1), im = axis("m", bm), jn = axis("n", bn), kk = axis("k", bk);
            auto gb = axis("gb", batches), gm = axis("gm", ceil_div(m, bm)), gn = axis("gn", ceil_div(n, bn));
            for (auto &nest : parallel(shape(gb, gm, gn))) {
                auto bi = nest.index(gb), mi = nest.index(gm) * bm, nj = nest.index(gn) * bn;
                auto acc = zeros<float>(shape(batch, im, jn));
                for (auto &step : nest.pipeline(shape(ceil_div(k, bk)), {.window = 2u, .interval = 1u})) {
                    auto a = A.tile(coord(bi, mi, step.index() * bk), shape(batch, im, kk)).load();
                    auto b = B.tile(coord(bi, step.index() * bk, nj), shape(batch, kk, jn)).load();
                    acc = mma(a, b, acc, {.allow_reassociation = true});
                }
                C(coord(bi, mi, nj), shape(batch, im, jn)).store(cast<T>(acc));
            }
        });
        f.kernel = definition.capture(tensor_shape(batches, m, k), tensor_shape(batches, k, n), tensor_shape(1), tensor_shape(batches, m, n));
        f.input_shapes = {vector<int64_t>{batches, m, k}, {batches, k, n}, {1}};
        f.output_shape = {batches, m, n};
        f.algorithm = "tile_bmm_singleton_batch_typed_inputs_fp32_accumulator_reassociation";
    } else if (gemm) {
        auto m = o.dimensions[0], n = o.dimensions[1], k = o.dimensions[2];
        auto bm = o.tile[0], bn = o.tile[1], bk = o.tile[2];
        auto definition = tile_kernel("workload_gemm", [=](TensorView<const T, 2> A, TensorView<const T, 2> B,
                                                           TensorView<const T, 1> Unused, TensorView<T, 2> C) {
            static_cast<void>(Unused);
            auto im = axis("m", bm), jn = axis("n", bn), kk = axis("k", bk);
            auto gm = axis("gm", ceil_div(m, bm)), gn = axis("gn", ceil_div(n, bn));
            for (auto &nest : parallel(shape(gm, gn))) {
                auto mi = nest.index(gm) * bm, nj = nest.index(gn) * bn;
                auto acc = zeros<float>(shape(im, jn));
                for (auto &step : nest.pipeline(shape(ceil_div(k, bk)), {.window = 2u, .interval = 1u})) {
                    auto a = A.tile(coord(mi, step.index() * bk), shape(im, kk)).load();
                    auto b = B.tile(coord(step.index() * bk, nj), shape(kk, jn)).load();
                    if (op == "gemv") {
                        acc += reduce(cast<float>(a) * cast<float>(b), kk, add);
                    } else {
                        acc = mma(a, b, acc, {.allow_reassociation = true});
                    }
                }
                C(coord(mi, nj), shape(im, jn)).store(cast<T>(acc));
            }
        });
        f.kernel = definition.capture(tensor_shape(m, k), tensor_shape(k, n), tensor_shape(1), tensor_shape(m, n));
        f.input_shapes = {vector<int64_t>{m, k}, {k, n}, {1}};
        f.output_shape = {m, n};
        f.algorithm = op == "gemv" ? "tile_gemv_product_tree_sum" : "tile_mma_typed_inputs_fp32_accumulator_reassociation";
    } else if (embedding) {
        auto vocabulary = o.dimensions[0], width = o.dimensions[1], tokens = o.dimensions[2];
        if (std::any_of(o.dimensions.begin(), o.dimensions.end(), [](int64_t d) noexcept { return d > 65536; }) ||
            (o.tile[0] != 1 && o.tile[0] != 4 && o.tile[0] != 8) || o.tile[2] != 1 || o.tile[1] <= 0 || o.tile[1] > 16384 ||
            !product_bounded(std::array{vocabulary, width}, 1ull << 24u) ||
            !product_bounded(std::array{tokens, width}, 1ull << 24u) ||
            (o.pattern != "random" && o.pattern != "adversarial")) {
            f.error = "embedding requires V,D,T<=65536, tile=(BR1/4/8,BD<=16384,1), tensors<=2^24 and random/adversarial pattern";
            return f;
        }
        if (o.backend == "cuda" && (width + o.tile[1] - 1) / o.tile[1] > 65535) {
            f.error = "embedding feature launch grid exceeds CUDA limits";
            return f;
        }
        f.input_ids.emplace(static_cast<size_t>(tokens));
        uint64_t id_state = o.seed ^ 0x454d42454444494eull;
        for (int64_t token = 0; token < tokens; token++) {
            (*f.input_ids)[token] = o.pattern == "adversarial" ?
                (token % 3 == 0 ? vocabulary - 1 : token % 3 == 1 ? int64_t{0} : vocabulary / 2) :
                static_cast<int64_t>(random_bits(id_state) % static_cast<uint64_t>(vocabulary));
        }
        if (!tile_embedding::valid_row_indices(span<const int64_t>{*f.input_ids}, vocabulary)) {
            f.error = "embedding input IDs must be exact INT64 values in [0,V)";
            return f;
        }
        f.kernel = tile_embedding::embedding_rows<T>(vocabulary, width, tokens, o.tile[1], o.tile[0]);
        f.input_shapes = {vector<int64_t>{vocabulary, width}, {1}, {1}};
        f.output_shape = {tokens, width};
        f.algorithm = o.tile[0] == 1 ? "uniform_int64_row_gather" : "serial_grouped_uniform_int64_row_gather";
    } else if (first_max) {
        auto r = o.dimensions[0], n = o.dimensions[1], padded = o.tile[1];
        if (o.tile[0] != 1 || o.tile[2] != 1 || padded < n || padded > 16384 ||
            !product_bounded(o.dimensions, 1ull << 24u)) {
            f.error = "argmax requires tile=(1,padded_width>=N,1), width<=16384 and bounded input allocation";
            return f;
        }
        f.kernel = tile_selection::stable_argmax<T>(r, n, padded);
        f.input_shapes = {vector<int64_t>{r, n}, {1}, {1}};
        f.output_shape = {r, 1};
        f.ranking = true;
        f.algorithm = "stable_first_index_argmax";
    } else if (rank) {
        auto r = o.dimensions[0], n = o.dimensions[1], k = o.dimensions[2];
        auto padded = o.tile[1];
        if (k > n || n > 16384 || (op == "sort" && k != n) || o.tile[0] != 1 || o.tile[2] != 1 ||
            padded < n || padded > 16384 || !std::has_single_bit(static_cast<uint64_t>(padded))) {
            f.error = "ranking requires K<=N (sort K=N), tile=(1,power_of_two>=N,1), padded width<=16384";
            return f;
        }
        auto definition = tile_kernel("workload_padded_ranking", [=](TensorView<const T, 2> X,
                                                                     TensorView<T, 2> Values,
                                                                     TensorView<int64_t, 2> Indices) {
            auto local_row = axis("row", 1), column = axis("column", padded);
            for (auto &nest : parallel(shape(r))) {
                auto origin = coord(nest.index(), 0);
                auto loaded = cast<float>(X.tile(origin, shape(local_row, column)).load());
                auto finite = ite(iota(column) < n, loaded, -std::numeric_limits<float>::infinity());
                // Keep the sorting network, but expose its needed prefix to
                // native Tile IR before the store mask. Power-of-two padding
                // also covers a logical K such as 7 without native K=7 Tiles.
                auto algorithm = o.ranking_algorithm == "packed_fp32" ? SortAlgorithm::PACKED_FP32 : SortAlgorithm::DEFAULT;
                auto ranked = op == "topk" ?
                    compute::tile::topk(finite, column, std::bit_ceil(static_cast<uint64_t>(k)), true, algorithm) :
                    compute::tile::sort(finite, column, true, algorithm);
                // This remains full-sort-prefix, not a selection algorithm.
                Values(origin, ranked.values.space()).store(cast<T>(ranked.values));
                Indices(origin, ranked.indices.space()).store(ranked.indices);
            }
        });
        if (chunked) {
            auto chunk = o.ranking_algorithm == "chunked_bitonic_c256" ? 256ll : 512ll;
            auto plan = tile_sort_pipeline::plan(r, n, chunk);
            if (!plan.error.empty()) { f.error = plan.error; return f; }
            if (plan.padded != padded) { f.error = "chunked sort requires tile width exactly bit_ceil(N)"; return f; }
            f.pipeline_chunk = chunk;
            for (auto &&stage : plan.stages) { f.pipeline_widths.emplace_back(stage.width); }
            if (plan.stages.size() == 1u) {
                f.kernel = tile_sort_pipeline::initialize<T, T, int64_t>(plan);
            } else {
                f.scratch_elements = static_cast<size_t>(r * padded);
                f.kernel = tile_sort_pipeline::initialize<T>(plan);
                for (auto i = size_t{1u}; i < plan.stages.size(); i++) {
                    auto &&stage = plan.stages[i];
                    if (stage.final) { f.continuation_kernels.emplace_back(tile_sort_pipeline::merge_whole<T, int64_t>(plan, stage)); }
                    else { f.continuation_kernels.emplace_back(tile_sort_pipeline::merge_whole<>(plan, stage)); }
                }
            }
        } else if (o.ranking_algorithm == "repeated_extrema") {
            f.kernel = tile_selection::repeated_extrema_topk<T>(r, n, k, padded);
        } else {
            f.kernel = definition.capture(tensor_shape(r, n), tensor_shape(r, k), tensor_shape(r, k));
        }
        f.input_shapes = {vector<int64_t>{r, n}, {1}, {1}};
        f.output_shape = {r, k};
        f.ranking = true;
        f.algorithm = o.ranking_algorithm == "packed_fp32" ? "stable_packed_fp32_full_sort_prefix" :
                      chunked ? "stable_chunked_bitonic_whole_tile_merge" :
                      o.ranking_algorithm == "repeated_extrema" ? "stable_repeated_extrema" :
                      op == "sort" ? "padded_bitonic_full_sort" : "padded_bitonic_full_sort_prefix";
    } else {
        auto &d = o.dimensions;
        if (d[1] % d[2] != 0 || d[4] < d[3] || o.tile[0] > 128 || o.tile[1] > 256 || o.tile[2] != 1 ||
            !product_bounded(std::array{d[0], d[1], d[3], d[4], d[5] + d[6]}, 1ull << 28u)) {
            f.error = "attention requires Hq divisible by Hkv, K>=Q, tile=(BQ,BK,1), bounded oracle work";
            return f;
        }
        auto batches = d[0], heads = d[1], kv_heads = d[2], queries = d[3], keys = d[4], channels = d[5], value_channels = d[6];
        auto bq = o.tile[0], bk = o.tile[1];
        auto scale = 1.0f / std::sqrt(static_cast<float>(channels));
        auto definition = tile_kernel("workload_attention", [=](TensorView<const T, 4> Q, TensorView<const T, 4> K,
                                                                TensorView<const T, 4> V, TensorView<T, 4> O) {
            auto batch = axis("batch", batches), head = axis("head", heads), query_block = axis("query_block", ceil_div(queries, bq));
            auto b = axis("b", 1), h = axis("h", 1), m = axis("m", bq), n = axis("n", bk);
            auto channel = axis("d", channels), dv = axis("dv", value_channels);
            for (auto &nest : parallel(shape(batch, head, query_block))) {
                auto b0 = nest.index(batch), h0 = nest.index(head), q0 = nest.index(query_block) * bq;
                auto kh = h0 / (heads / kv_heads);
                auto query = Q.tile(coord(b0, h0, q0, 0), shape(b, h, m, channel)).load();
                auto row_max = full<float>(shape(b, h, m), -1e30f);
                auto row_sum = zeros<float>(shape(b, h, m));
                auto acc = zeros<float>(shape(b, h, m, dv));
                for (auto &step : nest.pipeline(shape(ceil_div(keys, bk)), {.window = 2u, .interval = 1u})) {
                    auto k0 = step.index() * bk;
                    auto key = K.tile(coord(b0, kh, k0, 0), shape(b, h, n, channel)).load();
                    auto stored_value = V.tile(coord(b0, kh, k0, 0), shape(b, h, n, dv)).load();
                    auto score = mma(query, key, zeros<float>(shape(b, h, m, n)), {.allow_reassociation = true}) * scale;
                    auto valid = (iota(n) + k0 < keys) && (iota(n) + k0 <= iota(m) + q0 + keys - queries);
                    auto masked = ite(valid, score, -1e30f);
                    auto next_max = max(row_max, reduce(masked, n, maximum));
                    auto alpha = exp(row_max - next_max);
                    auto probability = ite(valid, exp(masked - next_max), 0.0f);
                    row_sum = row_sum * alpha + reduce(probability, n, add);
                    if (tensorcore_attention) {
                        // Explicit numerical contract: only the PV contribution
                        // is narrowed; the denominator and accumulator stay FP32.
                        acc = mma(cast<T>(probability), stored_value, acc * alpha, {.allow_reassociation = true});
                    } else {
                        acc = mma(probability, cast<float>(stored_value), acc * alpha, {.allow_reassociation = true});
                    }
                    row_max = next_max;
                }
                O(coord(b0, h0, q0, 0), shape(b, h, m, dv)).store(cast<T>(acc / row_sum));
            }
        });
        f.input_shapes = {vector<int64_t>{batches, heads, queries, channels}, {batches, kv_heads, keys, channels}, {batches, kv_heads, keys, value_channels}};
        f.output_shape = {batches, heads, queries, value_channels};
        f.kernel = definition.capture(tensor_shape(batches, heads, queries, channels), tensor_shape(batches, kv_heads, keys, channels),
                                      tensor_shape(batches, kv_heads, keys, value_channels), tensor_shape(batches, heads, queries, value_channels));
        f.algorithm = tensorcore_attention ? "causal_online_softmax_gqa_narrow_pv" : "causal_online_softmax_gqa";
    }
    for (auto &shape : f.input_shapes) {
        if (!product_bounded(shape, 1ull << 24u)) {
            f.error = "input allocation exceeds 2^24 elements";
            return f;
        }
    }
    auto state = o.seed;
    for (size_t input = 0; input < 3u; input++) {
        auto &data = f.inputs[input];
        data.resize(volume(f.input_shapes[input]));
        for (size_t i = 0; i < data.size(); i++) {
            auto x = random_value(state);
            if (o.pattern == "cancellation") { x = (i % 2u == 0u ? 1.0f : -1.0f) * (1.0f + std::abs(x) * 0x1p-12f); }
            if (o.pattern == "adversarial") {
                x = rank ? static_cast<float>((i + o.seed) % 7u) - 3.0f :
                           (i % 17u == 0u ? 16.0f : i % 19u == 0u ? -16.0f :
                                                                    x * 0x1p-8f);
            }
            if (embedding && input == 0u && o.pattern == "adversarial") {
                if (i % 23u == 0u) { x = -0.0f; }
                if (i % 23u == 1u) { x = 0.0f; }
            }
            if (rows && input > 0u) {
                if (op == "rmsnorm" || op == "layernorm") { x = input == 1u ? 1.0f + .2f * x : .1f * x; }
                if (op == "rope") { x = input == 1u ? std::cos(x) : std::sin(x); }
            }
            data[i] = x;
        }
    }
    if ((gemm || bmm) && o.pattern == "cancellation") {
        auto base = bmm ? 1u : 0u;
        auto batches = bmm ? o.dimensions[0] : int64_t{1};
        auto m = o.dimensions[base], n = o.dimensions[base + 1u], k = o.dimensions[base + 2u];
        for (int64_t batch = 0; batch < batches; batch++) {
            auto a_offset = batch * m * k, b_offset = batch * k * n;
            for (int64_t t = 0; t + 1 < k; t += 2) {
                for (int64_t i = 0; i < m; i++) { f.inputs[0][a_offset + i * k + t + 1] = f.inputs[0][a_offset + i * k + t]; }
                for (int64_t j = 0; j < n; j++) { f.inputs[1][b_offset + (t + 1) * n + j] = -f.inputs[1][b_offset + t * n + j] * (1.0f - 0x1p-20f); }
            }
        }
    }
    // These float vectors hold exact decoded storage values, never an
    // unquantized surrogate. Export/runtime convert back to identical bits.
    for (auto &input : f.inputs) {
        for (auto &value : input) { value = static_cast<float>(T{value}); }
    }
    if (rank && std::any_of(f.inputs[0].begin(), f.inputs[0].end(), [](float value) noexcept { return std::isnan(value); })) {
        f.error = "ranking inputs must not contain NaNs";
        return f;
    }
    f.expected.resize(volume(f.output_shape));
    f.bound.resize(f.expected.size());
    if (tensorcore_attention) { f.probability_rounding_bound.resize(f.expected.size()); }
    if (rows) {
        row_oracle(f, o);
    } else if (gemm || bmm) {
        auto base = bmm ? 1u : 0u;
        auto batches = bmm ? o.dimensions[0] : int64_t{1};
        auto m = o.dimensions[base], n = o.dimensions[base + 1u], k = o.dimensions[base + 2u];
        for (int64_t batch = 0; batch < batches; batch++) {
            auto a_offset = batch * m * k, b_offset = batch * k * n, c_offset = batch * m * n;
            for (int64_t i = 0; i < m; i++) {
                for (int64_t j = 0; j < n; j++) {
                    double value = 0.0, absolute = 0.0;
                    for (int64_t t = 0; t < k; t++) {
                        auto product = static_cast<double>(f.inputs[0][a_offset + i * k + t]) * f.inputs[1][b_offset + t * n + j];
                        value += product;
                        absolute += std::abs(product);
                    }
                    f.expected[c_offset + i * n + j] = value;
                    f.bound[c_offset + i * n + j] = sum_bound(k, absolute);
                }
            }
        }
    } else if (embedding) {
        auto width = o.dimensions[1], tokens = o.dimensions[2];
        for (int64_t token = 0; token < tokens; token++) {
            auto id = (*f.input_ids)[token];
            for (int64_t column = 0; column < width; column++) {
                auto output = token * width + column;
                f.expected[output] = static_cast<double>(f.inputs[0][id * width + column]);
                f.bound[output] = 0.0;
            }
        }
    } else if (first_max) {
        auto r = o.dimensions[0], n = o.dimensions[1];
        f.expected_indices.resize(f.expected.size());
        for (int64_t row = 0; row < r; row++) {
            auto winner = int64_t{0};
            for (int64_t col = 1; col < n; col++) {
                if (f.inputs[0][row * n + col] > f.inputs[0][row * n + winner]) { winner = col; }
            }
            f.expected_indices[row] = winner;
            f.expected[row] = f.inputs[0][row * n + winner];
            f.bound[row] = 0.0;
        }
    } else if (rank) {
        auto r = o.dimensions[0], n = o.dimensions[1], k = o.dimensions[2];
        f.expected_indices.resize(f.expected.size());
        vector<int64_t> order(n);
        for (int64_t row = 0; row < r; row++) {
            std::iota(order.begin(), order.end(), int64_t{0});
            std::sort(order.begin(), order.end(), [&](auto a, auto b) {
                auto x = f.inputs[0][row * n + a], y = f.inputs[0][row * n + b];
                return x == y ? a < b : x > y;
            });
            for (int64_t i = 0; i < k; i++) {
                f.expected_indices[row * k + i] = order[i];
                f.expected[row * k + i] = f.inputs[0][row * n + order[i]];
            }
        }
    } else {
        auto &d = o.dimensions;
        auto scale = 1.0f / std::sqrt(static_cast<float>(d[5]));
        for (int64_t b = 0; b < d[0]; b++) {
            for (int64_t h = 0; h < d[1]; h++) {
                auto kh = h / (d[1] / d[2]);
                for (int64_t q = 0; q < d[3]; q++) {
                    vector<double> scores(d[4] - d[3] + q + 1);
                    auto peak = -std::numeric_limits<double>::infinity();
                    for (size_t k = 0; k < scores.size(); k++) {
                        auto dot = 0.0;
                        for (int64_t c = 0; c < d[5]; c++) { dot += static_cast<double>(f.inputs[0][((b * d[1] + h) * d[3] + q) * d[5] + c]) * f.inputs[1][((b * d[2] + kh) * d[4] + k) * d[5] + c]; }
                        scores[k] = dot * scale;
                        peak = std::max(peak, scores[k]);
                    }
                    double denominator = 0.0;
                    for (auto &score : scores) {
                        score = std::exp(score - peak);
                        denominator += score;
                    }
                    for (int64_t c = 0; c < d[6]; c++) {
                        double value = 0.0;
                        for (size_t k = 0; k < scores.size(); k++) { value += scores[k] / denominator * f.inputs[2][((b * d[2] + kh) * d[4] + k) * d[6] + c]; }
                        auto index = ((b * d[1] + h) * d[3] + q) * d[6] + c;
                        f.expected[index] = value;
                        f.bound[index] = 5e-5 + 5e-5 * std::abs(value);
                        if (tensorcore_attention) {
                            double maximum_value = 0.0, absolute_sum = 0.0;
                            for (size_t k = 0; k < scores.size(); k++) {
                                auto magnitude = std::abs(static_cast<double>(f.inputs[2][((b * d[2] + kh) * d[4] + k) * d[6] + c]));
                                maximum_value = std::max(maximum_value, magnitude);
                                absolute_sum += magnitude;
                            }
                            auto unit = static_cast<double>(static_cast<float>(std::numeric_limits<T>::epsilon())) * .5;
                            auto eta = static_cast<double>(static_cast<float>(std::numeric_limits<T>::denorm_min()));
                            auto count = 2.0 * static_cast<double>(scores.size()) + 2.0;
                            auto gamma = count * 0x1p-24 / (1.0 - count * 0x1p-24);
                            auto amplification = (1.0 + gamma) / (1.0 - gamma);
                            // One RNE probability narrowing, positive online
                            // normalization, <=K rescalings and <=K sum stages.
                            // max|V| bounds cancellation independently of |Y|.
                            f.probability_rounding_bound[index] = amplification * (unit * maximum_value + .5 * eta * absolute_sum);
                        }
                    }
                }
            }
        }
    }
    if constexpr (!std::is_same_v<T, float>) {
        if (!rank && !embedding && op != "reduce_max") {
            // If float arithmetic error is <= e, one RNE storage conversion
            // adds <= u*(abs(reference)+e) + half the smallest subnormal.
            auto unit_roundoff = static_cast<double>(static_cast<float>(std::numeric_limits<T>::epsilon())) * .5;
            auto subnormal = static_cast<double>(static_cast<float>(std::numeric_limits<T>::denorm_min())) * .5;
            for (size_t i = 0; i < f.bound.size(); i++) {
                f.bound[i] += unit_roundoff * (std::abs(f.expected[i]) + f.bound[i]) + subnormal;
            }
        }
    }
    if (tensorcore_attention) {
        f.strict_bound = f.bound;
        auto output_unit = static_cast<double>(static_cast<float>(std::numeric_limits<T>::epsilon())) * .5;
        for (size_t i = 0; i < f.bound.size(); i++) {
            f.bound[i] += (1.0 + output_unit) * f.probability_rounding_bound[i];
        }
    }
    return f;
}

}// namespace luisa::test::tile_workloads
