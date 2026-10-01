#pragma once

// Actual Tile DSL fixtures shared by the CUDA/SIMD workload benchmark.
// Exported inputs, rather than an operation-name-dependent device substitute,
// define the independent Python comparison.
#include "tile_llm_test_utils.h"
#include "tile_rank_test_utils.h"
#include <luisa/tile/algorithms.h>
#include <luisa/core/stl/optional.h>
#include <array>
#include <bit>
#include <cmath>
#include <numeric>

namespace luisa::test::tile_workloads {

struct Options {
    string backend, lowering, operation, precision, pattern;
    vector<int64_t> dimensions;
    std::array<int64_t, 3u> tile{};
    uint64_t seed{0u};
    uint32_t samples{7u}, sample_ms{100u}, warmup_ms{500u}, graph_batch{0u};
};

struct Fixture {
    optional<compute::tile::Kernel> kernel;
    std::array<vector<int64_t>, 3u> input_shapes;
    std::array<vector<float>, 3u> inputs;
    vector<int64_t> output_shape;
    vector<double> expected, bound;
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
           op == "reduce_max" || op == "scan";
}

inline void build_rows(Fixture &f, const Options &o) {
    using namespace compute::tile;
    auto rows = o.dimensions[0], width = o.dimensions[1], tile_width = o.tile[1];
    auto op = o.operation;
    auto norm = op == "rmsnorm" || op == "layernorm";
    auto reduction = op == "reduce_sum" || op == "reduce_max";
    f.input_shapes = {vector<int64_t>{rows, width}, {norm ? 1 : rows, op == "rope" ? width / 2 : width}, {norm ? 1 : rows, op == "rope" ? width / 2 : width}};
    f.output_shape = {rows, reduction ? 1 : width};
    f.algorithm = op == "scan" ? "quadratic_reference_inclusive_scan" : "whole_row_tile";
    if (op == "rope") {
        // A physical pair axis makes a padded half-width Tile safe: stores
        // cannot overlap the other half when the logical half has a tail.
        auto definition = tile_kernel("workload_rope", [=](TensorView<const float, 3> X,
                                                           TensorView<const float, 3> C,
                                                           TensorView<const float, 3> S,
                                                           TensorView<float, 3> Y) {
            auto m = axis("m", 1), pair = axis("pair", 1), n = axis("n", tile_width);
            for (auto &nest : parallel(shape(rows))) {
                auto r = nest.index();
                auto x = X.tile(coord(r, 0, 0), shape(m, pair, n)).load();
                auto y = X.tile(coord(r, 1, 0), shape(m, pair, n)).load();
                auto c = C.tile(coord(r, 0, 0), shape(m, pair, n)).load();
                auto s = S.tile(coord(r, 0, 0), shape(m, pair, n)).load();
                Y(coord(r, 0, 0), shape(m, pair, n)).store(x * c - y * s);
                Y(coord(r, 1, 0), shape(m, pair, n)).store(x * s + y * c);
            }
        });
        f.kernel = definition.capture(tensor_shape(rows, 2, width / 2), tensor_shape(rows, 1, width / 2),
                                      tensor_shape(rows, 1, width / 2), tensor_shape(rows, 2, width / 2));
        return;
    }
    auto definition = tile_kernel("workload_rows", [=](TensorView<const float, 2> X,
                                                       TensorView<const float, 2> U,
                                                       TensorView<const float, 2> V,
                                                       TensorView<float, 2> Y) {
        auto m = axis("m", 1), n = axis("n", tile_width);
        for (auto &nest : parallel(shape(rows))) {
            auto row = nest.index();
            auto x = X.tile(coord(row, 0), shape(m, n)).load();
            auto valid = iota(n) < width;
            if (op == "reduce_sum" || op == "reduce_max") {
                auto result = op == "reduce_sum" ? reduce(x, n, add) : reduce(ite(valid, x, -std::numeric_limits<float>::infinity()), n, maximum);
                Y(coord(row, 0), shape(m, axis("out", 1))).store(result);
            } else if (op == "scan") {
                auto k = axis("prefix", tile_width);
                auto result = map<float>(shape(m, n), [&](const Nest &item) {
                    auto sum = Scalar<float>{0.0f};
                    for (auto &part : item.reduce(shape(k), reduction::fold_left)) {
                        auto index = part.index(k);
                        sum += ite(index <= item.index(n), x.at(coord(0, index)), 0.0f);
                    }
                    return sum;
                });
                Y(coord(row, 0), shape(m, n)).store(result);
            } else {
                auto result = x;
                if (op == "rmsnorm" || op == "layernorm") {
                    auto centered = op == "layernorm" ? x - reduce(x, n, add) / static_cast<float>(width) : x;
                    auto square = ite(valid, centered * centered, 0.0f);
                    auto variance = reduce(square, n, add) / static_cast<float>(width);
                    result = centered / sqrt(variance + 1e-5f) * U.tile(coord(0, 0), shape(m, n)).load();
                    if (op == "layernorm") { result += V.tile(coord(0, 0), shape(m, n)).load(); }
                } else if (op == "swiglu") {
                    result = x / (1.0f + exp(-x)) * U.tile(coord(row, 0), shape(m, n)).load();
                } else if (op == "gelu_residual") {
                    result = 0.5f * x * (1.0f + tanh(0.7978845608f * (x + 0.044715f * x * x * x))) + U.tile(coord(row, 0), shape(m, n)).load();
                } else {
                    auto mask = op == "masked_softmax" ? valid && (iota(n) <= row % width) : valid;
                    auto score = ite(mask, x, -1e30f);
                    auto exponential = ite(mask, exp(score - reduce(score, n, maximum)), 0.0f);
                    result = exponential / reduce(exponential, n, add);
                }
                Y(coord(row, 0), shape(m, n)).store(result);
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
            } else if (o.operation == "scan") {
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

[[nodiscard]] inline Fixture make_fixture(const Options &o) {
    using namespace compute::tile;
    Fixture f;
    auto op = string_view{o.operation};
    auto rows = is_row(op);
    auto gemm = op == "gemm" || op == "gemv";
    auto rank = op == "sort" || op == "topk";
    auto attention = op == "attention";
    if (!rows && !gemm && !rank && !attention) {
        f.error = "operation has no benchmark fixture";
        return f;
    }
    auto count = rows ? 2u : attention ? 7u :
                                         3u;
    if (o.dimensions.size() != count || !product_bounded(o.dimensions, attention ? (1ull << 31u) : (1ull << 28u))) {
        f.error = "invalid dimensions or fixture work bound exceeded";
        return f;
    }
    if (rows) {
        auto width = o.dimensions[1];
        auto logical_width = op == "rope" ? width / 2 : width;
        if (o.tile[0] != 1 || o.tile[2] != 1 || o.tile[1] < logical_width || o.tile[1] > 16384 ||
            (op == "rope" && width % 2 != 0) || (op == "scan" && o.tile[1] > 1024)) {
            f.error = "row schedule requires tile=(1,padded_width,1); scan width <=1024 and RoPE width even";
            return f;
        }
        build_rows(f, o);
    } else if (gemm) {
        auto m = o.dimensions[0], n = o.dimensions[1], k = o.dimensions[2];
        if ((op == "gemv" && n != 1) || o.tile[0] > 128 || o.tile[1] > 128 || o.tile[2] > 256) {
            f.error = "GEMV requires N=1; GEMM tile exceeds benchmark limits";
            return f;
        }
        auto bm = o.tile[0], bn = o.tile[1], bk = o.tile[2];
        auto definition = tile_kernel("workload_gemm", [=](TensorView<const float, 2> A, TensorView<const float, 2> B,
                                                           TensorView<const float, 1> Unused, TensorView<float, 2> C) {
            static_cast<void>(Unused);
            auto im = axis("m", bm), jn = axis("n", bn), kk = axis("k", bk);
            auto gm = axis("gm", ceil_div(m, bm)), gn = axis("gn", ceil_div(n, bn));
            for (auto &nest : parallel(shape(gm, gn))) {
                auto mi = nest.index(gm) * bm, nj = nest.index(gn) * bn;
                auto acc = zeros<float>(shape(im, jn));
                for (auto &step : nest.pipeline(shape(ceil_div(k, bk)), {.window = 2u, .interval = 1u})) {
                    auto a = A.tile(coord(mi, step.index() * bk), shape(im, kk)).load();
                    auto b = B.tile(coord(step.index() * bk, nj), shape(kk, jn)).load();
                    acc = mma(a, b, acc);
                }
                C(coord(mi, nj), shape(im, jn)).store(acc);
            }
        });
        f.kernel = definition.capture(tensor_shape(m, k), tensor_shape(k, n), tensor_shape(1), tensor_shape(m, n));
        f.input_shapes = {vector<int64_t>{m, k}, {k, n}, {1}};
        f.output_shape = {m, n};
        f.algorithm = "tile_mma_fp32";
    } else if (rank) {
        auto r = o.dimensions[0], n = o.dimensions[1], k = o.dimensions[2];
        if (k > n || n > 1024 || r * n * n > (1ll << 26) || (op == "sort" && k != n) ||
            o.tile[0] != 1 || o.tile[1] != n || o.tile[2] != 1) {
            f.error = "quadratic ranking requires N<=1024, K<=N (sort K=N), tile=(1,N,1), work<=2^26";
            return f;
        }
        auto ranking = tile_rank::rows(op == "sort" ? tile_rank::Operation::SORT : tile_rank::Operation::TOPK, r, n, k, true);
        f.kernel = std::move(ranking.kernel);
        f.input_shapes = {vector<int64_t>{r, n}, {1}, {1}};
        f.output_shape = {r, k};
        f.ranking = true;
        f.algorithm = "quadratic_reference_stable_descending_rank";
    } else {
        auto &d = o.dimensions;
        if (d[1] % d[2] != 0 || d[4] < d[3] || o.tile[0] > 128 || o.tile[1] > 256 || o.tile[2] != 1 ||
            !product_bounded(std::array{d[0], d[1], d[3], d[4], d[5] + d[6]}, 1ull << 28u)) {
            f.error = "attention requires Hq divisible by Hkv, K>=Q, tile=(BQ,BK,1), bounded oracle work";
            return f;
        }
        auto fixture = tile_llm::attention(d[0], d[1], d[2], d[3], d[4], d[5], d[6], o.tile[0], o.tile[1]);
        f.kernel = std::move(fixture.kernel);
        for (size_t i = 0u; i < 3u; i++) { f.input_shapes[i] = std::move(fixture.shapes[i]); }
        f.output_shape = std::move(fixture.shapes[3]);
        f.algorithm = "causal_online_softmax_gqa";
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
            if (rows && input > 0u) {
                if (op == "rmsnorm" || op == "layernorm") { x = input == 1u ? 1.0f + .2f * x : .1f * x; }
                if (op == "rope") { x = input == 1u ? std::cos(x) : std::sin(x); }
            }
            data[i] = x;
        }
    }
    f.expected.resize(volume(f.output_shape));
    f.bound.resize(f.expected.size());
    if (rows) {
        row_oracle(f, o);
    } else if (gemm) {
        auto m = o.dimensions[0], n = o.dimensions[1], k = o.dimensions[2];
        if (o.pattern == "cancellation") {
            for (int64_t t = 0; t + 1 < k; t += 2) {
                for (int64_t i = 0; i < m; i++) { f.inputs[0][i * k + t + 1] = f.inputs[0][i * k + t]; }
                for (int64_t j = 0; j < n; j++) { f.inputs[1][(t + 1) * n + j] = -f.inputs[1][t * n + j] * (1.0f - 0x1p-20f); }
            }
        }
        for (int64_t i = 0; i < m; i++) {
            for (int64_t j = 0; j < n; j++) {
                double value = 0.0, absolute = 0.0;
                for (int64_t t = 0; t < k; t++) {
                    auto product = static_cast<double>(f.inputs[0][i * k + t]) * f.inputs[1][t * n + j];
                    value += product;
                    absolute += std::abs(product);
                }
                f.expected[i * n + j] = value;
                f.bound[i * n + j] = sum_bound(k, absolute);
            }
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
                    }
                }
            }
        }
    }
    return f;
}

}// namespace luisa::test::tile_workloads
