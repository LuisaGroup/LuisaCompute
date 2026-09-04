#pragma once

#include <array>
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <luisa/core/mathematics.h>
#include <luisa/tile/dsl.h>

namespace luisa::test::tile_llm {

struct Case {
    compute::tile::Kernel kernel;
    std::array<vector<int64_t>, 4u> shapes;
    std::array<vector<float>, 3u> inputs;
    vector<double> expected;
};

enum class RowOp { RMS_NORM,
                   LAYER_NORM,
                   SWIGLU,
                   ROPE,
                   MASKED_SOFTMAX,
                   GELU_RESIDUAL };

[[nodiscard]] inline Case rows(RowOp op, int64_t count, int64_t width) {
    using namespace compute::tile;
    if (count <= 0 || width <= 0 || (op == RowOp::ROPE && width % 2 != 0)) { throw std::invalid_argument{"invalid LLM row shape"}; }
    auto definition = tile_kernel("llm_rows", [=](TensorView<const float, 2> X, TensorView<const float, 2> U,
                                                  TensorView<const float, 2> V, TensorView<float, 2> Y) {
        auto m = axis("m", 1), n = axis("n", op == RowOp::ROPE ? width / 2 : width);
        for (auto &nest : parallel(shape(count))) {
            auto row = nest.index();
            auto x = X.tile(coord(row, 0), shape(m, n)).load();
            if (op == RowOp::ROPE) {
                auto other = X.tile(coord(row, width / 2), shape(m, n)).load();
                auto c = U.tile(coord(row, 0), shape(m, n)).load();
                auto s = V.tile(coord(row, 0), shape(m, n)).load();
                Y(coord(row, 0), shape(m, n)).store(x * c - other * s);
                Y(coord(row, width / 2), shape(m, n)).store(x * s + other * c);
            } else {
                auto y = x;
                if (op == RowOp::RMS_NORM || op == RowOp::LAYER_NORM) {
                    auto centered = op == RowOp::LAYER_NORM ? x - reduce(x, n, add) / static_cast<float>(width) : x;
                    auto variance = reduce(centered * centered, n, add) / static_cast<float>(width);
                    auto gamma = U.tile(coord(0, 0), shape(m, n)).load();
                    y = centered / sqrt(variance + 1e-5f) * gamma;
                    if (op == RowOp::LAYER_NORM) { y += V.tile(coord(0, 0), shape(m, n)).load(); }
                } else if (op == RowOp::SWIGLU) {
                    y = x / (1.0f + exp(-x)) * U.tile(coord(row, 0), shape(m, n)).load();
                } else if (op == RowOp::GELU_RESIDUAL) {
                    y = 0.5f * x * (1.0f + tanh(0.7978845608f * (x + 0.044715f * x * x * x))) + U.tile(coord(row, 0), shape(m, n)).load();
                } else {
                    auto valid = iota(n) <= row % width;
                    auto score = ite(valid, x, -1e30f);
                    auto e = ite(valid, exp(score - reduce(score, n, maximum)), 0.0f);
                    y = e / reduce(e, n, add);
                }
                Y(coord(row, 0), shape(m, n)).store(y);
            }
        }
    });
    auto aux_rows = op == RowOp::RMS_NORM || op == RowOp::LAYER_NORM ? 1 : count;
    auto aux_width = op == RowOp::ROPE ? width / 2 : width;
    Case result{definition.capture(tensor_shape(count, width), tensor_shape(aux_rows, aux_width),
                                   tensor_shape(aux_rows, aux_width), tensor_shape(count, width))};
    result.shapes = {vector<int64_t>{count, width}, {aux_rows, aux_width}, {aux_rows, aux_width}, {count, width}};
    result.inputs[0].resize(count * width);
    result.inputs[1].resize(aux_rows * aux_width);
    result.inputs[2].resize(aux_rows * aux_width);
    result.expected.resize(count * width);
    for (size_t i = 0u; i < result.inputs[0].size(); i++) { result.inputs[0][i] = std::sin(static_cast<float>(i) * .173f) * 1.7f; }
    for (size_t i = 0u; i < result.inputs[1].size(); i++) {
        result.inputs[1][i] = op == RowOp::ROPE ? std::cos(static_cast<float>(i) * .013f) : 1.0f + .2f * std::cos(static_cast<float>(i) * .113f);
        result.inputs[2][i] = std::sin(static_cast<float>(i) * .013f);
    }
    auto &x = result.inputs[0];
    for (int64_t row = 0; row < count; row++) {
        double mean = 0.0, variance = 0.0, denominator = 0.0, peak = -1e30;
        for (int64_t col = 0; col < width; col++) {
            mean += x[row * width + col] / static_cast<double>(width);
            if (col <= row % width) { peak = std::max(peak, static_cast<double>(x[row * width + col])); }
        }
        for (int64_t col = 0; col < width; col++) {
            auto centered = x[row * width + col] - (op == RowOp::LAYER_NORM ? mean : 0.0);
            variance += centered * centered / width;
            if (col <= row % width) { denominator += std::exp(x[row * width + col] - peak); }
        }
        for (int64_t col = 0; col < width; col++) {
            auto i = row * width + col;
            auto value = static_cast<double>(x[i]);
            auto &u = result.inputs[1];
            auto &v = result.inputs[2];
            if (op == RowOp::RMS_NORM || op == RowOp::LAYER_NORM) {
                value = (value - (op == RowOp::LAYER_NORM ? mean : 0.0)) / std::sqrt(variance + static_cast<double>(1e-5f)) * u[col];
                if (op == RowOp::LAYER_NORM) { value += v[col]; }
            } else if (op == RowOp::SWIGLU) {
                value = value / (1.0 + std::exp(-value)) * u[i];
            } else if (op == RowOp::GELU_RESIDUAL) {
                value = .5 * value * (1.0 + std::tanh(static_cast<double>(0.7978845608f) * (value + static_cast<double>(0.044715f) * value * value * value))) + u[i];
            } else if (op == RowOp::ROPE) {
                auto j = row * (width / 2) + col % (width / 2);
                auto left = static_cast<double>(x[row * width + col % (width / 2)]);
                auto right = static_cast<double>(x[row * width + col % (width / 2) + width / 2]);
                value = col < width / 2 ? left * u[j] - right * v[j] : left * v[j] + right * u[j];
            } else {
                value = col <= row % width ? std::exp(value - peak) / denominator : 0.0;
            }
            result.expected[i] = value;
        }
    }
    return result;
}

// Causal prefill and decode share the same online softmax program. The query
// positions are the final Q positions in the KV sequence; Hq/Hkv implements GQA.
[[nodiscard]] inline Case attention(int64_t batches, int64_t heads, int64_t kv_heads,
                                    int64_t queries, int64_t keys, int64_t channels, int64_t value_channels) {
    using namespace compute::tile;
    if (batches <= 0 || kv_heads <= 0 || heads % kv_heads || queries <= 0 || keys < queries || channels <= 0 || value_channels <= 0) {
        throw std::invalid_argument{"invalid attention shape"};
    }
    auto scale = 1.0f / std::sqrt(static_cast<float>(channels));
    auto definition = tile_kernel("llm_attention", [=](TensorView<const float, 4> Q, TensorView<const float, 4> K,
                                                       TensorView<const float, 4> V, TensorView<float, 4> O) {
        constexpr int64_t bq = 2, bk = 3;
        auto batch = axis("batch", batches), head = axis("head", heads), query_block = axis("query_block", ceil_div(queries, bq));
        auto b = axis("b", 1), h = axis("h", 1), m = axis("m", bq), n = axis("n", bk);
        auto d = axis("d", channels), dv = axis("dv", value_channels);
        for (auto &nest : parallel(shape(batch, head, query_block))) {
            auto b0 = nest.index(batch), h0 = nest.index(head), q0 = nest.index(query_block) * bq;
            auto kh = h0 / (heads / kv_heads);
            auto query = Q.tile(coord(b0, h0, q0, 0), shape(b, h, m, d)).load();
            auto row_max = full<float>(shape(b, h, m), -1e30f);
            auto row_sum = zeros<float>(shape(b, h, m));
            auto acc = zeros<float>(shape(b, h, m, dv));
            for (auto &step : nest.pipeline(shape(ceil_div(keys, bk)), {.stages = 2u, .initiation_interval = 1u})) {
                auto k0 = step.index() * bk;
                step.stage("load");
                auto key = K.tile(coord(b0, kh, k0, 0), shape(b, h, n, d)).load();
                auto value = V.tile(coord(b0, kh, k0, 0), shape(b, h, n, dv)).load();
                step.stage("score");
                auto score = mma(query, key, zeros<float>(shape(b, h, m, n))) * scale;
                auto valid = (iota(n) + k0 < keys) && (iota(n) + k0 <= iota(m) + q0 + keys - queries);
                auto masked = ite(valid, score, -1e30f);
                auto next_max = max(row_max, reduce(masked, n, maximum));
                auto alpha = exp(row_max - next_max);
                auto probability = ite(valid, exp(masked - next_max), 0.0f);
                step.stage("update");
                row_sum = row_sum * alpha + reduce(probability, n, add);
                acc = mma(probability, value, acc * alpha);
                row_max = next_max;
            }
            O(coord(b0, h0, q0, 0), shape(b, h, m, dv)).store(acc / row_sum);
        }
    });
    Case result{definition.capture(tensor_shape(batches, heads, queries, channels), tensor_shape(batches, kv_heads, keys, channels),
                                   tensor_shape(batches, kv_heads, keys, value_channels), tensor_shape(batches, heads, queries, value_channels))};
    result.shapes = {vector<int64_t>{batches, heads, queries, channels}, {batches, kv_heads, keys, channels}, {batches, kv_heads, keys, value_channels}, {batches, heads, queries, value_channels}};
    for (size_t input = 0u; input < 3u; input++) {
        auto &shape = result.shapes[input];
        auto &data = result.inputs[input];
        data.resize(shape[0] * shape[1] * shape[2] * shape[3]);
        for (size_t i = 0u; i < data.size(); i++) { data[i] = std::sin(static_cast<float>(i) * (.117f + .031f * input) + .3f * input); }
    }
    result.expected.resize(batches * heads * queries * value_channels);
    for (int64_t b = 0; b < batches; b++) {
        for (int64_t h = 0; h < heads; h++) {
            auto kh = h / (heads / kv_heads);
            for (int64_t q = 0; q < queries; q++) {
                vector<double> scores(keys - queries + q + 1);
                auto peak = -1e30;
                for (size_t k = 0u; k < scores.size(); k++) {
                    auto score = 0.0;
                    for (int64_t d = 0; d < channels; d++) {
                        score += static_cast<double>(result.inputs[0][((b * heads + h) * queries + q) * channels + d]) * result.inputs[1][((b * kv_heads + kh) * keys + k) * channels + d];
                    }
                    scores[k] = score * scale;
                    peak = std::max(peak, scores[k]);
                }
                auto denominator = 0.0;
                for (auto score : scores) { denominator += std::exp(score - peak); }
                for (int64_t d = 0; d < value_channels; d++) {
                    auto value = 0.0;
                    for (size_t k = 0u; k < scores.size(); k++) {
                        value += std::exp(scores[k] - peak) / denominator * result.inputs[2][((b * kv_heads + kh) * keys + k) * value_channels + d];
                    }
                    result.expected[((b * heads + h) * queries + q) * value_channels + d] = value;
                }
            }
        }
    }
    return result;
}

}// namespace luisa::test::tile_llm
