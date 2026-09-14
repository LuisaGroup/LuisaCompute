// End-to-end execution tests for neural-network Tile DSL PoCs that stress
// Tile reductions, loop-carried pipelines, and dimension-identity broadcasting.

#include "ut/ut.hpp"
#include "tile_tirx_test_utils.h"

#include <tvm/ffi/function.h>
#include <tvm/ffi/string.h>
#include <tvm/runtime/tensor.h>

#include <luisa/core/mathematics.h>
#include <luisa/core/stl/string.h>
#include <luisa/core/stl/vector.h>
#include <luisa/tile/bridge/tirx/compiler.h>
#include <luisa/tile/bridge/tirx/lower.h>
#include <luisa/tile/dsl.h>
#include <luisa/tile/algorithms.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <initializer_list>
#include <limits>

using namespace luisa;
using namespace luisa::compute::tile;
using namespace luisa::compute::tile::bridge::tirx;
using namespace boost::ut;
using namespace boost::ut::literals;

namespace {

using luisa::test::tile_tirx::Runtime;

void test_bias_gelu_residual(Runtime &runtime) {
    constexpr int64_t rows = 3;
    constexpr int64_t columns = 8;
    constexpr float gelu_scale = 0.7978845608028654f;
    constexpr float gelu_cubic = 0.044715f;
    auto definition = tile_kernel(
        "tile_poc_bias_gelu_residual",
        [=](TensorView<const float, 2> x,
            TensorView<const float, 1> bias,
            TensorView<const float, 2> residual,
            TensorView<float, 2> y) {
            constexpr auto bm = 2;
            constexpr auto bn = 4;
            auto gm = axis("block_m", ceil_div(x.extent<0>(), bm));
            auto gn = axis("block_n", ceil_div(x.extent<1>(), bn));
            auto m = axis("m", bm);
            auto n = axis("n", bn);
            for (auto &nest : parallel(shape(gm, gn))) {
                auto origin = coord(nest.index(gm) * bm, nest.index(gn) * bn);
                auto value = x[origin, shape(m, n)] + bias[coord(nest.index(gn) * bn), shape(n)];
                auto cubic = value * value * value;
                auto gelu = 0.5f * value * (1.0f + tanh(gelu_scale * (value + gelu_cubic * cubic)));
                y(origin, shape(m, n)).store(gelu + residual[origin, shape(m, n)]);
            }
        });
    auto kernel = definition.capture(
        tensor_shape("x", rows, columns), tensor_shape("bias", columns),
        tensor_shape("residual", rows, columns), tensor_shape("y", rows, columns));
    auto executable = runtime.build(kernel);
    expect(executable.ok()) << executable.error;
    if (!executable.ok()) { return; }

    luisa::vector<float> x(rows * columns);
    luisa::vector<float> bias(columns);
    luisa::vector<float> residual(rows * columns);
    for (auto i = 0u; i < x.size(); i++) {
        x[i] = static_cast<float>(static_cast<int32_t>((i * 5u) % 23u) - 11) * 0.2f;
        residual[i] = static_cast<float>(static_cast<int32_t>((i * 3u) % 13u) - 6) * 0.05f;
    }
    for (auto i = 0u; i < bias.size(); i++) {
        bias[i] = static_cast<float>(static_cast<int32_t>(i) - 3) * 0.1f;
    }
    auto x_tensor = runtime.upload<float>({rows, columns}, x);
    auto bias_tensor = runtime.upload<float>({columns}, bias);
    auto residual_tensor = runtime.upload<float>({rows, columns}, residual);
    auto y_tensor = runtime.allocate<float>({rows, columns});
    (*executable.entry)(x_tensor, bias_tensor, residual_tensor, y_tensor);
    auto y = runtime.download<float>(y_tensor, x.size());
    for (auto r = 0; r < rows; r++) {
        for (auto c = 0; c < columns; c++) {
            auto index = static_cast<size_t>(r * columns + c);
            auto value = x[index] + bias[static_cast<size_t>(c)];
            auto expected = 0.5f * value *
                                (1.0f + std::tanh(gelu_scale *
                                                  (value + gelu_cubic * value * value * value))) +
                            residual[index];
            expect(std::abs(y[index] - expected) < 2e-5f);
        }
    }
}

void test_whole_tensor_reduction(Runtime &runtime) {
    constexpr int64_t rows = 5;
    constexpr int64_t columns = 7;
    auto definition = tile_kernel(
        "tile_poc_whole_tensor_reduction",
        [=](TensorView<const float, 2> x,
            TensorView<float, 1> sum_out,
            TensorView<float, 1> square_sum_out,
            TensorView<float, 1> maximum_out) {
            auto program = axis("program", 1);
            auto row = axis("row", x.extent<0>());
            auto column = axis("column", x.extent<1>());
            auto output = axis("output", 1);
            for (auto &nest : parallel(shape(program))) {
                auto value = x[coord(0, 0), shape(row, column)];
                auto destination = coord(nest.index());
                sum_out(destination, shape(output)).store(reduce(value, shape(row, column), add));
                square_sum_out(destination, shape(output)).store(reduce(value * value, shape(row, column), add));
                maximum_out(destination, shape(output)).store(reduce(value, shape(row, column), maximum));
            }
        });
    auto kernel = definition.capture(
        tensor_shape("x", rows, columns), tensor_shape("sum", 1u),
        tensor_shape("square_sum", 1u), tensor_shape("maximum", 1u));
    auto executable = runtime.build(kernel);
    expect(executable.ok()) << executable.error;
    if (!executable.ok()) { return; }

    luisa::vector<float> x(rows * columns);
    auto expected_sum = 0.0f;
    auto expected_square_sum = 0.0f;
    auto expected_maximum = -std::numeric_limits<float>::infinity();
    for (auto i = 0u; i < x.size(); i++) {
        x[i] = static_cast<float>(static_cast<int32_t>((i * 11u) % 29u) - 14) * 0.125f;
        expected_sum += x[i];
        expected_square_sum += x[i] * x[i];
        expected_maximum = std::max(expected_maximum, x[i]);
    }
    auto x_tensor = runtime.upload<float>({rows, columns}, x);
    auto sum_tensor = runtime.allocate<float>({1});
    auto square_sum_tensor = runtime.allocate<float>({1});
    auto maximum_tensor = runtime.allocate<float>({1});
    (*executable.entry)(x_tensor, sum_tensor, square_sum_tensor, maximum_tensor);
    auto sum = runtime.download<float>(sum_tensor, 1u);
    auto square_sum = runtime.download<float>(square_sum_tensor, 1u);
    auto maximum = runtime.download<float>(maximum_tensor, 1u);
    expect(std::abs(sum[0u] - expected_sum) < 1e-5f);
    expect(std::abs(square_sum[0u] - expected_square_sum) < 1e-5f);
    expect(std::abs(maximum[0u] - expected_maximum) < 1e-5f);
}

void test_sparse_softmax_cross_entropy(Runtime &runtime) {
    constexpr int64_t rows = 4;
    constexpr int64_t classes = 6;
    auto definition = tile_kernel(
        "tile_poc_sparse_softmax_cross_entropy",
        [=](TensorView<const float, 2> logits,
            TensorView<const int64_t, 1> labels,
            TensorView<float, 1> losses,
            TensorView<float, 2> gradient) {
            auto row = axis("row", logits.extent<0>());
            auto r = axis("local_row", 1);
            auto c = axis("class", logits.extent<1>());
            for (auto &nest : parallel(shape(row))) {
                auto origin = coord(nest.index(), 0);
                auto value = logits[origin, shape(r, c)];
                auto label = labels[coord(nest.index()), shape(r)];
                auto peak = reduce(value, c, maximum);
                auto exponentials = exp(value - peak);
                auto total = reduce(exponentials, c, add);
                auto selected = gather(value, label, c);
                losses(coord(nest.index()), shape(r)).store(luisa::compute::tile::log(total) + peak - selected);
                auto one_hot = ite(iota(c) == label, 1.0f, 0.0f);
                gradient(origin, shape(r, c)).store(exponentials / total - one_hot);
            }
        });
    auto kernel = definition.capture(
        tensor_shape("logits", rows, classes), tensor_shape("labels", rows),
        tensor_shape("losses", rows), tensor_shape("gradient", rows, classes));
    auto executable = runtime.build(kernel);
    expect(executable.ok()) << executable.error;
    if (!executable.ok()) { return; }

    luisa::vector<float> logits(rows * classes);
    luisa::vector<int64_t> labels{0, 5, 2, 3};
    for (auto i = 0u; i < logits.size(); i++) {
        logits[i] = static_cast<float>(static_cast<int32_t>((i * 7u) % 17u) - 8) * 0.35f;
    }
    auto logits_tensor = runtime.upload<float>({rows, classes}, logits);
    auto labels_tensor = runtime.upload<int64_t>({rows}, labels);
    auto losses_tensor = runtime.allocate<float>({rows});
    auto gradient_tensor = runtime.allocate<float>({rows, classes});
    (*executable.entry)(logits_tensor, labels_tensor, losses_tensor, gradient_tensor);
    auto losses = runtime.download<float>(losses_tensor, rows);
    auto gradient = runtime.download<float>(gradient_tensor, logits.size());
    for (auto r = 0; r < rows; r++) {
        auto begin = static_cast<size_t>(r * classes);
        auto peak = -std::numeric_limits<float>::infinity();
        for (auto c = 0; c < classes; c++) {
            peak = std::max(peak, logits[begin + static_cast<size_t>(c)]);
        }
        auto exponential_sum = 0.0f;
        for (auto c = 0; c < classes; c++) {
            exponential_sum += std::exp(logits[begin + static_cast<size_t>(c)] - peak);
        }
        auto expected_loss = std::log(exponential_sum) + peak -
                             logits[begin + static_cast<size_t>(labels[static_cast<size_t>(r)])];
        expect(std::abs(losses[static_cast<size_t>(r)] - expected_loss) < 2e-5f);
        for (auto c = 0; c < classes; c++) {
            auto expected = std::exp(logits[begin + static_cast<size_t>(c)] - peak) /
                                exponential_sum -
                            (c == labels[static_cast<size_t>(r)] ? 1.0f : 0.0f);
            expect(std::abs(gradient[begin + static_cast<size_t>(c)] - expected) < 2e-5f);
        }
    }
}

void test_flash_attention_online_softmax(Runtime &runtime) {
    constexpr int64_t batches = 1;
    constexpr int64_t heads = 2;
    constexpr int64_t queries = 4;
    constexpr int64_t keys = 5;
    constexpr int64_t channels = 4;
    constexpr int64_t value_channels = 3;
    constexpr float scale = 0.5f;
    auto definition = tile_kernel(
        "tile_poc_flash_attention",
        [=](TensorView<const float, 4> q,
            TensorView<const float, 4> k,
            TensorView<const float, 4> v,
            TensorView<float, 4> result) {
            constexpr auto bq = 2;
            constexpr auto bk = 3;
            auto batch = axis("batch", result.extent<0>());
            auto head = axis("head", result.extent<1>());
            auto query_blocks = axis("query_blocks", ceil_div(result.extent<2>(), bq));
            auto key_blocks = axis("key_blocks", ceil_div(k.extent<2>(), bk));
            auto b = axis("local_batch", 1);
            auto h = axis("local_head", 1);
            auto m = axis("query", bq);
            auto s = axis("key", bk);
            auto d = axis("channel", q.extent<3>());
            auto dv = axis("value_channel", result.extent<3>());
            for (auto &nest : parallel(shape(batch, head, query_blocks))) {
                auto b0 = nest.index(batch);
                auto h0 = nest.index(head);
                auto q0 = nest.index(query_blocks) * bq;
                auto query = q[coord(b0, h0, q0, 0), shape(b, h, m, d)];
                auto row_max = full<float>(shape(b, h, m), -1e30f);
                auto row_sum = zeros<float>(shape(b, h, m));
                auto acc = zeros<float>(shape(b, h, m, dv));
                for (auto &step : nest.pipeline(shape(key_blocks), {.stages = 2, .initiation_interval = 1})) {
                    auto k0 = step.index() * bk;
                    step.stage("load");
                    auto key = k[coord(b0, h0, k0, 0), shape(b, h, s, d)];
                    auto value = v[coord(b0, h0, k0, 0), shape(b, h, s, dv)];
                    step.stage("score");
                    auto score = mma(query, key, zeros<float>(shape(b, h, m, s))) * scale;
                    auto valid = (iota(s) + k0 < k.extent<2>()) && (iota(s) + k0 <= iota(m) + q0);
                    auto masked = ite(valid, score, -1e30f);
                    auto next_max = max(row_max, reduce(masked, s, maximum));
                    auto alpha = exp(row_max - next_max);
                    auto probability = ite(valid, exp(masked - next_max), 0.0f);
                    step.stage("update");
                    row_sum = row_sum * alpha + reduce(probability, s, add);
                    acc = mma(probability, value, acc * alpha);
                    row_max = next_max;
                }
                result(coord(b0, h0, q0, 0), shape(b, h, m, dv)).store(acc / row_sum);
            }
        });
    auto kernel = definition.capture(
        tensor_shape("q", batches, heads, queries, channels),
        tensor_shape("k", batches, heads, keys, channels),
        tensor_shape("v", batches, heads, keys, value_channels),
        tensor_shape("output", batches, heads, queries, value_channels));
    expect(eq(luisa::test::tile_tirx::count_operations(kernel.function().body(), OperationKind::MMA), 2u));
    auto executable = runtime.build(kernel);
    expect(executable.ok()) << executable.error;
    if (!executable.ok()) { return; }

    luisa::vector<float> q(batches * heads * queries * channels);
    luisa::vector<float> k(batches * heads * keys * channels);
    luisa::vector<float> v(batches * heads * keys * value_channels);
    for (auto i = 0u; i < q.size(); i++) {
        q[i] = static_cast<float>(static_cast<int32_t>((i * 5u) % 17u) - 8) * 0.15f;
    }
    for (auto i = 0u; i < k.size(); i++) {
        k[i] = static_cast<float>(static_cast<int32_t>((i * 7u) % 19u) - 9) * 0.125f;
    }
    for (auto i = 0u; i < v.size(); i++) {
        v[i] = static_cast<float>(static_cast<int32_t>((i * 11u) % 23u) - 11) * 0.1f;
    }
    auto q_tensor = runtime.upload<float>({batches, heads, queries, channels}, q);
    auto k_tensor = runtime.upload<float>({batches, heads, keys, channels}, k);
    auto v_tensor = runtime.upload<float>({batches, heads, keys, value_channels}, v);
    auto output_tensor = runtime.allocate<float>({batches, heads, queries, value_channels});
    (*executable.entry)(q_tensor, k_tensor, v_tensor, output_tensor);
    auto actual = runtime.download<float>(output_tensor, batches * heads * queries * value_channels);

    auto q_index = [](int64_t b, int64_t h, int64_t qi, int64_t c) noexcept {
        return static_cast<size_t>(((b * heads + h) * queries + qi) * channels + c);
    };
    auto k_index = [](int64_t b, int64_t h, int64_t ki, int64_t c) noexcept {
        return static_cast<size_t>(((b * heads + h) * keys + ki) * channels + c);
    };
    auto v_index = [](int64_t b, int64_t h, int64_t ki, int64_t c) noexcept {
        return static_cast<size_t>(((b * heads + h) * keys + ki) * value_channels + c);
    };
    auto o_index = [](int64_t b, int64_t h, int64_t qi, int64_t c) noexcept {
        return static_cast<size_t>(((b * heads + h) * queries + qi) * value_channels + c);
    };
    for (auto b = 0; b < batches; b++) {
        for (auto h = 0; h < heads; h++) {
            for (auto qi = 0; qi < queries; qi++) {
                luisa::vector<float> scores(static_cast<size_t>(qi + 1));
                auto peak = -std::numeric_limits<float>::infinity();
                for (auto ki = 0; ki <= qi; ki++) {
                    auto score = 0.0f;
                    for (auto c = 0; c < channels; c++) {
                        score += q[q_index(b, h, qi, c)] * k[k_index(b, h, ki, c)];
                    }
                    score *= scale;
                    scores[static_cast<size_t>(ki)] = score;
                    peak = std::max(peak, score);
                }
                auto denominator = 0.0f;
                for (auto score : scores) { denominator += std::exp(score - peak); }
                for (auto c = 0; c < value_channels; c++) {
                    auto expected = 0.0f;
                    for (auto ki = 0; ki <= qi; ki++) {
                        expected += std::exp(scores[static_cast<size_t>(ki)] - peak) /
                                    denominator * v[v_index(b, h, ki, c)];
                    }
                    expect(std::abs(actual[o_index(b, h, qi, c)] - expected) < 3e-5f);
                }
            }
        }
    }
}

}// namespace

int main(int argc, char *argv[]) {
    auto runtime = Runtime{argc > 1 ? argv[1] : "cpu"};
    boost::ut::detail::cfg::parse_arg_with_fallback(
        argc > 1 ? argc - 1 : argc,
        const_cast<const char **>(argc > 1 ? argv + 1 : argv));
    "tile_tirx_poc_bias_gelu_residual"_test = [&] { test_bias_gelu_residual(runtime); };
    "tile_tirx_poc_whole_tensor_reduction"_test = [&] { test_whole_tensor_reduction(runtime); };
    "tile_tirx_poc_sparse_softmax_cross_entropy"_test = [&] { test_sparse_softmax_cross_entropy(runtime); };
    "tile_tirx_poc_flash_attention"_test = [&] { test_flash_attention_online_softmax(runtime); };
}
