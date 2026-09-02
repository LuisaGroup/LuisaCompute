// End-to-end execution tests for neural-network Tile DSL PoCs that stress
// nested reductions, loop-carried pipelines, and ordinary scalar lifting.

#include "ut/ut.hpp"

#include <tvm/ffi/function.h>
#include <tvm/ffi/string.h>
#include <tvm/runtime/tensor.h>

#include <luisa/core/stl/string.h>
#include <luisa/core/stl/vector.h>
#include <luisa/tile/bridge/tirx/compiler.h>
#include <luisa/tile/bridge/tirx/lower.h>
#include <luisa/tile/dsl.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <initializer_list>
#include <limits>
#include <type_traits>
#include <utility>

using namespace luisa;
using namespace luisa::compute::tile;
using namespace luisa::compute::tile::bridge::tirx;
using namespace boost::ut;
using namespace boost::ut::literals;

namespace {

struct Executable {
    tvm::ffi::Optional<tvm::ffi::Module> module;
    tvm::ffi::Optional<tvm::ffi::Function> entry;
    luisa::string error;

    [[nodiscard]] bool ok() const noexcept {
        return error.empty() && module.has_value() && entry.has_value();
    }
};

[[nodiscard]] Executable build(Kernel &kernel) {
    Executable result;
    if (!kernel.valid()) {
        result.error = "Tile DSL capture or verification failed";
        for (auto &&diagnostic : kernel.diagnostics()) {
            result.error.append(": ");
            result.error.append(diagnostic);
        }
        return result;
    }
    auto native = lower(kernel.function());
    if (!native) {
        result.error = std::move(native.error);
        return result;
    }
    auto compilation = compile(std::move(native.value), kernel.function().name());
    if (!compilation) {
        result.error = luisa::string{compilation.error()};
        return result;
    }
    result.module = compilation.module();
    auto entry_name = kernel.function().name();
    result.entry = result.module.value()->GetFunction(
        tvm::ffi::String{entry_name.data(), entry_name.size()}, true);
    if (!result.entry) { result.error = "compiled module has no requested entry function"; }
    return result;
}

template<typename T>
[[nodiscard]] constexpr DLDataType dl_data_type() noexcept {
    if constexpr (std::is_same_v<T, float>) {
        return DLDataType{kDLFloat, 32, 1};
    } else if constexpr (std::is_same_v<T, int64_t>) {
        return DLDataType{kDLInt, 64, 1};
    } else {
        static_assert(std::is_same_v<T, void>, "unsupported test tensor element type");
    }
}

template<typename T>
[[nodiscard]] tvm::runtime::Tensor upload(
    std::initializer_list<int64_t> shape,
    const luisa::vector<T> &values) {
    auto tensor = tvm::runtime::Tensor::Empty(
        tvm::ffi::Shape{shape}, dl_data_type<T>(), tvm::Device{kDLCPU, 0});
    tensor.CopyFromBytes(values.data(), values.size() * sizeof(T));
    return tensor;
}

template<typename T>
[[nodiscard]] tvm::runtime::Tensor allocate(std::initializer_list<int64_t> shape) {
    return tvm::runtime::Tensor::Empty(
        tvm::ffi::Shape{shape}, dl_data_type<T>(), tvm::Device{kDLCPU, 0});
}

template<typename T>
[[nodiscard]] luisa::vector<T> download(const tvm::runtime::Tensor &tensor, size_t count) {
    luisa::vector<T> values(count);
    tensor.CopyToBytes(values.data(), values.size() * sizeof(T));
    return values;
}

void test_bias_gelu_residual() {
    constexpr int64_t rows = 3;
    constexpr int64_t columns = 8;
    constexpr float gelu_scale = 0.7978845608028654f;
    constexpr float gelu_cubic = 0.044715f;
    auto kernel = define("tile_poc_bias_gelu_residual", [] {
        auto row = axis("row", rows);
        auto column = axis("column", columns);
        auto x = input<float>("x", shape(row, column));
        auto bias = input<float>("bias", shape(column));
        auto residual = input<float>("residual", shape(row, column));
        auto y = output<float>("y", shape(row, column));
        for (auto &element : parallel(shape(row, column))) {
            auto value = x[element].load() + bias(element[column]).load();
            auto cubic = value * value * value;
            auto gelu = 0.5f * value *
                        (1.0f + tanh(gelu_scale * (value + gelu_cubic * cubic)));
            y[element] = gelu + residual[element].load();
        }
    });
    auto executable = build(kernel);
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
    auto x_tensor = upload<float>({rows, columns}, x);
    auto bias_tensor = upload<float>({columns}, bias);
    auto residual_tensor = upload<float>({rows, columns}, residual);
    auto y_tensor = allocate<float>({rows, columns});
    (*executable.entry)(x_tensor, bias_tensor, residual_tensor, y_tensor);
    auto y = download<float>(y_tensor, x.size());
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

void test_whole_tensor_reduction() {
    constexpr int64_t rows = 5;
    constexpr int64_t columns = 7;
    auto kernel = define("tile_poc_whole_tensor_reduction", [] {
        auto row = axis("row", rows);
        auto column = axis("column", columns);
        auto result = axis("result", 1);
        auto x = input<float>("x", shape(row, column));
        auto sum_out = output<float>("sum", shape(result));
        auto square_sum_out = output<float>("square_sum", shape(result));
        auto maximum_out = output<float>("maximum", shape(result));
        auto sum = Scalar<float>{0.0f};
        auto square_sum = Scalar<float>{0.0f};
        auto maximum = Scalar<float>{-1e30f};
        for (auto &element : reduce(shape(row, column))) {
            auto value = x[element].load();
            sum += value;
            square_sum += value * value;
            maximum = max(maximum, value);
        }
        auto zero = Scalar<int64_t>{0};
        sum_out(zero) = sum;
        square_sum_out(zero) = square_sum;
        maximum_out(zero) = maximum;
    });
    auto executable = build(kernel);
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
    auto x_tensor = upload<float>({rows, columns}, x);
    auto sum_tensor = allocate<float>({1});
    auto square_sum_tensor = allocate<float>({1});
    auto maximum_tensor = allocate<float>({1});
    (*executable.entry)(x_tensor, sum_tensor, square_sum_tensor, maximum_tensor);
    auto sum = download<float>(sum_tensor, 1u);
    auto square_sum = download<float>(square_sum_tensor, 1u);
    auto maximum = download<float>(maximum_tensor, 1u);
    expect(std::abs(sum[0u] - expected_sum) < 1e-5f);
    expect(std::abs(square_sum[0u] - expected_square_sum) < 1e-5f);
    expect(std::abs(maximum[0u] - expected_maximum) < 1e-5f);
}

void test_sparse_softmax_cross_entropy() {
    constexpr int64_t rows = 4;
    constexpr int64_t classes = 6;
    auto kernel = define("tile_poc_sparse_softmax_cross_entropy", [] {
        auto row = axis("row", rows);
        auto klass = axis("class", classes);
        auto logits = input<float>("logits", shape(row, klass));
        auto labels = input<int64_t>("labels", shape(row));
        auto losses = output<float>("losses", shape(row));
        auto gradient = output<float>("gradient", shape(row, klass));
        for (auto &sample : parallel(shape(row))) {
            auto label = labels(sample[row]).load();
            auto peak = Scalar<float>{-1e30f};
            for (auto &item : sample.reduce(shape(klass))) {
                peak = max(peak, logits(sample[row], item[klass]).load());
            }
            auto exponential_sum = Scalar<float>{0.0f};
            for (auto &item : sample.reduce(shape(klass))) {
                exponential_sum += exp(logits(sample[row], item[klass]).load() - peak);
            }
            auto selected = logits(sample[row], label).load();
            losses(sample[row]) = luisa::compute::tile::log(exponential_sum) + peak - selected;
            for (auto &item : sample.parallel(shape(klass))) {
                auto probability = exp(logits(sample[row], item[klass]).load() - peak) /
                                   exponential_sum;
                gradient(sample[row], item[klass]) =
                    probability - select(item[klass] == label, 1.0f, 0.0f);
            }
        }
    });
    auto executable = build(kernel);
    expect(executable.ok()) << executable.error;
    if (!executable.ok()) { return; }

    luisa::vector<float> logits(rows * classes);
    luisa::vector<int64_t> labels{0, 5, 2, 3};
    for (auto i = 0u; i < logits.size(); i++) {
        logits[i] = static_cast<float>(static_cast<int32_t>((i * 7u) % 17u) - 8) * 0.35f;
    }
    auto logits_tensor = upload<float>({rows, classes}, logits);
    auto labels_tensor = upload<int64_t>({rows}, labels);
    auto losses_tensor = allocate<float>({rows});
    auto gradient_tensor = allocate<float>({rows, classes});
    (*executable.entry)(logits_tensor, labels_tensor, losses_tensor, gradient_tensor);
    auto losses = download<float>(losses_tensor, rows);
    auto gradient = download<float>(gradient_tensor, logits.size());
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

void test_flash_attention_online_softmax() {
    constexpr int64_t batches = 1;
    constexpr int64_t heads = 2;
    constexpr int64_t queries = 4;
    constexpr int64_t keys = 5;
    constexpr int64_t channels = 4;
    constexpr int64_t value_channels = 3;
    constexpr float scale = 0.5f;
    auto kernel = define("tile_poc_flash_attention", [] {
        auto batch = axis("batch", batches);
        auto head = axis("head", heads);
        auto query = axis("query", queries);
        auto key = axis("key", keys);
        auto channel = axis("channel", channels);
        auto value_channel = axis("value_channel", value_channels);
        auto q = input<float>("q", shape(batch, head, query, channel));
        auto k = input<float>("k", shape(batch, head, key, channel));
        auto v = input<float>("v", shape(batch, head, key, value_channel));
        auto result = output<float>("output", shape(batch, head, query, value_channel));
        for (auto &element : parallel(shape(batch, head, query, value_channel))) {
            auto row_max = Scalar<float>{-1e30f};
            auto row_sum = Scalar<float>{0.0f};
            auto accumulator = Scalar<float>{0.0f};
            for (auto &key_step : element.pipeline(
                     shape(key), PipelinePolicy{.stages = 2u, .initiation_interval = 1u})) {
                key_step.stage("score");
                auto score = Scalar<float>{0.0f};
                for (auto &dot : key_step.reduce(shape(channel))) {
                    score += q(element[batch], element[head], element[query], dot[channel]).load() *
                             k(element[batch], element[head], key_step[key], dot[channel]).load();
                }
                score *= scale;
                score = select(key_step[key] <= element[query], score, -1e30f);

                key_step.stage("update");
                auto next_max = max(row_max, score);
                auto old_scale = exp(row_max - next_max);
                auto probability = exp(score - next_max);
                row_sum = row_sum * old_scale + probability;
                accumulator = accumulator * old_scale +
                              probability * v(element[batch], element[head], key_step[key], element[value_channel]).load();
                row_max = next_max;
            }
            result[element] = accumulator / row_sum;
        }
    });
    auto executable = build(kernel);
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
    auto q_tensor = upload<float>({batches, heads, queries, channels}, q);
    auto k_tensor = upload<float>({batches, heads, keys, channels}, k);
    auto v_tensor = upload<float>({batches, heads, keys, value_channels}, v);
    auto output_tensor = allocate<float>({batches, heads, queries, value_channels});
    (*executable.entry)(q_tensor, k_tensor, v_tensor, output_tensor);
    auto actual = download<float>(output_tensor, batches * heads * queries * value_channels);

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
    boost::ut::detail::cfg::parse_arg_with_fallback(argc, const_cast<const char **>(argv));
    "tile_tirx_poc_bias_gelu_residual"_test = test_bias_gelu_residual;
    "tile_tirx_poc_whole_tensor_reduction"_test = test_whole_tensor_reduction;
    "tile_tirx_poc_sparse_softmax_cross_entropy"_test = test_sparse_softmax_cross_entropy;
    "tile_tirx_poc_flash_attention"_test = test_flash_attention_online_softmax;
}
