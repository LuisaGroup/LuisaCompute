// End-to-end execution tests for the Tile DSL PoC kernel gallery.
// This test covers portable scalar reference realizations of:
// - row statistics and common losses
// - softmax, LayerNorm, and RMSNorm
// - pipelined GEMM
// - padded, strided, multi-channel Conv2D

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
    } else if constexpr (std::is_same_v<T, int32_t>) {
        return DLDataType{kDLInt, 32, 1};
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

void test_row_statistics_and_losses() {
    constexpr int64_t rows = 3;
    constexpr int64_t columns = 5;
    constexpr float huber_delta = 0.75f;
    constexpr float epsilon = 1e-4f;
    auto kernel = define("tile_poc_statistics_losses", [] {
        auto row = axis("row", rows);
        auto column = axis("column", columns);
        auto x = input<float>("x", shape(row, column));
        auto target = input<float>("target", shape(row, column));
        auto probability = input<float>("probability", shape(row, column));
        auto sum_out = output<float>("sum", shape(row));
        auto mean_out = output<float>("mean", shape(row));
        auto peak_out = output<float>("peak", shape(row));
        auto argmax_out = output<int64_t>("argmax", shape(row));
        auto mse_out = output<float>("mse", shape(row));
        auto mae_out = output<float>("mae", shape(row));
        auto huber_out = output<float>("huber", shape(row));
        auto bce_out = output<float>("bce", shape(row));

        for (auto &row_nest : parallel(shape(row))) {
            auto sum = Scalar<float>{0.0f};
            auto peak = Scalar<float>{-1e30f};
            auto best_index = Scalar<int64_t>{0};
            auto mse = Scalar<float>{0.0f};
            auto mae = Scalar<float>{0.0f};
            auto huber = Scalar<float>{0.0f};
            auto bce = Scalar<float>{0.0f};
            for (auto &item : row_nest.reduce(shape(column))) {
                auto index = item[column];
                auto value = x(row_nest[row], index).load();
                auto expected = target(row_nest[row], index).load();
                auto p = probability(row_nest[row], index).load();
                auto difference = value - expected;
                auto magnitude = abs(difference);
                auto better = (value > peak) || ((value == peak) && (index < best_index));
                sum += value;
                peak = select(better, value, peak);
                best_index = select(better, index, best_index);
                mse += difference * difference;
                mae += magnitude;
                huber += select(
                    magnitude <= huber_delta,
                    0.5f * difference * difference,
                    huber_delta * (magnitude - 0.5f * huber_delta));
                auto clamped = min(max(p, epsilon), 1.0f - epsilon);
                bce -= expected * luisa::compute::tile::log(clamped) +
                       (1.0f - expected) * luisa::compute::tile::log(1.0f - clamped);
            }
            auto denominator = static_cast<float>(columns);
            sum_out(row_nest[row]) = sum;
            mean_out(row_nest[row]) = sum / denominator;
            peak_out(row_nest[row]) = peak;
            argmax_out(row_nest[row]) = best_index;
            mse_out(row_nest[row]) = mse / denominator;
            mae_out(row_nest[row]) = mae / denominator;
            huber_out(row_nest[row]) = huber / denominator;
            bce_out(row_nest[row]) = bce / denominator;
        }
    });
    auto executable = build(kernel);
    expect(executable.ok()) << executable.error;
    if (!executable.ok()) { return; }

    luisa::vector<float> x{
        -1.0f, 2.0f, 2.0f, 0.5f, -3.0f,
        4.0f, -2.0f, 1.0f, 3.0f, 0.0f,
        -5.0f, -1.0f, -3.0f, -1.0f, -2.0f};
    luisa::vector<float> target{
        0.0f, 1.0f, 0.0f, 1.0f, 0.0f,
        1.0f, 0.0f, 1.0f, 0.0f, 1.0f,
        0.0f, 1.0f, 0.0f, 1.0f, 0.0f};
    luisa::vector<float> probability{
        0.1f, 0.8f, 0.6f, 0.4f, 0.2f,
        0.9f, 0.3f, 0.7f, 0.2f, 0.55f,
        0.05f, 0.65f, 0.15f, 0.75f, 0.25f};
    auto x_tensor = upload<float>({rows, columns}, x);
    auto target_tensor = upload<float>({rows, columns}, target);
    auto probability_tensor = upload<float>({rows, columns}, probability);
    auto sum_tensor = allocate<float>({rows});
    auto mean_tensor = allocate<float>({rows});
    auto peak_tensor = allocate<float>({rows});
    auto argmax_tensor = allocate<int64_t>({rows});
    auto mse_tensor = allocate<float>({rows});
    auto mae_tensor = allocate<float>({rows});
    auto huber_tensor = allocate<float>({rows});
    auto bce_tensor = allocate<float>({rows});
    (*executable.entry)(x_tensor, target_tensor, probability_tensor,
                        sum_tensor, mean_tensor, peak_tensor, argmax_tensor,
                        mse_tensor, mae_tensor, huber_tensor, bce_tensor);

    auto sums = download<float>(sum_tensor, rows);
    auto means = download<float>(mean_tensor, rows);
    auto peaks = download<float>(peak_tensor, rows);
    auto argmax = download<int64_t>(argmax_tensor, rows);
    auto mses = download<float>(mse_tensor, rows);
    auto maes = download<float>(mae_tensor, rows);
    auto hubers = download<float>(huber_tensor, rows);
    auto bces = download<float>(bce_tensor, rows);
    for (auto r = 0; r < rows; r++) {
        auto sum = 0.0f;
        auto peak = -std::numeric_limits<float>::infinity();
        auto best_index = int64_t{0};
        auto mse = 0.0f;
        auto mae = 0.0f;
        auto huber = 0.0f;
        auto bce = 0.0f;
        for (auto c = 0; c < columns; c++) {
            auto index = static_cast<size_t>(r * columns + c);
            auto value = x[index];
            auto difference = value - target[index];
            auto magnitude = std::abs(difference);
            if (value > peak) {
                peak = value;
                best_index = c;
            }
            sum += value;
            mse += difference * difference;
            mae += magnitude;
            huber += magnitude <= huber_delta ?
                         0.5f * difference * difference :
                         huber_delta * (magnitude - 0.5f * huber_delta);
            auto p = std::clamp(probability[index], epsilon, 1.0f - epsilon);
            bce -= target[index] * std::log(p) + (1.0f - target[index]) * std::log(1.0f - p);
        }
        auto denominator = static_cast<float>(columns);
        expect(std::abs(sums[r] - sum) < 1e-5f);
        expect(std::abs(means[r] - sum / denominator) < 1e-5f);
        expect(std::abs(peaks[r] - peak) < 1e-5f);
        expect(eq(argmax[r], best_index));
        expect(std::abs(mses[r] - mse / denominator) < 1e-5f);
        expect(std::abs(maes[r] - mae / denominator) < 1e-5f);
        expect(std::abs(hubers[r] - huber / denominator) < 1e-5f);
        expect(std::abs(bces[r] - bce / denominator) < 1e-5f);
    }
}

void test_softmax_layernorm_rmsnorm() {
    constexpr int64_t rows = 4;
    constexpr int64_t columns = 7;
    constexpr float epsilon = 1e-5f;
    auto kernel = define("tile_poc_softmax_norm", [] {
        auto row = axis("row", rows);
        auto column = axis("column", columns);
        auto x = input<float>("x", shape(row, column));
        auto softmax_out = output<float>("softmax", shape(row, column));
        auto layernorm_out = output<float>("layernorm", shape(row, column));
        auto rmsnorm_out = output<float>("rmsnorm", shape(row, column));
        for (auto &row_nest : parallel(shape(row))) {
            auto peak = Scalar<float>{-1e30f};
            for (auto &item : row_nest.reduce(shape(column))) {
                peak = max(peak, x(row_nest[row], item[column]).load());
            }
            auto exponential_sum = Scalar<float>{0.0f};
            auto sum = Scalar<float>{0.0f};
            auto square_sum = Scalar<float>{0.0f};
            for (auto &item : row_nest.reduce(shape(column))) {
                auto value = x(row_nest[row], item[column]).load();
                exponential_sum += exp(value - peak);
                sum += value;
                square_sum += value * value;
            }
            auto denominator = static_cast<float>(columns);
            auto mean = sum / denominator;
            auto variance = max(square_sum / denominator - mean * mean, 0.0f);
            auto inverse_stddev = 1.0f / sqrt(variance + epsilon);
            auto inverse_rms = 1.0f / sqrt(square_sum / denominator + epsilon);
            for (auto &item : row_nest.parallel(shape(column))) {
                auto value = x(row_nest[row], item[column]).load();
                softmax_out(row_nest[row], item[column]) = exp(value - peak) / exponential_sum;
                layernorm_out(row_nest[row], item[column]) = (value - mean) * inverse_stddev;
                rmsnorm_out(row_nest[row], item[column]) = value * inverse_rms;
            }
        }
    });
    auto executable = build(kernel);
    expect(executable.ok()) << executable.error;
    if (!executable.ok()) { return; }

    luisa::vector<float> x(rows * columns);
    for (auto i = 0u; i < x.size(); i++) {
        x[i] = static_cast<float>(static_cast<int32_t>((i * 7u) % 19u) - 9) * 0.3f;
    }
    auto x_tensor = upload<float>({rows, columns}, x);
    auto softmax_tensor = allocate<float>({rows, columns});
    auto layernorm_tensor = allocate<float>({rows, columns});
    auto rmsnorm_tensor = allocate<float>({rows, columns});
    (*executable.entry)(x_tensor, softmax_tensor, layernorm_tensor, rmsnorm_tensor);
    auto softmax_values = download<float>(softmax_tensor, x.size());
    auto layernorm_values = download<float>(layernorm_tensor, x.size());
    auto rmsnorm_values = download<float>(rmsnorm_tensor, x.size());
    for (auto r = 0; r < rows; r++) {
        auto begin = static_cast<size_t>(r * columns);
        auto peak = -std::numeric_limits<float>::infinity();
        auto sum = 0.0f;
        auto square_sum = 0.0f;
        for (auto c = 0; c < columns; c++) {
            auto value = x[begin + static_cast<size_t>(c)];
            peak = std::max(peak, value);
            sum += value;
            square_sum += value * value;
        }
        auto exponential_sum = 0.0f;
        for (auto c = 0; c < columns; c++) {
            exponential_sum += std::exp(x[begin + static_cast<size_t>(c)] - peak);
        }
        auto denominator = static_cast<float>(columns);
        auto mean = sum / denominator;
        auto variance = std::max(square_sum / denominator - mean * mean, 0.0f);
        auto inverse_stddev = 1.0f / std::sqrt(variance + epsilon);
        auto inverse_rms = 1.0f / std::sqrt(square_sum / denominator + epsilon);
        for (auto c = 0; c < columns; c++) {
            auto index = begin + static_cast<size_t>(c);
            expect(std::abs(softmax_values[index] - std::exp(x[index] - peak) / exponential_sum) < 2e-5f);
            expect(std::abs(layernorm_values[index] - (x[index] - mean) * inverse_stddev) < 2e-5f);
            expect(std::abs(rmsnorm_values[index] - x[index] * inverse_rms) < 2e-5f);
        }
    }
}

void test_pipelined_gemm() {
    constexpr int64_t m_size = 5;
    constexpr int64_t n_size = 4;
    constexpr int64_t k_size = 7;
    auto kernel = define("tile_poc_pipelined_gemm", [] {
        auto m = axis("m", m_size);
        auto n = axis("n", n_size);
        auto k = axis("k", k_size);
        auto a = input<float>("a", shape(m, k));
        auto b = input<float>("b", shape(k, n));
        auto c = output<float>("c", shape(m, n));
        for (auto &output : parallel(shape(m, n))) {
            auto accumulator = Scalar<float>{0.0f};
            for (auto &step : output.pipeline(
                     shape(k), PipelinePolicy{.stages = 2u, .initiation_interval = 1u})) {
                step.stage("load");
                auto lhs = a(output[m], step[k]).load();
                auto rhs = b(step[k], output[n]).load();
                step.stage("compute");
                accumulator += lhs * rhs;
            }
            c(output[m], output[n]) = accumulator;
        }
    });
    auto executable = build(kernel);
    expect(executable.ok()) << executable.error;
    if (!executable.ok()) { return; }

    luisa::vector<float> a(m_size * k_size);
    luisa::vector<float> b(k_size * n_size);
    for (auto i = 0u; i < a.size(); i++) { a[i] = static_cast<float>((i % 11u) + 1u) * 0.1f; }
    for (auto i = 0u; i < b.size(); i++) { b[i] = static_cast<float>(static_cast<int32_t>(i % 9u) - 4) * 0.2f; }
    auto a_tensor = upload<float>({m_size, k_size}, a);
    auto b_tensor = upload<float>({k_size, n_size}, b);
    auto c_tensor = allocate<float>({m_size, n_size});
    (*executable.entry)(a_tensor, b_tensor, c_tensor);
    auto c = download<float>(c_tensor, m_size * n_size);
    for (auto m = 0; m < m_size; m++) {
        for (auto n = 0; n < n_size; n++) {
            auto expected = 0.0f;
            for (auto k = 0; k < k_size; k++) {
                expected += a[static_cast<size_t>(m * k_size + k)] *
                            b[static_cast<size_t>(k * n_size + n)];
            }
            expect(std::abs(c[static_cast<size_t>(m * n_size + n)] - expected) < 1e-5f);
        }
    }
}

void test_padded_strided_conv2d() {
    constexpr int64_t batch_size = 1;
    constexpr int64_t input_height = 5;
    constexpr int64_t input_width = 6;
    constexpr int64_t input_channels = 2;
    constexpr int64_t filter_height = 3;
    constexpr int64_t filter_width = 3;
    constexpr int64_t output_channels = 3;
    constexpr int64_t stride = 2;
    constexpr int64_t dilation = 1;
    constexpr int64_t padding = 1;
    constexpr int64_t output_height = 3;
    constexpr int64_t output_width = 3;
    auto kernel = define("tile_poc_conv2d", [] {
        auto batch = axis("batch", batch_size);
        auto input_y = axis("input_y", input_height);
        auto input_x = axis("input_x", input_width);
        auto input_channel = axis("input_channel", input_channels);
        auto filter_y = axis("filter_y", filter_height);
        auto filter_x = axis("filter_x", filter_width);
        auto output_y = axis("output_y", output_height);
        auto output_x = axis("output_x", output_width);
        auto output_channel = axis("output_channel", output_channels);
        auto x = input<float>("x", shape(batch, input_y, input_x, input_channel));
        auto weights = input<float>("weights", shape(filter_y, filter_x, input_channel, output_channel));
        auto bias = input<float>("bias", shape(output_channel));
        auto y = output<float>("y", shape(batch, output_y, output_x, output_channel));
        for (auto &output : parallel(shape(batch, output_y, output_x, output_channel))) {
            auto accumulator = Scalar<float>{0.0f};
            for (auto &tap : output.reduce(shape(filter_y, filter_x, input_channel))) {
                auto source_y = output[output_y] * stride + tap[filter_y] * dilation - padding;
                auto source_x = output[output_x] * stride + tap[filter_x] * dilation - padding;
                auto valid = (source_y >= 0) && (source_y < input_height) &&
                             (source_x >= 0) && (source_x < input_width);
                auto source = x(output[batch], source_y, source_x, tap[input_channel]).load(valid, 0.0f);
                auto weight = weights(tap[filter_y], tap[filter_x], tap[input_channel], output[output_channel]).load();
                accumulator += source * weight;
            }
            auto value = accumulator + bias(output[output_channel]).load();
            y(output[batch], output[output_y], output[output_x], output[output_channel]) = max(value, 0.0f);
        }
    });
    auto executable = build(kernel);
    expect(executable.ok()) << executable.error;
    if (!executable.ok()) { return; }

    luisa::vector<float> x(batch_size * input_height * input_width * input_channels);
    luisa::vector<float> weights(filter_height * filter_width * input_channels * output_channels);
    luisa::vector<float> bias(output_channels);
    for (auto i = 0u; i < x.size(); i++) {
        x[i] = static_cast<float>(static_cast<int32_t>((i * 5u) % 17u) - 8) * 0.125f;
    }
    for (auto i = 0u; i < weights.size(); i++) {
        weights[i] = static_cast<float>(static_cast<int32_t>((i * 3u) % 13u) - 6) * 0.0625f;
    }
    for (auto i = 0u; i < bias.size(); i++) { bias[i] = static_cast<float>(i) * 0.1f - 0.05f; }
    auto x_tensor = upload<float>({batch_size, input_height, input_width, input_channels}, x);
    auto weights_tensor = upload<float>({filter_height, filter_width, input_channels, output_channels}, weights);
    auto bias_tensor = upload<float>({output_channels}, bias);
    auto y_tensor = allocate<float>({batch_size, output_height, output_width, output_channels});
    (*executable.entry)(x_tensor, weights_tensor, bias_tensor, y_tensor);
    auto y = download<float>(y_tensor, batch_size * output_height * output_width * output_channels);

    auto x_index = [](int64_t n, int64_t iy, int64_t ix, int64_t c) noexcept {
        return static_cast<size_t>(((n * input_height + iy) * input_width + ix) * input_channels + c);
    };
    auto weight_index = [](int64_t fy, int64_t fx, int64_t c, int64_t oc) noexcept {
        return static_cast<size_t>(((fy * filter_width + fx) * input_channels + c) * output_channels + oc);
    };
    auto y_index = [](int64_t n, int64_t oy, int64_t ox, int64_t oc) noexcept {
        return static_cast<size_t>(((n * output_height + oy) * output_width + ox) * output_channels + oc);
    };
    for (auto n = 0; n < batch_size; n++) {
        for (auto oy = 0; oy < output_height; oy++) {
            for (auto ox = 0; ox < output_width; ox++) {
                for (auto oc = 0; oc < output_channels; oc++) {
                    auto expected = bias[static_cast<size_t>(oc)];
                    for (auto fy = 0; fy < filter_height; fy++) {
                        for (auto fx = 0; fx < filter_width; fx++) {
                            auto iy = oy * stride + fy * dilation - padding;
                            auto ix = ox * stride + fx * dilation - padding;
                            if (iy < 0 || iy >= input_height || ix < 0 || ix >= input_width) { continue; }
                            for (auto c = 0; c < input_channels; c++) {
                                expected += x[x_index(n, iy, ix, c)] * weights[weight_index(fy, fx, c, oc)];
                            }
                        }
                    }
                    expected = std::max(expected, 0.0f);
                    expect(std::abs(y[y_index(n, oy, ox, oc)] - expected) < 2e-5f);
                }
            }
        }
    }
}

}// namespace

int main(int argc, char *argv[]) {
    boost::ut::detail::cfg::parse_arg_with_fallback(argc, const_cast<const char **>(argv));
    "tile_tirx_poc_statistics_losses"_test = test_row_statistics_and_losses;
    "tile_tirx_poc_softmax_norm"_test = test_softmax_layernorm_rmsnorm;
    "tile_tirx_poc_pipelined_gemm"_test = test_pipelined_gemm;
    "tile_tirx_poc_conv2d"_test = test_padded_strided_conv2d;
}
