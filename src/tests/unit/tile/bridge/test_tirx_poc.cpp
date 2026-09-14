// End-to-end execution tests for the Tile DSL PoC kernel gallery.
// Every memory access selects a subtile; arithmetic stays in Tile SSA.
// This test covers:
// - row statistics and common losses
// - softmax, LayerNorm, and RMSNorm
// - pipelined GEMM
// - padded, strided, multi-channel Conv2D

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
#include <array>
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

void test_row_statistics_and_losses(Runtime &runtime) {
    constexpr int64_t rows = 3;
    constexpr int64_t columns = 5;
    constexpr float huber_delta = 0.75f;
    constexpr float epsilon = 1e-4f;
    auto definition = tile_kernel(
        "tile_poc_statistics_losses",
        [=](TensorView<const float, 2> x,
            TensorView<const float, 2> target,
            TensorView<const float, 2> probability,
            TensorView<float, 1> sum_out,
            TensorView<float, 1> mean_out,
            TensorView<float, 1> peak_out,
            TensorView<int64_t, 1> argmax_out,
            TensorView<float, 1> mse_out,
            TensorView<float, 1> mae_out,
            TensorView<float, 1> huber_out,
            TensorView<float, 1> bce_out) {
            auto row = axis("row", x.extent<0>());
            auto r = axis("local_row", 1);
            auto c = axis("column", x.extent<1>());
            for (auto &nest : parallel(shape(row))) {
                auto origin = coord(nest.index(), 0);
                auto value = x[origin, shape(r, c)];
                auto expected = target[origin, shape(r, c)];
                auto probability_tile = probability[origin, shape(r, c)];
                auto difference = value - expected;
                auto magnitude = abs(difference);
                auto p = min(max(probability_tile, epsilon), 1.0f - epsilon);
                auto huber = ite(magnitude <= huber_delta,
                                 0.5f * difference * difference,
                                 huber_delta * (magnitude - 0.5f * huber_delta));
                auto bce = -(expected * luisa::compute::tile::log(p) + (1.0f - expected) * luisa::compute::tile::log(1.0f - p));
                auto sum = reduce(value, c, add);
                auto denominator = static_cast<float>(x.extent<1>());
                auto destination = coord(nest.index());
                sum_out(destination, shape(r)).store(sum);
                mean_out(destination, shape(r)).store(sum / denominator);
                peak_out(destination, shape(r)).store(reduce(value, c, maximum));
                argmax_out(destination, shape(r)).store(argmax(value, c));
                mse_out(destination, shape(r)).store(reduce(difference * difference, c, add) / denominator);
                mae_out(destination, shape(r)).store(reduce(magnitude, c, add) / denominator);
                huber_out(destination, shape(r)).store(reduce(huber, c, add) / denominator);
                bce_out(destination, shape(r)).store(reduce(bce, c, add) / denominator);
            }
        });
    auto kernel = definition.capture(
        tensor_shape("x", rows, columns),
        tensor_shape("target", rows, columns),
        tensor_shape("probability", rows, columns),
        tensor_shape("sum", rows), tensor_shape("mean", rows),
        tensor_shape("peak", rows), tensor_shape("argmax", rows),
        tensor_shape("mse", rows), tensor_shape("mae", rows),
        tensor_shape("huber", rows), tensor_shape("bce", rows));
    auto executable = runtime.build(kernel);
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
    auto x_tensor = runtime.upload<float>({rows, columns}, x);
    auto target_tensor = runtime.upload<float>({rows, columns}, target);
    auto probability_tensor = runtime.upload<float>({rows, columns}, probability);
    auto sum_tensor = runtime.allocate<float>({rows});
    auto mean_tensor = runtime.allocate<float>({rows});
    auto peak_tensor = runtime.allocate<float>({rows});
    auto argmax_tensor = runtime.allocate<int64_t>({rows});
    auto mse_tensor = runtime.allocate<float>({rows});
    auto mae_tensor = runtime.allocate<float>({rows});
    auto huber_tensor = runtime.allocate<float>({rows});
    auto bce_tensor = runtime.allocate<float>({rows});
    (*executable.entry)(x_tensor, target_tensor, probability_tensor,
                        sum_tensor, mean_tensor, peak_tensor, argmax_tensor,
                        mse_tensor, mae_tensor, huber_tensor, bce_tensor);

    auto sums = runtime.download<float>(sum_tensor, rows);
    auto means = runtime.download<float>(mean_tensor, rows);
    auto peaks = runtime.download<float>(peak_tensor, rows);
    auto argmax = runtime.download<int64_t>(argmax_tensor, rows);
    auto mses = runtime.download<float>(mse_tensor, rows);
    auto maes = runtime.download<float>(mae_tensor, rows);
    auto hubers = runtime.download<float>(huber_tensor, rows);
    auto bces = runtime.download<float>(bce_tensor, rows);
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

void test_softmax_layernorm_rmsnorm(Runtime &runtime) {
    constexpr int64_t rows = 4;
    constexpr int64_t columns = 7;
    constexpr float epsilon = 1e-5f;
    auto definition = tile_kernel(
        "tile_poc_softmax_norm",
        [=](TensorView<const float, 2> x,
            TensorView<float, 2> softmax_out,
            TensorView<float, 2> layernorm_out,
            TensorView<float, 2> rmsnorm_out) {
            auto row = axis("row", x.extent<0>());
            auto r = axis("local_row", 1);
            auto c = axis("column", x.extent<1>());
            for (auto &nest : parallel(shape(row))) {
                auto origin = coord(nest.index(), 0);
                auto value = x[origin, shape(r, c)];
                auto peak = reduce(value, c, maximum);
                auto exponentials = exp(value - peak);
                auto denominator = static_cast<float>(x.extent<1>());
                auto mean = reduce(value, c, add) / denominator;
                auto mean_square = reduce(value * value, c, add) / denominator;
                auto variance = max(mean_square - mean * mean, 0.0f);
                softmax_out(origin, shape(r, c)).store(exponentials / reduce(exponentials, c, add));
                layernorm_out(origin, shape(r, c)).store((value - mean) / sqrt(variance + epsilon));
                rmsnorm_out(origin, shape(r, c)).store(value / sqrt(mean_square + epsilon));
            }
        });
    auto kernel = definition.capture(
        tensor_shape("x", rows, columns), tensor_shape("softmax", rows, columns),
        tensor_shape("layernorm", rows, columns), tensor_shape("rmsnorm", rows, columns));
    auto executable = runtime.build(kernel);
    expect(executable.ok()) << executable.error;
    if (!executable.ok()) { return; }

    luisa::vector<float> x(rows * columns);
    for (auto i = 0u; i < x.size(); i++) {
        x[i] = static_cast<float>(static_cast<int32_t>((i * 7u) % 19u) - 9) * 0.3f;
    }
    auto x_tensor = runtime.upload<float>({rows, columns}, x);
    auto softmax_tensor = runtime.allocate<float>({rows, columns});
    auto layernorm_tensor = runtime.allocate<float>({rows, columns});
    auto rmsnorm_tensor = runtime.allocate<float>({rows, columns});
    (*executable.entry)(x_tensor, softmax_tensor, layernorm_tensor, rmsnorm_tensor);
    auto softmax_values = runtime.download<float>(softmax_tensor, x.size());
    auto layernorm_values = runtime.download<float>(layernorm_tensor, x.size());
    auto rmsnorm_values = runtime.download<float>(rmsnorm_tensor, x.size());
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

void test_pipelined_gemm(Runtime &runtime) {
    auto definition = tile_kernel(
        "tile_poc_pipelined_gemm",
        [=](TensorView<const float, 2> a,
            TensorView<const float, 2> b,
            TensorView<float, 2> c) {
            constexpr auto bm = 4;
            constexpr auto bn = 4;
            constexpr auto bk = 4;
            auto gm = axis("block_m", ceil_div(a.extent<0>(), bm));
            auto gn = axis("block_n", ceil_div(b.extent<1>(), bn));
            auto kt = axis("k_tiles", ceil_div(a.extent<1>(), bk));
            auto m = axis("m", bm);
            auto n = axis("n", bn);
            auto k = axis("k", bk);
            for (auto &nest : parallel(shape(gm, gn))) {
                auto m0 = nest.index(gm) * bm;
                auto n0 = nest.index(gn) * bn;
                auto acc = zeros<float>(shape(m, n));
                for (auto &step : nest.pipeline(shape(kt), {.stages = 2, .initiation_interval = 1})) {
                    step.stage("load");
                    auto a_tile = a[coord(m0, step.index() * bk), shape(m, k)];
                    auto b_tile = b[coord(step.index() * bk, n0), shape(k, n)];
                    step.stage("compute");
                    acc = mma(a_tile, b_tile, acc);
                }
                c(coord(m0, n0), shape(m, n)).store(acc);
            }
        });
    constexpr std::array<std::array<int64_t, 3>, 8> cases{{{1, 1, 1}, {5, 4, 7}, {4, 13, 3}, {13, 3, 9}, {9, 11, 10}, {16, 16, 16}, {31, 17, 29}, {65, 33, 17}}};
    for (auto [m_size, n_size, k_size] : cases) {
        auto kernel = definition.capture(
            tensor_shape("a", m_size, k_size), tensor_shape("b", k_size, n_size),
            tensor_shape("c", m_size, n_size));
        expect(eq(luisa::test::tile_tirx::count_operations(kernel.function().body(), OperationKind::MMA), 1u));
        auto executable = runtime.build(kernel);
        expect(executable.ok()) << executable.error;
        if (!executable.ok()) { return; }

        luisa::vector<float> a(m_size * k_size);
        luisa::vector<float> b(k_size * n_size);
        for (auto i = 0u; i < a.size(); i++) { a[i] = static_cast<float>((i % 11u) + 1u) * 0.1f; }
        for (auto i = 0u; i < b.size(); i++) { b[i] = static_cast<float>(static_cast<int32_t>(i % 9u) - 4) * 0.2f; }
        auto a_tensor = runtime.upload<float>({m_size, k_size}, a);
        auto b_tensor = runtime.upload<float>({k_size, n_size}, b);
        auto c_tensor = runtime.allocate<float>({m_size, n_size});
        (*executable.entry)(a_tensor, b_tensor, c_tensor);
        auto c = runtime.download<float>(c_tensor, m_size * n_size);
        for (auto m = 0; m < m_size; m++) {
            for (auto n = 0; n < n_size; n++) {
                auto expected = 0.0f;
                for (auto k = 0; k < k_size; k++) {
                    expected += a[static_cast<size_t>(m * k_size + k)] *
                                b[static_cast<size_t>(k * n_size + n)];
                }
                expect(std::abs(c[static_cast<size_t>(m * n_size + n)] - expected) < 1e-5f)
                    << "at (" << m << "," << n << "): actual " << c[static_cast<size_t>(m * n_size + n)] << ", expected " << expected;
            }
        }
    }
}

void test_padded_strided_conv2d(Runtime &runtime) {
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
    auto definition = tile_kernel(
        "tile_poc_conv2d",
        [=](TensorView<const float, 4> x,
            TensorView<const float, 4> weights,
            TensorView<const float, 1> bias,
            TensorView<float, 4> y) {
            auto batch = axis("batch", y.extent<0>());
            auto output_y = axis("output_y", y.extent<1>());
            auto output_x = axis("output_x", y.extent<2>());
            auto b = axis("local_batch", 1);
            auto oy = axis("local_y", 1);
            auto ox = axis("local_x", 1);
            auto fy = axis("filter_y", weights.extent<0>());
            auto fx = axis("filter_x", weights.extent<1>());
            auto ic = axis("input_channel", x.extent<3>());
            auto oc = axis("output_channel", y.extent<3>());
            auto wy = axis("window_y", (weights.extent<0>() - 1) * dilation + 1);
            auto wx = axis("window_x", (weights.extent<1>() - 1) * dilation + 1);
            for (auto &nest : parallel(shape(batch, output_y, output_x))) {
                auto n0 = nest.index(batch);
                auto y0 = nest.index(output_y);
                auto x0 = nest.index(output_x);
                auto window = x[coord(n0, y0 * stride - padding, x0 * stride - padding, 0), shape(b, wy, wx, ic)];
                auto taps = reindex(window, shape(b, fy, fx, ic), [&](const Nest &element) {
                    return coord(element.index(b), element.index(fy) * dilation, element.index(fx) * dilation, element.index(ic));
                });
                auto filter = weights[coord(0, 0, 0, 0), shape(fy, fx, ic, oc)];
                auto bias_tile = bias[coord(0), shape(oc)];
                auto value = reduce(taps * filter, shape(fy, fx, ic), add) + bias_tile;
                auto result = reshape(max(value, 0.0f), shape(b, oy, ox, oc));
                y(coord(n0, y0, x0, 0), shape(b, oy, ox, oc)).store(result);
            }
        });
    auto kernel = definition.capture(
        tensor_shape("x", batch_size, input_height, input_width, input_channels),
        tensor_shape("weights", filter_height, filter_width, input_channels, output_channels),
        tensor_shape("bias", output_channels),
        tensor_shape("y", batch_size, output_height, output_width, output_channels));
    auto executable = runtime.build(kernel);
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
    auto x_tensor = runtime.upload<float>({batch_size, input_height, input_width, input_channels}, x);
    auto weights_tensor = runtime.upload<float>({filter_height, filter_width, input_channels, output_channels}, weights);
    auto bias_tensor = runtime.upload<float>({output_channels}, bias);
    auto y_tensor = runtime.allocate<float>({batch_size, output_height, output_width, output_channels});
    (*executable.entry)(x_tensor, weights_tensor, bias_tensor, y_tensor);
    auto y = runtime.download<float>(y_tensor, batch_size * output_height * output_width * output_channels);

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
    auto runtime = Runtime{argc > 1 ? argv[1] : "cpu"};
    boost::ut::detail::cfg::parse_arg_with_fallback(
        argc > 1 ? argc - 1 : argc,
        const_cast<const char **>(argc > 1 ? argv + 1 : argv));
    "tile_tirx_poc_statistics_losses"_test = [&] { test_row_statistics_and_losses(runtime); };
    "tile_tirx_poc_softmax_norm"_test = [&] { test_softmax_layernorm_rmsnorm(runtime); };
    "tile_tirx_poc_pipelined_gemm"_test = [&] { test_pipelined_gemm(runtime); };
    "tile_tirx_poc_conv2d"_test = [&] { test_padded_strided_conv2d(runtime); };
}
