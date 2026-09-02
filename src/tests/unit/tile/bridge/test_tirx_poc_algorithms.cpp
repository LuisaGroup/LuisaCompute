// End-to-end Tile DSL PoCs for stencil, ranking, and irregular algorithms.

#include "ut/ut.hpp"
#include "tile_tirx_test_utils.h"

#include <tvm/ffi/function.h>
#include <tvm/ffi/string.h>
#include <tvm/runtime/tensor.h>

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
#include <utility>

using namespace luisa;
using namespace luisa::compute::tile;
using namespace luisa::compute::tile::bridge::tirx;
using namespace boost::ut;
using namespace boost::ut::literals;

namespace {

using luisa::test::tile_tirx::Runtime;

void test_depthwise_convolution_and_max_pool(Runtime &runtime) {
    constexpr int64_t height = 5;
    constexpr int64_t width = 6;
    constexpr int64_t channels = 3;
    constexpr int64_t filter_size = 3;
    constexpr int64_t padding = 1;
    auto definition = tile_kernel(
        "tile_poc_depthwise_pool",
        [](TensorView<const float, 3> source,
           TensorView<const float, 3> weights,
           TensorView<float, 3> convolution,
           TensorView<float, 3> pool) {
            auto y = axis("y", source.extent<0>());
            auto x = axis("x", source.extent<1>());
            auto c = axis("channel", source.extent<2>());
            auto fy = axis("filter_y", weights.extent<0>());
            auto fx = axis("filter_x", weights.extent<1>());
            auto oy = axis("local_y", 1);
            auto ox = axis("local_x", 1);
            for (auto &nest : parallel(shape(y, x))) {
                auto window = source[coord(nest[y] - padding, nest[x] - padding, 0), shape(fy, fx, c)];
                auto filter = weights[coord(0, 0, 0), shape(fy, fx, c)];
                auto conv = reduce(window * filter, shape(fy, fx), add);
                auto pooled = reduce(window, shape(fy, fx), maximum);
                auto destination = coord(nest[y], nest[x], 0);
                convolution(destination, shape(oy, ox, c)).store(reshape(conv, shape(oy, ox, c)));
                pool(destination, shape(oy, ox, c)).store(reshape(pooled, shape(oy, ox, c)));
            }
        });
    auto kernel = definition.capture(
        tensor_shape("input", height, width, channels),
        tensor_shape("weights", filter_size, filter_size, channels),
        tensor_shape("convolution", height, width, channels),
        tensor_shape("pool", height, width, channels));
    auto executable = runtime.build(kernel);
    expect(executable.ok()) << executable.error;
    if (!executable.ok()) { return; }

    luisa::vector<float> input(height * width * channels);
    luisa::vector<float> weights(filter_size * filter_size * channels);
    for (auto i = 0u; i < input.size(); i++) {
        input[i] = static_cast<float>(static_cast<int32_t>((i * 7u) % 23u) - 11) * 0.1f;
    }
    for (auto i = 0u; i < weights.size(); i++) {
        weights[i] = static_cast<float>(static_cast<int32_t>((i * 5u) % 17u) - 8) * 0.0625f;
    }
    auto input_tensor = runtime.upload<float>({height, width, channels}, input);
    auto weights_tensor = runtime.upload<float>({filter_size, filter_size, channels}, weights);
    auto convolution_tensor = runtime.allocate<float>({height, width, channels});
    auto pool_tensor = runtime.allocate<float>({height, width, channels});
    (*executable.entry)(input_tensor, weights_tensor, convolution_tensor, pool_tensor);
    auto convolution = runtime.download<float>(convolution_tensor, input.size());
    auto pool = runtime.download<float>(pool_tensor, input.size());
    auto input_index = [](int64_t iy, int64_t ix, int64_t c) noexcept {
        return static_cast<size_t>((iy * width + ix) * channels + c);
    };
    auto weight_index = [](int64_t fy, int64_t fx, int64_t c) noexcept {
        return static_cast<size_t>((fy * filter_size + fx) * channels + c);
    };
    for (auto iy = 0; iy < height; iy++) {
        for (auto ix = 0; ix < width; ix++) {
            for (auto c = 0; c < channels; c++) {
                auto expected_convolution = 0.0f;
                auto expected_pool = -std::numeric_limits<float>::infinity();
                for (auto fy = 0; fy < filter_size; fy++) {
                    for (auto fx = 0; fx < filter_size; fx++) {
                        auto sy = iy + fy - padding;
                        auto sx = ix + fx - padding;
                        auto value = sy >= 0 && sy < height && sx >= 0 && sx < width ?
                                         input[input_index(sy, sx, c)] :
                                         0.0f;
                        expected_convolution += value * weights[weight_index(fy, fx, c)];
                        expected_pool = std::max(expected_pool, value);
                    }
                }
                auto index = input_index(iy, ix, c);
                expect(std::abs(convolution[index] - expected_convolution) < 2e-5f);
                expect(std::abs(pool[index] - expected_pool) < 1e-6f);
            }
        }
    }
}

void test_sobel_and_ordered_median(Runtime &runtime) {
    constexpr int64_t height = 6;
    constexpr int64_t width = 7;
    constexpr int64_t tap_count = 9;
    auto definition = tile_kernel(
        "tile_poc_sobel_median",
        [](TensorView<const float, 2> source,
           TensorView<float, 2> gradient_x,
           TensorView<float, 2> gradient_y,
           TensorView<float, 2> median) {
            auto y = axis("y", source.extent<0>());
            auto x = axis("x", source.extent<1>());
            auto fy = axis("filter_y", 3);
            auto fx = axis("filter_x", 3);
            auto tap = axis("tap", tap_count);
            auto oy = axis("local_y", 1);
            auto ox = axis("local_x", 1);
            for (auto &nest : parallel(shape(y, x))) {
                auto window = source[coord(nest[y] - 1, nest[x] - 1), shape(fy, fx)];
                auto dy = iota(fy) - 1;
                auto dx = iota(fx) - 1;
                auto weight_x = cast<float>(dx * select(dy == 0, int64_t{2}, int64_t{1}));
                auto weight_y = cast<float>(dy * select(dx == 0, int64_t{2}, int64_t{1}));
                auto gx = reduce(window * weight_x, shape(fy, fx), add);
                auto gy = reduce(window * weight_y, shape(fy, fx), add);
                auto ordered = sort(reshape(window, shape(tap)), tap);
                auto middle = gather(ordered.values, full<int64_t>(IndexSpace{}, 4), tap);
                auto destination = coord(nest[y], nest[x]);
                gradient_x(destination, shape(oy, ox)).store(gx);
                gradient_y(destination, shape(oy, ox)).store(gy);
                median(destination, shape(oy, ox)).store(middle);
            }
        });
    auto kernel = definition.capture(
        tensor_shape("input", height, width), tensor_shape("gradient_x", height, width),
        tensor_shape("gradient_y", height, width), tensor_shape("median", height, width));
    auto executable = runtime.build(kernel);
    expect(executable.ok()) << executable.error;
    if (!executable.ok()) { return; }

    luisa::vector<float> input(height * width);
    for (auto i = 0u; i < input.size(); i++) {
        input[i] = static_cast<float>(static_cast<int32_t>((i * 13u) % 31u) - 15) * 0.125f;
    }
    auto input_tensor = runtime.upload<float>({height, width}, input);
    auto gx_tensor = runtime.allocate<float>({height, width});
    auto gy_tensor = runtime.allocate<float>({height, width});
    auto median_tensor = runtime.allocate<float>({height, width});
    (*executable.entry)(input_tensor, gx_tensor, gy_tensor, median_tensor);
    auto gx = runtime.download<float>(gx_tensor, input.size());
    auto gy = runtime.download<float>(gy_tensor, input.size());
    auto median = runtime.download<float>(median_tensor, input.size());
    auto load = [&](int64_t iy, int64_t ix) noexcept {
        return iy >= 0 && iy < height && ix >= 0 && ix < width ?
                   input[static_cast<size_t>(iy * width + ix)] :
                   0.0f;
    };
    for (auto iy = 0; iy < height; iy++) {
        for (auto ix = 0; ix < width; ix++) {
            auto expected_gx = 0.0f;
            auto expected_gy = 0.0f;
            std::array<std::pair<float, int64_t>, tap_count> window;
            auto tap_index = int64_t{0};
            for (auto dy = -1; dy <= 1; dy++) {
                for (auto dx = -1; dx <= 1; dx++) {
                    auto value = load(iy + dy, ix + dx);
                    expected_gx += value * static_cast<float>(dx * (dy == 0 ? 2 : 1));
                    expected_gy += value * static_cast<float>(dy * (dx == 0 ? 2 : 1));
                    window[static_cast<size_t>(tap_index)] = {value, tap_index};
                    tap_index++;
                }
            }
            std::sort(window.begin(), window.end(), [](auto lhs, auto rhs) noexcept {
                return lhs.first < rhs.first || (lhs.first == rhs.first && lhs.second < rhs.second);
            });
            auto index = static_cast<size_t>(iy * width + ix);
            expect(std::abs(gx[index] - expected_gx) < 1e-5f);
            expect(std::abs(gy[index] - expected_gy) < 1e-5f);
            expect(std::abs(median[index] - window[4u].first) < 1e-6f);
        }
    }
}

void test_stable_sort_and_topk(Runtime &runtime) {
    constexpr int64_t rows = 3;
    constexpr int64_t columns = 8;
    constexpr int64_t top_k = 3;
    auto definition = tile_kernel(
        "tile_poc_stable_sort",
        [](TensorView<const float, 2> source,
           TensorView<float, 2> sorted_value,
           TensorView<int64_t, 2> sorted_index) {
            auto row = axis("row", source.extent<0>());
            auto r = axis("local_row", 1);
            auto c = axis("column", source.extent<1>());
            auto rank = axis("output_rank", sorted_value.extent<1>());
            for (auto &nest : parallel(shape(row))) {
                auto value = source[coord(nest.index(), 0), shape(r, c)];
                auto ranked = topk(value, c, sorted_value.extent<1>());
                auto destination = coord(nest.index(), 0);
                sorted_value(destination, shape(r, rank)).store(ranked.values);
                sorted_index(destination, shape(r, rank)).store(ranked.indices);
            }
        });
    auto kernel = definition.capture(
        tensor_shape("input", rows, columns), tensor_shape("sorted_value", rows, columns),
        tensor_shape("sorted_index", rows, columns));
    auto executable = runtime.build(kernel);
    expect(executable.ok()) << executable.error;
    if (!executable.ok()) { return; }

    luisa::vector<float> input{
        2.0f, -1.0f, 2.0f, 5.0f, 5.0f, 0.0f, 3.0f, 2.0f,
        -4.0f, 8.0f, 1.0f, 8.0f, 2.0f, 2.0f, 7.0f, -1.0f,
        0.5f, 0.5f, -2.0f, 4.0f, 1.5f, 4.0f, 3.0f, 0.5f};
    auto input_tensor = runtime.upload<float>({rows, columns}, input);
    auto value_tensor = runtime.allocate<float>({rows, columns});
    auto index_tensor = runtime.allocate<int64_t>({rows, columns});
    (*executable.entry)(input_tensor, value_tensor, index_tensor);
    auto values = runtime.download<float>(value_tensor, input.size());
    auto indices = runtime.download<int64_t>(index_tensor, input.size());
    auto topk_kernel = definition.capture(
        tensor_shape("input", rows, columns), tensor_shape("topk_value", rows, top_k),
        tensor_shape("topk_index", rows, top_k));
    auto topk_executable = runtime.build(topk_kernel);
    expect(topk_executable.ok()) << topk_executable.error;
    if (!topk_executable.ok()) { return; }
    auto topk_value_tensor = runtime.allocate<float>({rows, top_k});
    auto topk_index_tensor = runtime.allocate<int64_t>({rows, top_k});
    (*topk_executable.entry)(input_tensor, topk_value_tensor, topk_index_tensor);
    auto topk_values = runtime.download<float>(topk_value_tensor, rows * top_k);
    auto topk_indices = runtime.download<int64_t>(topk_index_tensor, rows * top_k);
    for (auto row = 0; row < rows; row++) {
        luisa::vector<std::pair<float, int64_t>> expected;
        expected.reserve(columns);
        for (auto column = 0; column < columns; column++) {
            expected.emplace_back(
                input[static_cast<size_t>(row * columns + column)], column);
        }
        std::sort(expected.begin(), expected.end(), [](auto lhs, auto rhs) noexcept {
            return lhs.first > rhs.first || (lhs.first == rhs.first && lhs.second < rhs.second);
        });
        for (auto rank = 0; rank < columns; rank++) {
            auto index = static_cast<size_t>(row * columns + rank);
            expect(std::abs(values[index] - expected[static_cast<size_t>(rank)].first) < 1e-6f);
            expect(eq(indices[index], expected[static_cast<size_t>(rank)].second));
        }
        for (auto rank = 0; rank < top_k; rank++) {
            auto index = static_cast<size_t>(row * top_k + rank);
            expect(eq(topk_values[index], expected[static_cast<size_t>(rank)].first));
            expect(eq(topk_indices[index], expected[static_cast<size_t>(rank)].second));
        }
    }
}

void test_segmented_accumulation(Runtime &runtime) {
    constexpr int64_t item_count = 13;
    constexpr int64_t bucket_count = 5;
    auto definition = tile_kernel(
        "tile_poc_segmented_accumulation",
        [](TensorView<const int64_t, 1> ids,
           TensorView<const float, 1> values,
           TensorView<float, 1> sums,
           TensorView<int64_t, 1> counts) {
            auto item = axis("item", ids.extent<0>());
            auto bucket = axis("bucket", sums.extent<0>());
            auto output = axis("output", 1);
            for (auto &nest : parallel(shape(bucket))) {
                auto id = ids[coord(0), shape(item)];
                auto value = values[coord(0), shape(item)];
                auto matches = id == nest.index();
                auto sum = reduce(select(matches, value, 0.0f), item, add);
                auto count = reduce(cast<int64_t>(matches), item, add);
                sums(coord(nest.index()), shape(output)).store(sum);
                counts(coord(nest.index()), shape(output)).store(count);
            }
        });
    auto kernel = definition.capture(
        tensor_shape("ids", item_count), tensor_shape("values", item_count),
        tensor_shape("sums", bucket_count), tensor_shape("counts", bucket_count));
    auto executable = runtime.build(kernel);
    expect(executable.ok()) << executable.error;
    if (!executable.ok()) { return; }

    luisa::vector<int64_t> ids{3, 1, 0, 3, 4, 1, 3, 2, 0, 2, 2, 4, 1};
    luisa::vector<float> values{1.0f, -2.0f, 3.0f, 4.5f, 1.5f, 2.0f, -1.0f,
                                8.0f, 0.5f, -3.0f, 2.5f, 7.0f, 1.25f};
    auto ids_tensor = runtime.upload<int64_t>({item_count}, ids);
    auto values_tensor = runtime.upload<float>({item_count}, values);
    auto sums_tensor = runtime.allocate<float>({bucket_count});
    auto counts_tensor = runtime.allocate<int64_t>({bucket_count});
    (*executable.entry)(ids_tensor, values_tensor, sums_tensor, counts_tensor);
    auto sums = runtime.download<float>(sums_tensor, bucket_count);
    auto counts = runtime.download<int64_t>(counts_tensor, bucket_count);
    for (auto bucket = 0; bucket < bucket_count; bucket++) {
        auto expected_sum = 0.0f;
        auto expected_count = int64_t{0};
        for (auto item = 0; item < item_count; item++) {
            if (ids[static_cast<size_t>(item)] == bucket) {
                expected_sum += values[static_cast<size_t>(item)];
                expected_count++;
            }
        }
        expect(std::abs(sums[static_cast<size_t>(bucket)] - expected_sum) < 1e-6f);
        expect(eq(counts[static_cast<size_t>(bucket)], expected_count));
    }
}

void test_all_structured_regions(Runtime &runtime) {
    constexpr int64_t rows = 3;
    constexpr int64_t phases = 2;
    constexpr int64_t steps = 3;
    constexpr int64_t lanes = 4;
    auto definition = tile_kernel(
        "tile_poc_all_structured_regions",
        [](TensorView<const float, 4> source, TensorView<float, 1> result) {
            auto row = axis("row", source.extent<0>());
            auto phase = axis("phase", source.extent<1>());
            auto step = axis("step", source.extent<2>());
            auto lane = axis("lane", source.extent<3>());
            auto r = axis("local_row", 1);
            auto p = axis("local_phase", 1);
            auto s = axis("local_step", 1);
            for (auto &nest : parallel(shape(row))) {
                auto sum = zeros<float>(shape(r));
                for (auto &phase_nest : nest.serial(shape(phase))) {
                    for (auto &step_nest : phase_nest.pipeline(shape(step), {.stages = 2, .initiation_interval = 1})) {
                        step_nest.stage("load");
                        auto value = source[coord(nest[row], phase_nest[phase], step_nest[step], 0), shape(r, p, s, lane)];
                        step_nest.stage("consume");
                        sum += reshape(reduce(value, lane, add), shape(r));
                    }
                }
                result(coord(nest[row]), shape(r)).store(sum);
            }
        });
    auto kernel = definition.capture(
        tensor_shape("input", rows, phases, steps, lanes), tensor_shape("result", rows));
    auto executable = runtime.build(kernel);
    expect(executable.ok()) << executable.error;
    if (!executable.ok()) { return; }

    luisa::vector<float> input(rows * phases * steps * lanes);
    for (auto i = 0u; i < input.size(); i++) {
        input[i] = static_cast<float>(static_cast<int32_t>((i * 3u) % 19u) - 9) * 0.25f;
    }
    auto input_tensor = runtime.upload<float>({rows, phases, steps, lanes}, input);
    auto result_tensor = runtime.allocate<float>({rows});
    (*executable.entry)(input_tensor, result_tensor);
    auto result = runtime.download<float>(result_tensor, rows);
    auto row_stride = phases * steps * lanes;
    for (auto row = 0; row < rows; row++) {
        auto expected = 0.0f;
        for (auto i = 0; i < row_stride; i++) {
            expected += input[static_cast<size_t>(row * row_stride + i)];
        }
        expect(std::abs(result[static_cast<size_t>(row)] - expected) < 1e-6f);
    }
}

}// namespace

int main(int argc, char *argv[]) {
    auto runtime = Runtime{argc > 1 ? argv[1] : "cpu"};
    boost::ut::detail::cfg::parse_arg_with_fallback(
        argc > 1 ? argc - 1 : argc,
        const_cast<const char **>(argc > 1 ? argv + 1 : argv));
    "tile_tirx_poc_depthwise_pool"_test = [&] { test_depthwise_convolution_and_max_pool(runtime); };
    "tile_tirx_poc_sobel_median"_test = [&] { test_sobel_and_ordered_median(runtime); };
    "tile_tirx_poc_stable_sort_topk"_test = [&] { test_stable_sort_and_topk(runtime); };
    "tile_tirx_poc_segmented_accumulation"_test = [&] { test_segmented_accumulation(runtime); };
    "tile_tirx_poc_all_structured_regions"_test = [&] { test_all_structured_regions(runtime); };
}
