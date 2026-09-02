// End-to-end Tile DSL PoCs for stencil, ranking, and irregular algorithms.

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
#include <array>
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

void test_depthwise_convolution_and_max_pool() {
    constexpr int64_t height = 5;
    constexpr int64_t width = 6;
    constexpr int64_t channels = 3;
    constexpr int64_t filter_size = 3;
    constexpr int64_t padding = 1;
    auto kernel = define("tile_poc_depthwise_pool", [] {
        auto y = axis("y", height);
        auto x = axis("x", width);
        auto channel = axis("channel", channels);
        auto filter_y = axis("filter_y", filter_size);
        auto filter_x = axis("filter_x", filter_size);
        auto source = input<float>("input", shape(y, x, channel));
        auto weights = input<float>("weights", shape(filter_y, filter_x, channel));
        auto convolution = output<float>("convolution", shape(y, x, channel));
        auto pool = output<float>("pool", shape(y, x, channel));
        for (auto &element : parallel(shape(y, x, channel))) {
            auto convolution_sum = Scalar<float>{0.0f};
            auto pool_maximum = Scalar<float>{-1e30f};
            for (auto &tap : element.reduce(shape(filter_y, filter_x))) {
                auto source_y = element[y] + tap[filter_y] - padding;
                auto source_x = element[x] + tap[filter_x] - padding;
                auto valid = (source_y >= 0) && (source_y < height) &&
                             (source_x >= 0) && (source_x < width);
                auto value = source(source_y, source_x, element[channel]).load(valid, 0.0f);
                convolution_sum += value * weights(tap[filter_y], tap[filter_x], element[channel]).load();
                pool_maximum = max(pool_maximum, value);
            }
            convolution[element] = convolution_sum;
            pool[element] = pool_maximum;
        }
    });
    auto executable = build(kernel);
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
    auto input_tensor = upload<float>({height, width, channels}, input);
    auto weights_tensor = upload<float>({filter_size, filter_size, channels}, weights);
    auto convolution_tensor = allocate<float>({height, width, channels});
    auto pool_tensor = allocate<float>({height, width, channels});
    (*executable.entry)(input_tensor, weights_tensor, convolution_tensor, pool_tensor);
    auto convolution = download<float>(convolution_tensor, input.size());
    auto pool = download<float>(pool_tensor, input.size());
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

void test_sobel_and_ordered_median() {
    constexpr int64_t height = 6;
    constexpr int64_t width = 7;
    constexpr int64_t tap_count = 9;
    auto kernel = define("tile_poc_sobel_median", [] {
        auto y = axis("y", height);
        auto x = axis("x", width);
        auto tap = axis("tap", tap_count);
        auto candidate = axis("candidate", tap_count);
        auto other = axis("other", tap_count);
        auto source = input<float>("input", shape(y, x));
        auto gradient_x = output<float>("gradient_x", shape(y, x));
        auto gradient_y = output<float>("gradient_y", shape(y, x));
        auto median = output<float>("median", shape(y, x));
        for (auto &element : parallel(shape(y, x))) {
            auto gx = Scalar<float>{0.0f};
            auto gy = Scalar<float>{0.0f};
            for (auto &sample : element.reduce(shape(tap))) {
                auto dy = sample[tap] / 3 - 1;
                auto dx = sample[tap] % 3 - 1;
                auto source_y = element[y] + dy;
                auto source_x = element[x] + dx;
                auto valid = (source_y >= 0) && (source_y < height) &&
                             (source_x >= 0) && (source_x < width);
                auto value = source(source_y, source_x).load(valid, 0.0f);
                auto weight_x = dx * select(dy == 0, int64_t{2}, int64_t{1});
                auto weight_y = dy * select(dx == 0, int64_t{2}, int64_t{1});
                gx += value * cast<float>(weight_x);
                gy += value * cast<float>(weight_y);
            }

            auto middle = Scalar<float>{0.0f};
            for (auto &candidate_nest : element.serial(shape(candidate))) {
                auto candidate_dy = candidate_nest[candidate] / 3 - 1;
                auto candidate_dx = candidate_nest[candidate] % 3 - 1;
                auto candidate_y = element[y] + candidate_dy;
                auto candidate_x = element[x] + candidate_dx;
                auto candidate_valid = (candidate_y >= 0) && (candidate_y < height) &&
                                       (candidate_x >= 0) && (candidate_x < width);
                auto candidate_value = source(candidate_y, candidate_x).load(candidate_valid, 0.0f);
                auto rank = Scalar<int64_t>{0};
                for (auto &other_nest : candidate_nest.reduce(shape(other))) {
                    auto other_dy = other_nest[other] / 3 - 1;
                    auto other_dx = other_nest[other] % 3 - 1;
                    auto other_y = element[y] + other_dy;
                    auto other_x = element[x] + other_dx;
                    auto other_valid = (other_y >= 0) && (other_y < height) &&
                                       (other_x >= 0) && (other_x < width);
                    auto other_value = source(other_y, other_x).load(other_valid, 0.0f);
                    auto before = (other_value < candidate_value) ||
                                  ((other_value == candidate_value) &&
                                   (other_nest[other] < candidate_nest[candidate]));
                    rank += cast<int64_t>(before);
                }
                middle = select(rank == 4, candidate_value, middle);
            }
            gradient_x[element] = gx;
            gradient_y[element] = gy;
            median[element] = middle;
        }
    });
    auto executable = build(kernel);
    expect(executable.ok()) << executable.error;
    if (!executable.ok()) { return; }

    luisa::vector<float> input(height * width);
    for (auto i = 0u; i < input.size(); i++) {
        input[i] = static_cast<float>(static_cast<int32_t>((i * 13u) % 31u) - 15) * 0.125f;
    }
    auto input_tensor = upload<float>({height, width}, input);
    auto gx_tensor = allocate<float>({height, width});
    auto gy_tensor = allocate<float>({height, width});
    auto median_tensor = allocate<float>({height, width});
    (*executable.entry)(input_tensor, gx_tensor, gy_tensor, median_tensor);
    auto gx = download<float>(gx_tensor, input.size());
    auto gy = download<float>(gy_tensor, input.size());
    auto median = download<float>(median_tensor, input.size());
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

void test_stable_sort_and_topk() {
    constexpr int64_t rows = 3;
    constexpr int64_t columns = 8;
    constexpr int64_t top_k = 3;
    auto kernel = define("tile_poc_stable_sort", [] {
        auto row = axis("row", rows);
        auto column = axis("column", columns);
        auto output_rank = axis("output_rank", columns);
        auto candidate = axis("candidate", columns);
        auto other = axis("other", columns);
        auto source = input<float>("input", shape(row, column));
        auto sorted_value = output<float>("sorted_value", shape(row, output_rank));
        auto sorted_index = output<int64_t>("sorted_index", shape(row, output_rank));
        for (auto &output_element : parallel(shape(row, output_rank))) {
            auto selected_value = Scalar<float>{-1e30f};
            auto selected_index = Scalar<int64_t>{-1};
            for (auto &candidate_nest : output_element.serial(shape(candidate))) {
                auto candidate_index = candidate_nest[candidate];
                auto candidate_value = source(output_element[row], candidate_index).load();
                auto rank = Scalar<int64_t>{0};
                for (auto &other_nest : candidate_nest.reduce(shape(other))) {
                    auto other_index = other_nest[other];
                    auto other_value = source(output_element[row], other_index).load();
                    auto before = (other_value > candidate_value) ||
                                  ((other_value == candidate_value) &&
                                   (other_index < candidate_index));
                    rank += cast<int64_t>(before);
                }
                auto matches = rank == output_element[output_rank];
                selected_value = select(matches, candidate_value, selected_value);
                selected_index = select(matches, candidate_index, selected_index);
            }
            sorted_value[output_element] = selected_value;
            sorted_index[output_element] = selected_index;
        }
    });
    auto executable = build(kernel);
    expect(executable.ok()) << executable.error;
    if (!executable.ok()) { return; }

    luisa::vector<float> input{
        2.0f, -1.0f, 2.0f, 5.0f, 5.0f, 0.0f, 3.0f, 2.0f,
        -4.0f, 8.0f, 1.0f, 8.0f, 2.0f, 2.0f, 7.0f, -1.0f,
        0.5f, 0.5f, -2.0f, 4.0f, 1.5f, 4.0f, 3.0f, 0.5f};
    auto input_tensor = upload<float>({rows, columns}, input);
    auto value_tensor = allocate<float>({rows, columns});
    auto index_tensor = allocate<int64_t>({rows, columns});
    (*executable.entry)(input_tensor, value_tensor, index_tensor);
    auto values = download<float>(value_tensor, input.size());
    auto indices = download<int64_t>(index_tensor, input.size());
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
            auto index = static_cast<size_t>(row * columns + rank);
            expect(eq(indices[index], expected[static_cast<size_t>(rank)].second));
        }
    }
}

void test_segmented_accumulation() {
    constexpr int64_t item_count = 13;
    constexpr int64_t bucket_count = 5;
    auto kernel = define("tile_poc_segmented_accumulation", [] {
        auto item = axis("item", item_count);
        auto bucket = axis("bucket", bucket_count);
        auto ids = input<int64_t>("ids", shape(item));
        auto values = input<float>("values", shape(item));
        auto sums = output<float>("sums", shape(bucket));
        auto counts = output<int64_t>("counts", shape(bucket));
        for (auto &segment : parallel(shape(bucket))) {
            auto sum = Scalar<float>{0.0f};
            auto count = Scalar<int64_t>{0};
            for (auto &source : segment.reduce(shape(item))) {
                auto matches = ids(source[item]).load() == segment[bucket];
                sum += select(matches, values(source[item]).load(), 0.0f);
                count += cast<int64_t>(matches);
            }
            sums[segment] = sum;
            counts[segment] = count;
        }
    });
    auto executable = build(kernel);
    expect(executable.ok()) << executable.error;
    if (!executable.ok()) { return; }

    luisa::vector<int64_t> ids{3, 1, 0, 3, 4, 1, 3, 2, 0, 2, 2, 4, 1};
    luisa::vector<float> values{1.0f, -2.0f, 3.0f, 4.5f, 1.5f, 2.0f, -1.0f,
                                8.0f, 0.5f, -3.0f, 2.5f, 7.0f, 1.25f};
    auto ids_tensor = upload<int64_t>({item_count}, ids);
    auto values_tensor = upload<float>({item_count}, values);
    auto sums_tensor = allocate<float>({bucket_count});
    auto counts_tensor = allocate<int64_t>({bucket_count});
    (*executable.entry)(ids_tensor, values_tensor, sums_tensor, counts_tensor);
    auto sums = download<float>(sums_tensor, bucket_count);
    auto counts = download<int64_t>(counts_tensor, bucket_count);
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

void test_all_structured_regions() {
    constexpr int64_t rows = 3;
    constexpr int64_t phases = 2;
    constexpr int64_t steps = 3;
    constexpr int64_t lanes = 4;
    auto kernel = define("tile_poc_all_structured_regions", [] {
        auto row = axis("row", rows);
        auto phase = axis("phase", phases);
        auto step = axis("step", steps);
        auto lane = axis("lane", lanes);
        auto source = input<float>("input", shape(row, phase, step, lane));
        auto result = output<float>("result", shape(row));
        for (auto &row_nest : parallel(shape(row))) {
            auto sum = Scalar<float>{0.0f};
            for (auto &phase_nest : row_nest.serial(shape(phase))) {
                for (auto &step_nest : phase_nest.pipeline(
                         shape(step), PipelinePolicy{.stages = 2u, .initiation_interval = 1u})) {
                    step_nest.stage("load");
                    for (auto &lane_nest : step_nest.reduce(shape(lane))) {
                        sum += source(row_nest[row], phase_nest[phase], step_nest[step], lane_nest[lane]).load();
                    }
                    step_nest.stage("consume");
                }
            }
            result(row_nest[row]) = sum;
        }
    });
    auto executable = build(kernel);
    expect(executable.ok()) << executable.error;
    if (!executable.ok()) { return; }

    luisa::vector<float> input(rows * phases * steps * lanes);
    for (auto i = 0u; i < input.size(); i++) {
        input[i] = static_cast<float>(static_cast<int32_t>((i * 3u) % 19u) - 9) * 0.25f;
    }
    auto input_tensor = upload<float>({rows, phases, steps, lanes}, input);
    auto result_tensor = allocate<float>({rows});
    (*executable.entry)(input_tensor, result_tensor);
    auto result = download<float>(result_tensor, rows);
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
    boost::ut::detail::cfg::parse_arg_with_fallback(argc, const_cast<const char **>(argv));
    "tile_tirx_poc_depthwise_pool"_test = test_depthwise_convolution_and_max_pool;
    "tile_tirx_poc_sobel_median"_test = test_sobel_and_ordered_median;
    "tile_tirx_poc_stable_sort_topk"_test = test_stable_sort_and_topk;
    "tile_tirx_poc_segmented_accumulation"_test = test_segmented_accumulation;
    "tile_tirx_poc_all_structured_regions"_test = test_all_structured_regions;
}
