// Deterministic softmax conformance tests.
//
// Both implementations are exercised on nonuniform input and every output is
// compared with an independent, stable double-precision host implementation.

#include "ut/ut.hpp"
#include "test_device.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <string_view>

#include <luisa/luisa-compute.h>

using namespace luisa;
using namespace luisa::compute;
using namespace boost::ut;
using namespace boost::ut::literals;

namespace {

constexpr auto batch_block_size = 1024u;

template<typename T>
struct DispatchPack {
    Kernel1D<void(Buffer<T>)> kernel;
    uint dispatch_size;
};

template<typename T>
struct BatchDispatchPack {
    Kernel1D<void(Buffer<T>, Buffer<T>, uint, bool)> calc_sum;
    Kernel1D<void(Buffer<T>, Buffer<T>)> final;
};

template<typename T>
[[nodiscard]] BatchDispatchPack<T> batch_softmax_kernel(uint size) {
    auto calc_sum = Kernel1D([](BufferVar<T> input, BufferVar<T> output,
                                UInt size, Bool compute_exp) noexcept {
        set_block_size(batch_block_size, 1u, 1u);
        Shared<float> shared_values{batch_block_size};
        auto lane = thread_id().x;
        auto index = dispatch_id().x;
        Float value;
        $if (index < size) {
            value = ite(compute_exp,
                        exp(cast<float>(input.read(index))),
                        cast<float>(input.read(index)));
        }
        $else {
            value = 0.0f;
        };
        shared_values.write(lane, value);
        sync_block();

        auto active_count = def(batch_block_size / 2u);
        $while (active_count > 0u) {
            $if (lane < active_count) {
                value = shared_values.read(lane * 2u) +
                        shared_values.read(lane * 2u + 1u);
            };
            sync_block();
            $if (lane < active_count) {
                shared_values.write(lane, value);
            };
            active_count /= 2u;
            sync_block();
        };
        $if (lane == 0u) {
            output.write(block_id().x, shared_values.read(0u));
        };
    });

    auto final = Kernel1D([size](BufferVar<T> buffer,
                                 BufferVar<T> sum_buffer) noexcept {
        auto index = dispatch_id().x;
        $if (index < size) {
            auto value = exp(cast<float>(buffer.read(index))) /
                         cast<float>(sum_buffer.read(0u));
            buffer.write(index, cast<T>(value));
        };
    });
    return BatchDispatchPack<T>{std::move(calc_sum), std::move(final)};
}

template<typename T>
[[nodiscard]] DispatchPack<T> single_block_softmax_kernel(uint size) {
    LUISA_ASSERT(size > 0u && size <= batch_block_size,
                 "Single-block softmax size must be in [1, {}], got {}.",
                 batch_block_size, size);
    auto block_size = std::max(next_pow2(size), 32u);
    auto kernel = Kernel1D([size, block_size](BufferVar<T> input) noexcept {
        set_block_size(block_size, 1u, 1u);
        Shared<float> shared_values{block_size};
        auto lane = thread_id().x;
        auto index = dispatch_id().x;
        Float value;
        $if (index < size) {
            value = exp(cast<float>(input.read(index)));
        }
        $else {
            value = 0.0f;
        };
        shared_values.write(lane, value);
        sync_block();

        auto active_count = def(block_size / 2u);
        $while (active_count > 0u) {
            $if (lane < active_count) {
                value = shared_values.read(lane * 2u) +
                        shared_values.read(lane * 2u + 1u);
            };
            sync_block();
            $if (lane < active_count) {
                shared_values.write(lane, value);
            };
            active_count /= 2u;
            sync_block();
        };

        $if (index < size) {
            auto normalized = exp(cast<float>(input.read(index))) /
                              shared_values.read(0u);
            input.write(index, cast<T>(normalized));
        };
    });
    return DispatchPack<T>{
        .kernel = std::move(kernel),
        .dispatch_size = block_size};
}

[[nodiscard]] luisa::vector<float> make_input(uint size, float phase) {
    luisa::vector<float> input(size);
    for (auto i = 0u; i < size; i++) {
        auto x = static_cast<float>(i);
        auto centered = static_cast<float>(static_cast<int>(i % 19u) - 9);
        input[i] = 1.75f * std::sin(x * 0.017f + phase) +
                   0.65f * std::cos(x * 0.031f - phase * 0.5f) +
                   centered * 0.0275f;
    }
    return input;
}

[[nodiscard]] luisa::vector<float> reference_softmax(
    luisa::span<const float> input) {
    auto maximum = -std::numeric_limits<double>::infinity();
    for (auto value : input) {
        maximum = std::max(maximum, static_cast<double>(value));
    }
    auto denominator = 0.0;
    for (auto value : input) {
        denominator += std::exp(static_cast<double>(value) - maximum);
    }
    luisa::vector<float> reference(input.size());
    for (auto i = 0u; i < input.size(); i++) {
        reference[i] = static_cast<float>(
            std::exp(static_cast<double>(input[i]) - maximum) / denominator);
    }
    return reference;
}

void validate_softmax(luisa::span<const float> input,
                      luisa::span<const float> actual,
                      std::string_view label) {
    auto reference = reference_softmax(input);
    auto probabilities_valid = true;
    auto values_match = true;
    auto first_mismatch = actual.size();
    auto actual_sum = 0.0;
    constexpr auto absolute_tolerance = 5.0e-6;
    constexpr auto relative_tolerance = 2.0e-4;
    for (auto i = 0u; i < actual.size(); i++) {
        auto value = static_cast<double>(actual[i]);
        auto expected = static_cast<double>(reference[i]);
        auto valid = std::isfinite(value) && value >= 0.0;
        probabilities_valid &= valid;
        if (std::isfinite(value)) { actual_sum += value; }
        auto difference = std::abs(value - expected);
        auto tolerance = absolute_tolerance +
                         relative_tolerance * std::abs(expected);
        auto matches = difference <= tolerance;
        values_match &= matches;
        if ((!valid || !matches) && first_mismatch == actual.size()) {
            first_mismatch = i;
        }
    }
    if (first_mismatch != actual.size()) {
        LUISA_WARNING(
            "{} softmax mismatch at {}: got {}, expected {}.",
            label, first_mismatch, actual[first_mismatch],
            reference[first_mismatch]);
    }
    expect(probabilities_valid)
        << label << " softmax must produce only finite nonnegative probabilities";
    expect(values_match)
        << label << " softmax must match the stable CPU oracle at every element";
    expect(static_cast<bool>(probabilities_valid &&
                             std::abs(actual_sum - 1.0) <= 2.0e-4))
        << label << " softmax probabilities must sum to one; sum=" << actual_sum;
}

void test_batch_softmax(Device &device, Stream &stream) {
    constexpr auto size = 3073u;
    constexpr auto partial_sum_count =
        (size + batch_block_size - 1u) / batch_block_size;
    auto input = make_input(size, 0.35f);
    luisa::vector<float> actual(size);
    luisa::vector<float> partial_sums(
        partial_sum_count, std::numeric_limits<float>::quiet_NaN());
    auto buffer = device.create_buffer<float>(size);
    auto sum_buffer = device.create_buffer<float>(partial_sum_count);
    auto kernels = batch_softmax_kernel<float>(size);
    auto sum_shader = device.compile(kernels.calc_sum);
    auto final_shader = device.compile(kernels.final);

    stream << buffer.copy_from(luisa::span{input})
           << sum_buffer.copy_from(luisa::span{partial_sums})
           << sum_shader(buffer, sum_buffer, size, true).dispatch(size)
           << sum_shader(sum_buffer, sum_buffer, partial_sum_count, false)
                  .dispatch(batch_block_size)
           << final_shader(buffer, sum_buffer).dispatch(size)
           << buffer.copy_to(luisa::span{actual})
           << synchronize();
    validate_softmax(luisa::span<const float>{input},
                     luisa::span<const float>{actual}, "batch");
}

void test_single_block_softmax(Device &device, Stream &stream) {
    constexpr auto size = 257u;
    auto input = make_input(size, -0.8f);
    luisa::vector<float> actual(size);
    auto buffer = device.create_buffer<float>(size);
    auto kernel = single_block_softmax_kernel<float>(size);
    auto shader = device.compile(kernel.kernel);

    stream << buffer.copy_from(luisa::span{input})
           << shader(buffer).dispatch(kernel.dispatch_size)
           << buffer.copy_to(luisa::span{actual})
           << synchronize();
    validate_softmax(luisa::span<const float>{input},
                     luisa::span<const float>{actual}, "single-block");
}

}// namespace

void test_softmax(Device &device) {
    auto stream = device.create_stream();
    test_batch_softmax(device, stream);
    test_single_block_softmax(device, stream);
}

int main(int argc, char *argv[]) {
    auto dc = luisa::test::create_device_from_ut(argc, argv);
    if (!dc) { return 0; }
    auto &device = dc->device;
    test_softmax(device);
}
