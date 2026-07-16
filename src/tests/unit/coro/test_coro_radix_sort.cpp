// Test for coroutine radix-sort helper.
// This test covers:
// - non-power-of-two bucket sorting used by small hint ranges
// - multi-pass radix sorting used by larger hint ranges
// - sort_switch output-buffer selection used by wavefront hint sorting

#include "ut/ut.hpp"
#include "coro_test_utils.h"

#include <luisa/coro/radix_sort.h>
#include <luisa/core/logging.h>
#include <luisa/dsl/sugar.h>
#include <luisa/runtime/buffer.h>
#include <luisa/runtime/device.h>
#include <luisa/runtime/stream.h>

using namespace luisa;
using namespace luisa::compute;
using namespace luisa::compute::coro;
using namespace boost::ut;
using namespace boost::ut::literals;

namespace {

enum class SortDispatch {
    direct,
    switch_buffers,
};

[[nodiscard]] auto make_bucket_keys(uint n, uint bucket_count) noexcept {
    luisa::vector<uint> keys(n);
    for (auto i = 0u; i < n; i++) {
        keys[i] = (i * 37u + (i / 7u) * 11u + 5u) % bucket_count;
    }
    return keys;
}

[[nodiscard]] auto make_radix_keys(uint n) noexcept {
    luisa::vector<uint> keys(n);
    for (auto i = 0u; i < n; i++) {
        keys[i] = ((i * 1103515245u + 12345u) ^ (i >> 3u) ^ (i * 97u)) & 2047u;
    }
    return keys;
}

[[nodiscard]] bool validate_sorted_key_values(
    luisa::string_view label,
    const luisa::vector<uint> &input_keys,
    const luisa::vector<uint> &output_keys,
    const luisa::vector<uint> &output_values) noexcept {
    auto n = static_cast<uint>(input_keys.size());
    if (output_keys.size() != n || output_values.size() != n) {
        LUISA_WARNING("{}: invalid output size keys={}, values={}, expected={}",
                      label, output_keys.size(), output_values.size(), n);
        return false;
    }
    luisa::vector<uint> seen(n, 0u);
    for (auto i = 0u; i < n; i++) {
        if (i != 0u && output_keys[i - 1u] > output_keys[i]) {
            LUISA_WARNING("{}: keys are not sorted at {}: {} > {}",
                          label, i, output_keys[i - 1u], output_keys[i]);
            return false;
        }
        auto value = output_values[i];
        if (value >= n) {
            LUISA_WARNING("{}: output value {} at {} is out of range {}",
                          label, value, i, n);
            return false;
        }
        if (seen[value] != 0u) {
            LUISA_WARNING("{}: duplicate output value {} at {}",
                          label, value, i);
            return false;
        }
        seen[value] = 1u;
        if (output_keys[i] != input_keys[value]) {
            LUISA_WARNING("{}: key/value mismatch at {}: key={}, value={}, input_key={}",
                          label, i, output_keys[i], value, input_keys[value]);
            return false;
        }
    }
    return true;
}

void run_sort_case(Device &device, const luisa::vector<uint> &keys, uint mode,
                   uint digit_count, uint low_bit, uint high_bit,
                   SortDispatch dispatch, luisa::string_view label) {
    auto n = static_cast<uint>(keys.size());
    Stream stream = device.create_stream();

    auto key_input = device.create_buffer<uint>(n);
    auto temp_key = device.create_buffer<uint>(n);
    auto temp_value = device.create_buffer<uint>(n);
    auto key_output = device.create_buffer<uint>(n);
    auto value_output = device.create_buffer<uint>(n);

    Callable get_key = [&](UInt index) noexcept {
        return key_input->read(index);
    };
    Callable get_value = [](UInt index) noexcept {
        return index;
    };

    auto storage = radix_sort::temp_storage{device, n, digit_count};
    auto sorter = radix_sort::instance<>{device, n, storage,
                                         &get_key, &get_value, &get_key,
                                         mode, digit_count, low_bit, high_bit};

    stream << key_input.copy_from(luisa::span{keys}) << synchronize();

    luisa::vector<uint> sorted_keys(n);
    luisa::vector<uint> sorted_values(n);
    if (dispatch == SortDispatch::direct) {
        sorter.sort(stream, temp_key.view(), temp_value.view(),
                    key_output.view(), value_output.view(), n);
        stream << key_output.copy_to(luisa::span{sorted_keys})
               << value_output.copy_to(luisa::span{sorted_values})
               << synchronize();
    } else {
        auto key_alt = device.create_buffer<uint>(n);
        auto value_alt = device.create_buffer<uint>(n);
        BufferView<uint> key_buffers[2] = {key_output.view(), key_alt.view()};
        BufferView<uint> value_buffers[2] = {value_output.view(), value_alt.view()};
        auto out_index = sorter.sort_switch(stream, key_buffers, value_buffers, n);
        expect(out_index < 2u) << "sort_switch should return one of the two output buffers";
        stream << key_buffers[out_index].copy_to(luisa::span{sorted_keys})
               << value_buffers[out_index].copy_to(luisa::span{sorted_values})
               << synchronize();
    }

    expect(validate_sorted_key_values(label, keys, sorted_keys, sorted_values))
        << label;
}

void reg_coro_radix_sort(luisa::test::coro_test::Options options) {

    "coro_radix_sort_bucket_direct_non_power_of_two"_test = [options] {
        auto dc = luisa::test::coro_test::create_device(options);
        auto keys = make_bucket_keys(4099u, 6u);
        run_sort_case(dc.device, keys, 1u, 6u, 0u, 31u, SortDispatch::direct,
                      "bucket direct non-power-of-two");
    };

    "coro_radix_sort_bucket_switch_non_power_of_two"_test = [options] {
        auto dc = luisa::test::coro_test::create_device(options);
        auto keys = make_bucket_keys(4099u, 6u);
        run_sort_case(dc.device, keys, 1u, 6u, 0u, 31u, SortDispatch::switch_buffers,
                      "bucket sort_switch non-power-of-two");
    };

    "coro_radix_sort_radix_direct_multiblock"_test = [options] {
        auto dc = luisa::test::coro_test::create_device(options);
        auto keys = make_radix_keys(5003u);
        run_sort_case(dc.device, keys, 0u, radix_sort::hist_block_size, 0u, 10u,
                      SortDispatch::direct, "radix direct multiblock");
    };

    "coro_radix_sort_radix_switch_multiblock"_test = [options] {
        auto dc = luisa::test::coro_test::create_device(options);
        auto keys = make_radix_keys(5003u);
        run_sort_case(dc.device, keys, 0u, radix_sort::hist_block_size, 0u, 10u,
                      SortDispatch::switch_buffers, "radix sort_switch multiblock");
    };
}

}// namespace

int main(int argc, char *argv[]) {
    auto options = luisa::test::coro_test::parse_options(argc, argv);
    reg_coro_radix_sort(options);
    return luisa::test::coro_test::run_tests(argc, argv);
}
