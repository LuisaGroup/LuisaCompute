// Test for coroutine radix-sort helper.
// This test covers:
// - non-power-of-two bucket sorting used by small hint ranges
// - multi-pass radix sorting used by larger hint ranges
// - sort_switch output-buffer selection used by wavefront hint sorting
// - required subgroup size preservation through the Vulkan shader cache

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

void run_repeated_uniform_bucket_case(Device &device, uint n,
                                      uint iteration_count) {
    constexpr uint digit_count = 2u;
    Stream stream = device.create_stream();

    auto key_input = device.create_buffer<uint>(n);
    auto temp_key = device.create_buffer<uint>(n);
    auto key_output = device.create_buffer<uint>(n);
    auto value_output = device.create_buffer<uint>(n);

    Callable get_key = [&](UInt index) noexcept {
        return key_input->read(index);
    };
    Callable get_value = [](UInt index) noexcept {
        return index;
    };

    auto storage = radix_sort::temp_storage{device, n, digit_count};
    auto sorter = radix_sort::instance<>{
        device, n, storage, &get_key, &get_value, &get_key,
        1u, digit_count};

    luisa::vector<uint> input_keys(n, 1u);
    luisa::vector<uint> output_keys(n);
    luisa::vector<uint> output_values(n);
    for (auto iteration = 0u; iteration < iteration_count; iteration++) {
        // Model a coroutine self-loop followed by its terminal transition:
        // every frame repeatedly has the same live token, then every frame
        // changes to the empty token at once.
        if (iteration + 1u == iteration_count) {
            std::fill(input_keys.begin(), input_keys.end(), 0u);
        }
        stream << key_input.copy_from(luisa::span{input_keys}) << synchronize();
        // Wavefront token gathering aliases the unused bucket-mode temporary
        // value argument with its output index queue.
        sorter.sort(stream, temp_key.view(), value_output.view(),
                    key_output.view(), value_output.view(), n);
        stream << key_output.copy_to(luisa::span{output_keys})
               << value_output.copy_to(luisa::span{output_values})
               << synchronize();
        auto label = luisa::format(
            "repeated uniform bucket iteration {}", iteration);
        expect(validate_sorted_key_values(
                   label, input_keys, output_keys, output_values))
            << label;
    }
}

void run_bucket_indirect_self_loop_case(Device &device, uint n,
                                        uint round_count) {
    constexpr uint digit_count = 2u;
    Stream stream = device.create_stream();

    auto token = device.create_buffer<uint>(n);
    auto counter = device.create_buffer<uint>(n);
    auto temp_key = device.create_buffer<uint>(n);
    auto key_output = device.create_buffer<uint>(n);
    auto index_queue = device.create_buffer<uint>(n);

    Callable get_key = [&](UInt index) noexcept {
        return token->read(index);
    };
    Callable get_value = [](UInt index) noexcept {
        return index;
    };
    auto storage = radix_sort::temp_storage{device, n, digit_count};
    auto sorter = radix_sort::instance<>{
        device, n, storage, &get_key, &get_value, &get_key,
        1u, digit_count};

    Kernel1D resume = [](BufferUInt indices, BufferUInt counters,
                         BufferUInt tokens, UInt offset, UInt count,
                         UInt terminal_round) noexcept {
        auto lane = dispatch_x();
        $if (lane >= count) { $return(); };
        auto frame = indices.read(offset + lane);
        auto next_counter = counters.read(frame) + 1u;
        counters.write(frame, next_counter);
        tokens.write(frame, ite(next_counter < terminal_round, 1u, 0u));
    };
    auto resume_shader = device.compile(resume);

    luisa::vector<uint> initial_token(n, 1u);
    luisa::vector<uint> initial_counter(n, 0u);
    stream << token.copy_from(luisa::span{initial_token})
           << counter.copy_from(luisa::span{initial_counter})
           << synchronize();

    uint offsets[digit_count]{};
    for (auto round = 0u; round < round_count; round++) {
        sorter.sort(stream, temp_key.view(), index_queue.view(),
                    key_output.view(), index_queue.view(), n);
        stream << storage.hist_buffer.view().subview(0u, digit_count)
                      .copy_to(luisa::span{offsets, digit_count})
               << synchronize();
        auto live_count = n - offsets[1u];
        expect(live_count == n)
            << "all self-loop frames must remain live before the terminal round";
        stream << resume_shader(index_queue, counter, token, offsets[1u],
                                live_count, round_count)
                      .dispatch(live_count);
    }

    sorter.sort(stream, temp_key.view(), index_queue.view(),
                key_output.view(), index_queue.view(), n);
    luisa::vector<uint> final_token(n);
    luisa::vector<uint> final_counter(n);
    stream << storage.hist_buffer.view().subview(0u, digit_count)
                  .copy_to(luisa::span{offsets, digit_count})
           << token.copy_to(luisa::span{final_token})
           << counter.copy_to(luisa::span{final_counter})
           << synchronize();
    auto correct = true;
    for (auto i = 0u; i < n; i++) {
        if (final_token[i] != 0u || final_counter[i] != round_count) {
            LUISA_WARNING(
                "indirect self-loop mismatch at {}: token={}, counter={}, expected counter={}",
                i, final_token[i], final_counter[i], round_count);
            correct = false;
            break;
        }
    }
    if (offsets[1u] != n) {
        LUISA_WARNING(
            "indirect self-loop retained {} live frames after {} rounds",
            n - offsets[1u], round_count);
    }
    expect(offsets[1u] == n)
        << "the terminal transition must drain the indirect queue";
    expect(correct)
        << "sorted indirect updates must visit each frame exactly once per round";
}

void reg_coro_radix_sort(luisa::test::coro_test::Options options) {

    "coro_radix_sort_repeated_compilation_preserves_required_subgroup_size"_test = [options] {
        auto dc = luisa::test::coro_test::create_device(options);
        if (dc.device.compute_warp_size() != radix_sort::warp_size) {
            // This test intentionally exercises the one-sweep subgroup
            // contract. The bucket regressions below must still run on such
            // devices because bucket sorting has no subgroup dependency.
            expect(true);
            return;
        }
        auto output = dc.device.create_buffer<uint>(1u);
        Kernel1D kernel = [](BufferUInt result) noexcept {
            set_block_size(radix_sort::warp_size);
            set_warp_size(radix_sort::warp_size);
            $if (dispatch_x() == 0u) {
                result.write(0u, warp_lane_count());
            };
        };
        auto first_shader = dc.device.compile(kernel);
        auto second_shader = dc.device.compile(kernel);
        static_cast<void>(first_shader);
        auto stream = dc.device.create_stream();
        uint subgroup_size{};
        stream << second_shader(output).dispatch(radix_sort::warp_size)
               << output.copy_to(luisa::span{&subgroup_size, 1u})
               << synchronize();
        expect(subgroup_size == radix_sort::warp_size)
            << "repeated compilation must preserve the required subgroup size";
    };

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

    "coro_radix_sort_bucket_repeated_full_contention_is_permutation"_test = [options] {
        auto dc = luisa::test::coro_test::create_device(options);
        run_repeated_uniform_bucket_case(dc.device, 12288u, 12u);
    };

    "coro_radix_sort_bucket_indirect_self_loop_drains"_test = [options] {
        auto dc = luisa::test::coro_test::create_device(options);
        run_bucket_indirect_self_loop_case(dc.device, 12288u, 11u);
    };

    "coro_radix_sort_radix_direct_multiblock"_test = [options] {
        auto dc = luisa::test::coro_test::create_device(options);
        if (dc.device.compute_warp_size() != radix_sort::warp_size) {
            expect(true);
            return;
        }
        auto keys = make_radix_keys(5003u);
        run_sort_case(dc.device, keys, 0u, radix_sort::hist_block_size, 0u, 10u,
                      SortDispatch::direct, "radix direct multiblock");
    };

    "coro_radix_sort_radix_switch_multiblock"_test = [options] {
        auto dc = luisa::test::coro_test::create_device(options);
        if (dc.device.compute_warp_size() != radix_sort::warp_size) {
            expect(true);
            return;
        }
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
