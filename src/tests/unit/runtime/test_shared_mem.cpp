/**
 * @file test/feat/runtime/test_buffer.cpp
 * @author sailing-innocent
 * @date 2023/11/05
 * @brief test shared memory
*/

#include "ut/ut.hpp"
#include "test_device.h"

#include <luisa/core/logging.h>
#include <luisa/runtime/device.h>
#include <luisa/runtime/context.h>
#include <luisa/runtime/stream.h>
#include <luisa/runtime/buffer.h>
#include <luisa/dsl/sugar.h>

using namespace luisa;
using namespace luisa::compute;
using namespace boost::ut;
using namespace boost::ut::literals;

namespace {

int test_shared_mem(Device &device) {
    uint block_size = 32u;
    uint n = 1024u;
    Buffer<int> a = device.create_buffer<int>(n);

    Kernel1D test_kernel = [&](BufferVar<int> arr) noexcept {
        set_block_size(block_size);
        auto idx = dispatch_id().x;
        $if (idx > n) { return; };
        Shared<int> *s_data = new Shared<int>(block_size);
        auto thread_idx = thread_id().x;
        (*s_data)[thread_idx] = static_cast<$int>(thread_idx);
        sync_block();
        arr->write(idx, (*s_data)[thread_idx]);
    };
    auto test_shader = device.compile(test_kernel);
    auto stream = device.create_stream();
    stream << test_shader(a).dispatch(n);
    stream << synchronize();
    luisa::vector<int> data(n, 0);
    stream << a.copy_to(luisa::span{data});
    stream << synchronize();

    for (uint i = 0u; i < n; i++) {
        boost::ut::expect(static_cast<bool>(data[i] == i % block_size));
    }
    return 0;
}

void test_shared_float4_and_coro_frame_alignment(Device &device) {
    constexpr auto block_size = 32u;
    constexpr auto block_count = 4u;
    constexpr auto element_count = block_size * block_count;

    auto input = device.create_buffer<float4>(element_count);
    auto output = device.create_buffer<float4>(element_count);
    luisa::vector<float4> host_input(element_count);
    luisa::vector<float4> host_output(element_count);
    for (auto i = 0u; i < element_count; i++) {
        auto base = static_cast<float>(i * 4u);
        host_input[i] = make_float4(base + 1.0f, base + 2.0f, base + 3.0f, base + 4.0f);
    }

    Kernel1D kernel = [block_size](BufferFloat4 in, BufferFloat4 out) noexcept {
        set_block_size(block_size, 1u, 1u);
        Shared<float4> shared_values{block_size};
        auto global_id = dispatch_x();
        auto lane = thread_x();
        auto previous_lane = (lane + block_size - 1u) % block_size;
        auto next_lane = (lane + 1u) % block_size;

        Float4 carry = in.read(global_id);
        shared_values.write(lane, carry);
        sync_block();

        Float4 next_value = shared_values.read(next_lane);
        sync_block();
        carry = carry * 2.0f + next_value;
        shared_values.write(lane, carry);
        sync_block();

        Float4 previous_value = shared_values.read(previous_lane);
        sync_block();
        carry += previous_value * 3.0f;
        shared_values.write(lane, carry);
        sync_block();

        out.write(global_id, carry + shared_values.read(next_lane) * 4.0f);
    };

    auto shader = device.compile(kernel);
    auto stream = device.create_stream();
    stream << input.copy_from(luisa::span{host_input})
           << shader(input, output).dispatch(element_count)
           << output.copy_to(luisa::span{host_output})
           << synchronize();

    auto valid = true;
    for (auto i = 0u; i < element_count; i++) {
        auto block_begin = i / block_size * block_size;
        auto lane = i % block_size;
        auto previous = block_begin + (lane + block_size - 1u) % block_size;
        auto next = block_begin + (lane + 1u) % block_size;
        auto next_next = block_begin + (lane + 2u) % block_size;
        auto expected = host_input[i] * 29.0f +
                        host_input[next] * 21.0f +
                        host_input[previous] * 6.0f +
                        host_input[next_next] * 4.0f;
        if (!all(host_output[i] == expected)) {
            LUISA_WARNING(
                "Shared/coroutine alignment mismatch at {}: got ({}, {}, {}, {}), expected ({}, {}, {}, {}).",
                i,
                host_output[i].x, host_output[i].y, host_output[i].z, host_output[i].w,
                expected.x, expected.y, expected.z, expected.w);
            valid = false;
            break;
        }
    }
    expect(valid) << "16-byte vector data must survive shared-memory accesses and all five block-synchronization coroutine suspensions";
}

}// namespace

int main(int argc, char *argv[]) {
    auto dc = luisa::test::create_device_from_ut(argc, argv);
    if (!dc) {
        return 0;
    }
    boost::ut::detail::cfg::parse_arg_with_fallback(argc, const_cast<const char **>(argv));

    auto &device = dc->device;
    test_shared_mem(device);
    test_shared_float4_and_coro_frame_alignment(device);
}
