// Test for XIR loop unrolling via the DSL.
// This test covers:
// - small constant trip count loop (fully unrolled)
// - nested loops with small trip counts
// - accumulator loop (reduction pattern)
// - loop with a conditional inside
// - variable trip count (not unrolled, still correct)

#include "ut/ut.hpp"
#include "test_device.h"

#include <luisa/runtime/device.h>
#include <luisa/runtime/stream.h>
#include <luisa/runtime/buffer.h>
#include <luisa/dsl/sugar.h>

#include <cmath>

using namespace luisa;
using namespace luisa::compute;
using namespace boost::ut;
using namespace boost::ut::literals;

namespace {

void check_close(luisa::span<const float> got,
                 luisa::span<const float> expected,
                 luisa::string_view what) noexcept {
    auto ok = true;
    for (auto i = 0u; i < expected.size(); i++) {
        if (std::abs(got[i] - expected[i]) > 1e-4f) {
            LUISA_WARNING("Mismatch in {} at [{}]: got {} expected {}",
                          what, i, got[i], expected[i]);
            ok = false;
        }
    }
    expect(ok) << "all elements must match expected values";
}

}// namespace

void test_small_constant_trip_count(Device &device) {
    constexpr uint n = 4u;
    auto in = device.create_buffer<float>(n);
    auto out = device.create_buffer<float>(n);
    Stream stream = device.create_stream();
    luisa::vector<float> host_in{1.0f, 2.0f, 3.0f, 4.0f};
    stream << in.copy_from(luisa::span{host_in}) << synchronize();

    Kernel1D kernel = [&](BufferFloat in_buf, BufferFloat out_buf) noexcept {
        $for (i, 4) {
            out_buf.write(i, in_buf.read(i) * 2.0f);
        };
    };
    auto shader = device.compile(kernel);
    stream << shader(in, out).dispatch(1u) << synchronize();

    luisa::vector<float> host_out(n);
    stream << out.copy_to(luisa::span{host_out}) << synchronize();
    check_close(host_out, luisa::vector<float>{2.0f, 4.0f, 6.0f, 8.0f},
                "small_constant_trip_count");
}

void test_nested_small_trip_counts(Device &device) {
    constexpr uint n = 6u;
    auto in = device.create_buffer<float>(n);
    auto out = device.create_buffer<float>(n);
    Stream stream = device.create_stream();
    luisa::vector<float> host_in(n);
    for (auto i = 0u; i < n; i++) { host_in[i] = static_cast<float>(i); }
    stream << in.copy_from(luisa::span{host_in}) << synchronize();

    Kernel1D kernel = [&](BufferFloat in_buf, BufferFloat out_buf) noexcept {
        $for (i, 3) {
            $for (j, 2) {
                out_buf.write(i * 2 + j, in_buf.read(i * 2 + j) + cast<float>(i * j));
            };
        };
    };
    auto shader = device.compile(kernel);
    stream << shader(in, out).dispatch(1u) << synchronize();

    luisa::vector<float> host_out(n);
    stream << out.copy_to(luisa::span{host_out}) << synchronize();
    luisa::vector<float> expected(n);
    for (auto i = 0u; i < 3u; i++) {
        for (auto j = 0u; j < 2u; j++) {
            expected[i * 2u + j] = host_in[i * 2u + j] + static_cast<float>(i * j);
        }
    }
    check_close(host_out, expected, "nested_small_trip_counts");
}

void test_accumulator_loop(Device &device) {
    constexpr uint n = 8u;
    auto in = device.create_buffer<float>(n);
    auto out = device.create_buffer<float>(1u);
    Stream stream = device.create_stream();
    luisa::vector<float> host_in(n);
    for (auto i = 0u; i < n; i++) { host_in[i] = static_cast<float>(i + 1u); }
    stream << in.copy_from(luisa::span{host_in}) << synchronize();

    Kernel1D kernel = [&](BufferFloat in_buf, BufferFloat out_buf) noexcept {
        $ sum = 0.0f;
        $for (i, 8) {
            sum += in_buf.read(i);
        };
        out_buf.write(0u, sum);
    };
    auto shader = device.compile(kernel);
    stream << shader(in, out).dispatch(1u) << synchronize();

    luisa::vector<float> host_out(1u);
    stream << out.copy_to(luisa::span{host_out}) << synchronize();
    check_close(host_out, luisa::vector<float>{36.0f}, "accumulator_loop");
}

void test_loop_with_conditional(Device &device) {
    constexpr uint n = 4u;
    auto in = device.create_buffer<float>(n);
    auto out = device.create_buffer<float>(n);
    Stream stream = device.create_stream();
    luisa::vector<float> host_in{-1.0f, 2.0f, -3.0f, 4.0f};
    stream << in.copy_from(luisa::span{host_in}) << synchronize();

    Kernel1D kernel = [&](BufferFloat in_buf, BufferFloat out_buf) noexcept {
        $for (i, 4) {
            $if (in_buf.read(i) > 0.0f) {
                out_buf.write(i, 1.0f);
            } $else {
                out_buf.write(i, 0.0f);
            };
        };
    };
    auto shader = device.compile(kernel);
    stream << shader(in, out).dispatch(1u) << synchronize();

    luisa::vector<float> host_out(n);
    stream << out.copy_to(luisa::span{host_out}) << synchronize();
    check_close(host_out, luisa::vector<float>{0.0f, 1.0f, 0.0f, 1.0f},
                "loop_with_conditional");
}

void test_variable_trip_count(Device &device) {
    constexpr uint n = 64u;
    auto in = device.create_buffer<float>(n);
    auto out = device.create_buffer<float>(n);
    Stream stream = device.create_stream();
    luisa::vector<float> host_in(n, 0.0f);
    host_in[0] = 17.0f;// n = 17
    stream << in.copy_from(luisa::span{host_in}) << synchronize();

    Kernel1D kernel = [&](BufferFloat in_buf, BufferFloat out_buf) noexcept {
        $ count = cast<uint>(in_buf.read(0));
        $for (i, count) {
            out_buf.write(i, 1.0f);
        };
    };
    auto shader = device.compile(kernel);
    stream << shader(in, out).dispatch(1u) << synchronize();

    luisa::vector<float> host_out(n);
    stream << out.copy_to(luisa::span{host_out}) << synchronize();
    luisa::vector<float> expected(n, 0.0f);
    for (auto i = 0u; i < 17u; i++) { expected[i] = 1.0f; }
    check_close(host_out, expected, "variable_trip_count");
}

int main(int argc, char *argv[]) {
    auto dc = luisa::test::create_device_from_ut(argc, argv);
    if (!dc) return 0;
    boost::ut::detail::cfg::parse_arg_with_fallback(
        argc, const_cast<const char **>(argv));
    auto &device = dc->device;
    "dsl_loop_unroll_small_constant_trip_count"_test = [&] {
        test_small_constant_trip_count(device);
    };
    "dsl_loop_unroll_nested_small_trip_counts"_test = [&] {
        test_nested_small_trip_counts(device);
    };
    "dsl_loop_unroll_accumulator"_test = [&] {
        test_accumulator_loop(device);
    };
    "dsl_loop_unroll_with_conditional"_test = [&] {
        test_loop_with_conditional(device);
    };
    "dsl_loop_unroll_variable_trip_count"_test = [&] {
        test_variable_trip_count(device);
    };
}
