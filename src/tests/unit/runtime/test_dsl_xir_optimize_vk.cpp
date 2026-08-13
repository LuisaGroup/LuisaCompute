// Comprehensive smoke test for the XIR optimization pipeline on Vulkan.
// Exercises multiple optimization passes simultaneously in realistic kernels:
// - consecutive buffer reads (fuse-consecutive-buffer-reads for byte buffers)
// - adjacent stores with shared arithmetic (slp-vectorization)
// - small-trip-count loops (ordinary structured-loop legalization)
// - scaled induction-variable access (indvar strength reduction)
// - dependent adjacent loops preserved by the production pipeline
// Verifies that all passes compose correctly and produce correct results.

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

void test_vectorized_elementwise_pipeline(Device &device) {
    constexpr uint n = 64u;
    auto a = device.create_buffer<float>(n * 4u);
    auto b = device.create_buffer<float>(n * 4u);
    auto out = device.create_buffer<float>(n * 4u);
    Stream stream = device.create_stream();
    luisa::vector<float> host_a(n * 4u), host_b(n * 4u);
    for (auto i = 0u; i < n * 4u; i++) {
        host_a[i] = static_cast<float>(i) * 0.5f;
        host_b[i] = static_cast<float>(i) * 0.25f;
    }
    stream << a.copy_from(luisa::span{host_a})
           << b.copy_from(luisa::span{host_b})
           << synchronize();

    Kernel1D kernel = [&](BufferFloat buf_a, BufferFloat buf_b, BufferFloat buf_out) noexcept {
        auto tid = dispatch_id().x;
        auto v0 = buf_a.read(tid * 4u + 0u);
        auto v1 = buf_a.read(tid * 4u + 1u);
        auto v2 = buf_a.read(tid * 4u + 2u);
        auto v3 = buf_a.read(tid * 4u + 3u);
        auto w0 = buf_b.read(tid * 4u + 0u);
        auto w1 = buf_b.read(tid * 4u + 1u);
        auto w2 = buf_b.read(tid * 4u + 2u);
        auto w3 = buf_b.read(tid * 4u + 3u);
        buf_out.write(tid * 4u + 0u, v0 + w0);
        buf_out.write(tid * 4u + 1u, v1 + w1);
        buf_out.write(tid * 4u + 2u, v2 + w2);
        buf_out.write(tid * 4u + 3u, v3 + w3);
    };
    auto shader = device.compile(kernel);
    stream << shader(a, b, out).dispatch(n) << synchronize();

    luisa::vector<float> host_out(n * 4u);
    stream << out.copy_to(luisa::span{host_out}) << synchronize();
    luisa::vector<float> expected(n * 4u);
    for (auto i = 0u; i < n * 4u; i++) { expected[i] = host_a[i] + host_b[i]; }
    check_close(host_out, expected, "vectorized_elementwise_pipeline");
}

void test_small_loop_accumulation(Device &device) {
    constexpr uint n = 8u;
    auto data = device.create_buffer<float>(n);
    auto out = device.create_buffer<float>(1u);
    Stream stream = device.create_stream();
    luisa::vector<float> host(n);
    for (auto i = 0u; i < n; i++) { host[i] = static_cast<float>(i + 1u); }
    stream << data.copy_from(luisa::span{host}) << synchronize();

    Kernel1D kernel = [&](BufferFloat in, BufferFloat result) noexcept {
        $ sum = 0.0f;
        $for (i, 8) {
            sum += in.read(i);
        };
        result.write(0u, sum);
    };
    auto shader = device.compile(kernel);
    stream << shader(data, out).dispatch(1u) << synchronize();

    luisa::vector<float> host_out(1u);
    stream << out.copy_to(luisa::span{host_out}) << synchronize();
    check_close(host_out, luisa::vector<float>{36.0f}, "small_loop_accumulation");
}

void test_while_loop_with_break(Device &device) {
    constexpr uint n = 32u;
    auto in = device.create_buffer<float>(n);
    auto out = device.create_buffer<float>(n);
    Stream stream = device.create_stream();
    luisa::vector<float> host(n);
    for (auto i = 0u; i < n; i++) { host[i] = static_cast<float>(i); }
    stream << in.copy_from(luisa::span{host}) << synchronize();

    Kernel1D kernel = [&](BufferFloat in_buf, BufferFloat out_buf) noexcept {
        $ i = 0u;
        $loop {
            $if (i >= 32u) { $break; };
            out_buf.write(i, in_buf.read(i) + 10.0f);
            i += 1u;
        };
    };
    auto shader = device.compile(kernel);
    stream << shader(in, out).dispatch(1u) << synchronize();

    luisa::vector<float> host_out(n);
    stream << out.copy_to(luisa::span{host_out}) << synchronize();
    luisa::vector<float> expected(n);
    for (auto i = 0u; i < n; i++) { expected[i] = host[i] + 10.0f; }
    check_close(host_out, expected, "while_loop_with_break");
}

void test_scaled_index_read(Device &device) {
    constexpr uint n = 16u;
    auto data = device.create_buffer<float>(n * 3u + 5u);
    auto out = device.create_buffer<float>(n);
    Stream stream = device.create_stream();
    luisa::vector<float> host(n * 3u + 5u);
    for (auto i = 0u; i < host.size(); i++) { host[i] = static_cast<float>(i); }
    stream << data.copy_from(luisa::span{host}) << synchronize();

    Kernel1D kernel = [&](BufferFloat in, BufferFloat out_buf, UInt stride, UInt offset) noexcept {
        $for (i, 16) {
            out_buf.write(i, in.read(i * stride + offset));
        };
    };
    auto shader = device.compile(kernel);
    stream << shader(data, out, 3u, 2u).dispatch(1u) << synchronize();

    luisa::vector<float> host_out(n);
    stream << out.copy_to(luisa::span{host_out}) << synchronize();
    luisa::vector<float> expected(n);
    for (auto i = 0u; i < n; i++) { expected[i] = static_cast<float>(i * 3u + 2u); }
    check_close(host_out, expected, "scaled_index_read");
}

void test_two_loops_with_dependence(Device &device) {
    constexpr uint n = 32u;
    auto temp = device.create_buffer<float>(n);
    auto out = device.create_buffer<float>(n);
    Stream stream = device.create_stream();

    Kernel1D kernel = [&](BufferFloat tmp, BufferFloat result) noexcept {
        $for (i, 32) {
            tmp.write(i, cast<float>(i) * 2.0f);
        };
        $for (i, 32) {
            result.write(i, tmp.read(i) + 1.0f);
        };
    };
    auto shader = device.compile(kernel);
    stream << shader(temp, out).dispatch(1u) << synchronize();

    luisa::vector<float> host_out(n);
    stream << out.copy_to(luisa::span{host_out}) << synchronize();
    luisa::vector<float> expected(n);
    for (auto i = 0u; i < n; i++) { expected[i] = static_cast<float>(i) * 2.0f + 1.0f; }
    check_close(host_out, expected, "two_loops_with_dependence");
}

int main(int argc, char *argv[]) {
    auto dc = luisa::test::create_device_from_ut(argc, argv);
    if (!dc) return 0;
    boost::ut::detail::cfg::parse_arg_with_fallback(
        argc, const_cast<const char **>(argv));
    auto &device = dc->device;
    "xir_optimize_vk_vectorized_elementwise_pipeline"_test = [&] {
        test_vectorized_elementwise_pipeline(device);
    };
    "xir_optimize_vk_small_loop_accumulation"_test = [&] {
        test_small_loop_accumulation(device);
    };
    "xir_optimize_vk_while_loop_with_break"_test = [&] {
        test_while_loop_with_break(device);
    };
    "xir_optimize_vk_scaled_index_read"_test = [&] {
        test_scaled_index_read(device);
    };
    "xir_optimize_vk_two_loops_with_dependence"_test = [&] {
        test_two_loops_with_dependence(device);
    };
}
