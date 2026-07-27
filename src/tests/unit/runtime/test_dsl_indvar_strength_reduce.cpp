// Test for XIR induction-variable strength reduction via the DSL.
// This test covers:
// - buffer access with a scaled induction variable
// - multiple buffer accesses with the same scaled IV
// - nested loop with outer IV in the inner index expression
// - non-constant stride (not strength-reduced, still correct)

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

void test_scaled_iv_access(Device &device) {
    constexpr uint n = 64u;
    auto in = device.create_buffer<float>(n * 4u + 8u);
    auto out = device.create_buffer<float>(n);
    Stream stream = device.create_stream();
    luisa::vector<float> host_in(n * 4u + 8u);
    for (auto i = 0u; i < host_in.size(); i++) { host_in[i] = static_cast<float>(i); }
    stream << in.copy_from(luisa::span{host_in}) << synchronize();

    Kernel1D kernel = [&](BufferFloat in_buf, BufferFloat out_buf, UInt base) noexcept {
        $for (i, 64) {
            out_buf.write(i, in_buf.read(i * 4u + base));
        };
    };
    auto shader = device.compile(kernel);
    stream << shader(in, out, 3u).dispatch(1u) << synchronize();

    luisa::vector<float> host_out(n);
    stream << out.copy_to(luisa::span{host_out}) << synchronize();
    luisa::vector<float> expected(n);
    for (auto i = 0u; i < n; i++) { expected[i] = static_cast<float>(i * 4u + 3u); }
    check_close(host_out, expected, "scaled_iv_access");
}

void test_multiple_accesses_same_scaled_iv(Device &device) {
    constexpr uint n = 64u;
    auto in1 = device.create_buffer<float>(n * 2u);
    auto in2 = device.create_buffer<float>(n * 2u);
    auto out = device.create_buffer<float>(n);
    Stream stream = device.create_stream();
    luisa::vector<float> host1(n * 2u), host2(n * 2u);
    for (auto i = 0u; i < n * 2u; i++) {
        host1[i] = static_cast<float>(i);
        host2[i] = static_cast<float>(i * 2u);
    }
    stream << in1.copy_from(luisa::span{host1})
           << in2.copy_from(luisa::span{host2})
           << synchronize();

    Kernel1D kernel = [&](BufferFloat a, BufferFloat b, BufferFloat out_buf) noexcept {
        $for (i, 64) {
            auto v = a.read(i * 2u) + b.read(i * 2u);
            out_buf.write(i, v);
        };
    };
    auto shader = device.compile(kernel);
    stream << shader(in1, in2, out).dispatch(1u) << synchronize();

    luisa::vector<float> host_out(n);
    stream << out.copy_to(luisa::span{host_out}) << synchronize();
    luisa::vector<float> expected(n);
    for (auto i = 0u; i < n; i++) { expected[i] = host1[i * 2u] + host2[i * 2u]; }
    check_close(host_out, expected, "multiple_accesses_same_scaled_iv");
}

void test_nested_outer_iv_in_inner_index(Device &device) {
    constexpr uint n = 32u;
    auto in = device.create_buffer<float>(n);
    auto out = device.create_buffer<float>(n);
    Stream stream = device.create_stream();
    luisa::vector<float> host_in(n);
    for (auto i = 0u; i < n; i++) { host_in[i] = static_cast<float>(i); }
    stream << in.copy_from(luisa::span{host_in}) << synchronize();

    Kernel1D kernel = [&](BufferFloat in_buf, BufferFloat out_buf) noexcept {
        $for (i, 8) {
            $for (j, 4) {
                out_buf.write(i * 4u + j, in_buf.read(i * 4u + j) * 2.0f);
            };
        };
    };
    auto shader = device.compile(kernel);
    stream << shader(in, out).dispatch(1u) << synchronize();

    luisa::vector<float> host_out(n);
    stream << out.copy_to(luisa::span{host_out}) << synchronize();
    luisa::vector<float> expected(n);
    for (auto i = 0u; i < n; i++) { expected[i] = host_in[i] * 2.0f; }
    check_close(host_out, expected, "nested_outer_iv_in_inner_index");
}

void test_variable_stride(Device &device) {
    constexpr uint n = 16u;
    auto in = device.create_buffer<float>(n * 4u);
    auto out = device.create_buffer<float>(n);
    Stream stream = device.create_stream();
    luisa::vector<float> host_in(n * 4u);
    for (auto i = 0u; i < n * 4u; i++) { host_in[i] = static_cast<float>(i); }
    stream << in.copy_from(luisa::span{host_in}) << synchronize();

    Kernel1D kernel = [&](BufferFloat in_buf, BufferFloat out_buf, UInt stride) noexcept {
        $for (i, 16) {
            out_buf.write(i, in_buf.read(i * stride));
        };
    };
    auto shader = device.compile(kernel);
    stream << shader(in, out, 3u).dispatch(1u) << synchronize();

    luisa::vector<float> host_out(n);
    stream << out.copy_to(luisa::span{host_out}) << synchronize();
    luisa::vector<float> expected(n);
    for (auto i = 0u; i < n; i++) { expected[i] = static_cast<float>(i * 3u); }
    check_close(host_out, expected, "variable_stride");
}

int main(int argc, char *argv[]) {
    auto dc = luisa::test::create_device_from_ut(argc, argv);
    if (!dc) return 0;
    boost::ut::detail::cfg::parse_arg_with_fallback(
        argc, const_cast<const char **>(argv));
    auto &device = dc->device;
    "dsl_indvar_scaled_iv_access"_test = [&] {
        test_scaled_iv_access(device);
    };
    "dsl_indvar_multiple_accesses_same_scaled_iv"_test = [&] {
        test_multiple_accesses_same_scaled_iv(device);
    };
    "dsl_indvar_nested_outer_iv_in_inner_index"_test = [&] {
        test_nested_outer_iv_in_inner_index(device);
    };
    "dsl_indvar_variable_stride"_test = [&] {
        test_variable_stride(device);
    };
}
