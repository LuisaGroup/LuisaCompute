// Test for XIR loop rotation via the DSL.
// This test covers:
// - while-style loop with counter (top-checked, rotatable)
// - nested while loops

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

void test_while_style_loop(Device &device) {
    constexpr uint n = 64u;
    auto in = device.create_buffer<float>(n);
    auto out = device.create_buffer<float>(n);
    Stream stream = device.create_stream();

    luisa::vector<float> host_in(n);
    for (auto i = 0u; i < n; i++) { host_in[i] = static_cast<float>(i) * 0.5f; }
    stream << in.copy_from(luisa::span{host_in}) << synchronize();

    Kernel1D kernel = [&](BufferFloat in_buf, BufferFloat out_buf) noexcept {
        $ i = 0u;
        $loop {
            $if (i >= 64u) { $break; };
            out_buf.write(i, in_buf.read(i) * 2.0f);
            i += 1u;
        };
    };
    auto shader = device.compile(kernel);
    stream << shader(in, out).dispatch(1u) << synchronize();

    luisa::vector<float> host_out(n);
    stream << out.copy_to(luisa::span{host_out}) << synchronize();
    luisa::vector<float> expected(n);
    for (auto i = 0u; i < n; i++) { expected[i] = host_in[i] * 2.0f; }
    check_close(host_out, expected, "while_style_loop");
}

void test_nested_while_loops(Device &device) {
    constexpr uint n = 32u;
    auto in = device.create_buffer<float>(n);
    auto out = device.create_buffer<float>(n);
    Stream stream = device.create_stream();

    luisa::vector<float> host_in(n);
    for (auto i = 0u; i < n; i++) { host_in[i] = static_cast<float>(i); }
    stream << in.copy_from(luisa::span{host_in}) << synchronize();

    Kernel1D kernel = [&](BufferFloat in_buf, BufferFloat out_buf) noexcept {
        $ i = 0u;
        $loop {
            $if (i >= 8u) { $break; };
            $ j = 0u;
            $loop {
                $if (j >= 4u) { $break; };
                out_buf.write(i * 4u + j, in_buf.read(i * 4u + j) * 2.0f);
                j += 1u;
            };
            i += 1u;
        };
    };
    auto shader = device.compile(kernel);
    stream << shader(in, out).dispatch(1u) << synchronize();

    luisa::vector<float> host_out(n);
    stream << out.copy_to(luisa::span{host_out}) << synchronize();
    luisa::vector<float> expected(n);
    for (auto i = 0u; i < n; i++) { expected[i] = host_in[i] * 2.0f; }
    check_close(host_out, expected, "nested_while_loops");
}

int main(int argc, char *argv[]) {
    // Loop rotation is opt-in in the SPIR-V pipeline; enable it so these
    // kernels exercise the rotated path end to end.
#ifdef _WIN32
    _putenv_s("LUISA_XIR_ENABLE_LOOP_ROTATION", "1");
#else
    setenv("LUISA_XIR_ENABLE_LOOP_ROTATION", "1", 1);
#endif
    auto dc = luisa::test::create_device_from_ut(argc, argv);
    if (!dc) return 0;
    boost::ut::detail::cfg::parse_arg_with_fallback(
        argc, const_cast<const char **>(argv));
    auto &device = dc->device;
    "dsl_loop_rotation_while_style"_test = [&] {
        test_while_style_loop(device);
    };
    "dsl_loop_rotation_nested_while"_test = [&] {
        test_nested_while_loops(device);
    };
}
