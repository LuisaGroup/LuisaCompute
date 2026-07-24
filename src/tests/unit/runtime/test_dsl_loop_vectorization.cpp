// Test for XIR loop vectorization via the DSL.
// This test covers:
// - element-wise loop over local arrays (vectorizable)
// - loop with a conditional (rejected or correct)
// - reduction loop over a buffer (stays scalar, still correct)
// - non-unit-stride loop (not vectorized, still correct)
// - trip count not a multiple of VF (vectorized + peeled scalar remainder)
// - reduction over a local array (vectorized with horizontal fold)

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

void test_elementwise_local_arrays(Device &device) {
    constexpr uint n = 16u;
    auto a = device.create_buffer<float>(n);
    auto b = device.create_buffer<float>(n);
    auto c = device.create_buffer<float>(n);
    Stream stream = device.create_stream();
    luisa::vector<float> host_a(n), host_b(n);
    for (auto i = 0u; i < n; i++) {
        host_a[i] = static_cast<float>(i);
        host_b[i] = static_cast<float>(i * 2u);
    }
    stream << a.copy_from(luisa::span{host_a})
           << b.copy_from(luisa::span{host_b})
           << synchronize();

    Kernel1D kernel = [&](BufferFloat in_a, BufferFloat in_b, BufferFloat out_c) noexcept {
        $array<float, 16> va;
        $array<float, 16> vb;
        $array<float, 16> vc;
        $for (i, 16) {
            va[i] = in_a.read(i);
            vb[i] = in_b.read(i);
        };
        $for (i, 16) {
            vc[i] = va[i] + vb[i];
        };
        $for (i, 16) {
            out_c.write(i, vc[i]);
        };
    };
    auto shader = device.compile(kernel);
    stream << shader(a, b, c).dispatch(1u) << synchronize();

    luisa::vector<float> host_c(n);
    stream << c.copy_to(luisa::span{host_c}) << synchronize();
    luisa::vector<float> expected(n);
    for (auto i = 0u; i < n; i++) { expected[i] = host_a[i] + host_b[i]; }
    check_close(host_c, expected, "elementwise_local_arrays");
}

void test_loop_with_conditional(Device &device) {
    constexpr uint n = 16u;
    auto a = device.create_buffer<float>(n);
    auto out = device.create_buffer<float>(n);
    Stream stream = device.create_stream();
    luisa::vector<float> host_a(n);
    for (auto i = 0u; i < n; i++) { host_a[i] = static_cast<float>(i) - 8.0f; }
    stream << a.copy_from(luisa::span{host_a}) << synchronize();

    Kernel1D kernel = [&](BufferFloat in_a, BufferFloat out_buf) noexcept {
        $for (i, 16) {
            auto v = in_a.read(i);
            $if (v > 0.0f) {
                out_buf.write(i, v);
            } $else {
                out_buf.write(i, 0.0f);
            };
        };
    };
    auto shader = device.compile(kernel);
    stream << shader(a, out).dispatch(1u) << synchronize();

    luisa::vector<float> host_out(n);
    stream << out.copy_to(luisa::span{host_out}) << synchronize();
    luisa::vector<float> expected(n);
    for (auto i = 0u; i < n; i++) { expected[i] = std::max(host_a[i], 0.0f); }
    check_close(host_out, expected, "loop_with_conditional");
}

void test_reduction_loop(Device &device) {
    constexpr uint n = 16u;
    auto a = device.create_buffer<float>(n);
    auto out = device.create_buffer<float>(1u);
    Stream stream = device.create_stream();
    luisa::vector<float> host_a(n);
    for (auto i = 0u; i < n; i++) { host_a[i] = static_cast<float>(i + 1u); }
    stream << a.copy_from(luisa::span{host_a}) << synchronize();

    Kernel1D kernel = [&](BufferFloat in_a, BufferFloat out_buf) noexcept {
        $ sum = 0.0f;
        $for (i, 16) {
            sum += in_a.read(i);
        };
        out_buf.write(0u, sum);
    };
    auto shader = device.compile(kernel);
    stream << shader(a, out).dispatch(1u) << synchronize();

    luisa::vector<float> host_out(1u);
    stream << out.copy_to(luisa::span{host_out}) << synchronize();
    check_close(host_out, luisa::vector<float>{136.0f}, "reduction_loop");
}

void test_remainder_local_arrays(Device &device) {
    // trip count 10 is not a multiple of the vector factor (4): the loop is
    // vectorized with a tightened bound and the trailing 2 iterations are
    // peeled as scalar epilogue code
    constexpr uint n = 10u;
    auto a = device.create_buffer<float>(n);
    auto b = device.create_buffer<float>(n);
    auto c = device.create_buffer<float>(n);
    Stream stream = device.create_stream();
    luisa::vector<float> host_a(n), host_b(n);
    for (auto i = 0u; i < n; i++) {
        host_a[i] = static_cast<float>(i);
        host_b[i] = static_cast<float>(i * 3u);
    }
    stream << a.copy_from(luisa::span{host_a})
           << b.copy_from(luisa::span{host_b})
           << synchronize();

    Kernel1D kernel = [&](BufferFloat in_a, BufferFloat in_b, BufferFloat out_c) noexcept {
        $array<float, 10> va;
        $array<float, 10> vb;
        $array<float, 10> vc;
        $for (i, 10) {
            va[i] = in_a.read(i);
            vb[i] = in_b.read(i);
        };
        $for (i, 10) {
            vc[i] = va[i] * vb[i];
        };
        $for (i, 10) {
            out_c.write(i, vc[i]);
        };
    };
    auto shader = device.compile(kernel);
    stream << shader(a, b, c).dispatch(1u) << synchronize();

    luisa::vector<float> host_c(n);
    stream << c.copy_to(luisa::span{host_c}) << synchronize();
    luisa::vector<float> expected(n);
    for (auto i = 0u; i < n; i++) { expected[i] = host_a[i] * host_b[i]; }
    check_close(host_c, expected, "remainder_local_arrays");
}

void test_reduction_local_array(Device &device) {
    // sum reduction over a local array: the per-iteration loaded values are
    // packed 4-wide and folded horizontally into the scalar accumulator
    constexpr uint n = 16u;
    auto a = device.create_buffer<float>(n);
    auto out = device.create_buffer<float>(1u);
    Stream stream = device.create_stream();
    luisa::vector<float> host_a(n);
    for (auto i = 0u; i < n; i++) { host_a[i] = static_cast<float>(i + 1u); }
    stream << a.copy_from(luisa::span{host_a}) << synchronize();

    Kernel1D kernel = [&](BufferFloat in_a, BufferFloat out_buf) noexcept {
        $array<float, 16> va;
        $for (i, 16) {
            va[i] = in_a.read(i);
        };
        $ sum = 0.0f;
        $for (i, 16) {
            sum += va[i];
        };
        out_buf.write(0u, sum);
    };
    auto shader = device.compile(kernel);
    stream << shader(a, out).dispatch(1u) << synchronize();

    luisa::vector<float> host_out(1u);
    stream << out.copy_to(luisa::span{host_out}) << synchronize();
    check_close(host_out, luisa::vector<float>{136.0f}, "reduction_local_array");
}

void test_non_unit_stride(Device &device) {
    constexpr uint n = 32u;
    auto a = device.create_buffer<float>(n);
    auto out = device.create_buffer<float>(16u);
    Stream stream = device.create_stream();
    luisa::vector<float> host_a(n);
    for (auto i = 0u; i < n; i++) { host_a[i] = static_cast<float>(i); }
    stream << a.copy_from(luisa::span{host_a}) << synchronize();

    Kernel1D kernel = [&](BufferFloat in_a, BufferFloat out_buf) noexcept {
        $for (i, 16) {
            out_buf.write(i, in_a.read(i * 2u));
        };
    };
    auto shader = device.compile(kernel);
    stream << shader(a, out).dispatch(1u) << synchronize();

    luisa::vector<float> host_out(16u);
    stream << out.copy_to(luisa::span{host_out}) << synchronize();
    luisa::vector<float> expected(16u);
    for (auto i = 0u; i < 16u; i++) { expected[i] = host_a[i * 2u]; }
    check_close(host_out, expected, "non_unit_stride");
}

int main(int argc, char *argv[]) {
    auto dc = luisa::test::create_device_from_ut(argc, argv);
    if (!dc) return 0;
    boost::ut::detail::cfg::parse_arg_with_fallback(
        argc, const_cast<const char **>(argv));
    auto &device = dc->device;
    "dsl_loop_vectorization_elementwise_local_arrays"_test = [&] {
        test_elementwise_local_arrays(device);
    };
    "dsl_loop_vectorization_loop_with_conditional"_test = [&] {
        test_loop_with_conditional(device);
    };
    "dsl_loop_vectorization_reduction_loop"_test = [&] {
        test_reduction_loop(device);
    };
    "dsl_loop_vectorization_non_unit_stride"_test = [&] {
        test_non_unit_stride(device);
    };
    "dsl_loop_vectorization_remainder_local_arrays"_test = [&] {
        test_remainder_local_arrays(device);
    };
    "dsl_loop_vectorization_reduction_local_array"_test = [&] {
        test_reduction_local_array(device);
    };
}
