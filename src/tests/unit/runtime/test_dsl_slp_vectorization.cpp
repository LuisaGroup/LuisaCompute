// Test for XIR SLP vectorization pass via the DSL.
// This test covers:
// - Adjacent scalar stores with isomorphic arithmetic (vectorizable)
// - Adjacent scalar stores with isomorphic unary ops (vectorizable)
// - Mixed ops in a store chain (not vectorizable, but correct)
// - Two independent vectorizable store pairs

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

constexpr uint n = 64u;

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

void test_adjacent_stores_same_arithmetic(Device &device) {
    // Writes x, x+1, x+2, x+3 through a local array with constant indices so
    // the SLP pass sees 4 adjacent scalar stores of isomorphic adds.
    auto in = device.create_buffer<float>(n);
    auto out = device.create_buffer<float>(n);
    Stream stream = device.create_stream();

    luisa::vector<float> host_in(n);
    for (auto i = 0u; i < n; i++) { host_in[i] = static_cast<float>(i) * 0.5f - 8.0f; }
    stream << in.copy_from(luisa::span{host_in}) << synchronize();

    Kernel1D kernel = [&](BufferFloat in_buf, BufferFloat out_buf) noexcept {
        auto tid = dispatch_id().x;
        $array<float, 4> v;
        auto x = in_buf.read(tid);
        v[0] = x + 0.0f;
        v[1] = x + 1.0f;
        v[2] = x + 2.0f;
        v[3] = x + 3.0f;
        $for (i, 4) { out_buf.write(tid * 4 + i, v[i]); };
    };
    auto shader = device.compile(kernel);
    auto out4 = device.create_buffer<float>(n * 4u);
    stream << shader(in, out4).dispatch(n) << synchronize();

    luisa::vector<float> host_out(n * 4u);
    stream << out4.copy_to(luisa::span{host_out}) << synchronize();
    luisa::vector<float> expected(n * 4u);
    for (auto t = 0u; t < n; t++) {
        for (auto i = 0u; i < 4u; i++) {
            expected[t * 4u + i] = host_in[t] + static_cast<float>(i);
        }
    }
    check_close(host_out, expected, "adjacent_stores_same_arithmetic");
}

void test_adjacent_stores_unary_ops(Device &device) {
    auto in = device.create_buffer<float>(n * 4u);
    auto out4 = device.create_buffer<float>(n * 4u);
    Stream stream = device.create_stream();

    luisa::vector<float> host_in(n * 4u);
    for (auto i = 0u; i < n * 4u; i++) { host_in[i] = static_cast<float>(i) - 128.0f; }
    stream << in.copy_from(luisa::span{host_in}) << synchronize();

    Kernel1D kernel = [&](BufferFloat in_buf, BufferFloat out_buf) noexcept {
        auto tid = dispatch_id().x;
        $array<float, 4> v;
        v[0] = abs(in_buf.read(tid * 4 + 0));
        v[1] = abs(in_buf.read(tid * 4 + 1));
        v[2] = abs(in_buf.read(tid * 4 + 2));
        v[3] = abs(in_buf.read(tid * 4 + 3));
        $for (i, 4) { out_buf.write(tid * 4 + i, v[i]); };
    };
    auto shader = device.compile(kernel);
    stream << shader(in, out4).dispatch(n) << synchronize();

    luisa::vector<float> host_out(n * 4u);
    stream << out4.copy_to(luisa::span{host_out}) << synchronize();
    luisa::vector<float> expected(n * 4u);
    for (auto t = 0u; t < n; t++) {
        for (auto i = 0u; i < 4u; i++) {
            expected[t * 4u + i] = std::abs(host_in[t * 4u + i]);
        }
    }
    check_close(host_out, expected, "adjacent_stores_unary_ops");
}

void test_mixed_ops_not_vectorized(Device &device) {
    auto in = device.create_buffer<float>(n);
    auto out4 = device.create_buffer<float>(n * 4u);
    Stream stream = device.create_stream();

    luisa::vector<float> host_in(n);
    for (auto i = 0u; i < n; i++) { host_in[i] = static_cast<float>(i) + 1.0f; }
    stream << in.copy_from(luisa::span{host_in}) << synchronize();

    Kernel1D kernel = [&](BufferFloat in_buf, BufferFloat out_buf) noexcept {
        auto tid = dispatch_id().x;
        $array<float, 4> v;
        v[0] = in_buf.read(tid) + 1.0f;
        v[1] = in_buf.read(tid) * 2.0f;
        v[2] = in_buf.read(tid) - 3.0f;
        v[3] = in_buf.read(tid) / 4.0f;
        $for (i, 4) { out_buf.write(tid * 4 + i, v[i]); };
    };
    auto shader = device.compile(kernel);
    stream << shader(in, out4).dispatch(n) << synchronize();

    luisa::vector<float> host_out(n * 4u);
    stream << out4.copy_to(luisa::span{host_out}) << synchronize();
    luisa::vector<float> expected(n * 4u);
    for (auto t = 0u; t < n; t++) {
        expected[t * 4u + 0] = host_in[t] + 1.0f;
        expected[t * 4u + 1] = host_in[t] * 2.0f;
        expected[t * 4u + 2] = host_in[t] - 3.0f;
        expected[t * 4u + 3] = host_in[t] / 4.0f;
    }
    check_close(host_out, expected, "mixed_ops_not_vectorized");
}

void test_two_vectorizable_pairs(Device &device) {
    auto in = device.create_buffer<float>(n * 2u);
    auto out4 = device.create_buffer<float>(n * 4u);
    Stream stream = device.create_stream();

    luisa::vector<float> host_in(n * 2u);
    for (auto i = 0u; i < n * 2u; i++) { host_in[i] = static_cast<float>(i) * 0.25f; }
    stream << in.copy_from(luisa::span{host_in}) << synchronize();

    Kernel1D kernel = [&](BufferFloat in_buf, BufferFloat out_buf) noexcept {
        auto tid = dispatch_id().x;
        $array<float, 2> a;
        a[0] = in_buf.read(tid * 2 + 0) + 1.0f;
        a[1] = in_buf.read(tid * 2 + 1) + 1.0f;
        $array<float, 2> b;
        b[0] = in_buf.read(tid * 2 + 0) * 2.0f;
        b[1] = in_buf.read(tid * 2 + 1) * 2.0f;
        out_buf.write(tid * 4 + 0, a[0]);
        out_buf.write(tid * 4 + 1, a[1]);
        out_buf.write(tid * 4 + 2, b[0]);
        out_buf.write(tid * 4 + 3, b[1]);
    };
    auto shader = device.compile(kernel);
    stream << shader(in, out4).dispatch(n) << synchronize();

    luisa::vector<float> host_out(n * 4u);
    stream << out4.copy_to(luisa::span{host_out}) << synchronize();
    luisa::vector<float> expected(n * 4u);
    for (auto t = 0u; t < n; t++) {
        auto i0 = t * 2u + 0u;
        auto i1 = t * 2u + 1u;
        expected[t * 4u + 0] = host_in[i0] + 1.0f;
        expected[t * 4u + 1] = host_in[i1] + 1.0f;
        expected[t * 4u + 2] = host_in[i0] * 2.0f;
        expected[t * 4u + 3] = host_in[i1] * 2.0f;
    }
    check_close(host_out, expected, "two_vectorizable_pairs");
}

int main(int argc, char *argv[]) {
    auto dc = luisa::test::create_device_from_ut(argc, argv);
    if (!dc) return 0;
    boost::ut::detail::cfg::parse_arg_with_fallback(
        argc, const_cast<const char **>(argv));
    auto &device = dc->device;
    "dsl_slp_adjacent_stores_same_arithmetic"_test = [&] {
        test_adjacent_stores_same_arithmetic(device);
    };
    "dsl_slp_adjacent_stores_unary_ops"_test = [&] {
        test_adjacent_stores_unary_ops(device);
    };
    "dsl_slp_mixed_ops_not_vectorized"_test = [&] {
        test_mixed_ops_not_vectorized(device);
    };
    "dsl_slp_two_vectorizable_pairs"_test = [&] {
        test_two_vectorizable_pairs(device);
    };
}
