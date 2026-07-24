// Test for XIR loop fusion via the DSL.
// This test covers:
// - two adjacent loops with the same trip count (fused)
// - two adjacent loops with different trip counts (not fused)
// - loops with a write/read dependence (not fused, still correct)

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

void test_adjacent_same_trip_count(Device &device) {
    constexpr uint n = 64u;
    auto a = device.create_buffer<float>(n);
    auto b1 = device.create_buffer<float>(n);
    auto b2 = device.create_buffer<float>(n);
    Stream stream = device.create_stream();
    luisa::vector<float> host_a(n);
    for (auto i = 0u; i < n; i++) { host_a[i] = static_cast<float>(i); }
    stream << a.copy_from(luisa::span{host_a}) << synchronize();

    Kernel1D kernel = [&](BufferFloat in, BufferFloat out1, BufferFloat out2) noexcept {
        $for (i, 64) {
            out1.write(i, in.read(i) * 2.0f);
        };
        $for (i, 64) {
            out2.write(i, in.read(i) + 1.0f);
        };
    };
    auto shader = device.compile(kernel);
    stream << shader(a, b1, b2).dispatch(1u) << synchronize();

    luisa::vector<float> host1(n), host2(n);
    stream << b1.copy_to(luisa::span{host1})
           << b2.copy_to(luisa::span{host2})
           << synchronize();
    luisa::vector<float> expected1(n), expected2(n);
    for (auto i = 0u; i < n; i++) {
        expected1[i] = host_a[i] * 2.0f;
        expected2[i] = host_a[i] + 1.0f;
    }
    check_close(host1, expected1, "adjacent_same_trip_count/out1");
    check_close(host2, expected2, "adjacent_same_trip_count/out2");
}

void test_adjacent_different_trip_counts(Device &device) {
    constexpr uint n = 64u;
    auto a = device.create_buffer<float>(n);
    auto b1 = device.create_buffer<float>(n);
    auto b2 = device.create_buffer<float>(32u);
    Stream stream = device.create_stream();
    luisa::vector<float> host_a(n);
    for (auto i = 0u; i < n; i++) { host_a[i] = static_cast<float>(i); }
    stream << a.copy_from(luisa::span{host_a}) << synchronize();

    Kernel1D kernel = [&](BufferFloat in, BufferFloat out1, BufferFloat out2) noexcept {
        $for (i, 64) {
            out1.write(i, in.read(i) * 2.0f);
        };
        $for (i, 32) {
            out2.write(i, in.read(i) + 1.0f);
        };
    };
    auto shader = device.compile(kernel);
    stream << shader(a, b1, b2).dispatch(1u) << synchronize();

    luisa::vector<float> host1(n), host2(32u);
    stream << b1.copy_to(luisa::span{host1})
           << b2.copy_to(luisa::span{host2})
           << synchronize();
    luisa::vector<float> expected1(n), expected2(32u);
    for (auto i = 0u; i < n; i++) { expected1[i] = host_a[i] * 2.0f; }
    for (auto i = 0u; i < 32u; i++) { expected2[i] = host_a[i] + 1.0f; }
    check_close(host1, expected1, "adjacent_different_trip_counts/out1");
    check_close(host2, expected2, "adjacent_different_trip_counts/out2");
}

void test_dependent_loops_not_fused(Device &device) {
    constexpr uint n = 64u;
    auto temp = device.create_buffer<float>(n);
    auto out = device.create_buffer<float>(n);
    Stream stream = device.create_stream();
    luisa::vector<float> host_in(n, 1.0f);
    stream << temp.copy_from(luisa::span{host_in}) << synchronize();

    Kernel1D kernel = [&](BufferFloat tmp, BufferFloat result) noexcept {
        $for (i, 64) {
            tmp.write(i, cast<float>(i) * 3.0f);
        };
        $for (i, 64) {
            result.write(i, tmp.read(i) + 1.0f);
        };
    };
    auto shader = device.compile(kernel);
    stream << shader(temp, out).dispatch(1u) << synchronize();

    luisa::vector<float> host_out(n);
    stream << out.copy_to(luisa::span{host_out}) << synchronize();
    luisa::vector<float> expected(n);
    for (auto i = 0u; i < n; i++) { expected[i] = static_cast<float>(i) * 3.0f + 1.0f; }
    check_close(host_out, expected, "dependent_loops_not_fused");
}

int main(int argc, char *argv[]) {
    auto dc = luisa::test::create_device_from_ut(argc, argv);
    if (!dc) return 0;
    boost::ut::detail::cfg::parse_arg_with_fallback(
        argc, const_cast<const char **>(argv));
    auto &device = dc->device;
    "dsl_loop_fusion_adjacent_same_trip_count"_test = [&] {
        test_adjacent_same_trip_count(device);
    };
    "dsl_loop_fusion_adjacent_different_trip_counts"_test = [&] {
        test_adjacent_different_trip_counts(device);
    };
    "dsl_loop_fusion_dependent_loops_not_fused"_test = [&] {
        test_dependent_loops_not_fused(device);
    };
}
