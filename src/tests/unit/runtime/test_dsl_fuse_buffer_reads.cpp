// Test for XIR fuse-consecutive-buffer-reads pass via the DSL.
// This test covers:
// - Consecutive byte-buffer reads fused into a wider vector read
// - Non-consecutive reads (not fused, still correct)
// - Reads separated by a write to the same buffer (not fused, still correct)
// - Two independent adjacent read pairs (two separate fusions)
// - Consecutive byte-buffer writes fused into a wider vector write

#include "ut/ut.hpp"
#include "test_device.h"

#include <luisa/runtime/device.h>
#include <luisa/runtime/stream.h>
#include <luisa/runtime/buffer.h>
#include <luisa/runtime/byte_buffer.h>
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

void test_consecutive_byte_reads(Device &device) {
    auto storage = device.create_buffer<float>(n * 4u);
    auto out = device.create_buffer<float>(n);
    Stream stream = device.create_stream();

    luisa::vector<float> host_in(n * 4u);
    for (auto i = 0u; i < n * 4u; i++) { host_in[i] = static_cast<float>(i) * 0.125f; }
    stream << storage.copy_from(luisa::span{host_in}) << synchronize();

    Kernel1D kernel = [&](ByteBufferVar buf, BufferFloat out_buf) noexcept {
        auto tid = dispatch_id().x;
        auto base = tid * 16u;
        auto v0 = buf.read<float>(base + 0u);
        auto v1 = buf.read<float>(base + 4u);
        auto v2 = buf.read<float>(base + 8u);
        auto v3 = buf.read<float>(base + 12u);
        out_buf.write(tid, v0 + v1 + v2 + v3);
    };
    auto shader = device.compile(kernel);
    stream << shader(ByteBufferView{storage}, out).dispatch(n) << synchronize();

    luisa::vector<float> host_out(n);
    stream << out.copy_to(luisa::span{host_out}) << synchronize();
    luisa::vector<float> expected(n);
    for (auto t = 0u; t < n; t++) {
        expected[t] = host_in[t * 4u + 0u] + host_in[t * 4u + 1u] +
                      host_in[t * 4u + 2u] + host_in[t * 4u + 3u];
    }
    check_close(host_out, expected, "consecutive_byte_reads");
}

void test_non_consecutive_byte_reads(Device &device) {
    auto storage = device.create_buffer<float>(n * 8u);
    auto out = device.create_buffer<float>(n);
    Stream stream = device.create_stream();

    luisa::vector<float> host_in(n * 8u);
    for (auto i = 0u; i < n * 8u; i++) { host_in[i] = static_cast<float>(i) * 0.25f - 3.0f; }
    stream << storage.copy_from(luisa::span{host_in}) << synchronize();

    Kernel1D kernel = [&](ByteBufferVar buf, BufferFloat out_buf) noexcept {
        auto tid = dispatch_id().x;
        auto base = tid * 32u;
        auto v0 = buf.read<float>(base + 0u);
        auto v1 = buf.read<float>(base + 8u);
        auto v2 = buf.read<float>(base + 16u);
        auto v3 = buf.read<float>(base + 24u);
        out_buf.write(tid, v0 + v1 + v2 + v3);
    };
    auto shader = device.compile(kernel);
    stream << shader(ByteBufferView{storage}, out).dispatch(n) << synchronize();

    luisa::vector<float> host_out(n);
    stream << out.copy_to(luisa::span{host_out}) << synchronize();
    luisa::vector<float> expected(n);
    for (auto t = 0u; t < n; t++) {
        expected[t] = host_in[t * 8u + 0u] + host_in[t * 8u + 2u] +
                      host_in[t * 8u + 4u] + host_in[t * 8u + 6u];
    }
    check_close(host_out, expected, "non_consecutive_byte_reads");
}

void test_reads_separated_by_write(Device &device) {
    auto storage = device.create_buffer<float>(n * 4u);
    auto out = device.create_buffer<float>(n);
    Stream stream = device.create_stream();

    luisa::vector<float> host_in(n * 4u, 1.0f);
    stream << storage.copy_from(luisa::span{host_in}) << synchronize();

    Kernel1D kernel = [&](ByteBufferVar buf, BufferFloat out_buf) noexcept {
        auto tid = dispatch_id().x;
        auto base = tid * 16u;
        auto v0 = buf.read<float>(base + 0u);
        buf.write(base + 8u, 5.0f);
        auto v1 = buf.read<float>(base + 4u);
        auto v2 = buf.read<float>(base + 8u);
        out_buf.write(tid, v0 + v1 + v2);
    };
    auto shader = device.compile(kernel);
    stream << shader(ByteBufferView{storage}, out).dispatch(n) << synchronize();

    luisa::vector<float> host_out(n);
    stream << out.copy_to(luisa::span{host_out}) << synchronize();
    luisa::vector<float> expected(n, 1.0f + 1.0f + 5.0f);
    check_close(host_out, expected, "reads_separated_by_write");
}

void test_two_adjacent_read_pairs(Device &device) {
    auto storage = device.create_buffer<float>(n * 8u);
    auto out = device.create_buffer<float>(n);
    Stream stream = device.create_stream();

    luisa::vector<float> host_in(n * 8u);
    for (auto i = 0u; i < n * 8u; i++) { host_in[i] = static_cast<float>(i % 17u); }
    stream << storage.copy_from(luisa::span{host_in}) << synchronize();

    Kernel1D kernel = [&](ByteBufferVar buf, BufferFloat out_buf) noexcept {
        auto tid = dispatch_id().x;
        auto base = tid * 32u;
        auto a0 = buf.read<float>(base + 0u);
        auto a1 = buf.read<float>(base + 4u);
        auto b0 = buf.read<float>(base + 16u);
        auto b1 = buf.read<float>(base + 20u);
        out_buf.write(tid, a0 * b0 + a1 * b1);
    };
    auto shader = device.compile(kernel);
    stream << shader(ByteBufferView{storage}, out).dispatch(n) << synchronize();

    luisa::vector<float> host_out(n);
    stream << out.copy_to(luisa::span{host_out}) << synchronize();
    luisa::vector<float> expected(n);
    for (auto t = 0u; t < n; t++) {
        expected[t] = host_in[t * 8u + 0u] * host_in[t * 8u + 4u] +
                      host_in[t * 8u + 1u] * host_in[t * 8u + 5u];
    }
    check_close(host_out, expected, "two_adjacent_read_pairs");
}

void test_consecutive_byte_writes(Device &device) {
    auto storage = device.create_buffer<float>(n * 4u);
    Stream stream = device.create_stream();

    Kernel1D kernel = [&](ByteBufferVar buf) noexcept {
        auto tid = dispatch_id().x;
        auto base = tid * 16u;
        auto x = cast<float>(tid);
        buf.write(base + 0u, x + 0.0f);
        buf.write(base + 4u, x + 1.0f);
        buf.write(base + 8u, x + 2.0f);
        buf.write(base + 12u, x + 3.0f);
    };
    auto shader = device.compile(kernel);
    stream << shader(ByteBufferView{storage}).dispatch(n) << synchronize();

    luisa::vector<float> host_out(n * 4u);
    stream << storage.copy_to(luisa::span{host_out}) << synchronize();
    luisa::vector<float> expected(n * 4u);
    for (auto t = 0u; t < n; t++) {
        for (auto i = 0u; i < 4u; i++) {
            expected[t * 4u + i] = static_cast<float>(t) + static_cast<float>(i);
        }
    }
    check_close(host_out, expected, "consecutive_byte_writes");
}

int main(int argc, char *argv[]) {
    auto dc = luisa::test::create_device_from_ut(argc, argv);
    if (!dc) return 0;
    boost::ut::detail::cfg::parse_arg_with_fallback(
        argc, const_cast<const char **>(argv));
    auto &device = dc->device;
    "dsl_fuse_consecutive_byte_reads"_test = [&] {
        test_consecutive_byte_reads(device);
    };
    "dsl_fuse_non_consecutive_byte_reads"_test = [&] {
        test_non_consecutive_byte_reads(device);
    };
    "dsl_fuse_reads_separated_by_write"_test = [&] {
        test_reads_separated_by_write(device);
    };
    "dsl_fuse_two_adjacent_read_pairs"_test = [&] {
        test_two_adjacent_read_pairs(device);
    };
    "dsl_fuse_consecutive_byte_writes"_test = [&] {
        test_consecutive_byte_writes(device);
    };
}
