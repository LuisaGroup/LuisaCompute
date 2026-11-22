// Test for same-batch dispatch -> readback ordering.
// This test covers:
// - a dispatch that writes a *captured* (bound, not an argument) buffer whose
//   contents are read back inside the same command batch
// - a dispatch that writes a kernel-argument buffer whose contents are read
//   back inside the same command batch
// - interleaved uploads, dispatches and downloads of independent buffers in one
//   batch, i.e. the command layout of the device-math reference test
//
// A backend that defers the copies of a batch must order the readback of a
// buffer against the dispatch that wrote it; otherwise the host reads the
// pre-dispatch (usually zero-initialized) contents of the buffer and every value
// of the dispatch is silently lost.

#include "ut/ut.hpp"
#include "test_device.h"

#include <luisa/core/stl/vector.h>
#include <luisa/runtime/buffer.h>
#include <luisa/runtime/device.h>
#include <luisa/runtime/stream.h>
#include <luisa/dsl/sugar.h>

using namespace luisa;
using namespace luisa::compute;
using namespace boost::ut;
using namespace boost::ut::literals;

namespace {

constexpr auto element_count = 256u;
constexpr auto xor_mask = 0x5a5a5a5au;

[[nodiscard]] luisa::vector<uint> make_input() noexcept {
    luisa::vector<uint> values(element_count);
    for (auto i = 0u; i < element_count; ++i) {
        values[i] = i * 3u + 1u;
    }
    return values;
}

void expect_values(luisa::span<const uint> actual,
                   luisa::span<const uint> expected,
                   const char *description) noexcept {
    expect(eq(actual.size(), expected.size()))
        << description << ": element count mismatch";
    auto matches = true;
    for (auto i = 0u; i < actual.size() && i < expected.size(); ++i) {
        if (actual[i] != expected[i]) {
            LUISA_WARNING("{}: element {} is {} but {} was expected",
                          description, i, actual[i], expected[i]);
            matches = false;
        }
    }
    expect(matches) << description << ": all elements must match";
}

// A dispatch whose output buffer is captured by the kernel (a "bound" argument)
// rather than passed through the shader invocation.
void test_captured_output_readback(Device &device) noexcept {
    auto input = device.create_buffer<uint>(element_count);
    auto output = device.create_buffer<uint>(element_count);
    auto host_input = make_input();
    luisa::vector<uint> host_output(element_count, 0u);

    Kernel1D kernel = [&](BufferUInt src) noexcept {
        auto index = dispatch_id().x;
        output->write(index, src.read(index) ^ xor_mask);
    };
    auto shader = device.compile(kernel);
    Stream stream = device.create_stream();

    luisa::vector<uint> expected(element_count);
    for (auto i = 0u; i < element_count; ++i) {
        expected[i] = host_input[i] ^ xor_mask;
    }

    // Upload, dispatch and download of the same buffer, all in one batch.
    stream << input.copy_from(luisa::span{host_input})
           << shader(input).dispatch(element_count)
           << output.copy_to(luisa::span{host_output})
           << synchronize();

    expect_values(luisa::span{host_output}, luisa::span{expected},
                  "captured-output dispatch readback");
}

// A dispatch whose output is a kernel argument, read back in the same batch.
void test_argument_output_readback(Device &device) noexcept {
    auto input = device.create_buffer<uint>(element_count);
    auto output = device.create_buffer<uint>(element_count);
    auto host_input = make_input();
    luisa::vector<uint> host_output(element_count, 0u);

    Kernel1D kernel = [](BufferUInt src, BufferUInt dst) noexcept {
        auto index = dispatch_id().x;
        dst.write(index, src.read(index) * 2u + 1u);
    };
    auto shader = device.compile(kernel);
    Stream stream = device.create_stream();

    luisa::vector<uint> expected(element_count);
    for (auto i = 0u; i < element_count; ++i) {
        expected[i] = host_input[i] * 2u + 1u;
    }

    stream << input.copy_from(luisa::span{host_input})
           << shader(input, output).dispatch(element_count)
           << output.copy_to(luisa::span{host_output})
           << synchronize();

    expect_values(luisa::span{host_output}, luisa::span{expected},
                  "argument-output dispatch readback");
}

// Several dispatches and readbacks interleaved with uploads in one batch: a
// backend that reorders or defers commands must keep every download after the
// dispatch that produced its contents while still letting independent commands
// overlap.
void test_interleaved_batch(Device &device) noexcept {
    auto input_a = device.create_buffer<uint>(element_count);
    auto input_b = device.create_buffer<uint>(element_count);
    auto output_a = device.create_buffer<uint>(element_count);
    auto output_b = device.create_buffer<uint>(element_count);
    auto output_c = device.create_buffer<uint>(element_count);

    auto host_input_a = make_input();
    luisa::vector<uint> host_input_b(element_count);
    for (auto i = 0u; i < element_count; ++i) {
        host_input_b[i] = i * 7u + 5u;
    }
    luisa::vector<uint> host_output_a(element_count, 0u);
    luisa::vector<uint> host_output_b(element_count, 0u);
    luisa::vector<uint> host_output_c(element_count, 0u);

    Kernel1D masked = [](BufferUInt src, BufferUInt dst) noexcept {
        auto index = dispatch_id().x;
        dst.write(index, src.read(index) ^ xor_mask);
    };
    Kernel1D scaled = [&](BufferUInt src) noexcept {
        auto index = dispatch_id().x;
        output_c->write(index, src.read(index) * 5u);
    };
    auto masked_shader = device.compile(masked);
    auto scaled_shader = device.compile(scaled);
    Stream stream = device.create_stream();

    luisa::vector<uint> expected_a(element_count);
    luisa::vector<uint> expected_b(element_count);
    luisa::vector<uint> expected_c(element_count);
    for (auto i = 0u; i < element_count; ++i) {
        expected_a[i] = host_input_a[i] ^ xor_mask;
        expected_b[i] = host_input_b[i] ^ xor_mask;
        expected_c[i] = expected_b[i] * 5u;
    }

    stream << input_a.copy_from(luisa::span{host_input_a})
           << masked_shader(input_a, output_a).dispatch(element_count)
           << output_a.copy_to(luisa::span{host_output_a})
           << input_b.copy_from(luisa::span{host_input_b})
           << masked_shader(input_b, output_b).dispatch(element_count)
           << output_b.copy_to(luisa::span{host_output_b})
           << scaled_shader(output_b).dispatch(element_count)
           << output_c.copy_to(luisa::span{host_output_c})
           << synchronize();

    expect_values(luisa::span{host_output_a}, luisa::span{expected_a},
                  "interleaved batch: first dispatch");
    expect_values(luisa::span{host_output_b}, luisa::span{expected_b},
                  "interleaved batch: second dispatch");
    expect_values(luisa::span{host_output_c}, luisa::span{expected_c},
                  "interleaved batch: captured-output dispatch");
}

// The command layout of the device-math reference test: the first command of the
// batch dispatches a kernel whose only buffer is captured and writes many
// constant values, its readback is followed by an upload and a second dispatch,
// and both readbacks are deferred to the next dispatch.
void test_single_thread_constant_output(Device &device) noexcept {
    constexpr auto slots = 310u;
    auto output = device.create_buffer<float>(slots);
    auto input = device.create_buffer<float>(4u);
    auto aux_output = device.create_buffer<float>(4u);
    luisa::vector<float> host_output(slots, 0.0f);
    luisa::vector<float> host_input{1.0f, -2.0f, 3.5f, -4.25f};
    luisa::vector<float> host_aux(4u, 0.0f);

    Kernel1D writer = [&] {
        for (auto i = 0u; i < slots; ++i) {
            output->write(i, static_cast<float>(i) + 0.5f);
        }
    };
    Kernel1D aux = [](BufferFloat src, BufferFloat dst) noexcept {
        auto index = dispatch_id().x;
        dst.write(index, src.read(index) * 3.0f);
    };
    auto writer_shader = device.compile(writer);
    auto aux_shader = device.compile(aux);
    Stream stream = device.create_stream();

    luisa::vector<float> expected(slots);
    for (auto i = 0u; i < slots; ++i) {
        expected[i] = static_cast<float>(i) + 0.5f;
    }
    luisa::vector<float> expected_aux(4u);
    for (auto i = 0u; i < 4u; ++i) {
        expected_aux[i] = host_input[i] * 3.0f;
    }

    stream << writer_shader().dispatch(1u)
           << output.copy_to(luisa::span{host_output})
           << input.copy_from(luisa::span{host_input})
           << aux_shader(input, aux_output).dispatch(4u)
           << aux_output.copy_to(luisa::span{host_aux})
           << synchronize();

    auto matches = true;
    for (auto i = 0u; i < slots; ++i) {
        if (host_output[i] != expected[i]) {
            LUISA_WARNING("single-thread constant output: element {} is {} but {} was expected",
                          i, host_output[i], expected[i]);
            matches = false;
        }
    }
    expect(matches) << "single-thread constant output: all elements must match";
    auto aux_matches = true;
    for (auto i = 0u; i < 4u; ++i) {
        if (host_aux[i] != expected_aux[i]) {
            LUISA_WARNING("deferred auxiliary dispatch: element {} is {} but {} was expected",
                          i, host_aux[i], expected_aux[i]);
            aux_matches = false;
        }
    }
    expect(aux_matches) << "deferred auxiliary dispatch: all elements must match";
}

}// namespace

int main(int argc, char *argv[]) {
    auto dc = luisa::test::create_device_from_ut(argc, argv);
    if (!dc) { return 0; }
    boost::ut::detail::cfg::parse_arg_with_fallback(
        argc, const_cast<const char **>(argv));
    auto &device = dc->device;

    "batch_readback_captured_output"_test = [&] {
        test_captured_output_readback(device);
    };
    "batch_readback_argument_output"_test = [&] {
        test_argument_output_readback(device);
    };
    "batch_readback_interleaved"_test = [&] {
        test_interleaved_batch(device);
    };
    "batch_readback_single_thread_constant_output"_test = [&] {
        test_single_thread_constant_output(device);
    };
}
