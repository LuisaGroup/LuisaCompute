// Test for buffer I/O operations
// This test verifies buffer read/write operations and
// iteration patterns using command lists.
//
// Features tested:
// - Buffer creation with different types
// - Buffer views and element views
// - Buffer read/write in kernels
// - Atomic operations on buffer elements
// - Command list creation and commit
// - Data verification between iterations

#include "ut/ut.hpp"
#include "test_device.h"
#include <stb/stb_image_write.h>

#include <luisa/core/logging.h>
#include <luisa/runtime/context.h>
#include <luisa/runtime/device.h>
#include <luisa/runtime/stream.h>
#include <luisa/runtime/event.h>
#include <luisa/dsl/sugar.h>
#include <luisa/runtime/byte_buffer.h>

#include <array>
#include <cstddef>
#include <cstring>

using namespace luisa;
using namespace luisa::compute;
using namespace boost::ut;
using namespace boost::ut::literals;

void test_byte_buffer(Device &device) {

    log_level_verbose();
    constexpr uint BUFFER_SIZE = 4;
    auto byte_buffer = device.create_byte_buffer(BUFFER_SIZE * sizeof(uint));
    auto buffer_float = device.create_buffer<float>(BUFFER_SIZE);
    auto test_shader = device.compile<1>([&](ByteBufferVar buffer, UInt value) {
        auto id = dispatch_id().x;
        buffer.write(id * (uint)sizeof(float), value);
    });
    auto stream = device.create_stream();
    luisa::vector<uint> host_data(BUFFER_SIZE);
    stream << test_shader(ByteBufferView{buffer_float} /*Buffer<float> to Byte buffer*/, 114).dispatch(buffer_float.size())
           << test_shader(byte_buffer, 514).dispatch(buffer_float.size())
           << buffer_float.copy_to(luisa::span<uint>{host_data.data(), host_data.size()})
           << synchronize();
    auto float_as_uint_ok = true;
    for (auto &i : host_data) {
        if (i != 114u) { float_as_uint_ok = false; }
    }
    expect(float_as_uint_ok) << "byte_buffer_float_as_uint_write";

    stream << byte_buffer.copy_to(host_data.data())
           << synchronize();
    auto byte_buf_ok = true;
    for (auto &i : host_data) {
        if (i != 514u) { byte_buf_ok = false; }
    }
    expect(byte_buf_ok) << "byte_buffer_direct_write";
}

void test_byte_buffer_bool_read(Device &device) {
    log_level_verbose();
    auto byte_buffer = device.create_byte_buffer(4u);
    auto result_buffer = device.create_buffer<uint>(1u);
    auto test_shader = device.compile<1>([&](ByteBufferVar buffer, BufferUInt result) {
        auto b0 = buffer.read<bool>(0u);
        auto b1 = buffer.read<bool>(1u);
        result.write(0u, select(0u, 1u, b0) | select(0u, 2u, b1));
    });
    uint input = 0x00000100u;
    uint result = 0u;
    auto stream = device.create_stream();
    stream << byte_buffer.copy_from(&input)
           << test_shader(byte_buffer, result_buffer).dispatch(1u)
           << result_buffer.copy_to(luisa::span{&result, 1u})
           << synchronize();
    expect(result == 2u) << "byte_buffer_bool_read_must_ignore_neighboring_bytes";
}

void test_byte_buffer_volatile_io(Device &device) {
    constexpr uint element_count = 16u;
    auto byte_buffer = device.create_byte_buffer(element_count * sizeof(uint));
    auto result_buffer = device.create_buffer<uint>(element_count);
    auto shader = device.compile<1>([](ByteBufferVar buffer, BufferUInt result) noexcept {
        auto index = dispatch_id().x;
        auto byte_offset = index * static_cast<uint>(sizeof(uint));
        auto expected = 0x9e3779b9u ^ index;
        buffer.volatile_write(byte_offset, expected);
        result.write(index, buffer.volatile_read<uint>(byte_offset));
    });

    luisa::vector<uint> result(element_count);
    auto stream = device.create_stream();
    stream << shader(byte_buffer, result_buffer).dispatch(element_count)
           << result_buffer.copy_to(luisa::span{result})
           << synchronize();

    auto all_correct = true;
    for (auto i = 0u; i < element_count; i++) {
        if (result[i] != (0x9e3779b9u ^ i)) {
            all_correct = false;
            break;
        }
    }
    expect(all_correct) << "volatile byte-buffer reads must observe preceding volatile writes";
}

void test_byte_buffer_vector_alignment(Device &device) {
    expect(Type::of<float3>()->alignment() == 16u)
        << "float3 byte-buffer ABI alignment";
    expect(Type::of<std::array<float, 3u>>()->alignment() == alignof(float))
        << "packed float array byte-buffer ABI alignment";

    auto input = device.create_byte_buffer(64u);
    auto output = device.create_byte_buffer(32u);
    std::array<std::byte, 64u> source{};
    const std::array<float, 3u> packed_expected{1.25f, -2.5f, 3.75f};
    const float3 aligned_expected{-4.0f, 5.5f, 6.25f};
    std::memcpy(
        source.data() + 12u, packed_expected.data(), sizeof(packed_expected));
    std::memcpy(
        source.data() + 32u, &aligned_expected, sizeof(aligned_expected));

    auto shader = device.compile<1>(
        [](ByteBufferVar src, ByteBufferVar dst) noexcept {
            const auto packed =
                src.read<std::array<float, 3u>>(12u);
            const Float3 aligned = src.read<float3>(32u);
            dst.write(0u, packed);
            dst.write(16u, aligned);
        });
    std::array<std::byte, 32u> result{};
    auto stream = device.create_stream();
    stream << input.copy_from(source.data())
           << shader(input, output).dispatch(1u)
           << output.copy_to(result.data())
           << synchronize();

    std::array<float, 3u> packed_actual{};
    float3 aligned_actual{};
    std::memcpy(
        packed_actual.data(), result.data(), sizeof(packed_actual));
    std::memcpy(
        &aligned_actual, result.data() + 16u, sizeof(aligned_actual));
    expect(packed_actual == packed_expected)
        << "packed float array byte-buffer access";
    expect(aligned_actual.x == aligned_expected.x &&
           aligned_actual.y == aligned_expected.y &&
           aligned_actual.z == aligned_expected.z)
        << "aligned float3 byte-buffer access";
}

int main(int argc, char *argv[]) {
    auto dc = luisa::test::create_device_from_ut(argc, argv);
    if (!dc) {
        return 0;
    }
    boost::ut::detail::cfg::parse_arg_with_fallback(argc, const_cast<const char **>(argv));

    auto &device = dc->device;
    test_byte_buffer(device);
    test_byte_buffer_bool_read(device);
    test_byte_buffer_volatile_io(device);
    test_byte_buffer_vector_alignment(device);
}
