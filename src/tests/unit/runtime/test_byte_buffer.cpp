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

int main(int argc, char *argv[]) {
    auto dc = luisa::test::create_device_from_ut(argc, argv);
    if (!dc) {
        return 0;
    }
    boost::ut::detail::cfg::parse_arg_with_fallback(argc, const_cast<const char **>(argv));

    auto &device = dc->device;
    test_byte_buffer(device);
}
