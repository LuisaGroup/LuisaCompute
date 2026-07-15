#include "ut/ut.hpp"
#include "test_device.h"
#include <luisa/runtime/context.h>
#include <luisa/runtime/device.h>
#include <luisa/runtime/stream.h>
#include <luisa/dsl/sugar.h>
#include <luisa/backends/ext/pinned_memory_ext.hpp>

using namespace luisa;
using namespace luisa::compute;
using namespace boost::ut;
using namespace boost::ut::literals;

void test_pinned_mem(Device &device) {
    constexpr uint buffer_size = 32;
    Stream stream = device.create_stream();
    auto ext = device.extension<PinnedMemoryExt>();
    // These buffer map memory in host, can directly copy data from host to device, or copy data from device to host.
    Buffer<uint> upload_buffer = ext->allocate_pinned_memory<uint>(
        buffer_size,
        // Use this buffer to upload data from host to device.
        PinnedMemoryOption{
            .write_combined = true});
    Buffer<uint> default_buffer = device.create_buffer<uint>(buffer_size);
    Buffer<uint> readback_buffer = ext->allocate_pinned_memory<uint>(
        buffer_size,
        // Use this buffer to read data from device back to host
        PinnedMemoryOption{
            .write_combined = false});
    auto shader = device.compile<1>([&]() {
        default_buffer->write(dispatch_id().x, upload_buffer->read(dispatch_id().x) + 256);
    });
    vector<uint> data;
    data.reserve(buffer_size);
    for (size_t i = 0; i < buffer_size; ++i) {
        data.emplace_back(i);
    }
    memcpy(upload_buffer.native_handle(), data.data(), luisa::size_bytes(data));
    stream
        << shader().dispatch(buffer_size)
        << readback_buffer.view().copy_from(default_buffer)
        << synchronize();
    expect(true) << "pinned memory test completed";
    memcpy(data.data(), readback_buffer.native_handle(), luisa::size_bytes(data));
    luisa::string result;
    for (auto &i : data) {
        result += std::to_string(i);
        result += " ";
    }
    LUISA_INFO("Result: {}", result);
}

int main(int argc, char *argv[]) {
    auto dc = luisa::test::create_device_from_ut(argc, argv);
    if (!dc) {
        return 0;
    }
    boost::ut::detail::cfg::parse_arg_with_fallback(argc, const_cast<const char **>(argv));

    auto &device = dc->device;
    test_pinned_mem(device);
}
