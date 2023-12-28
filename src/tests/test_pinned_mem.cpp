#include <luisa/runtime/context.h>
#include <luisa/runtime/device.h>
#include <luisa/runtime/stream.h>
#include <luisa/dsl/sugar.h>
#include <luisa/backends/ext/pinned_memory_ext.hpp>
using namespace luisa;
using namespace luisa::compute;

int main(int argc, char *argv[]) {
    Context context{argv[0]};
    if (argc <= 1) { exit(1); }
    constexpr uint buffer_size = 32;
    Device device = context.create_device(argv[1]);
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
    memcpy(upload_buffer.native_handle(), data.data(), data.size_bytes());
    stream
        << shader().dispatch(buffer_size)
        << readback_buffer.copy_from(default_buffer)
        << synchronize();
    memcpy(data.data(), readback_buffer.native_handle(), data.size_bytes());
    luisa::string result;
    for(auto & i : data){
        result += std::to_string(i);
        result += " ";
    }
    LUISA_INFO("Result: {}", result);
}