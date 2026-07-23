// Test for the pinned-memory device extension.
// This test covers:
// - Host writes through a write-combined mapped allocation read by a kernel
// - Device writes copied into a host-readable mapped allocation
// - HIP registration, device access, destruction, and re-registration of external host memory

#include "ut/ut.hpp"
#include "test_device.h"
#include <luisa/runtime/context.h>
#include <luisa/runtime/device.h>
#include <luisa/runtime/stream.h>
#include <luisa/dsl/sugar.h>
#include <luisa/backends/ext/pinned_memory_ext.hpp>
#include <array>

using namespace luisa;
using namespace luisa::compute;
using namespace boost::ut;
using namespace boost::ut::literals;

void test_allocated_pinned_memory(Device &device, PinnedMemoryExt *ext) {
    constexpr uint32_t buffer_size = 32u;
    Stream stream = device.create_stream();
    auto upload_buffer = ext->allocate_pinned_memory<uint>(
        buffer_size,
        PinnedMemoryOption{
            .write_combined = true});
    auto default_buffer = device.create_buffer<uint>(buffer_size);
    auto readback_buffer = ext->allocate_pinned_memory<uint>(
        buffer_size,
        PinnedMemoryOption{
            .write_combined = false});
    expect(upload_buffer.native_handle() != nullptr)
        << "upload pinned memory must expose a host pointer";
    expect(readback_buffer.native_handle() != nullptr)
        << "readback pinned memory must expose a host pointer";

    auto upload = static_cast<uint *>(upload_buffer.native_handle());
    for (uint32_t i = 0u; i < buffer_size; i++) { upload[i] = i; }

    auto shader = device.compile<1>([&]() {
        auto i = dispatch_id().x;
        default_buffer->write(i, upload_buffer->read(i) + 256u);
    });
    stream
        << shader().dispatch(buffer_size)
        << readback_buffer.view().copy_from(default_buffer)
        << synchronize();

    auto readback = static_cast<const uint *>(readback_buffer.native_handle());
    for (uint32_t i = 0u; i < buffer_size; i++) {
        expect(readback[i] == i + 256u)
            << "pinned readback mismatch at index " << i
            << ": expected " << i + 256u << ", got " << readback[i];
    }
}

void test_registered_pinned_memory(Device &device, PinnedMemoryExt *ext) {
    constexpr uint32_t buffer_size = 32u;
    alignas(64) std::array<uint, buffer_size> host_memory{};
    Stream stream = device.create_stream();

    // Register the same allocation twice in sequence. The second registration
    // verifies that destroying the first owning Buffer unregistered it.
    for (uint32_t pass = 0u; pass < 2u; pass++) {
        {
            auto [owner, view] = ext->pin_host_memory(
                host_memory.data(), host_memory.size());
            expect(owner.native_handle() == host_memory.data())
                << "registered pinned memory must expose its host pointer";
            auto base = 512u + pass * buffer_size;
            auto shader = device.compile<1>([&]() {
                auto i = dispatch_id().x;
                view->write(i, i + base);
            });
            stream << shader().dispatch(buffer_size) << synchronize();
            for (uint32_t i = 0u; i < buffer_size; i++) {
                expect(host_memory[i] == i + base)
                    << "registered pinned memory mismatch at index " << i;
            }
        }
    }
}

void test_pinned_mem(Device &device) {
    auto ext = device.extension<PinnedMemoryExt>();
    expect(ext != nullptr)
        << "backend '" << device.backend_name()
        << "' must provide PinnedMemoryExt for this test";
    if (ext == nullptr) { return; }
    test_allocated_pinned_memory(device, ext);
    if (device.backend_name() == "hip") {
        test_registered_pinned_memory(device, ext);
    }
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
