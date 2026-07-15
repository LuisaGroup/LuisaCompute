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
#include <luisa/luisa-compute.h>

using namespace luisa;
using namespace luisa::compute;
using namespace boost::ut;
using namespace boost::ut::literals;

// Test function for buffer I/O operations
void test_buffer_io(Device &device) noexcept {

    Stream stream = device.create_stream();

    // Create buffers of different types
    auto buffer0 = device.create_buffer<float>(4);
    auto buffer1 = device.create_buffer<float>(4);
    auto buffer2 = device.create_buffer<float3>(4);

    // Create a view starting at offset 2 with size 2
    auto buffer2view = buffer2.view(2, 2);
    // Create element-wise view for atomic operations
    auto buffer2_element = buffer2.view().as<float>();

    // Kernel to fill buffers with initial values
    auto filler = device.compile<1>([&] {
        auto id = thread_id().x;
        buffer0->write(id, 0.0f);
        buffer1->write(id, 0.0f);
        buffer2->write(id, float3{0.0f});
    });

    // Kernel that performs iterative operations
    auto iteration = device.compile<1>([&](UInt iter, BufferFloat2 result) {
        auto id = dispatch_id().x;
        auto res0 = buffer0->read(id);
        auto res1 = buffer1->read(id);

        // Perform atomic operations on buffer elements
        for (int i = 0; i < 3; i++) {
            buffer2_element->atomic(id * 4 + i).fetch_add(0.0f);
        }

        // Log values for debugging
        device_log("{} : res0 = {}, res1 = {}", id, res0, res1);
        result.write(iter * dispatch_size_x() + id, make_float2(res0, res1));

        // Increment values for next iteration
        buffer0->write(id, res0 + 1.0f);
        buffer1->write(id, res1 + 1.0f);
    });

    // Empty kernel for testing
    auto used = device.compile<1>([&]() {
    });

    // Create result buffer
    auto result_buffer = device.create_buffer<float2>(4 * 2);
    auto result_readback = luisa::vector<float2>(4 * 2);

    // Run multiple frames for verification
    for (size_t frame = 0; frame < 3; frame++) {
        auto cmdlist = CommandList::create();
        // Dispatch iteration kernel multiple times per frame
        for (size_t i = 0; i < 2; i++) {
            cmdlist << iteration(i, result_buffer).dispatch(4);
        }

        stream << cmdlist.commit()
               << used().dispatch(1)
               << result_buffer.copy_to(luisa::span{result_readback})
               << synchronize();

        // Verify results match expected values
        for (auto i = 0u; i < 2u; i++) {
            for (auto j = 0u; j < 4u; j++) {
                auto res = result_readback[i * 4 + j];
                boost::ut::expect(static_cast<bool>((res.x) == (frame * 2 + i)));
                boost::ut::expect(static_cast<bool>((res.y) == (frame * 2 + i)));
            }
        }
    }
}

int main(int argc, char *argv[]) {
    auto dc = luisa::test::create_device_from_ut(argc, argv);
    if (!dc) {
        return 0;
    }
    boost::ut::detail::cfg::parse_arg_with_fallback(argc, const_cast<const char **>(argv));

    auto &device = dc->device;
    test_buffer_io(device);
}
