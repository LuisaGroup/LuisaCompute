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

    constexpr auto element_count = 4u;
    constexpr auto frame_count = 3u;
    constexpr auto iterations_per_frame = 2u;
    Stream stream = device.create_stream();

    // Create buffers of different types
    auto buffer0 = device.create_buffer<float>(element_count);
    auto buffer1 = device.create_buffer<float>(element_count);
    auto buffer2 = device.create_buffer<float3>(element_count);

    // Create a view starting at offset 2 with size 2
    auto buffer2view = buffer2.view(2, 2);
    // Create element-wise view for atomic operations
    auto buffer2_element = buffer2.view().as<float>();
    auto buffer2_element_stride = static_cast<uint>(buffer2.stride() / sizeof(float));

    // Kernel to fill buffers with initial values
    auto filler = device.compile<1>([&] {
        auto id = dispatch_id().x;
        auto value = cast<float>(id);
        buffer0->write(id, value * 3.0f + 1.0f);
        buffer1->write(id, value * 5.0f + 2.0f);
        buffer2->write(id, make_float3(value + 10.0f,
                                       value + 20.0f,
                                       value + 30.0f));
    });

    // Update only the tail through a subview. The full-buffer readback below
    // verifies both the subview offset and its extent.
    auto update_subview = device.compile<1>([](BufferFloat3 tail) noexcept {
        auto id = dispatch_id().x;
        tail.write(id, tail.read(id) + make_float3(100.0f, 200.0f, 300.0f));
    });

    // Kernel that performs iterative operations
    auto iteration = device.compile<1>([&](UInt iter, BufferFloat2 result) {
        auto id = dispatch_id().x;
        auto res0 = buffer0->read(id);
        auto res1 = buffer1->read(id);

        // Perform observable atomic operations on every logical float3
        // component. Skip any ABI padding between float3 elements.
        for (int i = 0; i < 3; i++) {
            buffer2_element->atomic(id * buffer2_element_stride + i).fetch_add(1.0f);
        }

        result.write(iter * dispatch_size_x() + id, make_float2(res0, res1));

        // Increment values for next iteration
        buffer0->write(id, res0 + 1.0f);
        buffer1->write(id, res1 + 1.0f);
    });

    // Create result buffer
    auto result_buffer = device.create_buffer<float2>(element_count * iterations_per_frame);
    auto result_readback = luisa::vector<float2>(element_count * iterations_per_frame);

    stream << filler().dispatch(element_count)
           << update_subview(buffer2view).dispatch(buffer2view.size())
           << synchronize();

    // Run multiple frames for verification
    for (auto frame = 0u; frame < frame_count; frame++) {
        auto cmdlist = CommandList::create();
        // Dispatch iteration kernel multiple times per frame
        for (auto i = 0u; i < iterations_per_frame; i++) {
            cmdlist << iteration(i, result_buffer).dispatch(element_count);
        }

        stream << cmdlist.commit()
               << result_buffer.copy_to(luisa::span{result_readback})
               << synchronize();

        // Verify results match expected values
        for (auto i = 0u; i < iterations_per_frame; i++) {
            for (auto j = 0u; j < element_count; j++) {
                auto res = result_readback[i * element_count + j];
                auto iteration_index = frame * iterations_per_frame + i;
                auto expected0 = static_cast<float>(j * 3u + 1u + iteration_index);
                auto expected1 = static_cast<float>(j * 5u + 2u + iteration_index);
                boost::ut::expect(static_cast<bool>(res.x == expected0))
                    << "buffer0 iteration mismatch at frame " << frame
                    << ", iteration " << i << ", element " << j;
                boost::ut::expect(static_cast<bool>(res.y == expected1))
                    << "buffer1 iteration mismatch at frame " << frame
                    << ", iteration " << i << ", element " << j;
            }
        }
    }

    luisa::vector<float3> buffer2_readback(element_count);
    stream << buffer2.copy_to(luisa::span{buffer2_readback}) << synchronize();
    constexpr auto atomic_increment = static_cast<float>(
        frame_count * iterations_per_frame);
    for (auto i = 0u; i < element_count; i++) {
        auto view_delta = i >= 2u ? make_float3(100.0f, 200.0f, 300.0f) :
                                    make_float3(0.0f);
        auto value = static_cast<float>(i);
        auto expected = make_float3(value + 10.0f,
                                    value + 20.0f,
                                    value + 30.0f) +
                        view_delta + atomic_increment;
        auto actual = buffer2_readback[i];
        boost::ut::expect(static_cast<bool>(all(actual == expected)))
            << "float3 subview/atomic mismatch at element " << i;
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
