#include "ut/ut.hpp"
#include "test_device.h"
// Test for warp-level prefix scan operations.
//
// Prefix scan (parallel prefix sum) is a fundamental parallel primitive
// where each output element is the sum of all previous input elements.
//
// This test demonstrates:
// - warp_prefix_sum: Exclusive or inclusive prefix sum within a warp
// - Conditional execution with warp operations
//
// Example for warp of 8 threads with input [1, 1, 1, 1, 1, 1, 1, 1]:
//   Inclusive prefix sum: [1, 2, 3, 4, 5, 6, 7, 8]
//   Exclusive prefix sum: [0, 1, 2, 3, 4, 5, 6, 7]

#include <luisa/luisa-compute.h>
#include <luisa/dsl/sugar.h>
#include <algorithm>
#include <cmath>

using namespace luisa;
using namespace luisa::compute;
using namespace boost::ut;
using namespace boost::ut::literals;

void test_warp_prefix_scan(Device &device) {
    auto stream = device.create_stream();

    auto warp_size = device.compute_warp_size();
    constexpr auto warp_count = 4u;
    auto element_count = warp_size * warp_count;
    auto block_threads = std::max(32u, warp_size * 2u);
    luisa::vector<float> output(element_count, -1.0f);
    auto output_buffer = device.create_buffer<float>(element_count);

    // Only even lanes participate. warp_prefix_sum is exclusive, so an active
    // lane 2 * n must observe n preceding active values of 0.5.
    Kernel1D kernel = [warp_size, block_threads](BufferFloat output) noexcept {
        set_block_size(block_threads, 1u, 1u);
        set_warp_size(warp_size);
        auto index = dispatch_x();
        // Only threads with even indices execute the scan
        $if (thread_x() % 2u == 0u) {
            output.write(index, warp_prefix_sum(0.5f));
        };
    };
    auto shader = device.compile(kernel);

    stream << output_buffer.copy_from(luisa::span{output})
           << shader(output_buffer).dispatch(element_count)
           << output_buffer.copy_to(luisa::span{output})
           << synchronize();

    for (auto index = 0u; index < element_count; index++) {
        auto lane = index % warp_size;
        auto expected = lane % 2u == 0u ? static_cast<float>(lane / 2u) * 0.5f : -1.0f;
        expect(std::abs(output[index] - expected) < 1e-6f)
            << "warp prefix scan mismatch at index " << index;
    }
}

int main(int argc, char *argv[]) {
    auto dc = luisa::test::create_device_from_ut(argc, argv);
    if (!dc) {
        return 0;
    }
    boost::ut::detail::cfg::parse_arg_with_fallback(argc, const_cast<const char **>(argv));

    auto &device = dc->device;
    test_warp_prefix_scan(device);
}
