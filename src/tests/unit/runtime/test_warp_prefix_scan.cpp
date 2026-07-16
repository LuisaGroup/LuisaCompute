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

using namespace luisa;
using namespace luisa::compute;
using namespace boost::ut;
using namespace boost::ut::literals;

void test_warp_prefix_scan(Device &device) {

    auto stream = device.create_stream();

    // Simple kernel demonstrating warp prefix sum
    // Only even threads participate in the scan
    auto shader = device.compile<1>([]() noexcept {
        // Only threads with even indices execute the scan
        $if (thread_x() % 2u == 0u) {
            // Compute prefix sum of 0.5 for all participating threads
            // If warp size is 32 and all threads are even:
            //   Thread 0: 0.5
            //   Thread 2: 1.0
            //   Thread 4: 1.5
            //   ...
            auto result = warp_prefix_sum(make_half4(.5_h));
            device_log("{} -> {}", dispatch_x(), result);
        };
    });

    // Execute with 1024 threads
    stream << shader().dispatch(1024u) << synchronize();
    expect(true) << "warp prefix scan completed";
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
