// Test for Cluster Launch Control (CUDA Blackwell, SM 10.0+).
// This test covers:
// - Basic work-stealing loop correctness
// - Cancellation result query
// - Multiple work items per block
// - Edge case: more work items than blocks

#include "ut/ut.hpp"
#include "test_device.h"

#include <luisa/runtime/device.h>
#include <luisa/runtime/stream.h>
#include <luisa/runtime/buffer.h>
#include <luisa/dsl/sugar.h>
#include <luisa/dsl/cluster_launch_control.h>

using namespace luisa;
using namespace luisa::compute;
using namespace boost::ut;
using namespace boost::ut::literals;

void test_clc_vector_scalar_multiply(Device &device) {
    constexpr auto N = 1024 * 1024u;
    constexpr auto block_size = 256u;
    constexpr auto num_blocks = 64u;

    luisa::vector<float> host(N);
    for (auto i = 0u; i < N; ++i) {
        host[i] = static_cast<float>(i);
    }

    Buffer<float> buf = device.create_buffer<float>(N);
    Stream stream = device.create_stream();
    stream << buf.copy_from(luisa::span{host}) << synchronize();

    Kernel1D kernel = [&](BufferFloat b, UInt n) noexcept {
        set_block_size(block_size, 1u, 1u);

        Shared<uint4> result{1};
        Shared<ulong> bar{1};
        $int phase = 0;

        // Initialize barrier from thread 0
        $if (thread_x() == 0) {
            mbarrier_init(bar, 1u);
        };
        sync_block();

        // Shared scalar multiplier
        $shared<float> alpha{1};
        $if (thread_x() == 0) {
            alpha[0] = 2.0f;
        };
        sync_block();

        $float a = alpha[0];
        $int bx = block_x();

        $loop {
            sync_block();

            // Cancellation request from thread 0
            $if (thread_x() == 0) {
                fence_proxy_async_acquire();
                clc_try_cancel(result, bar);
                mbarrier_arrive_expect_tx(bar, 16u); // sizeof(uint4)
            };

            // Compute
            auto i = bx * block_size + thread_x();
            $if (i < n) {
                b.write(i, b.read(i) * a);
            };

            // Wait for cancellation
            $while (!mbarrier_try_wait_parity(bar, phase)) {};
            phase = phase ^ 1;

            // Check result
            $if (!clc_query_is_canceled(result)) {
                $break;
            };

            // Get next block index
            bx = clc_query_get_ctaid_x(result);

            fence_proxy_async_release();
        };
    };

    auto shader = device.compile(kernel);
    stream << shader(buf, N).dispatch(num_blocks) << synchronize();

    luisa::vector<float> result(N);
    stream << buf.copy_to(luisa::span{result}) << synchronize();

    bool all_correct = true;
    for (auto i = 0u; i < N; ++i) {
        auto expected = static_cast<float>(i) * 2.0f;
        if (std::abs(result[i] - expected) > 1e-4f) {
            LUISA_WARNING("Mismatch at [{}]: got {} expected {}", i, result[i], expected);
            all_correct = false;
            break;
        }
    }
    expect(all_correct) << "all elements must be multiplied by 2.0f";
}

void test_clc_work_stealing_helper(Device &device) {
    constexpr auto N = 512u;
    constexpr auto block_size = 64u;
    constexpr auto num_blocks = 4u; // fewer blocks than needed for work stealing

    luisa::vector<float> host(N);
    for (auto i = 0u; i < N; ++i) {
        host[i] = static_cast<float>(i);
    }

    Buffer<float> buf = device.create_buffer<float>(N);
    Stream stream = device.create_stream();
    stream << buf.copy_from(luisa::span{host}) << synchronize();

    Kernel1D kernel = [&](BufferFloat b, UInt n) noexcept {
        set_block_size(block_size, 1u, 1u);

        cluster_launch_control_work_stealing_1d([&](Int bx) noexcept {
            auto i = bx * block_size + thread_x();
            $if (i < n) {
                b.write(i, b.read(i) * 3.0f);
            };
        });
    };

    auto shader = device.compile(kernel);
    stream << shader(buf, N).dispatch(num_blocks) << synchronize();

    luisa::vector<float> result(N);
    stream << buf.copy_to(luisa::span{result}) << synchronize();

    bool all_correct = true;
    for (auto i = 0u; i < N; ++i) {
        auto expected = static_cast<float>(i) * 3.0f;
        if (std::abs(result[i] - expected) > 1e-4f) {
            LUISA_WARNING("Mismatch at [{}]: got {} expected {}", i, result[i], expected);
            all_correct = false;
            break;
        }
    }
    expect(all_correct) << "work-stealing helper must process all elements";
}

static inline const auto reg = [] {
    "clc_vector_scalar_multiply"_test = [] {
        auto dc = luisa::test::create_device_from_ut();
        if (!dc) return;
        test_clc_vector_scalar_multiply(dc->device);
    };
    "clc_work_stealing_helper"_test = [] {
        auto dc = luisa::test::create_device_from_ut();
        if (!dc) return;
        test_clc_work_stealing_helper(dc->device);
    };
    return 0;
}();

int main() {}
