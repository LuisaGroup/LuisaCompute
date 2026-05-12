// Test for softmax computation using parallel reduction.
//
// This test implements softmax normalization: softmax(x_i) = exp(x_i) / sum(exp(x_j))
// Using a two-pass algorithm with parallel reduction for the sum.
//
// Two implementations are provided:
// 1. Batch softmax: For large arrays that don't fit in a single block
// 2. Single-pass softmax: For smaller arrays using shared memory reduction

#include "ut/ut.hpp"
#include "test_device.h"
#include <luisa/runtime/context.h>
#include <luisa/runtime/device.h>
#include <luisa/runtime/stream.h>
#include <luisa/runtime/image.h>
#include <luisa/dsl/sugar.h>
#include <luisa/dsl/shared.h>
#include <luisa/core/clock.h>
#include <luisa/vstl/meta_lib.h>
#include <luisa/vstl/common.h>

using namespace luisa;
using namespace luisa::compute;
using namespace boost::ut;
using namespace boost::ut::literals;

// Dispatch pack for single-pass softmax
template<typename T>
struct DispatchPack {
    Kernel1D<void(Buffer<T>)> kernel;
    uint dispatch_size;
};

// Batch dispatch pack for two-pass softmax
template<typename T>
struct BatchDispatchPack {
    Kernel1D<void(Buffer<T>, Buffer<T>, uint, bool)> calc_sum;// First pass: compute partial sums
    Kernel1D<void(Buffer<T>, Buffer<T>)> final;               // Second pass: normalize
};

// Batch softmax kernel for large arrays
// Uses two-pass approach:
//   Pass 1: Compute exp(x) and partial sums using parallel reduction
//   Pass 2: Divide each exp(x) by total sum
template<typename T>
BatchDispatchPack<T> batch_softmax_kernel(uint2 size) {
    // First pass: compute exp and partial sums
    auto batch = Kernel1D([=](BufferVar<T> input, BufferVar<T> output, UInt size, Bool compute_exp) {
        auto block_size = 1024;
        set_block_size(block_size, 1, 1);
        Shared<float> shared_arr(block_size);
        auto thd_id = thread_id().x;
        Float value;
        auto id = dispatch_id().x;

        // Load and compute exp(x) or just x
        $if (id < size) {
            $if (compute_exp) {
                value = exp(Float(input.read(id)));
            }
            $else {
                value = Float(input.read(id));
            };
        }
        $else {
            value = 0.0f;
        };
        shared_arr[thd_id] = value;

        // Parallel reduction: sum all values in block
        UInt thd_size = block_size / 2u;
        sync_block();
        $while (thd_size > 0) {
            $if (thd_id < thd_size) {
                value = shared_arr[thd_id * 2] + shared_arr[thd_id * 2 + 1];
            };
            sync_block();
            $if (thd_id < thd_size) {
                shared_arr[thd_id] = value;
            };
            thd_size /= 2u;
            sync_block();
        };

        // Write block sum to output
        $if (thd_id == 0) {
            output.write(block_id().x, shared_arr[0]);
        };
    });

    // Second pass: normalize by total sum
    auto final = Kernel1D([=](BufferVar<T> buffer, BufferVar<T> sum_buffer) {
        auto id = dispatch_id().x;
        buffer.write(id, exp(buffer.read(id)) / sum_buffer.read(0u));
    });

    return BatchDispatchPack<T>{
        std::move(batch),
        std::move(final)};
}

// Single-pass softmax for smaller arrays that fit in shared memory
template<typename T>
DispatchPack<T> softmax_kernel(uint2 size) {
    // Validate size constraints
    if (size.x > 1024) {
        LUISA_ERROR("Softmax size can not be larger than 2048");
    }
    if (any(size == 0u)) {
        LUISA_ERROR("Softmax size can not be 0");
    }

    // Round up to next power of 2 for efficient reduction
    auto block_size = next_pow2(size.x);
    block_size = std::max<uint>(block_size, 32u);

    auto kernel = Kernel1D([=](BufferVar<T> input) {
        set_block_size(block_size, 1, 1);
        Shared<float> shared_arr(block_size);
        auto thd_id = thread_id().x;
        Float value;
        auto id = dispatch_id().x;

        // Load and compute exp(x), padding with 0 if out of bounds
        $if (id < size.x) {
            value = exp(Float(input.read(id)));
        }
        $else {
            value = 0.0f;
        };
        shared_arr[thd_id] = value;

        // Parallel reduction to compute sum of exp(x)
        UInt thd_size = block_size / 2u;
        sync_block();
        $while (thd_size > 0) {
            $if (thd_id < thd_size) {
                value = shared_arr[thd_id * 2] + shared_arr[thd_id * 2 + 1];
            };
            sync_block();
            $if (thd_id < thd_size) {
                shared_arr[thd_id] = value;
            };
            thd_size /= 2u;
            sync_block();
        };

        // Normalize and write output
        $if (id < size.x) {
            auto write_id = id;
            input.write(write_id, (exp(Float(input.read(write_id))) / shared_arr[0]).template cast<T>());
        };
    });

    return DispatchPack{
        .kernel = std::move(kernel),
        .dispatch_size = (size.x + block_size - 1u) & (~(block_size - 1u))};
}

void test_softmax(Device &device) {
    auto stream = device.create_stream();

    // Test with array larger than block size
    const auto size = 1024 * 3;
    auto pack = batch_softmax_kernel<float>(uint2(size, 1));
    auto sum_shader = device.compile(pack.calc_sum);
    auto final_shader = device.compile(pack.final);
    auto buffer = device.create_buffer<float>(size);
    auto temp_buffer = device.create_buffer<float>(size / 1024);

    // Initialize input with all ones (softmax should produce uniform distribution)
    luisa::vector<float> f(size);
    for (auto &i : f) {
        i = 1.0f;
    }

    // Execute softmax computation
    float sum;
    stream << buffer.copy_from(luisa::span{f})
           // Pass 1: Compute exp(x) and partial sums
           << sum_shader(buffer, temp_buffer, size, true).dispatch(size)
           // Pass 1b: Reduce partial sums to total sum
           << sum_shader(temp_buffer, temp_buffer, temp_buffer.size(), false).dispatch(1024)
           // Pass 2: Normalize
           << final_shader(buffer, temp_buffer).dispatch(size)
           << buffer.view(0, 1).copy_to(luisa::span{&sum, 1})
           << synchronize();

    // For uniform input of ones, softmax output should be 1/size
    // sum reads the first element: softmax(1.0) = exp(1.0) / (size * exp(1.0)) = 1/size
    auto expected = 1.0f / static_cast<float>(size);
    expect(std::abs(sum - expected) < 1e-5f) << "softmax_uniform_distribution";
}

int main(int argc, char *argv[]) {
    auto dc = luisa::test::create_device_from_ut(argc, argv);
    if (!dc) {
        return 0;
    }
    boost::ut::detail::cfg::parse_arg_with_fallback(argc, const_cast<const char **>(argv));

    auto &device = dc->device;
    test_softmax(device);
}
