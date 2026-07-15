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

    // Simple exp test
    auto exp_test = device.compile<1>([](BufferVar<float> buf) {
        buf.write(0u, exp(1.0f));
        buf.write(1u, exp(buf.read(1u)));
    });
    float exp_one, exp_one_from_buf;
    stream << buffer.copy_from(luisa::span{f})
           << exp_test(buffer).dispatch(1)
           << buffer.view(0, 1).copy_to(luisa::span{&exp_one, 1})
           << buffer.view(1, 1).copy_to(luisa::span{&exp_one_from_buf, 1})
           << synchronize();
    LUISA_INFO("exp(1.0) direct={}, exp(1.0) from buf={}", exp_one, exp_one_from_buf);

    // Proper minimal test: dispatch enough workgroups for all threads to pass bounds check
    auto minimal_test = device.compile<1>([](BufferVar<float> input, BufferVar<float> output) {
        set_block_size(32, 1, 1);
        Shared<float> shared_arr(32);
        auto thd_id = thread_id().x;
        auto id = dispatch_id().x;
        shared_arr[thd_id] = input.read(id);
        sync_block();
        $if (thd_id == 0) {
            auto v = shared_arr[0] + shared_arr[1] + shared_arr[2] + shared_arr[3];
            output.write(0u, v);
        };
    });
    float minimal_result;
    auto minimal_input = device.create_buffer<float>(32);
    luisa::vector<float> minimal_data(32);
    for (auto &i : minimal_data) i = 2.7182817f;
    stream << minimal_input.copy_from(luisa::span{minimal_data})
           << minimal_test(minimal_input, buffer).dispatch(32)
           << buffer.view(0, 1).copy_to(luisa::span{&minimal_result, 1})
           << synchronize();
    LUISA_INFO("minimal_test result={}, expected={}", minimal_result, 4.0f * 2.7182817f);

    // Minimal loop test
    auto minimal_loop = device.compile<1>([](BufferVar<float> input, BufferVar<float> output) {
        set_block_size(32, 1, 1);
        Shared<float> shared_arr(32);
        auto thd_id = thread_id().x;
        auto id = dispatch_id().x;
        Float value = input.read(id);
        shared_arr[thd_id] = value;
        sync_block();
        UInt thd_size = 16u;
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
        $if (thd_id == 0) {
            output.write(0u, shared_arr[0]);
        };
    });
    float minimal_loop_result;
    stream << minimal_input.copy_from(luisa::span{minimal_data})
           << minimal_loop(minimal_input, buffer).dispatch(32)
           << buffer.view(0, 1).copy_to(luisa::span{&minimal_loop_result, 1})
           << synchronize();
    LUISA_INFO("minimal_loop result={}, expected={}", minimal_loop_result, 32.0f * 2.7182817f);

    // Large block size loop test
    auto large_loop = device.compile<1>([](BufferVar<float> input, BufferVar<float> output) {
        set_block_size(1024, 1, 1);
        Shared<float> shared_arr(1024);
        auto thd_id = thread_id().x;
        auto id = dispatch_id().x;
        Float value = input.read(id);
        shared_arr[thd_id] = value;
        sync_block();
        UInt thd_size = 512u;
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
        $if (thd_id == 0) {
            output.write(0u, shared_arr[0]);
        };
    });
    float large_loop_result;
    auto large_input = device.create_buffer<float>(1024);
    luisa::vector<float> large_data(1024);
    for (auto &i : large_data) i = 2.7182817f;
    stream << large_input.copy_from(luisa::span{large_data})
           << large_loop(large_input, buffer).dispatch(1024)
           << buffer.view(0, 1).copy_to(luisa::span{&large_loop_result, 1})
           << synchronize();
    LUISA_INFO("large_loop result={}, expected={}", large_loop_result, 1024.0f * 2.7182817f);

    // Test with conditional (like original)
    auto conditional_loop = device.compile<1>([](BufferVar<float> input, BufferVar<float> output, UInt size) {
        set_block_size(1024, 1, 1);
        Shared<float> shared_arr(1024);
        auto thd_id = thread_id().x;
        auto id = dispatch_id().x;
        Float value;
        $if (id < size) {
            value = exp(input.read(id));
        }
        $else {
            value = 0.0f;
        };
        shared_arr[thd_id] = value;
        sync_block();
        UInt thd_size = 512u;
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
        $if (thd_id == 0) {
            output.write(0u, shared_arr[0]);
        };
    });
    float conditional_loop_result;
    luisa::vector<float> ones_data(1024);
    for (auto &i : ones_data) i = 1.0f;
    stream << large_input.copy_from(luisa::span{ones_data})
           << conditional_loop(large_input, buffer, 1024).dispatch(1024)
           << buffer.view(0, 1).copy_to(luisa::span{&conditional_loop_result, 1})
           << synchronize();
    LUISA_INFO("conditional_loop result={}, expected={}", conditional_loop_result, 1024.0f * 2.7182817f);

    // Test with block_id write (like original)
    auto block_id_write = device.compile<1>([](BufferVar<float> input, BufferVar<float> output, UInt size) {
        set_block_size(1024, 1, 1);
        Shared<float> shared_arr(1024);
        auto thd_id = thread_id().x;
        auto id = dispatch_id().x;
        Float value;
        $if (id < size) {
            value = exp(input.read(id));
        }
        $else {
            value = 0.0f;
        };
        shared_arr[thd_id] = value;
        sync_block();
        UInt thd_size = 512u;
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
        $if (thd_id == 0) {
            output.write(block_id().x, shared_arr[0]);
        };
    });
    float block_id_result;
    auto block_output = device.create_buffer<float>(1024);
    stream << large_input.copy_from(luisa::span{ones_data})
           << block_id_write(large_input, block_output, 1024).dispatch(1024)
           << block_output.view(0, 1).copy_to(luisa::span{&block_id_result, 1})
           << synchronize();
    LUISA_INFO("block_id_write result={}, expected={}", block_id_result, 1024.0f * 2.7182817f);

    // Exact replica of original sum_shader
    auto exact_replica = device.compile<1>([](BufferVar<float> input, BufferVar<float> output, UInt size, Bool compute_exp) {
        auto block_size = 1024;
        set_block_size(block_size, 1, 1);
        Shared<float> shared_arr(block_size);
        auto thd_id = thread_id().x;
        Float value;
        auto id = dispatch_id().x;

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

        $if (thd_id == 0) {
            output.write(block_id().x, shared_arr[0]);
        };
    });
    float exact_result;
    stream << large_input.copy_from(luisa::span{ones_data})
           << exact_replica(large_input, block_output, 1024, true).dispatch(1024)
           << block_output.view(0, 1).copy_to(luisa::span{&exact_result, 1})
           << synchronize();
    LUISA_INFO("exact_replica result={}, expected={}", exact_result, 1024.0f * 2.7182817f);

    // Test with 3072 workgroups (like original)
    float exact_result_3072;
    auto large_output = device.create_buffer<float>(3072);
    stream << large_input.copy_from(luisa::span{ones_data})
           << exact_replica(large_input, large_output, 1024, true).dispatch(3072)
           << large_output.view(0, 1).copy_to(luisa::span{&exact_result_3072, 1})
           << synchronize();
    LUISA_INFO("exact_replica_3072 result={}, expected={}", exact_result_3072, 1024.0f * 2.7182817f);

    // Test with same buffer as input and output (like original pass 1b)
    float same_buffer_result;
    auto same_buffer = device.create_buffer<float>(3);
    luisa::vector<float> same_data(3);
    for (auto &i : same_data) i = 2.7182817f;
    stream << same_buffer.copy_from(luisa::span{same_data})
           << exact_replica(same_buffer, same_buffer, 3, false).dispatch(1024)
           << same_buffer.view(0, 1).copy_to(luisa::span{&same_buffer_result, 1})
           << synchronize();
    LUISA_INFO("same_buffer result={}, expected={}", same_buffer_result, 3.0f * 2.7182817f);

    // Execute softmax computation
    float sum;
    float temp0;
    float buf0;
    stream << buffer.copy_from(luisa::span{f})
           // Pass 1: Compute exp(x) and partial sums
           << sum_shader(buffer, temp_buffer, size, true).dispatch(size)
           << temp_buffer.view(0, 1).copy_to(luisa::span{&temp0, 1})
           << buffer.view(0, 1).copy_to(luisa::span{&buf0, 1})
           // Pass 1b: Reduce partial sums to total sum
           << sum_shader(temp_buffer, temp_buffer, temp_buffer.size(), false).dispatch(1024)
           << temp_buffer.view(0, 1).copy_to(luisa::span{&temp0, 1})
           // Pass 2: Normalize
           << final_shader(buffer, temp_buffer).dispatch(size)
           << buffer.view(0, 1).copy_to(luisa::span{&sum, 1})
           << synchronize();
    LUISA_INFO("after_pass1: temp0={}, buf0={}", temp0, buf0);
    LUISA_INFO("after_pass1b: temp0={}", temp0);

    // For uniform input of ones, softmax output should be 1/size
    // sum reads the first element: softmax(1.0) = exp(1.0) / (size * exp(1.0)) = 1/size
    auto expected = 1.0f / static_cast<float>(size);
    LUISA_INFO("sum={}, expected={}", sum, expected);
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
