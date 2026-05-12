#include "ut/ut.hpp"
#include "test_device.h"
// Test for batched matrix multiplication with neural network training.
//
// This test implements:
// 1. GEMM (General Matrix Multiply) kernel with configurable batch sizes
// 2. A simple 2-layer neural network with backpropagation
// 3. LCG random number generation for weight initialization
//
// The network learns the identity function using gradient descent.

#include <luisa/runtime/context.h>
#include <luisa/runtime/device.h>
#include <luisa/runtime/stream.h>
#include <luisa/runtime/image.h>
#include <luisa/dsl/sugar.h>
#include <luisa/core/clock.h>
#include <luisa/vstl/meta_lib.h>
#include <luisa/vstl/common.h>
#include <fstream>

using namespace luisa;
using namespace luisa::compute;
using namespace boost::ut;
using namespace boost::ut::literals;

// Dispatch pack containing compiled kernel and dispatch dimensions
template<typename T>
struct DispatchPack {
    Kernel3D<void(Buffer<T>, Buffer<T>, Buffer<T>)> kernel;
    uint3 dispatch_size;
};

// GEMM kernel: C = sigmoid(A * B + bias) with batch support
// Supports batched multiplication where each batch can be independent
// or share weights depending on lhs_batch/rhs_batch flags
template<typename T>
DispatchPack<T> gemm_kernel(uint2 lhs_matrix_size, uint2 rhs_matrix_size, uint min_batch_size, bool lhs_batch, bool rhs_batch) {
    using VarType = Var<T>;

    // Determine the common dimension for matrix multiplication
    auto iterate_size = std::min(lhs_matrix_size.x, rhs_matrix_size.y);

    // Configure block size based on batch size for optimal occupancy
    uint3 block_size;
    if (min_batch_size == 1) {
        block_size = make_uint3(8, 8, 1);
    } else if (min_batch_size < 4) {
        block_size = make_uint3(8, 4, min_batch_size);
    } else if (min_batch_size < 8) {
        block_size = make_uint3(4, 4, min_batch_size);
    } else if (min_batch_size < 16) {
        block_size = make_uint3(4, 2, min_batch_size);
    } else if (min_batch_size < 32) {
        block_size = make_uint3(2, 2, min_batch_size);
    } else if (min_batch_size < 64) {
        block_size = make_uint3(2, 1, min_batch_size);
    } else if (min_batch_size < 64ull * 65535ull) {
        block_size = make_uint3(1, 1, 64);
    } else if (min_batch_size < 128ull * 65535ull) {
        block_size = make_uint3(1, 1, 128);
    } else if (min_batch_size < 256ull * 65535ull) {
        block_size = make_uint3(1, 1, 256);
    } else {
        block_size = make_uint3(1, 1, 512);
    }

    // Optimize thread layout based on matrix dimensions
    if (rhs_matrix_size.x < lhs_matrix_size.y) {
        luisa::swap(block_size.x, block_size.y);
    }
    block_size = block_size.zyx();

    // Calculate output matrix dimensions
    uint2 size = make_uint2(rhs_matrix_size.x, lhs_matrix_size.y);
    uint2 lhs_size = make_uint2(lhs_matrix_size.x, size.y);
    uint2 rhs_size = make_uint2(size.x, rhs_matrix_size.y);

    // Define the GEMM kernel
    auto kernel = Kernel3D([=](BufferVar<T> lhs, BufferVar<T> rhs, BufferVar<T> output) {
        // Helper lambdas for buffer access
        auto ReadTex = [&](BufferVar<T> &img, auto &&idx) {
            return img.read(idx);
        };
        auto WriteTex = [&](BufferVar<T> &img, auto &&idx, auto &&value) {
            img.write(idx, value);
        };

        set_block_size(block_size);

        // Get 3D dispatch ID (batch, row, col)
        UInt3 id = dispatch_id().zyx();
        Float r = 0.f;

        // Calculate batch offsets for batched multiplication
        UInt lhs_global_offset = lhs_batch ? ((lhs_matrix_size.x * lhs_matrix_size.y) * id.z) : 0u;
        UInt rhs_gloabal_offset = rhs_batch ? ((rhs_matrix_size.x * rhs_matrix_size.y) * id.z) : 0u;
        UInt output_global_offset = (size.x * size.y) * id.z;

        // Main matrix multiplication loop: accumulate dot product
        for (auto i : dynamic_range(iterate_size)) {
            auto lhs_val = ReadTex(lhs, lhs_global_offset + lhs_size.y * i + id.y);
            auto rhs_val = ReadTex(rhs, rhs_gloabal_offset + rhs_size.y * id.x + i);
            r += Float(lhs_val * rhs_val);
        };

        // Add bias terms if dimensions don't match (bias is stored in extra elements)
        if (lhs_matrix_size.x < rhs_matrix_size.y) {
            for (auto i : dynamic_range(iterate_size, rhs_matrix_size.y)) {
                r += Float(ReadTex(rhs, rhs_gloabal_offset + rhs_size.y * id.x + i));
            }
        } else if (lhs_matrix_size.x > rhs_matrix_size.y) {
            for (auto i : dynamic_range(iterate_size, lhs_matrix_size.x)) {
                r += Float(ReadTex(lhs, lhs_global_offset + lhs_size.y * i + id.y));
            }
        }

        // Apply sigmoid activation: 1 / (1 + exp(-x))
        r = 1.f / (1.f + exp(-r));

        // Write output
        WriteTex(output,
                 output_global_offset + size.y * id.x + id.y,
                 VarType(r));
    });
    return {kernel, make_uint3(min_batch_size, size.yx())};
}

// Get optimal block size for group reduction operations
static uint2 get_proper_dispatch_size(uint group_size) {
    uint2 block_size(1);
    // Find largest power of 2 that divides group_size
    for (uint i = 7; i >= 0; --i) {
        if (group_size % (1 << i) == 0) {
            block_size.y = (1 << i);
            break;
        }
    }
    block_size.x = 128 / block_size.y;
    return block_size;
}

// Linear Congruential Generator (LCG) for random number generation
// Uses the GL parameters: a = 1664525, c = 1013904223
template<typename T>
Kernel1D<void(Buffer<T>, uint, uint)> lcg_kernel() {
    return [](BufferVar<T> b, UInt seed, UInt buffer_size) {
        set_block_size(1024);

        // Tiny Encryption Algorithm (TEA) for seed mixing
        auto get_seed = [](UInt2 v) {
            uint s0 = 0;
            for (uint i = 0; i < 4; ++i) {
                s0 += 0x9e3779b9u;
                v.x += ((v.y << 4) + 0xa341316cu) ^ (v.y + s0) ^ ((v.y >> 5) + 0xc8013ea4u);
                v.y += ((v.x << 4) + 0xad90777du) ^ (v.x + s0) ^ ((v.x >> 5) + 0x7e95761eu);
            }
            return v.x;
        };

        // LCG random number generation
        auto lcg = [](UInt state) {
            constexpr uint lcg_a = 1664525u;
            constexpr uint lcg_c = 1013904223u;
            state = lcg_a * state + lcg_c;
            return cast<float>(state & 0x00ffffffu) *
                   (1.0f / static_cast<float>(0x01000000u));
        };

        // Generate random value for each buffer element
        b.write(dispatch_id().x, Var<T>(lcg(get_seed(make_uint2(seed, dispatch_id().x % buffer_size)))));
    };
}

// Zero-initialize buffer kernel
template<typename T>
Kernel1D<void(Buffer<T>)> zero_kernel() {
    return [](BufferVar<T> b) {
        set_block_size(1024);
        b.write(dispatch_id().x, T(0.f));
    };
}

// Fully connected layer kernel: output = sigmoid(input * weights + bias)
// Supports grouped weights for batched processing
template<typename T>
Kernel2D<void(Buffer<T>, Buffer<T>, Buffer<T>)> fully_connect_kernel(uint start_node_size, uint end_node_size, uint bias_size, bool weight_group) {
    using VarType = Var<T>;
    return [=](BufferVar<T> input_node, BufferVar<T> weight_node, BufferVar<T> output_node) {
        set_block_size(get_proper_dispatch_size(end_node_size));
        auto id = dispatch_id().xy();
        auto start_node_idx = start_node_size * id.x;
        auto weight_colume_size = start_node_size + bias_size;
        Float r = 0.0f;

        // Lambda to fetch weight with optional group offset
        auto get_weight = [&](auto &i) {
            VarType weight;
            auto weight_local_idx = i + weight_colume_size * id.y;
            if (weight_group) {
                weight_local_idx += weight_colume_size * end_node_size * id.x;
            }
            return weight_node.read(weight_local_idx);
        };

        // Compute weighted sum of inputs
        for (auto i : dynamic_range(start_node_size)) {
            auto input = input_node.read(start_node_idx + i);
            r += Float(input) * Float(get_weight(i));
        }

        // Add bias terms
        if (bias_size > 0) {
            for (auto i : dynamic_range(start_node_size, weight_colume_size)) {
                r += Float(get_weight(i));
            }
        }

        // Apply sigmoid activation
        r = 1.f / (1.f + exp(-r));
        output_node.write(end_node_size * id.x + id.y, VarType(r));
    };
}

// Metadata for fully connected layer
struct FullyConnectData {
    size_t start_buffer_size_bytes;
    size_t end_buffer_size_bytes;
    size_t weight_buffer_size_bytes;
    uint2 dispatch_size;
};

// Calculate buffer sizes for fully connected layer
FullyConnectData fully_connect_data(uint group_batch_size, uint start_node_size, uint end_node_size, uint bias_size, bool weight_group) {
    return FullyConnectData{
        .start_buffer_size_bytes = size_t(start_node_size) * size_t(group_batch_size),
        .end_buffer_size_bytes = size_t(end_node_size) * size_t(group_batch_size),
        .weight_buffer_size_bytes = size_t(start_node_size + bias_size) * size_t(end_node_size),
        .dispatch_size = uint2(group_batch_size, end_node_size)};
}

// Parallel reduction sum kernel using shared memory
// Computes sum across batch dimension using binary tree reduction
// (batch_size, buffer_group_size)
Kernel2D<void(Buffer<float>, Buffer<uint>, uint)> sum_kernel() {
    // Atomic float add using compare-exchange loop
    auto float_atomic_add = Callable([](
                                         BufferVar<uint> buffer,
                                         UInt index,
                                         Float value) {
        UInt old = buffer.read(index);
        $while (true) {
            UInt r = buffer.atomic(index).compare_exchange(old, (old.template as<float>() + value).template as<uint>());
            $if (r == old) {
                $break;
            };
            old = r;
        };
    });

    return [=, float_atomic_add = std::move(float_atomic_add)](BufferVar<float> buffer, BufferVar<uint> out_buffer, UInt buffer_size) {
        set_block_size(256);
        Shared<float> shared_arr(256);
        auto id = dispatch_id().xy();
        auto thd_id = thread_id().x;
        auto count = Float(min(dispatch_size().x - (id.x - thd_id), 256u));

        // Load data into shared memory
        $if (id.x < buffer_size) {
            shared_arr[thd_id] = buffer.read(id.x * dispatch_size().y + id.y) / count;
        }
        $else {
            shared_arr[thd_id] = 0;
        };

        // Binary tree reduction in shared memory
        UInt array_count = 128u;
        $while (array_count > 1) {
            sync_block();
            Float local_v;
            $if (thd_id < array_count) {
                local_v = shared_arr[thd_id * 2] + shared_arr[thd_id * 2 + 1];
            };
            sync_block();
            $if (thd_id < array_count) {
                shared_arr[thd_id] = local_v;
            };
            array_count /= 2;
        };

        // Write result using atomic add
        sync_block();
        auto result = shared_arr[0] + shared_arr[1];
        $if (thd_id == 0) {
            float_atomic_add(out_buffer, id.y, result);
        };
    };
}

// Backpropagation kernel for hidden layers
// Computes weight gradients and propagates error backwards
// (batch_size, from + bias)
Kernel2D<void(Buffer<float>, Buffer<float>, Buffer<float>, Buffer<float>, Buffer<float>, float, float)> back_prop(
    uint from_node_size,
    uint to_node_size,
    uint bias_size) {
    return [=](BufferVar<float> layer, BufferVar<float> from_layer_err, BufferVar<float> to_layer_err, BufferVar<float> layer_weight, BufferVar<float> layer_weight_delta, Float mobp, Float rate) {
        uint weight_width = from_node_size + bias_size;
        set_block_size(get_proper_dispatch_size(weight_width));
        uint weight_size = weight_width * to_node_size;
        auto id = dispatch_id().xy();
        auto j = id.y;
        Float z = 0.0f;

        // Update weights and compute backpropagated error
        for (auto i : dynamic_range(to_node_size)) {
            auto weight_idx = weight_size * id.x + j + i * to_node_size;
            auto err = to_layer_err.read(i + to_node_size * id.x);
            auto weight_value = layer_weight.read(weight_idx);
            auto delta = layer_weight_delta.read(weight_idx);

            // Compute gradient: delta = learning_rate * error * activation_derivative
            delta = rate * err;
            $if (j < from_node_size) {
                delta *= layer.read(j + from_node_size * id.x);
                // Accumulate weighted error for backpropagation
                z += err * weight_value;
            };

            // Apply momentum: delta = momentum * prev_delta + new_delta
            delta += mobp * delta;
            weight_value += delta;

            layer_weight_delta.write(weight_idx, delta);
            layer_weight.write(weight_idx, weight_value);
        }

        // Write backpropagated error with sigmoid derivative: err * y * (1 - y)
        $if (j < from_node_size) {
            auto idx = from_node_size * id.x + j;
            auto layer_val = layer.read(idx);
            from_layer_err.write(idx, z * layer_val * (1.f - layer_val));
        };
    };
}

// Backpropagation kernel for input layer (no backpropagation needed)
Kernel2D<void(Buffer<float>, Buffer<float>, Buffer<float>, Buffer<float>, float, float)> first_back_prop(
    uint from_node_size,
    uint to_node_size,
    uint bias_size) {
    return [=](BufferVar<float> layer, BufferVar<float> to_layer_err, BufferVar<float> layer_weight, BufferVar<float> layer_weight_delta, Float mobp, Float rate) {
        uint weight_width = from_node_size + bias_size;
        set_block_size(get_proper_dispatch_size(weight_width));
        uint weight_size = weight_width * to_node_size;
        auto id = dispatch_id().xy();
        auto j = id.y;

        // Update weights only (no error backpropagation)
        for (auto i : dynamic_range(to_node_size)) {
            auto weight_idx = weight_size * id.x + j + i * to_node_size;
            auto err = to_layer_err.read(i + to_node_size * id.x);
            auto weight_value = layer_weight.read(weight_idx);
            auto delta = layer_weight_delta.read(weight_idx);
            delta = rate * err;
            $if (j < from_node_size) {
                delta *= layer.read(j + from_node_size * id.x);
            };
            delta += mobp * delta;
            weight_value += delta;

            layer_weight_delta.write(weight_idx, delta);
            layer_weight.write(weight_idx, weight_value);
        }
    };
}

// Compute error at output layer using squared error derivative
Kernel2D<void(Buffer<float>, Buffer<float>, Buffer<float>)> last_layer_err(uint layer_size) {
    return [=](BufferVar<float> layer, BufferVar<float> layer_err, BufferVar<float> tar) {
        set_block_size(get_proper_dispatch_size(layer_size));
        auto id = dispatch_id().xy();
        auto layer_idx = layer_size * id.x + id.y;
        auto layer_val = layer.read(layer_idx);
        // Error = output * (1 - output) * (target - output) for sigmoid
        layer_err.write(layer_idx, layer_val * (1.f - layer_val) * (tar.read(layer_idx) - layer_val));
    };
}

void test_matrix_multiply(Device &device) {
    auto stream = device.create_stream();

    // Network architecture: 1 input -> 32 hidden -> 1 output
    auto batch_size = 512;
    auto hidden_size = 32;

    // Create buffers for network activations
    auto input_buffer = device.create_buffer<float>(1 * batch_size);
    auto lcg_shader = device.compile(lcg_kernel<float>());
    auto zero_shader = device.compile(zero_kernel<float>());
    auto sum_shader = device.compile(sum_kernel());

    float input_val = 0.5f;
    auto hidden_buffer = device.create_buffer<float>(hidden_size * batch_size);
    auto hidden_error = device.create_buffer<float>(hidden_buffer.size());
    auto out_buffer = device.create_buffer<float>(1 * batch_size);
    auto tar_buffer = device.create_buffer<float>(out_buffer.size());
    auto out_error = device.create_buffer<float>(hidden_buffer.size());

    // Helper to zero-initialize buffers
    auto make_zero = [&](auto &&buffer) {
        auto dispatch_count = (buffer.size() + zero_shader.block_size().x - 1) / zero_shader.block_size().x;
        return zero_shader(buffer).dispatch(buffer.size());
    };

    // Helper to fill buffer with random values
    uint seed = 0;
    auto make_lcg = [&](auto &&buffer) {
        return lcg_shader(buffer, seed++, buffer.size() / batch_size).dispatch(buffer.size());
    };

    // Create weight buffers with bias terms
    auto input_to_hidden_weight = device.create_buffer<float>((input_buffer.size() / batch_size + 1) * hidden_buffer.size() * batch_size);
    auto input_to_hidden_weight_delta = device.create_buffer<float>(input_to_hidden_weight.size());
    auto hidden_to_out_weight = device.create_buffer<float>((hidden_buffer.size() / batch_size + 1) * out_buffer.size() * batch_size);
    auto hidden_to_out_weight_delta = device.create_buffer<float>(hidden_to_out_weight.size());

    // Create GEMM kernels for forward pass
    auto input_hidden_kernel = gemm_kernel<float>(
        uint2(1, 1),
        uint2(hidden_buffer.size() / batch_size, 2),
        batch_size,
        true,
        true);
    auto input_hidden_shader = device.compile(input_hidden_kernel.kernel);

    auto hidden_output_kernel = gemm_kernel<float>(
        uint2(hidden_buffer.size() / batch_size, 1),
        uint2(1, hidden_buffer.size() / batch_size + 1),
        batch_size,
        true,
        true);
    auto hidden_output_shader = device.compile(hidden_output_kernel.kernel);

    // Compile backpropagation kernels
    auto input_hidden_back_prop = device.compile(first_back_prop(1, hidden_size, 1));
    auto hidden_output_back_prop = device.compile(back_prop(hidden_size, 1, 1));
    auto get_last_err = device.compile(last_layer_err(1));

    // Initialize network
    stream << make_zero(input_buffer) << make_zero(hidden_buffer) << make_zero(out_buffer)
           << make_zero(hidden_error) << make_zero(out_error)
           << make_lcg(input_to_hidden_weight) << make_lcg(hidden_to_out_weight)
           << make_zero(input_to_hidden_weight_delta) << make_zero(hidden_to_out_weight_delta) << synchronize();

    // Training loop: train network to learn identity function
    for (int i = 0; i < 5000; ++i) {
        stream << make_lcg(input_buffer)
               << tar_buffer.view().copy_from(input_buffer)
               // Forward pass
               << input_hidden_shader(input_buffer, input_to_hidden_weight, hidden_buffer).dispatch(input_hidden_kernel.dispatch_size)
               << hidden_output_shader(hidden_buffer, hidden_to_out_weight, out_buffer).dispatch(hidden_output_kernel.dispatch_size)
               // Compute output error
               << get_last_err(out_buffer, out_error, tar_buffer).dispatch(batch_size, 1)
               // Backward pass
               << hidden_output_back_prop(hidden_buffer, hidden_error, out_error, hidden_to_out_weight, hidden_to_out_weight_delta, 0.1f, 0.1f).dispatch(batch_size, hidden_size + 1)
               << input_hidden_back_prop(input_buffer, hidden_error, input_to_hidden_weight, input_to_hidden_weight_delta, 0.1f, 0.1f).dispatch(batch_size, 1 + 1);
    }
    stream << synchronize();

    // Test the trained network
    auto final_input_buffer = device.create_buffer<float>(input_buffer.size() / batch_size);
    auto final_hidden_buffer = device.create_buffer<float>(hidden_buffer.size() / batch_size);
    auto final_out_buffer = device.create_buffer<float>(out_buffer.size() / batch_size);
    auto final_input_to_hidden_weight = device.create_buffer<float>(input_to_hidden_weight.size() / batch_size);
    auto final_hidden_to_out_weight = device.create_buffer<float>(hidden_to_out_weight.size() / batch_size);
    float out_val;
    float hidden_weight;

    stream
        // Average weights across batch dimension
        << sum_shader(
               input_to_hidden_weight,
               final_input_to_hidden_weight.view().as<uint>(),
               batch_size)
               .dispatch((batch_size + sum_shader.block_size().x - 1) & (~(sum_shader.block_size().x - 1)), final_input_to_hidden_weight.size())
        << sum_shader(
               hidden_to_out_weight,
               final_hidden_to_out_weight.view().as<uint>(),
               batch_size)
               .dispatch((batch_size + sum_shader.block_size().x - 1) & (~(sum_shader.block_size().x - 1)), final_hidden_to_out_weight.size())
        << final_input_buffer.copy_from(luisa::span{&input_val, 1})
        // Final forward pass with averaged weights
        << hidden_output_shader(final_hidden_buffer, final_hidden_to_out_weight, final_out_buffer).dispatch(make_uint3(1u, hidden_output_kernel.dispatch_size.yz()))
        << final_out_buffer.copy_to(luisa::span{&out_val, 1})
        << final_input_to_hidden_weight.view(1, 1).copy_to(luisa::span{&hidden_weight, 1}) << [hidden_weight]() {
               LUISA_INFO("Final weight {}", hidden_weight);
           }
        << synchronize();
    expect(true) << "matrix multiply completed";
    LUISA_INFO("{}", out_val);
}

int main(int argc, char *argv[]) {
    auto dc = luisa::test::create_device_from_ut(argc, argv);
    if (!dc) {
        return 0;
    }
    boost::ut::detail::cfg::parse_arg_with_fallback(argc, const_cast<const char **>(argv));

    auto &device = dc->device;
    test_matrix_multiply(device);
}
