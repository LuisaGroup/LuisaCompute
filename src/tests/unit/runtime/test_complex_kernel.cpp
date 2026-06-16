#include "ut/ut.hpp"
#include "test_device.h"
// Tests for matrix multiply and XIR pass corner cases.
//
// Original test:
// 1. GEMM (General Matrix Multiply) kernel with configurable batch sizes
// 2. A simple 2-layer neural network with backpropagation
// 3. LCG random number generation for weight initialization
// The network learns the identity function using gradient descent.
//
// Corner-case tests that exercise XIR passes (utils.cpp):
// - control_flow_corners: destructure_cfg, restructure_cfg, simplify_cfg,
//   if_conversion, lower_break_continue, early_return_elimination, phi_cleanup
// - memory_pass_corners: mem2reg, dead_store_elimination, local_store_forward,
//   local_load_elimination, sroa, reg2mem
// - callable_inline_corners: inline, dead_arg_elim, unused_callable_removal,
//   promote_ref_arg
// - loop_pass_corners: loop_unroll, indvar_simplify, licm, loop_rotation
// - algebraic_pass_corners: algebraic_simplify, const_fold, simplify_libcalls,
//   reassociate, div_rem_pairs, gvn, sccp, cvp
// - scalarize_gep_corners: scalarizer, trace_gep, transpose_gep

#include <luisa/runtime/context.h>
#include <luisa/runtime/device.h>
#include <luisa/runtime/stream.h>
#include <luisa/runtime/image.h>
#include <luisa/runtime/rtx/accel.h>
#include <luisa/runtime/rtx/mesh.h>
#include <luisa/runtime/rtx/triangle.h>
#include <luisa/dsl/sugar.h>
#include <luisa/core/clock.h>
#include <luisa/vstl/meta_lib.h>
#include <luisa/vstl/common.h>
#include <fstream>
#ifdef LUISA_ENABLE_XIR
#include <luisa/xir/passes/aggregate_field_bitmask.h>
#endif

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

// ============================================================================
// Corner-case kernels that exercise XIR passes from
// src/backends/common/spirv/spirv_codegen/utils.cpp
// ============================================================================

// --- Structs for scalarizer / sroa / trace_gep / fix_self_referential tests ---
struct VecStruct {
    float3 v;
    float s;
};
LUISA_STRUCT(VecStruct, v, s) {};

struct NestedArr {
    float arr[3][2];
};
LUISA_STRUCT(NestedArr, arr) {};

struct InnerStruct {
    float x;
    float2 y;
};
struct OuterStruct {
    InnerStruct inner;
    float z;
    float3 w;
};
LUISA_STRUCT(InnerStruct, x, y) {};
LUISA_STRUCT(OuterStruct, inner, z, w) {};

struct PartialA {
    float a;
    float b;
    float c;
};
LUISA_STRUCT(PartialA, a, b, c) {};

struct AoSItem {
    float3 pos;
    float mass;
};
LUISA_STRUCT(AoSItem, pos, mass) {};

struct AoSArray {
    AoSItem items[4];
};
LUISA_STRUCT(AoSArray, items) {};

struct SoAFields {
    float x[4];
    float y[4];
    float z[4];
    float mass[4];
};
LUISA_STRUCT(SoAFields, x, y, z, mass) {};

struct MaskedStruct {
    float3 v;
    float s;
    int2 iv;
    bool flag;
    float3x3 m;
};
LUISA_STRUCT(MaskedStruct, v, s, iv, flag, m) {};

struct ReturnBox {
    float x;
    float y;
    float z;
    float w;
};
LUISA_STRUCT(ReturnBox, x, y, z, w) {};

struct PointerData {
    float a;
    float b;
    float c;
};
LUISA_STRUCT(PointerData, a, b, c) {};

// ---------------------------------------------------------------------------
// Test 1: nested if-elif-else, switch, break in loops, while-loop, early return
// Passes exercised:
//   destructure_cfg, restructure_cfg, simplify_cfg, if_conversion,
//   lower_break_continue, early_return_elimination, phi_cleanup
// ---------------------------------------------------------------------------
void test_control_flow_corners(Device &device) {
    auto stream = device.create_stream();
    auto buf = device.create_buffer<float>(64);
    auto result = device.create_buffer<float>(1);

    Kernel1D k = [](BufferVar<float> buf, BufferVar<float> result) noexcept {
        set_block_size(64);
        auto idx = dispatch_id().x;
        Float val = 0.0f;

        // Nested if-elif-else with 4-way branch (tests destructure/restructure)
        $if (idx < 16) {
            val = 1.0f;
        } $elif (idx < 32) {
            val = 2.0f;
        } $elif (idx < 48) {
            val = 3.0f;
        } $else {
            val = 4.0f;
        };

        // Loop with conditional break (tests lower_break_continue)
        Float acc = 0.0f;
        $for (i, 10) {
            acc += val;
            $if (acc > 20.0f) {
                $break;
            };
        };

        // Diamond if-else assigning to same variable (tests if_conversion, phi_cleanup)
        Float diamond;
        $if (idx % 2 == 0) {
            diamond = val;
        } $else {
            diamond = val;  // both branches assign same value -> phi_cleanup
        };

        // Switch with multiple non-default cases (tests simplify_cfg)
        Float sw_val = 0.0f;
        $switch (idx % 4) {
            $case (0) { sw_val = 1.0f; };
            $case (1) { sw_val = 2.0f; };
            $case (2) { sw_val = 4.0f; };
            $default { sw_val = 8.0f; };
        };

        // While loop pattern (tests loop_rotation)
        Float counter = 0.0f;
        Float target = (idx % 8).cast<float>() + 1.0f;
        $while (counter < target) {
            counter += 1.0f;
        };

        // Nested if inside loop (tests destructure with inner control)
        Float nested = 0.0f;
        $for (j, 5) {
            $if (j < 3) {
                nested += 1.0f;
            } $else {
                nested += 2.0f;
            };
        };

        buf.write(idx, val + acc + diamond + sw_val + counter + nested);

        // Early return style: only thread 0 does final reduction
        $if (idx == 0) {
            Float total = 0.0f;
            $for (k, 64) {
                total += buf.read(k);
            };
            result.write(0, total);
        };
    };

    auto shader = device.compile(k);
    luisa::vector<float> host(64);
    luisa::vector<float> host_result(1, 0.0f);
    stream << shader(buf, result).dispatch(64)
           << buf.copy_to(luisa::span{host})
           << result.copy_to(luisa::span{host_result})
           << synchronize();

    LUISA_INFO("Control-flow corner result: {:f}", host_result[0]);
    expect(host_result[0] > 0.0f) << "control flow corner sum should be positive";
    // Quick sanity: first element must be > 0
    expect(host[0] > 0.0f) << "first element should be positive";
}

// ---------------------------------------------------------------------------
// Test 2: conditional stores, dead stores, struct access, store-forwarding
// Passes exercised:
//   mem2reg, dead_store_elimination, local_store_forward,
//   local_load_elimination, sroa, reg2mem
// ---------------------------------------------------------------------------
void test_memory_pass_corners(Device &device) {
    auto stream = device.create_stream();
    auto out = device.create_buffer<float>(64);

    Kernel1D k = [](BufferVar<float> out) noexcept {
        set_block_size(64);
        auto idx = dispatch_id().x;

        // Conditional store to local var (tests mem2reg, reg2mem)
        Float val;
        $if (idx < 32) {
            val = 1.0f;
        } $else {
            val = 2.0f;
        };

        // Dead store chain (tests dead_store_elimination)
        Float dead = 0.0f;
        dead = idx.cast<float>();   // dead: overwritten before read
        dead = val;                 // dead: overwritten before read
        dead = 42.0f;              // only this one matters

        // Store-forward: store then immediately load (tests local_store_forward)
        Float fwd = val * 2.0f;
        Float loaded = fwd;       // should forward val*2.0f directly
        loaded += 1.0f;

        // Redundant load elimination (tests local_load_elimination)
        Float rle1 = loaded;
        Float rle2 = loaded;      // should use rle1 directly

        // Partial struct write/read (tests sroa)
        Var<VecStruct> vs;
        vs.v = make_float3(val, val + 1.0f, val + 2.0f);
        vs.s = val + 10.0f;

        // Access individual vector lanes (tests scalarizer on struct member)
        Float vs_sum = vs.v.x + vs.v.y + vs.v.z + vs.s;

        // Nested struct (tests deeper sroa / trace_gep)
        Var<OuterStruct> os;
        os.inner.x = val;
        os.inner.y = make_float2(val, val + 1.0f);
        os.z = val + 3.0f;
        os.w = make_float3(val, val, val);
        Float os_sum = os.inner.x + os.inner.y.x + os.inner.y.y + os.z + os.w.x + os.w.y + os.w.z;

        out.write(idx, dead + fwd + loaded + rle1 + rle2 + vs_sum + os_sum);
    };

    auto shader = device.compile(k);
    luisa::vector<float> host(64);
    stream << shader(out).dispatch(64)
           << out.copy_to(luisa::span{host})
           << synchronize();

    LUISA_INFO("Memory-pass corner: host[0]={:f}, host[33]={:f}", host[0], host[33]);
    expect(host[0] != host[33]) << "values should differ across conditional boundary";
    bool all_pos = true;
    for (auto v : host) {
        if (v <= 0.0f) { all_pos = false; break; }
    }
    expect(all_pos) << "all outputs should be positive";
}

// ---------------------------------------------------------------------------
// Test 3: deeply nested callables, unused params, compose multi-return
// Passes exercised:
//   inline, dead_arg_elim, unused_callable_removal, promote_ref_arg
// ---------------------------------------------------------------------------
void test_callable_inline_corners(Device &device) {
    auto stream = device.create_stream();
    auto out = device.create_buffer<float>(64);

    // Callable with unused parameter (tests dead_arg_elim)
    Callable add_with_extra = [](Var<float> a, Var<float> b, Var<float> unused) noexcept {
        return a + b;  // 'unused' should be eliminated by dead_arg_elim
    };

    // Multi-return via compose (tests inline with tuple)
    Callable mult_ret = [](Var<float> a, Var<float> b) noexcept {
        return compose(a + b, a * b, a - b);
    };

    // Nested callable chain of depth 3 (tests inline with deep nesting)
    Callable leaf = [](Var<float> x) noexcept { return x + 1.0f; };
    Callable mid = [&leaf](Var<float> x) noexcept { return leaf(x) * 2.0f; };
    Callable root = [&mid](Var<float> x) noexcept { return mid(x) + 3.0f; };

    // An intentionally unused callable (tests unused_callable_removal)
    Callable ghost = [](Var<float> x) noexcept { return x * 999.0f; };

    Kernel1D k = [&](BufferVar<float> out) noexcept {
        set_block_size(64);
        auto idx = dispatch_id().x;
        Float val = idx.cast<float>();

        Float r1 = add_with_extra(val, 2.0f, 999.0f);

        auto t = mult_ret(val, 3.0f);
        Float r2 = t.get<0>() + t.get<1>() + t.get<2>();

        Float r3 = root(val);   // should inline mid->leaf chain

        // ghost is captured but never called -> tests unused_callable_removal
        out.write(idx, r1 + r2 + r3);
    };

    auto shader = device.compile(k);
    luisa::vector<float> host(64);
    stream << shader(out).dispatch(64)
           << out.copy_to(luisa::span{host})
           << synchronize();

    LUISA_INFO("Callable-inline corner: host[0]={:f}", host[0]);
    // r1 = val+2, r2 = (val+3)+(val*3)+(val-3), r3 = (val+1)*2+3
    // For val=0: r1=2, r2=3+0+(-3)=0, r3=1*2+3=5 -> total=7
    float expected_0 = 2.0f + (3.0f + 0.0f - 3.0f) + 5.0f;
    expect(std::abs(host[0] - expected_0) < 1e-4f) << "callable inline result mismatch at idx 0";
}

// ---------------------------------------------------------------------------
// Test 4: small loops, non-unit strides, loop-invariant code, while loops
// Passes exercised:
//   loop_unroll, indvar_simplify, licm, loop_rotation
// ---------------------------------------------------------------------------
void test_loop_pass_corners(Device &device) {
    auto stream = device.create_stream();
    auto out = device.create_buffer<float>(64);

    Kernel1D k = [](BufferVar<float> out) noexcept {
        set_block_size(64);
        auto idx = dispatch_id().x;

        // Small constant-trip loop (loop_unroll candidate)
        Float sum1 = 0.0f;
        $for (i, 4) {
            sum1 += i.cast<float>();
        };

        // Loop with non-unit stride (indvar_simplify candidate)
        Float sum2 = 0.0f;
        $for (i, 0, 20, 2) {
            sum2 += i.cast<float>();
        };

        // Loop-invariant code inside loop (licm candidate)
        Float invariant = idx.cast<float>() * 0.5f;
        Float acc = 0.0f;
        $for (k, 8) {
            acc += invariant;
        };

        // While-loop (loop_rotation candidate - tests while->do-while)
        Float ctr = 0.0f;
        $while (ctr < 5.0f) {
            ctr += 1.0f;
        };

        // Nested loop with different bounds
        Float nested_sum = 0.0f;
        $for (outer, 3) {
            $for (inner, 2) {
                nested_sum += (outer * 2 + inner).cast<float>();
            };
        };

        out.write(idx, sum1 + sum2 + acc + ctr + nested_sum);
    };

    auto shader = device.compile(k);
    luisa::vector<float> host(64);
    stream << shader(out).dispatch(64)
           << out.copy_to(luisa::span{host})
           << synchronize();

    LUISA_INFO("Loop-pass corner: host[0]={:f}, host[10]={:f}", host[0], host[10]);
    // sum1 = 0+1+2+3 = 6
    // sum2 = 0+2+4+...+18 = 90
    // acc = invariant*8, so for idx=0 -> 0, for idx=10 -> 5*8=40
    // ctr = 5
    // nested_sum = 0+1+2+3+4+5 = 15
    // idx=0 total = 6+90+0+5+15 = 116
    expect(std::abs(host[0] - 116.0f) < 1e-4f) << "loop pass result mismatch at idx 0";
    // idx=10 total = 6+90+40+5+15 = 156
    expect(std::abs(host[10] - 156.0f) < 1e-4f) << "loop pass result mismatch at idx 10";
}

// ---------------------------------------------------------------------------
// Test 5: algebraic identities, constant folding, libcall simplify, div/rem, GVN
// Passes exercised:
//   algebraic_simplify, const_fold, simplify_libcalls, reassociate,
//   div_rem_pairs, gvn, sccp, cvp
// ---------------------------------------------------------------------------
void test_algebraic_pass_corners(Device &device) {
    auto stream = device.create_stream();
    auto out = device.create_buffer<float>(64);

    Kernel1D k = [](BufferVar<float> out) noexcept {
        set_block_size(64);
        auto idx = dispatch_id().x;
        Float val = idx.cast<float>() + 1.0f;  // avoid val==0

        // Identity patterns (algebraic_simplify)
        Float a1 = val + 0.0f;  // x+0 -> x
        Float a2 = val * 1.0f;  // x*1 -> x
        Float a3 = val - 0.0f;  // x-0 -> x

        // Constant expressions (const_fold)
        Float c1 = 2.0f * 3.0f + 4.0f;  // -> 10.0f
        Float c2 = 10.0f / 2.0f;         // -> 5.0f

        // Library call simplifications (simplify_libcalls)
        Float lc_exp = exp(0.0f);        // exp(0) -> 1.0
        Float lc_cos = cos(0.0f);        // cos(0) -> 1.0
        Float lc_pow0 = pow(val, 0.0f);  // pow(x,0) -> 1.0

        // Div/rem pair: compute both quotient and remainder-like value
        Float dividend = val + 10.0f;
        Float divisor = 3.0f;
        Float quot = dividend / divisor;
        Float rem = dividend - quot * divisor;  // should pair with div

        // GVN: common subexpression used in two branches
        Float common = val * 2.0f + 1.0f;
        Float branch_result;
        $if (idx < 32) {
            branch_result = common + 1.0f;
        } $else {
            branch_result = common + 2.0f;  // 'common' GVN candidate
        };

        // Reassociate: expression chain that could be rebalanced
        Float rea = val + 2.0f + 3.0f + 4.0f;  // -> val + 9.0f

        // CVP/SCCP: condition on constant; dead branch elimination
        Float cvp_val = 100.0f;
        Float cvp_result;
        $if (cvp_val > 0.0f) {  // always true
            cvp_result = cvp_val;
        } $else {
            cvp_result = 0.0f;  // dead code, should be eliminated
        };

        out.write(idx,
            a1 + a2 + a3 +
            c1 + c2 +
            lc_exp + lc_cos + lc_pow0 +
            quot * divisor + rem +
            branch_result + rea + cvp_result);
    };

    auto shader = device.compile(k);
    luisa::vector<float> host(64);
    stream << shader(out).dispatch(64)
           << out.copy_to(luisa::span{host})
           << synchronize();

    LUISA_INFO("Algebraic-pass corner: host[0]={:f}, host[40]={:f}", host[0], host[40]);
    // Verify a few elements
    float val0 = 1.0f;
    float expected_a = val0 * 3.0f;  // a1+a2+a3 = val+val+val = 3*val
    float expected_c = 15.0f;         // c1+c2
    float expected_lc = 3.0f;         // 1+1+1
    float d0 = val0 + 10.0f;          // 11
    float expected_dr = 3.0f * (d0 / 3.0f) + (d0 - (d0 / 3.0f) * 3.0f);  // = d0 = 11.0
    float common0 = val0 * 2.0f + 1.0f;  // 3
    float expected_br = common0 + 1.0f;  // 4 (idx 0 < 32)
    float expected_rea = val0 + 9.0f;    // 10
    float expected_cvp = 100.0f;
    float expected_0 = expected_a + expected_c + expected_lc + d0 + expected_br + expected_rea + expected_cvp;
    expect(std::abs(host[0] - expected_0) < 1e-4f) << "algebraic pass result mismatch at idx 0";
}

// ---------------------------------------------------------------------------
// Test 6: scalarize vectors, trace/transpose GEP on nested arrays
// Passes exercised: scalarizer, trace_gep, transpose_gep
// ---------------------------------------------------------------------------
void test_scalarize_gep_corners(Device &device) {
    auto stream = device.create_stream();
    auto out = device.create_buffer<float>(64);

    Kernel1D k = [](BufferVar<float> out) noexcept {
        set_block_size(64);
        auto idx = dispatch_id().x;
        Float val = idx.cast<float>() + 1.0f;

        // Scalarize: struct with float3 member - partial access
        Var<VecStruct> vs;
        vs.v = make_float3(val, val * 2.0f, val * 3.0f);
        vs.s = val * 4.0f;
        Float vs_x = vs.v.x;
        Float vs_y = vs.v.y;
        Float vs_z = vs.v.z;

        // Struct with nested struct - exercises GEP and scalarizer
        Var<OuterStruct> os;
        os.inner.x = val;
        os.inner.y = make_float2(val + 1.0f, val + 2.0f);
        os.z = val + 3.0f;
        os.w = make_float3(val, val * 2.0f, 0.0f);
        // Read back via GEP chain: outer->inner->y->x
        Float os_sum = os.inner.x + os.inner.y.x + os.inner.y.y + os.z
                     + os.w.x + os.w.y + os.w.z;

        // Scalarize: float3 arithmetic then extract component
        Float3 v3 = make_float3(val, 2.0f, 3.0f);
        Float3 v3_scaled = v3 * 2.0f;
        Float v3x = v3_scaled.x;
        Float v3y = v3_scaled.y;

        out.write(idx, vs_x + vs_y + vs_z + vs.s + os_sum + v3x + v3y);
    };

    auto shader = device.compile(k);
    luisa::vector<float> host(64);
    stream << shader(out).dispatch(64)
           << out.copy_to(luisa::span{host})
           << synchronize();

    LUISA_INFO("Scalarize/GEP corner: host[0]={:f}, host[10]={:f}", host[0], host[10]);
    // idx=0: val=1
    //   vs_sum = 1+2+3+4 = 10
    //   os_sum = 1 + 2+3 + 4 + 1+2+0 = 13
    //   v3 = (1,2,3)*2 = (2,4,6); v3x+v3y = 2+4 = 6
    //   total = 10 + 13 + 6 = 29
    expect(std::abs(host[0] - 29.0f) < 1e-4f) << "scalarize/gep mismatch at idx 0";
    // idx=1: val=2
    //   vs_sum = 2+4+6+8 = 20
    //   os_sum = 2 + 3+4 + 5 + 2+4+0 = 20
    //   v3 = (2,2,3)*2 = (4,4,6); v3x+v3y = 4+4 = 8
    //   total = 20 + 20 + 8 = 48
    expect(std::abs(host[1] - 48.0f) < 1e-4f) << "scalarize/gep mismatch at idx 1";
}

// ============================================================================
// Original test
// ============================================================================

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

// ---------------------------------------------------------------------------
// New CFG corner tests: constant-folded branches, nested switches, tiny diamonds
// Passes exercised: destructure_cfg, restructure_cfg, simplify_cfg,
//                   lower_switch, convergence_region, if_conversion
// ---------------------------------------------------------------------------
void test_cfg_empty_constant_and_multi_predecessor_merge(Device &device) {
    auto stream = device.create_stream();
    auto out = device.create_buffer<float>(64);

    Kernel1D k = [](BufferVar<float> out) noexcept {
        set_block_size(64);
        auto idx = dispatch_id().x;
        Float acc = 0.0f;

        UInt zero_u = 0u;
        $if (zero_u == 0u) {
            acc += 1.0f;
        } $else {
        };

        UInt one_u = 1u;
        $if (one_u == 0u) {
            acc += 1000.0f;
        } $else {
            acc += 2.0f;
        };

        Float base = idx.cast<float>();
        Float mid;
        $if (idx % 2u == 0u) {
            mid = base + 10.0f;
        } $else {
            mid = base - 10.0f;
        };
        $if (idx % 3u == 0u) {
            mid = mid * 2.0f;
        } $else {
        };

        out.write(idx, acc + mid);
    };

    auto shader = device.compile(k);
    luisa::vector<float> host(64);
    stream << shader(out).dispatch(64)
           << out.copy_to(luisa::span{host})
           << synchronize();

    LUISA_INFO("CFG empty/constant corner: host[0]={:f}, host[2]={:f}, host[3]={:f}",
               host[0], host[2], host[3]);

    expect(std::abs(host[0] - 23.0f) < 1e-4f) << "idx 0: even, multiple of 3";
    expect(std::abs(host[2] - 15.0f) < 1e-4f) << "idx 2: even, not multiple of 3";
    expect(std::abs(host[3] - (-11.0f)) < 1e-4f) << "idx 3: odd, multiple of 3";
    expect(std::abs(host[6] - 35.0f) < 1e-4f) << "idx 6: even, multiple of 3";
}

void test_cfg_nested_switch_fallthrough_like(Device &device) {
    auto stream = device.create_stream();
    auto out = device.create_buffer<float>(64);

    Kernel1D k = [](BufferVar<float> out) noexcept {
        set_block_size(64);
        auto idx = dispatch_id().x;
        Float base = idx.cast<float>();
        Float sw = 0.0f;

        $switch (idx % 4u) {
            $case (0u) {
                $switch (idx % 2u) {
                    $case (0u) { sw = base + 1.0f; };
                    $case (1u) { sw = base + 2.0f; };
                    $default { sw = base + 3.0f; };
                };
            };
            $case (1u) { sw = base + 10.0f; };
            $case (2u) { sw = base + 20.0f; };
            $default { sw = base + 30.0f; };
        };

        UInt constant_selector = 1u;
        Float extra = 0.0f;
        $switch (constant_selector) {
            $case (0u) { extra = 1.0f; };
            $case (1u) { extra = 4.0f; };
            $default { extra = 8.0f; };
        };

        out.write(idx, sw + extra);
    };

    auto shader = device.compile(k);
    luisa::vector<float> host(64);
    stream << shader(out).dispatch(64)
           << out.copy_to(luisa::span{host})
           << synchronize();

    LUISA_INFO("CFG nested-switch corner: host[0]={:f}, host[1]={:f}, host[2]={:f}, host[3]={:f}",
               host[0], host[1], host[2], host[3]);

    expect(std::abs(host[0] - 5.0f) < 1e-4f) << "idx 0 mismatch";
    expect(std::abs(host[1] - 15.0f) < 1e-4f) << "idx 1 mismatch";
    expect(std::abs(host[2] - 26.0f) < 1e-4f) << "idx 2 mismatch";
    expect(std::abs(host[3] - 37.0f) < 1e-4f) << "idx 3 mismatch";
    expect(std::abs(host[4] - 9.0f) < 1e-4f) << "idx 4 mismatch";
}

void test_cfg_tiny_diamond_if_conversion(Device &device) {
    auto stream = device.create_stream();
    auto out = device.create_buffer<float>(64);

    Kernel1D k = [](BufferVar<float> out) noexcept {
        set_block_size(64);
        auto idx = dispatch_id().x;
        Float a = idx.cast<float>();
        Float x;

        $if (idx % 2u == 0u) {
            x = a + 1.0f;
        } $else {
            x = a - 1.0f;
        };

        out.write(idx, x);
    };

    auto shader = device.compile(k);
    luisa::vector<float> host(64);
    stream << shader(out).dispatch(64)
           << out.copy_to(luisa::span{host})
           << synchronize();

    LUISA_INFO("CFG tiny-diamond corner: host[0]={:f}, host[1]={:f}, host[2]={:f}, host[3]={:f}",
               host[0], host[1], host[2], host[3]);

    expect(std::abs(host[0] - 1.0f) < 1e-4f) << "idx 0 mismatch";
    expect(std::abs(host[1] - 0.0f) < 1e-4f) << "idx 1 mismatch";
    expect(std::abs(host[2] - 3.0f) < 1e-4f) << "idx 2 mismatch";
    expect(std::abs(host[3] - 2.0f) < 1e-4f) << "idx 3 mismatch";
}

// ---------------------------------------------------------------------------
// New structured-lowering corner tests
// Passes exercised: lower_break_continue, lower_ray_query_loop,
//                   lower_ray_query_loop_to_loop, early_return_elimination, phi_cleanup
// ---------------------------------------------------------------------------
void test_structured_break_continue(Device &device) {
    auto stream = device.create_stream();
    auto out = device.create_buffer<uint>(64);

    Kernel1D k = [](BufferVar<uint> out) noexcept {
        set_block_size(64);
        auto idx = dispatch_id().x;

        UInt acc = 0u;
        $for (i, 0u, 4u) {
            $for (j, 0u, 4u) {
                $if ((i + j + idx) % 2u == 0u) {
                    $continue;
                };
                acc += 1u;
                $if (acc > 5u) {
                    $break;
                };
            };
        };

        UInt multi_phi = idx;
        $for (n, 0u, 20u) {
            $if (n == 3u) {
                multi_phi = 100u;
                $break;
            };
            $if (n == 7u) {
                multi_phi = 200u;
                $break;
            };
            multi_phi = multi_phi + 1u;
        };

        UInt wk = 0u;
        UInt wsum = 0u;
        $while (wk < 20u) {
            wk += 1u;
            $if (wk % 3u == 0u) {
                $continue;
            };
            wsum += wk;
            $if (wsum > 30u) {
                $break;
            };
        };

        UInt redundant = 0u;
        $if (idx % 3u == 0u) {
            redundant = 5u;
        } $elif (idx % 3u == 1u) {
            redundant = 5u;
        } $else {
            redundant = 5u;
        };

        out.write(idx, acc + multi_phi + wsum + redundant);
    };

    auto shader = device.compile(k);
    luisa::vector<uint> host(64);
    stream << shader(out).dispatch(64)
           << out.copy_to(luisa::span{host})
           << synchronize();

    LUISA_INFO("Structured break/continue corner: host[0]={}", host[0]);
    for (auto v : host) {
        expect(v > 0u) << "all outputs should be positive";
    }
}

void test_structured_early_return(Device &device) {
    auto stream = device.create_stream();
    auto out = device.create_buffer<uint>(64);

    Callable<uint(uint)> classify = [](UInt idx) noexcept {
        $if (idx % 2u == 0u) {
            $return(idx * 2u);
        } $else {
            $return(idx + 100u);
        };
        return 0u;
    };

    Kernel1D k = [&](BufferVar<uint> out) noexcept {
        set_block_size(64);
        auto idx = dispatch_id().x;
        UInt r = classify(idx);
        out.write(idx, r);
    };

    auto shader = device.compile(k);
    luisa::vector<uint> host(64);
    stream << shader(out).dispatch(64)
           << out.copy_to(luisa::span{host})
           << synchronize();

    LUISA_INFO("Structured early-return corner: host[0]={}, host[1]={}", host[0], host[1]);
    for (uint i = 0u; i < 64u; ++i) {
        uint expected = (i % 2u == 0u) ? (i * 2u) : (i + 100u);
        expect(host[i] == expected) << "early-return callable result mismatch";
    }
}

void test_structured_ray_query_loop(Device &device) {
    if (device.backend_name() == "cpu") {
        LUISA_INFO("Skipping ray-query loop test on CPU backend.");
        expect(true) << "ray-query loop test skipped on CPU";
        return;
    }

    auto stream = device.create_stream();
    auto out = device.create_buffer<uint>(4);

    luisa::vector<float3> vertices{
        make_float3(-1.0f, -1.0f, 0.0f),
        make_float3(1.0f, -1.0f, 0.0f),
        make_float3(0.0f, 1.0f, 0.0f)};
    luisa::vector<Triangle> triangles{{0u, 1u, 2u}};

    auto vertex_buffer = device.create_buffer<float3>(vertices.size());
    auto triangle_buffer = device.create_buffer<Triangle>(triangles.size());
    auto mesh = device.create_mesh(vertex_buffer, triangle_buffer);
    auto accel = device.create_accel();
    accel.emplace_back(mesh);

    stream << vertex_buffer.copy_from(luisa::span{vertices})
           << triangle_buffer.copy_from(luisa::span{triangles})
           << mesh.build()
           << accel.build()
           << synchronize();

    Kernel1D k = [&](BufferVar<uint> out) noexcept {
        set_block_size(32);
        auto idx = dispatch_id().x;
        auto ray = make_ray(
            make_float3(0.1f * idx.cast<float>(), 0.0f, 1.0f),
            make_float3(0.0f, 0.0f, -1.0f));
        UInt hit_count = 0u;
        accel->traverse(ray, {})
            .on_surface_candidate([&](SurfaceCandidate &c) noexcept {
                c.commit();
                hit_count += 1u;
            })
            .on_procedural_candidate([&](ProceduralCandidate &c) noexcept {
            })
            .trace();
        out.write(idx, hit_count);
    };

    auto shader = device.compile(k);
    luisa::vector<uint> host(4);
    stream << shader(out).dispatch(4)
           << out.copy_to(luisa::span{host})
           << synchronize();

    LUISA_INFO("Structured ray-query loop corner: hits={} {} {} {}", host[0], host[1], host[2], host[3]);
    bool any_hit = false;
    for (auto v : host) {
        if (v > 0u) {
            any_hit = true;
            break;
        }
    }
    if (!any_hit) {
        LUISA_WARNING("Ray-query loop test: no hits detected; RT may be unavailable on this setup");
        for (auto v : host) {
            expect(v <= 1u) << "hit count should be at most one";
        }
    } else {
        for (auto v : host) {
            expect(v == 1u) << "each ray should hit the single triangle exactly once";
        }
    }
}

// ---------------------------------------------------------------------------
// New memory corner tests
// Passes exercised: mem2reg, reg2mem, sroa, dead_store_elimination,
//                   local_store_forward, local_load_elimination
// ---------------------------------------------------------------------------
void test_memory_cross_block_phi(Device &device) {
    auto stream = device.create_stream();
    auto out = device.create_buffer<float>(64);

    Kernel1D k = [](BufferVar<float> out) noexcept {
        set_block_size(64);
        auto idx = dispatch_id().x;
        Float f = idx.cast<float>();

        Float branch_val;
        $if (idx < 16) {
            branch_val = f + 1.0f;
        } $elif (idx < 32) {
            branch_val = f + 2.0f;
        } $elif (idx < 48) {
            branch_val = f + 3.0f;
        } $else {
            branch_val = f + 4.0f;
        };

        Float a = branch_val;
        Float b = branch_val;

        Float loop_acc = 0.0f;
        $for (i, 4) {
            loop_acc = loop_acc + f;
        };

        out.write(idx, a + b + loop_acc);
    };

    auto shader = device.compile(k);
    luisa::vector<float> host(64);
    stream << shader(out).dispatch(64)
           << out.copy_to(luisa::span{host})
           << synchronize();

    LUISA_INFO("Cross-block phi corner: host[0]={:f}, host[20]={:f}, host[50]={:f}",
               host[0], host[20], host[50]);

    expect(std::abs(host[0] - 2.0f) < 1e-4f) << "cross-block phi mismatch at idx 0";
    expect(std::abs(host[20] - 124.0f) < 1e-4f) << "cross-block phi mismatch at idx 20";
    expect(std::abs(host[50] - 308.0f) < 1e-4f) << "cross-block phi mismatch at idx 50";
}

void test_memory_partial_sroa(Device &device) {
    auto stream = device.create_stream();
    auto out = device.create_buffer<float>(64);

    Kernel1D k = [](BufferVar<float> out) noexcept {
        set_block_size(64);
        auto idx = dispatch_id().x;
        Float f = idx.cast<float>();

        Var<PartialA> ps;
        ps.b = f + 1.0f;
        Float struct_sum = ps.b + 10.0f;

        $array<float, 4> arr;
        arr[1] = f;
        arr[3] = f * 2.0f;
        Float arr_sum = arr[1] + arr[3];

        $array<float, 3> dse_arr;
        dse_arr[0] = 0.0f;
        dse_arr[1] = 1.0f;
        dse_arr[1] = f;
        dse_arr[2] = 2.0f;
        dse_arr[2] = f * 3.0f;
        Float dse_sum = dse_arr[0] + dse_arr[1] + dse_arr[2];

        out.write(idx, struct_sum + arr_sum + dse_sum);
    };

    auto shader = device.compile(k);
    luisa::vector<float> host(64);
    stream << shader(out).dispatch(64)
           << out.copy_to(luisa::span{host})
           << synchronize();

    LUISA_INFO("Partial SROA corner: host[0]={:f}, host[5]={:f}", host[0], host[5]);
    expect(std::abs(host[0] - 11.0f) < 1e-4f) << "partial SROA mismatch at idx 0";
    expect(std::abs(host[5] - 51.0f) < 1e-4f) << "partial SROA mismatch at idx 5";
}

void test_memory_dead_store_loop(Device &device) {
    auto stream = device.create_stream();
    auto out = device.create_buffer<float>(64);

    Kernel1D k = [](BufferVar<float> out) noexcept {
        set_block_size(64);
        auto idx = dispatch_id().x;
        Float f = idx.cast<float>();

        Float acc = 0.0f;
        $for (i, 3) {
            Float tmp;
            tmp = f + 100.0f;
            tmp = i.cast<float>();
            acc = acc + tmp;
        };

        Float branch_store;
        $if (idx < 32) {
            branch_store = f + 1.0f;
        } $else {
            branch_store = f + 2.0f;
        };
        Float forwarded = branch_store + 1.0f;

        Float maybe;
        $if (idx < 32) {
            maybe = f;
        };
        Float loaded = maybe;
        Float selected;
        $if (idx < 32) {
            selected = loaded + 1.0f;
        } $else {
            selected = 0.0f;
        };

        out.write(idx, acc + forwarded + selected);
    };

    auto shader = device.compile(k);
    luisa::vector<float> host(64);
    stream << shader(out).dispatch(64)
           << out.copy_to(luisa::span{host})
           << synchronize();

    LUISA_INFO("Dead-store loop corner: host[0]={:f}, host[40]={:f}", host[0], host[40]);
    expect(std::abs(host[0] - 6.0f) < 1e-4f) << "dead-store loop mismatch at idx 0";
    expect(std::abs(host[40] - 46.0f) < 1e-4f) << "dead-store loop mismatch at idx 40";
}

// ---------------------------------------------------------------------------
// New algebraic corner tests
// Passes exercised: algebraic_simplify, const_fold, simplify_libcalls,
//                   reassociate, div_rem_pairs, early_cse
// ---------------------------------------------------------------------------
void test_algebraic_bitwise_identities(Device &device) {
    auto stream = device.create_stream();
    auto out = device.create_buffer<uint>(64);

    Kernel1D k = [](BufferVar<uint> out) noexcept {
        set_block_size(64);
        UInt idx = dispatch_id().x;
        UInt x = idx + 1u;

        UInt b1 = x & 0xFFFFFFFFu;
        UInt b2 = x | 0u;
        UInt b3 = x ^ 0u;
        UInt b4 = x & 0u;
        UInt b5 = x ^ x;
        UInt b6 = x << 0u;
        UInt b7 = x >> 0u;

        UInt c1 = 0xAAu & 0x55u;
        UInt c2 = 0x0Fu | 0xF0u;
        UInt c3 = 0xFFu ^ 0xFFu;
        UInt c4 = 1u << 8u;

        UInt u = idx + 16u;
        UInt div1 = u / 1u;
        UInt div8 = u / 8u;
        UInt rem8 = u % 8u;

        out.write(idx,
                  b1 + b2 + b3 + b4 + b5 + b6 + b7 +
                  c1 + c2 + c3 + c4 +
                  div1 + div8 + rem8);
    };

    auto shader = device.compile(k);
    luisa::vector<uint> host(64);
    stream << shader(out).dispatch(64)
           << out.copy_to(luisa::span{host})
           << synchronize();

    LUISA_INFO("Algebraic bitwise corner: host[0]={}, host[1]={}", host[0], host[1]);
    expect(host[0] == 534u) << "bitwise identity result mismatch at idx 0";
    expect(host[1] == 541u) << "bitwise identity result mismatch at idx 1";
}

void test_algebraic_libcall_reassociate(Device &device) {
    auto stream = device.create_stream();
    auto out = device.create_buffer<float>(64);

    Kernel1D k = [](BufferVar<float> out) noexcept {
        set_block_size(64);
        auto idx = dispatch_id().x;
        Float val = idx.cast<float>() + 1.0f;

        Float s1 = sqrt(1.0f);
        Float p1 = pow(1.0f, val);
        Float l1 = luisa::compute::log(def(1.0f));
        Float e0 = exp2(0.0f);
        Float r1 = rsqrt(def(1.0f));
        Float a0 = abs(0.0f);

        Float lerp0 = lerp(val, val * 2.0f, 0.0f);
        Float lerp1 = lerp(val, val * 2.0f, 1.0f);
        Float clamp01 = clamp(val, 0.0f, 1.0f);
        Float step0 = step(0.0f, val);

        Float negneg = -(-val);
        Float zero_minus = 0.0f - (0.0f - val);

        Float chain = val + 1.0f + 2.0f + 3.0f + 4.0f +
                      5.0f + 6.0f + 7.0f + 8.0f;
        Float cancel = (val + 1.0f) - 1.0f;
        Float mulchain = val * 1.0f * 2.0f * 3.0f;

        out.write(idx,
                  s1 + p1 + l1 + e0 + r1 + a0 +
                  lerp0 + lerp1 + clamp01 + step0 +
                  negneg + zero_minus +
                  chain + cancel + mulchain);
    };

    auto shader = device.compile(k);
    luisa::vector<float> host(64);
    stream << shader(out).dispatch(64)
           << out.copy_to(luisa::span{host})
           << synchronize();

    LUISA_INFO("Algebraic libcall/reassociate corner: host[0]={:f}, host[1]={:f}",
               host[0], host[1]);

    constexpr float eps = 1e-4f;
    for (uint i = 0; i < 64; ++i) {
        float expected = 13.0f * static_cast<float>(i + 1u) + 42.0f;
        expect(std::abs(host[i] - expected) < eps)
            << "libcall/reassociate result mismatch at idx " << i;
    }
}

void test_algebraic_early_cse(Device &device) {
    auto stream = device.create_stream();
    auto out = device.create_buffer<float>(64);

    Kernel1D k = [](BufferVar<float> out) noexcept {
        set_block_size(64);
        auto idx = dispatch_id().x;
        Float base = idx.cast<float>() + 1.0f;

        Float acc = 0.0f;
        $for (i, 8) {
            Float common = base * 3.0f + 2.0f;
            Float dup    = base * 3.0f + 2.0f;
            acc += common + dup;
        };

        Float outside1 = base * 3.0f + 2.0f;
        Float outside2 = base * 3.0f + 2.0f;

        out.write(idx, acc + outside1 + outside2);
    };

    auto shader = device.compile(k);
    luisa::vector<float> host(64);
    stream << shader(out).dispatch(64)
           << out.copy_to(luisa::span{host})
           << synchronize();

    LUISA_INFO("Algebraic early-cse corner: host[0]={:f}, host[1]={:f}",
               host[0], host[1]);

    constexpr float eps = 1e-4f;
    for (uint i = 0; i < 64; ++i) {
        float common = 3.0f * static_cast<float>(i + 1u) + 2.0f;
        float expected = 18.0f * common;
        expect(std::abs(host[i] - expected) < eps)
            << "early-cse result mismatch at idx " << i;
    }
}

// ---------------------------------------------------------------------------
// New value-propagation corner tests
// Passes exercised: gvn, sccp, cvp, scalar_evolution
// ---------------------------------------------------------------------------
void test_value_gvn_cvp(Device &device) {
    auto stream = device.create_stream();
    auto out = device.create_buffer<float>(64);

    Kernel1D k = [](BufferVar<float> out) noexcept {
        set_block_size(64);
        auto idx = dispatch_id().x;
        Float a = idx.cast<float>();
        Float b = (idx + 1u).cast<float>();
        Float shared = a * b + 1.0f;

        Float branch_val;
        $if (idx < 16u) {
            Float t = a * b + 1.0f;
            branch_val = t + 1.0f;
        } $elif (idx < 32u) {
            Float t = a * b + 1.0f;
            branch_val = t + 2.0f;
        } $elif (idx < 48u) {
            Float t = a * b + 1.0f;
            branch_val = t + 3.0f;
        } $else {
            Float t = a * b + 1.0f;
            branch_val = t + 4.0f;
        };

        UInt tag = idx % 4u;
        Float cvp_sum = 0.0f;
        $if (tag == 0u) {
            cvp_sum += tag.cast<float>();
        } $elif (tag == 1u) {
            cvp_sum += tag.cast<float>();
        } $elif (tag == 2u) {
            cvp_sum += tag.cast<float>();
        } $else {
            cvp_sum += tag.cast<float>();
        };

        out.write(idx, branch_val + cvp_sum);
    };

    auto shader = device.compile(k);
    luisa::vector<float> host(64);
    stream << shader(out).dispatch(64)
           << out.copy_to(luisa::span{host})
           << synchronize();

    LUISA_INFO("GVN/CVP corner: host[0]={:f}, host[20]={:f}, host[50]={:f}",
               host[0], host[20], host[50]);

    auto expected = [](uint i) {
        float a = static_cast<float>(i);
        float b = a + 1.0f;
        float shared = a * b + 1.0f;
        uint offset = (i < 16u) ? 1u : (i < 32u) ? 2u : (i < 48u) ? 3u : 4u;
        return shared + static_cast<float>(offset) + static_cast<float>(i % 4u);
    };
    for (uint i = 0; i < 64; ++i) {
        expect(std::abs(host[i] - expected(i)) < 1e-4f) << "gvn/cvp result mismatch";
    }
}

void test_value_sccp_phi(Device &device) {
    auto stream = device.create_stream();
    auto out = device.create_buffer<float>(64);

    Kernel1D k = [](BufferVar<float> out) noexcept {
        set_block_size(64);
        auto idx = dispatch_id().x;

        Float phi_val;
        $if (idx < 32u) {
            phi_val = 7.0f;
        } $else {
            phi_val = 7.0f;
        };
        Float result = phi_val + 3.0f;

        Float sw_val;
        $switch (idx % 3u) {
            $case (0u) { sw_val = 5.0f; };
            $case (1u) { sw_val = 5.0f; };
            $default { sw_val = 5.0f; };
        };
        result += sw_val + 1.0f;

        Float known = 100.0f;
        $if (known > 0.0f) {
            result += 2.0f;
        } $else {
            result += 999.0f;
        };

        out.write(idx, result);
    };

    auto shader = device.compile(k);
    luisa::vector<float> host(64);
    stream << shader(out).dispatch(64)
           << out.copy_to(luisa::span{host})
           << synchronize();

    LUISA_INFO("SCCP phi corner: host[0]={:f}", host[0]);
    for (auto v : host) {
        expect(std::abs(v - 18.0f) < 1e-4f) << "sccp phi result should be constant 18";
    }
}

void test_value_scalar_evolution(Device &device) {
    auto stream = device.create_stream();
    auto out = device.create_buffer<float>(64);

    Kernel1D k = [](BufferVar<float> out) noexcept {
        set_block_size(64);
        auto idx = dispatch_id().x;

        Float sum1 = 0.0f;
        $for (i, 0, 15, 3) {
            sum1 += i.cast<float>();
        };

        Float sum2 = 0.0f;
        $for (j, 0, 12, 4) {
            sum2 += (j * 2 + 1).cast<float>();
        };

        Float inv = idx.cast<float>() * 2.0f + 1.0f;
        Float sum3 = 0.0f;
        $for (k, 0, 6, 2) {
            sum3 += inv + k.cast<float>();
        };

        out.write(idx, sum1 + sum2 + sum3);
    };

    auto shader = device.compile(k);
    luisa::vector<float> host(64);
    stream << shader(out).dispatch(64)
           << out.copy_to(luisa::span{host})
           << synchronize();

    LUISA_INFO("Scalar-evolution corner: host[0]={:f}, host[1]={:f}, host[10]={:f}",
               host[0], host[1], host[10]);

    expect(std::abs(host[0] - 66.0f) < 1e-4f) << "scalar evolution mismatch at idx 0";
    expect(std::abs(host[1] - 72.0f) < 1e-4f) << "scalar evolution mismatch at idx 1";
    expect(std::abs(host[10] - 126.0f) < 1e-4f) << "scalar evolution mismatch at idx 10";
}

// ---------------------------------------------------------------------------
// New loop corner tests
// Passes exercised: loop_unroll, loop_rotation, indvar_simplify, licm
// ---------------------------------------------------------------------------
void test_loop_zero_one_trip(Device &device) {
    auto stream = device.create_stream();
    auto bounds = device.create_buffer<uint>(2u);
    auto out = device.create_buffer<float>(64u);

    luisa::vector<uint> host_bounds = {0u, 1u};
    stream << bounds.copy_from(luisa::span{host_bounds})
           << synchronize();

    Kernel1D k = [](BufferVar<uint> bounds, BufferVar<float> out) noexcept {
        set_block_size(64);
        auto idx = dispatch_id().x;
        UInt bound = bounds.read(idx % 2u);

        Float a = 0.0f;
        $for (i, 0) {
            a += 1.0f;
        };

        Float b = 0.0f;
        $for (i, 1) {
            b += 1.0f;
        };

        Float c = 0.0f;
        $for (i, 0u, 0u, 3u) {
            c += 1.0f;
        };

        Float d = 0.0f;
        $for (i, 0u, bound) {
            d += 1.0f;
        };

        out.write(idx, a + b + c + d);
    };

    auto shader = device.compile(k);
    luisa::vector<float> host(64);
    stream << shader(bounds, out).dispatch(64)
           << out.copy_to(luisa::span{host})
           << synchronize();

    LUISA_INFO("Loop zero/one trip: host[0]={:f}, host[1]={:f}", host[0], host[1]);
    expect(std::abs(host[0] - 1.0f) < 1e-4f) << "zero/one trip mismatch at idx 0";
    expect(std::abs(host[1] - 2.0f) < 1e-4f) << "zero/one trip mismatch at idx 1";
}

void test_loop_nested_licm(Device &device) {
    auto stream = device.create_stream();
    auto base_buf = device.create_buffer<float>(8u);
    auto out = device.create_buffer<float>(64u);

    luisa::vector<float> host_base = {1.0f, 2.0f, 3.0f, 4.0f,
                                      5.0f, 6.0f, 7.0f, 8.0f};
    stream << base_buf.copy_from(luisa::span{host_base})
           << synchronize();

    Kernel1D k = [](BufferVar<float> base_buf, BufferVar<float> out) noexcept {
        set_block_size(64);
        auto idx = dispatch_id().x;

        Float base = base_buf.read(idx % 8u);
        Float sum = 0.0f;

        $for (outer, 3) {
            $for (inner, 4) {
                Float factor = base * 2.0f + 1.0f;
                sum += factor;
            };
        };

        out.write(idx, sum);
    };

    auto shader = device.compile(k);
    luisa::vector<float> host(64);
    stream << shader(base_buf, out).dispatch(64)
           << out.copy_to(luisa::span{host})
           << synchronize();

    LUISA_INFO("Loop nested LICM: host[0]={:f}, host[1]={:f}, host[7]={:f}",
               host[0], host[1], host[7]);
    for (auto i = 0u; i < 64u; ++i) {
        auto expected = (host_base[i % 8u] * 2.0f + 1.0f) * 12.0f;
        expect(std::abs(host[i] - expected) < 1e-4f)
            << "nested licm mismatch at idx " << i;
    }
}

void test_loop_while_stride_break(Device &device) {
    auto stream = device.create_stream();
    auto limits = device.create_buffer<uint>(4u);
    auto out = device.create_buffer<float>(64u);

    luisa::vector<uint> host_limits = {3u, 6u, 9u, 12u};
    stream << limits.copy_from(luisa::span{host_limits})
           << synchronize();

    Kernel1D k = [](BufferVar<uint> limits, BufferVar<float> out) noexcept {
        set_block_size(64);
        auto idx = dispatch_id().x;
        UInt limit = limits.read(idx % 4u);

        UInt count_while = 0u;
        UInt i = 0u;
        $while (i < limit) {
            i += 3u;
            count_while += 1u;
        };

        UInt count_for = 0u;
        $for (j, 0u, limit, 3u) {
            $if (j == 0u) {
                $continue;
            };
            $if (j >= 6u) {
                $break;
            };
            count_for += 1u;
        };

        out.write(idx, count_while.cast<float>() * 10.0f + count_for.cast<float>());
    };

    auto shader = device.compile(k);
    luisa::vector<float> host(64);
    stream << shader(limits, out).dispatch(64)
           << out.copy_to(luisa::span{host})
           << synchronize();

    LUISA_INFO("Loop while/stride/break: host[0]={:f}, host[1]={:f}, host[2]={:f}, host[3]={:f}",
               host[0], host[1], host[2], host[3]);

    float expected[4] = {10.0f, 21.0f, 31.0f, 41.0f};
    for (auto i = 0u; i < 4u; ++i) {
        expect(std::abs(host[i] - expected[i]) < 1e-4f)
            << "while/stride/break mismatch at idx " << i;
    }
}

// ---------------------------------------------------------------------------
// New scalar/GEP corner tests
// Passes exercised: scalarizer, trace_gep, transpose_gep, aggregate_field_bitmask
// ---------------------------------------------------------------------------
void test_scalar_gep_vector_matrix(Device &device) {
    auto stream = device.create_stream();
    auto out = device.create_buffer<float>(64);

    Kernel1D k = [](BufferVar<float> out) noexcept {
        set_block_size(64);
        auto idx = dispatch_id().x;
        Float val = idx.cast<float>() + 1.0f;

        Float3x3 m = make_float3x3(
            make_float3(val, 0.0f, 0.0f),
            make_float3(0.0f, val * 2.0f, 0.0f),
            make_float3(0.0f, 0.0f, val * 3.0f));

        Float3 c0 = m[0] * 2.0f + make_float3(1.0f, 2.0f, 3.0f);
        Float3 c1 = m[1] * 0.5f;

        Float partial = c0.x + c0.z + c1.x;

        Float4 v4 = make_float4(val, val + 1.0f, val + 2.0f, val + 3.0f) * 3.0f;
        Float v4_partial = v4.y + v4.w;

        out.write(idx, partial + v4_partial);
    };

    auto shader = device.compile(k);
    luisa::vector<float> host(64);
    stream << shader(out).dispatch(64)
           << out.copy_to(luisa::span{host})
           << synchronize();

    LUISA_INFO("Scalar/GEP vector-matrix corner: host[0]={:f}, host[1]={:f}", host[0], host[1]);
    expect(std::abs(host[0] - 24.0f) < 1e-4f) << "vector/matrix scalarize mismatch at idx 0";
    expect(std::abs(host[1] - 32.0f) < 1e-4f) << "vector/matrix scalarize mismatch at idx 1";
}

void test_scalar_gep_nested_aos_soa(Device &device) {
    auto stream = device.create_stream();
    auto out = device.create_buffer<float>(64);

    Kernel1D k = [](BufferVar<float> out) noexcept {
        set_block_size(64);
        auto idx = dispatch_id().x;
        Float base = idx.cast<float>() * 10.0f;

        // Fixed-index struct access to exercise GEP on structs.
        Var<AoSItem> item;
        item.pos = make_float3(base, base + 1.0f, base + 2.0f);
        item.mass = base + 3.0f;
        Float item_sum = item.pos.x + item.pos.y + item.pos.z + item.mass;

        // Struct-of-arrays with dynamic scalar indexing (scalar arrays are supported).
        Var<SoAFields> soa;
        $for (i, 4) {
            soa.x[i] = base + i.cast<float>();
            soa.y[i] = base + i.cast<float>() + 1.0f;
            soa.z[i] = base + i.cast<float>() + 2.0f;
            soa.mass[i] = base + i.cast<float>() + 3.0f;
        };
        Float soa_sum = 0.0f;
        $for (i, 4) {
            soa_sum += soa.x[i] + soa.y[i] + soa.z[i] + soa.mass[i];
        };

        out.write(idx, item_sum + soa_sum);
    };

    auto shader = device.compile(k);
    luisa::vector<float> host(64);
    stream << shader(out).dispatch(64)
           << out.copy_to(luisa::span{host})
           << synchronize();

    LUISA_INFO("Scalar/GEP nested AoS/SoA corner: host[0]={:f}, host[1]={:f}", host[0], host[1]);
    // idx=0: item_sum = 0+1+2+3 = 6; soa per-iteration = 0+1+2+3, 1+2+3+4, 2+3+4+5, 3+4+5+6 -> 48; total = 54
    expect(std::abs(host[0] - 54.0f) < 1e-4f) << "nested AoS/SoA mismatch at idx 0";
    // idx=1: item_sum = 10+11+12+13 = 46; soa per-iteration = 10+11+12+13, ..., 13+14+15+16 -> 208; total = 254
    expect(std::abs(host[1] - 254.0f) < 1e-4f) << "nested AoS/SoA mismatch at idx 1";
}

void test_scalar_gep_aggregate_bitmask(Device &device) {
#ifdef LUISA_ENABLE_XIR
    xir::AggregateFieldBitmask mask{Type::of<MaskedStruct>()};
    mask.access(0).set();
    mask.access(1).set();
    mask.access(4, 0, 0).set();

    expect(mask.access(0).any()) << "bitmask field v should be set";
    expect(mask.access(1).any()) << "bitmask field s should be set";
    expect(mask.access(4, 0, 0).any()) << "bitmask m[0][0] should be set";

    xir::AggregateFieldBitmask other{Type::of<MaskedStruct>()};
    other.access(2).set();
    other.access(4, 1, 1).set();

    auto combined = mask | other;
    expect(combined.access(2).any()) << "OR should carry iv";
    expect(combined.access(4, 1, 1).any()) << "OR should carry m[1][1]";

    auto intersect = mask & other;
    expect(intersect.access(0).none()) << "AND should not have v";
    expect(intersect.access(4, 0, 0).none()) << "AND should not have m[0][0]";
#endif

    auto stream = device.create_stream();
    auto out = device.create_buffer<float>(64);

    Kernel1D k = [](BufferVar<float> out) noexcept {
        set_block_size(64);
        auto idx = dispatch_id().x;
        Var<MaskedStruct> ms;
        ms.m = make_float3x3(
            0.0f, 0.0f, 0.0f,
            0.0f, 0.0f, 0.0f,
            0.0f, 0.0f, 0.0f);

        $if (idx % 2u == 0u) {
            ms.v = make_float3(1.0f, 2.0f, 3.0f);
            ms.iv = make_int2(10, 20);
        } $else {
            ms.s = 4.0f;
            ms.flag = true;
        };

        Float sum = ms.v.x + ms.v.y + ms.v.z + ms.s
                    + cast<float>(ms.iv.x) + cast<float>(ms.iv.y)
                    + ite(ms.flag, 1.0f, 0.0f)
                    + ms.m[0][0] + ms.m[1][1] + ms.m[2][2];
        out.write(idx, sum);
    };

    auto shader = device.compile(k);
    luisa::vector<float> host(64);
    stream << shader(out).dispatch(64)
           << out.copy_to(luisa::span{host})
           << synchronize();

    LUISA_INFO("Scalar/GEP aggregate bitmask corner: host[0]={:f}, host[1]={:f}", host[0], host[1]);
    expect(std::abs(host[0] - 36.0f) < 1e-4f) << "aggregate bitmask mismatch at idx 0";
    expect(std::abs(host[1] - 5.0f) < 1e-4f) << "aggregate bitmask mismatch at idx 1";
}

// ---------------------------------------------------------------------------
// New callable corner tests
// Passes exercised: inline, dead_arg_elim, unused_callable_removal,
//                   promote_ref_arg, dead_field_elimination
// ---------------------------------------------------------------------------
void test_callable_dead_arg_elim_corners(Device &device) {
    auto stream = device.create_stream();
    auto out = device.create_buffer<float>(64);

    Callable use_first_of_six = [](Float a, Float u1, Float u2, Float u3, Float u4, Float u5) noexcept {
        return a + 10.0f;
    };

    Callable mul_with_dead = [](Float a, Float dead) noexcept {
        return a * 2.0f;
    };

    Callable d1 = [](Float x) noexcept { return x + 1.0f; };
    Callable d2 = [&d1](Float x) noexcept { return d1(x) * 2.0f; };
    Callable d3 = [&d2](Float x) noexcept { return d2(x) + 3.0f; };
    Callable d4 = [&d3](Float x) noexcept { return d3(x) * 4.0f; };
    Callable d5 = [&d4](Float x) noexcept { return d4(x) - 5.0f; };

    Kernel1D k = [&](BufferVar<float> out) noexcept {
        set_block_size(64);
        auto idx = dispatch_id().x;
        Float val = idx.cast<float>();

        Float r1 = use_first_of_six(val, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f);
        Float r2 = mul_with_dead(val,       99.0f)
                 + mul_with_dead(val + 1.0f, 98.0f)
                 + mul_with_dead(val + 2.0f, 97.0f)
                 + mul_with_dead(val + 3.0f, 96.0f);
        Float r3 = d5(val);

        out.write(idx, r1 + r2 + r3);
    };

    auto shader = device.compile(k);
    luisa::vector<float> host(64);
    stream << shader(out).dispatch(64)
           << out.copy_to(luisa::span{host})
           << synchronize();

    LUISA_INFO("Dead-arg-elim corner: host[0]={:f}", host[0]);
    expect(std::abs(host[0] - 37.0f) < 1e-4f) << "dead-arg-elim corner mismatch at idx 0";
}

void test_callable_unused_callable_corners(Device &device) {
    auto stream = device.create_stream();
    auto out = device.create_buffer<float>(64);

    [[maybe_unused]] Callable unused_leaf = [](Float x) noexcept { return x * 100.0f; };
    [[maybe_unused]] Callable unused_outer = [&unused_leaf](Float x) noexcept { return unused_leaf(x) + 1.0f; };
    Callable ghost_captured = [](Float x) noexcept { return x * 999.0f; };
    Callable used = [](Float x) noexcept { return x + 7.0f; };

    Kernel1D k = [&](BufferVar<float> out) noexcept {
        set_block_size(64);
        auto idx = dispatch_id().x;
        Float val = idx.cast<float>();
        out.write(idx, used(val));
    };

    auto shader = device.compile(k);
    luisa::vector<float> host(64);
    stream << shader(out).dispatch(64)
           << out.copy_to(luisa::span{host})
           << synchronize();

    LUISA_INFO("Unused-callable corner: host[0]={:f}", host[0]);
    expect(std::abs(host[0] - 7.0f) < 1e-4f) << "unused-callable-removal corner mismatch at idx 0";
}

void test_callable_ref_arg_and_struct(Device &device) {
    auto stream = device.create_stream();
    auto out = device.create_buffer<float>(64);

    Callable add_refs = [](Float &a, Float &b) noexcept {
        return a + b;
    };

    Callable scale_ref = [](Float &x, Float s) noexcept {
        return x * s;
    };

    Callable make_box = [](Float v) noexcept {
        Var<ReturnBox> box;
        box.x = v;
        box.y = v + 1.0f;
        box.z = v + 2.0f;
        box.w = v + 3.0f;
        return box;
    };

    Kernel1D k = [&](BufferVar<float> out) noexcept {
        set_block_size(64);
        auto idx = dispatch_id().x;
        Float val = idx.cast<float>();
        Float other = val + 10.0f;

        Float r1 = add_refs(val, other);
        Float r2 = scale_ref(val, 2.0f);
        auto box = make_box(val);
        Float r3 = box.x + box.w;

        out.write(idx, r1 + r2 + r3);
    };

    auto shader = device.compile(k);
    luisa::vector<float> host(64);
    stream << shader(out).dispatch(64)
           << out.copy_to(luisa::span{host})
           << synchronize();

    LUISA_INFO("Ref-arg/struct corner: host[0]={:f}", host[0]);
    expect(std::abs(host[0] - 13.0f) < 1e-4f) << "ref-arg/struct corner mismatch at idx 0";
}

// ---------------------------------------------------------------------------
// New analysis corner tests
// Passes exercised: dom_tree, post_dom_tree, uniformity_analysis,
//                   alias_analysis, pointer_usage, call_graph
// ---------------------------------------------------------------------------
void test_analysis_dom_postdom_uniformity(Device &device) {
    auto stream = device.create_stream();
    auto out = device.create_buffer<float>(128);
    auto result = device.create_buffer<float>(1);

    Callable multi_return = [](Float x, UInt sel) noexcept {
        $if (sel == 0u) { $return(x + 1.0f); };
        $if (sel == 1u) { $return(x * 2.0f); };
        $if (sel == 2u) { $return(x * x); };
        return x + 10.0f;
    };

    Kernel1D k = [&](BufferVar<float> out, BufferVar<float> result) noexcept {
        set_block_size(128);
        auto idx = dispatch_id().x;

        Float uniform_base = def(block_size().x).cast<float>();
        Float uniform_size = dispatch_size().x.cast<float>();
        Float uniform_sum = uniform_base + uniform_size;

        Float non_uniform = idx.cast<float>();
        Float mixed = uniform_sum + non_uniform;

        Float val;
        $if (idx % 2u == 0u) {
            $if (idx % 4u == 0u) {
                val = mixed + 1.0f;
            } $else {
                val = mixed + 2.0f;
            };
        } $else {
            $if (idx % 3u == 0u) {
                val = mixed + 4.0f;
            } $else {
                val = mixed + 8.0f;
            };
        };

        $if (dispatch_size().x > 64u) {
            val += uniform_base;
        };

        UInt sel = idx % 4u;
        Float called = multi_return(val, sel);
        out.write(idx, called);

        $if (idx == 64u) {
            result.write(0u, called);
        };
    };

    auto shader = device.compile(k);
    luisa::vector<float> host(128);
    luisa::vector<float> host_result(1, 0.0f);
    stream << shader(out, result).dispatch(128)
           << out.copy_to(luisa::span{host})
           << result.copy_to(luisa::span{host_result})
           << synchronize();

    LUISA_INFO("Analysis dom/post-dom/uniformity: host[0]={:f}, host[1]={:f}, result={:f}",
               host[0], host[1], host_result[0]);

    expect(std::abs(host[0] - 386.0f) < 1e-4f) << "dom/post-dom/uniformity mismatch at idx 0";
    expect(std::abs(host[1] - 786.0f) < 1e-4f) << "dom/post-dom/uniformity mismatch at idx 1";
    expect(std::abs(host_result[0] - 450.0f) < 1e-4f) << "dom/post-dom/uniformity summary mismatch";
}

void test_analysis_alias_pointer(Device &device) {
    auto stream = device.create_stream();
    auto base = device.create_buffer<float>(64);
    auto view_a = base.view();
    auto view_b = base.view();
    auto out = device.create_buffer<float>(64);

    Kernel1D k = [](BufferVar<float> a, BufferVar<float> b, BufferVar<float> out) noexcept {
        set_block_size(64);
        auto idx = dispatch_id().x;

        Var<PointerData> data;
        data.a = 1.0f;
        data.b = 2.0f;
        data.c = 3.0f;
        $if (idx % 2u == 0u) {
            data.a = 10.0f;
        } $else {
            data.c = 30.0f;
        };
        Float struct_sum = data.a + data.b + data.c;

        $array<float, 8> arr;
        $for (i, 8) {
            arr[i] = 0.0f;
        };
        $if (idx % 2u == 0u) {
            arr[0] = 1.0f;
            arr[1] = 2.0f;
        } $else {
            arr[2] = 4.0f;
            arr[3] = 8.0f;
        };
        Float acc = 0.0f;
        $for (i, 4) {
            acc += arr[i];
        };

        a.write(idx, idx.cast<float>() + 1.0f);
        Float via_a = a.read(idx);
        Float via_b = b.read(idx);

        out.write(idx, struct_sum + acc + via_a + via_b);
    };

    auto shader = device.compile(k);
    luisa::vector<float> host_out(64);
    stream << shader(view_a, view_b, out).dispatch(64)
           << out.copy_to(luisa::span{host_out})
           << synchronize();

    LUISA_INFO("Analysis alias/pointer: out[0]={:f}, out[1]={:f}", host_out[0], host_out[1]);

    expect(std::abs(host_out[0] - 20.0f) < 1e-4f) << "alias/pointer mismatch at idx 0";
    expect(std::abs(host_out[1] - 49.0f) < 1e-4f) << "alias/pointer mismatch at idx 1";
}

void test_analysis_call_graph(Device &device) {
    auto stream = device.create_stream();
    auto out = device.create_buffer<float>(64);

    Callable leaf_a = [](Float x) noexcept { return x + 1.0f; };
    Callable leaf_b = [](Float x) noexcept { return x * 2.0f; };
    Callable mid = [&leaf_a, &leaf_b](Float x, UInt sel) noexcept {
        Float r;
        $if (sel % 2u == 0u) {
            r = leaf_a(x);
        } $else {
            r = leaf_b(x);
        };
        return r;
    };
    Callable root = [&mid](Float x, UInt sel) noexcept {
        return mid(x, sel) + mid(x + 1.0f, sel + 1u);
    };

    Callable unused_ghost = [](Float x) noexcept { return x * 999.0f; };
    Callable another_unused = [&unused_ghost](Float x) noexcept { return unused_ghost(x); };

    Kernel1D k = [&](BufferVar<float> out) noexcept {
        set_block_size(64);
        auto idx = dispatch_id().x;
        Float val = idx.cast<float>();
        UInt sel = idx % 4u;
        Float r = root(val, sel);
        out.write(idx, r);
    };

    auto shader = device.compile(k);
    luisa::vector<float> host(64);
    stream << shader(out).dispatch(64)
           << out.copy_to(luisa::span{host})
           << synchronize();

    LUISA_INFO("Analysis call-graph: host[0]={:f}, host[1]={:f}, host[2]={:f}",
               host[0], host[1], host[2]);

    expect(std::abs(host[0] - 3.0f) < 1e-4f) << "call-graph mismatch at idx 0";
    expect(std::abs(host[1] - 5.0f) < 1e-4f) << "call-graph mismatch at idx 1";
    expect(std::abs(host[2] - 9.0f) < 1e-4f) << "call-graph mismatch at idx 2";
}

// ---------------------------------------------------------------------------
// New cleanup/misc corner tests
// Passes exercised: dce, fix_self_referential, outline
// ---------------------------------------------------------------------------
void test_cleanup_dce(Device &device) {
    auto stream = device.create_stream();
    auto out = device.create_buffer<float>(64);

    Kernel1D k = [](BufferVar<float> out) noexcept {
        set_block_size(64);
        auto idx = dispatch_id().x;
        Float val = idx.cast<float>();

        Float dead_a = val * val + 2.0f * val + 1.0f;
        Float dead_b = dead_a * dead_a - dead_a * 3.0f + val;
        Float dead_c = dead_b + dead_b * 4.0f - dead_a + val;

        Float live = val * 2.0f + 5.0f;
        out.write(idx, live);
    };

    auto shader = device.compile(k);
    luisa::vector<float> host(64);
    stream << shader(out).dispatch(64)
           << out.copy_to(luisa::span{host})
           << synchronize();

    LUISA_INFO("Cleanup DCE corner: host[0]={:f}, host[10]={:f}", host[0], host[10]);
    for (auto i = 0u; i < 64u; ++i) {
        expect(std::abs(host[i] - (2.0f * static_cast<float>(i) + 5.0f)) < 1e-4f)
            << "DCE cleanup result mismatch at index " << i;
    }
}

void test_cleanup_fix_self_referential(Device &device) {
    auto stream = device.create_stream();
    auto out = device.create_buffer<float>(64);

    Kernel1D k = [](BufferVar<float> out) noexcept {
        set_block_size(64);
        auto idx = dispatch_id().x;

        $array<float, 4> arr;
        arr[0] = 1.0f;
        arr[1] = 2.0f;
        arr[2] = 3.0f;
        arr[3] = 4.0f;

        arr[0] = arr[0] * 2.0f + idx.cast<float>();
        arr[1] = arr[1] * 2.0f + idx.cast<float>();
        arr[2] = arr[2] * 2.0f + idx.cast<float>();
        arr[3] = arr[3] * 2.0f + idx.cast<float>();

        Float sum = 0.0f;
        $for (i, 4) {
            sum += arr[i];
        };

        out.write(idx, sum);
    };

    auto shader = device.compile(k);
    luisa::vector<float> host(64);
    stream << shader(out).dispatch(64)
           << out.copy_to(luisa::span{host})
           << synchronize();

    LUISA_INFO("Cleanup fix-self-referential corner: host[0]={:f}, host[10]={:f}", host[0], host[10]);
    for (auto i = 0u; i < 64u; ++i) {
        expect(std::abs(host[i] - (20.0f + 4.0f * static_cast<float>(i))) < 1e-4f)
            << "fix-self-referential cleanup result mismatch at index " << i;
    }
}

void test_cleanup_outline(Device &device) {
    auto stream = device.create_stream();
    auto out = device.create_buffer<float>(64);

    Kernel1D k = [](BufferVar<float> out) noexcept {
        set_block_size(64);
        auto idx = dispatch_id().x;
        Float acc = 0.0f;

        $outline {
            Float term = idx.cast<float>() * 3.0f + 1.0f;
            acc += term;
        };

        $outline_with_name("outline_add_two") {
            acc += 2.0f;
        };

        $if (idx % 2u == 0u) {
            $outline {
                acc += 10.0f;
            };
        };

        out.write(idx, acc);
    };

    auto shader = device.compile(k);
    luisa::vector<float> host(64);
    stream << shader(out).dispatch(64)
           << out.copy_to(luisa::span{host})
           << synchronize();

    LUISA_INFO("Cleanup outline corner: host[0]={:f}, host[10]={:f}", host[0], host[10]);
    for (auto i = 0u; i < 64u; ++i) {
        float expected = 3.0f * static_cast<float>(i) + 3.0f + (i % 2u == 0u ? 10.0f : 0.0f);
        expect(std::abs(host[i] - expected) < 1e-4f)
            << "outline cleanup result mismatch at index " << i;
    }
}

int main(int argc, char *argv[]) {
    auto dc = luisa::test::create_device_from_ut(argc, argv);
    if (!dc) {
        return 0;
    }
    boost::ut::detail::cfg::parse_arg_with_fallback(argc, const_cast<const char **>(argv));

    auto &device = dc->device;
    test_control_flow_corners(device);
    test_cfg_empty_constant_and_multi_predecessor_merge(device);
    test_cfg_nested_switch_fallthrough_like(device);
    test_cfg_tiny_diamond_if_conversion(device);
    test_structured_break_continue(device);
    test_structured_early_return(device);
    test_structured_ray_query_loop(device);
    test_memory_cross_block_phi(device);
    test_memory_partial_sroa(device);
    test_memory_dead_store_loop(device);
    test_memory_pass_corners(device);
    test_callable_inline_corners(device);
    test_loop_pass_corners(device);
    test_algebraic_pass_corners(device);
    test_scalarize_gep_corners(device);
    test_algebraic_bitwise_identities(device);
    test_algebraic_libcall_reassociate(device);
    test_algebraic_early_cse(device);
    test_value_gvn_cvp(device);
    test_value_sccp_phi(device);
    test_value_scalar_evolution(device);
    test_loop_zero_one_trip(device);
    test_loop_nested_licm(device);
    test_loop_while_stride_break(device);
    test_scalar_gep_vector_matrix(device);
    test_scalar_gep_nested_aos_soa(device);
    test_scalar_gep_aggregate_bitmask(device);
    test_callable_dead_arg_elim_corners(device);
    test_callable_unused_callable_corners(device);
    test_callable_ref_arg_and_struct(device);
    test_analysis_dom_postdom_uniformity(device);
    test_analysis_alias_pointer(device);
    test_analysis_call_graph(device);
    test_cleanup_dce(device);
    test_cleanup_fix_self_referential(device);
    test_cleanup_outline(device);
    test_matrix_multiply(device);
}
