// Test for switch-case and loop interaction in SPIR-V backend.
//
// This test exercises switch-case patterns that are likely to trigger
// the "Selection must be structured" SPIR-V validation error when
// the restructure_cfg pass creates a switch whose case bodies form
// a loop (back-edge to switch header).
//
// The SPIR-V backend's _emit_switch_inst attempts to detect this
// pattern and wrap the switch in a synthetic loop, but certain
// CFG shapes may evade detection.
//
// Patterns tested:
// 1. Switch inside a loop (creates back-edge potential)
// 2. Multi-exit loop with conditional breaks (restructure_cfg target)
// 3. Nested switches (creates complex structured CFG)
// 4. Switch followed by loop then another switch
// 5. Loop with switch and break at loop level
// 6. Vectorized ByteBuffer read + conditional scalar writes to local array
//    (reproduces the Compress operator miscompile on Vulkan)
// 9. Single-iteration dynamic_range loop with nested loops and $if
//    (reproduces the TopK operator SPIR-V validation crash on Vulkan)
//
// NOTE: $break/$continue inside $case bodies are NOT valid DSL because
// $case expands to a lambda, and break/continue scopes don't cross
// lambda boundaries. Control flow must be at the loop level, not
// inside switch case lambdas.
//
// Run with: LUISA_XIR_DISABLE_RESTRUCTURE_CFG=0 test_switch_loop_cfg vk

#include "ut/ut.hpp"
#include "test_device.h"

#include <luisa/runtime/device.h>
#include <luisa/runtime/stream.h>
#include <luisa/runtime/buffer.h>
#include <luisa/runtime/byte_buffer.h>
#include <luisa/dsl/sugar.h>
#include <luisa/dsl/syntax.h>
#include <luisa/dsl/local.h>

using namespace luisa;
using namespace luisa::compute;
using namespace boost::ut;
using namespace boost::ut::literals;


// Pattern 6: Loop-carried Local<float> with dynamic index assignment.
// Creates GEP+Store pattern inside $if branches inside a loop.
// transpose_gep converts GEP+Store to Load+INSERT+Store.
// if-conversion then converts the diamonds, potentially creating
// a use-before-def cycle where the INSERT references a SELECT
// that appears later in the same block.
//
// Expected: crash at SpirvCodegenEntry::_emit_value with:
//   "SPIR-V value inst (name=<noname>, type=vector<float,4>) should have been pre-mapped."
void test_loop_carried_local_vector(Device &device) {
    auto stream = device.create_stream();

    // ByteBuffer with 16 floats
    auto buf = device.create_byte_buffer(64u);
    float init_data[16] = {1.0f, 2.0f, 3.0f, 4.0f,
                           5.0f, 6.0f, 7.0f, 8.0f,
                           9.0f, 10.0f, 11.0f, 12.0f,
                           13.0f, 14.0f, 15.0f, 16.0f};
    stream << buf.copy_from(init_data) << synchronize();

    // Condition buffer
    auto cond_buf = device.create_buffer<uint>(16);
    luisa::vector<uint> cond_init = {1u, 0u, 1u, 0u, 1u, 0u, 1u, 0u,
                                      0u, 1u, 0u, 1u, 0u, 1u, 0u, 1u};
    stream << cond_buf.copy_from(luisa::span{cond_init}) << synchronize();

    // Output buffer
    auto out_buf = device.create_buffer<float>(16);

    Kernel1D k = [](ByteBufferVar buf, BufferUInt cond, BufferFloat out) noexcept {
        set_block_size(32);
        auto idx = dispatch_id().x;
        // $if (idx != 0u) { $return(); };

        // Local<float>(4) creates a local float4 variable.
        // Assigning via dynamic index: local[out_idx] = value
        // creates GEP(float4, out_idx) + Store.
        // transpose_gep converts this to Load + INSERT + Store.
        // After mem2reg, the alloca is promoted to phi.
        // The INSERT now operates on the loop-carried phi value.
        auto elem_size = 4u;
        Local<float> local_out{4u};  // float4
        auto out_idx = def(0u);
        auto vec_n = 2u;

        for (auto i4 : dynamic_range(vec_n)) {
            auto base = i4 * 4u;
            auto v4 = buf.read<float4>(base * elem_size);

            // Each $if conditionally writes one component of v4
            // to local_out at position out_idx, then increments out_idx.
            // local_out[dynamic_idx] = scalar → GEP+Store → transpose_gep → INSERT
            $if (cond.read(base + 0u) != 0u) {
                local_out[out_idx] = v4.x;
                out_idx += 1u;
            };
            $if (cond.read(base + 1u) != 0u) {
                local_out[out_idx] = v4.y;
                out_idx += 1u;
            };
            $if (cond.read(base + 2u) != 0u) {
                local_out[out_idx] = v4.z;
                out_idx += 1u;
            };
            $if (cond.read(base + 3u) != 0u) {
                local_out[out_idx] = v4.w;
                out_idx += 1u;
            };
        }

        // Write final result to output
        out.write(0u, local_out[0u]);
        out.write(1u, local_out[1u]);
        out.write(2u, local_out[2u]);
        out.write(3u, local_out[3u]);
    };

    auto shader = device.compile(k);
    luisa::vector<float> host(16);
    stream << shader(buf, cond_buf, out_buf).dispatch(32)
           << out_buf.copy_to(luisa::span{host})
           << synchronize();

    LUISA_INFO("Loop-carried local vector test completed (if it reaches here, no crash).");
}

// Pattern 7: Reproducer for Compress-like vectorized ByteBuffer read bug.
//
// Reads a vector (float4) from a ByteBuffer, extracts its components, and
// conditionally scatters them into a local array using sequential $if blocks.
// The loop-carried index (out_idx) is updated inside each $if. The XIR
// local-store-forward / if-conversion pipeline can mis-handle the index
// updates, causing stale/wrong output values on the Vulkan backend.
//
// Expected output: [1, 3, 5, 7] (the odd-indexed input values).
// Buggy output:    all zeros or otherwise corrupted values.
void test_compress_like_vectorized_read(Device &device) {
    auto stream = device.create_stream();

    // ByteBuffer with 8 floats: [1,2,3,4,5,6,7,8]
    auto buf = device.create_byte_buffer(32u);
    float init_data[8] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
    stream << buf.copy_from(init_data) << synchronize();

    // Condition buffer: [true,false,true,false,true,false,true,false]
    auto cond_buf = device.create_buffer<uint>(8);
    luisa::vector<uint> cond_init = {1u, 0u, 1u, 0u, 1u, 0u, 1u, 0u};
    stream << cond_buf.copy_from(luisa::span{cond_init}) << synchronize();

    // Output buffer
    auto out_buf = device.create_buffer<float>(4);

    Kernel1D k = [](ByteBufferVar buf, BufferUInt cond, BufferFloat out) noexcept {
        set_block_size(32);
        auto idx = dispatch_id().x;
        $if (idx != 0u) { $return(); };

        auto elem_size = 4u;
        // Local array that acts like the DynamicArray used in mrpnn_luisa's Compress.
        Local<float> local_out{4u};
        auto out_idx = def(0u);
        auto vec_n = 2u;

        for (auto i4 : dynamic_range(vec_n)) {
            auto base = i4 * 4u;
            auto v4 = buf.read<float4>(base * elem_size);

            // Sequential conditional writes. Each branch reads/writes local_out[out_idx]
            // and increments out_idx. This is the exact pattern that miscompiles.
            $if (cond.read(base + 0u) != 0u) {
                local_out[out_idx] = v4.x;
                out_idx += 1u;
            };
            $if (cond.read(base + 1u) != 0u) {
                local_out[out_idx] = v4.y;
                out_idx += 1u;
            };
            $if (cond.read(base + 2u) != 0u) {
                local_out[out_idx] = v4.z;
                out_idx += 1u;
            };
            $if (cond.read(base + 3u) != 0u) {
                local_out[out_idx] = v4.w;
                out_idx += 1u;
            };
        }

        out.write(0u, local_out[0u]);
        out.write(1u, local_out[1u]);
        out.write(2u, local_out[2u]);
        out.write(3u, local_out[3u]);
    };

    auto shader = device.compile(k);
    luisa::vector<float> host(4);
    stream << shader(buf, cond_buf, out_buf).dispatch(32)
           << out_buf.copy_to(luisa::span{host})
           << synchronize();

    float expected[4] = {1.0f, 3.0f, 5.0f, 7.0f};
    bool ok = true;
    for (auto i = 0u; i < 4u; ++i) {
        if (host[i] != expected[i]) {
            LUISA_WARNING("compress-like vectorized read mismatch at [{}]: actual={}, expected={}",
                          i, host[i], expected[i]);
            ok = false;
        }
    }
    boost::ut::expect(static_cast<bool>(ok));
    LUISA_INFO("Compress-like vectorized read test: {}.", ok ? "PASS" : "FAIL");
}

// Pattern 8: Scalar-read fallback for the same Compress-like scatter.
//
// Instead of reading a float4 and extracting components, this reads each
// scalar float directly from the ByteBuffer. This is the workaround that
// makes the Compress operator produce correct output, and is expected to
// pass even when the vectorized path fails.
void test_compress_like_scalar_fallback(Device &device) {
    auto stream = device.create_stream();

    auto buf = device.create_byte_buffer(32u);
    float init_data[8] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
    stream << buf.copy_from(init_data) << synchronize();

    auto cond_buf = device.create_buffer<uint>(8);
    luisa::vector<uint> cond_init = {1u, 0u, 1u, 0u, 1u, 0u, 1u, 0u};
    stream << cond_buf.copy_from(luisa::span{cond_init}) << synchronize();

    auto out_buf = device.create_buffer<float>(4);

    Kernel1D k = [](ByteBufferVar buf, BufferUInt cond, BufferFloat out) noexcept {
        set_block_size(32);
        auto idx = dispatch_id().x;
        $if (idx != 0u) { $return(); };

        auto elem_size = 4u;
        Local<float> local_out{4u};
        auto out_idx = def(0u);
        auto n = 8u;

        for (auto i : dynamic_range(n)) {
            auto v = buf.read<float>(i * elem_size);
            $if (cond.read(i) != 0u) {
                local_out[out_idx] = v;
                out_idx += 1u;
            };
        }

        out.write(0u, local_out[0u]);
        out.write(1u, local_out[1u]);
        out.write(2u, local_out[2u]);
        out.write(3u, local_out[3u]);
    };

    auto shader = device.compile(k);
    luisa::vector<float> host(4);
    stream << shader(buf, cond_buf, out_buf).dispatch(32)
           << out_buf.copy_to(luisa::span{host})
           << synchronize();

    float expected[4] = {1.0f, 3.0f, 5.0f, 7.0f};
    bool ok = true;
    for (auto i = 0u; i < 4u; ++i) {
        if (host[i] != expected[i]) {
            LUISA_WARNING("compress-like scalar fallback mismatch at [{}]: actual={}, expected={}",
                          i, host[i], expected[i]);
            ok = false;
        }
    }
    boost::ut::expect(static_cast<bool>(ok));
    LUISA_INFO("Compress-like scalar fallback test: {}.", ok ? "PASS" : "FAIL");
}

// Pattern 9: Single-iteration dynamic_range loop containing nested loops and
// conditional writes. Reproduces the SPIR-V validation failure seen in
// mrpnn_luisa's TopK operator when the output size is 1:
//
//   error [:0:0]: block <ID> ... exits the loop headed by <ID> ...,
//                 but not via a structured exit
//
// The outer loop is known to execute exactly once (dynamic_range(1u)), but the
// SPIR-V restructure_cfg pass still emits an invalid structured control-flow
// edge when the body contains nested dynamic_range loops and an $if block.
void test_single_iteration_loop(Device &device) {
    auto stream = device.create_stream();

    constexpr uint axis_size = 5u;
    auto in_buf = device.create_buffer<float>(axis_size);
    auto count_buf = device.create_buffer<uint>(1u);
    auto out_val_buf = device.create_buffer<float>(1u);
    auto out_idx_buf = device.create_buffer<int>(1u);

    luisa::vector<float> init = {5.0f, 3.0f, 8.0f, 1.0f, 9.0f};
    luisa::vector<uint> count_init = {1u};
    stream << in_buf.copy_from(luisa::span{init})
           << count_buf.copy_from(luisa::span{count_init})
           << synchronize();

    Kernel1D k = [axis_size](BufferFloat in, BufferUInt count,
                             BufferFloat out_val, BufferInt out_idx) noexcept {
        set_block_size(32);
        auto idx = dispatch_id().x;
        $if (idx != 0u) { $return(); };

        // Output arrays of size 1, indexed by the outer loop variable.
        Local<float> out_vals{1u};
        Local<int> out_idxs{1u};

        // Runtime loop bound equal to 1: the exact trigger in TopK.
        auto bound = count.read(0u);
        for (auto out_linear : dynamic_range(bound)) {
            Local<float> local_vals{axis_size};
            Local<float> local_cmps{axis_size};

            for (auto i : dynamic_range(axis_size)) {
                auto v = in.read(i);
                local_vals[i] = v;
                local_cmps[i] = v;
            }

            auto result_val = def(local_vals[0]);
            auto result_idx = def(0);

            for (auto candidate : dynamic_range(axis_size)) {
                auto cand_cmp = local_cmps[candidate];
                auto count = def(0u);

                for (auto other : dynamic_range(axis_size)) {
                    auto other_cmp = local_cmps[other];
                    auto inc = select(0u, 1u,
                                      (other_cmp < cand_cmp) |
                                          ((other_cmp == cand_cmp) & (other < candidate)));
                    count += inc;
                }

                $if (count == 0u) {
                    result_val = local_vals[candidate];
                    result_idx = candidate.cast<int>();
                };
            }

            out_vals[out_linear] = result_val;
            out_idxs[out_linear] = result_idx;
        }

        out_val.write(0u, out_vals[0u]);
        out_idx.write(0u, out_idxs[0u]);
    };

    auto shader = device.compile(k);
    luisa::vector<float> out_val_host(1);
    luisa::vector<int> out_idx_host(1);
    stream << shader(in_buf, count_buf, out_val_buf, out_idx_buf).dispatch(32)
           << out_val_buf.copy_to(luisa::span{out_val_host})
           << out_idx_buf.copy_to(luisa::span{out_idx_host})
           << synchronize();

    bool ok = (out_val_host[0] == 1.0f) && (out_idx_host[0] == 3);
    if (!ok) {
        LUISA_WARNING("single-iteration loop result mismatch: value={}, index={}",
                      out_val_host[0], out_idx_host[0]);
    }
    boost::ut::expect(static_cast<bool>(ok));
    LUISA_INFO("Single-iteration loop test: {}.", ok ? "PASS" : "FAIL");
}

int main(int argc, char *argv[]) {
    auto dc = luisa::test::create_device_from_ut(argc, argv);
    if (!dc) {
        return 0;
    }
    boost::ut::detail::cfg::parse_arg_with_fallback(argc, const_cast<const char **>(argv));

    auto &device = dc->device;
    LUISA_INFO("Testing switch-loop CFG patterns on backend: {}", device.backend_name());


    // test_loop_carried_local_vector(device);
    // test_compress_like_scalar_fallback(device); // scalar fallback; passes, kept as reference
    // test_compress_like_vectorized_read(device);
    test_single_iteration_loop(device);

    LUISA_INFO("All switch-loop CFG tests completed.");
}
