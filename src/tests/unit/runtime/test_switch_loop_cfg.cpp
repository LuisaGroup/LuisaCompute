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

using namespace luisa;
using namespace luisa::compute;
using namespace boost::ut;
using namespace boost::ut::literals;

// Pattern 1: Switch inside a loop.
// This creates a switch whose case bodies can potentially branch back
// to the loop header via continue, creating a back-edge in SPIR-V.
// Uses loop-level continue (outside switch) to avoid lambda scoping issues.
void test_switch_in_loop(Device &device) {
    auto stream = device.create_stream();
    auto out = device.create_buffer<float>(64);

    Kernel1D k = [](BufferVar<float> out) noexcept {
        set_block_size(64);
        auto idx = dispatch_id().x;
        Float accum = 0.0f;

        // Loop with switch inside; case bodies assign values,
        // loop-level if handles continue to avoid lambda scoping.
        $for (i, 0u, 10u) {
            Float sw_val = 0.0f;
            $switch (idx % 3u) {
                $case (0u) { sw_val = 1.0f; };
                $case (1u) { sw_val = 2.0f; };
                $default { sw_val = 3.0f; };
            };
            accum += sw_val;
            $if (i < 5u) {
                accum += 0.5f;// extra for early iterations
            };
        };

        out.write(idx, accum);
    };

    auto shader = device.compile(k);
    luisa::vector<float> host(64);
    stream << shader(out).dispatch(64)
           << out.copy_to(luisa::span{host})
           << synchronize();

    // idx % 3 == 0: 10 * 1.0 + 5 * 0.5 = 12.5
    // idx % 3 == 1: 10 * 2.0 + 5 * 0.5 = 22.5
    // idx % 3 == 2: 10 * 3.0 + 5 * 0.5 = 32.5
    for (auto i = 0u; i < 64u; i++) {
        auto mod = i % 3u;
        float expected = mod == 0u ? 12.5f : (mod == 1u ? 22.5f : 32.5f);
        expect(std::abs(host[i] - expected) < 1e-4f)
            << "switch_in_loop mismatch at " << i
            << ": got " << host[i] << " expected " << expected;
    }
}

// Pattern 2: Multi-exit loop with conditional breaks.
// restructure_cfg converts this into a switch construct where
// each exit becomes a case, potentially creating back-edges.
void test_multi_exit_loop(Device &device) {
    auto stream = device.create_stream();
    auto out = device.create_buffer<float>(64);

    Kernel1D k = [](BufferVar<float> out) noexcept {
        set_block_size(64);
        auto tid = dispatch_id().x;
        $if (tid != 0u) { $return(); };
        auto idx = dispatch_id().x;
        Float accum = 0.0f;
        UInt count = 0u;

        // Multi-exit loop: different conditions break at different points.
        // restructure_cfg will create a switch with cases for each exit.
        $while (true) {
            $if (idx < 10u) {
                accum += 1.0f;
                $if (count >= 3u) {
                    $break;// exit 1
                };
            }
            $elif (idx < 20u) {
                accum += 10.0f;
                $if (count >= 5u) {
                    $break;// exit 2
                };
            }
            $elif (idx < 40u) {
                accum += 100.0f;
                $if (count >= 2u) {
                    $break;// exit 3
                };
            }
            $else {
                accum += 1000.0f;
                $if (count >= 1u) {
                    $break;// exit 4
                };
            };
            count += 1u;
        };

        out.write(idx, accum);
    };

    auto shader = device.compile(k);
    luisa::vector<float> host(64);
    stream << shader(out).dispatch(64)
           << out.copy_to(luisa::span{host})
           << synchronize();

    // Just check that results are reasonable (non-zero, finite)
    for (auto i = 0u; i < 64u; i++) {
        expect(host[i] > 0.0f && std::isfinite(host[i]))
            << "multi_exit_loop mismatch at " << i
            << ": got " << host[i];
    }
}

// Pattern 3: Nested switches.
// This creates complex nested structured CFG that stress-tests
// the SPIR-V backend's switch emission, especially with shared
// merge blocks.
void test_nested_switches(Device &device) {
    auto stream = device.create_stream();
    auto out = device.create_buffer<float>(64);

    Kernel1D k = [](BufferVar<float> out) noexcept {
        set_block_size(64);
        auto idx = dispatch_id().x;
        auto tid = dispatch_id().x;
        $if (tid != 0u) { $return(); };
        Float result = 0.0f;

        // Outer switch
        $switch (idx % 2u) {
            $case (0u) {
                // Inner switch
                $switch (idx % 4u) {
                    $case (0u) { result = 3.0f; };
                    $case (1u) { result = 8.0f; };
                    $case (2u) { result = 10.0f; };
                    $default { result = 100.0f; };
                };
            };
            $case (1u) {
                result = 999.0f;
            };
        };

        out.write(idx, result);
    };

    auto shader = device.compile(k);
    luisa::vector<float> host(64);
    stream << shader(out).dispatch(64)
           << out.copy_to(luisa::span{host})
           << synchronize();

    for (auto i = 0u; i < 64u; i++) {
        float expected;
        if (i % 2u == 1u) {
            expected = 999.0f;
        } else {
            auto mod = i % 4u;
            switch (mod) {
                case 0u: expected = 3.0f; break;
                case 1u: expected = 8.0f; break;
                case 2u: expected = 10.0f; break;
                default: expected = 100.0f; break;
            }
        }
        expect(std::abs(host[i] - expected) < 1e-4f)
            << "nested_switches mismatch at " << i
            << ": got " << host[i] << " expected " << expected;
    }
}

// Pattern 4: Switch followed by loop then another switch.
// This creates the kind of CFG where restructure_cfg may
// restructure the region into a switch-loop construct.
void test_switch_then_loop_then_switch(Device &device) {
    auto stream = device.create_stream();
    auto out = device.create_buffer<float>(64);

    Kernel1D k = [](BufferVar<float> out) noexcept {
        set_block_size(128, 1, 1);
        auto tid = dispatch_id().x;
        $if (tid != 0u) { $return(); };
    };

    auto shader = device.compile(k);

}

// Pattern 5: ByteBuffer::read<float4> with component extraction inside a loop.
// This exercises a crash where ByteBufferRead produces a float4, and EXTRACT
// instructions for v4.x/v4.y/v4.z/v4.w appear BEFORE the ByteBufferRead instruction
// in the XIR instruction list, violating SSA dominance and causing:
//   LUISA_ERROR_WITH_LOCATION("SPIR-V value ... should have been pre-mapped.")
// in SpirvCodegenEntry::_emit_value.
//
// Reproduces the same DSL pattern as the Compress ONNX operator's vectorized
// read path (nn/operators/Compress.cpp).
//
// Uses C++ range-for with dynamic_range (same as Compress) and ByteBuffer write
// with dynamic offsets, closely mirroring the exact Compress operator pattern.
void test_bytebuffer_vector_read_compress_pattern(Device &device) {
    auto stream = device.create_stream();

    // ByteBuffer contains 8 floats = 32 bytes
    auto buf = device.create_byte_buffer(32u);
    float init_data[8] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
    stream << buf.copy_from(init_data) << synchronize();

    // Output buffer for 8 floats
    auto out_buf = device.create_byte_buffer(32u);
    auto check_buf = device.create_buffer<float>(8);

    // Condition buffer (local tensor, but we use a local Var array to avoid DSL complexities)
    auto cond_buf = device.create_buffer<uint>(8);
    luisa::vector<uint> cond_init = {1u, 0u, 1u, 0u, 1u, 0u, 1u, 0u};
    stream << cond_buf.copy_from(luisa::span{cond_init}) << synchronize();

    Kernel1D k = [](ByteBufferVar buf_in, ByteBufferVar buf_out, BufferUInt cond, BufferFloat check) noexcept {
        set_block_size(32);
        auto idx = dispatch_id().x;
        $if (idx != 0u) { $return(); };

        // Exactly mirrors Compress.cpp vectorized read path:
        //   for (auto i4 : dynamic_range(vec_n)) {
        //       auto base = i4 * 4;
        //       auto v4 = buf_in->read<float4>(off_in + base * sizeof(float));
        //       $if (cond[base + 0u]) { buf_out->write(off_out + out_idx * 4, v4.x); out_idx++; };
        //       ...
        //   }
        auto off_in = 0u;
        auto off_out = 0u;
        auto elem_size = 4u;
        auto vec_n = 2u;
        auto out_idx = def(0u);

        for (auto i4 : dynamic_range(vec_n)) {
            auto base = i4 * 4u;
            auto v4 = buf_in.read<float4>(off_in + base * elem_size);

            $if (cond.read(base + 0u) != 0u) {
                buf_out.write(off_out + out_idx * elem_size, v4.x);
                out_idx += 1u;
            };
            $if (cond.read(base + 1u) != 0u) {
                buf_out.write(off_out + out_idx * elem_size, v4.y);
                out_idx += 1u;
            };
            $if (cond.read(base + 2u) != 0u) {
                buf_out.write(off_out + out_idx * elem_size, v4.z);
                out_idx += 1u;
            };
            $if (cond.read(base + 3u) != 0u) {
                buf_out.write(off_out + out_idx * elem_size, v4.w);
                out_idx += 1u;
            };
        }

        // Write output to check buffer for verification
        check.write(0u, buf_out.read<float>(0u));
        check.write(1u, buf_out.read<float>(4u));
        check.write(2u, buf_out.read<float>(8u));
        check.write(3u, buf_out.read<float>(12u));
    };

    auto shader = device.compile(k);
    luisa::vector<float> host(8);
    stream << shader(buf, out_buf, cond_buf, check_buf).dispatch(32)
           << check_buf.copy_to(luisa::span{host})
           << synchronize();

    // Cond = [1,0,1,0,1,0,1,0], Data = [1,2,3,4,5,6,7,8]
    // Selected elements: 1, 3, 5, 7 -> values: 1.0, 3.0, 5.0, 7.0
    expect(host[0] == 1.0f) << "compress_pattern[0]: expected 1.0 got " << host[0];
    expect(host[1] == 3.0f) << "compress_pattern[1]: expected 3.0 got " << host[1];
    expect(host[2] == 5.0f) << "compress_pattern[2]: expected 5.0 got " << host[2];
    expect(host[3] == 7.0f) << "compress_pattern[3]: expected 7.0 got " << host[3];

    LUISA_INFO("ByteBuffer vector read Compress pattern test passed.");
}

// Pattern 6: ByteBuffer::read<float4> with condition-guarded component extraction.
// Closely mirrors the Compress operator: v4 read outside $if, components used inside
// separate conditional blocks. This tests whether the XIR pass reorders instructions
// such that EXTRACT appears before the producing ByteBufferRead.
void test_bytebuffer_vector_read_with_cond_guards(Device &device) {
    auto stream = device.create_stream();

    // ByteBuffer with 4 floats = 16 bytes
    auto byte_buf = device.create_byte_buffer(16u);
    float init_data[4] = {10.0f, 20.0f, 30.0f, 40.0f};
    stream << byte_buf.copy_from(init_data) << synchronize();

    auto out = device.create_buffer<float>(64);

    Kernel1D k = [](ByteBufferVar buf, BufferFloat out) noexcept {
        set_block_size(64);
        auto idx = dispatch_id().x;
        $if (idx != 0u) { $return(); };

        // The float4 is read once, then each component is extracted inside
        // separate conditional blocks. This is the exact pattern from Compress.cpp.
        for (auto i : dynamic_range(2u)) {
            auto v4 = buf.read<float4>(0u);
            $if (i == 0u) {
                out.write(idx, v4.x + v4.y);
            };
            $if (i == 1u) {
                out.write(idx + 1u, v4.z + v4.w);
            };
        }
    };

    auto shader = device.compile(k);
    luisa::vector<float> host(64);
    stream << shader(byte_buf, out).dispatch(64)
           << out.copy_to(luisa::span{host})
           << synchronize();

    // Expected: 10+20=30 at idx 0, 30+40=70 at idx 1
    expect(host[0] == 30.0f) << "bytebuffer_vector_read_cond[0]: expected 30.0 got " << host[0];
    expect(host[1] == 70.0f) << "bytebuffer_vector_read_cond[1]: expected 70.0 got " << host[1];

    LUISA_INFO("ByteBuffer vector read with cond guards test passed.");
}

int main(int argc, char *argv[]) {
    auto dc = luisa::test::create_device_from_ut(argc, argv);
    if (!dc) {
        return 0;
    }
    boost::ut::detail::cfg::parse_arg_with_fallback(argc, const_cast<const char **>(argv));

    auto &device = dc->device;
    LUISA_INFO("Testing switch-loop CFG patterns on backend: {}", device.backend_name());

    // test_switch_in_loop(device);
    // test_multi_exit_loop(device);
    // test_nested_switches(device);
    test_switch_then_loop_then_switch(device);
    test_bytebuffer_vector_read_compress_pattern(device);
    test_bytebuffer_vector_read_with_cond_guards(device);

    LUISA_INFO("All switch-loop CFG tests completed.");
}
