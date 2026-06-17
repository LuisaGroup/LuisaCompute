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

int main(int argc, char *argv[]) {
    auto dc = luisa::test::create_device_from_ut(argc, argv);
    if (!dc) {
        return 0;
    }
    boost::ut::detail::cfg::parse_arg_with_fallback(argc, const_cast<const char **>(argv));

    auto &device = dc->device;
    LUISA_INFO("Testing switch-loop CFG patterns on backend: {}", device.backend_name());


    test_loop_carried_local_vector(device);

    LUISA_INFO("All switch-loop CFG tests completed.");
}
