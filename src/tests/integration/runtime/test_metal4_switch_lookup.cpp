#include "ut/ut.hpp"
#include "test_device.h"

#include <array>
#include <luisa/luisa-compute.h>
#include <luisa/dsl/dispatch_indirect.h>

using namespace luisa;
using namespace luisa::compute;
using namespace boost::ut;

int main(int argc, char *argv[]) {
    auto dc = luisa::test::create_device_from_ut(argc, argv);
    if (!dc) { return 0; }
    // Dense, nonlinear, full-width values force an LLVM switch lookup table,
    // rather than an affine expression or a packed-integer bitmap. The table
    // is introduced by optimization, not by the XIR emitter.
    constexpr std::array<uint, 16u> values{
        0x159ad328u, 0x378c41f0u, 0x8a65120bu, 0xa9f72d14u,
        0x714832c5u, 0xf39216abu, 0x04615c32u, 0x539a874cu,
        0xc814e932u, 0x64f29ba1u, 0x3a2789c5u, 0xe91834d6u,
        0x98735621u, 0xb549c12du, 0x31e7a498u, 0x621bc9d3u};
    constexpr uint count = 257u;
    constexpr uint default_value = 0xdeadbeefu;
    Kernel1D kernel = [&](BufferUInt output, UInt bias) noexcept {
        set_block_size(32u);
        UInt index = (dispatch_x() + bias) % 20u;
        UInt result = default_value;
        $switch (index) {
            for (uint i = 0u; i < values.size(); i++) {
                $case (i) { result = values[i]; };
            }
        };
        output.write(dispatch_x(), result);
    };
    auto shader = dc->device.compile(kernel, ShaderOption{.enable_cache = false});
    Kernel1D prepare = [](IndirectDispatchBufferVar commands) noexcept {
        commands.set_dispatch_count(1u);
        commands.set_kernel(0u, make_uint3(32u, 1u, 1u), make_uint3(count, 1u, 1u));
    };
    auto prepare_shader = dc->device.compile(prepare);
    auto commands = dc->device.create_indirect_dispatch_buffer(1u);
    auto output = dc->device.create_buffer<uint>(count);
    auto stream = dc->device.create_stream();
    std::array<uint, count> actual{};
    for (auto indirect : {false, true}) {
        for (auto bias : {0u, 3u, 17u}) {
            if (indirect) {
                stream << prepare_shader(commands).dispatch(1u)
                       << shader(output, bias).dispatch(commands);
            } else {
                stream << shader(output, bias).dispatch(count);
            }
            stream << output.copy_to(luisa::span{actual}) << synchronize();
            for (uint i = 0u; i < count; i++) {
                auto index = (i + bias) % 20u;
                expect(actual[i] == (index < values.size() ? values[index] : default_value))
                    << "Optimizer-generated constant tables must link and execute in both AIR entries";
            }
        }
    }
}
