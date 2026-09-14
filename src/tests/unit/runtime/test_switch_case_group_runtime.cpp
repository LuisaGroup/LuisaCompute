// Execution coverage for shared switch bodies, including exits and suspension.
#include "ut/ut.hpp"
#include "test_device.h"

#include <luisa/coro/schedulers/state_machine.h>
#include <luisa/coro/schedulers/wavefront.h>
#include <luisa/dsl/sugar.h>
#include <luisa/runtime/buffer.h>
#include <luisa/runtime/stream.h>

#include <array>

using namespace luisa;
using namespace luisa::compute;
using namespace luisa::compute::coro;
using namespace boost::ut;

namespace {

void test_exits(Device &device) {
    const std::array<int, 10u> selectors{-7, 0, 25, 8, 9, 42, 43, -1, 11, 200};
    const std::array<uint, 10u> expected{335u, 335u, 335u, 408u, 408u,
                                         999u, 999u, 404u, 404u, 404u};
    auto input = device.create_buffer<int>(selectors.size());
    auto output = device.create_buffer<uint>(selectors.size());
    Kernel1D kernel = [](BufferInt input, BufferUInt output) {
        auto tid = dispatch_x();
        auto selector = input.read(tid);
        UInt sum = 0u;
        $for (j, 4u) {
            $switch (selector) {
                $case (-7, 0, 25) {
                    $if (j == 1u) { $continue; };
                    sum += j + 10u;
                    $break;
                };
                $case (8, 9) { sum += 2u; };
                $case (42, 43) {
                    output.write(tid, 999u);
                    $return();
                };
                $default { sum += 1u; };
            };
            sum += 100u;
        };
        output.write(tid, sum);
    };
    auto shader = device.compile(kernel, ShaderOption{.enable_cache = false});
    auto stream = device.create_stream();
    std::array<uint, 10u> actual{};
    stream << input.copy_from(luisa::span{selectors})
           << shader(input, output).dispatch(selectors.size())
           << output.copy_to(luisa::span{actual}) << synchronize();
    for (auto i = 0u; i < actual.size(); i++) { expect(actual[i] == expected[i]); }
}

template<typename T>
void test_labels(Device &device, const std::array<T, 6u> &selectors) {
    auto input = device.create_buffer<T>(selectors.size());
    auto output = device.create_buffer<uint>(selectors.size());
    for (auto grouped : {false, true}) {
        LUISA_INFO("Switch selector={} grouped={}", Type::of<T>()->description(), grouped);
        Kernel1D kernel = [&](Var<Buffer<T>> input, BufferUInt output) {
            auto tid = dispatch_x();
            $switch (input.read(tid)) {
                if (grouped) {
                    $case (selectors[0], selectors[1], selectors[2]) { output.write(tid, 17u); };
                } else {
                    for (auto i = 0u; i < 3u; i++) {
                        $case (selectors[i]) { output.write(tid, 17u); };
                    }
                }
                $case (selectors[3]) { output.write(tid, 29u); };
                $default { output.write(tid, 41u); };
            };
        };
        auto shader = device.compile(kernel, ShaderOption{.enable_cache = false});
        auto stream = device.create_stream();
        std::array<uint, 6u> actual{};
        stream << input.copy_from(luisa::span{selectors})
               << shader(input, output).dispatch(selectors.size())
               << output.copy_to(luisa::span{actual}) << synchronize();
        const std::array<uint, 6u> expected{17u, 17u, 17u, 29u, 41u, 41u};
        for (auto i = 0u; i < actual.size(); i++) { expect(actual[i] == expected[i]); }
    }
}

void test_suspension(Device &device) {
    constexpr auto count = 64u;
    Coroutine<void(Buffer<uint>)> coroutine = [](BufferUInt output) {
        auto tid = dispatch_x();
        auto selector = tid % 4u;
        UInt sum = 0u;
        $for (j, 3u) {
            $switch (selector) {
                $case (0u, 2u) {
                    sum += j + 1u;
                    $suspend("shared_case");
                    sum += selector;
                };
                $case (1u) { sum += 10u; };
                $default { sum += 1u; };
            };
        };
        output.write(tid, sum);
    };
    expect(coroutine.graph().node_by_name("shared_case") != nullptr);
    auto output = device.create_buffer<uint>(count);
    auto stream = device.create_stream();
    for (auto wavefront : {false, true}) {
        if (wavefront) {
            WavefrontCoroScheduler<Buffer<uint>> scheduler{
                device, coroutine, WavefrontCoroSchedulerConfig{.thread_count = 32u, .gather_by_sorting = false}};
            stream << scheduler(output).dispatch(count) << synchronize();
        } else {
            StateMachineCoroScheduler<Buffer<uint>> scheduler{device, coroutine};
            stream << scheduler(output).dispatch(count) << synchronize();
        }
        std::array<uint, count> actual{};
        stream << output.copy_to(luisa::span{actual}) << synchronize();
        const std::array<uint, 4u> expected{6u, 30u, 12u, 3u};
        for (auto i = 0u; i < count; i++) { expect(actual[i] == expected[i % 4u]); }
    }
}

}// namespace

int main(int argc, char *argv[]) {
    auto dc = luisa::test::create_device(argc, argv);
    test_exits(dc.device);
    test_labels<int64_t>(dc.device, {-1, 0xffffffffll, 0x1234567800000000ll, 3, -2, 0});
    test_labels<uint64_t>(dc.device, {0xffffffffffffffffull, 0xffffffffull, 0x1234567800000000ull, 3u, 4u, 0u});
    test_labels<int16_t>(dc.device, {-32768, -1, 32767, 3, -2, 0});
    test_labels<uint16_t>(dc.device, {65535u, 32768u, 1u, 3u, 4u, 0u});
    test_labels<int8_t>(dc.device, {-128, -1, 127, 3, -2, 0});
    test_labels<uint8_t>(dc.device, {255u, 128u, 1u, 3u, 4u, 0u});
    test_suspension(dc.device);
}
