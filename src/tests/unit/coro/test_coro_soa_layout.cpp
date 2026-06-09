#include "ut/ut.hpp"
#include <luisa/core/logging.h>
#include <luisa/dsl/coro_func.h>
#include <luisa/dsl/sugar.h>
#include <luisa/coro/schedulers/wavefront.h>
#include <luisa/runtime/buffer.h>
#include <luisa/runtime/context.h>
#include <luisa/runtime/device.h>
#include <luisa/runtime/stream.h>

using namespace luisa;
using namespace luisa::compute;
using namespace luisa::compute::coro;
using namespace boost::ut;
using namespace boost::ut::literals;

void reg_coro_soa_layout(char *argv[]) {

    "soa_layout_enum_values"_test = [] {
        expect(static_cast<uint8_t>(FrameLayout::AoS) == 0u);
        expect(static_cast<uint8_t>(FrameLayout::SoA) == 1u);
    };

    "soa_layout_constructs_and_has_correct_layout"_test = [argv] {
        Context ctx{argv[0]};
        Device device = ctx.create_device("fallback");

        auto coro = Coroutine<void()>([] {
            $suspend("s1");
        });

        WavefrontCoroScheduler<> aos_scheduler{device, coro, FrameLayout::AoS};
        expect(aos_scheduler.layout() == FrameLayout::AoS);

        WavefrontCoroScheduler<> soa_scheduler{device, coro, FrameLayout::SoA};
        expect(soa_scheduler.layout() == FrameLayout::SoA);
    };

    "soa_layout_compiles_and_runs"_test = [argv] {
        constexpr uint N = 64u;

        Context ctx{argv[0]};
        Device device = ctx.create_device("fallback");
        Stream stream = device.create_stream();

        auto coro = Coroutine<void()>([] {
            $suspend("s1");
        });

        LUISA_INFO("Coroutine created, sub_count={}", coro.subroutine_count());
        expect(coro.subroutine_count() >= 2u);

        WavefrontCoroScheduler<> scheduler{device, coro, FrameLayout::SoA};
        LUISA_INFO("SoA Wavefront scheduler created, dispatching {} instances", N);

        scheduler().dispatch(N)(stream);
        stream << synchronize();
        LUISA_INFO("SoA dispatch complete");
        expect(true);
    };

    "soa_layout_1suspend_with_buffer"_test = [argv] {
        constexpr uint N = 128u;

        Context ctx{argv[0]};
        Device device = ctx.create_device("fallback");
        Stream stream = device.create_stream();

        auto output = device.create_buffer<uint>(N);

        auto coro = Coroutine<void(Buffer<uint>)>([](BufferUInt buf) {
            $suspend("s1");
            buf.write(0u, 42u);
        });

        LUISA_INFO("Coroutine created, sub_count={}", coro.subroutine_count());
        expect(coro.subroutine_count() >= 2u);

        WavefrontCoroScheduler<Buffer<uint>> scheduler{device, coro, FrameLayout::SoA};
        LUISA_INFO("SoA Wavefront scheduler created, dispatching {} instances", N);

        scheduler(output).dispatch(N)(stream);
        stream << synchronize();
        LUISA_INFO("SoA dispatch complete");
        expect(true);
    };

    "soa_layout_3suspend_smoke"_test = [argv] {
        constexpr uint N = 64u;

        Context ctx{argv[0]};
        Device device = ctx.create_device("fallback");
        Stream stream = device.create_stream();

        auto coro = Coroutine<void(int)>([](Var<int> unused) {
            $suspend("a");
            $suspend("b");
            $suspend("c");
        });

        LUISA_INFO("Coroutine created, sub_count={}", coro.subroutine_count());
        expect(coro.subroutine_count() >= 2u);

        WavefrontCoroScheduler<int> scheduler{device, coro, FrameLayout::SoA};
        LUISA_INFO("SoA Wavefront scheduler created, dispatching {} instances", N);

        scheduler(42).dispatch(N)(stream);
        stream << synchronize();
        LUISA_INFO("SoA dispatch complete");
        expect(true);
    };
}

int main(int argc, char *argv[]) {
    reg_coro_soa_layout(argv);
    return 0;
}
