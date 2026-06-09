#include "ut/ut.hpp"
#include <luisa/core/logging.h>
#include <luisa/dsl/coro_func.h>
#include <luisa/dsl/sugar.h>
#include <luisa/coro/schedulers/state_machine.h>
#include <luisa/coro/schedulers/wavefront.h>
#include <luisa/coro/schedulers/persistent.h>
#include <luisa/runtime/context.h>
#include <luisa/runtime/device.h>
#include <luisa/runtime/stream.h>

using namespace luisa;
using namespace luisa::compute;
using namespace luisa::compute::coro;
using namespace boost::ut;
using namespace boost::ut::literals;

void reg_coro_all_schedulers(char *argv[]) {

    // ══════════════════════════════════════════════════════════════════
    // 1-suspend coroutine — all 3 schedulers
    // ══════════════════════════════════════════════════════════════════

    "cross_1suspend_state_machine"_test = [argv] {
        constexpr uint N = 256u;

        Context ctx{argv[0]};
        Device device = ctx.create_device("fallback");
        Stream stream = device.create_stream();

        auto coro = Coroutine<void()>([] {
            $suspend("s1");
        });

        LUISA_INFO("1-suspend coroutine: sub_count={}", coro.subroutine_count());
        expect(coro.subroutine_count() >= 2u);
        expect(coro.graph().node_count() >= 2u);

        StateMachineCoroScheduler<> scheduler{device, coro};
        LUISA_INFO("StateMachineCoroScheduler: dispatching {} threads", N);
        scheduler().dispatch(N)(stream);
        stream << synchronize();
        LUISA_INFO("StateMachineCoroScheduler: dispatch complete");
        expect(true);
    };

    "cross_1suspend_wavefront"_test = [argv] {
        constexpr uint N = 64u;

        Context ctx{argv[0]};
        Device device = ctx.create_device("fallback");
        Stream stream = device.create_stream();

        auto coro = Coroutine<void()>([] {
            $suspend("s1");
        });

        LUISA_INFO("1-suspend coroutine: sub_count={}", coro.subroutine_count());
        expect(coro.subroutine_count() >= 2u);
        expect(coro.graph().node_count() >= 2u);

        WavefrontCoroScheduler<> scheduler{device, coro};
        LUISA_INFO("WavefrontCoroScheduler: dispatching {} instances", N);
        scheduler().dispatch(N)(stream);
        stream << synchronize();
        LUISA_INFO("WavefrontCoroScheduler: dispatch complete");
        expect(true);
    };

    "cross_1suspend_persistent"_test = [argv] {
        constexpr uint N = 64u;

        Context ctx{argv[0]};
        Device device = ctx.create_device("fallback");
        Stream stream = device.create_stream();

        auto coro = Coroutine<void()>([] {
            $suspend("s1");
        });

        LUISA_INFO("1-suspend coroutine: sub_count={}", coro.subroutine_count());
        expect(coro.subroutine_count() >= 2u);
        expect(coro.graph().node_count() >= 2u);

        PersistentThreadsCoroScheduler<> scheduler{device, coro, N};
        LUISA_INFO("PersistentThreadsCoroScheduler: dispatching 1 block of {} threads", N);
        scheduler().dispatch(1u)(stream);
        stream << synchronize();
        LUISA_INFO("PersistentThreadsCoroScheduler: dispatch complete");
        expect(true);
    };

    // ══════════════════════════════════════════════════════════════════
    // 3-suspend coroutine — all 3 schedulers
    // ══════════════════════════════════════════════════════════════════

    "cross_3suspend_state_machine"_test = [argv] {
        constexpr uint N = 256u;

        Context ctx{argv[0]};
        Device device = ctx.create_device("fallback");
        Stream stream = device.create_stream();

        auto coro = Coroutine<void()>([] {
            $suspend("a");
            $suspend("b");
            $suspend("c");
        });

        LUISA_INFO("3-suspend coroutine: sub_count={}", coro.subroutine_count());
        expect(coro.subroutine_count() >= 2u);
        expect(coro.graph().node_count() >= 2u);

        StateMachineCoroScheduler<> scheduler{device, coro};
        LUISA_INFO("StateMachineCoroScheduler: dispatching {} threads", N);
        scheduler().dispatch(N)(stream);
        stream << synchronize();
        LUISA_INFO("StateMachineCoroScheduler: dispatch complete");
        expect(true);
    };

    "cross_3suspend_wavefront"_test = [argv] {
        constexpr uint N = 64u;

        Context ctx{argv[0]};
        Device device = ctx.create_device("fallback");
        Stream stream = device.create_stream();

        auto coro = Coroutine<void()>([] {
            $suspend("a");
            $suspend("b");
            $suspend("c");
        });

        LUISA_INFO("3-suspend coroutine: sub_count={}", coro.subroutine_count());
        expect(coro.subroutine_count() >= 2u);
        expect(coro.graph().node_count() >= 2u);

        WavefrontCoroScheduler<> scheduler{device, coro};
        LUISA_INFO("WavefrontCoroScheduler: dispatching {} instances", N);
        scheduler().dispatch(N)(stream);
        stream << synchronize();
        LUISA_INFO("WavefrontCoroScheduler: dispatch complete");
        expect(true);
    };

    "cross_3suspend_persistent"_test = [argv] {
        constexpr uint N = 64u;

        Context ctx{argv[0]};
        Device device = ctx.create_device("fallback");
        Stream stream = device.create_stream();

        auto coro = Coroutine<void()>([] {
            $suspend("a");
            $suspend("b");
            $suspend("c");
        });

        LUISA_INFO("3-suspend coroutine: sub_count={}", coro.subroutine_count());
        expect(coro.subroutine_count() >= 2u);
        expect(coro.graph().node_count() >= 2u);

        PersistentThreadsCoroScheduler<> scheduler{device, coro, N};
        LUISA_INFO("PersistentThreadsCoroScheduler: dispatching 1 block of {} threads", N);
        scheduler().dispatch(1u)(stream);
        stream << synchronize();
        LUISA_INFO("PersistentThreadsCoroScheduler: dispatch complete");
        expect(true);
    };
}

int main(int argc, char *argv[]) {
    reg_coro_all_schedulers(argv);
    return 0;
}
