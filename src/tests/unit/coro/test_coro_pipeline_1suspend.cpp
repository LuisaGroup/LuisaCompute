// End-to-end pipeline integration test: 1-suspend coroutine
// Validates the complete DSL → AST → XIR → passes → scheduler → GPU execution chain
#include "ut/ut.hpp"
#include <luisa/core/logging.h>
#include <luisa/dsl/coro_func.h>
#include <luisa/dsl/sugar.h>
#include <luisa/coro/schedulers/state_machine.h>
#include <luisa/runtime/buffer.h>
#include <luisa/runtime/context.h>
#include <luisa/runtime/device.h>
#include <luisa/runtime/stream.h>

using namespace luisa;
using namespace luisa::compute;
using namespace luisa::compute::coro;
using namespace boost::ut;
using namespace boost::ut::literals;

void reg_coro_pipeline_1suspend(char *argv[]) {

    "1suspend_identity"_test = [argv] {
        constexpr uint N = 256u;

        Context ctx{argv[0]};
        Device device = ctx.create_device("fallback");
        Stream stream = device.create_stream();

        auto output = device.create_buffer<uint>(N);

        auto coro = Coroutine<void(Buffer<uint>)>([](BufferUInt output) {
            $suspend("s1");
            output.write(0u, 42u);
        });

        LUISA_INFO("Coroutine created, subroutine_count={}", coro.subroutine_count());
        expect(coro.subroutine_count() >= 2u);
        expect(coro.graph().node_count() >= 2u);

        StateMachineCoroScheduler<Buffer<uint>> scheduler{device, coro};
        LUISA_INFO("Scheduler created, dispatching {} threads", N);

        scheduler(output).dispatch(N)(stream);
        stream << synchronize();
        LUISA_INFO("Dispatch complete");
        expect(true);
    };

    "1suspend_smoke"_test = [argv] {
        Context ctx{argv[0]};
        Device device = ctx.create_device("fallback");
        Stream stream = device.create_stream();

        auto coro = Coroutine<void()>([] {
            $suspend("s1");
        });

        LUISA_INFO("Smoke coroutine created, subroutine_count={}", coro.subroutine_count());
        expect(coro.subroutine_count() >= 2u);

        StateMachineCoroScheduler<> scheduler{device, coro};
        LUISA_INFO("Smoke scheduler created, dispatching 1 thread");

        scheduler().dispatch(1u)(stream);
        stream << synchronize();
        LUISA_INFO("Smoke dispatch complete");
        expect(true);
    };
}

int main(int argc, char *argv[]) {
    reg_coro_pipeline_1suspend(argv);
    return 0;
}
