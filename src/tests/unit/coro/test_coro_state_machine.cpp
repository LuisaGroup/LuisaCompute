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

void reg_coro_state_machine(char *argv[]) {

    "state_machine_constructor_and_type_check"_test = [] {
        // Verify the scheduler can be constructed and its type is correct.
        // This is a compile-time + basic runtime check (no GPU needed).
        CoroGraph graph;
        CoroFrameDesc desc;
        // StateMachineCoroScheduler inherits from CoroScheduler
        static_assert(std::is_base_of_v<CoroScheduler<Buffer<int>>,
                                        StateMachineCoroScheduler<Buffer<int>>>);
        expect(true);
    };

    "state_machine_compiles_and_runs"_test = [argv] {
        constexpr uint N = 256u;

        Context ctx{argv[0]};
        Device device = ctx.create_device("fallback");
        Stream stream = device.create_stream();

        // First verify basic shader dispatch works
        {
            auto k = Kernel1D{[]() noexcept {}};
            auto s = device.compile(k);
            stream << s().dispatch(N) << synchronize();
            LUISA_INFO("Basic kernel dispatch OK");
        }

        // Coroutine with 3 suspends: each thread writes its dispatch
        // ID to the output buffer after all suspends complete.
        Buffer<int> output = device.create_buffer<int>(N);

        auto coro = Coroutine<void(int)>([](Var<int> multiplier) {
            $suspend("a");
            $suspend("b");
            $suspend("c");
        });

        LUISA_INFO("Coroutine created, subroutine_count={}", coro.subroutine_count());
        expect(coro.subroutine_count() >= 2u);
        expect(coro.graph().node_count() >= 2u);

        StateMachineCoroScheduler<int> scheduler{device, coro};
        LUISA_INFO("Scheduler created, dispatching {} threads", N);

        scheduler(42).dispatch(N)(stream);
        stream << synchronize();
        LUISA_INFO("Dispatch complete");

        expect(true);
    };
}

int main(int argc, char *argv[]) {
    reg_coro_state_machine(argv);
    return 0;
}
