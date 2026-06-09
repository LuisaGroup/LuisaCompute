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
        // Verify the scheduler inherits from CoroScheduler and can be
        // instantiated with compatible types (compile-time check).
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

        // Coroutine with 3 suspends
        Buffer<int> output = device.create_buffer<int>(N);

        auto coro = Coroutine<void(int)>([](Var<int> multiplier) {
            $suspend("a");
            $suspend("b");
            $suspend("c");
        });

        LUISA_INFO("Coroutine created, subroutine_count={}", coro.subroutine_count());
        expect(coro.subroutine_count() >= 2u);
        expect(coro.graph().node_count() >= 2u);

        // Debug: print node tokens
        for (size_t i = 0u; i < coro.graph().node_count(); ++i) {
            auto &node = coro.graph().node(i);
            LUISA_INFO("  Node[{}]: token={}, name='{}', terminal={}",
                       i, node.token, node.name, node.is_terminal);
        }

        StateMachineCoroScheduler<int> scheduler{device, coro};
        LUISA_INFO("Scheduler created, dispatching {} threads", N);

        // Use explicit dispatch syntax: scheduler(args...).dispatch(size)(stream)
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
