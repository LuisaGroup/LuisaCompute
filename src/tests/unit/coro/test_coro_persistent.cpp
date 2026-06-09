#include "ut/ut.hpp"
#include <luisa/core/logging.h>
#include <luisa/dsl/coro_func.h>
#include <luisa/dsl/sugar.h>
#include <luisa/coro/schedulers/persistent.h>
#include <luisa/runtime/buffer.h>
#include <luisa/runtime/context.h>
#include <luisa/runtime/device.h>
#include <luisa/runtime/stream.h>

using namespace luisa;
using namespace luisa::compute;
using namespace luisa::compute::coro;
using namespace boost::ut;
using namespace boost::ut::literals;

void reg_coro_persistent(char *argv[]) {

    "persistent_constructor_and_type_check"_test = [] {
        static_assert(std::is_base_of_v<CoroScheduler<Buffer<int>>,
                                        PersistentThreadsCoroScheduler<Buffer<int>>>);
        static_assert(std::is_base_of_v<CoroScheduler<>,
                                        PersistentThreadsCoroScheduler<>>);
        expect(true);
    };

    "persistent_compiles_and_runs"_test = [argv] {
        constexpr uint N = 64u;

        Context ctx{argv[0]};
        Device device = ctx.create_device("fallback");
        Stream stream = device.create_stream();

        // Basic shader dispatch smoke test
        {
            auto k = Kernel1D{[]() noexcept {}};
            auto s = device.compile(k);
            stream << s().dispatch(N) << synchronize();
            LUISA_INFO("Basic kernel dispatch OK");
        }

        // Coroutine with 2 suspends
        auto coro = Coroutine<void()>([] {
            $suspend("s1");
            $suspend("s2");
        });

        LUISA_INFO("Coroutine created, sub_count={}", coro.subroutine_count());
        expect(coro.subroutine_count() >= 2u);
        expect(coro.graph().node_count() >= 2u);

        PersistentThreadsCoroScheduler<> scheduler{device, coro, N};
        LUISA_INFO("Persistent scheduler created (block_size={}), dispatching 1 block",
                   scheduler.block_size());

        // Dispatch 1 block
        scheduler().dispatch(1u)(stream);
        stream << synchronize();
        LUISA_INFO("Persistent dispatch complete");
        expect(true);
    };

    "persistent_1suspend_smoke"_test = [argv] {
        constexpr uint N = 64u;

        Context ctx{argv[0]};
        Device device = ctx.create_device("fallback");
        Stream stream = device.create_stream();

        auto coro = Coroutine<void()>([] {
            $suspend("a");
        });

        LUISA_INFO("Coroutine created, sub_count={}", coro.subroutine_count());
        expect(coro.subroutine_count() >= 2u);

        PersistentThreadsCoroScheduler<> scheduler{device, coro, N};
        LUISA_INFO("Persistent 1-suspend scheduler created");

        scheduler().dispatch(1u)(stream);
        stream << synchronize();
        LUISA_INFO("Persistent 1-suspend dispatch complete");
        expect(true);
    };

    "persistent_3suspend_smoke"_test = [argv] {
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

        PersistentThreadsCoroScheduler<int> scheduler{device, coro, N};
        LUISA_INFO("Persistent 3-suspend scheduler created");

        scheduler(42).dispatch(1u)(stream);
        stream << synchronize();
        LUISA_INFO("Persistent 3-suspend dispatch complete");
        expect(true);
    };

    "persistent_with_buffer_arg_smoke"_test = [argv] {
        constexpr uint N = 64u;

        Context ctx{argv[0]};
        Device device = ctx.create_device("fallback");
        Stream stream = device.create_stream();

        auto output = device.create_buffer<uint>(N);

        auto coro = Coroutine<void(Buffer<uint>)>([](BufferUInt buf) {
            $suspend("a");
            buf.write(0u, 42u);
        });

        LUISA_INFO("Coroutine created, sub_count={}", coro.subroutine_count());
        expect(coro.subroutine_count() >= 2u);

        PersistentThreadsCoroScheduler<Buffer<uint>> scheduler{device, coro, N};
        LUISA_INFO("Persistent with buffer scheduler created");

        scheduler(output).dispatch(1u)(stream);
        stream << synchronize();
        LUISA_INFO("Persistent with buffer dispatch complete");
        expect(true);
    };
}

int main(int argc, char *argv[]) {
    reg_coro_persistent(argv);
    return 0;
}
