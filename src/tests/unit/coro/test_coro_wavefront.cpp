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

void reg_coro_wavefront(char *argv[]) {

    "wavefront_constructor_and_type_check"_test = [] {
        static_assert(std::is_base_of_v<CoroScheduler<Buffer<int>>,
                                        WavefrontCoroScheduler<Buffer<int>>>);
        expect(true);
    };

    "wavefront_compiles_and_runs"_test = [argv] {
        constexpr uint N = 64u;

        Context ctx{argv[0]};
        Device device = ctx.create_device("fallback");
        Stream stream = device.create_stream();

        // Verify basic shader dispatch works
        {
            auto k = Kernel1D{[]() noexcept {}};
            auto s = device.compile(k);
            stream << s().dispatch(N) << synchronize();
            LUISA_INFO("Basic kernel dispatch OK");
        }

        // Coroutine with 1 suspend — verify dispatch completes
        auto coro = Coroutine<void()>([] {
            $suspend("s1");
        });

        LUISA_INFO("Coroutine created, sub_count={}", coro.subroutine_count());
        expect(coro.subroutine_count() >= 2u);
        expect(coro.graph().node_count() >= 2u);

        WavefrontCoroScheduler<> scheduler{device, coro};
        LUISA_INFO("Wavefront scheduler created, dispatching {} instances", N);

        scheduler().dispatch(N)(stream);
        stream << synchronize();
        LUISA_INFO("Dispatch complete");
        expect(true);
    };

    "wavefront_1suspend_with_buffer"_test = [argv] {
        // Same coroutine pattern as StateMachine test — verify no crash
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

        WavefrontCoroScheduler<Buffer<uint>> scheduler{device, coro};
        LUISA_INFO("Wavefront scheduler created, dispatching {} instances", N);

        scheduler(output).dispatch(N)(stream);
        stream << synchronize();
        LUISA_INFO("Dispatch complete");
        expect(true);
    };

    "wavefront_3suspend_smoke"_test = [argv] {
        // Multi-suspend coroutine — verify dispatch completes
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

        WavefrontCoroScheduler<int> scheduler{device, coro};
        LUISA_INFO("Wavefront scheduler created, dispatching {} instances", N);

        scheduler(42).dispatch(N)(stream);
        stream << synchronize();
        LUISA_INFO("Dispatch complete");
        expect(true);
    };
}

int main(int argc, char *argv[]) {
    reg_coro_wavefront(argv);
    return 0;
}
