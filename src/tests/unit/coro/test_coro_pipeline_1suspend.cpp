// End-to-end pipeline integration test: 1-suspend coroutine
// Validates the complete DSL → AST → XIR → passes → scheduler → GPU execution chain
#include "ut/ut.hpp"
#include "coro_test_utils.h"

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

void reg_coro_pipeline_1suspend(luisa::test::coro_test::Options options) {

    "1suspend_identity"_test = [options] {
        constexpr uint N = 256u;

        auto dc = luisa::test::coro_test::create_device(options);
        auto &device = dc.device;
        Stream stream = device.create_stream();

        auto output = device.create_buffer<uint>(N);

        auto coro = Coroutine<void(Buffer<uint>)>([](BufferUInt output) {
            auto tid = dispatch_x();
            $suspend("s1");
            output.write(tid, tid + 42u);
        });

        LUISA_INFO("Coroutine created, subroutine_count={}", coro.subroutine_count());
        expect(coro.subroutine_count() >= 2u);
        expect(coro.graph().node_count() >= 2u);

        StateMachineCoroScheduler<Buffer<uint>> scheduler{device, coro};
        LUISA_INFO("Scheduler created, dispatching {} threads", N);

        scheduler(output).dispatch(N)(stream);
        luisa::vector<uint> host(N);
        stream << output.copy_to(host.data()) << synchronize();
        LUISA_INFO("Dispatch complete");
        auto ok = true;
        for (auto i = 0u; i < N; i++) {
            if (host[i] != i + 42u) {
                LUISA_WARNING("1suspend_identity mismatch at {}: got {}, expected {}",
                              i, host[i], i + 42u);
                ok = false;
                break;
            }
        }
        expect(ok) << "all coroutine instances should write their own output slot";
    };

    "1suspend_smoke"_test = [options] {
        auto dc = luisa::test::coro_test::create_device(options);
        auto &device = dc.device;
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
        expect(coro.graph().node_count() >= 2u);
    };
}

int main(int argc, char *argv[]) {
    auto options = luisa::test::coro_test::parse_options(argc, argv);
    reg_coro_pipeline_1suspend(options);
    return 0;
}
