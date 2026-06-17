#include "ut/ut.hpp"
#include "coro_test_utils.h"

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

void reg_coro_soa_layout(luisa::test::coro_test::Options options) {

    "soa_layout_constructs_and_has_correct_config"_test = [options] {
        auto dc = luisa::test::coro_test::create_device(options);
        auto &device = dc.device;

        auto coro = Coroutine<void()>([] {
            $suspend("s1");
        });

        WavefrontCoroScheduler<> aos_scheduler{device, coro,
            WavefrontCoroSchedulerConfig{.global_memory_soa = false}};
        expect(aos_scheduler.config().global_memory_soa == false);

        WavefrontCoroScheduler<> soa_scheduler{device, coro,
            WavefrontCoroSchedulerConfig{.global_memory_soa = true}};
        expect(soa_scheduler.config().global_memory_soa == true);
    };

    "soa_layout_compiles_and_runs"_test = [options] {
        constexpr uint N = 64u;

        auto dc = luisa::test::coro_test::create_device(options);
        auto &device = dc.device;
        Stream stream = device.create_stream();

        auto coro = Coroutine<void()>([] {
            $suspend("s1");
        });

        LUISA_INFO("Coroutine created, sub_count={}", coro.subroutine_count());
        expect(coro.subroutine_count() >= 2u);

        WavefrontCoroScheduler<> scheduler{device, coro,
            WavefrontCoroSchedulerConfig{.global_memory_soa = true}};
        LUISA_INFO("SoA Wavefront scheduler created, dispatching {} instances", N);

        scheduler().dispatch(N)(stream);
        stream << synchronize();
        LUISA_INFO("SoA dispatch complete");
        expect(scheduler.config().global_memory_soa == true);
        expect(scheduler.config().thread_count >= N);
    };

    "soa_layout_1suspend_with_buffer"_test = [options] {
        constexpr uint N = 128u;

        auto dc = luisa::test::coro_test::create_device(options);
        auto &device = dc.device;
        Stream stream = device.create_stream();

        auto output = device.create_buffer<uint>(N);

        auto coro = Coroutine<void(Buffer<uint>)>([](BufferUInt buf) {
            auto tid = dispatch_x();
            $suspend("s1");
            buf.write(tid, tid + 42u);
        });

        LUISA_INFO("Coroutine created, sub_count={}", coro.subroutine_count());
        expect(coro.subroutine_count() >= 2u);

        WavefrontCoroScheduler<Buffer<uint>> scheduler{device, coro,
            WavefrontCoroSchedulerConfig{.global_memory_soa = true}};
        LUISA_INFO("SoA Wavefront scheduler created, dispatching {} instances", N);

        scheduler(output).dispatch(N)(stream);
        luisa::vector<uint> host(N);
        stream << output.copy_to(host.data()) << synchronize();
        LUISA_INFO("SoA dispatch complete");
        auto ok = true;
        for (auto i = 0u; i < N; i++) {
            if (host[i] != i + 42u) {
                LUISA_WARNING("soa_layout_1suspend_with_buffer mismatch at {}: got {}, expected {}",
                              i, host[i], i + 42u);
                ok = false;
                break;
            }
        }
        expect(ok) << "all wavefront SoA coroutine instances should write expected values";
    };

    "soa_layout_3suspend_smoke"_test = [options] {
        constexpr uint N = 64u;

        auto dc = luisa::test::coro_test::create_device(options);
        auto &device = dc.device;
        Stream stream = device.create_stream();

        auto coro = Coroutine<void(int)>([](Var<int> unused) {
            $suspend("a");
            $suspend("b");
            $suspend("c");
        });

        LUISA_INFO("Coroutine created, sub_count={}", coro.subroutine_count());
        expect(coro.subroutine_count() >= 2u);

        WavefrontCoroScheduler<int> scheduler{device, coro,
            WavefrontCoroSchedulerConfig{.global_memory_soa = true}};
        LUISA_INFO("SoA Wavefront scheduler created, dispatching {} instances", N);

        scheduler(42).dispatch(N)(stream);
        stream << synchronize();
        LUISA_INFO("SoA dispatch complete");
        expect(scheduler.config().global_memory_soa == true);
        expect(scheduler.config().thread_count >= N);
    };
}

int main(int argc, char *argv[]) {
    auto options = luisa::test::coro_test::parse_options(argc, argv);
    reg_coro_soa_layout(options);
    return 0;
}
