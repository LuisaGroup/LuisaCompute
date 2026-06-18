#include "ut/ut.hpp"
#include "coro_test_utils.h"

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

void reg_coro_all_schedulers(luisa::test::coro_test::Options options) {

    auto expect_filled = [](luisa::span<const uint> host, uint base, luisa::string_view label) noexcept {
        auto ok = true;
        for (auto i = 0u; i < host.size(); i++) {
            auto expected = base + i;
            if (host[i] != expected) {
                LUISA_WARNING("{} mismatch at {}: got {}, expected {}",
                              label, i, host[i], expected);
                ok = false;
                break;
            }
        }
        expect(ok) << "all coroutine instances should write expected values";
    };

    // ══════════════════════════════════════════════════════════════════
    // 1-suspend coroutine — all 3 schedulers
    // ══════════════════════════════════════════════════════════════════

    "cross_1suspend_state_machine"_test = [options, expect_filled] {
        constexpr uint N = 256u;

        auto dc = luisa::test::coro_test::create_device(options);
        auto &device = dc.device;
        Stream stream = device.create_stream();
        auto output = device.create_buffer<uint>(N);

        auto coro = Coroutine<void(Buffer<uint>)>([](BufferUInt output) {
            auto tid = dispatch_x();
            $suspend("s1");
            output.write(tid, tid + 11u);
        });

        LUISA_INFO("1-suspend coroutine: sub_count={}", coro.subroutine_count());
        expect(coro.subroutine_count() >= 2u);
        expect(coro.graph().node_count() >= 2u);

        StateMachineCoroScheduler<Buffer<uint>> scheduler{device, coro};
        LUISA_INFO("StateMachineCoroScheduler: dispatching {} threads", N);
        scheduler(output).dispatch(N)(stream);
        luisa::vector<uint> host(N);
        stream << output.copy_to(luisa::span{host}) << synchronize();
        LUISA_INFO("StateMachineCoroScheduler: dispatch complete");
        expect_filled(host, 11u, "cross_1suspend_state_machine");
    };

    "cross_1suspend_wavefront"_test = [options, expect_filled] {
        constexpr uint N = 64u;

        auto dc = luisa::test::coro_test::create_device(options);
        auto &device = dc.device;
        Stream stream = device.create_stream();
        auto output = device.create_buffer<uint>(N);

        auto coro = Coroutine<void(Buffer<uint>)>([](BufferUInt output) {
            auto tid = dispatch_x();
            $suspend("s1");
            output.write(tid, tid + 12u);
        });

        LUISA_INFO("1-suspend coroutine: sub_count={}", coro.subroutine_count());
        expect(coro.subroutine_count() >= 2u);
        expect(coro.graph().node_count() >= 2u);

        WavefrontCoroScheduler<Buffer<uint>> scheduler{device, coro};
        LUISA_INFO("WavefrontCoroScheduler: dispatching {} instances", N);
        scheduler(output).dispatch(N)(stream);
        luisa::vector<uint> host(N);
        stream << output.copy_to(luisa::span{host}) << synchronize();
        LUISA_INFO("WavefrontCoroScheduler: dispatch complete");
        expect_filled(host, 12u, "cross_1suspend_wavefront");
    };

    "cross_1suspend_persistent"_test = [options, expect_filled] {
        constexpr uint N = 64u;

        auto dc = luisa::test::coro_test::create_device(options);
        auto &device = dc.device;
        Stream stream = device.create_stream();
        auto output = device.create_buffer<uint>(N);

        auto coro = Coroutine<void(Buffer<uint>)>([](BufferUInt output) {
            auto tid = dispatch_x();
            $suspend("s1");
            output.write(tid, tid + 13u);
        });

        LUISA_INFO("1-suspend coroutine: sub_count={}", coro.subroutine_count());
        expect(coro.subroutine_count() >= 2u);
        expect(coro.graph().node_count() >= 2u);

        PersistentThreadsCoroScheduler<Buffer<uint>> scheduler{
            device, coro,
            PersistentThreadsCoroSchedulerConfig{.thread_count = N, .block_size = N}};
        LUISA_INFO("PersistentThreadsCoroScheduler: dispatching {} logical instances", N);
        scheduler(output).dispatch(N)(stream);
        luisa::vector<uint> host(N);
        stream << output.copy_to(luisa::span{host}) << synchronize();
        LUISA_INFO("PersistentThreadsCoroScheduler: dispatch complete");
        expect_filled(host, 13u, "cross_1suspend_persistent");
    };

    // ══════════════════════════════════════════════════════════════════
    // 3-suspend coroutine — all 3 schedulers
    // ══════════════════════════════════════════════════════════════════

    "cross_3suspend_state_machine"_test = [options, expect_filled] {
        constexpr uint N = 256u;

        auto dc = luisa::test::coro_test::create_device(options);
        auto &device = dc.device;
        Stream stream = device.create_stream();
        auto output = device.create_buffer<uint>(N);

        auto coro = Coroutine<void(Buffer<uint>)>([](BufferUInt output) {
            auto tid = dispatch_x();
            $suspend("a");
            $suspend("b");
            $suspend("c");
            output.write(tid, tid + 31u);
        });

        LUISA_INFO("3-suspend coroutine: sub_count={}", coro.subroutine_count());
        expect(coro.subroutine_count() >= 2u);
        expect(coro.graph().node_count() >= 2u);

        StateMachineCoroScheduler<Buffer<uint>> scheduler{device, coro};
        LUISA_INFO("StateMachineCoroScheduler: dispatching {} threads", N);
        scheduler(output).dispatch(N)(stream);
        luisa::vector<uint> host(N);
        stream << output.copy_to(luisa::span{host}) << synchronize();
        LUISA_INFO("StateMachineCoroScheduler: dispatch complete");
        expect_filled(host, 31u, "cross_3suspend_state_machine");
    };

    "cross_3suspend_wavefront"_test = [options, expect_filled] {
        constexpr uint N = 64u;

        auto dc = luisa::test::coro_test::create_device(options);
        auto &device = dc.device;
        Stream stream = device.create_stream();
        auto output = device.create_buffer<uint>(N);

        auto coro = Coroutine<void(Buffer<uint>)>([](BufferUInt output) {
            auto tid = dispatch_x();
            $suspend("a");
            $suspend("b");
            $suspend("c");
            output.write(tid, tid + 32u);
        });

        LUISA_INFO("3-suspend coroutine: sub_count={}", coro.subroutine_count());
        expect(coro.subroutine_count() >= 2u);
        expect(coro.graph().node_count() >= 2u);

        WavefrontCoroScheduler<Buffer<uint>> scheduler{device, coro};
        LUISA_INFO("WavefrontCoroScheduler: dispatching {} instances", N);
        scheduler(output).dispatch(N)(stream);
        luisa::vector<uint> host(N);
        stream << output.copy_to(luisa::span{host}) << synchronize();
        LUISA_INFO("WavefrontCoroScheduler: dispatch complete");
        expect_filled(host, 32u, "cross_3suspend_wavefront");
    };

    "cross_3suspend_persistent"_test = [options, expect_filled] {
        constexpr uint N = 64u;

        auto dc = luisa::test::coro_test::create_device(options);
        auto &device = dc.device;
        Stream stream = device.create_stream();
        auto output = device.create_buffer<uint>(N);

        auto coro = Coroutine<void(Buffer<uint>)>([](BufferUInt output) {
            auto tid = dispatch_x();
            $suspend("a");
            $suspend("b");
            $suspend("c");
            output.write(tid, tid + 33u);
        });

        LUISA_INFO("3-suspend coroutine: sub_count={}", coro.subroutine_count());
        expect(coro.subroutine_count() >= 2u);
        expect(coro.graph().node_count() >= 2u);

        PersistentThreadsCoroScheduler<Buffer<uint>> scheduler{
            device, coro,
            PersistentThreadsCoroSchedulerConfig{.thread_count = N, .block_size = N}};
        LUISA_INFO("PersistentThreadsCoroScheduler: dispatching {} logical instances", N);
        scheduler(output).dispatch(N)(stream);
        luisa::vector<uint> host(N);
        stream << output.copy_to(luisa::span{host}) << synchronize();
        LUISA_INFO("PersistentThreadsCoroScheduler: dispatch complete");
        expect_filled(host, 33u, "cross_3suspend_persistent");
    };
}

int main(int argc, char *argv[]) {
    auto options = luisa::test::coro_test::parse_options(argc, argv);
    reg_coro_all_schedulers(options);
    return luisa::test::coro_test::run_tests(argc, argv);
}
