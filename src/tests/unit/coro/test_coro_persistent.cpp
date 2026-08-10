#include "ut/ut.hpp"
#include "coro_test_utils.h"

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

void reg_coro_persistent(luisa::test::coro_test::Options options) {

    "persistent_constructor_and_type_check"_test = [] {
        static_assert(std::is_base_of_v<CoroScheduler<Buffer<int>>,
                                        PersistentThreadsCoroScheduler<Buffer<int>>>);
        static_assert(std::is_base_of_v<CoroScheduler<>,
                                        PersistentThreadsCoroScheduler<>>);
        expect(std::is_base_of_v<CoroScheduler<Buffer<int>>,
                                 PersistentThreadsCoroScheduler<Buffer<int>>>);
        expect(std::is_base_of_v<CoroScheduler<>,
                                 PersistentThreadsCoroScheduler<>>);
    };

    "persistent_compiles_and_runs"_test = [options] {
        constexpr uint N = 64u;

        auto dc = luisa::test::coro_test::create_device(options);
        auto &device = dc.device;
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

        PersistentThreadsCoroScheduler<> scheduler{device, coro,
            PersistentThreadsCoroSchedulerConfig{.block_size = N}};
        LUISA_INFO("Persistent scheduler created (block_size={}), dispatching 1 block",
                   scheduler.config().block_size);

        // Dispatch 1 block
        scheduler().dispatch(1u)(stream);
        stream << synchronize();
        LUISA_INFO("Persistent dispatch complete");
        expect(scheduler.config().block_size == N);
    };

    "persistent_1suspend_smoke"_test = [options] {
        constexpr uint N = 64u;

        auto dc = luisa::test::coro_test::create_device(options);
        auto &device = dc.device;
        Stream stream = device.create_stream();

        auto coro = Coroutine<void()>([] {
            $suspend("a");
        });

        LUISA_INFO("Coroutine created, sub_count={}", coro.subroutine_count());
        expect(coro.subroutine_count() >= 2u);

        PersistentThreadsCoroScheduler<> scheduler{device, coro,
            PersistentThreadsCoroSchedulerConfig{.block_size = N}};
        LUISA_INFO("Persistent 1-suspend scheduler created");

        scheduler().dispatch(1u)(stream);
        stream << synchronize();
        LUISA_INFO("Persistent 1-suspend dispatch complete");
        expect(scheduler.config().block_size == N);
    };

    "persistent_3suspend_smoke"_test = [options] {
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

        PersistentThreadsCoroScheduler<int> scheduler{device, coro,
            PersistentThreadsCoroSchedulerConfig{.block_size = N}};
        LUISA_INFO("Persistent 3-suspend scheduler created");

        scheduler(42).dispatch(1u)(stream);
        stream << synchronize();
        LUISA_INFO("Persistent 3-suspend dispatch complete");
        expect(scheduler.config().block_size == N);
    };

    "persistent_with_buffer_arg_smoke"_test = [options] {
        constexpr uint N = 64u;

        auto dc = luisa::test::coro_test::create_device(options);
        auto &device = dc.device;
        Stream stream = device.create_stream();

        auto output = device.create_buffer<uint>(N);

        auto coro = Coroutine<void(Buffer<uint>)>([](BufferUInt buf) {
            auto tid = dispatch_x();
            $suspend("a");
            buf.write(tid, tid + 42u);
        });

        LUISA_INFO("Coroutine created, sub_count={}", coro.subroutine_count());
        expect(coro.subroutine_count() >= 2u);

        PersistentThreadsCoroScheduler<Buffer<uint>> scheduler{
            device, coro,
            PersistentThreadsCoroSchedulerConfig{.thread_count = N, .block_size = N}};
        LUISA_INFO("Persistent with buffer scheduler created");

        scheduler(output).dispatch(N)(stream);
        luisa::vector<uint> host(N);
        stream << output.copy_to(luisa::span{host}) << synchronize();
        LUISA_INFO("Persistent with buffer dispatch complete");
        auto ok = true;
        for (auto i = 0u; i < N; i++) {
            if (host[i] != i + 42u) {
                LUISA_WARNING("persistent_with_buffer_arg_smoke mismatch at {}: got {}, expected {}",
                              i, host[i], i + 42u);
                ok = false;
                break;
            }
        }
        expect(ok) << "all persistent coroutine instances should write expected values";
    };

    "persistent_oversubscribed_dispatch_id_after_suspend"_test = [options] {
        constexpr uint N = 256u;
        constexpr uint worker_count = 64u;

        auto dc = luisa::test::coro_test::create_device(options);
        auto &device = dc.device;
        Stream stream = device.create_stream();

        auto output = device.create_buffer<uint>(N);

        auto coro = Coroutine<void(Buffer<uint>)>([](BufferUInt buf) {
            $suspend("a");
            auto tid = dispatch_x();
            buf.write(tid, tid + 77u);
        });

        PersistentThreadsCoroScheduler<Buffer<uint>> scheduler{
            device, coro,
            PersistentThreadsCoroSchedulerConfig{.thread_count = worker_count, .block_size = worker_count}};

        scheduler(output).dispatch(N)(stream);
        luisa::vector<uint> host(N);
        stream << output.copy_to(luisa::span{host}) << synchronize();

        auto ok = true;
        for (auto i = 0u; i < N; i++) {
            if (host[i] != i + 77u) {
                LUISA_WARNING("persistent_oversubscribed_dispatch_id_after_suspend mismatch at {}: got {}, expected {}",
                              i, host[i], i + 77u);
                ok = false;
                break;
            }
        }
        expect(ok) << "persistent scheduler must preserve logical dispatch id across suspension";
    };

    "persistent_more_continuations_than_block_threads"_test = [options] {
        constexpr uint N = 37u;
        constexpr uint block_size = 32u;
        constexpr uint suspend_count = block_size + 1u;

        auto dc = luisa::test::coro_test::create_device(options);
        auto &device = dc.device;
        Stream stream = device.create_stream();
        auto output = device.create_buffer<uint>(N);

        auto coro = Coroutine<void(Buffer<uint>)>([](BufferUInt buf) {
            auto tid = dispatch_x();
            auto value = tid * 11u;
            for (auto i = 0u; i < suspend_count; i++) {
                value += i + 1u;
                auto name = luisa::format("s{}", i);
                $suspend(name.c_str());
            }
            buf.write(tid, value);
        });
        expect(coro.subroutine_count() > block_size)
            << "the regression must cover more counters than block threads";

        for (auto global_memory_ext : {false, true}) {
            luisa::vector<uint> zero(N);
            stream << output.copy_from(luisa::span{zero}) << synchronize();
            PersistentThreadsCoroScheduler<Buffer<uint>> scheduler{
                device, coro,
                PersistentThreadsCoroSchedulerConfig{
                    .thread_count = 64u,
                    .block_size = block_size,
                    .fetch_size = 3u,
                    .shared_memory_soa = true,
                    .global_memory_ext = global_memory_ext}};
            scheduler(output).dispatch(N)(stream);

            luisa::vector<uint> host(N);
            stream << output.copy_to(luisa::span{host}) << synchronize();
            auto correct = true;
            for (auto i = 0u; i < N; i++) {
                correct &= host[i] == i * 11u +
                                           suspend_count * (suspend_count + 1u) / 2u;
            }
            expect(correct)
                << "all continuation counters must be initialized and scheduled "
                   "even when their count exceeds the block size";
        }
    };

    "persistent_zero_sized_dispatch_is_noop"_test = [options] {
        auto dc = luisa::test::coro_test::create_device(options);
        auto &device = dc.device;
        Stream stream = device.create_stream();
        auto output = device.create_buffer<uint>(1u);

        auto coro = Coroutine<void(Buffer<uint>)>([](BufferUInt buf) {
            $suspend("resume");
            buf.write(0u, 0u);
        });
        PersistentThreadsCoroScheduler<Buffer<uint>> scheduler{
            device, coro,
            PersistentThreadsCoroSchedulerConfig{
                .thread_count = 32u,
                .block_size = 32u,
                .global_memory_ext = true}};

        uint value = 0x12345678u;
        stream << output.copy_from(luisa::span{&value, 1u});
        scheduler(output).dispatch(0u)(stream);
        stream << output.copy_to(luisa::span{&value, 1u}) << synchronize();
        expect(value == 0x12345678u)
            << "a zero-sized logical dispatch must enqueue no scheduler work";
    };
}

int main(int argc, char *argv[]) {
    auto options = luisa::test::coro_test::parse_options(argc, argv);
    reg_coro_persistent(options);
    return luisa::test::coro_test::run_tests(argc, argv);
}
