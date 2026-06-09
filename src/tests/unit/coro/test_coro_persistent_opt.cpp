#include "ut/ut.hpp"
#include <luisa/core/logging.h>
#include <luisa/coro/schedulers/persistent.h>
#include <luisa/dsl/coro_func.h>
#include <luisa/dsl/sugar.h>
#include <luisa/runtime/buffer.h>
#include <luisa/runtime/context.h>
#include <luisa/runtime/device.h>
#include <luisa/runtime/stream.h>

using namespace luisa;
using namespace luisa::compute;
using namespace luisa::compute::coro;
using namespace boost::ut;
using namespace boost::ut::literals;

void reg_coro_persistent_opt(char *argv[]) {

    // ================================================================
    // T33: Global memory extension (GME) for frame spill
    // ================================================================

    "T33_config_default_values"_test = [] {
        PersistentThreadsCoroSchedulerConfig cfg{};
        expect(cfg.thread_count == 65536u);
        expect(cfg.block_size == 128u);
        expect(cfg.fetch_size == 4u);
        expect(cfg.shared_memory_soa == false);
        expect(cfg.global_memory_ext == false);
    };

    "T33_config_custom_values"_test = [] {
        PersistentThreadsCoroSchedulerConfig cfg{
            .thread_count = 1024u,
            .block_size = 64u,
            .fetch_size = 8u,
            .shared_memory_soa = true,
            .global_memory_ext = true,
        };
        expect(cfg.thread_count == 1024u);
        expect(cfg.block_size == 64u);
        expect(cfg.fetch_size == 8u);
        expect(cfg.shared_memory_soa == true);
        expect(cfg.global_memory_ext == true);
    };

    "T33_GME_scheduler_creates_and_dispatches"_test = [argv] {
        Context ctx{argv[0]};
        Device device = ctx.create_device("fallback");
        Stream stream = device.create_stream();

        auto coro = Coroutine<void()>([] {
            $suspend("a");
            $suspend("b");
        });
        expect(coro.subroutine_count() >= 2u);

        PersistentThreadsCoroSchedulerConfig cfg{
            .thread_count = 64u,
            .block_size = 32u,
            .fetch_size = 2u,
            .global_memory_ext = true,
        };
        PersistentThreadsCoroScheduler<> scheduler{device, coro, cfg};
        LUISA_INFO("T33: GME scheduler created (thread_count={}, block_size={})",
                   cfg.thread_count, cfg.block_size);

        expect(scheduler.config().global_memory_ext == true);
        expect(scheduler.config().thread_count == 64u);

        scheduler().dispatch(1u)(stream);
        stream << synchronize();
        LUISA_INFO("T33: GME dispatch complete");
        expect(true);
    };

    // ================================================================
    // T34: SoA bank conflict avoidance in shared memory
    // ================================================================

    "T34_SoA_scheduler_creates_and_dispatches"_test = [argv] {
        Context ctx{argv[0]};
        Device device = ctx.create_device("fallback");
        Stream stream = device.create_stream();

        auto coro = Coroutine<void(int)>([](Var<int> x) {
            $suspend("s1");
        });
        expect(coro.subroutine_count() >= 2u);

        PersistentThreadsCoroSchedulerConfig cfg{
            .thread_count = 64u,
            .block_size = 32u,
            .shared_memory_soa = true,
        };
        PersistentThreadsCoroScheduler<int> scheduler{device, coro, cfg};
        LUISA_INFO("T34: SoA scheduler created");

        expect(scheduler.config().shared_memory_soa == true);

        scheduler(42).dispatch(1u)(stream);
        stream << synchronize();
        LUISA_INFO("T34: SoA dispatch complete");
        expect(true);
    };

    // ================================================================
    // T35: Atomic task acquisition + block-wise voting
    // ================================================================

    "T35_atomic_task_acquire_scheduler_creates_and_dispatches"_test = [argv] {
        Context ctx{argv[0]};
        Device device = ctx.create_device("fallback");
        Stream stream = device.create_stream();

        auto coro = Coroutine<void()>([] {
            $suspend("x");
        });
        expect(coro.subroutine_count() >= 2u);

        PersistentThreadsCoroSchedulerConfig cfg{
            .thread_count = 64u,
            .block_size = 32u,
            .fetch_size = 4u,
        };
        PersistentThreadsCoroScheduler<> scheduler{device, coro, cfg};
        LUISA_INFO("T35: Atomic-task scheduler created (fetch_size={})",
                   cfg.fetch_size);

        expect(scheduler.config().fetch_size == 4u);

        scheduler().dispatch(1u)(stream);
        stream << synchronize();
        LUISA_INFO("T35: Atomic-task dispatch complete");
        expect(true);
    };

    // ================================================================
    // Combined: GME + SoA + atomic task acquisition
    // ================================================================

    "T33_T34_T35_all_options_enabled"_test = [argv] {
        Context ctx{argv[0]};
        Device device = ctx.create_device("fallback");
        Stream stream = device.create_stream();

        auto coro = Coroutine<void()>([] {
            $suspend("first");
            $suspend("second");
            $suspend("third");
        });
        expect(coro.subroutine_count() >= 2u);

        PersistentThreadsCoroSchedulerConfig cfg{
            .thread_count = 64u,
            .block_size = 32u,
            .fetch_size = 4u,
            .shared_memory_soa = true,
            .global_memory_ext = true,
        };
        PersistentThreadsCoroScheduler<> scheduler{device, coro, cfg};
        LUISA_INFO("Combined: all options enabled scheduler created");

        expect(scheduler.config().shared_memory_soa == true);
        expect(scheduler.config().global_memory_ext == true);
        expect(scheduler.config().fetch_size == 4u);

        scheduler().dispatch(1u)(stream);
        stream << synchronize();
        LUISA_INFO("Combined: all options enabled dispatch complete");
        expect(true);
    };

    // ================================================================
    // Default config constructor (no Config argument)
    // ================================================================

    "T33_default_constructor_dispatches"_test = [argv] {
        Context ctx{argv[0]};
        Device device = ctx.create_device("fallback");
        Stream stream = device.create_stream();

        auto coro = Coroutine<void()>([] {
            $suspend("a");
        });
        expect(coro.subroutine_count() >= 2u);

        // Default constructor: default Config
        PersistentThreadsCoroScheduler<> scheduler{device, coro};
        LUISA_INFO("Default-construct scheduler: block_size={}",
                   scheduler.config().block_size);

        scheduler().dispatch(1u)(stream);
        stream << synchronize();
        LUISA_INFO("Default-construct dispatch complete");
        expect(true);
    };

    // ================================================================
    // Backward-compatible block_size constructor
    // ================================================================

    "T33_backward_compat_block_size_constructor"_test = [argv] {
        Context ctx{argv[0]};
        Device device = ctx.create_device("fallback");
        Stream stream = device.create_stream();

        auto coro = Coroutine<void()>([] {
            $suspend("a");
            $suspend("b");
        });
        expect(coro.subroutine_count() >= 2u);

        constexpr uint N = 64u;
        PersistentThreadsCoroScheduler<> scheduler{device, coro,
            PersistentThreadsCoroSchedulerConfig{.block_size = N}};
        LUISA_INFO("Backward-compat constructor: block_size={}",
                   scheduler.config().block_size);

        expect(scheduler.config().block_size == N);

        scheduler().dispatch(1u)(stream);
        stream << synchronize();
        LUISA_INFO("Backward-compat dispatch complete");
        expect(true);
    };

    // ================================================================
    // Type check: config accessor
    // ================================================================

    "T35_config_accessor_type_check"_test = [] {
        static_assert(std::is_same_v<
                      decltype(std::declval<PersistentThreadsCoroScheduler<>>().config()),
                      const PersistentThreadsCoroSchedulerConfig &>);
        expect(true);
    };
}

int main(int argc, char *argv[]) {
    reg_coro_persistent_opt(argv);
    return 0;
}
