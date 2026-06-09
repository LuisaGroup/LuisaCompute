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

void reg_coro_persistent_integration(char *argv[]) {

    // ================================================================
    // T36: Persistent integration test — config matrix
    // ================================================================

    // Simple 1-suspend coroutine shared by all config tests
    auto make_coro = [] {
        auto coro = Coroutine<void()>([] {
            $suspend("step");
        });
        expect(coro.subroutine_count() >= 2u);
        return coro;
    };

    // ----------------------------------------------------------------
    // 1. Default config
    // ----------------------------------------------------------------
    "T36_default_config_dispatches"_test = [argv, &make_coro] {
        Context ctx{argv[0]};
        Device device = ctx.create_device("fallback");
        Stream stream = device.create_stream();

        auto coro = make_coro();
        PersistentThreadsCoroScheduler<> scheduler{device, coro};
        LUISA_INFO("T36: default config — thread_count={}, block_size={}",
                   scheduler.config().thread_count,
                   scheduler.config().block_size);

        expect(scheduler.config().global_memory_ext == false);
        expect(scheduler.config().shared_memory_soa == false);
        expect(scheduler.config().thread_count == 65536u);
        expect(scheduler.config().block_size == 128u);

        scheduler().dispatch(1u)(stream);
        stream << synchronize();
        LUISA_INFO("T36: default config dispatch complete");
    };

    // ----------------------------------------------------------------
    // 2. GME on
    // ----------------------------------------------------------------
    "T36_GME_on_dispatches"_test = [argv, &make_coro] {
        Context ctx{argv[0]};
        Device device = ctx.create_device("fallback");
        Stream stream = device.create_stream();

        auto coro = make_coro();
        PersistentThreadsCoroSchedulerConfig cfg{
            .thread_count = 64u,
            .block_size = 32u,
            .global_memory_ext = true,
        };
        PersistentThreadsCoroScheduler<> scheduler{device, coro, cfg};
        LUISA_INFO("T36: GME on — thread_count={}, block_size={}",
                   cfg.thread_count, cfg.block_size);

        expect(scheduler.config().global_memory_ext == true);
        expect(scheduler.config().shared_memory_soa == false);

        scheduler().dispatch(1u)(stream);
        stream << synchronize();
        LUISA_INFO("T36: GME-on dispatch complete");
    };

    // ----------------------------------------------------------------
    // 3. SoA on
    // ----------------------------------------------------------------
    "T36_SoA_on_dispatches"_test = [argv, &make_coro] {
        Context ctx{argv[0]};
        Device device = ctx.create_device("fallback");
        Stream stream = device.create_stream();

        auto coro = make_coro();
        PersistentThreadsCoroSchedulerConfig cfg{
            .thread_count = 64u,
            .block_size = 32u,
            .shared_memory_soa = true,
        };
        PersistentThreadsCoroScheduler<> scheduler{device, coro, cfg};
        LUISA_INFO("T36: SoA on — thread_count={}, block_size={}",
                   cfg.thread_count, cfg.block_size);

        expect(scheduler.config().shared_memory_soa == true);
        expect(scheduler.config().global_memory_ext == false);

        scheduler().dispatch(1u)(stream);
        stream << synchronize();
        LUISA_INFO("T36: SoA-on dispatch complete");
    };

    // ----------------------------------------------------------------
    // 4. GME + SoA both on (all optimizations)
    // ----------------------------------------------------------------
    "T36_GME_SoA_all_on_dispatches"_test = [argv, &make_coro] {
        Context ctx{argv[0]};
        Device device = ctx.create_device("fallback");
        Stream stream = device.create_stream();

        auto coro = make_coro();
        PersistentThreadsCoroSchedulerConfig cfg{
            .thread_count = 64u,
            .block_size = 32u,
            .fetch_size = 4u,
            .shared_memory_soa = true,
            .global_memory_ext = true,
        };
        PersistentThreadsCoroScheduler<> scheduler{device, coro, cfg};
        LUISA_INFO("T36: GME+SoA all on — thread_count={}, block_size={}, fetch_size={}",
                   cfg.thread_count, cfg.block_size, cfg.fetch_size);

        expect(scheduler.config().shared_memory_soa == true);
        expect(scheduler.config().global_memory_ext == true);
        expect(scheduler.config().fetch_size == 4u);

        scheduler().dispatch(1u)(stream);
        stream << synchronize();
        LUISA_INFO("T36: GME+SoA all-on dispatch complete");
    };

    // ----------------------------------------------------------------
    // 5. Custom block_size and thread_count
    // ----------------------------------------------------------------
    "T36_custom_block_thread_dispatches"_test = [argv, &make_coro] {
        Context ctx{argv[0]};
        Device device = ctx.create_device("fallback");
        Stream stream = device.create_stream();

        auto coro = make_coro();
        PersistentThreadsCoroSchedulerConfig cfg{
            .thread_count = 256u,
            .block_size = 64u,
            .fetch_size = 2u,
        };
        PersistentThreadsCoroScheduler<> scheduler{device, coro, cfg};
        LUISA_INFO("T36: custom — thread_count={}, block_size={}, fetch_size={}",
                   cfg.thread_count, cfg.block_size, cfg.fetch_size);

        expect(scheduler.config().thread_count == 256u);
        expect(scheduler.config().block_size == 64u);
        expect(scheduler.config().fetch_size == 2u);

        scheduler().dispatch(1u)(stream);
        stream << synchronize();
        LUISA_INFO("T36: custom block/thread dispatch complete");
    };

    // ----------------------------------------------------------------
    // 6. Backward compatibility: block_size-only constructor
    // ----------------------------------------------------------------
    "T36_backward_compat_block_size_dispatches"_test = [argv, &make_coro] {
        Context ctx{argv[0]};
        Device device = ctx.create_device("fallback");
        Stream stream = device.create_stream();

        auto coro = make_coro();
        constexpr uint N = 64u;
        PersistentThreadsCoroScheduler<> scheduler{device, coro, N};
        LUISA_INFO("T36: backward compat — block_size={}",
                   scheduler.config().block_size);

        expect(scheduler.config().block_size == N);
        // Other config values should be defaults
        expect(scheduler.config().global_memory_ext == false);
        expect(scheduler.config().shared_memory_soa == false);

        scheduler().dispatch(1u)(stream);
        stream << synchronize();
        LUISA_INFO("T36: backward compat dispatch complete");
    };

    // ----------------------------------------------------------------
    // 7. GME + custom thread_count (different combine)
    // ----------------------------------------------------------------
    "T36_GME_custom_thread_dispatches"_test = [argv, &make_coro] {
        Context ctx{argv[0]};
        Device device = ctx.create_device("fallback");
        Stream stream = device.create_stream();

        auto coro = make_coro();
        PersistentThreadsCoroSchedulerConfig cfg{
            .thread_count = 512u,
            .block_size = 128u,
            .fetch_size = 8u,
            .global_memory_ext = true,
        };
        PersistentThreadsCoroScheduler<> scheduler{device, coro, cfg};
        LUISA_INFO("T36: GME+custom — thread_count={}, block_size={}, fetch_size={}",
                   cfg.thread_count, cfg.block_size, cfg.fetch_size);

        expect(scheduler.config().thread_count == 512u);
        expect(scheduler.config().global_memory_ext == true);

        scheduler().dispatch(1u)(stream);
        stream << synchronize();
        LUISA_INFO("T36: GME+custom dispatch complete");
    };

    // ----------------------------------------------------------------
    // 8. Minimal thread count (boundary)
    // ----------------------------------------------------------------
    "T36_minimal_thread_count_dispatches"_test = [argv, &make_coro] {
        Context ctx{argv[0]};
        Device device = ctx.create_device("fallback");
        Stream stream = device.create_stream();

        auto coro = make_coro();
        PersistentThreadsCoroSchedulerConfig cfg{
            .thread_count = 32u,
            .block_size = 32u,
        };
        PersistentThreadsCoroScheduler<> scheduler{device, coro, cfg};
        LUISA_INFO("T36: minimal — thread_count={}, block_size={}",
                   cfg.thread_count, cfg.block_size);

        expect(scheduler.config().thread_count == 32u);
        expect(scheduler.config().block_size == 32u);

        scheduler().dispatch(1u)(stream);
        stream << synchronize();
        LUISA_INFO("T36: minimal thread count dispatch complete");
    };
}

int main(int argc, char *argv[]) {
    reg_coro_persistent_integration(argv);
    return 0;
}
