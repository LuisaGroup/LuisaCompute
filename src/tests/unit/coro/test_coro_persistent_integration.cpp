#include "ut/ut.hpp"
#include "coro_test_utils.h"

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

void reg_coro_persistent_integration(luisa::test::coro_test::Options options) {

    // ================================================================
    // T36: Persistent integration test — config matrix
    // ================================================================

    // Simple 1-suspend coroutine shared by all config tests
    auto make_coro = [] {
        auto coro = Coroutine<void(Buffer<uint>)>([](BufferUInt output) {
            auto tid = dispatch_x();
            auto value = tid * 17u + 5u;
            $suspend("step");
            value = value * 3u + 7u;
            output.write(tid, value);
        });
        expect(coro.subroutine_count() >= 2u);
        return coro;
    };

    auto dispatch_and_validate = [](Device &device, Stream &stream, auto &scheduler,
                                    luisa::string_view label) noexcept {
        constexpr auto dispatch_size = 64u;
        auto output = device.create_buffer<uint>(dispatch_size);
        luisa::vector<uint> initial(dispatch_size, ~0u);
        stream << output.copy_from(luisa::span{initial});
        scheduler(output).dispatch(dispatch_size)(stream);

        luisa::vector<uint> host(dispatch_size);
        stream << output.copy_to(luisa::span{host}) << synchronize();
        auto ok = true;
        for (auto i = 0u; i < dispatch_size; i++) {
            auto expected = (i * 17u + 5u) * 3u + 7u;
            if (host[i] != expected) {
                LUISA_WARNING("{} mismatch at {}: got {}, expected {}",
                              label, i, host[i], expected);
                ok = false;
                break;
            }
        }
        expect(ok) << "persistent scheduler must preserve frame values across suspension";
    };

    // ----------------------------------------------------------------
    // 1. Default config
    // ----------------------------------------------------------------
    "T36_default_config_dispatches"_test = [options, make_coro, dispatch_and_validate] {
        auto dc = luisa::test::coro_test::create_device(options);
        auto &device = dc.device;
        Stream stream = device.create_stream();

        auto coro = make_coro();
        PersistentThreadsCoroScheduler<Buffer<uint>> scheduler{device, coro};
        LUISA_INFO("T36: default config — thread_count={}, block_size={}",
                   scheduler.config().thread_count,
                   scheduler.config().block_size);

        expect(scheduler.config().global_memory_ext == false);
        expect(scheduler.config().shared_memory_soa == false);
        expect(scheduler.config().thread_count == 65536u);
        expect(scheduler.config().block_size == 128u);
        expect(64u < scheduler.config().thread_count)
            << "regression requires logical work to be smaller than the default worker capacity";

        dispatch_and_validate(device, stream, scheduler, "T36_default_config_dispatches");
        LUISA_INFO("T36: default config dispatch complete");
    };

    // ----------------------------------------------------------------
    // 2. GME on
    // ----------------------------------------------------------------
    "T36_GME_on_dispatches"_test = [options, make_coro, dispatch_and_validate] {
        auto dc = luisa::test::coro_test::create_device(options);
        auto &device = dc.device;
        Stream stream = device.create_stream();

        auto coro = make_coro();
        PersistentThreadsCoroSchedulerConfig cfg{
            .thread_count = 64u,
            .block_size = 32u,
            .global_memory_ext = true,
        };
        PersistentThreadsCoroScheduler<Buffer<uint>> scheduler{device, coro, cfg};
        LUISA_INFO("T36: GME on — thread_count={}, block_size={}",
                   cfg.thread_count, cfg.block_size);

        expect(scheduler.config().global_memory_ext == true);
        expect(scheduler.config().shared_memory_soa == false);

        dispatch_and_validate(device, stream, scheduler, "T36_GME_on_dispatches");
        LUISA_INFO("T36: GME-on dispatch complete");
    };

    // ----------------------------------------------------------------
    // 3. SoA on
    // ----------------------------------------------------------------
    "T36_SoA_on_dispatches"_test = [options, make_coro, dispatch_and_validate] {
        auto dc = luisa::test::coro_test::create_device(options);
        auto &device = dc.device;
        Stream stream = device.create_stream();

        auto coro = make_coro();
        PersistentThreadsCoroSchedulerConfig cfg{
            .thread_count = 64u,
            .block_size = 32u,
            .shared_memory_soa = true,
        };
        PersistentThreadsCoroScheduler<Buffer<uint>> scheduler{device, coro, cfg};
        LUISA_INFO("T36: SoA on — thread_count={}, block_size={}",
                   cfg.thread_count, cfg.block_size);

        expect(scheduler.config().shared_memory_soa == true);
        expect(scheduler.config().global_memory_ext == false);

        dispatch_and_validate(device, stream, scheduler, "T36_SoA_on_dispatches");
        LUISA_INFO("T36: SoA-on dispatch complete");
    };

    // ----------------------------------------------------------------
    // 4. GME + SoA both on (all optimizations)
    // ----------------------------------------------------------------
    "T36_GME_SoA_all_on_dispatches"_test = [options, make_coro, dispatch_and_validate] {
        auto dc = luisa::test::coro_test::create_device(options);
        auto &device = dc.device;
        Stream stream = device.create_stream();

        auto coro = make_coro();
        PersistentThreadsCoroSchedulerConfig cfg{
            .thread_count = 64u,
            .block_size = 32u,
            .fetch_size = 4u,
            .shared_memory_soa = true,
            .global_memory_ext = true,
        };
        PersistentThreadsCoroScheduler<Buffer<uint>> scheduler{device, coro, cfg};
        LUISA_INFO("T36: GME+SoA all on — thread_count={}, block_size={}, fetch_size={}",
                   cfg.thread_count, cfg.block_size, cfg.fetch_size);

        expect(scheduler.config().shared_memory_soa == true);
        expect(scheduler.config().global_memory_ext == true);
        expect(scheduler.config().fetch_size == 4u);

        dispatch_and_validate(device, stream, scheduler, "T36_GME_SoA_all_on_dispatches");
        LUISA_INFO("T36: GME+SoA all-on dispatch complete");
    };

    // ----------------------------------------------------------------
    // 5. Custom block_size and thread_count
    // ----------------------------------------------------------------
    "T36_custom_block_thread_dispatches"_test = [options, make_coro, dispatch_and_validate] {
        auto dc = luisa::test::coro_test::create_device(options);
        auto &device = dc.device;
        Stream stream = device.create_stream();

        auto coro = make_coro();
        PersistentThreadsCoroSchedulerConfig cfg{
            .thread_count = 256u,
            .block_size = 64u,
            .fetch_size = 2u,
        };
        PersistentThreadsCoroScheduler<Buffer<uint>> scheduler{device, coro, cfg};
        LUISA_INFO("T36: custom — thread_count={}, block_size={}, fetch_size={}",
                   cfg.thread_count, cfg.block_size, cfg.fetch_size);

        expect(scheduler.config().thread_count == 256u);
        expect(scheduler.config().block_size == 64u);
        expect(scheduler.config().fetch_size == 2u);

        dispatch_and_validate(device, stream, scheduler, "T36_custom_block_thread_dispatches");
        LUISA_INFO("T36: custom block/thread dispatch complete");
    };

    // ----------------------------------------------------------------
    // 6. Backward compatibility: block_size-only constructor
    // ----------------------------------------------------------------
    "T36_backward_compat_block_size_dispatches"_test = [options, make_coro, dispatch_and_validate] {
        auto dc = luisa::test::coro_test::create_device(options);
        auto &device = dc.device;
        Stream stream = device.create_stream();

        auto coro = make_coro();
        constexpr uint N = 64u;
        PersistentThreadsCoroScheduler<Buffer<uint>> scheduler{device, coro,
                                                               PersistentThreadsCoroSchedulerConfig{.block_size = N}};
        LUISA_INFO("T36: backward compat — block_size={}",
                   scheduler.config().block_size);

        expect(scheduler.config().block_size == N);
        // Other config values should be defaults
        expect(scheduler.config().global_memory_ext == false);
        expect(scheduler.config().shared_memory_soa == false);

        dispatch_and_validate(device, stream, scheduler, "T36_backward_compat_block_size_dispatches");
        LUISA_INFO("T36: backward compat dispatch complete");
    };

    // ----------------------------------------------------------------
    // 7. GME + custom thread_count (different combine)
    // ----------------------------------------------------------------
    "T36_GME_custom_thread_dispatches"_test = [options, make_coro, dispatch_and_validate] {
        auto dc = luisa::test::coro_test::create_device(options);
        auto &device = dc.device;
        Stream stream = device.create_stream();

        auto coro = make_coro();
        PersistentThreadsCoroSchedulerConfig cfg{
            .thread_count = 512u,
            .block_size = 128u,
            .fetch_size = 8u,
            .global_memory_ext = true,
        };
        PersistentThreadsCoroScheduler<Buffer<uint>> scheduler{device, coro, cfg};
        LUISA_INFO("T36: GME+custom — thread_count={}, block_size={}, fetch_size={}",
                   cfg.thread_count, cfg.block_size, cfg.fetch_size);

        expect(scheduler.config().thread_count == 512u);
        expect(scheduler.config().global_memory_ext == true);

        dispatch_and_validate(device, stream, scheduler, "T36_GME_custom_thread_dispatches");
        LUISA_INFO("T36: GME+custom dispatch complete");
    };

    // ----------------------------------------------------------------
    // 8. Minimal thread count (boundary)
    // ----------------------------------------------------------------
    "T36_minimal_thread_count_dispatches"_test = [options, make_coro, dispatch_and_validate] {
        auto dc = luisa::test::coro_test::create_device(options);
        auto &device = dc.device;
        Stream stream = device.create_stream();

        auto coro = make_coro();
        PersistentThreadsCoroSchedulerConfig cfg{
            .thread_count = 32u,
            .block_size = 32u,
        };
        PersistentThreadsCoroScheduler<Buffer<uint>> scheduler{device, coro, cfg};
        LUISA_INFO("T36: minimal — thread_count={}, block_size={}",
                   cfg.thread_count, cfg.block_size);

        expect(scheduler.config().thread_count == 32u);
        expect(scheduler.config().block_size == 32u);

        dispatch_and_validate(device, stream, scheduler, "T36_minimal_thread_count_dispatches");
        LUISA_INFO("T36: minimal thread count dispatch complete");
    };

    // ----------------------------------------------------------------
    // 9. Fit an oversized workgroup to an explicit shared-memory budget
    // ----------------------------------------------------------------
    "T36_shared_memory_budget_reduces_block_and_dispatches"_test =
        [options, make_coro, dispatch_and_validate] {
            auto dc = luisa::test::coro_test::create_device(options);
            auto &device = dc.device;
            Stream stream = device.create_stream();

            auto coro = make_coro();
            PersistentThreadsCoroSchedulerConfig baseline_config{
                .thread_count = 128u,
                .block_size = 128u,
            };
            PersistentThreadsCoroScheduler<Buffer<uint>> baseline{
                device, coro, baseline_config};
            auto baseline_shared_memory =
                baseline.static_shared_memory_size_bytes();
            expect(baseline.config().block_size == 128u);
            expect(baseline_shared_memory > 0u);
            LUISA_ASSERT(
                baseline_shared_memory > 0u,
                "Persistent scheduler must report non-zero static shared memory.");

            auto fitted_config = baseline_config;
            fitted_config.shared_memory_limit_bytes =
                baseline_shared_memory - 1u;
            PersistentThreadsCoroScheduler<Buffer<uint>> fitted{
                device, coro, fitted_config};

            expect(fitted.config().block_size <
                   baseline.config().block_size)
                << "an over-budget workgroup must be rebuilt at a smaller block size";
            expect(fitted.config().thread_count %
                       fitted.config().block_size ==
                   0u)
                << "resource fitting must preserve complete persistent workgroups";
            expect(fitted.static_shared_memory_size_bytes() <=
                   fitted_config.shared_memory_limit_bytes)
                << "the rebuilt kernel must satisfy the explicit shared-memory budget";

            dispatch_and_validate(
                device, stream, fitted,
                "T36_shared_memory_budget_reduces_block_and_dispatches");
        };
}

int main(int argc, char *argv[]) {
    auto options = luisa::test::coro_test::parse_options(argc, argv);
    reg_coro_persistent_integration(options);
    return luisa::test::coro_test::run_tests(argc, argv);
}
