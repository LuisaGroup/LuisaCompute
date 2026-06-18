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

void reg_coro_persistent_opt(luisa::test::coro_test::Options options) {

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

    "T33_GME_scheduler_creates_and_dispatches"_test = [options] {
        auto dc = luisa::test::coro_test::create_device(options);
        auto &device = dc.device;
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
    };

    "T33_GME_spills_and_preserves_frame_fields"_test = [options] {
        constexpr uint N = 257u;
        constexpr uint thread_count = 64u;
        constexpr uint block_size = 32u;

        auto dc = luisa::test::coro_test::create_device(options);
        auto &device = dc.device;
        Stream stream = device.create_stream();

        auto output = device.create_buffer<uint>(N);

        auto coro = Coroutine<void(Buffer<uint>)>([](BufferUInt buf) {
            auto tid = dispatch_x();
            auto value = tid * 17u + 3u;
            $suspend("first");
            value = value ^ (tid + 11u);
            $suspend("second");
            value += tid * 5u + 7u;
            $suspend("third");
            buf.write(tid, value);
        });
        expect(coro.subroutine_count() >= 4u);

        for (auto shared_soa : {false, true}) {
            luisa::vector<uint> zero(N);
            stream << output.copy_from(luisa::span{zero}) << synchronize();

            PersistentThreadsCoroSchedulerConfig cfg{
                .thread_count = thread_count,
                .block_size = block_size,
                .fetch_size = 4u,
                .shared_memory_soa = shared_soa,
                .global_memory_ext = true,
            };
            PersistentThreadsCoroScheduler<Buffer<uint>> scheduler{device, coro, cfg};
            scheduler(output).dispatch(N)(stream);

            luisa::vector<uint> host(N);
            stream << output.copy_to(luisa::span{host}) << synchronize();

            auto ok = true;
            for (auto i = 0u; i < N; i++) {
                auto expected = ((i * 17u + 3u) ^ (i + 11u)) + i * 5u + 7u;
                if (host[i] != expected) {
                    LUISA_WARNING("persistent GME spill mismatch shared_soa={} at {}: got {}, expected {}",
                                  shared_soa, i, host[i], expected);
                    ok = false;
                    break;
                }
            }
            expect(ok) << "persistent global-memory extension must spill and restore frame fields";
            expect(scheduler.config().global_memory_ext == true);
            expect(scheduler.config().shared_memory_soa == shared_soa);
        }
    };

    "T33_no_GME_preserves_frame_fields_across_oversubscribed_dispatch"_test = [options] {
        constexpr uint N = 257u;
        constexpr uint thread_count = 64u;
        constexpr uint block_size = 32u;

        auto dc = luisa::test::coro_test::create_device(options);
        auto &device = dc.device;
        Stream stream = device.create_stream();

        auto output = device.create_buffer<uint>(N);

        auto coro = Coroutine<void(Buffer<uint>)>([](BufferUInt buf) {
            auto tid = dispatch_x();
            auto value = tid * 23u + 9u;
            $suspend("first");
            value = value * 3u + (tid & 15u);
            $suspend("second");
            value = (value ^ (tid * 7u + 5u)) + 41u;
            $suspend("third");
            buf.write(tid, value);
        });
        expect(coro.subroutine_count() >= 4u);

        for (auto shared_soa : {false, true}) {
            luisa::vector<uint> zero(N);
            stream << output.copy_from(luisa::span{zero}) << synchronize();

            PersistentThreadsCoroSchedulerConfig cfg{
                .thread_count = thread_count,
                .block_size = block_size,
                .fetch_size = 2u,
                .shared_memory_soa = shared_soa,
                .global_memory_ext = false,
            };
            PersistentThreadsCoroScheduler<Buffer<uint>> scheduler{device, coro, cfg};
            scheduler(output).dispatch(N)(stream);

            luisa::vector<uint> host(N);
            stream << output.copy_to(luisa::span{host}) << synchronize();

            auto ok = true;
            for (auto i = 0u; i < N; i++) {
                auto expected = ((i * 23u + 9u) * 3u + (i & 15u)) ^ (i * 7u + 5u);
                expected += 41u;
                if (host[i] != expected) {
                    LUISA_WARNING("persistent no-GME mismatch shared_soa={} at {}: got {}, expected {}",
                                  shared_soa, i, host[i], expected);
                    ok = false;
                    break;
                }
            }
            expect(ok) << "persistent scheduler must preserve frames without global-memory spill";
            expect(scheduler.config().global_memory_ext == false);
            expect(scheduler.config().shared_memory_soa == shared_soa);
        }
    };

    "T33_GME_repeated_dispatch_reuses_scheduler_without_stale_frames"_test = [options] {
        constexpr uint N = 173u;

        auto dc = luisa::test::coro_test::create_device(options);
        auto &device = dc.device;
        Stream stream = device.create_stream();

        auto output = device.create_buffer<uint>(N);

        auto coro = Coroutine<void(Buffer<uint>, uint)>([](BufferUInt buf, UInt salt) {
            auto tid = dispatch_x();
            auto value = tid * 13u + salt;
            $suspend("first");
            value = (value ^ (salt * 17u + tid)) + 29u;
            $suspend("second");
            buf.write(tid, value);
        });

        PersistentThreadsCoroSchedulerConfig cfg{
            .thread_count = 64u,
            .block_size = 32u,
            .fetch_size = 3u,
            .shared_memory_soa = true,
            .global_memory_ext = true,
        };
        PersistentThreadsCoroScheduler<Buffer<uint>, uint> scheduler{device, coro, cfg};

        for (auto pass = 0u; pass < 2u; pass++) {
            auto salt = pass == 0u ? 101u : 907u;
            luisa::vector<uint> zero(N);
            stream << output.copy_from(luisa::span{zero}) << synchronize();

            scheduler(output, salt).dispatch(N)(stream);

            luisa::vector<uint> host(N);
            stream << output.copy_to(luisa::span{host}) << synchronize();

            auto ok = true;
            for (auto i = 0u; i < N; i++) {
                auto expected = (i * 13u + salt) ^ (salt * 17u + i);
                expected += 29u;
                if (host[i] != expected) {
                    LUISA_WARNING("persistent repeated GME dispatch mismatch pass={} at {}: got {}, expected {}",
                                  pass, i, host[i], expected);
                    ok = false;
                    break;
                }
            }
            expect(ok) << "persistent GME scheduler must not read stale spilled frames on later dispatches";
        }
    };

    // ================================================================
    // T34: SoA bank conflict avoidance in shared memory
    // ================================================================

    "T34_SoA_scheduler_creates_and_dispatches"_test = [options] {
        auto dc = luisa::test::coro_test::create_device(options);
        auto &device = dc.device;
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
    };

    // ================================================================
    // T35: Atomic task acquisition + block-wise voting
    // ================================================================

    "T35_atomic_task_acquire_scheduler_creates_and_dispatches"_test = [options] {
        auto dc = luisa::test::coro_test::create_device(options);
        auto &device = dc.device;
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
    };

    // ================================================================
    // Combined: GME + SoA + atomic task acquisition
    // ================================================================

    "T33_T34_T35_all_options_enabled"_test = [options] {
        auto dc = luisa::test::coro_test::create_device(options);
        auto &device = dc.device;
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
    };

    // ================================================================
    // Default config constructor (no Config argument)
    // ================================================================

    "T33_default_constructor_dispatches"_test = [options] {
        auto dc = luisa::test::coro_test::create_device(options);
        auto &device = dc.device;
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
        expect(scheduler.config().block_size == PersistentThreadsCoroSchedulerConfig{}.block_size);
    };

    // ================================================================
    // Backward-compatible block_size constructor
    // ================================================================

    "T33_backward_compat_block_size_constructor"_test = [options] {
        auto dc = luisa::test::coro_test::create_device(options);
        auto &device = dc.device;
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
    };

    // ================================================================
    // Type check: config accessor
    // ================================================================

    "T35_config_accessor_type_check"_test = [] {
        static_assert(std::is_same_v<
                      decltype(std::declval<PersistentThreadsCoroScheduler<>>().config()),
                      const PersistentThreadsCoroSchedulerConfig &>);
        expect(std::is_same_v<
               decltype(std::declval<PersistentThreadsCoroScheduler<>>().config()),
               const PersistentThreadsCoroSchedulerConfig &>);
    };

    "T35_fetch_size_and_thread_alignment_preserve_correctness"_test = [options] {
        constexpr uint N = 229u;

        auto dc = luisa::test::coro_test::create_device(options);
        auto &device = dc.device;
        Stream stream = device.create_stream();

        auto output = device.create_buffer<uint>(N);

        auto coro = Coroutine<void(Buffer<uint>)>([](BufferUInt buf) {
            auto tid = dispatch_x();
            auto value = tid + 3u;
            $suspend("first");
            value = value * 19u + (tid & 7u);
            $suspend("second");
            buf.write(tid, value ^ 0x5a5au);
        });

        for (auto fetch_size : {1u, 3u, 5u}) {
            luisa::vector<uint> zero(N);
            stream << output.copy_from(luisa::span{zero}) << synchronize();

            PersistentThreadsCoroSchedulerConfig cfg{
                .thread_count = 70u,
                .block_size = 32u,
                .fetch_size = fetch_size,
                .shared_memory_soa = fetch_size == 3u,
                .global_memory_ext = fetch_size == 5u,
            };
            PersistentThreadsCoroScheduler<Buffer<uint>> scheduler{device, coro, cfg};
            expect(scheduler.config().thread_count == 96u) << "thread_count should align to block_size";
            expect(scheduler.config().fetch_size == fetch_size);

            scheduler(output).dispatch(N)(stream);

            luisa::vector<uint> host(N);
            stream << output.copy_to(luisa::span{host}) << synchronize();

            auto ok = true;
            for (auto i = 0u; i < N; i++) {
                auto expected = ((i + 3u) * 19u + (i & 7u)) ^ 0x5a5au;
                if (host[i] != expected) {
                    LUISA_WARNING("persistent fetch/alignment mismatch fetch_size={} at {}: got {}, expected {}",
                                  fetch_size, i, host[i], expected);
                    ok = false;
                    break;
                }
            }
            expect(ok) << "persistent fetch-size variations and aligned thread counts must preserve results";
        }
    };
}

int main(int argc, char *argv[]) {
    auto options = luisa::test::coro_test::parse_options(argc, argv);
    reg_coro_persistent_opt(options);
    return luisa::test::coro_test::run_tests(argc, argv);
}
