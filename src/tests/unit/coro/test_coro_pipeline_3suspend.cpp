// End-to-end pipeline integration test: 3-suspend counter coroutine
// Validates multi-suspend correctness through the full DSL → AST → XIR → scheduler chain.
//
#include "ut/ut.hpp"
#include "coro_test_utils.h"

#include <algorithm>

#include <luisa/core/logging.h>
#include <luisa/dsl/coro_func.h>
#include <luisa/dsl/sugar.h>
#include <luisa/coro/schedulers/persistent.h>
#include <luisa/coro/schedulers/state_machine.h>
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

enum class TestSchedulerKind {
    state_machine,
    wavefront,
    persistent,
};

[[nodiscard]] static constexpr auto scheduler_name(TestSchedulerKind kind) noexcept {
    switch (kind) {
        case TestSchedulerKind::state_machine: return "state_machine";
        case TestSchedulerKind::wavefront: return "wavefront";
        case TestSchedulerKind::persistent: return "persistent";
    }
    return "unknown";
}

template<typename F>
static void for_each_scheduler(F &&f) noexcept {
    f(TestSchedulerKind::state_machine);
    f(TestSchedulerKind::wavefront);
    f(TestSchedulerKind::persistent);
}

template<typename... Args>
static void dispatch_with_scheduler(Device &device, const Coroutine<void(Args...)> &coro,
                                    TestSchedulerKind kind, Stream &stream,
                                    uint dispatch_size,
                                    compute::detail::prototype_to_shader_invocation_t<Args>... args) noexcept {
    // Scheduler-owned shaders and frame/queue buffers are referenced by
    // asynchronous commands. Keep the concrete scheduler alive until those
    // commands finish; destroying it after submission is a resource-lifetime
    // violation on every backend, even if a synchronous allocator happens to
    // hide it on some devices.
    switch (kind) {
        case TestSchedulerKind::state_machine: {
            StateMachineCoroScheduler<Args...> scheduler{device, coro};
            scheduler(args...).dispatch(dispatch_size)(stream);
            stream << synchronize();
            break;
        }
        case TestSchedulerKind::wavefront: {
            auto capacity = std::max(dispatch_size, 128u);
            WavefrontCoroSchedulerConfig cfg{
                .thread_count = capacity,
            };
            WavefrontCoroScheduler<Args...> scheduler{device, coro, cfg};
            scheduler(args...).dispatch(dispatch_size)(stream);
            stream << synchronize();
            break;
        }
        case TestSchedulerKind::persistent: {
            auto capacity = std::max(dispatch_size, 128u);
            PersistentThreadsCoroSchedulerConfig cfg{
                .thread_count = capacity,
                .block_size = 128u,
            };
            PersistentThreadsCoroScheduler<Args...> scheduler{device, coro, cfg};
            scheduler(args...).dispatch(dispatch_size)(stream);
            stream << synchronize();
            break;
        }
    }
}

void reg_coro_pipeline_3suspend(luisa::test::coro_test::Options options) {

    "3suspend_identity"_test = [options] {
        constexpr uint N = 256u;

        for_each_scheduler([&](auto scheduler_kind) noexcept {
            auto dc = luisa::test::coro_test::create_device(options);
            auto &device = dc.device;
            Stream stream = device.create_stream();

            auto output = device.create_buffer<uint>(N);

            auto coro = Coroutine<void(Buffer<uint>)>([](BufferUInt output) {
                auto tid = dispatch_x();
                auto value = tid * 3u + 5u;
                $suspend("s1");
                value += tid ^ 7u;
                $suspend("s2");
                value = value * 2u + 1u;
                $suspend("s3");
                output.write(tid, value);
            });

            LUISA_INFO("Coroutine created for {}, subroutine_count={}",
                       scheduler_name(scheduler_kind), coro.subroutine_count());
            expect(coro.subroutine_count() >= 2u);
            expect(coro.graph().node_count() >= 2u);

            LUISA_INFO("Scheduler {} created, dispatching {} threads",
                       scheduler_name(scheduler_kind), N);

            dispatch_with_scheduler(device, coro, scheduler_kind, stream, N, output);
            luisa::vector<uint> host(N);
            stream << output.copy_to(luisa::span{host}) << synchronize();
            LUISA_INFO("Dispatch complete for {}", scheduler_name(scheduler_kind));
            auto ok = true;
            for (auto i = 0u; i < N; i++) {
                auto expected = ((i * 3u + 5u) + (i ^ 7u)) * 2u + 1u;
                if (host[i] != expected) {
                    LUISA_WARNING("3suspend_identity {} mismatch at {}: got {}, expected {}",
                                  scheduler_name(scheduler_kind), i, host[i], expected);
                    ok = false;
                    break;
                }
            }
            expect(ok) << "all coroutine instances should write their own output slot";
        });
    };

    "3suspend_large_dispatch"_test = [options] {
        constexpr uint N = 4096u;

        for_each_scheduler([&](auto scheduler_kind) noexcept {
            auto dc = luisa::test::coro_test::create_device(options);
            auto &device = dc.device;
            Stream stream = device.create_stream();

            auto output = device.create_buffer<uint>(N);

            auto coro = Coroutine<void(Buffer<uint>)>([](BufferUInt output) {
                auto tid = dispatch_x();
                auto value = tid + 99u;
                $suspend("s1");
                value = value * 17u + (tid & 31u);
                $suspend("s2");
                value = value ^ (tid * 13u + 3u);
                $suspend("s3");
                output.write(tid, value);
            });

            LUISA_INFO("Coroutine created for {}, subroutine_count={}",
                       scheduler_name(scheduler_kind), coro.subroutine_count());
            expect(coro.subroutine_count() >= 2u);

            LUISA_INFO("Scheduler {} created, dispatching {} threads",
                       scheduler_name(scheduler_kind), N);

            dispatch_with_scheduler(device, coro, scheduler_kind, stream, N, output);
            luisa::vector<uint> host(N);
            stream << output.copy_to(luisa::span{host}) << synchronize();
            LUISA_INFO("Large dispatch complete for {}", scheduler_name(scheduler_kind));
            auto ok = true;
            for (auto i = 0u; i < N; i++) {
                auto expected = ((i + 99u) * 17u + (i & 31u)) ^ (i * 13u + 3u);
                if (host[i] != expected) {
                    LUISA_WARNING("3suspend_large_dispatch {} mismatch at {}: got {}, expected {}",
                                  scheduler_name(scheduler_kind), i, host[i], expected);
                    ok = false;
                    break;
                }
            }
            expect(ok) << "all coroutine instances should write their own output slot";
        });
    };

    "3suspend_smoke"_test = [options] {
        for_each_scheduler([&](auto scheduler_kind) noexcept {
            auto dc = luisa::test::coro_test::create_device(options);
            auto &device = dc.device;
            Stream stream = device.create_stream();

            auto coro = Coroutine<void()>([] {
                $suspend("s1");
                $suspend("s2");
                $suspend("s3");
            });

            LUISA_INFO("Smoke coroutine created for {}, subroutine_count={}",
                       scheduler_name(scheduler_kind), coro.subroutine_count());
            expect(coro.subroutine_count() >= 2u);

            LUISA_INFO("Smoke scheduler {} created, dispatching 1 thread",
                       scheduler_name(scheduler_kind));

            dispatch_with_scheduler(device, coro, scheduler_kind, stream, 1u);
            stream << synchronize();
            LUISA_INFO("Smoke dispatch complete for {}", scheduler_name(scheduler_kind));
            expect(coro.graph().node_count() >= 2u);
        });
    };
}

int main(int argc, char *argv[]) {
    auto options = luisa::test::coro_test::parse_options(argc, argv);
    reg_coro_pipeline_3suspend(options);
    return luisa::test::coro_test::run_tests(argc, argv);
}
