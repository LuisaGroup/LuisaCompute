// Test for wavefront coroutine scheduling.
// This test covers bounded frame pools, AoS/SoA storage, compaction,
// token and hint sorting, and oversubscribed dispatches.

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

void reg_coro_wavefront(luisa::test::coro_test::Options options) {

    "wavefront_constructor_and_type_check"_test = [] {
        static_assert(std::is_base_of_v<CoroScheduler<Buffer<int>>,
                                        WavefrontCoroScheduler<Buffer<int>>>);
        expect(std::is_base_of_v<CoroScheduler<Buffer<int>>,
                                 WavefrontCoroScheduler<Buffer<int>>>);
    };

    "wavefront_compiles_and_runs"_test = [options] {
        constexpr uint N = 64u;

        auto dc = luisa::test::coro_test::create_device(options);
        auto &device = dc.device;
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

        WavefrontCoroScheduler<> scheduler{
            device, coro, WavefrontCoroSchedulerConfig{.thread_count = N}};
        LUISA_INFO("Wavefront scheduler created, dispatching {} instances", N);

        scheduler().dispatch(N)(stream);
        stream << synchronize();
        LUISA_INFO("Dispatch complete");
        expect(scheduler.config().thread_count == N)
            << "the smoke test should use a bounded frame pool";
    };

    "wavefront_1suspend_with_buffer"_test = [options] {
        // Same coroutine pattern as StateMachine test — verify no crash
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

        WavefrontCoroScheduler<Buffer<uint>> scheduler{
            device, coro, WavefrontCoroSchedulerConfig{.thread_count = N}};
        LUISA_INFO("Wavefront scheduler created, dispatching {} instances", N);

        scheduler(output).dispatch(N)(stream);
        luisa::vector<uint> host(N);
        stream << output.copy_to(luisa::span{host}) << synchronize();
        LUISA_INFO("Dispatch complete");
        auto ok = true;
        for (auto i = 0u; i < N; i++) {
            if (host[i] != i + 42u) {
                LUISA_WARNING("wavefront_1suspend_with_buffer mismatch at {}: got {}, expected {}",
                              i, host[i], i + 42u);
                ok = false;
                break;
            }
        }
        expect(ok) << "all wavefront coroutine instances should write expected values";
        expect(scheduler.config().thread_count == N)
            << "the buffer test should use a bounded frame pool";
    };

    "wavefront_3suspend_smoke"_test = [options] {
        // Multi-suspend coroutine — verify dispatch completes
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

        WavefrontCoroScheduler<int> scheduler{
            device, coro, WavefrontCoroSchedulerConfig{.thread_count = N}};
        LUISA_INFO("Wavefront scheduler created, dispatching {} instances", N);

        scheduler(42).dispatch(N)(stream);
        stream << synchronize();
        LUISA_INFO("Dispatch complete");
        expect(scheduler.config().thread_count == N)
            << "the multi-suspend smoke test should use a bounded frame pool";
    };

    "wavefront_fixed_capacity_pool_runs_oversubscribed_dispatch"_test = [options] {
        constexpr uint N = 257u;
        constexpr uint capacity = 64u;

        auto dc = luisa::test::coro_test::create_device(options);
        auto &device = dc.device;
        Stream stream = device.create_stream();

        auto output = device.create_buffer<uint>(N);

        auto coro = Coroutine<void(Buffer<uint>)>([](BufferUInt buf) {
            auto tid = dispatch_x();
            auto v = tid * 3u + 5u;
            $suspend("first");
            v += 7u;
            $suspend("second");
            v = v * 2u + tid;
            buf.write(tid, v);
        });

        for (auto soa : {false, true}) {
            for (auto compaction : {false, true}) {
                luisa::vector<uint> zero(N);
                stream << output.copy_from(luisa::span{zero}) << synchronize();

                WavefrontCoroSchedulerConfig cfg{
                    .thread_count = capacity,
                    .global_memory_soa = soa,
                    .gather_by_sorting = false,
                    .frame_buffer_compaction = compaction,
                };
                WavefrontCoroScheduler<Buffer<uint>> scheduler{device, coro, cfg};
                scheduler(output).dispatch(N)(stream);

                luisa::vector<uint> host(N);
                stream << output.copy_to(luisa::span{host}) << synchronize();

                auto ok = true;
                for (auto i = 0u; i < N; i++) {
                    auto expected = (i * 3u + 12u) * 2u + i;
                    if (host[i] != expected) {
                        LUISA_WARNING("wavefront fixed-capacity mismatch soa={}, compaction={} at {}: got {}, expected {}",
                                      soa, compaction, i, host[i], expected);
                        ok = false;
                        break;
                    }
                }
                expect(ok) << "wavefront must use thread_count as frame-pool capacity, not dispatch limit";
                expect(scheduler.config().thread_count == capacity);
            }
        }
    };

    "wavefront_large_pool_activates_only_logical_dispatch"_test = [options] {
        constexpr uint N = 13u;
        constexpr uint capacity = 256u;

        auto dc = luisa::test::coro_test::create_device(options);
        auto &device = dc.device;
        Stream stream = device.create_stream();
        auto output = device.create_buffer<uint>(N);

        auto coro = Coroutine<void(Buffer<uint>)>([](BufferUInt buf) {
            auto tid = dispatch_x();
            auto value = tid + 1u;
            $suspend("only");
            buf.write(tid, value * 3u);
        });
        WavefrontCoroScheduler<Buffer<uint>> scheduler{
            device, coro,
            WavefrontCoroSchedulerConfig{
                .thread_count = capacity,
                .gather_by_sorting = false,
                .frame_buffer_compaction = false}};

        scheduler(output).dispatch(N)(stream);
        expect(scheduler.config().thread_count == capacity)
            << "the allocated pool ceiling must remain unchanged";
        expect(scheduler.active_frame_capacity() == N)
            << "a small dispatch must not initialize or scan the entire pool";

        luisa::vector<uint> host(N);
        stream << output.copy_to(luisa::span{host}) << synchronize();
        auto correct = true;
        for (auto i = 0u; i < N; i++) {
            correct &= host[i] == (i + 1u) * 3u;
        }
        expect(correct);
    };

    "wavefront_sorting_gather_preserves_config_and_correctness"_test = [options] {
        constexpr uint N = 193u;
        constexpr uint capacity = 64u;

        auto dc = luisa::test::coro_test::create_device(options);
        auto &device = dc.device;
        Stream stream = device.create_stream();

        auto output = device.create_buffer<uint>(N);

        auto coro = Coroutine<void(Buffer<uint>)>([](BufferUInt buf) {
            auto tid = dispatch_x();
            auto v = tid + 11u;
            $suspend("first");
            v = v * 5u + 1u;
            $suspend("second");
            buf.write(tid, v ^ (tid * 17u));
        });

        WavefrontCoroSchedulerConfig cfg{
            .thread_count = capacity,
            .global_memory_soa = true,
            .gather_by_sorting = true,
            .frame_buffer_compaction = true,
        };
        WavefrontCoroScheduler<Buffer<uint>> scheduler{device, coro, cfg};
        expect(scheduler.config().gather_by_sorting) << "sorting gather should not be silently disabled";

        scheduler(output).dispatch(N)(stream);
        luisa::vector<uint> host(N);
        stream << output.copy_to(luisa::span{host}) << synchronize();

        auto ok = true;
        for (auto i = 0u; i < N; i++) {
            auto expected = ((i + 11u) * 5u + 1u) ^ (i * 17u);
            if (host[i] != expected) {
                LUISA_WARNING("wavefront sorting mismatch at {}: got {}, expected {}",
                              i, host[i], expected);
                ok = false;
                break;
            }
        }
        expect(ok) << "sorting gather must preserve coroutine results";
    };

    "wavefront_sorting_gather_handles_aos_without_compaction"_test = [options] {
        constexpr uint N = 211u;
        constexpr uint capacity = 96u;

        auto dc = luisa::test::coro_test::create_device(options);
        auto &device = dc.device;
        Stream stream = device.create_stream();

        auto output = device.create_buffer<uint>(N);

        auto coro = Coroutine<void(Buffer<uint>)>([](BufferUInt buf) {
            auto tid = dispatch_x();
            auto v = tid * 9u + 4u;
            $suspend("first");
            v = (v ^ (tid * 5u + 1u)) + 23u;
            $suspend("second");
            v = v * 3u + (tid & 7u);
            buf.write(tid, v);
        });

        WavefrontCoroSchedulerConfig cfg{
            .thread_count = capacity,
            .global_memory_soa = false,
            .gather_by_sorting = true,
            .frame_buffer_compaction = false,
        };
        WavefrontCoroScheduler<Buffer<uint>> scheduler{device, coro, cfg};
        expect(scheduler.config().global_memory_soa == false);
        expect(scheduler.config().gather_by_sorting == true);
        expect(scheduler.config().frame_buffer_compaction == false);

        scheduler(output).dispatch(N)(stream);
        luisa::vector<uint> host(N);
        stream << output.copy_to(luisa::span{host}) << synchronize();

        auto ok = true;
        for (auto i = 0u; i < N; i++) {
            auto expected = (((i * 9u + 4u) ^ (i * 5u + 1u)) + 23u) * 3u + (i & 7u);
            if (host[i] != expected) {
                LUISA_WARNING("wavefront sorted AoS/no-compaction mismatch at {}: got {}, expected {}",
                              i, host[i], expected);
                ok = false;
                break;
            }
        }
        expect(ok) << "sorted gather must work with AoS frame storage and disabled compaction";
    };

    "wavefront_sorting_gather_option_matrix_preserves_frame_fields"_test = [options] {
        constexpr uint N = 257u;
        constexpr uint capacity = 64u;

        auto dc = luisa::test::coro_test::create_device(options);
        auto &device = dc.device;
        Stream stream = device.create_stream();

        auto output = device.create_buffer<uint>(N);

        auto coro = Coroutine<void(Buffer<uint>)>([](BufferUInt buf) {
            auto tid = dispatch_x();
            auto value = tid * 5u + 13u;
            $suspend("first");
            value = (value ^ (tid * 3u + 7u)) + 19u;
            $suspend("second");
            value = value * 11u + (tid & 31u);
            $suspend("third");
            buf.write(tid, value ^ (tid * 29u + 3u));
        });

        for (auto soa : {false, true}) {
            for (auto compaction : {false, true}) {
                luisa::vector<uint> zero(N);
                stream << output.copy_from(luisa::span{zero}) << synchronize();

                WavefrontCoroSchedulerConfig cfg{
                    .thread_count = capacity,
                    .global_memory_soa = soa,
                    .gather_by_sorting = true,
                    .frame_buffer_compaction = compaction,
                };
                WavefrontCoroScheduler<Buffer<uint>> scheduler{device, coro, cfg};
                scheduler(output).dispatch(N)(stream);

                luisa::vector<uint> host(N);
                stream << output.copy_to(luisa::span{host}) << synchronize();

                auto ok = true;
                for (auto i = 0u; i < N; i++) {
                    auto value = ((i * 5u + 13u) ^ (i * 3u + 7u)) + 19u;
                    value = value * 11u + (i & 31u);
                    auto expected = value ^ (i * 29u + 3u);
                    if (host[i] != expected) {
                        LUISA_WARNING("wavefront sorted matrix mismatch soa={}, compaction={} at {}: got {}, expected {}",
                                      soa, compaction, i, host[i], expected);
                        ok = false;
                        break;
                    }
                }
                expect(ok) << "sorted wavefront gather must preserve frame fields for every layout/compaction combination";
            }
        }
    };

    "wavefront_large_self_loop_queue_preserves_frame_fields"_test = [options] {
        constexpr uint N = 12288u;
        constexpr uint rounds = 11u;
        constexpr uint capacity = N;

        auto dc = luisa::test::coro_test::create_device(options);
        auto &device = dc.device;
        Stream stream = device.create_stream();

        auto output = device.create_buffer<uint>(N);

        auto coro = Coroutine<void(Buffer<uint>)>([](BufferUInt buf) {
            auto tid = dispatch_x();
            auto value = tid * 747796405u + 2891336453u;
            $for (i, 11u) {
                value = value * 1664525u + 1013904223u + i;
                $suspend("loop");
                value = (value ^ (tid + i * 2246822519u)) * 3266489917u;
            };
            buf.write(tid, value ^ (tid * 668265263u));
        });

        auto expected_at = [](uint tid) noexcept {
            auto value = tid * 747796405u + 2891336453u;
            for (auto i = 0u; i < rounds; i++) {
                value = value * 1664525u + 1013904223u + i;
                value = (value ^ (tid + i * 2246822519u)) * 3266489917u;
            }
            return value ^ (tid * 668265263u);
        };

        for (auto gather_by_sorting : {false, true}) {
            luisa::vector<uint> zero(N);
            stream << output.copy_from(luisa::span{zero}) << synchronize();

            WavefrontCoroSchedulerConfig cfg{
                .thread_count = capacity,
                .global_memory_soa = true,
                .gather_by_sorting = gather_by_sorting,
                .frame_buffer_compaction = true,
            };
            WavefrontCoroScheduler<Buffer<uint>> scheduler{device, coro, cfg};
            scheduler(output).dispatch(N)(stream);

            luisa::vector<uint> host(N);
            stream << output.copy_to(luisa::span{host}) << synchronize();

            auto ok = true;
            for (auto i = 0u; i < N; i++) {
                auto expected = expected_at(i);
                if (host[i] != expected) {
                    LUISA_WARNING("wavefront large self-loop mismatch gather_by_sorting={} at {}: got {}, expected {}",
                                  gather_by_sorting, i, host[i], expected);
                    ok = false;
                    break;
                }
            }
            expect(ok) << "large self-loop queues must preserve frame fields and drain correctly";
        }
    };

    "wavefront_hint_fields_are_resolved_and_correct"_test = [options] {
        constexpr uint N = 129u;
        constexpr uint capacity = 64u;

        auto dc = luisa::test::coro_test::create_device(options);
        auto &device = dc.device;
        Stream stream = device.create_stream();

        auto output = device.create_buffer<uint>(N);

        auto coro = Coroutine<void(Buffer<uint>)>([](BufferUInt buf) {
            auto tid = dispatch_x();
            auto coro_hint = (N - 1u) - tid;
            coro_hint.set_name("coro_hint");
            auto v = tid * 7u + 3u;
            $suspend("sort_me");
            v += coro_hint;
            $suspend("done");
            buf.write(tid, v);
        });

        WavefrontCoroSchedulerConfig cfg{
            .thread_count = capacity,
            .global_memory_soa = true,
            .gather_by_sorting = false,
            .frame_buffer_compaction = true,
            .hint_range = N,
            .hint_fields = {"sort_me"},
        };
        WavefrontCoroScheduler<Buffer<uint>> scheduler{device, coro, cfg};
        auto native_hint_sort =
            device.compute_warp_size() == radix_sort::warp_size;
        expect(scheduler.config().hint_fields.size() ==
               static_cast<size_t>(native_hint_sort))
            << "one-sweep hint sorting must be enabled exactly on its "
               "declared subgroup capability";
        if (native_hint_sort) {
            expect(scheduler.config().hint_fields.front() == "sort_me")
                << "hint field should resolve by suspend name";
        }

        scheduler(output).dispatch(N)(stream);
        luisa::vector<uint> host(N);
        stream << output.copy_to(luisa::span{host}) << synchronize();

        auto ok = true;
        for (auto i = 0u; i < N; i++) {
            auto expected = i * 7u + 3u + (N - 1u) - i;
            if (host[i] != expected) {
                LUISA_WARNING("wavefront hint mismatch at {}: got {}, expected {}",
                              i, host[i], expected);
                ok = false;
                break;
            }
        }
        expect(ok) << "hint sorting must preserve coroutine results";
    };

    "wavefront_hint_sort_handles_non_power_of_two_full_bucket"_test = [options] {
        constexpr uint N = 65u;
        constexpr uint capacity = 65u;

        auto dc = luisa::test::coro_test::create_device(options);
        auto &device = dc.device;
        Stream stream = device.create_stream();

        auto output = device.create_buffer<uint>(N);

        auto coro = Coroutine<void(Buffer<uint>)>([](BufferUInt buf) {
            auto tid = dispatch_x();
            auto coro_hint = (tid * 13u) & 63u;
            coro_hint.set_name("coro_hint");
            auto v = tid + 1u;
            $suspend("sort_me");
            buf.write(tid, v + coro_hint);
        });

        WavefrontCoroSchedulerConfig cfg{
            .thread_count = capacity,
            .global_memory_soa = true,
            .gather_by_sorting = false,
            .frame_buffer_compaction = true,
            .hint_range = 64u,
            .hint_fields = {"sort_me"},
        };
        WavefrontCoroScheduler<Buffer<uint>> scheduler{device, coro, cfg};
        expect(scheduler.config().hint_fields.size() == 1u);

        scheduler(output).dispatch(N)(stream);
        luisa::vector<uint> host(N);
        stream << output.copy_to(luisa::span{host}) << synchronize();

        auto ok = true;
        for (auto i = 0u; i < N; i++) {
            auto expected = i + 1u + ((i * 13u) & 63u);
            if (host[i] != expected) {
                LUISA_WARNING("wavefront hint padded-sort mismatch at {}: got {}, expected {}",
                              i, host[i], expected);
                ok = false;
                break;
            }
        }
        expect(ok) << "hint sorting scratch buffers must cover padded sort size";
    };

    "wavefront_hint_sort_works_after_sorted_token_gather"_test = [options] {
        constexpr uint N = 150u;
        constexpr uint capacity = 64u;

        auto dc = luisa::test::coro_test::create_device(options);
        auto &device = dc.device;
        Stream stream = device.create_stream();

        auto output = device.create_buffer<uint>(N);

        auto coro = Coroutine<void(Buffer<uint>)>([](BufferUInt buf) {
            auto tid = dispatch_x();
            auto coro_hint = (tid * 37u + 19u) & 255u;
            coro_hint.set_name("coro_hint");
            auto v = tid * 2u + 5u;
            $suspend("sort_me");
            v = (v + coro_hint) ^ (tid * 3u + 1u);
            $suspend("done");
            buf.write(tid, v + coro_hint);
        });

        WavefrontCoroSchedulerConfig cfg{
            .thread_count = capacity,
            .global_memory_soa = false,
            .gather_by_sorting = true,
            .frame_buffer_compaction = true,
            .hint_range = 256u,
            .hint_fields = {"sort_me"},
        };
        WavefrontCoroScheduler<Buffer<uint>> scheduler{device, coro, cfg};
        expect(scheduler.config().hint_fields.size() ==
               static_cast<size_t>(
                   device.compute_warp_size() ==
                   radix_sort::warp_size));
        expect(scheduler.config().gather_by_sorting == true);

        scheduler(output).dispatch(N)(stream);
        luisa::vector<uint> host(N);
        stream << output.copy_to(luisa::span{host}) << synchronize();

        auto ok = true;
        for (auto i = 0u; i < N; i++) {
            auto hint = (i * 37u + 19u) & 255u;
            auto expected = ((i * 2u + 5u + hint) ^ (i * 3u + 1u)) + hint;
            if (host[i] != expected) {
                LUISA_WARNING("wavefront sorted-token hint mismatch at {}: got {}, expected {}",
                              i, host[i], expected);
                ok = false;
                break;
            }
        }
        expect(ok) << "hint sorting must compose with sorted token gathering";
    };

    "wavefront_hint_sort_radix_range_matrix_preserves_indices"_test = [options] {
        constexpr uint N = 241u;
        constexpr uint capacity = 80u;
        constexpr uint hint_range = 1024u;

        auto dc = luisa::test::coro_test::create_device(options);
        auto &device = dc.device;
        Stream stream = device.create_stream();

        auto output = device.create_buffer<uint>(N);

        auto coro = Coroutine<void(Buffer<uint>)>([](BufferUInt buf) {
            auto tid = dispatch_x();
            auto coro_hint = (tid * 149u + 73u) & 1023u;
            coro_hint.set_name("coro_hint");
            auto value = tid * 17u + 5u;
            $suspend("sort_me");
            value = (value + coro_hint) ^ (tid * 11u + 31u);
            $suspend("done");
            buf.write(tid, value + coro_hint * 3u);
        });

        for (auto gather_by_sorting : {false, true}) {
            for (auto soa : {false, true}) {
                luisa::vector<uint> zero(N);
                stream << output.copy_from(luisa::span{zero}) << synchronize();

                WavefrontCoroSchedulerConfig cfg{
                    .thread_count = capacity,
                    .global_memory_soa = soa,
                    .gather_by_sorting = gather_by_sorting,
                    .frame_buffer_compaction = true,
                    .hint_range = hint_range,
                    .hint_fields = {"sort_me"},
                };
                WavefrontCoroScheduler<Buffer<uint>> scheduler{device, coro, cfg};
                expect(scheduler.config().hint_fields.size() ==
                       static_cast<size_t>(
                           device.compute_warp_size() ==
                           radix_sort::warp_size));

                scheduler(output).dispatch(N)(stream);
                luisa::vector<uint> host(N);
                stream << output.copy_to(luisa::span{host}) << synchronize();

                auto ok = true;
                for (auto i = 0u; i < N; i++) {
                    auto hint = (i * 149u + 73u) & 1023u;
                    auto value = (i * 17u + 5u + hint) ^ (i * 11u + 31u);
                    auto expected = value + hint * 3u;
                    if (host[i] != expected) {
                        LUISA_WARNING("wavefront radix hint mismatch gather_by_sorting={}, soa={} at {}: got {}, expected {}",
                                      gather_by_sorting, soa, i, host[i], expected);
                        ok = false;
                        break;
                    }
                }
                expect(ok) << "radix-range hint sorting must preserve frame indices and values";
            }
        }
    };
}

int main(int argc, char *argv[]) {
    auto options = luisa::test::coro_test::parse_options(argc, argv);
    reg_coro_wavefront(options);
    return luisa::test::coro_test::run_tests(argc, argv);
}
