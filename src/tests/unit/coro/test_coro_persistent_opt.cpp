#include "ut/ut.hpp"
#include "coro_test_utils.h"

#include <luisa/core/logging.h>
#include <luisa/coro/schedulers/persistent.h>
#include <luisa/coro/coro_frame_storage.h>
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

namespace {

void dump_coro_debug_info(const Coroutine<void(Buffer<uint>)> &coro, luisa::string_view tag) noexcept {
    LUISA_INFO("{}: subroutines={} nodes={} fields={} payload={}B struct={}B",
               tag, coro.subroutine_count(), coro.graph().node_count(),
               coro.frame().frame_field_count(), coro.frame().total_size(),
               coro.frame().frame_type()->size());
    LUISA_INFO("{}:\n{}", tag, coro.frame().dump());
    LUISA_INFO("{}:\n{}", tag, coro.graph().dump());
    for (auto i = 0u; i < coro.frame().field_count(); i++) {
        auto &&field = coro.frame().field(i);
        LUISA_INFO("{}: user field {} frame_index={} name={} type={} offset={} size={}",
                   tag, i, i + CoroFrameDesc::reserved_field_count, field.name,
                   field.type->description(), field.offset, field.size);
    }
    for (auto i = 0u; i < coro.subroutine_count(); i++) {
        auto &&node = coro.graph().node(i);
        LUISA_INFO("{}: node {} token={} in={} out={} targets={}",
                   tag, i, node.token, node.input_fields, node.output_fields, node.targets);
    }
}

}// namespace

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
        expect(cfg.global_memory_frames == false);
    };

    "T33_config_custom_values"_test = [] {
        PersistentThreadsCoroSchedulerConfig cfg{
            .thread_count = 1024u,
            .block_size = 64u,
            .fetch_size = 8u,
            .shared_memory_soa = true,
            .global_memory_ext = true,
            .global_memory_frames = true,
        };
        expect(cfg.thread_count == 1024u);
        expect(cfg.block_size == 64u);
        expect(cfg.fetch_size == 8u);
        expect(cfg.shared_memory_soa == true);
        expect(cfg.global_memory_ext == true);
        expect(cfg.global_memory_frames == true);
    };

    "T33_worker_and_fetch_sizes_are_runtime_dispatch_policy"_test = [options] {
        auto dc = luisa::test::coro_test::create_device(options);
        auto &device = dc.device;
        Stream stream = device.create_stream();

        constexpr uint N = 257u;
        auto output = device.create_buffer<uint>(N);
        auto coro = Coroutine<void(Buffer<uint>)>([](BufferUInt buffer) {
            auto i = dispatch_x();
            auto value = i * 17u + 3u;
            $suspend("first");
            value ^= i + 11u;
            $suspend("second");
            // The payload keeps a non-trivial value live across both
            // continuations; the counter proves that the runtime task
            // partition executes each logical instance exactly once.
            $if (value == ((i * 17u + 3u) ^ (i + 11u))) {
                buffer.atomic(i).fetch_add(1u);
            };
        });

        auto make_scheduler = [&](uint worker_count, uint fetch_size) {
            return PersistentThreadsCoroScheduler<Buffer<uint>>{
                device, coro,
                PersistentThreadsCoroSchedulerConfig{
                    .thread_count = worker_count,
                    .block_size = 32u,
                    .fetch_size = fetch_size,
                    .shared_memory_soa = true,
                    .global_memory_ext = true,
                }};
        };
        auto fine = make_scheduler(64u, 1u);
        auto coarse = make_scheduler(96u, 17u);
        expect(fine.main_shader_structure_hash() ==
               coarse.main_shader_structure_hash())
            << "worker and fetch sizes must not enter persistent shader identity";

        luisa::vector<uint> zero(N);
        luisa::vector<uint> fine_result(N);
        luisa::vector<uint> coarse_result(N);
        stream << output.copy_from(luisa::span{zero}) << synchronize();
        fine(output).dispatch(N)(stream);
        stream << output.copy_to(luisa::span{fine_result}) << synchronize();
        stream << output.copy_from(luisa::span{zero}) << synchronize();
        coarse(output).dispatch(N)(stream);
        stream << output.copy_to(luisa::span{coarse_result}) << synchronize();
        expect(fine_result == coarse_result)
            << "runtime fetch partitions must preserve the logical task set";
        expect(std::all_of(
            fine_result.begin(), fine_result.end(),
            [](auto count) noexcept { return count == 1u; }))
            << "every logical task must execute exactly once";
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

    "T33_soa_shared_storage_spills_to_global_frame"_test = [options] {
        constexpr uint N = 64u;
        constexpr uint block_size = 32u;

        auto dc = luisa::test::coro_test::create_device(options);
        auto &device = dc.device;
        Stream stream = device.create_stream();

        CoroFrameDesc desc;
        desc.add_field("a", Type::of<uint>());
        desc.add_field("b", Type::of<uint>());
        auto layout = CoroFrameStorageLayout::make_aos(desc, N);
        auto global_frames = device.create_byte_buffer(layout.size_bytes);
        auto output = device.create_buffer<uint>(N * 6u);

        Kernel1D kernel = [&desc, layout](ByteBufferVar global_frames, BufferUInt out) noexcept {
            set_block_size(block_size, 1u, 1u);
            CoroFrameSharedStorage frames{&desc, block_size, true};
            auto local_id = thread_x();
            auto global_id = dispatch_x();

            auto frame = CoroFrame::create(&desc);
            frame.coro_id = make_uint3(global_id, global_id + 100u, global_id + 200u);
            frame.target_token = global_id + 300u;
            auto a = frame.get<uint>(0u);
            auto b = frame.get<uint>(1u);
            a = global_id * 17u + 3u;
            b = global_id * 23u + 5u;
            frames.write(local_id, frame);
            sync_block();

            auto spilled = frames.read(local_id);
            coro_frame_store(global_frames, global_id, spilled, layout, false, luisa::nullopt, true);
            sync_block();

            auto zero = CoroFrame::create(&desc);
            zero.coro_id = make_uint3(0u);
            zero.target_token = 0u;
            auto za = zero.get<uint>(0u);
            auto zb = zero.get<uint>(1u);
            za = 0u;
            zb = 0u;
            frames.write(local_id, zero);
            sync_block();

            auto restored = coro_frame_load(&desc, global_frames, global_id, layout, false, luisa::nullopt, true);
            frames.write(local_id, restored);
            sync_block();

            auto result = frames.read(local_id);
            auto base = global_id * 6u;
            out.write(base + 0u, result.coro_id.x);
            out.write(base + 1u, result.coro_id.y);
            out.write(base + 2u, result.coro_id.z);
            out.write(base + 3u, result.target_token);
            out.write(base + 4u, result.get<uint>(0u));
            out.write(base + 5u, result.get<uint>(1u));
        };

        auto shader = device.compile(kernel);
        stream << shader(global_frames, output).dispatch(N);

        luisa::vector<uint> host(N * 6u);
        stream << output.copy_to(luisa::span{host}) << synchronize();

        auto ok = true;
        for (auto i = 0u; i < N; i++) {
            auto base = i * 6u;
            uint expected[6]{
                i,
                i + 100u,
                i + 200u,
                i + 300u,
                i * 17u + 3u,
                i * 23u + 5u};
            for (auto j = 0u; j < 6u; j++) {
                if (host[base + j] != expected[j]) {
                    LUISA_WARNING("SoA shared/global frame mismatch i={} field={}: got {}, expected {}",
                                  i, j, host[base + j], expected[j]);
                    ok = false;
                    break;
                }
            }
            if (!ok) { break; }
        }
        expect(ok) << "SoA shared frame storage must spill and restore scalar header fields";
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

        for (auto global_memory_frames : {false, true}) {
            for (auto shared_soa : {false, true}) {
                luisa::vector<uint> zero(N);
                stream << output.copy_from(luisa::span{zero}) << synchronize();

                PersistentThreadsCoroSchedulerConfig cfg{
                    .thread_count = thread_count,
                    .block_size = block_size,
                    .fetch_size = 4u,
                    .shared_memory_soa = shared_soa,
                    .global_memory_ext = !global_memory_frames,
                    .global_memory_frames = global_memory_frames,
                };
                PersistentThreadsCoroScheduler<Buffer<uint>> scheduler{device, coro, cfg};
                scheduler(output).dispatch(N)(stream);

                luisa::vector<uint> host(N);
                stream << output.copy_to(luisa::span{host}) << synchronize();

                auto ok = true;
                for (auto i = 0u; i < N; i++) {
                    auto expected = ((i * 17u + 3u) ^ (i + 11u)) + i * 5u + 7u;
                    if (host[i] != expected) {
                        LUISA_WARNING("persistent GME spill mismatch global_frames={} shared_soa={} at {}: got {}, expected {}",
                                      global_memory_frames, shared_soa, i, host[i], expected);
                        ok = false;
                        break;
                    }
                }
                if (!ok) {
                    auto zero_count = 0u;
                    auto expected_count = 0u;
                    auto other_count = 0u;
                    auto first_expected = N;
                    auto last_expected = 0u;
                    for (auto i = 0u; i < N; i++) {
                        auto expected = ((i * 17u + 3u) ^ (i + 11u)) + i * 5u + 7u;
                        if (host[i] == 0u) {
                            zero_count++;
                        } else if (host[i] == expected) {
                            expected_count++;
                            first_expected = std::min(first_expected, i);
                            last_expected = std::max(last_expected, i);
                        } else {
                            other_count++;
                        }
                    }
                    LUISA_WARNING("persistent GME summary global_frames={} shared_soa={}: zero={} expected={} other={} expected_range=[{}, {}]",
                                  global_memory_frames, shared_soa, zero_count, expected_count, other_count, first_expected, last_expected);
                }
                expect(ok) << "persistent global-memory frame representation must preserve every continuation field";
                expect(scheduler.config().global_memory_ext == true);
                expect(scheduler.config().global_memory_frames ==
                       global_memory_frames);
                expect(scheduler.config().shared_memory_soa == shared_soa);
            }
        }
    };

    "T33_shared_frame_lower_bound_selects_global_frames"_test = [options] {
        auto dc = luisa::test::coro_test::create_device(options);
        auto &device = dc.device;
        Stream stream = device.create_stream();
        auto block_quantum =
            std::max(device.compute_warp_size(), 32u);
        auto output_count = block_quantum * 2u + 3u;
        auto output = device.create_buffer<uint>(output_count);

        auto coro = Coroutine<void(Buffer<uint>)>([](BufferUInt buf) {
            auto tid = dispatch_x();
            auto value = tid * 17u + 3u;
            $suspend("live-frame-a");
            value = value ^ (tid + 11u);
            $suspend("live-frame-b");
            buf.write(tid, value + 7u);
        });

        PersistentThreadsCoroSchedulerConfig shared_config{
            .thread_count = block_quantum,
            .block_size = block_quantum,
            .global_memory_ext = true,
        };
        PersistentThreadsCoroScheduler<Buffer<uint>> shared{
            device, coro, shared_config};

        auto global_config = shared_config;
        global_config.global_memory_frames = true;
        PersistentThreadsCoroScheduler<Buffer<uint>> global{
            device, coro, global_config};
        expect(shared.static_shared_memory_size_bytes() >
               global.static_shared_memory_size_bytes())
            << "removing per-slot shared frames must reduce the irreducible "
               "one-quantum shared-memory lower bound";

        auto automatic_config = shared_config;
        automatic_config.shared_memory_limit_bytes =
            global.static_shared_memory_size_bytes();
        PersistentThreadsCoroScheduler<Buffer<uint>> automatic{
            device, coro, automatic_config};
        expect(automatic.config().global_memory_frames)
            << "when one shared-frame wave cannot fit, resource normalization "
               "must change frame representation instead of naming a backend";
        expect(automatic.static_shared_memory_size_bytes() <=
               automatic_config.shared_memory_limit_bytes);

        luisa::vector<uint> zero(output_count);
        stream << output.copy_from(luisa::span{zero});
        automatic(output).dispatch(output_count)(stream);
        luisa::vector<uint> host(output_count);
        stream << output.copy_to(luisa::span{host}) << synchronize();
        auto correct = true;
        for (auto i = 0u; i < output_count; ++i) {
            correct &= host[i] ==
                       ((i * 17u + 3u) ^ (i + 11u)) + 7u;
        }
        expect(correct)
            << "automatic global-frame selection must preserve the slot/token "
               "state-transition result";
    };

    "T33_GME_soa_progress_by_suspend_count"_test = [options] {
        constexpr uint N = 257u;

        auto dc = luisa::test::coro_test::create_device(options);
        auto &device = dc.device;
        Stream stream = device.create_stream();
        auto output = device.create_buffer<uint>(N);

        auto test_case = [&](auto make_coro, luisa::string_view tag, uint expected_offset) {
            auto coro = make_coro();
            luisa::vector<uint> zero(N);
            stream << output.copy_from(luisa::span{zero}) << synchronize();
            PersistentThreadsCoroSchedulerConfig cfg{
                .thread_count = 64u,
                .block_size = 32u,
                .fetch_size = 4u,
                .shared_memory_soa = true,
                .global_memory_ext = true,
            };
            PersistentThreadsCoroScheduler<Buffer<uint>> scheduler{device, coro, cfg};
            scheduler(output).dispatch(N)(stream);
            luisa::vector<uint> host(N);
            stream << output.copy_to(luisa::span{host}) << synchronize();
            auto ok = true;
            for (auto i = 0u; i < N; i++) {
                auto expected = i + expected_offset;
                if (host[i] != expected) {
                    LUISA_WARNING("{} mismatch at {}: got {}, expected {}",
                                  tag, i, host[i], expected);
                    ok = false;
                    break;
                }
            }
            if (!ok) { dump_coro_debug_info(coro, tag); }
            if (!ok) {
                uint zero_count = 0u;
                uint final_count = 0u;
                uint other_count = 0u;
                for (auto i = 0u; i < N; i++) {
                    auto expected = i + expected_offset;
                    if (host[i] == 0u) {
                        zero_count++;
                    } else if (host[i] == expected) {
                        final_count++;
                    } else {
                        other_count++;
                    }
                }
                LUISA_WARNING("{} summary: zero={} final={} other={}",
                              tag, zero_count, final_count, other_count);
                bool in_final = false;
                uint range_begin = 0u;
                for (auto i = 0u; i <= N; i++) {
                    auto is_final = i < N && host[i] == i + expected_offset;
                    if (is_final && !in_final) {
                        range_begin = i;
                        in_final = true;
                    } else if (!is_final && in_final) {
                        LUISA_WARNING("{} final range [{}, {})", tag, range_begin, i);
                        in_final = false;
                    }
                }
            }
            return ok;
        };

        auto one = test_case([] {
            return Coroutine<void(Buffer<uint>)>([](BufferUInt buf) {
                auto tid = dispatch_x();
                auto value = tid + 1u;
                $suspend("a");
                buf.write(tid, value);
            });
        },
                             "one_suspend", 1u);
        auto two = test_case([] {
            return Coroutine<void(Buffer<uint>)>([](BufferUInt buf) {
                auto tid = dispatch_x();
                auto value = tid + 1u;
                $suspend("a");
                value += 1u;
                $suspend("b");
                buf.write(tid, value);
            });
        },
                             "two_suspend", 2u);
        auto three = test_case([] {
            return Coroutine<void(Buffer<uint>)>([](BufferUInt buf) {
                auto tid = dispatch_x();
                auto value = tid + 1u;
                $suspend("a");
                value += 1u;
                $suspend("b");
                value += 1u;
                $suspend("c");
                buf.write(tid, value);
            });
        },
                               "three_suspend", 3u);
        expect(one);
        expect(two);
        expect(three);
    };

    "T33_GME_soa_three_suspend_progress_standalone"_test = [options] {
        constexpr uint N = 257u;

        auto dc = luisa::test::coro_test::create_device(options);
        auto &device = dc.device;
        Stream stream = device.create_stream();
        auto output = device.create_buffer<uint>(N);

        auto coro = Coroutine<void(Buffer<uint>)>([](BufferUInt buf) {
            auto tid = dispatch_x();
            auto value = tid + 1u;
            $suspend("a");
            value += 1u;
            $suspend("b");
            value += 1u;
            $suspend("c");
            buf.write(tid, value);
        });

        luisa::vector<uint> zero(N);
        stream << output.copy_from(luisa::span{zero}) << synchronize();
        PersistentThreadsCoroSchedulerConfig cfg{
            .thread_count = 64u,
            .block_size = 32u,
            .fetch_size = 4u,
            .shared_memory_soa = true,
            .global_memory_ext = true,
        };
        PersistentThreadsCoroScheduler<Buffer<uint>> scheduler{device, coro, cfg};
        scheduler(output).dispatch(N)(stream);
        luisa::vector<uint> host(N);
        stream << output.copy_to(luisa::span{host}) << synchronize();

        auto ok = true;
        for (auto i = 0u; i < N; i++) {
            auto expected = i + 3u;
            if (host[i] != expected) {
                LUISA_WARNING("standalone three-suspend mismatch at {}: got {}, expected {}",
                              i, host[i], expected);
                ok = false;
                break;
            }
        }
        expect(ok) << "three-suspend GME/SoA scheduler must progress when compiled standalone";
    };

    "T33_GME_single_block_spills_and_preserves_frame_fields"_test = [options] {
        constexpr uint N = 65u;
        constexpr uint thread_count = 32u;
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

        luisa::vector<uint> zero(N);
        stream << output.copy_from(luisa::span{zero}) << synchronize();

        PersistentThreadsCoroSchedulerConfig cfg{
            .thread_count = thread_count,
            .block_size = block_size,
            .fetch_size = 4u,
            .shared_memory_soa = true,
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
                LUISA_WARNING("persistent single-block GME spill mismatch at {}: got {}, expected {}",
                              i, host[i], expected);
                ok = false;
                break;
            }
        }
        expect(ok) << "persistent single-block GME must spill and restore frame fields";
    };

    "T33_no_GME_preserves_frame_fields_across_oversubscribed_dispatch"_test = [options] {
        constexpr uint N = 257u;
        constexpr uint thread_count = 64u;
        constexpr uint block_size = 32u;

        auto dc = luisa::test::coro_test::create_device(options);
        auto &device = dc.device;
        Stream stream = device.create_stream();

        auto output = device.create_buffer<uint>(N);

        for (auto shared_soa : {false, true}) {
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
            if (!ok) {
                auto zero_count = 0u;
                auto expected_count = 0u;
                auto other_count = 0u;
                for (auto i = 0u; i < N; i++) {
                    auto expected = ((i * 23u + 9u) * 3u + (i & 15u)) ^ (i * 7u + 5u);
                    expected += 41u;
                    if (host[i] == 0u) {
                        zero_count++;
                    } else if (host[i] == expected) {
                        expected_count++;
                    } else {
                        other_count++;
                    }
                }
                LUISA_WARNING("persistent no-GME summary shared_soa={}: zero={} expected={} other={}",
                              shared_soa, zero_count, expected_count, other_count);
                auto in_expected = false;
                auto begin = 0u;
                for (auto i = 0u; i <= N; i++) {
                    auto expected = i < N ? (((i * 23u + 9u) * 3u + (i & 15u)) ^ (i * 7u + 5u)) + 41u : 0u;
                    auto is_expected = i < N && host[i] == expected;
                    if (is_expected && !in_expected) {
                        begin = i;
                        in_expected = true;
                    } else if (!is_expected && in_expected) {
                        LUISA_WARNING("persistent no-GME expected range shared_soa={}: [{}, {})",
                                      shared_soa, begin, i);
                        in_expected = false;
                    }
                }
            }
            expect(ok) << "persistent scheduler must preserve frames without global-memory spill";
            expect(scheduler.config().global_memory_ext == false);
            expect(scheduler.config().shared_memory_soa == shared_soa);
        }
    };

    "T33_no_GME_soa_passes_without_prior_aos_scheduler"_test = [options] {
        constexpr uint N = 257u;
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

        luisa::vector<uint> zero(N);
        stream << output.copy_from(luisa::span{zero}) << synchronize();
        PersistentThreadsCoroSchedulerConfig cfg{
            .thread_count = 64u,
            .block_size = 32u,
            .fetch_size = 2u,
            .shared_memory_soa = true,
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
                LUISA_WARNING("standalone no-GME SoA mismatch at {}: got {}, expected {}", i, host[i], expected);
                ok = false;
                break;
            }
        }
        expect(ok) << "standalone no-GME SoA persistent scheduler should complete all work";
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

    "T35_fetch_size_5_gme_only_preserves_correctness"_test = [options] {
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

        luisa::vector<uint> zero(N);
        stream << output.copy_from(luisa::span{zero}) << synchronize();

        PersistentThreadsCoroSchedulerConfig cfg{
            .thread_count = 70u,
            .block_size = 32u,
            .fetch_size = 5u,
            .global_memory_ext = true,
        };
        PersistentThreadsCoroScheduler<Buffer<uint>> scheduler{device, coro, cfg};
        expect(scheduler.config().thread_count == 96u) << "thread_count should align to block_size";

        scheduler(output).dispatch(N)(stream);

        luisa::vector<uint> host(N);
        stream << output.copy_to(luisa::span{host}) << synchronize();

        auto ok = true;
        for (auto i = 0u; i < N; i++) {
            auto expected = ((i + 3u) * 19u + (i & 7u)) ^ 0x5a5au;
            if (host[i] != expected) {
                LUISA_WARNING("persistent fetch/alignment GME-only mismatch at {}: got {}, expected {}",
                              i, host[i], expected);
                ok = false;
                break;
            }
        }
        expect(ok) << "persistent fetch-size=5 GME-only variant must preserve results";
    };
}

int main(int argc, char *argv[]) {
    auto options = luisa::test::coro_test::parse_options(argc, argv);
    reg_coro_persistent_opt(options);
    return luisa::test::coro_test::run_tests(argc, argv);
}
