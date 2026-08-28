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

#include <algorithm>
#include <array>
#include <bit>

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

    "T33_sparse_aos_load_updates_only_the_active_sum_variant"_test = [options] {
        constexpr uint N = 64u;
        constexpr auto active_user_field =
            CoroFrameDesc::reserved_field_count;
        constexpr std::array<size_t, 1u> active_fields{
            active_user_field};

        CoroFrameDesc desc;
        desc.add_field("active", Type::of<uint>());
        desc.add_field("inactive_a", Type::of<uint>());
        desc.add_field("inactive_b", Type::of<uint>());
        auto layout = CoroFrameStorageLayout::make_aos(desc, N);

        auto dc = luisa::test::coro_test::create_device(options);
        auto &device = dc.device;
        Stream stream = device.create_stream();
        auto frames = device.create_byte_buffer(layout.size_bytes);
        auto output = device.create_buffer<uint>(N * 20u);

        Kernel1D store = [&desc, layout](ByteBufferVar frames) noexcept {
            auto tid = dispatch_x();
            auto frame = CoroFrame::create(&desc);
            frame.coro_id = make_uint3(tid + 1u, tid + 2u, tid + 3u);
            frame.dispatch_size_x = tid + 4u;
            frame.dispatch_size_y = tid + 5u;
            frame.dispatch_size_z = tid + 6u;
            frame.target_token = tid + 7u;
            auto active = frame.get<uint>(0u);
            auto inactive_a = frame.get<uint>(1u);
            auto inactive_b = frame.get<uint>(2u);
            active = tid + 8u;
            inactive_a = tid + 9u;
            inactive_b = tid + 10u;
            coro_frame_store(
                frames, tid, frame, layout, false,
                luisa::nullopt, true);
        };
        Kernel1D load = [&desc, layout, active_fields](
                            ByteBufferVar frames,
                            BufferUInt output) noexcept {
            auto tid = dispatch_x();
            auto frame = CoroFrame::create(&desc);
            frame.coro_id = make_uint3(101u, 102u, 103u);
            frame.dispatch_size_x = 104u;
            frame.dispatch_size_y = 105u;
            frame.dispatch_size_z = 106u;
            frame.target_token = 107u;
            auto active = frame.get<uint>(0u);
            auto inactive_a = frame.get<uint>(1u);
            auto inactive_b = frame.get<uint>(2u);
            active = 108u;
            inactive_a = 109u;
            inactive_b = 110u;
            coro_frame_load_into(
                frame, frames, tid, layout, false,
                luisa::span{active_fields}, true);

            auto base = tid * 20u;
            output.write(base + 0u, frame.coro_id.x);
            output.write(base + 1u, frame.coro_id.y);
            output.write(base + 2u, frame.coro_id.z);
            output.write(base + 3u, frame.dispatch_size_x);
            output.write(base + 4u, frame.dispatch_size_y);
            output.write(base + 5u, frame.dispatch_size_z);
            output.write(base + 6u, frame.target_token);
            output.write(base + 7u, frame.get<uint>(0u));
            output.write(base + 8u, frame.get<uint>(1u));
            output.write(base + 9u, frame.get<uint>(2u));

            auto exact = CoroFrame::create(&desc);
            exact.coro_id = make_uint3(201u, 202u, 203u);
            exact.dispatch_size_x = 204u;
            exact.dispatch_size_y = 205u;
            exact.dispatch_size_z = 206u;
            exact.target_token = 207u;
            auto exact_active = exact.get<uint>(0u);
            auto exact_inactive_a = exact.get<uint>(1u);
            auto exact_inactive_b = exact.get<uint>(2u);
            exact_active = 208u;
            exact_inactive_a = 209u;
            exact_inactive_b = 210u;
            coro_frame_load_into(
                exact, frames, tid, layout, false,
                luisa::span{active_fields}, true, false);
            output.write(base + 10u, exact.coro_id.x);
            output.write(base + 11u, exact.coro_id.y);
            output.write(base + 12u, exact.coro_id.z);
            output.write(base + 13u, exact.dispatch_size_x);
            output.write(base + 14u, exact.dispatch_size_y);
            output.write(base + 15u, exact.dispatch_size_z);
            output.write(base + 16u, exact.target_token);
            output.write(base + 17u, exact.get<uint>(0u));
            output.write(base + 18u, exact.get<uint>(1u));
            output.write(base + 19u, exact.get<uint>(2u));
        };

        auto store_shader = device.compile(store);
        auto load_shader = device.compile(load);
        stream << store_shader(frames).dispatch(N)
               << load_shader(frames, output).dispatch(N);
        luisa::vector<uint> host(output.size());
        stream << output.copy_to(luisa::span{host}) << synchronize();

        auto correct = true;
        for (auto i = 0u; i < N && correct; i++) {
            const std::array<uint, 20u> expected{
                i + 1u, i + 2u, i + 3u, i + 4u, i + 5u,
                i + 6u, i + 7u, i + 8u, 109u, 110u,
                201u, 202u, 203u, 204u, 205u,
                206u, 207u, i + 8u, 209u, 210u};
            for (auto j = 0u; j < expected.size(); j++) {
                if (host[i * expected.size() + j] != expected[j]) {
                    LUISA_WARNING(
                        "sparse AoS load mismatch instance={} field={}: "
                        "got {}, expected {}",
                        i, j, host[i * expected.size() + j], expected[j]);
                    correct = false;
                    break;
                }
            }
        }
        expect(correct)
            << "token-indexed loads must restore reserved and active fields "
               "without overwriting another sum variant's inactive payload";
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

    "T33_GME_preserves_state_live_through_an_intermediate_continuation"_test = [options] {
        constexpr uint N = 257u;
        auto dc = luisa::test::coro_test::create_device(options);
        auto &device = dc.device;
        Stream stream = device.create_stream();
        auto output = device.create_buffer<uint>(N * 2u);

        auto coro = Coroutine<void(Buffer<uint>)>([](BufferUInt output) {
            auto tid = dispatch_x();
            auto early = make_float3(
                cast<float>(tid) + 0.25f,
                cast<float>(tid) + 0.5f,
                cast<float>(tid) + 0.75f);
            // `late` is already part of the queued frame, but the first
            // continuation neither reads nor redefines it. It must survive
            // relocation through that continuation for the second one.
            auto late = def(tid * 17u + 11u);
            $suspend("early_float3");
            output.write(tid * 2u, early.x.as<uint>());
            $suspend("late_uint");
            output.write(tid * 2u + 1u, late);
        });

        auto *early_node = coro.graph().node_by_name("early_float3");
        auto *late_node = coro.graph().node_by_name("late_uint");
        expect(early_node != nullptr);
        expect(late_node != nullptr);
        if (early_node != nullptr && late_node != nullptr) {
            expect(early_node->input_fields != late_node->input_fields)
                << "different continuation payload types must form distinct "
                   "token variants";
            auto relocation_fields = coro_frame_collect_relocation_fields(
                coro.graph(), coro.frame().frame_field_count());
            auto early_index = early_node->index;
            auto late_only = std::find_if(
                late_node->input_fields.begin(),
                late_node->input_fields.end(),
                [&](auto field) noexcept {
                    return std::find(
                               early_node->input_fields.begin(),
                               early_node->input_fields.end(), field) ==
                           early_node->input_fields.end();
                });
            expect(late_only != late_node->input_fields.end())
                << "the fixture must contain a field first consumed after "
                   "the intermediate continuation";
            if (late_only != late_node->input_fields.end()) {
                expect(std::find(
                           relocation_fields[early_index].begin(),
                           relocation_fields[early_index].end(),
                           *late_only) !=
                       relocation_fields[early_index].end())
                    << "a field read later and not redefined on the edge must "
                       "remain live through the intermediate queue token";
            }
            expect(early_node->input_fields.size() <
                   coro.frame().frame_field_count());
            expect(late_node->input_fields.size() <
                   coro.frame().frame_field_count());
        }

        PersistentThreadsCoroSchedulerConfig config{
            .thread_count = 64u,
            .block_size = 32u,
            .fetch_size = 4u,
            .shared_memory_soa = true,
            .global_memory_ext = true,
        };
        PersistentThreadsCoroScheduler<Buffer<uint>> scheduler{
            device, coro, config};
        scheduler(output).dispatch(N)(stream);
        luisa::vector<uint> host(output.size());
        stream << output.copy_to(luisa::span{host}) << synchronize();

        auto correct = true;
        for (auto i = 0u; i < N && correct; i++) {
            auto expected_early =
                std::bit_cast<uint>(static_cast<float>(i) + 0.25f);
            auto expected_late = i * 17u + 11u;
            if (host[i * 2u] != expected_early ||
                host[i * 2u + 1u] != expected_late) {
                LUISA_WARNING(
                    "disjoint persistent live-in mismatch at {}: "
                    "got ({}, {}), expected ({}, {})",
                    i, host[i * 2u], host[i * 2u + 1u],
                    expected_early, expected_late);
                correct = false;
            }
        }
        expect(correct)
            << "GME exchange must preserve both immediate live-ins and state "
               "that is live through an intermediate continuation";
    };

    "T33_persistent_frame_io_is_exact_per_graph_edge"_test = [options] {
        constexpr uint N = 64u;
        auto dc = luisa::test::coro_test::create_device(options);
        auto &device = dc.device;
        Stream stream = device.create_stream();
        auto output = device.create_buffer<uint>(N);

        auto coro = Coroutine<void(Buffer<uint>)>([](BufferUInt output) {
            auto tid = dispatch_x();
            $suspend("route_payload");
            $if ((tid & 1u) == 0u) {
                // Distinct physical types prevent the mutually exclusive
                // payloads from sharing a colored frame slot.
                auto even = make_uint4(
                    tid + 1u, tid + 2u, tid + 3u, tid + 4u);
                $suspend("even_payload");
                output.write(
                    tid, even.x + even.y + even.z + even.w);
            }
            $else {
                auto base = cast<float>(tid);
                auto odd = make_float3(
                    base + 5.0f, base + 6.0f, base + 7.0f);
                $suspend("odd_payload");
                // This builtin is first used in the late continuation. Its
                // immutable field must survive through route_payload, but it
                // must not be rewritten on route_payload -> odd_payload.
                output.write(
                    tid,
                    cast<uint>(odd.x + odd.y + odd.z) +
                        dispatch_size_x());
            };
        });

        auto *route = coro.graph().node_by_name("route_payload");
        auto *even = coro.graph().node_by_name("even_payload");
        auto *odd = coro.graph().node_by_name("odd_payload");
        expect(route != nullptr && even != nullptr && odd != nullptr);
        if (route == nullptr || even == nullptr || odd == nullptr) { return; }

        PersistentThreadsCoroSchedulerConfig config{
            .thread_count = N,
            .block_size = 32u,
            .fetch_size = 4u,
            .global_memory_ext = true,
            .global_memory_frames = true,
        };
        PersistentThreadsCoroScheduler<Buffer<uint>> scheduler{
            device, coro, config};
        auto &&plan = scheduler.frame_io_plan();
        auto contains = [](luisa::span<const size_t> fields,
                           size_t field) noexcept {
            return std::find(fields.begin(), fields.end(), field) !=
                   fields.end();
        };

        auto even_store = plan.output(route->index, even->index);
        auto odd_store = plan.output(route->index, odd->index);
        auto even_only = std::find_if(
            even_store.begin(), even_store.end(),
            [&](auto field) noexcept {
                return field >= CoroFrameDesc::reserved_field_count &&
                       !contains(odd_store, field);
            });
        auto odd_only = std::find_if(
            odd_store.begin(), odd_store.end(),
            [&](auto field) noexcept {
                return field >= CoroFrameDesc::reserved_field_count &&
                       !contains(even_store, field);
            });
        expect(even_only != even_store.end());
        expect(odd_only != odd_store.end());
        expect(plan.output(even->index, odd->index).empty())
            << "an impossible continuation edge must generate no frame writes";
        expect(even_store.size() < route->output_fields.size());
        expect(odd_store.size() < route->output_fields.size())
            << "each transition must use its edge projection, not the "
               "source-node union";

        constexpr size_t dispatch_size_x_field = 3u;
        expect(contains(plan.input(odd->index), dispatch_size_x_field));
        expect(contains(plan.relocation(route->index),
                        dispatch_size_x_field));
        expect(contains(plan.output(0u, route->index),
                        dispatch_size_x_field));
        expect(!contains(odd_store, dispatch_size_x_field))
            << "immutable invocation identity is initialized once and "
               "retained, not rewritten on every transition";

        constexpr size_t target_token_field = 6u;
        for (auto node = 0u; node < coro.subroutine_count(); ++node) {
            expect(!contains(plan.input(node), target_token_field));
            expect(!contains(plan.relocation(node), target_token_field));
            for (auto target = 0u;
                 target < coro.subroutine_count(); ++target) {
                expect(!contains(plan.output(node, target),
                                 target_token_field));
            }
        }

        scheduler(output).dispatch(N)(stream);
        luisa::vector<uint> host(N);
        stream << output.copy_to(luisa::span{host}) << synchronize();
        auto correct = true;
        for (auto tid = 0u; tid < N; ++tid) {
            auto expected = (tid & 1u) == 0u ?
                                4u * tid + 10u :
                                3u * tid + 18u + N;
            if (host[tid] != expected) {
                LUISA_WARNING(
                    "edge-exact persistent I/O mismatch at {}: got {}, "
                    "expected {}",
                    tid, host[tid], expected);
                correct = false;
                break;
            }
        }
        expect(correct)
            << "the exact edge plan must preserve both tagged payloads and "
               "dormant immutable identity through global-frame exchange";
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
