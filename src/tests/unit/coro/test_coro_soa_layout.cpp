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

#include <algorithm>
#include <bit>
#include <utility>

using namespace luisa;
using namespace luisa::compute;
using namespace luisa::compute::coro;
using namespace boost::ut;
using namespace boost::ut::literals;

struct CoroSoASevenWords {
    uint a;
    uint b;
    uint c;
    uint d;
    uint e;
    uint f;
    uint g;
};

LUISA_STRUCT(CoroSoASevenWords, a, b, c, d, e, f, g) {};

void reg_coro_soa_layout(luisa::test::coro_test::Options options) {

    "mutually_exclusive_suspend_edges_share_frame_storage"_test = [] {
        auto coro = Coroutine<void(Buffer<uint>)>([](BufferUInt output) {
            auto index = dispatch_x();
            // Both values are defined before the control split on purpose:
            // their source-level ranges overlap, but no dynamic continuation
            // ever needs them together. Frame interference is therefore an
            // edge-live property, not a lexical/source-range property.
            auto left = def(index * 3u + 1u);
            auto right = def(index * 5u + 2u);
            $if ((index & 1u) == 0u) {
                $suspend("A", coro_frame_export(
                                  "branch_left_payload", left));
                output.write(index, left);
            }
            $else {
                $suspend("B", coro_frame_export(
                                  "branch_right_payload", right));
                output.write(index, right);
            };
            $suspend("C");
        });

        const auto *entry = &coro.graph().node(0u);
        const auto *a = coro.graph().node_by_name("A");
        const auto *b = coro.graph().node_by_name("B");
        const auto *c = coro.graph().node_by_name("C");
        expect(a != nullptr);
        expect(b != nullptr);
        expect(c != nullptr);
        if (a != nullptr && b != nullptr && c != nullptr) {
            expect(coro.graph().edge(entry->index, a->index) != nullptr);
            expect(coro.graph().edge(entry->index, b->index) != nullptr);
            expect(coro.graph().edge(a->index, c->index) != nullptr);
            expect(coro.graph().edge(b->index, c->index) != nullptr);
            expect(coro.graph().edge(a->index, b->index) == nullptr);
            expect(coro.graph().edge(b->index, a->index) == nullptr);
        }

        auto left_field =
            coro.frame().field_index("branch_left_payload");
        auto right_field =
            coro.frame().field_index("branch_right_payload");
        expect(left_field != static_cast<size_t>(-1));
        expect(right_field != static_cast<size_t>(-1));
        expect(left_field == right_field)
            << "mutually exclusive edge-live values of the same type must "
               "reuse one physical frame slot";
    };

    "runtime_soa_layout_is_linear_and_capacity_invariant"_test = [] {
        auto coro = Coroutine<void(Buffer<float4>)>([](BufferFloat4 output) {
            auto i = dispatch_x();
            auto scalar = def(cast<float>(i) + 0.5f);
            auto vector = make_float3(scalar, scalar + 1.0f, scalar + 2.0f);
            auto tag = def(i * 7u + 3u);
            $suspend("mixed_alignment");
            output.write(i, make_float4(vector * scalar, cast<float>(tag)));
        });

        constexpr size_t small_capacity = 17u;
        constexpr size_t large_capacity = 257u;
        auto small = CoroFrameStorageLayout::make_runtime_soa(
            coro.frame(), small_capacity);
        auto large = CoroFrameStorageLayout::make_runtime_soa(
            coro.frame(), large_capacity);
        expect(small.has_runtime_capacity());
        expect(large.has_runtime_capacity());
        expect(small.frame_stride == large.frame_stride);
        expect(small.field_strides == large.field_strides);
        expect(small.field_capacity_strides ==
               large.field_capacity_strides);
        expect(small.size_bytes == small.frame_stride * small_capacity);
        expect(large.size_bytes == large.frame_stride * large_capacity);

        for (auto capacity : {small_capacity, large_capacity}) {
            luisa::vector<std::pair<size_t, size_t>> ranges;
            ranges.reserve(coro.frame().frame_field_count());
            for (auto i = 0u; i < coro.frame().frame_field_count(); i++) {
                auto alignment = std::max<size_t>(
                    coro.frame().frame_field_type(i)->alignment(), 4u);
                auto begin = small.field_offsets[i] +
                             capacity * small.field_capacity_strides[i];
                auto end = begin + capacity * small.field_strides[i];
                expect(begin % alignment == 0u)
                    << "every field array base must satisfy its ABI alignment";
                ranges.emplace_back(begin, end);
            }
            std::sort(ranges.begin(), ranges.end());
            auto cursor = size_t{0u};
            for (auto [begin, end] : ranges) {
                expect(begin == cursor)
                    << "runtime SoA field arrays must form one disjoint partition";
                expect(end >= begin);
                cursor = end;
            }
            expect(cursor == small.frame_stride * capacity);
        }
    };

    "selective_frame_copy_does_not_touch_inactive_fields"_test = [options] {
        constexpr uint frame_capacity = 4u;
        constexpr uint source_index = 3u;
        constexpr uint destination_index = 1u;
        constexpr uint output_count = 6u;

        CoroFrameDesc desc;
        desc.add_field("selected_uint", Type::of<uint>());
        desc.add_field("inactive_float", Type::of<float>());
        desc.add_field("selected_uint_2", Type::of<uint>());
        desc.add_field("inactive_float3", Type::of<float3>());
        auto selected_fields = luisa::vector<size_t>{0u, 2u};

        auto dc = luisa::test::coro_test::create_device(options);
        auto &device = dc.device;
        auto stream = device.create_stream();

        for (auto soa : {false, true}) {
            auto layout = soa ?
                              CoroFrameStorageLayout::make_runtime_soa(
                                  desc, frame_capacity) :
                              CoroFrameStorageLayout::make_aos(
                                  desc, frame_capacity);
            auto frames = device.create_byte_buffer(layout.size_bytes);
            auto output = device.create_buffer<uint>(output_count);
            Kernel1D copy = [&desc, layout, soa,
                             selected_fields](ByteBufferVar frames,
                                              BufferUInt output) noexcept {
                auto frame_buf = Expr<ByteBuffer>{frames};
                coro_frame_write_field(
                    frame_buf, source_index, frame_capacity,
                    layout, soa, 0u, 101u);
                coro_frame_write_field(
                    frame_buf, source_index, frame_capacity,
                    layout, soa, 1u, 2.5f);
                coro_frame_write_field(
                    frame_buf, source_index, frame_capacity,
                    layout, soa, 2u, 303u);
                coro_frame_write_field(
                    frame_buf, source_index, frame_capacity,
                    layout, soa, 3u, make_float3(4.5f, 5.5f, 6.5f));

                coro_frame_write_field(
                    frame_buf, destination_index, frame_capacity,
                    layout, soa, 0u, 11u);
                coro_frame_write_field(
                    frame_buf, destination_index, frame_capacity,
                    layout, soa, 1u, -2.0f);
                coro_frame_write_field(
                    frame_buf, destination_index, frame_capacity,
                    layout, soa, 2u, 33u);
                coro_frame_write_field(
                    frame_buf, destination_index, frame_capacity,
                    layout, soa, 3u, make_float3(-4.0f, -5.0f, -6.0f));

                coro_frame_copy_fields(
                    frame_buf, source_index, destination_index,
                    frame_capacity, &desc, layout, soa,
                    luisa::span<const size_t>{selected_fields});

                output.write(0u, coro_frame_read_field<uint>(
                                     frame_buf, destination_index,
                                     frame_capacity, layout, soa, 0u));
                output.write(1u, coro_frame_read_field<float>(
                                     frame_buf, destination_index,
                                     frame_capacity, layout, soa, 1u)
                                     .as<uint>());
                output.write(2u, coro_frame_read_field<uint>(
                                     frame_buf, destination_index,
                                     frame_capacity, layout, soa, 2u));
                auto inactive = coro_frame_read_field<float3>(
                    frame_buf, destination_index, frame_capacity,
                    layout, soa, 3u);
                output.write(3u, inactive.x.as<uint>());
                output.write(4u, inactive.y.as<uint>());
                output.write(5u, inactive.z.as<uint>());
            };
            auto shader = device.compile(copy);
            luisa::vector<uint> host(output_count);
            stream << shader(frames, output).dispatch(1u)
                   << output.copy_to(luisa::span{host})
                   << synchronize();

            auto expected = luisa::vector<uint>{
                101u, std::bit_cast<uint>(-2.0f), 303u,
                std::bit_cast<uint>(-4.0f),
                std::bit_cast<uint>(-5.0f),
                std::bit_cast<uint>(-6.0f)};
            expect(host == expected)
                << "selective relocation must copy certified fields and "
                   "leave every inactive destination field unchanged";
        }
    };

    "runtime_soa_large_mixed_frame_roundtrip"_test = [options] {
        constexpr uint active_count = 4096u;
        constexpr uint frame_capacity = 65536u;
        constexpr uint float2_count = 4u;
        constexpr uint struct_count = 2u;
        constexpr uint float_count = 40u;
        constexpr uint uint_count = 60u;
        constexpr uint bool_count = 35u;
        constexpr uint output_word_count =
            3u + 2u * float2_count + 7u * struct_count +
            float_count + uint_count + bool_count;

        CoroFrameDesc desc;
        desc.add_field("direction", Type::of<float3>());
        for (auto i = 0u; i < float2_count; i++) {
            desc.add_field(luisa::format("pair_{}", i), Type::of<float2>());
        }
        for (auto i = 0u; i < struct_count; i++) {
            desc.add_field(luisa::format("words_{}", i), Type::of<CoroSoASevenWords>());
        }
        for (auto i = 0u; i < float_count; i++) {
            desc.add_field(luisa::format("float_{}", i), Type::of<float>());
        }
        for (auto i = 0u; i < uint_count; i++) {
            desc.add_field(luisa::format("uint_{}", i), Type::of<uint>());
        }
        for (auto i = 0u; i < bool_count; i++) {
            desc.add_field(luisa::format("bool_{}", i), Type::of<bool>());
        }

        auto layout = CoroFrameStorageLayout::make_runtime_soa(
            desc, frame_capacity);
        expect(layout.frame_stride == 672u)
            << "the regression frame should retain the production-sized ABI";

        auto dc = luisa::test::coro_test::create_device(options);
        auto &device = dc.device;
        Stream stream = device.create_stream();
        auto frames = device.create_byte_buffer(layout.size_bytes);
        auto output =
            device.create_buffer<uint>(active_count * output_word_count);

        Kernel1D store = [&desc, layout, active_count](ByteBufferVar frames) noexcept {
            auto tid = dispatch_x();
            auto frame = CoroFrame::create(&desc);
            frame.coro_id = make_uint3(tid, tid + 1u, tid + 2u);
            frame.dispatch_size_x = active_count;
            frame.dispatch_size_y = 1u;
            frame.dispatch_size_z = 1u;
            frame.target_token = 1u;

            auto direction = frame.get<float3>(0u);
            direction = make_float3(
                cast<float>(tid) + 0.25f,
                cast<float>(tid) + 0.5f,
                cast<float>(tid) + 0.75f);
            auto field = 1u;
            for (auto i = 0u; i < float2_count; i++, field++) {
                auto pair = frame.get<float2>(field);
                auto base = cast<float>(tid * 17u + i * 3u);
                pair = make_float2(base + 0.125f, base + 0.625f);
            }
            for (auto i = 0u; i < struct_count; i++, field++) {
                auto words = frame.get<CoroSoASevenWords>(field);
                auto base = tid * 101u + i * 19u;
                words.a = base + 1u;
                words.b = base + 2u;
                words.c = base + 3u;
                words.d = base + 4u;
                words.e = base + 5u;
                words.f = base + 6u;
                words.g = base + 7u;
            }
            for (auto i = 0u; i < float_count; i++, field++) {
                auto value = frame.get<float>(field);
                value = cast<float>(tid * 13u + i * 5u) + 0.375f;
            }
            for (auto i = 0u; i < uint_count; i++, field++) {
                auto value = frame.get<uint>(field);
                value = tid * 29u + i * 11u + 7u;
            }
            for (auto i = 0u; i < bool_count; i++, field++) {
                auto value = frame.get<bool>(field);
                value = ((tid + i * 3u) & 1u) != 0u;
            }
            coro_frame_store(
                frames, tid, frame_capacity, frame, layout, true);
        };

        Kernel1D load = [&desc, layout, output_word_count](
                            ByteBufferVar frames, BufferUInt output) noexcept {
            auto tid = dispatch_x();
            auto frame = coro_frame_load(
                &desc, frames, tid, frame_capacity, layout, true);
            auto output_index = tid * output_word_count;
            auto emit = [&](auto value) noexcept {
                output.write(output_index, value);
                output_index += 1u;
            };
            auto direction = frame.get<float3>(0u);
            emit(direction.x.as<uint>());
            emit(direction.y.as<uint>());
            emit(direction.z.as<uint>());
            auto field = 1u;
            for (auto i = 0u; i < float2_count; i++, field++) {
                auto pair = frame.get<float2>(field);
                emit(pair.x.as<uint>());
                emit(pair.y.as<uint>());
            }
            for (auto i = 0u; i < struct_count; i++, field++) {
                auto words = frame.get<CoroSoASevenWords>(field);
                emit(words.a);
                emit(words.b);
                emit(words.c);
                emit(words.d);
                emit(words.e);
                emit(words.f);
                emit(words.g);
            }
            for (auto i = 0u; i < float_count; i++, field++) {
                emit(frame.get<float>(field).as<uint>());
            }
            for (auto i = 0u; i < uint_count; i++, field++) {
                emit(frame.get<uint>(field));
            }
            for (auto i = 0u; i < bool_count; i++, field++) {
                emit(select(0u, 1u, frame.get<bool>(field)));
            }
        };

        auto store_shader = device.compile(store);
        auto load_shader = device.compile(load);
        stream << store_shader(frames).dispatch(active_count)
               << load_shader(frames, output).dispatch(active_count);
        luisa::vector<uint> host(output.size());
        stream << output.copy_to(luisa::span{host}) << synchronize();

        auto expected_float_bits = [](float value) noexcept {
            return std::bit_cast<uint>(value);
        };
        auto correct = true;
        for (auto tid = 0u; tid < active_count && correct; tid++) {
            auto output_index = tid * output_word_count;
            auto check = [&](uint expected) noexcept {
                if (host[output_index] != expected) {
                    LUISA_WARNING(
                        "runtime SoA mixed-frame mismatch at instance {}, word {}: "
                        "got {}, expected {}",
                        tid, output_index - tid * output_word_count,
                        host[output_index], expected);
                    correct = false;
                }
                output_index++;
            };
            check(expected_float_bits(static_cast<float>(tid) + 0.25f));
            check(expected_float_bits(static_cast<float>(tid) + 0.5f));
            check(expected_float_bits(static_cast<float>(tid) + 0.75f));
            for (auto i = 0u; i < float2_count; i++) {
                auto base = static_cast<float>(tid * 17u + i * 3u);
                check(expected_float_bits(base + 0.125f));
                check(expected_float_bits(base + 0.625f));
            }
            for (auto i = 0u; i < struct_count; i++) {
                auto base = tid * 101u + i * 19u;
                for (auto lane = 1u; lane <= 7u; lane++) {
                    check(base + lane);
                }
            }
            for (auto i = 0u; i < float_count; i++) {
                check(expected_float_bits(
                    static_cast<float>(tid * 13u + i * 5u) + 0.375f));
            }
            for (auto i = 0u; i < uint_count; i++) {
                check(tid * 29u + i * 11u + 7u);
            }
            for (auto i = 0u; i < bool_count; i++) {
                check(((tid + i * 3u) & 1u) != 0u ? 1u : 0u);
            }
        }
        expect(correct)
            << "runtime-capacity SoA must preserve every mixed frame field";
    };

    "wavefront_soa_large_live_frame_matches_aos"_test = [options] {
        constexpr uint instance_count = 4096u;
        constexpr uint uint_value_count = 80u;
        constexpr uint float_value_count = 80u;
        constexpr uint output_word_count =
            uint_value_count + float_value_count;

        auto coro = Coroutine<void(
            Buffer<uint>, Buffer<float>, Buffer<uint>, Buffer<uint>)>(
            [uint_value_count, float_value_count, output_word_count](
                BufferUInt uint_input, BufferFloat float_input,
                BufferUInt side_effect, BufferUInt output) noexcept {
                auto tid = dispatch_x();
                luisa::vector<UInt> uint_values;
                uint_values.reserve(uint_value_count);
                for (auto i = 0u; i < uint_value_count; i++) {
                    uint_values.emplace_back(def(uint_input.read(
                        tid * uint_value_count + i)));
                }
                luisa::vector<Float> float_values;
                float_values.reserve(float_value_count);
                for (auto i = 0u; i < float_value_count; i++) {
                    float_values.emplace_back(def(float_input.read(
                        tid * float_value_count + i)));
                }
                $suspend("loaded_state");
                side_effect.atomic(tid).fetch_add(
                    uint_values.front() ^
                    float_values.front().as<uint>());
                $suspend("after_side_effect");
                auto output_index = tid * output_word_count;
                for (auto &&value : uint_values) {
                    output.write(output_index, value);
                    output_index += 1u;
                }
                for (auto &&value : float_values) {
                    output.write(output_index, value.as<uint>());
                    output_index += 1u;
                }
            });

        auto dc = luisa::test::coro_test::create_device(options);
        auto &device = dc.device;
        Stream stream = device.create_stream();
        auto uint_input = device.create_buffer<uint>(
            instance_count * uint_value_count);
        auto float_input = device.create_buffer<float>(
            instance_count * float_value_count);
        auto side_effect = device.create_buffer<uint>(instance_count);
        auto soa_output = device.create_buffer<uint>(
            instance_count * output_word_count);
        auto aos_output = device.create_buffer<uint>(
            instance_count * output_word_count);

        luisa::vector<uint> host_uint(uint_input.size());
        for (auto i = 0u; i < host_uint.size(); i++) {
            host_uint[i] = i * 747796405u + 2891336453u;
        }
        luisa::vector<float> host_float(float_input.size());
        for (auto i = 0u; i < host_float.size(); i++) {
            host_float[i] =
                static_cast<float>(i % 16384u) * 0.0625f + 0.03125f;
        }
        luisa::vector<uint> zero_side_effect(instance_count);
        stream << uint_input.copy_from(luisa::span{host_uint})
               << float_input.copy_from(luisa::span{host_float})
               << side_effect.copy_from(luisa::span{zero_side_effect})
               << synchronize();

        LUISA_INFO(
            "Large wavefront frame regression: fields={} bytes={}",
            coro.frame().frame_field_count(),
            coro.frame().frame_type()->size());
        expect(coro.frame().frame_type()->size() >= 640u)
            << "the regression must actually carry a production-sized frame";

        auto run = [&](bool soa, Buffer<uint> &output) noexcept {
            WavefrontCoroSchedulerConfig config{
                .thread_count = instance_count,
                .global_memory_soa = soa,
                .gather_by_sorting = true,
                .frame_buffer_compaction = true};
            WavefrontCoroScheduler<
                Buffer<uint>, Buffer<float>, Buffer<uint>, Buffer<uint>>
                scheduler{device, coro, config};
            scheduler(uint_input, float_input, side_effect, output)
                .dispatch(instance_count)(stream);
            stream << synchronize();
        };
        run(true, soa_output);
        stream << side_effect.copy_from(luisa::span{zero_side_effect})
               << synchronize();
        run(false, aos_output);

        luisa::vector<uint> host_soa(soa_output.size());
        luisa::vector<uint> host_aos(aos_output.size());
        stream << soa_output.copy_to(luisa::span{host_soa})
               << aos_output.copy_to(luisa::span{host_aos})
               << synchronize();
        auto correct = true;
        for (auto i = 0u; i < host_soa.size(); i++) {
            auto lane = i % output_word_count;
            auto instance = i / output_word_count;
            auto expected = lane < uint_value_count ?
                                host_uint[instance * uint_value_count + lane] :
                                std::bit_cast<uint>(host_float[
                                    instance * float_value_count +
                                    lane - uint_value_count]);
            if (host_soa[i] != expected || host_aos[i] != expected) {
                LUISA_WARNING(
                    "large wavefront frame mismatch at instance {}, word {}: "
                    "SoA={}, AoS={}, expected={}",
                    instance, lane, host_soa[i], host_aos[i], expected);
                correct = false;
                break;
            }
        }
        expect(correct)
            << "wavefront SoA and AoS must preserve all live fields equally";
    };

    "wavefront_shader_structure_does_not_hash_pool_capacity"_test = [options] {
        auto dc = luisa::test::coro_test::create_device(options);
        auto &device = dc.device;
        auto coro = Coroutine<void(Buffer<uint>)>([](BufferUInt output) {
            auto value = def(dispatch_x() * 11u + 5u);
            $suspend("live_value");
            output.write(dispatch_x(), value);
        });
        auto make_config = [](uint capacity) noexcept {
            return WavefrontCoroSchedulerConfig{
                .thread_count = capacity,
                .global_memory_soa = true,
                .gather_by_sorting = false,
                .frame_buffer_compaction = true};
        };
        WavefrontCoroScheduler<Buffer<uint>> small{
            device, coro, make_config(17u)};
        WavefrontCoroScheduler<Buffer<uint>> large{
            device, coro, make_config(257u)};
        auto small_hashes = small.shader_structure_hashes();
        auto large_hashes = large.shader_structure_hashes();
        auto small_infos = small.shader_infos();
        auto large_infos = large.shader_infos();
        expect(!small_hashes.empty());
        expect(small_hashes.size() == large_hashes.size());
        expect(small_infos.size() == small_hashes.size());
        expect(large_infos.size() == large_hashes.size());
        expect(std::equal(small_hashes.begin(), small_hashes.end(),
                          large_hashes.begin(), large_hashes.end()))
            << "frame-pool capacity is an allocation/runtime parameter and "
               "must not invalidate scheduler shader caches";
        auto semantic_map_matches = true;
        auto has_entry = false;
        auto has_live_value = false;
        for (auto i = 0u; i < small_infos.size(); ++i) {
            semantic_map_matches &=
                small_infos[i].structural_hash == small_hashes[i] &&
                large_infos[i].structural_hash == large_hashes[i] &&
                small_infos[i].stage == large_infos[i].stage;
            has_entry |=
                small_infos[i].stage == "wavefront_generate/<entry>";
            has_live_value |=
                small_infos[i].stage ==
                "wavefront_resume_1/live_value";
        }
        expect(semantic_map_matches)
            << "semantic profiler labels must be a one-to-one, capacity-"
               "independent attribution of scheduler structural hashes";
        expect(has_entry);
        expect(has_live_value)
            << "continuation profiler labels must use the CoroGraph node "
               "name rather than compilation order alone";
    };

    "wavefront_compaction_policy_is_structural"_test = [options] {
        auto dc = luisa::test::coro_test::create_device(options);
        auto &device = dc.device;
        auto coro = Coroutine<void(Buffer<uint>)>([](BufferUInt output) {
            auto value = def(dispatch_x() * 13u + 7u);
            $suspend("live_value");
            output.write(dispatch_x(), value);
        });
        auto make_scheduler = [&](bool compact) {
            return WavefrontCoroScheduler<Buffer<uint>>{
                device, coro,
                WavefrontCoroSchedulerConfig{
                    .thread_count = 257u,
                    .global_memory_soa = true,
                    .gather_by_sorting = false,
                    .frame_buffer_compaction = compact,
                    .execution_block_size = 32u}};
        };
        auto compact = make_scheduler(true);
        auto sparse = make_scheduler(false);
        auto compact_hashes = compact.shader_structure_hashes();
        auto sparse_hashes = sparse.shader_structure_hashes();
        expect(compact_hashes.size() == sparse_hashes.size());
        expect(!std::equal(compact_hashes.begin(), compact_hashes.end(),
                           sparse_hashes.begin(), sparse_hashes.end()))
            << "frame compaction changes generate-kernel control flow and "
               "must remain a host structural specialization";
    };

    "wavefront_execution_block_size_is_structural"_test = [options] {
        auto dc = luisa::test::coro_test::create_device(options);
        auto &device = dc.device;
        auto coro = Coroutine<void(Buffer<uint>)>([](BufferUInt output) {
            auto value = def(dispatch_x() * 7u + 3u);
            $suspend("live_value");
            output.write(dispatch_x(), value);
        });
        auto make_config = [](uint block_size) noexcept {
            return WavefrontCoroSchedulerConfig{
                .thread_count = 257u,
                .global_memory_soa = true,
                .gather_by_sorting = false,
                .frame_buffer_compaction = true,
                .execution_block_size = block_size};
        };
        WavefrontCoroScheduler<Buffer<uint>> narrow{
            device, coro, make_config(32u)};
        WavefrontCoroScheduler<Buffer<uint>> wide{
            device, coro, make_config(256u)};
        auto narrow_hashes = narrow.shader_structure_hashes();
        auto wide_hashes = wide.shader_structure_hashes();
        expect(narrow_hashes.size() == wide_hashes.size());
        expect(!std::equal(narrow_hashes.begin(), narrow_hashes.end(),
                           wide_hashes.begin(), wide_hashes.end()))
            << "execution block size changes generate/resume kernel structure "
               "and must invalidate their shader cache entries";

        constexpr uint n = 257u;
        auto output = device.create_buffer<uint>(n);
        auto stream = device.create_stream();
        narrow(output).dispatch(n)(stream);
        luisa::vector<uint> host(n);
        stream << output.copy_to(luisa::span{host}) << synchronize();
        auto correct = true;
        for (auto i = 0u; i < n; ++i) {
            correct &= host[i] == i * 7u + 3u;
        }
        expect(correct)
            << "a non-default execution block size must preserve the exact "
               "logical-frame mapping, including a partial final block";
    };

    "soa_layout_constructs_and_has_correct_config"_test = [options] {
        auto dc = luisa::test::coro_test::create_device(options);
        auto &device = dc.device;

        auto coro = Coroutine<void()>([] {
            $suspend("s1");
        });

        WavefrontCoroScheduler<> aos_scheduler{device, coro,
            WavefrontCoroSchedulerConfig{.global_memory_soa = false}};
        expect(aos_scheduler.config().global_memory_soa == false);

        WavefrontCoroScheduler<> soa_scheduler{device, coro,
            WavefrontCoroSchedulerConfig{.global_memory_soa = true}};
        expect(soa_scheduler.config().global_memory_soa == true);
    };

    "soa_layout_compiles_and_runs"_test = [options] {
        constexpr uint N = 64u;

        auto dc = luisa::test::coro_test::create_device(options);
        auto &device = dc.device;
        Stream stream = device.create_stream();

        auto coro = Coroutine<void()>([] {
            $suspend("s1");
        });

        LUISA_INFO("Coroutine created, sub_count={}", coro.subroutine_count());
        expect(coro.subroutine_count() >= 2u);

        WavefrontCoroScheduler<> scheduler{device, coro,
            WavefrontCoroSchedulerConfig{.global_memory_soa = true}};
        LUISA_INFO("SoA Wavefront scheduler created, dispatching {} instances", N);

        scheduler().dispatch(N)(stream);
        stream << synchronize();
        LUISA_INFO("SoA dispatch complete");
        expect(scheduler.config().global_memory_soa == true);
        expect(scheduler.config().thread_count >= N);
    };

    "soa_layout_1suspend_with_buffer"_test = [options] {
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

        WavefrontCoroScheduler<Buffer<uint>> scheduler{device, coro,
            WavefrontCoroSchedulerConfig{.global_memory_soa = true}};
        LUISA_INFO("SoA Wavefront scheduler created, dispatching {} instances", N);

        scheduler(output).dispatch(N)(stream);
        luisa::vector<uint> host(N);
        stream << output.copy_to(luisa::span{host}) << synchronize();
        LUISA_INFO("SoA dispatch complete");
        auto ok = true;
        for (auto i = 0u; i < N; i++) {
            if (host[i] != i + 42u) {
                LUISA_WARNING("soa_layout_1suspend_with_buffer mismatch at {}: got {}, expected {}",
                              i, host[i], i + 42u);
                ok = false;
                break;
            }
        }
        expect(ok) << "all wavefront SoA coroutine instances should write expected values";
    };

    "soa_layout_3suspend_smoke"_test = [options] {
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

        WavefrontCoroScheduler<int> scheduler{device, coro,
            WavefrontCoroSchedulerConfig{.global_memory_soa = true}};
        LUISA_INFO("SoA Wavefront scheduler created, dispatching {} instances", N);

        scheduler(42).dispatch(N)(stream);
        stream << synchronize();
        LUISA_INFO("SoA dispatch complete");
        expect(scheduler.config().global_memory_soa == true);
        expect(scheduler.config().thread_count >= N);
    };
}

int main(int argc, char *argv[]) {
    auto options = luisa::test::coro_test::parse_options(argc, argv);
    reg_coro_soa_layout(options);
    return luisa::test::coro_test::run_tests(argc, argv);
}
