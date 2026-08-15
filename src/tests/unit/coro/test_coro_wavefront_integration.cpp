// Test for end-to-end coroutine wavefront scheduling and device execution.

#include "ut/ut.hpp"
#include "coro_test_utils.h"

#include <luisa/core/logging.h>
#include <luisa/dsl/coro_func.h>
#include <luisa/dsl/sugar.h>
#include <luisa/coro/schedulers/wavefront.h>
#include <luisa/coro/schedulers/state_machine.h>
#include <luisa/coro/schedulers/compaction.h>
#include <luisa/runtime/buffer.h>
#include <luisa/runtime/context.h>
#include <luisa/runtime/device.h>
#include <luisa/runtime/stream.h>

#include <algorithm>
#include <iterator>
#include <limits>

using namespace luisa;
using namespace luisa::compute;
using namespace luisa::compute::coro;
using namespace boost::ut;
using namespace boost::ut::literals;

namespace {

constexpr uint kTestInstances = 64u;
constexpr uint kCompactionCapacity = 32u;
constexpr uint kCompactionInstances = 56u;

[[nodiscard]] auto make_tagged_union_compaction_coroutine() {
    return Coroutine<void(Buffer<uint>, Buffer<uint>)>([](
                                                           BufferUInt output,
                                                           BufferUInt execution_count) {
        auto tid = dispatch_x();
        auto base = def(tid * 13u + 7u);
        auto value = def(base);
        auto route = tid % 3u;
        $if (route == 1u) {
            // The A and B payloads deliberately have different types. They
            // therefore occupy distinct physical frame fields even though
            // the control-flow alternatives are mutually exclusive.
            auto a0 = make_uint4(base + 1u, base + 2u, base + 3u, base + 4u);
            auto a1 = make_uint4(base + 5u, base + 6u, base + 7u, base + 8u);
            auto a2 = make_uint4(base + 9u, base + 10u, base + 11u, base + 12u);
            auto a3 = make_uint4(base + 13u, base + 14u, base + 15u, base + 16u);
            auto a4 = make_uint4(base + 17u, base + 18u, base + 19u, base + 20u);
            auto a5 = make_uint4(base + 21u, base + 22u, base + 23u, base + 24u);
            auto a6 = make_uint4(base + 25u, base + 26u, base + 27u, base + 28u);
            auto a7 = make_uint4(base + 29u, base + 30u, base + 31u, base + 32u);
            $suspend(
                "relocate_a",
                coro_frame_export("relocate_a_0", a0),
                coro_frame_export("relocate_a_1", a1),
                coro_frame_export("relocate_a_2", a2),
                coro_frame_export("relocate_a_3", a3),
                coro_frame_export("relocate_a_4", a4),
                coro_frame_export("relocate_a_5", a5),
                coro_frame_export("relocate_a_6", a6),
                coro_frame_export("relocate_a_7", a7));
            value = a0.x + a0.y + a0.z + a0.w +
                    a1.x + a1.y + a1.z + a1.w +
                    a2.x + a2.y + a2.z + a2.w +
                    a3.x + a3.y + a3.z + a3.w +
                    a4.x + a4.y + a4.z + a4.w +
                    a5.x + a5.y + a5.z + a5.w +
                    a6.x + a6.y + a6.z + a6.w +
                    a7.x + a7.y + a7.z + a7.w;
        }
        $elif (route == 2u) {
            auto fbase = cast<float>(base);
            auto b0 = make_float4(fbase + 33.0f, fbase + 34.0f, fbase + 35.0f, fbase + 36.0f);
            auto b1 = make_float4(fbase + 37.0f, fbase + 38.0f, fbase + 39.0f, fbase + 40.0f);
            auto b2 = make_float4(fbase + 41.0f, fbase + 42.0f, fbase + 43.0f, fbase + 44.0f);
            auto b3 = make_float4(fbase + 45.0f, fbase + 46.0f, fbase + 47.0f, fbase + 48.0f);
            auto b4 = make_float4(fbase + 49.0f, fbase + 50.0f, fbase + 51.0f, fbase + 52.0f);
            auto b5 = make_float4(fbase + 53.0f, fbase + 54.0f, fbase + 55.0f, fbase + 56.0f);
            auto b6 = make_float4(fbase + 57.0f, fbase + 58.0f, fbase + 59.0f, fbase + 60.0f);
            auto b7 = make_float4(fbase + 61.0f, fbase + 62.0f, fbase + 63.0f, fbase + 64.0f);
            $suspend(
                "relocate_b",
                coro_frame_export("relocate_b_0", b0),
                coro_frame_export("relocate_b_1", b1),
                coro_frame_export("relocate_b_2", b2),
                coro_frame_export("relocate_b_3", b3),
                coro_frame_export("relocate_b_4", b4),
                coro_frame_export("relocate_b_5", b5),
                coro_frame_export("relocate_b_6", b6),
                coro_frame_export("relocate_b_7", b7));
            value = cast<uint>(
                b0.x + b0.y + b0.z + b0.w +
                b1.x + b1.y + b1.z + b1.w +
                b2.x + b2.y + b2.z + b2.w +
                b3.x + b3.y + b3.z + b3.w +
                b4.x + b4.y + b4.z + b4.w +
                b5.x + b5.y + b5.z + b5.w +
                b6.x + b6.y + b6.z + b6.w +
                b7.x + b7.y + b7.z + b7.w);
        };
        execution_count.atomic(tid).fetch_add(1u);
        output.write(tid, value);
    });
}

void verify_relocation_partition(
    const Coroutine<void(Buffer<uint>, Buffer<uint>)> &coro) {
    auto exact = coro_frame_collect_relocation_fields(
        coro.graph(), coro.frame().frame_field_count());
    auto partition = coro_frame_partition_relocation_fields(
        coro.graph(), coro.frame().frame_field_count());
    expect(partition.residual_fields.size() == exact.size());
    for (auto token = 1u; token < exact.size(); ++token) {
        auto reconstructed = partition.common_fields;
        reconstructed.insert(
            reconstructed.end(),
            partition.residual_fields[token].begin(),
            partition.residual_fields[token].end());
        std::sort(reconstructed.begin(), reconstructed.end());
        reconstructed.erase(
            std::unique(reconstructed.begin(), reconstructed.end()),
            reconstructed.end());
        expect(reconstructed == exact[token])
            << "C union D[t] must reconstruct the exact relocation proof R[t]";
        auto intersection = luisa::vector<size_t>{};
        std::set_intersection(
            partition.common_fields.begin(), partition.common_fields.end(),
            partition.residual_fields[token].begin(),
            partition.residual_fields[token].end(),
            std::back_inserter(intersection));
        expect(intersection.empty())
            << "C and D[t] must form a disjoint partition of R[t]";
    }
    auto *a = coro.graph().node_by_name("relocate_a");
    auto *b = coro.graph().node_by_name("relocate_b");
    expect(a != nullptr);
    expect(b != nullptr);
    if (a != nullptr && b != nullptr) {
        expect(!partition.residual_fields[a->index].empty());
        expect(!partition.residual_fields[b->index].empty());
        expect(partition.residual_fields[a->index] !=
               partition.residual_fields[b->index])
            << "the regression requires two genuinely different tagged-union arms";
    }
}

bool expect_sequence(luisa::span<const int> host, int base, luisa::string_view label) noexcept {
    for (auto i = 0u; i < host.size(); i++) {
        auto expected = static_cast<int>(i) + base;
        if (host[i] != expected) {
            LUISA_WARNING("{} mismatch at {}: got {}, expected {}",
                          label, i, host[i], expected);
            return false;
        }
    }
    return true;
}

/// Verify that compact_frame_buffer works correctly on mock frame data.
/// This exercises the standalone compaction utility alongside the scheduler.
void verify_compaction_utility() {
    // Mock: 8 instances, stride=3 uints, 3 alive
    constexpr size_t capacity = 8u;
    constexpr size_t stride = 3u;

    luisa::vector<uint32_t> buf(capacity * stride);
    for (size_t i = 0u; i < capacity; ++i) {
        auto base = i * stride;
        for (size_t j = 0u; j < stride; ++j) {
            buf[base + j] = static_cast<uint32_t>(i + 1u);
        }
    }

    // Instances 1, 3, 5 alive (odd indices except 7)
    luisa::vector<bool> alive = {false, true, false, true, false, true, false, false};

    CompactionResult result;
    compact_frame_buffer(buf, alive, stride, result);

    expect(result.compacted);
    expect(result.alive_count_before == 3u);
    expect(result.alive_count_after == 3u);
    expect(result.load_factor_before < 0.5_d);

    // Verify the 3 alive instances are now at positions 0,1,2
    expect(buf[0u * stride] == 2u) << "instance 1 should be at pos 0";
    expect(buf[1u * stride] == 4u) << "instance 3 should be at pos 1";
    expect(buf[2u * stride] == 6u) << "instance 5 should be at pos 2";
}

void verify_scheduler_compaction(Device &device, bool soa, luisa::string_view label) {
    Stream stream = device.create_stream();
    auto stage = device.create_buffer<int>(kCompactionInstances);
    auto output = device.create_buffer<int>(kCompactionInstances);
    auto execution_count = device.create_buffer<int>(kCompactionInstances);
    luisa::vector<int> initial(kCompactionInstances, std::numeric_limits<int>::min());
    luisa::vector<int> zero(kCompactionInstances, 0);
    stream << stage.copy_from(luisa::span{initial})
           << output.copy_from(luisa::span{initial})
           << execution_count.copy_from(luisa::span{zero});

    auto coro = Coroutine<void(Buffer<int>, Buffer<int>, Buffer<int>)>([](
                                                                          BufferInt stage,
                                                                          BufferInt output,
                                                                          BufferInt execution_count) {
        auto tid = dispatch_x();
        auto state = tid.cast<int>() * 17 + 5;
        $if ((tid & 3u) == 3u) {
            state = state * 3 + 7;
            $suspend("relocate");
        };
        execution_count.atomic(tid).fetch_add(1);
        stage.write(tid, state);
        output.write(tid, state + tid.cast<int>());
    });

    WavefrontCoroScheduler<Buffer<int>, Buffer<int>, Buffer<int>> scheduler{
        device, coro, WavefrontCoroSchedulerConfig{
                          .thread_count = kCompactionCapacity,
                          .global_memory_soa = soa,
                          .gather_by_sorting = false,
                          .frame_buffer_compaction = true,
                          .report_stats = true}};
    expect(scheduler.config().thread_count == kCompactionCapacity);
    expect(scheduler.config().frame_buffer_compaction);
    scheduler(stage, output, execution_count).dispatch(kCompactionInstances)(stream);

    luisa::vector<int> host_stage(kCompactionInstances);
    luisa::vector<int> host_output(kCompactionInstances);
    luisa::vector<int> host_execution_count(kCompactionInstances);
    stream << stage.copy_to(luisa::span{host_stage})
           << output.copy_to(luisa::span{host_output})
           << execution_count.copy_to(luisa::span{host_execution_count})
           << synchronize();

    auto stage_ok = true;
    auto output_ok = true;
    auto execution_count_ok = true;
    for (auto i = 0u; i < kCompactionInstances; i++) {
        auto survives = (i & 3u) == 3u;
        auto expected_stage = survives ? static_cast<int>(i) * 51 + 22 : static_cast<int>(i) * 17 + 5;
        auto expected_output = expected_stage + static_cast<int>(i);
        if (host_stage[i] != expected_stage) {
            LUISA_WARNING("{} stage mismatch at {}: got {}, expected {}",
                          label, i, host_stage[i], expected_stage);
            stage_ok = false;
        }
        if (host_output[i] != expected_output) {
            LUISA_WARNING("{} output mismatch at {}: got {}, expected {}",
                          label, i, host_output[i], expected_output);
            output_ok = false;
        }
        if (host_execution_count[i] != 1) {
            LUISA_WARNING("{} execution-count mismatch at {}: got {}, expected 1",
                          label, i, host_execution_count[i]);
            execution_count_ok = false;
        }
    }
    expect(stage_ok) << "the compaction-enabled scheduler must preserve suspended frame state and logical IDs";
    expect(output_ok) << "the scheduler must produce the expected output for every logical instance";
    expect(execution_count_ok) << "every logical instance must complete exactly once";
}

void verify_tagged_union_scheduler_compaction(
    Device &device, bool soa, luisa::string_view label) {
    auto coro = make_tagged_union_compaction_coroutine();
    verify_relocation_partition(coro);

    Stream stream = device.create_stream();
    auto output = device.create_buffer<uint>(kCompactionInstances);
    auto execution_count = device.create_buffer<uint>(kCompactionInstances);
    luisa::vector<uint> initial(
        kCompactionInstances, std::numeric_limits<uint>::max());
    luisa::vector<uint> zero(kCompactionInstances, 0u);
    stream << output.copy_from(luisa::span{initial})
           << execution_count.copy_from(luisa::span{zero});

    WavefrontCoroScheduler<Buffer<uint>, Buffer<uint>> scheduler{
        device, coro, WavefrontCoroSchedulerConfig{
                          .thread_count = kCompactionCapacity,
                          .global_memory_soa = soa,
                          .gather_by_sorting = false,
                          .frame_buffer_compaction = true,
                          .report_stats = true}};
    scheduler(output, execution_count)
        .dispatch(kCompactionInstances)(stream);

    luisa::vector<uint> host_output(kCompactionInstances);
    luisa::vector<uint> host_execution_count(kCompactionInstances);
    stream << output.copy_to(luisa::span{host_output})
           << execution_count.copy_to(luisa::span{host_execution_count})
           << synchronize();

    auto correct = true;
    for (auto i = 0u; i < kCompactionInstances; ++i) {
        auto base = i * 13u + 7u;
        auto route = i % 3u;
        auto expected = route == 0u ? base :
                        route == 1u ? 32u * base + 528u :
                                      32u * base + 1552u;
        if (host_output[i] != expected || host_execution_count[i] != 1u) {
            LUISA_WARNING(
                "{} mismatch at {}: value {} (expected {}), executions {}",
                label, i, host_output[i], expected,
                host_execution_count[i]);
            correct = false;
        }
    }
    expect(correct)
        << "compaction must preserve only the active tagged-union arm while "
           "refilling a capacity-limited frame pool";
}

}// namespace

void reg_coro_wavefront_integration(luisa::test::coro_test::Options options) {

    // =====================================================================
    // Test 1: StateMachine baseline
    // =====================================================================
    "sm_baseline"_test = [options] {
        auto dc = luisa::test::coro_test::create_device(options);
        auto &device = dc.device;
        Stream stream = device.create_stream();

        auto coro = Coroutine<void(Buffer<int>)>([](BufferInt buf) {
            auto tid = dispatch_x();
            $suspend("1");
            $suspend("2");
            buf.write(tid, tid.cast<int>() + 42);
        });

        LUISA_INFO("StateMachine: sub_count={}, node_count={}",
                   coro.subroutine_count(), coro.graph().node_count());
        expect(coro.subroutine_count() > 0u);
        expect(coro.graph().node_count() > 0u);

        auto output = device.create_buffer<int>(kTestInstances);
        StateMachineCoroScheduler<Buffer<int>> scheduler{device, coro};
        scheduler(output).dispatch(kTestInstances)(stream);
        luisa::vector<int> host(kTestInstances);
        stream << output.copy_to(luisa::span{host}) << synchronize();
        LUISA_INFO("StateMachine: dispatch complete — PASSED");
        expect(expect_sequence(host, 42, "sm_baseline"));
    };

    // =====================================================================
    // Test 2: Wavefront AoS, no compaction
    // =====================================================================
    "wf_aos_no_compaction"_test = [options] {
        auto dc = luisa::test::coro_test::create_device(options);
        auto &device = dc.device;
        Stream stream = device.create_stream();

        auto coro = Coroutine<void(Buffer<int>)>([](BufferInt buf) {
            auto tid = dispatch_x();
            $suspend("1");
            $suspend("2");
            buf.write(tid, tid.cast<int>() + 42);
        });

        LUISA_INFO("Wavefront AoS (no comp): sub_count={}, node_count={}",
                   coro.subroutine_count(), coro.graph().node_count());
        expect(coro.subroutine_count() > 0u);
        expect(coro.graph().node_count() > 0u);

        auto output = device.create_buffer<int>(kTestInstances);
        WavefrontCoroScheduler<Buffer<int>> scheduler{
            device, coro, WavefrontCoroSchedulerConfig{
                              .thread_count = kTestInstances,
                              .global_memory_soa = false,
                              .frame_buffer_compaction = false}};
        expect(scheduler.config().thread_count == kTestInstances);
        expect(!scheduler.config().frame_buffer_compaction);
        scheduler(output).dispatch(kTestInstances)(stream);
        luisa::vector<int> host(kTestInstances);
        stream << output.copy_to(luisa::span{host}) << synchronize();
        LUISA_INFO("Wavefront AoS (no comp): dispatch complete — PASSED");
        expect(expect_sequence(host, 42, "wf_aos_no_compaction"));
    };

    // =====================================================================
    // Test 3: Wavefront AoS, with compaction
    // =====================================================================
    "wf_aos_with_compaction"_test = [options] {
        auto dc = luisa::test::coro_test::create_device(options);
        auto &device = dc.device;
        verify_scheduler_compaction(device, false, "wf_aos_with_compaction");

        verify_compaction_utility();
    };

    // =====================================================================
    // Test 4: Wavefront SoA, no compaction
    // =====================================================================
    "wf_soa_no_compaction"_test = [options] {
        auto dc = luisa::test::coro_test::create_device(options);
        auto &device = dc.device;
        Stream stream = device.create_stream();

        auto coro = Coroutine<void(Buffer<int>)>([](BufferInt buf) {
            auto tid = dispatch_x();
            $suspend("1");
            $suspend("2");
            buf.write(tid, tid.cast<int>() + 42);
        });

        LUISA_INFO("Wavefront SoA (no comp): sub_count={}, node_count={}",
                   coro.subroutine_count(), coro.graph().node_count());
        expect(coro.subroutine_count() > 0u);
        expect(coro.graph().node_count() > 0u);

        auto output = device.create_buffer<int>(kTestInstances);
        WavefrontCoroScheduler<Buffer<int>> scheduler{
            device, coro, WavefrontCoroSchedulerConfig{
                              .thread_count = kTestInstances,
                              .global_memory_soa = true,
                              .frame_buffer_compaction = false}};
        expect(scheduler.config().thread_count == kTestInstances);
        expect(!scheduler.config().frame_buffer_compaction);
        scheduler(output).dispatch(kTestInstances)(stream);
        luisa::vector<int> host(kTestInstances);
        stream << output.copy_to(luisa::span{host}) << synchronize();
        LUISA_INFO("Wavefront SoA (no comp): dispatch complete — PASSED");
        expect(expect_sequence(host, 42, "wf_soa_no_compaction"));
    };

    // =====================================================================
    // Test 5: Wavefront SoA, with compaction
    // =====================================================================
    "wf_soa_with_compaction"_test = [options] {
        auto dc = luisa::test::coro_test::create_device(options);
        auto &device = dc.device;
        verify_scheduler_compaction(device, true, "wf_soa_with_compaction");
    };

    "wf_aos_tagged_union_compaction"_test = [options] {
        auto dc = luisa::test::coro_test::create_device(options);
        verify_tagged_union_scheduler_compaction(
            dc.device, false, "wf_aos_tagged_union_compaction");
    };

    "wf_soa_tagged_union_compaction"_test = [options] {
        auto dc = luisa::test::coro_test::create_device(options);
        verify_tagged_union_scheduler_compaction(
            dc.device, true, "wf_soa_tagged_union_compaction");
    };
}

int main(int argc, char *argv[]) {
    auto options = luisa::test::coro_test::parse_options(argc, argv);
    reg_coro_wavefront_integration(options);
    return luisa::test::coro_test::run_tests(argc, argv);
}
