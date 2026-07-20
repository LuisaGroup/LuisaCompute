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
}

int main(int argc, char *argv[]) {
    auto options = luisa::test::coro_test::parse_options(argc, argv);
    reg_coro_wavefront_integration(options);
    return luisa::test::coro_test::run_tests(argc, argv);
}
