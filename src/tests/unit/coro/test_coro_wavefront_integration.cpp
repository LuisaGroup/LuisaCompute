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

using namespace luisa;
using namespace luisa::compute;
using namespace luisa::compute::coro;
using namespace boost::ut;
using namespace boost::ut::literals;

namespace {

constexpr uint kTestInstances = 64u;

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
            device, coro, WavefrontCoroSchedulerConfig{.global_memory_soa = false}};
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
        Stream stream = device.create_stream();

        auto coro = Coroutine<void(Buffer<int>)>([](BufferInt buf) {
            auto tid = dispatch_x();
            $suspend("1");
            $suspend("2");
            buf.write(tid, tid.cast<int>() + 42);
        });

        LUISA_INFO("Wavefront AoS (comp): sub_count={}, node_count={}",
                   coro.subroutine_count(), coro.graph().node_count());
        expect(coro.subroutine_count() > 0u);
        expect(coro.graph().node_count() > 0u);

        auto output = device.create_buffer<int>(kTestInstances);
        WavefrontCoroScheduler<Buffer<int>> scheduler{
            device, coro, WavefrontCoroSchedulerConfig{.global_memory_soa = false}};
        scheduler(output).dispatch(kTestInstances)(stream);
        luisa::vector<int> host(kTestInstances);
        stream << output.copy_to(luisa::span{host}) << synchronize();
        LUISA_INFO("Wavefront AoS (comp): dispatch complete — PASSED");
        expect(expect_sequence(host, 42, "wf_aos_with_compaction"));

        // Additionally verify the standalone compaction utility works
        verify_compaction_utility();
        LUISA_INFO("Wavefront AoS (comp): compaction utility PASSED");
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
            device, coro, WavefrontCoroSchedulerConfig{.global_memory_soa = true}};
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
        Stream stream = device.create_stream();

        auto coro = Coroutine<void(Buffer<int>)>([](BufferInt buf) {
            auto tid = dispatch_x();
            $suspend("1");
            $suspend("2");
            buf.write(tid, tid.cast<int>() + 42);
        });

        LUISA_INFO("Wavefront SoA (comp): sub_count={}, node_count={}",
                   coro.subroutine_count(), coro.graph().node_count());
        expect(coro.subroutine_count() > 0u);
        expect(coro.graph().node_count() > 0u);

        auto output = device.create_buffer<int>(kTestInstances);
        WavefrontCoroScheduler<Buffer<int>> scheduler{
            device, coro, WavefrontCoroSchedulerConfig{.global_memory_soa = true}};
        scheduler(output).dispatch(kTestInstances)(stream);
        luisa::vector<int> host(kTestInstances);
        stream << output.copy_to(luisa::span{host}) << synchronize();
        LUISA_INFO("Wavefront SoA (comp): dispatch complete — PASSED");
        expect(expect_sequence(host, 42, "wf_soa_with_compaction"));

        // Additionally verify the standalone compaction utility works
        verify_compaction_utility();
        LUISA_INFO("Wavefront SoA (comp): compaction utility PASSED");
    };
}

int main(int argc, char *argv[]) {
    auto options = luisa::test::coro_test::parse_options(argc, argv);
    reg_coro_wavefront_integration(options);
    return luisa::test::coro_test::run_tests(argc, argv);
}
