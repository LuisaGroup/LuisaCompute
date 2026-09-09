#include "ut/ut.hpp"
#include "coro_test_utils.h"

#include <luisa/luisa-compute.h>
#include <luisa/coro/schedulers/graph_wavefront.h>
#include <luisa/coro/schedulers/state_machine.h>
#include <luisa/coro/schedulers/wavefront.h>

using namespace luisa;
using namespace luisa::compute;
using namespace luisa::compute::coro;
using namespace boost::ut;
using namespace boost::ut::literals;

int main(int argc, char *argv[]) {
    auto options = luisa::test::coro_test::parse_options(argc, argv);
    "packed_word_defined_entry_and_dormant_lane_replay"_test = [options] {
        constexpr uint count = 4096u;
        auto dc = luisa::test::coro_test::create_device(options);
        auto &device = dc.device;
        auto stream = device.create_stream();
        auto input = device.create_buffer<uint>(count);
        auto output = device.create_buffer<uint>(count);
        auto coroutine = Coroutine<void(Buffer<uint>, Buffer<uint>)>{
            [](BufferUInt input, BufferUInt output) noexcept {
                auto tid = dispatch_x();
                auto first = (input.read(tid) & 1u) != 0u;
                $suspend("first");
                auto second = (input.read(tid) & 2u) != 0u;
                // first is dormant while this continuation writes second;
                // both bits must coexist in the same physical word.
                $suspend("second", coro_frame_export(
                                       "coro_hint", (input.read(tid) >> 2u) % 3u));
                $for (iteration, (input.read(tid) >> 4u) % 7u) {
                    $suspend("wait");
                };
                output.write(dispatch_x(), ite(first, 1u, 0u) |
                                               ite(second, 2u, 0u));
            }};
        luisa::vector<uint> source(count);
        luisa::vector<uint> poison(count, 0xdeadbeefu);
        luisa::vector<uint> actual(count);
        auto check_replay = [&](auto &scheduler, luisa::string_view label) {
            auto passed = true;
            for (auto replay = 0u; replay < 4u; ++replay) {
                for (auto i = 0u; i < count; ++i) {
                    // Flip every low-bit truth-table case on each replay.
                    source[i] = (i * 37u) ^ replay;
                }
                stream << input.copy_from(luisa::span{source})
                       << output.copy_from(luisa::span{poison});
                stream << scheduler(input, output).dispatch(count);
                stream << output.copy_to(luisa::span{actual})
                       << synchronize();
                for (auto i = 0u; i < count; ++i) {
                    if (actual[i] != (source[i] & 3u)) {
                        LUISA_WARNING("{} replay={} lane={} got={} expected={}",
                                      label, replay, i, actual[i], source[i] & 3u);
                        passed = false;
                        break;
                    }
                }
            }
            expect(passed) << label;
        };

        StateMachineCoroScheduler<Buffer<uint>, Buffer<uint>> state_machine{
            device, coroutine};
        check_replay(state_machine, "state-machine undefined initial frame");
        WavefrontCoroScheduler<Buffer<uint>, Buffer<uint>> wavefront{
            device, coroutine};
        check_replay(wavefront, "wavefront packed partial update");
        for (auto soa : {false, true}) {
            for (auto sort : {false, true}) {
                for (auto tail : {0u, 96u}) {
                    GraphWavefrontCoroSchedulerConfig config;
                    config.thread_count = 128u;
                    config.worker_count = 64u;
                    config.execution_block_size = 32u;
                    config.global_memory_soa = soa;
                    config.selective_scheduling = true;
                    config.counter_readback_batch_size = 1u;
                    config.counter_readback_pipeline_depth = 1u;
                    config.refill_continuations = {"first"};
                    config.tail_megakernel_threshold = tail;
                    if (sort) {
                        config.hint_fields = {"second"};
                        config.hint_range = 3u;
                    }
                    config.shader_option.enable_cache = false;
                    GraphWavefrontCoroScheduler<Buffer<uint>, Buffer<uint>> graph{
                        device, coroutine, config};
                    auto label = luisa::format("graph soa={} sorting={} tail={}", soa, sort, tail);
                    check_replay(graph, label);
                    if (tail != 0u) {
                        expect(graph.last_dispatch_stats().tail_dispatch_count != 0u)
                            << "the regression must execute the tail";
                    }
                }
            }
        }
    };
    return luisa::test::coro_test::run_tests(argc, argv);
}
