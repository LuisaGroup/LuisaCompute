// Test for wavefront coroutine scheduling.
// This test covers bounded frame pools, AoS/SoA storage, compaction,
// token and hint sorting, and oversubscribed dispatches.

#include "ut/ut.hpp"
#include "coro_test_utils.h"

#include <luisa/core/logging.h>
#include <luisa/dsl/coro_func.h>
#include <luisa/dsl/sugar.h>
#include <luisa/coro/schedulers/graph_wavefront.h>
#include <luisa/coro/schedulers/wavefront.h>
#include <luisa/runtime/buffer.h>
#include <luisa/runtime/context.h>
#include <luisa/runtime/device.h>
#include <luisa/runtime/stream.h>

#include <algorithm>
#include <limits>

using namespace luisa;
using namespace luisa::compute;
using namespace luisa::compute::coro;
using namespace boost::ut;
using namespace boost::ut::literals;

namespace {

class TestWavefrontAuxiliaryWork final
    : public WavefrontCoroAuxiliaryWork<
          Buffer<uint>, Buffer<uint>, Buffer<uint>> {

private:
    Buffer<uint> _items;
    Buffer<uint> _count;
    Shader1D<Buffer<uint>, Buffer<uint>, Buffer<uint>, uint>
        _consumer;
    luisa::vector<WavefrontCoroAuxiliaryProducer> _producers;
    uint _capacity{};
    uint _host_count{};
    uint _zero{};

public:
    TestWavefrontAuxiliaryWork(
        Device &device, uint capacity,
        luisa::vector<WavefrontCoroAuxiliaryProducer> producers)
        : _items{device.create_buffer<uint>(capacity)},
          _count{device.create_buffer<uint>(1u)},
          _producers{std::move(producers)},
          _capacity{capacity} {
        auto *items = &_items;
        Kernel1D consume = [items](
                               BufferUInt,
                               BufferUInt auxiliary_visits,
                               BufferUInt,
                               UInt count) noexcept {
            auto item_buffer = Expr<Buffer<uint>>{*items};
            auto x = dispatch_x();
            $if(x < count) {
                auto logical_id = item_buffer.read(x);
                auxiliary_visits.atomic(logical_id).fetch_add(1u);
            };
        };
        _consumer = device.compile(consume);
    }

    [[nodiscard]] luisa::string_view name() const noexcept override {
        return "test_side_work";
    }
    [[nodiscard]] uint capacity() const noexcept override {
        return _capacity;
    }
    [[nodiscard]] luisa::span<const WavefrontCoroAuxiliaryProducer>
    producers() const noexcept override {
        return _producers;
    }
    void reset(Stream &stream) noexcept override {
        _host_count = 0u;
        stream << _count.copy_from(luisa::span{&_zero, 1u});
    }
    void enqueue_count_readback(Stream &stream) noexcept override {
        stream << _count.copy_to(luisa::span{&_host_count, 1u});
    }
    [[nodiscard]] uint host_count() const noexcept override {
        return _host_count;
    }
    void dispatch(Stream &stream,
                  BufferView<uint> main_visits,
                  BufferView<uint> auxiliary_visits,
                  BufferView<uint> overflow) noexcept override {
        auto count = _host_count;
        LUISA_ASSERT(count != 0u && count <= _capacity,
                     "Invalid test auxiliary dispatch count {}.", count);
        stream << _consumer(
                      main_visits, auxiliary_visits, overflow, count)
                      .dispatch(count)
               << _count.copy_from(luisa::span{&_zero, 1u});
        _host_count = 0u;
    }

    [[nodiscard]] auto &items() noexcept { return _items; }
    [[nodiscard]] auto &count() noexcept { return _count; }
};

}// namespace

void reg_coro_wavefront(luisa::test::coro_test::Options options) {

    "graph_wavefront_uses_coro_graph_consumers_and_batches_counter_readback"_test = [options] {
        constexpr uint N = 257u;
        constexpr uint capacity = 32u;

        auto dc = luisa::test::coro_test::create_device(options);
        auto &device = dc.device;
        Stream stream = device.create_stream();
        auto output = device.create_buffer<uint>(N);
        auto loop_visits = device.create_buffer<uint>(N);

        auto coroutine = Coroutine<void(Buffer<uint>, Buffer<uint>)>{
            [](BufferUInt values, BufferUInt visits) noexcept {
                auto tid = dispatch_x();
                auto value = tid * 3u + 5u;
                $suspend("first");
                value += 7u;
                auto iteration = def(0u);
                auto iteration_count = tid % 3u + 1u;
                $while(iteration < iteration_count) {
                    $suspend("loop");
                    visits.atomic(tid).fetch_add(1u);
                    value += iteration + 1u;
                    iteration += 1u;
                };
                $suspend("last");
                values.write(tid, value);
            }};

        struct ReadbackCase {
            uint batch_size;
            uint pipeline_depth;
            uint worker_count;
            bool soa;
        };
        luisa::vector<uint64_t> reference_shader_hashes[2];
        for (auto [batch_size, pipeline_depth, worker_count, soa] :
             {ReadbackCase{1u, 1u, 1u, false},
              ReadbackCase{4u, 2u, 5u, true},
              ReadbackCase{8u, 4u, capacity, true}}) {
            luisa::vector<uint> zeros(N);
            stream << output.copy_from(luisa::span{zeros})
                   << loop_visits.copy_from(luisa::span{zeros})
                   << synchronize();

            GraphWavefrontCoroScheduler<Buffer<uint>, Buffer<uint>> scheduler{
                device, coroutine,
                GraphWavefrontCoroSchedulerConfig{
                    .thread_count = capacity,
                    .global_memory_soa = soa,
                    .execution_block_size = 32u,
                    .worker_count = worker_count,
                    .counter_readback_batch_size = batch_size,
                    .counter_readback_pipeline_depth = pipeline_depth,
                    .tail_megakernel_threshold = 0u,
                    .report_stats = true}};
            scheduler(output, loop_visits).dispatch(N)(stream);

            luisa::vector<uint> host_output(N);
            luisa::vector<uint> host_visits(N);
            stream << output.copy_to(luisa::span{host_output})
                   << loop_visits.copy_to(luisa::span{host_visits})
                   << synchronize();

            auto exact = true;
            auto mismatch_count = 0u;
            for (auto tid = 0u; tid < N; ++tid) {
                auto iterations = tid % 3u + 1u;
                auto expected = tid * 3u + 12u;
                for (auto i = 0u; i < iterations; ++i) {
                    expected += i + 1u;
                }
                if (host_output[tid] != expected ||
                    host_visits[tid] != iterations) {
                    if (mismatch_count < 8u) {
                        LUISA_WARNING(
                            "Graph wavefront mismatch: batch={} tid={} "
                            "output={}/{} visits={}/{}.",
                            batch_size, tid, host_output[tid], expected,
                            host_visits[tid], iterations);
                    }
                    mismatch_count++;
                    exact = false;
                }
            }
            expect(exact)
                << "CoroGraph consumers must execute every logical instance "
                   "and self-loop iteration exactly once";
            expect(scheduler.node_count() == coroutine.graph().node_count());
            expect(scheduler.active_frame_capacity() == capacity);
            expect(scheduler.last_dispatch_stats().worker_count ==
                   worker_count);

            auto shader_hashes = scheduler.shader_structure_hashes();
            expect(!shader_hashes.empty());
            auto &reference = reference_shader_hashes[soa ? 1u : 0u];
            if (reference.empty()) {
                reference.assign(
                    shader_hashes.begin(), shader_hashes.end());
            } else {
                expect(shader_hashes.size() ==
                       reference.size());
                expect(std::equal(shader_hashes.begin(), shader_hashes.end(),
                                  reference.begin(), reference.end()))
                    << "worker count and readback policy are runtime scheduler "
                       "parameters and must not invalidate shader caches";
            }

            auto &&stats = scheduler.last_dispatch_stats();
            auto frame_field_count =
                static_cast<uint>(coroutine.frame().frame_field_count());
            expect(stats.input_field_count.size() ==
                   coroutine.graph().node_count());
            expect(stats.max_transition_output_field_count.size() ==
                   coroutine.graph().node_count());
            auto has_partial_input = false;
            auto has_partial_output = false;
            for (auto node = 1u; node < coroutine.graph().node_count(); ++node) {
                has_partial_input |=
                    stats.input_field_count[node] < frame_field_count;
                has_partial_output |=
                    stats.max_transition_output_field_count[node] <
                    frame_field_count;
            }
            expect(has_partial_input)
                << "continuations must load CoroGraph-certified inputs, not "
                   "the complete frame";
            expect(has_partial_output)
                << "continuations must store edge-certified outputs, not "
                   "the complete frame";
            expect(stats.generated_count == N);
            expect(stats.counter_snapshot_count == stats.sweep_count);
            expect(stats.counter_snapshot_count ==
                   stats.counter_readback_count * batch_size);
            if (batch_size > 1u) {
                expect(stats.counter_readback_count <
                       stats.counter_snapshot_count)
                    << "one contiguous readback must amortize multiple graph "
                       "sweeps";
            }
        }
    };

    "graph_wavefront_selective_actions_preserve_queues_and_self_edges"_test = [options] {
        constexpr uint N = 257u;
        constexpr uint capacity = 32u;

        auto dc = luisa::test::coro_test::create_device(options);
        auto &device = dc.device;
        auto stream = device.create_stream();
        auto output = device.create_buffer<uint>(N);
        auto visits = device.create_buffer<uint>(N);
        auto coroutine = Coroutine<void(Buffer<uint>, Buffer<uint>)>{
            [](BufferUInt values, BufferUInt loop_visits) noexcept {
                auto tid = dispatch_x();
                auto value = tid + 1u;
                $suspend("first");
                auto i = def(0u);
                $while (i < tid % 4u) {
                    $suspend("loop");
                    loop_visits.atomic(tid).fetch_add(1u);
                    value += i + 3u;
                    i += 1u;
                };
                $suspend("last");
                values.write(tid, value);
            }};
        luisa::vector<uint> zeros(N);
        stream << output.copy_from(luisa::span{zeros})
               << visits.copy_from(luisa::span{zeros}) << synchronize();

        GraphWavefrontCoroScheduler<Buffer<uint>, Buffer<uint>> scheduler{
            device, coroutine,
            GraphWavefrontCoroSchedulerConfig{
                .thread_count = capacity,
                .global_memory_soa = true,
                .execution_block_size = 32u,
                .worker_count = 5u,
                .selective_scheduling = true,
                .counter_readback_batch_size = 1u,
                .counter_readback_pipeline_depth = 1u,
                .tail_megakernel_threshold = 0u,
                .report_stats = true}};
        scheduler(output, visits).dispatch(N)(stream);

        luisa::vector<uint> host_output(N);
        luisa::vector<uint> host_visits(N);
        stream << output.copy_to(luisa::span{host_output})
               << visits.copy_to(luisa::span{host_visits}) << synchronize();
        for (auto tid = 0u; tid < N; ++tid) {
            auto expected = tid + 1u;
            for (auto i = 0u; i < tid % 4u; ++i) {
                expected += i + 3u;
            }
            expect(host_output[tid] == expected);
            expect(host_visits[tid] == tid % 4u);
        }
        expect(scheduler.last_dispatch_stats().generated_count == N);
        auto &&stats = scheduler.last_dispatch_stats();
        expect(stats.entry_dispatch_count != 0u);
        for (auto node = 1u; node < coroutine.graph().node_count(); ++node) {
            expect(stats.continuation_executed_count[node] <=
                   stats.queued_count_sum[node])
                << "selective execution cannot consume more frames than "
                   "were observed in that queue";
            expect(stats.continuation_executed_count[node] != 0u);
        }
    };

    "graph_wavefront_selective_hint_sort_preserves_queue_bijection"_test =
        [options] {
            constexpr uint N = 193u;
            constexpr uint capacity = 32u;
            constexpr uint hint_range = 64u;

            auto dc = luisa::test::coro_test::create_device(options);
            auto &device = dc.device;
            auto stream = device.create_stream();
            auto output = device.create_buffer<uint>(N);
            auto visits = device.create_buffer<uint>(N);
            auto coroutine = Coroutine<void(Buffer<uint>, Buffer<uint>)>{
                [](BufferUInt values, BufferUInt loop_visits) noexcept {
                    auto tid = dispatch_x();
                    auto value = tid * 17u + 5u;
                    auto iteration = def(0u);
                    auto iteration_count = tid % 5u + 1u;
                    $while (iteration < iteration_count) {
                        auto coro_hint =
                            (tid * 13u + iteration * 7u) & 63u;
                        $suspend(
                            "sort_me",
                            coro_frame_export("coro_hint", coro_hint));
                        loop_visits.atomic(tid).fetch_add(1u);
                        value = (value ^ (coro_hint + 1u)) + iteration * 3u;
                        iteration += 1u;
                    };
                    $suspend("finish");
                    values.write(tid, value);
                }};

            luisa::vector<uint> zeros(N);
            stream << output.copy_from(luisa::span{zeros})
                   << visits.copy_from(luisa::span{zeros}) << synchronize();
            GraphWavefrontCoroScheduler<Buffer<uint>, Buffer<uint>> scheduler{
                device, coroutine,
                GraphWavefrontCoroSchedulerConfig{
                    .thread_count = capacity,
                    .global_memory_soa = true,
                    .execution_block_size = 32u,
                    .worker_count = 5u,
                    .selective_scheduling = true,
                    .counter_readback_batch_size = 1u,
                    .counter_readback_pipeline_depth = 1u,
                    .tail_megakernel_threshold = 0u,
                    .report_stats = true,
                    .hint_range = hint_range,
                    .hint_fields = {"sort_me", "finish"}}};
            expect(scheduler.config().hint_fields.size() == 1u)
                << "small-range graph hint sorting is subgroup independent "
                   "and must reject a target without the exported hint";
            if (scheduler.config().hint_fields.size() == 1u) {
                expect(scheduler.config().hint_fields.front() == "sort_me");
            }
            scheduler(output, visits).dispatch(N)(stream);

            luisa::vector<uint> host_output(N);
            luisa::vector<uint> host_visits(N);
            stream << output.copy_to(luisa::span{host_output})
                   << visits.copy_to(luisa::span{host_visits}) << synchronize();
            auto exact = true;
            auto expected_resumes = uint64_t{0u};
            for (auto tid = 0u; tid < N; ++tid) {
                auto expected = tid * 17u + 5u;
                auto iteration_count = tid % 5u + 1u;
                expected_resumes += iteration_count;
                for (auto iteration = 0u; iteration < iteration_count;
                     ++iteration) {
                    auto hint = (tid * 13u + iteration * 7u) & 63u;
                    expected = (expected ^ (hint + 1u)) + iteration * 3u;
                }
                exact &= host_output[tid] == expected;
                exact &= host_visits[tid] == iteration_count;
            }
            expect(exact)
                << "sorting stable frame-slot indices must neither omit, "
                   "duplicate, nor cross-associate coroutine frames";

            auto *sort_node = coroutine.graph().node_by_name("sort_me");
            auto *finish_node = coroutine.graph().node_by_name("finish");
            expect(sort_node != nullptr && finish_node != nullptr);
            if (sort_node == nullptr || finish_node == nullptr) { return; }
            auto &&stats = scheduler.last_dispatch_stats();
            expect(stats.continuation_executed_count[sort_node->index] ==
                   expected_resumes)
                << "every loop self-edge must resume exactly once";
            expect(stats.continuation_hint_sort_count[sort_node->index] != 0u);
            expect(stats.continuation_hint_sort_count[sort_node->index] ==
                   stats.continuation_dispatch_count[sort_node->index])
                << "every selected hinted queue is sorted from its exact "
                   "host-observed cardinality";
            expect(stats.continuation_hint_sort_count[finish_node->index] ==
                   0u)
                << "unconfigured continuation queues must remain unsorted";
        };

    "graph_wavefront_tail_megakernel_finishes_residual_frames"_test =
        [options] {
            constexpr uint N = 66u;
            constexpr uint capacity = 16u;

            auto dc = luisa::test::coro_test::create_device(options);
            auto &device = dc.device;
            Stream stream = device.create_stream();
            auto output = device.create_buffer<uint>(N);

            auto coroutine = Coroutine<void(Buffer<uint>)>{
                [](BufferUInt values) noexcept {
                    auto tid = dispatch_x();
                    $if ((tid & 1u) == 0u) {
                        values.write(tid, 11u);
                        $return();
                    };
                    $suspend("odd_tail");
                    values.write(tid, 22u);
                }};

            luisa::vector<uint> zeros(N);
            stream << output.copy_from(luisa::span{zeros})
                   << synchronize();

            GraphWavefrontCoroScheduler<Buffer<uint>> scheduler{
                device, coroutine,
                GraphWavefrontCoroSchedulerConfig{
                    .thread_count = capacity,
                    .global_memory_soa = true,
                    .execution_block_size = 32u,
                    .counter_readback_batch_size = 1u,
                    .counter_readback_pipeline_depth = 1u,
                    .tail_megakernel_threshold = capacity,
                    .report_stats = true}};
            scheduler(output).dispatch(N)(stream);

            luisa::vector<uint> host_output(N);
            stream << output.copy_to(luisa::span{host_output})
                   << synchronize();
            for (auto tid = 0u; tid < N; ++tid) {
                expect(host_output[tid] == ((tid & 1u) == 0u ? 11u : 22u))
                    << "entry termination and tail continuation must each "
                       "execute exactly once";
            }
            auto &&stats = scheduler.last_dispatch_stats();
            expect(stats.generated_count == N);
            expect(stats.tail_dispatch_count == 1u)
                << "an exact, non-speculative snapshot must switch a small "
                   "residual set to the graph-derived state machine";
            expect(stats.tail_instance_count != 0u &&
                   stats.tail_instance_count <= capacity)
                << "the tail must contain exactly the latest bounded active "
                   "set, independent of entry batch boundaries";
        };

    "graph_wavefront_tail_flattens_multiple_token_queues_bijectively"_test =
        [options] {
            constexpr uint N = 32u;

            auto dc = luisa::test::coro_test::create_device(options);
            auto &device = dc.device;
            Stream stream = device.create_stream();
            auto output = device.create_buffer<uint>(N);

            auto coroutine = Coroutine<void(Buffer<uint>)>{
                [](BufferUInt values) noexcept {
                    auto tid = dispatch_x();
                    auto value = tid * 5u;
                    $if ((tid & 1u) == 0u) {
                        value += 3u;
                        $suspend("even_tail");
                        value += 7u;
                    }
                    $else {
                        value += 11u;
                        $suspend("odd_tail");
                        value += 13u;
                    };
                    values.write(tid, value);
                }};

            luisa::vector<uint> zeros(N);
            stream << output.copy_from(luisa::span{zeros})
                   << synchronize();
            GraphWavefrontCoroScheduler<Buffer<uint>> scheduler{
                device, coroutine,
                GraphWavefrontCoroSchedulerConfig{
                    .thread_count = N * 2u,
                    .global_memory_soa = true,
                    .execution_block_size = 32u,
                    .counter_readback_batch_size = 1u,
                    .counter_readback_pipeline_depth = 1u,
                    .tail_megakernel_threshold = N,
                    .report_stats = true}};
            scheduler(output).dispatch(N)(stream);

            luisa::vector<uint> host_output(N);
            stream << output.copy_to(luisa::span{host_output})
                   << synchronize();
            for (auto tid = 0u; tid < N; ++tid) {
                auto expected = tid * 5u +
                                ((tid & 1u) == 0u ? 10u : 24u);
                expect(host_output[tid] == expected)
                    << "flattening multiple token queues must neither omit "
                       "nor duplicate a frame";
            }
            auto &&stats = scheduler.last_dispatch_stats();
            expect(scheduler.active_frame_capacity() == N)
                << "the ownership pool is the active dispatch capacity, not "
                   "the larger physical queue storage capacity";
            expect(stats.tail_dispatch_count == 1u);
            expect(stats.tail_instance_count == N)
                << "the exact first snapshot contains both token queues";
        };

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

    "wavefront_reports_semantic_continuation_work"_test = [options] {
        constexpr uint N = 64u;

        auto dc = luisa::test::coro_test::create_device(options);
        auto &device = dc.device;
        Stream stream = device.create_stream();
        auto output = device.create_buffer<uint>(N);

        auto coro = Coroutine<void(Buffer<uint>)>([](BufferUInt buf) {
            auto tid = dispatch_x();
            $if ((tid & 1u) == 0u) {
                $suspend("even");
            }
            $else {
                $suspend("odd");
            };
            $suspend("join");
            buf.write(tid, tid + 1u);
        });

        WavefrontCoroScheduler<Buffer<uint>> scheduler{
            device, coro,
            WavefrontCoroSchedulerConfig{
                .thread_count = N,
                .gather_by_sorting = false,
                .frame_buffer_compaction = true,
                .report_stats = true}};
        scheduler(output).dispatch(N)(stream);
        stream << synchronize();

        auto &&stats = scheduler.last_dispatch_stats();
        expect(stats.collected);
        expect(stats.generated_count == N);
        expect(stats.resumed_count == 2u * N);
        expect(stats.continuations.size() == coro.subroutine_count());

        auto expect_node = [&](luisa::string_view name,
                               uint64_t executed,
                               uint peak) noexcept {
            auto *node = coro.graph().node_by_name(name);
            expect(node != nullptr);
            if (node == nullptr) { return; }
            auto &&node_stats = stats.continuations[node->index];
            expect(node_stats.index == node->index);
            expect(node_stats.token == node->token);
            expect(node_stats.name == name);
            expect(node_stats.dispatch_count == 1u);
            expect(node_stats.executed_count == executed);
            expect(node_stats.peak_queued_count == peak);
        };
        auto &&entry = stats.continuations.front();
        expect(entry.name == "<entry>");
        expect(entry.dispatch_count == 1u);
        expect(entry.executed_count == N);
        expect_node("even", N / 2u, N / 2u);
        expect_node("odd", N / 2u, N / 2u);
        expect_node("join", N, N);
    };

    "wavefront_auxiliary_work_is_bounded_and_exact"_test = [options] {
        constexpr uint N = 20u;
        constexpr uint capacity = 8u;

        static_assert(wavefront_auxiliary_queue_can_admit(8u, 0u, 8u, 1u));
        static_assert(wavefront_auxiliary_queue_can_admit(8u, 2u, 3u, 2u));
        static_assert(!wavefront_auxiliary_queue_can_admit(8u, 1u, 8u, 1u));
        static_assert(!wavefront_auxiliary_queue_can_admit(8u, 9u, 0u, 1u));
        static_assert(wavefront_auxiliary_queue_can_admit(
            std::numeric_limits<uint>::max(), 0u,
            std::numeric_limits<uint>::max(), 1u));
        static_assert(!wavefront_auxiliary_queue_can_admit(
            std::numeric_limits<uint>::max(), 0u,
            std::numeric_limits<uint>::max(),
            std::numeric_limits<uint>::max()));

        auto dc = luisa::test::coro_test::create_device(options);
        auto &device = dc.device;
        Stream stream = device.create_stream();
        for (auto incremental : {false, true}) {
            auto main_visits = device.create_buffer<uint>(N);
            auto auxiliary_visits = device.create_buffer<uint>(N);
            auto overflow = device.create_buffer<uint>(1u);
            luisa::vector<uint> zero_visits(N);
            uint zero = 0u;
            stream << main_visits.copy_from(luisa::span{zero_visits})
                   << auxiliary_visits.copy_from(luisa::span{zero_visits})
                   << overflow.copy_from(luisa::span{&zero, 1u})
                   << synchronize();

            auto auxiliary = luisa::make_shared<TestWavefrontAuxiliaryWork>(
                device, capacity,
                luisa::vector<WavefrontCoroAuxiliaryProducer>{
                    {.continuation = "sparse_publish",
                     .max_emitted_per_invocation = 1u},
                    {.continuation = "dense_publish",
                     .max_emitted_per_invocation = 1u}});
            auto *items = &auxiliary->items();
            auto *item_count = &auxiliary->count();
            auto coro = Coroutine<void(
                Buffer<uint>, Buffer<uint>, Buffer<uint>)>(
                [items, item_count, capacity](BufferUInt main_output,
                                              BufferUInt,
                                              BufferUInt overflow_output) {
                    auto item_buffer = Expr<Buffer<uint>>{*items};
                    auto count_buffer = Expr<Buffer<uint>>{*item_count};
                    auto tid = dispatch_x();
                    $suspend("sparse_publish");
                    $if((tid % capacity) == 0u) {
                        auto slot = count_buffer.atomic(0u).fetch_add(1u);
                        $if(slot < capacity) {
                            item_buffer.write(slot, tid);
                        }
                        $else {
                            overflow_output.atomic(0u).fetch_add(1u);
                        };
                    };
                    $suspend("dense_publish");
                    auto slot = count_buffer.atomic(0u).fetch_add(1u);
                    $if(slot < capacity) {
                        item_buffer.write(slot, tid);
                    }
                    $else {
                        overflow_output.atomic(0u).fetch_add(1u);
                    };
                    main_output.atomic(tid).fetch_add(1u);
                });

            WavefrontCoroScheduler<
                Buffer<uint>, Buffer<uint>, Buffer<uint>> scheduler{
                device, coro,
                WavefrontCoroSchedulerConfig{
                    .thread_count = capacity,
                    .gather_by_sorting = false,
                    .frame_buffer_compaction = true,
                    .report_stats = true,
                    .execution_block_size = 32u,
                    .largest_continuation_first = true,
                    .incremental_continuation_counts = incremental}};
            scheduler.register_auxiliary_work(auxiliary);
            scheduler(main_visits, auxiliary_visits, overflow)
                .dispatch(N)(stream);

            luisa::vector<uint> host_main(N);
            luisa::vector<uint> host_auxiliary(N);
            uint host_overflow = ~0u;
            stream << main_visits.copy_to(luisa::span{host_main})
                   << auxiliary_visits.copy_to(luisa::span{host_auxiliary})
                   << overflow.copy_to(luisa::span{&host_overflow, 1u})
                   << synchronize();

            expect(host_overflow == 0u)
                << "admission control must prevent auxiliary queue overflow";
            expect(std::all_of(host_main.begin(), host_main.end(),
                               [](auto count) noexcept { return count == 1u; }))
                << "auxiliary scheduling must not perturb main coroutine work";
            auto auxiliary_exact = true;
            for (auto i = 0u; i < N; ++i) {
                auto expected = 1u + static_cast<uint>((i % capacity) == 0u);
                auxiliary_exact &= host_auxiliary[i] == expected;
            }
            expect(auxiliary_exact)
                << "every published auxiliary item must execute exactly once";

            auto &&stats = scheduler.last_dispatch_stats();
            expect(stats.auxiliary_work.size() == 1u);
            if (!stats.auxiliary_work.empty()) {
                auto &&side = stats.auxiliary_work.front();
                expect(side.name == "test_side_work");
                expect(side.executed_count == N + 3u);
                expect(side.dispatch_count == 5u)
                    << "full frame batches must drain sparse work before the "
                       "dense producer, while a safe tail may coalesce";
                expect(side.peak_queued_count == capacity);
            }
        }
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
            auto v = tid * 7u + 3u;
            $suspend("sort_me", coro_frame_export(
                                     "coro_hint", coro_hint));
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

    "wavefront_debug_name_is_not_scheduler_abi"_test = [options] {
        constexpr uint N = 17u;

        auto dc = luisa::test::coro_test::create_device(options);
        auto &device = dc.device;
        Stream stream = device.create_stream();
        auto output = device.create_buffer<uint>(N);

        auto coro = Coroutine<void(Buffer<uint>)>([](BufferUInt buf) {
            auto tid = dispatch_x();
            auto hint = def(tid * 5u + 1u);
            hint.set_name("coro_hint");
            $suspend("sort_me");
            buf.write(tid, hint);
        });

        WavefrontCoroSchedulerConfig cfg{
            .thread_count = N,
            .global_memory_soa = true,
            .gather_by_sorting = false,
            .frame_buffer_compaction = true,
            .hint_range = N,
            .hint_fields = {"sort_me"},
        };
        WavefrontCoroScheduler<Buffer<uint>> scheduler{device, coro, cfg};
        expect(scheduler.config().hint_fields.empty())
            << "an ordinary diagnostic name must not become scheduler ABI";

        scheduler(output).dispatch(N)(stream);
        luisa::vector<uint> host(N);
        stream << output.copy_to(luisa::span{host}) << synchronize();
        auto correct = true;
        for (auto i = 0u; i < N; ++i) {
            correct &= host[i] == i * 5u + 1u;
        }
        expect(correct)
            << "disabling an undeclared hint must preserve coroutine values";
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
            auto v = tid + 1u;
            $suspend("sort_me", coro_frame_export(
                                     "coro_hint", coro_hint));
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
            auto v = tid * 2u + 5u;
            $suspend("sort_me", coro_frame_export(
                                     "coro_hint", coro_hint));
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
            auto value = tid * 17u + 5u;
            $suspend("sort_me", coro_frame_export(
                                     "coro_hint", coro_hint));
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

    "wavefront_largest_continuation_first_is_greedy_and_complete"_test = [options] {
        constexpr uint N = 65u;

        auto dc = luisa::test::coro_test::create_device(options);
        auto &device = dc.device;
        Stream stream = device.create_stream();

        auto order = device.create_buffer<uint>(N + 1u);
        auto visits = device.create_buffer<uint>(N);
        luisa::vector<uint> zero_order(N + 1u);
        luisa::vector<uint> zero_visits(N);
        stream << order.copy_from(luisa::span{zero_order})
               << visits.copy_from(luisa::span{zero_visits})
               << synchronize();

        auto coro = Coroutine<void(Buffer<uint>, Buffer<uint>)>(
            [](BufferUInt order_buffer, BufferUInt visit_buffer) {
                auto tid = dispatch_x();
                $if (tid == 0u) {
                    $suspend("small");
                    auto slot = order_buffer.atomic(0u).fetch_add(1u);
                    order_buffer.write(slot + 1u, 1u);
                }
                $else {
                    $suspend("large");
                    auto slot = order_buffer.atomic(0u).fetch_add(1u);
                    order_buffer.write(slot + 1u, 2u);
                };
                visit_buffer.atomic(tid).fetch_add(1u);
            });

        WavefrontCoroSchedulerConfig cfg{
            .thread_count = N,
            .global_memory_soa = true,
            .gather_by_sorting = true,
            .frame_buffer_compaction = true,
            .execution_block_size = 32u,
            .largest_continuation_first = true,
        };
        WavefrontCoroScheduler<Buffer<uint>, Buffer<uint>> scheduler{
            device, coro, cfg};
        scheduler(order, visits).dispatch(N)(stream);

        luisa::vector<uint> host_order(N + 1u);
        luisa::vector<uint> host_visits(N);
        stream << order.copy_to(luisa::span{host_order})
               << visits.copy_to(luisa::span{host_visits})
               << synchronize();

        auto greedy = host_order[0u] == N;
        for (auto i = 1u; i < N; ++i) {
            greedy &= host_order[i] == 2u;
        }
        greedy &= host_order[N] == 1u;
        expect(greedy)
            << "the largest queue must run before the earlier small token";
        expect(std::all_of(host_visits.begin(), host_visits.end(),
                           [](auto count) noexcept { return count == 1u; }))
            << "greedy scheduling must neither lose nor duplicate frames";
    };

    "wavefront_single_frame_default_refill_makes_progress"_test = [options] {
        constexpr uint N = 3u;

        auto dc = luisa::test::coro_test::create_device(options);
        auto &device = dc.device;
        Stream stream = device.create_stream();

        auto visits = device.create_buffer<uint>(N);
        luisa::vector<uint> zero(N);
        stream << visits.copy_from(luisa::span{zero}) << synchronize();

        auto coro = Coroutine<void(Buffer<uint>)>([](BufferUInt visit_buffer) {
            auto tid = dispatch_x();
            $suspend("checkpoint");
            visit_buffer.atomic(tid).fetch_add(1u);
        });

        WavefrontCoroSchedulerConfig cfg{
            .thread_count = 1u,
            .global_memory_soa = true,
            .gather_by_sorting = true,
            .frame_buffer_compaction = true,
            .execution_block_size = 32u,
            .largest_continuation_first = true,
            .refill_continuations = {"checkpoint"},
        };
        WavefrontCoroScheduler<Buffer<uint>> scheduler{device, coro, cfg};
        scheduler(visits).dispatch(N)(stream);

        luisa::vector<uint> host(N);
        stream << visits.copy_to(luisa::span{host}) << synchronize();
        expect(std::all_of(host.begin(), host.end(),
                           [](auto count) noexcept { return count == 1u; }))
            << "a one-frame scheduler must escape its empty state and drain "
               "all logical invocations";
    };

    "wavefront_refill_waits_for_named_alignment_queue"_test = [options] {
        constexpr uint N = 8u;
        constexpr uint capacity = 4u;
        constexpr uint log_capacity = 16u;

        auto dc = luisa::test::coro_test::create_device(options);
        auto &device = dc.device;
        Stream stream = device.create_stream();

        auto log = device.create_buffer<uint>(log_capacity);
        auto visits = device.create_buffer<uint>(N);
        luisa::vector<uint> zero_log(log_capacity);
        luisa::vector<uint> zero_visits(N);
        stream << log.copy_from(luisa::span{zero_log})
               << visits.copy_from(luisa::span{zero_visits})
               << synchronize();

        auto coro = Coroutine<void(Buffer<uint>, Buffer<uint>)>(
            [](BufferUInt event_log, BufferUInt visit_buffer) {
                auto tid = dispatch_x();
                auto entry_slot = event_log.atomic(0u).fetch_add(1u);
                event_log.write(entry_slot + 1u, 100u + tid);
                $suspend("refill");
                $if ((tid & 3u) == 0u) {
                    $suspend("blocked");
                    auto blocked_slot = event_log.atomic(0u).fetch_add(1u);
                    event_log.write(blocked_slot + 1u, 300u + tid);
                };
                visit_buffer.atomic(tid).fetch_add(1u);
            });

        WavefrontCoroSchedulerConfig cfg{
            .thread_count = capacity,
            .global_memory_soa = true,
            .gather_by_sorting = true,
            .frame_buffer_compaction = true,
            .execution_block_size = 32u,
            .largest_continuation_first = true,
            .refill_continuations = {"refill"},
            .refill_threshold = 2u,
        };
        WavefrontCoroScheduler<Buffer<uint>, Buffer<uint>> scheduler{
            device, coro, cfg};
        scheduler(log, visits).dispatch(N)(stream);

        luisa::vector<uint> host_log(log_capacity);
        luisa::vector<uint> host_visits(N);
        stream << log.copy_to(luisa::span{host_log})
               << visits.copy_to(luisa::span{host_visits})
               << synchronize();

        auto first_blocked = log_capacity;
        auto second_batch_entry = log_capacity;
        auto event_count = std::min(host_log[0u], log_capacity - 1u);
        for (auto i = 0u; i < event_count; ++i) {
            auto event = host_log[i + 1u];
            if (event == 300u) { first_blocked = std::min(first_blocked, i); }
            if (event >= 104u && event <= 107u) {
                second_batch_entry = std::min(second_batch_entry, i);
            }
        }
        expect(first_blocked < second_batch_entry)
            << "low occupancy at an unlisted continuation must drain that "
               "queue before admitting the next entry batch";
        expect(std::all_of(host_visits.begin(), host_visits.end(),
                           [](auto count) noexcept { return count == 1u; }))
            << "alignment-aware refill must neither lose nor duplicate frames";
    };

    "wavefront_incremental_counts_conserve_sparse_transitions"_test =
        [options] {
            constexpr uint N = 257u;
            constexpr uint capacity = 31u;

            auto dc = luisa::test::coro_test::create_device(options);
            auto &device = dc.device;
            Stream stream = device.create_stream();

            auto coro = Coroutine<void(Buffer<uint>, Buffer<uint>)>(
                [](BufferUInt values, BufferUInt visit_count) {
                    auto tid = dispatch_x();
                    auto value = tid * 17u + 3u;
                    $suspend(17u, "refill");
                    $if((tid % 3u) == 0u) {
                        $suspend(101u, "thirds");
                        value += 5u;
                    }
                    $else {
                        $suspend(307u, "others");
                        value += 11u;
                    };
                    auto remaining = def(tid & 3u);
                    $while(remaining != 0u) {
                        // One static suspension in a dynamic loop induces a
                        // continuation self-edge. Its queue count must remain
                        // unchanged for that transition, not underflow or
                        // double-count the frame.
                        $suspend(509u, "loop");
                        value += remaining;
                        remaining -= 1u;
                    };
                    values.write(tid, value);
                    visit_count.atomic(tid).fetch_add(1u);
                });

            auto output = device.create_buffer<uint>(N);
            auto visits = device.create_buffer<uint>(N);
            luisa::vector<uint> zero(N);
            stream << output.copy_from(luisa::span{zero})
                   << visits.copy_from(luisa::span{zero})
                   << synchronize();

            WavefrontCoroSchedulerConfig cfg{
                .thread_count = capacity,
                .global_memory_soa = true,
                .gather_by_sorting = true,
                .frame_buffer_compaction = true,
                .execution_block_size = 32u,
                .largest_continuation_first = true,
                .refill_continuations = {"refill"},
                .refill_threshold = capacity / 2u,
                .incremental_continuation_counts = true,
            };
            WavefrontCoroScheduler<Buffer<uint>, Buffer<uint>> scheduler{
                device, coro, cfg};
            scheduler(output, visits).dispatch(N)(stream);

            luisa::vector<uint> host_output(N);
            luisa::vector<uint> host_visits(N);
            stream << output.copy_to(luisa::span{host_output})
                   << visits.copy_to(luisa::span{host_visits})
                   << synchronize();

            auto correct = true;
            for (auto i = 0u; i < N; ++i) {
                auto expected = i * 17u + 3u +
                                (i % 3u == 0u ? 5u : 11u);
                for (auto r = i & 3u; r != 0u; --r) {
                    expected += r;
                }
                correct &= host_output[i] == expected;
                correct &= host_visits[i] == 1u;
            }
            expect(correct)
                << "incremental queue counts must preserve every sparse-token "
                   "branch and self-loop transition under refill/compaction";
        };

    "wavefront_incremental_counts_do_not_perturb_user_kernels"_test =
        [options] {
            auto dc = luisa::test::coro_test::create_device(options);
            auto &device = dc.device;
            auto coro = Coroutine<void(Buffer<uint>)>([](BufferUInt output) {
                auto tid = dispatch_x();
                auto value = def(tid * 7u + 3u);
                $suspend("first");
                $if ((tid & 1u) != 0u) {
                    $suspend("odd");
                    value += 11u;
                };
                output.write(tid, value);
            });

            auto config = WavefrontCoroSchedulerConfig{
                .thread_count = 64u,
                .global_memory_soa = true,
                .gather_by_sorting = false,
                .frame_buffer_compaction = true,
                .execution_block_size = 32u,
                .largest_continuation_first = true,
            };
            auto incremental_config = config;
            incremental_config.incremental_continuation_counts = true;
            WavefrontCoroScheduler<Buffer<uint>> materialized{
                device, coro, config};
            WavefrontCoroScheduler<Buffer<uint>> incremental{
                device, coro, incremental_config};

            auto materialized_infos = materialized.shader_infos();
            auto incremental_infos = incremental.shader_infos();
            auto user_kernel_count = 0u;
            auto user_kernels_match = true;
            for (auto &&expected : materialized_infos) {
                auto is_user_kernel =
                    expected.stage.starts_with("wavefront_generate/") ||
                    expected.stage.starts_with("wavefront_resume_");
                if (!is_user_kernel) { continue; }
                user_kernel_count++;
                auto actual = std::find_if(
                    incremental_infos.begin(), incremental_infos.end(),
                    [&](auto &&info) noexcept {
                        return info.stage == expected.stage;
                    });
                user_kernels_match &=
                    actual != incremental_infos.end() &&
                    actual->structural_hash == expected.structural_hash;
            }
            auto has_generated_publisher = std::any_of(
                incremental_infos.begin(), incremental_infos.end(),
                [](auto &&info) noexcept {
                    return info.stage ==
                           "wavefront_publish_generated_count";
                });
            auto has_resumed_publisher = std::any_of(
                incremental_infos.begin(), incremental_infos.end(),
                [](auto &&info) noexcept {
                    return info.stage ==
                           "wavefront_publish_resumed_count";
                });
            expect(user_kernel_count == coro.subroutine_count());
            expect(user_kernels_match)
                << "incremental queue accounting must not alter a user "
                   "continuation's AST, argument ABI, or cache identity";
            expect(has_generated_publisher);
            expect(has_resumed_publisher)
                << "incremental accounting must be isolated in scheduler-"
                   "owned publication kernels";
        };
}

int main(int argc, char *argv[]) {
    auto options = luisa::test::coro_test::parse_options(argc, argv);
    reg_coro_wavefront(options);
    return luisa::test::coro_test::run_tests(argc, argv);
}
