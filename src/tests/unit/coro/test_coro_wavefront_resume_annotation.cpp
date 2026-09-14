#include "ut/ut.hpp"

#include "coro_test_utils.h"

#include <luisa/coro/schedulers/wavefront.h>
#include <luisa/dsl/sugar.h>

using namespace luisa;
using namespace luisa::compute;
using namespace luisa::compute::coro;
using namespace boost::ut;

namespace {

// The keys are a permutation of [0, count), so a scatter is sufficient to
// isolate the queue handoff. This is deliberately not another radix sorter.
class Permute final : public WavefrontCoroSchedulerExtensionHandler {
  private:
    Buffer<uint> _sorted;
    Shader1D<ByteBuffer, Buffer<uint>, uint, uint> _scatter;
    Shader1D<ByteBuffer, Buffer<uint>, uint, uint> _copy;

  public:
    Permute(WavefrontCoroExtensionPrepareContext &context, const WavefrontCoroExtensionStage &stage,
            BufferView<uint> witness)
        : _sorted{context.device.create_buffer<uint>(context.frame_capacity)} {
        auto *desc = &context.frame_desc;
        auto *key = &stage.binding("key");
        auto fields = stage.dataflow->reconstruct_slots;
        auto load_key = [desc, key, fields, layout = context.frame_layout,
                         soa = context.global_memory_soa](const ByteBufferVar &storage, UInt index,
                                                          UInt capacity) {
            auto frame = CoroFrame::create(desc);
            coro_frame_load_into(frame, storage, index, capacity, layout, soa, span{fields}, false,
                                 false);
            return key->read<uint>(frame);
        };
        Kernel1D scatter = [this, load_key](ByteBufferVar storage, BufferUInt indices,
                                            UInt capacity, UInt count) {
            auto x = dispatch_x();
            $if(x >= count) { $return(); };
            auto index = indices.read(x);
            _sorted->write(load_key(storage, index, capacity), index);
        };
        Kernel1D copy = [this, load_key, witness](ByteBufferVar storage, BufferUInt indices,
                                                  UInt capacity, UInt count) {
            auto x = dispatch_x();
            $if(x >= count) { $return(); };
            auto index = _sorted->read(x);
            witness->write(x, load_key(storage, index, capacity));
        };
        _scatter = context.device.compile(scatter);
        _copy = context.device.compile(copy);
    }
    [[nodiscard]] string_view name() const noexcept override { return "permutation"; }
    [[nodiscard]] WavefrontCoroExtensionExecution execution() const noexcept override {
        return WavefrontCoroExtensionExecution::before_resume;
    }
    [[nodiscard]] BufferView<uint>
    dispatch_queue(const WavefrontCoroExtensionDispatchContext &context) noexcept override {
        context.stream << _scatter(context.frame_buffer, context.frame_indices,
                                   context.frame_capacity, context.frame_count)
                              .dispatch(context.frame_count)
                       << _copy(context.frame_buffer, context.frame_indices, context.frame_capacity,
                                context.frame_count)
                              .dispatch(context.frame_count);
        return _sorted.view().subview(0u, context.frame_count);
    }
};

// A semantic write followed by two read-only queue operations. Ranking is
// quadratic but tiny and deterministic; only the queue protocol is under test.
class ChainHandler final : public WavefrontCoroSchedulerExtensionHandler {
  private:
    string _schema;
    Buffer<uint> _sorted;
    Shader1D<ByteBuffer, Buffer<uint>, uint, uint> _shader;

  public:
    ChainHandler(WavefrontCoroExtensionPrepareContext &context,
                 const WavefrontCoroExtensionStage &stage, BufferView<uint> expected,
                 BufferView<uint> diagnostics)
        : _schema{stage.extension->schema()},
          _sorted{context.device.create_buffer<uint>(context.frame_capacity)} {
        auto *desc = &context.frame_desc;
        auto *value = &stage.binding("key");
        auto fields = stage.dataflow->reconstruct_slots;
        auto writes = stage.dataflow->required_def.slots;
        Kernel1D operation = [this, desc, value, fields, writes, expected, diagnostics,
                              layout = context.frame_layout, soa = context.global_memory_soa](
                                 ByteBufferVar storage, BufferUInt indices, UInt capacity,
                                 UInt count) {
            auto x = dispatch_x();
            $if(x >= count) { $return(); };
            auto load = [&](UInt index) {
                auto frame = CoroFrame::create(desc);
                coro_frame_load_into(frame, storage, index, capacity, layout, soa, span{fields},
                                     false, false);
                return frame;
            };
            auto index = indices.read(x);
            auto frame = load(index);
            auto key = value->read<uint>(frame);
            if (_schema == "luisa.test.resume.add") {
                value->write<uint>(frame, key + 5u);
                coro_frame_store(storage, index, capacity, frame, layout, soa, span{writes}, false,
                                 false);
            } else if (_schema == "luisa.test.resume.rank") {
                UInt rank = 0u;
                $for(j, count) {
                    auto other = load(indices.read(j));
                    auto other_key = value->read<uint>(other);
                    rank += ((other_key < key) | ((other_key == key) & (j < x))).cast<uint>();
                };
                _sorted->write(rank, index);
            } else {
                $if(x != 0u) {
                    auto previous = load(indices.read(x - 1u));
                    $if(value->read<uint>(previous) > key) {
                        diagnostics->atomic(0u).fetch_add(1u);
                    };
                };
                expected->write(x, key);
            }
        };
        _shader = context.device.compile(operation);
    }
    [[nodiscard]] string_view name() const noexcept override { return _schema; }
    [[nodiscard]] WavefrontCoroExtensionExecution execution() const noexcept override {
        return _schema == "luisa.test.resume.add" ? WavefrontCoroExtensionExecution::stage
                                                  : WavefrontCoroExtensionExecution::before_resume;
    }
    [[nodiscard]] BufferView<uint>
    dispatch_queue(const WavefrontCoroExtensionDispatchContext &context) noexcept override {
        context.stream << _shader(context.frame_buffer, context.frame_indices,
                                  context.frame_capacity, context.frame_count)
                              .dispatch(context.frame_count);
        return _schema == "luisa.test.resume.rank" ? _sorted.view().subview(0u, context.frame_count)
                                                   : context.frame_indices;
    }
};

// Same schema and target do not prove handler equivalence. This handler's
// selected binding is captured by the facade separately for each boundary.
class BoundaryReadWitness final : public WavefrontCoroSchedulerExtensionHandler {
  private:
    Shader1D<ByteBuffer, Buffer<uint>, uint, uint> _shader;

  public:
    BoundaryReadWitness(WavefrontCoroExtensionPrepareContext &context,
                        const WavefrontCoroExtensionStage &stage,
                        string_view selected_binding, BufferView<uint> witness) {
        auto *desc = &context.frame_desc;
        auto *route = &stage.binding("route");
        auto *value = &stage.binding(selected_binding);
        auto fields = stage.dataflow->reconstruct_slots;
        Kernel1D read = [desc, route, value, fields, witness,
                         layout = context.frame_layout, soa = context.global_memory_soa](
                            ByteBufferVar storage, BufferUInt indices, UInt capacity,
                            UInt count) {
            auto x = dispatch_x();
            $if(x >= count) { $return(); };
            auto frame = CoroFrame::create(desc);
            coro_frame_load_into(frame, storage, indices.read(x), capacity, layout, soa,
                                 span{fields}, false, false);
            witness->write(route->read<uint>(frame), value->read<uint>(frame));
        };
        _shader = context.device.compile(read);
    }
    [[nodiscard]] string_view name() const noexcept override { return "boundary-local-read"; }
    [[nodiscard]] WavefrontCoroExtensionExecution execution() const noexcept override {
        return WavefrontCoroExtensionExecution::before_resume;
    }
    void dispatch(const WavefrontCoroExtensionDispatchContext &context) noexcept override {
        context.stream << _shader(context.frame_buffer, context.frame_indices,
                                  context.frame_capacity, context.frame_count)
                              .dispatch(context.frame_count);
    }
};

class SelfSnapshotWitness final : public WavefrontCoroSchedulerExtensionHandler {
  private:
    bool _record;
    uint _serial{0u};
    Shader1D<ByteBuffer, Buffer<uint>, uint, uint, uint> _shader;

  public:
    SelfSnapshotWitness(WavefrontCoroExtensionPrepareContext &context,
                        const WavefrontCoroExtensionStage &stage, bool record,
                        BufferView<uint> epochs) : _record{record} {
        if (!record) { return; }
        auto *desc = &context.frame_desc;
        auto *route = &stage.binding("route");
        auto fields = stage.dataflow->reconstruct_slots;
        Kernel1D mark = [desc, route, fields, epochs, layout = context.frame_layout,
                         soa = context.global_memory_soa](ByteBufferVar storage,
                                                          BufferUInt indices, UInt capacity,
                                                          UInt count, UInt serial) {
            auto x = dispatch_x();
            $if(x >= count) { $return(); };
            auto frame = CoroFrame::create(desc);
            coro_frame_load_into(frame, storage, indices.read(x), capacity, layout, soa,
                                 span{fields}, false, false);
            epochs->write(route->read<uint>(frame), serial);
        };
        _shader = context.device.compile(mark);
    }
    [[nodiscard]] string_view name() const noexcept override { return "self-snapshot-witness"; }
    [[nodiscard]] WavefrontCoroExtensionExecution execution() const noexcept override {
        return WavefrontCoroExtensionExecution::before_resume;
    }
    void dispatch(const WavefrontCoroExtensionDispatchContext &context) noexcept override {
        if (_record) {
            context.stream << _shader(context.frame_buffer, context.frame_indices,
                                      context.frame_capacity, context.frame_count, _serial++)
                                  .dispatch(context.frame_count);
        }
    }
};

} // namespace

int main(int argc, char *argv[]) {
    auto options = luisa::test::coro_test::parse_options(argc, argv);
    "resume_annotation_owns_descriptors_after_source_destruction"_test = [options] {
        constexpr uint N = 16u;
        auto dc = luisa::test::coro_test::create_device(options);
        auto &device = dc.device;
        auto stream = device.create_stream();
        auto output = device.create_buffer<uint>(N);
        auto witness = device.create_buffer<uint>(N);
        // The source is a compilation input, just as a Kernel is to compile().
        // Both delayed facade registration and runtime descriptors must remain
        // valid after it has been destroyed, without an application keepalive.
        auto make_scheduler = [&] {
            Coroutine<void(Buffer<uint>)> source{[=](BufferUInt output) {
                auto id = dispatch_x();
                $suspend("ordered", coro_sort_by(N - 1u - id, N));
                output.write(thread_x(), id);
            }};
            return make_unique<WavefrontCoroScheduler<Buffer<uint>>>(
                device, source,
                WavefrontCoroSchedulerConfig{
                    .thread_count = N, .gather_by_sorting = false, .execution_block_size = 32u});
        };
        auto scheduler = make_scheduler();
        scheduler->register_extension_handler(
            stream,
            [&](auto &context, auto &stage) -> unique_ptr<WavefrontCoroSchedulerExtensionHandler> {
                return make_unique<Permute>(context, stage, witness.view());
            });
        stream << (*scheduler)(output).dispatch(N);
        vector<uint> host(N);
        stream << output.copy_to(span{host}) << synchronize();
        for (auto i = 0u; i < N; ++i) {
            expect(host[i] == N - 1u - i);
        }
    };
    "resume_annotation_retains_logical_queue_tie_priority"_test = [options] {
        struct Identity final : WavefrontCoroSchedulerExtensionHandler {
            [[nodiscard]] string_view name() const noexcept override { return "identity"; }
            [[nodiscard]] WavefrontCoroExtensionExecution execution() const noexcept override {
                return WavefrontCoroExtensionExecution::before_resume;
            }
            void dispatch(const WavefrontCoroExtensionDispatchContext &) noexcept override {}
        };
        auto dc = luisa::test::coro_test::create_device(options);
        auto &device = dc.device;
        auto stream = device.create_stream();
        auto order = device.create_buffer<uint>(16u);
        auto count = device.create_buffer<uint>(1u);
        for (auto annotated : {false, true}) {
            auto coro = Coroutine<void(Buffer<uint>, Buffer<uint>)>(
                [annotated](BufferUInt output, BufferUInt sequence) {
                    auto id = dispatch_x();
                    $if(id < 8u) {
                        if (annotated) {
                            $suspend("first",
                                     coro_annotation("luisa.test.identity").read("id", id));
                        } else {
                            $suspend("first");
                        }
                    }
                    $else { $suspend("second"); };
                    output.write(id, sequence.atomic(0u).fetch_add(1u));
                });
            expect(coro.graph().node_by_name("first")->index <
                   coro.graph().node_by_name("second")->index);
            WavefrontCoroScheduler<Buffer<uint>, Buffer<uint>> scheduler{
                device,
                coro,
                {.thread_count = 16u,
                 .gather_by_sorting = false,
                 .execution_block_size = 32u,
                 .largest_continuation_first = true,
                 .incremental_continuation_counts = true,
                 .fused_continuation_counts = true}};
            scheduler.register_extension_handler(
                stream, [](auto &, auto &) -> unique_ptr<WavefrontCoroSchedulerExtensionHandler> {
                    return make_unique<Identity>();
                });
            uint zero = 0u;
            vector<uint> host(16u);
            stream << count.copy_from(span{&zero, 1u});
            stream << scheduler(order, count).dispatch(16u);
            stream << order.copy_to(span{host}) << synchronize();
            for (auto i = 0u; i < 16u; ++i) {
                expect((host[i] < 8u) == (i < 8u))
                    << "a resume annotation must not replace logical queue tie priority";
            }
        }
    };
    "resume_annotation_retains_logical_queue_cardinality"_test = [options] {
        struct Identity final : WavefrontCoroSchedulerExtensionHandler {
            [[nodiscard]] string_view name() const noexcept override { return "identity"; }
            [[nodiscard]] WavefrontCoroExtensionExecution execution() const noexcept override {
                return WavefrontCoroExtensionExecution::before_resume;
            }
            void dispatch(const WavefrontCoroExtensionDispatchContext &) noexcept override {}
        };
        constexpr uint N = 15u;
        auto dc = luisa::test::coro_test::create_device(options);
        auto &device = dc.device;
        auto stream = device.create_stream();
        auto input = device.create_buffer<uint>(N);
        auto output = device.create_buffer<uint>(N);
        auto sequence = device.create_buffer<uint>(1u);
        // 0: plain target; 1: both annotated entries; 2: the producer's
        // annotation is ignored, mixing a bare target queue with an Extension.
        for (auto annotation_mode = 0u; annotation_mode < 3u; ++annotation_mode) {
            auto annotated = annotation_mode != 0u;
            Coroutine<void(Buffer<uint>, Buffer<uint>, Buffer<uint>)> coro{
                [annotated](BufferUInt routes, BufferUInt order, BufferUInt count) {
                    // The population split is supplied through a runtime buffer,
                    // not specialized into the source or discovered by profiling.
                    auto route = routes.read(dispatch_x());
                    $if((route >= 4u) & (route < 12u)) { $suspend("producer"); };
                    $if(route < 8u) {
                        if (annotated) {
                            $suspend("target",
                                     coro_annotation("luisa.test.identity").read("route", route));
                        } else {
                            $suspend("target");
                        }
                    }
                    $else { $suspend("rival"); };
                    order.write(route, count.atomic(0u).fetch_add(1u));
                }};
            const auto *target = coro.graph().node_by_name("target");
            expect(target != nullptr);
            auto incoming = 0u;
            for (auto &&boundary : coro.graph().boundaries()) {
                incoming += boundary.to_index == target->index;
            }
            expect(incoming == 2u) << "entry and producer must reach the same static target";
            for (auto soa : {false, true}) {
                for (auto accounting = 0u; accounting < 3u; ++accounting) {
                    WavefrontCoroScheduler<Buffer<uint>, Buffer<uint>, Buffer<uint>> scheduler{
                        device,
                        coro,
                        {.thread_count = N,
                         .global_memory_soa = soa,
                         .gather_by_sorting = false,
                         .frame_buffer_compaction = accounting == 0u || !soa,
                         .report_stats = true,
                         .execution_block_size = 32u,
                         .largest_continuation_first = true,
                         .incremental_continuation_counts = accounting != 0u,
                         .fused_continuation_counts = accounting == 2u}};
                    scheduler.register_extension_handler(
                        stream,
                        [&](auto &, auto &stage) -> unique_ptr<WavefrontCoroSchedulerExtensionHandler> {
                            if (annotation_mode == 2u &&
                                stage.boundary->from_index != coro.graph().entry_index()) {
                                return nullptr;
                            }
                            return make_unique<Identity>();
                        });
                    for (auto reverse : {false, true}) {
                        vector<uint> routes(N), order(N, ~0u);
                        for (auto i = 0u; i < N; ++i) { routes[i] = reverse ? N - 1u - i : i; }
                        uint zero = 0u;
                        stream << input.copy_from(span{routes}) << output.copy_from(span{order})
                               << sequence.copy_from(span{&zero, 1u});
                        // All 15 inputs are admitted once; there is no refill
                        // or tail that can change the counterexample.
                        stream << scheduler(input, output, sequence).dispatch(N);
                        stream << output.copy_to(span{order}) << synchronize();
                        // Initial queues: target-A=4, producer=8, rival=3.
                        // Producer adds target-B=4 and makes rival=7. Logical
                        // target=8 must win although each physical entry has 4.
                        for (auto i = 0u; i < N; ++i) {
                            expect(order[i] < N);
                            expect((order[i] < 8u) == (i < 8u))
                                << "annotation entry queues must not fragment logical target cardinality";
                        }
                        expect(scheduler.last_dispatch_stats().generated_count == N);
                        expect(scheduler.last_dispatch_stats().resumed_count == N + 8u);
                        expect(scheduler.last_dispatch_stats().continuations[target->index]
                                   .peak_queued_count == 8u)
                            << "logical continuation peak must sum all incoming physical queues";
                    }
                }
            }
        }
    };
    "resume_annotation_snapshots_self_rescheduling_membership"_test = [options] {
        constexpr uint N = 15u;
        auto dc = luisa::test::coro_test::create_device(options);
        auto &device = dc.device;
        auto stream = device.create_stream();
        auto input = device.create_buffer<uint>(N);
        auto output = device.create_buffer<uint>(N);
        auto epochs = device.create_buffer<uint>(N);
        Coroutine<void(Buffer<uint>, Buffer<uint>)> coro{
            [](BufferUInt routes, BufferUInt visits) {
                auto route = routes.read(dispatch_x());
                $if(route >= 8u) { $suspend("producer"); };
                UInt iteration = 0u;
                auto twice = (route < 3u) | ((route >= 8u) & (route < 12u));
                auto limit = select(1u, 2u, twice);
                $while(iteration < limit) {
                    $if(route < 12u) {
                        $suspend("target",
                                 coro_annotation("luisa.test.snapshot").read("route", route));
                    }
                    $else { $suspend("rival"); };
                    iteration += 1u;
                };
                visits.write(route, iteration);
            }};
        const auto *target = coro.graph().node_by_name("target");
        expect(target != nullptr);
        for (auto soa : {false, true}) {
            for (auto accounting = 0u; accounting < 3u; ++accounting) {
                WavefrontCoroScheduler<Buffer<uint>, Buffer<uint>> scheduler{
                    device,
                    coro,
                    {.thread_count = N,
                     .global_memory_soa = soa,
                     .gather_by_sorting = false,
                     .frame_buffer_compaction = accounting == 0u || !soa,
                     .report_stats = true,
                     .execution_block_size = 32u,
                     .largest_continuation_first = true,
                     .incremental_continuation_counts = accounting != 0u,
                     .fused_continuation_counts = accounting == 2u}};
                scheduler.register_extension_handler(
                    stream,
                    [&](auto &context, auto &stage) -> unique_ptr<WavefrontCoroSchedulerExtensionHandler> {
                        return make_unique<SelfSnapshotWitness>(
                            context, stage, stage.boundary->from_index == target->index,
                            epochs.view());
                    });
                vector<uint> routes(N), actual(N, ~0u), observed(N, ~0u);
                for (auto i = 0u; i < N; ++i) { routes[i] = N - 1u - i; }
                stream << input.copy_from(span{routes}) << output.copy_from(span{actual})
                       << epochs.copy_from(span{observed});
                stream << scheduler(input, output).dispatch(N);
                stream << output.copy_to(span{actual}) << epochs.copy_to(span{observed})
                       << synchronize();
                // Initial target=8 beats producer=7, leaving three self paths.
                // Producer then adds four target paths and three rivals.
                // The logical target group (old self=3 + incoming=4) wins.
                // Incoming paths reschedule themselves, but must not join the
                // already snapshotted old-self batch. No ordering within a
                // batch, or among that group's distinct members, is assumed.
                for (auto i = 0u; i < N; ++i) {
                    auto twice = i < 3u || (i >= 8u && i < 12u);
                    expect(actual[i] == (twice ? 2u : 1u));
                    auto expected_epoch = i < 3u ? 0u : (twice ? 1u : ~0u);
                    expect(observed[i] == expected_epoch)
                        << "new self work must remain outside the current logical-group snapshot";
                }
                expect(scheduler.last_dispatch_stats().generated_count == N);
                expect(scheduler.last_dispatch_stats().resumed_count == 29u);
            }
        }
    };
    "resume_annotation_retains_boundary_local_binding_choice"_test = [options] {
        constexpr uint N = 15u;
        auto dc = luisa::test::coro_test::create_device(options);
        auto &device = dc.device;
        auto stream = device.create_stream();
        auto input = device.create_buffer<uint>(N);
        auto output = device.create_buffer<uint>(N);
        auto witness = device.create_buffer<uint>(N);
        Coroutine<void(Buffer<uint>, Buffer<uint>)> coro{
            [](BufferUInt routes, BufferUInt result) {
                auto route = routes.read(dispatch_x());
                auto left = route * 3u + 7u;
                auto right = route * 11u + 13u;
                $if((route >= 4u) & (route < 12u)) { $suspend("producer"); };
                $if(route < 8u) {
                    $suspend("target",
                             coro_annotation("luisa.test.boundary.read")
                                 .read("route", route).read("left", left).read("right", right));
                }
                $else { $suspend("rival"); };
                result.write(route, left + right);
            }};
        for (auto soa : {false, true}) {
            WavefrontCoroScheduler<Buffer<uint>, Buffer<uint>> scheduler{
                device,
                coro,
                {.thread_count = N,
                 .global_memory_soa = soa,
                 .gather_by_sorting = false,
                 .frame_buffer_compaction = false,
                 .execution_block_size = 32u,
                 .largest_continuation_first = true,
                 .incremental_continuation_counts = true,
                 .fused_continuation_counts = true}};
            scheduler.register_extension_handler(
                stream,
                [&](auto &context, auto &stage) -> unique_ptr<WavefrontCoroSchedulerExtensionHandler> {
                    // Both values are read by the same annotation and cannot
                    // occupy one colored slot. The same handler name/schema
                    // intentionally chooses different bindings at each edge.
                    auto left = stage.binding("left").pieces();
                    auto right = stage.binding("right").pieces();
                    expect(left.size() == 1u && right.size() == 1u);
                    expect(left[0u].field_index != right[0u].field_index);
                    auto selected = stage.boundary->from_index == coro.graph().entry_index()
                                        ? string_view{"left"} : string_view{"right"};
                    return make_unique<BoundaryReadWitness>(context, stage, selected, witness.view());
                });
            vector<uint> routes(N), actual(N, ~0u), observed(N, ~0u);
            for (auto i = 0u; i < N; ++i) { routes[i] = N - 1u - i; }
            stream << input.copy_from(span{routes}) << output.copy_from(span{actual})
                   << witness.copy_from(span{observed});
            stream << scheduler(input, output).dispatch(N);
            stream << output.copy_to(span{actual}) << witness.copy_to(span{observed})
                   << synchronize();
            for (auto i = 0u; i < N; ++i) {
                expect(actual[i] == i * 14u + 20u);
                auto expected = i < 4u ? i * 3u + 7u : (i < 8u ? i * 11u + 13u : ~0u);
                expect(observed[i] == expected)
                    << "same schema/name must not replace a boundary's captured binding choice";
            }
        }
    };
    "resume_annotation_keeps_queue_permutation"_test = [options] {
        constexpr uint N = 16u;
        auto dc = luisa::test::coro_test::create_device(options);
        auto &device = dc.device;
        auto stream = device.create_stream();
        auto output = device.create_buffer<uint>(2u * N);
        auto witness = device.create_buffer<uint>(N);
        auto coro = Coroutine<void(Buffer<uint>)>([=](BufferUInt output) {
            auto tid = dispatch_x();
            auto key = N - 1u - tid;
            $suspend("ordered", coro_sort_by(key, N));
            // Unlike dispatch_x(), thread_x() is the physical resumed lane.
            // Everything fits in one block, so no atomic arrival-order oracle
            // or assumption about gather's ordering is needed.
            output.write(thread_x(), tid);
            output.write(N + tid, tid * 7u + 3u);
        });
        for (auto soa : {false, true}) {
            for (auto mode = 0u; mode < 4u; ++mode) {
                WavefrontCoroScheduler<Buffer<uint>> scheduler{
                    device,
                    coro,
                    {.thread_count = N,
                     .global_memory_soa = soa,
                     .gather_by_sorting = mode == 3u,
                     .frame_buffer_compaction = true,
                     .report_stats = true,
                     .execution_block_size = 32u,
                     .largest_continuation_first = mode != 0u,
                     .incremental_continuation_counts = mode == 1u || mode == 2u,
                     .fused_continuation_counts = mode == 2u}};
                scheduler.register_extension_handler(
                    stream,
                    [&](auto &context,
                        auto &stage) -> unique_ptr<WavefrontCoroSchedulerExtensionHandler> {
                        if (stage.extension->schema() != "luisa.coro.schedule.sort") {
                            return nullptr;
                        }
                        return make_unique<Permute>(context, stage, witness.view());
                    });
                stream << scheduler(output).dispatch(N);
                vector<uint> host(2u * N), sorted_keys(N);
                stream << output.copy_to(span{host}) << witness.copy_to(span{sorted_keys})
                       << synchronize();
                for (auto i = 0u; i < N; ++i) {
                    expect(sorted_keys[i] == i) << "handler really sorted the selected queue";
                    expect(host[i] == N - 1u - i) << "continuation receives the handler's order";
                    expect(host[N + i] == i * 7u + 3u) << "logical frame state is preserved";
                }
                expect(scheduler.last_dispatch_stats().resumed_count == N);
            }
        }
    };
    "resume_annotation_chain_preserves_each_boundary_and_relocation"_test = [options] {
        constexpr uint N = 67u, capacity = 19u;
        auto dc = luisa::test::coro_test::create_device(options);
        auto &device = dc.device;
        auto stream = device.create_stream();
        auto expected = device.create_buffer<uint>(capacity);
        auto output = device.create_buffer<uint>(2u * N);
        auto diagnostics = device.create_buffer<uint>(2u);
        auto coro = Coroutine<void(Buffer<uint>)>(
            [=, expected = expected.view(), diagnostics = diagnostics.view()](BufferUInt output) {
                auto tid = dispatch_x();
                auto state = tid + 1u;
                auto dormant = make_uint3(tid + 7u, tid * 3u + 1u, tid * 11u);
                // Several incoming scopes and a self edge reach the same static
                // annotation site; its colored binding plans stay boundary-local.
                $if((tid & 1u) != 0u) { $suspend("prelude"); };
                $for(iteration, tid % 3u + 1u) {
                    $suspend("ordered_cycle",
                             coro_stage("luisa.test.resume.add").read_write("key", state),
                             coro_annotation("luisa.test.resume.rank").read("key", state),
                             coro_annotation("luisa.test.resume.ignored").read("key", state),
                             coro_annotation("luisa.test.resume.audit").read("key", state));
                    // read binds a boundary snapshot, while read_write binds the
                    // lvalue updated by the add stage. Sorting reads the former;
                    // the resumed continuation receives the latter (+5).
                    $if(expected->read(thread_x()) + 5u != state) {
                        diagnostics->atomic(1u).fetch_add(1u);
                    };
                };
                output.write(tid, state);
                output.write(N + tid, dormant.x + dormant.y + dormant.z);
            });
        expect(coro.graph().boundary_count() > 1u);
        for (auto soa : {false, true}) {
            for (auto compact : {false, true}) {
                for (auto mode = 0u; mode < 4u; ++mode) {
                    WavefrontCoroScheduler<Buffer<uint>> scheduler{
                        device,
                        coro,
                        {.thread_count = capacity,
                         .global_memory_soa = soa,
                         .gather_by_sorting = mode == 3u,
                         .frame_buffer_compaction = compact,
                         .report_stats = true,
                         .execution_block_size = 32u,
                         .largest_continuation_first = mode != 0u,
                         .refill_continuations = {"ordered_cycle"},
                         .incremental_continuation_counts = mode == 1u || mode == 2u,
                         .fused_continuation_counts = mode == 2u}};
                    scheduler.register_extension_handler(
                        stream,
                        [&](auto &context,
                            auto &stage) -> unique_ptr<WavefrontCoroSchedulerExtensionHandler> {
                            if (stage.extension->schema() == "luisa.test.resume.ignored") {
                                return nullptr;
                            }
                            return make_unique<ChainHandler>(context, stage, expected.view(),
                                                             diagnostics.view());
                        });
                    for (auto repeat = 0u; repeat < 2u; ++repeat) {
                        std::array<uint, 2u> errors{};
                        vector<uint> host(2u * N, 0u);
                        stream << diagnostics.copy_from(span{errors})
                               << output.copy_from(span{host});
                        stream << scheduler(output).dispatch(N);
                        stream << output.copy_to(span{host}) << diagnostics.copy_to(span{errors})
                               << synchronize();
                        expect(errors[0u] == 0u) << "audit receives rank handler's sorted queue";
                        expect(errors[1u] == 0u) << "continuation receives audit's exact queue";
                        auto visits = 0u;
                        for (auto i = 0u; i < N; ++i) {
                            auto iterations = i % 3u + 1u;
                            expect(host[i] == i + 1u + 5u * iterations);
                            expect(host[N + i] == i * 15u + 8u);
                            visits += iterations;
                        }
                        auto &&stats = scheduler.last_dispatch_stats();
                        expect(stats.generated_count == N);
                        expect(stats.resumed_count == visits + N / 2u);
                        expect(stats.extension_count == 3u * visits);
                        for (auto &&stat : stats.extensions) {
                            if (stat.schema == "luisa.test.resume.ignored") {
                                expect(stat.executed_count == 0u);
                            }
                        }
                    }
                }
            }
        }
    };
    return luisa::test::coro_test::run_tests(argc, argv);
}
