#include "ut/ut.hpp"

#include "coro_test_utils.h"

#include <luisa/coro/schedulers/wavefront.h>
#include <luisa/dsl/sugar.h>

#include <array>

using namespace luisa;
using namespace luisa::compute;
using namespace luisa::compute::coro;
using namespace boost::ut;

namespace {

class ProducerAnnotation final : public WavefrontCoroSchedulerExtensionHandler {
public:
    [[nodiscard]] string_view name() const noexcept override { return "producer annotation"; }
    [[nodiscard]] WavefrontCoroExtensionExecution execution() const noexcept override {
        return WavefrontCoroExtensionExecution::before_resume;
    }
    void dispatch(const WavefrontCoroExtensionDispatchContext &) noexcept override {}
};

// Two main frames publish three times into a four-slot append pool. Each
// consumer releases two slots. The client deliberately reclaims its append
// extent only when empty: dead holes are not yet available producer storage.
// Only allocator counters are needed to witness this scheduler contract.
class AppendPool final : public WavefrontCoroAuxiliaryWork<Buffer<uint>> {
private:
    Buffer<uint> _counts;
    Shader1D<uint> _consume;
    std::array<uint, 2u> _host{}, _zeros{};
    luisa::vector<WavefrontCoroAuxiliaryProducer> _producers{
        {.continuation = "producer", .max_emitted_per_invocation = 1u}};

public:
    uint premature_preparations{};
    uint empty_reclaims{};
    luisa::vector<uint> drain_counts;

    explicit AppendPool(Device &device, bool declare_prelude = false)
        : _counts{device.create_buffer<uint>(2u)} {
        if (declare_prelude) {
            _producers.emplace_back(WavefrontCoroAuxiliaryProducer{
                .continuation = "prelude", .max_emitted_per_invocation = 1u});
        }
        Kernel1D consume = [this](UInt count) {
            set_block_size(32u);
            _counts->atomic(0u).fetch_sub(count);
        };
        _consume = device.compile(consume);
    }
    [[nodiscard]] string_view name() const noexcept override { return "append pool"; }
    [[nodiscard]] uint capacity() const noexcept override { return 4u; }
    [[nodiscard]] span<const WavefrontCoroAuxiliaryProducer> producers() const noexcept override {
        return _producers;
    }
    [[nodiscard]] BufferView<uint> counters() const noexcept { return _counts.view(); }
    void reset(Stream &stream) noexcept override {
        _host = {};
        premature_preparations = empty_reclaims = 0u;
        drain_counts.clear();
        stream << _counts.copy_from(_zeros.data());
    }
    void enqueue_count_readback(Stream &stream) noexcept override {
        stream << _counts.copy_to(_host.data());
    }
    [[nodiscard]] uint host_count() const noexcept override { return _host[0]; }

    void prepare_for_admission(Stream &stream) noexcept override {
        if (_host[0] == 0u && _host[1] != 0u) {
            ++empty_reclaims;
            _host[1] = 0u;
            stream << _counts.view().subview(1u, 1u).copy_from(&_host[1]);
        }
    }
    [[nodiscard]] uint host_available_slots() const noexcept override { return 4u - _host[1]; }
    void prepare_for_producer(Stream &stream, uint required) noexcept override {
        if (required > host_available_slots()) {
            ++premature_preparations;
            // Keep the old-scheduler witness memory-safe, but record that it
            // forced a reclaim the client policy did not make available.
            _host[1] = _host[0];
            stream << _counts.view().subview(1u, 1u).copy_from(&_host[1]);
        }
    }
    void dispatch_stage(uint stage, Stream &stream, BufferView<uint>) noexcept override {
        LUISA_ASSERT(stage == 0u, "Invalid allocator stage.");
        drain_counts.emplace_back(_host[0]);
        stream << _consume(std::min(_host[0], 2u)).dispatch(1u);
    }
    void dispatch(Stream &, BufferView<uint>) noexcept override {
        LUISA_ERROR("Stage-aware allocator must use dispatch_stage.");
    }
};

} // namespace

int main(int argc, char *argv[]) {
    auto options = luisa::test::coro_test::parse_options(argc, argv);
    auto dc = luisa::test::coro_test::create_device(options);
    auto &device = dc.device;
    auto stream = device.create_stream();
    "auxiliary_admission_obeys_materializable_storage"_test = [&] {
        for (auto soa : {false, true}) {
            for (auto annotation : {false, true}) {
                for (auto mode = 0u; mode < 3u; ++mode) {
                    auto pool = luisa::make_shared<AppendPool>(device);
                    auto counts = pool->counters();
                    Coroutine<void(Buffer<uint>)> coroutine{[counts, annotation](BufferUInt visits) {
                        UInt iteration = 0u;
                        $while (iteration < 3u) {
                            if (annotation) {
                                $suspend("producer", coro_annotation("test.admission", 1u).build());
                            } else {
                                $suspend("producer");
                            }
                            counts->atomic(0u).fetch_add(1u);
                            counts->atomic(1u).fetch_add(1u);
                            iteration += 1u;
                        };
                        visits.write(dispatch_x(), iteration);
                    }};
                    WavefrontCoroSchedulerConfig config;
                    config.thread_count = 2u;
                    config.global_memory_soa = soa;
                    config.execution_block_size = 32u;
                    config.largest_continuation_first = true;
                    config.incremental_continuation_counts = mode != 0u;
                    config.fused_continuation_counts = mode == 2u;
                    WavefrontCoroScheduler<Buffer<uint>> scheduler{device, coroutine, config};
                    if (annotation) {
                        scheduler.register_extension_handler(stream, [](auto &, auto &) {
                            return luisa::make_unique<ProducerAnnotation>();
                        });
                    }
                    scheduler.register_auxiliary_work(pool);
                    auto visits = device.create_buffer<uint>(2u);
                    for (auto repeat = 0u; repeat < 2u; ++repeat) {
                        std::array<uint, 2u> actual{};
                        stream << visits.copy_from(actual.data());
                        stream << scheduler(visits).dispatch(2u);
                        stream << visits.copy_to(actual.data()) << synchronize();
                        expect(eq(actual[0], 3u));
                        expect(eq(actual[1], 3u));
                        expect(eq(pool->host_count(), 0u));
                        expect(eq(pool->premature_preparations, 0u));
                        expect(eq(pool->empty_reclaims, 1u));
                        expect(pool->drain_counts == luisa::vector<uint>{4u, 2u, 2u});
                    }
                }
            }
        }
    };
    "auxiliary_admission_uses_aggregate_annotation_population"_test = [&] {
        for (auto mode = 0u; mode < 3u; ++mode) {
            auto pool = luisa::make_shared<AppendPool>(device, true);
            auto counts = pool->counters();
            Coroutine<void(Buffer<uint>)> coroutine{[counts](BufferUInt visits) {
                auto id = dispatch_x();
                $if(id >= 2u) {
                    $suspend("prelude");
                    $if(id == 2u) {
                        counts->atomic(0u).fetch_add(1u);
                        counts->atomic(1u).fetch_add(1u);
                    };
                };
                $suspend("producer", coro_annotation("test.admission").read("id", id));
                counts->atomic(0u).fetch_add(1u);
                counts->atomic(1u).fetch_add(1u);
                visits.write(id, 1u);
            }};
            expect(coroutine.graph().node_by_name("prelude")->index <
                   coroutine.graph().node_by_name("producer")->index)
                << "initial equal populations select the prelude by logical tie priority";
            WavefrontCoroScheduler<Buffer<uint>> scheduler{
                device,
                coroutine,
                {.thread_count = 4u,
                 .gather_by_sorting = false,
                 .frame_buffer_compaction = mode != 1u,
                 .report_stats = true,
                 .execution_block_size = 32u,
                 .largest_continuation_first = true,
                 .incremental_continuation_counts = mode != 0u,
                 .fused_continuation_counts = mode == 2u}};
            scheduler.register_extension_handler(stream, [](auto &, auto &) {
                return luisa::make_unique<ProducerAnnotation>();
            });
            scheduler.register_auxiliary_work(pool);
            auto visits = device.create_buffer<uint>(4u);
            std::array<uint, 4u> actual{};
            stream << visits.copy_from(actual.data());
            stream << scheduler(visits).dispatch(4u);
            stream << visits.copy_to(actual.data()) << synchronize();
            for (auto value : actual) { expect(value == 1u); }
            expect(pool->host_count() == 0u);
            expect(pool->premature_preparations == 0u);
            // Prelude leaves one live item and two source queues with two
            // producer paths each. Aggregate required=4 cannot fit available=3:
            // drain the one old item before either producer batch starts.
            expect(pool->drain_counts == luisa::vector<uint>{1u, 4u, 2u});
            expect(scheduler.last_dispatch_stats().resumed_count == 6u);
        }
    };
    return luisa::test::coro_test::run_tests(argc, argv);
}
