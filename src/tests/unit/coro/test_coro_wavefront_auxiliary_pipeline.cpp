#include "ut/ut.hpp"
#include "coro_test_utils.h"

#include <luisa/coro/schedulers/wavefront.h>
#include <luisa/dsl/sugar.h>

#include <algorithm>
#include <array>
#include <numeric>

using namespace luisa;
using namespace luisa::compute;
using namespace luisa::compute::coro;
using namespace boost::ut;

namespace {

// One pool, three independently scheduled stages. This deliberately uses a
// tiny stable-slot allocator: no renderer, tracing, or floating-point code is
// involved in the admission and scheduling counterexample.
class Pipeline final : public WavefrontCoroAuxiliaryWork<Buffer<uint>, Buffer<uint>> {
private:
    uint _capacity;
    Buffer<uint3> _items;
    Buffer<uint> _tokens;
    Buffer<uint> _free_slots;
    Buffer<uint> _free_count;
    Buffer<uint> _counts;
    Shader1D<uint> _initialize;
    std::array<Shader1D<Buffer<uint>, Buffer<uint>, uint>, 3u> _consume;
    std::array<uint, 3u> _host_counts{};
    std::array<uint, 3u> _zeros{};
    luisa::vector<uint> _host_tokens;
    std::array<WavefrontCoroAuxiliaryProducer, 2u> _producers{{
        {.continuation = "sparse_publish", .max_emitted_per_invocation = 1u},
        {.continuation = "dense_publish", .max_emitted_per_invocation = 1u}}};
    mutable uint _forced_stage{3u};

public:
    uint legacy_dispatches{};
    uint preparations{};
    mutable uint forced_dispatches{};
    std::array<uint64_t, 3u> executed{};

    Pipeline(Device &device, uint capacity)
        : _capacity{capacity},
          _items{device.create_buffer<uint3>(capacity)},
          _tokens{device.create_buffer<uint>(capacity)},
          _free_slots{device.create_buffer<uint>(capacity)},
          _free_count{device.create_buffer<uint>(1u)},
          _counts{device.create_buffer<uint>(3u)},
          _host_tokens(capacity) {
        Kernel1D initialize = [this](UInt capacity) {
            auto x = dispatch_x();
            $if(x < capacity) {
                _tokens->write(x, 0u);
                _free_slots->write(x, x);
            };
        };
        _initialize = device.compile(initialize);
        for (auto stage = 0u; stage < 3u; ++stage) {
            Kernel1D consume = [this, stage](BufferUInt visits, BufferUInt diagnostics,
                                             UInt capacity) {
                set_block_size(32u);
                auto slot = dispatch_x();
                $if(slot >= capacity) { $return(); };
                $if(_tokens->read(slot) == stage + 1u) {
                    auto item = _items->read(slot);
                    visits.atomic(item.x * 3u + stage).fetch_add(1u);
                    UInt next = 0u;
                    if (stage == 0u) {
                        // A disjoint bypass edge and a two-visit loop.
                        next = select(2u, 3u, item.x % 3u == 0u);
                    } else if (stage == 1u) {
                        next = select(3u, 2u, item.z != 0u);
                        item.z = 0u;
                        _items->write(slot, item);
                    } else {
                        // The payload must survive every stage and coexist
                        // with younger items published by the same main path.
                        $if(item.y != (item.x ^ 0x9e3779b9u)) {
                            diagnostics.atomic(0u).fetch_add(1u);
                        };
                        auto free_index = _free_count->atomic(0u).fetch_add(1u);
                        $if(free_index < capacity) { _free_slots->write(free_index, slot); }
                        $else { diagnostics.atomic(0u).fetch_add(1u); };
                    }
                    _tokens->write(slot, next);
                    _counts->atomic(stage).fetch_sub(1u);
                    $if(next != 0u) { _counts->atomic(next - 1u).fetch_add(1u); };
                };
            };
            _consume[stage] = device.compile(consume);
        }
    }

    void publish(UInt id, UInt stage, const BufferUInt &diagnostics) const noexcept {
        auto free_count = _free_count->atomic(0u).fetch_sub(1u);
        $if((free_count != 0u) & (free_count <= _capacity)) {
            auto slot = _free_slots->read(free_count - 1u);
            _items->write(slot, make_uint3(id, id ^ 0x9e3779b9u, 1u));
            _tokens->write(slot, stage + 1u);
            _counts->atomic(stage).fetch_add(1u);
        }
        $else { diagnostics.atomic(0u).fetch_add(1u); };
    }

    void observe_independent_main(const BufferUInt &diagnostics) const noexcept {
        $if(_free_count->read(0u) != _capacity) {
            diagnostics.atomic(1u).fetch_add(1u);
        };
    }
    [[nodiscard]] string_view name() const noexcept override { return "pipeline"; }
    [[nodiscard]] uint capacity() const noexcept override { return _capacity; }
    [[nodiscard]] span<const WavefrontCoroAuxiliaryProducer> producers() const noexcept override {
        return _producers;
    }
    void reset(Stream &stream) noexcept override {
        _host_counts = {};
        legacy_dispatches = preparations = forced_dispatches = 0u;
        executed = {};
        _forced_stage = 3u;
        stream << _initialize(_capacity).dispatch(_capacity)
               << _free_count.copy_from(span{&_capacity, 1u})
               << _counts.copy_from(span{_zeros});
    }
    void enqueue_count_readback(Stream &stream) noexcept override {
        stream << _counts.copy_to(span{_host_counts})
               << _tokens.copy_to(span{_host_tokens});
    }
    [[nodiscard]] uint host_count() const noexcept override {
        for (auto stage = 0u; stage < 3u; ++stage) {
            LUISA_ASSERT(_host_counts[stage] ==
                             std::count(_host_tokens.begin(), _host_tokens.end(), stage + 1u),
                         "Maintained stage count must equal live slot ownership.");
        }
        return std::accumulate(_host_counts.begin(), _host_counts.end(), 0u);
    }
    [[nodiscard]] uint stage_count() const noexcept override { return 3u; }
    [[nodiscard]] string_view stage_name(uint stage) const noexcept override {
        constexpr string_view names[]{"prepare", "iterate", "finish"};
        return names[stage];
    }
    [[nodiscard]] uint stage_host_count(uint stage) const noexcept override {
        return _host_counts[stage];
    }
    [[nodiscard]] uint admission_stage() const noexcept override {
        ++forced_dispatches;
        _forced_stage = WavefrontCoroAuxiliaryWork::admission_stage();
        return _forced_stage;
    }
    void prepare_for_producer(Stream &, uint required) noexcept override {
        LUISA_ASSERT(required <= _capacity - host_count(),
                     "All in-flight stages must participate in admission.");
        ++preparations;
    }
    void dispatch_stage(uint stage, Stream &stream, BufferView<uint> visits,
                        BufferView<uint> diagnostics) noexcept override {
        LUISA_ASSERT(stage < 3u && _host_counts[stage] != 0u, "Empty selected stage.");
        LUISA_ASSERT(_forced_stage == 3u || _forced_stage == stage,
                     "Blocked producers must honor stage drain priority.");
        _forced_stage = 3u;
        executed[stage] += _host_counts[stage];
        stream << _consume[stage](visits, diagnostics, _capacity).dispatch(_capacity);
    }
    // Baseline compatibility lets the regression run to completion before the
    // scheduler understands stages. Its aggregate scheduling is then detected
    // by assertions, rather than by a hang or an intentional device fault.
    void dispatch(Stream &stream, BufferView<uint> visits,
                  BufferView<uint> diagnostics) noexcept override {
        ++legacy_dispatches;
        dispatch_stage(WavefrontCoroAuxiliaryWork::admission_stage(), stream, visits, diagnostics);
    }
};

}// namespace

int main(int argc, char *argv[]) {
    auto options = luisa::test::coro_test::parse_options(argc, argv);
    "wavefront_multi_stage_shared_capacity"_test = [options] {
        constexpr uint N = 67u;
        constexpr uint capacity = 8u;
        auto dc = luisa::test::coro_test::create_device(options);
        auto &device = dc.device;
        auto stream = device.create_stream();
        for (auto soa : {false, true}) {
            for (auto incremental : {false, true}) {
                auto visits = device.create_buffer<uint>(N * 3u);
                auto diagnostics = device.create_buffer<uint>(N + 2u);
                auto work = luisa::make_shared<Pipeline>(device, capacity);
                auto coro = Coroutine<void(Buffer<uint>, Buffer<uint>)>(
                    [work, capacity](BufferUInt, BufferUInt diagnostics) {
                        auto id = dispatch_x();
                        $suspend("sparse_publish");
                        $if(id % capacity == 0u) { work->publish(id, 2u, diagnostics); };
                        $suspend("dense_publish");
                        work->publish(id, 0u, diagnostics);
                        $suspend("main_after_publish");
                        work->observe_independent_main(diagnostics);
                        diagnostics.atomic(id + 2u).fetch_add(1u);
                    });
                WavefrontCoroScheduler<Buffer<uint>, Buffer<uint>> scheduler{
                    device, coro,
                    {.thread_count = capacity,
                     .global_memory_soa = soa,
                     .gather_by_sorting = false,
                     .frame_buffer_compaction = true,
                     .report_stats = true,
                     .execution_block_size = 32u,
                     .largest_continuation_first = true,
                     .incremental_continuation_counts = incremental}};
                scheduler.register_auxiliary_work(work);
                // Reuse also checks reset after a completely drained dispatch.
                for (auto repeat = 0u; repeat < 2u; ++repeat) {
                    luisa::vector<uint> host_visits(N * 3u, 0u);
                    luisa::vector<uint> host_diagnostics(N + 2u, 0u);
                    stream << visits.copy_from(span{host_visits})
                           << diagnostics.copy_from(span{host_diagnostics});
                    scheduler(visits, diagnostics).dispatch(N)(stream);
                    stream << visits.copy_to(span{host_visits})
                           << diagnostics.copy_to(span{host_diagnostics})
                           << synchronize();
                    expect(host_diagnostics[0] == 0u) << "no overflow or overwritten payload";
                    expect(host_diagnostics[1] != 0u) << "main can resume with live side work";
                    std::array<uint64_t, 3u> expected{};
                    for (auto i = 0u; i < N; ++i) {
                        const std::array<uint, 3u> counts{
                            1u, i % 3u == 0u ? 0u : 2u, 1u + uint(i % capacity == 0u)};
                        expect(host_diagnostics[i + 2u] == 1u);
                        for (auto stage = 0u; stage < 3u; ++stage) {
                            expect(host_visits[i * 3u + stage] == counts[stage]);
                            expected[stage] += counts[stage];
                        }
                    }
                    expect(work->legacy_dispatches == 0u) << "scheduler must select individual stages";
                    expect(work->preparations != 0u);
                    expect(work->forced_dispatches != 0u);
                    expect(work->host_count() == 0u);
                    auto &&stats = scheduler.last_dispatch_stats();
                    expect(stats.auxiliary_work.size() == 3u);
                    if (stats.auxiliary_work.size() == 3u) {
                        for (auto stage = 0u; stage < 3u; ++stage) {
                            auto &&stat = stats.auxiliary_work[stage];
                            expect(stat.name == work->stage_name(stage));
                            expect(stat.executed_count == expected[stage]);
                            expect(stat.executed_count == work->executed[stage]);
                        }
                    }
                }
            }
        }
    };
    return luisa::test::coro_test::run_tests(argc, argv);
}
