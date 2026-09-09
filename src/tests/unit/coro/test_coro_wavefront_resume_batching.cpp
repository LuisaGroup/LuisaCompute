#include "ut/ut.hpp"
#include "coro_test_utils.h"

#include <luisa/coro/schedulers/wavefront.h>
#include <luisa/coro/schedulers/wavefront_extension_batch.h>
#include <luisa/dsl/sugar.h>

#include <array>

using namespace luisa;
using namespace luisa::compute;
using namespace luisa::compute::coro;
using namespace boost::ut;

namespace {

struct Observations {
    vector<uint> ranked_counts;
    uint prefix_visits{};
    bool capture_first_rank_slots{};
    vector<uint> first_rank_slots;
};

// Tiny deterministic rank, not a second production sorting implementation.
// Interleaved keys from two predecessors require a genuinely joint queue to
// produce the complete 0..7 permutation in one target launch.
class Rank final : public WavefrontCoroSchedulerExtensionHandler {
  private:
    string _identity;
    shared_ptr<Observations> _observations;
    Buffer<uint> _ordered;
    Shader1D<ByteBuffer, Buffer<uint>, uint, uint> _shader;

  public:
    Rank(WavefrontCoroExtensionPrepareContext &context,
         const WavefrontCoroExtensionStage &stage, string identity,
         shared_ptr<Observations> observations)
        : _identity{std::move(identity)}, _observations{std::move(observations)},
          _ordered{context.device.create_buffer<uint>(context.frame_capacity)} {
        auto *desc = &context.frame_desc;
        auto *key = &stage.binding("key");
        auto fields = stage.dataflow->reconstruct_slots;
        Kernel1D rank = [this, desc, key, fields, layout = context.frame_layout,
                         soa = context.global_memory_soa](ByteBufferVar storage,
                                                          BufferUInt indices, UInt capacity,
                                                          UInt count) {
            auto x = dispatch_x();
            $if(x >= count) { $return(); };
            auto load_key = [&](UInt slot) {
                auto frame = CoroFrame::create(desc);
                coro_frame_load_into(frame, storage, indices.read(slot), capacity, layout,
                                     soa, span{fields}, false, false);
                return key->read<uint>(frame);
            };
            auto value = load_key(x);
            UInt position = 0u;
            $for(j, count) {
                auto other = load_key(j);
                position += ((other < value) | ((other == value) & (j < x))).cast<uint>();
            };
            _ordered->write(position, indices.read(x));
        };
        _shader = context.device.compile(rank);
    }
    [[nodiscard]] string_view name() const noexcept override { return "test-rank"; }
    [[nodiscard]] WavefrontCoroExtensionExecution execution() const noexcept override {
        return WavefrontCoroExtensionExecution::before_resume;
    }
    [[nodiscard]] string_view batching_identity() const noexcept override { return _identity; }
    [[nodiscard]] BufferView<uint>
    dispatch_queue(const WavefrontCoroExtensionDispatchContext &context) noexcept override {
        _observations->ranked_counts.emplace_back(context.frame_count);
        if (_observations->capture_first_rank_slots && _observations->first_rank_slots.empty()) {
            _observations->first_rank_slots.resize(context.frame_count);
            context.stream << context.frame_indices.copy_to(span{_observations->first_rank_slots})
                           << synchronize();
        }
        context.stream << _shader(context.frame_buffer, context.frame_indices,
                                  context.frame_capacity, context.frame_count)
                              .dispatch(context.frame_count);
        return _ordered.view().subview(0u, context.frame_count);
    }
};

class AddPrefix final : public WavefrontCoroSchedulerExtensionHandler {
  private:
    shared_ptr<Observations> _observations;
    Shader1D<ByteBuffer, Buffer<uint>, uint, uint> _shader;

  public:
    AddPrefix(WavefrontCoroExtensionPrepareContext &context,
              const WavefrontCoroExtensionStage &stage, shared_ptr<Observations> observations)
        : _observations{std::move(observations)} {
        auto *desc = &context.frame_desc;
        auto *value = &stage.binding("state");
        auto fields = stage.dataflow->reconstruct_slots;
        auto writes = stage.dataflow->required_def.slots;
        Kernel1D add = [desc, value, fields, writes, layout = context.frame_layout,
                        soa = context.global_memory_soa](ByteBufferVar storage,
                                                         BufferUInt indices, UInt capacity,
                                                         UInt count) {
            auto x = dispatch_x();
            $if(x >= count) { $return(); };
            auto index = indices.read(x);
            auto frame = CoroFrame::create(desc);
            coro_frame_load_into(frame, storage, index, capacity, layout, soa,
                                 span{fields}, false, false);
            value->write<uint>(frame, value->read<uint>(frame) + 5u);
            coro_frame_store(storage, index, capacity, frame, layout, soa,
                             span{writes}, false, false);
        };
        _shader = context.device.compile(add);
    }
    [[nodiscard]] string_view name() const noexcept override { return "test-add-prefix"; }
    void dispatch(const WavefrontCoroExtensionDispatchContext &context) noexcept override {
        _observations->prefix_visits += context.frame_count;
        context.stream << _shader(context.frame_buffer, context.frame_indices,
                                  context.frame_capacity, context.frame_count)
                              .dispatch(context.frame_count);
    }
};

enum class Permission { none, compatible, different, one_empty };

struct CountMode {
    bool incremental;
    bool fused;
    bool compact;
};

constexpr std::array count_modes{
    CountMode{false, false, false}, // Full queue snapshot/count/gather.
    CountMode{true, false, true},   // Separate publication, compact policy.
    CountMode{true, true, false},   // Fused publication, no relocation.
    CountMode{true, true, true}};   // Fused publication, compact policy.

// Public CoroGraph records can be copied as metadata fixtures without mutating
// the source Coroutine or inventing its private typed slot projections.
CoroGraph::Boundary copy_boundary(const CoroGraph::Boundary &source) {
    vector<CoroSuspendExtensionPtr> extensions;
    for (auto &&extension : source.extensions) { extensions.emplace_back(extension->clone()); }
    return {.index = source.index, .from_index = source.from_index,
            .to_index = source.to_index, .token = source.token,
            .extensions = std::move(extensions), .bindings = source.bindings,
            .source_store = source.source_store, .target_live = source.target_live,
            .stages = source.stages};
}

class Policy final : public WavefrontCoroSchedulerExtensionHandler {
  private:
    string_view _identity;
    WavefrontCoroExtensionExecution _execution;

  public:
    explicit Policy(string_view identity,
                    WavefrontCoroExtensionExecution execution =
                        WavefrontCoroExtensionExecution::before_resume)
        : _identity{identity}, _execution{execution} {}
    [[nodiscard]] string_view name() const noexcept override { return "metadata-policy"; }
    [[nodiscard]] string_view batching_identity() const noexcept override { return _identity; }
    [[nodiscard]] WavefrontCoroExtensionExecution execution() const noexcept override { return _execution; }
};

bool compatible(const CoroGraph::Boundary &a, const CoroGraph::Boundary &b,
                const Policy &ah, const Policy &bh, bool reverse = false) {
    std::array<WavefrontCoroExtensionStage, 2u> as, bs;
    for (auto i = 0u; i < 2u; ++i) {
        as[i] = {.queue_index = i + 2u, .boundary = &a,
                 .extension = a.extensions[i].get(), .dataflow = &a.stages[i]};
        bs[i] = {.queue_index = i + 4u, .boundary = &b,
                 .extension = b.extensions[i].get(), .dataflow = &b.stages[i]};
    }
    using Entry = luisa::compute::coro::detail::WavefrontCoroResumeBatchEntry;
    std::array<Entry, 2u> lhs{{{&as[0], &ah}, {&as[1], &ah}}};
    std::array<Entry, 2u> rhs{{{&bs[reverse ? 1u : 0u], &bh},
                             {&bs[reverse ? 0u : 1u], &bh}}};
    return luisa::compute::coro::detail::wavefront_coro_resume_batch_compatible(
        span<const Entry>{lhs}, span<const Entry>{rhs});
}

void set_threshold(CoroGraph::Boundary &boundary, double threshold) {
    const auto &old = *boundary.extensions[0u];
    vector<CoroSuspendBinding> bindings{old.bindings().begin(), old.bindings().end()};
    vector<CoroSuspendAttribute> attributes{old.attributes().begin(), old.attributes().end()};
    attributes.emplace_back(CoroSuspendAttribute{.name = "threshold", .value = threshold});
    boundary.extensions[0u] = make_coro_suspend_annotation_data(
        string{old.schema()}, old.version(), old.fallback(), std::move(bindings),
        std::move(attributes));
}

void run_case(const luisa::test::coro_test::Options &options, Permission permission,
              uint visits, bool prefix, bool soa,
              CountMode mode = {true, true, false}, bool refill = false) {
    constexpr uint capacity = 15u, target_population = 8u;
    const uint N = capacity + (refill ? 4u : 0u);
    auto dc = luisa::test::coro_test::create_device(options);
    auto &device = dc.device;
    auto stream = device.create_stream();
    auto input = device.create_buffer<uint>(N + 1u);
    auto output = device.create_buffer<uint>(N);
    auto ordered = device.create_buffer<uint>(target_population);
    auto observations = make_shared<Observations>();
    observations->capture_first_rank_slots = refill;
    Coroutine<void(Buffer<uint>, Buffer<uint>, Buffer<uint>)> coro{
        [prefix, N, refill](BufferUInt inputs, BufferUInt result, BufferUInt order) {
            auto route = inputs.read(dispatch_x());
            auto limit = inputs.read(N);
            UInt state = route * 7u + 3u;
            $if((route >= 4u) & (route < 12u)) { $suspend("prelude"); };
            $if(route < 8u) {
                // Entry contributes evens, the prelude contributes odds.
                auto key = select(route * 2u, (route - 4u) * 2u + 1u, route >= 4u);
                UInt iteration = 0u;
                $while(iteration < limit) {
                    if (prefix) {
                        $suspend("target",
                                 coro_stage("luisa.test.batch.add").read_write("state", state),
                                 coro_annotation("luisa.test.batch.rank").read("key", key));
                    } else {
                        $suspend("target",
                                 coro_annotation("luisa.test.batch.rank").read("key", key));
                    }
                    order.write(thread_x(), key);
                    iteration += 1u;
                };
                state += iteration;
            }
            $else {
                if (refill) {
                    // Prelude paths 8..11 terminate, opening holes while the
                    // two compatible target entries are still queued.
                    $if(route >= 12u) { $suspend("rival"); };
                } else { $suspend("rival"); }
            };
            result.write(route, state);
        }};
    const auto *target = coro.graph().node_by_name("target");
    expect(target != nullptr);
    WavefrontCoroScheduler<Buffer<uint>, Buffer<uint>, Buffer<uint>> scheduler{
        device,
        coro,
        {.thread_count = capacity,
         .global_memory_soa = soa,
         .gather_by_sorting = false,
         .frame_buffer_compaction = mode.compact,
         .report_stats = true,
         .execution_block_size = 32u,
         .largest_continuation_first = true,
         .refill_threshold = refill ? capacity : 0u,
         .incremental_continuation_counts = mode.incremental,
         .fused_continuation_counts = mode.fused}};
    scheduler.register_extension_handler(
        stream,
        [&](auto &context, auto &stage) -> unique_ptr<WavefrontCoroSchedulerExtensionHandler> {
            if (stage.extension->schema() == "luisa.test.batch.add") {
                return make_unique<AddPrefix>(context, stage, observations);
            }
            string identity;
            switch (permission) {
                case Permission::none: break;
                case Permission::compatible: identity = "luisa.test.rank.uint/v1"; break;
                case Permission::different:
                    identity = stage.boundary->from_index == coro.graph().entry_index()
                                   ? "luisa.test.rank.entry/v1" : "luisa.test.rank.other/v1";
                    break;
                case Permission::one_empty:
                    if (stage.boundary->from_index == coro.graph().entry_index()) {
                        identity = "luisa.test.rank.uint/v1";
                    }
                    break;
            }
            return make_unique<Rank>(context, stage, std::move(identity), observations);
        });
    vector<uint> inputs(N + 1u), actual(N, ~0u), permutation(target_population, ~0u);
    for (auto i = 0u; i < N; ++i) {
        inputs[i] = i < capacity ? capacity - 1u - i : N + capacity - 1u - i;
    }
    inputs[N] = visits;
    stream << input.copy_from(span{inputs}) << output.copy_from(span{actual})
           << ordered.copy_from(span{permutation});
    stream << scheduler(input, output, ordered).dispatch(N);
    stream << output.copy_to(span{actual}) << ordered.copy_to(span{permutation})
           << synchronize();
    for (auto i = 0u; i < N; ++i) {
        auto expected = i * 7u + 3u + (i < 8u ? visits * (prefix ? 6u : 1u) : 0u);
        expect(actual[i] == expected) << "every path retains its native continuation state";
    }
    auto total_ranked = 0u;
    for (auto count : observations->ranked_counts) { total_ranked += count; }
    expect(total_ranked == target_population * visits);
    expect(observations->prefix_visits == (prefix ? target_population * visits : 0u));
    expect(scheduler.last_dispatch_stats().continuations[target->index].executed_count ==
           target_population * visits);
    if (refill) {
        const auto &stats = scheduler.last_dispatch_stats();
        expect(stats.generated_count == N);
        expect(stats.continuations[coro.graph().entry_index()].dispatch_count == 2u);
        expect(stats.compact_scan_count > 0u) << "the relocation kernel must actually execute";
        expect(observations->first_rank_slots.size() == target_population);
        // Original physical slots 11..14 contain target IDs3..0. They move
        // into dead prelude slots3..6; the other target IDs7..4 stay at7..10.
        // Observe the actual Handler queue, not merely the compaction flag.
        std::array<bool, target_population> seen{};
        for (auto slot : observations->first_rank_slots) {
            expect(slot >= 3u && slot < 11u) << "joint target members survive refill relocation";
            if (slot >= 3u && slot < 11u) {
                expect(!seen[slot - 3u]) << "relocated alias queue contains each member once";
                seen[slot - 3u] = true;
            }
        }
    }
    if (!prefix) {
        if (permission == Permission::compatible) {
            expect(observations->ranked_counts == vector<uint>(visits, target_population))
                << "compatible source queues must receive one joint rank operation per visit";
            expect(scheduler.last_dispatch_stats().continuations[target->index].dispatch_count == visits)
                << "joint sorting must feed one target resume, not separate sorted subsets";
            for (auto i = 0u; i < target_population; ++i) {
                expect(permutation[i] == i) << "one resume receives the complete joint permutation";
            }
        } else {
            expect(visits == 1u);
            expect(observations->ranked_counts == vector<uint>{4u, 4u})
                << "absent or distinct permission preserves independently invoked source handlers";
            expect(scheduler.last_dispatch_stats().continuations[target->index].dispatch_count == 2u);
        }
    }
}

} // namespace

int main(int argc, char *argv[]) {
    auto options = luisa::test::coro_test::parse_options(argc, argv);
    "resume_batching_rejects_incompatible_metadata"_test = [] {
        Coroutine<void(Buffer<uint>)> source{[](BufferUInt input) {
            auto id = dispatch_x();
            auto left = input.read(id * 2u);
            auto right = input.read(id * 2u + 1u);
            $suspend("target",
                     coro_annotation("luisa.test.batch.rank").read("key", left).read("other", right),
                     coro_annotation("luisa.test.batch.audit").read("key", left));
            input.write(id * 2u, left + right);
        }};
        expect(source.graph().boundary_count() == 1u);
        if (source.graph().boundary_count() != 1u) { return; }
        const auto &original = source.graph().boundary(0u);
        expect(original.extensions.size() == 2u && original.stages.size() == 2u);
        if (original.extensions.size() != 2u || original.stages.size() != 2u) { return; }
        Policy enabled{"luisa.test.compatible/v1"}, different{"luisa.test.other/v1"}, disabled{""};
        Policy independent{"luisa.test.compatible/v1", WavefrontCoroExtensionExecution::stage};
        auto clone = copy_boundary(original);
        expect(compatible(original, clone, enabled, enabled));
        expect(!compatible(original, clone, enabled, disabled));
        expect(!compatible(original, clone, enabled, different));
        expect(!compatible(original, clone, enabled, independent));
        expect(!compatible(original, clone, enabled, enabled, true))
            << "a suffix is an ordered operation chain, not a set of schemas";

        clone.to_index += 1u;
        expect(!compatible(original, clone, enabled, enabled));
        clone = copy_boundary(original);
        set_threshold(clone, 1.0);
        expect(!compatible(original, clone, enabled, enabled));
        auto positive_zero = copy_boundary(original), negative_zero = copy_boundary(original);
        set_threshold(positive_zero, 0.0);
        set_threshold(negative_zero, -0.0);
        expect(!compatible(positive_zero, negative_zero, enabled, enabled))
            << "signed zero in captured metadata must not be collapsed by numeric equality";
        auto same_attributes = copy_boundary(positive_zero);
        expect(compatible(positive_zero, same_attributes, enabled, enabled));

        clone = copy_boundary(original);
        const auto key = original.extensions[0u]->bindings()[0u].index;
        const auto other = original.extensions[0u]->bindings()[1u].index;
        expect(original.bindings[key].pieces().size() == 1u &&
               original.bindings[other].pieces().size() == 1u);
        if (original.bindings[key].pieces().size() == 1u &&
            original.bindings[other].pieces().size() == 1u) {
            expect(original.bindings[key].pieces()[0u].field_index !=
                   original.bindings[other].pieces()[0u].field_index);
        }
        clone.bindings[key] = clone.bindings[other];
        expect(!compatible(original, clone, enabled, enabled))
            << "same schema and logical type do not replace a distinct physical field";

        clone = copy_boundary(original);
        expect(!clone.stages[0u].reconstruct_slots.empty());
        if (!clone.stages[0u].reconstruct_slots.empty()) {
            clone.stages[0u].reconstruct_slots.pop_back();
            expect(!compatible(original, clone, enabled, enabled));
        }
        clone = copy_boundary(original);
        expect(!clone.stages[0u].live_in.slots.empty());
        if (!clone.stages[0u].live_in.slots.empty()) {
            clone.stages[0u].live_in.slots.pop_back();
            expect(!compatible(original, clone, enabled, enabled))
                << "queue resident/relocation certificates must agree";
        }
    };
    "resume_batching_opt_in_joint_permutation"_test = [options] {
        for (auto soa : {false, true}) {
            for (auto mode : count_modes) {
                run_case(options, Permission::compatible, 1u, false, soa, mode);
            }
        }
    };
    "resume_batching_default_and_distinct_identity_keep_members"_test = [options] {
        for (auto permission : {Permission::none, Permission::different, Permission::one_empty}) {
            run_case(options, permission, 1u, false, true);
        }
    };
    "resume_batching_self_edge_conserves_membership"_test = [options] {
        for (auto mode : count_modes) {
            run_case(options, Permission::compatible, 2u, false, true, mode);
        }
    };
    "resume_batching_relocates_joint_alias_members_at_refill"_test = [options] {
        // N19 exceeds capacity15. The first producer frees four interior
        // slots, forcing pending aliased target members to move before the
        // next four inputs are generated. Their original IDs, keys and live
        // payloads must all remain valid through the one joint resume.
        for (auto soa : {false, true}) {
            for (auto fused : {false, true}) {
                run_case(options, Permission::compatible, 1u, false, soa,
                         CountMode{true, fused, true}, true);
            }
        }
    };
    "resume_batching_preserves_semantic_prefix"_test = [options] {
        run_case(options, Permission::compatible, 2u, true, true);
    };
    return luisa::test::coro_test::run_tests(argc, argv);
}
