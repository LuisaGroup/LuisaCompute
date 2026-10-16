// Guarded slot projections are not flat binding-equivalence certificates.
// Use verified compiler metadata; never clear the writable carrier effects
// merely to make a guarded annotation eligible for before-resume batching.
#include "ut/ut.hpp"
#include <algorithm>
#include <array>

#include <luisa/core/logging.h>
#include <luisa/coro/schedulers/wavefront_extension_batch.h>
#include <luisa/xir/builder.h>
#include <luisa/xir/function.h>
#include <luisa/xir/module.h>
#include <luisa/xir/passes/coro_cfg_distill.h>
#include <luisa/xir/passes/coro_materialize.h>
#include <luisa/xir/passes/coro_split.h>
#include <luisa/xir/verifier.h>

using namespace luisa;
using namespace luisa::compute;
using namespace luisa::compute::coro;
using namespace luisa::compute::xir;
using namespace boost::ut;
using namespace boost::ut::literals;

namespace {
void require(bool condition, const char *message) noexcept {
    expect(condition) << message;
    LUISA_ASSERT(condition, "{}", message);
}
template<typename A, typename B>
bool equal_slots(A a, B b) { return std::equal(a.begin(), a.end(), b.begin(), b.end()); }

class Policy final : public WavefrontCoroSchedulerExtensionHandler {
public:
    string_view name() const noexcept override { return "guarded-projection-metadata"; }
    WavefrontCoroExtensionExecution execution() const noexcept override {
        return WavefrontCoroExtensionExecution::before_resume;
    }
    string_view batching_identity() const noexcept override { return "guarded-projection-test-v1"; }
};

struct GraphFixture {
    // Keep every callable owned by the graph's metadata alive. Declaration
    // order also destroys graph before its module.
    luisa::unique_ptr<Module> module;
    CoroGraph graph;
};

GraphFixture make_graph(bool projected, bool swapped) {
    auto module = luisa::make_unique<Module>();
    auto *kernel = module->create_kernel();
    auto *entry = kernel->create_body_block();
    auto *resume = kernel->create_basic_block();
    XIRBuilder b;
    b.set_insertion_point(entry);
    auto *seed = b.static_cast_if_necessary(Type::of<uint>(), b.clock());
    auto *one = module->create_constant_one(Type::of<uint>());
    auto *zero = module->create_constant_zero(Type::of<uint>());
    auto *other = b.call(Type::of<uint>(), ArithmeticOp::BINARY_ADD, {seed, one});
    auto *bit = b.call(Type::of<uint>(), ArithmeticOp::BINARY_BIT_AND, {seed, one});
    auto *condition = b.call(Type::of<bool>(), ArithmeticOp::BINARY_EQUAL, {bit, zero});
    auto *inverse = b.call(Type::of<bool>(), ArithmeticOp::BINARY_NOT_EQUAL, {bit, zero});
    constexpr auto lifetime = CoroSuspendBindingLifetime::resumed;
    CoroSuspendBinding logical{"key", CoroSuspendBindingAccess::read, lifetime, 0u};
    auto annotation = make_coro_suspend_annotation_data(
        "test.guarded-read-metadata", 1u, CoroSuspendFallback::reject, {logical}, {});
    vector<CoroSuspendExtensionPtr> extensions;
    vector<Value *> values;
    AllocaInst *a = nullptr;
    AllocaInst *c = nullptr;
    if (projected) {
        a = b.alloca_local(Type::of<uint>());
        c = b.alloca_local(Type::of<uint>());
        b.store(a, seed);
        b.store(c, other);
        // Logical reads still use verifier-required read/write candidates.
        // The two fixtures only change which guard selects each candidate.
        CoroSuspendBindingProjection projection{
            logical, {{0u, swapped ? 3u : 1u}, {2u, swapped ? 1u : 3u}}};
        extensions.emplace_back(make_coro_suspend_projected_extension(
            std::move(annotation),
            {{"candidate_a", CoroSuspendBindingAccess::read_write, lifetime, 0u},
             {"condition", CoroSuspendBindingAccess::read, lifetime, 1u},
             {"candidate_b", CoroSuspendBindingAccess::read_write, lifetime, 2u},
             {"inverse", CoroSuspendBindingAccess::read, lifetime, 3u}},
            {projection}));
        values = {a, condition, c, inverse};
    } else {
        extensions.emplace_back(std::move(annotation));
        values = {seed};
    }
    b.coro_suspend(5u, "guarded-boundary", nullptr, {}, {}, std::move(extensions), values);
    b.set_insertion_point(resume);
    b.coro_resume(5u, nullptr);
    if (projected) {
        static_cast<void>(b.load(Type::of<uint>(), a));
        static_cast<void>(b.load(Type::of<uint>(), c));
    }
    b.return_void();
    require(xir_verify_module(module.get()).succeeded(), "original XIR verifies");
    auto cfg = coro_cfg_distill_pass_run_on_function(kernel);
    require(cfg.succeeded(), "distill succeeds");
    auto split = coro_split_pass_run_on_module_with_cfg_and_frame_info(module.get(), cfg, nullptr);
    auto materialized = coro_materialize_pass_run_on_module_with_cfg(module.get(), cfg, split);
    auto graph = CoroGraph::from_module(*module, materialized, cfg, split);
    require(graph.node_count() == 2u && graph.boundary_count() == 1u,
            "one real suspend and one real resume");
    return {std::move(module), std::move(graph)};
}

bool self_compatible(const CoroGraph::Boundary &boundary) {
    Policy policy;
    WavefrontCoroExtensionStage stage{
        .queue_index = 2u, .boundary = &boundary,
        .extension = boundary.extensions.at(0u).get(), .dataflow = &boundary.stages.at(0u)};
    using Entry = luisa::compute::coro::detail::WavefrontCoroResumeBatchEntry;
    std::array<Entry, 1u> entries{{{&stage, &policy}}};
    return luisa::compute::coro::detail::wavefront_coro_resume_batch_compatible(
        span<const Entry>{entries}, span<const Entry>{entries});
}

void check_guarded(const CoroGraph::Boundary &boundary) {
    require(boundary.bindings.size() == 1u, "compiler carrier bindings remain private");
    const auto &binding = boundary.bindings.at(0u);
    expect(binding.access() == CoroSuspendBindingAccess::read) << "logical schema is read-only";
    expect(binding.materialized()) << "real guarded binding is materialized";
    expect(binding.pieces().empty()) << "guarded binding has no flat pieces";
    expect(!binding.use_slots().empty()) << "genuine carrier uses";
    expect(!binding.def_slots().empty()) << "writable candidate effects retained";
    expect(!binding.reconstruct_slots().empty()) << "genuine reconstruction plan";
    expect(!boundary.stages.at(0u).def.slots.empty()) << "genuine writable stage certificate";
    // This is a conservative-negative regression, not a newly reproduced
    // unsound merge: the writable stage certificate also forbids batching.
    expect(!self_compatible(boundary)) << "guarded access is never batched";
}
}// namespace

void reg_coro_wavefront_resume_batching_guarded() {
    "resume_batch_accepts_identical_direct_read_binding"_test = [] {
        auto fixture = make_graph(false, false);
        const auto &flat = fixture.graph.boundary(0u);
        expect(flat.bindings.at(0u).materialized());
        expect(!flat.bindings.at(0u).pieces().empty());
        expect(flat.stages.at(0u).def.slots.empty());
        expect(self_compatible(flat));
    };
    "resume_batch_rejects_verified_guarded_aliases_with_equal_slot_unions"_test = [] {
        auto guarded = make_graph(true, false);
        auto swapped = make_graph(true, true);
        check_guarded(guarded.graph.boundary(0u));
        check_guarded(swapped.graph.boundary(0u));
        const auto &a = guarded.graph.boundary(0u).bindings.at(0u);
        const auto &b = swapped.graph.boundary(0u).bindings.at(0u);
        expect(equal_slots(a.use_slots(), b.use_slots())) << "same carrier use union";
        expect(equal_slots(a.def_slots(), b.def_slots())) << "same carrier def union";
        expect(equal_slots(a.reconstruct_slots(), b.reconstruct_slots())) << "same reconstruction union";
    };
}

int main(int, char **) {
    reg_coro_wavefront_resume_batching_guarded();
    return 0;
}
