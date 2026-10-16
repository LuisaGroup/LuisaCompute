// Host-only compiler regression candidate. No Context, Device, dispatch,
// renderer, profile, or measured trace participates in these expectations.
// A red result is missing static precision, not an execution miscompile.
#include "ut/ut.hpp"

#include <cstdio>
#include <set>
#include <luisa/dsl/coro_func.h>
#include <luisa/dsl/sugar.h>
#include <luisa/xir/basic_block.h>
#include <luisa/xir/builder.h>
#include <luisa/xir/constant.h>
#include <luisa/xir/function.h>
#include <luisa/xir/instructions/alloca.h>
#include <luisa/xir/instructions/branch.h>
#include <luisa/xir/instructions/clock.h>
#include <luisa/xir/instructions/coro.h>
#include <luisa/xir/instructions/gep.h>
#include <luisa/xir/instructions/load.h>
#include <luisa/xir/instructions/store.h>
#include <luisa/xir/module.h>
#include <luisa/xir/passes/coro_cfg_distill.h>
#include <luisa/xir/passes/coro_split.h>
#include <luisa/xir/verifier.h>

using namespace luisa;
using namespace luisa::compute;
using namespace boost::ut;
using namespace boost::ut::literals;

namespace {
void check_split_token_effects(xir::Module &module, const xir::CoroCfgDistillResult &cfg) {
    auto split = xir::coro_split_pass_run_on_module_with_cfg_and_frame_info(&module, cfg, nullptr);
    expect(split.succeeded());
    if (!split.succeeded()) { return; }
    expect(xir::xir_verify_module(&module).succeeded());
    for (auto &subroutine : split.subroutines) {
        std::set<uint> actual, expected;
        for (const auto &edge : cfg.transition_edges) {
            if (edge.from_scope == subroutine.scope_index) { expected.insert(edge.token); }
        }
        subroutine.callable->definition()->traverse_basic_blocks([&](xir::BasicBlock *block) {
            for (auto *instruction : block->instructions()) {
                if (!instruction->isa<xir::StoreInst>()) { continue; }
                auto *store = static_cast<xir::StoreInst *>(instruction);
                if (!store->variable()->isa<xir::GEPInst>() || !store->value()->isa<xir::Constant>()) { continue; }
                auto *address = static_cast<xir::GEPInst *>(store->variable());
                if (address->base() != subroutine.frame_argument || address->index_count() != 1u ||
                    !address->index(0u)->isa<xir::Constant>() ||
                    static_cast<xir::Constant *>(address->index(0u))->as<uint>() != 6u) { continue; }
                auto token = static_cast<xir::Constant *>(store->value())->as<uint>();
                if (token != ~0u) { actual.insert(token); }
            }
        });
        expect(actual == expected) << "the actual cloned token stores equal the certified transition set";
    }
}

bool has_transition(const xir::CoroCfgDistillResult &cfg, uint from, uint to) {
    for (const auto &edge : cfg.transition_edges) {
        if (cfg.scopes.at(edge.from_scope).trigger_token == from &&
            cfg.scopes.at(edge.to_scope).trigger_token == to) { return true; }
    }
    return false;
}

void check_raw(bool dynamic_entry, bool dynamic_resume) {
    xir::Module module;
    auto *kernel = module.create_kernel();
    auto *argument = kernel->create_value_argument(Type::of<bool>());
    auto *entry = kernel->create_body_block();
    auto *head = kernel->create_basic_block();
    auto *yield_a = kernel->create_basic_block();
    auto *resume_a = kernel->create_basic_block();
    auto *yield_b = kernel->create_basic_block();
    auto *resume_b = kernel->create_basic_block();
    auto *yes = module.create_constant_one(Type::of<bool>());
    auto *no = module.create_constant_zero(Type::of<bool>());
    xir::XIRBuilder b;
    b.set_insertion_point(entry);
    auto *search = b.alloca_local(Type::of<bool>());
    b.store(search, dynamic_entry ? static_cast<xir::Value *>(argument) : yes);
    b.br(head);
    b.set_insertion_point(head);
    b.cond_br(b.load(Type::of<bool>(), search), yield_a, yield_b);
    b.set_insertion_point(yield_a);
    b.coro_suspend(1u, "resolve", nullptr);
    b.set_insertion_point(resume_a);
    b.coro_resume(1u, nullptr);
    b.store(search, dynamic_resume ? static_cast<xir::Value *>(argument) : no);
    b.br(head);
    b.set_insertion_point(yield_b);
    b.coro_suspend(2u, "shade", nullptr);
    b.set_insertion_point(resume_b);
    b.coro_resume(2u, nullptr);
    b.return_void();

    // Six blocks, one raw conditional branch, zero Loop/If/Phi instructions.
    // Structural merge/update block references cannot explain a false edge.
    expect(xir::xir_verify_module(&module).succeeded());
    auto cfg = xir::coro_cfg_distill_pass_run_on_function(kernel);
    expect(cfg.succeeded());
    if (!cfg.succeeded()) { return; }
    std::fprintf(stderr, "raw dynamic_entry=%d dynamic_resume=%d scopes=%zu edges=%zu slots=%zu\n",
                 dynamic_entry, dynamic_resume, cfg.scopes.size(),
                 cfg.transition_edges.size(), cfg.frame_slots.size());
    for (const auto &edge : cfg.transition_edges) {
        std::fprintf(stderr, "  %u -> %u\n",
                     cfg.scopes.at(edge.from_scope).trigger_token,
                     cfg.scopes.at(edge.to_scope).trigger_token);
    }
    expect(has_transition(cfg, 0u, 1u));
    expect(has_transition(cfg, 1u, 2u));
    expect(has_transition(cfg, 0u, 2u) == dynamic_entry)
        << "only a dynamic initial condition can bypass the mandatory first yield";
    expect(has_transition(cfg, 1u, 1u) == dynamic_resume)
        << "a store in the resume must replace facts from the preceding iteration";
    if (!dynamic_entry && !dynamic_resume && !cfg.scopes.front().selected_successors.empty()) {
        auto tampered = cfg;
        auto &proof = tampered.scopes.front().selected_successors.front();
        auto *branch = static_cast<xir::ConditionalBranchInst *>(proof.block->terminator());
        proof.successor = proof.successor == branch->true_block() ? branch->false_block() : branch->true_block();
        auto rejected = xir::coro_split_pass_run_on_module_with_cfg_and_frame_info(&module, tampered, nullptr);
        expect(!rejected.succeeded()) << "a selected-arm certificate is sealed with its scope owner";
    }
    check_split_token_effects(module, cfg);
}

void check_external_write(CoroSuspendBindingLifetime lifetime, bool old_snapshot) {
    xir::Module module;
    auto *kernel = module.create_kernel();
    auto *entry = kernel->create_body_block();
    auto *resume = kernel->create_basic_block();
    auto *yes = kernel->create_basic_block();
    auto *no = kernel->create_basic_block();
    auto *resume_yes = kernel->create_basic_block();
    auto *resume_no = kernel->create_basic_block();
    xir::XIRBuilder b;
    b.set_insertion_point(entry);
    auto *state = b.alloca_local(Type::of<bool>());
    b.store(state, module.create_constant_one(Type::of<bool>()));
    auto *snapshot = b.load(Type::of<bool>(), state);
    vector<CoroSuspendExtensionPtr> extensions;
    extensions.emplace_back(make_coro_suspend_extension_data(
        "test.boolean-write", 1u, CoroSuspendFallback::reject,
        {{"state", CoroSuspendBindingAccess::read_write, lifetime, 0u}}, {}));
    vector<xir::Value *> bindings{state};
    b.coro_suspend(1u, "write", nullptr, {}, {}, std::move(extensions), bindings);
    b.set_insertion_point(resume);
    b.coro_resume(1u, nullptr);
    b.cond_br(old_snapshot ? snapshot : b.load(Type::of<bool>(), state), yes, no);
    b.set_insertion_point(yes);
    b.coro_suspend(2u, "yes", nullptr);
    b.set_insertion_point(no);
    b.coro_suspend(3u, "no", nullptr);
    b.set_insertion_point(resume_yes);
    b.coro_resume(2u, nullptr);
    b.return_void();
    b.set_insertion_point(resume_no);
    b.coro_resume(3u, nullptr);
    b.return_void();
    expect(xir::xir_verify_module(&module).succeeded());
    auto cfg = xir::coro_cfg_distill_pass_run_on_function(kernel);
    expect(cfg.succeeded());
    if (!cfg.succeeded()) { return; }
    expect(has_transition(cfg, 1u, 2u));
    expect(has_transition(cfg, 1u, 3u) == !old_snapshot)
        << "write invalidates the mutable slot, not its prior SSA snapshot";
    check_split_token_effects(module, cfg);
}

void check_ordinary_resume_entry() {
    xir::Module module;
    auto *kernel = module.create_kernel();
    auto *entry = kernel->create_body_block();
    auto *suspend = kernel->create_basic_block();
    auto *resume = kernel->create_basic_block();
    xir::XIRBuilder b;
    b.set_insertion_point(entry);
    auto *discarded = b.clock();
    b.cond_br(module.create_constant_one(Type::of<bool>()), resume, suspend);
    b.set_insertion_point(suspend);
    vector<string> export_names{"discarded"};
    vector<xir::Value *> export_values{discarded};
    b.coro_suspend(1u, "bypass", nullptr, export_names, export_values);
    b.set_insertion_point(resume);
    b.coro_resume(1u, nullptr);
    b.return_void();
    expect(xir::xir_verify_module(&module).succeeded());
    auto cfg = xir::coro_cfg_distill_pass_run_on_function(kernel);
    expect(cfg.succeeded());
    if (!cfg.succeeded()) { return; }
    expect(cfg.scopes.size() == 2u);
    expect(cfg.scopes.front().suspend_points.empty());
    expect(has_transition(cfg, 0u, 1u));
    expect(cfg.frame_values.empty()) << "a value used only by the discarded suspension is not transported";
    check_split_token_effects(module, cfg);
}

void check_reference_escape() {
    xir::Module module;
    auto *callee = module.create_callable(nullptr);
    auto *destination = callee->create_reference_argument(Type::of<bool>());
    auto *condition = callee->create_value_argument(Type::of<bool>());
    auto *callee_entry = callee->create_body_block();
    auto *write = callee->create_basic_block();
    auto *done = callee->create_basic_block();
    xir::XIRBuilder b;
    b.set_insertion_point(callee_entry);
    b.cond_br(condition, write, done);
    b.set_insertion_point(write);
    b.store(destination, module.create_constant_one(Type::of<bool>()));
    b.br(done);
    b.set_insertion_point(done);
    b.return_void();
    auto *kernel = module.create_kernel();
    auto *input = kernel->create_value_argument(Type::of<bool>());
    auto *entry = kernel->create_body_block();
    auto *yes = kernel->create_basic_block();
    auto *no = kernel->create_basic_block();
    auto *resume_yes = kernel->create_basic_block();
    auto *resume_no = kernel->create_basic_block();
    b.set_insertion_point(entry);
    auto *state = b.alloca_local(Type::of<bool>());
    b.store(state, module.create_constant_zero(Type::of<bool>()));
    static_cast<void>(b.call(nullptr, callee, {state, input}));
    b.cond_br(b.load(Type::of<bool>(), state), yes, no);
    b.set_insertion_point(yes);
    b.coro_suspend(1u, "yes", nullptr);
    b.set_insertion_point(no);
    b.coro_suspend(2u, "no", nullptr);
    b.set_insertion_point(resume_yes);
    b.coro_resume(1u, nullptr);
    b.return_void();
    b.set_insertion_point(resume_no);
    b.coro_resume(2u, nullptr);
    b.return_void();
    expect(xir::xir_verify_module(&module).succeeded());
    auto cfg = xir::coro_cfg_distill_pass_run_on_function(kernel);
    expect(cfg.succeeded());
    if (!cfg.succeeded()) { return; }
    expect(has_transition(cfg, 0u, 1u));
    expect(has_transition(cfg, 0u, 2u));
    check_split_token_effects(module, cfg);
}

bool has_named_transition(const coro::CoroGraph &graph, string_view from, string_view to) {
    auto *source = from.empty() ? &graph.node(graph.entry_index()) : graph.node_by_name(from);
    auto *target = graph.node_by_name(to);
    expect(source != nullptr && target != nullptr);
    return source != nullptr && target != nullptr && graph.edge(source->index, target->index) != nullptr;
}

void dump_graph(const coro::CoroGraph &graph) {
    for (const auto &edge : graph.boundaries()) {
        std::fprintf(stderr, "  %s -> %s\n",
                     graph.node(edge.from_index).name.c_str(),
                     graph.node(edge.to_index).name.c_str());
    }
}
}// namespace

int main(int argc, char *argv[]) {
    boost::ut::detail::cfg::parse_arg_with_fallback(argc, const_cast<const char **>(argv));
    "raw_local_condition_has_scope_specific_reachability"_test = [] {
        check_raw(false, false);
    };
    "raw_dynamic_entry_preserves_a_real_bypass"_test = [] {
        check_raw(true, false);
    };
    "raw_dynamic_resume_preserves_a_real_self_edge"_test = [] {
        check_raw(false, true);
    };
    "writable_extensions_preserve_unknown_memory_and_old_snapshots"_test = [] {
        for (auto lifetime : {CoroSuspendBindingLifetime::boundary, CoroSuspendBindingLifetime::queued,
                              CoroSuspendBindingLifetime::resumed}) {
            check_external_write(lifetime, false);
            check_external_write(lifetime, true);
        }
    };
    "ordinary_resume_entry_survives_an_unreachable_matching_suspend"_test = [] {
        check_ordinary_resume_entry();
    };
    "a_conditional_reference_call_is_not_a_no_write_certificate"_test = [] {
        check_reference_escape();
    };
    "diamond_join_retains_both_late_discovered_boolean_values"_test = [] {
        Coroutine<void(bool)> source{[](Bool input) {
            Bool state = false;
            $if(input) { state = true; };
            $suspend("join");
            $if(state) { $suspend("yes"); }
            $else { $suspend("no"); };
        }};
        const auto &graph = source.graph();
        expect(has_named_transition(graph, "join", "yes"));
        expect(has_named_transition(graph, "join", "no"));
    };
    "reexecuted_boolean_definitions_do_not_freeze_the_previous_iteration"_test = [] {
        Coroutine<void()> source{[] {
            Bool state = true;
            UInt iteration = 0u;
            $loop {
                $if(state) { $suspend("yes"); }
                $else { $suspend("no"); };
                state = !state;
                iteration += 1u;
                $if(iteration == 4u) { $break; };
            };
        }};
        const auto &graph = source.graph();
        expect(has_named_transition(graph, "", "yes"));
        expect(!has_named_transition(graph, "", "no"));
        expect(has_named_transition(graph, "yes", "no"));
        expect(has_named_transition(graph, "no", "yes"));
        expect(!has_named_transition(graph, "yes", "yes"));
        expect(!has_named_transition(graph, "no", "no"));
    };
    "dsl_mandatory_inner_yield_is_not_bypassed_on_outer_reentry"_test = [] {
        Coroutine<void()> source{[] {
            UInt iterations = 0u;
            $loop {
                Bool search = true;
                $while(search) {
                    $suspend("resolve");
                    search = false;
                };
                $suspend("shade");
                iterations += 1u;
                $if(iterations == 2u) { $break; };
            };
        }};
        const auto &graph = source.graph();
        dump_graph(graph);
        expect(has_named_transition(graph, "", "resolve"));
        expect(has_named_transition(graph, "resolve", "shade"));
        expect(has_named_transition(graph, "shade", "resolve"));
        expect(!has_named_transition(graph, "", "shade"));
        expect(!has_named_transition(graph, "shade", "shade"));
        expect(!has_named_transition(graph, "resolve", "resolve"));
    };
    "dsl_lamp_and_background_cannot_enter_surface_directly"_test = [] {
        Coroutine<void(uint, bool)> source{[](UInt kind, Bool lamp_dies) {
            Bool search = true;
            Bool terminated = false;
            $while(search & !terminated) {
                $suspend("resolve");
                $if(kind == 1u) {
                    $suspend("lamp");
                    terminated = lamp_dies;
                }
                $else {
                    search = false;
                    $if(kind == 2u) {
                        $suspend("background");
                        terminated = true;
                    };
                };
            };
            $if(!terminated) { $suspend("shade"); };
        }};
        const auto &graph = source.graph();
        dump_graph(graph);
        expect(has_named_transition(graph, "", "resolve"));
        expect(has_named_transition(graph, "resolve", "lamp"));
        expect(has_named_transition(graph, "resolve", "background"));
        expect(has_named_transition(graph, "resolve", "shade"));
        expect(has_named_transition(graph, "lamp", "resolve"));
        expect(!has_named_transition(graph, "", "shade"));
        expect(!has_named_transition(graph, "lamp", "shade"));
        expect(!has_named_transition(graph, "background", "shade"));
    };
    "shared_callable_mutation_preserves_both_real_branch_targets"_test = [] {
        Callable<void(bool &)> shared = [](Bool &value) {
            $suspend("shared");
            value = !value;
        };
        Coroutine<void(bool)> source{[&](Bool input) {
            Bool state = input;
            shared(state);
            $if(state) { $suspend("left"); }
            $else { $suspend("right"); };
            shared(state);
            $suspend("done");
        }};
        const auto &graph = source.graph();
        expect(graph.call_graph().analysis_state_count != 0u);
        expect(has_named_transition(graph, "shared", "left"));
        expect(has_named_transition(graph, "shared", "right"));
        expect(has_named_transition(graph, "shared", "done"));
    };
}
