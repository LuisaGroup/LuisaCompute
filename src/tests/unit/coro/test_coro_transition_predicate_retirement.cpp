// Host-only compiler precision regression. Every chain result is an observable
// reference-array store; this is not dead renderer code or a scene-size rule.
#include "ut/ut.hpp"

#include <cstdio>
#include <luisa/xir/basic_block.h>
#include <luisa/xir/builder.h>
#include <luisa/xir/constant.h>
#include <luisa/xir/function.h>
#include <luisa/xir/instructions/alloca.h>
#include <luisa/xir/instructions/arithmetic.h>
#include <luisa/xir/instructions/branch.h>
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
enum class Gate { constant, dynamic, retained_snapshot };

bool has_transition(const xir::CoroCfgDistillResult &cfg, uint from, uint to) {
    for (const auto &edge : cfg.transition_edges) {
        if (cfg.scopes.at(edge.from_scope).trigger_token == from &&
            cfg.scopes.at(edge.to_scope).trigger_token == to) { return true; }
    }
    return false;
}

void check_chain(uint count, Gate gate_kind) {
    xir::Module module;
    auto *kernel = module.create_kernel();
    auto *boolean = Type::of<bool>();
    auto *output_type = Type::array(boolean, count);
    auto *output = kernel->create_reference_argument(output_type);
    auto *gate_input = kernel->create_value_argument(boolean);
    auto *entry = kernel->create_body_block();
    auto *yes = kernel->create_basic_block();
    auto *no = kernel->create_basic_block();
    auto *resume_yes = kernel->create_basic_block();
    auto *resume_no = kernel->create_basic_block();
    xir::XIRBuilder b;
    b.set_insertion_point(entry);
    auto *gate = b.alloca_local(boolean);
    b.store(gate, gate_kind == Gate::dynamic ?
                      static_cast<xir::Value *>(gate_input) : module.create_constant_one(boolean));
    vector<xir::Value *> snapshots, inverted;
    for (auto i = 0u; i < count; ++i) {
        auto *input = kernel->create_value_argument(boolean);
        auto *slot = b.alloca_local(boolean);
        b.store(slot, input);
        snapshots.emplace_back(b.load(boolean, slot));
    }
    // The early snapshots precede all their late copies in ROBDD variable
    // order. Retaining every dead relation makes this equality family
    // exponential, although its semantic live relation frontier is linear.
    for (auto i = 0u; i < count; ++i) {
        auto *value = b.call(boolean, xir::ArithmeticOp::UNARY_BIT_NOT, {snapshots[i]});
        inverted.emplace_back(value);
        auto *index = module.create_constant(Type::of<uint>(), &i);
        b.store(b.gep(boolean, output, {index}), value);
    }
    xir::Value *condition = b.load(boolean, gate);
    if (gate_kind == Gate::retained_snapshot) {
        // These two old SSA values are genuinely live beyond their output
        // store. Last-use projection must not forget their relation early.
        auto *restored = b.call(boolean, xir::ArithmeticOp::UNARY_BIT_NOT, {inverted.front()});
        condition = b.call(boolean, xir::ArithmeticOp::BINARY_EQUAL,
                           {snapshots.front(), restored});
    }
    b.cond_br(condition, yes, no);
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
    xir::CoroCfgDistillStats stats;
    auto cfg = xir::coro_cfg_distill_pass_run_on_function(kernel, {.stats = &stats});
    expect(cfg.succeeded());
    if (!cfg.succeeded()) { return; }
    std::fprintf(stderr, "chain count=%u gate=%u widened=%d states=%zu edges=%zu\n",
                 count, static_cast<uint>(gate_kind), stats.reachability_widened,
                 stats.reachability_state_count, cfg.transition_edges.size());
    expect(!stats.reachability_widened)
        << "dead intra-block Boolean relations must not exhaust the relation budget";
    expect(has_transition(cfg, 0u, 1u));
    expect(has_transition(cfg, 0u, 2u) == (gate_kind == Gate::dynamic));

    auto split = xir::coro_split_pass_run_on_module_with_cfg_and_frame_info(&module, cfg, nullptr);
    expect(split.succeeded());
    if (!split.succeeded()) { return; }
    expect(xir::xir_verify_module(&module).succeeded());
    auto outputs = 0u;
    for (const auto &subroutine : split.subroutines) {
        subroutine.callable->definition()->traverse_basic_blocks([&](xir::BasicBlock *block) {
            for (auto *instruction : block->instructions()) {
                if (!instruction->isa<xir::StoreInst>()) { continue; }
                auto *pointer = static_cast<xir::StoreInst *>(instruction)->variable();
                if (pointer->isa<xir::GEPInst>() &&
                    static_cast<xir::GEPInst *>(pointer)->base()->type() == output_type) { ++outputs; }
            }
        });
    }
    expect(outputs == count) << "projection changes facts, not observable output stores";
}
}// namespace

int main(int argc, char *argv[]) {
    boost::ut::detail::cfg::parse_arg_with_fallback(argc, const_cast<const char **>(argv));
    "independent_observable_boolean_chains_retire_dead_relations"_test = [] {
        for (auto count : {3u, 8u, 24u}) { check_chain(count, Gate::constant); }
    };
    "retirement_preserves_both_dynamic_gate_targets"_test = [] {
        for (auto count : {3u, 8u, 24u}) { check_chain(count, Gate::dynamic); }
    };
    "snapshots_with_a_later_use_keep_their_relation"_test = [] {
        check_chain(24u, Gate::retained_snapshot);
    };
}
