// Test for reconstructing structured XIR control flow from explicit CFGs.

#include "ut/ut.hpp"
#include <luisa/luisa-compute.h>
#include <luisa/ast/type_registry.h>
#include <luisa/xir/basic_block.h>
#include <luisa/xir/builder.h>
#include <luisa/xir/function.h>
#include <luisa/xir/instructions/branch.h>
#include <luisa/xir/instructions/if.h>
#include <luisa/xir/instructions/phi.h>
#include <luisa/xir/instructions/loop.h>
#include <luisa/xir/instructions/return.h>
#include <luisa/xir/instructions/switch.h>
#include <luisa/xir/module.h>
#include <luisa/xir/passes/destructure_cfg.h>
#include <luisa/xir/passes/algebraic_simplify.h>
#include <luisa/xir/passes/const_fold.h>
#include <luisa/xir/passes/dce.h>
#include <luisa/xir/passes/dead_store_elimination.h>
#include <luisa/xir/passes/gvn.h>
#include <luisa/xir/passes/if_conversion.h>
#include <luisa/xir/passes/local_load_elimination.h>
#include <luisa/xir/passes/local_store_forward.h>
#include <luisa/xir/passes/lower_ray_query_loop_to_loop.h>
#include <luisa/xir/passes/lower_switch.h>
#include <luisa/xir/passes/mem2reg.h>
#include <luisa/xir/passes/phi_cleanup.h>
#include <luisa/xir/passes/reg2mem.h>
#include <luisa/xir/passes/restructure_cfg.h>
#include <luisa/xir/passes/sccp.h>
#include <luisa/xir/passes/simplify_cfg.h>
#include <luisa/xir/passes/unused_callable_removal.h>
#include <luisa/xir/translators/ast2xir.h>
#include <luisa/xir/verifier.h>

using namespace luisa;
using namespace luisa::compute;
using namespace luisa::compute::xir;
using namespace boost::ut;
using namespace boost::ut::literals;

namespace {

[[nodiscard]] KernelFunction *make_kernel_with_body(Module &m, BasicBlock *&body_out) noexcept {
    auto *k = m.create_kernel();
    body_out = k->create_body_block();
    return k;
}

[[nodiscard]] size_t count_terminator_kind(FunctionDefinition *def,
                                           DerivedInstructionTag tag) noexcept {
    size_t n = 0u;
    def->traverse_basic_blocks([&](BasicBlock *bb) noexcept {
        if (!bb->is_terminated()) { return; }
        if (bb->terminator()->derived_instruction_tag() == tag) { ++n; }
    });
    return n;
}

[[nodiscard]] size_t count_non_canonical_loop_prepare(FunctionDefinition *def) noexcept {
    size_t n = 0u;
    def->traverse_basic_blocks([&](BasicBlock *bb) noexcept {
        if (!bb->is_terminated()) { return; }
        auto *term = bb->terminator();
        if (!term->isa<LoopInst>()) { return; }
        auto *loop = static_cast<LoopInst *>(term);
        auto *prepare = loop->prepare_block();
        if (prepare == nullptr || !prepare->is_terminated()) {
            ++n;
            return;
        }
        auto *prepare_term = prepare->terminator();
        if (!prepare_term->isa<ConditionalBranchInst>()) {
            ++n;
            return;
        }
        auto *cond_br = static_cast<ConditionalBranchInst *>(prepare_term);
        if (cond_br->true_block() != loop->body_block() ||
            cond_br->false_block() != loop->merge_block()) {
            ++n;
        }
    });
    return n;
}

[[nodiscard]] size_t count_non_canonical_loop_update(FunctionDefinition *def) noexcept {
    size_t n = 0u;
    def->traverse_basic_blocks([&](BasicBlock *bb) noexcept {
        if (!bb->is_terminated()) { return; }
        auto *term = bb->terminator();
        if (!term->isa<LoopInst>()) { return; }
        auto *loop = static_cast<LoopInst *>(term);
        auto *prepare = loop->prepare_block();
        auto *update = loop->update_block();
        if (prepare == nullptr || update == nullptr || !update->is_terminated()) {
            ++n;
            return;
        }
        bool branches_to_prepare = false;
        update->traverse_successors(false, [&](BasicBlock *succ) noexcept {
            if (succ == prepare) { branches_to_prepare = true; }
        });
        if (!branches_to_prepare) { ++n; }
    });
    return n;
}

[[nodiscard]] size_t count_phi(FunctionDefinition *def) noexcept {
    size_t n = 0u;
    def->traverse_instructions([&](Instruction *inst) noexcept {
        if (inst->isa<PhiInst>()) { ++n; }
    });
    return n;
}

[[nodiscard]] bool branch_chain_reaches(BasicBlock *from, BasicBlock *to) noexcept {
    luisa::unordered_set<BasicBlock *> visited;
    auto *cur = from;
    while (cur != nullptr && visited.emplace(cur).second) {
        if (cur == to) { return true; }
        if (!cur->is_terminated() || !cur->terminator()->isa<BranchInst>()) { return false; }
        cur = static_cast<BranchInst *>(cur->terminator())->target_block();
    }
    return false;
}

void run_spirv_normalize_before_restructure(Module *m) noexcept {
    auto algebraic_options = AlgebraicSimplifyOptions{};
    (void)lower_ray_query_loop_to_loop_pass_run_on_module(m);
    (void)lower_switch_pass_run_on_module(m);
    (void)destructure_cfg_pass_run_on_module(m);
    (void)mem2reg_pass_run_on_module(m);
    (void)algebraic_simplify_pass_run_on_module(m, algebraic_options);
    (void)const_fold_pass_run_on_module(m);
    (void)sccp_pass_run_on_module(m);
    (void)dce_pass_run_on_module(m);
    (void)local_store_forward_pass_run_on_module(m);
    (void)local_load_elimination_pass_run_on_module(m);
    (void)dead_store_elimination_pass_run_on_module(m);
    (void)dce_pass_run_on_module(m);
    (void)gvn_pass_run_on_module(m);
    (void)if_conversion_pass_run_on_module(m);
    (void)phi_cleanup_pass_run_on_module(m);
    (void)unused_callable_removal_pass_run_on_module(m);
    (void)simplify_cfg_pass_run_on_module(m);
    (void)reg2mem_pass_run_on_module(m);
}

void expect_no_structured_cfg(FunctionDefinition *def) noexcept {
    expect(count_terminator_kind(def, DerivedInstructionTag::LOOP) == 0u);
    expect(count_terminator_kind(def, DerivedInstructionTag::SIMPLE_LOOP) == 0u);
    expect(count_terminator_kind(def, DerivedInstructionTag::IF) == 0u);
}

}// namespace

void reg_restructure_cfg() {

    "restructure_empty_function_noop"_test = [] {
        Module m;
        BasicBlock *body;
        auto *k = make_kernel_with_body(m, body);
        XIRBuilder b;
        b.set_insertion_point(body);
        b.return_void();
        auto info = restructure_cfg_pass_run_on_function(k);
        expect(info.restructured_if_count == 0u);
        expect(info.restructured_loop_count == 0u);
        expect(info.irreducible_region_count == 0u);
    };

    "restructure_external_function_skipped"_test = [] {
        Module m;
        auto *ext = m.create_external_function(Type::of<void>());
        auto info = restructure_cfg_pass_run_on_function(ext);
        expect(info.restructured_if_count == 0u);
        expect(info.restructured_loop_count == 0u);
    };

    "restructure_switch_with_duplicate_targets"_test = [] {
        Module m;
        BasicBlock *body;
        auto *kernel = make_kernel_with_body(m, body);
        auto *selector = kernel->create_value_argument(Type::of<int>());
        XIRBuilder b;
        b.set_insertion_point(body);
        auto *sw = b.switch_(selector);
        auto *shared_target = sw->create_default_block();
        sw->add_case(1, shared_target);
        sw->add_case(2, shared_target);
        auto *merge = sw->create_merge_block();
        b.set_insertion_point(shared_target);
        b.br(merge);
        b.set_insertion_point(merge);
        b.return_void();

        size_t successor_count = 0u;
        body->traverse_successors(false, [&](BasicBlock *successor) noexcept {
            expect(successor == shared_target);
            ++successor_count;
        });
        expect(successor_count == 1u);
        size_t predecessor_count = 0u;
        shared_target->traverse_predecessors(false, [&](BasicBlock *predecessor) noexcept {
            expect(predecessor == body);
            ++predecessor_count;
        });
        expect(predecessor_count == 1u);

        // Dominance and post-dominance traversal must accept multiple switch
        // operands that represent the same CFG successor. The restructurer then
        // gives each switch label a unique proxy as required by code generation.
        (void)restructure_cfg_pass_run_on_function(kernel);
        expect(body->terminator()->isa<SwitchInst>());
        auto *normalized = static_cast<SwitchInst *>(body->terminator());
        expect(normalized->case_count() == 2u);
        auto *default_target = normalized->default_block();
        auto *case_0_target = normalized->case_block(0u);
        auto *case_1_target = normalized->case_block(1u);
        expect(default_target != case_0_target);
        expect(default_target != case_1_target);
        expect(case_0_target != case_1_target);
        expect(branch_chain_reaches(default_target, merge));
        expect(branch_chain_reaches(case_0_target, merge));
        expect(branch_chain_reaches(case_1_target, merge));
    };

    "restructure_irreducible_scc_rejected_atomically"_test = [] {
        Module m;
        BasicBlock *body;
        auto *kernel = make_kernel_with_body(m, body);
        auto *condition0 = kernel->create_value_argument(Type::of<bool>());
        auto *condition1 = kernel->create_value_argument(Type::of<bool>());
        auto *definition = kernel->definition();
        auto *left = definition->create_basic_block();
        auto *right = definition->create_basic_block();
        auto *exit = definition->create_basic_block();
        XIRBuilder b;
        b.set_insertion_point(body);
        auto *entry_branch = b.cond_br(condition0, left, right);
        b.set_insertion_point(left);
        auto *left_branch = b.br(right);
        b.set_insertion_point(right);
        auto *right_branch = b.cond_br(condition1, left, exit);
        b.set_insertion_point(exit);
        auto *exit_return = b.return_void();
        auto block_count = definition->basic_blocks().count_size();

        auto info = restructure_cfg_pass_run_on_function(kernel);

        expect(!info.succeeded());
        expect(info.irreducible_region_count == 1u);
        expect(info.restructured_loop_count == 0u);
        expect(info.restructured_if_count == 0u);
        expect(definition->basic_blocks().count_size() == block_count);
        expect(body->terminator() == entry_branch);
        expect(left->terminator() == left_branch);
        expect(right->terminator() == right_branch);
        expect(exit->terminator() == exit_return);
        expect(entry_branch->true_block() == left);
        expect(entry_branch->false_block() == right);
        expect(right_branch->true_block() == left);
        expect(right_branch->false_block() == exit);
    };

    "restructure_if_from_destructured"_test = [] {
        Module m;
        BasicBlock *body;
        auto *k = make_kernel_with_body(m, body);
        XIRBuilder b;
        b.set_insertion_point(body);
        auto *cond = m.create_constant_one(Type::of<bool>());
        auto *if_inst = b.if_(cond);
        auto *t = if_inst->create_true_block();
        auto *f = if_inst->create_false_block();
        auto *merge = if_inst->create_merge_block();
        b.set_insertion_point(t);
        b.br(merge);
        b.set_insertion_point(f);
        b.br(merge);
        b.set_insertion_point(merge);
        b.return_void();
        (void)destructure_cfg_pass_run_on_function(k);
        auto *def = k->definition();
        expect(count_terminator_kind(def, DerivedInstructionTag::IF) == 0u);
        auto info = restructure_cfg_pass_run_on_function(k);
        expect(info.restructured_if_count == 1u);
        expect(count_terminator_kind(def, DerivedInstructionTag::IF) == 1u);
        expect(count_terminator_kind(def, DerivedInstructionTag::CONDITIONAL_BRANCH) == 0u);
    };

    "restructure_simple_loop_from_destructured"_test = [] {
        Module m;
        BasicBlock *body;
        auto *k = make_kernel_with_body(m, body);
        XIRBuilder b;
        b.set_insertion_point(body);
        auto *sl = b.simple_loop();
        auto *lbody = sl->create_body_block();
        auto *merge = sl->create_merge_block();
        auto *cond = m.create_constant_zero(Type::of<bool>());
        auto *cont = k->definition()->create_basic_block();
        b.set_insertion_point(lbody);
        b.cond_br(cond, merge, cont);
        b.set_insertion_point(cont);
        b.continue_(lbody);
        b.set_insertion_point(merge);
        b.return_void();
        (void)destructure_cfg_pass_run_on_function(k);
        auto *def = k->definition();
        expect(count_terminator_kind(def, DerivedInstructionTag::SIMPLE_LOOP) == 0u);
        auto info = restructure_cfg_pass_run_on_function(k);
        expect(info.restructured_loop_count == 1u);
        expect(count_terminator_kind(def, DerivedInstructionTag::SIMPLE_LOOP) == 1u);
    };

    "restructure_simple_loop_latch_conditional_to_break_continue"_test = [] {
        Module m;
        BasicBlock *body;
        auto *k = make_kernel_with_body(m, body);
        auto *def = k->definition();
        auto *cond = k->create_value_argument(Type::of<bool>());
        auto *loop_body = def->create_basic_block();
        auto *work = def->create_basic_block();
        auto *latch = def->create_basic_block();
        auto *break_proxy = def->create_basic_block();
        auto *merge = def->create_basic_block();

        XIRBuilder b;
        b.set_insertion_point(body);
        auto *sl = b.simple_loop();
        sl->set_body_block(loop_body);
        sl->set_merge_block(merge);
        b.set_insertion_point(loop_body);
        b.br(work);
        b.set_insertion_point(work);
        b.br(latch);
        b.set_insertion_point(latch);
        b.cond_br(cond, break_proxy, loop_body);
        b.set_insertion_point(break_proxy);
        b.br(merge);
        b.set_insertion_point(merge);
        b.return_void();

        expect(count_terminator_kind(def, DerivedInstructionTag::CONDITIONAL_BRANCH) == 1u);
        auto info = restructure_cfg_pass_run_on_function(k);
        expect(info.irreducible_region_count == 0u);
        expect(count_terminator_kind(def, DerivedInstructionTag::CONDITIONAL_BRANCH) == 0u);
        expect(count_terminator_kind(def, DerivedInstructionTag::IF) == 1u);
        expect(count_terminator_kind(def, DerivedInstructionTag::BREAK) == 1u);
        expect(count_terminator_kind(def, DerivedInstructionTag::CONTINUE) <= 1u);
    };

    "restructure_simple_loop_nested_latch_conditional_to_break_continue"_test = [] {
        Module m;
        BasicBlock *body;
        auto *k = make_kernel_with_body(m, body);
        auto *def = k->definition();
        auto *cond = k->create_value_argument(Type::of<bool>());
        auto *loop_body = def->create_basic_block();
        auto *then_block = def->create_basic_block();
        auto *else_block = def->create_basic_block();
        auto *inner_merge = def->create_basic_block();
        auto *latch = def->create_basic_block();
        auto *break_proxy = def->create_basic_block();
        auto *merge = def->create_basic_block();

        XIRBuilder b;
        b.set_insertion_point(body);
        auto *sl = b.simple_loop();
        sl->set_body_block(loop_body);
        sl->set_merge_block(merge);
        b.set_insertion_point(loop_body);
        auto *inner_if = b.if_(cond);
        inner_if->set_true_target(then_block);
        inner_if->set_false_target(else_block);
        inner_if->set_merge_block(inner_merge);
        b.set_insertion_point(then_block);
        b.br(inner_merge);
        b.set_insertion_point(else_block);
        b.br(inner_merge);
        b.set_insertion_point(inner_merge);
        b.br(latch);
        b.set_insertion_point(latch);
        b.cond_br(cond, break_proxy, loop_body);
        b.set_insertion_point(break_proxy);
        b.br(merge);
        b.set_insertion_point(merge);
        b.return_void();

        expect(count_terminator_kind(def, DerivedInstructionTag::CONDITIONAL_BRANCH) == 1u);
        auto info = restructure_cfg_pass_run_on_function(k);
        expect(info.irreducible_region_count == 0u);
        expect(count_terminator_kind(def, DerivedInstructionTag::CONDITIONAL_BRANCH) == 0u);
        expect(count_terminator_kind(def, DerivedInstructionTag::IF) == 2u);
        expect(count_terminator_kind(def, DerivedInstructionTag::BREAK) == 1u);
        expect(count_terminator_kind(def, DerivedInstructionTag::CONTINUE) == 1u);
    };

    "restructure_loop_body_break_or_continue_through_proxy_chain"_test = [] {
        Module m;
        BasicBlock *body;
        auto *k = make_kernel_with_body(m, body);
        auto *def = k->definition();
        auto *cond = k->create_value_argument(Type::of<bool>());
        auto *header = def->create_basic_block();
        auto *loop_body = def->create_basic_block();
        auto *break_block = def->create_basic_block();
        auto *continue_proxy_0 = def->create_basic_block();
        auto *continue_proxy_1 = def->create_basic_block();
        auto *update = def->create_basic_block();
        auto *merge = def->create_basic_block();

        XIRBuilder b;
        b.set_insertion_point(body);
        auto *loop = b.loop();
        loop->set_prepare_block(header);
        loop->set_body_block(loop_body);
        loop->set_update_block(update);
        loop->set_merge_block(merge);
        b.set_insertion_point(header);
        b.cond_br(cond, loop_body, merge);
        b.set_insertion_point(loop_body);
        b.cond_br(cond, break_block, continue_proxy_0);
        b.set_insertion_point(break_block);
        b.break_(merge);
        b.set_insertion_point(continue_proxy_0);
        b.br(continue_proxy_1);
        b.set_insertion_point(continue_proxy_1);
        b.continue_(update);
        b.set_insertion_point(update);
        b.br(header);
        b.set_insertion_point(merge);
        b.return_void();

        expect(count_terminator_kind(def, DerivedInstructionTag::CONDITIONAL_BRANCH) == 2u);
        auto info = restructure_cfg_pass_run_on_function(k);
        expect(info.irreducible_region_count == 0u);
        expect(count_terminator_kind(def, DerivedInstructionTag::CONDITIONAL_BRANCH) == 1u);
        expect(count_terminator_kind(def, DerivedInstructionTag::IF) == 1u);
        expect(count_terminator_kind(def, DerivedInstructionTag::BREAK) == 1u);
        expect(count_terminator_kind(def, DerivedInstructionTag::CONTINUE) == 1u);
    };

    "restructure_module_runs_all_functions"_test = [] {
        Module m;
        constexpr size_t kFns = 3u;
        for (size_t i = 0; i < kFns; ++i) {
            BasicBlock *body;
            auto *k = make_kernel_with_body(m, body);
            XIRBuilder b;
            b.set_insertion_point(body);
            auto *cond = m.create_constant_one(Type::of<bool>());
            auto *if_inst = b.if_(cond);
            auto *t = if_inst->create_true_block();
            auto *f = if_inst->create_false_block();
            auto *merge = if_inst->create_merge_block();
            b.set_insertion_point(t);
            b.br(merge);
            b.set_insertion_point(f);
            b.br(merge);
            b.set_insertion_point(merge);
            b.return_void();
        }
        (void)destructure_cfg_pass_run_on_module(&m);
        auto info = restructure_cfg_pass_run_on_module(&m);
        expect(info.restructured_if_count == kFns);
        for (auto *f : m.function_list()) {
            auto *def = f->definition();
            if (def == nullptr) { continue; }
            expect(count_terminator_kind(def, DerivedInstructionTag::IF) == 1u);
            expect(count_terminator_kind(def, DerivedInstructionTag::CONDITIONAL_BRANCH) == 0u);
        }
    };

    "restructure_idempotent_if"_test = [] {
        Module m;
        BasicBlock *body;
        auto *k = make_kernel_with_body(m, body);
        XIRBuilder b;
        b.set_insertion_point(body);
        auto *cond = m.create_constant_one(Type::of<bool>());
        auto *if_inst = b.if_(cond);
        auto *t = if_inst->create_true_block();
        auto *f = if_inst->create_false_block();
        auto *merge = if_inst->create_merge_block();
        b.set_insertion_point(t);
        b.br(merge);
        b.set_insertion_point(f);
        b.br(merge);
        b.set_insertion_point(merge);
        b.return_void();
        (void)destructure_cfg_pass_run_on_function(k);
        auto first = restructure_cfg_pass_run_on_function(k);
        auto second = restructure_cfg_pass_run_on_function(k);
        expect(first.restructured_if_count == 1u);
        expect(second.restructured_if_count == 0u);
    };

    "restructure_empty_module_noop"_test = [] {
        Module m;
        auto info = restructure_cfg_pass_run_on_module(&m);
        expect(info.restructured_if_count == 0u);
        expect(info.restructured_loop_count == 0u);
    };

    "restructure_if_preserves_true_false_blocks"_test = [] {
        Module m;
        BasicBlock *body;
        auto *k = make_kernel_with_body(m, body);
        XIRBuilder b;
        b.set_insertion_point(body);
        auto *cond = m.create_constant_one(Type::of<bool>());
        auto *if_inst = b.if_(cond);
        auto *t = if_inst->create_true_block();
        auto *f = if_inst->create_false_block();
        auto *merge = if_inst->create_merge_block();
        b.set_insertion_point(t);
        b.br(merge);
        b.set_insertion_point(f);
        b.br(merge);
        b.set_insertion_point(merge);
        b.return_void();
        (void)destructure_cfg_pass_run_on_function(k);
        (void)restructure_cfg_pass_run_on_function(k);
        auto *def = k->definition();
        auto *new_term = def->body_block()->terminator();
        expect(new_term != nullptr);
        expect(new_term->isa<IfInst>());
        auto *rebuilt = static_cast<IfInst *>(new_term);
        // Empty true/false arms branched directly to merge, so during
        // destructure they collapse and restructure retargets them to a
        // fresh structural merge. The arms thus may equal either the
        // original blocks or the structural merge itself.
        auto *rt = rebuilt->true_block();
        auto *rf = rebuilt->false_block();
        auto *rm = rebuilt->merge_block();
        expect(rt != nullptr);
        expect(rf != nullptr);
        expect(rm != nullptr);
        // The structural merge must reach the original merge block. It is
        // either the original merge itself or a freshly-synthesized block
        // whose sole terminator is `br merge`.
        auto *rm_term = rm->terminator();
        expect(rm == merge ||
               (rm_term != nullptr &&
                rm_term->isa<BranchInst>() &&
                static_cast<BranchInst *>(rm_term)->target_block() == merge));
    };

    "restructure_nested_if_from_destructured"_test = [] {
        Module m;
        BasicBlock *body;
        auto *k = make_kernel_with_body(m, body);
        XIRBuilder b;
        b.set_insertion_point(body);
        auto *cond = m.create_constant_one(Type::of<bool>());
        auto *outer = b.if_(cond);
        auto *ot = outer->create_true_block();
        auto *of = outer->create_false_block();
        auto *omerge = outer->create_merge_block();
        b.set_insertion_point(ot);
        auto *inner = b.if_(cond);
        auto *it = inner->create_true_block();
        auto *if_ = inner->create_false_block();
        auto *imerge = inner->create_merge_block();
        b.set_insertion_point(it);
        b.br(imerge);
        b.set_insertion_point(if_);
        b.br(imerge);
        b.set_insertion_point(imerge);
        b.br(omerge);
        b.set_insertion_point(of);
        b.br(omerge);
        b.set_insertion_point(omerge);
        b.return_void();
        (void)destructure_cfg_pass_run_on_function(k);
        auto *def = k->definition();
        expect(count_terminator_kind(def, DerivedInstructionTag::IF) == 0u);
        auto info = restructure_cfg_pass_run_on_function(k);
        expect(info.restructured_if_count == 2u);
        expect(count_terminator_kind(def, DerivedInstructionTag::IF) == 2u);
        expect(count_terminator_kind(def, DerivedInstructionTag::CONDITIONAL_BRANCH) == 0u);
    };

    "restructure_nested_loop_does_not_capture_outer_tail"_test = [] {
        Module m;
        BasicBlock *body;
        auto *k = make_kernel_with_body(m, body);
        auto *def = k->definition();
        auto *outer_header = def->create_basic_block();
        auto *outer_body = def->create_basic_block();
        auto *inner_header = def->create_basic_block();
        auto *inner_body = def->create_basic_block();
        auto *inner_latch = def->create_basic_block();
        auto *after_inner = def->create_basic_block();
        auto *outer_latch = def->create_basic_block();
        auto *outer_exit = def->create_basic_block();
        auto *cond = m.create_constant_one(Type::of<bool>());
        XIRBuilder b;
        b.set_insertion_point(body);
        b.br(outer_header);
        b.set_insertion_point(outer_header);
        b.cond_br(cond, outer_body, outer_exit);
        b.set_insertion_point(outer_body);
        b.br(inner_header);
        b.set_insertion_point(inner_header);
        b.cond_br(cond, inner_body, after_inner);
        b.set_insertion_point(inner_body);
        b.br(inner_latch);
        b.set_insertion_point(inner_latch);
        b.br(inner_header);
        b.set_insertion_point(after_inner);
        b.cond_br(cond, outer_exit, outer_latch);
        b.set_insertion_point(outer_latch);
        b.br(outer_header);
        b.set_insertion_point(outer_exit);
        b.return_void();
        auto info = restructure_cfg_pass_run_on_function(k);
        auto loop_count = count_terminator_kind(def, DerivedInstructionTag::LOOP) +
                          count_terminator_kind(def, DerivedInstructionTag::SIMPLE_LOOP);
        expect(info.restructured_loop_count == 2u);
        expect(loop_count == 2u);
        expect(count_terminator_kind(def, DerivedInstructionTag::SWITCH) == 0u);
        expect(count_non_canonical_loop_prepare(def) == 0u);
        expect(count_non_canonical_loop_update(def) == 0u);
    };

    "restructure_outer_update_path_with_inner_loop"_test = [] {
        Module m;
        BasicBlock *body;
        auto *k = make_kernel_with_body(m, body);
        auto *def = k->definition();
        XIRBuilder b;
        auto *cond = k->create_value_argument(Type::of<bool>());
        auto *outer_continue_entry = def->create_basic_block();
        auto *after_inner = def->create_basic_block();

        b.set_insertion_point(body);
        auto *outer = b.loop();
        auto *outer_prepare = outer->create_prepare_block();
        auto *outer_body = outer->create_body_block();
        auto *outer_update = outer->create_update_block();
        auto *outer_merge = outer->create_merge_block();

        b.set_insertion_point(outer_prepare);
        b.cond_br(cond, outer_body, outer_merge);

        b.set_insertion_point(outer_body);
        auto *body_if = b.if_(cond);
        auto *body_then = body_if->create_true_block();
        auto *body_else = body_if->create_false_block();
        auto *body_if_merge = body_if->create_merge_block();
        b.set_insertion_point(body_then);
        b.br(body_if_merge);
        b.set_insertion_point(body_else);
        b.br(body_if_merge);
        b.set_insertion_point(body_if_merge);
        auto *continue_if = b.if_(cond);
        auto *break_path = continue_if->create_true_block();
        auto *continue_path = continue_if->create_false_block();
        auto *continue_if_merge = continue_if->create_merge_block();
        b.set_insertion_point(break_path);
        b.break_(outer_merge);
        b.set_insertion_point(continue_path);
        b.br(continue_if_merge);
        b.set_insertion_point(continue_if_merge);
        b.br(outer_continue_entry);

        b.set_insertion_point(outer_continue_entry);
        auto *inner = b.loop();
        auto *inner_prepare = inner->create_prepare_block();
        auto *inner_body = inner->create_body_block();
        auto *inner_update = inner->create_update_block();
        auto *inner_merge = inner->create_merge_block();
        b.set_insertion_point(inner_prepare);
        b.cond_br(cond, inner_body, inner_merge);
        b.set_insertion_point(inner_body);
        b.br(inner_update);
        b.set_insertion_point(inner_update);
        b.br(inner_prepare);
        b.set_insertion_point(inner_merge);
        b.br(after_inner);
        b.set_insertion_point(after_inner);
        b.br(outer_update);

        b.set_insertion_point(outer_update);
        b.br(outer_prepare);
        b.set_insertion_point(outer_merge);
        b.return_void();

        expect(count_terminator_kind(def, DerivedInstructionTag::LOOP) == 2u);
        run_spirv_normalize_before_restructure(&m);
        expect_no_structured_cfg(def);
        auto info = restructure_cfg_pass_run_on_function(k);
        expect(info.iteration_limit_count == 0u);
        expect(info.restructured_loop_count == 2u);
        expect(count_terminator_kind(def, DerivedInstructionTag::LOOP) +
                   count_terminator_kind(def, DerivedInstructionTag::SIMPLE_LOOP) ==
               2u);
        expect(count_non_canonical_loop_prepare(def) == 0u);
        expect(count_non_canonical_loop_update(def) == 0u);
    };

    "restructure_outer_loop_break_or_update_if_shape"_test = [] {
        Module m;
        BasicBlock *body;
        auto *k = make_kernel_with_body(m, body);
        auto *def = k->definition();
        auto *cond = k->create_value_argument(Type::of<bool>());
        auto *header = def->create_basic_block();
        auto *body_entry = def->create_basic_block();
        auto *first_then = def->create_basic_block();
        auto *first_else = def->create_basic_block();
        auto *first_merge = def->create_basic_block();
        auto *break_block = def->create_basic_block();
        auto *update_path = def->create_basic_block();
        auto *if_merge_on_update_path = def->create_basic_block();
        auto *latch = def->create_basic_block();
        auto *exit = def->create_basic_block();

        XIRBuilder b;
        b.set_insertion_point(body);
        b.br(header);
        b.set_insertion_point(header);
        b.cond_br(cond, body_entry, exit);
        b.set_insertion_point(body_entry);
        b.cond_br(cond, first_then, first_else);
        b.set_insertion_point(first_then);
        b.br(first_merge);
        b.set_insertion_point(first_else);
        b.br(first_merge);
        b.set_insertion_point(first_merge);
        b.cond_br(cond, break_block, update_path);
        b.set_insertion_point(break_block);
        b.br(exit);
        b.set_insertion_point(update_path);
        b.br(if_merge_on_update_path);
        b.set_insertion_point(if_merge_on_update_path);
        b.br(latch);
        b.set_insertion_point(latch);
        b.br(header);
        b.set_insertion_point(exit);
        b.return_void();

        run_spirv_normalize_before_restructure(&m);
        expect_no_structured_cfg(def);
        auto info = restructure_cfg_pass_run_on_function(k);
        expect(info.restructured_loop_count == 1u);
        expect(count_terminator_kind(def, DerivedInstructionTag::LOOP) == 1u);
        expect(count_non_canonical_loop_prepare(def) == 0u);
        expect(count_non_canonical_loop_update(def) == 0u);
    };

    "restructure_full_pipeline_loop_with_inner_phi_diamond"_test = [] {
        Module m;
        BasicBlock *body;
        auto *k = make_kernel_with_body(m, body);
        auto *def = k->definition();
        auto *cond = k->create_value_argument(Type::of<bool>());
        auto *header = def->create_basic_block();
        auto *loop_body = def->create_basic_block();
        auto *then_block = def->create_basic_block();
        auto *else_block = def->create_basic_block();
        auto *diamond_merge = def->create_basic_block();
        auto *break_block = def->create_basic_block();
        auto *continue_block = def->create_basic_block();
        auto *inner_header = def->create_basic_block();
        auto *inner_body = def->create_basic_block();
        auto *inner_latch = def->create_basic_block();
        auto *inner_merge = def->create_basic_block();
        auto *outer_latch = def->create_basic_block();
        auto *exit = def->create_basic_block();
        auto *zero = m.create_constant_zero(Type::of<int>());
        int one_v = 1;
        auto *one = m.create_constant(Type::of<int>(), &one_v);
        int two_v = 2;
        auto *two = m.create_constant(Type::of<int>(), &two_v);

        XIRBuilder b;
        b.set_insertion_point(body);
        b.br(header);
        b.set_insertion_point(header);
        b.cond_br(cond, loop_body, exit);
        b.set_insertion_point(loop_body);
        b.cond_br(cond, then_block, else_block);
        b.set_insertion_point(then_block);
        auto *then_value = b.call(Type::of<int>(), ArithmeticOp::BINARY_ADD, {zero, one});
        b.br(diamond_merge);
        b.set_insertion_point(else_block);
        auto *else_value = b.call(Type::of<int>(), ArithmeticOp::BINARY_ADD, {zero, two});
        b.br(diamond_merge);
        b.set_insertion_point(diamond_merge);
        auto *phi = b.phi(Type::of<int>());
        phi->add_incoming(then_value, then_block);
        phi->add_incoming(else_value, else_block);
        auto *break_cond = b.call(Type::of<bool>(), ArithmeticOp::BINARY_EQUAL, {phi, zero});
        b.cond_br(break_cond, break_block, continue_block);
        b.set_insertion_point(break_block);
        b.br(exit);
        b.set_insertion_point(continue_block);
        b.br(inner_header);
        b.set_insertion_point(inner_header);
        b.cond_br(cond, inner_body, inner_merge);
        b.set_insertion_point(inner_body);
        b.br(inner_latch);
        b.set_insertion_point(inner_latch);
        b.br(inner_header);
        b.set_insertion_point(inner_merge);
        b.br(outer_latch);
        b.set_insertion_point(outer_latch);
        b.br(header);
        b.set_insertion_point(exit);
        b.return_void();

        expect(count_phi(def) == 1u);
        run_spirv_normalize_before_restructure(&m);
        expect_no_structured_cfg(def);
        expect(count_phi(def) == 0u);
        auto info = restructure_cfg_pass_run_on_function(k);
        expect(info.restructured_loop_count == 2u);
        expect(count_terminator_kind(def, DerivedInstructionTag::LOOP) +
                   count_terminator_kind(def, DerivedInstructionTag::SIMPLE_LOOP) ==
               2u);
        expect(count_non_canonical_loop_prepare(def) == 0u);
        expect(count_non_canonical_loop_update(def) == 0u);
    };

    "restructure_full_pipeline_ast_kernel_nested_loop_break"_test = [] {
        Kernel1D kernel = [](BufferFloat buf, Float t) noexcept {
            auto idx = dispatch_id().x;
            Float x = buf.read(idx);
            Float acc = def(0.0f);
            Float state = def(x);
            Bool flag = def(false);
            $for (i, 8u) {
                auto hit = state > t;
                $if (hit & (state != acc)) {
                    flag = flag | hit;
                    hit = false;
                    Float tmp = def(0.0f);
                    $for (j, 4u) {
                        tmp += state * cast<float>(j + 1u);
                    };
                    state = tmp * 0.25f;
                };
                acc += state;
                state += 1.0f;
                $if (hit) { $break; };
            };
            buf.write(idx, ite(flag, acc, state));
        };

        auto m = ast_to_xir_translate(kernel.function()->function(), {});
        expect(m != nullptr);
        run_spirv_normalize_before_restructure(m.get());
        for (auto *f : m->function_list()) {
            if (auto *def = f->definition(); def != nullptr) {
                expect_no_structured_cfg(def);
                expect(count_phi(def) == 0u);
            }
        }
        auto info = restructure_cfg_pass_run_on_module(m.get());
        expect(info.iteration_limit_count == 0u);
        expect(info.restructured_loop_count > 0u);
        for (auto *f : m->function_list()) {
            if (auto *def = f->definition(); def != nullptr) {
                expect(count_non_canonical_loop_prepare(def) == 0u);
                expect(count_non_canonical_loop_update(def) == 0u);
            }
        }
        expect(xir_verify_module(
                   m.get(), {.require_unique_merge_blocks = true})
                   .succeeded());
    };
    "restructure_converts_remaining_divergent_conditional"_test = [] {
        Module m;
        BasicBlock *body;
        auto *k = make_kernel_with_body(m, body);
        auto *def = k->definition();
        auto *cond = k->create_value_argument(Type::of<bool>());
        auto *ret_a = def->create_basic_block();
        auto *ret_b = def->create_basic_block();
        XIRBuilder b;
        b.set_insertion_point(body);
        b.cond_br(cond, ret_a, ret_b);
        b.set_insertion_point(ret_a);
        b.return_void();
        b.set_insertion_point(ret_b);
        b.return_void();
        // Skip full pipeline; reg2mem is a no-op here (no phis).
        (void)reg2mem_pass_run_on_function(k);
        expect(count_terminator_kind(def, DerivedInstructionTag::CONDITIONAL_BRANCH) == 1u);
        auto info = restructure_cfg_pass_run_on_function(k);
        expect(info.restructured_if_count >= 1u) << "conditional branch should be structurized";
        expect(count_terminator_kind(def, DerivedInstructionTag::CONDITIONAL_BRANCH) == 0u);
    };

    "restructure_fixup_nested_if_cross_hierarchy"_test = [] {
        Module m;
        BasicBlock *body;
        auto *k = make_kernel_with_body(m, body);
        auto *def = k->definition();
        auto *cond = k->create_value_argument(Type::of<bool>());
        auto *outer_merge = def->create_basic_block();
        auto *inner_merge = def->create_basic_block();
        XIRBuilder b;
        b.set_insertion_point(body);
        auto *outer = b.if_(cond);
        auto *ot = outer->create_true_block();
        auto *of = outer->create_false_block();
        b.set_insertion_point(ot);
        auto *inner = b.if_(cond);
        auto *it = inner->create_true_block();
        auto *if_ = inner->create_false_block();
        b.set_insertion_point(it);
        b.br(inner_merge);
        b.set_insertion_point(if_);
        b.br(outer_merge);
        b.set_insertion_point(inner_merge);
        b.br(outer_merge);
        b.set_insertion_point(of);
        b.br(outer_merge);
        b.set_insertion_point(outer_merge);
        b.return_void();
        (void)destructure_cfg_pass_run_on_function(k);
        (void)reg2mem_pass_run_on_function(k);
        expect_no_structured_cfg(def);
        expect(count_terminator_kind(def, DerivedInstructionTag::CONDITIONAL_BRANCH) >= 2u);
        auto info = restructure_cfg_pass_run_on_function(k);
        expect(info.iteration_limit_count == 0u);
        expect(info.restructured_if_count >= 2u);
        expect(count_terminator_kind(def, DerivedInstructionTag::CONDITIONAL_BRANCH) == 0u);
    };

    "restructure_routes_nested_switch_exits_through_merges"_test = [] {
        Module m;
        BasicBlock *body;
        auto *k = make_kernel_with_body(m, body);
        auto *def = k->definition();
        auto *selector = m.create_constant_zero(Type::of<uint32_t>());
        XIRBuilder b;

        b.set_insertion_point(body);
        auto *outer = b.switch_(selector);
        auto *outer_default = outer->create_default_block();
        auto *outer_case = outer->create_case_block(1);
        auto *outer_merge = outer->create_merge_block();
        auto *ret = def->create_basic_block();

        b.set_insertion_point(outer_default);
        auto *inner = b.switch_(selector);
        auto *inner_default = inner->create_default_block();
        auto *inner_case = inner->create_case_block(1);
        auto *inner_merge = inner->create_merge_block();

        b.set_insertion_point(inner_default);
        b.br(ret);
        b.set_insertion_point(inner_case);
        b.br(ret);
        b.set_insertion_point(inner_merge);
        b.unreachable_();

        b.set_insertion_point(outer_case);
        b.br(ret);
        b.set_insertion_point(outer_merge);
        b.unreachable_();
        b.set_insertion_point(ret);
        b.return_void();

        auto info = restructure_cfg_pass_run_on_function(k);
        expect(info.iteration_limit_count == 0u);
        expect(info.irreducible_region_count == 0u);
        expect(inner->merge_block() != inner_merge);
        expect(outer->merge_block() != outer_merge);
        expect(static_cast<BranchInst *>(inner_default->terminator())->target_block() == inner->merge_block());
        expect(static_cast<BranchInst *>(inner_case->terminator())->target_block() == inner->merge_block());
        expect(static_cast<BranchInst *>(inner->merge_block()->terminator())->target_block() == outer->merge_block());
        expect(static_cast<BranchInst *>(outer_case->terminator())->target_block() == outer->merge_block());
        expect(static_cast<BranchInst *>(outer->merge_block()->terminator())->target_block() == ret);
    };

    "restructure_routes_nested_if_exits_through_merges"_test = [] {
        Module m;
        BasicBlock *body;
        auto *k = make_kernel_with_body(m, body);
        auto *def = k->definition();
        auto *cond = k->create_value_argument(Type::of<bool>());
        XIRBuilder b;

        b.set_insertion_point(body);
        auto *outer = b.if_(cond);
        auto *outer_then = outer->create_true_block();
        auto *outer_else = outer->create_false_block();
        auto *outer_merge = outer->create_merge_block();
        auto *ret = def->create_basic_block();

        b.set_insertion_point(outer_then);
        auto *inner = b.if_(cond);
        auto *inner_then = inner->create_true_block();
        auto *inner_else = inner->create_false_block();
        auto *inner_merge = inner->create_merge_block();

        b.set_insertion_point(inner_then);
        b.br(inner_merge);
        b.set_insertion_point(inner_else);
        b.br(outer_merge);
        b.set_insertion_point(inner_merge);
        b.br(outer_merge);

        b.set_insertion_point(outer_else);
        b.br(outer_merge);
        b.set_insertion_point(outer_merge);
        b.br(ret);
        b.set_insertion_point(ret);
        b.return_void();

        auto info = restructure_cfg_pass_run_on_function(k);
        expect(info.iteration_limit_count == 0u);
        expect(info.irreducible_region_count == 0u);
        expect(inner->merge_block() != inner_merge);
        expect(outer->merge_block() != outer_merge);
        expect(static_cast<BranchInst *>(inner_then->terminator())->target_block() == inner->merge_block());
        expect(static_cast<BranchInst *>(inner_else->terminator())->target_block() == inner->merge_block());
        expect(static_cast<BranchInst *>(inner->merge_block()->terminator())->target_block() == outer->merge_block());
        expect(static_cast<BranchInst *>(outer_else->terminator())->target_block() == outer->merge_block());
        expect(branch_chain_reaches(outer->merge_block(), ret));
    };

    "restructure_structurizes_raw_branch_inside_structured_if"_test = [] {
        Module m;
        BasicBlock *body;
        auto *k = make_kernel_with_body(m, body);
        auto *condition = k->create_value_argument(Type::of<bool>());
        XIRBuilder b;
        b.set_insertion_point(body);
        auto *outer = b.if_(condition);
        auto *outer_true = outer->create_true_block();
        auto *outer_false = outer->create_false_block();
        auto *outer_merge = outer->create_merge_block();
        auto *inner_tail = k->create_basic_block();
        b.set_insertion_point(outer_true);
        b.cond_br(condition, outer_merge, inner_tail);
        b.set_insertion_point(inner_tail);
        b.br(outer_merge);
        b.set_insertion_point(outer_false);
        b.br(outer_merge);
        b.set_insertion_point(outer_merge);
        b.return_void();

        auto info = restructure_cfg_pass_run_on_function(k);
        expect(info.succeeded());
        expect(info.unstructured_branch_count == 0u);
        expect(info.invalid_construct_count == 0u);
        expect(info.iteration_limit_count == 0u);
        expect(count_terminator_kind(k->definition(), DerivedInstructionTag::CONDITIONAL_BRANCH) == 0u);
        expect(xir_verify_module(
                   &m, {.require_unique_merge_blocks = true})
                   .succeeded());
    };

    "restructure_structurizes_one_sided_branch_before_nested_selection"_test = [] {
        Module m;
        BasicBlock *body;
        auto *k = make_kernel_with_body(m, body);
        auto *condition = k->create_value_argument(Type::of<bool>());
        XIRBuilder b;
        b.set_insertion_point(body);
        auto *outer = b.if_(condition);
        auto *raw_header = outer->create_true_block();
        auto *outer_false = outer->create_false_block();
        auto *outer_merge = outer->create_merge_block();
        auto *nested_header = k->create_basic_block();

        b.set_insertion_point(raw_header);
        b.cond_br(condition, nested_header, outer_merge);
        b.set_insertion_point(nested_header);
        auto *nested = b.if_(condition);
        auto *nested_true = nested->create_true_block();
        auto *nested_false = nested->create_false_block();
        auto *nested_merge = nested->create_merge_block();
        b.set_insertion_point(nested_true);
        b.br(nested_merge);
        b.set_insertion_point(nested_false);
        b.br(nested_merge);
        b.set_insertion_point(nested_merge);
        b.br(outer_merge);
        b.set_insertion_point(outer_false);
        b.br(outer_merge);
        b.set_insertion_point(outer_merge);
        b.return_void();

        auto info = restructure_cfg_pass_run_on_function(k);
        auto verification = xir_verify_module(
            &m, {.require_unique_merge_blocks = true});
        expect(info.succeeded());
        expect(info.restructured_if_count >= 1u);
        expect(raw_header->terminator()->isa<IfInst>());
        auto *structured = static_cast<IfInst *>(raw_header->terminator());
        expect(structured->merge_block() != outer_merge);
        expect(nested_header->terminator() == nested);
        expect(count_terminator_kind(k->definition(), DerivedInstructionTag::CONDITIONAL_BRANCH) == 0u);
        expect(verification.succeeded());
    };

    "restructure_structurizes_loop_early_exit_ladder"_test = [] {
        Module m;
        BasicBlock *body;
        auto *k = make_kernel_with_body(m, body);
        auto *condition = k->create_value_argument(Type::of<bool>());
        auto *def = k->definition();
        XIRBuilder b;
        b.set_insertion_point(body);
        auto *loop = b.loop();
        auto *prepare = loop->create_prepare_block();
        auto *first_guard = loop->create_body_block();
        auto *update = loop->create_update_block();
        auto *merge = loop->create_merge_block();
        auto *first_break = def->create_basic_block();
        auto *second_guard = def->create_basic_block();
        auto *second_break = def->create_basic_block();
        auto *third_guard = def->create_basic_block();
        auto *third_break = def->create_basic_block();
        auto *continue_block = def->create_basic_block();

        b.set_insertion_point(prepare);
        b.cond_br(condition, first_guard, merge);
        b.set_insertion_point(first_guard);
        b.cond_br(condition, first_break, second_guard);
        b.set_insertion_point(first_break);
        b.break_(merge);
        b.set_insertion_point(second_guard);
        b.cond_br(condition, second_break, third_guard);
        b.set_insertion_point(second_break);
        b.break_(merge);
        b.set_insertion_point(third_guard);
        b.cond_br(condition, third_break, continue_block);
        b.set_insertion_point(third_break);
        b.break_(merge);
        b.set_insertion_point(continue_block);
        b.continue_(update);
        b.set_insertion_point(update);
        b.br(prepare);
        b.set_insertion_point(merge);
        b.return_void();

        auto info = restructure_cfg_pass_run_on_function(k);
        auto verification = xir_verify_module(
            &m, {.require_unique_merge_blocks = true});
        expect(info.succeeded());
        expect(info.iteration_limit_count == 0u);
        expect(first_guard->terminator()->isa<IfInst>());
        expect(second_guard->terminator()->isa<IfInst>());
        expect(third_guard->terminator()->isa<IfInst>());
        expect(count_terminator_kind(def, DerivedInstructionTag::CONDITIONAL_BRANCH) == 1u);
        expect(count_non_canonical_loop_prepare(def) == 0u);
        expect(verification.succeeded());
    };

    "spirv_normalization_lowers_switch_before_if_conversion"_test = [] {
        Module m;
        BasicBlock *body;
        auto *k = make_kernel_with_body(m, body);
        auto *selector = k->create_value_argument(Type::of<uint32_t>());
        XIRBuilder b;
        b.set_insertion_point(body);
        auto *sw = b.switch_(selector);
        auto *case_block = sw->create_case_block(1);
        auto *default_block = sw->create_default_block();
        auto *merge = sw->create_merge_block();
        b.set_insertion_point(case_block);
        b.br(merge);
        b.set_insertion_point(default_block);
        b.br(merge);
        b.set_insertion_point(merge);
        b.return_void();

        auto lower_info = lower_switch_pass_run_on_module(&m);
        auto destructure_info = destructure_cfg_pass_run_on_module(&m);
        auto if_conversion_info = if_conversion_pass_run_on_module(&m);
        expect(lower_info.succeeded());
        expect(lower_info.lowered_switch_count == 1u);
        expect(destructure_info.succeeded());
        expect(if_conversion_info.succeeded());
        expect(count_terminator_kind(k->definition(), DerivedInstructionTag::SWITCH) == 0u);
        expect(count_terminator_kind(k->definition(), DerivedInstructionTag::IF) == 0u);
        expect(xir_verify_module(&m).succeeded());
    };

    "restructure_reports_unhandled_raw_conditional"_test = [] {
        Module m;
        BasicBlock *body;
        auto *k = make_kernel_with_body(m, body);
        auto *condition = k->create_value_argument(Type::of<bool>());
        auto *return_block = k->create_basic_block();
        XIRBuilder b;
        b.set_insertion_point(body);
        auto *branch = b.cond_br(condition, return_block, nullptr);
        b.set_insertion_point(return_block);
        b.return_void();

        auto info = restructure_cfg_pass_run_on_function(k);
        expect(!info.succeeded());
        expect(info.unstructured_branch_count == 1u);
        expect(info.invalid_construct_count >= 1u);
        expect(body->terminator() == branch);
    };
}

int main(int argc, char *argv[]) {
    boost::ut::detail::cfg::parse_arg_with_fallback(argc, const_cast<const char **>(argv));
    reg_restructure_cfg();
    return 0;
}
