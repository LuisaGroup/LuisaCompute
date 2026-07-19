// Test for lowering structured break and continue instructions.

#include "ut/ut.hpp"
#include <luisa/ast/type_registry.h>
#include <luisa/xir/basic_block.h>
#include <luisa/xir/builder.h>
#include <luisa/xir/function.h>
#include <luisa/xir/instructions/branch.h>
#include <luisa/xir/instructions/break.h>
#include <luisa/xir/instructions/continue.h>
#include <luisa/xir/instructions/loop.h>
#include <luisa/xir/module.h>
#include <luisa/xir/passes/lower_break_continue.h>

using namespace luisa;
using namespace luisa::compute;
using namespace luisa::compute::xir;
using namespace boost::ut;
using namespace boost::ut::literals;

namespace {

struct LoopSkeleton {
    LoopInst *loop;
    BasicBlock *prepare;
    BasicBlock *body;
    BasicBlock *update;
    BasicBlock *merge;
};

[[nodiscard]] LoopSkeleton make_loop_skeleton(XIRBuilder &b) noexcept {
    auto *loop = b.loop();
    LoopSkeleton s{};
    s.loop = loop;
    s.prepare = loop->create_prepare_block();
    s.body = loop->create_body_block();
    s.update = loop->create_update_block();
    s.merge = loop->create_merge_block();
    return s;
}

[[nodiscard]] KernelFunction *make_kernel_with_body(Module &m, BasicBlock *&body_out) noexcept {
    auto *k = m.create_kernel();
    body_out = k->create_body_block();
    return k;
}

[[nodiscard]] size_t count_break_or_continue(FunctionDefinition *def) noexcept {
    size_t n = 0u;
    def->traverse_basic_blocks([&](BasicBlock *bb) noexcept {
        if (!bb->is_terminated()) { return; }
        auto t = bb->terminator();
        if (t->isa<BreakInst>() || t->isa<ContinueInst>()) { ++n; }
    });
    return n;
}

[[nodiscard]] size_t count_branches_to(FunctionDefinition *def, BasicBlock *target) noexcept {
    size_t n = 0u;
    def->traverse_basic_blocks([&](BasicBlock *bb) noexcept {
        if (!bb->is_terminated()) { return; }
        auto t = bb->terminator();
        if (t->isa<BranchInst>()) {
            auto br = static_cast<BranchInst *>(t);
            if (br->target_block() == target) { ++n; }
        }
    });
    return n;
}

}// namespace

void reg_lower_break_continue() {

    "lower_bc_noop_when_no_break_or_continue"_test = [] {
        Module m;
        BasicBlock *body;
        auto *k = make_kernel_with_body(m, body);
        XIRBuilder b;
        b.set_insertion_point(body);
        b.return_void();
        auto info = lower_break_continue_pass_run_on_function(k);
        expect(info.lowered_break_count == 0u);
        expect(info.lowered_continue_count == 0u);
    };

    "lower_bc_single_break_in_loop"_test = [] {
        Module m;
        BasicBlock *body;
        auto *k = make_kernel_with_body(m, body);
        XIRBuilder b;
        b.set_insertion_point(body);
        auto s = make_loop_skeleton(b);
        b.set_insertion_point(s.prepare);
        b.br(s.body);
        b.set_insertion_point(s.body);
        b.break_(s.merge);
        b.set_insertion_point(s.update);
        b.br(s.prepare);
        b.set_insertion_point(s.merge);
        b.return_void();

        auto info = lower_break_continue_pass_run_on_function(k);
        expect(info.lowered_break_count == 1u);
        expect(info.lowered_continue_count == 0u);
        auto *def = k->definition();
        expect(count_break_or_continue(def) == 0u);
        expect(count_branches_to(def, s.merge) >= 1u);
        expect(s.body->is_terminated() == true);
        expect(s.body->terminator()->isa<BranchInst>());
    };

    "lower_bc_single_continue_in_loop"_test = [] {
        Module m;
        BasicBlock *body;
        auto *k = make_kernel_with_body(m, body);
        XIRBuilder b;
        b.set_insertion_point(body);
        auto s = make_loop_skeleton(b);
        b.set_insertion_point(s.prepare);
        b.br(s.body);
        b.set_insertion_point(s.body);
        b.continue_(s.update);
        b.set_insertion_point(s.update);
        b.br(s.prepare);
        b.set_insertion_point(s.merge);
        b.return_void();

        auto info = lower_break_continue_pass_run_on_function(k);
        expect(info.lowered_break_count == 0u);
        expect(info.lowered_continue_count == 1u);
        auto *def = k->definition();
        expect(count_break_or_continue(def) == 0u);
        expect(count_branches_to(def, s.update) >= 1u);
        expect(s.body->terminator()->isa<BranchInst>());
    };

    "lower_bc_break_and_continue_same_loop"_test = [] {
        Module m;
        BasicBlock *body;
        auto *k = make_kernel_with_body(m, body);
        XIRBuilder b;
        b.set_insertion_point(body);
        auto s = make_loop_skeleton(b);
        auto *brk_bb = k->create_basic_block();
        auto *cont_bb = k->create_basic_block();
        auto *cond = m.create_constant_one(Type::of<bool>());
        b.set_insertion_point(s.prepare);
        b.br(s.body);
        b.set_insertion_point(s.body);
        b.cond_br(cond, brk_bb, cont_bb);
        b.set_insertion_point(brk_bb);
        b.break_(s.merge);
        b.set_insertion_point(cont_bb);
        b.continue_(s.update);
        b.set_insertion_point(s.update);
        b.br(s.prepare);
        b.set_insertion_point(s.merge);
        b.return_void();

        auto info = lower_break_continue_pass_run_on_function(k);
        expect(info.lowered_break_count == 1u);
        expect(info.lowered_continue_count == 1u);
        auto *def = k->definition();
        expect(count_break_or_continue(def) == 0u);
        expect(brk_bb->terminator()->isa<BranchInst>());
        expect(cont_bb->terminator()->isa<BranchInst>());
    };

    "lower_bc_multiple_breaks_one_loop"_test = [] {
        Module m;
        BasicBlock *body;
        auto *k = make_kernel_with_body(m, body);
        XIRBuilder b;
        b.set_insertion_point(body);
        auto s = make_loop_skeleton(b);
        constexpr size_t kBreaks = 5u;
        luisa::vector<BasicBlock *> brk_blocks;
        luisa::vector<BasicBlock *> dispatch_blocks;
        brk_blocks.reserve(kBreaks);
        dispatch_blocks.reserve(kBreaks);
        for (size_t i = 0; i < kBreaks; ++i) {
            brk_blocks.emplace_back(k->create_basic_block());
            dispatch_blocks.emplace_back(k->create_basic_block());
        }
        auto *cond = m.create_constant_one(Type::of<bool>());
        b.set_insertion_point(s.prepare);
        b.br(s.body);
        b.set_insertion_point(s.body);
        b.br(dispatch_blocks[0]);
        for (size_t i = 0; i < kBreaks; ++i) {
            b.set_insertion_point(dispatch_blocks[i]);
            auto *next = (i + 1u < kBreaks) ? dispatch_blocks[i + 1u] : s.update;
            b.cond_br(cond, brk_blocks[i], next);
            b.set_insertion_point(brk_blocks[i]);
            b.break_(s.merge);
        }
        b.set_insertion_point(s.update);
        b.br(s.prepare);
        b.set_insertion_point(s.merge);
        b.return_void();

        auto info = lower_break_continue_pass_run_on_function(k);
        expect(info.lowered_break_count == kBreaks);
        expect(info.lowered_continue_count == 0u);
        auto *def = k->definition();
        expect(count_break_or_continue(def) == 0u);
        for (auto *bb : brk_blocks) {
            expect(bb->terminator()->isa<BranchInst>());
        }
    };

    "lower_bc_nested_loops_break_targets_inner_then_outer"_test = [] {
        Module m;
        BasicBlock *body;
        auto *k = make_kernel_with_body(m, body);
        XIRBuilder b;
        b.set_insertion_point(body);

        auto outer = make_loop_skeleton(b);
        b.set_insertion_point(outer.prepare);
        b.br(outer.body);

        b.set_insertion_point(outer.body);
        auto inner = make_loop_skeleton(b);
        b.set_insertion_point(outer.update);
        b.br(outer.prepare);
        b.set_insertion_point(outer.merge);
        b.return_void();

        b.set_insertion_point(inner.prepare);
        b.br(inner.body);
        b.set_insertion_point(inner.body);
        b.break_(inner.merge);
        b.set_insertion_point(inner.update);
        b.br(inner.prepare);
        b.set_insertion_point(inner.merge);
        b.break_(outer.merge);

        auto info = lower_break_continue_pass_run_on_function(k);
        expect(info.lowered_break_count == 2u);
        expect(info.lowered_continue_count == 0u);
        auto *def = k->definition();
        expect(count_break_or_continue(def) == 0u);
        expect(inner.body->terminator()->isa<BranchInst>());
        expect(inner.merge->terminator()->isa<BranchInst>());
        expect(count_branches_to(def, inner.merge) >= 1u);
        expect(count_branches_to(def, outer.merge) >= 1u);
    };

    "lower_bc_module_runs_all_functions"_test = [] {
        Module m;
        constexpr size_t kFns = 3u;
        luisa::vector<BasicBlock *> merges;
        for (size_t i = 0; i < kFns; ++i) {
            BasicBlock *body;
            auto *k = make_kernel_with_body(m, body);
            XIRBuilder b;
            b.set_insertion_point(body);
            auto s = make_loop_skeleton(b);
            merges.emplace_back(s.merge);
            b.set_insertion_point(s.prepare);
            b.br(s.body);
            b.set_insertion_point(s.body);
            b.break_(s.merge);
            b.set_insertion_point(s.update);
            b.br(s.prepare);
            b.set_insertion_point(s.merge);
            b.return_void();
        }
        auto info = lower_break_continue_pass_run_on_module(&m);
        expect(info.lowered_break_count == kFns);
        expect(info.lowered_continue_count == 0u);
        for (auto f : m.function_list()) {
            auto def = f->definition();
            if (def == nullptr) { continue; }
            expect(count_break_or_continue(def) == 0u);
        }
    };

    "lower_bc_external_function_skipped"_test = [] {
        Module m;
        auto *ext = m.create_external_function(Type::of<void>());
        auto info = lower_break_continue_pass_run_on_function(ext);
        expect(info.lowered_break_count == 0u);
        expect(info.lowered_continue_count == 0u);
    };

    "lower_bc_idempotent"_test = [] {
        Module m;
        BasicBlock *body;
        auto *k = make_kernel_with_body(m, body);
        XIRBuilder b;
        b.set_insertion_point(body);
        auto s = make_loop_skeleton(b);
        b.set_insertion_point(s.prepare);
        b.br(s.body);
        b.set_insertion_point(s.body);
        b.break_(s.merge);
        b.set_insertion_point(s.update);
        b.br(s.prepare);
        b.set_insertion_point(s.merge);
        b.return_void();

        auto first = lower_break_continue_pass_run_on_function(k);
        auto second = lower_break_continue_pass_run_on_function(k);
        expect(first.lowered_break_count == 1u);
        expect(first.lowered_continue_count == 0u);
        expect(second.lowered_break_count == 0u);
        expect(second.lowered_continue_count == 0u);
    };

    "lower_bc_branch_target_preserved"_test = [] {
        Module m;
        BasicBlock *body;
        auto *k = make_kernel_with_body(m, body);
        XIRBuilder b;
        b.set_insertion_point(body);
        auto s = make_loop_skeleton(b);
        b.set_insertion_point(s.prepare);
        b.br(s.body);
        b.set_insertion_point(s.body);
        b.break_(s.merge);
        b.set_insertion_point(s.update);
        b.br(s.prepare);
        b.set_insertion_point(s.merge);
        b.return_void();

        (void)lower_break_continue_pass_run_on_function(k);
        auto *new_term = s.body->terminator();
        expect(new_term->isa<BranchInst>());
        auto *br = static_cast<BranchInst *>(new_term);
        expect(br->target_block() == s.merge);
        expect(br->is_terminator() == true);
    };

    "lower_bc_does_not_touch_plain_branch"_test = [] {
        Module m;
        BasicBlock *body;
        auto *k = make_kernel_with_body(m, body);
        XIRBuilder b;
        b.set_insertion_point(body);
        auto s = make_loop_skeleton(b);
        b.set_insertion_point(s.prepare);
        b.br(s.body);
        b.set_insertion_point(s.body);
        b.br(s.update);
        b.set_insertion_point(s.update);
        b.br(s.prepare);
        b.set_insertion_point(s.merge);
        b.return_void();

        auto *body_term_before = s.body->terminator();
        auto info = lower_break_continue_pass_run_on_function(k);
        expect(info.lowered_break_count == 0u);
        expect(info.lowered_continue_count == 0u);
        expect(s.body->terminator() == body_term_before);
    };

    "lower_bc_empty_module_runs_cleanly"_test = [] {
        Module m;
        auto info = lower_break_continue_pass_run_on_module(&m);
        expect(info.lowered_break_count == 0u);
        expect(info.lowered_continue_count == 0u);
    };

    "lower_bc_null_target_rejects_function_atomically"_test = [] {
        Module m;
        BasicBlock *body;
        auto *k = make_kernel_with_body(m, body);
        auto *valid_block = k->create_basic_block();
        auto *target = k->create_basic_block();
        XIRBuilder b;
        b.set_insertion_point(body);
        auto *invalid_break = b.break_(nullptr);
        b.set_insertion_point(valid_block);
        auto *valid_continue = b.continue_(target);
        b.set_insertion_point(target);
        b.return_void();

        auto info = lower_break_continue_pass_run_on_function(k);
        expect(!info.succeeded());
        expect(info.rejected_break_count == 1u);
        expect(info.rejected_continue_count == 0u);
        expect(info.lowered_break_count == 0u);
        expect(info.lowered_continue_count == 0u);
        expect(body->terminator() == invalid_break);
        expect(valid_block->terminator() == valid_continue);
    };

    "lower_bc_foreign_target_rejects_module_atomically"_test = [] {
        Module m;
        BasicBlock *body0;
        BasicBlock *body1;
        auto *k0 = make_kernel_with_body(m, body0);
        auto *k1 = make_kernel_with_body(m, body1);
        auto *target0 = k0->create_basic_block();
        XIRBuilder b;
        b.set_insertion_point(body0);
        auto *valid_break = b.break_(target0);
        b.set_insertion_point(target0);
        b.return_void();
        b.set_insertion_point(body1);
        auto *foreign_continue = b.continue_(target0);

        auto info = lower_break_continue_pass_run_on_module(&m);
        expect(!info.succeeded());
        expect(info.rejected_continue_count == 1u);
        expect(info.lowered_break_count == 0u);
        expect(info.lowered_continue_count == 0u);
        expect(body0->terminator() == valid_break);
        expect(body1->terminator() == foreign_continue);
        expect(target0->parent_function() != k1);
    };
}

int main(int argc, char *argv[]) {
    boost::ut::detail::cfg::parse_arg_with_fallback(argc, const_cast<const char **>(argv));
    reg_lower_break_continue();
    return 0;
}
