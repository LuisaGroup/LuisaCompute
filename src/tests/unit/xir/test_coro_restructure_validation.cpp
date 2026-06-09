#include "ut/ut.hpp"
#include <luisa/ast/type_registry.h>
#include <luisa/xir/basic_block.h>
#include <luisa/xir/builder.h>
#include <luisa/xir/function.h>
#include <luisa/xir/instructions/branch.h>
#include <luisa/xir/instructions/coro.h>
#include <luisa/xir/instructions/if.h>
#include <luisa/xir/instructions/loop.h>
#include <luisa/xir/instructions/return.h>
#include <luisa/xir/module.h>
#include <luisa/xir/passes/coro_split.h>
#include <luisa/xir/passes/coro_reg2mem.h>
#include <luisa/xir/passes/restructure_cfg.h>

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

[[nodiscard]] size_t count_callable_count(Module &m) noexcept {
    size_t n = 0u;
    for (auto *f : m.function_list()) {
        if (f->isa<CallableFunction>()) { n++; }
    }
    return n;
}

[[nodiscard]] bool callable_is_structured(FunctionDefinition *def) noexcept {
    bool ok = true;
    def->traverse_basic_blocks([&](BasicBlock *bb) noexcept {
        if (!bb->is_terminated()) {
            ok = false;
            return;
        }
        auto tag = bb->terminator()->derived_instruction_tag();
        if (tag == DerivedInstructionTag::BRANCH ||
            tag == DerivedInstructionTag::CONDITIONAL_BRANCH) {
            ok = false;
        }
    });
    return ok;
}

void verify_structured_callables(Module &m) noexcept {
    for (auto *f : m.function_list()) {
        if (!f->isa<CallableFunction>()) { continue; }
        auto *def = f->definition();
        if (def == nullptr) { continue; }
        auto structured = callable_is_structured(def);
        auto br_count = count_terminator_kind(def, DerivedInstructionTag::BRANCH);
        auto cbr_count = count_terminator_kind(def, DerivedInstructionTag::CONDITIONAL_BRANCH);
        auto if_count = count_terminator_kind(def, DerivedInstructionTag::IF);
        auto loop_count = count_terminator_kind(def, DerivedInstructionTag::LOOP);
        expect(structured)
            << "callable has unstructured terminators"
            << " br=" << br_count << " cbr=" << cbr_count
            << " if=" << if_count << " loop=" << loop_count;
        def->traverse_basic_blocks([&](BasicBlock *bb) noexcept {
            expect(bb->is_terminated()) << "block not terminated";
        });
    }
}

void verify_callable_has_unstructured_remainder(Module &m,
                                                size_t expected_br,
                                                size_t expected_cbr,
                                                size_t expected_if,
                                                size_t expected_loop) noexcept {
    bool found = false;
    for (auto *f : m.function_list()) {
        if (!f->isa<CallableFunction>()) { continue; }
        auto *def = f->definition();
        if (def == nullptr) { continue; }
        auto br_count = count_terminator_kind(def, DerivedInstructionTag::BRANCH);
        auto cbr_count = count_terminator_kind(def, DerivedInstructionTag::CONDITIONAL_BRANCH);
        auto if_count = count_terminator_kind(def, DerivedInstructionTag::IF);
        auto loop_count = count_terminator_kind(def, DerivedInstructionTag::LOOP);
        if (br_count == expected_br && cbr_count == expected_cbr) {
            if (if_count == expected_if && loop_count == expected_loop) {
                found = true;
                break;
            }
        }
    }
    expect(found) << "no callable matched expected unstructured shape"
                  << " br=" << expected_br << " cbr=" << expected_cbr
                  << " if=" << expected_if << " loop=" << expected_loop;
}

void run_coro_pipeline_through_restructure(Module &m) noexcept {
    auto split_count = coro_split_pass_run_on_module(&m);
    auto reg2mem_info = coro_reg2mem_pass_run_on_module(&m);
    (void)split_count;
    (void)reg2mem_info;
    auto restructure_info = restructure_cfg_pass_run_on_module(&m);
    static_cast<void>(restructure_info);
}

}// namespace

void reg_coro_restructure_validation() {

    "simple_split_restructure"_test = [] {
        // given: a coroutine kernel with one suspend point
        Module m;
        BasicBlock *body;
        auto *k = make_kernel_with_body(m, body);
        XIRBuilder b;
        auto *cond = m.create_constant_one(Type::of<bool>());

        auto *suspend_bb = k->create_basic_block();
        auto *resume_bb = k->create_basic_block();

        b.set_insertion_point(body);
        b.cond_br(cond, suspend_bb, resume_bb);

        b.set_insertion_point(suspend_bb);
        b.coro_suspend(1u, "simple", nullptr);

        b.set_insertion_point(resume_bb);
        b.coro_resume(1u, nullptr);
        b.return_void();

        run_coro_pipeline_through_restructure(m);

        expect(count_callable_count(m) >= 2u);
        verify_structured_callables(m);
    };

    "three_suspend_restructure"_test = [] {
        // given: three suspend points generating multiple continuations
        Module m;
        BasicBlock *body;
        auto *k = make_kernel_with_body(m, body);
        XIRBuilder b;
        auto *cond = m.create_constant_one(Type::of<bool>());

        auto *s1 = k->create_basic_block();
        auto *r1 = k->create_basic_block();
        b.set_insertion_point(body);
        b.cond_br(cond, s1, r1);
        b.set_insertion_point(s1);
        b.coro_suspend(1u, "s1", nullptr);
        b.set_insertion_point(r1);
        b.coro_resume(1u, nullptr);

        auto *s2 = k->create_basic_block();
        auto *r2 = k->create_basic_block();
        b.cond_br(cond, s2, r2);
        b.set_insertion_point(s2);
        b.coro_suspend(2u, "s2", nullptr);
        b.set_insertion_point(r2);
        b.coro_resume(2u, nullptr);

        auto *s3 = k->create_basic_block();
        auto *r3 = k->create_basic_block();
        b.cond_br(cond, s3, r3);
        b.set_insertion_point(s3);
        b.coro_suspend(3u, "s3", nullptr);
        b.set_insertion_point(r3);
        b.coro_resume(3u, nullptr);
        b.return_void();

        run_coro_pipeline_through_restructure(m);

        expect(count_callable_count(m) == 4u);
        verify_structured_callables(m);
    };

    "adjacent_suspend_restructure"_test = [] {
        // given: two suspends back-to-back
        Module m;
        BasicBlock *body;
        auto *k = make_kernel_with_body(m, body);
        XIRBuilder b;
        auto *cond = m.create_constant_one(Type::of<bool>());

        auto *s1 = k->create_basic_block();
        auto *r1 = k->create_basic_block();

        b.set_insertion_point(body);
        b.cond_br(cond, s1, r1);
        b.set_insertion_point(s1);
        b.coro_suspend(1u, "first", nullptr);

        b.set_insertion_point(r1);
        b.coro_resume(1u, nullptr);

        auto *s2 = k->create_basic_block();
        auto *r2 = k->create_basic_block();
        b.cond_br(cond, s2, r2);
        b.set_insertion_point(s2);
        b.coro_suspend(2u, "second", nullptr);
        b.set_insertion_point(r2);
        b.coro_resume(2u, nullptr);
        b.return_void();

        run_coro_pipeline_through_restructure(m);

        expect(count_callable_count(m) == 3u);
        verify_structured_callables(m);
    };

    "restructure_requirements_met"_test = [] {
        // given: a single-suspend coroutine; verify coro_reg2mem then
        // restructure_cfg order produces phi-free structured output
        Module m;
        BasicBlock *body;
        auto *k = make_kernel_with_body(m, body);
        XIRBuilder b;
        auto *cond = m.create_constant_one(Type::of<bool>());

        auto *suspend_bb = k->create_basic_block();
        auto *resume_bb = k->create_basic_block();

        b.set_insertion_point(body);
        b.cond_br(cond, suspend_bb, resume_bb);

        b.set_insertion_point(suspend_bb);
        b.coro_suspend(1u, "req", nullptr);

        b.set_insertion_point(resume_bb);
        b.coro_resume(1u, nullptr);
        b.return_void();

        auto split_count = coro_split_pass_run_on_module(&m);
        expect(split_count == 2u);

        auto reg2mem_info = coro_reg2mem_pass_run_on_module(&m);
        expect(reg2mem_info.callable_count >= 1u);

        auto info = restructure_cfg_pass_run_on_module(&m);
        expect(info.restructured_if_count + info.restructured_loop_count > 0u);
        expect(info.irreducible_region_count == 0u);

        verify_structured_callables(m);
    };

    "conditional_suspend_restructure_failure"_test = [] {
        // FAILURE DOCUMENTED: suspending inside one branch of a conditional
        // creates cross-scope fallback blocks that restructure_cfg cannot
        // fully structurize. The skip-check cond_br is converted to an
        // IfInst, but residual BranchInst remain (br=1, cbr=0, if=2, loop=0).
        Module m;
        BasicBlock *body;
        auto *k = make_kernel_with_body(m, body);
        XIRBuilder b;
        auto *always_true = m.create_constant_one(Type::of<bool>());

        auto *branch_a = k->create_basic_block();
        auto *branch_b = k->create_basic_block();
        auto *merge = k->create_basic_block();
        auto *s1 = k->create_basic_block();
        auto *r1 = k->create_basic_block();

        b.set_insertion_point(body);
        b.cond_br(always_true, branch_a, branch_b);

        b.set_insertion_point(branch_a);
        b.cond_br(always_true, s1, r1);
        b.set_insertion_point(s1);
        b.coro_suspend(1u, "cond_suspend", nullptr);

        b.set_insertion_point(r1);
        b.coro_resume(1u, nullptr);
        b.br(merge);

        b.set_insertion_point(branch_b);
        b.br(merge);

        b.set_insertion_point(merge);
        b.return_void();

        run_coro_pipeline_through_restructure(m);

        expect(count_callable_count(m) >= 1u);
        verify_callable_has_unstructured_remainder(m, 1u, 0u, 2u, 0u);
        verify_callable_has_unstructured_remainder(m, 1u, 0u, 1u, 0u);
    };

    "loop_suspend_restructure_failure"_test = [] {
        // FAILURE DOCUMENTED: suspending inside a loop body creates a
        // continuation whose back-edge is preserved through the split.
        // The skip-check becomes an IfInst but the loop itself is NOT
        // restructured into a LoopInst (loop=0). Residual BranchInst
        // remain from the loop header→body and latch→header edges.
        Module m;
        BasicBlock *body;
        auto *k = make_kernel_with_body(m, body);
        XIRBuilder b;
        auto *always_true = m.create_constant_one(Type::of<bool>());
        auto *loop_cond = m.create_constant_one(Type::of<bool>());

        auto *loop_hdr = k->create_basic_block();
        auto *loop_body = k->create_basic_block();
        auto *s1 = k->create_basic_block();
        auto *r1 = k->create_basic_block();
        auto *exit = k->create_basic_block();

        b.set_insertion_point(body);
        b.br(loop_hdr);

        b.set_insertion_point(loop_hdr);
        b.cond_br(loop_cond, loop_body, exit);

        b.set_insertion_point(loop_body);
        b.cond_br(always_true, s1, r1);

        b.set_insertion_point(s1);
        b.coro_suspend(1u, "loop_suspend", nullptr);

        b.set_insertion_point(r1);
        b.coro_resume(1u, nullptr);
        b.br(loop_hdr);

        b.set_insertion_point(exit);
        b.return_void();

        run_coro_pipeline_through_restructure(m);

        expect(count_callable_count(m) >= 1u);
        verify_callable_has_unstructured_remainder(m, 1u, 0u, 2u, 0u);
        verify_callable_has_unstructured_remainder(m, 1u, 0u, 1u, 0u);
    };

    "suspend_after_break_restructure_failure"_test = [] {
        // FAILURE DOCUMENTED: a loop with a break path, then a suspend
        // after loop exit. The split creates continuations with complex
        // cross-scope edges. restructure_cfg partially structurizes
        // (some IfInst, one LoopInst) but leaves many BranchInst and
        // ConditionalBranchInst behind (br=5, cbr=1, if=2, loop=1).
        Module m;
        BasicBlock *body;
        auto *k = make_kernel_with_body(m, body);
        XIRBuilder b;
        auto *cond = m.create_constant_one(Type::of<bool>());

        auto *loop_hdr = k->create_basic_block();
        auto *loop_body = k->create_basic_block();
        auto *break_bb = k->create_basic_block();
        auto *loop_latch = k->create_basic_block();
        auto *loop_exit = k->create_basic_block();
        auto *s1 = k->create_basic_block();
        auto *r1 = k->create_basic_block();

        b.set_insertion_point(body);
        b.br(loop_hdr);

        b.set_insertion_point(loop_hdr);
        b.cond_br(cond, loop_body, loop_exit);

        b.set_insertion_point(loop_body);
        b.cond_br(cond, break_bb, loop_latch);

        b.set_insertion_point(break_bb);
        b.br(loop_exit);

        b.set_insertion_point(loop_latch);
        b.br(loop_hdr);

        b.set_insertion_point(loop_exit);
        b.cond_br(cond, s1, r1);

        b.set_insertion_point(s1);
        b.coro_suspend(1u, "after_break", nullptr);

        b.set_insertion_point(r1);
        b.coro_resume(1u, nullptr);
        b.return_void();

        run_coro_pipeline_through_restructure(m);

        expect(count_callable_count(m) >= 1u);
        verify_callable_has_unstructured_remainder(m, 5u, 1u, 2u, 1u);
    };

    "nested_loop_suspend_restructure_failure"_test = [] {
        // FAILURE DOCUMENTED: suspending inside a nested loop body creates
        // multi-level back-edge complexity after split. The outer/inner
        // loops are NOT restructured (loop=0). restructure_cfg creates
        // IfInst from the skip-checks but leaves all loop BranchInst.
        Module m;
        BasicBlock *body;
        auto *k = make_kernel_with_body(m, body);
        XIRBuilder b;
        auto *cond = m.create_constant_one(Type::of<bool>());

        auto *outer_hdr = k->create_basic_block();
        auto *outer_body = k->create_basic_block();
        auto *outer_latch = k->create_basic_block();
        auto *outer_exit = k->create_basic_block();

        auto *inner_hdr = k->create_basic_block();
        auto *inner_body = k->create_basic_block();
        auto *inner_latch = k->create_basic_block();
        auto *inner_exit = k->create_basic_block();

        auto *s1 = k->create_basic_block();
        auto *r1 = k->create_basic_block();

        b.set_insertion_point(body);
        b.br(outer_hdr);

        b.set_insertion_point(outer_hdr);
        b.cond_br(cond, outer_body, outer_exit);

        b.set_insertion_point(outer_body);
        b.br(inner_hdr);

        b.set_insertion_point(inner_hdr);
        b.cond_br(cond, inner_body, inner_exit);

        b.set_insertion_point(inner_body);
        b.cond_br(cond, s1, r1);

        b.set_insertion_point(s1);
        b.coro_suspend(1u, "nested", nullptr);

        b.set_insertion_point(r1);
        b.coro_resume(1u, nullptr);
        b.br(inner_latch);

        b.set_insertion_point(inner_latch);
        b.br(inner_hdr);

        b.set_insertion_point(inner_exit);
        b.br(outer_latch);

        b.set_insertion_point(outer_latch);
        b.br(outer_hdr);

        b.set_insertion_point(outer_exit);
        b.return_void();

        run_coro_pipeline_through_restructure(m);

        expect(count_callable_count(m) >= 1u);
        verify_callable_has_unstructured_remainder(m, 4u, 0u, 3u, 0u);
        verify_callable_has_unstructured_remainder(m, 4u, 0u, 3u, 0u);
        verify_callable_has_unstructured_remainder(m, 2u, 0u, 1u, 0u);
    };

    "branch_targets_remapped_restructure_failure"_test = [] {
        // FAILURE DOCUMENTED: a loop-back within a single scope creates
        // a self-loop after the skip-check. restructure_cfg converts the
        // skip-check to an IfInst but leaves the loop-back BranchInst
        // and the loop-header ConditionalBranchInst unreconstructed.
        Module m;
        BasicBlock *body;
        auto *k = make_kernel_with_body(m, body);
        XIRBuilder b;
        auto *cond = m.create_constant_one(Type::of<bool>());

        auto *suspend_bb = k->create_basic_block();
        auto *resume_bb = k->create_basic_block();

        b.set_insertion_point(body);
        b.cond_br(cond, suspend_bb, resume_bb);

        b.set_insertion_point(suspend_bb);
        b.coro_suspend(10u, "loop_check", nullptr);

        b.set_insertion_point(resume_bb);
        b.coro_resume(10u, nullptr);

        auto *loop_body = k->create_basic_block();
        auto *loop_exit = k->create_basic_block();
        b.cond_br(cond, loop_body, loop_exit);

        b.set_insertion_point(loop_body);
        b.br(resume_bb);

        b.set_insertion_point(loop_exit);
        b.return_void();

        run_coro_pipeline_through_restructure(m);

        expect(count_callable_count(m) >= 1u);
        verify_callable_has_unstructured_remainder(m, 3u, 1u, 1u, 0u);
    };

    "terminal_coro_restructure_failure"_test = [] {
        // FAILURE DOCUMENTED: a coroutine ending with CoroTerminateInst
        // creates a terminal scope whose skip-check is structurized to
        // an IfInst, but a residual BranchInst to the terminate block
        // remains unreconstructed.
        Module m;
        BasicBlock *body;
        auto *k = make_kernel_with_body(m, body);
        XIRBuilder b;
        auto *cond = m.create_constant_one(Type::of<bool>());

        auto *suspend_bb = k->create_basic_block();
        auto *resume_bb = k->create_basic_block();

        b.set_insertion_point(body);
        b.cond_br(cond, suspend_bb, resume_bb);

        b.set_insertion_point(suspend_bb);
        b.coro_suspend(1u, "mid", nullptr);

        b.set_insertion_point(resume_bb);
        b.coro_resume(1u, nullptr);

        auto *term_bb = k->create_basic_block();
        b.br(term_bb);

        b.set_insertion_point(term_bb);
        b.coro_terminate();

        run_coro_pipeline_through_restructure(m);

        expect(count_callable_count(m) >= 2u);
        verify_callable_has_unstructured_remainder(m, 1u, 0u, 1u, 0u);
    };

}

int main(int argc, char *argv[]) {
    (void)argc;
    (void)argv;
    reg_coro_restructure_validation();
    return 0;
}
