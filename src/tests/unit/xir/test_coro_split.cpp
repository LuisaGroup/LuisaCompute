// Test for coroutine splitting, state transitions, and malformed-input rejection.

#include "ut/ut.hpp"
#include <luisa/ast/type_registry.h>
#include <luisa/dsl/coro_frame.h>
#include <luisa/xir/basic_block.h>
#include <luisa/xir/builder.h>
#include <luisa/xir/function.h>
#include <luisa/xir/instructions/arithmetic.h>
#include <luisa/xir/instructions/branch.h>
#include <luisa/xir/instructions/coro.h>
#include <luisa/xir/instructions/clock.h>
#include <luisa/xir/instructions/loop.h>
#include <luisa/xir/instructions/return.h>
#include <luisa/xir/instructions/store.h>
#include <luisa/xir/instructions/switch.h>
#include <luisa/xir/metadata/signature_constraint.h>
#include <luisa/xir/module.h>
#include <luisa/xir/passes/coro_cfg_distill.h>
#include <luisa/xir/passes/coro_materialize.h>
#include <luisa/xir/passes/coro_split.h>
#include <luisa/xir/passes/destructure_cfg.h>
#include <luisa/xir/passes/dom_tree.h>
#include <luisa/xir/verifier.h>

#include <algorithm>

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

size_t count_callables(Module &m) noexcept {
    size_t n = 0u;
    for (auto *f : m.function_list()) {
        if (f->isa<CallableFunction>()) { n++; }
    }
    return n;
}

size_t count_instructions_with_tag(Module &m, DerivedInstructionTag tag) noexcept {
    size_t n = 0u;
    for (auto *f : m.function_list()) {
        if (auto *def = f->definition()) {
            def->traverse_instructions([&](Instruction *inst) noexcept {
                if (inst->derived_instruction_tag() == tag) { n++; }
            });
        }
    }
    return n;
}

bool has_frame_arg(CallableFunction *cf) noexcept {
    auto &args = cf->arguments();
    auto *first = args.front();
    return first != nullptr && !first->is_sentinel() && first->is_reference();
}

bool has_return(Module &m) noexcept {
    for (auto *f : m.function_list()) {
        if (auto *def = f->definition()) {
            bool found = false;
            def->traverse_instructions([&](Instruction *inst) noexcept {
                if (inst->derived_instruction_tag() == DerivedInstructionTag::RETURN) { found = true; }
            });
            if (found) { return true; }
        }
    }
    return false;
}

struct StructuredSwitchCoroutine {
    KernelFunction *function;
    SwitchInst *switch_inst;
    BasicBlock *resume_block;
    BasicBlock *merge_block;
};

[[nodiscard]] StructuredSwitchCoroutine make_structured_switch_coroutine(Module &m,
                                                                         uint32_t token) noexcept {
    BasicBlock *body;
    auto *k = make_kernel_with_body(m, body);
    XIRBuilder b;
    b.set_insertion_point(body);
    auto *selector = m.create_constant_zero(Type::of<int>());
    auto *sw = b.switch_(selector);
    auto *case_block = sw->create_case_block(0);
    auto *default_block = sw->create_default_block();
    auto *merge_block = sw->create_merge_block();
    auto *resume_block = k->create_basic_block();

    b.set_insertion_point(case_block);
    b.coro_suspend(token, "structured-switch", nullptr);
    b.set_insertion_point(default_block);
    b.br(merge_block);
    b.set_insertion_point(resume_block);
    b.coro_resume(token, nullptr);
    b.br(merge_block);
    b.set_insertion_point(merge_block);
    b.return_void();
    return {k, sw, resume_block, merge_block};
}

}// namespace

void reg_coro_split() {

    "no_coroutine_no_split"_test = [] {
        // given: a function with no coroutine instructions
        Module m;
        BasicBlock *body;
        auto *k = make_kernel_with_body(m, body);
        XIRBuilder b;
        b.set_insertion_point(body);
        b.return_void();

        // when
        auto count = coro_split_pass_run_on_module(&m);

        // then: 0 functions created
        expect(count == 0u);
        expect(count_callables(m) == 0u);
    };

    "single_suspend_creates_two_callables"_test = [] {
        // given: a function with one CoroSuspendInst
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
        b.coro_suspend(42u, "checkpoint", nullptr);

        b.set_insertion_point(resume_bb);
        b.coro_resume(42u, nullptr);
        b.return_void();

        // when
        auto count = coro_split_pass_run_on_module(&m);

        // then: 2 callables created
        expect(count == 2u);
        expect(count_callables(m) == 2u);
    };

    "created_functions_have_frame_parameter"_test = [] {
        // given: a single-suspend coroutine
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
        b.coro_suspend(1u, "a", nullptr);

        b.set_insertion_point(resume_bb);
        b.coro_resume(1u, nullptr);
        b.return_void();

        // when
        auto frame_count = coro_split_pass_run_on_module(&m);
        static_cast<void>(frame_count);

        // then: both callables have a frame reference argument
        for (auto *f : m.function_list()) {
            if (auto *cf = static_cast<CallableFunction *>(f); f->isa<CallableFunction>()) {
                expect(has_frame_arg(cf));
            }
        }
    };

    "no_coro_suspend_in_generated_functions"_test = [] {
        // given: a single-suspend coroutine
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
        b.coro_suspend(7u, "s", nullptr);

        b.set_insertion_point(resume_bb);
        b.coro_resume(7u, nullptr);
        b.return_void();

        // when
        auto susp_count = coro_split_pass_run_on_module(&m);
        static_cast<void>(susp_count);

        // then: no CoroSuspendInst in callables; returns exist instead
        auto suspends = count_instructions_with_tag(m, DerivedInstructionTag::CORO_SUSPEND);
        // only the original kernel may still have the suspend; callables must not
        // Check callables specifically
        size_t callable_suspends = 0u;
        for (auto *f : m.function_list()) {
            if (f->isa<CallableFunction>() && f->definition() != nullptr) {
                auto *def = static_cast<FunctionDefinition *>(f);
                def->traverse_instructions([&](Instruction *inst) noexcept {
                    if (inst->derived_instruction_tag() == DerivedInstructionTag::CORO_SUSPEND) {
                        callable_suspends++;
                    }
                });
            }
        }
        expect(callable_suspends == 0u);
        // returns must exist
        expect(has_return(m));
    };

    "three_suspends_four_callables"_test = [] {
        // given: three suspend points
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

        // when
        auto count = coro_split_pass_run_on_module(&m);

        // then
        expect(count == 4u);
        expect(count_callables(m) == 4u);

        // no suspends in callables
        size_t callable_suspends = 0u;
        for (auto *f : m.function_list()) {
            if (f->isa<CallableFunction>() && f->definition() != nullptr) {
                auto *def = static_cast<FunctionDefinition *>(f);
                def->traverse_instructions([&](Instruction *inst) noexcept {
                    if (inst->derived_instruction_tag() == DerivedInstructionTag::CORO_SUSPEND) {
                        callable_suspends++;
                    }
                });
            }
        }
        expect(callable_suspends == 0u);
    };

    "coro_resume_present_in_continuation"_test = [] {
        // given: a single-suspend coroutine
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
        b.coro_suspend(99u, "p99", nullptr);

        b.set_insertion_point(resume_bb);
        b.coro_resume(99u, nullptr);
        b.return_void();

        // when
        auto resume_count = coro_split_pass_run_on_module(&m);
        static_cast<void>(resume_count);

        // then: one of the callables has CoroResumeInst
        bool found_resume = false;
        for (auto *f : m.function_list()) {
            if (f->isa<CallableFunction>() && f->definition() != nullptr) {
                auto *def = static_cast<FunctionDefinition *>(f);
                def->traverse_instructions([&](Instruction *inst) noexcept {
                    if (inst->derived_instruction_tag() == DerivedInstructionTag::CORO_RESUME) {
                        found_resume = true;
                    }
                });
            }
        }
        expect(found_resume);
    };

    "terminal_coro_replaced"_test = [] {
        // given: coroutine ending with CoroTerminateInst
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

        // when
        auto count = coro_split_pass_run_on_module(&m);

        // then: no CoroTerminateInst in callables
        expect(count == 2u);
        size_t callable_terms = 0u;
        for (auto *f : m.function_list()) {
            if (f->isa<CallableFunction>() && f->definition() != nullptr) {
                auto *def = static_cast<FunctionDefinition *>(f);
                def->traverse_instructions([&](Instruction *inst) noexcept {
                    if (inst->derived_instruction_tag() == DerivedInstructionTag::CORO_TERMINATE) {
                        callable_terms++;
                    }
                });
            }
        }
        expect(callable_terms == 0u);
        // returns should exist in the terminal scope function
        expect(has_return(m));
    };

    "branch_targets_remapped"_test = [] {
        // given: a CFG where after suspend, the resume block
        // has a conditional branch back to itself (simulating a loop within a scope)
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

        // scope 3: body block with a conditional branch within itself
        b.set_insertion_point(resume_bb);
        b.coro_resume(10u, nullptr);

        auto *loop_body = k->create_basic_block();
        auto *loop_exit = k->create_basic_block();
        b.cond_br(cond, loop_body, loop_exit);

        b.set_insertion_point(loop_body);
        b.br(resume_bb);// loop back to the check

        b.set_insertion_point(loop_exit);
        b.return_void();

        // when
        auto count = coro_split_pass_run_on_module(&m);

        // then: split produced callables; check they have valid terminators (no dangling block refs)
        expect(count >= 1u);
        // basic sanity: all blocks in callables should be terminated
        for (auto *f : m.function_list()) {
            if (f->isa<CallableFunction>() && f->definition() != nullptr) {
                auto *def = static_cast<FunctionDefinition *>(f);
                def->traverse_basic_blocks([&](BasicBlock *bb) noexcept {
                    expect(bb->is_terminated());
                });
            }
        }
    };

    "continuations_do_not_emit_skip_check"_test = [] {
        // given: a single-suspend coroutine
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
        b.coro_suspend(1u, "first", nullptr);

        b.set_insertion_point(resume_bb);
        b.coro_resume(1u, nullptr);
        b.return_void();

        // when
        auto cfg = coro_cfg_distill_pass_run_on_function(k);
        auto split = coro_split_pass_run_on_module_with_cfg_and_frame_info(&m, cfg, nullptr);

        // then: callables exist, each is well-formed, and no synthetic skip-check branch is emitted
        expect(split.subroutines.size() == 2u);
        expect(count_callables(m) == 2u);

        // verify each callable has a body block and all blocks are terminated
        for (auto *f : m.function_list()) {
            if (f->isa<CallableFunction>() && f->definition() != nullptr) {
                auto *def = static_cast<FunctionDefinition *>(f);
                expect(def->body_block() != nullptr);
                def->traverse_basic_blocks([&](BasicBlock *bb) noexcept {
                    expect(bb->is_terminated());
                });
            }
        }

        size_t continuation_cond_branches = 0u;
        auto *cont = split.subroutines[1u].callable;
        cont->definition()->traverse_instructions([&](Instruction *inst) noexcept {
            if (inst->derived_instruction_tag() == DerivedInstructionTag::CONDITIONAL_BRANCH) {
                continuation_cond_branches++;
            }
        });
        expect(continuation_cond_branches == 0u);
    };

    "module_pass_handles_mixed_functions"_test = [] {
        // given: module with a coroutine kernel AND a non-coroutine callable
        Module m;

        // coroutine kernel
        {
            BasicBlock *body;
            auto *k = make_kernel_with_body(m, body);
            XIRBuilder b;
            auto *cond = m.create_constant_one(Type::of<bool>());
            auto *suspend_bb = k->create_basic_block();
            auto *resume_bb = k->create_basic_block();
            b.set_insertion_point(body);
            b.cond_br(cond, suspend_bb, resume_bb);
            b.set_insertion_point(suspend_bb);
            b.coro_suspend(1u, "s", nullptr);
            b.set_insertion_point(resume_bb);
            b.coro_resume(1u, nullptr);
            b.return_void();
        }

        // non-coroutine callable
        {
            auto *c = m.create_callable(nullptr);
            auto *cb = c->create_body_block();
            XIRBuilder b;
            b.set_insertion_point(cb);
            b.return_void();
        }

        // when
        auto count = coro_split_pass_run_on_module(&m);

        // then: only coroutine functions produce splits
        expect(count == 2u);
        // callable count = 2 (from coro split) + 1 (original non-coroutine)
        expect(count_callables(m) >= 2u);
    };

    "original_kernel_preserved"_test = [] {
        // given: a coroutine kernel
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
        b.coro_suspend(1u, "a", nullptr);

        b.set_insertion_point(resume_bb);
        b.coro_resume(1u, nullptr);
        b.return_void();

        // when
        auto resume_count = coro_split_pass_run_on_module(&m);
        static_cast<void>(resume_count);

        // then: original kernel still in function_list
        bool kernel_found = false;
        for (auto *f : m.function_list()) {
            if (f == k) { kernel_found = true; }
        }
        expect(kernel_found);
    };

    // ── edge case: adjacent suspends ─────────────────────────────────
    "adjacent_suspends_split"_test = [] {
        // given: two suspends back-to-back (resume→suspend with no
        // non-coro instructions between)
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

        // when
        auto count = coro_split_pass_run_on_module(&m);

        // then: 3 callables created
        expect(count == 3u);
        expect(count_callables(m) == 3u);

        // no CoroSuspendInst inside generated callables
        size_t callable_suspends = 0u;
        for (auto *f : m.function_list()) {
            if (f->isa<CallableFunction>() && f->definition() != nullptr) {
                auto *def = static_cast<FunctionDefinition *>(f);
                def->traverse_instructions([&](Instruction *inst) noexcept {
                    if (inst->derived_instruction_tag() == DerivedInstructionTag::CORO_SUSPEND) {
                        callable_suspends++;
                    }
                });
            }
        }
        expect(callable_suspends == 0u);

        // each callable has a frame-arg and all blocks terminated
        for (auto *f : m.function_list()) {
            if (auto *cf = static_cast<CallableFunction *>(f);
                f->isa<CallableFunction>()) {
                expect(has_frame_arg(cf));
            }
            if (f->isa<CallableFunction>() && f->definition() != nullptr) {
                auto *def = static_cast<FunctionDefinition *>(f);
                def->traverse_basic_blocks(
                    [&](BasicBlock *bb) noexcept {
                        expect(bb->is_terminated());
                    });
            }
        }
    };

    // ── edge case: suspend in one branch of conditional ──────────────
    "conditional_suspend_split"_test = [] {
        // given: if/else where suspend lives only in the true branch;
        // the false branch skips it and merges afterward
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

        // when
        auto count = coro_split_pass_run_on_module(&m);

        // then: split produces callables, all blocks well-formed
        expect(count >= 1u);
        for (auto *f : m.function_list()) {
            if (f->isa<CallableFunction>() && f->definition() != nullptr) {
                auto *def = static_cast<FunctionDefinition *>(f);
                def->traverse_basic_blocks(
                    [&](BasicBlock *bb) noexcept {
                        expect(bb->is_terminated());
                    });
            }
        }
    };

    // ── edge case: suspend inside a loop body ────────────────────────
    "loop_suspend_split"_test = [] {
        // given: a loop whose body reaches a suspend; after resume
        // the code branches back to the loop header
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

        // when
        auto count = coro_split_pass_run_on_module(&m);

        // then: split creates callables, loop back-edge survives
        expect(count >= 1u);
        expect(count_callables(m) >= 1u);

        // every callable block must be terminated
        for (auto *f : m.function_list()) {
            if (f->isa<CallableFunction>() && f->definition() != nullptr) {
                auto *def = static_cast<FunctionDefinition *>(f);
                expect(def->body_block() != nullptr);
                def->traverse_basic_blocks(
                    [&](BasicBlock *bb) noexcept {
                        expect(bb->is_terminated());
                    });
            }
        }

        // returns must exist (terminal scope ends properly)
        expect(has_return(m));
    };

    "cross_module_cfg_is_rejected_without_mutation"_test = [] {
        Module source;
        Module destination;
        BasicBlock *body;
        auto *kernel = make_kernel_with_body(source, body);
        auto *suspend_block = kernel->create_basic_block();
        auto *resume_block = kernel->create_basic_block();
        XIRBuilder b;
        b.set_insertion_point(body);
        auto *entry_term = b.cond_br(source.create_constant_one(Type::of<bool>()),
                                     suspend_block, resume_block);
        b.set_insertion_point(suspend_block);
        auto *suspend = b.coro_suspend(13u, "cross-module", nullptr);
        b.set_insertion_point(resume_block);
        b.coro_resume(13u, nullptr);
        b.return_void();
        auto cfg = coro_cfg_distill_pass_run_on_function(kernel);

        auto info = coro_split_pass_run_on_module_with_cfg_and_frame_info(
            &destination, cfg, nullptr);

        expect(!info.succeeded());
        expect(info.invalid_cfg_error_count == 1u);
        expect(info.structured_cfg_error_count == 0u);
        expect(info.subroutines.empty());
        expect(count_callables(destination) == 0u);
        expect(count_callables(source) == 0u);
        expect(body->terminator() == entry_term);
        expect(suspend_block->terminator() == suspend);
        expect(count_instructions_with_tag(source, DerivedInstructionTag::CORO_SUSPEND) == 1u);
    };

    "overlapping_distilled_scopes_are_rejected_atomically"_test = [] {
        Module m;
        BasicBlock *body;
        auto *kernel = make_kernel_with_body(m, body);
        auto *resume = kernel->create_basic_block();
        XIRBuilder b;
        b.set_insertion_point(body);
        auto *suspend = b.coro_suspend(1u, "overlap", nullptr);
        b.set_insertion_point(resume);
        auto *resume_inst = b.coro_resume(1u, nullptr);
        auto *return_inst = b.return_void();
        auto cfg = coro_cfg_distill_pass_run_on_function(kernel);
        cfg.scopes[1u].blocks.emplace_back(body);

        auto info = coro_split_pass_run_on_module_with_cfg_and_frame_info(&m, cfg, nullptr);

        expect(!info.succeeded());
        expect(info.invalid_cfg_error_count == 1u);
        expect(info.subroutines.empty());
        expect(count_callables(m) == 0u);
        expect(m.function_list().count_size() == 1u);
        expect(body->terminator() == suspend);
        expect(resume_inst->parent_block() == resume);
        expect(resume->terminator() == return_inst);
        expect(count_instructions_with_tag(m, DerivedInstructionTag::CORO_SUSPEND) == 1u);
    };

    "mismatched_distilled_suspend_metadata_is_rejected_atomically"_test = [] {
        Module m;
        BasicBlock *body;
        auto *kernel = make_kernel_with_body(m, body);
        auto *resume_one = kernel->create_basic_block();
        auto *resume_two = kernel->create_basic_block();
        XIRBuilder b;
        b.set_insertion_point(body);
        auto *suspend_one = b.coro_suspend(1u, "first", nullptr);
        b.set_insertion_point(resume_one);
        auto *resume_one_inst = b.coro_resume(1u, nullptr);
        auto *suspend_two = b.coro_suspend(2u, "second", nullptr);
        b.set_insertion_point(resume_two);
        auto *resume_two_inst = b.coro_resume(2u, nullptr);
        auto *return_inst = b.return_void();
        auto cfg = coro_cfg_distill_pass_run_on_function(kernel);
        cfg.scopes[0u].suspend_points[0u].token = 2u;
        cfg.scopes[1u].suspend_points[0u].token = 1u;

        auto info = coro_split_pass_run_on_module_with_cfg_and_frame_info(&m, cfg, nullptr);

        expect(!info.succeeded());
        expect(info.invalid_cfg_error_count == 1u);
        expect(info.subroutines.empty());
        expect(count_callables(m) == 0u);
        expect(m.function_list().count_size() == 1u);
        expect(body->terminator() == suspend_one);
        expect(resume_one_inst->parent_block() == resume_one);
        expect(resume_one->terminator() == suspend_two);
        expect(resume_two_inst->parent_block() == resume_two);
        expect(resume_two->terminator() == return_inst);
    };

    "tampered_distilled_transition_liveness_is_rejected_atomically"_test = [] {
        Module m;
        BasicBlock *body;
        auto *kernel = make_kernel_with_body(m, body);
        auto *resume = kernel->create_basic_block();
        auto *true_exit = kernel->create_basic_block();
        auto *false_exit = kernel->create_basic_block();
        XIRBuilder b;
        b.set_insertion_point(body);
        auto *state = b.alloca_local(Type::of<bool>());
        b.store(state, m.create_constant_one(Type::of<bool>()));
        auto *suspend =
            b.coro_suspend(5u, "liveness", nullptr);
        b.set_insertion_point(resume);
        b.coro_resume(5u, nullptr);
        b.cond_br(b.load(Type::of<bool>(), state),
                  true_exit, false_exit);
        b.set_insertion_point(true_exit);
        b.return_void();
        b.set_insertion_point(false_exit);
        b.return_void();
        auto cfg = coro_cfg_distill_pass_run_on_function(kernel);
        expect(cfg.succeeded());
        expect(!cfg.transition_edges.empty());
        auto found_stored_state = false;
        for (auto &edge : cfg.transition_edges) {
            if (std::find(edge.store_values.begin(),
                          edge.store_values.end(),
                          state) != edge.store_values.end()) {
                edge.store_values.clear();
                found_stored_state = true;
                break;
            }
        }
        expect(found_stored_state);

        auto info =
            coro_split_pass_run_on_module_with_cfg_and_frame_info(
                &m, cfg, nullptr);

        expect(!info.succeeded());
        expect(info.invalid_cfg_error_count == 1u);
        expect(info.subroutines.empty());
        expect(count_callables(m) == 0u);
        expect(body->terminator() == suspend);
        expect(count_instructions_with_tag(
                   m, DerivedInstructionTag::CORO_SUSPEND) == 1u);
        expect(xir_verify_module(&m).succeeded());
    };

    "tampered_distilled_scope_edges_are_rejected_atomically"_test = [] {
        Module m;
        BasicBlock *body;
        auto *kernel = make_kernel_with_body(m, body);
        auto *resume = kernel->create_basic_block();
        XIRBuilder b;
        b.set_insertion_point(body);
        auto *suspend =
            b.coro_suspend(7u, "edge", nullptr);
        b.set_insertion_point(resume);
        b.coro_resume(7u, nullptr);
        b.return_void();
        auto cfg = coro_cfg_distill_pass_run_on_function(kernel);
        expect(cfg.succeeded());
        expect(!cfg.edges.empty());
        expect(!cfg.edges.front().empty());
        cfg.edges.front().clear();

        auto info =
            coro_split_pass_run_on_module_with_cfg_and_frame_info(
                &m, cfg, nullptr);

        expect(!info.succeeded());
        expect(info.invalid_cfg_error_count == 1u);
        expect(info.subroutines.empty());
        expect(count_callables(m) == 0u);
        expect(body->terminator() == suspend);
        expect(xir_verify_module(&m).succeeded());
    };

    "stale_distill_certificate_after_source_edit_is_rejected_atomically"_test = [] {
        Module m;
        BasicBlock *body;
        auto *kernel = make_kernel_with_body(m, body);
        auto *resume = kernel->create_basic_block();
        XIRBuilder b;
        auto *one = m.create_constant_one(Type::of<int>());
        auto *zero = m.create_constant_zero(Type::of<int>());
        b.set_insertion_point(body);
        auto *state = b.alloca_local(Type::of<int>());
        auto *store = b.store(state, one);
        auto *suspend = b.coro_suspend(11u, "stale-source", nullptr);
        b.set_insertion_point(resume);
        b.coro_resume(11u, nullptr);
        auto *loaded = b.load(Type::of<int>(), state);
        static_cast<void>(loaded);
        b.return_void();

        auto cfg = coro_cfg_distill_pass_run_on_function(kernel);
        expect(cfg.succeeded());
        // Editing an operand preserves pointer identity and CFG shape, so a
        // shallow block/instruction snapshot cannot detect this stale result.
        // The certificate must bind the complete operand relation as well.
        store->set_value(zero);

        auto info =
            coro_split_pass_run_on_module_with_cfg_and_frame_info(
                &m, cfg, nullptr);

        expect(!info.succeeded());
        expect(info.invalid_cfg_error_count == 1u);
        expect(info.subroutines.empty());
        expect(count_callables(m) == 0u);
        expect(body->terminator() == suspend);
        expect(store->value() == zero);
        expect(xir_verify_module(&m).succeeded());
    };

    "split_preserves_metadata_on_manually_cloned_control_flow"_test = [] {
        Module m;
        BasicBlock *body;
        auto *kernel = make_kernel_with_body(m, body);
        auto *suspend_block = kernel->create_basic_block();
        auto *resume_block = kernel->create_basic_block();
        XIRBuilder b;
        b.set_insertion_point(body);
        auto *branch = b.cond_br(
            m.create_constant_one(Type::of<bool>()),
            suspend_block, resume_block);
        branch->add_comment("split-branch-metadata");
        b.set_insertion_point(suspend_block);
        auto *suspend =
            b.coro_suspend(11u, "metadata", nullptr);
        suspend->add_comment("split-suspend-metadata");
        b.set_insertion_point(resume_block);
        b.coro_resume(11u, nullptr);
        b.return_void();

        auto info = coro_split_pass_run_on_module_info(&m);

        expect(info.succeeded());
        expect(info.subroutines.size() == 2u);
        auto *entry_callable = info.subroutines.front().callable;
        size_t cloned_metadata_count = 0u;
        entry_callable->traverse_instructions(
            [&](Instruction *inst) noexcept {
                for ([[maybe_unused]] auto *metadata :
                     inst->metadata_list()) {
                    ++cloned_metadata_count;
                }
            });
        expect(cloned_metadata_count >= 2u);
        expect(xir_verify_module(&m).succeeded());
    };

    "split_clones_function_block_argument_and_alloca_metadata"_test = [] {
        Module m;
        auto *source = m.create_callable(nullptr);
        source->set_name("source_coroutine");
        source->add_comment("function provenance");
        auto *condition =
            source->create_value_argument(Type::of<bool>());
        condition->set_name("source_condition");
        condition->add_comment("argument provenance");
        auto *entry = source->create_body_block();
        auto *suspend_block = source->create_basic_block();
        auto *resume_block = source->create_basic_block();
        entry->set_name("source_entry");
        suspend_block->set_name("source_suspend");
        resume_block->set_name("source_resume");
        XIRBuilder b;
        b.set_insertion_point(entry);
        auto *state = b.alloca_local(Type::of<int>());
        state->set_name("source_state");
        state->add_comment("alloca provenance");
        b.store(state, m.create_constant_one(Type::of<int>()));
        b.cond_br(condition, suspend_block, resume_block);
        b.set_insertion_point(suspend_block);
        b.coro_suspend(37u, "metadata", nullptr);
        b.set_insertion_point(resume_block);
        auto *resume = b.coro_resume(37u, nullptr);
        resume->set_name("resume_boundary");
        resume->add_comment("resume provenance");
        auto *loaded = b.load(Type::of<int>(), state);
        static_cast<void>(loaded);
        b.return_void();

        expect(xir_verify_module(&m).succeeded());
        auto split = coro_split_pass_run_on_module_info(&m);

        expect(split.succeeded());
        expect(split.subroutines.size() == 2u);
        for (auto &subroutine : split.subroutines) {
            auto *callable = subroutine.callable;
            expect(callable->name().has_value());
            if (callable->name()) {
                expect(*callable->name() == "source_coroutine");
            }
            expect(callable->metadata_list().count_size() == 2u);
            auto argument = callable->arguments().begin();
            ++argument;// synthetic frame argument
            expect(argument != callable->arguments().end());
            if (argument != callable->arguments().end()) {
                expect((*argument)->name().has_value());
                if ((*argument)->name()) {
                    expect(*(*argument)->name() ==
                           "source_condition");
                }
                expect((*argument)->metadata_list().count_size() ==
                       2u);
            }
            size_t named_state_count = 0u;
            callable->traverse_instructions(
                [&](Instruction *inst) noexcept {
                    if (inst->isa<AllocaInst>() && inst->name() &&
                        *inst->name() == "source_state") {
                        ++named_state_count;
                        expect(inst->metadata_list().count_size() ==
                               2u);
                    }
                });
            expect(named_state_count == 1u);
        }
        auto *resume_callable = split.subroutines[1u].callable;
        expect(resume_callable->body_block()->name().has_value());
        if (resume_callable->body_block()->name()) {
            expect(*resume_callable->body_block()->name() ==
                   "source_resume");
        }
        CoroResumeInst *cloned_resume = nullptr;
        resume_callable->traverse_instructions(
            [&](Instruction *inst) noexcept {
                if (inst->isa<CoroResumeInst>()) {
                    cloned_resume =
                        static_cast<CoroResumeInst *>(inst);
                }
            });
        expect(cloned_resume != nullptr);
        if (cloned_resume == nullptr) { return; }
        expect(cloned_resume->isa<CoroResumeInst>());
        expect(cloned_resume->name().has_value());
        if (cloned_resume->name()) {
            expect(*cloned_resume->name() == "resume_boundary");
        }
        auto split_verification = xir_verify_module(&m);
        expect(split_verification.succeeded())
            << (split_verification.errors.empty() ?
                    "unknown XIR verification error" :
                    split_verification.errors.front().message.c_str());

        auto cfg = coro_cfg_distill_pass_run_on_function(source);
        auto materialized =
            coro_materialize_pass_run_on_module_with_cfg(
                &m, cfg, split);
        expect(materialized.succeeded());
        expect(!cloned_resume->is_linked());
        expect(resume_callable->body_block()->name().has_value());
        if (resume_callable->body_block()->name()) {
            expect(*resume_callable->body_block()->name() ==
                   "resume_boundary");
        }
        expect(resume_callable->body_block()
                   ->metadata_list()
                   .count_size() == 3u);
        auto materialized_verification = xir_verify_module(&m);
        expect(materialized_verification.succeeded())
            << (materialized_verification.errors.empty() ?
                    "unknown XIR verification error" :
                    materialized_verification.errors.front().message.c_str());
    };

    "continuation_local_alloca_precedes_its_first_store"_test = [] {
        Module m;
        auto *source = m.create_callable(nullptr);
        auto *entry = source->create_body_block();
        auto *resume_block = source->create_basic_block();
        XIRBuilder b;
        b.set_insertion_point(entry);
        auto *continuation_local =
            b.alloca_local(Type::of<int>());
        continuation_local->set_name("continuation_local");
        b.coro_suspend(39u, "local", nullptr);
        b.set_insertion_point(resume_block);
        b.coro_resume(39u, nullptr);
        b.store(
            continuation_local,
            m.create_constant_one(Type::of<int>()));
        static_cast<void>(
            b.load(Type::of<int>(), continuation_local));
        b.return_void();

        auto cfg = coro_cfg_distill_pass_run_on_function(source);
        expect(cfg.succeeded());
        expect(cfg.scopes.size() == 2u);
        expect(std::none_of(
            cfg.frame_values.begin(),
            cfg.frame_values.end(),
            [&](const auto &field) noexcept {
                return field.value == continuation_local;
            }));

        auto split = coro_split_pass_run_on_module_info(&m);

        expect(split.succeeded());
        expect(split.subroutines.size() == 2u);
        auto *continuation = split.subroutines[1u].callable;
        AllocaInst *cloned_alloca = nullptr;
        StoreInst *first_store = nullptr;
        for (auto *inst :
             continuation->body_block()->instructions()) {
            if (inst->isa<AllocaInst>() &&
                inst->name() &&
                *inst->name() == "continuation_local") {
                cloned_alloca =
                    static_cast<AllocaInst *>(inst);
            }
            if (first_store == nullptr &&
                inst->isa<StoreInst>()) {
                first_store = static_cast<StoreInst *>(inst);
            }
        }
        expect(cloned_alloca != nullptr);
        expect(first_store != nullptr);
        if (cloned_alloca != nullptr &&
            first_store != nullptr) {
            expect(first_store->variable() ==
                   cloned_alloca);
            Instruction *cursor = cloned_alloca;
            auto precedes = false;
            while (cursor != nullptr &&
                   !cursor->is_sentinel()) {
                if (cursor == first_store) {
                    precedes = true;
                    break;
                }
                cursor = cursor->next();
            }
            expect(precedes);
        }
        expect(xir_verify_module(&m).succeeded());
    };

    "signature_constrained_coroutine_is_rejected_atomically"_test = [] {
        Module m;
        auto *source = m.create_callable(nullptr);
        static_cast<void>(
            source->create_metadata<SignatureConstraintMD>());
        auto *entry = source->create_body_block();
        auto *resume = source->create_basic_block();
        XIRBuilder b;
        b.set_insertion_point(entry);
        auto *suspend =
            b.coro_suspend(41u, "constrained", nullptr);
        b.set_insertion_point(resume);
        b.coro_resume(41u, nullptr);
        b.return_void();
        auto original_function_count =
            m.function_list().count_size();

        expect(xir_verify_module(&m).succeeded());
        auto info = coro_split_pass_run_on_module_info(&m);

        expect(!info.succeeded());
        expect(info.invalid_cfg_error_count == 1u);
        expect(info.subroutines.empty());
        expect(m.function_list().count_size() ==
               original_function_count);
        expect(entry->terminator() == suspend);
        expect(xir_verify_module(&m).succeeded());
    };

    "structured_switch_is_rejected_without_mutation"_test = [] {
        Module m;
        auto original = make_structured_switch_coroutine(m, 17u);
        CoroCfgDistillResult cfg;
        cfg.scopes.resize(2u);
        cfg.scopes[0u].blocks.emplace_back(original.function->body_block());
        cfg.scopes[1u].blocks.emplace_back(original.resume_block);

        auto info = coro_split_pass_run_on_module_with_cfg_and_frame_info(&m, cfg, nullptr);

        expect(!info.succeeded());
        expect(info.structured_cfg_error_count == 1u);
        expect(info.subroutines.empty());
        expect(count_callables(m) == 0u);
        expect(original.function->body_block()->terminator() == original.switch_inst);
        expect(original.switch_inst->merge_block() == original.merge_block);
        expect(count_instructions_with_tag(m, DerivedInstructionTag::CORO_SUSPEND) == 1u);
    };

    "structured_loop_rejection_is_module_atomic"_test = [] {
        Module m;
        XIRBuilder b;
        auto *cond = m.create_constant_one(Type::of<bool>());

        // This definition is valid plain CFG and would be split if processed.
        BasicBlock *plain_body;
        auto *plain = make_kernel_with_body(m, plain_body);
        auto *plain_suspend = plain->create_basic_block();
        auto *plain_resume = plain->create_basic_block();
        b.set_insertion_point(plain_body);
        b.cond_br(cond, plain_suspend, plain_resume);
        b.set_insertion_point(plain_suspend);
        b.coro_suspend(3u, "plain", nullptr);
        b.set_insertion_point(plain_resume);
        b.coro_resume(3u, nullptr);
        b.return_void();

        // A later definition contains a real structured Loop/Continue region.
        BasicBlock *loop_entry;
        auto *structured = make_kernel_with_body(m, loop_entry);
        b.set_insertion_point(loop_entry);
        auto *loop = b.loop();
        auto *prepare = loop->create_prepare_block();
        auto *loop_body = loop->create_body_block();
        auto *update = loop->create_update_block();
        auto *merge = loop->create_merge_block();
        auto *resume = structured->create_basic_block();
        b.set_insertion_point(prepare);
        b.br(loop_body);
        b.set_insertion_point(loop_body);
        b.coro_suspend(9u, "structured-loop", nullptr);
        b.set_insertion_point(resume);
        b.coro_resume(9u, nullptr);
        b.continue_(update);
        b.set_insertion_point(update);
        b.break_(merge);
        b.set_insertion_point(merge);
        b.return_void();

        auto info = coro_split_pass_run_on_module_info(&m);

        expect(!info.succeeded());
        expect(info.structured_cfg_error_count == 1u);
        expect(info.subroutines.empty());
        expect(count_callables(m) == 0u);
        expect(plain_body->terminator()->derived_instruction_tag() ==
               DerivedInstructionTag::CONDITIONAL_BRANCH);
        expect(loop_entry->terminator() == loop);
        expect(resume->terminator()->derived_instruction_tag() ==
               DerivedInstructionTag::CONTINUE);
        expect(count_instructions_with_tag(m, DerivedInstructionTag::CORO_SUSPEND) == 2u);
    };

    "explicit_switch_destructure_allows_split"_test = [] {
        Module m;
        auto original = make_structured_switch_coroutine(m, 23u);

        auto destructured = destructure_cfg_pass_run_on_function(original.function);
        auto cfg = coro_cfg_distill_pass_run_on_function(original.function);
        auto info = coro_split_pass_run_on_module_with_cfg_and_frame_info(&m, cfg, nullptr);
        auto materialized = coro_materialize_pass_run_on_module_with_cfg(&m, cfg, info);

        expect(destructured.destructured_switch_count == 1u);
        expect(info.succeeded());
        expect(info.structured_cfg_error_count == 0u);
        expect(info.subroutines.size() == 2u);
        expect(materialized.succeeded());
        expect(materialized.callable_count == 2u);
        expect(count_callables(m) == 2u);
    };

    "default_frame_type_contains_distilled_live_fields"_test = [] {
        Module m;
        BasicBlock *body;
        auto *k = make_kernel_with_body(m, body);
        XIRBuilder b;
        b.set_insertion_point(body);
        auto *lhs = b.alloca_local(Type::of<float>());
        auto *rhs = b.alloca_local(Type::of<float>());
        lhs->set_name("state");
        rhs->set_name("state");
        b.store(lhs, m.create_constant_one(Type::of<float>()));
        b.store(rhs, m.create_constant_one(Type::of<float>()));
        b.coro_suspend(1u, "checkpoint", nullptr);
        auto *resume = k->create_basic_block();
        b.set_insertion_point(resume);
        b.coro_resume(1u, nullptr);
        auto *sum = b.call(Type::of<float>(), ArithmeticOp::BINARY_ADD,
                           {b.load(Type::of<float>(), lhs), b.load(Type::of<float>(), rhs)});
        static_cast<void>(sum);
        b.return_void();

        auto cfg = coro_cfg_distill_pass_run_on_function(k);
        auto info = coro_split_pass_run_on_module_with_cfg_and_frame_info(&m, cfg, nullptr);

        expect(info.succeeded());
        expect(info.subroutines.size() == 2u);
        expect(cfg.frame_values.size() == 2u);
        expect(cfg.frame_values[0u].name != cfg.frame_values[1u].name);
        for (auto &subroutine : info.subroutines) {
            auto members = subroutine.frame_argument->type()->members();
            expect(members.size() == 9u);
            expect(members[7u] == Type::of<float>());
            expect(members[8u] == Type::of<float>());
        }
        auto materialized = coro_materialize_pass_run_on_module_with_cfg(&m, cfg, info);
        expect(materialized.succeeded());
        expect(materialized.frame_fields.size() == 2u);
        expect(materialized.name_to_field.size() == 2u);
        expect(materialized.frame_fields[0u].index == 7u);
        expect(materialized.frame_fields[1u].index == 8u);
        expect(materialized.frame_fields[0u].name != materialized.frame_fields[1u].name);
    };

    "mismatched_explicit_frame_type_is_rejected_atomically"_test = [] {
        Module m;
        BasicBlock *body;
        auto *k = make_kernel_with_body(m, body);
        XIRBuilder b;
        b.set_insertion_point(body);
        auto *state = b.alloca_local(Type::of<float>());
        b.store(state, m.create_constant_one(Type::of<float>()));
        auto *suspend = b.coro_suspend(1u, "checkpoint", nullptr);
        auto *resume = k->create_basic_block();
        b.set_insertion_point(resume);
        b.coro_resume(1u, nullptr);
        auto *loaded = b.load(Type::of<float>(), state);
        static_cast<void>(loaded);
        b.return_void();
        auto cfg = coro_cfg_distill_pass_run_on_function(k);
        luisa::vector<const Type *> fields(7u, Type::of<uint>());
        auto *undersized_frame = Type::structure(fields);

        auto info = coro_split_pass_run_on_module_with_cfg_and_frame_info(&m, cfg, undersized_frame);

        expect(!info.succeeded());
        expect(info.invalid_cfg_error_count == 1u);
        expect(info.subroutines.empty());
        expect(count_callables(m) == 0u);
        expect(body->terminator() == suspend);
    };

    "invalid_coroutine_tokens_are_rejected_atomically"_test = [] {
        auto run_invalid = [](uint32_t suspend_token, uint32_t resume_token, bool emit_suspend, bool emit_resume) {
            Module m;
            BasicBlock *body;
            auto *k = make_kernel_with_body(m, body);
            XIRBuilder b;
            b.set_insertion_point(body);
            if (emit_suspend) {
                b.coro_suspend(suspend_token, "invalid", nullptr);
            } else {
                b.coro_resume(resume_token, nullptr);
                b.return_void();
            }
            if (emit_suspend && emit_resume) {
                auto *resume = k->create_basic_block();
                b.set_insertion_point(resume);
                b.coro_resume(resume_token, nullptr);
                b.return_void();
            }
            auto info = coro_split_pass_run_on_module_info(&m);
            expect(!info.succeeded());
            expect(info.invalid_cfg_error_count >= 1u);
            expect(info.subroutines.empty());
            expect(count_callables(m) == 0u);
        };
        run_invalid(0u, 0u, true, true);
        run_invalid(1u, 0u, true, false);
        run_invalid(0u, 1u, false, true);
        run_invalid(0xffffffffu, 0xffffffffu, true, true);

        Module m;
        BasicBlock *body;
        auto *k = make_kernel_with_body(m, body);
        auto *lhs = k->create_basic_block();
        auto *rhs = k->create_basic_block();
        auto *resume = k->create_basic_block();
        XIRBuilder b;
        b.set_insertion_point(body);
        b.cond_br(m.create_constant_one(Type::of<bool>()), lhs, rhs);
        b.set_insertion_point(lhs);
        b.coro_suspend(1u, "lhs", nullptr);
        b.set_insertion_point(rhs);
        b.coro_suspend(1u, "rhs", nullptr);
        b.set_insertion_point(resume);
        b.coro_resume(1u, nullptr);
        b.return_void();
        auto info = coro_split_pass_run_on_module_info(&m);
        expect(!info.succeeded());
        expect(info.invalid_cfg_error_count >= 1u);
        expect(info.subroutines.empty());
        expect(count_callables(m) == 0u);
    };

    "scope_cloning_orders_dominating_definitions_before_uses"_test = [] {
        Module m;
        BasicBlock *body;
        auto *k = make_kernel_with_body(m, body);
        auto *use_block = k->create_basic_block();
        auto *definition_block = k->create_basic_block();
        auto *suspend_block = k->create_basic_block();
        auto *resume_block = k->create_basic_block();
        XIRBuilder b;
        b.set_insertion_point(body);
        auto *state = b.alloca_local(Type::of<float>());
        b.store(state, m.create_constant_one(Type::of<float>()));
        b.br(definition_block);
        b.set_insertion_point(definition_block);
        auto *loaded = b.load(Type::of<float>(), state);
        b.br(use_block);
        b.set_insertion_point(use_block);
        auto *sum = b.call(Type::of<float>(), ArithmeticOp::BINARY_ADD,
                           {loaded, m.create_constant_one(Type::of<float>())});
        b.store(state, sum);
        b.br(suspend_block);
        b.set_insertion_point(suspend_block);
        b.coro_suspend(1u, "checkpoint", nullptr);
        b.set_insertion_point(resume_block);
        b.coro_resume(1u, nullptr);
        auto *resumed_state = b.load(Type::of<float>(), state);
        static_cast<void>(resumed_state);
        b.return_void();

        auto info = coro_split_pass_run_on_module_info(&m);

        expect(info.succeeded());
        expect(info.subroutines.size() == 2u);
        expect(count_callables(m) == 2u);
        auto *subroutine = info.subroutines[0u].callable;
        ArithmeticInst *cloned_sum = nullptr;
        subroutine->traverse_instructions([&](Instruction *inst) noexcept {
            if (inst->isa<ArithmeticInst>() &&
                static_cast<ArithmeticInst *>(inst)->op() == ArithmeticOp::BINARY_ADD) {
                expect(cloned_sum == nullptr);
                cloned_sum = static_cast<ArithmeticInst *>(inst);
            }
        });
        expect(cloned_sum != nullptr);
        if (cloned_sum == nullptr) { return; }
        expect(cloned_sum->operand_count() == 2u);
        auto *cloned_definition = cloned_sum->operand(0u);
        expect(cloned_definition != nullptr);
        expect(cloned_definition != loaded);
        expect(cloned_definition->isa<LoadInst>());
        if (cloned_definition == nullptr || !cloned_definition->isa<Instruction>()) { return; }
        auto *definition_inst = static_cast<Instruction *>(cloned_definition);
        expect(definition_inst->parent_function() == subroutine);
        expect(cloned_sum->parent_function() == subroutine);
        auto *definition_parent = definition_inst->parent_block();
        auto *use_parent = cloned_sum->parent_block();
        if (definition_parent == use_parent) {
            auto saw_definition = false;
            auto ordered = false;
            for (auto *inst : definition_parent->instructions()) {
                if (inst == definition_inst) { saw_definition = true; }
                if (inst == cloned_sum) { ordered = saw_definition; }
            }
            expect(ordered);
        } else {
            auto dom_tree = compute_dom_tree(subroutine);
            expect(dom_tree.dominates(definition_parent, use_parent));
        }
        expect(xir_verify_function(subroutine).succeeded());
    };

    "replayed_value_is_cloned_per_non_dominating_use_region"_test = [] {
        Module m;
        BasicBlock *entry;
        auto *kernel = make_kernel_with_body(m, entry);
        auto *x = kernel->create_argument(Type::of<float>(), false);
        auto *branch = kernel->create_argument(Type::of<bool>(), false);
        auto *resume = kernel->create_basic_block();
        auto *left = kernel->create_basic_block();
        auto *right = kernel->create_basic_block();
        auto *merge = kernel->create_basic_block();
        XIRBuilder b;
        auto *one = m.create_constant_one(Type::of<float>());
        b.set_insertion_point(entry);
        auto *replay = b.call(
            Type::of<float>(), ArithmeticOp::BINARY_ADD,
            {x, one});
        b.coro_suspend(53u, "branch-replay", nullptr);
        b.set_insertion_point(resume);
        b.coro_resume(53u, nullptr);
        b.cond_br(branch, left, right);
        b.set_insertion_point(left);
        static_cast<void>(b.call(
            Type::of<float>(), ArithmeticOp::BINARY_MUL,
            {replay, one}));
        b.br(merge);
        b.set_insertion_point(right);
        static_cast<void>(b.call(
            Type::of<float>(), ArithmeticOp::BINARY_SUB,
            {replay, one}));
        b.br(merge);
        b.set_insertion_point(merge);
        b.return_void();

        auto cfg = coro_cfg_distill_pass_run_on_function(kernel);
        expect(cfg.succeeded());
        expect(cfg.frame_values.empty());
        auto split = coro_split_pass_run_on_module_info(&m);

        expect(split.succeeded());
        expect(split.subroutines.size() == 2u);
        if (split.subroutines.size() != 2u) { return; }
        auto *continuation = split.subroutines[1u].callable;
        luisa::vector<ArithmeticInst *> replayed_adds;
        continuation->traverse_instructions(
            [&](Instruction *instruction) noexcept {
                if (instruction->isa<ArithmeticInst>() &&
                    static_cast<ArithmeticInst *>(instruction)->op() ==
                        ArithmeticOp::BINARY_ADD) {
                    replayed_adds.emplace_back(
                        static_cast<ArithmeticInst *>(instruction));
                }
            });
        expect(replayed_adds.size() == 2u);
        if (replayed_adds.size() == 2u) {
            expect(replayed_adds[0u]->parent_block() !=
                   replayed_adds[1u]->parent_block());
        }
        auto verification = xir_verify_module(&m);
        expect(verification.succeeded())
            << (verification.errors.empty() ?
                    "unknown XIR verification error" :
                    verification.errors.front().message.c_str());
    };

    "split_materializes_colored_frame_slot_once"_test = [] {
        Module m;
        BasicBlock *entry;
        auto *kernel = make_kernel_with_body(m, entry);
        auto *resume_first = kernel->create_basic_block();
        auto *resume_second = kernel->create_basic_block();
        XIRBuilder b;
        auto *one = m.create_constant_one(Type::of<uint64_t>());
        b.set_insertion_point(entry);
        auto *first = b.clock();
        b.coro_suspend(79u, "first", nullptr);
        b.set_insertion_point(resume_first);
        b.coro_resume(79u, nullptr);
        static_cast<void>(b.call(
            Type::of<uint64_t>(), ArithmeticOp::BINARY_ADD,
            {first, one}));
        auto *second = b.clock();
        b.coro_suspend(83u, "second", nullptr);
        b.set_insertion_point(resume_second);
        b.coro_resume(83u, nullptr);
        static_cast<void>(b.call(
            Type::of<uint64_t>(), ArithmeticOp::BINARY_ADD,
            {second, one}));
        b.return_void();

        auto cfg = coro_cfg_distill_pass_run_on_function(kernel);
        expect(cfg.succeeded());
        expect(cfg.frame_values.size() == 2u);
        expect(cfg.frame_slots.size() == 1u);
        auto split =
            coro_split_pass_run_on_module_with_cfg_and_frame_info(
                &m, cfg, nullptr);

        expect(split.succeeded());
        expect(split.subroutines.size() == 3u);
        for (auto &subroutine : split.subroutines) {
            expect(subroutine.frame_argument != nullptr);
            if (subroutine.frame_argument != nullptr) {
                expect(subroutine.frame_argument->type()->members().size() ==
                       CoroFrameDesc::reserved_field_count + 1u);
            }
        }
        auto materialized =
            coro_materialize_pass_run_on_module_with_cfg(
                &m, cfg, split);
        expect(materialized.succeeded());
        expect(materialized.register_count == 2u);
        expect(materialized.frame_fields.size() == 1u);
        expect(materialized.frame_field_count ==
               CoroFrameDesc::reserved_field_count + 1u);
        expect(materialized.name_to_field.size() == 2u);
        expect(materialized.name_to_type.size() == 2u);
        if (cfg.frame_values.size() == 2u) {
            expect(materialized.name_to_field.at(
                       cfg.frame_values[0u].name) ==
                   materialized.name_to_field.at(
                       cfg.frame_values[1u].name));
        }
        expect(xir_verify_module(&m).succeeded());
    };

    "split_spills_and_restores_only_the_live_aggregate_path"_test = [] {
        Module m;
        BasicBlock *entry;
        auto *kernel = make_kernel_with_body(m, entry);
        auto *resume = kernel->create_basic_block();
        auto *pair = Type::structure(
            {Type::of<float>(), Type::of<float>()});
        uint32_t zero_value = 0u;
        uint32_t one_value = 1u;
        auto *zero = m.create_constant(
            Type::of<uint32_t>(), &zero_value);
        auto *one = m.create_constant(
            Type::of<uint32_t>(), &one_value);
        XIRBuilder b;
        b.set_insertion_point(entry);
        auto *state = b.alloca_local(pair);
        state->set_name("path_state");
        auto *first = b.gep(Type::of<float>(), state, {zero});
        auto *second = b.gep(Type::of<float>(), state, {one});
        b.store(first, m.create_constant_zero(Type::of<float>()));
        b.store(second, m.create_constant_one(Type::of<float>()));
        b.coro_suspend(137u, "path", nullptr);
        b.set_insertion_point(resume);
        b.coro_resume(137u, nullptr);
        auto *resumed_second = b.gep(Type::of<float>(), state, {one});
        static_cast<void>(b.load(Type::of<float>(), resumed_second));
        b.return_void();

        expect(xir_verify_module(&m).succeeded());
        auto cfg = coro_cfg_distill_pass_run_on_function(kernel);
        expect(cfg.succeeded());
        expect(cfg.frame_values.size() == 1u);
        if (cfg.frame_values.size() == 1u) {
            expect(cfg.frame_values.front().value == state);
            expect(cfg.frame_values.front().access_chain ==
                   luisa::vector<uint32_t>{1u});
            expect(cfg.frame_values.front().type == Type::of<float>());
        }

        auto split =
            coro_split_pass_run_on_module_with_cfg_and_frame_info(
                &m, cfg, nullptr);
        expect(split.succeeded());
        expect(split.subroutines.size() == 2u);
        for (auto &subroutine : split.subroutines) {
            expect(subroutine.frame_argument != nullptr);
            if (subroutine.frame_argument != nullptr) {
                auto members = subroutine.frame_argument->type()->members();
                expect(members.size() ==
                       CoroFrameDesc::reserved_field_count + 1u);
                expect(members.back() == Type::of<float>());
            }
        }
        auto materialized =
            coro_materialize_pass_run_on_module_with_cfg(
                &m, cfg, split);
        expect(materialized.succeeded());
        expect(materialized.register_count == 1u);
        expect(materialized.frame_fields.size() == 1u);
        expect(materialized.name_to_type.at("path_state.1") ==
               Type::of<float>());
        auto verification = xir_verify_module(&m);
        expect(verification.succeeded())
            << (verification.errors.empty() ?
                    "unknown XIR verification error" :
                    verification.errors.front().message.c_str());
    };

    "split_materializes_padded_aggregate_as_packed_abi_fields"_test = [] {
        Module m;
        BasicBlock *entry;
        auto *kernel = make_kernel_with_body(m, entry);
        auto *resume = kernel->create_basic_block();
        auto *padded = Type::structure(
            {Type::of<float2>(), Type::of<float>()});
        XIRBuilder b;
        b.set_insertion_point(entry);
        auto *state = b.alloca_local(padded);
        state->set_name("padded_state");
        b.store(state, m.create_constant_zero(padded));
        b.coro_suspend(229u, "padded", nullptr);
        b.set_insertion_point(resume);
        b.coro_resume(229u, nullptr);
        static_cast<void>(b.load(padded, state));
        b.return_void();

        expect(xir_verify_module(&m).succeeded());
        auto cfg = coro_cfg_distill_pass_run_on_function(kernel);
        expect(cfg.succeeded());
        expect(cfg.frame_values.size() == 2u);
        expect(cfg.frame_slots.size() == 2u);

        auto split =
            coro_split_pass_run_on_module_with_cfg_and_frame_info(
                &m, cfg, nullptr);
        expect(split.succeeded());
        expect(split.subroutines.size() == 2u);
        for (auto &subroutine : split.subroutines) {
            expect(subroutine.frame_argument != nullptr);
            if (subroutine.frame_argument != nullptr) {
                auto members = subroutine.frame_argument->type()->members();
                expect(members.size() ==
                       CoroFrameDesc::reserved_field_count + 2u);
                expect(members[CoroFrameDesc::reserved_field_count] ==
                       Type::of<float2>());
                expect(members[CoroFrameDesc::reserved_field_count + 1u] ==
                       Type::of<float>());
            }
        }
        auto materialized =
            coro_materialize_pass_run_on_module_with_cfg(
                &m, cfg, split);
        expect(materialized.succeeded());
        expect(materialized.register_count == 2u);
        expect(materialized.frame_fields.size() == 2u);
        expect(materialized.name_to_type.at("padded_state.0") ==
               Type::of<float2>());
        expect(materialized.name_to_type.at("padded_state.1") ==
               Type::of<float>());
        auto verification = xir_verify_module(&m);
        expect(verification.succeeded())
            << (verification.errors.empty() ?
                    "unknown XIR verification error" :
                    verification.errors.front().message.c_str());
    };
}

int main(int argc, char *argv[]) {
    (void)argc;
    (void)argv;
    reg_coro_split();
    return 0;
}
