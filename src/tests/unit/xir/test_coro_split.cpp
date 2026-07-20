// Test for coroutine splitting, state transitions, and malformed-input rejection.

#include "ut/ut.hpp"
#include <luisa/ast/type_registry.h>
#include <luisa/xir/basic_block.h>
#include <luisa/xir/builder.h>
#include <luisa/xir/function.h>
#include <luisa/xir/instructions/arithmetic.h>
#include <luisa/xir/instructions/branch.h>
#include <luisa/xir/instructions/coro.h>
#include <luisa/xir/instructions/loop.h>
#include <luisa/xir/instructions/return.h>
#include <luisa/xir/instructions/switch.h>
#include <luisa/xir/module.h>
#include <luisa/xir/passes/coro_cfg_distill.h>
#include <luisa/xir/passes/coro_materialize.h>
#include <luisa/xir/passes/coro_split.h>
#include <luisa/xir/passes/destructure_cfg.h>
#include <luisa/xir/passes/dom_tree.h>
#include <luisa/xir/passes/lower_switch.h>
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

    "explicit_switch_then_structured_normalization_allows_split"_test = [] {
        Module m;
        auto original = make_structured_switch_coroutine(m, 23u);

        auto lowered = lower_switch_pass_run_on_function(original.function);
        auto destructured = destructure_cfg_pass_run_on_function(original.function);
        auto cfg = coro_cfg_distill_pass_run_on_function(original.function);
        auto info = coro_split_pass_run_on_module_with_cfg_and_frame_info(&m, cfg, nullptr);
        auto materialized = coro_materialize_pass_run_on_module_with_cfg(&m, cfg, info);

        expect(lowered.lowered_switch_count == 1u);
        expect(destructured.destructured_if_count >= 1u);
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
        b.return_(sum);

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
        b.return_(b.load(Type::of<float>(), state));
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
        b.return_(b.load(Type::of<float>(), state));

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
}

int main(int argc, char *argv[]) {
    (void)argc;
    (void)argv;
    reg_coro_split();
    return 0;
}
