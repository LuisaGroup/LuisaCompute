#include "ut/ut.hpp"
#include <luisa/ast/type_registry.h>
#include <luisa/xir/basic_block.h>
#include <luisa/xir/builder.h>
#include <luisa/xir/function.h>
#include <luisa/xir/instructions/branch.h>
#include <luisa/xir/instructions/coro.h>
#include <luisa/xir/module.h>
#include <luisa/xir/passes/coro_cfg_distill.h>

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

}// namespace

void reg_coro_cfg_distill() {

    "no_suspend_single_scope"_test = [] {
        // given: a function with no coroutine instructions
        Module m;
        BasicBlock *body;
        auto *k = make_kernel_with_body(m, body);
        XIRBuilder b;
        b.set_insertion_point(body);
        b.return_void();

        // when
        auto result = coro_cfg_distill_pass_run_on_function(k);

        // then: 1 scope, no suspend info, not terminal
        expect(result.scopes.size() == 1u);
        expect(result.scopes[0].blocks.size() == 1u);
        expect(result.scopes[0].blocks[0] == body);
        expect(!result.scopes[0].suspend_token.has_value());
        expect(!result.scopes[0].suspend_name.has_value());
        expect(!result.scopes[0].is_terminal);
        expect(result.edges.size() == 1u);
        expect(result.edges[0].empty());
    };

    "single_suspend_two_scopes"_test = [] {
        // given: CFG with one suspend point
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
        auto result = coro_cfg_distill_pass_run_on_function(k);

        // then: 2 scopes, scope 0 has suspend, scope 1 is continuation
        expect(result.scopes.size() == 2u);

        // scope 0
        expect(result.scopes[0].scope_id == 0);
        expect(result.scopes[0].blocks.size() >= 1u);// body is in scope 0
        expect(result.scopes[0].suspend_token.has_value());
        expect(*result.scopes[0].suspend_token == 42u);
        expect(result.scopes[0].suspend_name.has_value());
        expect(*result.scopes[0].suspend_name == "checkpoint");
        expect(!result.scopes[0].is_terminal);

        // scope 1
        expect(result.scopes[1].scope_id == 1);
        expect(!result.scopes[1].suspend_token.has_value());
        expect(!result.scopes[1].is_terminal);

        // edges
        expect(result.edges.size() == 2u);
    };

    "three_suspends_four_scopes"_test = [] {
        // given: CFG with three suspend points (linear chain)
        Module m;
        BasicBlock *body;
        auto *k = make_kernel_with_body(m, body);
        XIRBuilder b;
        auto *cond = m.create_constant_one(Type::of<bool>());

        // suspend 1
        auto *s1 = k->create_basic_block();
        auto *r1 = k->create_basic_block();

        b.set_insertion_point(body);
        b.cond_br(cond, s1, r1);

        b.set_insertion_point(s1);
        b.coro_suspend(1u, "s1", nullptr);

        b.set_insertion_point(r1);
        b.coro_resume(1u, nullptr);

        // suspend 2
        auto *s2 = k->create_basic_block();
        auto *r2 = k->create_basic_block();
        b.cond_br(cond, s2, r2);

        b.set_insertion_point(s2);
        b.coro_suspend(2u, "s2", nullptr);

        b.set_insertion_point(r2);
        b.coro_resume(2u, nullptr);

        // suspend 3
        auto *s3 = k->create_basic_block();
        auto *r3 = k->create_basic_block();
        b.cond_br(cond, s3, r3);

        b.set_insertion_point(s3);
        b.coro_suspend(3u, "s3", nullptr);

        b.set_insertion_point(r3);
        b.coro_resume(3u, nullptr);
        b.return_void();

        // when
        auto result = coro_cfg_distill_pass_run_on_function(k);

        // then: 4 scopes
        expect(result.scopes.size() == 4u);

        // verify suspend tokens
        expect(result.scopes[0].suspend_token.has_value());
        expect(*result.scopes[0].suspend_token == 1u);
        expect(result.scopes[1].suspend_token.has_value());
        expect(*result.scopes[1].suspend_token == 2u);
        expect(result.scopes[2].suspend_token.has_value());
        expect(*result.scopes[2].suspend_token == 3u);
        expect(!result.scopes[3].suspend_token.has_value());

        // verify no scope is terminal except possibly the last
        expect(!result.scopes[0].is_terminal);
        expect(!result.scopes[1].is_terminal);
        expect(!result.scopes[2].is_terminal);
        expect(!result.scopes[3].is_terminal);

        // verify scope block counts
        for (size_t i = 0; i < 4u; ++i) {
            expect(result.scopes[i].blocks.size() >= 1u);
        }

        // edges
        expect(result.edges.size() == 4u);
    };

    "suspend_token_values_match"_test = [] {
        // given: two suspend points with distinct tokens and names
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
        b.coro_suspend(100u, "alpha", nullptr);

        b.set_insertion_point(r1);
        b.coro_resume(100u, nullptr);

        auto *s2 = k->create_basic_block();
        auto *r2 = k->create_basic_block();
        b.cond_br(cond, s2, r2);

        b.set_insertion_point(s2);
        b.coro_suspend(200u, "beta", nullptr);

        b.set_insertion_point(r2);
        b.coro_resume(200u, nullptr);
        b.return_void();

        // when
        auto result = coro_cfg_distill_pass_run_on_function(k);

        // then: token values match the suspend instructions
        expect(result.scopes.size() == 3u);

        expect(result.scopes[0].suspend_token.has_value());
        expect(*result.scopes[0].suspend_token == 100u);
        expect(result.scopes[0].suspend_name.has_value());
        expect(*result.scopes[0].suspend_name == "alpha");

        expect(result.scopes[1].suspend_token.has_value());
        expect(*result.scopes[1].suspend_token == 200u);
        expect(result.scopes[1].suspend_name.has_value());
        expect(*result.scopes[1].suspend_name == "beta");

        expect(!result.scopes[2].suspend_token.has_value());
    };

    "terminal_scope"_test = [] {
        // given: a coroutine that ends with CoroTerminateInst
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
        b.coro_suspend(1u, "middle", nullptr);

        b.set_insertion_point(r1);
        b.coro_resume(1u, nullptr);

        auto *term_bb = k->create_basic_block();
        b.br(term_bb);

        b.set_insertion_point(term_bb);
        b.coro_terminate();

        // when
        auto result = coro_cfg_distill_pass_run_on_function(k);

        // then: last scope is terminal
        expect(result.scopes.size() >= 1u);
        auto &last = result.scopes.back();
        expect(last.is_terminal);
    };

    "scope_contains_suspend_block"_test = [] {
        // given: a single-suspend CFG — verify the suspend block is in the first scope
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
        b.coro_suspend(7u, "test", nullptr);

        b.set_insertion_point(resume_bb);
        b.coro_resume(7u, nullptr);
        b.return_void();

        // when
        auto result = coro_cfg_distill_pass_run_on_function(k);

        // then: suspend block is in scope 0, resume block is in scope 1
        expect(result.scopes.size() == 2u);

        bool suspend_found = false;
        for (auto *bb : result.scopes[0].blocks) {
            if (bb == suspend_bb) { suspend_found = true; }
        }
        expect(suspend_found);

        bool resume_found = false;
        for (auto *bb : result.scopes[1].blocks) {
            if (bb == resume_bb) { resume_found = true; }
        }
        expect(resume_found);
    };

    "module_pass_iterates_all_functions"_test = [] {
        // given: module with a kernel and a callable (neither with coroutine instructions)
        Module m;
        {
            auto *k = m.create_kernel();
            auto *body = k->create_body_block();
            XIRBuilder b;
            b.set_insertion_point(body);
            b.return_void();
        }
        {
            auto *c = m.create_callable(nullptr);
            auto *body = c->create_body_block();
            XIRBuilder b;
            b.set_insertion_point(body);
            b.return_void();
        }

        // when
        auto count = coro_cfg_distill_pass_run_on_module(&m);

        // then: processes both definition functions
        expect(count == 2u);
    };

    // ── edge case: adjacent suspends ───────────────────────────────────
    "adjacent_suspends"_test = [] {
        // given: two suspends with minimal code between them (resume
        // of first immediately branches to second suspend)
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
        b.coro_suspend(10u, "first", nullptr);

        // scope 1: resume, then immediately branch to next suspend
        b.set_insertion_point(r1);
        b.coro_resume(10u, nullptr);

        auto *s2 = k->create_basic_block();
        auto *r2 = k->create_basic_block();
        b.cond_br(cond, s2, r2);

        b.set_insertion_point(s2);
        b.coro_suspend(20u, "second", nullptr);

        b.set_insertion_point(r2);
        b.coro_resume(20u, nullptr);
        b.return_void();

        // when
        auto result = coro_cfg_distill_pass_run_on_function(k);

        // then: 3 scopes (body→s1 | r1→s2 | r2)
        expect(result.scopes.size() == 3u);

        expect(result.scopes[0].suspend_token.has_value());
        expect(*result.scopes[0].suspend_token == 10u);
        expect(result.scopes[0].suspend_name.has_value());
        expect(*result.scopes[0].suspend_name == "first");

        expect(result.scopes[1].suspend_token.has_value());
        expect(*result.scopes[1].suspend_token == 20u);
        expect(result.scopes[1].suspend_name.has_value());
        expect(*result.scopes[1].suspend_name == "second");

        expect(!result.scopes[2].suspend_token.has_value());

        // no scope marked terminal
        expect(!result.scopes[0].is_terminal);
        expect(!result.scopes[1].is_terminal);
        expect(!result.scopes[2].is_terminal);

        // r1 and s2 should coexist in scope 1 (adjacent)
        bool r1_in_scope1 = false;
        bool s2_in_scope1 = false;
        for (auto *bb : result.scopes[1].blocks) {
            if (bb == r1) { r1_in_scope1 = true; }
            if (bb == s2) { s2_in_scope1 = true; }
        }
        expect(r1_in_scope1);
        expect(s2_in_scope1);

        expect(result.edges.size() == 3u);
    };

    // ── edge case: suspend inside a conditional branch ────────────────
    "suspend_in_conditional"_test = [] {
        // given: an if/else where only one branch contains a suspend;
        // the other branch skips it entirely and merges afterward
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

        // branch A: contains a suspend
        b.set_insertion_point(branch_a);
        b.cond_br(always_true, s1, r1);

        b.set_insertion_point(s1);
        b.coro_suspend(1u, "in_branch", nullptr);

        b.set_insertion_point(r1);
        b.coro_resume(1u, nullptr);
        b.br(merge);

        // branch B: no suspend, goes straight to merge
        b.set_insertion_point(branch_b);
        b.br(merge);

        b.set_insertion_point(merge);
        b.return_void();

        // when
        auto result = coro_cfg_distill_pass_run_on_function(k);

        // then: at least 2 scopes, exactly one scope has a suspend
        expect(result.scopes.size() >= 2u);

        size_t suspend_scopes = 0u;
        for (auto &scope : result.scopes) {
            if (scope.suspend_token.has_value()) { suspend_scopes++; }
        }
        expect(suspend_scopes == 1u);

        // suspend block must live in the scope that owns the suspend
        bool s1_in_suspend_scope = false;
        for (auto &scope : result.scopes) {
            if (scope.suspend_token.has_value()) {
                for (auto *bb : scope.blocks) {
                    if (bb == s1) { s1_in_suspend_scope = true; }
                }
            }
        }
        expect(s1_in_suspend_scope);

        // merge block must appear in at least one scope
        bool merge_found = false;
        for (auto &scope : result.scopes) {
            for (auto *bb : scope.blocks) {
                if (bb == merge) { merge_found = true; }
            }
        }
        expect(merge_found);
    };

    // ── edge case: suspend inside a loop ──────────────────────────────
    "suspend_in_loop"_test = [] {
        // given: a loop whose body contains a suspend point;
        // the back-edge goes through the resume block back to the header
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
        b.coro_suspend(1u, "in_loop", nullptr);

        b.set_insertion_point(r1);
        b.coro_resume(1u, nullptr);
        b.br(loop_hdr);// back-edge through resume

        b.set_insertion_point(exit);
        b.return_void();

        // when
        auto result = coro_cfg_distill_pass_run_on_function(k);

        // then: at least 2 scopes, suspend is found, no crash on cycle
        expect(result.scopes.size() >= 2u);

        bool suspend_found = false;
        for (auto &scope : result.scopes) {
            if (scope.suspend_token.has_value() &&
                *scope.suspend_token == 1u) {
                suspend_found = true;
            }
        }
        expect(suspend_found);

        // edges array matches scope count
        expect(result.edges.size() == result.scopes.size());

        // all blocks from the kernel appear in at least one scope
        size_t total_blocks = 0u;
        for (auto &scope : result.scopes) {
            total_blocks += scope.blocks.size();
        }
        expect(total_blocks >= 6u);// body, loop_hdr, loop_body, s1, r1, exit
    };
}

int main(int argc, char *argv[]) {
    (void)argc;
    (void)argv;
    reg_coro_cfg_distill();
    return 0;
}
