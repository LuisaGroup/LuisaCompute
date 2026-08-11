// Test for coroutine CFG distillation and malformed-graph rejection.

#include "ut/ut.hpp"
#include <luisa/ast/type_registry.h>
#include <luisa/xir/basic_block.h>
#include <luisa/xir/builder.h>
#include <luisa/xir/function.h>
#include <luisa/xir/instructions/alloca.h>
#include <luisa/xir/instructions/branch.h>
#include <luisa/xir/instructions/coro.h>
#include <luisa/xir/instructions/indexed_branch.h>
#include <luisa/xir/instructions/phi.h>
#include <luisa/xir/instructions/switch.h>
#include <luisa/xir/module.h>
#include <luisa/xir/passes/coro_cfg_distill.h>
#include <luisa/xir/passes/coro_reg2mem.h>
#include <luisa/xir/passes/destructure_cfg.h>
#include <luisa/xir/translators/xir2text.h>
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

    "loop_must_kill_uses_greatest_fixed_point"_test = [] {
        // Definite definition is a must property. The loop header equation is
        //
        //   K_header = K_entry intersect K_backedge.
        //
        // Since state is initialized before the loop and no path can undo a
        // definition, the greatest fixed point contains state. Initializing
        // the backedge to the empty set instead selects the smaller, invalid
        // fixed point and spuriously promotes this loop-local value into the
        // coroutine frame.
        Module m;
        BasicBlock *body;
        auto *k = make_kernel_with_body(m, body);
        XIRBuilder b;
        auto *loop_cond = m.create_constant_one(Type::of<bool>());
        auto *zero = m.create_constant_zero(Type::of<int>());

        auto *loop_header = k->create_basic_block();
        auto *loop_backedge = k->create_basic_block();
        auto *suspend_block = k->create_basic_block();
        auto *resume_block = k->create_basic_block();

        b.set_insertion_point(body);
        auto *state = b.alloca_local(Type::of<int>());
        state->set_name("loop_local_state");
        b.store(state, zero);
        b.br(loop_header);

        b.set_insertion_point(loop_header);
        auto *loaded = b.load(Type::of<int>(), state);
        static_cast<void>(loaded);
        b.cond_br(loop_cond, loop_backedge, suspend_block);

        b.set_insertion_point(loop_backedge);
        b.br(loop_header);

        b.set_insertion_point(suspend_block);
        b.coro_suspend(1u, "after-loop", nullptr);

        b.set_insertion_point(resume_block);
        b.coro_resume(1u, nullptr);
        b.return_void();

        auto result = coro_cfg_distill_pass_run_on_function(k);

        expect(result.succeeded());
        expect(result.scopes.size() == 2u);
        expect(std::find(result.scopes[0u].external_values.begin(),
                         result.scopes[0u].external_values.end(),
                         state) == result.scopes[0u].external_values.end());
        expect(std::find_if(result.frame_values.begin(),
                            result.frame_values.end(),
                            [&](auto &field) noexcept {
                                return field.value == state;
                            }) == result.frame_values.end());
        auto suspend_kills_state = false;
        for (auto &edge : result.transition_edges) {
            if (edge.is_suspend && edge.token == 1u &&
                std::find(edge.killed_values.begin(),
                          edge.killed_values.end(),
                          state) != edge.killed_values.end()) {
                suspend_kills_state = true;
            }
        }
        expect(suspend_kills_state);
    };

    "for_if_suspend_liveness"_test = [] {
        // given: for (...) { if (...) { suspend } } with a local updated
        // before the suspend and used after resume
        Module m;
        BasicBlock *body;
        auto *k = make_kernel_with_body(m, body);
        XIRBuilder b;
        auto *loop_cond = m.create_constant_one(Type::of<bool>());
        auto *if_cond = m.create_constant_one(Type::of<bool>());
        auto *zero = m.create_constant_zero(Type::of<int>());
        auto *one = m.create_constant_one(Type::of<int>());

        auto *loop_hdr = k->create_basic_block();
        auto *loop_body = k->create_basic_block();
        auto *suspend_bb = k->create_basic_block();
        auto *resume_bb = k->create_basic_block();
        auto *after_if = k->create_basic_block();
        auto *exit = k->create_basic_block();

        b.set_insertion_point(body);
        auto *state = b.alloca_local(Type::of<int>());
        state->set_name("state");
        b.store(state, zero);
        b.br(loop_hdr);

        b.set_insertion_point(loop_hdr);
        b.cond_br(loop_cond, loop_body, exit);

        b.set_insertion_point(loop_body);
        auto *old_state = b.load(Type::of<int>(), state);
        auto *new_state = b.call(Type::of<int>(), ArithmeticOp::BINARY_ADD, {old_state, one});
        b.store(state, new_state);
        b.cond_br(if_cond, suspend_bb, after_if);

        b.set_insertion_point(suspend_bb);
        b.coro_suspend(1u, "in_for_if", nullptr);

        b.set_insertion_point(resume_bb);
        b.coro_resume(1u, nullptr);
        b.br(after_if);

        b.set_insertion_point(after_if);
        auto *reloaded_state = b.load(Type::of<int>(), state);
        auto *next_state = b.call(Type::of<int>(), ArithmeticOp::BINARY_ADD, {reloaded_state, one});
        b.store(state, next_state);
        b.br(loop_hdr);

        b.set_insertion_point(exit);
        b.return_void();

        // when
        auto result = coro_cfg_distill_pass_run_on_function(k);

        // then: the updated local is stored on suspend edges into the
        // continuation scope, including the loop-carried self edge
        expect(result.scopes.size() == 2u);
        auto has_state_store = [](const CoroCfgDistillResult::Edge &edge) noexcept {
            for (auto &name : edge.store_variables) {
                if (name == "state") { return true; }
            }
            return false;
        };
        bool entry_edge_ok = false;
        bool loop_edge_ok = false;
        for (auto &edge : result.transition_edges) {
            if (edge.token != 1u) { continue; }
            if (edge.from_scope == 0u && edge.to_scope == 1u) {
                entry_edge_ok = has_state_store(edge);
            }
            if (edge.from_scope == 1u && edge.to_scope == 1u) {
                loop_edge_ok = has_state_store(edge);
            }
        }
        expect(entry_edge_ok);
        expect(loop_edge_ok);
    };

    "per_edge_store_excludes_post_suspend_touches"_test = [] {
        Module m;
        BasicBlock *body;
        auto *k = make_kernel_with_body(m, body);
        XIRBuilder b;
        auto *cond = m.create_constant_one(Type::of<bool>());
        auto *zero = m.create_constant_zero(Type::of<int>());
        auto *one = m.create_constant_one(Type::of<int>());
        auto two_value = 2;
        auto *two = m.create_constant(Type::of<int>(), &two_value);

        auto *suspend_bb = k->create_basic_block();
        auto *resume_bb = k->create_basic_block();

        b.set_insertion_point(body);
        auto *state = b.alloca_local(Type::of<int>());
        state->set_name("state");
        auto *late = b.alloca_local(Type::of<int>());
        late->set_name("late");
        b.store(state, one);
        b.store(late, zero);
        b.cond_br(cond, suspend_bb, resume_bb);

        b.set_insertion_point(suspend_bb);
        b.coro_suspend(1u, "s", nullptr);

        b.set_insertion_point(resume_bb);
        b.coro_resume(1u, nullptr);
        b.store(late, two);
        auto *a = b.load(Type::of<int>(), state);
        auto *c = b.load(Type::of<int>(), late);
        auto *sum = b.call(Type::of<int>(), ArithmeticOp::BINARY_ADD, {a, c});
        static_cast<void>(sum);
        b.return_void();

        auto result = coro_cfg_distill_pass_run_on_function(k);
        const CoroCfgDistillResult::Edge *suspend_edge = nullptr;
        for (auto &edge : result.transition_edges) {
            if (edge.is_suspend && edge.token == 1u) {
                suspend_edge = &edge;
                break;
            }
        }
        expect(suspend_edge != nullptr);
        bool stores_state = false;
        bool stores_late = false;
        if (suspend_edge != nullptr) {
            for (auto &name : suspend_edge->store_variables) {
                if (name == "state") { stores_state = true; }
                if (name == "late") { stores_late = true; }
            }
        }
        expect(stores_state);
        expect(!stores_late);
    };

    "cross_scope_branch_has_transition_store"_test = [] {
        Module m;
        BasicBlock *body;
        auto *k = make_kernel_with_body(m, body);
        XIRBuilder b;
        auto *cond = m.create_constant_one(Type::of<bool>());
        auto *one = m.create_constant_one(Type::of<int>());

        auto *suspend_bb = k->create_basic_block();
        auto *resume_bb = k->create_basic_block();
        auto *skip_bb = k->create_basic_block();
        auto *merge_bb = k->create_basic_block();

        b.set_insertion_point(body);
        auto *state = b.alloca_local(Type::of<int>());
        state->set_name("state");
        b.store(state, one);
        b.cond_br(cond, suspend_bb, skip_bb);

        b.set_insertion_point(suspend_bb);
        b.coro_suspend(1u, "s", nullptr);

        b.set_insertion_point(skip_bb);
        b.br(resume_bb);

        b.set_insertion_point(resume_bb);
        b.coro_resume(1u, nullptr);
        b.br(merge_bb);

        b.set_insertion_point(merge_bb);
        auto *loaded = b.load(Type::of<int>(), state);
        static_cast<void>(loaded);
        b.return_void();

        auto result = coro_cfg_distill_pass_run_on_function(k);
        bool found_branch_edge = false;
        bool branch_stores_state = false;
        bool found_suspend_edge = false;
        for (auto &edge : result.transition_edges) {
            if (edge.is_suspend) {
                found_suspend_edge = true;
            } else if (edge.from_scope == 0u && edge.to_scope == 1u && edge.exit_block == skip_bb) {
                found_branch_edge = true;
                for (auto &name : edge.store_variables) {
                    if (name == "state") { branch_stores_state = true; }
                }
            }
        }
        expect(found_suspend_edge);
        expect(found_branch_edge);
        expect(branch_stores_state);
    };

    "distilled_scopes_may_share_bypass_merge_blocks"_test = [] {
        // Scope regions are rooted reachability sets rather than a partition.
        // The shared merge is reached directly by the entry scope and through
        // the resume root by the continuation scope. Dense dataflow must use
        // an explicit (scope, block) membership relation; assigning the block
        // one global local index loses one of these two executions.
        Module m;
        BasicBlock *body;
        auto *k = make_kernel_with_body(m, body);
        XIRBuilder b;
        auto *cond = m.create_constant_one(Type::of<bool>());
        auto *one = m.create_constant_one(Type::of<int>());

        auto *suspend_block = k->create_basic_block();
        auto *bypass_block = k->create_basic_block();
        auto *resume_block = k->create_basic_block();
        auto *shared_merge = k->create_basic_block();

        b.set_insertion_point(body);
        auto *state = b.alloca_local(Type::of<int>());
        state->set_name("shared_merge_state");
        b.store(state, one);
        b.cond_br(cond, suspend_block, bypass_block);

        b.set_insertion_point(suspend_block);
        b.coro_suspend(1u, "shared-merge", nullptr);

        b.set_insertion_point(bypass_block);
        b.br(shared_merge);

        b.set_insertion_point(resume_block);
        b.coro_resume(1u, nullptr);
        // This value is reachable only from the logical resume root. Ordinary
        // raw-CFG traversal from the function body cannot see it, but the
        // shared coroutine value domain must still assign it a coordinate.
        auto *resume_only = b.call(
            Type::of<int>(), ArithmeticOp::BINARY_ADD, {one, one});
        static_cast<void>(resume_only);
        b.br(shared_merge);

        b.set_insertion_point(shared_merge);
        auto *loaded = b.load(Type::of<int>(), state);
        static_cast<void>(loaded);
        b.return_void();

        auto result = coro_cfg_distill_pass_run_on_function(k);
        expect(result.succeeded());
        expect(result.scopes.size() == 2u);
        auto merge_membership_count = size_t{0u};
        for (auto &scope : result.scopes) {
            if (std::find(scope.blocks.begin(), scope.blocks.end(),
                          shared_merge) != scope.blocks.end()) {
                ++merge_membership_count;
            }
        }
        expect(merge_membership_count == 2u);
        auto suspend_stores_state = false;
        for (auto &edge : result.transition_edges) {
            if (!edge.is_suspend || edge.token != 1u) { continue; }
            suspend_stores_state =
                std::find(edge.store_values.begin(),
                          edge.store_values.end(), state) !=
                edge.store_values.end();
        }
        expect(suspend_stores_state);
    };

    "frame_values_sorted_by_alignment_and_size"_test = [] {
        Module m;
        BasicBlock *body;
        auto *k = make_kernel_with_body(m, body);
        XIRBuilder b;
        auto *one_i = m.create_constant_one(Type::of<int>());
        auto *one_f = m.create_constant_one(Type::of<float>());
        auto *cond = m.create_constant_one(Type::of<bool>());
        auto *float3_ty = Type::of<float3>();

        auto *suspend_bb = k->create_basic_block();
        auto *resume_bb = k->create_basic_block();

        b.set_insertion_point(body);
        auto *small = b.alloca_local(Type::of<int>());
        small->set_name("small");
        auto *medium = b.alloca_local(Type::of<float>());
        medium->set_name("medium");
        auto *large = b.alloca_local(float3_ty);
        large->set_name("large");
        b.store(small, one_i);
        b.store(medium, one_f);
        auto *large_value = b.call(float3_ty, ArithmeticOp::AGGREGATE, {one_f, one_f, one_f});
        b.store(large, large_value);
        b.cond_br(cond, suspend_bb, resume_bb);

        b.set_insertion_point(suspend_bb);
        b.coro_suspend(1u, "s", nullptr);

        b.set_insertion_point(resume_bb);
        b.coro_resume(1u, nullptr);
        auto *loaded_small = b.load(Type::of<int>(), small);
        auto *loaded_medium = b.load(Type::of<float>(), medium);
        auto *loaded_large = b.load(float3_ty, large);
        auto *loaded_large_x = b.call(Type::of<float>(), ArithmeticOp::EXTRACT, {loaded_large, m.create_constant_zero(Type::of<uint32_t>())});
        auto *medium_i = b.static_cast_(Type::of<int>(), loaded_medium);
        auto *large_i = b.static_cast_(Type::of<int>(), loaded_large_x);
        auto *sum0 = b.call(Type::of<int>(), ArithmeticOp::BINARY_ADD, {loaded_small, medium_i});
        auto *sum1 = b.call(Type::of<int>(), ArithmeticOp::BINARY_ADD, {sum0, large_i});
        static_cast<void>(sum1);
        b.return_void();

        auto result = coro_cfg_distill_pass_run_on_function(k);
        expect(result.frame_values.size() == 3u);
        expect(result.frame_values[0u].name == "large");
        expect(result.frame_values[0u].type == float3_ty);
        expect(result.frame_values[1u].type->alignment() >= result.frame_values[2u].type->alignment());
    };

    "partial_aggregate_store_preserves_live_in_frame_value"_test = [] {
        Module m;
        BasicBlock *body;
        auto *k = make_kernel_with_body(m, body);
        auto *pair_type = Type::structure({Type::of<float>(), Type::of<float>()});
        XIRBuilder b;
        b.set_insertion_point(body);
        auto *state = b.alloca_local(pair_type);
        state->set_name("state");
        b.store(state, m.create_constant_zero(pair_type));
        b.coro_suspend(1u, "first", nullptr);

        auto *resume_first = k->create_basic_block();
        b.set_insertion_point(resume_first);
        b.coro_resume(1u, nullptr);
        uint32_t first_index = 0u;
        auto *first = m.create_constant(Type::of<uint32_t>(), &first_index);
        auto *first_ptr = b.gep(Type::of<float>(), state, {first});
        b.store(first_ptr, m.create_constant_one(Type::of<float>()));
        b.coro_suspend(2u, "second", nullptr);

        auto *resume_second = k->create_basic_block();
        b.set_insertion_point(resume_second);
        b.coro_resume(2u, nullptr);
        uint32_t second_index = 1u;
        auto *second = m.create_constant(Type::of<uint32_t>(), &second_index);
        auto *second_ptr = b.gep(Type::of<float>(), state, {second});
        auto *value = b.load(Type::of<float>(), second_ptr);
        static_cast<void>(value);
        b.return_void();

        auto result = coro_cfg_distill_pass_run_on_function(k);
        expect(result.scopes.size() == 3u);
        expect(result.scopes[1u].live_in_values.size() == 1u);
        expect(result.scopes[1u].live_in_values[0u] == state);
        bool stored_on_second_suspend = false;
        for (auto &edge : result.transition_edges) {
            if (edge.is_suspend && edge.from_scope == 1u && edge.to_scope == 2u) {
                for (auto *stored : edge.store_values) {
                    if (stored == state) { stored_on_second_suspend = true; }
                }
            }
        }
        expect(stored_on_second_suspend);
    };

    "duplicate_alloca_names_get_distinct_frame_field_names"_test = [] {
        Module m;
        BasicBlock *body;
        auto *k = make_kernel_with_body(m, body);
        XIRBuilder b;
        b.set_insertion_point(body);
        auto *lhs = b.alloca_local(Type::of<float>());
        auto *rhs = b.alloca_local(Type::of<float>());
        lhs->set_name("duplicate");
        rhs->set_name("duplicate");
        b.store(lhs, m.create_constant_one(Type::of<float>()));
        b.store(rhs, m.create_constant_one(Type::of<float>()));
        b.coro_suspend(1u, "checkpoint", nullptr);

        auto *resume = k->create_basic_block();
        b.set_insertion_point(resume);
        b.coro_resume(1u, nullptr);
        auto *lhs_value = b.load(Type::of<float>(), lhs);
        auto *rhs_value = b.load(Type::of<float>(), rhs);
        auto *sum = b.call(Type::of<float>(), ArithmeticOp::BINARY_ADD,
                           {lhs_value, rhs_value});
        static_cast<void>(sum);
        b.return_void();

        auto result = coro_cfg_distill_pass_run_on_function(k);
        expect(result.frame_values.size() == 2u);
        expect(result.frame_values[0u].name != result.frame_values[1u].name);
    };

    "structured_switch_is_rejected_until_destructured"_test = [] {
        Module m;
        auto *k = m.create_kernel();
        auto *selector = k->create_value_argument(Type::of<uint32_t>());
        auto *entry = k->create_body_block();
        auto *case_block = k->create_basic_block();
        auto *default_block = k->create_basic_block();
        auto *merge = k->create_basic_block();
        XIRBuilder b;
        b.set_insertion_point(entry);
        auto *switch_inst = b.switch_(selector);
        switch_inst->set_default_block(default_block);
        switch_inst->add_case(7u, case_block);
        switch_inst->set_merge_block(merge);
        b.set_insertion_point(case_block);
        b.br(merge);
        b.set_insertion_point(default_block);
        b.br(merge);
        b.set_insertion_point(merge);
        b.return_void();

        expect(xir_verify_module(&m).succeeded());
        auto before = xir_to_text_translate(&m, true);
        auto rejected = coro_cfg_distill_pass_run_on_function(k);
        expect(!rejected.succeeded());
        expect(rejected.structured_cfg_error_count == 1u);
        expect(rejected.invalid_input_error_count == 0u);
        expect(rejected.invalid_cfg_error_count == 0u);
        expect(rejected.scopes.empty());
        expect(xir_to_text_translate(&m, true) == before);
        expect(xir_verify_module(&m).succeeded());

        auto destructured = destructure_cfg_pass_run_on_function(k);
        expect(destructured.succeeded());
        expect(destructured.destructured_switch_count == 1u);
        expect(entry->terminator()->isa<IndexedBranchInst>());
        auto accepted = coro_cfg_distill_pass_run_on_function(k);
        expect(accepted.succeeded());
        expect(accepted.scopes.size() == 1u);
        expect(xir_verify_module(&m).succeeded());
    };

    "null_and_declaration_inputs_fail_closed"_test = [] {
        Module m;
        auto *external = m.create_external_function(nullptr);
        auto null_result =
            coro_cfg_distill_pass_run_on_function(nullptr);
        auto external_result =
            coro_cfg_distill_pass_run_on_function(external);
        expect(!null_result.succeeded());
        expect(null_result.invalid_input_error_count == 1u);
        expect(null_result.scopes.empty());
        expect(!external_result.succeeded());
        expect(external_result.invalid_input_error_count == 1u);
        expect(external_result.scopes.empty());
        expect(coro_cfg_distill_pass_run_on_module(nullptr) == 0u);
    };

    "missing_and_duplicate_coroutine_tokens_fail_closed"_test = [] {
        {
            Module m;
            auto *k = m.create_kernel();
            auto *body = k->create_body_block();
            XIRBuilder b;
            b.set_insertion_point(body);
            b.coro_suspend(7u, "missing_resume", nullptr);

            expect(xir_verify_module(&m).succeeded());
            auto *before = body->terminator();
            auto result = coro_cfg_distill_pass_run_on_function(k);
            expect(!result.succeeded());
            expect(result.invalid_cfg_error_count == 1u);
            expect(result.scopes.empty());
            expect(body->terminator() == before);
            expect(xir_verify_module(&m).succeeded());
        }
        {
            Module m;
            auto *k = m.create_kernel();
            auto *body = k->create_body_block();
            auto *resume0 = k->create_basic_block();
            auto *resume1 = k->create_basic_block();
            XIRBuilder b;
            b.set_insertion_point(body);
            b.coro_suspend(9u, "duplicate_resume", nullptr);
            b.set_insertion_point(resume0);
            b.coro_resume(9u, nullptr);
            b.return_void();
            b.set_insertion_point(resume1);
            b.coro_resume(9u, nullptr);
            b.return_void();

            expect(xir_verify_module(&m).succeeded());
            auto *before_suspend = body->terminator();
            auto *before_resume0 = resume0->terminator();
            auto *before_resume1 = resume1->terminator();
            auto result = coro_cfg_distill_pass_run_on_function(k);
            expect(!result.succeeded());
            expect(result.invalid_cfg_error_count == 1u);
            expect(result.scopes.empty());
            expect(body->terminator() == before_suspend);
            expect(resume0->terminator() == before_resume0);
            expect(resume1->terminator() == before_resume1);
            expect(xir_verify_module(&m).succeeded());
        }
    };

    "phi_cfg_is_rejected_until_reg2mem_makes_edges_explicit"_test = [] {
        Module m;
        auto *callable = m.create_callable(nullptr);
        auto *condition =
            callable->create_value_argument(Type::of<bool>());
        auto *entry = callable->create_body_block();
        auto *left = callable->create_basic_block();
        auto *right = callable->create_basic_block();
        auto *join = callable->create_basic_block();
        auto *resume = callable->create_basic_block();
        XIRBuilder b;
        b.set_insertion_point(entry);
        b.cond_br(condition, left, right);
        b.set_insertion_point(left);
        b.br(join);
        b.set_insertion_point(right);
        b.br(join);
        b.set_insertion_point(join);
        auto *phi = b.phi(Type::of<int>());
        phi->add_incoming(
            m.create_constant_zero(Type::of<int>()), left);
        phi->add_incoming(
            m.create_constant_one(Type::of<int>()), right);
        phi->set_name("edge_selected_value");
        auto *sum = b.call(
            Type::of<int>(), ArithmeticOp::BINARY_ADD,
            {phi, m.create_constant_one(Type::of<int>())});
        static_cast<void>(sum);
        b.coro_suspend(17u, "phi", nullptr);
        b.set_insertion_point(resume);
        b.coro_resume(17u, nullptr);
        b.return_void();

        expect(xir_verify_module(&m).succeeded());
        auto rejected =
            coro_cfg_distill_pass_run_on_function(callable);
        expect(!rejected.succeeded());
        expect(rejected.invalid_cfg_error_count == 1u);
        expect(rejected.scopes.empty());
        expect(phi->is_linked());

        auto lowered = coro_reg2mem_pass_run_on_module(&m);
        expect(lowered.lowered_phi_count == 1u);
        expect(!phi->is_linked());
        expect(xir_verify_module(&m).succeeded());
        auto accepted =
            coro_cfg_distill_pass_run_on_function(callable);
        expect(accepted.succeeded());
        expect(accepted.scopes.size() == 2u);
    };

    "two_resume_identities_cannot_share_one_block"_test = [] {
        Module m;
        BasicBlock *entry;
        auto *kernel = make_kernel_with_body(m, entry);
        auto *left_suspend = kernel->create_basic_block();
        auto *right_suspend = kernel->create_basic_block();
        auto *shared_resume = kernel->create_basic_block();
        XIRBuilder b;
        b.set_insertion_point(entry);
        b.cond_br(
            kernel->create_value_argument(Type::of<bool>()),
            left_suspend, right_suspend);
        b.set_insertion_point(left_suspend);
        b.coro_suspend(21u, "left", nullptr);
        b.set_insertion_point(right_suspend);
        b.coro_suspend(22u, "right", nullptr);
        b.set_insertion_point(shared_resume);
        auto *first = b.coro_resume(21u, nullptr);
        auto *second = b.coro_resume(22u, nullptr);
        b.return_void();

        expect(xir_verify_module(&m).succeeded());
        auto result =
            coro_cfg_distill_pass_run_on_function(kernel);
        expect(!result.succeeded());
        expect(result.invalid_cfg_error_count == 1u);
        expect(result.scopes.empty());
        expect(first->is_linked());
        expect(second->is_linked());
        expect(xir_verify_module(&m).succeeded());
    };

    "entry_block_cannot_alias_a_resume_root"_test = [] {
        Module m;
        BasicBlock *entry;
        auto *kernel = make_kernel_with_body(m, entry);
        auto *suspend = kernel->create_basic_block();
        auto *exit = kernel->create_basic_block();
        XIRBuilder b;
        b.set_insertion_point(entry);
        auto *resume = b.coro_resume(27u, nullptr);
        b.cond_br(
            kernel->create_value_argument(Type::of<bool>()),
            suspend, exit);
        b.set_insertion_point(suspend);
        auto *suspend_inst =
            b.coro_suspend(27u, "entry-alias", nullptr);
        b.set_insertion_point(exit);
        b.return_void();

        expect(xir_verify_module(&m).succeeded());
        auto result =
            coro_cfg_distill_pass_run_on_function(kernel);
        expect(!result.succeeded());
        expect(result.invalid_cfg_error_count == 1u);
        expect(result.scopes.empty());
        expect(resume->is_linked());
        expect(suspend->terminator() == suspend_inst);
        expect(xir_verify_module(&m).succeeded());
    };

    "non_void_coroutine_is_rejected_before_continuation_abi_change"_test = [] {
        Module m;
        auto *callable = m.create_callable(Type::of<int>());
        auto *entry = callable->create_body_block();
        auto *resume = callable->create_basic_block();
        XIRBuilder b;
        b.set_insertion_point(entry);
        auto *suspend =
            b.coro_suspend(31u, "non-void", nullptr);
        b.set_insertion_point(resume);
        b.coro_resume(31u, nullptr);
        b.return_(m.create_constant_one(Type::of<int>()));

        expect(xir_verify_module(&m).succeeded());
        auto result =
            coro_cfg_distill_pass_run_on_function(callable);
        expect(!result.succeeded());
        expect(result.invalid_cfg_error_count == 1u);
        expect(result.scopes.empty());
        expect(entry->terminator() == suspend);
        expect(xir_verify_module(&m).succeeded());
    };
}

int main(int argc, char *argv[]) {
    (void)argc;
    (void)argv;
    reg_coro_cfg_distill();
    return 0;
}
