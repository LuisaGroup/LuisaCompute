#include "ut/ut.hpp"
#include <luisa/coro/coro_graph.h>
#include <luisa/xir/basic_block.h>
#include <luisa/xir/builder.h>
#include <luisa/xir/function.h>
#include <luisa/xir/instructions/branch.h>
#include <luisa/xir/instructions/coro.h>
#include <luisa/xir/instructions/return.h>
#include <luisa/xir/module.h>
#include <luisa/xir/passes/coro_cfg_distill.h>
#include <luisa/xir/passes/coro_materialize.h>
#include <luisa/xir/passes/coro_split.h>

using namespace luisa;
using namespace luisa::compute;
using namespace luisa::compute::xir;
using namespace luisa::compute::coro;
using namespace boost::ut;
using namespace boost::ut::literals;

namespace {

[[nodiscard]] KernelFunction *make_kernel_with_body(Module &m, BasicBlock *&body_out) noexcept {
    auto *k = m.create_kernel();
    body_out = k->create_body_block();
    return k;
}

}// namespace

void reg_coro_graph() {

    "one_suspend_two_nodes_one_edge"_test = [] {
        // given: a kernel with one suspend point
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
        b.coro_suspend(1u, "checkpoint", nullptr);

        b.set_insertion_point(resume_bb);
        b.coro_resume(1u, nullptr);
        b.return_void();

        // Run cfg-distill (read-only) on the kernel before splitting
        auto cfg = coro_cfg_distill_pass_run_on_function(k);

        // Run the coroutine pipeline
        (void)coro_split_pass_run_on_module(&m);
        auto info = coro_materialize_pass_run_on_module(&m);

        // when: build CoroGraph
        auto graph = CoroGraph::from_module(m, info, cfg);

        // then: 2 nodes, 1 edge
        expect(graph.node_count() == 2u);
        expect(graph.edge_count() == 1u);

        // entry node (index 0)
        auto &n0 = graph.node(0u);
        expect(n0.index == 0u);
        expect(n0.token == 0u);// entry token
        expect(n0.name.empty());
        expect(!n0.is_terminal);
        expect(n0.callable != nullptr);

        // scope 1 node
        auto &n1 = graph.node(1u);
        expect(n1.index == 1u);
        expect(n1.token == 1u);
        expect(n1.name == "checkpoint");
        expect(!n1.is_terminal);
        expect(n1.callable != nullptr);

        // edge from 0 → 1
        auto *e = graph.edge(0u, 1u);
        expect(e != nullptr);
        expect(e->from_index == 0u);
        expect(e->to_index == 1u);

        // token lookup
        auto *t0 = graph.node_by_token(0u);
        expect(t0 != nullptr);
        expect(t0->index == 0u);

        auto *t1 = graph.node_by_token(1u);
        expect(t1 != nullptr);
        expect(t1->index == 1u);

        // name lookup
        auto *nm = graph.node_by_name("checkpoint");
        expect(nm != nullptr);
        expect(nm->index == 1u);
    };

    "three_suspends_four_nodes"_test = [] {
        // given: a kernel with three suspend points (linear chain)
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
        b.coro_suspend(10u, "alpha", nullptr);
        b.set_insertion_point(r1);
        b.coro_resume(10u, nullptr);

        // suspend 2
        auto *s2 = k->create_basic_block();
        auto *r2 = k->create_basic_block();
        b.cond_br(cond, s2, r2);
        b.set_insertion_point(s2);
        b.coro_suspend(20u, "beta", nullptr);
        b.set_insertion_point(r2);
        b.coro_resume(20u, nullptr);

        // suspend 3
        auto *s3 = k->create_basic_block();
        auto *r3 = k->create_basic_block();
        b.cond_br(cond, s3, r3);
        b.set_insertion_point(s3);
        b.coro_suspend(30u, "gamma", nullptr);
        b.set_insertion_point(r3);
        b.coro_resume(30u, nullptr);
        b.return_void();

        // Run passes
        auto cfg = coro_cfg_distill_pass_run_on_function(k);
        (void)coro_split_pass_run_on_module(&m);
        auto info = coro_materialize_pass_run_on_module(&m);

        // when
        auto graph = CoroGraph::from_module(m, info, cfg);

        // then: 4 nodes, 3 edges
        expect(graph.node_count() == 4u);
        expect(graph.edge_count() == 3u);

        // node 0: entry
        auto &n0 = graph.node(0u);
        expect(n0.index == 0u);
        expect(n0.token == 0u);
        expect(!n0.is_terminal);

        // node 1: alpha
        auto &n1 = graph.node(1u);
        expect(n1.index == 1u);
        expect(n1.token == 10u);
        expect(n1.name == "alpha");
        expect(!n1.is_terminal);

        // node 2: beta
        auto &n2 = graph.node(2u);
        expect(n2.index == 2u);
        expect(n2.token == 20u);
        expect(n2.name == "beta");
        expect(!n2.is_terminal);

        // node 3: gamma (last, non-terminal because no CoroTerminateInst)
        auto &n3 = graph.node(3u);
        expect(n3.index == 3u);
        expect(n3.token == 30u);
        expect(n3.name == "gamma");
        expect(!n3.is_terminal);

        // Verify all edges exist
        expect(graph.edge(0u, 1u) != nullptr);
        expect(graph.edge(1u, 2u) != nullptr);
        expect(graph.edge(2u, 3u) != nullptr);

        // Name lookup for all
        expect(graph.node_by_name("alpha") != nullptr);
        expect(graph.node_by_name("beta") != nullptr);
        expect(graph.node_by_name("gamma") != nullptr);
        expect(graph.node_by_name("nonexistent") == nullptr);

        // Token lookup for all
        expect(graph.node_by_token(0u) != nullptr);
        expect(graph.node_by_token(10u) != nullptr);
        expect(graph.node_by_token(20u) != nullptr);
        expect(graph.node_by_token(30u) != nullptr);
        expect(graph.node_by_token(999u) == nullptr);
    };

    "terminal_scope_has_terminal_flag"_test = [] {
        // given: a kernel that ends with CoroTerminateInst
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
        b.coro_suspend(1u, "middle", nullptr);

        b.set_insertion_point(resume_bb);
        b.coro_resume(1u, nullptr);

        // terminal block
        auto *term_bb = k->create_basic_block();
        b.br(term_bb);

        b.set_insertion_point(term_bb);
        b.coro_terminate();

        // Run passes
        auto cfg = coro_cfg_distill_pass_run_on_function(k);
        (void)coro_split_pass_run_on_module(&m);
        auto info = coro_materialize_pass_run_on_module(&m);

        // when
        auto graph = CoroGraph::from_module(m, info, cfg);

        // then: last scope is terminal
        expect(graph.node_count() >= 1u);
        auto &last = graph.node(graph.node_count() - 1u);
        expect(last.is_terminal);

        // entry is not terminal
        auto &entry = graph.node(0u);
        expect(!entry.is_terminal);
    };

    "named_token_lookup_by_string_name"_test = [] {
        // given: a kernel with two suspends, each with distinct names
        Module m;
        BasicBlock *body;
        auto *k = make_kernel_with_body(m, body);
        XIRBuilder b;
        auto *cond = m.create_constant_one(Type::of<bool>());

        // suspend "first_half"
        auto *s1 = k->create_basic_block();
        auto *r1 = k->create_basic_block();
        b.set_insertion_point(body);
        b.cond_br(cond, s1, r1);
        b.set_insertion_point(s1);
        b.coro_suspend(100u, "first_half", nullptr);
        b.set_insertion_point(r1);
        b.coro_resume(100u, nullptr);

        // suspend "second_half"
        auto *s2 = k->create_basic_block();
        auto *r2 = k->create_basic_block();
        b.cond_br(cond, s2, r2);
        b.set_insertion_point(s2);
        b.coro_suspend(200u, "second_half", nullptr);
        b.set_insertion_point(r2);
        b.coro_resume(200u, nullptr);
        b.return_void();

        // Run passes
        auto cfg = coro_cfg_distill_pass_run_on_function(k);
        (void)coro_split_pass_run_on_module(&m);
        auto info = coro_materialize_pass_run_on_module(&m);

        // when
        auto graph = CoroGraph::from_module(m, info, cfg);

        // then: find nodes by their suspend names
        auto *first = graph.node_by_name("first_half");
        expect(first != nullptr);
        expect(first->token == 100u);
        expect(first->name == "first_half");

        auto *second = graph.node_by_name("second_half");
        expect(second != nullptr);
        expect(second->token == 200u);
        expect(second->name == "second_half");

        // Entry node should not be findable by name (empty name)
        // node_by_name with empty string should not return entry
        auto *entry = graph.node_by_name("");
        expect(entry == nullptr);

        // Nonexistent name returns nullptr
        auto *missing = graph.node_by_name("third_half");
        expect(missing == nullptr);
    };
}

int main(int /*argc*/, char * /*argv*/[]) {
    reg_coro_graph();
    return 0;
}
