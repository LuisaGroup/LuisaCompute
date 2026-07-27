// Test for XIR post-dominator analysis on branching and malformed CFGs.

#include "ut/ut.hpp"
#include <luisa/ast/type_registry.h>
#include <luisa/xir/basic_block.h>
#include <luisa/xir/builder.h>
#include <luisa/xir/function.h>
#include <luisa/xir/module.h>
#include <luisa/xir/passes/dom_tree.h>
#include <luisa/xir/passes/post_dom_tree.h>

#include <algorithm>

using namespace luisa;
using namespace luisa::compute;
using namespace luisa::compute::xir;
using namespace boost::ut;
using namespace boost::ut::literals;

namespace {

[[nodiscard]] KernelFunction *make_kernel_with_body(Module &m, BasicBlock *&body) noexcept {
    auto *kernel = m.create_kernel();
    body = kernel->create_body_block();
    return kernel;
}

}// namespace

void register_post_dom_tree_tests() {

    "post_dom_tree_coro_terminate_is_exit"_test = [] {
        Module m;
        BasicBlock *body;
        auto *kernel = make_kernel_with_body(m, body);
        auto *terminate = kernel->definition()->create_basic_block();
        XIRBuilder b;
        b.set_insertion_point(body);
        b.br(terminate);
        b.set_insertion_point(terminate);
        b.coro_terminate();

        auto tree = compute_post_dom_tree(kernel);
        expect(tree.contains(body));
        expect(tree.contains(terminate));
        expect(tree.immediate_post_dominator(body) == terminate);
        expect(tree.immediate_post_dominator(terminate) == nullptr);
        expect(tree.immediate_post_dominator(nullptr) == nullptr);
        expect(tree.post_dominates(terminate, body));
    };

    "post_dom_tree_return_does_not_hide_infinite_path"_test = [] {
        Module m;
        BasicBlock *body;
        auto *kernel = make_kernel_with_body(m, body);
        auto *definition = kernel->definition();
        auto *return_block = definition->create_basic_block();
        auto *cycle = definition->create_basic_block();
        auto *condition = kernel->create_value_argument(Type::of<bool>());
        XIRBuilder b;
        b.set_insertion_point(body);
        b.cond_br(condition, return_block, cycle);
        b.set_insertion_point(return_block);
        b.return_void();
        b.set_insertion_point(cycle);
        b.br(cycle);

        auto tree = compute_post_dom_tree(kernel);
        expect(tree.contains(body));
        expect(tree.contains(return_block));
        expect(tree.contains(cycle));
        expect(tree.immediate_post_dominator(body) == nullptr);
        expect(tree.immediate_post_dominator(cycle) == nullptr);
        expect(!tree.post_dominates(return_block, body));
        expect(!tree.post_dominates(cycle, body));
    };

    "post_dom_tree_mixed_exit_self_cycle_has_virtual_exit"_test = [] {
        Module m;
        BasicBlock *body;
        auto *kernel = make_kernel_with_body(m, body);
        auto *return_block = kernel->create_basic_block();
        auto *condition = kernel->create_value_argument(Type::of<bool>());
        XIRBuilder b;
        b.set_insertion_point(body);
        b.cond_br(condition, return_block, body);
        b.set_insertion_point(return_block);
        b.return_void();

        auto tree = compute_post_dom_tree(kernel);
        expect(tree.immediate_post_dominator(body) == nullptr);
        expect(!tree.post_dominates(return_block, body));
    };

    "dom_tree_queries_reject_blocks_outside_the_analysis"_test = [] {
        Module m;
        BasicBlock *body;
        auto *kernel = make_kernel_with_body(m, body);
        XIRBuilder b;
        b.set_insertion_point(body);
        b.return_void();

        auto dom_tree = compute_dom_tree(kernel);
        auto post_dom_tree = compute_post_dom_tree(kernel);
        auto *foreign = reinterpret_cast<BasicBlock *>(uintptr_t{0xdead});
        expect(!dom_tree.dominates(foreign, foreign));
        expect(!dom_tree.strictly_dominates(foreign, foreign));
        expect(!post_dom_tree.post_dominates(foreign, foreign));
        expect(!post_dom_tree.strictly_post_dominates(foreign, foreign));
        expect(!post_dom_tree.post_dominates(nullptr, foreign));
    };

    "dom_tree_entry_backedges_compute_root_frontier"_test = [] {
        Module m;
        BasicBlock *body;
        auto *kernel = make_kernel_with_body(m, body);
        auto *left = kernel->create_basic_block();
        auto *right = kernel->create_basic_block();
        auto *condition = kernel->create_value_argument(Type::of<bool>());
        XIRBuilder b;
        b.set_insertion_point(body);
        b.cond_br(condition, left, right);
        b.set_insertion_point(left);
        b.br(body);
        b.set_insertion_point(right);
        b.br(body);

        auto tree = compute_dom_tree(kernel);
        auto frontiers = tree.root()->frontiers();
        expect(std::find(frontiers.begin(), frontiers.end(), tree.root()) != frontiers.end());
    };

    "dom_trees_ignore_unreachable_predecessors"_test = [] {
        Module m;
        BasicBlock *body;
        auto *kernel = make_kernel_with_body(m, body);
        auto *orphan = kernel->create_basic_block();
        XIRBuilder b;
        b.set_insertion_point(body);
        b.return_void();
        b.set_insertion_point(orphan);
        b.br(body);

        auto dom_tree = compute_dom_tree(kernel);
        auto post_dom_tree = compute_post_dom_tree(kernel);
        expect(dom_tree.contains(body));
        expect(!dom_tree.contains(orphan));
        expect(post_dom_tree.contains(body));
        expect(!post_dom_tree.contains(orphan));
    };
}

int main(int argc, char *argv[]) {
    boost::ut::detail::cfg::parse_arg_with_fallback(argc, const_cast<const char **>(argv));
    register_post_dom_tree_tests();
    return 0;
}
