#include "ut/ut.hpp"
#include <luisa/ast/type_registry.h>
#include <luisa/xir/basic_block.h>
#include <luisa/xir/builder.h>
#include <luisa/xir/function.h>
#include <luisa/xir/module.h>
#include <luisa/xir/passes/post_dom_tree.h>

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
}

int main(int argc, char *argv[]) {
    boost::ut::detail::cfg::parse_arg_with_fallback(argc, const_cast<const char **>(argv));
    register_post_dom_tree_tests();
    return 0;
}
