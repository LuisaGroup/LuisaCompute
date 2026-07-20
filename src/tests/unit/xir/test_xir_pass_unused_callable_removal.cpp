#include "ut/ut.hpp"

#include <luisa/ast/type.h>
#include <luisa/xir/builder.h>
#include <luisa/xir/module.h>
#include <luisa/xir/passes/unused_callable_removal.h>

using namespace luisa::compute;
using namespace luisa::compute::xir;
using namespace boost::ut;
using namespace boost::ut::literals;

int main() {

    "unused_callable_keeps_disconnected_owned_reference"_test = [] {
        Module m;
        auto *callee = m.create_callable(Type::of<void>());
        auto *callee_body = callee->create_body_block();
        XIRBuilder b;
        b.set_insertion_point(callee_body);
        b.return_void();

        auto *kernel = m.create_kernel();
        auto *body = kernel->create_body_block();
        b.set_insertion_point(body);
        b.return_void();
        auto *disconnected = kernel->create_basic_block();
        b.set_insertion_point(disconnected);
        auto *call = b.call(nullptr, callee, {});
        b.return_void();

        auto info = unused_callable_removal_pass_run_on_module(&m);
        expect(info.removed_callable_count == 0u);
        expect(callee->is_linked());
        expect(call->callee() == callee);
    };

    "unused_callable_chain_removed_callers_first"_test = [] {
        Module m;
        // Deliberately create the callee first so module order alone is unsafe.
        auto *leaf = m.create_callable(Type::of<void>());
        auto *leaf_body = leaf->create_body_block();
        XIRBuilder b;
        b.set_insertion_point(leaf_body);
        b.return_void();

        auto *caller = m.create_callable(Type::of<void>());
        auto *caller_body = caller->create_body_block();
        b.set_insertion_point(caller_body);
        b.call(nullptr, leaf, {});
        b.return_void();

        auto info = unused_callable_removal_pass_run_on_module(&m);
        expect(info.removed_callable_count == 2u);
        expect(m.function_list().empty());
    };

    "unused_recursive_scc_is_conservatively_retained"_test = [] {
        Module m;
        auto *a = m.create_callable(Type::of<void>());
        auto *b_fn = m.create_callable(Type::of<void>());
        auto *a_body = a->create_body_block();
        auto *b_body = b_fn->create_body_block();
        XIRBuilder builder;
        builder.set_insertion_point(a_body);
        builder.call(nullptr, b_fn, {});
        builder.return_void();
        builder.set_insertion_point(b_body);
        builder.call(nullptr, a, {});
        builder.return_void();

        auto info = unused_callable_removal_pass_run_on_module(&m);
        expect(info.removed_callable_count == 0u);
        expect(a->is_linked());
        expect(b_fn->is_linked());
    };
}
