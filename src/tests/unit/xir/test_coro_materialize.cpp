#include "ut/ut.hpp"
#include <luisa/ast/type_registry.h>
#include <luisa/xir/basic_block.h>
#include <luisa/xir/builder.h>
#include <luisa/xir/function.h>
#include <luisa/xir/instructions/alloca.h>
#include <luisa/xir/instructions/coroutine.h>
#include <luisa/xir/instructions/loop.h>
#include <luisa/xir/instructions/if.h>
#include <luisa/xir/module.h>
#include <luisa/xir/passes/coro_materialize.h>
#include <luisa/xir/translators/xir2ast.h>

using namespace luisa;
using namespace luisa::compute;
using namespace luisa::compute::xir;
using namespace luisa::compute::xir::coro;
using namespace boost::ut;
using namespace boost::ut::literals;

namespace {

[[nodiscard]] KernelFunction *make_kernel_with_body(Module &m, BasicBlock *&body_out) noexcept {
    auto *k = m.create_kernel();
    body_out = k->create_body_block();
    return k;
}

}// namespace

void reg_coro_materialize() {

    "materialize_flat_coroutine"_test = [] {
        Module m;
        BasicBlock *body;
        auto *k = make_kernel_with_body(m, body);
        XIRBuilder b;
        b.set_insertion_point(body);
        auto *a = b.alloca_local(Type::of<float>());
        b.coro_register(a, "a");
        b.coro_suspend(1u);
        b.coro_suspend(2u);
        b.load(Type::of<float>(), a);
        b.return_void();

        auto result = coro_materialize_run_on_function(k);
        expect(result.ok);
        expect(result.split_info.is_supported);
        expect(result.split_info.changed);
        // 3 scopes = 3 continuations (entry + 2 resumed)
        expect(result.split_info.continuations.size() == 3_u);
        // Each continuation should have a callable
        for (auto &cont : result.split_info.continuations) {
            expect(cont.callable != nullptr);
        }
        // Frame type should exist with target_token + 1 alloca slot
        expect(result.split_info.frame_type != nullptr);
        expect(result.split_info.frame_slots.size() == 1_u);
    };

    "materialize_loop_with_suspend"_test = [] {
        Module m;
        BasicBlock *body;
        auto *k = make_kernel_with_body(m, body);
        XIRBuilder b;
        b.set_insertion_point(body);
        auto *a = b.alloca_local(Type::of<float>());
        b.coro_register(a, "a");
        auto *loop = b.loop();
        auto *prepare = loop->create_prepare_block();
        auto *loop_body = loop->create_body_block();
        auto *update = loop->create_update_block();
        auto *merge = loop->create_merge_block();
        b.set_insertion_point(prepare);
        b.br(loop_body);
        b.set_insertion_point(loop_body);
        b.coro_suspend(1u);
        b.br(update);
        b.set_insertion_point(update);
        b.br(prepare);
        b.set_insertion_point(merge);
        b.return_void();

        auto result = coro_materialize_run_on_function(k);
        expect(result.ok);
        expect(result.split_info.is_supported);
        // At least 2 scopes: entry + continuation for token 1
        expect(result.split_info.continuations.size() >= 2_u);
        // Each continuation should survive xir2ast roundtrip
        for (auto &cont : result.split_info.continuations) {
            expect(cont.callable != nullptr);
            auto ast = xir_to_ast_translate(*cont.callable->definition(), {});
            expect(ast != nullptr);
        }
    };

    "materialize_if_with_suspend"_test = [] {
        Module m;
        BasicBlock *body;
        auto *k = make_kernel_with_body(m, body);
        XIRBuilder b;
        b.set_insertion_point(body);
        auto *a = b.alloca_local(Type::of<bool>());
        b.coro_register(a, "cond");
        auto *cond = b.load(Type::of<bool>(), a);
        auto *if_inst = b.if_(cond);
        auto *true_block = if_inst->create_true_block();
        auto *false_block = if_inst->create_false_block();
        auto *merge_block = if_inst->create_merge_block();
        b.set_insertion_point(true_block);
        b.coro_suspend(1u);
        b.br(merge_block);
        b.set_insertion_point(false_block);
        b.br(merge_block);
        b.set_insertion_point(merge_block);
        b.return_void();

        auto result = coro_materialize_run_on_function(k);
        expect(result.ok);
        expect(result.split_info.is_supported);
        // Entry + continuation for token 1 = at least 2 scopes
        expect(result.split_info.continuations.size() >= 2_u);
        for (auto &cont : result.split_info.continuations) {
            expect(cont.callable != nullptr);
        }
    };
}

int main() {
    reg_coro_materialize();
    return 0;
}
