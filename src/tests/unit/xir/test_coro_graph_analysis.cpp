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
#include <luisa/xir/passes/coro_graph_analysis.h>

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

void reg_coro_graph_analysis() {

    "preliminary_graph_flat_coroutine"_test = [] {
        Module m;
        BasicBlock *body;
        auto *k = make_kernel_with_body(m, body);
        XIRBuilder b;
        b.set_insertion_point(body);
        b.alloca_local(Type::of<float>());
        b.coro_suspend(1u);
        b.coro_suspend(2u);
        b.return_void();

        auto pg = coro_preliminary_graph_build(k);
        expect(pg.entry_scope.valid());
        expect(pg.diagnostics.empty());
        // Should have instructions for: alloca, suspend(1), suspend(2), return, entry, entry_scope
        expect(pg.instructions.size() >= 6_u);
        // Two suspends should be terminators
        size_t suspend_count = 0;
        for (auto &instr : pg.instructions) {
            if (instr.tag == CoroInstruction::Tag::SUSPEND) { ++suspend_count; }
        }
        expect(suspend_count == 2_u);
    };

    "preliminary_graph_loop_with_suspend"_test = [] {
        Module m;
        BasicBlock *body;
        auto *k = make_kernel_with_body(m, body);
        XIRBuilder b;
        b.set_insertion_point(body);
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

        auto pg = coro_preliminary_graph_build(k);
        expect(pg.entry_scope.valid());
        // Should find the loop node and the suspend inside it
        bool found_loop = false;
        bool found_suspend = false;
        for (auto &instr : pg.instructions) {
            if (instr.tag == CoroInstruction::Tag::LOOP) found_loop = true;
            if (instr.tag == CoroInstruction::Tag::SUSPEND) found_suspend = true;
        }
        expect(found_loop);
        expect(found_suspend);
    };

    "graph_split_flat_produces_scopes"_test = [] {
        Module m;
        BasicBlock *body;
        auto *k = make_kernel_with_body(m, body);
        XIRBuilder b;
        b.set_insertion_point(body);
        b.coro_suspend(1u);
        b.coro_suspend(2u);
        b.return_void();

        auto info = coro_graph_run_on_function(k);
        expect(info.ok);
        // Entry scope + 2 continuation scopes = 3 scopes
        expect(info.scopes.size() == 3_u);
        // Token 1 and 2 should map to scopes
        expect(info.token_to_scope.contains(1u));
        expect(info.token_to_scope.contains(2u));
    };

    "graph_split_loop_with_suspend_produces_first_flag"_test = [] {
        Module m;
        BasicBlock *body;
        auto *k = make_kernel_with_body(m, body);
        XIRBuilder b;
        b.set_insertion_point(body);
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

        auto info = coro_graph_run_on_function(k);
        expect(info.ok);
        // Should have at least 2 scopes (entry + continuation for token 1)
        expect(info.scopes.size() >= 2_u);
        expect(info.token_to_scope.contains(1u));
        // The continuation scope should contain a MAKE_FIRST_FLAG node
        auto cont_scope = info.token_to_scope.at(1u);
        bool has_first_flag = false;
        for (auto ref : info.scopes[cont_scope.index].instructions) {
            if (info.preliminary.instructions[ref.index].tag == CoroInstruction::Tag::MAKE_FIRST_FLAG) {
                has_first_flag = true;
            }
        }
        expect(has_first_flag);
    };
}

int main() {
    reg_coro_graph_analysis();
    return 0;
}
