#include "ut/ut.hpp"
#include <luisa/ast/type_registry.h>
#include <luisa/xir/basic_block.h>
#include <luisa/xir/builder.h>
#include <luisa/xir/function.h>
#include <luisa/xir/instructions/alloca.h>
#include <luisa/xir/instructions/branch.h>
#include <luisa/xir/instructions/coroutine.h>
#include <luisa/xir/instructions/loop.h>
#include <luisa/xir/instructions/if.h>
#include <luisa/xir/module.h>
#include <luisa/xir/passes/coroutine.h>
#include <luisa/xir/passes/coroutine_split.h>
#include <luisa/xir/translators/xir2ast.h>

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

void reg_coroutine_split() {

    "coroutine_split_ignores_plain_kernel"_test = [] {
        Module m;
        BasicBlock *body;
        auto *k = make_kernel_with_body(m, body);
        XIRBuilder b;
        b.set_insertion_point(body);
        b.return_void();
        auto info = coroutine_split_run_on_function(k);
        expect(!info.is_supported);
        expect(info.continuations.empty());
        expect(info.frame_slots.empty());
        // No diagnostics when there are no markers — just nothing to do.
    };

    "coroutine_split_produces_plan_for_flat_coroutine"_test = [] {
        Module m;
        BasicBlock *body;
        auto *k = make_kernel_with_body(m, body);
        XIRBuilder b;
        b.set_insertion_point(body);
        auto *a = b.alloca_local(Type::of<float>());
        auto *c = b.alloca_local(Type::of<int>());
        b.coro_register(a, "a");
        b.coro_register(c, "c");
        b.coro_suspend(1u);
        b.coro_suspend(2u);
        b.load(Type::of<float>(), a);
        b.load(Type::of<int>(), c);
        b.return_void();

        auto info = coroutine_split_run_on_function(k);
        expect(info.is_supported);
        expect(info.changed);
        expect(info.frame_type != nullptr);
        // Slot 0 is target_token; the two allocas occupy slots 1 and 2.
        expect(info.frame_slots.size() == 2_u);
        // 2 suspends produce 3 continuations: entry + two resumed slices.
        expect(info.continuations.size() == 3_u);
        for (auto &&cont : info.continuations) {
            expect(cont.callable != nullptr);
            auto args_begin = cont.callable->arguments().begin();
            auto args_end = cont.callable->arguments().end();
            expect(args_begin != args_end);
            // ABI: each continuation takes the frame as a reference arg first.
            expect((*args_begin)->is_reference());
            // Each continuation must round-trip through xir_to_ast_translate
            // so the DSL-side wrapper layer can use it as a Callable.
            auto ast = xir_to_ast_translate(*cont.callable->definition(), {});
            expect(ast != nullptr);
        }
    };

    "coroutine_split_rejects_loop_containing_suspends"_test = [] {
        Module m;
        BasicBlock *body;
        auto *k = make_kernel_with_body(m, body);
        XIRBuilder b;
        b.set_insertion_point(body);
        // Build a structured loop in the body. Even if no suspend lives
        // inside the loop, the presence of a structured-CFG container in the
        // body block disqualifies the input from the flat coroutine subset
        // until condition replay lands. We add a suspend AFTER the loop so
        // is_coroutine=true but is_supported=false.
        auto *loop = b.loop();
        auto *prepare = loop->create_prepare_block();
        auto *loop_body = loop->create_body_block();
        auto *update = loop->create_update_block();
        auto *merge = loop->create_merge_block();
        b.set_insertion_point(prepare);
        b.br(merge);
        b.set_insertion_point(loop_body);
        b.br(update);
        b.set_insertion_point(update);
        b.br(prepare);
        b.set_insertion_point(merge);
        b.coro_suspend(1u);
        b.return_void();

        auto info = coroutine_split_run_on_function(k);
        expect(!info.is_supported);
        // The pass must surface a diagnostic that names the unsupported case
        // so callers can fall back to coroutine_lower.
        auto found_diag = false;
        for (auto &&d : info.diagnostics) {
            if (d.find("structured control-flow") != luisa::string::npos ||
                d.find("outside the entry body block") != luisa::string::npos) {
                found_diag = true;
            }
        }
        expect(found_diag);
    };
}

int main() {
    reg_coroutine_split();
    return 0;
}