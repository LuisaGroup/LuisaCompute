#include "ut/ut.hpp"
#include <luisa/xir/basic_block.h>
#include <luisa/xir/builder.h>
#include <luisa/xir/function.h>
#include <luisa/xir/instructions/branch.h>
#include <luisa/xir/instructions/coro.h>
#include <luisa/xir/instructions/return.h>
#include <luisa/xir/module.h>
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

[[nodiscard]] size_t count_callables(Module &m) noexcept {
    size_t n = 0u;
    for (auto *f : m.function_list()) {
        if (f->isa<CallableFunction>()) { n++; }
    }
    return n;
}

[[nodiscard]] bool all_blocks_terminated(Module &m) noexcept {
    for (auto *f : m.function_list()) {
        if (auto *def = f->definition()) {
            bool all_ok = true;
            def->traverse_basic_blocks([&](BasicBlock *bb) noexcept {
                if (!bb->is_terminated()) { all_ok = false; }
            });
            if (!all_ok) { return false; }
        }
    }
    return true;
}

}// namespace

void reg_coro_pipeline() {

    "coro_pipeline_does_not_crash"_test = [] {
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

        xir_to_ast_normalize_module(&m);

        expect(all_blocks_terminated(m));
    };

    "non_coroutine_module_unchanged"_test = [] {
        Module m;
        BasicBlock *body;
        auto *k = make_kernel_with_body(m, body);
        XIRBuilder b;
        b.set_insertion_point(body);
        b.return_void();

        xir_to_ast_normalize_module(&m);

        expect(all_blocks_terminated(m));
        expect(count_callables(m) == 0u);
    };

    "non_coroutine_with_control_flow"_test = [] {
        Module m;
        BasicBlock *body;
        auto *k = make_kernel_with_body(m, body);
        XIRBuilder b;

        auto *alt = k->create_basic_block();
        b.set_insertion_point(body);
        b.br(alt);

        b.set_insertion_point(alt);
        b.return_void();

        xir_to_ast_normalize_module(&m);

        expect(all_blocks_terminated(m));
    };
}

int main(int argc, char *argv[]) {
    (void)argc;
    (void)argv;
    reg_coro_pipeline();
    return 0;
}
