#include "ut/ut.hpp"
#include <luisa/ast/type_registry.h>
#include <luisa/xir/basic_block.h>
#include <luisa/xir/builder.h>
#include <luisa/xir/function.h>
#include <luisa/xir/instructions/alloca.h>
#include <luisa/xir/instructions/branch.h>
#include <luisa/xir/instructions/coroutine.h>
#include <luisa/xir/module.h>
#include <luisa/xir/passes/coroutine.h>

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

void reg_coroutine_analysis() {

    "coroutine_analysis_ignores_plain_kernel"_test = [] {
        Module m;
        BasicBlock *body;
        auto *k = make_kernel_with_body(m, body);
        XIRBuilder b;
        b.set_insertion_point(body);
        b.return_void();
        auto info = coroutine_analysis_run_on_function(k);
        expect(!info.is_coroutine);
        expect(info.registers.empty());
        expect(info.suspends.empty());
        expect(info.continuations.empty());
        expect(info.transitions.empty());
        expect(info.diagnostics.empty());
    };

    "coroutine_analysis_collects_markers_and_transition"_test = [] {
        Module m;
        BasicBlock *body;
        auto *k = make_kernel_with_body(m, body);
        auto *tail = k->definition()->create_basic_block();
        XIRBuilder b;
        b.set_insertion_point(body);
        auto *local = b.alloca_local(Type::of<float>());
        b.coro_register(local, "local");
        b.coro_suspend(1u);
        b.br(tail);
        b.set_insertion_point(tail);
        b.load(Type::of<float>(), local);
        b.return_void();

        auto info = coroutine_analysis_run_on_function(k);
        expect(info.is_coroutine);
        expect(info.registers.size() == 1u);
        expect(info.suspends.size() == 1u);
        expect(info.continuations.size() == 2u);
        expect(info.transitions.size() == 1u);
        expect(!info.transitions.front().exits);
        expect(info.frame_candidates.size() == 1u);
        expect(info.frame_candidates.front().alloca == local);
        expect(info.frame_candidates.front().live_across_suspend_ids.size() == 1u);
        expect(info.diagnostics.empty());
    };

    "coroutine_analysis_continues_after_same_block_suspend"_test = [] {
        Module m;
        BasicBlock *body;
        auto *k = make_kernel_with_body(m, body);
        XIRBuilder b;
        b.set_insertion_point(body);
        auto *local = b.alloca_local(Type::of<float>());
        b.coro_suspend(1u);
        b.load(Type::of<float>(), local);
        b.return_void();

        auto info = coroutine_analysis_run_on_function(k);
        expect(info.is_coroutine);
        expect(info.suspends.size() == 1u);
        expect(info.continuations.size() == 2u);
        expect(info.transitions.size() == 1u);
        expect(!info.transitions.front().exits);
        expect(info.transitions.front().to_continuation == 1u);
        expect(info.continuations[1u].entry_block == body);
        expect(info.continuations[1u].entry_inst != nullptr);
        expect(info.frame_candidates.size() == 1u);
        expect(info.frame_candidates.front().alloca == local);
        expect(info.diagnostics.empty());
    };

    "coroutine_module_analysis_remaps_ids"_test = [] {
        Module m;
        BasicBlock *body_a;
        auto *k_a = make_kernel_with_body(m, body_a);
        BasicBlock *body_b;
        auto *k_b = make_kernel_with_body(m, body_b);
        XIRBuilder b;
        b.set_insertion_point(body_a);
        b.coro_suspend(1u);
        b.return_void();
        b.set_insertion_point(body_b);
        b.coro_suspend(1u);
        b.return_void();
        static_cast<void>(k_a);
        static_cast<void>(k_b);

        auto info = coroutine_analysis_run_on_module(&m);
        expect(info.is_coroutine);
        expect(info.suspends.size() == 2u);
        expect(info.continuations.size() == 2u);
        expect(info.transitions.size() == 2u);
        expect(info.suspends[0u].id == 0u);
        expect(info.suspends[1u].id == 1u);
        expect(info.continuations[0u].id == 0u);
        expect(info.continuations[1u].id == 1u);
        expect(info.transitions[0u].suspend_id == 0u);
        expect(info.transitions[1u].suspend_id == 1u);
        expect(info.diagnostics.empty());
    };
}

int main(int argc, char *argv[]) {
    boost::ut::detail::cfg::parse_arg_with_fallback(argc, const_cast<const char **>(argv));
    reg_coroutine_analysis();
    return 0;
}
