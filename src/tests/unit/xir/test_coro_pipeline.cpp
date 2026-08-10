#include "ut/ut.hpp"
#include <luisa/xir/basic_block.h>
#include <luisa/xir/builder.h>
#include <luisa/xir/function.h>
#include <luisa/xir/instructions/branch.h>
#include <luisa/xir/instructions/coro.h>
#include <luisa/xir/instructions/return.h>
#include <luisa/xir/instructions/switch.h>
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

[[nodiscard]] size_t count_tag_in_owned_blocks(Module &m, DerivedInstructionTag tag) noexcept {
    size_t count = 0u;
    for (auto *function : m.function_list()) {
        if (auto *definition = function->definition()) {
            for (auto *block : definition->basic_blocks()) {
                for (auto *inst : block->instructions()) {
                    count += inst->derived_instruction_tag() == tag ? 1u : 0u;
                }
            }
        }
    }
    return count;
}

}// namespace

void reg_coro_pipeline() {

    "coro_pipeline_does_not_crash"_test = [] {
        Module m;
        BasicBlock *body;
        auto *k = make_kernel_with_body(m, body);
        XIRBuilder b;
        // Keep both suspend and resume roots executable through phase-A DCE.
        auto *cond = k->create_value_argument(Type::of<bool>());

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
        expect(count_callables(m) >= 2u);
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

    "structured_switch_coroutine_is_explicitly_normalized"_test = [] {
        Module m;
        BasicBlock *body;
        auto *kernel = make_kernel_with_body(m, body);
        XIRBuilder b;
        b.set_insertion_point(body);
        // Keep both switch arms executable through phase-A DCE so the test
        // actually reaches coro split/materialize. The raw coroutine interval
        // uses IndexedBranchInst, and the final structured boundary must
        // reconstruct a native SwitchInst rather than binary-lowering it.
        auto *selector = kernel->create_value_argument(Type::of<int>());
        auto *sw = b.switch_(selector);
        auto *suspend_block = sw->create_case_block(0);
        auto *default_block = sw->create_default_block();
        auto *merge_block = sw->create_merge_block();
        auto *resume_block = kernel->create_basic_block();
        b.set_insertion_point(suspend_block);
        b.coro_suspend(51u, "switch", nullptr);
        b.set_insertion_point(default_block);
        b.br(resume_block);
        b.set_insertion_point(resume_block);
        b.coro_resume(51u, nullptr);
        b.br(merge_block);
        b.set_insertion_point(merge_block);
        b.return_void();

        expect(count_tag_in_owned_blocks(m, DerivedInstructionTag::SWITCH) == 1u);
        xir_to_ast_normalize_module(&m);

        expect(count_tag_in_owned_blocks(
                   m, DerivedInstructionTag::INDEXED_BRANCH) == 0u);
        expect(count_tag_in_owned_blocks(
                   m, DerivedInstructionTag::SWITCH) >= 1u);
        expect(count_callables(m) >= 2u);
        size_t checked_continuations = 0u;
        for (auto *function : m.function_list()) {
            if (!function->isa<CallableFunction>() || function->definition() == nullptr) { continue; }
            ++checked_continuations;
            for (auto *block : function->definition()->basic_blocks()) {
                expect(block->is_terminated());
                for (auto *inst : block->instructions()) {
                    auto tag = inst->derived_instruction_tag();
                    expect(tag != DerivedInstructionTag::CORO_SUSPEND);
                    expect(tag != DerivedInstructionTag::CORO_RESUME);
                    expect(tag != DerivedInstructionTag::CORO_TERMINATE);
                }
            }
        }
        expect(checked_continuations >= 2u);
        expect(all_blocks_terminated(m));
    };

    "generic_pipeline_projects_only_live_sparse_suspend_tokens"_test = [] {
        Module m;
        BasicBlock *entry;
        auto *kernel = make_kernel_with_body(m, entry);
        auto *dead_suspend = kernel->create_basic_block();
        auto *dead_resume = kernel->create_basic_block();
        auto *live_suspend = kernel->create_basic_block();
        auto *live_resume = kernel->create_basic_block();
        constexpr uint32_t dead_token = 17u;
        constexpr uint32_t live_token = 91u;
        XIRBuilder b;

        b.set_insertion_point(entry);
        b.cond_br(
            m.create_constant_zero(Type::of<bool>()),
            dead_suspend, live_suspend);
        b.set_insertion_point(dead_suspend);
        b.coro_suspend(dead_token, "dead", nullptr);
        b.set_insertion_point(dead_resume);
        b.coro_resume(dead_token, nullptr);
        b.br(live_suspend);
        b.set_insertion_point(live_suspend);
        b.coro_suspend(live_token, "live", nullptr);
        b.set_insertion_point(live_resume);
        b.coro_resume(live_token, nullptr);
        b.coro_terminate();

        xir_to_ast_normalize_module(&m);

        expect(count_callables(m) == 2u)
            << "entry plus the live sparse-token continuation must be materialized; the dead token must not be";
        for (auto *function : m.function_list()) {
            if (!function->isa<CallableFunction>() ||
                function->definition() == nullptr) {
                continue;
            }
            for (auto *block : function->definition()->basic_blocks()) {
                expect(block->is_terminated());
                for (auto *inst : block->instructions()) {
                    auto tag = inst->derived_instruction_tag();
                    expect(tag != DerivedInstructionTag::CORO_SUSPEND);
                    expect(tag != DerivedInstructionTag::CORO_RESUME);
                    expect(tag != DerivedInstructionTag::CORO_TERMINATE);
                }
            }
        }
    };

    "generic_pipeline_lowers_empty_live_token_set_to_entry_callable"_test = [] {
        Module m;
        BasicBlock *entry;
        auto *kernel = make_kernel_with_body(m, entry);
        auto *dead_suspend = kernel->create_basic_block();
        auto *dead_resume = kernel->create_basic_block();
        auto *terminal = kernel->create_basic_block();
        constexpr uint32_t dead_token = 37u;
        XIRBuilder b;

        b.set_insertion_point(entry);
        b.cond_br(
            m.create_constant_zero(Type::of<bool>()),
            dead_suspend, terminal);
        b.set_insertion_point(dead_suspend);
        b.coro_suspend(dead_token, "dead-only", nullptr);
        b.set_insertion_point(dead_resume);
        b.coro_resume(dead_token, nullptr);
        b.br(terminal);
        b.set_insertion_point(terminal);
        b.coro_terminate();

        xir_to_ast_normalize_module(&m);

        expect(count_callables(m) == 1u)
            << "T_live = empty is a one-scope coroutine, not a failed split";
        expect(all_blocks_terminated(m));
        size_t continuation_count = 0u;
        for (auto *function : m.function_list()) {
            if (!function->isa<CallableFunction>() ||
                function->definition() == nullptr) {
                continue;
            }
            ++continuation_count;
            for (auto *block : function->definition()->basic_blocks()) {
                for (auto *inst : block->instructions()) {
                    auto tag = inst->derived_instruction_tag();
                    expect(tag != DerivedInstructionTag::CORO_SUSPEND);
                    expect(tag != DerivedInstructionTag::CORO_RESUME);
                    expect(tag != DerivedInstructionTag::CORO_TERMINATE);
                }
            }
        }
        expect(continuation_count == 1u);
    };
}

int main(int argc, char *argv[]) {
    (void)argc;
    (void)argv;
    reg_coro_pipeline();
    return 0;
}
