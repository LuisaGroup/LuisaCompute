#include "ut/ut.hpp"
#include <luisa/ast/type_registry.h>
#include <luisa/xir/basic_block.h>
#include <luisa/xir/builder.h>
#include <luisa/xir/function.h>
#include <luisa/xir/instructions/branch.h>
#include <luisa/xir/instructions/return.h>
#include <luisa/xir/instructions/switch.h>
#include <luisa/xir/module.h>
#include <luisa/xir/passes/simplify_cfg.h>

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

[[nodiscard]] size_t count_blocks(FunctionDefinition *def) noexcept {
    size_t n = 0u;
    def->traverse_basic_blocks([&](BasicBlock *) noexcept { ++n; });
    return n;
}

[[nodiscard]] size_t count_isa_branch(FunctionDefinition *def) noexcept {
    size_t n = 0u;
    def->traverse_basic_blocks([&](BasicBlock *bb) noexcept {
        if (bb->is_terminated() && bb->terminator()->isa<BranchInst>()) ++n;
    });
    return n;
}

[[nodiscard]] size_t count_isa_cond_branch(FunctionDefinition *def) noexcept {
    size_t n = 0u;
    def->traverse_basic_blocks([&](BasicBlock *bb) noexcept {
        if (bb->is_terminated() && bb->terminator()->isa<ConditionalBranchInst>()) ++n;
    });
    return n;
}

}// namespace

void reg_simplify_cfg() {

    "simplify_empty_function"_test = [] {
        Module m;
        BasicBlock *body;
        auto *k = make_kernel_with_body(m, body);
        XIRBuilder b;
        b.set_insertion_point(body);
        b.return_void();
        auto info = simplify_cfg_pass_run_on_function(k);
        expect(info.folded_constant_cond_br_count == 0u);
        expect(info.threaded_empty_block_count == 0u);
        expect(info.removed_unreachable_block_count == 0u);
    };

    "fold_constant_true_cond_br"_test = [] {
        Module m;
        BasicBlock *body;
        auto *k = make_kernel_with_body(m, body);
        auto *def = k->definition();
        auto *t = def->create_basic_block();
        auto *f = def->create_basic_block();
        XIRBuilder b;
        b.set_insertion_point(body);
        auto *cond = m.create_constant_one(Type::of<bool>());
        b.cond_br(cond, t, f);
        b.set_insertion_point(t);
        b.return_void();
        b.set_insertion_point(f);
        b.return_void();
        auto info = simplify_cfg_pass_run_on_function(k);
        expect(info.folded_constant_cond_br_count == 1u);
        expect(count_isa_cond_branch(def) == 0u);
        expect(count_isa_branch(def) == 1u);
        expect(info.removed_unreachable_block_count == 1u);
    };

    "fold_constant_false_cond_br"_test = [] {
        Module m;
        BasicBlock *body;
        auto *k = make_kernel_with_body(m, body);
        auto *def = k->definition();
        auto *t = def->create_basic_block();
        auto *f = def->create_basic_block();
        XIRBuilder b;
        b.set_insertion_point(body);
        auto *cond = m.create_constant_zero(Type::of<bool>());
        b.cond_br(cond, t, f);
        b.set_insertion_point(t);
        b.return_void();
        b.set_insertion_point(f);
        b.return_void();
        auto info = simplify_cfg_pass_run_on_function(k);
        expect(info.folded_constant_cond_br_count == 1u);
        expect(info.removed_unreachable_block_count == 1u);
        expect(count_isa_cond_branch(def) == 0u);
    };

    "thread_empty_block"_test = [] {
        Module m;
        BasicBlock *body;
        auto *k = make_kernel_with_body(m, body);
        auto *def = k->definition();
        auto *mid = def->create_basic_block();
        auto *tail = def->create_basic_block();
        XIRBuilder b;
        b.set_insertion_point(body);
        b.br(mid);
        b.set_insertion_point(mid);
        b.br(tail);
        b.set_insertion_point(tail);
        b.return_void();
        auto info = simplify_cfg_pass_run_on_function(k);
        expect(info.threaded_empty_block_count >= 1u);
        expect(count_blocks(def) == 2u);
    };

    "remove_unreachable_block"_test = [] {
        Module m;
        BasicBlock *body;
        auto *k = make_kernel_with_body(m, body);
        auto *def = k->definition();
        auto *dead = def->create_basic_block();
        XIRBuilder b;
        b.set_insertion_point(body);
        b.return_void();
        b.set_insertion_point(dead);
        b.return_void();
        auto info = simplify_cfg_pass_run_on_function(k);
        expect(info.removed_unreachable_block_count == 1u);
        expect(count_blocks(def) == 1u);
    };

    "switch_default_redirect_via_thread"_test = [] {
        Module m;
        BasicBlock *body;
        auto *k = make_kernel_with_body(m, body);
        auto *def = k->definition();
        XIRBuilder b;
        b.set_insertion_point(body);
        auto *val = m.create_constant_zero(Type::of<int>());
        auto *sw = b.switch_(val);
        auto *def_bb = sw->create_default_block();
        auto *case_bb = sw->create_case_block(0);
        auto *real_default = def->create_basic_block();
        auto *real_case = def->create_basic_block();
        b.set_insertion_point(def_bb);
        b.br(real_default);
        b.set_insertion_point(case_bb);
        b.br(real_case);
        b.set_insertion_point(real_default);
        b.return_void();
        b.set_insertion_point(real_case);
        b.return_void();
        auto info = simplify_cfg_pass_run_on_function(k);
        expect(info.threaded_empty_block_count >= 2u);
        expect(sw->default_block() == real_default);
        expect(sw->case_block(0) == real_case);
    };

    "idempotent_on_simplified"_test = [] {
        Module m;
        BasicBlock *body;
        auto *k = make_kernel_with_body(m, body);
        XIRBuilder b;
        b.set_insertion_point(body);
        b.return_void();
        auto info1 = simplify_cfg_pass_run_on_function(k);
        auto info2 = simplify_cfg_pass_run_on_function(k);
        expect(info1.folded_constant_cond_br_count == 0u);
        expect(info2.folded_constant_cond_br_count == 0u);
        expect(info2.threaded_empty_block_count == 0u);
        expect(info2.removed_unreachable_block_count == 0u);
    };

    "module_entry_point"_test = [] {
        Module m;
        BasicBlock *body;
        auto *k = make_kernel_with_body(m, body);
        auto *def = k->definition();
        auto *t = def->create_basic_block();
        auto *f = def->create_basic_block();
        XIRBuilder b;
        b.set_insertion_point(body);
        auto *cond = m.create_constant_one(Type::of<bool>());
        b.cond_br(cond, t, f);
        b.set_insertion_point(t);
        b.return_void();
        b.set_insertion_point(f);
        b.return_void();
        auto info = simplify_cfg_pass_run_on_module(&m);
        expect(info.folded_constant_cond_br_count == 1u);
    };
}

int main(int argc, char *argv[]) {
    boost::ut::detail::cfg::parse_arg_with_fallback(argc, const_cast<const char **>(argv));
    reg_simplify_cfg();
    return 0;
}
