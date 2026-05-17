#include "ut/ut.hpp"
#include <luisa/ast/type_registry.h>
#include <luisa/xir/basic_block.h>
#include <luisa/xir/builder.h>
#include <luisa/xir/function.h>
#include <luisa/xir/instructions/branch.h>
#include <luisa/xir/instructions/if.h>
#include <luisa/xir/instructions/loop.h>
#include <luisa/xir/instructions/return.h>
#include <luisa/xir/module.h>
#include <luisa/xir/passes/destructure_cfg.h>
#include <luisa/xir/passes/restructure_cfg.h>

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

[[nodiscard]] size_t count_terminator_kind(FunctionDefinition *def,
                                           DerivedInstructionTag tag) noexcept {
    size_t n = 0u;
    def->traverse_basic_blocks([&](BasicBlock *bb) noexcept {
        if (!bb->is_terminated()) { return; }
        if (bb->terminator()->derived_instruction_tag() == tag) { ++n; }
    });
    return n;
}

}// namespace

void reg_restructure_cfg() {

    "restructure_empty_function_noop"_test = [] {
        Module m;
        BasicBlock *body;
        auto *k = make_kernel_with_body(m, body);
        XIRBuilder b;
        b.set_insertion_point(body);
        b.return_void();
        auto info = restructure_cfg_pass_run_on_function(k);
        expect(info.restructured_if_count == 0u);
        expect(info.restructured_loop_count == 0u);
        expect(info.irreducible_region_count == 0u);
    };

    "restructure_external_function_skipped"_test = [] {
        Module m;
        auto *ext = m.create_external_function(Type::of<void>());
        auto info = restructure_cfg_pass_run_on_function(ext);
        expect(info.restructured_if_count == 0u);
        expect(info.restructured_loop_count == 0u);
    };

    "restructure_if_from_destructured"_test = [] {
        Module m;
        BasicBlock *body;
        auto *k = make_kernel_with_body(m, body);
        XIRBuilder b;
        b.set_insertion_point(body);
        auto *cond = m.create_constant_one(Type::of<bool>());
        auto *if_inst = b.if_(cond);
        auto *t = if_inst->create_true_block();
        auto *f = if_inst->create_false_block();
        auto *merge = if_inst->create_merge_block();
        b.set_insertion_point(t);
        b.br(merge);
        b.set_insertion_point(f);
        b.br(merge);
        b.set_insertion_point(merge);
        b.return_void();
        (void)destructure_cfg_pass_run_on_function(k);
        auto *def = k->definition();
        expect(count_terminator_kind(def, DerivedInstructionTag::IF) == 0u);
        auto info = restructure_cfg_pass_run_on_function(k);
        expect(info.restructured_if_count == 1u);
        expect(count_terminator_kind(def, DerivedInstructionTag::IF) == 1u);
        expect(count_terminator_kind(def, DerivedInstructionTag::CONDITIONAL_BRANCH) == 0u);
    };

    "restructure_simple_loop_from_destructured"_test = [] {
        Module m;
        BasicBlock *body;
        auto *k = make_kernel_with_body(m, body);
        XIRBuilder b;
        b.set_insertion_point(body);
        auto *sl = b.simple_loop();
        auto *lbody = sl->create_body_block();
        auto *merge = sl->create_merge_block();
        auto *cond = m.create_constant_zero(Type::of<bool>());
        auto *cont = k->definition()->create_basic_block();
        b.set_insertion_point(lbody);
        b.cond_br(cond, merge, cont);
        b.set_insertion_point(cont);
        b.continue_(lbody);
        b.set_insertion_point(merge);
        b.return_void();
        (void)destructure_cfg_pass_run_on_function(k);
        auto *def = k->definition();
        expect(count_terminator_kind(def, DerivedInstructionTag::SIMPLE_LOOP) == 0u);
        auto info = restructure_cfg_pass_run_on_function(k);
        expect(info.restructured_loop_count == 1u);
        expect(count_terminator_kind(def, DerivedInstructionTag::SIMPLE_LOOP) == 1u);
    };

    "restructure_module_runs_all_functions"_test = [] {
        Module m;
        constexpr size_t kFns = 3u;
        for (size_t i = 0; i < kFns; ++i) {
            BasicBlock *body;
            auto *k = make_kernel_with_body(m, body);
            XIRBuilder b;
            b.set_insertion_point(body);
            auto *cond = m.create_constant_one(Type::of<bool>());
            auto *if_inst = b.if_(cond);
            auto *t = if_inst->create_true_block();
            auto *f = if_inst->create_false_block();
            auto *merge = if_inst->create_merge_block();
            b.set_insertion_point(t);
            b.br(merge);
            b.set_insertion_point(f);
            b.br(merge);
            b.set_insertion_point(merge);
            b.return_void();
        }
        (void)destructure_cfg_pass_run_on_module(&m);
        auto info = restructure_cfg_pass_run_on_module(&m);
        expect(info.restructured_if_count == kFns);
        for (auto *f : m.function_list()) {
            auto *def = f->definition();
            if (def == nullptr) { continue; }
            expect(count_terminator_kind(def, DerivedInstructionTag::IF) == 1u);
            expect(count_terminator_kind(def, DerivedInstructionTag::CONDITIONAL_BRANCH) == 0u);
        }
    };

    "restructure_idempotent_if"_test = [] {
        Module m;
        BasicBlock *body;
        auto *k = make_kernel_with_body(m, body);
        XIRBuilder b;
        b.set_insertion_point(body);
        auto *cond = m.create_constant_one(Type::of<bool>());
        auto *if_inst = b.if_(cond);
        auto *t = if_inst->create_true_block();
        auto *f = if_inst->create_false_block();
        auto *merge = if_inst->create_merge_block();
        b.set_insertion_point(t);
        b.br(merge);
        b.set_insertion_point(f);
        b.br(merge);
        b.set_insertion_point(merge);
        b.return_void();
        (void)destructure_cfg_pass_run_on_function(k);
        auto first = restructure_cfg_pass_run_on_function(k);
        auto second = restructure_cfg_pass_run_on_function(k);
        expect(first.restructured_if_count == 1u);
        expect(second.restructured_if_count == 0u);
    };

    "restructure_empty_module_noop"_test = [] {
        Module m;
        auto info = restructure_cfg_pass_run_on_module(&m);
        expect(info.restructured_if_count == 0u);
        expect(info.restructured_loop_count == 0u);
    };

    "restructure_if_preserves_true_false_blocks"_test = [] {
        Module m;
        BasicBlock *body;
        auto *k = make_kernel_with_body(m, body);
        XIRBuilder b;
        b.set_insertion_point(body);
        auto *cond = m.create_constant_one(Type::of<bool>());
        auto *if_inst = b.if_(cond);
        auto *t = if_inst->create_true_block();
        auto *f = if_inst->create_false_block();
        auto *merge = if_inst->create_merge_block();
        b.set_insertion_point(t);
        b.br(merge);
        b.set_insertion_point(f);
        b.br(merge);
        b.set_insertion_point(merge);
        b.return_void();
        (void)destructure_cfg_pass_run_on_function(k);
        (void)restructure_cfg_pass_run_on_function(k);
        auto *def = k->definition();
        auto *new_term = def->body_block()->terminator();
        expect(new_term != nullptr);
        expect(new_term->isa<IfInst>());
        auto *rebuilt = static_cast<IfInst *>(new_term);
        expect(rebuilt->true_block() == t);
        expect(rebuilt->false_block() == f);
        expect(rebuilt->merge_block() == merge);
    };

    "restructure_nested_if_from_destructured"_test = [] {
        Module m;
        BasicBlock *body;
        auto *k = make_kernel_with_body(m, body);
        XIRBuilder b;
        b.set_insertion_point(body);
        auto *cond = m.create_constant_one(Type::of<bool>());
        auto *outer = b.if_(cond);
        auto *ot = outer->create_true_block();
        auto *of = outer->create_false_block();
        auto *omerge = outer->create_merge_block();
        b.set_insertion_point(ot);
        auto *inner = b.if_(cond);
        auto *it = inner->create_true_block();
        auto *if_ = inner->create_false_block();
        auto *imerge = inner->create_merge_block();
        b.set_insertion_point(it);
        b.br(imerge);
        b.set_insertion_point(if_);
        b.br(imerge);
        b.set_insertion_point(imerge);
        b.br(omerge);
        b.set_insertion_point(of);
        b.br(omerge);
        b.set_insertion_point(omerge);
        b.return_void();
        (void)destructure_cfg_pass_run_on_function(k);
        auto *def = k->definition();
        expect(count_terminator_kind(def, DerivedInstructionTag::IF) == 0u);
        auto info = restructure_cfg_pass_run_on_function(k);
        expect(info.restructured_if_count == 2u);
        expect(count_terminator_kind(def, DerivedInstructionTag::IF) == 2u);
        expect(count_terminator_kind(def, DerivedInstructionTag::CONDITIONAL_BRANCH) == 0u);
    };
}

int main(int argc, char *argv[]) {
    boost::ut::detail::cfg::parse_arg_with_fallback(argc, const_cast<const char **>(argv));
    reg_restructure_cfg();
    return 0;
}
