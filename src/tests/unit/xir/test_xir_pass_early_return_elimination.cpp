#include "ut/ut.hpp"
#include <luisa/ast/type_registry.h>
#include <luisa/xir/basic_block.h>
#include <luisa/xir/builder.h>
#include <luisa/xir/function.h>
#include <luisa/xir/instructions/alloca.h>
#include <luisa/xir/instructions/if.h>
#include <luisa/xir/instructions/return.h>
#include <luisa/xir/module.h>
#include <luisa/xir/passes/early_return_elimination.h>

using namespace luisa;
using namespace luisa::compute;
using namespace luisa::compute::xir;
using namespace boost::ut;
using namespace boost::ut::literals;

namespace {

[[nodiscard]] size_t count_terminator_kind(FunctionDefinition *def, DerivedInstructionTag tag) noexcept {
    size_t n = 0u;
    def->traverse_basic_blocks([&](BasicBlock *bb) noexcept {
        if (!bb->is_terminated()) { return; }
        if (bb->terminator()->derived_instruction_tag() == tag) { ++n; }
    });
    return n;
}

[[nodiscard]] size_t count_alloca(FunctionDefinition *def) noexcept {
    size_t n = 0u;
    def->traverse_basic_blocks([&](BasicBlock *bb) noexcept {
        bb->traverse_instructions([&](Instruction *inst) noexcept {
            if (inst->isa<AllocaInst>()) { ++n; }
        });
    });
    return n;
}

}// namespace

void reg_early_return_elimination() {

    "no_early_return_void"_test = [] {
        Module m;
        auto *k = m.create_kernel();
        auto *body = k->create_body_block();
        XIRBuilder b;
        b.set_insertion_point(body);
        b.return_void();
        auto info = early_return_elimination_pass_run_on_function(k);
        expect(info.removed_return_count == 0u);
        expect(count_terminator_kind(k->definition(), DerivedInstructionTag::RETURN) == 1u);
    };

    "single_early_return_void_in_if"_test = [] {
        Module m;
        auto *k = m.create_kernel();
        auto *body = k->create_body_block();
        XIRBuilder b;
        b.set_insertion_point(body);
        auto *cond = m.create_constant_one(Type::of<bool>());
        auto *if_inst = b.if_(cond);
        auto *t = if_inst->create_true_block();
        auto *f = if_inst->create_false_block();
        auto *merge = if_inst->create_merge_block();
        b.set_insertion_point(t);
        b.return_void();
        b.set_insertion_point(f);
        b.br(merge);
        b.set_insertion_point(merge);
        b.return_void();
        auto info = early_return_elimination_pass_run_on_function(k);
        expect(info.removed_return_count == 1u);
        auto *def = k->definition();
        expect(count_terminator_kind(def, DerivedInstructionTag::RETURN) == 1u);
        expect(count_alloca(def) >= 1u);
    };

    "single_early_return_nonvoid_in_if"_test = [] {
        Module m;
        auto *c = m.create_callable(Type::of<int>());
        auto *body = c->create_body_block();
        XIRBuilder b;
        b.set_insertion_point(body);
        auto *cond = m.create_constant_one(Type::of<bool>());
        auto *if_inst = b.if_(cond);
        auto *t = if_inst->create_true_block();
        auto *f = if_inst->create_false_block();
        auto *merge = if_inst->create_merge_block();
        auto *v0 = m.create_constant_zero(Type::of<int>());
        auto *v1 = m.create_constant_one(Type::of<int>());
        b.set_insertion_point(t);
        b.return_(v0);
        b.set_insertion_point(f);
        b.br(merge);
        b.set_insertion_point(merge);
        b.return_(v1);
        auto info = early_return_elimination_pass_run_on_function(c);
        expect(info.removed_return_count == 1u);
        auto *def = c->definition();
        expect(count_terminator_kind(def, DerivedInstructionTag::RETURN) == 1u);
        expect(count_alloca(def) >= 2u);
    };

    "two_early_returns_void_in_nested_if"_test = [] {
        Module m;
        auto *k = m.create_kernel();
        auto *body = k->create_body_block();
        XIRBuilder b;
        b.set_insertion_point(body);
        auto *cond = m.create_constant_one(Type::of<bool>());
        auto *outer_if = b.if_(cond);
        auto *ot = outer_if->create_true_block();
        auto *of_ = outer_if->create_false_block();
        auto *outer_merge = outer_if->create_merge_block();
        b.set_insertion_point(ot);
        auto *inner_if = b.if_(cond);
        auto *it = inner_if->create_true_block();
        auto *if2 = inner_if->create_false_block();
        auto *inner_merge = inner_if->create_merge_block();
        b.set_insertion_point(it);
        b.return_void();
        b.set_insertion_point(if2);
        b.br(inner_merge);
        b.set_insertion_point(inner_merge);
        b.return_void();
        b.set_insertion_point(of_);
        b.br(outer_merge);
        b.set_insertion_point(outer_merge);
        b.return_void();
        auto info = early_return_elimination_pass_run_on_function(k);
        expect(info.removed_return_count == 2u);
        auto *def = k->definition();
        expect(count_terminator_kind(def, DerivedInstructionTag::RETURN) == 1u);
    };

    "early_return_in_false_branch"_test = [] {
        Module m;
        auto *k = m.create_kernel();
        auto *body = k->create_body_block();
        XIRBuilder b;
        b.set_insertion_point(body);
        auto *cond = m.create_constant_zero(Type::of<bool>());
        auto *if_inst = b.if_(cond);
        auto *t = if_inst->create_true_block();
        auto *f = if_inst->create_false_block();
        auto *merge = if_inst->create_merge_block();
        b.set_insertion_point(t);
        b.br(merge);
        b.set_insertion_point(f);
        b.return_void();
        b.set_insertion_point(merge);
        b.return_void();
        auto info = early_return_elimination_pass_run_on_function(k);
        expect(info.removed_return_count == 1u);
        auto *def = k->definition();
        expect(count_terminator_kind(def, DerivedInstructionTag::RETURN) == 1u);
    };

    "no_early_return_nonvoid"_test = [] {
        Module m;
        auto *c = m.create_callable(Type::of<float>());
        auto *body = c->create_body_block();
        XIRBuilder b;
        b.set_insertion_point(body);
        auto *v = m.create_constant_one(Type::of<float>());
        b.return_(v);
        auto info = early_return_elimination_pass_run_on_function(c);
        expect(info.removed_return_count == 0u);
        expect(count_terminator_kind(c->definition(), DerivedInstructionTag::RETURN) == 1u);
    };

    "module_run_multiple_functions"_test = [] {
        Module m;
        auto *k1 = m.create_kernel();
        auto *body1 = k1->create_body_block();
        auto *k2 = m.create_kernel();
        auto *body2 = k2->create_body_block();
        XIRBuilder b;
        auto *cond = m.create_constant_one(Type::of<bool>());
        b.set_insertion_point(body1);
        auto *if1 = b.if_(cond);
        auto *t1 = if1->create_true_block();
        auto *f1 = if1->create_false_block();
        auto *merge1 = if1->create_merge_block();
        b.set_insertion_point(t1);
        b.return_void();
        b.set_insertion_point(f1);
        b.br(merge1);
        b.set_insertion_point(merge1);
        b.return_void();
        b.set_insertion_point(body2);
        b.return_void();
        auto info = early_return_elimination_pass_run_on_module(&m);
        expect(info.removed_return_count == 1u);
        expect(count_terminator_kind(k1->definition(), DerivedInstructionTag::RETURN) == 1u);
        expect(count_terminator_kind(k2->definition(), DerivedInstructionTag::RETURN) == 1u);
    };
}

int main(int argc, char *argv[]) {
    boost::ut::detail::cfg::parse_arg_with_fallback(argc, const_cast<const char **>(argv));
    reg_early_return_elimination();
    return 0;
}
