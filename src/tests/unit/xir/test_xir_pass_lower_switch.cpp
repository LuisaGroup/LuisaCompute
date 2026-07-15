#include "ut/ut.hpp"
#include <luisa/xir/basic_block.h>
#include <luisa/xir/builder.h>
#include <luisa/xir/function.h>
#include <luisa/xir/instructions/branch.h>
#include <luisa/xir/instructions/arithmetic.h>
#include <luisa/xir/instructions/if.h>
#include <luisa/xir/instructions/return.h>
#include <luisa/xir/instructions/switch.h>
#include <luisa/xir/constant.h>
#include <luisa/xir/module.h>
#include <luisa/xir/passes/lower_switch.h>

#include <limits>

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

[[nodiscard]] size_t count_terminator_kind(FunctionDefinition *def, DerivedInstructionTag tag) noexcept {
    size_t n = 0u;
    def->traverse_basic_blocks([&](BasicBlock *bb) noexcept {
        if (bb->is_terminated() && bb->terminator()->derived_instruction_tag() == tag) { ++n; }
    });
    return n;
}

} // namespace

void reg_lower_switch() {

    "lower_empty_switch_to_branch"_test = [] {
        Module m;
        BasicBlock *body;
        auto *k = make_kernel_with_body(m, body);
        auto *def = k->definition();
        int v0 = 0;
        auto *val = m.create_constant(Type::of<int>(), &v0);
        XIRBuilder b;
        b.set_insertion_point(body);
        auto *sw = b.switch_(val);
        auto *def_bb = sw->create_default_block();
        auto *merge = sw->create_merge_block();
        b.set_insertion_point(def_bb);
        b.br(merge);
        b.set_insertion_point(merge);
        b.return_void();
        expect(count_terminator_kind(def, DerivedInstructionTag::SWITCH) == 1u);
        auto info = lower_switch_pass_run_on_function(k);
        expect(info.lowered_switch_count == 1u);
        expect(count_terminator_kind(def, DerivedInstructionTag::SWITCH) == 0u);
        expect(body->terminator()->isa<BranchInst>());
        expect(static_cast<BranchInst *>(body->terminator())->target_block() == def_bb);
    };

    "lower_switch_to_if_chain"_test = [] {
        Module m;
        BasicBlock *body;
        auto *k = make_kernel_with_body(m, body);
        auto *def = k->definition();
        int v0 = 0;
        auto *val = m.create_constant(Type::of<int>(), &v0);
        XIRBuilder b;
        b.set_insertion_point(body);
        auto *sw = b.switch_(val);
        auto *c0 = sw->create_case_block(0);
        auto *c1 = sw->create_case_block(1);
        auto *def_bb = sw->create_default_block();
        auto *merge = sw->create_merge_block();
        b.set_insertion_point(c0);
        b.br(merge);
        b.set_insertion_point(c1);
        b.br(merge);
        b.set_insertion_point(def_bb);
        b.br(merge);
        b.set_insertion_point(merge);
        b.return_void();
        expect(count_terminator_kind(def, DerivedInstructionTag::SWITCH) == 1u);
        expect(count_terminator_kind(def, DerivedInstructionTag::IF) == 0u);
        auto info = lower_switch_pass_run_on_function(k);
        expect(info.lowered_switch_count == 1u);
        expect(count_terminator_kind(def, DerivedInstructionTag::SWITCH) == 0u);
        expect(count_terminator_kind(def, DerivedInstructionTag::IF) == 2u);
        auto *first = static_cast<IfInst *>(body->terminator());
        expect(first->true_block() == c0);
        expect(first->merge_block() == merge);
        auto *second_header = first->false_block();
        expect(second_header->terminator()->isa<IfInst>());
        auto *second = static_cast<IfInst *>(second_header->terminator());
        expect(second->true_block() == c1);
        expect(second->false_block() == def_bb);
        expect(second->merge_block() == merge);
    };

    "lower_switch_preserves_merge_and_case_flow"_test = [] {
        Module m;
        BasicBlock *body;
        auto *k = make_kernel_with_body(m, body);
        auto *def = k->definition();
        int v0 = 0;
        auto *val = m.create_constant(Type::of<int>(), &v0);
        XIRBuilder b;
        b.set_insertion_point(body);
        auto *sw = b.switch_(val);
        auto *c0 = sw->create_case_block(0);
        auto *def_bb = sw->create_default_block();
        auto *merge = sw->create_merge_block();
        b.set_insertion_point(c0);
        b.br(merge);
        b.set_insertion_point(def_bb);
        b.br(merge);
        b.set_insertion_point(merge);
        b.return_void();
        auto info = lower_switch_pass_run_on_function(k);
        expect(info.lowered_switch_count == 1u);
        auto *lowered = static_cast<IfInst *>(body->terminator());
        expect(lowered->merge_block() == merge);
        expect(lowered->true_block() == c0);
        expect(lowered->false_block() == def_bb);
        expect(static_cast<BranchInst *>(c0->terminator())->target_block() == merge);
        expect(static_cast<BranchInst *>(def_bb->terminator())->target_block() == merge);
    };

    "lower_switch_uses_exact_selector_typed_case_constants"_test = [] {
        auto check = []<typename T>(SwitchInst::case_value_type case_value, T expected) noexcept {
            Module m;
            BasicBlock *body;
            auto *k = make_kernel_with_body(m, body);
            auto *selector = k->create_value_argument(Type::of<T>());
            XIRBuilder b;
            b.set_insertion_point(body);
            auto *sw = b.switch_(selector);
            auto *case_bb = sw->create_case_block(case_value);
            auto *default_bb = sw->create_default_block();
            auto *merge = sw->create_merge_block();
            b.set_insertion_point(case_bb);
            b.br(merge);
            b.set_insertion_point(default_bb);
            b.br(merge);
            b.set_insertion_point(merge);
            b.return_void();

            auto info = lower_switch_pass_run_on_function(k);
            expect(info.lowered_switch_count == 1u);
            auto *if_inst = static_cast<IfInst *>(body->terminator());
            expect(if_inst->condition()->isa<ArithmeticInst>());
            auto *eq = static_cast<ArithmeticInst *>(if_inst->condition());
            expect(eq->op() == ArithmeticOp::BINARY_EQUAL);
            expect(eq->operand(0) == selector);
            expect(eq->operand(1)->isa<Constant>());
            auto *case_const = static_cast<Constant *>(eq->operand(1));
            expect(case_const->type() == Type::of<T>());
            expect(case_const->as<T>() == expected);
        };

        check.template operator()<int64_t>(-123456789, int64_t{-123456789});
        check.template operator()<uint64_t>(-1, std::numeric_limits<uint64_t>::max());
    };

    "lower_switch_preserves_null_merge_marker"_test = [] {
        Module m;
        BasicBlock *body;
        auto *k = make_kernel_with_body(m, body);
        auto *selector = k->create_value_argument(Type::of<int>());
        XIRBuilder b;
        b.set_insertion_point(body);
        auto *sw = b.switch_(selector);
        auto *case_bb = sw->create_case_block(7);
        auto *default_bb = sw->create_default_block();
        b.set_insertion_point(case_bb);
        b.return_void();
        b.set_insertion_point(default_bb);
        b.return_void();

        auto info = lower_switch_pass_run_on_function(k);
        expect(info.lowered_switch_count == 1u);
        auto *lowered = static_cast<IfInst *>(body->terminator());
        expect(lowered->merge_block() == nullptr);
        expect(lowered->true_block() == case_bb);
        expect(lowered->false_block() == default_bb);
    };
}

int main(int argc, char *argv[]) {
    boost::ut::detail::cfg::parse_arg_with_fallback(argc, const_cast<const char **>(argv));
    reg_lower_switch();
    return 0;
}
