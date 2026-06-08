#include "ut/ut.hpp"
#include <luisa/xir/basic_block.h>
#include <luisa/xir/builder.h>
#include <luisa/xir/function.h>
#include <luisa/xir/instructions/branch.h>
#include <luisa/xir/instructions/if.h>
#include <luisa/xir/instructions/return.h>
#include <luisa/xir/instructions/switch.h>
#include <luisa/xir/module.h>
#include <luisa/xir/passes/lower_switch.h>

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
        lower_switch_pass_run_on_function(k);
        expect(count_terminator_kind(def, DerivedInstructionTag::SWITCH) == 0u);
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
        expect(count_terminator_kind(def, DerivedInstructionTag::IF) >= 2u);
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
        lower_switch_pass_run_on_function(k);
        // merge block must still be reachable
        bool found_merge = false;
        def->traverse_basic_blocks([&](BasicBlock *bb) noexcept { if (bb == merge) found_merge = true; });
        expect(found_merge);
    };
}

int main(int argc, char *argv[]) {
    boost::ut::detail::cfg::parse_arg_with_fallback(argc, const_cast<const char **>(argv));
    reg_lower_switch();
    return 0;
}
