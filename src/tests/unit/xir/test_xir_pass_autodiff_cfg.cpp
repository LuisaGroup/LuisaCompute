#include "ut/ut.hpp"
#include <luisa/ast/type_registry.h>
#include <luisa/xir/basic_block.h>
#include <luisa/xir/builder.h>
#include <luisa/xir/function.h>
#include <luisa/xir/instructions/if.h>
#include <luisa/xir/instructions/switch.h>
#include <luisa/xir/module.h>
#include <luisa/xir/passes/autodiff.h>

using namespace luisa;
using namespace luisa::compute;
using namespace luisa::compute::xir;
using namespace boost::ut;
using namespace boost::ut::literals;

namespace {

[[nodiscard]] KernelFunction *make_kernel_with_body(Module &m, BasicBlock *&body) noexcept {
    auto *kernel = m.create_kernel();
    body = kernel->create_body_block();
    return kernel;
}

[[nodiscard]] size_t count_instructions(FunctionDefinition *definition,
                                        DerivedInstructionTag tag) noexcept {
    size_t count = 0u;
    definition->traverse_instructions([&](Instruction *inst) noexcept {
        if (inst->derived_instruction_tag() == tag) { ++count; }
    });
    return count;
}

}// namespace

void register_autodiff_cfg_tests() {

    "forward_autodiff_null_merges_respect_enclosing_scope_boundary"_test = [] {
        Module m;
        BasicBlock *body;
        auto *kernel = make_kernel_with_body(m, body);
        auto *definition = kernel->definition();
        auto *x = kernel->create_value_argument(Type::of<float>());
        auto *condition = kernel->create_value_argument(Type::of<bool>());
        auto *selector = kernel->create_value_argument(Type::of<int>());
        XIRBuilder b;
        b.set_insertion_point(body);
        auto *scope = b.forward_autodiff_scope(1u);
        auto *scope_entry = scope->create_entry_block();
        auto *scope_merge = scope->create_merge_block();

        b.set_insertion_point(scope_entry);
        b.call(Type::of<void>(), AutodiffIntrinsicOp::AUTODIFF_PROPAGATE_GRADIENT,
               {x, m.create_constant_one(Type::of<float>())});
        auto *outer_if = b.if_(condition);
        auto *if_true = outer_if->create_true_block();
        auto *if_false = outer_if->create_false_block();
        expect(outer_if->merge_block() == nullptr);

        b.set_insertion_point(if_true);
        auto *inner_switch = b.switch_(selector);
        auto *default_block = inner_switch->create_default_block();
        auto *case_block = inner_switch->create_case_block(3);
        expect(inner_switch->merge_block() == nullptr);
        b.set_insertion_point(default_block);
        b.br(scope_merge);
        b.set_insertion_point(case_block);
        b.br(scope_merge);
        b.set_insertion_point(if_false);
        b.br(scope_merge);

        // This is outside the autodiff scope. Incorrect null-merge traversal
        // reaches it and creates a second gradient slot for `outside_value`.
        b.set_insertion_point(scope_merge);
        auto *outside_value = b.call(Type::of<float>(), ArithmeticOp::SIN, {x});
        static_cast<void>(outside_value);
        b.return_void();

        auto info = autodiff_pass_run_on_function(kernel);
        expect(info.transformed_scope_count == 1u);
        expect(count_instructions(definition, DerivedInstructionTag::AUTODIFF_SCOPE) == 0u);
        expect(count_instructions(definition, DerivedInstructionTag::AUTODIFF_INTRINSIC) == 0u);
        expect(count_instructions(definition, DerivedInstructionTag::IF) == 1u);
        expect(count_instructions(definition, DerivedInstructionTag::SWITCH) == 1u);
        expect(count_instructions(definition, DerivedInstructionTag::ALLOCA) == 1u);
    };

    "autodiff_null_merge_if_preprocesses_and_lowers_epilogue"_test = [] {
        Module m;
        BasicBlock *body;
        auto *kernel = make_kernel_with_body(m, body);
        auto *definition = kernel->definition();
        auto *x = kernel->create_value_argument(Type::of<float>());
        auto *condition = kernel->create_value_argument(Type::of<bool>());
        XIRBuilder b;
        b.set_insertion_point(body);
        auto *gradient_out = b.alloca_local(Type::of<float>());
        auto *scope = b.autodiff_scope();
        auto *scope_entry = scope->create_entry_block();
        auto *scope_merge = scope->create_merge_block();

        b.set_insertion_point(scope_entry);
        b.call(Type::of<void>(), AutodiffIntrinsicOp::AUTODIFF_REQUIRES_GRADIENT, {x});
        auto *y = b.call(Type::of<float>(), ArithmeticOp::SIN, {x});
        b.call(Type::of<void>(), AutodiffIntrinsicOp::AUTODIFF_GRADIENT_MARKER,
               {y, m.create_constant_one(Type::of<float>())});
        b.call(Type::of<void>(), AutodiffIntrinsicOp::AUTODIFF_BACKWARD, {});
        // This terminator is moved into the epilogue by split_at_backward.
        // Keeping its local merge null also makes preprocessing use the
        // enclosing autodiff-scope merge as the traversal boundary.
        auto *epilogue_if = b.if_(condition);
        auto *true_block = epilogue_if->create_true_block();
        auto *false_block = epilogue_if->create_false_block();
        expect(epilogue_if->merge_block() == nullptr);

        b.set_insertion_point(true_block);
        auto *true_gradient = b.call(Type::of<float>(), AutodiffIntrinsicOp::AUTODIFF_GRADIENT, {x});
        b.store(gradient_out, true_gradient);
        b.br(scope_merge);
        b.set_insertion_point(false_block);
        auto *false_gradient = b.call(Type::of<float>(), AutodiffIntrinsicOp::AUTODIFF_GRADIENT, {x});
        b.store(gradient_out, false_gradient);
        b.br(scope_merge);
        b.set_insertion_point(scope_merge);
        b.return_void();

        auto info = autodiff_pass_run_on_function(kernel);
        expect(info.transformed_scope_count == 1u);
        expect(count_instructions(definition, DerivedInstructionTag::AUTODIFF_SCOPE) == 0u);
        expect(count_instructions(definition, DerivedInstructionTag::AUTODIFF_INTRINSIC) == 0u);
        expect(count_instructions(definition, DerivedInstructionTag::IF) == 1u);
    };

    "autodiff_null_merge_switch_preprocesses_and_lowers_epilogue"_test = [] {
        Module m;
        BasicBlock *body;
        auto *kernel = make_kernel_with_body(m, body);
        auto *definition = kernel->definition();
        auto *x = kernel->create_value_argument(Type::of<float>());
        auto *selector = kernel->create_value_argument(Type::of<int>());
        XIRBuilder b;
        b.set_insertion_point(body);
        auto *gradient_out = b.alloca_local(Type::of<float>());
        auto *scope = b.autodiff_scope();
        auto *scope_entry = scope->create_entry_block();
        auto *scope_merge = scope->create_merge_block();

        b.set_insertion_point(scope_entry);
        b.call(Type::of<void>(), AutodiffIntrinsicOp::AUTODIFF_REQUIRES_GRADIENT, {x});
        auto *y = b.call(Type::of<float>(), ArithmeticOp::COS, {x});
        b.call(Type::of<void>(), AutodiffIntrinsicOp::AUTODIFF_GRADIENT_MARKER,
               {y, m.create_constant_one(Type::of<float>())});
        b.call(Type::of<void>(), AutodiffIntrinsicOp::AUTODIFF_BACKWARD, {});
        auto *epilogue_switch = b.switch_(selector);
        auto *default_block = epilogue_switch->create_default_block();
        auto *case_block = epilogue_switch->create_case_block(7);
        expect(epilogue_switch->merge_block() == nullptr);

        b.set_insertion_point(default_block);
        auto *default_gradient = b.call(Type::of<float>(), AutodiffIntrinsicOp::AUTODIFF_GRADIENT, {x});
        b.store(gradient_out, default_gradient);
        b.br(scope_merge);
        b.set_insertion_point(case_block);
        auto *case_gradient = b.call(Type::of<float>(), AutodiffIntrinsicOp::AUTODIFF_GRADIENT, {x});
        b.store(gradient_out, case_gradient);
        b.br(scope_merge);
        b.set_insertion_point(scope_merge);
        b.return_void();

        auto info = autodiff_pass_run_on_function(kernel);
        expect(info.transformed_scope_count == 1u);
        expect(count_instructions(definition, DerivedInstructionTag::AUTODIFF_SCOPE) == 0u);
        expect(count_instructions(definition, DerivedInstructionTag::AUTODIFF_INTRINSIC) == 0u);
        expect(count_instructions(definition, DerivedInstructionTag::SWITCH) == 1u);
    };
}

int main(int argc, char *argv[]) {
    boost::ut::detail::cfg::parse_arg_with_fallback(argc, const_cast<const char **>(argv));
    register_autodiff_cfg_tests();
    return 0;
}
