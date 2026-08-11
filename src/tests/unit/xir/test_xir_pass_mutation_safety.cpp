// Test for mutation safety across XIR transformation passes.
// This test covers verifier-preserving success and fail-closed rejection paths.

#include "ut/ut.hpp"

#include <luisa/ast/type_registry.h>
#include <luisa/xir/builder.h>
#include <luisa/xir/instructions/arithmetic.h>
#include <luisa/xir/instructions/autodiff.h>
#include <luisa/xir/instructions/call.h>
#include <luisa/xir/instructions/load.h>
#include <luisa/xir/instructions/outline.h>
#include <luisa/xir/instructions/ray_query.h>
#include <luisa/xir/instructions/store.h>
#include <luisa/xir/module.h>
#include <luisa/xir/passes/dce.h>
#include <luisa/xir/passes/fix_self_referential.h>
#include <luisa/xir/passes/if_conversion.h>
#include <luisa/xir/passes/inline.h>
#include <luisa/xir/passes/lex_scope_analysis.h>
#include <luisa/xir/passes/restructure_cfg.h>
#include <luisa/xir/passes/scalarizer.h>
#include <luisa/xir/verifier.h>

using namespace luisa;
using namespace luisa::compute;
using namespace luisa::compute::xir;
using namespace boost::ut;
using namespace boost::ut::literals;

namespace {

[[nodiscard]] size_t count_instructions(FunctionDefinition *function,
                                        DerivedInstructionTag tag) noexcept {
    auto count = 0u;
    function->traverse_instructions([&](Instruction *inst) noexcept {
        count += inst->derived_instruction_tag() == tag;
    });
    return count;
}

[[nodiscard]] KernelFunction *
make_plain_diamond(Module &module) noexcept {
    auto *kernel = module.create_kernel();
    auto *entry = kernel->create_body_block();
    auto *condition =
        kernel->create_value_argument(Type::of<bool>());
    auto *true_block = kernel->create_basic_block();
    auto *false_block = kernel->create_basic_block();
    auto *merge = kernel->create_basic_block();
    XIRBuilder builder;
    builder.set_insertion_point(entry);
    builder.cond_br(condition, true_block, false_block);
    builder.set_insertion_point(true_block);
    builder.br(merge);
    builder.set_insertion_point(false_block);
    builder.br(merge);
    builder.set_insertion_point(merge);
    builder.return_void();
    return kernel;
}

[[nodiscard]] KernelFunction *
make_iteration_limited_loop(Module &module) noexcept {
    auto *kernel = module.create_kernel();
    auto *entry = kernel->create_body_block();
    auto *header_condition =
        kernel->create_value_argument(Type::of<bool>());
    auto *latch_condition =
        kernel->create_value_argument(Type::of<bool>());
    auto *header = kernel->create_basic_block();
    auto *loop_body = kernel->create_basic_block();
    auto *header_exit = kernel->create_basic_block();
    auto *latch_exit = kernel->create_basic_block();
    XIRBuilder builder;
    builder.set_insertion_point(entry);
    builder.br(header);
    builder.set_insertion_point(header);
    builder.cond_br(
        header_condition, loop_body, header_exit);
    builder.set_insertion_point(loop_body);
    builder.cond_br(
        latch_condition, header, latch_exit);
    builder.set_insertion_point(header_exit);
    builder.return_void();
    builder.set_insertion_point(latch_exit);
    builder.return_void();
    return kernel;
}

}// namespace

void reg_xir_pass_mutation_safety() {
    "restructure_in_place_matches_transactional_success_once"_test = [] {
        Module transactional_module;
        auto *transactional_kernel =
            make_plain_diamond(transactional_module);
        Module in_place_module;
        auto *in_place_kernel =
            make_plain_diamond(in_place_module);
        Module function_in_place_module;
        auto *function_in_place_kernel =
            make_plain_diamond(function_in_place_module);

        auto transactional =
            restructure_cfg_pass_run_on_module(
                &transactional_module);
        auto in_place =
            restructure_cfg_pass_run_on_module(
                &in_place_module, nullptr,
                {.mutation_mode =
                     RestructureCFGMutationMode::
                         IN_PLACE_DISCARDABLE});
        auto function_in_place =
            restructure_cfg_pass_run_on_function(
                function_in_place_kernel,
                {.mutation_mode =
                     RestructureCFGMutationMode::
                         IN_PLACE_DISCARDABLE});

        expect(transactional.succeeded());
        expect(in_place.succeeded());
        expect(function_in_place.succeeded());
        expect(transactional.changed());
        expect(in_place.changed());
        expect(function_in_place.changed());
        expect(transactional.restructured_if_count == 1u);
        expect(in_place.restructured_if_count ==
               transactional.restructured_if_count);
        expect(in_place.restructured_loop_count ==
               transactional.restructured_loop_count);
        expect(in_place.restructured_switch_count ==
               transactional.restructured_switch_count);
        expect(in_place.canonicalized_cfg_count ==
               transactional.canonicalized_cfg_count);
        expect(transactional.boundary_verifier_count == 2u);
        expect(in_place.boundary_verifier_count == 2u);
        expect(
            transactional.definition_transform_invocation_count ==
            2u);
        expect(
            in_place.definition_transform_invocation_count == 1u);
        expect(
            function_in_place
                .definition_transform_invocation_count == 1u);
        expect(function_in_place.boundary_verifier_count == 2u);
        expect(
            count_instructions(
                transactional_kernel,
                DerivedInstructionTag::CONDITIONAL_BRANCH) ==
            0u);
        expect(
            count_instructions(
                in_place_kernel,
                DerivedInstructionTag::CONDITIONAL_BRANCH) ==
            0u);
        expect(count_instructions(
                   transactional_kernel,
                   DerivedInstructionTag::IF) == 1u);
        expect(count_instructions(
                   in_place_kernel,
                   DerivedInstructionTag::IF) == 1u);
        expect(count_instructions(
                   function_in_place_kernel,
                   DerivedInstructionTag::IF) == 1u);
        expect(
            transactional_kernel->basic_blocks().count_size() ==
            in_place_kernel->basic_blocks().count_size());
        auto output_requirements = XIRVerificationOptions{
            .require_no_phi = true,
            .require_unique_merge_blocks = true,
            .require_canonical_break_continue_targets = true};
        expect(xir_verify_module(
                   &transactional_module,
                   output_requirements)
                   .succeeded());
        expect(xir_verify_module(
                   &in_place_module,
                   output_requirements)
                   .succeeded());
        expect(xir_verify_module(
                   &function_in_place_module,
                   output_requirements)
                   .succeeded());
    };

    "restructure_mutation_mode_defines_failure_atomicity"_test = [] {
        Module transactional_module;
        auto *transactional_kernel =
            make_iteration_limited_loop(
                transactional_module);
        auto transactional_block_count =
            transactional_kernel->basic_blocks().count_size();
        Module in_place_module;
        auto *in_place_kernel =
            make_iteration_limited_loop(in_place_module);

        auto transactional =
            restructure_cfg_pass_run_on_module(
                &transactional_module, nullptr,
                {.main_iteration_limit = 1u,
                 .post_iteration_limit = 64u});
        auto in_place =
            restructure_cfg_pass_run_on_module(
                &in_place_module, nullptr,
                {.main_iteration_limit = 1u,
                 .post_iteration_limit = 64u,
                 .mutation_mode =
                     RestructureCFGMutationMode::
                         IN_PLACE_DISCARDABLE});

        expect(!transactional.succeeded());
        expect(!in_place.succeeded());
        expect(transactional.iteration_limit_count == 1u);
        expect(in_place.iteration_limit_count == 1u);
        expect(!transactional.changed());
        expect(in_place.changed());
        expect(
            transactional.definition_transform_invocation_count ==
            1u);
        expect(
            in_place.definition_transform_invocation_count == 1u);
        expect(transactional.boundary_verifier_count == 1u);
        expect(in_place.boundary_verifier_count == 1u);
        expect(
            transactional_kernel->basic_blocks().count_size() ==
            transactional_block_count);
        expect(count_instructions(
                   transactional_kernel,
                   DerivedInstructionTag::CONDITIONAL_BRANCH) ==
               2u);
        expect(count_instructions(
                   in_place_kernel,
                   DerivedInstructionTag::CONDITIONAL_BRANCH) ==
               0u);
        expect(xir_verify_module(
                   &transactional_module)
                   .succeeded());
        // The in-place input is intentionally not passed to another transform
        // after failure; its ownership contract requires module disposal.
    };

    "dce_preserves_structured_merge_ownership"_test = [] {
        Module module;
        auto *kernel = module.create_kernel();
        auto *body = kernel->create_body_block();
        auto *condition = kernel->create_value_argument(Type::of<bool>());
        XIRBuilder builder;
        builder.set_insertion_point(body);
        auto *if_inst = builder.if_(condition);
        auto *true_block = if_inst->create_true_block();
        auto *false_block = if_inst->create_false_block();
        auto *merge = if_inst->create_merge_block();
        builder.set_insertion_point(true_block);
        builder.return_void();
        builder.set_insertion_point(false_block);
        builder.return_void();
        builder.set_insertion_point(merge);
        builder.return_void();
        expect(xir_verify_module(&module).succeeded());
        (void)dce_pass_run_on_function(kernel);
        expect(if_inst->merge_block() == merge);
        expect(merge->parent_function() == kernel);
        expect(xir_verify_module(&module).succeeded());
    };

    "inline_multiblock_return_storage_dominates_cloned_returns"_test = [] {
        Module module;
        auto *callee = module.create_callable(Type::of<int>());
        auto *callee_body = callee->create_body_block();
        auto *true_block = callee->create_basic_block();
        auto *false_block = callee->create_basic_block();
        auto *condition = callee->create_value_argument(Type::of<bool>());
        auto *one = module.create_constant_one(Type::of<int>());
        auto *zero = module.create_constant_zero(Type::of<int>());
        XIRBuilder builder;
        builder.set_insertion_point(callee_body);
        builder.cond_br(condition, true_block, false_block);
        builder.set_insertion_point(true_block);
        builder.return_(one);
        builder.set_insertion_point(false_block);
        builder.return_(zero);

        auto *kernel = module.create_kernel();
        auto *kernel_body = kernel->create_body_block();
        auto *kernel_condition = kernel->create_value_argument(Type::of<bool>());
        builder.set_insertion_point(kernel_body);
        auto *call = builder.call(Type::of<int>(), callee, {kernel_condition});
        auto call_lock = call->lock();
        auto *sum = builder.call(Type::of<int>(), ArithmeticOp::BINARY_ADD,
                                 {call, one});
        auto *storage = builder.alloca_local(Type::of<int>());
        builder.store(storage, sum);
        builder.return_void();

        expect(xir_verify_module(&module).succeeded());
        auto info = inline_pass_run_on_module(&module);
        expect(info.inlined_call_count == 1u);
        expect(call_lock->use_list().empty());
        expect(count_instructions(kernel, DerivedInstructionTag::CALL) == 0u);
        expect(xir_verify_module(&module).succeeded());
    };

    "inline_rejects_mismatched_return_before_mutation"_test = [] {
        Module module;
        auto *callee = module.create_callable(nullptr);
        auto *callee_body = callee->create_body_block();
        XIRBuilder builder;
        builder.set_insertion_point(callee_body);
        builder.return_void();
        auto *kernel = module.create_kernel();
        auto *body = kernel->create_body_block();
        builder.set_insertion_point(body);
        auto *call = builder.call(Type::of<int>(), callee, {});
        auto *storage = builder.alloca_local(Type::of<int>());
        builder.store(storage, call);
        builder.return_void();
        auto before = body->instructions().count_size();
        auto info = inline_pass_run_on_module(&module);
        expect(info.inlined_call_count == 0u);
        expect(info.rejected_malformed_call_count == 1u);
        expect(call->is_linked());
        expect(body->instructions().count_size() == before);
    };

    "inline_rejects_lvalue_for_value_formal_atomically"_test = [] {
        Module module;
        auto *callee = module.create_callable(Type::of<int>());
        auto *formal = callee->create_value_argument(Type::of<int>());
        auto *callee_body = callee->create_body_block();
        XIRBuilder builder;
        builder.set_insertion_point(callee_body);
        builder.return_(formal);

        auto *kernel = module.create_kernel();
        auto *body = kernel->create_body_block();
        builder.set_insertion_point(body);
        auto *local = builder.alloca_local(Type::of<int>());
        auto *call = builder.call(Type::of<int>(), callee, {local});
        builder.return_void();
        auto before = body->instructions().count_size();

        auto info = inline_pass_run_on_module(&module);
        expect(info.inlined_call_count == 0u);
        expect(info.removed_callable_count == 0u);
        expect(info.rejected_malformed_call_count == 1u);
        expect(call->is_linked());
        expect(call->argument(0u) == local);
        expect(local->is_linked());
        expect(body->instructions().count_size() == before);
    };

    "inline_rejects_rvalue_for_reference_formal_atomically"_test = [] {
        Module module;
        auto *callee = module.create_callable(Type::of<int>());
        auto *formal = callee->create_reference_argument(Type::of<int>());
        auto *callee_body = callee->create_body_block();
        XIRBuilder builder;
        builder.set_insertion_point(callee_body);
        auto *loaded = builder.load(Type::of<int>(), formal);
        builder.return_(loaded);

        auto *kernel = module.create_kernel();
        auto *body = kernel->create_body_block();
        auto *actual = module.create_constant_one(Type::of<int>());
        builder.set_insertion_point(body);
        auto *call = builder.call(Type::of<int>(), callee, {actual});
        builder.return_void();
        auto before = body->instructions().count_size();

        auto info = inline_pass_run_on_module(&module);
        expect(info.inlined_call_count == 0u);
        expect(info.removed_callable_count == 0u);
        expect(info.rejected_malformed_call_count == 1u);
        expect(call->is_linked());
        expect(call->argument(0u) == actual);
        expect(body->instructions().count_size() == before);
    };

    "inline_rejects_non_argument_resource_actual_atomically"_test = [] {
        Module module;
        auto *resource_type = Type::buffer(Type::of<int>());
        auto *callee = module.create_callable(nullptr);
        callee->create_resource_argument(resource_type);
        auto *callee_body = callee->create_body_block();
        XIRBuilder builder;
        builder.set_insertion_point(callee_body);
        builder.return_void();

        auto *kernel = module.create_kernel();
        auto *body = kernel->create_body_block();
        auto *actual = module.create_undefined(resource_type);
        builder.set_insertion_point(body);
        auto *call = builder.call(nullptr, callee, {actual});
        builder.return_void();
        auto before = body->instructions().count_size();

        auto info = inline_pass_run_on_module(&module);
        expect(info.inlined_call_count == 0u);
        expect(info.removed_callable_count == 0u);
        expect(info.rejected_malformed_call_count == 1u);
        expect(call->is_linked());
        expect(call->argument(0u) == actual);
        expect(body->instructions().count_size() == before);
    };

    "inline_accepts_matching_argument_categories"_test = [] {
        Module module;
        auto *resource_type = Type::buffer(Type::of<int>());
        auto *callee = module.create_callable(Type::of<int>());
        auto *value_formal = callee->create_value_argument(Type::of<int>());
        callee->create_reference_argument(Type::of<int>());
        callee->create_resource_argument(resource_type);
        auto *callee_body = callee->create_body_block();
        XIRBuilder builder;
        builder.set_insertion_point(callee_body);
        builder.return_(value_formal);

        auto *kernel = module.create_kernel();
        auto *resource_actual = kernel->create_resource_argument(resource_type);
        auto *body = kernel->create_body_block();
        auto *value_actual = module.create_constant_one(Type::of<int>());
        builder.set_insertion_point(body);
        auto *reference_actual = builder.alloca_local(Type::of<int>());
        builder.store(reference_actual, value_actual);
        auto *call = builder.call(Type::of<int>(), callee,
                                  {value_actual, reference_actual, resource_actual});
        auto call_lock = call->lock();
        builder.store(reference_actual, call);
        builder.return_void();

        expect(xir_verify_module(&module).succeeded());
        auto info = inline_pass_run_on_module(&module);
        expect(info.inlined_call_count == 1u);
        expect(info.removed_callable_count == 1u);
        expect(info.rejected_malformed_call_count == 0u);
        expect(call_lock->use_list().empty());
        expect(count_instructions(kernel, DerivedInstructionTag::CALL) == 0u);
        expect(xir_verify_module(&module).succeeded());
    };

    "inline_all_counts_each_malformed_call_once"_test = [] {
        Module module;
        auto *invalid_callee = module.create_callable(Type::of<int>());
        auto *invalid_formal = invalid_callee->create_value_argument(Type::of<int>());
        auto *invalid_body = invalid_callee->create_body_block();
        XIRBuilder builder;
        builder.set_insertion_point(invalid_body);
        builder.return_(invalid_formal);

        auto *valid_callee = module.create_callable(Type::of<int>());
        auto *valid_body = valid_callee->create_body_block();
        builder.set_insertion_point(valid_body);
        builder.return_(module.create_constant_one(Type::of<int>()));

        auto *kernel = module.create_kernel();
        auto *body = kernel->create_body_block();
        builder.set_insertion_point(body);
        auto *local = builder.alloca_local(Type::of<int>());
        auto *invalid_call = builder.call(Type::of<int>(), invalid_callee, {local});
        auto *valid_call = builder.call(Type::of<int>(), valid_callee, {});
        auto valid_call_lock = valid_call->lock();
        builder.store(local, valid_call);
        builder.return_void();

        auto info = inline_all_pass_run_on_module(&module);
        expect(info.inlined_call_count == 1u);
        expect(info.removed_callable_count == 1u);
        expect(info.rejected_malformed_call_count == 1u);
        expect(invalid_call->is_linked());
        expect(invalid_call->argument(0u) == local);
        expect(valid_call_lock->use_list().empty());
        expect(count_instructions(kernel, DerivedInstructionTag::CALL) == 1u);
    };

    "if_conversion_rejects_speculative_integer_division"_test = [] {
        Module module;
        auto *kernel = module.create_kernel();
        auto *body = kernel->create_body_block();
        auto *true_block = kernel->create_basic_block();
        auto *false_block = kernel->create_basic_block();
        auto *merge = kernel->create_basic_block();
        auto *condition = kernel->create_value_argument(Type::of<bool>());
        auto *divisor = kernel->create_value_argument(Type::of<int>());
        auto *one = module.create_constant_one(Type::of<int>());
        XIRBuilder builder;
        builder.set_insertion_point(body);
        builder.cond_br(condition, true_block, false_block);
        builder.set_insertion_point(true_block);
        auto *quotient = builder.call(Type::of<int>(), ArithmeticOp::BINARY_DIV,
                                      {one, divisor});
        builder.br(merge);
        builder.set_insertion_point(false_block);
        builder.br(merge);
        builder.set_insertion_point(merge);
        builder.phi(Type::of<int>(), {{quotient, true_block}, {one, false_block}});
        builder.return_void();
        expect(xir_verify_module(&module).succeeded());
        auto info = if_conversion_pass_run_on_function(kernel);
        expect(info.converted_diamond_count == 0u);
        expect(quotient->is_linked());
        expect(body->terminator()->derived_instruction_tag() ==
               DerivedInstructionTag::CONDITIONAL_BRANCH);
        expect(xir_verify_module(&module).succeeded());
    };

    "if_conversion_rejects_speculative_numeric_cast"_test = [] {
        Module module;
        auto *kernel = module.create_kernel();
        auto *body = kernel->create_body_block();
        auto *true_block = kernel->create_basic_block();
        auto *false_block = kernel->create_basic_block();
        auto *merge = kernel->create_basic_block();
        auto *condition = kernel->create_value_argument(Type::of<bool>());
        auto *floating = kernel->create_value_argument(Type::of<float>());
        auto *one = module.create_constant_one(Type::of<int>());
        XIRBuilder builder;
        builder.set_insertion_point(body);
        builder.cond_br(condition, true_block, false_block);
        builder.set_insertion_point(true_block);
        auto *converted = builder.static_cast_(Type::of<int>(), floating);
        builder.br(merge);
        builder.set_insertion_point(false_block);
        builder.br(merge);
        builder.set_insertion_point(merge);
        builder.phi(Type::of<int>(),
                    {{converted, true_block}, {one, false_block}});
        builder.return_void();

        expect(xir_verify_module(&module).succeeded());
        auto info = if_conversion_pass_run_on_function(kernel);
        expect(info.converted_diamond_count == 0u);
        expect(converted->parent_block() == true_block);
        expect(body->terminator()->derived_instruction_tag() ==
               DerivedInstructionTag::CONDITIONAL_BRANCH);
        expect(xir_verify_module(&module).succeeded());
    };

    "if_conversion_accepts_total_integer_to_float_cast"_test = [] {
        Module module;
        auto *kernel = module.create_kernel();
        auto *body = kernel->create_body_block();
        auto *true_block = kernel->create_basic_block();
        auto *false_block = kernel->create_basic_block();
        auto *merge = kernel->create_basic_block();
        auto *condition =
            kernel->create_value_argument(Type::of<bool>());
        auto *integer =
            kernel->create_value_argument(Type::of<int>());
        auto *zero =
            module.create_constant_zero(Type::of<float>());
        XIRBuilder builder;
        builder.set_insertion_point(body);
        builder.cond_br(condition, true_block, false_block);
        builder.set_insertion_point(true_block);
        auto *converted =
            builder.static_cast_(Type::of<float>(), integer);
        builder.br(merge);
        builder.set_insertion_point(false_block);
        builder.br(merge);
        builder.set_insertion_point(merge);
        builder.phi(
            Type::of<float>(),
            {{converted, true_block}, {zero, false_block}});
        builder.return_void();

        expect(xir_verify_module(&module).succeeded());
        auto info = if_conversion_pass_run_on_function(kernel);
        expect(info.converted_diamond_count == 1u);
        expect(converted->parent_block() == body);
        expect(body->terminator()->derived_instruction_tag() ==
               DerivedInstructionTag::BRANCH);
        expect(xir_verify_module(&module).succeeded());
    };

    "if_conversion_rejects_speculative_dynamic_extract"_test = [] {
        Module module;
        auto *kernel = module.create_kernel();
        auto *body = kernel->create_body_block();
        auto *true_block = kernel->create_basic_block();
        auto *false_block = kernel->create_basic_block();
        auto *merge = kernel->create_basic_block();
        auto *condition = kernel->create_value_argument(Type::of<bool>());
        auto *aggregate = kernel->create_value_argument(Type::of<int4>());
        auto *index = kernel->create_value_argument(Type::of<uint>());
        auto *one = module.create_constant_one(Type::of<int>());
        XIRBuilder builder;
        builder.set_insertion_point(body);
        builder.cond_br(condition, true_block, false_block);
        builder.set_insertion_point(true_block);
        auto *extracted = builder.call(
            Type::of<int>(), ArithmeticOp::EXTRACT,
            {aggregate, index});
        builder.br(merge);
        builder.set_insertion_point(false_block);
        builder.br(merge);
        builder.set_insertion_point(merge);
        builder.phi(Type::of<int>(),
                    {{extracted, true_block}, {one, false_block}});
        builder.return_void();

        expect(xir_verify_module(&module).succeeded());
        auto info = if_conversion_pass_run_on_function(kernel);
        expect(info.converted_diamond_count == 0u);
        expect(extracted->parent_block() == true_block);
        expect(body->terminator()->derived_instruction_tag() ==
               DerivedInstructionTag::CONDITIONAL_BRANCH);
        expect(xir_verify_module(&module).succeeded());
    };

    "if_conversion_rejects_incomplete_phi_atomically"_test = [] {
        Module module;
        auto *kernel = module.create_kernel();
        auto *body = kernel->create_body_block();
        auto *true_block = kernel->create_basic_block();
        auto *false_block = kernel->create_basic_block();
        auto *merge = kernel->create_basic_block();
        auto *condition = kernel->create_value_argument(Type::of<bool>());
        auto *one = module.create_constant_one(Type::of<int>());
        XIRBuilder builder;
        builder.set_insertion_point(body);
        builder.cond_br(condition, true_block, false_block);
        builder.set_insertion_point(true_block);
        auto *value = builder.call(Type::of<int>(), ArithmeticOp::BINARY_ADD,
                                   {one, one});
        builder.br(merge);
        builder.set_insertion_point(false_block);
        builder.br(merge);
        builder.set_insertion_point(merge);
        auto *phi = builder.phi(Type::of<int>(), {{value, true_block}});
        builder.return_void();
        auto info = if_conversion_pass_run_on_function(kernel);
        expect(info.converted_diamond_count == 0u);
        expect(value->parent_block() == true_block);
        expect(phi->incoming_count() == 1u);
        expect(body->terminator()->derived_instruction_tag() ==
               DerivedInstructionTag::CONDITIONAL_BRANCH);
    };

    "scalarizer_does_not_share_extracts_across_sibling_blocks"_test = [] {
        Module module;
        auto *kernel = module.create_kernel();
        auto *body = kernel->create_body_block();
        auto *true_block = kernel->create_basic_block();
        auto *false_block = kernel->create_basic_block();
        auto *merge = kernel->create_basic_block();
        auto *condition = kernel->create_value_argument(Type::of<bool>());
        auto *input = kernel->create_value_argument(Type::of<float2>());
        auto *zero = module.create_constant_zero(Type::of<float2>());
        XIRBuilder builder;
        builder.set_insertion_point(body);
        auto *storage = builder.alloca_local(Type::of<float2>());
        builder.cond_br(condition, true_block, false_block);
        builder.set_insertion_point(true_block);
        auto *true_value = builder.call(Type::of<float2>(), ArithmeticOp::BINARY_ADD,
                                        {input, zero});
        builder.store(storage, true_value);
        builder.br(merge);
        builder.set_insertion_point(false_block);
        auto *false_value = builder.call(Type::of<float2>(), ArithmeticOp::BINARY_SUB,
                                         {input, zero});
        builder.store(storage, false_value);
        builder.br(merge);
        builder.set_insertion_point(merge);
        builder.return_void();
        expect(xir_verify_module(&module).succeeded());
        auto info = scalarizer_pass_run_on_function(kernel);
        expect(info.scalarized_inst_count == 2u);
        expect(xir_verify_module(&module).succeeded());
        expect(count_instructions(kernel, DerivedInstructionTag::ARITHMETIC) > 2u);
    };

    "fix_self_referential_uses_exact_unique_store_target"_test = [] {
        Module module;
        auto *kernel = module.create_kernel();
        auto *body = kernel->create_body_block();
        auto *index = module.create_constant_zero(Type::of<uint>());
        auto *scalar = module.create_constant_one(Type::of<float>());
        XIRBuilder builder;
        builder.set_insertion_point(body);
        auto *storage = builder.alloca_local(Type::of<float2>());
        auto *insert = builder.call(
            Type::of<float2>(), ArithmeticOp::INSERT,
            {module.create_undefined(Type::of<float2>()), scalar, index});
        insert->set_operand(0u, insert);
        builder.store(storage, insert);
        builder.return_void();
        auto info = fix_self_referential_pass_run_on_function(kernel);
        expect(info.fixed_count == 1u);
        expect(info.unresolved_count == 0u);
        expect(info.succeeded());
        expect(insert->operand(0u)->isa<LoadInst>());
        expect(static_cast<LoadInst *>(insert->operand(0u))->variable() == storage);
        expect(xir_verify_module(&module).succeeded());
    };

    "fix_self_referential_rejects_ambiguous_store_targets"_test = [] {
        Module module;
        auto *kernel = module.create_kernel();
        auto *body = kernel->create_body_block();
        auto *index = module.create_constant_zero(Type::of<uint>());
        auto *scalar = module.create_constant_one(Type::of<float>());
        XIRBuilder builder;
        builder.set_insertion_point(body);
        auto *first_storage = builder.alloca_local(Type::of<float2>());
        auto *second_storage = builder.alloca_local(Type::of<float2>());
        auto *insert = builder.call(
            Type::of<float2>(), ArithmeticOp::INSERT,
            {module.create_undefined(Type::of<float2>()), scalar, index});
        insert->set_operand(0u, insert);
        builder.store(first_storage, insert);
        builder.store(second_storage, insert);
        builder.return_void();
        auto info = fix_self_referential_pass_run_on_function(kernel);
        expect(info.fixed_count == 0u);
        expect(info.unresolved_count == 1u);
        expect(!info.succeeded());
        expect(insert->operand(0u) == insert);
        expect(count_instructions(kernel, DerivedInstructionTag::LOAD) == 0u);
    };

    "fix_self_referential_rejects_non_aggregate_insert_self_operands"_test = [] {
        for (auto self_operand : {1u, 2u}) {
            Module module;
            auto *kernel = module.create_kernel();
            auto *body = kernel->create_body_block();
            auto *index = module.create_constant_zero(Type::of<uint>());
            auto *scalar = module.create_constant_one(Type::of<float>());
            XIRBuilder builder;
            builder.set_insertion_point(body);
            auto *storage = builder.alloca_local(Type::of<float2>());
            auto *insert = builder.call(
                Type::of<float2>(), ArithmeticOp::INSERT,
                {module.create_undefined(Type::of<float2>()), scalar, index});
            insert->set_operand(self_operand, insert);
            builder.store(storage, insert);
            builder.return_void();
            auto instruction_count = body->instructions().count_size();

            auto info = fix_self_referential_pass_run_on_function(kernel);

            expect(info.fixed_count == 0u);
            expect(info.unresolved_count == 1u);
            expect(!info.succeeded());
            expect(insert->operand(self_operand) == insert);
            expect(body->instructions().count_size() == instruction_count);
            expect(count_instructions(kernel, DerivedInstructionTag::LOAD) == 0u);
        }
    };

    "fix_self_referential_rejects_mixed_insert_self_operands_atomically"_test = [] {
        Module module;
        auto *kernel = module.create_kernel();
        auto *body = kernel->create_body_block();
        auto *index = module.create_constant_zero(Type::of<uint>());
        auto *scalar = module.create_constant_one(Type::of<float>());
        XIRBuilder builder;
        builder.set_insertion_point(body);
        auto *storage = builder.alloca_local(Type::of<float2>());
        auto *insert = builder.call(
            Type::of<float2>(), ArithmeticOp::INSERT,
            {module.create_undefined(Type::of<float2>()), scalar, index});
        insert->set_operand(0u, insert);
        insert->set_operand(1u, insert);
        builder.store(storage, insert);
        builder.return_void();
        auto instruction_count = body->instructions().count_size();

        auto info = fix_self_referential_pass_run_on_function(kernel);

        expect(info.fixed_count == 0u);
        expect(info.unresolved_count == 2u);
        expect(!info.succeeded());
        expect(insert->operand(0u) == insert);
        expect(insert->operand(1u) == insert);
        expect(body->instructions().count_size() == instruction_count);
        expect(count_instructions(kernel, DerivedInstructionTag::LOAD) == 0u);
    };

    "fix_self_referential_function_late_failure_rolls_back_all_candidates"_test = [] {
        Module module;
        auto *kernel = module.create_kernel();
        auto *body = kernel->create_body_block();
        auto *index = module.create_constant_zero(Type::of<uint>());
        auto *scalar = module.create_constant_one(Type::of<float>());
        XIRBuilder builder;
        builder.set_insertion_point(body);
        auto *valid_storage =
            builder.alloca_local(Type::of<float2>());
        auto *valid = builder.call(
            Type::of<float2>(), ArithmeticOp::INSERT,
            {module.create_undefined(Type::of<float2>()),
             scalar, index});
        valid->set_operand(0u, valid);
        builder.store(valid_storage, valid);
        auto *invalid_storage =
            builder.alloca_local(Type::of<float2>());
        auto *invalid = builder.call(
            Type::of<float2>(), ArithmeticOp::INSERT,
            {module.create_undefined(Type::of<float2>()),
             scalar, index});
        invalid->set_operand(1u, invalid);
        builder.store(invalid_storage, invalid);
        builder.return_void();
        auto instruction_count =
            body->instructions().count_size();

        auto info =
            fix_self_referential_pass_run_on_function(kernel);

        expect(!info.succeeded());
        expect(info.fixed_count == 0u);
        expect(info.unresolved_count == 1u);
        expect(valid->operand(0u) == valid);
        expect(invalid->operand(1u) == invalid);
        expect(body->instructions().count_size() ==
               instruction_count);
        expect(count_instructions(
                   kernel, DerivedInstructionTag::LOAD) == 0u);
    };

    "fix_self_referential_module_late_failure_is_atomic"_test = [] {
        Module module;
        auto *index =
            module.create_constant_zero(Type::of<uint>());
        auto *scalar =
            module.create_constant_one(Type::of<float>());
        auto make_candidate =
            [&](bool valid_candidate) noexcept {
                auto *kernel = module.create_kernel();
                auto *body = kernel->create_body_block();
                XIRBuilder builder;
                builder.set_insertion_point(body);
                auto *storage =
                    builder.alloca_local(Type::of<float2>());
                auto *insert = builder.call(
                    Type::of<float2>(), ArithmeticOp::INSERT,
                    {module.create_undefined(Type::of<float2>()),
                     scalar, index});
                insert->set_operand(
                    valid_candidate ? 0u : 2u, insert);
                builder.store(storage, insert);
                builder.return_void();
                return std::pair{kernel, insert};
            };
        auto [valid_kernel, valid] = make_candidate(true);
        auto [invalid_kernel, invalid] = make_candidate(false);
        auto valid_count =
            valid_kernel->body_block()->instructions().count_size();
        auto invalid_count =
            invalid_kernel->body_block()->instructions().count_size();

        auto info =
            fix_self_referential_pass_run_on_module(&module);

        expect(!info.succeeded());
        expect(info.fixed_count == 0u);
        expect(info.unresolved_count == 1u);
        expect(valid->operand(0u) == valid);
        expect(invalid->operand(2u) == invalid);
        expect(valid_kernel->body_block()
                   ->instructions().count_size() == valid_count);
        expect(invalid_kernel->body_block()
                   ->instructions().count_size() == invalid_count);
        expect(count_instructions(
                   valid_kernel,
                   DerivedInstructionTag::LOAD) == 0u);
        expect(count_instructions(
                   invalid_kernel,
                   DerivedInstructionTag::LOAD) == 0u);
    };

    "fix_self_referential_preserves_legal_loop_phi_self_reference"_test = [] {
        Module module;
        auto *kernel = module.create_kernel();
        auto *body = kernel->create_body_block();
        auto *header = kernel->create_basic_block();
        auto *latch = kernel->create_basic_block();
        auto *exit = kernel->create_basic_block();
        auto *condition = kernel->create_value_argument(Type::of<bool>());
        auto *zero = module.create_constant_zero(Type::of<uint>());
        XIRBuilder builder;
        builder.set_insertion_point(body);
        builder.br(header);
        builder.set_insertion_point(header);
        auto *phi = builder.phi(Type::of<uint>());
        phi->add_incoming(zero, body);
        phi->add_incoming(phi, latch);
        builder.cond_br(condition, latch, exit);
        builder.set_insertion_point(latch);
        builder.br(header);
        builder.set_insertion_point(exit);
        builder.return_void();

        expect(xir_verify_module(&module).succeeded());
        auto info = fix_self_referential_pass_run_on_function(kernel);
        expect(info.fixed_count == 0u);
        expect(info.unresolved_count == 0u);
        expect(info.succeeded());
        expect(phi->incoming(1u).value == phi);
        expect(xir_verify_module(&module).succeeded());
    };

    "fix_self_referential_accepts_null_or_bodyless_function"_test = [] {
        Module module;
        auto *kernel = module.create_kernel();
        auto null_info = fix_self_referential_pass_run_on_function(nullptr);
        auto bodyless_info = fix_self_referential_pass_run_on_function(kernel);
        expect(null_info.fixed_count == 0u);
        expect(null_info.unresolved_count == 0u);
        expect(bodyless_info.fixed_count == 0u);
        expect(bodyless_info.unresolved_count == 0u);
    };

    "lex_scope_walks_autodiff_and_outline_regions"_test = [] {
        for (auto use_autodiff : {false, true}) {
            Module module;
            auto *kernel = module.create_kernel();
            auto *body = kernel->create_body_block();
            auto *one = module.create_constant_one(Type::of<int>());
            XIRBuilder builder;
            builder.set_insertion_point(body);
            BasicBlock *nested = nullptr;
            BasicBlock *merge = nullptr;
            if (use_autodiff) {
                auto *scope = builder.autodiff_scope();
                nested = scope->create_entry_block();
                merge = scope->create_merge_block();
            } else {
                auto *outline = builder.outline();
                nested = outline->create_target_block();
                merge = outline->create_merge_block();
            }
            builder.set_insertion_point(nested);
            auto *nested_value = builder.call(
                Type::of<int>(), ArithmeticOp::BINARY_ADD, {one, one});
            builder.br(merge);
            builder.set_insertion_point(merge);
            builder.call(Type::of<int>(), ArithmeticOp::BINARY_ADD,
                         {nested_value, one});
            builder.return_void();
            auto info = lex_scope_analysis_pass_run_on_function(kernel, {});
            expect(info.lexical_scope_breakers.contains(nested_value));
        }
    };

    "lex_scope_walks_ray_query_dispatch_handlers"_test = [] {
        Module module;
        auto *kernel = module.create_kernel();
        auto *body = kernel->create_body_block();
        auto *one = module.create_constant_one(Type::of<int>());
        XIRBuilder builder;
        builder.set_insertion_point(body);
        auto *query = builder.alloca_local(Type::of<int>());
        auto *loop = builder.ray_query_loop();
        auto *dispatch_block = loop->create_dispatch_block();
        auto *merge = loop->create_merge_block();
        builder.set_insertion_point(dispatch_block);
        auto *dispatch_value = builder.call(
            Type::of<int>(), ArithmeticOp::BINARY_ADD, {one, one});
        auto *dispatch = builder.ray_query_dispatch(query);
        dispatch->set_exit_block(merge);
        auto *surface = dispatch->create_on_surface_candidate_block();
        auto *procedural = dispatch->create_on_procedural_candidate_block();
        builder.set_insertion_point(surface);
        builder.call(Type::of<int>(), ArithmeticOp::BINARY_ADD,
                     {dispatch_value, one});
        builder.br(dispatch_block);
        builder.set_insertion_point(procedural);
        builder.br(dispatch_block);
        builder.set_insertion_point(merge);
        builder.return_void();
        auto info = lex_scope_analysis_pass_run_on_function(kernel, {});
        expect(info.lexical_scope_breakers.empty());
    };

    "lex_scope_null_and_declaration_inputs_are_empty"_test = [] {
        Module module;
        auto *declaration = module.create_external_function(nullptr);
        auto null_info =
            lex_scope_analysis_pass_run_on_function(nullptr, {});
        auto declaration_info =
            lex_scope_analysis_pass_run_on_function(declaration, {});
        expect(null_info.lexical_scope_breakers.empty());
        expect(null_info.lexical_scope_breaks_ordered.empty());
        expect(declaration_info.lexical_scope_breakers.empty());
        expect(declaration_info.lexical_scope_breaks_ordered.empty());
    };
}

int main(int argc, char *argv[]) {
    boost::ut::detail::cfg::parse_arg_with_fallback(
        argc, const_cast<const char **>(argv));
    reg_xir_pass_mutation_safety();
    return 0;
}
