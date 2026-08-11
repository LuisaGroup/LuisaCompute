// Tests for coroutine-semantic local-allocation lifetime contraction.

#include "ut/ut.hpp"

#include <algorithm>

#include <luisa/xir/basic_block.h>
#include <luisa/xir/builder.h>
#include <luisa/xir/function.h>
#include <luisa/xir/instructions/alloca.h>
#include <luisa/xir/instructions/arithmetic.h>
#include <luisa/xir/instructions/branch.h>
#include <luisa/xir/instructions/coro.h>
#include <luisa/xir/instructions/gep.h>
#include <luisa/xir/instructions/phi.h>
#include <luisa/xir/module.h>
#include <luisa/xir/passes/coro_alloca_scope.h>
#include <luisa/xir/passes/coro_cfg_distill.h>
#include <luisa/xir/verifier.h>

using namespace luisa;
using namespace luisa::compute;
using namespace luisa::compute::xir;
using namespace boost::ut;
using namespace boost::ut::literals;

namespace {

[[nodiscard]] KernelFunction *make_kernel(
    Module &module, BasicBlock *&entry) noexcept {
    auto *kernel = module.create_kernel();
    entry = kernel->create_body_block();
    return kernel;
}

[[nodiscard]] bool frame_contains(
    const CoroCfgDistillResult &cfg, Value *value) noexcept {
    return std::any_of(
        cfg.frame_values.begin(), cfg.frame_values.end(),
        [value](auto &&field) noexcept { return field.value == value; });
}

[[nodiscard]] size_t instruction_index(
    BasicBlock *block, Instruction *needle) noexcept {
    auto index = size_t{0u};
    for (auto *instruction : block->instructions()) {
        if (instruction == needle) { return index; }
        ++index;
    }
    return index;
}

}// namespace

void register_coro_alloca_scope_tests() {

    "continuation_loop_begins_fresh_dynamic_scratch_lifetime"_test = [] {
        Module module;
        BasicBlock *entry;
        auto *kernel = make_kernel(module, entry);
        auto *index = kernel->create_value_argument(Type::of<uint>());
        auto *suspend = kernel->create_basic_block();
        auto *resume = kernel->create_basic_block();
        auto *array_type = Type::array(Type::of<uint>(), 4u);
        XIRBuilder builder;

        builder.set_insertion_point(entry);
        auto *scratch = builder.alloca_local(array_type);
        scratch->set_name("phase_scratch");
        builder.br(suspend);

        builder.set_insertion_point(suspend);
        builder.coro_suspend(7u, "phase", nullptr);

        builder.set_insertion_point(resume);
        auto *resume_inst = builder.coro_resume(7u, nullptr);
        auto *element = builder.gep(Type::of<uint>(), scratch, {index});
        builder.store(element, module.create_constant_one(Type::of<uint>()));
        static_cast<void>(builder.load(Type::of<uint>(), element));
        builder.br(suspend);

        expect(xir_verify_module(&module).succeeded());
        auto before = coro_cfg_distill_pass_run_on_function(kernel);
        expect(before.succeeded());
        expect(frame_contains(before, scratch));

        auto info = coro_alloca_scope_pass_run_on_function(kernel);
        expect(info.invalid_semantic_cfg_count == 0u);
        expect(info.scanned_local_alloca_count == 1u);
        expect(info.contracted_alloca_count == 1u);
        expect(info.cross_block_contraction_count == 1u);
        expect(info.intra_block_contraction_count == 0u);
        expect(info.definite_initialization_proof_count == 1u);
        expect(info.rejected_prior_lifetime_observation_count == 0u);
        expect(scratch->parent_block() == resume);
        expect(instruction_index(resume, resume_inst) <
               instruction_index(resume, scratch));
        expect(instruction_index(resume, scratch) <
               instruction_index(resume, element));
        expect(xir_verify_module(&module).succeeded());

        auto after = coro_cfg_distill_pass_run_on_function(kernel);
        expect(after.succeeded());
        expect(!frame_contains(after, scratch));
    };

    "sibling_uses_contract_to_nearest_common_dominator"_test = [] {
        Module module;
        BasicBlock *entry;
        auto *kernel = make_kernel(module, entry);
        auto *condition = kernel->create_value_argument(Type::of<bool>());
        auto *left = kernel->create_basic_block();
        auto *right = kernel->create_basic_block();
        XIRBuilder builder;

        builder.set_insertion_point(entry);
        auto *scratch = builder.alloca_local(Type::of<uint>());
        auto *unrelated = builder.clock();
        auto *branch = builder.cond_br(condition, left, right);
        builder.set_insertion_point(left);
        static_cast<void>(builder.load(Type::of<uint>(), scratch));
        builder.return_void();
        builder.set_insertion_point(right);
        static_cast<void>(builder.load(Type::of<uint>(), scratch));
        builder.return_void();

        expect(xir_verify_module(&module).succeeded());
        auto info = coro_alloca_scope_pass_run_on_function(kernel);
        expect(info.contracted_alloca_count == 1u);
        expect(info.intra_block_contraction_count == 1u);
        expect(info.cross_block_contraction_count == 0u);
        expect(scratch->parent_block() == entry);
        expect(instruction_index(entry, unrelated) <
               instruction_index(entry, scratch));
        expect(instruction_index(entry, scratch) <
               instruction_index(entry, branch));
        expect(xir_verify_module(&module).succeeded());
    };

    "loop_carried_state_prevents_lifetime_contraction"_test = [] {
        Module module;
        BasicBlock *entry;
        auto *kernel = make_kernel(module, entry);
        auto *header = kernel->create_basic_block();
        auto *initialize = kernel->create_basic_block();
        auto *consume = kernel->create_basic_block();
        XIRBuilder builder;

        builder.set_insertion_point(entry);
        auto *state = builder.alloca_local(Type::of<uint>());
        builder.br(header);

        builder.set_insertion_point(header);
        auto *first_iteration = builder.phi(
            Type::of<bool>(),
            {{module.create_constant_one(Type::of<bool>()), entry},
             {module.create_constant_zero(Type::of<bool>()), consume}});
        builder.cond_br(first_iteration, initialize, consume);

        builder.set_insertion_point(initialize);
        builder.store(
            state, module.create_constant_one(Type::of<uint>()));
        builder.br(consume);

        builder.set_insertion_point(consume);
        static_cast<void>(builder.load(Type::of<uint>(), state));
        builder.br(header);

        expect(xir_verify_module(&module).succeeded());
        auto original_index = instruction_index(entry, state);
        auto info = coro_alloca_scope_pass_run_on_function(kernel);
        expect(info.definite_initialization_proof_count == 0u);
        expect(info.rejected_prior_lifetime_observation_count == 1u);
        expect(info.contracted_alloca_count == 0u);
        expect(state->parent_block() == entry);
        expect(instruction_index(entry, state) == original_index);
        expect(xir_verify_module(&module).succeeded());
    };

    "all_paths_initialize_before_observation"_test = [] {
        Module module;
        BasicBlock *entry;
        auto *kernel = make_kernel(module, entry);
        auto *condition = kernel->create_value_argument(Type::of<bool>());
        auto *branch = kernel->create_basic_block();
        auto *left = kernel->create_basic_block();
        auto *right = kernel->create_basic_block();
        auto *join = kernel->create_basic_block();
        XIRBuilder builder;

        builder.set_insertion_point(entry);
        auto *scratch = builder.alloca_local(Type::of<uint>());
        builder.br(branch);
        builder.set_insertion_point(branch);
        auto *branch_term = builder.cond_br(condition, left, right);
        builder.set_insertion_point(left);
        builder.store(
            scratch, module.create_constant_one(Type::of<uint>()));
        builder.br(join);
        builder.set_insertion_point(right);
        builder.store(
            scratch, module.create_constant_zero(Type::of<uint>()));
        builder.br(join);
        builder.set_insertion_point(join);
        static_cast<void>(builder.load(Type::of<uint>(), scratch));
        builder.return_void();

        expect(xir_verify_module(&module).succeeded());
        auto info = coro_alloca_scope_pass_run_on_function(kernel);
        expect(info.definite_initialization_proof_count == 1u);
        expect(info.rejected_prior_lifetime_observation_count == 0u);
        expect(info.contracted_alloca_count == 1u);
        expect(info.cross_block_contraction_count == 1u);
        expect(scratch->parent_block() == branch);
        expect(instruction_index(branch, scratch) <
               instruction_index(branch, branch_term));
        expect(xir_verify_module(&module).succeeded());
    };

    "disjoint_subaggregate_stores_jointly_initialize_parent"_test = [] {
        Module module;
        BasicBlock *entry;
        auto *kernel = make_kernel(module, entry);
        auto *phase = kernel->create_basic_block();
        auto *array_type = Type::array(Type::of<uint>(), 2u);
        XIRBuilder builder;

        builder.set_insertion_point(entry);
        auto *scratch = builder.alloca_local(array_type);
        builder.br(phase);
        builder.set_insertion_point(phase);
        auto *element_0 = builder.gep(
            Type::of<uint>(), scratch,
            {module.create_constant_zero(Type::of<uint>())});
        builder.store(
            element_0, module.create_constant_zero(Type::of<uint>()));
        auto *element_1 = builder.gep(
            Type::of<uint>(), scratch,
            {module.create_constant_one(Type::of<uint>())});
        builder.store(
            element_1, module.create_constant_one(Type::of<uint>()));
        static_cast<void>(builder.load(array_type, scratch));
        builder.return_void();

        expect(xir_verify_module(&module).succeeded());
        auto info = coro_alloca_scope_pass_run_on_function(kernel);
        expect(info.definite_initialization_proof_count == 1u);
        expect(info.rejected_prior_lifetime_observation_count == 0u);
        expect(info.contracted_alloca_count == 1u);
        expect(scratch->parent_block() == phase);
        expect(xir_verify_module(&module).succeeded());
    };

    "partial_parent_initialization_prevents_contraction"_test = [] {
        Module module;
        BasicBlock *entry;
        auto *kernel = make_kernel(module, entry);
        auto *phase = kernel->create_basic_block();
        auto *array_type = Type::array(Type::of<uint>(), 2u);
        XIRBuilder builder;

        builder.set_insertion_point(entry);
        auto *scratch = builder.alloca_local(array_type);
        builder.br(phase);
        builder.set_insertion_point(phase);
        auto *element_0 = builder.gep(
            Type::of<uint>(), scratch,
            {module.create_constant_zero(Type::of<uint>())});
        builder.store(
            element_0, module.create_constant_zero(Type::of<uint>()));
        static_cast<void>(builder.load(array_type, scratch));
        builder.return_void();

        expect(xir_verify_module(&module).succeeded());
        auto info = coro_alloca_scope_pass_run_on_function(kernel);
        expect(info.definite_initialization_proof_count == 0u);
        expect(info.rejected_prior_lifetime_observation_count == 1u);
        expect(info.contracted_alloca_count == 0u);
        expect(scratch->parent_block() == entry);
        expect(xir_verify_module(&module).succeeded());
    };

    "single_full_definition_moves_with_lifetime_start"_test = [] {
        Module module;
        BasicBlock *entry;
        auto *kernel = make_kernel(module, entry);
        auto *phase = kernel->create_basic_block();
        XIRBuilder builder;

        builder.set_insertion_point(entry);
        auto *scratch = builder.alloca_local(Type::of<uint>());
        auto *definition = builder.store(
            scratch, module.create_constant_one(Type::of<uint>()));
        auto *unrelated = builder.clock();
        builder.br(phase);
        builder.set_insertion_point(phase);
        auto *observation = builder.load(Type::of<uint>(), scratch);
        builder.return_void();

        expect(xir_verify_module(&module).succeeded());
        auto info = coro_alloca_scope_pass_run_on_function(kernel);
        expect(info.delayed_first_definition_count == 1u);
        expect(info.cross_block_first_definition_delay_count == 1u);
        expect(info.intra_block_first_definition_delay_count == 0u);
        expect(info.contracted_alloca_count == 1u);
        expect(scratch->parent_block() == phase);
        expect(definition->parent_block() == phase);
        expect(unrelated->parent_block() == entry);
        expect(instruction_index(phase, scratch) <
               instruction_index(phase, definition));
        expect(instruction_index(phase, definition) <
               instruction_index(phase, observation));
        expect(xir_verify_module(&module).succeeded());
    };

    "single_definition_can_initialize_each_new_loop_lifetime"_test = [] {
        Module module;
        BasicBlock *entry;
        auto *kernel = make_kernel(module, entry);
        auto *repeat = kernel->create_value_argument(Type::of<bool>());
        auto *header = kernel->create_basic_block();
        auto *done = kernel->create_basic_block();
        XIRBuilder builder;

        builder.set_insertion_point(entry);
        auto *scratch = builder.alloca_local(Type::of<uint>());
        auto *definition = builder.store(
            scratch, module.create_constant_one(Type::of<uint>()));
        builder.br(header);
        builder.set_insertion_point(header);
        auto *unrelated = builder.clock();
        auto *observation = builder.load(Type::of<uint>(), scratch);
        builder.cond_br(repeat, header, done);
        builder.set_insertion_point(done);
        builder.return_void();

        expect(xir_verify_module(&module).succeeded());
        auto info = coro_alloca_scope_pass_run_on_function(kernel);
        expect(info.delayed_first_definition_count == 1u);
        expect(info.cross_block_first_definition_delay_count == 1u);
        expect(scratch->parent_block() == header);
        expect(definition->parent_block() == header);
        expect(instruction_index(header, unrelated) <
               instruction_index(header, scratch));
        expect(instruction_index(header, scratch) <
               instruction_index(header, definition));
        expect(instruction_index(header, definition) <
               instruction_index(header, observation));
        expect(xir_verify_module(&module).succeeded());
    };

    "multiple_definitions_are_not_first_definition_delayed"_test = [] {
        Module module;
        BasicBlock *entry;
        auto *kernel = make_kernel(module, entry);
        auto *phase = kernel->create_basic_block();
        XIRBuilder builder;

        builder.set_insertion_point(entry);
        auto *scratch = builder.alloca_local(Type::of<uint>());
        auto *unrelated = builder.clock();
        auto *first_definition = builder.store(
            scratch, module.create_constant_zero(Type::of<uint>()));
        builder.br(phase);
        builder.set_insertion_point(phase);
        auto *second_definition = builder.store(
            scratch, module.create_constant_one(Type::of<uint>()));
        static_cast<void>(builder.load(Type::of<uint>(), scratch));
        builder.return_void();

        expect(xir_verify_module(&module).succeeded());
        auto info = coro_alloca_scope_pass_run_on_function(kernel);
        expect(info.delayed_first_definition_count == 0u);
        expect(first_definition->parent_block() == entry);
        expect(second_definition->parent_block() == phase);
        expect(scratch->parent_block() == entry);
        expect(instruction_index(entry, unrelated) <
               instruction_index(entry, scratch));
        expect(instruction_index(entry, scratch) <
               instruction_index(entry, first_definition));
        expect(xir_verify_module(&module).succeeded());
    };

    "correlated_runtime_predicate_proves_guarded_initialization"_test = [] {
        Module module;
        BasicBlock *entry;
        auto *kernel = make_kernel(module, entry);
        auto *condition = kernel->create_value_argument(Type::of<bool>());
        auto *start = kernel->create_basic_block();
        auto *initialize = kernel->create_basic_block();
        auto *skip_initialize = kernel->create_basic_block();
        auto *retest = kernel->create_basic_block();
        auto *observe = kernel->create_basic_block();
        auto *done = kernel->create_basic_block();
        XIRBuilder builder;

        builder.set_insertion_point(entry);
        auto *scratch = builder.alloca_local(Type::of<uint>());
        builder.br(start);
        builder.set_insertion_point(start);
        auto *start_branch = builder.cond_br(
            condition, initialize, skip_initialize);
        builder.set_insertion_point(initialize);
        builder.store(
            scratch, module.create_constant_one(Type::of<uint>()));
        builder.br(retest);
        builder.set_insertion_point(skip_initialize);
        builder.br(retest);
        builder.set_insertion_point(retest);
        builder.cond_br(condition, observe, done);
        builder.set_insertion_point(observe);
        static_cast<void>(builder.load(Type::of<uint>(), scratch));
        builder.br(done);
        builder.set_insertion_point(done);
        builder.return_void();

        expect(xir_verify_module(&module).succeeded());
        auto info = coro_alloca_scope_pass_run_on_function(kernel);
        expect(info.definite_initialization_proof_count == 1u);
        expect(info.guarded_initialization_proof_count == 1u);
        expect(info.rejected_prior_lifetime_observation_count == 0u);
        expect(info.contracted_alloca_count == 1u);
        expect(scratch->parent_block() == start);
        expect(instruction_index(start, scratch) <
               instruction_index(start, start_branch));
        expect(xir_verify_module(&module).succeeded());
    };

    "structurally_numbered_predicates_prove_the_same_guard"_test = [] {
        Module module;
        BasicBlock *entry;
        auto *kernel = make_kernel(module, entry);
        auto *value = kernel->create_value_argument(Type::of<uint>());
        auto *limit = kernel->create_value_argument(Type::of<uint>());
        auto *start = kernel->create_basic_block();
        auto *initialize = kernel->create_basic_block();
        auto *skip_initialize = kernel->create_basic_block();
        auto *retest = kernel->create_basic_block();
        auto *observe = kernel->create_basic_block();
        auto *done = kernel->create_basic_block();
        XIRBuilder builder;

        builder.set_insertion_point(entry);
        auto *scratch = builder.alloca_local(Type::of<uint>());
        builder.br(start);
        builder.set_insertion_point(start);
        auto *first_test = builder.call(
            Type::of<bool>(), ArithmeticOp::BINARY_LESS,
            {value, limit});
        builder.cond_br(first_test, initialize, skip_initialize);
        builder.set_insertion_point(initialize);
        builder.store(
            scratch, module.create_constant_one(Type::of<uint>()));
        builder.br(retest);
        builder.set_insertion_point(skip_initialize);
        builder.br(retest);
        builder.set_insertion_point(retest);
        auto *second_test = builder.call(
            Type::of<bool>(), ArithmeticOp::BINARY_LESS,
            {value, limit});
        builder.cond_br(second_test, observe, done);
        builder.set_insertion_point(observe);
        static_cast<void>(builder.load(Type::of<uint>(), scratch));
        builder.br(done);
        builder.set_insertion_point(done);
        builder.return_void();

        expect(first_test != second_test);
        expect(xir_verify_module(&module).succeeded());
        auto info = coro_alloca_scope_pass_run_on_function(kernel);
        expect(info.guarded_initialization_proof_count == 1u);
        expect(info.contracted_alloca_count == 1u);
        expect(scratch->parent_block() == start);
        expect(xir_verify_module(&module).succeeded());
    };

    "unrelated_runtime_predicate_cannot_prove_initialization"_test = [] {
        Module module;
        BasicBlock *entry;
        auto *kernel = make_kernel(module, entry);
        auto *initialize_condition =
            kernel->create_value_argument(Type::of<bool>());
        auto *observe_condition =
            kernel->create_value_argument(Type::of<bool>());
        auto *start = kernel->create_basic_block();
        auto *initialize = kernel->create_basic_block();
        auto *skip_initialize = kernel->create_basic_block();
        auto *retest = kernel->create_basic_block();
        auto *observe = kernel->create_basic_block();
        auto *done = kernel->create_basic_block();
        XIRBuilder builder;

        builder.set_insertion_point(entry);
        auto *scratch = builder.alloca_local(Type::of<uint>());
        builder.br(start);
        builder.set_insertion_point(start);
        builder.cond_br(
            initialize_condition, initialize, skip_initialize);
        builder.set_insertion_point(initialize);
        builder.store(
            scratch, module.create_constant_one(Type::of<uint>()));
        builder.br(retest);
        builder.set_insertion_point(skip_initialize);
        builder.br(retest);
        builder.set_insertion_point(retest);
        builder.cond_br(observe_condition, observe, done);
        builder.set_insertion_point(observe);
        static_cast<void>(builder.load(Type::of<uint>(), scratch));
        builder.br(done);
        builder.set_insertion_point(done);
        builder.return_void();

        expect(xir_verify_module(&module).succeeded());
        auto info = coro_alloca_scope_pass_run_on_function(kernel);
        expect(info.definite_initialization_proof_count == 0u);
        expect(info.guarded_initialization_proof_count == 0u);
        expect(info.rejected_prior_lifetime_observation_count == 1u);
        expect(info.contracted_alloca_count == 0u);
        expect(scratch->parent_block() == entry);
        expect(xir_verify_module(&module).succeeded());
    };

    "opposite_runtime_predicate_cannot_prove_initialization"_test = [] {
        Module module;
        BasicBlock *entry;
        auto *kernel = make_kernel(module, entry);
        auto *condition = kernel->create_value_argument(Type::of<bool>());
        auto *start = kernel->create_basic_block();
        auto *initialize = kernel->create_basic_block();
        auto *skip_initialize = kernel->create_basic_block();
        auto *retest = kernel->create_basic_block();
        auto *observe = kernel->create_basic_block();
        auto *done = kernel->create_basic_block();
        XIRBuilder builder;

        builder.set_insertion_point(entry);
        auto *scratch = builder.alloca_local(Type::of<uint>());
        builder.br(start);
        builder.set_insertion_point(start);
        builder.cond_br(condition, initialize, skip_initialize);
        builder.set_insertion_point(initialize);
        builder.store(
            scratch, module.create_constant_one(Type::of<uint>()));
        builder.br(retest);
        builder.set_insertion_point(skip_initialize);
        builder.br(retest);
        builder.set_insertion_point(retest);
        builder.cond_br(condition, done, observe);
        builder.set_insertion_point(observe);
        static_cast<void>(builder.load(Type::of<uint>(), scratch));
        builder.br(done);
        builder.set_insertion_point(done);
        builder.return_void();

        expect(xir_verify_module(&module).succeeded());
        auto info = coro_alloca_scope_pass_run_on_function(kernel);
        expect(info.definite_initialization_proof_count == 0u);
        expect(info.guarded_initialization_proof_count == 0u);
        expect(info.rejected_prior_lifetime_observation_count == 1u);
        expect(info.contracted_alloca_count == 0u);
        expect(scratch->parent_block() == entry);
        expect(xir_verify_module(&module).succeeded());
    };

    "phi_pointer_use_is_an_atomic_noop"_test = [] {
        Module module;
        BasicBlock *entry;
        auto *kernel = make_kernel(module, entry);
        auto *join = kernel->create_basic_block();
        XIRBuilder builder;
        builder.set_insertion_point(entry);
        auto *scratch = builder.alloca_local(Type::of<uint>());
        auto *clock = builder.clock();
        builder.br(join);
        builder.set_insertion_point(join);
        static_cast<void>(builder.phi(
            Type::of<uint>(), {{scratch, entry}}));
        builder.return_void();

        auto original_index = instruction_index(entry, scratch);
        auto info = coro_alloca_scope_pass_run_on_function(kernel);
        expect(info.rejected_phi_use_count == 1u);
        expect(info.contracted_alloca_count == 0u);
        expect(scratch->parent_block() == entry);
        expect(instruction_index(entry, scratch) == original_index);
        expect(instruction_index(entry, scratch) <
               instruction_index(entry, clock));
    };

    "invalid_suspend_resume_pair_is_an_atomic_noop"_test = [] {
        Module module;
        BasicBlock *entry;
        auto *kernel = make_kernel(module, entry);
        auto *resume_a = kernel->create_basic_block();
        auto *resume_b = kernel->create_basic_block();
        XIRBuilder builder;
        builder.set_insertion_point(entry);
        auto *scratch = builder.alloca_local(Type::of<uint>());
        builder.coro_suspend(17u, "ambiguous", nullptr);
        builder.set_insertion_point(resume_a);
        builder.coro_resume(17u, nullptr);
        static_cast<void>(builder.load(Type::of<uint>(), scratch));
        builder.return_void();
        builder.set_insertion_point(resume_b);
        builder.coro_resume(17u, nullptr);
        builder.return_void();

        auto info = coro_alloca_scope_pass_run_on_function(kernel);
        expect(info.invalid_semantic_cfg_count == 1u);
        expect(info.contracted_alloca_count == 0u);
        expect(scratch->parent_block() == entry);
    };

    "null_and_declaration_are_noops"_test = [] {
        expect(!coro_alloca_scope_pass_run_on_function(nullptr).changed());
        Module module;
        auto *callable = module.create_callable(nullptr);
        expect(!coro_alloca_scope_pass_run_on_function(callable).changed());
    };
}

int main() {
    register_coro_alloca_scope_tests();
    return 0;
}
