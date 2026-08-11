// Tests for coroutine-semantic local-state rematerialization.

#include "ut/ut.hpp"

#include <luisa/xir/basic_block.h>
#include <luisa/xir/builder.h>
#include <luisa/xir/function.h>
#include <luisa/xir/instructions/alloca.h>
#include <luisa/xir/instructions/arithmetic.h>
#include <luisa/xir/instructions/branch.h>
#include <luisa/xir/instructions/coro.h>
#include <luisa/xir/instructions/gep.h>
#include <luisa/xir/instructions/load.h>
#include <luisa/xir/module.h>
#include <luisa/xir/passes/coro_cfg_distill.h>
#include <luisa/xir/passes/coro_rematerialize.h>
#include <luisa/xir/passes/dce.h>
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

[[nodiscard]] size_t count_loads_from(
    FunctionDefinition *definition, Value *root) noexcept {
    auto count = size_t{0u};
    for (auto *block : definition->basic_blocks()) {
        for (auto *instruction : block->instructions()) {
            if (!instruction->isa<LoadInst>()) { continue; }
            auto *pointer = static_cast<LoadInst *>(instruction)->variable();
            while (pointer != nullptr && pointer->isa<GEPInst>()) {
                pointer = static_cast<GEPInst *>(pointer)->base();
            }
            if (pointer == root) { ++count; }
        }
    }
    return count;
}

}// namespace

void register_coro_rematerialize_tests() {

    "unique_constant_store_dominates_across_suspend"_test = [] {
        Module module;
        BasicBlock *entry;
        auto *kernel = make_kernel(module, entry);
        auto *resume = kernel->create_basic_block();
        XIRBuilder builder;
        builder.set_insertion_point(entry);
        auto *state = builder.alloca_local(Type::of<float>());
        state->set_name("immutable_state");
        builder.store(state, module.create_constant_one(Type::of<float>()));
        builder.coro_suspend(11u, "state", nullptr);
        builder.set_insertion_point(resume);
        builder.coro_resume(11u, nullptr);
        static_cast<void>(builder.load(Type::of<float>(), state));
        builder.return_void();

        expect(xir_verify_module(&module).succeeded());
        auto info =
            coro_rematerialize_local_state_pass_run_on_function(kernel);

        expect(info.invalid_semantic_cfg_count == 0u);
        expect(info.promoted_alloca_count == 1u);
        expect(info.replaced_load_count == 1u);
        expect(info.inserted_extract_count == 0u);
        expect(info.initializer_replay_instruction_cost == 0u);
        expect(count_loads_from(kernel, state) == 0u);
        expect(xir_verify_module(&module).succeeded());
        static_cast<void>(dce_pass_run_on_function(kernel));
        auto cfg = coro_cfg_distill_pass_run_on_function(kernel);
        expect(cfg.succeeded());
        expect(std::none_of(
            cfg.frame_values.begin(), cfg.frame_values.end(),
            [&](const auto &field) noexcept {
                return field.value == state;
            }));
    };

    "conditional_store_does_not_dominate_resume"_test = [] {
        Module module;
        BasicBlock *entry;
        auto *kernel = make_kernel(module, entry);
        auto *condition = kernel->create_argument(
            Type::of<bool>(), false);
        auto *store_block = kernel->create_basic_block();
        auto *skip_block = kernel->create_basic_block();
        auto *suspend = kernel->create_basic_block();
        auto *resume = kernel->create_basic_block();
        XIRBuilder builder;
        builder.set_insertion_point(entry);
        auto *state = builder.alloca_local(Type::of<float>());
        builder.cond_br(condition, store_block, skip_block);
        builder.set_insertion_point(store_block);
        builder.store(state, module.create_constant_one(Type::of<float>()));
        builder.br(suspend);
        builder.set_insertion_point(skip_block);
        builder.br(suspend);
        builder.set_insertion_point(suspend);
        builder.coro_suspend(13u, "conditional", nullptr);
        builder.set_insertion_point(resume);
        builder.coro_resume(13u, nullptr);
        static_cast<void>(builder.load(Type::of<float>(), state));
        builder.return_void();

        expect(xir_verify_module(&module).succeeded());
        auto info =
            coro_rematerialize_local_state_pass_run_on_function(kernel);

        expect(info.replayable_single_store_count == 1u);
        expect(info.promoted_alloca_count == 0u);
        expect(info.replaced_load_count == 0u);
        expect(count_loads_from(kernel, state) == 1u);
        expect(xir_verify_module(&module).succeeded());
    };

    "same_block_store_after_load_does_not_dominate"_test = [] {
        Module module;
        BasicBlock *entry;
        auto *kernel = make_kernel(module, entry);
        XIRBuilder builder;
        builder.set_insertion_point(entry);
        auto *state = builder.alloca_local(Type::of<float>());
        static_cast<void>(builder.load(Type::of<float>(), state));
        builder.store(state, module.create_constant_one(Type::of<float>()));
        builder.return_void();

        expect(xir_verify_module(&module).succeeded());
        auto info =
            coro_rematerialize_local_state_pass_run_on_function(kernel);

        expect(info.replayable_single_store_count == 1u);
        expect(info.promoted_alloca_count == 0u);
        expect(info.replaced_load_count == 0u);
        expect(count_loads_from(kernel, state) == 1u);
        expect(xir_verify_module(&module).succeeded());
    };

    "sequential_replayable_phases_are_rematerialized"_test = [] {
        Module module;
        BasicBlock *entry;
        auto *kernel = make_kernel(module, entry);
        auto *resume_first = kernel->create_basic_block();
        auto *resume_second = kernel->create_basic_block();
        XIRBuilder builder;
        builder.set_insertion_point(entry);
        auto *state = builder.alloca_local(Type::of<uint>());
        builder.store(state, module.create_constant_zero(Type::of<uint>()));
        builder.coro_suspend(17u, "first", nullptr);
        builder.set_insertion_point(resume_first);
        builder.coro_resume(17u, nullptr);
        static_cast<void>(builder.load(Type::of<uint>(), state));
        builder.store(state, module.create_constant_one(Type::of<uint>()));
        builder.coro_suspend(19u, "second", nullptr);
        builder.set_insertion_point(resume_second);
        builder.coro_resume(19u, nullptr);
        static_cast<void>(builder.load(Type::of<uint>(), state));
        builder.return_void();

        auto info =
            coro_rematerialize_local_state_pass_run_on_function(kernel);

        expect(info.replayable_multi_store_count == 1u);
        expect(info.promoted_multi_store_alloca_count == 1u);
        expect(info.promoted_alloca_count == 1u);
        expect(info.replaced_load_count == 2u);
        expect(count_loads_from(kernel, state) == 0u);
        expect(xir_verify_module(&module).succeeded());
    };

    "branch_stores_of_one_exact_value_are_rematerialized"_test = [] {
        Module module;
        BasicBlock *entry;
        auto *kernel = make_kernel(module, entry);
        auto *condition = kernel->create_argument(
            Type::of<bool>(), false);
        auto *left = kernel->create_basic_block();
        auto *right = kernel->create_basic_block();
        auto *suspend = kernel->create_basic_block();
        auto *resume = kernel->create_basic_block();
        auto *zero = module.create_constant_zero(Type::of<uint>());
        XIRBuilder builder;
        builder.set_insertion_point(entry);
        auto *state = builder.alloca_local(Type::of<uint>());
        builder.cond_br(condition, left, right);
        builder.set_insertion_point(left);
        builder.store(state, zero);
        builder.br(suspend);
        builder.set_insertion_point(right);
        builder.store(state, zero);
        builder.br(suspend);
        builder.set_insertion_point(suspend);
        builder.coro_suspend(21u, "uniform_join", nullptr);
        builder.set_insertion_point(resume);
        builder.coro_resume(21u, nullptr);
        static_cast<void>(builder.load(Type::of<uint>(), state));
        builder.return_void();

        expect(xir_verify_module(&module).succeeded());
        auto info =
            coro_rematerialize_local_state_pass_run_on_function(kernel);

        expect(info.replayable_multi_store_count == 1u);
        expect(info.promoted_multi_store_alloca_count == 1u);
        expect(info.replaced_load_count == 1u);
        expect(count_loads_from(kernel, state) == 0u);
        expect(xir_verify_module(&module).succeeded());
    };

    "branch_stores_of_distinct_values_conflict"_test = [] {
        Module module;
        BasicBlock *entry;
        auto *kernel = make_kernel(module, entry);
        auto *condition = kernel->create_argument(
            Type::of<bool>(), false);
        auto *left = kernel->create_basic_block();
        auto *right = kernel->create_basic_block();
        auto *suspend = kernel->create_basic_block();
        auto *resume = kernel->create_basic_block();
        XIRBuilder builder;
        builder.set_insertion_point(entry);
        auto *state = builder.alloca_local(Type::of<uint>());
        builder.cond_br(condition, left, right);
        builder.set_insertion_point(left);
        builder.store(
            state, module.create_constant_zero(Type::of<uint>()));
        builder.br(suspend);
        builder.set_insertion_point(right);
        builder.store(
            state, module.create_constant_one(Type::of<uint>()));
        builder.br(suspend);
        builder.set_insertion_point(suspend);
        builder.coro_suspend(25u, "conflicting_join", nullptr);
        builder.set_insertion_point(resume);
        builder.coro_resume(25u, nullptr);
        static_cast<void>(builder.load(Type::of<uint>(), state));
        builder.return_void();

        expect(xir_verify_module(&module).succeeded());
        auto info =
            coro_rematerialize_local_state_pass_run_on_function(kernel);

        expect(info.replayable_multi_store_count == 1u);
        expect(info.unresolved_load_count == 1u);
        expect(info.promoted_alloca_count == 0u);
        expect(info.replaced_load_count == 0u);
        expect(count_loads_from(kernel, state) == 1u);
        expect(xir_verify_module(&module).succeeded());
    };

    "loop_backedge_conflicts_with_entry_definition"_test = [] {
        Module module;
        BasicBlock *entry;
        auto *kernel = make_kernel(module, entry);
        auto *condition = kernel->create_argument(
            Type::of<bool>(), false);
        auto *header = kernel->create_basic_block();
        auto *body = kernel->create_basic_block();
        auto *suspend = kernel->create_basic_block();
        auto *resume = kernel->create_basic_block();
        XIRBuilder builder;
        builder.set_insertion_point(entry);
        auto *state = builder.alloca_local(Type::of<uint>());
        builder.store(
            state, module.create_constant_zero(Type::of<uint>()));
        builder.br(header);
        builder.set_insertion_point(header);
        builder.cond_br(condition, body, suspend);
        builder.set_insertion_point(body);
        builder.store(
            state, module.create_constant_one(Type::of<uint>()));
        builder.br(header);
        builder.set_insertion_point(suspend);
        builder.coro_suspend(27u, "loop_carried", nullptr);
        builder.set_insertion_point(resume);
        builder.coro_resume(27u, nullptr);
        static_cast<void>(builder.load(Type::of<uint>(), state));
        builder.return_void();

        expect(xir_verify_module(&module).succeeded());
        auto info =
            coro_rematerialize_local_state_pass_run_on_function(kernel);

        expect(info.replayable_multi_store_count == 1u);
        expect(info.unresolved_load_count == 1u);
        expect(info.promoted_alloca_count == 0u);
        expect(count_loads_from(kernel, state) == 1u);
        expect(xir_verify_module(&module).succeeded());
    };

    "partial_store_is_a_conservative_barrier"_test = [] {
        Module module;
        BasicBlock *entry;
        auto *kernel = make_kernel(module, entry);
        auto *resume = kernel->create_basic_block();
        auto *pair = Type::array(Type::of<float>(), 2u);
        auto *index = module.create_constant_zero(Type::of<uint32_t>());
        XIRBuilder builder;
        builder.set_insertion_point(entry);
        auto *state = builder.alloca_local(pair);
        builder.store(state, module.create_constant_zero(pair));
        auto *field = builder.gep(Type::of<float>(), state, {index});
        builder.store(
            field, module.create_constant_one(Type::of<float>()));
        builder.coro_suspend(28u, "partial_store", nullptr);
        builder.set_insertion_point(resume);
        builder.coro_resume(28u, nullptr);
        static_cast<void>(builder.load(pair, state));
        builder.return_void();

        expect(xir_verify_module(&module).succeeded());
        auto info =
            coro_rematerialize_local_state_pass_run_on_function(kernel);

        expect(info.replayable_single_store_count == 0u);
        expect(info.replayable_multi_store_count == 0u);
        expect(info.promoted_alloca_count == 0u);
        expect(count_loads_from(kernel, state) == 1u);
        expect(xir_verify_module(&module).succeeded());
    };

    "clock_initialized_state_is_forwarded_without_replay"_test = [] {
        Module module;
        BasicBlock *entry;
        auto *kernel = make_kernel(module, entry);
        auto *output = kernel->create_reference_argument(
            Type::of<uint64_t>());
        auto *resume = kernel->create_basic_block();
        XIRBuilder builder;
        builder.set_insertion_point(entry);
        auto *state = builder.alloca_local(Type::of<uint64_t>());
        auto *clock = builder.clock();
        builder.store(state, clock);
        builder.coro_suspend(23u, "clock", nullptr);
        builder.set_insertion_point(resume);
        builder.coro_resume(23u, nullptr);
        auto *loaded = builder.load(Type::of<uint64_t>(), state);
        auto *one = module.create_constant_one(Type::of<uint64_t>());
        auto *use = builder.call(
            Type::of<uint64_t>(), ArithmeticOp::BINARY_ADD,
            {loaded, one});
        builder.store(output, use);
        builder.return_void();

        auto info =
            coro_rematerialize_local_state_pass_run_on_function(kernel);

        expect(info.replayable_single_store_count == 0u);
        expect(info.nonreplayable_candidate_count == 1u);
        expect(info.promoted_nonreplayable_alloca_count == 1u);
        expect(info.promoted_alloca_count == 1u);
        expect(info.replaced_load_count == 1u);
        expect(count_loads_from(kernel, state) == 0u);
        // Forward the one dynamic clock result itself; cloning/replaying the
        // impure instruction would change program semantics.
        expect(use->operand(0) == clock);
        expect(xir_verify_module(&module).succeeded());
        static_cast<void>(dce_pass_run_on_function(kernel));
        auto cfg = coro_cfg_distill_pass_run_on_function(kernel);
        expect(cfg.succeeded());
        expect(std::any_of(
            cfg.frame_values.begin(), cfg.frame_values.end(),
            [&](const auto &field) noexcept {
                return field.value == clock;
            }));
        expect(std::none_of(
            cfg.frame_values.begin(), cfg.frame_values.end(),
            [&](const auto &field) noexcept {
                return field.value == state;
            }));
    };

    "scope_local_nonreplayable_state_is_retained"_test = [] {
        Module module;
        BasicBlock *entry;
        auto *kernel = make_kernel(module, entry);
        auto *output = kernel->create_reference_argument(
            Type::of<uint64_t>());
        XIRBuilder builder;
        builder.set_insertion_point(entry);
        auto *state = builder.alloca_local(Type::of<uint64_t>());
        builder.store(state, builder.clock());
        auto *loaded = builder.load(Type::of<uint64_t>(), state);
        builder.store(output, loaded);
        builder.return_void();

        auto info =
            coro_rematerialize_local_state_pass_run_on_function(kernel);

        expect(info.nonreplayable_candidate_count == 1u);
        expect(info.rejected_nonreplayable_scope_local_count == 1u);
        expect(info.promoted_nonreplayable_alloca_count == 0u);
        expect(info.promoted_alloca_count == 0u);
        expect(count_loads_from(kernel, state) == 1u);
        expect(xir_verify_module(&module).succeeded());
    };

    "loop_reexecuted_store_resets_cross_suspend_state"_test = [] {
        Module module;
        BasicBlock *entry;
        auto *kernel = make_kernel(module, entry);
        auto *output = kernel->create_reference_argument(
            Type::of<uint64_t>());
        auto *initial_resume = kernel->create_basic_block();
        auto *body = kernel->create_basic_block();
        auto *backedge_resume = kernel->create_basic_block();
        XIRBuilder builder;
        builder.set_insertion_point(entry);
        builder.coro_suspend(30u, "enter_loop", nullptr);
        builder.set_insertion_point(initial_resume);
        builder.coro_resume(30u, nullptr);
        builder.br(body);
        builder.set_insertion_point(body);
        auto *state = builder.alloca_local(Type::of<uint64_t>());
        builder.store(state, builder.clock());
        auto *loaded = builder.load(Type::of<uint64_t>(), state);
        builder.store(output, loaded);
        builder.coro_suspend(31u, "loop_backedge", nullptr);
        builder.set_insertion_point(backedge_resume);
        builder.coro_resume(31u, nullptr);
        builder.br(body);

        auto info =
            coro_rematerialize_local_state_pass_run_on_function(kernel);

        expect(info.nonreplayable_candidate_count == 1u);
        expect(info.rejected_nonreplayable_scope_local_count == 1u);
        expect(info.promoted_nonreplayable_alloca_count == 0u);
        expect(info.promoted_alloca_count == 0u);
        expect(count_loads_from(kernel, state) == 1u);
        expect(xir_verify_module(&module).succeeded());
    };

    "phased_nonreplayable_state_tracks_each_suspend_edge"_test = [] {
        Module module;
        BasicBlock *entry;
        auto *kernel = make_kernel(module, entry);
        auto *first_output = kernel->create_reference_argument(
            Type::of<uint64_t>());
        auto *second_output = kernel->create_reference_argument(
            Type::of<uint64_t>());
        auto *first_resume = kernel->create_basic_block();
        auto *second_resume = kernel->create_basic_block();
        XIRBuilder builder;
        builder.set_insertion_point(entry);
        auto *state = builder.alloca_local(Type::of<uint64_t>());
        auto *first_clock = builder.clock();
        builder.store(state, first_clock);
        builder.coro_suspend(26u, "first_clock", nullptr);
        builder.set_insertion_point(first_resume);
        builder.coro_resume(26u, nullptr);
        auto *first = builder.load(Type::of<uint64_t>(), state);
        auto *write_first = builder.store(first_output, first);
        auto *second_clock = builder.clock();
        builder.store(state, second_clock);
        builder.coro_suspend(27u, "second_clock", nullptr);
        builder.set_insertion_point(second_resume);
        builder.coro_resume(27u, nullptr);
        auto *second = builder.load(Type::of<uint64_t>(), state);
        auto *write_second = builder.store(second_output, second);
        builder.return_void();

        auto info =
            coro_rematerialize_local_state_pass_run_on_function(kernel);

        expect(info.nonreplayable_candidate_count == 1u);
        expect(info.reaching_dataflow_alloca_count == 1u);
        expect(info.promoted_nonreplayable_alloca_count == 1u);
        expect(info.promoted_multi_store_alloca_count == 1u);
        expect(info.replaced_load_count == 2u);
        expect(write_first->value() == first_clock);
        expect(write_second->value() == second_clock);
        expect(count_loads_from(kernel, state) == 0u);
        expect(xir_verify_module(&module).succeeded());
    };

    "projected_nonreplayable_state_is_retained"_test = [] {
        Module module;
        BasicBlock *entry;
        auto *kernel = make_kernel(module, entry);
        auto *resume = kernel->create_basic_block();
        auto *pair = Type::array(Type::of<uint64_t>(), 2u);
        auto *index = module.create_constant_zero(Type::of<uint32_t>());
        XIRBuilder builder;
        builder.set_insertion_point(entry);
        auto *state = builder.alloca_local(pair);
        auto *clock = builder.clock();
        auto *zero = module.create_constant_zero(Type::of<uint64_t>());
        auto *value = builder.call(
            pair, ArithmeticOp::AGGREGATE,
            std::array<Value *, 2u>{clock, zero});
        builder.store(state, value);
        builder.coro_suspend(24u, "projected_clock", nullptr);
        builder.set_insertion_point(resume);
        builder.coro_resume(24u, nullptr);
        auto *field = builder.gep(Type::of<uint64_t>(), state, {index});
        static_cast<void>(builder.load(Type::of<uint64_t>(), field));
        builder.return_void();

        auto info =
            coro_rematerialize_local_state_pass_run_on_function(kernel);

        expect(info.replayable_single_store_count == 0u);
        expect(info.nonreplayable_candidate_count == 1u);
        expect(info.rejected_nonreplayable_projection_count == 1u);
        expect(info.promoted_alloca_count == 0u);
        expect(count_loads_from(kernel, state) == 1u);
        expect(xir_verify_module(&module).succeeded());
    };

    "cross_local_nonreplayable_copy_chain_is_rewritten_atomically"_test = [] {
        Module module;
        BasicBlock *entry;
        auto *kernel = make_kernel(module, entry);
        auto *output = kernel->create_reference_argument(
            Type::of<uint64_t>());
        auto *resume = kernel->create_basic_block();
        XIRBuilder builder;
        builder.set_insertion_point(entry);
        // Deliberately create the producer first. Candidate discovery then
        // visits its load before the consumer load whose reaching value is the
        // producer load; candidate-order rewriting would leave a dangling
        // reference after detaching the producer.
        auto *producer = builder.alloca_local(Type::of<uint64_t>());
        auto *consumer = builder.alloca_local(Type::of<uint64_t>());
        auto *clock = builder.clock();
        builder.store(producer, clock);
        auto *snapshot = builder.load(Type::of<uint64_t>(), producer);
        builder.store(consumer, snapshot);
        builder.coro_suspend(25u, "copy_chain", nullptr);
        builder.set_insertion_point(resume);
        builder.coro_resume(25u, nullptr);
        auto *loaded = builder.load(Type::of<uint64_t>(), consumer);
        auto *producer_after_suspend = builder.load(
            Type::of<uint64_t>(), producer);
        auto *sum = builder.call(
            Type::of<uint64_t>(), ArithmeticOp::BINARY_ADD,
            {loaded, producer_after_suspend});
        builder.store(output, sum);
        builder.return_void();

        auto info =
            coro_rematerialize_local_state_pass_run_on_function(kernel);

        expect(info.rejected_forwarding_cycle_count == 0u);
        expect(info.promoted_nonreplayable_alloca_count == 2u);
        expect(info.promoted_alloca_count == 2u);
        expect(info.replaced_load_count == 3u);
        expect(count_loads_from(kernel, producer) == 0u);
        expect(count_loads_from(kernel, consumer) == 0u);
        expect(xir_verify_module(&module).succeeded());
        static_cast<void>(dce_pass_run_on_function(kernel));
        expect(xir_verify_module(&module).succeeded());
        auto cfg = coro_cfg_distill_pass_run_on_function(kernel);
        expect(cfg.succeeded());
        expect(std::any_of(
            cfg.frame_values.begin(), cfg.frame_values.end(),
            [&](const auto &field) noexcept {
                return field.value == clock;
            }));
        expect(std::none_of(
            cfg.frame_values.begin(), cfg.frame_values.end(),
            [&](const auto &field) noexcept {
                return field.value == producer ||
                       field.value == consumer ||
                       field.value == snapshot ||
                       field.value == loaded ||
                       field.value == producer_after_suspend;
            }));
    };

    "aggregate_gep_load_becomes_snapshot_extract"_test = [] {
        Module module;
        BasicBlock *entry;
        auto *kernel = make_kernel(module, entry);
        auto *resume = kernel->create_basic_block();
        auto *pair = Type::array(Type::of<float>(), 2u);
        uint32_t second_index = 1u;
        auto *second = module.create_constant(
            Type::of<uint32_t>(), &second_index);
        XIRBuilder builder;
        builder.set_insertion_point(entry);
        auto *state = builder.alloca_local(pair);
        builder.store(state, module.create_constant_zero(pair));
        builder.coro_suspend(29u, "aggregate", nullptr);
        builder.set_insertion_point(resume);
        builder.coro_resume(29u, nullptr);
        auto *pointer = builder.gep(Type::of<float>(), state, {second});
        static_cast<void>(builder.load(Type::of<float>(), pointer));
        builder.return_void();

        auto info =
            coro_rematerialize_local_state_pass_run_on_function(kernel);

        expect(info.promoted_alloca_count == 1u);
        expect(info.replaced_load_count == 1u);
        expect(info.inserted_extract_count == 1u);
        expect(count_loads_from(kernel, state) == 0u);
        auto extract_count = size_t{0u};
        for (auto *block : kernel->basic_blocks()) {
            for (auto *instruction : block->instructions()) {
                if (instruction->isa<ArithmeticInst>() &&
                    static_cast<ArithmeticInst *>(instruction)->op() ==
                        ArithmeticOp::EXTRACT) {
                    ++extract_count;
                }
            }
        }
        expect(extract_count == 1u);
        expect(xir_verify_module(&module).succeeded());
    };

    "semantic_dominance_converges_on_loop_backedge"_test = [] {
        Module module;
        BasicBlock *entry;
        auto *kernel = make_kernel(module, entry);
        auto *condition = kernel->create_argument(
            Type::of<bool>(), false);
        auto *header = kernel->create_basic_block();
        auto *body = kernel->create_basic_block();
        auto *suspend = kernel->create_basic_block();
        auto *resume = kernel->create_basic_block();
        XIRBuilder builder;
        builder.set_insertion_point(entry);
        auto *state = builder.alloca_local(Type::of<float>());
        builder.store(state, module.create_constant_one(Type::of<float>()));
        builder.br(header);
        builder.set_insertion_point(header);
        builder.cond_br(condition, body, suspend);
        builder.set_insertion_point(body);
        builder.br(header);
        builder.set_insertion_point(suspend);
        builder.coro_suspend(37u, "loop", nullptr);
        builder.set_insertion_point(resume);
        builder.coro_resume(37u, nullptr);
        static_cast<void>(builder.load(Type::of<float>(), state));
        builder.return_void();

        expect(xir_verify_module(&module).succeeded());
        auto info =
            coro_rematerialize_local_state_pass_run_on_function(kernel);

        expect(info.invalid_semantic_cfg_count == 0u);
        expect(info.promoted_alloca_count == 1u);
        expect(info.replaced_load_count == 1u);
        expect(count_loads_from(kernel, state) == 0u);
        expect(xir_verify_module(&module).succeeded());
    };

    "projected_expression_must_fit_result_replay_budget"_test = [] {
        Module module;
        BasicBlock *entry;
        auto *kernel = make_kernel(module, entry);
        auto *resume = kernel->create_basic_block();
        auto *pair = Type::array(Type::of<float>(), 2u);
        auto *zero_index =
            module.create_constant_zero(Type::of<uint32_t>());
        auto *one_index =
            module.create_constant_one(Type::of<uint32_t>());
        auto *one = module.create_constant_one(Type::of<float>());
        XIRBuilder builder;
        builder.set_insertion_point(entry);
        auto *state = builder.alloca_local(pair);
        auto *first_insert = builder.call(
            pair, ArithmeticOp::INSERT,
            {module.create_constant_zero(pair), one, zero_index});
        auto *second_insert = builder.call(
            pair, ArithmeticOp::INSERT,
            {first_insert, one, one_index});
        builder.store(state, second_insert);
        builder.coro_suspend(41u, "projection_budget", nullptr);
        builder.set_insertion_point(resume);
        builder.coro_resume(41u, nullptr);
        static_cast<void>(builder.load(pair, state));
        auto *pointer = builder.gep(
            Type::of<float>(), state, {zero_index});
        static_cast<void>(builder.load(Type::of<float>(), pointer));
        builder.return_void();

        expect(xir_verify_module(&module).succeeded());
        auto info =
            coro_rematerialize_local_state_pass_run_on_function(kernel);

        // The two INSERTs fit the 4-op aggregate budget, but adding EXTRACT
        // exceeds the projected float's 2-op budget. Retaining local storage
        // is therefore the only representation proven profitable here.
        expect(info.replayable_single_store_count == 1u);
        expect(info.rejected_projected_replay_cost_count == 1u);
        expect(info.promoted_alloca_count == 0u);
        expect(info.replaced_load_count == 0u);
        // The cheap whole-object load is retained too: candidate analysis is
        // atomic rather than committing loads one at a time.
        expect(count_loads_from(kernel, state) == 2u);
        expect(xir_verify_module(&module).succeeded());
    };

    "unmatched_suspend_resume_graph_is_atomic_noop"_test = [] {
        Module module;
        BasicBlock *entry;
        auto *kernel = make_kernel(module, entry);
        XIRBuilder builder;
        builder.set_insertion_point(entry);
        auto *state = builder.alloca_local(Type::of<float>());
        builder.store(state, module.create_constant_one(Type::of<float>()));
        builder.coro_suspend(31u, "unmatched", nullptr);

        auto info =
            coro_rematerialize_local_state_pass_run_on_function(kernel);

        expect(info.invalid_semantic_cfg_count == 1u);
        expect(info.promoted_alloca_count == 0u);
        expect(info.replaced_load_count == 0u);
    };

    "duplicate_resume_token_graph_is_atomic_noop"_test = [] {
        Module module;
        BasicBlock *entry;
        auto *kernel = make_kernel(module, entry);
        auto *first_resume = kernel->create_basic_block();
        auto *second_resume = kernel->create_basic_block();
        XIRBuilder builder;
        builder.set_insertion_point(entry);
        auto *state = builder.alloca_local(Type::of<float>());
        builder.store(state, module.create_constant_one(Type::of<float>()));
        builder.coro_suspend(43u, "ambiguous", nullptr);
        builder.set_insertion_point(first_resume);
        builder.coro_resume(43u, nullptr);
        static_cast<void>(builder.load(Type::of<float>(), state));
        builder.return_void();
        builder.set_insertion_point(second_resume);
        builder.coro_resume(43u, nullptr);
        builder.return_void();

        auto info =
            coro_rematerialize_local_state_pass_run_on_function(kernel);

        expect(info.invalid_semantic_cfg_count == 1u);
        expect(info.promoted_alloca_count == 0u);
        expect(info.replaced_load_count == 0u);
        expect(count_loads_from(kernel, state) == 1u);
    };

    "unterminated_owned_block_is_atomic_noop"_test = [] {
        Module module;
        BasicBlock *entry;
        auto *kernel = make_kernel(module, entry);
        XIRBuilder builder;
        builder.set_insertion_point(entry);
        auto *state = builder.alloca_local(Type::of<float>());
        builder.store(state, module.create_constant_one(Type::of<float>()));
        static_cast<void>(builder.load(Type::of<float>(), state));

        auto info =
            coro_rematerialize_local_state_pass_run_on_function(kernel);

        expect(info.invalid_semantic_cfg_count == 1u);
        expect(info.promoted_alloca_count == 0u);
        expect(info.replaced_load_count == 0u);
        expect(count_loads_from(kernel, state) == 1u);
    };
}

int main() {
    register_coro_rematerialize_tests();
    return 0;
}
