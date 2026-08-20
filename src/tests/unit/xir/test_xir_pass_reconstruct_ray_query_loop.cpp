// Tests for reconstructing canonical proceed loops as RayQueryLoopInst.

#include "ut/ut.hpp"

#include <luisa/ast/type_registry.h>
#include <luisa/dsl/rtx/ray_query.h>
#include <luisa/xir/basic_block.h>
#include <luisa/xir/builder.h>
#include <luisa/xir/function.h>
#include <luisa/xir/instructions/arithmetic.h>
#include <luisa/xir/instructions/branch.h>
#include <luisa/xir/instructions/loop.h>
#include <luisa/xir/instructions/phi.h>
#include <luisa/xir/instructions/ray_query.h>
#include <luisa/xir/module.h>
#include <luisa/xir/passes/lower_ray_query_to_loop.h>
#include <luisa/xir/passes/reconstruct_ray_query_loop.h>
#include <luisa/xir/verifier.h>

using namespace luisa;
using namespace luisa::compute;
using namespace luisa::compute::xir;
using namespace boost::ut;
using namespace boost::ut::literals;

namespace {

[[nodiscard]] size_t count_blocks(FunctionDefinition *definition) noexcept {
    size_t count = 0u;
    for ([[maybe_unused]] auto *block : definition->basic_blocks()) {
        ++count;
    }
    return count;
}

[[nodiscard]] size_t count_terminators(
    FunctionDefinition *definition,
    DerivedInstructionTag tag) noexcept {
    size_t count = 0u;
    definition->traverse_basic_blocks([&](BasicBlock *block) noexcept {
        if (block->is_terminated() &&
            block->terminator()->derived_instruction_tag() == tag) {
            ++count;
        }
    });
    return count;
}

}// namespace

void register_tests() {
    "lower_reconstruct_round_trip_preserves_query_handlers_and_phi"_test = [] {
        Module module;
        auto *kernel = module.create_kernel();
        auto *body = kernel->create_body_block();
        XIRBuilder builder;
        builder.set_insertion_point(body);
        auto *query = builder.alloca_local(Type::of<RayQueryAll>());
        auto *ray_query = builder.ray_query_loop();
        ray_query->set_name("round_trip_ray_query");
        ray_query->add_comment("preserve loop provenance");
        auto *dispatch_block = ray_query->create_dispatch_block();
        auto *merge = ray_query->create_merge_block();
        builder.set_insertion_point(dispatch_block);
        auto *dispatch = builder.ray_query_dispatch(query);
        dispatch->set_name("round_trip_candidate_dispatch");
        dispatch->add_comment("preserve dispatch provenance");
        dispatch->set_exit_block(merge);
        auto *surface =
            dispatch->create_on_surface_candidate_block();
        auto *procedural =
            dispatch->create_on_procedural_candidate_block();
        builder.set_insertion_point(surface);
        builder.call(
            RayQueryObjectWriteOp::RAY_QUERY_OBJECT_COMMIT_TRIANGLE,
            {query});
        builder.br(dispatch_block);
        builder.set_insertion_point(procedural);
        builder.br(dispatch_block);
        builder.set_insertion_point(merge);
        auto *exit_phi = builder.phi(Type::of<int>());
        exit_phi->add_incoming(
            module.create_constant_one(Type::of<int>()), dispatch_block);
        builder.return_void();
        auto original_block_count = count_blocks(kernel->definition());

        expect(xir_verify_module(&module).succeeded());
        auto lower =
            lower_ray_query_to_loop_pass_run_on_function(kernel);
        expect(lower.succeeded());
        expect(lower.lowered_ray_query_loop_count == 1u);
        expect(count_terminators(
                   kernel->definition(), DerivedInstructionTag::LOOP) == 1u);
        expect(xir_verify_module(&module).succeeded());

        auto reconstruct =
            reconstruct_ray_query_loop_pass_run_on_function(kernel);
        expect(reconstruct.succeeded());
        expect(reconstruct.error_count == 0u);
        expect(reconstruct.ignored_loop_count == 0u);
        expect(reconstruct.reconstructed_ray_query_loop_count == 1u);
        expect(count_blocks(kernel->definition()) == original_block_count);
        expect(count_terminators(
                   kernel->definition(), DerivedInstructionTag::LOOP) == 0u);
        expect(count_terminators(
                   kernel->definition(),
                   DerivedInstructionTag::RAY_QUERY_LOOP) == 1u);
        expect(body->terminator()->isa<RayQueryLoopInst>());
        auto *reconstructed_loop =
            static_cast<RayQueryLoopInst *>(body->terminator());
        expect(reconstructed_loop->name().has_value());
        if (reconstructed_loop->name()) {
            expect(*reconstructed_loop->name() ==
                   "round_trip_ray_query");
        }
        auto *reconstructed_dispatch =
            static_cast<RayQueryDispatchInst *>(
                reconstructed_loop->dispatch_block()->terminator());
        expect(reconstructed_dispatch->query_object() == query);
        expect(reconstructed_dispatch->exit_block() == merge);
        expect(reconstructed_dispatch->on_surface_candidate_block() ==
               surface);
        expect(reconstructed_dispatch->name().has_value());
        if (reconstructed_dispatch->name()) {
            expect(*reconstructed_dispatch->name() ==
                   "round_trip_candidate_dispatch");
        }
        expect(static_cast<BranchInst *>(surface->terminator())
                   ->target_block() ==
               reconstructed_loop->dispatch_block());
        auto *reconstructed_procedural =
            reconstructed_dispatch->on_procedural_candidate_block();
        expect(reconstructed_procedural->instructions().front() ==
               reconstructed_procedural->terminator());
        expect(static_cast<BranchInst *>(
                   reconstructed_procedural->terminator())
                   ->target_block() ==
               reconstructed_loop->dispatch_block());
        expect(exit_phi->incoming_count() == 1u);
        expect(exit_phi->incoming(0u).block ==
               reconstructed_loop->dispatch_block());
        expect(xir_verify_module(&module).succeeded());

        auto second_lower =
            lower_ray_query_to_loop_pass_run_on_function(kernel);
        expect(second_lower.succeeded());
        expect(second_lower.lowered_ray_query_loop_count == 1u);
        expect(xir_verify_module(&module).succeeded());
    };

    "compacted_noop_handlers_round_trip"_test = [] {
        Module module;
        auto *kernel = module.create_kernel();
        auto *body = kernel->create_body_block();
        XIRBuilder builder;
        builder.set_insertion_point(body);
        auto *query = builder.alloca_local(Type::of<RayQueryAny>());
        auto *ray_query = builder.ray_query_loop();
        auto *dispatch_block = ray_query->create_dispatch_block();
        auto *merge = ray_query->create_merge_block();
        builder.set_insertion_point(dispatch_block);
        auto *dispatch = builder.ray_query_dispatch(query);
        dispatch->set_exit_block(merge);
        auto *surface =
            dispatch->create_on_surface_candidate_block();
        auto *procedural =
            dispatch->create_on_procedural_candidate_block();
        builder.set_insertion_point(surface);
        builder.br(dispatch_block);
        builder.set_insertion_point(procedural);
        builder.br(dispatch_block);
        builder.set_insertion_point(merge);
        builder.return_void();
        auto original_block_count = count_blocks(kernel->definition());

        expect(xir_verify_module(&module).succeeded());
        auto lower =
            lower_ray_query_to_loop_pass_run_on_function(kernel);
        expect(lower.succeeded());
        auto *lowered_loop =
            static_cast<LoopInst *>(body->terminator());
        expect(lowered_loop->body_block()->terminator()->isa<BranchInst>());
        expect(static_cast<BranchInst *>(
                   lowered_loop->body_block()->terminator())
                   ->target_block() == lowered_loop->update_block());
        expect(xir_verify_module(&module).succeeded());

        auto reconstruct =
            reconstruct_ray_query_loop_pass_run_on_function(kernel);
        expect(reconstruct.succeeded());
        expect(reconstruct.reconstructed_ray_query_loop_count == 1u);
        expect(count_blocks(kernel->definition()) == original_block_count);
        auto *reconstructed =
            static_cast<RayQueryLoopInst *>(body->terminator());
        auto *reconstructed_dispatch =
            static_cast<RayQueryDispatchInst *>(
                reconstructed->dispatch_block()->terminator());
        expect(reconstructed_dispatch->query_object() == query);
        auto *reconstructed_surface =
            reconstructed_dispatch->on_surface_candidate_block();
        auto *reconstructed_procedural =
            reconstructed_dispatch->on_procedural_candidate_block();
        expect(reconstructed_surface != reconstructed_procedural);
        for (auto *handler : {
                 reconstructed_surface, reconstructed_procedural}) {
            expect(handler->instructions().front() == handler->terminator());
            expect(handler->terminator()->isa<BranchInst>());
            expect(static_cast<BranchInst *>(handler->terminator())
                       ->target_block() == reconstructed->dispatch_block());
        }
        expect(xir_verify_module(&module).succeeded());
    };

    "multiple_handler_exits_round_trip"_test = [] {
        Module module;
        auto *kernel = module.create_kernel();
        auto *condition =
            kernel->create_value_argument(Type::of<bool>());
        auto *body = kernel->create_body_block();
        XIRBuilder builder;
        builder.set_insertion_point(body);
        auto *query = builder.alloca_local(Type::of<RayQueryAll>());
        auto *ray_query = builder.ray_query_loop();
        auto *dispatch_block = ray_query->create_dispatch_block();
        auto *merge = ray_query->create_merge_block();
        builder.set_insertion_point(dispatch_block);
        auto *dispatch = builder.ray_query_dispatch(query);
        dispatch->set_exit_block(merge);
        auto *surface =
            dispatch->create_on_surface_candidate_block();
        auto *procedural =
            dispatch->create_on_procedural_candidate_block();
        auto *left = kernel->create_basic_block();
        auto *right = kernel->create_basic_block();
        builder.set_insertion_point(surface);
        builder.cond_br(condition, left, right);
        builder.set_insertion_point(left);
        builder.br(dispatch_block);
        builder.set_insertion_point(right);
        builder.br(dispatch_block);
        builder.set_insertion_point(procedural);
        builder.br(dispatch_block);
        builder.set_insertion_point(merge);
        builder.return_void();
        auto original_block_count = count_blocks(kernel->definition());

        expect(xir_verify_module(&module).succeeded());
        auto lower =
            lower_ray_query_to_loop_pass_run_on_function(kernel);
        expect(lower.succeeded());
        expect(lower.lowered_ray_query_loop_count == 1u);
        expect(xir_verify_module(&module).succeeded());

        auto reconstruct =
            reconstruct_ray_query_loop_pass_run_on_function(kernel);
        expect(reconstruct.succeeded());
        expect(reconstruct.reconstructed_ray_query_loop_count == 1u);
        expect(count_blocks(kernel->definition()) == original_block_count);
        auto *reconstructed =
            static_cast<RayQueryLoopInst *>(body->terminator());
        auto *reconstructed_dispatch =
            static_cast<RayQueryDispatchInst *>(
                reconstructed->dispatch_block()->terminator());
        expect(reconstructed_dispatch->on_surface_candidate_block() ==
               surface);
        expect(static_cast<BranchInst *>(left->terminator())
                   ->target_block() == reconstructed->dispatch_block());
        expect(static_cast<BranchInst *>(right->terminator())
                   ->target_block() == reconstructed->dispatch_block());
        expect(xir_verify_module(&module).succeeded());
    };

    "ordinary_loop_is_ignored"_test = [] {
        Module module;
        auto *kernel = module.create_kernel();
        auto *condition =
            kernel->create_value_argument(Type::of<bool>());
        auto *body = kernel->create_body_block();
        XIRBuilder builder;
        builder.set_insertion_point(body);
        auto *loop = builder.loop();
        auto *prepare = loop->create_prepare_block();
        auto *loop_body = loop->create_body_block();
        auto *update = loop->create_update_block();
        auto *merge = loop->create_merge_block();
        builder.set_insertion_point(prepare);
        builder.cond_br(condition, loop_body, merge);
        builder.set_insertion_point(loop_body);
        builder.br(update);
        builder.set_insertion_point(update);
        builder.br(prepare);
        builder.set_insertion_point(merge);
        builder.return_void();

        expect(xir_verify_module(&module).succeeded());
        auto info =
            reconstruct_ray_query_loop_pass_run_on_function(kernel);
        expect(info.succeeded());
        expect(info.reconstructed_ray_query_loop_count == 0u);
        expect(info.ignored_loop_count == 1u);
        expect(body->terminator() == loop);
        expect(xir_verify_module(&module).succeeded());
    };

    "misplaced_proceed_is_rejected_atomically"_test = [] {
        Module module;
        auto *kernel = module.create_kernel();
        auto *condition =
            kernel->create_value_argument(Type::of<bool>());
        auto *body = kernel->create_body_block();
        XIRBuilder builder;
        builder.set_insertion_point(body);
        auto *query = builder.alloca_local(Type::of<RayQueryAll>());
        auto *loop = builder.loop();
        auto *prepare = loop->create_prepare_block();
        auto *loop_body = loop->create_body_block();
        auto *update = loop->create_update_block();
        auto *merge = loop->create_merge_block();
        builder.set_insertion_point(prepare);
        builder.cond_br(condition, loop_body, merge);
        builder.set_insertion_point(loop_body);
        auto *proceed = builder.call(
            RayQueryObjectWriteOp::RAY_QUERY_OBJECT_PROCEED, {query});
        builder.br(update);
        builder.set_insertion_point(update);
        builder.br(prepare);
        builder.set_insertion_point(merge);
        builder.return_void();
        auto block_count = count_blocks(kernel->definition());

        expect(xir_verify_module(&module).succeeded());
        auto info =
            reconstruct_ray_query_loop_pass_run_on_function(kernel);
        expect(!info.succeeded());
        expect(info.error_count == 1u);
        expect(info.reconstructed_ray_query_loop_count == 0u);
        expect(info.ignored_loop_count == 0u);
        expect(count_blocks(kernel->definition()) == block_count);
        expect(body->terminator() == loop);
        expect(loop_body->instructions().front() == proceed);
        expect(xir_verify_module(&module).succeeded());
    };

    "nested_handler_loop_survives_ray_query_reconstruction"_test = [] {
        Module module;
        auto *kernel = module.create_kernel();
        auto *condition =
            kernel->create_value_argument(Type::of<bool>());
        auto *body = kernel->create_body_block();
        XIRBuilder builder;
        builder.set_insertion_point(body);
        auto *query = builder.alloca_local(Type::of<RayQueryAll>());
        auto *ray_query = builder.ray_query_loop();
        auto *dispatch_block = ray_query->create_dispatch_block();
        auto *merge = ray_query->create_merge_block();
        builder.set_insertion_point(dispatch_block);
        auto *dispatch = builder.ray_query_dispatch(query);
        dispatch->set_exit_block(merge);
        auto *surface =
            dispatch->create_on_surface_candidate_block();
        auto *procedural =
            dispatch->create_on_procedural_candidate_block();

        builder.set_insertion_point(surface);
        auto *handler_loop = builder.loop();
        auto *handler_prepare = handler_loop->create_prepare_block();
        auto *handler_body = handler_loop->create_body_block();
        auto *handler_update = handler_loop->create_update_block();
        auto *handler_merge = handler_loop->create_merge_block();
        builder.set_insertion_point(handler_prepare);
        builder.cond_br(condition, handler_body, handler_merge);
        builder.set_insertion_point(handler_body);
        builder.br(handler_update);
        builder.set_insertion_point(handler_update);
        builder.br(handler_prepare);
        builder.set_insertion_point(handler_merge);
        builder.br(dispatch_block);
        builder.set_insertion_point(procedural);
        builder.br(dispatch_block);
        builder.set_insertion_point(merge);
        builder.return_void();

        expect(xir_verify_module(&module).succeeded());
        auto lower =
            lower_ray_query_to_loop_pass_run_on_function(kernel);
        expect(lower.succeeded());
        expect(lower.lowered_ray_query_loop_count == 1u);
        expect(count_terminators(
                   kernel->definition(), DerivedInstructionTag::LOOP) == 2u);
        expect(xir_verify_module(&module).succeeded());

        auto reconstruct =
            reconstruct_ray_query_loop_pass_run_on_function(kernel);
        expect(reconstruct.succeeded());
        expect(reconstruct.reconstructed_ray_query_loop_count == 1u);
        expect(reconstruct.ignored_loop_count == 1u);
        expect(count_terminators(
                   kernel->definition(), DerivedInstructionTag::LOOP) == 1u);
        expect(count_terminators(
                   kernel->definition(),
                   DerivedInstructionTag::RAY_QUERY_LOOP) == 1u);
        auto *reconstructed =
            static_cast<RayQueryLoopInst *>(body->terminator());
        auto *reconstructed_dispatch =
            static_cast<RayQueryDispatchInst *>(
                reconstructed->dispatch_block()->terminator());
        expect(reconstructed_dispatch->on_surface_candidate_block() ==
               surface);
        expect(surface->terminator() == handler_loop);
        expect(xir_verify_module(&module).succeeded());
    };

    "module_reconstruction_rejects_late_near_match_atomically"_test = [] {
        Module module;
        XIRBuilder builder;

        auto *valid = module.create_kernel();
        auto *valid_body = valid->create_body_block();
        builder.set_insertion_point(valid_body);
        auto *valid_query =
            builder.alloca_local(Type::of<RayQueryAll>());
        auto *valid_ray_query = builder.ray_query_loop();
        auto *valid_dispatch_block =
            valid_ray_query->create_dispatch_block();
        auto *valid_merge = valid_ray_query->create_merge_block();
        builder.set_insertion_point(valid_dispatch_block);
        auto *valid_dispatch = builder.ray_query_dispatch(valid_query);
        valid_dispatch->set_exit_block(valid_merge);
        auto *valid_surface =
            valid_dispatch->create_on_surface_candidate_block();
        auto *valid_procedural =
            valid_dispatch->create_on_procedural_candidate_block();
        builder.set_insertion_point(valid_surface);
        builder.br(valid_dispatch_block);
        builder.set_insertion_point(valid_procedural);
        builder.br(valid_dispatch_block);
        builder.set_insertion_point(valid_merge);
        builder.return_void();
        auto lower =
            lower_ray_query_to_loop_pass_run_on_function(valid);
        expect(lower.succeeded());
        expect(lower.lowered_ray_query_loop_count == 1u);
        auto *valid_lowered_loop = valid_body->terminator();

        auto *malformed = module.create_kernel();
        auto *malformed_body = malformed->create_body_block();
        builder.set_insertion_point(malformed_body);
        auto *malformed_query =
            builder.alloca_local(Type::of<RayQueryAll>());
        auto *malformed_loop = builder.loop();
        auto *malformed_prepare =
            malformed_loop->create_prepare_block();
        auto *malformed_loop_body =
            malformed_loop->create_body_block();
        auto *malformed_update =
            malformed_loop->create_update_block();
        auto *malformed_merge =
            malformed_loop->create_merge_block();
        builder.set_insertion_point(malformed_prepare);
        builder.call(
            RayQueryObjectWriteOp::RAY_QUERY_OBJECT_PROCEED,
            {malformed_query});
        auto *terminated = builder.call(
            Type::of<bool>(),
            RayQueryObjectReadOp::RAY_QUERY_OBJECT_IS_TERMINATED,
            {malformed_query});
        auto *active = builder.call(
            Type::of<bool>(), ArithmeticOp::UNARY_BIT_NOT,
            {terminated});
        builder.call(
            Type::of<bool>(), ArithmeticOp::UNARY_BIT_NOT, {active});
        builder.cond_br(
            active, malformed_loop_body, malformed_merge);
        builder.set_insertion_point(malformed_loop_body);
        builder.br(malformed_update);
        builder.set_insertion_point(malformed_update);
        builder.br(malformed_prepare);
        builder.set_insertion_point(malformed_merge);
        builder.return_void();

        expect(xir_verify_module(&module).succeeded());
        auto info =
            reconstruct_ray_query_loop_pass_run_on_module(&module);
        expect(!info.succeeded());
        expect(info.error_count == 1u);
        expect(info.reconstructed_ray_query_loop_count == 0u);
        expect(valid_body->terminator() == valid_lowered_loop);
        expect(malformed_body->terminator() == malformed_loop);
        expect(count_terminators(
                   valid->definition(),
                   DerivedInstructionTag::RAY_QUERY_LOOP) == 0u);
        expect(xir_verify_module(&module).succeeded());
    };

    "malformed_ray_like_loop_is_rejected_atomically"_test = [] {
        Module module;
        auto *kernel = module.create_kernel();
        auto *body = kernel->create_body_block();
        XIRBuilder builder;
        builder.set_insertion_point(body);
        auto *query = builder.alloca_local(Type::of<RayQueryAll>());
        auto *loop = builder.loop();
        auto *prepare = loop->create_prepare_block();
        auto *loop_body = loop->create_body_block();
        auto *update = loop->create_update_block();
        auto *merge = loop->create_merge_block();
        builder.set_insertion_point(prepare);
        auto *proceed = builder.call(
            RayQueryObjectWriteOp::RAY_QUERY_OBJECT_PROCEED, {query});
        auto *terminated = builder.call(
            Type::of<bool>(),
            RayQueryObjectReadOp::RAY_QUERY_OBJECT_IS_TERMINATED,
            {query});
        auto *active = builder.call(
            Type::of<bool>(), ArithmeticOp::UNARY_BIT_NOT,
            {terminated});
        // The extra verifier-valid instruction makes this a deliberate
        // near-match rather than the exact reconstruction contract.
        builder.call(
            Type::of<bool>(), ArithmeticOp::UNARY_BIT_NOT, {active});
        auto *prepare_branch =
            builder.cond_br(active, loop_body, merge);
        builder.set_insertion_point(loop_body);
        auto *body_branch = builder.br(update);
        builder.set_insertion_point(update);
        auto *update_branch = builder.br(prepare);
        builder.set_insertion_point(merge);
        builder.return_void();
        auto block_count = count_blocks(kernel->definition());

        expect(xir_verify_module(&module).succeeded());
        auto info =
            reconstruct_ray_query_loop_pass_run_on_function(kernel);
        expect(!info.succeeded());
        expect(info.error_count == 1u);
        expect(info.reconstructed_ray_query_loop_count == 0u);
        expect(count_blocks(kernel->definition()) == block_count);
        expect(body->terminator() == loop);
        expect(prepare->instructions().front() == proceed);
        expect(prepare->terminator() == prepare_branch);
        expect(loop_body->terminator() == body_branch);
        expect(update->terminator() == update_branch);
        expect(xir_verify_module(&module).succeeded());
    };

    "canonical_shell_value_escape_is_rejected_atomically"_test = [] {
        Module module;
        auto *kernel = module.create_kernel();
        auto *body = kernel->create_body_block();
        XIRBuilder builder;
        builder.set_insertion_point(body);
        auto *query = builder.alloca_local(Type::of<RayQueryAll>());
        auto *ray_query = builder.ray_query_loop();
        auto *dispatch_block = ray_query->create_dispatch_block();
        auto *merge = ray_query->create_merge_block();
        builder.set_insertion_point(dispatch_block);
        auto *dispatch = builder.ray_query_dispatch(query);
        dispatch->set_exit_block(merge);
        auto *surface =
            dispatch->create_on_surface_candidate_block();
        auto *procedural =
            dispatch->create_on_procedural_candidate_block();
        builder.set_insertion_point(surface);
        builder.call(
            RayQueryObjectWriteOp::RAY_QUERY_OBJECT_COMMIT_TRIANGLE,
            {query});
        builder.br(dispatch_block);
        builder.set_insertion_point(procedural);
        builder.br(dispatch_block);
        builder.set_insertion_point(merge);
        builder.return_void();

        expect(xir_verify_module(&module).succeeded());
        auto lower =
            lower_ray_query_to_loop_pass_run_on_function(kernel);
        expect(lower.succeeded());
        auto *lowered_loop =
            static_cast<LoopInst *>(body->terminator());
        auto prepare_iter =
            lowered_loop->prepare_block()->instructions().begin();
        ++prepare_iter;
        ++prepare_iter;
        auto *active = *prepare_iter;
        expect(active->isa<ArithmeticInst>());
        builder.set_insertion_point(surface->terminator()->prev());
        auto *escaped_use = builder.call(
            Type::of<bool>(), ArithmeticOp::UNARY_BIT_NOT, {active});
        auto block_count = count_blocks(kernel->definition());
        expect(xir_verify_module(&module).succeeded());

        auto info =
            reconstruct_ray_query_loop_pass_run_on_function(kernel);
        expect(!info.succeeded());
        expect(info.error_count == 1u);
        expect(info.reconstructed_ray_query_loop_count == 0u);
        expect(count_blocks(kernel->definition()) == block_count);
        expect(body->terminator() == lowered_loop);
        expect(escaped_use->operand(0u) == active);
        expect(xir_verify_module(&module).succeeded());
    };
}

int main(int argc, char *argv[]) {
    boost::ut::detail::cfg::parse_arg_with_fallback(
        argc, const_cast<const char **>(argv));
    register_tests();
    return 0;
}
