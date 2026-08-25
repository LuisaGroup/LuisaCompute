// Tests for reconstructing canonical proceed loops as RayQueryLoopInst.

#include "ut/ut.hpp"

#include <luisa/ast/type_registry.h>
#include <luisa/dsl/sugar.h>
#include <luisa/xir/basic_block.h>
#include <luisa/xir/builder.h>
#include <luisa/xir/function.h>
#include <luisa/xir/instructions/arithmetic.h>
#include <luisa/xir/instructions/branch.h>
#include <luisa/xir/instructions/loop.h>
#include <luisa/xir/instructions/phi.h>
#include <luisa/xir/instructions/ray_query.h>
#include <luisa/xir/instructions/resource.h>
#include <luisa/xir/module.h>
#include <luisa/xir/passes/lower_ray_query_to_loop.h>
#include <luisa/xir/passes/lower_ray_query_to_pipeline.h>
#include <luisa/xir/passes/pass_pipeline.h>
#include <luisa/xir/passes/reconstruct_ray_query_loop.h>
#include <luisa/xir/translators/ast2xir.h>
#include <luisa/xir/translators/xir2text.h>
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

[[nodiscard]] KernelFunction *find_kernel(Module *module) noexcept {
    if (module == nullptr) { return nullptr; }
    for (auto *function : module->function_list()) {
        if (function->isa<KernelFunction>()) {
            return static_cast<KernelFunction *>(function);
        }
    }
    return nullptr;
}

[[nodiscard]] uint64_t report_value(
    const PassReport &report, luisa::string_view key) noexcept {
    for (auto &&entry : report.entries()) {
        if (entry.key == key) { return entry.value; }
    }
    return ~uint64_t{0u};
}

void expect_frontend_round_trip(
    luisa::unique_ptr<Module> module,
    size_t query_count,
    size_t ordinary_loop_count) {
    expect(module != nullptr);
    if (module == nullptr) { return; }
    expect(xir_verify_module(module.get()).succeeded());
    auto *kernel = find_kernel(module.get());
    expect(kernel != nullptr);
    if (kernel == nullptr) { return; }
    auto *definition = kernel->definition();
    expect(count_terminators(
               definition,
               DerivedInstructionTag::RAY_QUERY_LOOP) == query_count);
    expect(count_terminators(
               definition,
               DerivedInstructionTag::LOOP) == ordinary_loop_count);

    auto lower =
        lower_ray_query_to_loop_pass_run_on_function(kernel);
    expect(lower.succeeded());
    expect(lower.lowered_ray_query_loop_count == query_count);
    expect(count_terminators(
               definition,
               DerivedInstructionTag::RAY_QUERY_LOOP) == 0u);
    expect(count_terminators(
               definition,
               DerivedInstructionTag::LOOP) ==
           query_count + ordinary_loop_count);
    expect(xir_verify_module(module.get()).succeeded());

    auto reconstruct =
        reconstruct_ray_query_loop_pass_run_on_function(kernel);
    expect(reconstruct.succeeded());
    expect(reconstruct.error_count == 0u);
    expect(reconstruct.reconstructed_ray_query_loop_count ==
           query_count);
    expect(reconstruct.ignored_loop_count == ordinary_loop_count);
    expect(count_terminators(
               definition,
               DerivedInstructionTag::RAY_QUERY_LOOP) == query_count);
    expect(count_terminators(
               definition,
               DerivedInstructionTag::LOOP) == ordinary_loop_count);
    expect(xir_verify_module(module.get()).succeeded());
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

    "frontend_dsl_traverse_with_both_handlers_round_trips"_test = [] {
        Kernel1D kernel = [](AccelVar accel, BufferUInt output) noexcept {
            auto index = dispatch_x();
            auto ray = make_ray(
                make_float3(cast<float>(index), 0.0f, 0.0f),
                make_float3(0.0f, 0.0f, 1.0f), 0.0f, 100.0f);
            UInt callback_weight = 0u;
            auto committed =
                accel.traverse(ray, {})
                    .on_surface_candidate(
                        [&](SurfaceCandidate &candidate) noexcept {
                            auto hit = candidate.hit();
                            $if (hit->inst != ~0u) {
                                $if ((hit->prim & 1u) == 0u) {
                                    callback_weight += 1u;
                                    candidate.commit();
                                }
                                $else {
                                    candidate.terminate();
                                };
                            };
                        })
                    .on_procedural_candidate(
                        [&](ProceduralCandidate &candidate) noexcept {
                            auto hit = candidate.hit();
                            $if (hit->prim < 4u) {
                                callback_weight += 2u;
                                candidate.commit(1.0f);
                            };
                        })
                    .trace();
            output.write(
                index, callback_weight + committed->prim);
        };
        auto module = ast_to_xir_translate(
            kernel.function()->function(), {});
        expect_frontend_round_trip(std::move(module), 1u, 0u);
    };

    "frontend_dsl_query_any_with_empty_handlers_round_trips"_test = [] {
        Kernel1D kernel = [](AccelVar accel, BufferUInt output) noexcept {
            auto index = dispatch_x();
            auto ray = make_ray(
                make_float3(cast<float>(index), 0.0f, 0.0f),
                make_float3(0.0f, 0.0f, 1.0f));
            auto committed = accel.traverse_any(ray, {}).trace();
            output.write(index, committed->prim);
        };
        auto module = ast_to_xir_translate(
            kernel.function()->function(), {});
        auto *translated_kernel = find_kernel(module.get());
        expect(translated_kernel != nullptr);
        auto saw_query_any = false;
        if (translated_kernel != nullptr) {
            translated_kernel->definition()->traverse_instructions(
                [&](Instruction *instruction) noexcept {
                    if (instruction->isa<RayQueryDispatchInst>()) {
                        auto *dispatch =
                            static_cast<RayQueryDispatchInst *>(instruction);
                        saw_query_any |= dispatch->query_object()->type() ==
                                         Type::of<RayQueryAny>();
                    }
                });
        }
        expect(saw_query_any);
        expect_frontend_round_trip(std::move(module), 1u, 0u);
    };

    "frontend_dsl_two_queries_inside_loop_round_trip"_test = [] {
        Kernel1D kernel = [](AccelVar accel, BufferUInt output) noexcept {
            auto index = dispatch_x();
            auto ray = make_ray(
                make_float3(cast<float>(index), 0.0f, 0.0f),
                make_float3(0.0f, 0.0f, 1.0f));
            UInt result = 0u;
            for (auto iteration : dynamic_range(2u)) {
                auto all = accel.traverse(ray, {}).trace();
                auto any =
                    accel.traverse_any(ray, {})
                        .on_surface_candidate(
                            [&](SurfaceCandidate &candidate) noexcept {
                                $if (iteration == 1u) {
                                    candidate.commit();
                                };
                            })
                        .trace();
                result += all->prim + any->prim + iteration;
            }
            output.write(index, result);
        };
        auto module = ast_to_xir_translate(
            kernel.function()->function(), {});
        expect_frontend_round_trip(std::move(module), 2u, 1u);
    };

    "frontend_dsl_inline_query_preserves_structured_loop"_test = [] {
        Kernel1D kernel = [](AccelVar accel, BufferUInt output) noexcept {
            auto index = dispatch_x();
            auto ray = make_ray(
                make_float3(cast<float>(index), 0.0f, 0.0f),
                make_float3(0.0f, 0.0f, 1.0f), 0.0f, 100.0f);
            auto query = accel.query(ray, {});
            UInt callback_weight = 0u;
            $while (query.proceed()) {
                $if (query.is_surface_candidate()) {
                    auto candidate = query.surface_candidate();
                    auto hit = candidate.hit();
                    callback_weight += hit->prim;
                    $if ((hit->prim & 1u) == 0u) {
                        candidate.commit();
                    };
                }
                $else {
                    auto candidate = query.procedural_candidate();
                    auto hit = candidate.hit();
                    callback_weight += hit->prim + 1u;
                    candidate.commit(1.0f);
                };
            };
            auto committed = query.committed_hit();
            output.write(
                index, callback_weight + committed->prim);
        };
        const LoopStmt *marked_loop = nullptr;
        for (auto statement :
             kernel.function()->function().body()->statements()) {
            if (statement->tag() == Statement::Tag::LOOP) {
                marked_loop = static_cast<const LoopStmt *>(statement);
                break;
            }
        }
        expect(marked_loop != nullptr);
        if (marked_loop != nullptr) {
            expect(marked_loop->while_condition() != nullptr);
        }
        auto module = ast_to_xir_translate(
            kernel.function()->function(), {});
        expect(module != nullptr);
        if (module == nullptr) { return; }
        expect(xir_verify_module(module.get()).succeeded());
        auto *translated_kernel = find_kernel(module.get());
        expect(translated_kernel != nullptr);
        if (translated_kernel == nullptr) { return; }
        auto *definition = translated_kernel->definition();
        expect(count_terminators(
                   definition,
                   DerivedInstructionTag::SIMPLE_LOOP) == 0u);
        expect(count_terminators(
                   definition,
                   DerivedInstructionTag::RAY_QUERY_LOOP) == 1u);
        auto proceed_count = 0u;
        definition->traverse_instructions(
            [&](Instruction *instruction) noexcept {
                if (instruction->isa<RayQueryObjectWriteInst>() &&
                    static_cast<RayQueryObjectWriteInst *>(instruction)
                            ->op() ==
                        RayQueryObjectWriteOp::
                            RAY_QUERY_OBJECT_PROCEED) {
                    ++proceed_count;
                }
            });
        expect(proceed_count == 0u);

        auto reconstruct =
            reconstruct_ray_query_loop_pass_run_on_function(
                translated_kernel);
        expect(reconstruct.succeeded());
        expect(reconstruct.error_count == 0u);
        expect(reconstruct.reconstructed_ray_query_loop_count == 0u);
        expect(count_terminators(
                   definition,
                   DerivedInstructionTag::SIMPLE_LOOP) == 0u);
        expect(count_terminators(
                   definition,
                   DerivedInstructionTag::RAY_QUERY_LOOP) == 1u);
        expect(xir_verify_module(module.get()).succeeded());

        auto legacy_module = ast_to_xir_translate(
            kernel.function()->function(),
            {.preserve_inline_ray_query_loops = false});
        expect(legacy_module != nullptr);
        auto *legacy_kernel = find_kernel(legacy_module.get());
        expect(legacy_kernel != nullptr);
        if (legacy_kernel != nullptr) {
            auto *legacy_definition = legacy_kernel->definition();
            expect(count_terminators(
                       legacy_definition,
                       DerivedInstructionTag::SIMPLE_LOOP) == 1u);
            expect(count_terminators(
                       legacy_definition,
                       DerivedInstructionTag::RAY_QUERY_LOOP) == 0u);
            auto legacy_reconstruct =
                reconstruct_ray_query_loop_pass_run_on_function(
                    legacy_kernel);
            expect(legacy_reconstruct.succeeded());
            expect(legacy_reconstruct.error_count == 0u);
            expect(legacy_reconstruct.reconstructed_ray_query_loop_count ==
                   1u);
            expect(xir_verify_module(legacy_module.get()).succeeded());
            expect(xir_to_text_translate(module.get(), false) ==
                   xir_to_text_translate(legacy_module.get(), false));
        }

        auto lower = lower_ray_query_to_loop_pass_run_on_function(
            translated_kernel);
        expect(lower.succeeded());
        expect(lower.lowered_ray_query_loop_count == 1u);
        expect(count_terminators(
                   definition,
                   DerivedInstructionTag::RAY_QUERY_LOOP) == 0u);
        expect(count_terminators(
                   definition,
                   DerivedInstructionTag::LOOP) == 1u);
        expect(xir_verify_module(module.get()).succeeded());
    };

    "frontend_dsl_inline_query_any_motion_reconstructs_to_pipeline"_test = [] {
        Kernel1D kernel = [](AccelVar accel, BufferUInt output) noexcept {
            auto index = dispatch_x();
            auto ray = make_ray(
                make_float3(cast<float>(index), 0.0f, 0.0f),
                make_float3(0.0f, 0.0f, 1.0f));
            auto query = accel.query_any_motion(ray, 0.5f, {});
            UInt callback_weight = 0u;
            $while (query.proceed()) {
                $if (query.is_procedural_candidate()) {
                    auto candidate = query.procedural_candidate();
                    callback_weight += candidate.hit()->prim;
                    candidate.commit(1.0f);
                }
                $else {
                    auto candidate = query.surface_candidate();
                    callback_weight += candidate.hit()->prim + 1u;
                    candidate.commit();
                    query.terminate();
                };
            };
            output.write(
                index,
                callback_weight + query.committed_hit()->prim);
        };
        auto module = ast_to_xir_translate(
            kernel.function()->function(), {});
        expect(module != nullptr);
        if (module == nullptr) { return; }
        auto *translated_kernel = find_kernel(module.get());
        expect(translated_kernel != nullptr);
        if (translated_kernel == nullptr) { return; }
        expect(xir_verify_module(module.get()).succeeded());

        expect(count_terminators(
                   translated_kernel->definition(),
                   DerivedInstructionTag::SIMPLE_LOOP) == 0u);
        expect(count_terminators(
                   translated_kernel->definition(),
                   DerivedInstructionTag::RAY_QUERY_LOOP) == 1u);
        auto reconstruct =
            reconstruct_ray_query_loop_pass_run_on_function(
                translated_kernel);
        expect(reconstruct.succeeded());
        expect(reconstruct.reconstructed_ray_query_loop_count == 0u);
        auto saw_query_any = false;
        translated_kernel->definition()->traverse_instructions(
            [&](Instruction *instruction) noexcept {
                if (instruction->isa<RayQueryDispatchInst>()) {
                    auto *dispatch = static_cast<RayQueryDispatchInst *>(
                        instruction);
                    saw_query_any |= dispatch->query_object()->type() ==
                                     Type::of<RayQueryAny>();
                }
            });
        expect(saw_query_any);
        expect(xir_verify_module(module.get()).succeeded());

        auto pipeline =
            lower_ray_query_to_pipeline_pass_run_on_function(
                translated_kernel);
        expect(pipeline.succeeded());
        expect(pipeline.lowered_loop_count == 1u);
        expect(count_terminators(
                   translated_kernel->definition(),
                   DerivedInstructionTag::RAY_QUERY_LOOP) == 0u);
        expect(xir_verify_module(module.get()).succeeded());
    };

    "frontend_dsl_inline_query_motion_uses_all_motion_opcode"_test = [] {
        Kernel1D kernel = [](AccelVar accel, BufferUInt output) noexcept {
            auto index = dispatch_x();
            auto ray = make_ray(
                make_float3(cast<float>(index), 0.0f, 0.0f),
                make_float3(0.0f, 0.0f, 1.0f));
            auto query = accel.query_motion(ray, 0.5f, {});
            UInt callback_weight = 0u;
            $while (query.proceed()) {
                $if (query.is_surface_candidate()) {
                    auto candidate = query.surface_candidate();
                    callback_weight += candidate.hit()->prim;
                    candidate.commit();
                }
                $else {
                    auto candidate = query.procedural_candidate();
                    callback_weight += candidate.hit()->prim + 1u;
                    candidate.commit(1.0f);
                };
            };
            output.write(
                index,
                callback_weight + query.committed_hit()->prim);
        };
        auto module = ast_to_xir_translate(
            kernel.function()->function(), {});
        expect(module != nullptr);
        if (module == nullptr) { return; }
        auto *translated_kernel = find_kernel(module.get());
        expect(translated_kernel != nullptr);
        if (translated_kernel == nullptr) { return; }
        auto saw_query_all_motion = false;
        translated_kernel->definition()->traverse_instructions(
            [&](Instruction *instruction) noexcept {
                if (instruction->isa<ResourceQueryInst>()) {
                    auto *query = static_cast<ResourceQueryInst *>(
                        instruction);
                    saw_query_all_motion |=
                        query->op() == ResourceQueryOp::
                                           RAY_TRACING_QUERY_ALL_MOTION_BLUR;
                }
            });
        expect(saw_query_all_motion);
        expect_frontend_round_trip(std::move(module), 1u, 0u);
    };

    "frontend_dsl_inline_query_nested_in_ordinary_while_is_preserved"_test = [] {
        Kernel1D kernel = [](AccelVar accel, BufferUInt output) noexcept {
            auto index = dispatch_x();
            auto ray = make_ray(
                make_float3(cast<float>(index), 0.0f, 0.0f),
                make_float3(0.0f, 0.0f, 1.0f));
            UInt outer_iteration = 0u;
            UInt result = 0u;
            $while (outer_iteration < 2u) {
                auto query = accel.query(ray, {});
                $while (query.proceed()) {
                    $if (query.is_surface_candidate()) {
                        auto candidate = query.surface_candidate();
                        for (auto inner : dynamic_range(2u)) {
                            result += candidate.hit()->prim + inner;
                        }
                        candidate.commit();
                    }
                    $else {
                        auto candidate = query.procedural_candidate();
                        result += candidate.hit()->prim;
                        candidate.commit(1.0f);
                    };
                };
                result += query.committed_hit()->prim;
                outer_iteration += 1u;
            };
            output.write(index, result);
        };

        auto direct = ast_to_xir_translate(
            kernel.function()->function(), {});
        auto legacy = ast_to_xir_translate(
            kernel.function()->function(),
            {.preserve_inline_ray_query_loops = false});
        expect(direct != nullptr);
        expect(legacy != nullptr);
        if (direct == nullptr || legacy == nullptr) { return; }
        auto *direct_kernel = find_kernel(direct.get());
        auto *legacy_kernel = find_kernel(legacy.get());
        expect(direct_kernel != nullptr);
        expect(legacy_kernel != nullptr);
        if (direct_kernel == nullptr || legacy_kernel == nullptr) { return; }

        expect(count_terminators(
                   direct_kernel->definition(),
                   DerivedInstructionTag::SIMPLE_LOOP) == 1u);
        expect(count_terminators(
                   direct_kernel->definition(),
                   DerivedInstructionTag::RAY_QUERY_LOOP) == 1u);
        expect(count_terminators(
                   legacy_kernel->definition(),
                   DerivedInstructionTag::SIMPLE_LOOP) == 2u);
        expect(count_terminators(
                   legacy_kernel->definition(),
                   DerivedInstructionTag::RAY_QUERY_LOOP) == 0u);

        auto direct_reconstruct =
            reconstruct_ray_query_loop_pass_run_on_function(
                direct_kernel);
        auto legacy_reconstruct =
            reconstruct_ray_query_loop_pass_run_on_function(
                legacy_kernel);
        expect(direct_reconstruct.succeeded());
        expect(direct_reconstruct.reconstructed_ray_query_loop_count ==
               0u);
        expect(legacy_reconstruct.succeeded());
        expect(legacy_reconstruct.reconstructed_ray_query_loop_count ==
               1u);
        expect(xir_verify_module(direct.get()).succeeded());
        expect(xir_verify_module(legacy.get()).succeeded());
        expect(xir_to_text_translate(direct.get(), false) ==
               xir_to_text_translate(legacy.get(), false));
    };

    "frontend_dsl_inline_query_without_candidate_split_rejects_atomically"_test = [] {
        Kernel1D kernel = [](AccelVar accel, BufferUInt output) noexcept {
            auto index = dispatch_x();
            auto ray = make_ray(
                make_float3(cast<float>(index), 0.0f, 0.0f),
                make_float3(0.0f, 0.0f, 1.0f));
            auto query = accel.query(ray, {});
            UInt candidate_count = 0u;
            $while (query.proceed()) {
                candidate_count += 1u;
            };
            output.write(index, candidate_count);
        };
        auto module = ast_to_xir_translate(
            kernel.function()->function(), {});
        expect(module != nullptr);
        if (module == nullptr) { return; }
        auto *translated_kernel = find_kernel(module.get());
        expect(translated_kernel != nullptr);
        if (translated_kernel == nullptr) { return; }
        expect(xir_verify_module(module.get()).succeeded());
        auto *definition = translated_kernel->definition();
        auto info = reconstruct_ray_query_loop_pass_run_on_function(
            translated_kernel);
        expect(!info.succeeded());
        expect(info.error_count == 1u);
        expect(info.reconstructed_ray_query_loop_count == 0u);
        expect(count_terminators(
                   definition,
                   DerivedInstructionTag::SIMPLE_LOOP) == 1u);
        expect(count_terminators(
                   definition,
                   DerivedInstructionTag::RAY_QUERY_LOOP) == 0u);
        expect(xir_verify_module(module.get()).succeeded());
    };

    "frontend_dsl_inline_queries_preflight_atomically"_test = [] {
        Kernel1D kernel = [](AccelVar accel, BufferUInt output) noexcept {
            auto index = dispatch_x();
            auto ray = make_ray(
                make_float3(cast<float>(index), 0.0f, 0.0f),
                make_float3(0.0f, 0.0f, 1.0f));
            UInt result = 0u;
            auto valid = accel.query(ray, {});
            $while (valid.proceed()) {
                $if (valid.is_surface_candidate()) {
                    auto candidate = valid.surface_candidate();
                    for (auto iteration : dynamic_range(2u)) {
                        result += iteration;
                    }
                    candidate.commit();
                }
                $else {
                    auto candidate = valid.procedural_candidate();
                    candidate.commit(1.0f);
                };
            };
            auto malformed = accel.query_any(ray, {});
            $while (malformed.proceed() & (index == index)) {
                $if (malformed.is_surface_candidate()) {
                    result += 1u;
                    malformed.surface_candidate().terminate();
                }
                $else {
                    result += 2u;
                    malformed.procedural_candidate().terminate();
                };
            };
            output.write(
                index,
                result + valid.committed_hit()->prim +
                    malformed.committed_hit()->prim);
        };
        auto module = ast_to_xir_translate(
            kernel.function()->function(), {});
        expect(module != nullptr);
        if (module == nullptr) { return; }
        auto *translated_kernel = find_kernel(module.get());
        expect(translated_kernel != nullptr);
        if (translated_kernel == nullptr) { return; }
        expect(xir_verify_module(module.get()).succeeded());
        auto *definition = translated_kernel->definition();
        expect(count_terminators(
                   definition,
                   DerivedInstructionTag::SIMPLE_LOOP) == 2u);
        expect(count_terminators(
                   definition,
                   DerivedInstructionTag::LOOP) == 1u);

        auto info = reconstruct_ray_query_loop_pass_run_on_function(
            translated_kernel);
        expect(!info.succeeded());
        expect(info.error_count == 1u);
        expect(info.reconstructed_ray_query_loop_count == 0u);
        expect(count_terminators(
                   definition,
                   DerivedInstructionTag::SIMPLE_LOOP) == 2u);
        expect(count_terminators(
                   definition,
                   DerivedInstructionTag::RAY_QUERY_LOOP) == 0u);
        expect(count_terminators(
                   definition,
                   DerivedInstructionTag::LOOP) == 1u);
        expect(xir_verify_module(module.get()).succeeded());
    };

    "same_function_late_malformed_loop_rejects_atomically"_test = [] {
        Kernel1D kernel = [](AccelVar accel, BufferUInt output) noexcept {
            auto index = dispatch_x();
            auto ray = make_ray(
                make_float3(cast<float>(index), 0.0f, 0.0f),
                make_float3(0.0f, 0.0f, 1.0f));
            auto first = accel.traverse(ray, {}).trace();
            auto second = accel.traverse_any(ray, {}).trace();
            output.write(index, first->prim + second->prim);
        };
        auto module = ast_to_xir_translate(
            kernel.function()->function(), {});
        auto *translated_kernel = find_kernel(module.get());
        expect(translated_kernel != nullptr);
        if (translated_kernel == nullptr) { return; }
        auto lower = lower_ray_query_to_loop_pass_run_on_function(
            translated_kernel);
        expect(lower.succeeded());
        expect(lower.lowered_ray_query_loop_count == 2u);
        luisa::vector<LoopInst *> lowered_loops;
        translated_kernel->definition()->traverse_basic_blocks(
            [&](BasicBlock *block) noexcept {
                if (block->is_terminated() &&
                    block->terminator()->isa<LoopInst>()) {
                    lowered_loops.emplace_back(
                        static_cast<LoopInst *>(block->terminator()));
                }
            });
        expect(lowered_loops.size() == 2u);
        if (lowered_loops.size() != 2u) { return; }
        auto *first_loop = lowered_loops.front();
        auto *malformed_loop = lowered_loops.back();
        auto *prepare = malformed_loop->prepare_block();
        auto *active = prepare->terminator()->prev();
        expect(active->isa<ArithmeticInst>());
        XIRBuilder builder;
        builder.set_insertion_point(active);
        builder.call(
            Type::of<bool>(), ArithmeticOp::UNARY_BIT_NOT, {active});
        expect(xir_verify_module(module.get()).succeeded());

        auto info = reconstruct_ray_query_loop_pass_run_on_function(
            translated_kernel);
        expect(!info.succeeded());
        expect(info.error_count == 1u);
        expect(info.reconstructed_ray_query_loop_count == 0u);
        expect(first_loop->parent_block()->terminator() == first_loop);
        expect(malformed_loop->parent_block()->terminator() ==
               malformed_loop);
        expect(count_terminators(
                   translated_kernel->definition(),
                   DerivedInstructionTag::RAY_QUERY_LOOP) == 0u);
        expect(count_terminators(
                   translated_kernel->definition(),
                   DerivedInstructionTag::LOOP) == 2u);
        expect(xir_verify_module(module.get()).succeeded());
    };

    "module_report_null_and_idempotence_are_stable"_test = [] {
        Kernel1D kernel = [](AccelVar accel, BufferUInt output) noexcept {
            auto index = dispatch_x();
            auto ray = make_ray(
                make_float3(cast<float>(index), 0.0f, 0.0f),
                make_float3(0.0f, 0.0f, 1.0f));
            auto committed = accel.traverse(ray, {}).trace();
            output.write(index, committed->prim);
        };
        auto module = ast_to_xir_translate(
            kernel.function()->function(), {});
        auto lower = lower_ray_query_to_loop_pass_run_on_module(
            module.get());
        expect(lower.succeeded());
        expect(lower.lowered_ray_query_loop_count == 1u);

        PassReport first_report;
        auto first = reconstruct_ray_query_loop_pass_run_on_module(
            module.get(), &first_report);
        expect(first.succeeded());
        expect(first.reconstructed_ray_query_loop_count == 1u);
        expect(first_report.entries().size() == 3u);
        expect(report_value(
                   first_report,
                   "reconstructed_ray_query_loop") == 1u);
        expect(report_value(first_report, "ignored_loop") == 0u);
        expect(report_value(first_report, "error") == 0u);
        expect(xir_verify_module(module.get()).succeeded());

        PassReport second_report;
        auto second = reconstruct_ray_query_loop_pass_run_on_module(
            module.get(), &second_report);
        expect(second.succeeded());
        expect(!second.changed());
        expect(second_report.entries().size() == 3u);
        expect(report_value(
                   second_report,
                   "reconstructed_ray_query_loop") == 0u);
        expect(report_value(second_report, "ignored_loop") == 0u);
        expect(report_value(second_report, "error") == 0u);
        expect(xir_verify_module(module.get()).succeeded());

        auto null_function =
            reconstruct_ray_query_loop_pass_run_on_function(nullptr);
        expect(null_function.succeeded());
        expect(!null_function.changed());
        PassReport null_report;
        auto null_module =
            reconstruct_ray_query_loop_pass_run_on_module(
                nullptr, &null_report);
        expect(null_module.succeeded());
        expect(!null_module.changed());
        expect(null_report.entries().size() == 3u);
        expect(report_value(
                   null_report,
                   "reconstructed_ray_query_loop") == 0u);
        expect(report_value(null_report, "ignored_loop") == 0u);
        expect(report_value(null_report, "error") == 0u);
    };
}

int main(int argc, char *argv[]) {
    boost::ut::detail::cfg::parse_arg_with_fallback(
        argc, const_cast<const char **>(argv));
    register_tests();
    return 0;
}
