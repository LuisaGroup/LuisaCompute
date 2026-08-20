// Test for lowering ray-query loop control flow and invalid-shape rejection.

#include "ut/ut.hpp"

#include <luisa/ast/type_registry.h>
#include <luisa/dsl/rtx/ray_query.h>
#include <luisa/xir/basic_block.h>
#include <luisa/xir/builder.h>
#include <luisa/xir/function.h>
#include <luisa/xir/instructions/alloca.h>
#include <luisa/xir/instructions/branch.h>
#include <luisa/xir/instructions/call.h>
#include <luisa/xir/instructions/load.h>
#include <luisa/xir/instructions/phi.h>
#include <luisa/xir/instructions/ray_query.h>
#include <luisa/xir/instructions/return.h>
#include <luisa/xir/module.h>
#include <luisa/xir/passes/lower_ray_query_loop.h>
#include <luisa/xir/passes/lower_ray_query_to_pipeline.h>
#include <luisa/xir/verifier.h>

using namespace luisa;
using namespace luisa::compute;
using namespace luisa::compute::xir;
using namespace boost::ut;
using namespace boost::ut::literals;

namespace {

[[nodiscard]] size_t count_functions(Module &m) noexcept {
    size_t n = 0u;
    for ([[maybe_unused]] auto *function : m.function_list()) { ++n; }
    return n;
}

[[nodiscard]] size_t count_blocks(FunctionDefinition *def) noexcept {
    size_t n = 0u;
    for ([[maybe_unused]] auto *block : def->basic_blocks()) { ++n; }
    return n;
}

struct RayQueryFixture {
    KernelFunction *kernel;
    BasicBlock *body;
    Value *query;
    RayQueryLoopInst *loop;
    BasicBlock *dispatch;
    BasicBlock *merge;
    RayQueryDispatchInst *dispatch_inst;
    BasicBlock *surface;
    BasicBlock *procedural;
};

[[nodiscard]] RayQueryFixture make_fixture(Module &m) noexcept {
    auto *kernel = m.create_kernel();
    auto *body = kernel->create_body_block();
    XIRBuilder b;
    b.set_insertion_point(body);
    auto *query = b.alloca_local(Type::of<RayQueryAll>());
    auto *loop = b.ray_query_loop();
    auto *dispatch = loop->create_dispatch_block();
    auto *merge = loop->create_merge_block();
    b.set_insertion_point(dispatch);
    auto *dispatch_inst = b.ray_query_dispatch(query);
    dispatch_inst->set_exit_block(merge);
    auto *surface = dispatch_inst->create_on_surface_candidate_block();
    auto *procedural = dispatch_inst->create_on_procedural_candidate_block();
    b.set_insertion_point(procedural);
    b.br(dispatch);
    b.set_insertion_point(merge);
    b.return_void();
    return {kernel, body, query, loop, dispatch, merge, dispatch_inst, surface, procedural};
}

}// namespace

void register_tests() {
    "single_exit_handlers_are_outlined"_test = [] {
        Module m;
        auto f = make_fixture(m);
        XIRBuilder b;
        b.set_insertion_point(f.surface);
        auto *surface_exit = b.br(f.dispatch);
        f.loop->set_name("source_ray_query_loop");
        f.loop->add_comment("preserve lowering provenance");
        f.dispatch_inst->set_name("source_candidate_dispatch");
        f.dispatch_inst->add_comment("preserve dispatch provenance");
        f.surface->set_name("source_surface_handler");
        surface_exit->set_name("source_surface_exit");

        expect(xir_verify_module(&m).succeeded());
        auto info =
            lower_ray_query_to_pipeline_pass_run_on_function(f.kernel);
        expect(info.lowered_loop_count == 1u);
        expect(info.error_count == 0u);
        expect(info.succeeded());
        RayQueryPipelineInst *pipeline = nullptr;
        f.body->traverse_instructions([&](Instruction *inst) noexcept {
            if (inst->isa<RayQueryPipelineInst>()) {
                expect(pipeline == nullptr);
                pipeline = static_cast<RayQueryPipelineInst *>(inst);
            }
        });
        expect(pipeline != nullptr);
        if (pipeline == nullptr) { return; }
        expect(pipeline->query_object() == f.query);
        expect(pipeline->query_object()->type() == Type::of<RayQueryAll>());
        expect(pipeline->captured_argument_count() == 0u);
        expect(pipeline->name().has_value());
        if (pipeline->name()) {
            expect(*pipeline->name() == "source_ray_query_loop");
        }
        expect(pipeline->metadata_list().count_size() == 4u);
        auto *surface_callback = pipeline->on_surface_function();
        expect(surface_callback->definition()->body_block()->name().has_value());
        if (surface_callback->definition()->body_block()->name()) {
            expect(*surface_callback->definition()->body_block()->name() ==
                   "source_surface_handler");
        }
        expect(surface_callback->definition()
                   ->body_block()
                   ->terminator()
                   ->name()
                   .has_value());
        if (surface_callback->definition()
                ->body_block()
                ->terminator()
                ->name()) {
            expect(*surface_callback->definition()
                        ->body_block()
                        ->terminator()
                        ->name() == "source_surface_exit");
        }
        for (auto *callback : {pipeline->on_surface_function(),
                               pipeline->on_procedural_function()}) {
            expect(callback != nullptr);
            expect(callback->isa<CallableFunction>());
            expect(callback->type() == nullptr);
            expect(callback->arguments().count_size() == 1u);
            auto *query_argument = callback->arguments().front();
            expect(query_argument->is_reference());
            expect(query_argument->type() == Type::of<RayQueryAll>());
            expect(callback->definition()->body_block()->terminator()->isa<ReturnInst>());
        }
        expect(count_functions(m) == 3u);
        expect(xir_verify_module(&m).succeeded());
    };

    "non_query_lvalue_is_rejected_atomically"_test = [] {
        Module m;
        auto *kernel = m.create_kernel();
        auto *body = kernel->create_body_block();
        XIRBuilder b;
        b.set_insertion_point(body);
        auto *not_a_query = b.alloca_local(Type::of<int>());
        auto *loop = b.ray_query_loop();
        auto *dispatch = loop->create_dispatch_block();
        auto *merge = loop->create_merge_block();
        b.set_insertion_point(dispatch);
        auto *dispatch_inst = b.ray_query_dispatch(not_a_query);
        dispatch_inst->set_exit_block(merge);
        auto *surface =
            dispatch_inst->create_on_surface_candidate_block();
        auto *procedural =
            dispatch_inst->create_on_procedural_candidate_block();
        b.set_insertion_point(surface);
        auto *surface_exit = b.br(dispatch);
        b.set_insertion_point(procedural);
        auto *procedural_exit = b.br(dispatch);
        b.set_insertion_point(merge);
        b.return_void();
        auto function_count = count_functions(m);
        auto block_count = count_blocks(kernel->definition());

        auto info = lower_ray_query_to_pipeline_pass_run_on_function(kernel);

        expect(!info.succeeded());
        expect(info.error_count == 1u);
        expect(info.lowered_loop_count == 0u);
        expect(count_functions(m) == function_count);
        expect(count_blocks(kernel->definition()) == block_count);
        expect(body->terminator() == loop);
        expect(dispatch->terminator() == dispatch_inst);
        expect(surface->terminator() == surface_exit);
        expect(procedural->terminator() == procedural_exit);
    };

    "ray_query_any_is_outlined_with_exact_callback_abi"_test = [] {
        Module m;
        auto *kernel = m.create_kernel();
        auto *body = kernel->create_body_block();
        XIRBuilder b;
        b.set_insertion_point(body);
        auto *query = b.alloca_local(Type::of<RayQueryAny>());
        auto *loop = b.ray_query_loop();
        auto *dispatch = loop->create_dispatch_block();
        auto *merge = loop->create_merge_block();
        b.set_insertion_point(dispatch);
        auto *dispatch_inst = b.ray_query_dispatch(query);
        dispatch_inst->set_exit_block(merge);
        auto *surface =
            dispatch_inst->create_on_surface_candidate_block();
        auto *procedural =
            dispatch_inst->create_on_procedural_candidate_block();
        b.set_insertion_point(surface);
        b.br(dispatch);
        b.set_insertion_point(procedural);
        b.br(dispatch);
        b.set_insertion_point(merge);
        b.return_void();

        expect(xir_verify_module(&m).succeeded());
        auto info = lower_ray_query_to_pipeline_pass_run_on_function(kernel);
        expect(info.succeeded());
        expect(info.error_count == 0u);
        expect(info.lowered_loop_count == 1u);
        RayQueryPipelineInst *pipeline = nullptr;
        body->traverse_instructions([&](Instruction *inst) noexcept {
            if (inst->isa<RayQueryPipelineInst>()) {
                pipeline = static_cast<RayQueryPipelineInst *>(inst);
            }
        });
        expect(pipeline != nullptr);
        if (pipeline == nullptr) { return; }
        expect(pipeline->query_object()->type() == Type::of<RayQueryAny>());
        for (auto *callback : {pipeline->on_surface_function(),
                               pipeline->on_procedural_function()}) {
            expect(callback->arguments().front()->is_reference());
            expect(callback->arguments().front()->type() ==
                   Type::of<RayQueryAny>());
        }
        expect(xir_verify_module(&m).succeeded());
    };

    "outlined_handler_diamond_preserves_block_phi_and_exit_metadata"_test = [] {
        Module m;
        auto f = make_fixture(m);
        auto *condition =
            f.kernel->create_value_argument(Type::of<bool>());
        auto *left = f.kernel->create_basic_block();
        auto *right = f.kernel->create_basic_block();
        auto *join = f.kernel->create_basic_block();
        XIRBuilder b;
        f.surface->set_name("surface_entry");
        left->set_name("surface_left");
        right->set_name("surface_right");
        join->set_name("surface_join");
        b.set_insertion_point(f.surface);
        b.cond_br(condition, left, right);
        b.set_insertion_point(left);
        b.br(join);
        b.set_insertion_point(right);
        b.br(join);
        b.set_insertion_point(join);
        auto *phi = b.phi(Type::of<int>());
        phi->add_incoming(m.create_constant_zero(Type::of<int>()), left);
        phi->add_incoming(m.create_constant_one(Type::of<int>()), right);
        phi->set_name("surface_join_value");
        auto *exit = b.br(f.dispatch);
        exit->set_name("surface_join_exit");

        expect(xir_verify_module(&m).succeeded());
        auto info = lower_ray_query_to_pipeline_pass_run_on_function(f.kernel);
        expect(info.succeeded());
        expect(info.lowered_loop_count == 1u);

        RayQueryPipelineInst *pipeline = nullptr;
        f.body->traverse_instructions([&](Instruction *inst) noexcept {
            if (inst->isa<RayQueryPipelineInst>()) {
                pipeline = static_cast<RayQueryPipelineInst *>(inst);
            }
        });
        expect(pipeline != nullptr);
        auto *callback = pipeline->on_surface_function();
        size_t named_block_count = 0u;
        size_t named_phi_count = 0u;
        size_t named_return_count = 0u;
        callback->definition()->traverse_basic_blocks(
            [&](BasicBlock *block) noexcept {
                named_block_count += block->name().has_value() ? 1u : 0u;
                block->traverse_instructions(
                    [&](Instruction *inst) noexcept {
                        if (inst->isa<PhiInst>() && inst->name() &&
                            *inst->name() == "surface_join_value") {
                            ++named_phi_count;
                        }
                        if (inst->isa<ReturnInst>() && inst->name() &&
                            *inst->name() == "surface_join_exit") {
                            ++named_return_count;
                        }
                    });
            });
        expect(named_block_count == 4u);
        expect(named_phi_count == 1u);
        expect(named_return_count == 1u);
        expect(xir_verify_module(&m).succeeded());
    };

    "module_rejection_is_atomic_across_functions"_test = [] {
        Module m;
        auto valid = make_fixture(m);
        XIRBuilder b;
        b.set_insertion_point(valid.surface);
        auto *valid_surface_exit = b.br(valid.dispatch);

        auto invalid = make_fixture(m);
        // The fixture deliberately leaves the surface block unterminated.
        auto function_count = count_functions(m);
        auto valid_block_count = count_blocks(valid.kernel->definition());
        auto invalid_block_count = count_blocks(invalid.kernel->definition());

        auto info = lower_ray_query_to_pipeline_pass_run_on_module(&m);
        expect(info.lowered_loop_count == 0u);
        expect(info.error_count == 1u);
        expect(!info.succeeded());
        expect(count_functions(m) == function_count);
        expect(count_blocks(valid.kernel->definition()) == valid_block_count);
        expect(count_blocks(invalid.kernel->definition()) == invalid_block_count);
        expect(valid.body->terminator() == valid.loop);
        expect(valid.dispatch->terminator() == valid.dispatch_inst);
        expect(valid.surface->terminator() == valid_surface_exit);
        expect(invalid.body->terminator() == invalid.loop);
        expect(invalid.dispatch->terminator() == invalid.dispatch_inst);
    };

    "null_module_and_function_are_noops"_test = [] {
        auto function_info =
            lower_ray_query_to_pipeline_pass_run_on_function(nullptr);
        auto module_info =
            lower_ray_query_to_pipeline_pass_run_on_module(nullptr);
        expect(function_info.lowered_loop_count == 0u);
        expect(function_info.error_count == 0u);
        expect(module_info.lowered_loop_count == 0u);
        expect(module_info.error_count == 0u);
    };

    "captured_callback_abi_is_exact_and_verifier_valid"_test = [] {
        Module m;
        auto *kernel = m.create_kernel();
        auto *value = kernel->create_value_argument(Type::of<int>());
        auto *body = kernel->create_body_block();
        XIRBuilder b;
        b.set_insertion_point(body);
        auto *query = b.alloca_local(Type::of<RayQueryAll>());
        auto *mutable_state = b.alloca_local(Type::of<int>());
        auto *loop = b.ray_query_loop();
        auto *dispatch = loop->create_dispatch_block();
        auto *merge = loop->create_merge_block();
        b.set_insertion_point(dispatch);
        auto *dispatch_inst = b.ray_query_dispatch(query);
        dispatch_inst->set_exit_block(merge);
        auto *surface = dispatch_inst->create_on_surface_candidate_block();
        auto *procedural = dispatch_inst->create_on_procedural_candidate_block();
        b.set_insertion_point(surface);
        b.store(mutable_state, value);
        b.br(dispatch);
        b.set_insertion_point(procedural);
        b.br(dispatch);
        b.set_insertion_point(merge);
        // Make the callback write observable. A write-only local is dead state
        // and may legitimately be localized instead of entering the ABI.
        b.load(Type::of<int>(), mutable_state);
        b.return_void();

        expect(xir_verify_module(&m).succeeded());
        size_t localized_alloca_count = 0u;
        auto info = lower_ray_query_to_pipeline_pass_run_on_function(
            kernel,
            {.localized_alloca_count = &localized_alloca_count});
        expect(info.succeeded());
        expect(info.lowered_loop_count == 1u);
        expect(localized_alloca_count == 0u);
        expect(info.error_count == 0u);

        RayQueryPipelineInst *pipeline = nullptr;
        body->traverse_instructions([&](Instruction *inst) noexcept {
            if (inst->isa<RayQueryPipelineInst>()) {
                expect(pipeline == nullptr);
                pipeline = static_cast<RayQueryPipelineInst *>(inst);
            }
        });
        expect(pipeline != nullptr);
        if (pipeline == nullptr) { return; }
        expect(pipeline->query_object() == query);
        expect(pipeline->captured_argument_count() == 2u);
        expect(pipeline->captured_argument(0u) == mutable_state);
        expect(pipeline->captured_argument(1u) == value);
        for (auto *callback : {pipeline->on_surface_function(),
                               pipeline->on_procedural_function()}) {
            expect(callback != nullptr);
            expect(callback->isa<CallableFunction>());
            expect(callback->type() == nullptr);
            expect(callback->arguments().count_size() == 3u);
            auto argument = callback->arguments().begin();
            auto *query_argument = *argument;
            ++argument;
            auto *mutable_argument = *argument;
            ++argument;
            auto *value_argument = *argument;
            expect(query_argument->is_reference());
            expect(query_argument->type() == Type::of<RayQueryAll>());
            expect(mutable_argument->is_reference());
            expect(mutable_argument->type() == Type::of<int>());
            expect(value_argument->is_value());
            expect(value_argument->type() == Type::of<int>());
        }
        expect(count_functions(m) == 3u);
        expect(xir_verify_module(&m).succeeded());
    };

    "definitely_initialized_handler_scratch_is_not_captured"_test = [] {
        Module m;
        auto f = make_fixture(m);
        auto *value = f.kernel->create_value_argument(Type::of<int>());
        XIRBuilder b;
        b.set_insertion_point(f.loop->prev());
        auto *scratch = b.alloca_local(Type::of<int>());
        scratch->set_name("surface_invocation_scratch");
        auto *persistent = b.alloca_local(Type::of<int>());
        persistent->set_name("cross_candidate_state");
        // This default definition is killed by the handler's unconditional
        // store and must not force scratch into the callback environment.
        b.store(scratch, m.create_constant_zero(Type::of<int>()));
        b.set_insertion_point(f.surface);
        b.store(scratch, value);
        auto *reloaded = b.load(Type::of<int>(), scratch);
        b.store(persistent, reloaded);
        b.br(f.dispatch);
        // An observation after the loop makes persistent state a real capture;
        // scratch remains wholly internal to one surface-handler invocation.
        b.set_insertion_point(f.merge->terminator()->prev());
        b.load(Type::of<int>(), persistent);

        expect(xir_verify_module(&m).succeeded());
        auto info = lower_ray_query_loop_pass_run_on_function(f.kernel);
        expect(info.succeeded());
        expect(info.lowered_loop_count == 1u);
        expect(info.localized_alloca_count == 1u);

        RayQueryPipelineInst *pipeline = nullptr;
        f.body->traverse_instructions([&](Instruction *inst) noexcept {
            if (inst->isa<RayQueryPipelineInst>()) {
                pipeline = static_cast<RayQueryPipelineInst *>(inst);
            }
        });
        expect(pipeline != nullptr);
        if (pipeline == nullptr) { return; }
        expect(pipeline->captured_argument_count() == 2u);
        auto captures_scratch = false;
        auto captures_persistent = false;
        for (auto i = 0u; i < pipeline->captured_argument_count(); ++i) {
            auto *captured = pipeline->captured_argument(i);
            captures_scratch |= captured == scratch;
            captures_persistent |= captured == persistent;
        }
        expect(!captures_scratch);
        expect(captures_persistent);

        auto *surface = pipeline->on_surface_function();
        size_t localized_count = 0u;
        surface->definition()->traverse_basic_blocks(
            [&](BasicBlock *block) noexcept {
                block->traverse_instructions(
                    [&](Instruction *inst) noexcept {
                        if (inst->isa<AllocaInst>() && inst->name() &&
                            *inst->name() == "surface_invocation_scratch") {
                            ++localized_count;
                        }
                    });
            });
        expect(localized_count == 1u);
        expect(xir_verify_module(&m).succeeded());
    };

    "large_handler_scratch_ignores_unrelated_instruction_volume"_test = [] {
        Module m;
        auto f = make_fixture(m);
        auto *seed = f.kernel->create_value_argument(Type::of<int>());
        auto *large_type = Type::array(Type::of<int>(), 4096u);
        XIRBuilder b;
        b.set_insertion_point(f.loop->prev());
        auto *scratch = b.alloca_local(large_type);
        scratch->set_name("large_surface_invocation_scratch");

        b.set_insertion_point(f.surface);
        b.store(scratch, m.create_constant_zero(large_type));
        b.load(large_type, scratch);
        // This is a complexity regression, not dead-code decoration: the
        // scratch proof runs before the lowering pass's DCE. Instructions
        // unrelated to `scratch` denote the identity path effect and must not
        // allocate or clear two 4096-bit aggregate masks apiece.
        Value *noise = seed;
        auto *one = m.create_constant_one(Type::of<int>());
        for (auto i = 0u; i < 8192u; ++i) {
            noise = b.call(
                Type::of<int>(), ArithmeticOp::BINARY_ADD,
                {noise, one});
        }
        static_cast<void>(noise);
        b.br(f.dispatch);

        expect(xir_verify_module(&m).succeeded());
        auto info = lower_ray_query_loop_pass_run_on_function(f.kernel);
        expect(info.succeeded());
        expect(info.lowered_loop_count == 1u);
        expect(info.localized_alloca_count == 1u);
        expect(xir_verify_module(&m).succeeded());
    };

    "independently_initialized_cross_handler_scratch_is_duplicated"_test = [] {
        Module m;
        auto f = make_fixture(m);
        XIRBuilder b;
        b.set_insertion_point(f.loop->prev());
        auto *scratch = b.alloca_local(Type::of<float3>());
        scratch->set_name("cross_handler_invocation_scratch");
        b.store(scratch, m.create_constant_zero(Type::of<float3>()));

        b.set_insertion_point(f.surface);
        b.store(scratch, m.create_constant_one(Type::of<float3>()));
        b.load(Type::of<float3>(), scratch);
        b.br(f.dispatch);
        b.set_insertion_point(f.procedural->terminator()->prev());
        b.store(scratch, m.create_constant_zero(Type::of<float3>()));
        b.load(Type::of<float3>(), scratch);

        expect(xir_verify_module(&m).succeeded());
        auto info = lower_ray_query_loop_pass_run_on_function(f.kernel);
        expect(info.succeeded());
        expect(info.lowered_loop_count == 1u);
        // One source object is removed from the ABI and recreated once in
        // each independently invoked candidate handler.
        expect(info.localized_alloca_count == 1u);

        RayQueryPipelineInst *pipeline = nullptr;
        f.body->traverse_instructions([&](Instruction *inst) noexcept {
            if (inst->isa<RayQueryPipelineInst>()) {
                pipeline = static_cast<RayQueryPipelineInst *>(inst);
            }
        });
        expect(pipeline != nullptr);
        if (pipeline == nullptr) { return; }
        auto captures_scratch = false;
        for (auto i = 0u; i < pipeline->captured_argument_count(); ++i) {
            captures_scratch |= pipeline->captured_argument(i) == scratch;
        }
        expect(!captures_scratch);
        for (auto *handler : {pipeline->on_surface_function(),
                              pipeline->on_procedural_function()}) {
            auto localized_count = 0u;
            handler->definition()->traverse_instructions(
                [&](Instruction *inst) noexcept {
                    if (inst->isa<AllocaInst>() && inst->name() &&
                        *inst->name() ==
                            "cross_handler_invocation_scratch") {
                        ++localized_count;
                    }
                });
            expect(localized_count == 1u);
        }
        expect(xir_verify_module(&m).succeeded());
    };

    "cross_handler_scratch_with_one_incoming_read_stays_captured"_test = [] {
        Module m;
        auto f = make_fixture(m);
        XIRBuilder b;
        b.set_insertion_point(f.loop->prev());
        auto *scratch = b.alloca_local(Type::of<float3>());
        scratch->set_name("cross_handler_persistent_state");
        b.store(scratch, m.create_constant_zero(Type::of<float3>()));

        b.set_insertion_point(f.surface);
        b.store(scratch, m.create_constant_one(Type::of<float3>()));
        b.load(Type::of<float3>(), scratch);
        b.br(f.dispatch);
        b.set_insertion_point(f.procedural->terminator()->prev());
        // This read can observe the surface handler's previous candidate and
        // therefore forbids replacing the shared lifetime by two locals.
        b.load(Type::of<float3>(), scratch);

        expect(xir_verify_module(&m).succeeded());
        auto info = lower_ray_query_loop_pass_run_on_function(f.kernel);
        expect(info.succeeded());
        expect(info.lowered_loop_count == 1u);
        expect(info.localized_alloca_count == 0u);
        RayQueryPipelineInst *pipeline = nullptr;
        f.body->traverse_instructions([&](Instruction *inst) noexcept {
            if (inst->isa<RayQueryPipelineInst>()) {
                pipeline = static_cast<RayQueryPipelineInst *>(inst);
            }
        });
        expect(pipeline != nullptr);
        if (pipeline != nullptr) {
            auto captures_scratch = false;
            for (auto i = 0u; i < pipeline->captured_argument_count(); ++i) {
                captures_scratch |=
                    pipeline->captured_argument(i) == scratch;
            }
            expect(captures_scratch);
        }
        expect(xir_verify_module(&m).succeeded());
    };

    "whole_aggregate_store_localizes_subfield_reads"_test = [] {
        Module m;
        auto f = make_fixture(m);
        auto *value =
            f.kernel->create_value_argument(Type::of<float3>());
        XIRBuilder b;
        b.set_insertion_point(f.loop->prev());
        auto *scratch = b.alloca_local(Type::of<float3>());
        scratch->set_name("surface_aggregate_scratch");
        auto *observed = b.alloca_local(Type::of<float>());
        b.store(
            scratch,
            m.create_constant_zero(Type::of<float3>()));
        b.set_insertion_point(f.surface);
        b.store(scratch, value);
        auto *x = b.gep(
            Type::of<float>(), scratch,
            {m.create_constant_zero(Type::of<uint>())});
        b.store(observed, b.load(Type::of<float>(), x));
        b.br(f.dispatch);
        b.set_insertion_point(f.merge->terminator()->prev());
        b.load(Type::of<float>(), observed);

        expect(xir_verify_module(&m).succeeded());
        auto info = lower_ray_query_loop_pass_run_on_function(f.kernel);
        expect(info.succeeded());
        expect(info.lowered_loop_count == 1u);
        expect(info.localized_alloca_count == 1u);

        RayQueryPipelineInst *pipeline = nullptr;
        f.body->traverse_instructions([&](Instruction *inst) noexcept {
            if (inst->isa<RayQueryPipelineInst>()) {
                pipeline = static_cast<RayQueryPipelineInst *>(inst);
            }
        });
        expect(pipeline != nullptr);
        if (pipeline == nullptr) { return; }
        auto captures_scratch = false;
        for (auto i = 0u; i < pipeline->captured_argument_count(); ++i) {
            captures_scratch |=
                pipeline->captured_argument(i) == scratch;
        }
        expect(!captures_scratch);

        AllocaInst *localized = nullptr;
        GEPInst *localized_x = nullptr;
        pipeline->on_surface_function()
            ->definition()
            ->traverse_basic_blocks([&](BasicBlock *block) noexcept {
                block->traverse_instructions(
                    [&](Instruction *inst) noexcept {
                        if (inst->isa<AllocaInst>() && inst->name() &&
                            *inst->name() ==
                                "surface_aggregate_scratch") {
                            localized =
                                static_cast<AllocaInst *>(inst);
                        }
                        if (inst->isa<GEPInst>() &&
                            inst->type() == Type::of<float>()) {
                            localized_x = static_cast<GEPInst *>(inst);
                        }
                    });
            });
        expect(localized != nullptr);
        expect(localized_x != nullptr);
        if (localized_x != nullptr) {
            expect(localized_x->base() == localized);
        }
        expect(xir_verify_module(&m).succeeded());
    };

    "partial_aggregate_store_does_not_prove_other_fields"_test = [] {
        Module m;
        auto f = make_fixture(m);
        auto *value =
            f.kernel->create_value_argument(Type::of<float>());
        XIRBuilder b;
        b.set_insertion_point(f.loop->prev());
        auto *scratch = b.alloca_local(Type::of<float3>());
        auto *observed = b.alloca_local(Type::of<float>());
        b.store(
            scratch,
            m.create_constant_zero(Type::of<float3>()));
        b.set_insertion_point(f.surface);
        auto *zero = m.create_constant_zero(Type::of<uint>());
        auto *one = m.create_constant_one(Type::of<uint>());
        auto *x = b.gep(Type::of<float>(), scratch, {zero});
        auto *y = b.gep(Type::of<float>(), scratch, {one});
        b.store(x, value);
        b.store(observed, b.load(Type::of<float>(), y));
        b.br(f.dispatch);
        b.set_insertion_point(f.merge->terminator()->prev());
        b.load(Type::of<float>(), observed);

        expect(xir_verify_module(&m).succeeded());
        auto info = lower_ray_query_loop_pass_run_on_function(f.kernel);
        expect(info.succeeded());
        expect(info.lowered_loop_count == 1u);
        expect(info.localized_alloca_count == 0u);
        RayQueryPipelineInst *pipeline = nullptr;
        f.body->traverse_instructions([&](Instruction *inst) noexcept {
            if (inst->isa<RayQueryPipelineInst>()) {
                pipeline = static_cast<RayQueryPipelineInst *>(inst);
            }
        });
        expect(pipeline != nullptr);
        if (pipeline != nullptr) {
            auto captures_scratch = false;
            for (auto i = 0u;
                 i < pipeline->captured_argument_count(); ++i) {
                captures_scratch |=
                    pipeline->captured_argument(i) == scratch;
            }
            expect(captures_scratch);
        }
        expect(xir_verify_module(&m).succeeded());
    };

    "component_initialized_scratch_through_callable_is_localized"_test = [] {
        Module m;
        auto f = make_fixture(m);
        auto *helper = m.create_callable(nullptr);
        auto *argument =
            helper->create_reference_argument(Type::of<float3>());
        XIRBuilder b;
        b.set_insertion_point(helper->create_body_block());
        auto *zero = m.create_constant_zero(Type::of<uint>());
        auto *one = m.create_constant_one(Type::of<uint>());
        const uint two_value = 2u;
        auto *two = m.create_constant(Type::of<uint>(), &two_value);
        auto *x = b.gep(Type::of<float>(), argument, {zero});
        auto *y = b.gep(Type::of<float>(), argument, {one});
        auto *z = b.gep(Type::of<float>(), argument, {two});
        b.store(x, m.create_constant_zero(Type::of<float>()));
        b.store(y, m.create_constant_zero(Type::of<float>()));
        b.store(z, m.create_constant_zero(Type::of<float>()));
        b.load(Type::of<float>(), x);
        b.load(Type::of<float>(), y);
        b.load(Type::of<float>(), z);
        b.return_void();

        b.set_insertion_point(f.loop->prev());
        auto *scratch = b.alloca_local(Type::of<float3>());
        scratch->set_name("interprocedural_component_scratch");
        // This incoming value is deliberately observable only if the helper
        // can read a field before its own per-invocation definitions.
        b.store(scratch, m.create_constant_zero(Type::of<float3>()));
        b.set_insertion_point(f.surface);
        b.call(nullptr, helper, {scratch});
        b.br(f.dispatch);

        expect(xir_verify_module(&m).succeeded());
        auto info = lower_ray_query_loop_pass_run_on_function(f.kernel);
        expect(info.succeeded());
        expect(info.lowered_loop_count == 1u);
        expect(info.localized_alloca_count == 1u);

        RayQueryPipelineInst *pipeline = nullptr;
        f.body->traverse_instructions([&](Instruction *inst) noexcept {
            if (inst->isa<RayQueryPipelineInst>()) {
                pipeline = static_cast<RayQueryPipelineInst *>(inst);
            }
        });
        expect(pipeline != nullptr);
        if (pipeline == nullptr) { return; }
        auto captures_scratch = false;
        for (auto i = 0u; i < pipeline->captured_argument_count(); ++i) {
            captures_scratch |= pipeline->captured_argument(i) == scratch;
        }
        expect(!captures_scratch);

        AllocaInst *localized = nullptr;
        CallInst *localized_call = nullptr;
        pipeline->on_surface_function()
            ->definition()
            ->traverse_basic_blocks([&](BasicBlock *block) noexcept {
                block->traverse_instructions(
                    [&](Instruction *inst) noexcept {
                        if (inst->isa<AllocaInst>() && inst->name() &&
                            *inst->name() ==
                                "interprocedural_component_scratch") {
                            localized = static_cast<AllocaInst *>(inst);
                        }
                        if (inst->isa<CallInst>() &&
                            static_cast<CallInst *>(inst)->callee() == helper) {
                            localized_call = static_cast<CallInst *>(inst);
                        }
                    });
            });
        expect(localized != nullptr);
        expect(localized_call != nullptr);
        if (localized_call != nullptr) {
            expect(localized_call->argument(0u) == localized);
        }
        expect(xir_verify_module(&m).succeeded());
    };

    "callable_read_of_uninitialized_component_stays_captured"_test = [] {
        Module m;
        auto f = make_fixture(m);
        auto *helper = m.create_callable(nullptr);
        auto *argument =
            helper->create_reference_argument(Type::of<float3>());
        XIRBuilder b;
        b.set_insertion_point(helper->create_body_block());
        auto *zero = m.create_constant_zero(Type::of<uint>());
        auto *one = m.create_constant_one(Type::of<uint>());
        auto *x = b.gep(Type::of<float>(), argument, {zero});
        auto *y = b.gep(Type::of<float>(), argument, {one});
        b.store(x, m.create_constant_zero(Type::of<float>()));
        // y still observes the value carried between candidate invocations.
        b.load(Type::of<float>(), y);
        b.return_void();

        b.set_insertion_point(f.loop->prev());
        auto *state = b.alloca_local(Type::of<float3>());
        b.store(state, m.create_constant_zero(Type::of<float3>()));
        b.set_insertion_point(f.surface);
        b.call(nullptr, helper, {state});
        b.br(f.dispatch);

        expect(xir_verify_module(&m).succeeded());
        auto info = lower_ray_query_loop_pass_run_on_function(f.kernel);
        expect(info.succeeded());
        expect(info.lowered_loop_count == 1u);
        expect(info.localized_alloca_count == 0u);
        RayQueryPipelineInst *pipeline = nullptr;
        f.body->traverse_instructions([&](Instruction *inst) noexcept {
            if (inst->isa<RayQueryPipelineInst>()) {
                pipeline = static_cast<RayQueryPipelineInst *>(inst);
            }
        });
        expect(pipeline != nullptr);
        if (pipeline != nullptr) {
            auto captures_state = false;
            for (auto i = 0u; i < pipeline->captured_argument_count(); ++i) {
                captures_state |= pipeline->captured_argument(i) == state;
            }
            expect(captures_state);
        }
        expect(xir_verify_module(&m).succeeded());
    };

    "load_before_store_keeps_cross_candidate_state_captured"_test = [] {
        Module m;
        auto f = make_fixture(m);
        XIRBuilder b;
        b.set_insertion_point(f.loop->prev());
        auto *state = b.alloca_local(Type::of<int>());
        b.set_insertion_point(f.surface);
        auto *previous = b.load(Type::of<int>(), state);
        b.store(state, previous);
        b.br(f.dispatch);

        expect(xir_verify_module(&m).succeeded());
        auto info = lower_ray_query_loop_pass_run_on_function(f.kernel);
        expect(info.succeeded());
        expect(info.localized_alloca_count == 0u);
        RayQueryPipelineInst *pipeline = nullptr;
        f.body->traverse_instructions([&](Instruction *inst) noexcept {
            if (inst->isa<RayQueryPipelineInst>()) {
                pipeline = static_cast<RayQueryPipelineInst *>(inst);
            }
        });
        expect(pipeline != nullptr);
        if (pipeline != nullptr) {
            expect(pipeline->captured_argument_count() == 1u);
            expect(pipeline->captured_argument(0u) == state);
        }
        expect(xir_verify_module(&m).succeeded());
    };

    "conditional_store_does_not_prove_handler_local_initialization"_test = [] {
        Module m;
        auto f = make_fixture(m);
        auto *condition = f.kernel->create_value_argument(Type::of<bool>());
        auto *stored = f.kernel->create_basic_block();
        auto *unstored = f.kernel->create_basic_block();
        auto *join = f.kernel->create_basic_block();
        XIRBuilder b;
        b.set_insertion_point(f.loop->prev());
        auto *state = b.alloca_local(Type::of<int>());
        // The original program is initialized, but one handler path observes
        // this outside definition. Recreating state inside the callback would
        // therefore be invalid unless the conditional store becomes a must.
        b.store(state, m.create_constant_zero(Type::of<int>()));
        b.set_insertion_point(f.surface);
        b.cond_br(condition, stored, unstored);
        b.set_insertion_point(stored);
        b.store(state, m.create_constant_one(Type::of<int>()));
        b.br(join);
        b.set_insertion_point(unstored);
        b.br(join);
        b.set_insertion_point(join);
        auto *value = b.load(Type::of<int>(), state);
        b.store(state, value);
        b.br(f.dispatch);

        expect(xir_verify_module(&m).succeeded());
        auto info = lower_ray_query_loop_pass_run_on_function(f.kernel);
        expect(info.succeeded());
        expect(info.localized_alloca_count == 0u);
        RayQueryPipelineInst *pipeline = nullptr;
        f.body->traverse_instructions([&](Instruction *inst) noexcept {
            if (inst->isa<RayQueryPipelineInst>()) {
                pipeline = static_cast<RayQueryPipelineInst *>(inst);
            }
        });
        expect(pipeline != nullptr);
        if (pipeline != nullptr) {
            expect(pipeline->captured_argument_count() == 2u);
            auto captures_state = false;
            for (auto i = 0u; i < pipeline->captured_argument_count(); ++i) {
                auto *captured = pipeline->captured_argument(i);
                captures_state |= captured == state;
            }
            expect(captures_state);
        }
        expect(xir_verify_module(&m).succeeded());
    };

    "stores_on_every_diamond_arm_prove_handler_local_initialization"_test = [] {
        Module m;
        auto f = make_fixture(m);
        auto *condition = f.kernel->create_value_argument(Type::of<bool>());
        auto *left = f.kernel->create_basic_block();
        auto *right = f.kernel->create_basic_block();
        auto *join = f.kernel->create_basic_block();
        XIRBuilder b;
        b.set_insertion_point(f.loop->prev());
        auto *scratch = b.alloca_local(Type::of<int>());
        auto *observed = b.alloca_local(Type::of<int>());
        b.store(scratch, m.create_constant_zero(Type::of<int>()));
        b.set_insertion_point(f.surface);
        b.cond_br(condition, left, right);
        b.set_insertion_point(left);
        b.store(scratch, m.create_constant_one(Type::of<int>()));
        b.br(join);
        b.set_insertion_point(right);
        b.store(scratch, m.create_constant_zero(Type::of<int>()));
        b.br(join);
        b.set_insertion_point(join);
        b.store(observed, b.load(Type::of<int>(), scratch));
        b.br(f.dispatch);
        b.set_insertion_point(f.merge->terminator()->prev());
        b.load(Type::of<int>(), observed);

        expect(xir_verify_module(&m).succeeded());
        auto info = lower_ray_query_loop_pass_run_on_function(f.kernel);
        expect(info.succeeded());
        expect(info.lowered_loop_count == 1u);
        expect(info.localized_alloca_count == 1u);
        RayQueryPipelineInst *pipeline = nullptr;
        f.body->traverse_instructions([&](Instruction *inst) noexcept {
            if (inst->isa<RayQueryPipelineInst>()) {
                pipeline = static_cast<RayQueryPipelineInst *>(inst);
            }
        });
        expect(pipeline != nullptr);
        if (pipeline != nullptr) {
            auto captures_scratch = false;
            auto captures_observed = false;
            for (auto i = 0u; i < pipeline->captured_argument_count(); ++i) {
                captures_scratch |= pipeline->captured_argument(i) == scratch;
                captures_observed |= pipeline->captured_argument(i) == observed;
            }
            expect(!captures_scratch);
            expect(captures_observed);
        }
        expect(xir_verify_module(&m).succeeded());
    };

    "localized_handler_scratch_does_not_consume_capture_budget"_test = [] {
        Module m;
        auto f = make_fixture(m);
        XIRBuilder b;
        b.set_insertion_point(f.loop->prev());
        auto *scratch = b.alloca_local(Type::of<int>());
        scratch->set_name("capture_budget_local_scratch");
        b.store(scratch, m.create_constant_zero(Type::of<int>()));
        b.set_insertion_point(f.surface);
        b.store(scratch, m.create_constant_one(Type::of<int>()));
        b.load(Type::of<int>(), scratch);
        b.br(f.dispatch);

        expect(xir_verify_module(&m).succeeded());
        size_t localized_alloca_count = 0u;
        size_t skipped_loop_count = 0u;
        auto info = lower_ray_query_to_pipeline_pass_run_on_function(
            f.kernel,
            {.max_captured_argument_count = 0u,
             .skipped_loop_count = &skipped_loop_count,
             .localized_alloca_count = &localized_alloca_count});

        expect(info.succeeded());
        expect(info.lowered_loop_count == 1u);
        expect(localized_alloca_count == 1u);
        expect(skipped_loop_count == 0u);
        RayQueryPipelineInst *pipeline = nullptr;
        f.body->traverse_instructions([&](Instruction *inst) noexcept {
            if (inst->isa<RayQueryPipelineInst>()) {
                pipeline = static_cast<RayQueryPipelineInst *>(inst);
            }
        });
        expect(pipeline != nullptr);
        if (pipeline != nullptr) {
            expect(pipeline->captured_argument_count() == 0u);
            auto localized_count = 0u;
            pipeline->on_surface_function()
                ->definition()
                ->traverse_instructions(
                    [&](Instruction *inst) noexcept {
                        if (inst->isa<AllocaInst>() && inst->name() &&
                            *inst->name() ==
                                "capture_budget_local_scratch") {
                            localized_count++;
                        }
                    });
            expect(localized_count == 1u);
        }
        expect(xir_verify_module(&m).succeeded());
    };

    "capture_bound_selectively_retains_captured_loops"_test = [] {
        Module m;
        auto capture_free = make_fixture(m);
        XIRBuilder b;
        b.set_insertion_point(capture_free.surface);
        b.br(capture_free.dispatch);

        auto captured = make_fixture(m);
        auto *value =
            captured.kernel->create_value_argument(Type::of<int>());
        auto *state = [&] {
            b.set_insertion_point(captured.body->instructions().front());
            return b.alloca_local(Type::of<int>());
        }();
        b.set_insertion_point(captured.surface);
        b.store(state, value);
        b.br(captured.dispatch);

        auto function_count = count_functions(m);
        expect(xir_verify_module(&m).succeeded());
        size_t skipped_loop_count = 0u;
        auto info = lower_ray_query_to_pipeline_pass_run_on_module(
            &m, nullptr,
            {.max_captured_argument_count = 0u,
             .skipped_loop_count = &skipped_loop_count});
        expect(info.succeeded());
        expect(info.error_count == 0u);
        expect(info.lowered_loop_count == 1u);
        expect(skipped_loop_count == 1u);
        expect(count_functions(m) == function_count + 2u);
        expect(capture_free.body->terminator() == nullptr ||
               !capture_free.body->terminator()->isa<RayQueryLoopInst>());
        expect(captured.body->terminator() == captured.loop);
        expect(captured.dispatch->terminator() == captured.dispatch_inst);
        expect(xir_verify_module(&m).succeeded());
    };

    "profitability_filter_retains_one_small_handler_loop"_test = [] {
        Module m;
        auto f = make_fixture(m);
        XIRBuilder b;
        b.set_insertion_point(f.surface);
        b.br(f.dispatch);
        auto function_count = count_functions(m);
        size_t skipped_loop_count = 0u;

        auto info = lower_ray_query_to_pipeline_pass_run_on_function(
            f.kernel,
            {.min_handler_instruction_count = 3u,
             .min_small_handler_loop_count = 2u,
             .skipped_loop_count = &skipped_loop_count});

        expect(info.succeeded());
        expect(info.lowered_loop_count == 0u);
        expect(skipped_loop_count == 1u);
        expect(count_functions(m) == function_count);
        expect(f.body->terminator() == f.loop);
        expect(xir_verify_module(&m).succeeded());
    };

    "profitability_filter_lowers_one_large_handler_loop"_test = [] {
        Module m;
        auto f = make_fixture(m);
        XIRBuilder b;
        b.set_insertion_point(f.surface);
        static_cast<void>(b.alloca_local(Type::of<int>()));
        b.br(f.dispatch);
        auto function_count = count_functions(m);
        size_t skipped_loop_count = 0u;

        auto info = lower_ray_query_to_pipeline_pass_run_on_function(
            f.kernel,
            {.min_handler_instruction_count = 3u,
             .min_small_handler_loop_count = 2u,
             .skipped_loop_count = &skipped_loop_count});

        expect(info.succeeded());
        expect(info.lowered_loop_count == 1u);
        expect(skipped_loop_count == 0u);
        expect(count_functions(m) == function_count + 2u);
        expect(xir_verify_module(&m).succeeded());
    };

    "profitability_filter_does_not_batch_across_functions"_test = [] {
        Module m;
        auto first = make_fixture(m);
        auto second = make_fixture(m);
        XIRBuilder b;
        b.set_insertion_point(first.surface);
        b.br(first.dispatch);
        b.set_insertion_point(second.surface);
        b.br(second.dispatch);
        auto function_count = count_functions(m);
        size_t skipped_loop_count = 0u;

        auto info = lower_ray_query_to_pipeline_pass_run_on_module(
            &m, nullptr,
            {.min_handler_instruction_count = 3u,
             .min_small_handler_loop_count = 2u,
             .skipped_loop_count = &skipped_loop_count});

        expect(info.succeeded());
        expect(info.lowered_loop_count == 0u);
        expect(skipped_loop_count == 2u);
        expect(count_functions(m) == function_count);
        expect(first.body->terminator() == first.loop);
        expect(second.body->terminator() == second.loop);
        expect(xir_verify_module(&m).succeeded());
    };

    "profitability_filter_batches_two_small_handler_loops"_test = [] {
        Module m;
        auto first = make_fixture(m);
        XIRBuilder b;
        b.set_insertion_point(first.surface);
        b.br(first.dispatch);
        first.merge->terminator()->remove_self();

        b.set_insertion_point(first.merge);
        auto *query = b.alloca_local(Type::of<RayQueryAll>());
        auto *second = b.ray_query_loop();
        auto *dispatch = second->create_dispatch_block();
        auto *merge = second->create_merge_block();
        b.set_insertion_point(dispatch);
        auto *dispatch_inst = b.ray_query_dispatch(query);
        dispatch_inst->set_exit_block(merge);
        auto *surface =
            dispatch_inst->create_on_surface_candidate_block();
        auto *procedural =
            dispatch_inst->create_on_procedural_candidate_block();
        b.set_insertion_point(surface);
        b.br(dispatch);
        b.set_insertion_point(procedural);
        b.br(dispatch);
        b.set_insertion_point(merge);
        b.return_void();
        auto function_count = count_functions(m);
        size_t skipped_loop_count = 0u;
        expect(xir_verify_module(&m).succeeded());

        auto info = lower_ray_query_to_pipeline_pass_run_on_function(
            first.kernel,
            {.min_handler_instruction_count = 3u,
             .min_small_handler_loop_count = 2u,
             .skipped_loop_count = &skipped_loop_count});

        expect(info.succeeded());
        expect(info.lowered_loop_count == 2u);
        expect(skipped_loop_count == 0u);
        expect(count_functions(m) == function_count + 4u);
        expect(xir_verify_module(&m).succeeded());
    };

    "multiple_handler_exits_are_outlined_to_returns"_test = [] {
        Module m;
        auto f = make_fixture(m);
        auto *cond = f.kernel->create_value_argument(Type::of<bool>());
        auto *left = f.kernel->create_basic_block();
        auto *right = f.kernel->create_basic_block();
        XIRBuilder b;
        b.set_insertion_point(f.surface);
        b.cond_br(cond, left, right);
        b.set_insertion_point(left);
        b.br(f.dispatch);
        b.set_insertion_point(right);
        b.br(f.dispatch);
        auto function_count = count_functions(m);

        auto info = lower_ray_query_to_pipeline_pass_run_on_function(f.kernel);
        expect(info.lowered_loop_count == 1u);
        expect(info.error_count == 0u);
        expect(info.succeeded());
        expect(count_functions(m) == function_count + 2u);
        RayQueryPipelineInst *pipeline = nullptr;
        f.body->traverse_instructions([&](Instruction *inst) noexcept {
            if (inst->isa<RayQueryPipelineInst>()) {
                pipeline = static_cast<RayQueryPipelineInst *>(inst);
            }
        });
        expect(pipeline != nullptr);
        if (pipeline == nullptr) { return; }
        auto *surface_callback = pipeline->on_surface_function();
        expect(surface_callback->arguments().count_size() == 2u);
        size_t return_count = 0u;
        surface_callback->definition()->traverse_instructions(
            [&](Instruction *inst) noexcept {
                return_count += inst->isa<ReturnInst>() ? 1u : 0u;
            });
        expect(return_count == 2u);
        expect(xir_verify_module(&m).succeeded());
    };

    "shared_handler_tail_with_phi_is_rejected_before_outlining"_test = [] {
        Module m;
        auto f = make_fixture(m);
        auto *shared_tail = f.kernel->create_basic_block();
        auto *zero = m.create_constant_zero(Type::of<int>());
        auto *one = m.create_constant_one(Type::of<int>());
        XIRBuilder b;
        b.set_insertion_point(f.surface);
        auto *surface_exit = b.br(shared_tail);
        // Replace the fixture's original procedural exit with the shared tail.
        f.procedural->terminator()->remove_self();
        b.set_insertion_point(f.procedural);
        auto *procedural_exit = b.br(shared_tail);
        b.set_insertion_point(shared_tail);
        auto *join_phi = b.phi(Type::of<int>());
        join_phi->add_incoming(zero, f.surface);
        join_phi->add_incoming(one, f.procedural);
        auto *tail_exit = b.br(f.dispatch);
        auto function_count = count_functions(m);
        auto block_count = count_blocks(f.kernel->definition());

        auto info = lower_ray_query_to_pipeline_pass_run_on_function(f.kernel);
        expect(info.lowered_loop_count == 0u);
        expect(info.error_count == 1u);
        expect(!info.succeeded());
        expect(count_functions(m) == function_count);
        expect(count_blocks(f.kernel->definition()) == block_count);
        expect(f.body->terminator() == f.loop);
        expect(f.dispatch->terminator() == f.dispatch_inst);
        expect(f.surface->terminator() == surface_exit);
        expect(f.procedural->terminator() == procedural_exit);
        expect(shared_tail->terminator() == tail_exit);
        expect(join_phi->is_linked());
        expect(join_phi->incoming_count() == 2u);
        expect(join_phi->incoming(0u).block == f.surface);
        expect(join_phi->incoming(1u).block == f.procedural);
    };

    "invalid_later_loop_keeps_earlier_valid_loop_unchanged"_test = [] {
        Module m;
        auto *kernel = m.create_kernel();
        auto *body = kernel->create_body_block();
        XIRBuilder b;

        // First loop is fully valid and would be outlined without a
        // function-wide preflight.
        b.set_insertion_point(body);
        auto *query0 = b.alloca_local(Type::of<RayQueryAll>());
        auto *loop0 = b.ray_query_loop();
        auto *dispatch0 = loop0->create_dispatch_block();
        auto *merge0 = loop0->create_merge_block();
        b.set_insertion_point(dispatch0);
        auto *dispatch_inst0 = b.ray_query_dispatch(query0);
        dispatch_inst0->set_exit_block(merge0);
        auto *surface0 = dispatch_inst0->create_on_surface_candidate_block();
        auto *procedural0 = dispatch_inst0->create_on_procedural_candidate_block();
        b.set_insertion_point(surface0);
        auto *surface_exit0 = b.br(dispatch0);
        b.set_insertion_point(procedural0);
        auto *procedural_exit0 = b.br(dispatch0);

        // The later loop bypasses candidate dispatch and reaches the loop
        // merge directly. It must reject the complete function before the
        // first callback or alloca move is created.
        b.set_insertion_point(merge0);
        auto *query1 = b.alloca_local(Type::of<RayQueryAll>());
        auto *loop1 = b.ray_query_loop();
        auto *dispatch1 = loop1->create_dispatch_block();
        auto *merge1 = loop1->create_merge_block();
        b.set_insertion_point(dispatch1);
        auto *dispatch_inst1 = b.ray_query_dispatch(query1);
        dispatch_inst1->set_exit_block(merge1);
        auto *surface1 = dispatch_inst1->create_on_surface_candidate_block();
        auto *procedural1 = dispatch_inst1->create_on_procedural_candidate_block();
        auto *left = kernel->create_basic_block();
        auto *right = kernel->create_basic_block();
        auto *cond = kernel->create_value_argument(Type::of<bool>());
        b.set_insertion_point(surface1);
        auto *split = b.cond_br(cond, left, right);
        b.set_insertion_point(left);
        auto *left_exit = b.br(dispatch1);
        b.set_insertion_point(right);
        auto *right_exit = b.br(merge1);
        b.set_insertion_point(procedural1);
        auto *procedural_exit1 = b.br(dispatch1);
        b.set_insertion_point(merge1);
        b.return_void();
        auto function_count = count_functions(m);
        auto block_count = count_blocks(kernel->definition());

        auto info = lower_ray_query_to_pipeline_pass_run_on_function(kernel);
        expect(info.lowered_loop_count == 0u);
        expect(info.error_count == 1u);
        expect(!info.succeeded());
        expect(count_functions(m) == function_count);
        expect(count_blocks(kernel->definition()) == block_count);
        expect(body->terminator() == loop0);
        expect(dispatch0->terminator() == dispatch_inst0);
        expect(surface0->terminator() == surface_exit0);
        expect(procedural0->terminator() == procedural_exit0);
        expect(merge0->terminator() == loop1);
        expect(dispatch1->terminator() == dispatch_inst1);
        expect(surface1->terminator() == split);
        expect(left->terminator() == left_exit);
        expect(right->terminator() == right_exit);
        expect(procedural1->terminator() == procedural_exit1);
        expect(query0->parent_block() == body);
        expect(query1->parent_block() == merge0);
    };

    "null_handler_is_rejected_before_outlining"_test = [] {
        Module m;
        auto *kernel = m.create_kernel();
        auto *body = kernel->create_body_block();
        XIRBuilder b;
        b.set_insertion_point(body);
        auto *query = b.alloca_local(Type::of<RayQueryAll>());
        auto *loop = b.ray_query_loop();
        auto *dispatch = loop->create_dispatch_block();
        auto *merge = loop->create_merge_block();
        b.set_insertion_point(dispatch);
        auto *dispatch_inst = b.ray_query_dispatch(query);
        dispatch_inst->set_exit_block(merge);
        auto *surface = dispatch_inst->create_on_surface_candidate_block();
        b.set_insertion_point(surface);
        b.br(dispatch);
        b.set_insertion_point(merge);
        b.return_void();
        auto function_count = count_functions(m);
        auto block_count = count_blocks(kernel->definition());

        auto info = lower_ray_query_to_pipeline_pass_run_on_function(kernel);

        expect(info.lowered_loop_count == 0u);
        expect(info.error_count == 1u);
        expect(!info.succeeded());
        expect(count_functions(m) == function_count);
        expect(count_blocks(kernel->definition()) == block_count);
        expect(body->terminator() == loop);
        expect(dispatch->terminator() == dispatch_inst);
        expect(dispatch_inst->on_procedural_candidate_block() == nullptr);
    };

    "merge_phi_is_rejected_before_outlining"_test = [] {
        Module m;
        auto f = make_fixture(m);
        XIRBuilder b;
        b.set_insertion_point(f.surface);
        b.br(f.dispatch);
        b.set_insertion_point(f.merge->instructions().head_sentinel());
        auto *phi = b.phi(Type::of<int>());
        phi->add_incoming(m.create_constant_zero(Type::of<int>()), f.dispatch);
        auto function_count = count_functions(m);
        auto block_count = count_blocks(f.kernel->definition());

        auto info = lower_ray_query_to_pipeline_pass_run_on_function(f.kernel);
        expect(!info.succeeded());
        expect(info.error_count == 1u);
        expect(info.lowered_loop_count == 0u);
        expect(count_functions(m) == function_count);
        expect(count_blocks(f.kernel->definition()) == block_count);
        expect(f.body->terminator() == f.loop);
        expect(phi->is_linked());
    };

    "external_handler_predecessor_is_rejected_before_outlining"_test = [] {
        Module m;
        auto f = make_fixture(m);
        XIRBuilder b;
        b.set_insertion_point(f.surface);
        auto *surface_exit = b.br(f.dispatch);
        auto *external = f.kernel->create_basic_block();
        b.set_insertion_point(external);
        auto *external_edge = b.br(f.surface);
        auto function_count = count_functions(m);
        auto block_count = count_blocks(f.kernel->definition());

        auto info = lower_ray_query_to_pipeline_pass_run_on_function(f.kernel);
        expect(!info.succeeded());
        expect(info.error_count == 1u);
        expect(info.lowered_loop_count == 0u);
        expect(count_functions(m) == function_count);
        expect(count_blocks(f.kernel->definition()) == block_count);
        expect(f.surface->terminator() == surface_exit);
        expect(external->terminator() == external_edge);
    };

    "external_dispatch_predecessor_is_rejected_before_outlining"_test = [] {
        Module m;
        auto f = make_fixture(m);
        XIRBuilder b;
        b.set_insertion_point(f.surface);
        auto *surface_exit = b.br(f.dispatch);
        auto *external = f.kernel->create_basic_block();
        b.set_insertion_point(external);
        auto *external_edge = b.br(f.dispatch);
        auto function_count = count_functions(m);
        auto block_count = count_blocks(f.kernel->definition());

        auto info = lower_ray_query_to_pipeline_pass_run_on_function(f.kernel);

        expect(!info.succeeded());
        expect(info.error_count == 1u);
        expect(info.lowered_loop_count == 0u);
        expect(count_functions(m) == function_count);
        expect(count_blocks(f.kernel->definition()) == block_count);
        expect(f.body->terminator() == f.loop);
        expect(f.dispatch->terminator() == f.dispatch_inst);
        expect(f.surface->terminator() == surface_exit);
        expect(external->terminator() == external_edge);
    };
}

int main(int argc, char *argv[]) {
    boost::ut::detail::cfg::parse_arg_with_fallback(argc, const_cast<const char **>(argv));
    register_tests();
    return 0;
}
