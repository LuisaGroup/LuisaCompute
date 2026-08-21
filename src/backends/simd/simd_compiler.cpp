#include "simd_compiler.h"

#include <array>
#include <iterator>
#include <limits>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <utility>

#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/Module.h>
#include <llvm/Support/raw_ostream.h>

#include <luisa/ast/function.h>
#include <luisa/core/logging.h>
#include <luisa/xir/function.h>
#include <luisa/xir/debug_printer.h>
#include <luisa/xir/instructions/call.h>
#include <luisa/xir/instructions/ray_query.h>
#include <luisa/xir/module.h>
#include <luisa/xir/passes/dce.h>
#include <luisa/xir/passes/destructure_cfg.h>
#include <luisa/xir/passes/fast_math_simplify.h>
#include <luisa/xir/passes/inline.h>
#include <luisa/xir/passes/local_load_elimination.h>
#include <luisa/xir/passes/local_store_forward.h>
#include <luisa/xir/passes/lower_ray_query_to_loop.h>
#include <luisa/xir/passes/lower_ray_query_to_pipeline.h>
#include <luisa/xir/passes/reconstruct_ray_query_loop.h>
#include <luisa/xir/passes/mem2reg.h>
#include <luisa/xir/passes/sroa.h>
#include <luisa/xir/translators/ast2xir.h>

#include "../common/env_flag.h"
#include "llvm/llvm_schedule_codegen.h"
#include "schedule/block_barrier.h"
#include "schedule/loop_unswitch.h"
#include "schedule/predicated_if_conversion.h"
#include "schedule/xir_to_schedule.h"

namespace luisa::compute::simd {

namespace {

[[nodiscard]] ::llvm::Function *build_w1_ray_query_handler_thunk(
    ::llvm::Module &module, ::llvm::Function *on_surface,
    ::llvm::Function *on_procedural,
    std::string_view name) {
    if (on_surface == nullptr || on_procedural == nullptr ||
        on_surface->arg_size() < 4u ||
        on_surface->arg_size() != on_procedural->arg_size() ||
        !on_surface->getReturnType()->isVoidTy() ||
        !on_procedural->getReturnType()->isVoidTy()) {
        return nullptr;
    }
    for (auto i = 0u; i < on_surface->arg_size(); i++) {
        if (on_surface->getFunctionType()->getParamType(i) !=
            on_procedural->getFunctionType()->getParamType(i)) {
            return nullptr;
        }
    }
    auto &context = module.getContext();
    auto *pointer_type = ::llvm::PointerType::getUnqual(context);
    auto *wrapper_type = ::llvm::FunctionType::get(
        ::llvm::Type::getVoidTy(context),
        {pointer_type, pointer_type, pointer_type,
         ::llvm::Type::getInt32Ty(context)},
        false);
    auto *wrapper = ::llvm::Function::Create(
        wrapper_type, ::llvm::GlobalValue::InternalLinkage,
        name, module);
    wrapper->setDSOLocal(true);
    wrapper->addParamAttr(0u, ::llvm::Attribute::NonNull);
    wrapper->addParamAttr(2u, ::llvm::Attribute::NonNull);

    auto argument = wrapper->arg_begin();
    auto *state = &*argument++;
    state->setName("state");
    auto *capture = &*argument++;
    capture->setName("capture");
    auto *launch_config = &*argument++;
    launch_config->setName("launch_config");
    auto *candidate_kind = &*argument;
    candidate_kind->setName("candidate_kind");

    auto *entry = ::llvm::BasicBlock::Create(
        context, "entry", wrapper);
    ::llvm::IRBuilder<> builder{entry};
    auto *state_pointer = builder.CreateAlloca(
        pointer_type, nullptr, "state.pointer");
    state_pointer->setAlignment(::llvm::Align{alignof(void *)});
    auto *state_store = builder.CreateStore(state, state_pointer);
    state_store->setAlignment(::llvm::Align{alignof(void *)});

    auto call_arguments = std::vector<::llvm::Value *>{
        builder.getInt32(1u), builder.getInt64(1u),
        state_pointer, launch_config};
    auto capture_count = on_surface->arg_size() - 4u;
    call_arguments.reserve(on_surface->arg_size());
    if (capture_count != 0u) {
        auto capture_types = std::vector<::llvm::Type *>{};
        capture_types.reserve(capture_count);
        for (auto i = 0u; i < capture_count; i++) {
            capture_types.emplace_back(
                on_surface->getFunctionType()->getParamType(i + 4u));
        }
        auto *capture_type = ::llvm::StructType::get(
            context, capture_types, false);
        auto *captured = builder.CreateLoad(
            capture_type, capture, "captures");
        captured->setAlignment(::llvm::Align{1u});
        for (auto i = 0u; i < capture_count; i++) {
            call_arguments.emplace_back(
                builder.CreateExtractValue(captured, {i}));
        }
    }
    auto *surface_block = ::llvm::BasicBlock::Create(
        context, "surface", wrapper);
    auto *procedural_block = ::llvm::BasicBlock::Create(
        context, "procedural", wrapper);
    auto *invalid_block = ::llvm::BasicBlock::Create(
        context, "invalid", wrapper);
    auto *dispatch = builder.CreateSwitch(
        candidate_kind, invalid_block, 2u);
    dispatch->addCase(
        builder.getInt32(static_cast<uint32_t>(
            SIMDHostRayQueryCandidateKind::surface)),
        surface_block);
    dispatch->addCase(
        builder.getInt32(static_cast<uint32_t>(
            SIMDHostRayQueryCandidateKind::procedural)),
        procedural_block);

    builder.SetInsertPoint(surface_block);
    builder.CreateCall(on_surface, call_arguments);
    builder.CreateRetVoid();
    builder.SetInsertPoint(procedural_block);
    builder.CreateCall(on_procedural, call_arguments);
    builder.CreateRetVoid();
    builder.SetInsertPoint(invalid_block);
    builder.CreateUnreachable();
    return wrapper;
}

void strip_debug_call_metadata_for_legalization(
    xir::Module *module) noexcept {
    // The generic XIR inliner conservatively retains a call when metadata has
    // no unique replacement owner. DSL $outline sites carry source comments,
    // but the SIMD backend requires every ordinary callable to be legalized
    // away before scheduling. Name/location/comment metadata is diagnostic
    // only, so discard it at this backend boundary while preserving semantic
    // metadata (which continues to produce a precise unsupported-call error).
    for (auto *function : module->function_list()) {
        auto *definition = function->definition();
        if (definition == nullptr) { continue; }
        for (auto *block : definition->basic_blocks()) {
            for (auto *instruction : block->instructions()) {
                if (!instruction->isa<xir::CallInst>()) { continue; }
                auto *metadata = instruction->metadata_list().head();
                while (metadata != nullptr) {
                    auto *next = metadata->next();
                    switch (metadata->derived_metadata_tag()) {
                        case xir::DerivedMetadataTag::NAME:
                        case xir::DerivedMetadataTag::LOCATION:
                        case xir::DerivedMetadataTag::COMMENT:
                            static_cast<void>(metadata->remove_self());
                            break;
                        default: break;
                    }
                    metadata = next;
                }
            }
        }
    }
}

[[nodiscard]] bool ray_query_handler_is_empty(
    const xir::Function *function) noexcept {
    auto *definition = function == nullptr ? nullptr :
                                             function->definition();
    if (definition == nullptr) { return false; }
    auto empty = true;
    definition->traverse_instructions(
        [&](const xir::Instruction *instruction) noexcept {
            switch (instruction->derived_instruction_tag()) {
                case xir::DerivedInstructionTag::BRANCH:
                case xir::DerivedInstructionTag::RETURN: break;
                default: empty = false; break;
            }
        });
    return empty;
}

// Embree may visit accepted candidates in a provider-defined order. Running a
// JIT handler from that filter is therefore legal only when the handler cannot
// observe ordering or communicate across candidates: it has no captures, is
// empty or may read only the current triangle hit, performs pure SSA/control
// work, and may only commit that same hit. An empty handler rejects every
// non-opaque triangle while the runtime still auto-commits opaque triangles,
// exactly matching the ordered query loop. Stateful handlers retain that loop.
[[nodiscard]] bool ray_query_surface_filter_is_order_independent(
    const xir::RayQueryPipelineInst *pipeline) noexcept {
    if (pipeline == nullptr ||
        pipeline->captured_argument_count() != 0u ||
        !ray_query_handler_is_empty(
            pipeline->on_procedural_function())) {
        return false;
    }
    auto *surface = pipeline->on_surface_function();
    auto *definition = surface == nullptr ? nullptr :
                                            surface->definition();
    if (definition == nullptr) { return false; }
    auto surface_is_empty = ray_query_handler_is_empty(surface);
    auto argument_count = size_t{0u};
    for ([[maybe_unused]] auto *argument : surface->arguments()) {
        argument_count++;
    }
    if (argument_count != 1u) { return false; }
    auto *query = surface->arguments().front();
    auto valid = true;
    auto saw_commit = false;
    definition->traverse_instructions(
        [&](const xir::Instruction *instruction) noexcept {
            if (!valid) { return; }
            switch (instruction->derived_instruction_tag()) {
                case xir::DerivedInstructionTag::BRANCH:
                case xir::DerivedInstructionTag::CONDITIONAL_BRANCH:
                case xir::DerivedInstructionTag::INDEXED_BRANCH:
                case xir::DerivedInstructionTag::RETURN:
                case xir::DerivedInstructionTag::PHI:
                case xir::DerivedInstructionTag::ARITHMETIC:
                case xir::DerivedInstructionTag::CAST: break;
                case xir::DerivedInstructionTag::RAY_QUERY_OBJECT_READ: {
                    auto *read = static_cast<
                        const xir::RayQueryObjectReadInst *>(instruction);
                    valid = read->op() ==
                                xir::RayQueryObjectReadOp::
                                    RAY_QUERY_OBJECT_TRIANGLE_CANDIDATE_HIT &&
                            read->operand_count() == 1u &&
                            read->operand(0u) == query;
                    break;
                }
                case xir::DerivedInstructionTag::RAY_QUERY_OBJECT_WRITE: {
                    auto *write = static_cast<
                        const xir::RayQueryObjectWriteInst *>(instruction);
                    valid = write->op() ==
                                xir::RayQueryObjectWriteOp::
                                    RAY_QUERY_OBJECT_COMMIT_TRIANGLE &&
                            write->operand_count() == 1u &&
                            write->operand(0u) == query;
                    saw_commit |= valid;
                    break;
                }
                default: valid = false; break;
            }
        });
    return valid && (surface_is_empty || saw_commit);
}

}// namespace

SIMDCompiledKernel compile_simd_kernel(
    const xir::Function *function, uint32_t warp_width,
    std::string_view entry_name, bool enable_fast_math,
    bool enable_uniform_buffer_broadcast,
    bool enable_lane_affine_buffer, bool capture_assembly,
    uint32_t dispatch_worker_count,
    bool enable_packet_batch_entry,
    bool enable_block_batch_entry) {
    SIMDCompiledKernel result{
        .warp_width = warp_width,
    };
    auto schedule_options = schedule::XIRToScheduleOptions{
        .logical_warp_width = warp_width,
        .enable_cohort_uniform_induction =
            !detail::env_flag(
                "LUISA_SIMD_DISABLE_COHORT_UNIFORM_INDUCTION"),
        .cohort_uniform_induction_min_loop_block_count =
            detail::env_flag(
                "LUISA_SIMD_FORCE_STRUCTURED_EARLY_EXIT_LOOP") ?
                4u :
                25u};
    auto schedule_result = schedule::lower_xir_to_schedule(
        function, schedule_options);
    if (!schedule_result.succeeded()) {
        result.diagnostics.reserve(schedule_result.diagnostics.size());
        for (auto &&diagnostic : schedule_result.diagnostics) {
            result.diagnostics.emplace_back(
                std::string{schedule::to_string(diagnostic.code)} +
                ": " + diagnostic.message);
        }
        return result;
    }
    struct PipelineSchedules {
        schedule::Function on_surface;
        schedule::Function on_procedural;
        const xir::Function *on_surface_xir{nullptr};
        std::vector<schedule::ValueClass> parameter_value_classes{};
        bool embree_surface_filter_safe{false};
        bool surface_handler_empty{false};
    };
    std::vector<PipelineSchedules> pipeline_schedules;
    pipeline_schedules.reserve(
        schedule_result.ray_query_pipelines.size());
    for (auto pipeline_index = size_t{0u};
         pipeline_index <
         schedule_result.ray_query_pipelines.size();
         pipeline_index++) {
        auto *pipeline =
            schedule_result.ray_query_pipelines[pipeline_index];
        const schedule::Instruction *pipeline_instruction = nullptr;
        for (auto &&block : schedule_result.function->blocks()) {
            for (auto &&instruction : block.instructions) {
                if (instruction.opcode ==
                        schedule::Opcode::ray_query_pipeline &&
                    instruction.source_op &&
                    *instruction.source_op ==
                        static_cast<uint32_t>(pipeline_index)) {
                    if (pipeline_instruction != nullptr) {
                        result.diagnostics.emplace_back(
                            "ray-query pipeline has duplicate Schedule IR sites");
                        return result;
                    }
                    pipeline_instruction = &instruction;
                }
            }
        }
        if (pipeline_instruction == nullptr ||
            pipeline_instruction->operands.size() !=
                pipeline->captured_argument_count() + 1u) {
            result.diagnostics.emplace_back(
                "ray-query pipeline capture operands do not match its XIR ABI");
            return result;
        }
        std::vector<schedule::ValueClass> parameter_value_classes;
        parameter_value_classes.reserve(
            pipeline_instruction->operands.size());
        // The query object is always one lane-local reference. Captured
        // arguments retain their caller-proven class so uniform resources and
        // scalar expressions do not become gratuitous vectors in callbacks.
        parameter_value_classes.emplace_back(
            schedule::ValueClass::varying);
        for (auto operand :
             std::span{pipeline_instruction->operands}.subspan(1u)) {
            auto *value = schedule_result.function->value(operand);
            if (value == nullptr ||
                value->value_class == schedule::ValueClass::mask ||
                value->value_class == schedule::ValueClass::token) {
                result.diagnostics.emplace_back(
                    "ray-query pipeline capture has an invalid Schedule IR class");
                return result;
            }
            parameter_value_classes.emplace_back(value->value_class);
        }
        auto lower_handler = [&](const xir::Function *handler,
                                 std::string_view kind)
            -> std::optional<schedule::Function> {
            auto handler_options = schedule_options;
            handler_options.parameter_value_classes =
                parameter_value_classes;
            auto lowered = schedule::lower_xir_to_schedule(
                handler, handler_options);
            if (!lowered.succeeded()) {
                for (auto &&diagnostic : lowered.diagnostics) {
                    result.diagnostics.emplace_back(
                        "ray-query pipeline " +
                        std::to_string(pipeline_index) + " " +
                        std::string{kind} + ": " +
                        std::string{schedule::to_string(
                            diagnostic.code)} +
                        ": " + diagnostic.message);
                }
                return std::nullopt;
            }
            return std::move(*lowered.function);
        };
        auto surface = lower_handler(
            pipeline->on_surface_function(), "surface handler");
        auto procedural = lower_handler(
            pipeline->on_procedural_function(),
            "procedural handler");
        if (!surface || !procedural) { return result; }
        auto embree_surface_filter_safe =
            ray_query_surface_filter_is_order_independent(pipeline);
        pipeline_schedules.emplace_back(PipelineSchedules{
            .on_surface = std::move(*surface),
            .on_procedural = std::move(*procedural),
            .on_surface_xir = pipeline->on_surface_function(),
            .parameter_value_classes =
                std::move(parameter_value_classes),
            .embree_surface_filter_safe =
                embree_surface_filter_safe,
            .surface_handler_empty =
                embree_surface_filter_safe &&
                ray_query_handler_is_empty(
                    pipeline->on_surface_function()),
        });
    }
    result.direct_ray_query_pipeline_count =
        pipeline_schedules.size();
    result.resident_ray_query_pipeline_count =
        warp_width == 1u ? pipeline_schedules.size() : 0u;
    if (warp_width != 1u) {
        for (auto &&pipeline : pipeline_schedules) {
            result.surface_filter_ray_query_pipeline_count +=
                pipeline.embree_surface_filter_safe;
        }
    }
    auto jit = std::make_unique<LLVMJIT>(capture_assembly);
    if (!jit->succeeded()) {
        result.diagnostics.emplace_back(jit->error());
        return result;
    }
    auto native_paired_leaf_gather =
        jit->supports_native_paired_leaf_gather(warp_width);
    auto use_paired_leaf_gather =
        !detail::env_flag(
            "LUISA_SIMD_DISABLE_PAIRED_LEAF_GATHER") &&
        native_paired_leaf_gather;
    auto use_biased_narrow_buffer_gather =
        !detail::env_flag(
            "LUISA_SIMD_DISABLE_BIASED_NARROW_BUFFER_GATHER") &&
        jit->supports_native_biased_narrow_buffer_gather(warp_width);
    auto use_gathered_native_texture_read =
        native_paired_leaf_gather;
    auto use_native_half4_texture_packet =
        jit->supports_native_half_conversion(warp_width);
    auto use_native_predicated_loop =
        jit->supports_native_predicated_loop(warp_width);
    auto use_inlined_packet_batch =
        enable_packet_batch_entry &&
        jit->supports_inlined_packet_batch(warp_width);
    auto use_native_vector_compress =
        jit->supports_native_vector_compress(warp_width);
    result.native_predicated_loop = use_native_predicated_loop;
    if (detail::env_flag("LUISA_SIMD_REPORT_SCHEDULE")) {
        LUISA_INFO(
            "SIMD Schedule IR [{} W{}]:\n{}",
            entry_name.empty() ? "simd_kernel" : entry_name,
            warp_width,
            schedule::to_string(*schedule_result.function));
        for (auto pipeline_index = size_t{0u};
             pipeline_index < pipeline_schedules.size();
             pipeline_index++) {
            LUISA_INFO(
                "SIMD ray-query surface Schedule IR [{} W{} #{}]:\n{}",
                entry_name.empty() ? "simd_kernel" : entry_name,
                warp_width, pipeline_index,
                schedule::to_string(
                    pipeline_schedules[pipeline_index].on_surface));
            LUISA_INFO(
                "SIMD ray-query procedural Schedule IR [{} W{} #{}]:\n{}",
                entry_name.empty() ? "simd_kernel" : entry_name,
                warp_width, pipeline_index,
                schedule::to_string(
                    pipeline_schedules[pipeline_index].on_procedural));
        }
    }

    auto context = std::make_unique<::llvm::LLVMContext>();
    auto module = std::make_unique<::llvm::Module>(
        "luisa-simd-kernel", *context);
    auto static_block_size = std::array<uint32_t, 3u>{};
    if (function->isa<xir::KernelFunction>()) {
        auto size = static_cast<const xir::KernelFunction *>(function)
                        ->block_size();
        static_block_size = {size.x, size.y, size.z};
    }
    std::vector<LLVMSIMDRayQueryPipelineHandlers>
        pipeline_handlers;
    pipeline_handlers.reserve(pipeline_schedules.size());
    std::vector<SIMDLLVMPrintFormat> pipeline_print_formats;
    // The scheduler oracle is diagnostic-only and substantially larger than
    // the compact handler. Keep it out of ordinary production JIT modules;
    // retaining it explicitly lets candidate/oracle processes execute
    // byte-identical modules, while DISABLE alone remains a convenient
    // correctness switch.
    auto retain_acyclic_surface_filter_scheduler_oracle =
        detail::env_flag(
            "LUISA_SIMD_RETAIN_ACYCLIC_SURFACE_FILTER_SCHEDULER_ORACLE") ||
        detail::env_flag(
            "LUISA_SIMD_DISABLE_ACYCLIC_SURFACE_FILTER_PREDICATION");
    auto handler_name_base = entry_name.empty() ?
                                 std::string{"simd_kernel"} :
                                 std::string{entry_name};
    for (auto pipeline_index = size_t{0u};
         pipeline_index < pipeline_schedules.size();
         pipeline_index++) {
        auto lower_handler = [&](const schedule::Function &handler,
                                 std::string_view kind) {
            auto name = handler_name_base + ".ray_query." +
                        std::to_string(pipeline_index) + "." +
                        std::string{kind} + ".simd_w" +
                        std::to_string(warp_width);
            return lower_ray_query_handler_schedule_to_llvm(
                *module, handler, warp_width, name,
                enable_fast_math, static_block_size,
                enable_uniform_buffer_broadcast,
                enable_lane_affine_buffer,
                use_paired_leaf_gather,
                dispatch_worker_count,
                use_native_predicated_loop,
                pipeline_print_formats.size());
        };
        auto surface = lower_handler(
            pipeline_schedules[pipeline_index].on_surface,
            "surface");
        if (!surface.succeeded()) {
            result.diagnostics.emplace_back(
                "ray-query surface handler LLVM lowering failed: " +
                surface.error);
            return result;
        }
        auto *surface_entry = surface.entry;
        pipeline_print_formats.insert(
            pipeline_print_formats.end(),
            std::make_move_iterator(surface.print_formats.begin()),
            std::make_move_iterator(surface.print_formats.end()));
        auto procedural = lower_handler(
            pipeline_schedules[pipeline_index].on_procedural,
            "procedural");
        if (!procedural.succeeded()) {
            result.diagnostics.emplace_back(
                "ray-query procedural handler LLVM lowering failed: " +
                procedural.error);
            return result;
        }
        auto *procedural_entry = procedural.entry;
        pipeline_print_formats.insert(
            pipeline_print_formats.end(),
            std::make_move_iterator(procedural.print_formats.begin()),
            std::make_move_iterator(procedural.print_formats.end()));
        auto *candidate_w1 = static_cast<::llvm::Function *>(nullptr);
        auto *surface_filter_entry =
            static_cast<::llvm::Function *>(nullptr);
        auto *surface_filter_scheduler_oracle_entry =
            static_cast<::llvm::Function *>(nullptr);
        auto *surface_filter_w4_entry =
            static_cast<::llvm::Function *>(nullptr);
        auto *surface_filter_w8_entry =
            static_cast<::llvm::Function *>(nullptr);
        if (warp_width >= 2u &&
            pipeline_schedules[pipeline_index]
                .embree_surface_filter_safe) {
            auto name = handler_name_base + ".ray_query." +
                        std::to_string(pipeline_index) +
                        ".surface_filter.simd_w" +
                        std::to_string(warp_width);
            auto surface_filter =
                lower_ray_query_surface_filter_handler_schedule_to_llvm(
                    *module,
                    pipeline_schedules[pipeline_index].on_surface,
                    warp_width, name, enable_fast_math,
                    static_block_size,
                    enable_uniform_buffer_broadcast,
                    enable_lane_affine_buffer,
                    use_paired_leaf_gather,
                    dispatch_worker_count,
                    use_native_predicated_loop,
                    pipeline_print_formats.size(), true);
            if (!surface_filter.succeeded()) {
                result.diagnostics.emplace_back(
                    "direct surface-filter handler LLVM lowering failed: " +
                    surface_filter.error);
                return result;
            }
            surface_filter_entry = surface_filter.entry;
            result.predicated_acyclic_surface_filter_handler_count +=
                surface_filter.predicated_acyclic_control_flow;
            // A sparse logical W16 direct-output query may call Embree W4/W8
            // only when its candidate handler uses the same physical width.
            // Compile those two capture-free variants from the already-audited
            // handler XIR. Empty handlers need no callback, and general
            // handlers that did not select the bounded acyclic lowering retain
            // exact-width W16 traversal to avoid speculative code growth.
            if (warp_width == 16u && use_native_vector_compress &&
                !pipeline_schedules[pipeline_index]
                     .surface_handler_empty &&
                surface_filter.predicated_acyclic_control_flow) {
                auto lower_narrow_surface_filter =
                    [&](uint32_t physical_width) {
                        auto narrow_options = schedule_options;
                        narrow_options.logical_warp_width =
                            physical_width;
                        narrow_options.parameter_value_classes =
                            pipeline_schedules[pipeline_index]
                                .parameter_value_classes;
                        auto narrow_schedule =
                            schedule::lower_xir_to_schedule(
                                pipeline_schedules[pipeline_index]
                                    .on_surface_xir,
                                narrow_options);
                        if (!narrow_schedule.succeeded()) {
                            LLVMScheduleCodegenResult failed{};
                            failed.error =
                                "narrow Schedule IR lowering failed";
                            if (!narrow_schedule.diagnostics.empty()) {
                                failed.error += ": " +
                                                narrow_schedule
                                                    .diagnostics.front()
                                                    .message;
                            }
                            return failed;
                        }
                        auto narrow_name =
                            handler_name_base + ".ray_query." +
                            std::to_string(pipeline_index) +
                            ".surface_filter.narrow.simd_w" +
                            std::to_string(physical_width) +
                            "_for_w16";
                        return lower_ray_query_surface_filter_handler_schedule_to_llvm(
                            *module,
                            *narrow_schedule.function,
                            physical_width, narrow_name,
                            enable_fast_math, static_block_size,
                            enable_uniform_buffer_broadcast,
                            enable_lane_affine_buffer,
                            use_paired_leaf_gather,
                            dispatch_worker_count,
                            use_native_predicated_loop,
                            pipeline_print_formats.size(), true);
                    };
                auto surface_filter_w4 =
                    lower_narrow_surface_filter(4u);
                auto surface_filter_w8 =
                    lower_narrow_surface_filter(8u);
                if (!surface_filter_w4.succeeded() ||
                    !surface_filter_w8.succeeded() ||
                    !surface_filter_w4.print_formats.empty() ||
                    !surface_filter_w8.print_formats.empty()) {
                    result.diagnostics.emplace_back(
                        "narrow direct surface-filter handler LLVM lowering failed: " +
                        (!surface_filter_w4.error.empty() ?
                             surface_filter_w4.error :
                             surface_filter_w8.error));
                    return result;
                }
                surface_filter_w4_entry = surface_filter_w4.entry;
                surface_filter_w8_entry = surface_filter_w8.entry;
            }
            if (surface_filter.predicated_acyclic_control_flow &&
                retain_acyclic_surface_filter_scheduler_oracle) {
                auto oracle_name = handler_name_base + ".ray_query." +
                                   std::to_string(pipeline_index) +
                                   ".surface_filter.scheduler_oracle.simd_w" +
                                   std::to_string(warp_width);
                auto scheduler_oracle =
                    lower_ray_query_surface_filter_handler_schedule_to_llvm(
                        *module,
                        pipeline_schedules[pipeline_index].on_surface,
                        warp_width, oracle_name, enable_fast_math,
                        static_block_size,
                        enable_uniform_buffer_broadcast,
                        enable_lane_affine_buffer,
                        use_paired_leaf_gather,
                        dispatch_worker_count,
                        use_native_predicated_loop,
                        pipeline_print_formats.size(), false);
                if (!scheduler_oracle.succeeded() ||
                    scheduler_oracle.predicated_acyclic_control_flow) {
                    result.diagnostics.emplace_back(
                        "direct surface-filter scheduler oracle LLVM lowering failed: " +
                        scheduler_oracle.error);
                    return result;
                }
                surface_filter_scheduler_oracle_entry =
                    scheduler_oracle.entry;
            }
            if (!surface_filter.print_formats.empty()) {
                result.diagnostics.emplace_back(
                    "direct surface-filter handler unexpectedly emitted print formats");
                return result;
            }
        }
        if (warp_width == 1u) {
            candidate_w1 = build_w1_ray_query_handler_thunk(
                *module, surface_entry, procedural_entry,
                surface_entry->getName().str() +
                    ".candidate.callback");
            if (candidate_w1 == nullptr) {
                result.diagnostics.emplace_back(
                    "failed to build W1 ray-query handler callback thunk");
                return result;
            }
        }
        pipeline_handlers.emplace_back(
            LLVMSIMDRayQueryPipelineHandlers{
                .on_surface = surface_entry,
                .on_procedural = procedural_entry,
                .on_surface_filter = surface_filter_entry,
                .on_surface_filter_scheduler_oracle =
                    surface_filter_scheduler_oracle_entry,
                .on_surface_filter_w4 = surface_filter_w4_entry,
                .on_surface_filter_w8 = surface_filter_w8_entry,
                .on_candidate_w1 = candidate_w1,
                .embree_surface_filter_safe =
                    pipeline_schedules[pipeline_index]
                        .embree_surface_filter_safe,
                .surface_handler_empty =
                    pipeline_schedules[pipeline_index]
                        .surface_handler_empty,
            });
    }
    auto llvm_result = lower_schedule_to_llvm(
        *module, *schedule_result.function, warp_width, entry_name,
        enable_fast_math, static_block_size,
        enable_uniform_buffer_broadcast,
        enable_lane_affine_buffer,
        use_paired_leaf_gather,
        dispatch_worker_count,
        use_native_predicated_loop,
        enable_packet_batch_entry,
        use_inlined_packet_batch,
        enable_block_batch_entry,
        pipeline_handlers,
        pipeline_print_formats.size(),
        use_native_vector_compress,
        use_biased_narrow_buffer_gather,
        use_gathered_native_texture_read,
        use_native_half4_texture_packet);
    if (!llvm_result.succeeded()) {
        result.diagnostics.emplace_back(llvm_result.error);
        return result;
    }
    result.argument_buffer_size = llvm_result.argument_buffer_size;
    result.print_formats = std::move(pipeline_print_formats);
    result.print_formats.insert(
        result.print_formats.end(),
        std::make_move_iterator(llvm_result.print_formats.begin()),
        std::make_move_iterator(llvm_result.print_formats.end()));
    result.schedule_block_count = llvm_result.schedule_block_count;
    result.convergence_point_count =
        llvm_result.convergence_point_count;
    result.scalar_frame_metadata =
        llvm_result.scalar_frame_metadata;
    result.state_slot_count = llvm_result.state_slot_count;
    result.coalesced_state_slot_count =
        llvm_result.coalesced_state_slot_count;
    result.general_colored_state_slot_count =
        llvm_result.general_colored_state_slot_count;
    result.spilled_instruction_count =
        llvm_result.spilled_instruction_count;
    result.cold_state_slot_count =
        llvm_result.cold_state_slot_count;
    result.stack_pinned_state_slot_count =
        llvm_result.stack_pinned_state_slot_count;
    result.ray_query_count = llvm_result.ray_query_count;
    result.ray_query_scratch_slot_count =
        llvm_result.ray_query_scratch_slot_count;
    result.ray_query_scratch_bytes =
        llvm_result.ray_query_scratch_bytes;
    result.ray_query_status_slot_count =
        llvm_result.ray_query_status_slot_count;
    result.ray_query_state_handle_slot_count =
        llvm_result.ray_query_state_handle_slot_count;
    result.compact_surface_filter_state_count =
        llvm_result.compact_surface_filter_state_count;
    result.output_only_empty_surface_filter_state_count =
        llvm_result.output_only_empty_surface_filter_state_count;
    result.direct_output_surface_filter_state_count =
        llvm_result.direct_output_surface_filter_state_count;
    result.uniform_buffer_broadcast_count =
        llvm_result.uniform_buffer_broadcast_count;
    result.contiguous_buffer_read_count =
        llvm_result.contiguous_buffer_read_count;
    result.contiguous_buffer_write_count =
        llvm_result.contiguous_buffer_write_count;
    result.transposed_buffer_read_count =
        llvm_result.transposed_buffer_read_count;
    result.transposed_buffer_write_count =
        llvm_result.transposed_buffer_write_count;
    result.paired_leaf_gather_count =
        llvm_result.paired_leaf_gather_count;
    result.biased_narrow_buffer_gather_count =
        llvm_result.biased_narrow_buffer_gather_count;
    result.interleaved_scalar_buffer_read_group_count =
        llvm_result.interleaved_scalar_buffer_read_group_count;
    result.interleaved_scalar_buffer_read_count =
        llvm_result.interleaved_scalar_buffer_read_count;
    result.interleaved_scalar_buffer_read_alias_guard_count =
        llvm_result.interleaved_scalar_buffer_read_alias_guard_count;
    result.guarded_native_texture_read_count =
        llvm_result.guarded_native_texture_read_count;
    result.guarded_gathered_native_texture_read_count =
        llvm_result.guarded_gathered_native_texture_read_count;
    result.guarded_int1_texture_read_count =
        llvm_result.guarded_int1_texture_read_count;
    result.guarded_half4_texture_read_count =
        llvm_result.guarded_half4_texture_read_count;
    result.guarded_native_texture_write_count =
        llvm_result.guarded_native_texture_write_count;
    result.guarded_byte4_texture_write_count =
        llvm_result.guarded_byte4_texture_write_count;
    result.guarded_int1_texture_write_count =
        llvm_result.guarded_int1_texture_write_count;
    result.guarded_half4_texture_write_count =
        llvm_result.guarded_half4_texture_write_count;
    result.predicated_memory_diamond_count =
        llvm_result.predicated_memory_diamond_count;
    result.predicated_memory_instruction_count =
        llvm_result.predicated_memory_instruction_count;
    result.local_predicated_diamond_count =
        llvm_result.local_predicated_diamond_count;
    result.local_predicated_two_sided_diamond_count =
        llvm_result.local_predicated_two_sided_diamond_count;
    result.local_predicated_assignment_diamond_count =
        llvm_result.local_predicated_assignment_diamond_count;
    result.local_predicated_block_count =
        llvm_result.local_predicated_block_count;
    result.local_predicated_instruction_count =
        llvm_result.local_predicated_instruction_count;
    result.nested_predicated_region_count =
        llvm_result.nested_predicated_region_count;
    result.nested_predicated_block_count =
        llvm_result.nested_predicated_block_count;
    result.nested_predicated_instruction_count =
        llvm_result.nested_predicated_instruction_count;
    result.chained_predicated_region_count =
        llvm_result.chained_predicated_region_count;
    result.chained_predicated_transition_count =
        llvm_result.chained_predicated_transition_count;
    result.chained_predicated_block_count =
        llvm_result.chained_predicated_block_count;
    result.chained_predicated_nested_tail_count =
        llvm_result.chained_predicated_nested_tail_count;
    result.chained_predicated_terminal_block_count =
        llvm_result.chained_predicated_terminal_block_count;
    result.chained_predicated_terminal_instruction_count =
        llvm_result.chained_predicated_terminal_instruction_count;
    result.predicated_loop_count =
        llvm_result.predicated_loop_count;
    result.predicated_loop_block_count =
        llvm_result.predicated_loop_block_count;
    result.predicated_loop_instruction_count =
        llvm_result.predicated_loop_instruction_count;
    result.predicated_loop_batch_iteration_count =
        llvm_result.predicated_loop_batch_iteration_count;
    result.structured_early_exit_loop_count =
        llvm_result.structured_early_exit_loop_count;
    result.structured_early_exit_loop_block_count =
        llvm_result.structured_early_exit_loop_block_count;
    result.structured_early_exit_loop_instruction_count =
        llvm_result.structured_early_exit_loop_instruction_count;
    result.structured_early_exit_loop_absorbed_block_count =
        llvm_result.structured_early_exit_loop_absorbed_block_count;
    result.cohort_uniform_loop_branch_count =
        llvm_result.cohort_uniform_loop_branch_count;
    result.coherent_mask_reuse_count =
        llvm_result.coherent_mask_reuse_count;
    result.all_on_region_version_count =
        llvm_result.all_on_region_version_count;
    result.all_on_region_block_count =
        llvm_result.all_on_region_block_count;
    result.all_on_region_instruction_count =
        llvm_result.all_on_region_instruction_count;
    result.convergence_token_guard_count =
        llvm_result.convergence_token_guard_count;
    result.return_frame_guard_count =
        llvm_result.return_frame_guard_count;
    result.direct_divergent_child_count =
        llvm_result.direct_divergent_child_count;
    result.unit_dimension_mask_elision_count =
        llvm_result.unit_dimension_mask_elision_count;
    result.linear_1d_thread_id_count =
        llvm_result.linear_1d_thread_id_count;
    result.linear_1d_packet_tail_narrowing_count =
        llvm_result.linear_1d_packet_tail_narrowing_count;
    result.linear_1d_block_coalescing_count =
        llvm_result.linear_1d_block_coalescing_count;
    result.shared_memory_size = llvm_result.shared_memory_size;
    result.block_barrier_count = llvm_result.block_barrier_count;
    result.block_barrier_loop_epoch_count =
        llvm_result.block_barrier_loop_epoch_count;
    result.cooperative_block = llvm_result.cooperative_block;
    result.direct_control_flow = llvm_result.direct_control_flow;
    auto llvm_entry_name = llvm_result.entry->getName().str();
    auto llvm_packet_batch_entry_name =
        llvm_result.packet_batch_entry == nullptr ?
            std::string{} :
            llvm_result.packet_batch_entry->getName().str();
    auto llvm_block_batch_entry_name =
        llvm_result.block_batch_entry == nullptr ?
            std::string{} :
            llvm_result.block_batch_entry->getName().str();
    result.jit = std::move(jit);
    result.target_triple = result.jit->target_triple();
    if (capture_assembly) {
        ::llvm::raw_string_ostream llvm_ir_stream{result.llvm_ir};
        module->print(llvm_ir_stream, nullptr);
        llvm_ir_stream.flush();
        result.assembly = result.jit->emit_assembly_copy(*module);
        if (result.assembly.empty()) {
            result.diagnostics.emplace_back(result.jit->error());
            result.jit.reset();
            return result;
        }
    }
    if (!result.jit->add_module(
            std::move(module), std::move(context))) {
        result.diagnostics.emplace_back(result.jit->error());
        result.jit.reset();
        return result;
    }
    if (!llvm_block_batch_entry_name.empty()) {
        result.block_batch_entry = result.jit->lookup(
            llvm_block_batch_entry_name);
        if (result.block_batch_entry == nullptr) {
            result.diagnostics.emplace_back(result.jit->error());
            result.jit.reset();
        }
    } else if (llvm_packet_batch_entry_name.empty()) {
        result.entry = result.jit->lookup(llvm_entry_name);
        if (result.entry == nullptr) {
            result.diagnostics.emplace_back(result.jit->error());
            result.jit.reset();
        }
    } else {
        result.packet_batch_entry = result.jit->lookup(
            llvm_packet_batch_entry_name);
        if (result.packet_batch_entry == nullptr) {
            result.diagnostics.emplace_back(result.jit->error());
            result.jit.reset();
        }
    }
    return result;
}

SIMDCompiledKernel compile_simd_kernel(
    const compute::Function &kernel, uint32_t warp_width,
    std::string_view entry_name, bool enable_fast_math,
    bool capture_assembly,
    uint32_t dispatch_worker_count,
    bool enable_packet_batch_entry,
    bool enable_block_batch_entry) {
    auto *translation = xir::ast_to_xir_translate_begin(
        {.preserve_inline_ray_query_loops =
             !detail::env_flag(
                 "LUISA_SIMD_DISABLE_FRONTEND_RAY_QUERY_PRESERVATION")});
    auto *xir_kernel = xir::ast_to_xir_translate_add_function(
        translation, kernel);
    auto module = xir::ast_to_xir_translate_finalize(translation);
    if (module == nullptr || xir_kernel == nullptr) {
        SIMDCompiledKernel result{.warp_width = warp_width};
        result.diagnostics.emplace_back("AST to XIR translation failed");
        return result;
    }
    auto aggregate_promotion_info = xir::SROAInfo{};
    auto promote_aggregate_allocas = [&]() noexcept {
        if (detail::env_flag(
                "LUISA_SIMD_DISABLE_AGGREGATE_PROMOTION")) {
            return;
        }
        auto info = xir::sroa_pass_run_on_module(module.get());
        aggregate_promotion_info.decomposed_alloca_count +=
            info.decomposed_alloca_count;
        aggregate_promotion_info.inserted_alloca_count +=
            info.inserted_alloca_count;
    };

    // AST callable expansion may leave single-store handler-local scratch
    // allocas in the kernel entry block. Forward those values before deciding
    // whether a structured traversal has captures: otherwise private call ABI
    // temporaries look like user-visible callback state and unnecessarily
    // force the explicit proceed-loop path.
    static_cast<void>(xir::local_store_forward_pass_run_on_module(module.get()));
    static_cast<void>(xir::dce_pass_run_on_module(module.get()));
    auto force_captured_ray_query_pipeline = detail::env_flag(
        "LUISA_SIMD_FORCE_CAPTURED_RAY_QUERY_PIPELINE");
    auto enable_captured_ray_query_pipeline =
        !detail::env_flag(
            "LUISA_SIMD_DISABLE_CAPTURED_RAY_QUERY_PIPELINE") &&
        (warp_width == 1u || warp_width == 4u ||
         force_captured_ray_query_pipeline);
    auto enable_ray_query_pipeline_profitability =
        !force_captured_ray_query_pipeline &&
        !detail::env_flag(
            "LUISA_SIMD_DISABLE_RAY_QUERY_PIPELINE_PROFITABILITY");
    // A single 9--12-instruction handler regresses at W4/W8/W16 because the
    // outlined callback costs more than the scheduler states it removes.
    // Two query sites amortize that boundary (the measured cutout renderer),
    // while one 108-instruction handler does so by itself (procedural). W1's
    // resident provider and W2's packet path remain profitable without this
    // gate. The diagnostic override retains a same-binary direct oracle.
    auto ray_query_pipeline_options =
        xir::LowerRayQueryToPipelineOptions{
            .max_captured_argument_count =
                enable_captured_ray_query_pipeline ?
                    (force_captured_ray_query_pipeline ?
                         std::numeric_limits<size_t>::max() :
                         4u) :
                    0u,
            .min_handler_instruction_count =
                enable_ray_query_pipeline_profitability &&
                        warp_width >= 4u ?
                    24u :
                    0u,
            .min_small_handler_loop_count = 2u};
    auto direct_ray_query_pipeline =
        xir::LowerRayQueryToPipelineInfo{};
    if (!detail::env_flag(
            "LUISA_SIMD_DISABLE_DIRECT_RAY_QUERY_PIPELINE")) {
        direct_ray_query_pipeline =
            xir::lower_ray_query_to_pipeline_pass_run_on_module(
                module.get(), nullptr, ray_query_pipeline_options);
        if (!direct_ray_query_pipeline.succeeded()) {
            SIMDCompiledKernel result{.warp_width = warp_width};
            result.diagnostics.emplace_back(
                "XIR direct ray-query pipeline lowering failed (errors=" +
                std::to_string(
                    direct_ray_query_pipeline.error_count) +
                ")");
            return result;
        }
    }
    auto inline_ray_query =
        xir::reconstruct_ray_query_loop_pass_run_on_module(
            module.get());
    if (!inline_ray_query.succeeded()) {
        SIMDCompiledKernel result{.warp_width = warp_width};
        result.diagnostics.emplace_back(
            "XIR explicit ray-query reconstruction failed (errors=" +
            std::to_string(inline_ray_query.error_count) + ")");
        return result;
    }
    auto reconstructed_ray_query_pipeline =
        xir::LowerRayQueryToPipelineInfo{};
    if (!detail::env_flag(
            "LUISA_SIMD_DISABLE_DIRECT_RAY_QUERY_PIPELINE")) {
        reconstructed_ray_query_pipeline =
            xir::lower_ray_query_to_pipeline_pass_run_on_module(
                module.get(), nullptr, ray_query_pipeline_options);
        if (!reconstructed_ray_query_pipeline.succeeded()) {
            SIMDCompiledKernel result{.warp_width = warp_width};
            result.diagnostics.emplace_back(
                "XIR reconstructed ray-query pipeline lowering failed (errors=" +
                std::to_string(
                    reconstructed_ray_query_pipeline.error_count) +
                ")");
            return result;
        }
    }
    // Single-block callables can be folded before CFG legalization. A second
    // pass after destructuring handles multi-block callables without cloning
    // structured regions into the caller.
    static_cast<void>(xir::inline_all_pass_run_on_module(module.get()));
    static_cast<void>(xir::local_store_forward_pass_run_on_module(module.get()));
    static_cast<void>(xir::local_load_elimination_pass_run_on_module(module.get()));
    static_cast<void>(xir::dce_pass_run_on_module(module.get()));

    auto ray_query =
        xir::lower_ray_query_to_loop_pass_run_on_module(module.get());
    if (!ray_query.succeeded()) {
        SIMDCompiledKernel result{.warp_width = warp_width};
        result.diagnostics.emplace_back(
            "XIR ray-query loop lowering failed (errors=" +
            std::to_string(ray_query.error_count) + ")");
        return result;
    }
    promote_aggregate_allocas();
    static_cast<void>(xir::mem2reg_pass_run_on_module(module.get()));
    static_cast<void>(xir::dce_pass_run_on_module(module.get()));

    auto destructure = xir::destructure_cfg_pass_run_on_module(module.get());
    if (!destructure.succeeded()) {
        SIMDCompiledKernel result{.warp_width = warp_width};
        result.diagnostics.emplace_back(
            "XIR CFG destructuring failed (errors=" +
            std::to_string(destructure.error_count) +
            ", leaked_blocks=" +
            std::to_string(destructure.leaked_block_count) + ")");
        return result;
    }
    strip_debug_call_metadata_for_legalization(module.get());
    static_cast<void>(xir::inline_all_pass_run_on_module(module.get()));
    promote_aggregate_allocas();
    static_cast<void>(xir::mem2reg_pass_run_on_module(module.get()));
    static_cast<void>(xir::dce_pass_run_on_module(module.get()));
    // A block barrier is a control/memory phase boundary, not an ordinary
    // movable side effect. Isolate it before speculative if-conversion or
    // loop unswitching inspect regions so neither rewrite can hoist, sink, or
    // clone instructions across synchronization.
    auto barrier_canonicalization =
        schedule::canonicalize_block_barriers(xir_kernel);
    if (!barrier_canonicalization.succeeded()) {
        SIMDCompiledKernel result{.warp_width = warp_width};
        result.diagnostics.emplace_back(
            "XIR block-barrier canonicalization failed: " +
            barrier_canonicalization.error);
        return result;
    }
    auto fast_math_info = xir::FastMathSimplifyInfo{};
    if (enable_fast_math) {
        fast_math_info =
            xir::fast_math_simplify_pass_run_on_module(
                module.get(), {.enable_fast_math = true});
        if (fast_math_info.changed()) {
            static_cast<void>(xir::dce_pass_run_on_module(module.get()));
        }
    }
    auto predication_info =
        schedule::PredicatedIfConversionInfo{};
    if (detail::env_flag("LUISA_SIMD_REPORT_XIR")) {
        luisa::string text;
        xir::XIRDebugPrinter printer;
        printer.emit_function(text, xir_kernel);
        LUISA_INFO(
            "SIMD XIR before scheduling rewrites [{} W{}]:\n{}",
            entry_name.empty() ? "simd_kernel" : entry_name,
            warp_width, text);
    }
    if (warp_width != 1u &&
        !detail::env_flag(
            "LUISA_SIMD_DISABLE_PREDICATED_IF")) {
        // Transparent select/Phi forwarding has a stable real-graphics win
        // at W4/W8. W2 regresses and W16 is neutral, so those widths retain
        // the single-pass policy.
        auto enable_refinement =
            (warp_width == 4u || warp_width == 8u) &&
            !detail::env_flag(
                "LUISA_SIMD_DISABLE_PREDICATED_IF_REFINEMENT");
        // A fourth float3 select-ladder layer costs fourteen register units.
        // It is profitable on the measured W8 voxel kernel but regresses W4;
        // all other widths retain the original cost-twelve boundary.
        auto enable_deep_refinement =
            enable_refinement && warp_width == 8u &&
            !detail::env_flag(
                "LUISA_SIMD_DISABLE_DEEP_PREDICATED_IF_REFINEMENT");
        auto enable_wide_refinement =
            enable_deep_refinement &&
            !detail::env_flag(
                "LUISA_SIMD_DISABLE_WIDE_PREDICATED_IF_REFINEMENT");
        auto max_speculation_cost =
            enable_deep_refinement ? 16u :
                                     12u;
        predication_info =
            schedule::predicate_small_varying_diamonds(
                xir_kernel, enable_refinement, max_speculation_cost,
                warp_width != 1u &&
                    !detail::env_flag(
                        "LUISA_SIMD_DISABLE_WIDENED_PREDICATED_UPDATE"),
                enable_wide_refinement,
                (warp_width == 8u ||
                 detail::env_flag(
                     "LUISA_SIMD_FORCE_RAY_QUERY_FILTER_PREDICATION")) &&
                    !detail::env_flag(
                        "LUISA_SIMD_DISABLE_RAY_QUERY_FILTER_PREDICATION"));
    }
    if (predication_info.changed()) {
        static_cast<void>(xir::dce_pass_run_on_module(module.get()));
    }
    auto loop_unswitch_info = schedule::SIMDLoopUnswitchInfo{};
    if (!detail::env_flag(
            "LUISA_SIMD_DISABLE_LOOP_UNSWITCH")) {
        loop_unswitch_info =
            schedule::unswitch_invariant_varying_loop_condition(
                xir_kernel,
                !detail::env_flag(
                    "LUISA_SIMD_DISABLE_GUARDED_LOOP_UNSWITCH"));
    }
    if (loop_unswitch_info.changed()) {
        static_cast<void>(xir::dce_pass_run_on_module(module.get()));
    }
    if (detail::env_flag("LUISA_SIMD_REPORT_XIR")) {
        luisa::string text;
        xir::XIRDebugPrinter printer;
        printer.emit_function(text, xir_kernel);
        LUISA_INFO(
            "SIMD XIR after scheduling rewrites [{} W{}]:\n{}",
            entry_name.empty() ? "simd_kernel" : entry_name,
            warp_width, text);
    }
    auto result = compile_simd_kernel(
        xir_kernel, warp_width, entry_name, enable_fast_math,
        !detail::env_flag(
            "LUISA_SIMD_DISABLE_UNIFORM_BUFFER_BROADCAST"),
        !detail::env_flag(
            "LUISA_SIMD_DISABLE_LANE_AFFINE_BUFFER"),
        capture_assembly, dispatch_worker_count,
        enable_packet_batch_entry,
        enable_block_batch_entry);
    result.post_reconstruction_ray_query_pipeline_count =
        reconstructed_ray_query_pipeline.lowered_loop_count;
    result.fast_math_identity_count = fast_math_info.identity_count;
    result.fast_math_radix_pow_count = fast_math_info.radix_pow_count;
    result.decomposed_aggregate_alloca_count =
        aggregate_promotion_info.decomposed_alloca_count;
    result.inserted_aggregate_leaf_alloca_count =
        aggregate_promotion_info.inserted_alloca_count;
    result.predicated_diamond_count =
        predication_info.if_conversion.converted_diamond_count;
    result.predicated_instruction_count =
        predication_info.if_conversion.hoisted_inst_count;
    result.predicated_phi_count =
        predication_info.if_conversion.replaced_phi_count;
    result.predicated_refinement_round_count =
        predication_info.refinement_round_count;
    result.predicated_forwarded_phi_count =
        predication_info.forwarded_phi_count;
    result.predicated_forwarding_block_count =
        predication_info.removed_forwarding_block_count;
    result.predicated_widened_update_diamond_count =
        predication_info.widened_update_diamond_count;
    result.predicated_wide_select_ladder_diamond_count =
        predication_info.wide_select_ladder_diamond_count;
    result.predicated_ray_query_filter_diamond_count =
        predication_info.ray_query_filter_diamond_count;
    result.factored_select_count =
        predication_info.select_factoring.factored_select_count;
    result.unswitched_loop_count =
        loop_unswitch_info.unswitch.unswitched_loop_count;
    result.guarded_unswitched_loop_count =
        loop_unswitch_info.unswitch.guarded_dynamic_loop_count;
    result.unswitched_cloned_block_count =
        loop_unswitch_info.unswitch.cloned_block_count;
    result.unswitched_cloned_instruction_count =
        loop_unswitch_info.unswitch.cloned_instruction_count;
    result.unswitched_live_out_count =
        loop_unswitch_info.unswitch.merged_live_out_count;
    return result;
}

}// namespace luisa::compute::simd
