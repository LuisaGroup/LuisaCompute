#include "simd_compiler.h"

#include <algorithm>
#include <array>
#include <iterator>
#include <limits>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>

#include <llvm/ADT/FloatingPointMode.h>
#include <llvm/Config/llvm-config.h>
#include <llvm/IR/Attributes.h>
#include <llvm/IR/Intrinsics.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/Module.h>
#include <llvm/Support/raw_ostream.h>

#include <luisa/ast/function.h>
#include <luisa/ast/type_registry.h>
#include <luisa/core/logging.h>
#include <luisa/xir/function.h>
#include <luisa/xir/debug_printer.h>
#include <luisa/xir/instructions/alloca.h>
#include <luisa/xir/instructions/arithmetic.h>
#include <luisa/xir/instructions/branch.h>
#include <luisa/xir/instructions/call.h>
#include <luisa/xir/instructions/gep.h>
#include <luisa/xir/instructions/load.h>
#include <luisa/xir/instructions/ray_query.h>
#include <luisa/xir/instructions/resource.h>
#include <luisa/xir/instructions/return.h>
#include <luisa/xir/instructions/store.h>
#include <luisa/xir/instructions/thread_group.h>
#include <luisa/xir/metadata/contiguous_copy.h>
#include <luisa/xir/metadata/name.h>
#include <luisa/xir/metadata/no_inline.h>
#include <luisa/xir/metadata/strided_mma.h>
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
#include <luisa/core/stl/memory.h>
#include <luisa/core/stl/optional.h>
#include <luisa/core/stl/string.h>

#include "../common/env_flag.h"
#include "../common/llvm_target_options.h"
#include "llvm/llvm_schedule_codegen.h"
#include "schedule/block_barrier.h"
#include "schedule/buffer_read_latency_hiding.h"
#include "schedule/loop_unswitch.h"
#include "schedule/predicated_if_conversion.h"
#include "schedule/xir_to_schedule.h"

namespace luisa::compute::simd {

namespace {

[[nodiscard]] ::llvm::Function *build_w1_ray_query_handler_thunk(
    ::llvm::Module &module, ::llvm::Function *on_surface,
    ::llvm::Function *on_procedural,
    luisa::string_view name) {
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

[[nodiscard]] ::llvm::Function *build_live_ray_query_handler_thunk(
    ::llvm::Module &module, ::llvm::Function *on_surface,
    ::llvm::Function *on_procedural,
    luisa::string_view name) {
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
    auto *i32_type = ::llvm::Type::getInt32Ty(context);
    auto *i64_type = ::llvm::Type::getInt64Ty(context);
    auto *wrapper_type = ::llvm::FunctionType::get(
        ::llvm::Type::getVoidTy(context),
        {i32_type, i64_type, pointer_type, pointer_type,
         pointer_type, i32_type},
        false);
    auto *wrapper = ::llvm::Function::Create(
        wrapper_type, ::llvm::GlobalValue::InternalLinkage,
        name, module);
    wrapper->setDSOLocal(true);
    wrapper->addParamAttr(2u, ::llvm::Attribute::NonNull);
    wrapper->addParamAttr(4u, ::llvm::Attribute::NonNull);

    auto argument = wrapper->arg_begin();
    auto *lane_count = &*argument++;
    lane_count->setName("lane_count");
    auto *active_lane_mask = &*argument++;
    active_lane_mask->setName("active_lane_mask");
    auto *states = &*argument++;
    states->setName("states");
    auto *capture = &*argument++;
    capture->setName("capture");
    auto *launch_config = &*argument++;
    launch_config->setName("launch_config");
    auto *candidate_kind = &*argument;
    candidate_kind->setName("candidate_kind");

    auto *entry = ::llvm::BasicBlock::Create(
        context, "entry", wrapper);
    ::llvm::IRBuilder<> builder{entry};
    std::vector<::llvm::Value *> call_arguments{
        lane_count, active_lane_mask, states, launch_config};
    auto capture_count = on_surface->arg_size() - 4u;
    call_arguments.reserve(on_surface->arg_size());
    if (capture_count != 0u) {
        std::vector<::llvm::Type *> capture_types;
        capture_types.reserve(capture_count);
        for (auto i = 0u; i < capture_count; i++) {
            capture_types.emplace_back(
                on_surface->getFunctionType()->getParamType(i + 4u));
        }
        auto *capture_type = ::llvm::StructType::get(
            context, capture_types, false);
        for (auto i = 0u; i < capture_count; i++) {
            auto *field_ptr = builder.CreateStructGEP(
                capture_type, capture, i);
            call_arguments.emplace_back(
                builder.CreateLoad(capture_types[i], field_ptr));
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

[[nodiscard]] bool can_extract_live_capture_lane(
    ::llvm::Type *src_ty, ::llvm::Type *dst_ty, uint32_t warp_width) noexcept {
    if (src_ty == nullptr || dst_ty == nullptr) { return false; }
    if (src_ty == dst_ty) { return true; }
    if (auto *src_vec = ::llvm::dyn_cast<::llvm::FixedVectorType>(src_ty)) {
        if (src_vec->getNumElements() != warp_width) { return false; }
        if (auto *dst_vec = ::llvm::dyn_cast<::llvm::FixedVectorType>(dst_ty)) {
            return dst_vec->getNumElements() == 1u &&
                   dst_vec->getElementType() == src_vec->getElementType();
        }
        return dst_ty == src_vec->getElementType();
    }
    if (auto *src_st = ::llvm::dyn_cast<::llvm::StructType>(src_ty)) {
        auto *dst_st = ::llvm::dyn_cast<::llvm::StructType>(dst_ty);
        if (dst_st == nullptr ||
            src_st->getNumElements() != dst_st->getNumElements()) {
            return false;
        }
        for (auto i = 0u; i < src_st->getNumElements(); ++i) {
            if (!can_extract_live_capture_lane(
                    src_st->getElementType(i), dst_st->getElementType(i),
                    warp_width)) {
                return false;
            }
        }
        return true;
    }
    if (auto *src_arr = ::llvm::dyn_cast<::llvm::ArrayType>(src_ty)) {
        auto *dst_arr = ::llvm::dyn_cast<::llvm::ArrayType>(dst_ty);
        return dst_arr != nullptr &&
               src_arr->getNumElements() == dst_arr->getNumElements() &&
               can_extract_live_capture_lane(
                   src_arr->getElementType(), dst_arr->getElementType(),
                   warp_width);
    }
    return false;
}

[[nodiscard]] ::llvm::Value *extract_live_capture_lane(
    ::llvm::IRBuilder<> &builder, ::llvm::Value *src, ::llvm::Type *dst_ty,
    ::llvm::Value *lane_i32, uint32_t warp_width) {
    auto *src_ty = src->getType();
    if (src_ty == dst_ty) { return src; }
    if (auto *src_vec = ::llvm::dyn_cast<::llvm::FixedVectorType>(src_ty)) {
        auto *elem = builder.CreateExtractElement(src, lane_i32);
        if (auto *dst_vec = ::llvm::dyn_cast<::llvm::FixedVectorType>(dst_ty)) {
            return builder.CreateInsertElement(
                ::llvm::PoisonValue::get(dst_vec), elem, builder.getInt32(0u));
        }
        return elem;
    }
    if (auto *src_st = ::llvm::dyn_cast<::llvm::StructType>(src_ty)) {
        auto *dst_st = ::llvm::cast<::llvm::StructType>(dst_ty);
        ::llvm::Value *out = ::llvm::PoisonValue::get(dst_st);
        for (auto i = 0u; i < src_st->getNumElements(); ++i) {
            auto *field = extract_live_capture_lane(
                builder, builder.CreateExtractValue(src, {i}),
                dst_st->getElementType(i), lane_i32, warp_width);
            out = builder.CreateInsertValue(out, field, {i});
        }
        return out;
    }
    if (auto *src_arr = ::llvm::dyn_cast<::llvm::ArrayType>(src_ty)) {
        auto *dst_arr = ::llvm::cast<::llvm::ArrayType>(dst_ty);
        ::llvm::Value *out = ::llvm::PoisonValue::get(dst_arr);
        for (auto i = 0u; i < src_arr->getNumElements(); ++i) {
            auto *elem = extract_live_capture_lane(
                builder, builder.CreateExtractValue(src, {i}),
                dst_arr->getElementType(), lane_i32, warp_width);
            out = builder.CreateInsertValue(out, elem, {i});
        }
        return out;
    }
    return nullptr;
}

[[nodiscard]] ::llvm::Function *build_live_ray_query_w1_handler_thunk(
    ::llvm::Module &module, ::llvm::Function *on_surface_w,
    ::llvm::Function *on_surface_w1, ::llvm::Function *on_procedural_w1,
    uint32_t warp_width, luisa::string_view name) {
    if (on_surface_w == nullptr || on_surface_w1 == nullptr ||
        on_procedural_w1 == nullptr || on_surface_w->arg_size() < 4u ||
        on_surface_w->arg_size() != on_surface_w1->arg_size() ||
        on_surface_w1->arg_size() != on_procedural_w1->arg_size() ||
        !on_surface_w1->getReturnType()->isVoidTy() ||
        !on_procedural_w1->getReturnType()->isVoidTy()) {
        return nullptr;
    }
    for (auto i = 0u; i < on_surface_w1->arg_size(); i++) {
        if (on_surface_w1->getFunctionType()->getParamType(i) !=
            on_procedural_w1->getFunctionType()->getParamType(i)) {
            return nullptr;
        }
    }
    auto capture_count = on_surface_w->arg_size() - 4u;
    for (auto i = 0u; i < capture_count; i++) {
        if (!can_extract_live_capture_lane(
                on_surface_w->getFunctionType()->getParamType(i + 4u),
                on_surface_w1->getFunctionType()->getParamType(i + 4u),
                warp_width)) {
            return nullptr;
        }
    }
    auto &context = module.getContext();
    auto *pointer_type = ::llvm::PointerType::getUnqual(context);
    auto *i32_type = ::llvm::Type::getInt32Ty(context);
    auto *i64_type = ::llvm::Type::getInt64Ty(context);
    auto *wrapper_type = ::llvm::FunctionType::get(
        ::llvm::Type::getVoidTy(context),
        {i32_type, i64_type, pointer_type, pointer_type,
         pointer_type, i32_type},
        false);
    auto *wrapper = ::llvm::Function::Create(
        wrapper_type, ::llvm::GlobalValue::InternalLinkage,
        name, module);
    wrapper->setDSOLocal(true);
    wrapper->addParamAttr(2u, ::llvm::Attribute::NonNull);
    wrapper->addParamAttr(4u, ::llvm::Attribute::NonNull);
    on_surface_w1->setLinkage(::llvm::GlobalValue::InternalLinkage);
    on_surface_w1->setDSOLocal(true);
    on_procedural_w1->setLinkage(::llvm::GlobalValue::InternalLinkage);
    on_procedural_w1->setDSOLocal(true);

    auto argument = wrapper->arg_begin();
    auto *lane_count = &*argument++;
    lane_count->setName("lane_count");
    auto *active_lane_mask = &*argument++;
    active_lane_mask->setName("active_lane_mask");
    auto *states = &*argument++;
    states->setName("states");
    auto *capture = &*argument++;
    capture->setName("capture");
    auto *launch_config = &*argument++;
    launch_config->setName("launch_config");
    auto *candidate_kind = &*argument;
    candidate_kind->setName("candidate_kind");

    auto *entry = ::llvm::BasicBlock::Create(context, "entry", wrapper);
    ::llvm::IRBuilder<> builder{entry};
    auto *cttz_decl =
#if LLVM_VERSION_MAJOR >= 22
        ::llvm::Intrinsic::getOrInsertDeclaration(
#else
        ::llvm::Intrinsic::getDeclaration(
#endif
            &module, ::llvm::Intrinsic::cttz, {i64_type});
    auto *lane_i64 = builder.CreateCall(
        cttz_decl, {active_lane_mask, builder.getTrue()}, "live.lane.i64");
    auto *lane_i32 = builder.CreateTrunc(lane_i64, i32_type, "live.lane");
    auto *state_slot = builder.CreateInBoundsGEP(
        pointer_type, states, {lane_i64}, "live.state.slot");

    ::llvm::Value *w1_launch_config = launch_config;
    if (!on_surface_w1->getArg(3u)->use_empty() ||
        !on_procedural_w1->getArg(3u)->use_empty()) {
        auto *cfg_bytes_ty = ::llvm::ArrayType::get(
            builder.getInt8Ty(), sizeof(SIMDPacketLaunchConfig));
        auto *local_cfg = builder.CreateAlloca(
            cfg_bytes_ty, nullptr, "live.w1.launch_config");
        local_cfg->setAlignment(::llvm::Align{alignof(SIMDPacketLaunchConfig)});
        builder.CreateMemCpy(
            local_cfg, ::llvm::Align{alignof(SIMDPacketLaunchConfig)},
            launch_config, ::llvm::Align{alignof(SIMDPacketLaunchConfig)},
            sizeof(SIMDPacketLaunchConfig));
        auto *tid_ptr = builder.CreateConstInBoundsGEP1_64(
            builder.getInt8Ty(), local_cfg,
            offsetof(SIMDPacketLaunchConfig, thread_index),
            "live.w1.thread_index.ptr");
        auto *base_tid = builder.CreateLoad(i32_type, tid_ptr);
        builder.CreateStore(builder.CreateAdd(base_tid, lane_i32), tid_ptr);
        w1_launch_config = local_cfg;
    }

    std::vector<::llvm::Value *> call_arguments{
        builder.getInt32(1u), builder.getInt64(1u),
        state_slot, w1_launch_config};
    call_arguments.reserve(on_surface_w1->arg_size());
    if (capture_count != 0u) {
        std::vector<::llvm::Type *> capture_types;
        capture_types.reserve(capture_count);
        for (auto i = 0u; i < capture_count; i++) {
            capture_types.emplace_back(
                on_surface_w->getFunctionType()->getParamType(i + 4u));
        }
        auto *capture_type = ::llvm::StructType::get(
            context, capture_types, false);
        for (auto i = 0u; i < capture_count; i++) {
            auto *field_ptr = builder.CreateStructGEP(
                capture_type, capture, i);
            auto *loaded = builder.CreateLoad(capture_types[i], field_ptr);
            auto *dst_ty =
                on_surface_w1->getFunctionType()->getParamType(i + 4u);
            call_arguments.emplace_back(
                extract_live_capture_lane(
                    builder, loaded, dst_ty, lane_i32, warp_width));
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
    builder.CreateCall(on_surface_w1, call_arguments);
    builder.CreateRetVoid();
    builder.SetInsertPoint(procedural_block);
    builder.CreateCall(on_procedural_w1, call_arguments);
    builder.CreateRetVoid();
    builder.SetInsertPoint(invalid_block);
    builder.CreateUnreachable();
    return wrapper;
}

void strip_debug_call_metadata_for_legalization(
    xir::Module *module) noexcept {
    // The generic XIR inliner conservatively retains a call when metadata has
    // no unique replacement owner (including block/return diagnostic metadata
    // around commented switch cases and returns). DSL $outline sites and
    // switch cases carry source comments, but the SIMD backend requires every
    // ordinary callable to be legalized away before scheduling.
    // Name/location/comment metadata is diagnostic only, so discard it at this
    // backend boundary while preserving semantic metadata (which continues to
    // produce a precise unsupported-call error).
    auto strip_diagnostic_metadata = [](auto *owner) noexcept {
        auto *metadata = owner->metadata_list().head();
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
    };
    for (auto *function : module->function_list()) {
        auto *definition = function->definition();
        if (definition == nullptr) { continue; }
        for (auto *block : definition->basic_blocks()) {
            strip_diagnostic_metadata(block);
            for (auto *instruction : block->instructions()) {
                auto tag = instruction->derived_instruction_tag();
                if (tag != xir::DerivedInstructionTag::CALL &&
                    tag != xir::DerivedInstructionTag::RETURN) {
                    continue;
                }
                strip_diagnostic_metadata(instruction);
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

struct SurfaceExclusionMatch {
    bool terminate_on_commit{false};
    std::array<uint32_t, 4u> capture_indices{
        std::numeric_limits<uint32_t>::max(),
        std::numeric_limits<uint32_t>::max(),
        std::numeric_limits<uint32_t>::max(),
        std::numeric_limits<uint32_t>::max()};
    std::array<uint32_t, 4u> constants{
        ~uint32_t{0u}, ~uint32_t{0u}, ~uint32_t{0u}, ~uint32_t{0u}};
};

[[nodiscard]] luisa::optional<SurfaceExclusionMatch>
match_ray_query_surface_exclusion(
    const xir::RayQueryPipelineInst *pipeline) noexcept {
    if (pipeline == nullptr ||
        detail::env_flag("LUISA_SIMD_DISABLE_LIVE_RAY_QUERY_PIPELINE") ||
        !ray_query_handler_is_empty(pipeline->on_procedural_function())) {
        return luisa::nullopt;
    }
    auto *surface = pipeline->on_surface_function();
    auto *def = surface == nullptr ? nullptr : surface->definition();
    if (def == nullptr) { return luisa::nullopt; }

    std::vector<const xir::Argument *> args;
    for (auto *arg : def->arguments()) { args.emplace_back(arg); }
    auto capture_count = pipeline->captured_argument_count();
    if (args.size() != 1u + capture_count) { return luisa::nullopt; }
    const auto *query_arg = args[0u];
    std::unordered_map<const xir::Value *, uint32_t> capture_arg_map;
    std::unordered_map<const xir::Value *, uint32_t> ref_capture_arg_map;
    for (auto i = size_t{0u}; i < capture_count; i++) {
        if (args[1u + i]->is_resource() ||
            args[1u + i]->type() != Type::of<uint32_t>()) {
            return luisa::nullopt;
        }
        if (args[1u + i]->is_reference()) {
            ref_capture_arg_map.emplace(
                args[1u + i], static_cast<uint32_t>(1u + i));
        } else {
            capture_arg_map.emplace(
                args[1u + i], static_cast<uint32_t>(1u + i));
        }
    }

    auto const_u32 = [](const xir::Value *v) noexcept
        -> luisa::optional<uint32_t> {
        if (v == nullptr || !v->isa<xir::Constant>()) {
            return luisa::nullopt;
        }
        auto *c = static_cast<const xir::Constant *>(v);
        if (c->type() == Type::of<uint32_t>()) {
            return *static_cast<const uint32_t *>(c->data());
        }
        if (c->type() == Type::of<int32_t>()) {
            auto s = *static_cast<const int32_t *>(c->data());
            if (s >= 0) { return static_cast<uint32_t>(s); }
        }
        return luisa::nullopt;
    };
    auto const_bool = [](const xir::Value *v) noexcept
        -> luisa::optional<bool> {
        if (v == nullptr || !v->isa<xir::Constant>() ||
            v->type() == nullptr || !v->type()->is_bool()) {
            return luisa::nullopt;
        }
        return *static_cast<const bool *>(
            static_cast<const xir::Constant *>(v)->data());
    };
    auto classify_action_block = [&](const xir::BasicBlock *bb,
                                     const xir::BasicBlock *merge_target) noexcept
        -> int {
        if (bb == nullptr || !bb->is_terminated()) { return -1; }
        if (bb == merge_target) { return 0; }
        uint32_t commit_cnt = 0u;
        uint32_t term_cnt = 0u;
        const auto *term = bb->terminator();
        for (const auto *ix : bb->instructions()) {
            if (ix == term) { break; }
            if (!ix->isa<xir::RayQueryObjectWriteInst>()) { return -1; }
            const auto *rqw = static_cast<const xir::RayQueryObjectWriteInst *>(ix);
            if (rqw->operand_count() != 1u || rqw->operand(0u) != query_arg) {
                return -1;
            }
            if (rqw->op() == xir::RayQueryObjectWriteOp::RAY_QUERY_OBJECT_COMMIT_TRIANGLE &&
                term_cnt == 0u) {
                ++commit_cnt;
            } else if (rqw->op() == xir::RayQueryObjectWriteOp::RAY_QUERY_OBJECT_TERMINATE) {
                ++term_cnt;
            } else {
                return -1;
            }
        }
        if (term->isa<xir::ReturnInst>()) {
            if (static_cast<const xir::ReturnInst *>(term)->return_value() != nullptr) {
                return -1;
            }
        } else if (term->isa<xir::BranchInst>()) {
            const auto *target = static_cast<const xir::BranchInst *>(term)->target_block();
            if (target == nullptr) { return -1; }
            if (target != merge_target) {
                if (!target->is_terminated() ||
                    target->instructions().front() != target->terminator() ||
                    !target->terminator()->isa<xir::ReturnInst>() ||
                    static_cast<const xir::ReturnInst *>(target->terminator())->return_value() != nullptr) {
                    return -1;
                }
            }
        } else {
            return -1;
        }
        if (commit_cnt == 0u && term_cnt == 0u) { return 0; }
        if (commit_cnt == 1u && term_cnt == 0u) { return 1; }
        if (commit_cnt == 1u && term_cnt == 1u) { return 2; }
        return -1;
    };

    const auto *entry_bb = def->body_block();
    if (entry_bb == nullptr || !entry_bb->is_terminated()) {
        return luisa::nullopt;
    }
    if (entry_bb->terminator()->isa<xir::ReturnInst>()) {
        auto kind = classify_action_block(entry_bb, nullptr);
        if (kind == 1 || kind == 2) {
            SurfaceExclusionMatch match{};
            match.terminate_on_commit = (kind == 2);
            return match;
        }
        return luisa::nullopt;
    }
    if (!entry_bb->terminator()->isa<xir::ConditionalBranchInst>()) {
        return luisa::nullopt;
    }
    const auto *cbr = static_cast<const xir::ConditionalBranchInst *>(
        entry_bb->terminator());
    const auto *t_bb = cbr->true_block();
    const auto *f_bb = cbr->false_block();
    if (t_bb == nullptr || f_bb == nullptr) { return luisa::nullopt; }

    const xir::BasicBlock *merge_bb = nullptr;
    if (t_bb->is_terminated() &&
        t_bb->instructions().front() == t_bb->terminator() &&
        t_bb->terminator()->isa<xir::ReturnInst>()) {
        merge_bb = t_bb;
    } else if (f_bb->is_terminated() &&
               f_bb->instructions().front() == f_bb->terminator() &&
               f_bb->terminator()->isa<xir::ReturnInst>()) {
        merge_bb = f_bb;
    }
    auto t_kind = classify_action_block(t_bb, merge_bb);
    auto f_kind = classify_action_block(f_bb, merge_bb);
    bool accept_when_cond_true = false;
    bool terminate_on_commit = false;
    if ((t_kind == 1 || t_kind == 2) && f_kind == 0) {
        accept_when_cond_true = true;
        terminate_on_commit = (t_kind == 2);
    } else if (t_kind == 0 && (f_kind == 1 || f_kind == 2)) {
        accept_when_cond_true = false;
        terminate_on_commit = (f_kind == 2);
    } else {
        return luisa::nullopt;
    }

    const xir::AllocaInst *hit_alloca = nullptr;
    const xir::Value *hit_ssa = nullptr;
    std::unordered_map<const xir::Value *, uint32_t> gep_hit_field;
    std::unordered_set<const xir::Value *> hit_inst_vals;
    std::unordered_set<const xir::Value *> hit_prim_vals;
    for (const auto *ix : entry_bb->instructions()) {
        if (ix == cbr) { break; }
        switch (ix->derived_instruction_tag()) {
            case xir::DerivedInstructionTag::ALLOCA: {
                const auto *al = static_cast<const xir::AllocaInst *>(ix);
                if (hit_alloca == nullptr &&
                    al->op() == xir::AllocaOp::LOCAL) {
                    hit_alloca = al;
                    break;
                }
                return luisa::nullopt;
            }
            case xir::DerivedInstructionTag::RAY_QUERY_OBJECT_READ: {
                const auto *rqr =
                    static_cast<const xir::RayQueryObjectReadInst *>(ix);
                if (hit_ssa == nullptr &&
                    rqr->op() == xir::RayQueryObjectReadOp::
                                     RAY_QUERY_OBJECT_TRIANGLE_CANDIDATE_HIT &&
                    rqr->operand_count() == 1u &&
                    rqr->operand(0u) == query_arg) {
                    hit_ssa = rqr;
                    break;
                }
                return luisa::nullopt;
            }
            case xir::DerivedInstructionTag::STORE: {
                const auto *st = static_cast<const xir::StoreInst *>(ix);
                if (hit_alloca != nullptr &&
                    st->variable() == hit_alloca && st->value() == hit_ssa) {
                    break;
                }
                return luisa::nullopt;
            }
            case xir::DerivedInstructionTag::GEP: {
                const auto *gep = static_cast<const xir::GEPInst *>(ix);
                if (hit_alloca != nullptr && gep->base() == hit_alloca &&
                    gep->index_uses().size() == 1u) {
                    if (auto idx = const_u32(gep->index_uses()[0u]->value());
                        idx && (*idx == 0u || *idx == 1u)) {
                        gep_hit_field.emplace(gep, *idx);
                        break;
                    }
                }
                return luisa::nullopt;
            }
            case xir::DerivedInstructionTag::LOAD: {
                const auto *ld = static_cast<const xir::LoadInst *>(ix);
                if (auto it = gep_hit_field.find(ld->variable());
                    it != gep_hit_field.end()) {
                    if (it->second == 0u) {
                        hit_inst_vals.emplace(ld);
                    } else {
                        hit_prim_vals.emplace(ld);
                    }
                    break;
                }
                if (auto it = ref_capture_arg_map.find(ld->variable());
                    it != ref_capture_arg_map.end()) {
                    capture_arg_map.emplace(ld, it->second);
                    break;
                }
                return luisa::nullopt;
            }
            case xir::DerivedInstructionTag::ARITHMETIC: {
                const auto *ar = static_cast<const xir::ArithmeticInst *>(ix);
                if (ar->op() == xir::ArithmeticOp::EXTRACT &&
                    ar->operand_count() == 2u &&
                    ar->operand(0u) == hit_ssa) {
                    if (auto idx = const_u32(ar->operand(1u));
                        idx && (*idx == 0u || *idx == 1u)) {
                        if (*idx == 0u) {
                            hit_inst_vals.emplace(ar);
                        } else {
                            hit_prim_vals.emplace(ar);
                        }
                        break;
                    }
                }
                if (ar->type() != nullptr && ar->type()->is_bool()) { break; }
                return luisa::nullopt;
            }
            default: return luisa::nullopt;
        }
    }
    if (hit_inst_vals.empty() || hit_prim_vals.empty()) {
        return luisa::nullopt;
    }

    auto strip_negation = [&](const xir::Value *v) noexcept
        -> const xir::Value * {
        if (v == nullptr || v->type() == nullptr || !v->type()->is_bool() ||
            !v->isa<xir::ArithmeticInst>()) {
            return nullptr;
        }
        const auto *ar = static_cast<const xir::ArithmeticInst *>(v);
        if (ar->op() == xir::ArithmeticOp::UNARY_BIT_NOT &&
            ar->operand_count() == 1u) {
            return ar->operand(0u);
        }
        if ((ar->op() == xir::ArithmeticOp::BINARY_BIT_XOR ||
             ar->op() == xir::ArithmeticOp::BINARY_NOT_EQUAL) &&
            ar->operand_count() == 2u) {
            if (auto c0 = const_bool(ar->operand(0u)); c0 && *c0) {
                return ar->operand(1u);
            }
            if (auto c1 = const_bool(ar->operand(1u)); c1 && *c1) {
                return ar->operand(0u);
            }
        }
        if (ar->op() == xir::ArithmeticOp::BINARY_EQUAL &&
            ar->operand_count() == 2u) {
            if (auto c0 = const_bool(ar->operand(0u)); c0 && !*c0) {
                return ar->operand(1u);
            }
            if (auto c1 = const_bool(ar->operand(1u)); c1 && !*c1) {
                return ar->operand(0u);
            }
        }
        return nullptr;
    };
    auto resolve_excl_operand = [&](const xir::Value *v, uint32_t &cap_out,
                                    uint32_t &const_out) noexcept -> bool {
        if (auto it = capture_arg_map.find(v); it != capture_arg_map.end()) {
            cap_out = it->second;
            const_out = ~uint32_t{0u};
            return true;
        }
        if (auto c = const_u32(v)) {
            cap_out = std::numeric_limits<uint32_t>::max();
            const_out = *c;
            return true;
        }
        return false;
    };
    auto match_hit_cmp = [&](auto &&self, const xir::Value *v, bool negated,
                             bool &is_inst_out, uint32_t &cap_out,
                             uint32_t &const_out) noexcept -> bool {
        if (const auto *inner = strip_negation(v)) {
            return self(
                self, inner, !negated, is_inst_out, cap_out, const_out);
        }
        if (v == nullptr || v->type() == nullptr || !v->type()->is_bool() ||
            !v->isa<xir::ArithmeticInst>()) {
            return false;
        }
        const auto *ar = static_cast<const xir::ArithmeticInst *>(v);
        const auto expected_op = negated ? xir::ArithmeticOp::BINARY_NOT_EQUAL :
                                           xir::ArithmeticOp::BINARY_EQUAL;
        if (ar->op() != expected_op || ar->operand_count() != 2u) {
            return false;
        }
        for (uint32_t swap = 0u; swap < 2u; ++swap) {
            const auto *lhs = ar->operand(swap);
            const auto *rhs = ar->operand(1u - swap);
            bool lhs_inst = hit_inst_vals.contains(lhs);
            bool lhs_prim = hit_prim_vals.contains(lhs);
            if ((lhs_inst || lhs_prim) &&
                resolve_excl_operand(rhs, cap_out, const_out)) {
                is_inst_out = lhs_inst;
                return true;
            }
        }
        return false;
    };
    auto match_pair = [&](auto &&self, const xir::Value *v, bool negated,
                          uint32_t &inst_cap, uint32_t &inst_const,
                          uint32_t &prim_cap, uint32_t &prim_const) noexcept
        -> bool {
        if (const auto *inner = strip_negation(v)) {
            return self(
                self, inner, !negated, inst_cap, inst_const, prim_cap,
                prim_const);
        }
        if (v == nullptr || v->type() == nullptr || !v->type()->is_bool() ||
            !v->isa<xir::ArithmeticInst>()) {
            return false;
        }
        const auto *ar = static_cast<const xir::ArithmeticInst *>(v);
        const auto expected_op = negated ? xir::ArithmeticOp::BINARY_BIT_OR :
                                           xir::ArithmeticOp::BINARY_BIT_AND;
        if (ar->op() != expected_op || ar->operand_count() != 2u) {
            return false;
        }
        bool i0 = false;
        bool i1 = false;
        uint32_t c0 = 0u, k0 = 0u, c1 = 0u, k1 = 0u;
        if (!match_hit_cmp(match_hit_cmp, ar->operand(0u), negated, i0, c0, k0) ||
            !match_hit_cmp(match_hit_cmp, ar->operand(1u), negated, i1, c1, k1) ||
            i0 == i1) {
            return false;
        }
        inst_cap = i0 ? c0 : c1;
        inst_const = i0 ? k0 : k1;
        prim_cap = i0 ? c1 : c0;
        prim_const = i0 ? k1 : k0;
        return true;
    };
    SurfaceExclusionMatch match{};
    match.terminate_on_commit = terminate_on_commit;
    auto match_root = [&](auto &&self, const xir::Value *v,
                          bool want_accepted) noexcept -> bool {
        if (const auto *inner = strip_negation(v)) {
            return self(self, inner, !want_accepted);
        }
        if (match_pair(
                match_pair, v, want_accepted,
                match.capture_indices[0u], match.constants[0u],
                match.capture_indices[1u], match.constants[1u])) {
            return true;
        }
        if (v != nullptr && v->type() != nullptr && v->type()->is_bool() &&
            v->isa<xir::ArithmeticInst>()) {
            const auto *ar = static_cast<const xir::ArithmeticInst *>(v);
            const auto expected_op =
                want_accepted ? xir::ArithmeticOp::BINARY_BIT_AND :
                                xir::ArithmeticOp::BINARY_BIT_OR;
            if (ar->op() == expected_op && ar->operand_count() == 2u &&
                match_pair(
                    match_pair, ar->operand(0u), want_accepted,
                    match.capture_indices[0u], match.constants[0u],
                    match.capture_indices[1u], match.constants[1u]) &&
                match_pair(
                    match_pair, ar->operand(1u), want_accepted,
                    match.capture_indices[2u], match.constants[2u],
                    match.capture_indices[3u], match.constants[3u])) {
                return true;
            }
        }
        return false;
    };
    if (!match_root(match_root, cbr->condition(), accept_when_cond_true)) {
        return luisa::nullopt;
    }
    return match;
}

[[nodiscard]] bool ray_query_pipeline_is_live_embree_safe(
    const xir::Function *caller,
    const xir::RayQueryPipelineInst *pipeline) noexcept {
    if (caller == nullptr || caller->definition() == nullptr ||
        pipeline == nullptr ||
        detail::env_flag("LUISA_SIMD_DISABLE_LIVE_RAY_QUERY_PIPELINE")) {
        return false;
    }
    const auto *query_obj = pipeline->query_object();
    if (query_obj == nullptr || query_obj->type() == nullptr) {
        return false;
    }
    const bool is_all = query_obj->type() == Type::custom("LC_RayQueryAll");
    const bool is_any = query_obj->type() == Type::custom("LC_RayQueryAny");
    if (!is_all && !is_any) { return false; }

    auto check_handler = [&](const xir::Function *fn, bool &has_ref_captures,
                             bool &has_terminate) noexcept -> bool {
        if (fn == nullptr || fn->definition() == nullptr) { return false; }
        std::unordered_set<const xir::Value *> ref_capture_args;
        size_t arg_idx = 0u;
        for (const auto *arg : fn->definition()->arguments()) {
            if (arg_idx++ > 0u && arg->is_reference()) {
                ref_capture_args.emplace(arg);
            }
        }
        bool ok = true;
        fn->definition()->traverse_instructions(
            [&](const xir::Instruction *inst) noexcept {
                if (!ok) { return; }
                // A reference capture that is only ever directly loaded is
                // immutable across candidates; any store, GEP, or other use
                // conservatively marks has_ref_captures = true.
                if (!ref_capture_args.empty()) {
                    const bool is_load =
                        inst->derived_instruction_tag() ==
                        xir::DerivedInstructionTag::LOAD;
                    for (size_t oi = 0u; oi < inst->operand_count(); ++oi) {
                        if (ref_capture_args.contains(inst->operand(oi)) &&
                            !(is_load && oi == 0u)) {
                            has_ref_captures = true;
                        }
                    }
                }
                switch (inst->derived_instruction_tag()) {
                    case xir::DerivedInstructionTag::CALL:
                    case xir::DerivedInstructionTag::THREAD_GROUP:
                    case xir::DerivedInstructionTag::PRINT:
                    case xir::DerivedInstructionTag::ASSERT:
                    case xir::DerivedInstructionTag::RESOURCE_WRITE:
                    case xir::DerivedInstructionTag::ATOMIC:
                    case xir::DerivedInstructionTag::RAY_QUERY_PIPELINE:
                        ok = false;
                        break;
                    case xir::DerivedInstructionTag::ALLOCA:
                        if (static_cast<const xir::AllocaInst *>(inst)->op() ==
                            xir::AllocaOp::SHARED) {
                            ok = false;
                        }
                        break;
                    case xir::DerivedInstructionTag::RAY_QUERY_OBJECT_WRITE:
                        if (static_cast<const xir::RayQueryObjectWriteInst *>(inst)
                                ->op() ==
                            xir::RayQueryObjectWriteOp::RAY_QUERY_OBJECT_TERMINATE) {
                            has_terminate = true;
                        }
                        break;
                    default: break;
                }
            });
        return ok;
    };

    bool has_ref_captures = false;
    bool has_terminate = false;
    if (!check_handler(
            pipeline->on_surface_function(), has_ref_captures, has_terminate) ||
        !check_handler(
            pipeline->on_procedural_function(), has_ref_captures,
            has_terminate)) {
        return false;
    }
    if (is_all) {
        return !has_ref_captures && !has_terminate;
    }
    // For RayQueryAny: live unordered Embree execution is observationally
    // sound when the caller either never loads any field of
    // RAY_QUERY_OBJECT_COMMITTED_HIT on this local query object (even with
    // mutable captures / explicit terminate), or (for capture-immutable
    // surface-only queries where every terminate follows a commit) only
    // inspects field 3 (`hit_type` / `.miss()`), since existence of an
    // accepted surface hit is permutation-invariant.
    if (!query_obj->isa<xir::AllocaInst>() ||
        static_cast<const xir::AllocaInst *>(query_obj)->op() !=
            xir::AllocaOp::LOCAL) {
        return false;
    }
    auto terminate_always_preceded_by_commit =
        [](const xir::Function *fn) noexcept -> bool {
        if (fn == nullptr || fn->definition() == nullptr ||
            fn->arguments().empty()) {
            return false;
        }
        const auto *q_arg = fn->arguments().front();
        for (const auto *bb : fn->definition()->basic_blocks()) {
            bool saw_commit = false;
            for (const auto *ix : bb->instructions()) {
                if (!ix->isa<xir::RayQueryObjectWriteInst>()) { continue; }
                const auto *rqw =
                    static_cast<const xir::RayQueryObjectWriteInst *>(ix);
                if (rqw->operand_count() == 0u || rqw->operand(0u) != q_arg) {
                    return false;
                }
                if (rqw->op() ==
                    xir::RayQueryObjectWriteOp::RAY_QUERY_OBJECT_COMMIT_TRIANGLE) {
                    saw_commit = true;
                } else if (rqw->op() ==
                               xir::RayQueryObjectWriteOp::
                                   RAY_QUERY_OBJECT_TERMINATE &&
                           !saw_commit) {
                    return false;
                }
            }
        }
        return true;
    };
    const bool allow_miss_only_committed_read =
        !has_ref_captures &&
        ray_query_handler_is_empty(pipeline->on_procedural_function()) &&
        (!has_terminate ||
         terminate_always_preceded_by_commit(pipeline->on_surface_function()));
    auto is_const_index_3 = [](const xir::Value *v) noexcept -> bool {
        if (v == nullptr || !v->isa<xir::Constant>()) { return false; }
        const auto *c = static_cast<const xir::Constant *>(v);
        if (c->type() == Type::of<uint32_t>()) {
            return *static_cast<const uint32_t *>(c->data()) == 3u;
        }
        if (c->type() == Type::of<int32_t>()) {
            return *static_cast<const int32_t *>(c->data()) == 3;
        }
        return false;
    };
    auto committed_hit_is_unused =
        [](const xir::RayQueryObjectReadInst *rqr) noexcept -> bool {
        for (const auto *use : rqr->use_list()) {
            const auto *user = use->user();
            if (user == nullptr || !user->isa<xir::StoreInst>()) {
                return false;
            }
            const auto *st = static_cast<const xir::StoreInst *>(user);
            const auto *dst = st->variable();
            if (st->value() != rqr || dst == nullptr ||
                !dst->isa<xir::AllocaInst>() ||
                static_cast<const xir::AllocaInst *>(dst)->op() !=
                    xir::AllocaOp::LOCAL) {
                return false;
            }
            for (const auto *alloca_use : dst->use_list()) {
                const auto *au = alloca_use->user();
                if (au == nullptr || !au->isa<xir::StoreInst>()) {
                    return false;
                }
                const auto *st2 = static_cast<const xir::StoreInst *>(au);
                if (st2->variable() != dst || st2->value() == dst) {
                    return false;
                }
            }
        }
        return true;
    };
    auto committed_hit_uses_only_hit_type =
        [&](const xir::RayQueryObjectReadInst *rqr) noexcept -> bool {
        for (const auto *use : rqr->use_list()) {
            const auto *user = use->user();
            if (user == nullptr || !user->isa<xir::Instruction>()) {
                return false;
            }
            const auto *ui = static_cast<const xir::Instruction *>(user);
            if (ui->isa<xir::ArithmeticInst>()) {
                const auto *ar = static_cast<const xir::ArithmeticInst *>(ui);
                if (ar->op() == xir::ArithmeticOp::EXTRACT &&
                    ar->operand_count() == 2u &&
                    ar->operand(0u) == rqr &&
                    is_const_index_3(ar->operand(1u))) {
                    continue;
                }
                return false;
            }
            if (ui->isa<xir::StoreInst>()) {
                const auto *st = static_cast<const xir::StoreInst *>(ui);
                const auto *dst = st->variable();
                if (st->value() != rqr || dst == nullptr ||
                    !dst->isa<xir::AllocaInst>() ||
                    static_cast<const xir::AllocaInst *>(dst)->op() !=
                        xir::AllocaOp::LOCAL) {
                    return false;
                }
                for (const auto *alloca_use : dst->use_list()) {
                    const auto *au = alloca_use->user();
                    if (au == st) { continue; }
                    if (au == nullptr || !au->isa<xir::GEPInst>()) {
                        return false;
                    }
                    const auto *gep = static_cast<const xir::GEPInst *>(au);
                    if (gep->base() != dst ||
                        gep->index_uses().size() != 1u ||
                        !is_const_index_3(gep->index_uses()[0u]->value())) {
                        return false;
                    }
                    for (const auto *gep_use : gep->use_list()) {
                        const auto *gu = gep_use->user();
                        if (gu == nullptr || !gu->isa<xir::LoadInst>() ||
                            static_cast<const xir::LoadInst *>(gu)->variable() != gep) {
                            return false;
                        }
                    }
                }
                continue;
            }
            return false;
        }
        return true;
    };
    bool post_read_free = true;
    uint32_t pipeline_uses = 0u;
    caller->definition()->traverse_instructions(
        [&](const xir::Instruction *inst) noexcept {
            if (!post_read_free) { return; }
            bool mentions_query = false;
            for (const auto *use : inst->operand_uses()) {
                if (use->value() == query_obj) { mentions_query = true; }
            }
            if (!mentions_query) { return; }
            if (inst == pipeline) {
                ++pipeline_uses;
                return;
            }
            if (inst->isa<xir::StoreInst>()) {
                const auto *st = static_cast<const xir::StoreInst *>(inst);
                if (st->variable() != query_obj || st->value() == query_obj) {
                    post_read_free = false;
                }
                return;
            }
            if (inst->isa<xir::RayQueryObjectReadInst>()) {
                const auto *rqr =
                    static_cast<const xir::RayQueryObjectReadInst *>(inst);
                if (rqr->op() ==
                    xir::RayQueryObjectReadOp::RAY_QUERY_OBJECT_COMMITTED_HIT) {
                    if (!committed_hit_is_unused(rqr) &&
                        (!allow_miss_only_committed_read ||
                         !committed_hit_uses_only_hit_type(rqr))) {
                        post_read_free = false;
                    }
                }
                return;
            }
            post_read_free = false;
        });
    return post_read_free && pipeline_uses == 1u;
}

[[nodiscard]] bool is_simd_outlined_data_type(const Type *type) noexcept {
    if (type == nullptr || type->is_resource() || type->is_custom() ||
        type->is_cooperative_vector() || type->is_cooperative_vector_ref() ||
        type->is_cooperative_matrix_ref()) {
        return false;
    }
    if (type->is_scalar()) { return !type->is_float8(); }
    if (type->is_vector() || type->is_matrix() || type->is_array()) {
        return is_simd_outlined_data_type(type->element());
    }
    if (type->is_structure()) {
        return std::all_of(
            type->members().begin(), type->members().end(),
            [](const Type *m) noexcept {
                return is_simd_outlined_data_type(m);
            });
    }
    return false;
}

void auto_outline_shared_value_callables(
    xir::Module *module, uint32_t warp_width) noexcept {
    if (module == nullptr || warp_width <= 1u || warp_width > 64u ||
        detail::env_flag("LUISA_SIMD_DISABLE_OUTLINED_CALLABLES") ||
        detail::env_flag("LUISA_SIMD_DISABLE_AUTO_OUTLINE_MULTI_CALL")) {
        return;
    }
    size_t min_instructions = 40u;
    if (const char *raw = std::getenv("LUISA_SIMD_AUTO_OUTLINE_MIN_INSNS");
        raw != nullptr && *raw != '\0') {
        char *end = nullptr;
        auto parsed = std::strtoull(raw, &end, 10);
        if (end != raw && *end == '\0' && parsed > 0ull) {
            min_instructions = static_cast<size_t>(parsed);
        }
    }
    std::unordered_map<const xir::Function *, size_t> call_counts;
    for (auto *fn : module->function_list()) {
        if (fn->definition() == nullptr) { continue; }
        fn->definition()->traverse_instructions(
            [&](const xir::Instruction *inst) noexcept {
                if (inst->isa<xir::CallInst>()) {
                    auto *callee = static_cast<const xir::CallInst *>(inst)->callee();
                    if (callee != nullptr) { ++call_counts[callee]; }
                }
            });
    }
    for (auto *fn : module->function_list()) {
        if (!fn->isa<xir::CallableFunction>() ||
            fn->definition() == nullptr ||
            fn->find_metadata<xir::NoInlineMD>() != nullptr ||
            fn->find_metadata<xir::StridedMmaMD>() != nullptr ||
            fn->find_metadata<xir::ContiguousCopyMD>() != nullptr) {
            continue;
        }
        auto it = call_counts.find(fn);
        if (it == call_counts.end()) { continue; }
        bool has_reference_arg = false;
        for (const auto *arg : fn->arguments()) {
            if (arg->is_reference()) {
                has_reference_arg = true;
                break;
            }
        }
        if (has_reference_arg) { continue; }
        size_t inst_count = 0u;
        bool has_loop = false;
        bool has_ray_query = false;
        fn->definition()->traverse_instructions(
            [&](const xir::Instruction *inst) noexcept {
                ++inst_count;
                auto tag = inst->derived_instruction_tag();
                if (tag == xir::DerivedInstructionTag::LOOP ||
                    tag == xir::DerivedInstructionTag::SIMPLE_LOOP) {
                    has_loop = true;
                } else if (tag == xir::DerivedInstructionTag::RAY_QUERY_PIPELINE ||
                           tag == xir::DerivedInstructionTag::RAY_QUERY_OBJECT_READ ||
                           tag == xir::DerivedInstructionTag::RAY_QUERY_OBJECT_WRITE) {
                    has_ray_query = true;
                }
            });
        if (!has_ray_query &&
            ((it->second >= 2u && inst_count >= min_instructions) ||
             (has_loop && inst_count >= 16u))) {
            static_cast<void>(fn->find_or_create_metadata<xir::NoInlineMD>());
        }
    }
}

void sanitize_ineligible_outlined_callables(
    xir::Module *module, uint32_t warp_width) noexcept {
    if (module == nullptr) { return; }
    auto strip_noinline = [](xir::Function *fn) noexcept {
        if (fn != nullptr) {
            while (auto *md = fn->find_metadata<xir::NoInlineMD>()) {
                static_cast<void>(md->remove_self());
            }
        }
    };
    if (warp_width > 64u ||
        detail::env_flag("LUISA_SIMD_DISABLE_OUTLINED_CALLABLES")) {
        for (auto *fn : module->function_list()) {
            if (fn->isa<xir::CallableFunction>()) { strip_noinline(fn); }
        }
        return;
    }

    // 1. Ray-query handlers cannot invoke outlined callables; strip NoInlineMD
    // on any callable reachable from a RayQueryPipelineInst handler.
    std::unordered_set<const xir::Function *> handler_reachable;
    auto mark_reachable = [&](auto &&self, const xir::Function *fn) noexcept -> void {
        if (fn == nullptr || fn->definition() == nullptr ||
            !handler_reachable.emplace(fn).second) {
            return;
        }
        fn->definition()->traverse_instructions(
            [&](const xir::Instruction *inst) noexcept {
                if (inst->isa<xir::CallInst>()) {
                    self(self, static_cast<const xir::CallInst *>(inst)->callee());
                } else if (inst->isa<xir::RayQueryPipelineInst>()) {
                    const auto *rq = static_cast<const xir::RayQueryPipelineInst *>(inst);
                    self(self, rq->on_surface_function());
                    self(self, rq->on_procedural_function());
                }
            });
    };
    for (auto *fn : module->function_list()) {
        if (fn->definition() == nullptr) { continue; }
        fn->definition()->traverse_instructions(
            [&](const xir::Instruction *inst) noexcept {
                if (inst->isa<xir::RayQueryPipelineInst>()) {
                    const auto *rq = static_cast<const xir::RayQueryPipelineInst *>(inst);
                    mark_reachable(mark_reachable, rq->on_surface_function());
                    mark_reachable(mark_reachable, rq->on_procedural_function());
                }
            });
    }

    for (auto *fn : module->function_list()) {
        if (!fn->isa<xir::CallableFunction>() ||
            fn->find_metadata<xir::NoInlineMD>() == nullptr) {
            continue;
        }
        if (fn->definition() == nullptr ||
            handler_reachable.contains(fn) ||
            fn->find_metadata<xir::StridedMmaMD>() != nullptr ||
            fn->find_metadata<xir::ContiguousCopyMD>() != nullptr) {
            strip_noinline(fn);
            continue;
        }
        if (fn->type() != nullptr &&
            !is_simd_outlined_data_type(fn->type())) {
            strip_noinline(fn);
            continue;
        }
        bool eligible = true;
        for (const auto *arg : fn->arguments()) {
            const auto *t = arg->type();
            if (arg->is_reference() || arg->is_value()) {
                eligible &= is_simd_outlined_data_type(t);
            } else if (arg->is_resource()) {
                eligible &= t != nullptr &&
                            (t->is_buffer() || t->is_texture() ||
                             t->is_bindless_array() || t->is_accel());
            } else {
                eligible = false;
            }
        }
        if (eligible) {
            fn->definition()->traverse_instructions(
                [&](const xir::Instruction *inst) noexcept {
                    if (!eligible) { return; }
                    switch (inst->derived_instruction_tag()) {
                        case xir::DerivedInstructionTag::ALLOCA:
                            if (static_cast<const xir::AllocaInst *>(inst)->op() ==
                                xir::AllocaOp::SHARED) {
                                eligible = false;
                            }
                            break;
                        case xir::DerivedInstructionTag::THREAD_GROUP:
                            if (static_cast<const xir::ThreadGroupInst *>(inst)->op() ==
                                xir::ThreadGroupOp::SYNCHRONIZE_BLOCK) {
                                eligible = false;
                            }
                            break;
                        default: break;
                    }
                });
        }
        if (!eligible) { strip_noinline(fn); }
    }

    // 2. Break any unexpected cycle among NoInlineMD callables.
    std::unordered_map<const xir::Function *, uint8_t> color;
    auto break_cycles = [&](auto &&self, xir::Function *fn) noexcept -> void {
        if (fn == nullptr || fn->definition() == nullptr) { return; }
        auto state = color[fn];
        if (state == 2u) { return; }
        if (state == 1u) {
            strip_noinline(fn);
            return;
        }
        color[fn] = 1u;
        fn->definition()->traverse_instructions(
            [&](const xir::Instruction *inst) noexcept {
                if (inst->isa<xir::CallInst>()) {
                    auto *callee = const_cast<xir::Function *>(
                        static_cast<const xir::CallInst *>(inst)->callee());
                    self(self, callee);
                }
            });
        color[fn] = 2u;
    };
    for (auto *fn : module->function_list()) {
        if (fn->isa<xir::CallableFunction>() &&
            fn->find_metadata<xir::NoInlineMD>() != nullptr) {
            break_cycles(break_cycles, fn);
        }
    }
}

}// namespace

SIMDCompiledKernel compile_simd_kernel(
    const xir::Function *function, uint32_t warp_width,
    luisa::string_view entry_name, bool enable_fast_math,
    bool enable_uniform_buffer_broadcast,
    bool enable_lane_affine_buffer, bool capture_assembly,
    uint32_t dispatch_worker_count,
    bool enable_packet_batch_entry,
    bool enable_block_batch_entry, bool capture_ir,
    size_t private_stack_budget_bytes, bool enable_interleaved_private_arrays,
    bool enable_contiguous_private_access, bool capture_object) {
    SIMDCompiledKernel result{
        .enable_fast_math = enable_fast_math,
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
                25u,
        .enable_counted_loop_uniformity =
            !detail::env_flag("LUISA_SIMD_DISABLE_PREDICATED_MEMORY_EFFECTS"),
        .enable_cohort_private_access =
            !detail::env_flag("LUISA_SIMD_DISABLE_COHORT_PRIVATE_ACCESS")};
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
        const xir::Function *on_procedural_xir{nullptr};
        std::vector<schedule::ValueClass> parameter_value_classes{};
        bool embree_surface_filter_safe{false};
        bool surface_handler_empty{false};
        bool procedural_handler_empty{true};
        bool live_embree_safe{false};
        luisa::optional<SurfaceExclusionMatch> surface_exclusion{};
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
             luisa::span{pipeline_instruction->operands}.subspan(1u)) {
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
                                 luisa::string_view kind)
            -> luisa::optional<schedule::Function> {
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
                return luisa::nullopt;
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
        auto procedural_empty =
            ray_query_handler_is_empty(pipeline->on_procedural_function());
        auto live_embree_safe =
            !embree_surface_filter_safe &&
            ray_query_pipeline_is_live_embree_safe(function, pipeline);
        auto surface_exclusion =
            !embree_surface_filter_safe && live_embree_safe ?
                match_ray_query_surface_exclusion(pipeline) :
                luisa::nullopt;
        if (detail::env_flag("LUISA_SIMD_REPORT_OPTIMIZATIONS")) {
            LUISA_INFO(
                "SIMD ray_query pipeline [{}]: surface_filter={}, live_embree={}, "
                "surface_exclusion={}, captures={}, proc_empty={}",
                pipeline_schedules.size(), embree_surface_filter_safe,
                live_embree_safe, surface_exclusion.has_value(),
                pipeline->captured_argument_count(), procedural_empty);
        }
        pipeline_schedules.emplace_back(PipelineSchedules{
            .on_surface = std::move(*surface),
            .on_procedural = std::move(*procedural),
            .on_surface_xir = pipeline->on_surface_function(),
            .on_procedural_xir = pipeline->on_procedural_function(),
            .parameter_value_classes =
                std::move(parameter_value_classes),
            .embree_surface_filter_safe =
                embree_surface_filter_safe,
            .surface_handler_empty =
                embree_surface_filter_safe &&
                ray_query_handler_is_empty(
                    pipeline->on_surface_function()),
            .procedural_handler_empty = procedural_empty,
            .live_embree_safe = live_embree_safe,
            .surface_exclusion = surface_exclusion,
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
    auto jit = luisa::make_unique<LLVMJIT>(
        capture_assembly || capture_object);
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

    auto context = luisa::make_unique<::llvm::LLVMContext>();
    auto module = luisa::make_unique<::llvm::Module>(
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
                                 luisa::string_view kind) {
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
        auto *candidate_live = static_cast<::llvm::Function *>(nullptr);
        if (warp_width >= 2u &&
            pipeline_schedules[pipeline_index].live_embree_safe &&
            !pipeline_schedules[pipeline_index].surface_exclusion.has_value()) {
            if (!detail::env_flag("LUISA_SIMD_DISABLE_W1_LIVE_CALLBACK")) {
                auto &ps_live = pipeline_schedules[pipeline_index];
                auto w1_opts = schedule_options;
                w1_opts.logical_warp_width = 1u;
                w1_opts.parameter_value_classes =
                    ps_live.parameter_value_classes;
                auto w1_surf_sched = schedule::lower_xir_to_schedule(
                    ps_live.on_surface_xir, w1_opts);
                auto w1_proc_sched = schedule::lower_xir_to_schedule(
                    ps_live.on_procedural_xir, w1_opts);
                if (w1_surf_sched.succeeded() && w1_proc_sched.succeeded()) {
                    auto w1_surf_name = handler_name_base + ".ray_query." +
                                        std::to_string(pipeline_index) +
                                        ".surface.live_w1";
                    auto w1_proc_name = handler_name_base + ".ray_query." +
                                        std::to_string(pipeline_index) +
                                        ".procedural.live_w1";
                    auto w1_surf = lower_ray_query_handler_schedule_to_llvm(
                        *module, *w1_surf_sched.function, 1u, w1_surf_name,
                        enable_fast_math, static_block_size,
                        enable_uniform_buffer_broadcast,
                        enable_lane_affine_buffer,
                        use_paired_leaf_gather,
                        dispatch_worker_count,
                        use_native_predicated_loop,
                        pipeline_print_formats.size());
                    auto w1_proc = lower_ray_query_handler_schedule_to_llvm(
                        *module, *w1_proc_sched.function, 1u, w1_proc_name,
                        enable_fast_math, static_block_size,
                        enable_uniform_buffer_broadcast,
                        enable_lane_affine_buffer,
                        use_paired_leaf_gather,
                        dispatch_worker_count,
                        use_native_predicated_loop,
                        pipeline_print_formats.size());
                    if (w1_surf.succeeded() && w1_proc.succeeded() &&
                        w1_surf.print_formats.empty() &&
                        w1_proc.print_formats.empty()) {
                        candidate_live = build_live_ray_query_w1_handler_thunk(
                            *module, surface_entry, w1_surf.entry,
                            w1_proc.entry, warp_width,
                            surface_entry->getName().str() +
                                ".live.w1.callback");
                    }
                    if (candidate_live == nullptr) {
                        if (w1_surf.entry != nullptr) {
                            w1_surf.entry->eraseFromParent();
                        }
                        if (w1_proc.entry != nullptr) {
                            w1_proc.entry->eraseFromParent();
                        }
                    }
                }
            }
            if (candidate_live == nullptr) {
                candidate_live = build_live_ray_query_handler_thunk(
                    *module, surface_entry, procedural_entry,
                    surface_entry->getName().str() + ".live.callback");
            }
            if (candidate_live == nullptr) {
                result.diagnostics.emplace_back(
                    "failed to build live ray-query handler callback thunk");
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
        auto &ps = pipeline_schedules[pipeline_index];
        LLVMSIMDRayQueryPipelineHandlers handler_record{
            .on_surface = surface_entry,
            .on_procedural = procedural_entry,
            .on_surface_filter = surface_filter_entry,
            .on_surface_filter_scheduler_oracle =
                surface_filter_scheduler_oracle_entry,
            .on_surface_filter_w4 = surface_filter_w4_entry,
            .on_surface_filter_w8 = surface_filter_w8_entry,
            .on_candidate_w1 = candidate_w1,
            .embree_surface_filter_safe =
                ps.embree_surface_filter_safe,
            .surface_handler_empty =
                ps.surface_handler_empty,
            .on_candidate_live = candidate_live,
            .live_embree_safe = ps.live_embree_safe,
            .procedural_handler_empty = ps.procedural_handler_empty,
            .surface_exclusion_safe = ps.surface_exclusion.has_value(),
        };
        if (ps.surface_exclusion) {
            handler_record.surface_exclusion_terminate_on_commit =
                ps.surface_exclusion->terminate_on_commit;
            handler_record.surface_exclusion_capture_indices =
                ps.surface_exclusion->capture_indices;
            handler_record.surface_exclusion_constants =
                ps.surface_exclusion->constants;
        }
        pipeline_handlers.emplace_back(std::move(handler_record));
    }

    std::vector<LLVMSIMDOutlinedCallable> kernel_outlined_callables;
    uint32_t outlined_interleaved_private_arrays = 0u;
    size_t outlined_contiguous_private_read_count = 0u;
    size_t outlined_contiguous_private_write_count = 0u;
    if (!schedule_result.outlined_callables.empty()) {
        std::vector<const xir::Function *> post_order;
        std::unordered_set<const xir::Function *> visited;
        auto collect_post_order = [&](auto &&self,
                                      const xir::Function *fn) -> void {
            if (fn == nullptr || fn->definition() == nullptr ||
                !visited.emplace(fn).second) {
                return;
            }
            fn->definition()->traverse_instructions(
                [&](const xir::Instruction *inst) {
                    if (!inst->isa<xir::CallInst>()) { return; }
                    auto *callee =
                        static_cast<const xir::CallInst *>(inst)->callee();
                    if (callee != nullptr &&
                        callee->isa<xir::CallableFunction>() &&
                        callee->definition() != nullptr &&
                        callee->find_metadata<xir::NoInlineMD>() != nullptr) {
                        self(self, callee);
                    }
                });
            post_order.emplace_back(fn);
        };
        for (auto *callee : schedule_result.outlined_callables) {
            collect_post_order(collect_post_order, callee);
        }

        struct OutlinedScheduleRecord {
            const xir::Function *xir_fn{nullptr};
            std::vector<const Type *> param_types{};
            std::vector<uint32_t> param_tags{};
            std::vector<schedule::ValueClass> param_classes{};
            std::vector<bool> all_callers_uniform{};
            std::vector<uint8_t> param_soa_cells{};
            bool has_call_site{false};
            schedule::XIRToScheduleResult schedule{};
        };
        std::unordered_map<const xir::Function *, size_t> record_index;
        std::vector<OutlinedScheduleRecord> records(post_order.size());
        for (size_t i = 0u; i < post_order.size(); ++i) {
            const auto *fn = post_order[i];
            record_index.emplace(fn, i);
            auto &rec = records[i];
            rec.xir_fn = fn;
            for (const auto *arg : fn->arguments()) {
                rec.param_types.emplace_back(arg->type());
                rec.param_tags.emplace_back(
                    static_cast<uint32_t>(arg->derived_argument_tag()));
                rec.param_classes.emplace_back(
                    arg->is_resource() ? schedule::ValueClass::warp_uniform :
                                         schedule::ValueClass::varying);
                rec.all_callers_uniform.push_back(
                    arg->is_resource() || arg->is_value());
            }
        }
        auto accumulate_caller_sites =
            [&](const schedule::Function &caller_sched,
                const std::vector<const xir::Function *> &caller_table) {
                for (const auto &blk : caller_sched.blocks()) {
                    for (const auto &inst : blk.instructions) {
                        if (inst.opcode != schedule::Opcode::call ||
                            !inst.outlined_callable_id ||
                            *inst.outlined_callable_id >= caller_table.size()) {
                            continue;
                        }
                        const auto *callee =
                            caller_table[*inst.outlined_callable_id];
                        auto it = record_index.find(callee);
                        if (it == record_index.end()) { continue; }
                        auto &rec = records[it->second];
                        rec.has_call_site = true;
                        for (size_t k = 0u;
                             k < inst.operands.size() &&
                             k < rec.all_callers_uniform.size();
                             ++k) {
                            const auto *v = caller_sched.value(inst.operands[k]);
                            if (v == nullptr ||
                                v->value_class !=
                                    schedule::ValueClass::warp_uniform) {
                                rec.all_callers_uniform[k] = false;
                            }
                        }
                    }
                }
            };
        accumulate_caller_sites(
            *schedule_result.function, schedule_result.outlined_callables);

        for (auto idx = post_order.size(); idx > 0u; --idx) {
            auto &rec = records[idx - 1u];
            for (size_t k = 0u; k < rec.param_classes.size(); ++k) {
                auto tag = static_cast<xir::DerivedArgumentTag>(
                    rec.param_tags[k]);
                if (tag == xir::DerivedArgumentTag::VALUE &&
                    rec.has_call_site && rec.all_callers_uniform[k]) {
                    rec.param_classes[k] = schedule::ValueClass::warp_uniform;
                }
            }
            auto opts = schedule_options;
            opts.parameter_value_classes = rec.param_classes;
            rec.schedule = schedule::lower_xir_to_schedule(rec.xir_fn, opts);
            if (!rec.schedule.succeeded()) {
                for (auto &&diag : rec.schedule.diagnostics) {
                    result.diagnostics.emplace_back(
                        "outlined callable Schedule IR lowering failed: " +
                        diag.message);
                }
                return result;
            }
            accumulate_caller_sites(
                *rec.schedule.function, rec.schedule.outlined_callables);
        }

        // Greatest-fixed-point interprocedural UniformSoA reference ABI
        // analysis across the kernel and all outlined callables. Start
        // optimistic for every eligible REFERENCE parameter and shrink to AoS
        // (0) whenever a callee poisons its own parameter or any call site
        // cannot supply a matching UniformSoA pointer.
        for (auto &rec : records) {
            rec.param_soa_cells.assign(rec.param_types.size(), 0u);
            if (enable_interleaved_private_arrays && warp_width > 1u &&
                rec.has_call_site) {
                for (size_t k = 0u; k < rec.param_types.size(); ++k) {
                    if (static_cast<xir::DerivedArgumentTag>(rec.param_tags[k]) ==
                        xir::DerivedArgumentTag::REFERENCE) {
                        rec.param_soa_cells[k] =
                            uniform_leaf_cell_size(rec.param_types[k]);
                    }
                }
            }
        }
        if (enable_interleaved_private_arrays && warp_width > 1u) {
            auto build_callee_cells =
                [&](const std::vector<const xir::Function *> &table) {
                    std::vector<std::vector<uint8_t>> cells;
                    cells.reserve(table.size());
                    for (const auto *fn : table) {
                        auto it = record_index.find(fn);
                        if (it != record_index.end()) {
                            cells.emplace_back(records[it->second].param_soa_cells);
                        } else {
                            cells.emplace_back();
                        }
                    }
                    return cells;
                };
            auto verify_caller_sites =
                [&](const schedule::Function &caller_fn,
                    const std::vector<const xir::Function *> &caller_table,
                    const LocalSoAAnalysis &caller_soa) noexcept -> bool {
                auto shrunk = false;
                for (const auto &blk : caller_fn.blocks()) {
                    for (const auto &inst : blk.instructions) {
                        if (inst.opcode != schedule::Opcode::call ||
                            !inst.outlined_callable_id ||
                            *inst.outlined_callable_id >= caller_table.size()) {
                            continue;
                        }
                        auto it = record_index.find(
                            caller_table[*inst.outlined_callable_id]);
                        if (it == record_index.end()) { continue; }
                        auto &callee_rec = records[it->second];
                        for (size_t k = 0u;
                             k < inst.operands.size() &&
                             k < callee_rec.param_soa_cells.size();
                             ++k) {
                            auto expected = callee_rec.param_soa_cells[k];
                            if (expected == 0u) { continue; }
                            auto op = inst.operands[k].value;
                            if (op >= caller_soa.kind.size() ||
                                caller_soa.kind[op] != LocalSoAKind::uniform) {
                                callee_rec.param_soa_cells[k] = 0u;
                                shrunk = true;
                            }
                        }
                    }
                }
                return shrunk;
            };
            auto soa_changed = true;
            while (soa_changed) {
                soa_changed = false;
                auto kernel_cells = build_callee_cells(
                    schedule_result.outlined_callables);
                auto kernel_soa = analyze_local_soa(
                    *schedule_result.function, warp_width, {},
                    kernel_cells, true);
                soa_changed |= verify_caller_sites(
                    *schedule_result.function,
                    schedule_result.outlined_callables, kernel_soa);
                for (auto &rec : records) {
                    auto child_cells = build_callee_cells(
                        rec.schedule.outlined_callables);
                    auto rec_soa = analyze_local_soa(
                        *rec.schedule.function, warp_width,
                        rec.param_soa_cells, child_cells, true);
                    for (auto &&val : rec.schedule.function->values()) {
                        if (val.origin != schedule::ValueOrigin::parameter) {
                            continue;
                        }
                        auto *md = std::get_if<
                            schedule::ParameterValueMetadata>(&val.metadata);
                        if (md == nullptr ||
                            md->index >= rec.param_soa_cells.size() ||
                            rec.param_soa_cells[md->index] == 0u) {
                            continue;
                        }
                        if (val.id.value >= rec_soa.kind.size() ||
                            rec_soa.kind[val.id.value] != LocalSoAKind::uniform ||
                            rec_soa.cell_bytes[val.id.value] !=
                                rec.param_soa_cells[md->index]) {
                            rec.param_soa_cells[md->index] = 0u;
                            soa_changed = true;
                        }
                    }
                    soa_changed |= verify_caller_sites(
                        *rec.schedule.function,
                        rec.schedule.outlined_callables, rec_soa);
                }
            }
        }

        auto lower_outlined_ray_query_pipelines =
            [&](const xir::Function *owner_xir,
                const schedule::Function &owner_sched,
                luisa::span<const xir::RayQueryPipelineInst *const> xir_pipelines,
                uint32_t target_width,
                const std::string &name_prefix,
                bool record_kernel_stats,
                std::vector<LLVMSIMDRayQueryPipelineHandlers> &out_handlers,
                std::string &out_error) -> bool {
            out_handlers.clear();
            out_handlers.reserve(xir_pipelines.size());
            for (auto p_idx = size_t{0u}; p_idx < xir_pipelines.size(); ++p_idx) {
                auto *pipeline = xir_pipelines[p_idx];
                const schedule::Instruction *p_inst = nullptr;
                for (auto &&blk : owner_sched.blocks()) {
                    for (auto &&inst : blk.instructions) {
                        if (inst.opcode == schedule::Opcode::ray_query_pipeline &&
                            inst.source_op &&
                            *inst.source_op == static_cast<uint32_t>(p_idx)) {
                            if (p_inst != nullptr) {
                                out_error = "outlined ray-query pipeline has duplicate Schedule IR sites";
                                return false;
                            }
                            p_inst = &inst;
                        }
                    }
                }
                if (p_inst == nullptr ||
                    p_inst->operands.size() != pipeline->captured_argument_count() + 1u) {
                    out_error = "outlined ray-query pipeline capture operands do not match its XIR ABI";
                    return false;
                }
                std::vector<schedule::ValueClass> param_classes;
                param_classes.reserve(p_inst->operands.size());
                param_classes.emplace_back(schedule::ValueClass::varying);
                for (auto op : luisa::span{p_inst->operands}.subspan(1u)) {
                    auto *v = owner_sched.value(op);
                    if (v == nullptr ||
                        v->value_class == schedule::ValueClass::mask ||
                        v->value_class == schedule::ValueClass::token) {
                        out_error = "outlined ray-query pipeline capture has invalid Schedule IR class";
                        return false;
                    }
                    auto vc = v->value_class;
                    if (target_width == 1u &&
                        vc == schedule::ValueClass::cohort_uniform) {
                        vc = schedule::ValueClass::warp_uniform;
                    }
                    param_classes.emplace_back(vc);
                }
                auto h_opts = schedule_options;
                h_opts.logical_warp_width = target_width;
                h_opts.parameter_value_classes = param_classes;
                auto surf_sched = schedule::lower_xir_to_schedule(
                    pipeline->on_surface_function(), h_opts);
                auto proc_sched = schedule::lower_xir_to_schedule(
                    pipeline->on_procedural_function(), h_opts);
                if (!surf_sched.succeeded() || !proc_sched.succeeded()) {
                    out_error = "outlined ray-query handler Schedule IR lowering failed";
                    return false;
                }
                auto surf_name = name_prefix + ".rq." + std::to_string(p_idx) +
                                 ".surface.simd_w" + std::to_string(target_width);
                auto proc_name = name_prefix + ".rq." + std::to_string(p_idx) +
                                 ".procedural.simd_w" + std::to_string(target_width);
                auto surf_llvm = lower_ray_query_handler_schedule_to_llvm(
                    *module, *surf_sched.function, target_width, surf_name,
                    enable_fast_math, static_block_size,
                    enable_uniform_buffer_broadcast,
                    enable_lane_affine_buffer,
                    target_width == warp_width ? use_paired_leaf_gather : false,
                    dispatch_worker_count,
                    target_width == warp_width ? use_native_predicated_loop : false,
                    pipeline_print_formats.size());
                if (!surf_llvm.succeeded()) {
                    out_error = "outlined ray-query surface LLVM lowering failed: " +
                                surf_llvm.error;
                    return false;
                }
                pipeline_print_formats.insert(
                    pipeline_print_formats.end(),
                    std::make_move_iterator(surf_llvm.print_formats.begin()),
                    std::make_move_iterator(surf_llvm.print_formats.end()));
                auto proc_llvm = lower_ray_query_handler_schedule_to_llvm(
                    *module, *proc_sched.function, target_width, proc_name,
                    enable_fast_math, static_block_size,
                    enable_uniform_buffer_broadcast,
                    enable_lane_affine_buffer,
                    target_width == warp_width ? use_paired_leaf_gather : false,
                    dispatch_worker_count,
                    target_width == warp_width ? use_native_predicated_loop : false,
                    pipeline_print_formats.size());
                if (!proc_llvm.succeeded()) {
                    out_error = "outlined ray-query procedural LLVM lowering failed: " +
                                proc_llvm.error;
                    return false;
                }
                pipeline_print_formats.insert(
                    pipeline_print_formats.end(),
                    std::make_move_iterator(proc_llvm.print_formats.begin()),
                    std::make_move_iterator(proc_llvm.print_formats.end()));

                if (target_width == 1u) {
                    auto *cand_w1 = build_w1_ray_query_handler_thunk(
                        *module, surf_llvm.entry, proc_llvm.entry,
                        surf_llvm.entry->getName().str() + ".candidate.callback");
                    if (cand_w1 == nullptr) {
                        out_error = "failed to build outlined W1 ray-query handler thunk";
                        return false;
                    }
                    out_handlers.emplace_back(LLVMSIMDRayQueryPipelineHandlers{
                        .on_surface = surf_llvm.entry,
                        .on_procedural = proc_llvm.entry,
                        .on_candidate_w1 = cand_w1,
                        .procedural_handler_empty =
                            ray_query_handler_is_empty(pipeline->on_procedural_function()),
                    });
                    continue;
                }

                auto filter_safe =
                    ray_query_surface_filter_is_order_independent(pipeline);
                auto proc_empty =
                    ray_query_handler_is_empty(pipeline->on_procedural_function());
                auto live_safe =
                    !filter_safe &&
                    ray_query_pipeline_is_live_embree_safe(owner_xir, pipeline);
                auto surf_excl =
                    !filter_safe && live_safe ?
                        match_ray_query_surface_exclusion(pipeline) :
                        luisa::nullopt;
                auto surf_empty =
                    filter_safe &&
                    ray_query_handler_is_empty(pipeline->on_surface_function());
                ::llvm::Function *sf_entry = nullptr;
                if (filter_safe) {
                    auto sf_name = name_prefix + ".rq." + std::to_string(p_idx) +
                                   ".surface_filter.simd_w" + std::to_string(target_width);
                    auto sf_llvm = lower_ray_query_surface_filter_handler_schedule_to_llvm(
                        *module, *surf_sched.function, target_width, sf_name,
                        enable_fast_math, static_block_size,
                        enable_uniform_buffer_broadcast,
                        enable_lane_affine_buffer,
                        use_paired_leaf_gather, dispatch_worker_count,
                        use_native_predicated_loop,
                        pipeline_print_formats.size(), true);
                    if (!sf_llvm.succeeded() || !sf_llvm.print_formats.empty()) {
                        out_error = "outlined ray-query surface_filter LLVM lowering failed";
                        return false;
                    }
                    sf_entry = sf_llvm.entry;
                    if (record_kernel_stats) {
                        result.predicated_acyclic_surface_filter_handler_count +=
                            sf_llvm.predicated_acyclic_control_flow;
                    }
                }
                ::llvm::Function *cand_live = nullptr;
                if (live_safe && !surf_excl.has_value()) {
                    if (!detail::env_flag("LUISA_SIMD_DISABLE_W1_LIVE_CALLBACK")) {
                        auto w1_opts = schedule_options;
                        w1_opts.logical_warp_width = 1u;
                        auto w1_param_classes = param_classes;
                        for (auto &vc : w1_param_classes) {
                            if (vc == schedule::ValueClass::cohort_uniform) {
                                vc = schedule::ValueClass::warp_uniform;
                            }
                        }
                        w1_opts.parameter_value_classes = w1_param_classes;
                        auto w1_s = schedule::lower_xir_to_schedule(
                            pipeline->on_surface_function(), w1_opts);
                        auto w1_p = schedule::lower_xir_to_schedule(
                            pipeline->on_procedural_function(), w1_opts);
                        if (w1_s.succeeded() && w1_p.succeeded()) {
                            auto w1_sl = lower_ray_query_handler_schedule_to_llvm(
                                *module, *w1_s.function, 1u,
                                name_prefix + ".rq." + std::to_string(p_idx) + ".surface.live_w1",
                                enable_fast_math, static_block_size,
                                enable_uniform_buffer_broadcast,
                                enable_lane_affine_buffer,
                                use_paired_leaf_gather, dispatch_worker_count,
                                use_native_predicated_loop,
                                pipeline_print_formats.size());
                            auto w1_pl = lower_ray_query_handler_schedule_to_llvm(
                                *module, *w1_p.function, 1u,
                                name_prefix + ".rq." + std::to_string(p_idx) + ".procedural.live_w1",
                                enable_fast_math, static_block_size,
                                enable_uniform_buffer_broadcast,
                                enable_lane_affine_buffer,
                                use_paired_leaf_gather, dispatch_worker_count,
                                use_native_predicated_loop,
                                pipeline_print_formats.size());
                            if (w1_sl.succeeded() && w1_pl.succeeded() &&
                                w1_sl.print_formats.empty() &&
                                w1_pl.print_formats.empty()) {
                                cand_live = build_live_ray_query_w1_handler_thunk(
                                    *module, surf_llvm.entry, w1_sl.entry,
                                    w1_pl.entry, target_width,
                                    surf_llvm.entry->getName().str() + ".live.w1.callback");
                            }
                            if (cand_live == nullptr) {
                                if (w1_sl.entry != nullptr) { w1_sl.entry->eraseFromParent(); }
                                if (w1_pl.entry != nullptr) { w1_pl.entry->eraseFromParent(); }
                            }
                        }
                    }
                    if (cand_live == nullptr) {
                        cand_live = build_live_ray_query_handler_thunk(
                            *module, surf_llvm.entry, proc_llvm.entry,
                            surf_llvm.entry->getName().str() + ".live.callback");
                    }
                    if (cand_live == nullptr) {
                        out_error = "failed to build outlined live ray-query thunk";
                        return false;
                    }
                }
                LLVMSIMDRayQueryPipelineHandlers hr{
                    .on_surface = surf_llvm.entry,
                    .on_procedural = proc_llvm.entry,
                    .on_surface_filter = sf_entry,
                    .embree_surface_filter_safe = filter_safe,
                    .surface_handler_empty = surf_empty,
                    .on_candidate_live = cand_live,
                    .live_embree_safe = live_safe,
                    .procedural_handler_empty = proc_empty,
                    .surface_exclusion_safe = surf_excl.has_value(),
                };
                if (surf_excl) {
                    hr.surface_exclusion_terminate_on_commit =
                        surf_excl->terminate_on_commit;
                    hr.surface_exclusion_capture_indices =
                        surf_excl->capture_indices;
                    hr.surface_exclusion_constants =
                        surf_excl->constants;
                }
                if (record_kernel_stats) {
                    result.direct_ray_query_pipeline_count++;
                    result.surface_filter_ray_query_pipeline_count += filter_safe;
                }
                out_handlers.emplace_back(std::move(hr));
            }
            return true;
        };

        std::unordered_map<const xir::Function *, LLVMSIMDOutlinedCallable>
            lowered_map;
        lowered_map.reserve(records.size());
        const bool enable_sparse_w1 =
            warp_width > 1u && simd_sparse_w1_outlined_max_lanes() > 0u;
        const char *w1_all_lanes_filter =
            std::getenv("LUISA_SIMD_W1_ALL_LANES_NAME");
        const auto w1_all_lanes_name =
            w1_all_lanes_filter != nullptr ?
                std::string_view{w1_all_lanes_filter} :
                std::string_view{};
        for (size_t i = 0u; i < records.size(); ++i) {
            auto &rec = records[i];
            std::vector<LLVMSIMDOutlinedCallable> child_table;
            child_table.reserve(rec.schedule.outlined_callables.size());
            bool children_support_w1 = enable_sparse_w1;
            bool uses_lane_launch_index = false;
            for (const auto *child_fn : rec.schedule.outlined_callables) {
                auto it = lowered_map.find(child_fn);
                if (it == lowered_map.end()) {
                    result.diagnostics.emplace_back(
                        "outlined callable topological order violation");
                    return result;
                }
                child_table.emplace_back(it->second);
                if (it->second.entry_w1 == nullptr ||
                    it->second.io_frame_type_w1 == nullptr) {
                    children_support_w1 = false;
                }
                uses_lane_launch_index |= it->second.uses_lane_launch_index;
            }
            std::string fn_label = "outlined";
            if (const auto *name_md =
                    rec.xir_fn->find_metadata<xir::NameMD>();
                name_md != nullptr && !name_md->name().empty()) {
                fn_label = std::string{name_md->name()};
                for (char &ch : fn_label) {
                    if (!((ch >= 'a' && ch <= 'z') ||
                          (ch >= 'A' && ch <= 'Z') ||
                          (ch >= '0' && ch <= '9') || ch == '_')) {
                        ch = '_';
                    }
                }
            }
            auto llvm_name = handler_name_base + ".outlined." +
                             std::to_string(i) + "." + fn_label +
                             ".simd_w" + std::to_string(warp_width);
            std::vector<LLVMSIMDRayQueryPipelineHandlers> outlined_rq_handlers;
            if (!rec.schedule.ray_query_pipelines.empty()) {
                std::string rq_error;
                if (!lower_outlined_ray_query_pipelines(
                        rec.xir_fn, *rec.schedule.function,
                        rec.schedule.ray_query_pipelines,
                        warp_width, llvm_name, true,
                        outlined_rq_handlers, rq_error)) {
                    result.diagnostics.emplace_back(std::move(rq_error));
                    return result;
                }
            }
            auto lowered = lower_outlined_callable_schedule_to_llvm(
                *module, *rec.schedule.function, rec.xir_fn->type(),
                warp_width, llvm_name, enable_fast_math,
                static_block_size, enable_uniform_buffer_broadcast,
                enable_lane_affine_buffer, use_paired_leaf_gather,
                dispatch_worker_count, use_native_predicated_loop,
                pipeline_print_formats.size(), use_native_vector_compress,
                use_biased_narrow_buffer_gather,
                use_gathered_native_texture_read,
                use_native_half4_texture_packet,
                enable_interleaved_private_arrays,
                enable_contiguous_private_access, child_table,
                rec.param_soa_cells, 0u, outlined_rq_handlers);
            if (!lowered.succeeded()) {
                result.diagnostics.emplace_back(
                    "outlined callable LLVM lowering failed: " + lowered.error);
                return result;
            }
            outlined_interleaved_private_arrays +=
                lowered.interleaved_private_arrays;
            outlined_contiguous_private_read_count +=
                lowered.contiguous_private_read_count;
            outlined_contiguous_private_write_count +=
                lowered.contiguous_private_write_count;
            pipeline_print_formats.insert(
                pipeline_print_formats.end(),
                std::make_move_iterator(lowered.print_formats.begin()),
                std::make_move_iterator(lowered.print_formats.end()));

            ::llvm::Function *entry_w1 = nullptr;
            ::llvm::StructType *io_frame_type_w1 = nullptr;
            const bool prefer_w1 =
                !w1_all_lanes_name.empty() &&
                fn_label.find(w1_all_lanes_name) != std::string::npos;
            if (children_support_w1) {
                bool eligible_w1 = true;
                for (const auto &val : rec.schedule.function->values()) {
                    if (val.origin == schedule::ValueOrigin::special_register) {
                        auto *md = std::get_if<
                            schedule::SpecialRegisterValueMetadata>(&val.metadata);
                        if (md == nullptr) {
                            eligible_w1 = false;
                            break;
                        }
                        auto tag = static_cast<xir::DerivedSpecialRegisterTag>(md->tag);
                        if (tag == xir::DerivedSpecialRegisterTag::THREAD_ID ||
                            tag == xir::DerivedSpecialRegisterTag::DISPATCH_ID) {
                            uses_lane_launch_index = true;
                        } else if (tag != xir::DerivedSpecialRegisterTag::BLOCK_ID &&
                                   tag != xir::DerivedSpecialRegisterTag::BLOCK_SIZE &&
                                   tag != xir::DerivedSpecialRegisterTag::DISPATCH_SIZE &&
                                   tag != xir::DerivedSpecialRegisterTag::KERNEL_ID) {
                            eligible_w1 = false;
                            break;
                        }
                    }
                }
                if (eligible_w1) {
                    for (const auto &block : rec.schedule.function->blocks()) {
                        if (std::holds_alternative<schedule::BlockBarrierTerminator>(
                                block.terminator)) {
                            eligible_w1 = false;
                            break;
                        }
                        for (const auto &inst : block.instructions) {
                            if (inst.opcode == schedule::Opcode::warp_collective ||
                                inst.opcode == schedule::Opcode::opaque) {
                                eligible_w1 = false;
                                break;
                            }
                        }
                        if (!eligible_w1) { break; }
                    }
                }
                std::vector<LLVMSIMDRayQueryPipelineHandlers> outlined_rq_handlers_w1;
                auto llvm_name_w1 = handler_name_base + ".outlined." +
                                    std::to_string(i) + "." + fn_label +
                                    ".simd_w1";
                if (eligible_w1 && !rec.schedule.ray_query_pipelines.empty()) {
                    std::string rq_w1_error;
                    if (!lower_outlined_ray_query_pipelines(
                            rec.xir_fn, *rec.schedule.function,
                            rec.schedule.ray_query_pipelines,
                            1u, llvm_name_w1, false,
                            outlined_rq_handlers_w1, rq_w1_error)) {
                        eligible_w1 = false;
                    }
                }
                if (eligible_w1) {
                    std::vector<LLVMSIMDOutlinedCallable> child_table_w1;
                    child_table_w1.reserve(child_table.size());
                    for (const auto &child : child_table) {
                        auto w1_child = child;
                        w1_child.entry = child.entry_w1;
                        w1_child.io_frame_type = child.io_frame_type_w1;
                        w1_child.entry_w1 = nullptr;
                        w1_child.io_frame_type_w1 = nullptr;
                        child_table_w1.emplace_back(std::move(w1_child));
                    }
                    auto lowered_w1 = lower_outlined_callable_schedule_to_llvm(
                        *module, *rec.schedule.function, rec.xir_fn->type(),
                        1u, llvm_name_w1, enable_fast_math,
                        static_block_size, enable_uniform_buffer_broadcast,
                        enable_lane_affine_buffer, false,
                        dispatch_worker_count, false,
                        pipeline_print_formats.size(), false,
                        false, false, false,
                        enable_interleaved_private_arrays,
                        enable_contiguous_private_access, child_table_w1,
                        rec.param_soa_cells, warp_width,
                        outlined_rq_handlers_w1);
                    if (lowered_w1.succeeded()) {
                        entry_w1 = lowered_w1.entry;
                        io_frame_type_w1 = lowered_w1.outlined_io_frame_type;
                        pipeline_print_formats.insert(
                            pipeline_print_formats.end(),
                            std::make_move_iterator(lowered_w1.print_formats.begin()),
                            std::make_move_iterator(lowered_w1.print_formats.end()));
                    } else if (lowered_w1.entry != nullptr) {
                        lowered_w1.entry->eraseFromParent();
                    }
                }
            }

            lowered_map.emplace(
                rec.xir_fn,
                LLVMSIMDOutlinedCallable{
                    .entry = lowered.entry,
                    .io_frame_type = lowered.outlined_io_frame_type,
                    .entry_w1 = entry_w1,
                    .io_frame_type_w1 = io_frame_type_w1,
                    .uses_lane_launch_index = uses_lane_launch_index,
                    .prefer_w1_all_lanes = prefer_w1 && entry_w1 != nullptr,
                    .return_type = rec.xir_fn->type(),
                    .parameter_types = std::move(rec.param_types),
                    .parameter_argument_tags = std::move(rec.param_tags),
                    .parameter_value_classes = std::move(rec.param_classes),
                    .parameter_soa_cell_bytes = std::move(rec.param_soa_cells),
                });
        }
        kernel_outlined_callables.reserve(
            schedule_result.outlined_callables.size());
        for (const auto *callee : schedule_result.outlined_callables) {
            kernel_outlined_callables.emplace_back(lowered_map.at(callee));
        }
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
        use_native_half4_texture_packet,
        private_stack_budget_bytes, enable_interleaved_private_arrays, enable_contiguous_private_access,
        kernel_outlined_callables);
    if (!llvm_result.succeeded()) {
        result.diagnostics.emplace_back(llvm_result.error);
        return result;
    }
    result.argument_buffer_size = llvm_result.argument_buffer_size;
    result.private_workspace_size = llvm_result.private_workspace_size;
    result.interleaved_private_arrays =
        llvm_result.interleaved_private_arrays +
        outlined_interleaved_private_arrays;
    result.contiguous_private_read_count =
        llvm_result.contiguous_private_read_count +
        outlined_contiguous_private_read_count;
    result.contiguous_private_write_count =
        llvm_result.contiguous_private_write_count +
        outlined_contiguous_private_write_count;
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
    result.surface_exclusion_soa_state_count =
        llvm_result.surface_exclusion_soa_state_count;
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
    result.guarded_byte4_texture_read_count =
        llvm_result.guarded_byte4_texture_read_count;
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
    result.full_packet_specialization_count = llvm_result.full_packet_specialization_count;
    result.full_packet_cloned_instruction_count = llvm_result.full_packet_cloned_instruction_count;
    result.full_packet_simplification_requested = llvm_result.full_packet_simplification_requested;
    result.full_packet_specialization_decision = llvm_result.full_packet_specialization_decision;
    result.full_packet_source_instruction_count = llvm_result.full_packet_source_instruction_count;
    result.full_packet_candidate_instruction_count = llvm_result.full_packet_candidate_instruction_count;
    result.full_packet_simplification_round_count = llvm_result.full_packet_simplification_round_count;
    result.shared_memory_size = llvm_result.shared_memory_size;
    result.block_barrier_count = llvm_result.block_barrier_count;
    result.block_barrier_loop_epoch_count =
        llvm_result.block_barrier_loop_epoch_count;
    result.cooperative_block = llvm_result.cooperative_block;
    result.direct_control_flow = llvm_result.direct_control_flow;
    if (detail::env_flag("LUISA_SIMD_REPORT_OPTIMIZATIONS")) {
        auto function_count = size_t{0u};
        auto block_count = size_t{0u};
        auto instruction_count = size_t{0u};
        auto alloca_count = size_t{0u};
        for (auto &&function : *module) {
            if (function.isDeclaration()) { continue; }
            function_count++;
            for (auto &&block : function) {
                block_count++;
                for (auto &&instruction : block) {
                    instruction_count++;
                    alloca_count += instruction.getOpcode() ==
                                    ::llvm::Instruction::Alloca;
                }
            }
        }
        LUISA_INFO(
            "SIMD pre-JIT module '{}': functions={}, blocks={}, "
            "instructions={}, allocas={}, state_slots={}, "
            "coalesced_state_slots={}, general_colored_state_slots={}",
            entry_name, function_count, block_count,
            instruction_count, alloca_count,
            result.state_slot_count,
            result.coalesced_state_slot_count,
            result.general_colored_state_slot_count);
        LUISA_INFO(
            "SIMD pre-JIT state residency '{}': cold_state_slots={}, "
            "stack_pinned_state_slots={}",
            entry_name, result.cold_state_slot_count,
            result.stack_pinned_state_slot_count);
    }
    if (enable_fast_math) {
        // Runtime dispatch establishes f32 FTZ/DAZ for the complete JIT
        // call tree. Other types follow the target's dynamic environment
        // (for example, MXCSR also affects f64, but not every f16 op).
        // Permit FMA contraction on shader arithmetic while preserving
        // native-math helper and strict-MMA contraction policies.
        for (auto &function : *module) {
            if (function.isDeclaration()) { continue; }
#if LLVM_VERSION_MAJOR >= 23
            auto environment = ::llvm::DenormalFPEnv{
                ::llvm::DenormalMode::getDynamic(),
                ::llvm::DenormalMode::getPreserveSign()};
            ::llvm::AttrBuilder attributes{*context};
            attributes.addDenormalFPEnvAttr(environment);
            function.addFnAttrs(attributes);
#else
            function.addFnAttr("denormal-fp-math", "dynamic,dynamic");
            function.addFnAttr("denormal-fp-math-f32", "preserve-sign,preserve-sign");
#endif
            if (!function.hasFnAttribute("luisa.cpu.native_math") &&
                function.getFnAttribute("fp-contract").getValueAsString() != "off") {
                luisa::compute::detail::apply_llvm_fp_function_options(
                    function, luisa::compute::detail::LLVMFPOpFusion::Fast);
            }
        }
    }
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
    if (capture_assembly || capture_ir) {
        ::llvm::raw_string_ostream llvm_ir_stream{result.llvm_ir};
        module->print(llvm_ir_stream, nullptr);
        llvm_ir_stream.flush();
    }
    if (capture_assembly) {
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
        result.entry_symbol = llvm_block_batch_entry_name;
        result.block_batch_entry = result.jit->lookup(
            llvm_block_batch_entry_name);
        if (result.block_batch_entry == nullptr) {
            result.diagnostics.emplace_back(result.jit->error());
            result.jit.reset();
        }
    } else if (llvm_packet_batch_entry_name.empty()) {
        result.entry_symbol = llvm_entry_name;
        result.entry = result.jit->lookup(llvm_entry_name);
        if (result.entry == nullptr) {
            result.diagnostics.emplace_back(result.jit->error());
            result.jit.reset();
        }
    } else {
        result.entry_symbol = llvm_packet_batch_entry_name;
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
    luisa::string_view entry_name, bool enable_fast_math,
    bool capture_assembly,
    uint32_t dispatch_worker_count,
    bool enable_packet_batch_entry,
    bool enable_block_batch_entry,
    bool capture_object) {
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
    auto decompose_vector_allocas =
        warp_width >= 4u &&
        !detail::env_flag(
            "LUISA_SIMD_DISABLE_VECTOR_ALLOCA_PROMOTION");
    auto promote_aggregate_allocas = [&]() noexcept {
        if (detail::env_flag(
                "LUISA_SIMD_DISABLE_AGGREGATE_PROMOTION")) {
            return;
        }
        auto info = xir::sroa_pass_run_on_module(
            module.get(),
            {.decompose_vectors = decompose_vector_allocas,
             .single_block_vectors_only = true});
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
        (warp_width == 1u || warp_width == 4u || warp_width == 8u ||
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
    const bool live_w8_ray_query_pipeline =
        warp_width == 8u &&
        !detail::env_flag("LUISA_SIMD_DISABLE_LIVE_RAY_QUERY_PIPELINE");
    auto ray_query_pipeline_options =
        xir::LowerRayQueryToPipelineOptions{
            .max_captured_argument_count =
                enable_captured_ray_query_pipeline ?
                    (force_captured_ray_query_pipeline ?
                         std::numeric_limits<size_t>::max() :
                         (live_w8_ray_query_pipeline ? size_t{64u} :
                          warp_width == 8u ? size_t{16u} : size_t{4u})) :
                    0u,
            .min_handler_instruction_count =
                enable_ray_query_pipeline_profitability &&
                        warp_width >= 4u &&
                        !live_w8_ray_query_pipeline ?
                    24u :
                    0u,
            .min_small_handler_loop_count =
                live_w8_ray_query_pipeline ? size_t{1u} : size_t{2u}};
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
    // structured regions into the caller. Both passes override NoInlineMD and
    // diagnostic call-site metadata because the SIMD scheduler requires all
    // user callables ($outline / CallableFunction) to be inlined into the
    // entry kernel.
    constexpr auto simd_inline_options = xir::InlineOptions{
        .consume_call_site_diagnostic_metadata = true,
        .override_noinline = false,
    };
    auto_outline_shared_value_callables(module.get(), warp_width);
    sanitize_ineligible_outlined_callables(module.get(), warp_width);
    strip_debug_call_metadata_for_legalization(module.get());
    static_cast<void>(
        xir::inline_all_pass_run_on_module(module.get(), simd_inline_options));
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
    sanitize_ineligible_outlined_callables(module.get(), warp_width);
    strip_debug_call_metadata_for_legalization(module.get());
    static_cast<void>(
        xir::inline_all_pass_run_on_module(module.get(), simd_inline_options));
    promote_aggregate_allocas();
    static_cast<void>(xir::mem2reg_pass_run_on_module(module.get()));
    static_cast<void>(xir::dce_pass_run_on_module(module.get()));
    // Re-evaluate post-destructure/post-inline eligibility (e.g. if an inlined
    // ordinary helper brought a ray-query loop into an outlined callable).
    auto before_recheck_noinline = size_t{0u};
    for (auto *fn : module->function_list()) {
        if (fn->isa<xir::CallableFunction>() &&
            fn->find_metadata<xir::NoInlineMD>() != nullptr) {
            ++before_recheck_noinline;
        }
    }
    sanitize_ineligible_outlined_callables(module.get(), warp_width);
    auto after_recheck_noinline = size_t{0u};
    for (auto *fn : module->function_list()) {
        if (fn->isa<xir::CallableFunction>() &&
            fn->find_metadata<xir::NoInlineMD>() != nullptr) {
            ++after_recheck_noinline;
        }
    }
    if (after_recheck_noinline < before_recheck_noinline) {
        strip_debug_call_metadata_for_legalization(module.get());
        static_cast<void>(
            xir::inline_all_pass_run_on_module(module.get(), simd_inline_options));
        promote_aggregate_allocas();
        static_cast<void>(xir::mem2reg_pass_run_on_module(module.get()));
        static_cast<void>(xir::dce_pass_run_on_module(module.get()));
    }
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
    auto buffer_read_latency_info =
        schedule::BufferReadLatencyHidingInfo{};
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
    // On the measured W4/W8 voxel traversal, issuing the bounded DDA
    // continuation before the read overlaps it with the gather/hit-test
    // dependency chain without adding memory effects. W16 regresses from the
    // extra live vectors, so it deliberately remains outside this width policy
    // instead of relying on a target-feature guess.
    if ((warp_width == 4u || warp_width == 8u) &&
        !detail::env_flag(
            "LUISA_SIMD_DISABLE_BUFFER_READ_LATENCY_HIDING")) {
        buffer_read_latency_info =
            schedule::hide_innermost_buffer_read_latency(
                xir_kernel);
    }
    if (predication_info.changed() ||
        buffer_read_latency_info.changed()) {
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
    // Apply the same per-function SIMD scheduling rewrites to surviving
    // NoInlineMD outlined callables.
    auto any_outlined_changed = false;
    for (auto *fn : module->function_list()) {
        if (!fn->isa<xir::CallableFunction>() || fn->definition() == nullptr ||
            fn->find_metadata<xir::NoInlineMD>() == nullptr) {
            continue;
        }
        auto fn_barrier = schedule::canonicalize_block_barriers(fn);
        if (!fn_barrier.succeeded()) {
            SIMDCompiledKernel result{.warp_width = warp_width};
            result.diagnostics.emplace_back(
                "XIR outlined callable block-barrier canonicalization failed: " +
                fn_barrier.error);
            return result;
        }
        if (warp_width != 1u &&
            !detail::env_flag("LUISA_SIMD_DISABLE_PREDICATED_IF")) {
            auto enable_refinement =
                (warp_width == 4u || warp_width == 8u) &&
                !detail::env_flag(
                    "LUISA_SIMD_DISABLE_PREDICATED_IF_REFINEMENT");
            auto enable_deep_refinement =
                enable_refinement && warp_width == 8u &&
                !detail::env_flag(
                    "LUISA_SIMD_DISABLE_DEEP_PREDICATED_IF_REFINEMENT");
            auto enable_wide_refinement =
                enable_deep_refinement &&
                !detail::env_flag(
                    "LUISA_SIMD_DISABLE_WIDE_PREDICATED_IF_REFINEMENT");
            auto max_speculation_cost = enable_deep_refinement ? 16u : 12u;
            auto fn_pred = schedule::predicate_small_varying_diamonds(
                fn, enable_refinement, max_speculation_cost,
                warp_width != 1u &&
                    !detail::env_flag(
                        "LUISA_SIMD_DISABLE_WIDENED_PREDICATED_UPDATE"),
                enable_wide_refinement, false);
            any_outlined_changed |= fn_pred.changed();
        }
        if ((warp_width == 4u || warp_width == 8u) &&
            !detail::env_flag(
                "LUISA_SIMD_DISABLE_BUFFER_READ_LATENCY_HIDING")) {
            auto fn_lat = schedule::hide_innermost_buffer_read_latency(fn);
            any_outlined_changed |= fn_lat.changed();
        }
        if (!detail::env_flag("LUISA_SIMD_DISABLE_LOOP_UNSWITCH")) {
            auto fn_unsw = schedule::unswitch_invariant_varying_loop_condition(
                fn,
                !detail::env_flag(
                    "LUISA_SIMD_DISABLE_GUARDED_LOOP_UNSWITCH"));
            any_outlined_changed |= fn_unsw.changed();
        }
    }
    if (any_outlined_changed) {
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
        enable_block_batch_entry,
        false, 0u,
        !detail::env_flag(
            "LUISA_SIMD_DISABLE_INTERLEAVED_PRIVATE_ARRAYS"),
        true, capture_object);
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
    result.buffer_read_latency_hidden_diamond_count =
        buffer_read_latency_info.hidden_diamond_count;
    result.buffer_read_latency_moved_instruction_count =
        buffer_read_latency_info.moved_instruction_count;
    result.buffer_read_latency_generated_select_count =
        buffer_read_latency_info.generated_select_count;
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
