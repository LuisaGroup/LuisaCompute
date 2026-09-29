#include <luisa/core/logging.h>
#include <luisa/dsl/rtx/ray_query.h>
#include <luisa/runtime/rtx/hit.h>

#include "../optix_api.h"
#include "cuda_codegen_llvm_impl.h"

namespace luisa::compute::cuda {

namespace {
constexpr auto context_id_index = 0u;
constexpr auto context_query_index = 1u;
constexpr auto context_dispatch_size_index = 2u;
constexpr auto context_kernel_id_index = 3u;
constexpr auto context_capture_offset = 4u;

void validate_ray_query_handler(const xir::Function *function,
                                llvm::DenseSet<const xir::Function *> &visited) noexcept {
    if (function == nullptr || !visited.insert(function).second) { return; }
    if (auto definition = function->definition()) {
        for (auto block : definition->basic_blocks()) {
            for (auto inst : block->instructions()) {
                switch (inst->derived_instruction_tag()) {
                    case xir::DerivedInstructionTag::RAY_QUERY_PIPELINE:
                    case xir::DerivedInstructionTag::RAY_QUERY_LOOP:
                    case xir::DerivedInstructionTag::RAY_QUERY_DISPATCH:
                        LUISA_ERROR_WITH_LOCATION("CUDA LLVM does not support nested ray queries inside a candidate handler.");
                    case xir::DerivedInstructionTag::CALL:
                        validate_ray_query_handler(static_cast<const xir::CallInst *>(inst)->callee(), visited);
                        break;
                    case xir::DerivedInstructionTag::RESOURCE_QUERY: {
                        switch (static_cast<const xir::ResourceQueryInst *>(inst)->op()) {
                            case xir::ResourceQueryOp::RAY_TRACING_TRACE_CLOSEST:
                            case xir::ResourceQueryOp::RAY_TRACING_TRACE_ANY:
                            case xir::ResourceQueryOp::RAY_TRACING_TRACE_CLOSEST_MOTION_BLUR:
                            case xir::ResourceQueryOp::RAY_TRACING_TRACE_ANY_MOTION_BLUR:
                            case xir::ResourceQueryOp::RAY_TRACING_QUERY_ALL:
                            case xir::ResourceQueryOp::RAY_TRACING_QUERY_ANY:
                            case xir::ResourceQueryOp::RAY_TRACING_QUERY_ALL_MOTION_BLUR:
                            case xir::ResourceQueryOp::RAY_TRACING_QUERY_ANY_MOTION_BLUR:
                                LUISA_ERROR_WITH_LOCATION("CUDA LLVM candidate handlers cannot launch nested traversal with the depth-one OptiX pipeline.");
                            default: break;
                        }
                        break;
                    }
                    default: break;
                }
            }
        }
    }
}
}// namespace

void CUDACodegenLLVMImpl::_translate_ray_query_loop_inst(IB &, FunctionContext &, const xir::RayQueryLoopInst *) noexcept {
    LUISA_ERROR_WITH_LOCATION("CUDA LLVM requires lower_ray_query_to_pipeline before emitting ray queries.");
}

void CUDACodegenLLVMImpl::_translate_ray_query_dispatch_inst(IB &, FunctionContext &, const xir::RayQueryDispatchInst *) noexcept {
    LUISA_ERROR_WITH_LOCATION("CUDA LLVM cannot emit an unlowered ray-query dispatch.");
}

llvm::Value *CUDACodegenLLVMImpl::_load_ray_query_field(IB &b, llvm::Value *query, unsigned field) noexcept {
    auto type = _get_llvm_ray_query_type();
    return b.CreateLoad(type->getStructElementType(field), b.CreateStructGEP(type, query, field));
}

void CUDACodegenLLVMImpl::_store_ray_query_field(IB &b, llvm::Value *query, unsigned field, llvm::Value *value) noexcept {
    b.CreateStore(value, b.CreateStructGEP(_get_llvm_ray_query_type(), query, field));
}

llvm::Value *CUDACodegenLLVMImpl::_ray_query_surface_candidate(IB &b) noexcept {
    auto bary = _call_optix_get_triangle_barycentrics(b);
    if (_rt_analysis.curve_basis_set.any()) {
        auto kind = _call_optix_get_hit_kind(b);
        auto triangle = b.CreateOr(b.CreateICmpEQ(kind, b.getInt32(optix::HIT_KIND_TRIANGLE_FRONT_FACE)),
                                   b.CreateICmpEQ(kind, b.getInt32(optix::HIT_KIND_TRIANGLE_BACK_FACE)));
        auto curve_bary = _create_llvm_vector(b, {_call_optix_get_curve_parameter(b),
                                                  llvm::ConstantFP::get(b.getFloatTy(), -1.)});
        bary = b.CreateSelect(triangle, bary, curve_bary);
    }
    auto hit = static_cast<llvm::Value *>(llvm::Constant::getNullValue(_get_llvm_surface_hit_type()));
    hit = b.CreateInsertValue(hit, _call_optix_read_instance_index(b), llvm_surface_hit_type_inst_id_index);
    hit = b.CreateInsertValue(hit, _call_optix_read_primitive_index(b), llvm_surface_hit_type_prim_id_index);
    hit = b.CreateInsertValue(hit, bary, llvm_surface_hit_type_bary_index);
    return b.CreateInsertValue(hit, _call_optix_get_hit_distance(b), llvm_surface_hit_type_t_index);
}

llvm::Value *CUDACodegenLLVMImpl::_translate_ray_query_object_read_inst(IB &b, FunctionContext &func_ctx, const xir::RayQueryObjectReadInst *inst) noexcept {
    LUISA_ASSERT(inst->operand_count() == 1u && inst->operand(0)->is_lvalue(), "Invalid ray-query object read.");
    auto query = _get_llvm_value(b, func_ctx, inst->operand(0));
    switch (inst->op()) {
        case xir::RayQueryObjectReadOp::RAY_QUERY_OBJECT_WORLD_SPACE_RAY:
            return _load_ray_query_field(b, query, llvm_ray_query_type_ray_index);
        case xir::RayQueryObjectReadOp::RAY_QUERY_OBJECT_CANDIDATE_OBJECT_SPACE_RAY: {
            auto ray = _call_optix_get_object_space_ray(b);
            auto world_ray = _load_ray_query_field(b, query, llvm_ray_query_type_ray_index);
            // OptiX reports the candidate distance in AH, not the committed bound.
            return b.CreateInsertValue(ray, b.CreateExtractValue(world_ray, llvm_ray_type_t_max_index), llvm_ray_type_t_max_index);
        }
        case xir::RayQueryObjectReadOp::RAY_QUERY_OBJECT_PROCEDURAL_CANDIDATE_HIT: {
            auto hit = static_cast<llvm::Value *>(llvm::Constant::getNullValue(_get_llvm_procedural_hit_type()));
            hit = b.CreateInsertValue(hit, _call_optix_read_instance_index(b), llvm_procedural_hit_type_inst_id_index);
            return b.CreateInsertValue(hit, _call_optix_read_primitive_index(b), llvm_procedural_hit_type_prim_id_index);
        }
        case xir::RayQueryObjectReadOp::RAY_QUERY_OBJECT_TRIANGLE_CANDIDATE_HIT:
            return _ray_query_surface_candidate(b);
        case xir::RayQueryObjectReadOp::RAY_QUERY_OBJECT_COMMITTED_HIT:
            return _load_ray_query_field(b, query, llvm_ray_query_type_hit_index);
        case xir::RayQueryObjectReadOp::RAY_QUERY_OBJECT_IS_TRIANGLE_CANDIDATE:
            return b.CreateICmpEQ(_load_ray_query_field(b, query, llvm_ray_query_type_state_index), b.getInt8(llvm_ray_query_state_surface_candidate));
        case xir::RayQueryObjectReadOp::RAY_QUERY_OBJECT_IS_PROCEDURAL_CANDIDATE:
            return b.CreateICmpEQ(_load_ray_query_field(b, query, llvm_ray_query_type_state_index), b.getInt8(llvm_ray_query_state_procedural_candidate));
        case xir::RayQueryObjectReadOp::RAY_QUERY_OBJECT_IS_TERMINATED:
            return b.CreateOr(_load_ray_query_field(b, query, llvm_ray_query_type_terminated_index),
                              b.CreateICmpEQ(_load_ray_query_field(b, query, llvm_ray_query_type_state_index), b.getInt8(llvm_ray_query_state_surface_terminated)));
    }
    LUISA_ERROR_WITH_LOCATION("Invalid ray-query read operation.");
}

void CUDACodegenLLVMImpl::_commit_ray_query_hit(IB &b, llvm::Value *query, llvm::Value *t, bool procedural) noexcept {
    auto ray = _load_ray_query_field(b, query, llvm_ray_query_type_ray_index);
    auto t_min = b.CreateExtractValue(ray, llvm_ray_type_t_min_index);
    auto t_max = b.CreateExtractValue(ray, llvm_ray_type_t_max_index);
    auto valid = b.CreateAnd(b.CreateFCmpOGE(t, t_min), b.CreateFCmpOLE(t, t_max));
    auto hit = static_cast<llvm::Value *>(llvm::Constant::getNullValue(_get_llvm_committed_hit_type()));
    hit = b.CreateInsertValue(hit, _call_optix_read_instance_index(b), llvm_committed_hit_type_inst_id_index);
    hit = b.CreateInsertValue(hit, _call_optix_read_primitive_index(b), llvm_committed_hit_type_prim_id_index);
    if (!procedural) {
        auto surface = _ray_query_surface_candidate(b);
        hit = b.CreateInsertValue(hit, b.CreateExtractValue(surface, llvm_surface_hit_type_bary_index), llvm_committed_hit_type_bary_index);
    }
    hit = b.CreateInsertValue(hit, b.getInt32(static_cast<uint32_t>(procedural ? HitType::Procedural : HitType::Surface)), llvm_committed_hit_type_hit_kind_index);
    hit = b.CreateInsertValue(hit, t, llvm_committed_hit_type_t_index);
    auto previous = _load_ray_query_field(b, query, llvm_ray_query_type_hit_index);
    auto selected = static_cast<llvm::Value *>(llvm::Constant::getNullValue(_get_llvm_committed_hit_type()));
    for (auto field = 0u; field < 5u; field++) {
        selected = b.CreateInsertValue(selected, b.CreateSelect(valid, b.CreateExtractValue(hit, field), b.CreateExtractValue(previous, field)), field);
    }
    _store_ray_query_field(b, query, llvm_ray_query_type_hit_index, selected);
    _store_ray_query_field(b, query, llvm_ray_query_type_ray_index,
                           b.CreateInsertValue(ray, b.CreateSelect(valid, t, t_max), llvm_ray_type_t_max_index));
    auto committed = _load_ray_query_field(b, query, llvm_ray_query_type_committed_index);
    _store_ray_query_field(b, query, llvm_ray_query_type_committed_index, b.CreateOr(committed, valid));
}

void CUDACodegenLLVMImpl::_translate_ray_query_object_write_inst(IB &b, FunctionContext &func_ctx, const xir::RayQueryObjectWriteInst *inst) noexcept {
    LUISA_ASSERT(inst->operand_count() >= 1u && inst->operand(0)->is_lvalue(), "Invalid ray-query object write.");
    auto query = _get_llvm_value(b, func_ctx, inst->operand(0));
    switch (inst->op()) {
        case xir::RayQueryObjectWriteOp::RAY_QUERY_OBJECT_COMMIT_TRIANGLE:
            _commit_ray_query_hit(b, query, _call_optix_get_hit_distance(b), false);
            break;
        case xir::RayQueryObjectWriteOp::RAY_QUERY_OBJECT_COMMIT_PROCEDURAL:
            LUISA_ASSERT(inst->operand_count() == 2u, "Procedural commit requires a distance.");
            _commit_ray_query_hit(b, query, _get_llvm_value(b, func_ctx, inst->operand(1)), true);
            break;
        case xir::RayQueryObjectWriteOp::RAY_QUERY_OBJECT_TERMINATE:
            _store_ray_query_field(b, query, llvm_ray_query_type_terminated_index, b.getTrue());
            break;
        case xir::RayQueryObjectWriteOp::RAY_QUERY_OBJECT_PROCEED:
            LUISA_ERROR_WITH_LOCATION("CUDA LLVM requires reconstruct_ray_query_loop and lower_ray_query_to_pipeline for query.proceed().");
    }
}

void CUDACodegenLLVMImpl::_translate_ray_query_pipeline_inst(IB &b, FunctionContext &func_ctx, const xir::RayQueryPipelineInst *inst) noexcept {
    llvm::DenseSet<const xir::Function *> visited;
    validate_ray_query_handler(inst->on_surface_function(), visited);
    validate_ray_query_handler(inst->on_procedural_function(), visited);
    auto generic_pointer = [&b](llvm::Value *value) noexcept {
        return value->getType()->isPointerTy() && value->getType()->getPointerAddressSpace() != 0u ?
                   b.CreateAddrSpaceCast(value, b.getPtrTy()) :
                   value;
    };
    auto query = generic_pointer(_get_llvm_value(b, func_ctx, inst->query_object()));
    auto id = static_cast<uint32_t>(_ray_query_pipelines.size());
    llvm::SmallVector<llvm::Value *> fields{
        b.getInt32(id), query, _read_dispatch_size(b, func_ctx), _read_kernel_id(b, func_ctx)};
    for (auto capture : inst->captured_argument_uses()) {
        // Keep references as references, including aliases and subobject captures.
        fields.emplace_back(generic_pointer(_get_llvm_value(b, func_ctx, capture->value())));
    }
    llvm::SmallVector<llvm::Type *> field_types;
    for (auto field : fields) { field_types.emplace_back(field->getType()); }
    auto context_type = llvm::StructType::create(_llvm_context, field_types, "luisa.ray.query.context");
    _ray_query_pipelines.emplace_back(RayQueryPipeline{inst, context_type});
    auto context = _create_temp_in_alloca_block(func_ctx, context_type, 16u);
    for (auto i = 0u; i < fields.size(); i++) {
        b.CreateStore(fields[i], b.CreateStructGEP(context_type, context, i));
    }
    auto address = b.CreatePtrToInt(generic_pointer(context), b.getInt64Ty());
    auto lo = b.CreateTrunc(address, b.getInt32Ty());
    auto hi = b.CreateTrunc(b.CreateLShr(address, 32u), b.getInt32Ty());
    auto accel = _load_ray_query_field(b, query, llvm_ray_query_type_accel_index);
    auto ray = _load_ray_query_field(b, query, llvm_ray_query_type_ray_index);
    auto time = _load_ray_query_field(b, query, llvm_ray_query_type_time_index);
    auto mask = _load_ray_query_field(b, query, llvm_ray_query_type_mask_index);
    // Observe opaque hits too, so the committed bound remains exact inside
    // later callbacks. The AH entry preserves opacity by accepting those hits
    // without invoking the user's surface callback.
    auto flags = optix::RAY_FLAG_DISABLE_CLOSESTHIT | optix::RAY_FLAG_ENFORCE_ANYHIT;
    if (inst->query_object()->type() == Type::of<RayQueryAny>()) {
        flags |= optix::RAY_FLAG_TERMINATE_ON_FIRST_HIT;
    }
    _call_optix_trace(b, optix::PAYLOAD_TYPE_ID_1, 5u, flags, accel, ray, time, mask, {lo, hi});
    // Explicit commits are authoritative. OptiX's terminating intersection may
    // be synthetic when terminate() is used without accepting the candidate.
    _call_optix_hit_object_reset(b);
    _store_ray_query_field(b, query, llvm_ray_query_type_state_index, b.getInt8(llvm_ray_query_state_surface_terminated));
}

void CUDACodegenLLVMImpl::_materialize_ray_query_pipelines() noexcept {
    for (auto procedural : {false, true}) {
        auto name = procedural ? "__intersection__ray_query" : "__anyhit__ray_query";
        auto function = llvm::Function::Create(llvm::FunctionType::get(llvm::Type::getVoidTy(_llvm_context), false),
                                               llvm::Function::ExternalLinkage, name, _llvm_module.get());
        function->setCallingConv(llvm::CallingConv::PTX_Kernel);
        auto entry = llvm::BasicBlock::Create(_llvm_context, "entry", function);
        IB b{entry};
        // Dead queries still require these entry points because host metadata
        // describes the original kernel rather than the optimized XIR module.
        if (_ray_query_pipelines.empty()) {
            b.CreateRetVoid();
            continue;
        }
        b.CreateCall(_get_inline_asm("call (), _optix_set_payload_types, ($0);", "r", true), {b.getInt32(optix::PAYLOAD_TYPE_ID_1)});
        auto get_payload = _get_inline_asm("call ($0), _optix_get_payload, ($1);", "=r,r", true);
        auto lo = b.CreateZExt(b.CreateCall(get_payload, {b.getInt32(0)}), b.getInt64Ty());
        auto hi = b.CreateZExt(b.CreateCall(get_payload, {b.getInt32(1)}), b.getInt64Ty());
        auto context = b.CreateIntToPtr(b.CreateOr(lo, b.CreateShl(hi, 32u)), b.getPtrTy());
        auto header_type = llvm::StructType::get(_llvm_context, {b.getInt32Ty(), b.getPtrTy()});
        auto id = b.CreateLoad(b.getInt32Ty(), b.CreateStructGEP(header_type, context, context_id_index));
        auto query = b.CreateLoad(b.getPtrTy(), b.CreateStructGEP(header_type, context, context_query_index));
        auto exit = llvm::BasicBlock::Create(_llvm_context, "exit", function);
        auto terminate = llvm::BasicBlock::Create(_llvm_context, "terminate", function);
        auto dispatch = llvm::BasicBlock::Create(_llvm_context, "dispatch", function);
        auto finish = llvm::BasicBlock::Create(_llvm_context, "candidate.finish", function);
        if (!procedural) {
            auto surface = llvm::BasicBlock::Create(_llvm_context, "surface", function);
            auto reported = llvm::BasicBlock::Create(_llvm_context, "reported", function);
            b.CreateCondBr(b.CreateICmpUGT(_call_optix_get_hit_kind(b), b.getInt32(127u)), surface, reported);
            b.SetInsertPoint(reported);
            b.CreateCondBr(_load_ray_query_field(b, query, llvm_ray_query_type_terminated_index), terminate, exit);
            b.SetInsertPoint(surface);
            _store_ray_query_field(b, query, llvm_ray_query_type_committed_index, b.getFalse());
            _store_ray_query_field(b, query, llvm_ray_query_type_state_index, b.getInt8(llvm_ray_query_state_surface_candidate));
            auto accel = _load_ray_query_field(b, query, llvm_ray_query_type_accel_index);
            auto instance = _get_accel_instance_pointer(b, accel, _call_optix_read_instance_index(b));
            auto flags = b.CreateLoad(b.getInt32Ty(), b.CreateStructGEP(_get_llvm_accel_instance_type(), instance, llvm_accel_instance_type_flags_index));
            auto opaque = b.CreateICmpEQ(b.CreateAnd(flags, b.getInt32(optix::INSTANCE_FLAG_ENFORCE_ANYHIT)), b.getInt32(0));
            auto accept_opaque = llvm::BasicBlock::Create(_llvm_context, "opaque", function);
            b.CreateCondBr(opaque, accept_opaque, dispatch);
            b.SetInsertPoint(accept_opaque);
            _commit_ray_query_hit(b, query, _call_optix_get_hit_distance(b), false);
            b.CreateBr(finish);
        } else {
            _store_ray_query_field(b, query, llvm_ray_query_type_committed_index, b.getFalse());
            _store_ray_query_field(b, query, llvm_ray_query_type_state_index, b.getInt8(llvm_ray_query_state_procedural_candidate));
            b.CreateBr(dispatch);
        }
        b.SetInsertPoint(dispatch);
        auto invalid = llvm::BasicBlock::Create(_llvm_context, "invalid.context", function);
        auto select = b.CreateSwitch(id, invalid, static_cast<unsigned>(_ray_query_pipelines.size()));
        for (auto i = 0u; i < _ray_query_pipelines.size(); i++) {
            auto &&pipeline = _ray_query_pipelines[i];
            auto block = llvm::BasicBlock::Create(_llvm_context, "pipeline." + std::to_string(i), function);
            select->addCase(b.getInt32(i), block);
            b.SetInsertPoint(block);
            auto callback = procedural ? pipeline.inst->on_procedural_function() : pipeline.inst->on_surface_function();
            if (callback != nullptr) {
                auto callee = _get_or_declare_llvm_function(callback);
                LUISA_ASSERT(callee->arg_size() == pipeline.inst->captured_argument_count() + 3u,
                             "Invalid ray-query callback capture ABI.");
                llvm::SmallVector<llvm::Value *> args{query};
                auto load_field = [&](unsigned index) noexcept {
                    return b.CreateLoad(pipeline.context_type->getElementType(index), b.CreateStructGEP(pipeline.context_type, context, index));
                };
                for (auto j = 0u; j < pipeline.inst->captured_argument_count(); j++) {
                    args.emplace_back(load_field(context_capture_offset + j));
                }
                args.emplace_back(load_field(context_dispatch_size_index));
                args.emplace_back(load_field(context_kernel_id_index));
                auto call = b.CreateCall(callee, args);
                call->setCallingConv(callee->getCallingConv());
            }
            b.CreateBr(finish);
        }
        b.SetInsertPoint(invalid);
        b.CreateUnreachable();
        b.SetInsertPoint(finish);
        auto committed = _load_ray_query_field(b, query, llvm_ray_query_type_committed_index);
        auto terminated = _load_ray_query_field(b, query, llvm_ray_query_type_terminated_index);
        if (procedural) {
            auto report = llvm::BasicBlock::Create(_llvm_context, "report", function);
            b.CreateCondBr(b.CreateOr(committed, terminated), report, exit);
            b.SetInsertPoint(report);
            auto hit = _load_ray_query_field(b, query, llvm_ray_query_type_hit_index);
            auto ray = _load_ray_query_field(b, query, llvm_ray_query_type_ray_index);
            // Reporting TMin lets AH terminate an uncommitted procedural query.
            // Its synthetic hit never replaces the explicit committed result.
            auto t = b.CreateSelect(committed, b.CreateExtractValue(hit, llvm_committed_hit_type_t_index),
                                    b.CreateExtractValue(ray, llvm_ray_type_t_min_index));
            _call_optix_report_intersection(b, b.getInt32(0), t);
            b.CreateBr(exit);
            b.SetInsertPoint(terminate);
            b.CreateUnreachable();
        } else {
            auto accept = llvm::BasicBlock::Create(_llvm_context, "accept", function);
            auto ignore = llvm::BasicBlock::Create(_llvm_context, "ignore", function);
            b.CreateCondBr(terminated, terminate, accept);
            b.SetInsertPoint(accept);
            b.CreateCondBr(committed, exit, ignore);
            b.SetInsertPoint(ignore);
            _call_optix_ignore_intersection(b);
            b.CreateUnreachable();
            b.SetInsertPoint(terminate);
            _call_optix_terminate_ray(b);
            b.CreateUnreachable();
        }
        b.SetInsertPoint(exit);
        b.CreateRetVoid();
    }
}

}// namespace luisa::compute::cuda
