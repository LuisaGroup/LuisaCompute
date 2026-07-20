//
// Created by mike on 4/8/26.
//

#include <luisa/dsl/rtx/ray_query.h>

#include "hip_codegen_llvm_impl.h"

namespace luisa::compute::hip {

void HIPCodegenLLVMImpl::_translate_ray_query_loop_inst(IB &b, FunctionContext &func_ctx, const xir::RayQueryLoopInst *inst) noexcept {
    b.GetInsertBlock()->setName("ray.query.loop");
    auto llvm_dispatch_block = func_ctx.get_local_value<llvm::BasicBlock>(inst->dispatch_block());
    llvm_dispatch_block->setName("ray.query.dispatch");
    b.CreateBr(llvm_dispatch_block);
}

void HIPCodegenLLVMImpl::_translate_ray_query_dispatch_inst(IB &b, FunctionContext &func_ctx, const xir::RayQueryDispatchInst *inst) noexcept {
    // luisa.ray.query.proceed();
    // switch (luisa.ray.query.state()) {
    //    case surface: br surface_block
    //    case procedural: br procedural_block
    //    default: br exit_block
    // }
    auto llvm_state_ptr = _get_ray_query_state_pointer(
        b, func_ctx, inst->query_object());
    auto llvm_state = _advance_ray_query(b, llvm_state_ptr);
    auto llvm_exit_block = func_ctx.get_local_value<llvm::BasicBlock>(inst->exit_block());
    llvm_exit_block->setName("ray.query.exit");
    auto llvm_surface_block = func_ctx.get_local_value<llvm::BasicBlock>(inst->on_surface_candidate_block());
    llvm_surface_block->setName("ray.query.on.surface.candidate");
    auto llvm_procedural_block = func_ctx.get_local_value<llvm::BasicBlock>(inst->on_procedural_candidate_block());
    llvm_procedural_block->setName("ray.query.on.procedural.candidate");
    auto llvm_dispatch = b.CreateSwitch(llvm_state, llvm_exit_block, 2);
    llvm_dispatch->addCase(b.getInt8(llvm_ray_query_state_surface_candidate), llvm_surface_block);
    llvm_dispatch->addCase(b.getInt8(llvm_ray_query_state_procedural_candidate), llvm_procedural_block);
}

llvm::Value *HIPCodegenLLVMImpl::_translate_ray_query_object_read_inst(IB &b, FunctionContext &func_ctx, const xir::RayQueryObjectReadInst *inst) noexcept {
    LUISA_DEBUG_ASSERT(inst->operand_count() == 1);
    auto op = inst->op();
    auto llvm_state_ptr = _get_ray_query_state_pointer(
        b, func_ctx, inst->operand(0));

    switch (op) {
        case xir::RayQueryObjectReadOp::RAY_QUERY_OBJECT_IS_TRIANGLE_CANDIDATE:
            return _call_ray_query_intrinsic(b, llvm_state_ptr, llvm_ray_query_intrinsic_name_is_surface_candidate, b.getInt1Ty(), {});
        case xir::RayQueryObjectReadOp::RAY_QUERY_OBJECT_IS_PROCEDURAL_CANDIDATE:
            return _call_ray_query_intrinsic(b, llvm_state_ptr, llvm_ray_query_intrinsic_name_is_procedural_candidate, b.getInt1Ty(), {});
        case xir::RayQueryObjectReadOp::RAY_QUERY_OBJECT_IS_TERMINATED:
            return _call_ray_query_intrinsic(b, llvm_state_ptr, llvm_ray_query_intrinsic_name_is_terminated, b.getInt1Ty(), {});
        case xir::RayQueryObjectReadOp::RAY_QUERY_OBJECT_WORLD_SPACE_RAY: {
            auto ox = _call_ray_query_intrinsic(b, llvm_state_ptr, llvm_ray_query_intrinsic_name_ray_origin_x, b.getFloatTy(), {});
            auto oy = _call_ray_query_intrinsic(b, llvm_state_ptr, llvm_ray_query_intrinsic_name_ray_origin_y, b.getFloatTy(), {});
            auto oz = _call_ray_query_intrinsic(b, llvm_state_ptr, llvm_ray_query_intrinsic_name_ray_origin_z, b.getFloatTy(), {});
            auto tmin = _call_ray_query_intrinsic(b, llvm_state_ptr, llvm_ray_query_intrinsic_name_ray_tmin, b.getFloatTy(), {});
            auto dx = _call_ray_query_intrinsic(b, llvm_state_ptr, llvm_ray_query_intrinsic_name_ray_direction_x, b.getFloatTy(), {});
            auto dy = _call_ray_query_intrinsic(b, llvm_state_ptr, llvm_ray_query_intrinsic_name_ray_direction_y, b.getFloatTy(), {});
            auto dz = _call_ray_query_intrinsic(b, llvm_state_ptr, llvm_ray_query_intrinsic_name_ray_direction_z, b.getFloatTy(), {});
            auto tmax = _call_ray_query_intrinsic(b, llvm_state_ptr, llvm_ray_query_intrinsic_name_ray_tmax, b.getFloatTy(), {});
            auto llvm_f32x3_array_type = llvm::ArrayType::get(b.getFloatTy(), 3);
            auto origin = static_cast<llvm::Value *>(llvm::PoisonValue::get(llvm_f32x3_array_type));
            origin = b.CreateInsertValue(origin, ox, 0);
            origin = b.CreateInsertValue(origin, oy, 1);
            origin = b.CreateInsertValue(origin, oz, 2);
            auto direction = static_cast<llvm::Value *>(llvm::PoisonValue::get(llvm_f32x3_array_type));
            direction = b.CreateInsertValue(direction, dx, 0);
            direction = b.CreateInsertValue(direction, dy, 1);
            direction = b.CreateInsertValue(direction, dz, 2);
            auto result_type = _get_llvm_ray_type();
            auto result = static_cast<llvm::Value *>(llvm::PoisonValue::get(result_type));
            result = b.CreateInsertValue(result, origin, llvm_ray_type_origin_index);
            result = b.CreateInsertValue(result, tmin, llvm_ray_type_t_min_index);
            result = b.CreateInsertValue(result, direction, llvm_ray_type_direction_index);
            result = b.CreateInsertValue(result, tmax, llvm_ray_type_t_max_index);
            return result;
        }
        case xir::RayQueryObjectReadOp::RAY_QUERY_OBJECT_TRIANGLE_CANDIDATE_HIT: {
            auto inst_id = _call_ray_query_intrinsic(b, llvm_state_ptr, llvm_ray_query_intrinsic_name_candidate_inst_id, b.getInt32Ty(), {});
            auto prim_id = _call_ray_query_intrinsic(b, llvm_state_ptr, llvm_ray_query_intrinsic_name_candidate_prim_id, b.getInt32Ty(), {});
            auto u = _call_ray_query_intrinsic(b, llvm_state_ptr, llvm_ray_query_intrinsic_name_candidate_bary_u, b.getFloatTy(), {});
            auto v = _call_ray_query_intrinsic(b, llvm_state_ptr, llvm_ray_query_intrinsic_name_candidate_bary_v, b.getFloatTy(), {});
            auto t = _call_ray_query_intrinsic(b, llvm_state_ptr, llvm_ray_query_intrinsic_name_candidate_hit_t, b.getFloatTy(), {});
            auto bary = _create_llvm_vector(b, {u, v});
            auto result_type = _get_llvm_surface_hit_type();
            auto result = static_cast<llvm::Value *>(llvm::PoisonValue::get(result_type));
            result = b.CreateInsertValue(result, inst_id, llvm_surface_hit_type_inst_id_index);
            result = b.CreateInsertValue(result, prim_id, llvm_surface_hit_type_prim_id_index);
            result = b.CreateInsertValue(result, bary, llvm_surface_hit_type_bary_index);
            result = b.CreateInsertValue(result, t, llvm_surface_hit_type_t_index);
            return result;
        }
        case xir::RayQueryObjectReadOp::RAY_QUERY_OBJECT_PROCEDURAL_CANDIDATE_HIT: {
            auto inst_id = _call_ray_query_intrinsic(b, llvm_state_ptr, llvm_ray_query_intrinsic_name_candidate_inst_id, b.getInt32Ty(), {});
            auto prim_id = _call_ray_query_intrinsic(b, llvm_state_ptr, llvm_ray_query_intrinsic_name_candidate_prim_id, b.getInt32Ty(), {});
            auto result_type = _get_llvm_procedural_hit_type();
            auto result = static_cast<llvm::Value *>(llvm::PoisonValue::get(result_type));
            result = b.CreateInsertValue(result, inst_id, llvm_procedural_hit_type_inst_id_index);
            result = b.CreateInsertValue(result, prim_id, llvm_procedural_hit_type_prim_id_index);
            return result;
        }
        case xir::RayQueryObjectReadOp::RAY_QUERY_OBJECT_COMMITTED_HIT: {
            auto inst_id = _call_ray_query_intrinsic(b, llvm_state_ptr, llvm_ray_query_intrinsic_name_committed_inst_id, b.getInt32Ty(), {});
            auto prim_id = _call_ray_query_intrinsic(b, llvm_state_ptr, llvm_ray_query_intrinsic_name_committed_prim_id, b.getInt32Ty(), {});
            auto hit_kind = _call_ray_query_intrinsic(b, llvm_state_ptr, llvm_ray_query_intrinsic_name_committed_hit_kind, b.getInt32Ty(), {});
            // Read committed hit float fields through wrapper function calls,
            // then pass each result through an inline asm barrier to make
            // the value opaque to the optimizer.  Without this barrier, the
            // LLVM O2 pipeline (specifically FunctionAttrs + downstream DCE)
            // eliminates the entire barycentric-interpolation → hit-position
            // → shadow-ray computation chain because the float values only
            // feed into FP math (no observable memory side-effects), unlike
            // the integer fields which are used for buffer GEP+load.
            auto u_raw = _call_ray_query_intrinsic(b, llvm_state_ptr, llvm_ray_query_intrinsic_name_committed_bary_u, b.getFloatTy(), {});
            auto u = _create_opaque_float_barrier(b, u_raw, "committed.bary.u");
            auto v_raw = _call_ray_query_intrinsic(b, llvm_state_ptr, llvm_ray_query_intrinsic_name_committed_bary_v, b.getFloatTy(), {});
            auto v = _create_opaque_float_barrier(b, v_raw, "committed.bary.v");
            auto t_raw = _call_ray_query_intrinsic(b, llvm_state_ptr, llvm_ray_query_intrinsic_name_committed_hit_t, b.getFloatTy(), {});
            auto t = _create_opaque_float_barrier(b, t_raw, "committed.hit.t");
            auto bary = _create_llvm_vector(b, {u, v});
            auto result_type = _get_llvm_committed_hit_type();
            auto result = static_cast<llvm::Value *>(llvm::PoisonValue::get(result_type));
            result = b.CreateInsertValue(result, inst_id, llvm_committed_hit_type_inst_id_index);
            result = b.CreateInsertValue(result, prim_id, llvm_committed_hit_type_prim_id_index);
            result = b.CreateInsertValue(result, bary, llvm_committed_hit_type_bary_index);
            result = b.CreateInsertValue(result, hit_kind, llvm_committed_hit_type_hit_kind_index);
            result = b.CreateInsertValue(result, t, llvm_committed_hit_type_t_index);
            return result;
        }
        default: break;
    }
    LUISA_ERROR("Invalid op (code = {}) for RayQueryObjectReadInst.", luisa::to_underlying(op));
}

void HIPCodegenLLVMImpl::_translate_ray_query_object_write_inst(IB &b, FunctionContext &func_ctx, const xir::RayQueryObjectWriteInst *inst) noexcept {
    auto intrinsic = [op = inst->op()] {
        switch (op) {
            case xir::RayQueryObjectWriteOp::RAY_QUERY_OBJECT_COMMIT_TRIANGLE: return llvm_ray_query_intrinsic_name_commit_surface_hit;
            case xir::RayQueryObjectWriteOp::RAY_QUERY_OBJECT_COMMIT_PROCEDURAL: return llvm_ray_query_intrinsic_name_commit_procedural_hit;
            case xir::RayQueryObjectWriteOp::RAY_QUERY_OBJECT_TERMINATE: return llvm_ray_query_intrinsic_name_terminate;
            case xir::RayQueryObjectWriteOp::RAY_QUERY_OBJECT_PROCEED: return llvm_ray_query_intrinsic_name_proceed;
            default: break;
        }
        LUISA_ERROR("Invalid op (code = {}) for RayQueryObjectWriteInst.", luisa::to_underlying(op));
    }();
    LUISA_DEBUG_ASSERT(inst->type() == nullptr);
    LUISA_DEBUG_ASSERT(inst->operand_count() == 1 || inst->operand_count() == 2);
    auto llvm_state_ptr = _get_ray_query_state_pointer(
        b, func_ctx, inst->operand(0));
    llvm::SmallVector<llvm::Value *, 2> llvm_args;
    for (auto &&op_use : inst->operand_uses().subspan(1) /* skip the query object */) {
        llvm_args.emplace_back(_get_llvm_value(b, func_ctx, op_use->value()));
    }
    if (inst->op() ==
        xir::RayQueryObjectWriteOp::RAY_QUERY_OBJECT_PROCEED) {
        LUISA_DEBUG_ASSERT(llvm_args.empty());
        if (_uses_hardware_rt_stack) {
            (void)_advance_ray_query(b, llvm_state_ptr);
        } else {
            (void)_call_ray_query_intrinsic(
                b, llvm_state_ptr, intrinsic, b.getVoidTy(), llvm_args);
        }
    } else {
        (void)_call_ray_query_intrinsic(
            b, llvm_state_ptr, intrinsic, b.getVoidTy(), llvm_args);
    }
}

void HIPCodegenLLVMImpl::_translate_ray_query_pipeline_inst(IB &b, FunctionContext &func_ctx, const xir::RayQueryPipelineInst *inst) noexcept {
    auto query_object = inst->query_object();
    LUISA_ASSERT(
        query_object != nullptr &&
            (query_object->type() == Type::of<RayQueryAll>() ||
             query_object->type() == Type::of<RayQueryAny>()),
        "Invalid HIP ray-query pipeline object.");

    auto llvm_on_surface = _get_or_declare_llvm_function(
        inst->on_surface_function());
    auto llvm_on_procedural = _get_or_declare_llvm_function(
        inst->on_procedural_function());
    LUISA_ASSERT(
        llvm_on_surface->getReturnType()->isVoidTy() &&
            llvm_on_procedural->getReturnType()->isVoidTy(),
        "HIP ray-query candidate handlers must return void.");

    // Candidate handlers take the query object by reference. Materialize an
    // rvalue query if necessary; lowered AST ray queries normally already use
    // an alloca here, but accepting both forms keeps RayQueryPipelineInst's
    // documented operand contract intact.
    auto llvm_query_object = _get_llvm_value(b, func_ctx, query_object);
    llvm::Value *llvm_query_pointer;
    if (llvm_query_object->getType()->isPointerTy()) {
        llvm_query_pointer = llvm_query_object;
    } else {
        llvm_query_pointer = _create_temp_in_alloca_block(
            func_ctx, _get_llvm_type(query_object->type())->mem_type,
            _get_type_alignment(query_object->type()));
        _store_llvm_value(
            b, llvm_query_pointer, llvm_query_object,
            query_object->type());
    }
    if (llvm_query_pointer->getType()->getPointerAddressSpace() != 0u) {
        llvm_query_pointer = b.CreateAddrSpaceCast(
            llvm_query_pointer, b.getPtrTy(0),
            "ray.query.object.generic");
    }

    // Form the exact ordinary-callable ABI used by _translate_call_inst:
    // (query-ref, captures..., print?, dispatch-size, kernel-id, rt-stack...).
    llvm::SmallVector<llvm::Value *, 16> llvm_callback_args;
    llvm_callback_args.reserve(inst->captured_argument_count() + 8u);
    llvm_callback_args.emplace_back(llvm_query_pointer);
    for (auto captured_use : inst->captured_argument_uses()) {
        auto llvm_arg = _get_llvm_value(
            b, func_ctx, captured_use->value());
        if (llvm_arg->getType()->isPointerTy() &&
            llvm_arg->getType()->getPointerAddressSpace() != 0u) {
            llvm_arg = b.CreateAddrSpaceCast(
                llvm_arg, b.getPtrTy(0),
                "ray.query.capture.generic");
        }
        llvm_callback_args.emplace_back(llvm_arg);
    }
    if (_config.requires_printing) {
        auto llvm_print_buffer = static_cast<llvm::Value *>(
            llvm::PoisonValue::get(_get_llvm_print_buffer_type()));
        llvm_print_buffer = b.CreateInsertValue(
            llvm_print_buffer,
            func_ctx.llvm_print_buffer_capacity, 0u);
        llvm_print_buffer = b.CreateInsertValue(
            llvm_print_buffer,
            func_ctx.llvm_print_buffer_content, 1u);
        llvm_callback_args.emplace_back(llvm_print_buffer);
    }
    llvm_callback_args.emplace_back(_read_dispatch_size(b, func_ctx));
    llvm_callback_args.emplace_back(_read_kernel_id(b, func_ctx));
    if (_rt_analysis.uses_ray_tracing) {
        llvm_callback_args.emplace_back(func_ctx.llvm_rt_stack_size);
        llvm_callback_args.emplace_back(func_ctx.llvm_rt_stack_count);
        llvm_callback_args.emplace_back(func_ctx.llvm_rt_stack_data);
    }

    llvm::SmallVector<llvm::Type *, 16> llvm_callback_arg_types;
    llvm_callback_arg_types.reserve(llvm_callback_args.size());
    for (auto llvm_arg : llvm_callback_args) {
        llvm_callback_arg_types.emplace_back(llvm_arg->getType());
    }
    auto llvm_pipeline_type = llvm::FunctionType::get(
        b.getVoidTy(), llvm_callback_arg_types, false);
    LUISA_ASSERT(
        llvm_on_surface->getFunctionType() == llvm_pipeline_type &&
            llvm_on_procedural->getFunctionType() == llvm_pipeline_type,
        "HIP ray-query pipeline callback ABI mismatch.");

    // Keep the pipeline's control flow inside a private helper. Expanding it
    // directly in the containing LLVM block would invalidate XIR PHI incoming
    // block mappings whenever the pipeline precedes a branch.
    auto llvm_pipeline = llvm::Function::Create(
        llvm_pipeline_type, llvm::Function::PrivateLinkage,
        llvm::Twine{"luisa.ray.query.pipeline."} +
            llvm::Twine{_ray_query_pipeline_count++},
        _llvm_module.get());
    llvm_pipeline->addFnAttr(llvm::Attribute::AlwaysInline);
    llvm_pipeline->addFnAttr(llvm::Attribute::NoUnwind);

    auto llvm_entry = llvm::BasicBlock::Create(
        _llvm_context, "entry", llvm_pipeline);
    auto llvm_dispatch = llvm::BasicBlock::Create(
        _llvm_context, "dispatch", llvm_pipeline);
    auto llvm_surface = llvm::BasicBlock::Create(
        _llvm_context, "surface", llvm_pipeline);
    auto llvm_procedural = llvm::BasicBlock::Create(
        _llvm_context, "procedural", llvm_pipeline);
    auto llvm_exit = llvm::BasicBlock::Create(
        _llvm_context, "exit", llvm_pipeline);

    llvm::SmallVector<llvm::Value *, 16> llvm_pipeline_args;
    llvm_pipeline_args.reserve(llvm_pipeline->arg_size());
    for (auto &llvm_arg : llvm_pipeline->args()) {
        llvm_pipeline_args.emplace_back(&llvm_arg);
    }

    IB pipeline_b{llvm_entry};
    auto llvm_query = pipeline_b.CreateAlignedLoad(
        _get_llvm_ray_query_type(), llvm_pipeline_args.front(),
        llvm::Align{_get_type_alignment(query_object->type())},
        "ray.query.object");
    auto llvm_state_address = pipeline_b.CreateExtractValue(
        llvm_query, llvm_ray_query_type_state_index,
        "ray.query.state.address");
    auto llvm_state_pointer = pipeline_b.CreateIntToPtr(
        llvm_state_address,
        pipeline_b.getPtrTy(amdgpu_address_space_local),
        "ray.query.state");
    pipeline_b.CreateBr(llvm_dispatch);

    pipeline_b.SetInsertPoint(llvm_dispatch);
    auto llvm_state = _advance_ray_query(
        pipeline_b, llvm_state_pointer);
    auto llvm_switch = pipeline_b.CreateSwitch(
        llvm_state, llvm_exit, 2u);
    llvm_switch->addCase(
        pipeline_b.getInt8(llvm_ray_query_state_surface_candidate),
        llvm_surface);
    llvm_switch->addCase(
        pipeline_b.getInt8(llvm_ray_query_state_procedural_candidate),
        llvm_procedural);

    pipeline_b.SetInsertPoint(llvm_surface);
    auto llvm_surface_call = pipeline_b.CreateCall(
        llvm_on_surface, llvm_pipeline_args);
    llvm_surface_call->setCallingConv(
        llvm_on_surface->getCallingConv());
    pipeline_b.CreateBr(llvm_dispatch);

    pipeline_b.SetInsertPoint(llvm_procedural);
    auto llvm_procedural_call = pipeline_b.CreateCall(
        llvm_on_procedural, llvm_pipeline_args);
    llvm_procedural_call->setCallingConv(
        llvm_on_procedural->getCallingConv());
    pipeline_b.CreateBr(llvm_dispatch);

    pipeline_b.SetInsertPoint(llvm_exit);
    pipeline_b.CreateRetVoid();

    auto llvm_call = b.CreateCall(
        llvm_pipeline, llvm_callback_args);
    llvm_call->setCallingConv(llvm_pipeline->getCallingConv());
}

llvm::Value *HIPCodegenLLVMImpl::_get_ray_query_state_pointer(
    IB &b, const FunctionContext &func_ctx,
    const xir::Value *query_object) noexcept {
    LUISA_ASSERT(
        query_object != nullptr &&
            (query_object->type() == Type::of<RayQueryAll>() ||
             query_object->type() == Type::of<RayQueryAny>()),
        "Invalid HIP ray-query object operand.");
    auto llvm_query = _get_llvm_value(b, func_ctx, query_object);
    if (llvm_query->getType()->isPointerTy()) {
        llvm_query = _load_llvm_value(
            b, llvm_query, query_object->type());
    }
    LUISA_ASSERT(
        llvm_query->getType() == _get_llvm_ray_query_type(),
        "Invalid HIP ray-query LLVM object type.");
    auto llvm_state_address = b.CreateExtractValue(
        llvm_query, llvm_ray_query_type_state_index,
        "ray.query.state.address");
    return b.CreateIntToPtr(
        llvm_state_address,
        b.getPtrTy(amdgpu_address_space_local),
        "ray.query.state");
}

llvm::Value *HIPCodegenLLVMImpl::_advance_ray_query(
    IB &b, llvm::Value *llvm_state_ptr) noexcept {
    if (_uses_hardware_rt_stack) {
        return _call_ray_query_intrinsic(
            b, llvm_state_ptr,
            llvm_ray_query_intrinsic_name_advance,
            b.getInt8Ty(), {});
    }
    (void)_call_ray_query_intrinsic(
        b, llvm_state_ptr,
        llvm_ray_query_intrinsic_name_proceed,
        b.getVoidTy(), {});
    return _call_ray_query_intrinsic(
        b, llvm_state_ptr,
        llvm_ray_query_intrinsic_name_state,
        b.getInt8Ty(), {});
}

llvm::Value *HIPCodegenLLVMImpl::_call_ray_query_intrinsic(
    IB &b, llvm::Value *llvm_state_ptr, llvm::StringRef name,
    llvm::Type *ret, llvm::ArrayRef<llvm::Value *> args) noexcept {
    LUISA_ASSERT(
        llvm_state_ptr != nullptr &&
            llvm_state_ptr->getType()->isPointerTy(),
        "Invalid HIP ray-query state pointer.");
    if (!_uses_hardware_rt_stack) {
        if (llvm_state_ptr->getType()->getPointerAddressSpace() != 0u) {
            llvm_state_ptr = b.CreateAddrSpaceCast(
                llvm_state_ptr, b.getPtrTy(0),
                "rq.state.generic");
        }
    }
    llvm::SmallVector<llvm::Value *, 8> augmented_args;
    augmented_args.push_back(llvm_state_ptr);
    augmented_args.append(args.begin(), args.end());
    std::string motion_name;
    auto wrapper_name = name;
    if (_supports_hardware_rt_stack && _rt_analysis.uses_motion_ray_query) {
        static constexpr std::string_view prefix{"luisa_ray_query_"};
        LUISA_ASSERT(name.starts_with(prefix),
                     "Invalid HIP ray-query wrapper name '{}'.", name.str());
        motion_name = "luisa_motion_ray_query_";
        motion_name.append(name.drop_front(prefix.size()).str());
        wrapper_name = motion_name;
    }
    auto func = _llvm_module->getFunction(wrapper_name);
    if (func == nullptr) {
        llvm::SmallVector<llvm::Type *, 8> arg_types;
        for (auto arg : augmented_args) { arg_types.push_back(arg->getType()); }
        auto func_type = llvm::FunctionType::get(ret, arg_types, false);
        func = llvm::Function::Create(
            func_type, llvm::Function::ExternalLinkage,
            wrapper_name, _llvm_module.get());
    }
    return b.CreateCall(func, augmented_args);
}

llvm::Value *HIPCodegenLLVMImpl::_call_ray_query_intrinsic(
    IB &b, FunctionContext &func_ctx, llvm::StringRef name,
    llvm::Type *ret, llvm::ArrayRef<llvm::Value *> args) noexcept {
    return _call_ray_query_intrinsic(
        b, func_ctx.llvm_rq_state, name, ret, args);
}

llvm::Value *HIPCodegenLLVMImpl::_create_opaque_float_barrier(IB &b, llvm::Value *val, const llvm::Twine &name) noexcept {
    auto *float_ty = b.getFloatTy();
    auto *asm_func_ty = llvm::FunctionType::get(float_ty, {float_ty}, false);
    // The asm keeps a consumed value opaque to LLVM, but it has no observable
    // side effect of its own. Marking it side-effecting retained committed-hit
    // loads even when a query only inspected miss(), as traverse_any does.
    auto *ia = llvm::InlineAsm::get(asm_func_ty, "v_mov_b32 $0, $1", "=v,v", /*hasSideEffects=*/false);
    return b.CreateCall(asm_func_ty, ia, {val}, name);
}

}// namespace luisa::compute::hip
