//
// Created by mike on 4/8/26.
//

#include <luisa/dsl/rtx/ray_query.h>

#include "hip_codegen_llvm_impl.h"

namespace luisa::compute::hip {

size_t HIPCodegenLLVMImpl::_finalize_ray_query_pipeline_contexts() noexcept {
    size_t projected_argument_count = 0u;
    size_t separated_query_argument_count = 0u;
    size_t scalarized_context_count = 0u;
    size_t original_context_bytes = 0u;
    size_t projected_context_bytes = 0u;
    size_t max_projected_context_bytes = 0u;

    // Compute the least fixed point of interprocedural argument demand over
    // the local generated-Callable graph. A use that only forwards argument
    // (f, i) to local callee argument (g, j) contributes the equation
    // live(f, i) |= live(g, j); every other use is an observation and seeds
    // liveness. Starting from those seeds makes forwarding-only SCCs dead,
    // while a single consuming use propagates backwards through every caller.
    // Calls to declarations or interposable definitions are observations by
    // construction, so the analysis cannot erase an externally visible use.
    llvm::DenseMap<const llvm::Argument *, size_t> argument_indices;
    luisa::vector<const llvm::Argument *> arguments;
    for (auto &function : *_llvm_module) {
        if (function.isDeclaration()) { continue; }
        for (auto &argument : function.args()) {
            auto index = arguments.size();
            arguments.emplace_back(&argument);
            argument_indices.try_emplace(&argument, index);
        }
    }
    luisa::vector<luisa::vector<size_t>> reverse_dependencies(
        arguments.size());
    luisa::vector<bool> live_arguments(arguments.size(), false);
    luisa::vector<size_t> live_worklist;
    live_worklist.reserve(arguments.size());
    for (auto argument_index = 0u;
         argument_index < arguments.size(); ++argument_index) {
        auto argument = arguments[argument_index];
        auto directly_live = argument->getParent()
                                 ->getAttributes()
                                 .hasParamAttrs(argument->getArgNo());
        for (auto &use : argument->uses()) {
            auto call = llvm::dyn_cast<llvm::CallBase>(use.getUser());
            if (call != nullptr && call->isArgOperand(&use)) {
                auto callee = call->getCalledFunction();
                auto callee_argument_index = call->getArgOperandNo(&use);
                if (call->getAttributes().hasParamAttrs(
                        callee_argument_index)) {
                    directly_live = true;
                    break;
                }
                if (callee != nullptr && !callee->isDeclaration() &&
                    callee->hasLocalLinkage() &&
                    callee_argument_index < callee->arg_size()) {
                    auto callee_argument =
                        callee->getArg(callee_argument_index);
                    if (auto iter = argument_indices.find(callee_argument);
                        iter != argument_indices.end()) {
                        reverse_dependencies[iter->second]
                            .emplace_back(argument_index);
                        continue;
                    }
                }
            }
            directly_live = true;
            break;
        }
        if (directly_live) {
            live_arguments[argument_index] = true;
            live_worklist.emplace_back(argument_index);
        }
    }
    for (auto cursor = 0u; cursor < live_worklist.size(); ++cursor) {
        auto live_index = live_worklist[cursor];
        for (auto dependent : reverse_dependencies[live_index]) {
            if (!live_arguments[dependent]) {
                live_arguments[dependent] = true;
                live_worklist.emplace_back(dependent);
            }
        }
    }
    auto argument_is_live = [&](const llvm::Argument *argument) noexcept {
        auto iter = argument_indices.find(argument);
        LUISA_ASSERT(
            iter != argument_indices.end(),
            "Missing HIP generated-Callable argument demand state.");
        return live_arguments[iter->second];
    };

    for (auto &context : _llvm_ray_query_pipeline_contexts) {
        auto argument_count = context.stores.size();
        LUISA_ASSERT(
            argument_count != 0u &&
                context.loads.size() == argument_count &&
                context.on_surface->arg_size() == argument_count &&
                context.on_procedural->arg_size() == argument_count,
            "Malformed HIP synchronous ray-query callback environment.");
        LUISA_ASSERT(
            !context.on_surface->isDeclaration() &&
                !context.on_procedural->isDeclaration(),
            "HIP ray-query callback environment projection requires "
            "translated candidate handlers.");

        // Let A_i be callback ABI argument i. A_0 is the query reference: it is
        // intrinsic traversal identity and reaches the dispatcher through its
        // dedicated argument, never through user capture storage. For i > 0,
        // the environment stores A_i only when either handler demands its
        // corresponding formal argument. If both demand bits are false,
        // replacing both call operands with poison is semantics-preserving
        // under the fixed-point equations above. Taking the union is necessary
        // because candidate kind is selected dynamically inside traversal.
        llvm::SmallVector<uint32_t, 16> retained_indices;
        retained_indices.reserve(argument_count);
        for (auto i = 1u; i < argument_count; ++i) {
            auto surface_arg = context.on_surface->getArg(i);
            auto procedural_arg = context.on_procedural->getArg(i);
            if (argument_is_live(surface_arg) ||
                argument_is_live(procedural_arg)) {
                retained_indices.emplace_back(i);
            }
        }

        auto original_type = llvm::cast<llvm::StructType>(
            context.storage->getAllocatedType());
        auto original_bytes =
            _data_layout->getTypeAllocSize(original_type).getFixedValue();
        original_context_bytes += original_bytes;

        auto erase_original_field = [&](auto i) noexcept {
            auto old_store = context.stores[i];
            auto old_load = context.loads[i];
            auto old_store_gep = llvm::cast<llvm::GetElementPtrInst>(
                old_store->getPointerOperand());
            auto old_load_gep = llvm::cast<llvm::GetElementPtrInst>(
                old_load->getPointerOperand());
            old_store->eraseFromParent();
            old_load->eraseFromParent();
            LUISA_ASSERT(
                old_store_gep->use_empty() && old_load_gep->use_empty(),
                "HIP ray-query callback environment address escaped.");
            old_store_gep->eraseFromParent();
            old_load_gep->eraseFromParent();
        };
        auto erase_original_storage = [&]() noexcept {
            if (context.generic_storage != context.storage &&
                context.generic_storage->use_empty()) {
                llvm::cast<llvm::Instruction>(context.generic_storage)
                    ->eraseFromParent();
            }
            LUISA_ASSERT(
                context.storage->use_empty(),
                "HIP ray-query callback environment storage escaped.");
            context.storage->eraseFromParent();
        };

        auto dispatch_query =
            _llvm_ray_query_pipeline_dispatch->getArg(0u);
        auto dispatch_context =
            _llvm_ray_query_pipeline_dispatch->getArg(1u);
        LUISA_ASSERT(
            context.stores[0u]->getValueOperand()->getType() ==
                    dispatch_query->getType() &&
                context.loads[0u]->getType() == dispatch_query->getType() &&
                context.on_surface->getArg(0u)->getType() ==
                    dispatch_query->getType() &&
                context.on_procedural->getArg(0u)->getType() ==
                    dispatch_query->getType(),
            "HIP RayQuery callback query-reference ABI mismatch.");
        context.loads[0u]->replaceAllUsesWith(dispatch_query);
        separated_query_argument_count++;

        // An empty user environment is represented by null. The traversal
        // transports this value opaquely and the dispatcher has no remaining
        // load from it, so no zero-sized object or dummy capture is required.
        if (retained_indices.empty()) {
            context.trace_call->setArgOperand(
                1u, llvm::ConstantPointerNull::get(
                        llvm::cast<llvm::PointerType>(
                            context.trace_call->getArgOperand(1u)->getType())));
            for (auto i = 0u; i < argument_count; ++i) {
                if (i != 0u) {
                    auto old_load = context.loads[i];
                    old_load->replaceAllUsesWith(
                        llvm::PoisonValue::get(old_load->getType()));
                }
                erase_original_field(i);
            }
            erase_original_storage();
            projected_argument_count += argument_count - 1u;
            continue;
        }

        // The native traversal treats callback_context as an opaque value and
        // returns it unchanged to the generated dispatcher. If the projected
        // product consists of one generic pointer, use that pointer itself as
        // the context. This is the exact one-field-product isomorphism
        //   {p : ptr} stored behind &env  <->  p
        // and eliminates both private storage and its load without merging the
        // lifetime of p with any other captured object. A non-pointer scalar is
        // deliberately not encoded into a pointer: that would invent address
        // semantics and would not be representation-preserving.
        if (retained_indices.size() == 1u) {
            auto retained_index = retained_indices.front();
            auto retained_value =
                context.stores[retained_index]->getValueOperand();
            if (retained_value->getType()->isPointerTy() &&
                retained_value->getType()->getPointerAddressSpace() == 0u) {
                LUISA_ASSERT(
                    dispatch_context->getType() == retained_value->getType() &&
                        context.loads[retained_index]->getType() ==
                            retained_value->getType(),
                    "HIP scalar callback context pointer type mismatch.");
                context.trace_call->setArgOperand(1u, retained_value);
                for (auto i = 0u; i < argument_count; ++i) {
                    auto old_load = context.loads[i];
                    if (i == retained_index) {
                        old_load->replaceAllUsesWith(dispatch_context);
                    } else if (i != 0u) {
                        old_load->replaceAllUsesWith(
                            llvm::PoisonValue::get(old_load->getType()));
                    }
                    erase_original_field(i);
                }
                erase_original_storage();
                projected_argument_count += argument_count - 2u;
                scalarized_context_count++;
                continue;
            }
        }

        llvm::SmallVector<llvm::Type *, 16> retained_types;
        retained_types.reserve(retained_indices.size());
        llvm::SmallVector<int32_t, 16> projected_indices(
            argument_count, -1);
        for (auto projected_index = 0u;
             projected_index < retained_indices.size();
             ++projected_index) {
            auto original_index = retained_indices[projected_index];
            auto type = context.stores[original_index]
                            ->getValueOperand()
                            ->getType();
            LUISA_ASSERT(
                type == context.loads[original_index]->getType() &&
                    type == context.on_surface->getArg(original_index)->getType() &&
                    type == context.on_procedural->getArg(original_index)->getType(),
                "HIP ray-query callback environment argument type mismatch.");
            retained_types.emplace_back(type);
            projected_indices[original_index] =
                static_cast<int32_t>(projected_index);
        }

        auto projected_type = llvm::StructType::get(
            _llvm_context, retained_types, false);
        auto current_projected_context_bytes =
            _data_layout->getTypeAllocSize(projected_type).getFixedValue();
        projected_context_bytes += current_projected_context_bytes;
        max_projected_context_bytes = std::max(
            max_projected_context_bytes,
            current_projected_context_bytes);
        projected_argument_count +=
            argument_count - 1u - retained_indices.size();

        IB alloca_b{context.storage};
        auto projected_storage = alloca_b.CreateAlloca(
            projected_type, nullptr,
            context.storage->getName() + ".projected");
        projected_storage->setAlignment(
            _data_layout->getABITypeAlign(projected_type));

        IB trace_b{context.trace_call};
        llvm::Value *projected_generic_storage = projected_storage;
        if (projected_storage->getType()->getPointerAddressSpace() != 0u) {
            projected_generic_storage = trace_b.CreateAddrSpaceCast(
                projected_storage, trace_b.getPtrTy(0),
                "ray.query.context.projected.generic");
        }
        context.trace_call->setArgOperand(
            1u, projected_generic_storage);

        for (auto i = 0u; i < argument_count; ++i) {
            auto old_store = context.stores[i];
            auto old_load = context.loads[i];
            auto projected_index = projected_indices[i];
            if (i == 0u) {
                // The query-reference load was replaced by dispatch_query
                // above; only its obsolete environment field remains.
            } else if (projected_index >= 0) {
                auto type = old_store->getValueOperand()->getType();
                auto alignment = _data_layout->getABITypeAlign(type);

                IB store_b{old_store};
                auto projected_store_gep = store_b.CreateStructGEP(
                    projected_type, projected_storage,
                    static_cast<uint32_t>(projected_index),
                    "ray.query.context.projected.field");
                auto projected_store = store_b.CreateStore(
                    old_store->getValueOperand(), projected_store_gep);
                projected_store->setAlignment(alignment);

                IB load_b{old_load};
                auto projected_load_gep = load_b.CreateStructGEP(
                    projected_type,
                    dispatch_context,
                    static_cast<uint32_t>(projected_index),
                    "ray.query.context.projected.field");
                auto projected_load = load_b.CreateLoad(
                    type, projected_load_gep,
                    "ray.query.context.projected.value");
                projected_load->setAlignment(alignment);
                old_load->replaceAllUsesWith(projected_load);
            } else {
                old_load->replaceAllUsesWith(
                    llvm::PoisonValue::get(old_load->getType()));
            }
            erase_original_field(i);
        }
        erase_original_storage();
    }

    if (projected_argument_count != 0u ||
        separated_query_argument_count != 0u ||
        scalarized_context_count != 0u) {
        LUISA_VERBOSE(
            "Separated {} HIP RayQuery identity argument(s), projected {} "
            "unused callback ABI argument(s), and scalarized {} "
            "one-pointer environment(s), "
            "shrinking static environments from {} to {} bytes.",
            separated_query_argument_count,
            projected_argument_count,
            scalarized_context_count,
            original_context_bytes,
            projected_context_bytes);
    }
    _llvm_ray_query_pipeline_contexts.clear();
    return max_projected_context_bytes;
}

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

    if (_uses_synchronous_ray_query_pipeline) {
        // Materialize the exact callback environment once. The native HIPRT
        // filter/intersection callbacks receive only an opaque context pointer;
        // this typed struct restores the ordinary Callable ABI without an
        // indirect device-function call or callback-specific backend pattern.
        auto llvm_context_type = llvm::StructType::get(
            _llvm_context, llvm_callback_arg_types, false);
        auto llvm_context_pointer = _create_temp_in_alloca_block(
            func_ctx, llvm_context_type,
            _data_layout->getABITypeAlign(llvm_context_type).value());
        luisa::vector<llvm::StoreInst *> llvm_context_stores;
        llvm_context_stores.reserve(llvm_callback_args.size());
        for (auto i = 0u; i < llvm_callback_args.size(); ++i) {
            auto llvm_field = b.CreateStructGEP(
                llvm_context_type, llvm_context_pointer, i,
                "ray.query.context.field");
            llvm_context_stores.emplace_back(
                b.CreateStore(llvm_callback_args[i], llvm_field));
        }
        auto llvm_generic_context = llvm_context_pointer;
        if (llvm_generic_context->getType()->getPointerAddressSpace() != 0u) {
            llvm_generic_context = b.CreateAddrSpaceCast(
                llvm_generic_context, b.getPtrTy(0),
                "ray.query.context.generic");
        }

        // One direct switch is shared by all pipelines in the module. Each
        // case decodes its own typed context and directly invokes the two XIR
        // handlers, preserving reference captures and arbitrary side effects.
        if (_llvm_ray_query_pipeline_dispatch == nullptr) {
            auto llvm_dispatch_type = llvm::FunctionType::get(
                b.getVoidTy(),
                {b.getPtrTy(0), b.getPtrTy(0),
                 b.getInt32Ty(), b.getInt32Ty()},
                false);
            _llvm_ray_query_pipeline_dispatch = llvm::Function::Create(
                llvm_dispatch_type, llvm::Function::ExternalLinkage,
                "luisa_ray_query_pipeline_dispatch", _llvm_module.get());
            _llvm_ray_query_pipeline_dispatch->addFnAttr(
                llvm::Attribute::NoUnwind);
            auto llvm_dispatch_entry = llvm::BasicBlock::Create(
                _llvm_context, "entry", _llvm_ray_query_pipeline_dispatch);
            auto llvm_dispatch_invalid = llvm::BasicBlock::Create(
                _llvm_context, "invalid", _llvm_ray_query_pipeline_dispatch);
            IB dispatch_b{llvm_dispatch_entry};
            _llvm_ray_query_pipeline_switch = dispatch_b.CreateSwitch(
                _llvm_ray_query_pipeline_dispatch->getArg(2),
                llvm_dispatch_invalid, 0u);
            dispatch_b.SetInsertPoint(llvm_dispatch_invalid);
            dispatch_b.CreateUnreachable();
        }

        auto pipeline_index = static_cast<uint32_t>(
            _ray_query_pipeline_count++);
        auto llvm_pipeline_block = llvm::BasicBlock::Create(
            _llvm_context,
            llvm::Twine{"pipeline."} + llvm::Twine{pipeline_index},
            _llvm_ray_query_pipeline_dispatch);
        _llvm_ray_query_pipeline_switch->addCase(
            b.getInt32(pipeline_index), llvm_pipeline_block);
        IB dispatch_b{llvm_pipeline_block};
        auto llvm_dispatch_context =
            _llvm_ray_query_pipeline_dispatch->getArg(1);
        llvm::SmallVector<llvm::Value *, 16> llvm_decoded_args;
        llvm_decoded_args.reserve(llvm_callback_arg_types.size());
        luisa::vector<llvm::LoadInst *> llvm_context_loads;
        llvm_context_loads.reserve(llvm_callback_arg_types.size());
        for (auto i = 0u; i < llvm_callback_arg_types.size(); ++i) {
            auto llvm_field = dispatch_b.CreateStructGEP(
                llvm_context_type, llvm_dispatch_context, i,
                "ray.query.context.field");
            auto llvm_value = dispatch_b.CreateLoad(
                llvm_callback_arg_types[i], llvm_field,
                "ray.query.context.value");
            llvm_decoded_args.emplace_back(llvm_value);
            llvm_context_loads.emplace_back(llvm_value);
        }
        auto llvm_surface_block = llvm::BasicBlock::Create(
            _llvm_context, "surface", _llvm_ray_query_pipeline_dispatch);
        auto llvm_procedural_block = llvm::BasicBlock::Create(
            _llvm_context, "procedural", _llvm_ray_query_pipeline_dispatch);
        auto llvm_invalid_kind_block = llvm::BasicBlock::Create(
            _llvm_context, "invalid.kind", _llvm_ray_query_pipeline_dispatch);
        auto llvm_kind_switch = dispatch_b.CreateSwitch(
            _llvm_ray_query_pipeline_dispatch->getArg(3),
            llvm_invalid_kind_block, 2u);
        llvm_kind_switch->addCase(
            dispatch_b.getInt32(llvm_ray_query_state_surface_candidate),
            llvm_surface_block);
        llvm_kind_switch->addCase(
            dispatch_b.getInt32(llvm_ray_query_state_procedural_candidate),
            llvm_procedural_block);

        dispatch_b.SetInsertPoint(llvm_surface_block);
        auto llvm_surface_call = dispatch_b.CreateCall(
            llvm_on_surface, llvm_decoded_args);
        llvm_surface_call->setCallingConv(
            llvm_on_surface->getCallingConv());
        dispatch_b.CreateRetVoid();

        dispatch_b.SetInsertPoint(llvm_procedural_block);
        auto llvm_procedural_call = dispatch_b.CreateCall(
            llvm_on_procedural, llvm_decoded_args);
        llvm_procedural_call->setCallingConv(
            llvm_on_procedural->getCallingConv());
        dispatch_b.CreateRetVoid();

        dispatch_b.SetInsertPoint(llvm_invalid_kind_block);
        dispatch_b.CreateUnreachable();

        auto llvm_state_pointer = _get_ray_query_state_pointer(
            b, func_ctx, query_object);
        if (llvm_state_pointer->getType()->getPointerAddressSpace() != 0u) {
            llvm_state_pointer = b.CreateAddrSpaceCast(
                llvm_state_pointer, b.getPtrTy(0),
                "ray.query.state.generic");
        }
        LUISA_ASSERT(
            func_ctx.llvm_rt_stack_size != nullptr &&
                func_ctx.llvm_rt_stack_count != nullptr &&
                func_ctx.llvm_rt_stack_data != nullptr,
            "Synchronous HIP ray query requires a dynamic stack buffer.");
        auto llvm_trace_type = llvm::FunctionType::get(
            b.getVoidTy(),
            {b.getPtrTy(0), b.getPtrTy(0), b.getInt32Ty(),
             b.getInt32Ty(), b.getInt32Ty(), b.getPtrTy(0)},
            false);
        auto llvm_trace_name =
            query_object->type() == Type::of<RayQueryAny>() ?
                (_rt_analysis.writes_instance_opacity ?
                     "luisa_pipeline_ray_query_trace_any" :
                     "luisa_pipeline_ray_query_trace_any_stable_opacity") :
                (_rt_analysis.writes_instance_opacity ?
                     "luisa_pipeline_ray_query_trace_all" :
                     "luisa_pipeline_ray_query_trace_all_stable_opacity");
        auto llvm_trace = _llvm_module->getFunction(
            llvm_trace_name);
        if (llvm_trace == nullptr) {
            llvm_trace = llvm::Function::Create(
                llvm_trace_type, llvm::Function::ExternalLinkage,
                llvm_trace_name, _llvm_module.get());
        } else {
            LUISA_ASSERT(llvm_trace->getFunctionType() == llvm_trace_type,
                         "HIP synchronous ray-query trace ABI mismatch.");
        }
        auto llvm_trace_call = b.CreateCall(
            llvm_trace,
            {llvm_state_pointer, llvm_generic_context,
             b.getInt32(pipeline_index), func_ctx.llvm_rt_stack_size,
             func_ctx.llvm_rt_stack_count, func_ctx.llvm_rt_stack_data});
        _llvm_ray_query_pipeline_contexts.emplace_back(
            RayQueryPipelineContext{
                llvm::cast<llvm::AllocaInst>(llvm_context_pointer),
                llvm_generic_context,
                llvm_trace_call,
                llvm_on_surface,
                llvm_on_procedural,
                std::move(llvm_context_stores),
                std::move(llvm_context_loads)});
        return;
    }

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
    auto llvm_state_address = pipeline_b.CreateAlignedLoad(
        _get_llvm_ray_query_type(), llvm_pipeline_args.front(),
        llvm::Align{_get_type_alignment(query_object->type())},
        "ray.query.object");
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
    return b.CreateIntToPtr(
        llvm_query,
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
    if (!_uses_hardware_rt_stack ||
        _uses_synchronous_ray_query_pipeline) {
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
    if (_uses_synchronous_ray_query_pipeline) {
        static constexpr std::string_view prefix{"luisa_ray_query_"};
        LUISA_ASSERT(name.starts_with(prefix),
                     "Invalid HIP ray-query wrapper name '{}'.", name.str());
        motion_name = "luisa_pipeline_ray_query_";
        motion_name.append(name.drop_front(prefix.size()).str());
        wrapper_name = motion_name;
    } else if (_supports_hardware_rt_stack &&
               !_uses_hardware_rt_stack) {
        // On gfx12 the generic DynamicStack implementation is emitted under
        // the historical motion-query symbol family. It is also the required
        // reentrant path when a static query handler performs a nested trace;
        // selecting by the actual stack plan keeps those two reasons unified.
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
