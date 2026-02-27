//
// Created by mike on 11/1/25.
//

#include <llvm/IR/Dominators.h>
#include <llvm/Analysis/LoopInfo.h>
#include <llvm/Transforms/Utils/Cloning.h>
#include <llvm/Transforms/Utils/BasicBlockUtils.h>
#include <llvm/Transforms/Utils/ValueMapper.h>
#include <llvm/IR/Verifier.h>
#include <llvm/IR/InlineAsm.h>

#include "cuda_codegen_llvm_impl.h"

namespace luisa::compute::cuda {

llvm::Value *CUDACodegenLLVMImpl::_call_ray_query_intrinsic(IB &b, llvm::StringRef name, llvm::Type *ret, llvm::ArrayRef<llvm::Value *> args, bool sideeffect) noexcept {
    llvm::SmallVector<llvm::Type *> arg_types;
    for (auto arg : args) {
        arg_types.push_back(arg->getType());
    }
    llvm::FunctionType *ft = llvm::FunctionType::get(ret, arg_types, false);
    auto f = _llvm_module->getOrInsertFunction(name, ft);
    auto call = b.CreateCall(f, args);
    if (sideeffect) {
        // Mark the function as having side effects so LLVM won't remove it
        if (auto *fn = llvm::dyn_cast<llvm::Function>(f.getCallee())) {
            fn->removeFnAttr(llvm::Attribute::ReadNone);
            fn->removeFnAttr(llvm::Attribute::ReadOnly);
        }
    }
    return call;
}

llvm::Value *CUDACodegenLLVMImpl::_translate_ray_query_object_read_inst(IB &b, FunctionContext &func_ctx, const xir::RayQueryObjectReadInst *inst) noexcept {
    LUISA_DEBUG_ASSERT(inst->operand_count() == 1u, "Invalid ray query object read instruction.");
    switch (inst->op()) {
        case xir::RayQueryObjectReadOp::RAY_QUERY_OBJECT_WORLD_SPACE_RAY:
            return _call_ray_query_intrinsic(b, llvm_ray_query_intrinsic_name_world_space_ray, _get_llvm_ray_type(), {});
        case xir::RayQueryObjectReadOp::RAY_QUERY_OBJECT_PROCEDURAL_CANDIDATE_HIT:
            return _call_ray_query_intrinsic(b, llvm_ray_query_intrinsic_name_procedural_candidate_hit, _get_llvm_procedural_hit_type(), {});
        case xir::RayQueryObjectReadOp::RAY_QUERY_OBJECT_TRIANGLE_CANDIDATE_HIT:
            return _call_ray_query_intrinsic(b, llvm_ray_query_intrinsic_name_surface_candidate_hit, _get_llvm_surface_hit_type(), {});
        case xir::RayQueryObjectReadOp::RAY_QUERY_OBJECT_COMMITTED_HIT:
            return _call_ray_query_intrinsic(b, llvm_ray_query_intrinsic_name_committed_hit, _get_llvm_committed_hit_type(), {});
        default:
            LUISA_ERROR_WITH_LOCATION("Invalid ray query object read operation.");
    }
}

void CUDACodegenLLVMImpl::_translate_ray_query_object_write_inst(IB &b, FunctionContext &func_ctx, const xir::RayQueryObjectWriteInst *inst) noexcept {
    switch (inst->op()) {
        case xir::RayQueryObjectWriteOp::RAY_QUERY_OBJECT_COMMIT_TRIANGLE:
            _call_ray_query_intrinsic(b, llvm_ray_query_intrinsic_name_commit_surface_hit, b.getVoidTy(), {});
            break;
        case xir::RayQueryObjectWriteOp::RAY_QUERY_OBJECT_COMMIT_PROCEDURAL:
            _call_ray_query_intrinsic(b, llvm_ray_query_intrinsic_name_commit_procedural_hit, b.getVoidTy(),
                                      {_get_llvm_value(b, func_ctx, inst->operand(1))});
            break;
        case xir::RayQueryObjectWriteOp::RAY_QUERY_OBJECT_TERMINATE:
            _call_ray_query_intrinsic(b, llvm_ray_query_intrinsic_name_terminate, b.getVoidTy(), {});
            break;
        case xir::RayQueryObjectWriteOp::RAY_QUERY_OBJECT_PROCEED:
            _call_ray_query_intrinsic(b, llvm_ray_query_intrinsic_name_proceed, b.getVoidTy(), {});
            break;
        default:
            LUISA_ERROR_WITH_LOCATION("Invalid ray query object write operation.");
    }
}

void CUDACodegenLLVMImpl::_translate_ray_query_pipeline_inst(IB &b, FunctionContext &func_ctx, const xir::RayQueryPipelineInst *inst) noexcept {
    // Pipeline is handled during materialization
    LUISA_NOT_IMPLEMENTED();
}

void CUDACodegenLLVMImpl::_translate_ray_query_loop_inst(IB &b, FunctionContext &func_ctx, const xir::RayQueryLoopInst *inst) noexcept {
    b.GetInsertBlock()->setName("ray.query.loop");
    auto llvm_dispatch_block = func_ctx.get_local_value<llvm::BasicBlock>(inst->dispatch_block());
    llvm_dispatch_block->setName("ray.query.dispatch");
    b.CreateBr(llvm_dispatch_block);
}

void CUDACodegenLLVMImpl::_translate_ray_query_dispatch_inst(IB &b, FunctionContext &func_ctx, const xir::RayQueryDispatchInst *inst) noexcept {
    _call_ray_query_intrinsic(b, llvm_ray_query_intrinsic_name_proceed, b.getVoidTy(), {});
    auto llvm_state = _call_ray_query_intrinsic(b, llvm_ray_query_intrinsic_name_state, b.getInt8Ty(), {});
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

void CUDACodegenLLVMImpl::_lower_ray_query_intrinsics(llvm::Function *f) noexcept {
    IB b{_llvm_context};
    llvm::SmallVector<llvm::Instruction *, 16> to_remove;
    for (auto &bb : *f) {
        for (auto &inst : bb) {
            if (auto call = llvm::dyn_cast<llvm::CallInst>(&inst)) {
                auto callee = call->getCalledFunction();
                if (callee == nullptr) continue;
                auto name = callee->getName();
                b.SetInsertPoint(call);
                if (name == llvm::StringRef{llvm_ray_query_intrinsic_name_world_space_ray.data(), llvm_ray_query_intrinsic_name_world_space_ray.size()}) {
                    call->replaceAllUsesWith(_call_optix_get_world_space_ray(b));
                    to_remove.push_back(call);
                } else if (name == llvm::StringRef{llvm_ray_query_intrinsic_name_procedural_candidate_hit.data(), llvm_ray_query_intrinsic_name_procedural_candidate_hit.size()}) {
                    auto inst_id = _call_optix_read_instance_index(b);
                    auto prim_id = _call_optix_read_primitive_index(b);
                    auto hit = static_cast<llvm::Value *>(llvm::PoisonValue::get(_get_llvm_procedural_hit_type()));
                    hit = b.CreateInsertValue(hit, inst_id, llvm_procedural_hit_type_inst_id_index);
                    hit = b.CreateInsertValue(hit, prim_id, llvm_procedural_hit_type_prim_id_index);
                    call->replaceAllUsesWith(hit);
                    to_remove.push_back(call);
                } else if (name == llvm::StringRef{llvm_ray_query_intrinsic_name_surface_candidate_hit.data(), llvm_ray_query_intrinsic_name_surface_candidate_hit.size()}) {
                    auto inst_id = _call_optix_read_instance_index(b);
                    auto prim_id = _call_optix_read_primitive_index(b);
                    auto bary = _call_optix_get_triangle_barycentrics(b);
                    auto t = _call_optix_get_hit_distance(b);
                    auto hit = static_cast<llvm::Value *>(llvm::PoisonValue::get(_get_llvm_surface_hit_type()));
                    hit = b.CreateInsertValue(hit, inst_id, llvm_surface_hit_type_inst_id_index);
                    hit = b.CreateInsertValue(hit, prim_id, llvm_surface_hit_type_prim_id_index);
                    hit = b.CreateInsertValue(hit, bary, llvm_surface_hit_type_bary_index);
                    hit = b.CreateInsertValue(hit, t, llvm_surface_hit_type_t_index);
                    call->replaceAllUsesWith(hit);
                    to_remove.push_back(call);
                } else if (name == llvm::StringRef{llvm_ray_query_intrinsic_name_committed_hit.data(), llvm_ray_query_intrinsic_name_committed_hit.size()}) {
                    auto is_hit = _call_optix_hit_object_is_hit(b);
                    auto inst_id = b.CreateSelect(is_hit, _call_optix_hit_object_instance_index(b), b.getInt32(~0u));
                    auto prim_id = _call_optix_hit_object_primitive_index(b);
                    auto bary = _call_optix_hit_object_triangle_barycentrics(b);
                    auto hit_kind = _call_optix_hit_object_hit_kind(b);
                    auto kind = b.CreateSelect(is_hit,
                                               b.CreateSelect(b.CreateICmpUGT(hit_kind, b.getInt32(127u)),
                                                              b.getInt32(1u /* BUILTIN */),
                                                              b.getInt32(2u /* PROCEDURAL */)),
                                               b.getInt32(0u /* MISS */));
                    auto t = _call_optix_hit_object_ray_t_max(b);
                    auto hit = static_cast<llvm::Value *>(llvm::PoisonValue::get(_get_llvm_committed_hit_type()));
                    hit = b.CreateInsertValue(hit, inst_id, llvm_committed_hit_type_inst_id_index);
                    hit = b.CreateInsertValue(hit, prim_id, llvm_committed_hit_type_prim_id_index);
                    hit = b.CreateInsertValue(hit, bary, llvm_committed_hit_type_bary_index);
                    hit = b.CreateInsertValue(hit, kind, llvm_committed_hit_type_hit_kind_index);
                    hit = b.CreateInsertValue(hit, t, llvm_committed_hit_type_t_index);
                    call->replaceAllUsesWith(hit);
                    to_remove.push_back(call);
                } else if (name == llvm::StringRef{llvm_ray_query_intrinsic_name_is_terminated.data(), llvm_ray_query_intrinsic_name_is_terminated.size()}) {
                    call->replaceAllUsesWith(b.getInt1(true));
                    to_remove.push_back(call);
                } else if (name == llvm::StringRef{llvm_ray_query_intrinsic_name_is_surface_candidate.data(), llvm_ray_query_intrinsic_name_is_surface_candidate.size()}) {
                    auto llvm_state = _call_ray_query_intrinsic(b, llvm_ray_query_intrinsic_name_state, b.getInt8Ty(), {});
                    call->replaceAllUsesWith(b.CreateICmpEQ(llvm_state, b.getInt8(llvm_ray_query_state_surface_candidate)));
                    to_remove.push_back(call);
                } else if (name == llvm::StringRef{llvm_ray_query_intrinsic_name_is_procedural_candidate.data(), llvm_ray_query_intrinsic_name_is_procedural_candidate.size()}) {
                    auto llvm_state = _call_ray_query_intrinsic(b, llvm_ray_query_intrinsic_name_state, b.getInt8Ty(), {});
                    call->replaceAllUsesWith(b.CreateICmpEQ(llvm_state, b.getInt8(llvm_ray_query_state_procedural_candidate)));
                    to_remove.push_back(call);
                }
            }
        }
    }
    for (auto inst : to_remove) {
        inst->eraseFromParent();
    }
}

void CUDACodegenLLVMImpl::_materialize_ray_query_loops() noexcept {
    static auto dump_llvm_ir = [] {
        using namespace std::string_view_literals;
        auto env = getenv("LUISA_DUMP_LLVM_IR");
        return env != nullptr && env == "1"sv;
    }();
    llvm::SmallVector<llvm::Function *, 4> extracted_funcs;
    for (auto &F : *_llvm_module) {
        if (F.getName().starts_with("ray.query.loop.extracted")) {
            extracted_funcs.push_back(&F);
        }
    }
    if (extracted_funcs.empty()) {
        for (auto &F : *_llvm_module) {
            _lower_ray_query_intrinsics(&F);
        }
        return;
    }

    for (size_t i = 0; i < extracted_funcs.size(); ++i) {
        auto F = extracted_funcs[i];

        llvm::CallInst *loop_call = nullptr;
        for (auto U : F->users()) {
            if (auto CI = llvm::dyn_cast<llvm::CallInst>(U)) {
                loop_call = CI;
                break;
            }
        }
        LUISA_ASSERT(loop_call != nullptr, "Ray query loop extracted function has no call site.");

        // Find spawn call in the same function as loop_call
        llvm::CallInst *spawn_call = nullptr;
        auto *caller_func = loop_call->getFunction();
        LUISA_ASSERT(caller_func != nullptr, "Loop call is not inside a function.");

        // Search all blocks in the caller function for the spawn call
        for (auto &BB : *caller_func) {
            for (auto &I : BB) {
                if (auto CI = llvm::dyn_cast<llvm::CallInst>(&I)) {
                    if (CI->getCalledFunction() &&
                        CI->getCalledFunction()->getName() == llvm::StringRef{llvm_ray_query_intrinsic_name_spawn.data(), llvm_ray_query_intrinsic_name_spawn.size()}) {
                        spawn_call = CI;
                        LUISA_VERBOSE_WITH_LOCATION("Found spawn call in block: {}", BB.getName().str());
                        break;
                    }
                }
            }
            if (spawn_call != nullptr) break;
        }
        LUISA_ASSERT(spawn_call != nullptr, "Spawn call not found in function containing ray query loop call site.");

        // Debug: print spawn call location relative to loop call
        LUISA_VERBOSE_WITH_LOCATION("Caller function: {}", caller_func->getName().str());
        LUISA_VERBOSE_WITH_LOCATION("Extracted function: {}", F->getName().str());
        LUISA_VERBOSE_WITH_LOCATION("Loop call in block: {}", loop_call->getParent()->getName().str());
        LUISA_VERBOSE_WITH_LOCATION("Spawn call in block: {}", spawn_call->getParent()->getName().str());
        if (spawn_call->getFunction() == F) {
            LUISA_VERBOSE_WITH_LOCATION("Spawn call is INSIDE extracted function F");
        } else if (spawn_call->getFunction() == caller_func) {
            LUISA_VERBOSE_WITH_LOCATION("Spawn call is in caller function");
        } else {
            LUISA_VERBOSE_WITH_LOCATION("Spawn call is in UNKNOWN function: {}", spawn_call->getFunction()->getName().str());
        }

        // Create alloca in the entry block, not at loop_call position
        auto *entry_block = &caller_func->getEntryBlock();
        llvm::Instruction *insert_point = &*entry_block->getFirstNonPHIIt();
        IB entry_b{insert_point};

        // Collect all values that need to be stored in context:
        // 1. Loop call arguments (captured values)
        // 2. Spawn call arguments that don't dominate (to avoid dominance violations)
        auto arg_count = loop_call->arg_size();
        llvm::DominatorTree DT(*caller_func);

        // First pass: identify which spawn args don't dominate
        llvm::SmallVector<bool, 5> spawn_arg_needs_context;
        llvm::SmallVector<llvm::Value *, 5> spawn_args;
        for (unsigned j = 0; j < spawn_call->arg_size(); ++j) {
            auto arg = spawn_call->getArgOperand(j);
            spawn_args.push_back(arg);
            bool dominates = DT.dominates(arg, loop_call);
            spawn_arg_needs_context.push_back(!dominates);
            if (!dominates) {
                LUISA_VERBOSE_WITH_LOCATION("Spawn arg {} ({}) does not dominate loop call", j,
                                            arg->getName().empty() ? "unnamed" : arg->getName().str());
                if (auto inst = llvm::dyn_cast<llvm::Instruction>(arg)) {
                    auto parent_block = inst->getParent();
                    LUISA_VERBOSE_WITH_LOCATION("  Defined in block: {}", parent_block->getName().str());
                    LUISA_VERBOSE_WITH_LOCATION("  Instruction type: {}", inst->getOpcodeName());
                    // Check relative position to spawn call
                    if (parent_block == spawn_call->getParent()) {
                        bool before_spawn = inst->comesBefore(spawn_call);
                        LUISA_VERBOSE_WITH_LOCATION("  Comes before spawn call: {}", before_spawn);
                    }
                    // Check if it's a load from alloca (loop-carried dependency)
                    if (auto load = llvm::dyn_cast<llvm::LoadInst>(inst)) {
                        LUISA_VERBOSE_WITH_LOCATION("  Is load from: {}",
                                                    load->getPointerOperand()->getName().empty() ? "unnamed" : load->getPointerOperand()->getName().str());
                    }
                } else if (auto arg_val = llvm::dyn_cast<llvm::Argument>(arg)) {
                    LUISA_VERBOSE_WITH_LOCATION("  Is function argument {}", arg_val->getArgNo());
                } else {
                    LUISA_VERBOSE_WITH_LOCATION("  Value type: other");
                }
                // Check if this spawn arg is the same as any loop_call arg
                for (unsigned k = 0; k < loop_call->arg_size(); ++k) {
                    if (arg == loop_call->getArgOperand(k)) {
                        LUISA_VERBOSE_WITH_LOCATION("  Matches loop_call arg {}", k);
                        break;
                    }
                }
            }
        }

        // Build context type: [loop_call_args..., spawn_args_that_need_context...]
        llvm::SmallVector<llvm::Type *, 8> ctx_field_types;
        for (unsigned j = 0; j < arg_count; ++j) {
            ctx_field_types.push_back(loop_call->getArgOperand(j)->getType());
        }
        unsigned spawn_context_start = arg_count;
        for (unsigned j = 0; j < spawn_call->arg_size(); ++j) {
            if (spawn_arg_needs_context[j]) {
                ctx_field_types.push_back(spawn_args[j]->getType());
            }
        }

        auto ctx_type = llvm::StructType::get(_llvm_context, ctx_field_types);
        auto ctx_alloca = entry_b.CreateAlloca(ctx_type, nullptr, "rq_ctx");

        // For spawn args that don't dominate loop_call, we need to store them at the
        // spawn_call position (where they're available) instead of at loop_call position
        IB spawn_b{spawn_call};
        unsigned ctx_idx = spawn_context_start;
        for (unsigned j = 0; j < spawn_call->arg_size(); ++j) {
            if (spawn_arg_needs_context[j]) {
                auto field_ptr = spawn_b.CreateStructGEP(ctx_type, ctx_alloca, ctx_idx++);
                spawn_b.CreateStore(spawn_args[j], field_ptr);
            }
        }

        // Store loop_call arguments at loop_call position
        IB b{loop_call};
        for (unsigned j = 0; j < arg_count; ++j) {
            auto field_ptr = b.CreateStructGEP(ctx_type, ctx_alloca, j);
            b.CreateStore(loop_call->getArgOperand(j), field_ptr);
        }

        // Create payload values at spawn_call position (where context is fully populated)
        auto ctx_ptr_int = spawn_b.CreatePtrToInt(ctx_alloca, spawn_b.getInt64Ty());
        auto p_ctx_hi = spawn_b.CreateTrunc(spawn_b.CreateLShr(ctx_ptr_int, 32), spawn_b.getInt32Ty());
        auto p_ctx_lo = spawn_b.CreateTrunc(ctx_ptr_int, spawn_b.getInt32Ty());

        // Align with AST path packing:
        // r0 = (impl_tag << 24u) | (static_cast<lc_uint>(p_ctx >> 32u) & 0xffffffu);
        // r1 = static_cast<lc_uint>(p_ctx);
        auto r0 = spawn_b.CreateOr(spawn_b.CreateShl(spawn_b.getInt32(static_cast<uint32_t>(i)), 24),
                                   spawn_b.CreateAnd(p_ctx_hi, spawn_b.getInt32(0xffffffu)));
        auto r1 = p_ctx_lo;

        // Validate spawn call has expected number of arguments
        LUISA_ASSERT(spawn_call->arg_size() == 5, "Spawn call must have exactly 5 arguments, got {}", spawn_call->arg_size());

        auto accel = spawn_call->getArgOperand(0);
        auto ray = spawn_call->getArgOperand(1);
        auto time = spawn_call->getArgOperand(2);
        auto mask = spawn_call->getArgOperand(3);
        auto flags_val = spawn_call->getArgOperand(4);
        auto flags_int = llvm::dyn_cast<llvm::ConstantInt>(flags_val);
        LUISA_ASSERT(flags_int != nullptr, "Ray query flags must be constant.");

        _call_optix_trace(spawn_b, 2 /* LC_PAYLOAD_TYPE_RAY_QUERY */, 5 /* sbt_offset */,
                          static_cast<uint32_t>(flags_int->getZExtValue()), accel,
                          ray, time, mask, {r0, r1});

        // Fix reloads in the caller by identifying Argument->Argument sync patterns in F
        llvm::DenseMap<llvm::Value *, llvm::Value *> reload_map;
        for (auto &BB : *F) {
            for (auto &I : BB) {
                if (auto SI = llvm::dyn_cast<llvm::StoreInst>(&I)) {
                    auto val = SI->getValueOperand();
                    if (auto LI = llvm::dyn_cast<llvm::LoadInst>(val)) {
                        auto src = LI->getPointerOperand();
                        auto dst = SI->getPointerOperand();
                        if (auto src_arg = llvm::dyn_cast<llvm::Argument>(src)) {
                            if (auto dst_arg = llvm::dyn_cast<llvm::Argument>(dst)) {
                                reload_map[loop_call->getArgOperand(dst_arg->getArgNo())] =
                                    loop_call->getArgOperand(src_arg->getArgNo());
                            }
                        }
                    }
                }
            }
        }

        llvm::SmallVector<llvm::LoadInst *, 8> caller_reloads;
        for (auto &I : *loop_call->getParent()) {
            if (I.comesBefore(loop_call)) continue;
            if (auto LI = llvm::dyn_cast<llvm::LoadInst>(&I)) {
                if (reload_map.count(LI->getPointerOperand())) {
                    caller_reloads.push_back(LI);
                }
            }
        }
        for (auto LI : caller_reloads) {
            b.SetInsertPoint(LI);
            auto new_load = b.CreateLoad(LI->getType(), reload_map[LI->getPointerOperand()]);
            LI->replaceAllUsesWith(new_load);
            LI->eraseFromParent();
        }

        // Define return type: struct { float t, i8 committed, i8 terminated }
        auto result_type = llvm::StructType::get(_llvm_context, {b.getFloatTy(),
                                                                 b.getInt8Ty(),
                                                                 b.getInt8Ty()});

        auto create_intersection_func = [&](luisa::string name) {
            auto ft = llvm::FunctionType::get(result_type, {b.getPtrTy()}, false);
            auto new_f = llvm::Function::Create(ft, llvm::Function::ExternalLinkage, llvm::StringRef{name.data(), name.size()}, _llvm_module.get());
            new_f->addFnAttr(llvm::Attribute::AlwaysInline);
            return new_f;
        };

        auto triangle_f_name = luisa::format("lc_ray_query_triangle_intersection_{}", i);
        auto procedural_f_name = luisa::format("lc_ray_query_procedural_intersection_{}", i);
        auto triangle_f = create_intersection_func(triangle_f_name);
        auto procedural_f = create_intersection_func(procedural_f_name);

        auto transform_intersection_func = [&](llvm::Function *new_f, bool is_triangle) {
            llvm::ValueToValueMapTy vmap;
            IB nb{_llvm_context};
            auto entry_bb = llvm::BasicBlock::Create(_llvm_context, "entry", new_f);
            nb.SetInsertPoint(entry_bb);

            for (unsigned j = 0; j < arg_count; ++j) {
                auto field_ptr = nb.CreateStructGEP(ctx_type, new_f->getArg(0), j);
                vmap[F->getArg(j)] = nb.CreateLoad(ctx_field_types[j], field_ptr);
            }

            llvm::SmallVector<llvm::ReturnInst *, 4> returns;
            llvm::CloneFunctionInto(new_f, F, vmap, llvm::CloneFunctionChangeType::LocalChangesOnly, returns);

            // Fix entry block branch to the cloned entry block of F
            nb.SetInsertPoint(entry_bb);
            nb.CreateBr(llvm::cast<llvm::BasicBlock>(vmap[&F->getEntryBlock()]));

            auto &alloca_bb = *llvm::cast<llvm::BasicBlock>(vmap[&F->getEntryBlock()]);
            nb.SetInsertPoint(&*alloca_bb.getFirstNonPHIIt());
            auto t_hit_alloca = nb.CreateAlloca(nb.getFloatTy(), nullptr, "t_hit");
            auto committed_alloca = nb.CreateAlloca(nb.getInt8Ty(), nullptr, "committed");
            auto terminated_alloca = nb.CreateAlloca(nb.getInt8Ty(), nullptr, "terminated");
            nb.CreateStore(llvm::ConstantFP::get(nb.getFloatTy(), 0.0), t_hit_alloca);
            nb.CreateStore(nb.getInt8(0), committed_alloca);
            nb.CreateStore(nb.getInt8(0), terminated_alloca);

            llvm::SmallVector<llvm::Instruction *, 16> to_remove;
            for (auto &bb : *new_f) {
                if (&bb == entry_bb) continue;
                for (auto &inst : bb) {
                    if (auto call = llvm::dyn_cast<llvm::CallInst>(&inst)) {
                        auto callee = call->getCalledFunction();
                        if (callee == nullptr) {
                            if (auto v = call->getCalledOperand()) {
                                if (auto old_func = llvm::dyn_cast<llvm::Function>(v)) {
                                    callee = old_func;
                                }
                            }
                        }
                        if (callee == nullptr) continue;
                        auto name = callee->getName();
                        if (name == llvm::StringRef{llvm_ray_query_intrinsic_name_state.data(), llvm_ray_query_intrinsic_name_state.size()}) {
                            call->replaceAllUsesWith(nb.getInt8(is_triangle ? llvm_ray_query_state_surface_candidate : llvm_ray_query_state_procedural_candidate));
                            to_remove.push_back(call);
                        } else if (name == llvm::StringRef{llvm_ray_query_intrinsic_name_commit_surface_hit.data(), llvm_ray_query_intrinsic_name_commit_surface_hit.size()}) {
                            nb.SetInsertPoint(call);
                            nb.CreateStore(nb.getInt8(1), committed_alloca);
                            to_remove.push_back(call);
                        } else if (name == llvm::StringRef{llvm_ray_query_intrinsic_name_commit_procedural_hit.data(), llvm_ray_query_intrinsic_name_commit_procedural_hit.size()}) {
                            nb.SetInsertPoint(call);
                            nb.CreateStore(nb.getInt8(1), committed_alloca);
                            nb.CreateStore(call->getArgOperand(0), t_hit_alloca);
                            to_remove.push_back(call);
                        } else if (name == llvm::StringRef{llvm_ray_query_intrinsic_name_terminate.data(), llvm_ray_query_intrinsic_name_terminate.size()}) {
                            nb.SetInsertPoint(call);
                            nb.CreateStore(nb.getInt8(1), terminated_alloca);
                            to_remove.push_back(call);
                        } else if (name == llvm::StringRef{llvm_ray_query_intrinsic_name_dispatch.data(), llvm_ray_query_intrinsic_name_dispatch.size()}) {
                            to_remove.push_back(call);
                        }
                    }
                }
            }

            for (auto inst : to_remove) {
                inst->eraseFromParent();
            }

            auto ret_bb = llvm::BasicBlock::Create(_llvm_context, "return", new_f);
            nb.SetInsertPoint(ret_bb);
            auto t_hit = nb.CreateLoad(nb.getFloatTy(), t_hit_alloca);
            auto committed = nb.CreateLoad(nb.getInt8Ty(), committed_alloca);
            auto terminated = nb.CreateLoad(nb.getInt8Ty(), terminated_alloca);
            auto ret_val = static_cast<llvm::Value *>(llvm::PoisonValue::get(result_type));
            ret_val = nb.CreateInsertValue(ret_val, t_hit, 0);
            ret_val = nb.CreateInsertValue(ret_val, committed, 1);
            ret_val = nb.CreateInsertValue(ret_val, terminated, 2);
            nb.CreateRet(ret_val);

            llvm::DominatorTree DT(*new_f);
            llvm::SmallVector<std::pair<llvm::Instruction *, unsigned>, 8> backedges;
            for (auto &bb : *new_f) {
                if (&bb == ret_bb || &bb == entry_bb) continue;
                auto term = bb.getTerminator();
                for (unsigned j = 0; j < term->getNumSuccessors(); ++j) {
                    auto succ = term->getSuccessor(j);
                    if (DT.dominates(succ, &bb)) {
                        backedges.emplace_back(term, j);
                    }
                }
            }
            for (auto [term, j] : backedges) {
                nb.SetInsertPoint(term);
                nb.CreateStore(nb.getInt8(0), terminated_alloca);
                auto succ = term->getSuccessor(j);
                term->setSuccessor(j, ret_bb);
                succ->removePredecessor(term->getParent());
            }

            llvm::SmallVector<llvm::ReturnInst *, 4> old_returns;
            for (auto &bb : *new_f) {
                if (&bb == ret_bb) continue;
                if (auto ret = llvm::dyn_cast<llvm::ReturnInst>(bb.getTerminator())) {
                    old_returns.push_back(ret);
                }
            }
            for (auto ret : old_returns) {
                nb.SetInsertPoint(ret);
                nb.CreateStore(nb.getInt8(1), terminated_alloca);
                llvm::BranchInst::Create(ret_bb, ret->getParent());
                ret->eraseFromParent();
            }
        };

        transform_intersection_func(triangle_f, true);
        transform_intersection_func(procedural_f, false);

        loop_call->replaceAllUsesWith(llvm::UndefValue::get(loop_call->getType()));
        loop_call->eraseFromParent();
        spawn_call->eraseFromParent();
    }

    auto generate_anyhit_entry = [&]() {
        auto ft = llvm::FunctionType::get(llvm::Type::getVoidTy(_llvm_context), {}, false);
        auto entry_f = llvm::Function::Create(ft, llvm::Function::ExternalLinkage, "__anyhit__ray_query", _llvm_module.get());
        auto entry_bb = llvm::BasicBlock::Create(_llvm_context, "entry", entry_f);
        IB eb{entry_bb};

        _call_optix_set_payload_types(eb, 1u << 1u /* LC_PAYLOAD_TYPE_RAY_QUERY */);
        auto hit_kind = _call_optix_get_hit_kind(eb);
        auto is_triangle = eb.CreateICmpUGT(hit_kind, eb.getInt32(127u));

        auto triangle_bb = llvm::BasicBlock::Create(_llvm_context, "triangle", entry_f);
        auto procedural_bb = llvm::BasicBlock::Create(_llvm_context, "procedural", entry_f);
        auto terminate_check_bb = llvm::BasicBlock::Create(_llvm_context, "terminate_check", entry_f);
        eb.CreateCondBr(is_triangle, triangle_bb, procedural_bb);

        eb.SetInsertPoint(procedural_bb);
        auto proc_term = eb.CreateICmpEQ(hit_kind, eb.getInt32(0x02u /* LC_HIT_KIND_PROCEDURAL_TERMINATED */));
        eb.CreateBr(terminate_check_bb);

        eb.SetInsertPoint(triangle_bb);
        auto r0 = _call_optix_get_payload(eb, 0u);
        auto r1 = _call_optix_get_payload(eb, 1u);

        auto query_id = eb.CreateLShr(r0, 24);
        auto p_ctx_hi = eb.CreateAnd(r0, eb.getInt32(0xffffffu));
        auto p_ctx_lo = r1;
        auto p_ctx = eb.CreateIntToPtr(
            eb.CreateOr(eb.CreateShl(eb.CreateZExt(p_ctx_hi, eb.getInt64Ty()), 32),
                        eb.CreateZExt(p_ctx_lo, eb.getInt64Ty())),
            eb.getPtrTy());

        auto should_terminate_alloca = eb.CreateAlloca(eb.getInt1Ty(), nullptr, "should_terminate");
        eb.CreateStore(eb.getInt1(false), should_terminate_alloca);

        auto switch_exit_bb = llvm::BasicBlock::Create(_llvm_context, "switch_exit", entry_f);
        auto sw = eb.CreateSwitch(query_id, switch_exit_bb, extracted_funcs.size());

        for (size_t i = 0; i < extracted_funcs.size(); ++i) {
            auto case_bb = llvm::BasicBlock::Create(_llvm_context, llvm::Twine{"case_"} + llvm::Twine{i}, entry_f);
            eb.SetInsertPoint(case_bb);
            auto intersection_f_name = luisa::format("lc_ray_query_triangle_intersection_{}", i);
            auto intersection_f = _llvm_module->getFunction(llvm::StringRef{intersection_f_name.data(), intersection_f_name.size()});
            auto res = eb.CreateCall(intersection_f, {p_ctx});
            auto committed = eb.CreateExtractValue(res, 1);
            auto terminated = eb.CreateExtractValue(res, 2);

            auto ignore_bb = llvm::BasicBlock::Create(_llvm_context, llvm::Twine{"ignore_"} + llvm::Twine{i}, entry_f);
            auto store_terminate_bb = llvm::BasicBlock::Create(_llvm_context, llvm::Twine{"store_terminate_"} + llvm::Twine{i}, entry_f);
            eb.CreateCondBr(eb.CreateICmpEQ(committed, eb.getInt8(0)), ignore_bb, store_terminate_bb);
            eb.SetInsertPoint(ignore_bb);
            _call_optix_ignore_intersection(eb);
            eb.CreateBr(store_terminate_bb);
            eb.SetInsertPoint(store_terminate_bb);
            eb.CreateStore(eb.CreateICmpNE(terminated, eb.getInt8(0)), should_terminate_alloca);
            eb.CreateBr(switch_exit_bb);

            sw->addCase(eb.getInt32(static_cast<uint32_t>(i)), case_bb);
        }
        eb.SetInsertPoint(switch_exit_bb);
        auto tri_term = eb.CreateLoad(eb.getInt1Ty(), should_terminate_alloca);
        eb.CreateBr(terminate_check_bb);

        eb.SetInsertPoint(terminate_check_bb);
        auto phi_term = eb.CreatePHI(eb.getInt1Ty(), 2);
        phi_term->addIncoming(proc_term, procedural_bb);
        phi_term->addIncoming(tri_term, switch_exit_bb);

        auto do_terminate_bb = llvm::BasicBlock::Create(_llvm_context, "do_terminate", entry_f);
        auto exit_bb = llvm::BasicBlock::Create(_llvm_context, "exit", entry_f);
        eb.CreateCondBr(phi_term, do_terminate_bb, exit_bb);

        eb.SetInsertPoint(do_terminate_bb);
        _call_optix_terminate_ray(eb);
        eb.CreateBr(exit_bb);

        eb.SetInsertPoint(exit_bb);
        eb.CreateRetVoid();
    };

    auto generate_intersection_entry = [&]() {
        auto ft = llvm::FunctionType::get(llvm::Type::getVoidTy(_llvm_context), {}, false);
        auto entry_f = llvm::Function::Create(ft, llvm::Function::ExternalLinkage, "__intersection__ray_query", _llvm_module.get());
        auto entry_bb = llvm::BasicBlock::Create(_llvm_context, "entry", entry_f);
        IB eb{entry_bb};

        _call_optix_set_payload_types(eb, 1u << 1u /* LC_PAYLOAD_TYPE_RAY_QUERY */);
        auto r0 = _call_optix_get_payload(eb, 0u);
        auto r1 = _call_optix_get_payload(eb, 1u);

        auto query_id = eb.CreateLShr(r0, 24);
        auto p_ctx_hi = eb.CreateAnd(r0, eb.getInt32(0xffffffu));
        auto p_ctx_lo = r1;
        auto p_ctx = eb.CreateIntToPtr(
            eb.CreateOr(eb.CreateShl(eb.CreateZExt(p_ctx_hi, eb.getInt64Ty()), 32),
                        eb.CreateZExt(p_ctx_lo, eb.getInt64Ty())),
            eb.getPtrTy());

        auto exit_bb = llvm::BasicBlock::Create(_llvm_context, "exit", entry_f);
        auto sw = eb.CreateSwitch(query_id, exit_bb, extracted_funcs.size());

        for (size_t i = 0; i < extracted_funcs.size(); ++i) {
            auto case_bb = llvm::BasicBlock::Create(_llvm_context, llvm::Twine{"case_"} + llvm::Twine{i}, entry_f);
            eb.SetInsertPoint(case_bb);
            auto intersection_f_name = luisa::format("lc_ray_query_procedural_intersection_{}", i);
            auto intersection_f = _llvm_module->getFunction(llvm::StringRef{intersection_f_name.data(), intersection_f_name.size()});
            auto res = eb.CreateCall(intersection_f, {p_ctx});
            auto t_hit = eb.CreateExtractValue(res, 0);
            auto committed = eb.CreateExtractValue(res, 1);
            auto terminated = eb.CreateExtractValue(res, 2);

            auto report_bb = llvm::BasicBlock::Create(_llvm_context, llvm::Twine{"report_"} + llvm::Twine{i}, entry_f);
            eb.CreateCondBr(eb.CreateICmpNE(committed, eb.getInt8(0)), report_bb, exit_bb);
            eb.SetInsertPoint(report_bb);
            auto hit_kind_val = eb.CreateSelect(eb.CreateICmpNE(terminated, eb.getInt8(0)),
                                                eb.getInt32(0x02u /* LC_HIT_KIND_PROCEDURAL_TERMINATED */),
                                                eb.getInt32(0x01u /* LC_HIT_KIND_PROCEDURAL */));
            _call_optix_report_intersection(eb, hit_kind_val, t_hit);
            eb.CreateBr(exit_bb);
            sw->addCase(eb.getInt32(static_cast<uint32_t>(i)), case_bb);
        }

        eb.SetInsertPoint(exit_bb);
        eb.CreateRetVoid();
    };

    generate_anyhit_entry();
    generate_intersection_entry();

    // Lower ray query intrinsics in all functions
    for (auto &f : *_llvm_module) {
        _lower_ray_query_intrinsics(&f);
    }

    // Remove the original extracted functions
    for (auto f : extracted_funcs) {
        f->eraseFromParent();
    }
}

}// namespace luisa::compute::cuda
