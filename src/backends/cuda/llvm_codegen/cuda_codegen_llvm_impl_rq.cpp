//
// Created by mike on 11/1/25.
//

#include <llvm/IR/Dominators.h>
#include <llvm/Analysis/LoopInfo.h>

#include "cuda_codegen_llvm_impl.h"

namespace luisa::compute::cuda {

void CUDACodegenLLVMImpl::_translate_ray_query_loop_inst(IB &b, FunctionContext &func_ctx, const xir::RayQueryLoopInst *inst) noexcept {
    b.GetInsertBlock()->setName("ray.query.loop");
    auto llvm_dispatch_block = func_ctx.get_local_value<llvm::BasicBlock>(inst->dispatch_block());
    llvm_dispatch_block->setName("ray.query.dispatch");
    b.CreateBr(llvm_dispatch_block);
}

void CUDACodegenLLVMImpl::_translate_ray_query_dispatch_inst(IB &b, FunctionContext &func_ctx, const xir::RayQueryDispatchInst *inst) noexcept {
    // luisa.ray.query.proceed();
    // switch (luisa.ray.query.state()) {
    //    case surface: br surface_block
    //    case procedural: br procedural_block
    //    default: br exit_block
    // }
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

llvm::Value *CUDACodegenLLVMImpl::_translate_ray_query_object_read_inst(IB &b, FunctionContext &func_ctx, const xir::RayQueryObjectReadInst *inst) noexcept {
    auto intrinsic = [op = inst->op()] {
        switch (op) {
            case xir::RayQueryObjectReadOp::RAY_QUERY_OBJECT_WORLD_SPACE_RAY: return llvm_ray_query_intrinsic_name_world_space_ray;
            case xir::RayQueryObjectReadOp::RAY_QUERY_OBJECT_CANDIDATE_OBJECT_SPACE_RAY: return llvm_ray_query_intrinsic_name_object_space_ray;
            case xir::RayQueryObjectReadOp::RAY_QUERY_OBJECT_PROCEDURAL_CANDIDATE_HIT: return llvm_ray_query_intrinsic_name_procedural_candidate_hit;
            case xir::RayQueryObjectReadOp::RAY_QUERY_OBJECT_TRIANGLE_CANDIDATE_HIT: return llvm_ray_query_intrinsic_name_surface_candidate_hit;
            case xir::RayQueryObjectReadOp::RAY_QUERY_OBJECT_COMMITTED_HIT: return llvm_ray_query_intrinsic_name_committed_hit;
            case xir::RayQueryObjectReadOp::RAY_QUERY_OBJECT_IS_TRIANGLE_CANDIDATE: return llvm_ray_query_intrinsic_name_is_surface_candidate;
            case xir::RayQueryObjectReadOp::RAY_QUERY_OBJECT_IS_PROCEDURAL_CANDIDATE: return llvm_ray_query_intrinsic_name_is_procedural_candidate;
            case xir::RayQueryObjectReadOp::RAY_QUERY_OBJECT_IS_TERMINATED: return llvm_ray_query_intrinsic_name_is_terminated;
            default: break;
        }
        LUISA_ERROR("Invalid op (code = {}) for RayQueryObjectReadInst.", luisa::to_underlying(op));
    }();
    LUISA_DEBUG_ASSERT(inst->operand_count() == 1);
    auto llvm_ret_type = _get_llvm_type(inst->type())->reg_type;
    return _call_ray_query_intrinsic(b, intrinsic, llvm_ret_type, {});
}

void CUDACodegenLLVMImpl::_translate_ray_query_object_write_inst(IB &b, FunctionContext &func_ctx, const xir::RayQueryObjectWriteInst *inst) noexcept {
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
    llvm::SmallVector<llvm::Value *, 2> llvm_args;
    for (auto &&op_use : inst->operand_uses().subspan(1)/* skip the query object */) {
        llvm_args.emplace_back(_get_llvm_value(b, func_ctx, op_use->value()));
    }
    _call_ray_query_intrinsic(b, intrinsic, b.getVoidTy(), llvm_args);
}

void CUDACodegenLLVMImpl::_translate_ray_query_pipeline_inst(IB &b, FunctionContext &func_ctx, const xir::RayQueryPipelineInst *inst) noexcept {
    LUISA_NOT_IMPLEMENTED();
}

llvm::Value *CUDACodegenLLVMImpl::_call_ray_query_intrinsic(IB &b, llvm::StringRef name, llvm::Type *ret, llvm::ArrayRef<llvm::Value *> args) noexcept {
    auto func = _llvm_module->getFunction(name);
    if (func == nullptr) {
        llvm::SmallVector<llvm::Type *, 2> arg_types;
        for (auto arg : args) { arg_types.push_back(arg->getType()); }
        auto func_type = llvm::FunctionType::get(ret, arg_types, false);
        func = llvm::Function::Create(func_type, llvm::Function::ExternalLinkage, name, _llvm_module.get());
    }
    return b.CreateCall(func, args);
}

void CUDACodegenLLVMImpl::_materialize_ray_query_loops() noexcept {
    // TODO
}

}// namespace luisa::compute::cuda
