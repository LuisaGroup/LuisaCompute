//
// Created by mike on 3/18/26.
//

#include "hip_codegen_llvm_impl.h"

namespace luisa::compute::hip {

void HIPCodegenLLVMImpl::_translate_if_inst(IB &b, const FunctionContext &func_ctx, const xir::IfInst *inst) noexcept {
    auto llvm_cond = _get_llvm_value(b, func_ctx, inst->condition());
    auto llvm_true_block = func_ctx.get_local_value<llvm::BasicBlock>(inst->true_block());
    auto llvm_false_block = func_ctx.get_local_value<llvm::BasicBlock>(inst->false_block());
    b.CreateCondBr(llvm_cond, llvm_true_block, llvm_false_block);
}

void HIPCodegenLLVMImpl::_translate_switch_inst(IB &b, const FunctionContext &func_ctx, const xir::SwitchInst *inst) noexcept {
    auto llvm_value = _get_llvm_value(b, func_ctx, inst->value());
    auto llvm_default_block = func_ctx.get_local_value<llvm::BasicBlock>(inst->default_block());
    auto llvm_switch = b.CreateSwitch(llvm_value, llvm_default_block, inst->case_count());
    for (auto i = 0u; i < inst->case_count(); i++) {
        auto llvm_case_value = b.getIntN(
            llvm_value->getType()->getIntegerBitWidth(), inst->case_value(i));
        auto llvm_case_block = func_ctx.get_local_value<llvm::BasicBlock>(inst->case_block(i));
        llvm_switch->addCase(llvm_case_value, llvm_case_block);
    }
}

void HIPCodegenLLVMImpl::_translate_loop_inst(IB &b, const FunctionContext &func_ctx, const xir::LoopInst *inst) noexcept {
    auto llvm_prepare_block = func_ctx.get_local_value<llvm::BasicBlock>(inst->prepare_block());
    b.CreateBr(llvm_prepare_block);
}

void HIPCodegenLLVMImpl::_translate_simple_loop_inst(IB &b, const FunctionContext &func_ctx, const xir::SimpleLoopInst *inst) noexcept {
    auto llvm_body_block = func_ctx.get_local_value<llvm::BasicBlock>(inst->body_block());
    b.CreateBr(llvm_body_block);
}

void HIPCodegenLLVMImpl::_translate_branch_inst(IB &b, const FunctionContext &func_ctx, const xir::BranchInst *inst) noexcept {
    auto llvm_target = func_ctx.get_local_value<llvm::BasicBlock>(inst->target_block());
    b.CreateBr(llvm_target);
}

void HIPCodegenLLVMImpl::_translate_conditional_branch_inst(IB &b, const FunctionContext &func_ctx, const xir::ConditionalBranchInst *inst) noexcept {
    auto llvm_cond = _get_llvm_value(b, func_ctx, inst->condition());
    auto llvm_true_target = func_ctx.get_local_value<llvm::BasicBlock>(inst->true_block());
    auto llvm_false_target = func_ctx.get_local_value<llvm::BasicBlock>(inst->false_block());
    b.CreateCondBr(llvm_cond, llvm_true_target, llvm_false_target);
}

void HIPCodegenLLVMImpl::_translate_unreachable_inst(IB &b, FunctionContext &func_ctx, const xir::UnreachableInst *inst) noexcept {
    b.CreateUnreachable();
}

void HIPCodegenLLVMImpl::_translate_break_inst(IB &b, const FunctionContext &func_ctx, const xir::BreakInst *inst) noexcept {
    auto llvm_target = func_ctx.get_local_value<llvm::BasicBlock>(inst->target_block());
    b.CreateBr(llvm_target);
}

void HIPCodegenLLVMImpl::_translate_continue_inst(IB &b, const FunctionContext &func_ctx, const xir::ContinueInst *inst) noexcept {
    auto llvm_target = func_ctx.get_local_value<llvm::BasicBlock>(inst->target_block());
    b.CreateBr(llvm_target);
}

void HIPCodegenLLVMImpl::_translate_return_inst(IB &b, const FunctionContext &func_ctx, const xir::ReturnInst *inst) noexcept {
    if (auto v = inst->return_value()) {
        auto llvm_ret_v = _get_llvm_value(b, func_ctx, v);
        b.CreateRet(llvm_ret_v);
    } else {
        b.CreateRetVoid();
    }
}

}// namespace luisa::compute::hip
