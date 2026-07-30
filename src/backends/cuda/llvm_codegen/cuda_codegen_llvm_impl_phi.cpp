//
// Created by mike on 11/1/25.
//

#include "cuda_codegen_llvm_impl.h"

namespace luisa::compute::cuda {

llvm::PHINode *CUDACodegenLLVMImpl::_translate_phi_inst(IB &b, FunctionContext &func_ctx, const xir::PhiInst *inst) noexcept {
    func_ctx.pending_phi_nodes.emplace_back(inst);
    auto llvm_type = _get_llvm_type(inst->type());
    return b.CreatePHI(llvm_type->reg_type, inst->incoming_count(), inst->name().value_or(""));
}

void CUDACodegenLLVMImpl::_finalize_pending_phi_nodes(const FunctionContext &func_ctx, const luisa::unordered_set<const xir::BasicBlock *> &translated_blocks) noexcept {
    IB b{_llvm_context};
    for (auto phi : func_ctx.pending_phi_nodes) {
        auto llvm_phi = func_ctx.get_local_value<llvm::PHINode>(phi);
        for (auto i = 0u; i < phi->incoming_count(); i++) {
            auto [value, block] = phi->incoming(i);
            if (!translated_blocks.contains(block)) { continue; }
            auto llvm_value = _get_llvm_value(b, func_ctx, value);
            auto llvm_block = func_ctx.get_local_value<llvm::BasicBlock>(block);
            llvm_phi->addIncoming(llvm_value, llvm_block);
        }
    }
}

}// namespace luisa::compute::cuda
