//
// Created by mike on 11/1/25.
//

#include "cuda_codegen_llvm_impl.h"

namespace luisa::compute::cuda {

llvm::Value *CUDACodegenLLVMImpl::_translate_alloca_inst(IB &b, FunctionContext &func_ctx, const xir::AllocaInst *inst) noexcept {
    if (inst->is_local()) {
        if (auto iter = func_ctx.packed_spill_slots.find(inst); iter != func_ctx.packed_spill_slots.end()) {
            return iter->second;
        }
        auto llvm_type = _get_llvm_type(inst->type())->mem_type;
        auto llvm_alloca = b.CreateAlloca(llvm_type, nullptr, inst->name().value_or(""));
        llvm_alloca->setAlignment(llvm::Align{_get_type_alignment(inst->type())});
        return llvm_alloca;
    }
    auto llvm_type = _get_llvm_type(inst->type())->mem_type;
    // shared alloca's are mapped to global variables in address space nvvm_address_space_shared
    auto llvm_global = new llvm::GlobalVariable{
        *_llvm_module, llvm_type, false, llvm::GlobalValue::PrivateLinkage,
        llvm::UndefValue::get(llvm_type), inst->name().value_or("shared"), nullptr,
        llvm::GlobalValue::NotThreadLocal, nvptx_address_space_shared, false};
    llvm_global->setAlignment(llvm::Align{_get_type_alignment(inst->type())});
    llvm_global->setUnnamedAddr(llvm::GlobalValue::UnnamedAddr::Global);
    return llvm_global;
}

llvm::Value *CUDACodegenLLVMImpl::_translate_load_inst(IB &b, const FunctionContext &func_ctx, const xir::LoadInst *inst) noexcept {
    auto llvm_ptr = func_ctx.get_local_value<llvm::Value>(inst->variable());
    return _load_llvm_value(b, llvm_ptr, inst->type());
}

void CUDACodegenLLVMImpl::_translate_store_inst(IB &b, const FunctionContext &func_ctx, const xir::StoreInst *inst) noexcept {
    auto llvm_ptr = func_ctx.get_local_value<llvm::Value>(inst->variable());
    auto llvm_value = _get_llvm_value(b, func_ctx, inst->value());
    _store_llvm_value(b, llvm_ptr, llvm_value, inst->value()->type());
}

llvm::Value *CUDACodegenLLVMImpl::_translate_gep_inst(IB &b, FunctionContext &func_ctx, const xir::GEPInst *inst) noexcept {
    auto llvm_base_ptr = _get_llvm_value(b, func_ctx, inst->base());
    auto [llvm_elem_ptr, elem_type] = _lower_access_chain_address(b, func_ctx, llvm_base_ptr, inst->base()->type(), inst->operand_uses().subspan(1));
    LUISA_DEBUG_ASSERT(elem_type == inst->type());
    return llvm_elem_ptr;
}

llvm::Value *CUDACodegenLLVMImpl::_load_llvm_value(IB &b, llvm::Value *llvm_ptr, const Type *type) noexcept {
    auto llvm_type = _get_llvm_type(type);
    auto llvm_mem_v = b.CreateAlignedLoad(llvm_type->mem_type, llvm_ptr, llvm::Align{_get_type_alignment(type)});
    return _convert_llvm_mem_value_to_reg(b, llvm_mem_v, type);
}

void CUDACodegenLLVMImpl::_store_llvm_value(IB &b, llvm::Value *llvm_ptr, llvm::Value *llvm_value, const Type *type) noexcept {
    auto llvm_mem_v = _convert_llvm_reg_value_to_mem(b, llvm_value, type);
    b.CreateAlignedStore(llvm_mem_v, llvm_ptr, llvm::Align{_get_type_alignment(type)});
}

llvm::Value *CUDACodegenLLVMImpl::_create_temp_in_alloca_block(const FunctionContext &func_ctx, llvm::Type *t, size_t align) noexcept {
    IB b{func_ctx.llvm_alloca_block->getTerminator()};
    auto llvm_alloca = b.CreateAlloca(t);
    if (align != 0) { llvm_alloca->setAlignment(llvm::Align{align}); }
    return llvm_alloca;
}

}// namespace luisa::compute::cuda
