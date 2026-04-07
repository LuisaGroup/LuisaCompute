//
// Created by mike on 3/18/26.
//

#include "hip_codegen_llvm_impl.h"

namespace luisa::compute::hip {

llvm::Value *HIPCodegenLLVMImpl::_translate_alloca_inst(IB &b, FunctionContext &func_ctx, const xir::AllocaInst *inst) noexcept {
    auto llvm_type = _get_llvm_type(inst->type())->mem_type;
    if (inst->is_local()) {
        IB alloca_builder{func_ctx.llvm_alloca_block->getTerminator()};
        auto llvm_alloca = alloca_builder.CreateAlloca(llvm_type, nullptr, inst->name().value_or(""));
        llvm_alloca->setAlignment(llvm::Align{_get_type_alignment(inst->type())});
        return llvm_alloca;
    }
    auto llvm_global = new llvm::GlobalVariable{
        *_llvm_module, llvm_type, false, llvm::GlobalValue::PrivateLinkage,
        llvm::UndefValue::get(llvm_type), inst->name().value_or("shared"), nullptr,
        llvm::GlobalValue::NotThreadLocal, amdgpu_address_space_shared, false};
    llvm_global->setAlignment(llvm::Align{_get_type_alignment(inst->type())});
    llvm_global->setUnnamedAddr(llvm::GlobalValue::UnnamedAddr::Global);
    return llvm_global;
}

llvm::Value *HIPCodegenLLVMImpl::_translate_load_inst(IB &b, const FunctionContext &func_ctx, const xir::LoadInst *inst) noexcept {
    auto llvm_ptr = func_ctx.get_local_value<llvm::Value>(inst->variable());
    return _load_llvm_value(b, llvm_ptr, inst->type());
}

void HIPCodegenLLVMImpl::_translate_store_inst(IB &b, const FunctionContext &func_ctx, const xir::StoreInst *inst) noexcept {
    auto llvm_ptr = func_ctx.get_local_value<llvm::Value>(inst->variable());
    auto llvm_value = _get_llvm_value(b, func_ctx, inst->value());
    _store_llvm_value(b, llvm_ptr, llvm_value, inst->value()->type());
}

llvm::Value *HIPCodegenLLVMImpl::_translate_gep_inst(IB &b, FunctionContext &func_ctx, const xir::GEPInst *inst) noexcept {
    auto llvm_base_ptr = _get_llvm_value(b, func_ctx, inst->base());
    auto [llvm_elem_ptr, elem_type] = _lower_access_chain_address(b, func_ctx, llvm_base_ptr, inst->base()->type(), inst->operand_uses().subspan(1));
    LUISA_DEBUG_ASSERT(elem_type == inst->type());
    return llvm_elem_ptr;
}

llvm::Value *HIPCodegenLLVMImpl::_load_llvm_value(IB &b, llvm::Value *llvm_ptr, const Type *type) noexcept {
    auto llvm_type_info = _get_llvm_type(type);
    auto llvm_mem_v = b.CreateAlignedLoad(llvm_type_info->mem_type, llvm_ptr, llvm::Align{_get_type_alignment(type)});
    return _convert_llvm_mem_value_to_reg(b, llvm_mem_v, type);
}

void HIPCodegenLLVMImpl::_store_llvm_value(IB &b, llvm::Value *llvm_ptr, llvm::Value *llvm_value, const Type *type) noexcept {
    auto llvm_mem_v = _convert_llvm_reg_value_to_mem(b, llvm_value, type);
    b.CreateAlignedStore(llvm_mem_v, llvm_ptr, llvm::Align{_get_type_alignment(type)});
}

llvm::Value *HIPCodegenLLVMImpl::_create_temp_in_alloca_block(const FunctionContext &func_ctx, llvm::Type *t, size_t align) noexcept {
    IB b{func_ctx.llvm_alloca_block->getTerminator()};
    auto llvm_alloca = b.CreateAlloca(t);
    if (align != 0) { llvm_alloca->setAlignment(llvm::Align{align}); }
    return llvm_alloca;
}

}// namespace luisa::compute::hip
