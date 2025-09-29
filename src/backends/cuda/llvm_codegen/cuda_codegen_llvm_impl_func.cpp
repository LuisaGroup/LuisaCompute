//
// Created by mike on 9/25/25.
//

#include "cuda_codegen_llvm_impl.h"

namespace luisa::compute::cuda {

llvm::Function *CUDACodegenLLVMImpl::_get_llvm_function(const xir::Function *func) noexcept {
    if (auto iter = _xir_to_llvm_function.find(func); iter != _xir_to_llvm_function.end()) {
        return iter->second;
    }
    auto llvm_func = [this, func]() noexcept {
        switch (func->derived_function_tag()) {
            case xir::DerivedFunctionTag::KERNEL: return _create_llvm_kernel_function(static_cast<const xir::KernelFunction *>(func));
            case xir::DerivedFunctionTag::CALLABLE: return _create_llvm_callable_function(static_cast<const xir::CallableFunction *>(func));
            case xir::DerivedFunctionTag::EXTERNAL: return _create_llvm_external_function(static_cast<const xir::ExternalFunction *>(func));
            default: break;
        }
        LUISA_ERROR_WITH_LOCATION("Unsupported function type.");
    }();
    auto [iter, success] = _xir_to_llvm_function.try_emplace(func, llvm_func);
    LUISA_ASSERT(success, "Failed to insert LLVM function.");
    return iter->second;
}

llvm::Function *CUDACodegenLLVMImpl::_create_llvm_kernel_function(const xir::KernelFunction *func) noexcept {
    llvm::SmallVector<llvm::Type *> llvm_arg_members;
    llvm::SmallVector<size_t> llvm_arg_member_indices;
    auto current_offset = static_cast<size_t>(0u);
    static constexpr auto argument_alignment = 16u;
    for (auto arg : func->arguments()) {
        auto next_offset = luisa::align(current_offset, argument_alignment);
        if (next_offset > current_offset) {
            auto llvm_i8_type = llvm::Type::getInt8Ty(_llvm_context);
            llvm_arg_members.emplace_back(llvm::ArrayType::get(llvm_i8_type, next_offset - current_offset));
        }
        llvm_arg_member_indices.emplace_back(llvm_arg_members.size());
        llvm_arg_members.emplace_back(_get_llvm_type(arg->type())->mem_type);
        current_offset = next_offset + _data_layout->getTypeAllocSize(llvm_arg_members.back()).getFixedValue();
    }
    if (current_offset % argument_alignment != 0u) {
        auto llvm_i8_type = llvm::Type::getInt8Ty(_llvm_context);
        auto count = luisa::align(current_offset, argument_alignment) - current_offset;
        llvm_arg_members.emplace_back(llvm::ArrayType::get(llvm_i8_type, count));
    }
    auto llvm_arg_struct_type = llvm::StructType::create(_llvm_context, llvm_arg_members, "kernel.params.struct");
    auto [llvm_kernel, llvm_arg_struct] = [&]() noexcept -> std::pair<llvm::Function *, llvm::Value *> {
        auto llvm_void_type = llvm::Type::getVoidTy(_llvm_context);
        if (_config.enable_ray_tracing) {// ray tracing kernels use constant memory for args
            auto llvm_global_arg = new ::llvm::GlobalVariable{
                *_llvm_module, llvm_arg_struct_type, true, llvm::GlobalValue::ExternalLinkage,
                nullptr, "params", nullptr, llvm::GlobalValue::NotThreadLocal,
                nvptx_address_space_constant, true};
            llvm_global_arg->setAlignment(llvm::Align{argument_alignment});
            auto llvm_func_type = llvm::FunctionType::get(llvm_void_type, {}, false);
            auto llvm_func = llvm::Function::Create(llvm_func_type, llvm::Function::ExternalLinkage,
                                                    "__raygen__main", _llvm_module.get());
            return std::make_pair(llvm_func, llvm_global_arg);
        }
        // normal kernels use direct arguments
        auto llvm_func_type = llvm::FunctionType::get(llvm_void_type, {llvm_arg_struct_type}, false);
        auto llvm_func = llvm::Function::Create(llvm_func_type, llvm::Function::ExternalLinkage,
                                                "kernel_main", _llvm_module.get());
        auto llvm_arg = llvm_func->getArg(0);
        llvm_arg->setName("params");
        return std::make_pair(llvm_func, llvm_arg);
    }();
    llvm_kernel->setCallingConv(llvm::CallingConv::PTX_Kernel);
    FunctionContext func_ctx{llvm_kernel};
    // load arguments
    IB b{func_ctx.llvm_entry_block};
    if (llvm_arg_struct->getType()->isPointerTy()) {
        llvm_arg_struct = b.CreateAlignedLoad(llvm_arg_struct_type, llvm_arg_struct,
                                              llvm::Align{argument_alignment},
                                              "params.load");
    }
    // TODO: load args, note we need to convert from mem_type to reg_type
    LUISA_NOT_IMPLEMENTED();
    auto body = _translate_basic_block(func_ctx, func->body_block());
    // branch from entry to body
    b.CreateBr(body);
    return llvm_kernel;
}

llvm::Function *CUDACodegenLLVMImpl::_create_llvm_callable_function(const xir::CallableFunction *func) noexcept {
    llvm::SmallVector<llvm::Type *> llvm_arg_types;
    for (auto arg : func->arguments()) {
        if (arg->is_reference()) {
            llvm_arg_types.emplace_back(llvm::PointerType::get(_llvm_context, 0));
        } else {
            llvm_arg_types.emplace_back(_get_llvm_type(arg->type())->reg_type);
        }
    }
    auto llvm_ret_type = func->type() == nullptr ? llvm::Type::getVoidTy(_llvm_context) :
                                                   _get_llvm_type(func->type())->reg_type;
    auto llvm_func_type = llvm::FunctionType::get(llvm_ret_type, llvm_arg_types, false);
    auto llvm_func = llvm::Function::Create(llvm_func_type, llvm::Function::PrivateLinkage, 0,
                                            func->name().value_or("callable"), _llvm_module.get());
    FunctionContext func_ctx{llvm_func};
    auto llvm_arg_iter = llvm_func->arg_begin();
    for (auto arg : func->arguments()) {
        func_ctx.local_values.try_emplace(arg, llvm_arg_iter++);
    }
    auto body = _translate_basic_block(func_ctx, func->body_block());
    // branch from entry to body
    IB b{func_ctx.llvm_entry_block};
    b.CreateBr(body);
    return llvm_func;
}

llvm::Function *CUDACodegenLLVMImpl::_create_llvm_external_function(const xir::ExternalFunction *func) noexcept {
    LUISA_NOT_IMPLEMENTED();
}

llvm::BasicBlock *CUDACodegenLLVMImpl::_translate_basic_block(FunctionContext &func_ctx, const xir::BasicBlock *bb) noexcept {
    return nullptr;
}

}// namespace luisa::compute::cuda
