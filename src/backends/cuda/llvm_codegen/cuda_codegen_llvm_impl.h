//
// Created by mike on 9/19/25.
//

#pragma once

#include <llvm/IR/Module.h>
#include <llvm/IR/Instructions.h>
#include <llvm/IR/Operator.h>
#include <llvm/IR/Verifier.h>
#include <llvm/IR/Constants.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/Support/SourceMgr.h>
#include <llvm/IRReader/IRReader.h>
#include <llvm/Target/TargetMachine.h>

#include <luisa/core/logging.h>
#include <luisa/ast/type.h>
#include <luisa/xir/module.h>

#include "cuda_codegen_llvm_config.h"

namespace luisa::compute::cuda {

struct CUDACodegenLLVMConfig;

class CUDACodegenLLVMImpl {

public:
    static constexpr auto nvptx_target_triple = "nvptx64-nvidia-cuda";
    static constexpr auto nvptx_address_space_global = 1u;
    static constexpr auto nvptx_address_space_shared = 3u;
    static constexpr auto nvptx_address_space_constant = 4u;
    static constexpr auto nvptx_address_space_local = 5u;

    struct LLVMTypeInfo {
        llvm::Type *mem_type;                // The LLVM type used in memory (with proper alignment)
        llvm::Type *reg_type;                // The LLVM type used in registers
        luisa::vector<size_t> member_indices;// For struct type, the mapping from member index to LLVM struct field index
        luisa::vector<size_t> member_offsets;// For struct type, the byte offset of each member
    };

    using IB = llvm::IRBuilder<>;

    struct FunctionContext {
        llvm::Function *llvm_func;
        llvm::BasicBlock *llvm_alloca_block;
        llvm::BasicBlock *llvm_entry_block;
        llvm::DenseMap<const xir::Value *, llvm::Value *> local_values;
        explicit FunctionContext(llvm::Function *f) noexcept;
    };

private:
    CUDACodegenLLVMConfig _config;
    llvm::TargetMachine *_target_machine{nullptr};
    std::unique_ptr<llvm::DataLayout> _data_layout;
    llvm::LLVMContext _llvm_context;
    std::unique_ptr<llvm::Module> _llvm_module;

    llvm::Type *_llvm_buffer_type{nullptr};             // { ptr, i32 offset, i32 size } as defined in cuda_buffer.h
    llvm::Type *_llvm_texture_type{nullptr};            // { i64 handle, i64 storage } as defined in cuda_texture.h
    llvm::Type *_llvm_bindless_array_type{nullptr};     // { ptr slots, i64 capacity } as defined in cuda_bindless_array.h
    llvm::Type *_llvm_bindless_array_slot_type{nullptr};// { i64 buffer, i64 size, i64 tex2d, i64 tex3d } as defined in cuda_bindless_array.h
    llvm::Type *_llvm_accel_type{nullptr};              // { i64 handle, ptr instances } as defined in cuda_accel.h
    llvm::Type *_llvm_accel_instance_type{nullptr};     // { [ 12 x float ] affine, i32 user_id, i32 sbt_offset, i32 mask, i32 flags, i64 handle, i32 padding } as defined in optix_api.h
    llvm::DenseMap<const Type *, luisa::unique_ptr<LLVMTypeInfo>> _xir_to_llvm_type;

    llvm::DenseMap<const xir::Function *, llvm::Function *> _xir_to_llvm_function;
    llvm::DenseMap<const xir::Constant *, llvm::Constant *> _xir_to_llvm_global;

private:
    [[nodiscard]] static const llvm::Target *_get_nvptx_target() noexcept;
    void _initialize() noexcept;
    void _run_optimization_passes() noexcept;
    void _dump_module(const std::filesystem::path &path) const noexcept;
    [[nodiscard]] luisa::string _generate_ptx() const noexcept;

    // defined in cuda_codegen_llvm_impl_type.cpp
    [[nodiscard]] const LLVMTypeInfo *_get_llvm_type(const Type *type) noexcept;
    [[nodiscard]] llvm::Type *_get_llvm_buffer_type() noexcept;
    [[nodiscard]] llvm::Type *_get_llvm_texture_type() noexcept;
    [[nodiscard]] llvm::Type *_get_llvm_bindless_array_type() noexcept;
    [[nodiscard]] llvm::Type *_get_llvm_bindless_array_slot_type() noexcept;
    [[nodiscard]] llvm::Type *_get_llvm_accel_type() noexcept;
    [[nodiscard]] llvm::Type *_get_llvm_accel_instance_type() noexcept;

    // defined in cuda_codegen_llvm_impl_func.cpp
    [[nodiscard]] llvm::Function *_get_llvm_function(const xir::Function *func) noexcept;
    [[nodiscard]] llvm::Function *_create_llvm_kernel_function(const xir::KernelFunction *func) noexcept;
    [[nodiscard]] llvm::Function *_create_llvm_callable_function(const xir::CallableFunction *func) noexcept;
    [[nodiscard]] llvm::Function *_create_llvm_external_function(const xir::ExternalFunction *func) noexcept;
    [[nodiscard]] llvm::BasicBlock *_translate_basic_block(FunctionContext &func_ctx, const xir::BasicBlock *bb) noexcept;

    // defined in cuda_codegen_llvm_impl_const.cpp
    [[nodiscard]] llvm::Value *_get_llvm_literal(IB &b, const Type *type, const void *data) noexcept;
    [[nodiscard]] llvm::Value *_get_llvm_constant(IB &b, const xir::Constant *c) noexcept;

public:
    explicit CUDACodegenLLVMImpl(CUDACodegenLLVMConfig config) noexcept;
    [[nodiscard]] luisa::string generate(const xir::Module &xir_module) noexcept;
};

}// namespace luisa::compute::cuda
