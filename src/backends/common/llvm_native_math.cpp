#include "llvm_native_math.h"
#include "llvm_native_math_internal.h"

#include <string>

#include <llvm/IR/Attributes.h>
#include <llvm/IR/DerivedTypes.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/Module.h>

namespace luisa::compute::cpu {

namespace {

[[nodiscard]] ::llvm::FixedVectorType *f32_vector_type(
    ::llvm::Value *vector) noexcept {
    auto *type = ::llvm::dyn_cast<::llvm::FixedVectorType>(
        vector == nullptr ? nullptr : vector->getType());
    return type != nullptr && type->getElementType()->isFloatTy() ?
               type :
               nullptr;
}

void configure_native_function(
    ::llvm::Function *function, bool reads_memory) {
    function->addFnAttr(::llvm::Attribute::AlwaysInline);
    function->addFnAttr(::llvm::Attribute::NoUnwind);
    function->addFnAttr(::llvm::Attribute::NoRecurse);
    function->addFnAttr(::llvm::Attribute::WillReturn);
    function->addFnAttr("luisa.cpu.native_math");
    if (reads_memory) {
        function->setOnlyReadsMemory();
    } else {
        function->setDoesNotAccessMemory();
    }
}

[[nodiscard]] ::llvm::Function *get_or_create_trig_f32(
    ::llvm::Module &module, ::llvm::FixedVectorType *type,
    LLVMNativeMathMode mode, detail::NativeTrigKind kind) {
    auto width = type->getNumElements();
    auto operation = kind == detail::NativeTrigKind::sin ? "sin" :
                     kind == detail::NativeTrigKind::cos ? "cos" :
                                                           "tan";
    auto precise_suffix = kind == detail::NativeTrigKind::tan ?
                              "u35" :
                              "u10";
    auto suffix = mode == LLVMNativeMathMode::fast ?
                      "fast" :
                      precise_suffix;
    auto name = std::string{"__luisa_cpu_native_"} + operation +
                "_f32_v" + std::to_string(width) + "_" + suffix;
    auto *function = module.getFunction(name);
    if (function != nullptr) { return function; }

    auto *precise_function = mode == LLVMNativeMathMode::fast ?
                                 get_or_create_trig_f32(
                                     module, type, LLVMNativeMathMode::precise, kind) :
                                 nullptr;
    auto *function_type = ::llvm::FunctionType::get(
        type, {type}, false);
    function = ::llvm::Function::Create(
        function_type, ::llvm::GlobalValue::InternalLinkage,
        name, module);
    configure_native_function(function, true);
    if (mode == LLVMNativeMathMode::fast) {
        detail::build_fast_trig_f32(
            module, function, width, kind, precise_function);
    } else {
        detail::build_precise_trig_f32(
            module, function, width, kind);
    }
    return function;
}

[[nodiscard]] ::llvm::Value *emit_trig_f32(
    ::llvm::Module &module, ::llvm::IRBuilder<> &builder,
    ::llvm::Value *vector, LLVMNativeMathMode mode,
    detail::NativeTrigKind kind) {
    auto *type = f32_vector_type(vector);
    if (type == nullptr) { return nullptr; }
    auto *function = get_or_create_trig_f32(
        module, type, mode, kind);
    auto operation = kind == detail::NativeTrigKind::sin ? "sin" :
                     kind == detail::NativeTrigKind::cos ? "cos" :
                                                           "tan";
    return builder.CreateCall(
        function, {vector}, std::string{"native."} + operation);
}

[[nodiscard]] ::llvm::Value *emit_inverse_trig_f32(
    ::llvm::Module &module, ::llvm::IRBuilder<> &builder,
    ::llvm::Value *vector, LLVMNativeMathMode mode,
    detail::NativeInverseTrigKind kind) {
    auto *type = f32_vector_type(vector);
    if (type == nullptr) { return nullptr; }
    auto operation = kind == detail::NativeInverseTrigKind::asin ? "asin" :
                     kind == detail::NativeInverseTrigKind::acos ? "acos" :
                                                                   "atan";
    auto width = type->getNumElements();
    auto suffix = mode == LLVMNativeMathMode::fast ? "fast" : "u35";
    auto name = std::string{"__luisa_cpu_native_"} + operation +
                "_f32_v" + std::to_string(width) + "_" + suffix;
    auto *function = module.getFunction(name);
    if (function == nullptr) {
        auto *function_type = ::llvm::FunctionType::get(
            type, {type}, false);
        function = ::llvm::Function::Create(
            function_type, ::llvm::GlobalValue::InternalLinkage,
            name, module);
        configure_native_function(function, false);
        if (mode == LLVMNativeMathMode::fast) {
            detail::build_fast_inverse_trig_f32(
                module, function, width, kind);
        } else {
            detail::build_precise_inverse_trig_f32(
                module, function, width, kind);
        }
    }
    return builder.CreateCall(
        function, {vector}, std::string{"native."} + operation);
}

[[nodiscard]] ::llvm::Value *emit_exp_log_f32(
    ::llvm::Module &module, ::llvm::IRBuilder<> &builder,
    ::llvm::Value *vector, LLVMNativeMathMode mode, bool logarithm) {
    auto *type = f32_vector_type(vector);
    if (type == nullptr) { return nullptr; }
    auto width = type->getNumElements();
    auto suffix = mode == LLVMNativeMathMode::fast ?
                      "fast" :
                  logarithm ? "u35" :
                              "u10";
    auto operation = logarithm ? "log" : "exp";
    auto name = std::string{"__luisa_cpu_native_"} + operation +
                "_f32_v" + std::to_string(width) + "_" + suffix;
    auto *function = module.getFunction(name);
    if (function == nullptr) {
        auto *function_type = ::llvm::FunctionType::get(
            type, {type}, false);
        function = ::llvm::Function::Create(
            function_type, ::llvm::GlobalValue::InternalLinkage,
            name, module);
        configure_native_function(function, false);
        if (mode == LLVMNativeMathMode::fast) {
            detail::build_fast_exp_log_f32(
                module, function, width, logarithm);
        } else {
            detail::build_precise_exp_log_f32(
                module, function, width, logarithm);
        }
    }
    return builder.CreateCall(
        function, {vector}, std::string{"native."} + operation);
}

}// namespace

::llvm::Value *LLVMNativeMath::emit_sin_f32(
    ::llvm::Module &module, ::llvm::IRBuilder<> &builder,
    ::llvm::Value *vector, LLVMNativeMathMode mode) {
    return emit_trig_f32(
        module, builder, vector, mode, detail::NativeTrigKind::sin);
}

::llvm::Value *LLVMNativeMath::emit_cos_f32(
    ::llvm::Module &module, ::llvm::IRBuilder<> &builder,
    ::llvm::Value *vector, LLVMNativeMathMode mode) {
    return emit_trig_f32(
        module, builder, vector, mode, detail::NativeTrigKind::cos);
}

::llvm::Value *LLVMNativeMath::emit_tan_f32(
    ::llvm::Module &module, ::llvm::IRBuilder<> &builder,
    ::llvm::Value *vector, LLVMNativeMathMode mode) {
    return emit_trig_f32(
        module, builder, vector, mode, detail::NativeTrigKind::tan);
}

::llvm::Value *LLVMNativeMath::emit_asin_f32(
    ::llvm::Module &module, ::llvm::IRBuilder<> &builder,
    ::llvm::Value *vector, LLVMNativeMathMode mode) {
    return emit_inverse_trig_f32(
        module, builder, vector, mode,
        detail::NativeInverseTrigKind::asin);
}

::llvm::Value *LLVMNativeMath::emit_acos_f32(
    ::llvm::Module &module, ::llvm::IRBuilder<> &builder,
    ::llvm::Value *vector, LLVMNativeMathMode mode) {
    return emit_inverse_trig_f32(
        module, builder, vector, mode,
        detail::NativeInverseTrigKind::acos);
}

::llvm::Value *LLVMNativeMath::emit_atan_f32(
    ::llvm::Module &module, ::llvm::IRBuilder<> &builder,
    ::llvm::Value *vector, LLVMNativeMathMode mode) {
    return emit_inverse_trig_f32(
        module, builder, vector, mode,
        detail::NativeInverseTrigKind::atan);
}

::llvm::Value *LLVMNativeMath::emit_exp_f32(
    ::llvm::Module &module, ::llvm::IRBuilder<> &builder,
    ::llvm::Value *vector, LLVMNativeMathMode mode) {
    return emit_exp_log_f32(module, builder, vector, mode, false);
}

::llvm::Value *LLVMNativeMath::emit_log_f32(
    ::llvm::Module &module, ::llvm::IRBuilder<> &builder,
    ::llvm::Value *vector, LLVMNativeMathMode mode) {
    return emit_exp_log_f32(module, builder, vector, mode, true);
}

}// namespace luisa::compute::cpu
