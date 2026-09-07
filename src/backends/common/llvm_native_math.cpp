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

[[nodiscard]] ::llvm::Value *emit_atan2_f32(
    ::llvm::Module &module, ::llvm::IRBuilder<> &builder,
    ::llvm::Value *y, ::llvm::Value *x, LLVMNativeMathMode mode) {
    auto *type = f32_vector_type(y);
    if (type == nullptr || x == nullptr || x->getType() != type) {
        return nullptr;
    }
    auto width = type->getNumElements();
    auto suffix = mode == LLVMNativeMathMode::fast ? "fast" : "u35";
    auto name = std::string{"__luisa_cpu_native_atan2_f32_v"} +
                std::to_string(width) + "_" + suffix;
    auto *function = module.getFunction(name);
    if (function == nullptr) {
        auto *function_type = ::llvm::FunctionType::get(
            type, {type, type}, false);
        function = ::llvm::Function::Create(
            function_type, ::llvm::GlobalValue::InternalLinkage,
            name, module);
        configure_native_function(function, false);
        detail::build_atan2_f32(
            module, function, width, mode);
    }
    return builder.CreateCall(function, {y, x}, "native.atan2");
}

[[nodiscard]] ::llvm::Function *get_or_create_exp_log_f32(
    ::llvm::Module &module, ::llvm::FixedVectorType *type,
    LLVMNativeMathMode mode, detail::NativeExpLogKind kind) {
    auto width = type->getNumElements();
    auto logarithm = kind == detail::NativeExpLogKind::log ||
                     kind == detail::NativeExpLogKind::log2 ||
                     kind == detail::NativeExpLogKind::log10;
    auto suffix = mode == LLVMNativeMathMode::fast ?
                      "fast" :
                  logarithm ? "u35" :
                              "u10";
    auto operation = kind == detail::NativeExpLogKind::exp      ? "exp" :
                     kind == detail::NativeExpLogKind::exp_half ? "exp_half" :
                     kind == detail::NativeExpLogKind::exp2     ? "exp2" :
                     kind == detail::NativeExpLogKind::exp10    ? "exp10" :
                     kind == detail::NativeExpLogKind::log      ? "log" :
                     kind == detail::NativeExpLogKind::log2     ? "log2" :
                                                                  "log10";
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
                module, function, width, kind);
        } else {
            detail::build_precise_exp_log_f32(
                module, function, width, kind);
        }
    }
    return function;
}

[[nodiscard]] ::llvm::Value *emit_exp_log_f32(
    ::llvm::Module &module, ::llvm::IRBuilder<> &builder,
    ::llvm::Value *vector, LLVMNativeMathMode mode,
    detail::NativeExpLogKind kind) {
    auto *type = f32_vector_type(vector);
    if (type == nullptr) { return nullptr; }
    auto *function = get_or_create_exp_log_f32(
        module, type, mode, kind);
    auto operation = kind == detail::NativeExpLogKind::exp      ? "exp" :
                     kind == detail::NativeExpLogKind::exp_half ? "exp_half" :
                     kind == detail::NativeExpLogKind::exp2     ? "exp2" :
                     kind == detail::NativeExpLogKind::exp10    ? "exp10" :
                     kind == detail::NativeExpLogKind::log      ? "log" :
                     kind == detail::NativeExpLogKind::log2     ? "log2" :
                                                                  "log10";
    return builder.CreateCall(
        function, {vector}, std::string{"native."} + operation);
}

[[nodiscard]] ::llvm::Value *emit_pow_f32(
    ::llvm::Module &module, ::llvm::IRBuilder<> &builder,
    ::llvm::Value *base, ::llvm::Value *exponent,
    LLVMNativeMathMode mode) {
    auto *type = f32_vector_type(base);
    if (type == nullptr || exponent == nullptr ||
        exponent->getType() != type) {
        return nullptr;
    }
    auto width = type->getNumElements();
    auto suffix = mode == LLVMNativeMathMode::fast ? "fast" : "u10";
    auto name = std::string{"__luisa_cpu_native_pow_f32_v"} +
                std::to_string(width) + "_" + suffix;
    auto *function = module.getFunction(name);
    if (function == nullptr) {
        auto *function_type = ::llvm::FunctionType::get(
            type, {type, type}, false);
        function = ::llvm::Function::Create(
            function_type, ::llvm::GlobalValue::InternalLinkage,
            name, module);
        configure_native_function(function, false);
        auto *fast_log = mode == LLVMNativeMathMode::fast ?
                             get_or_create_exp_log_f32(
                                 module, type, mode,
                                 detail::NativeExpLogKind::log) :
                             nullptr;
        auto *fast_exp = mode == LLVMNativeMathMode::fast ?
                             get_or_create_exp_log_f32(
                                 module, type, mode,
                                 detail::NativeExpLogKind::exp) :
                             nullptr;
        detail::build_pow_f32(
            module, function, width, mode,
            fast_log, fast_exp);
    }
    return builder.CreateCall(
        function, {base, exponent}, "native.pow");
}

[[nodiscard]] ::llvm::Value *emit_hyperbolic_f32(
    ::llvm::Module &module, ::llvm::IRBuilder<> &builder,
    ::llvm::Value *vector, LLVMNativeMathMode mode,
    detail::NativeHyperbolicKind kind) {
    auto *type = f32_vector_type(vector);
    if (type == nullptr) { return nullptr; }
    auto operation = kind == detail::NativeHyperbolicKind::sinh  ? "sinh" :
                     kind == detail::NativeHyperbolicKind::cosh  ? "cosh" :
                     kind == detail::NativeHyperbolicKind::tanh  ? "tanh" :
                     kind == detail::NativeHyperbolicKind::asinh ? "asinh" :
                     kind == detail::NativeHyperbolicKind::acosh ? "acosh" :
                                                                   "atanh";
    auto width = type->getNumElements();
    auto suffix = mode == LLVMNativeMathMode::fast ?
                      "fast" :
                      "precise";
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
        auto exponential =
            kind == detail::NativeHyperbolicKind::sinh ||
            kind == detail::NativeHyperbolicKind::cosh ||
            kind == detail::NativeHyperbolicKind::tanh;
        auto *exp_half_function = exponential ?
                                      get_or_create_exp_log_f32(
                                          module, type, mode,
                                          detail::NativeExpLogKind::exp_half) :
                                      nullptr;
        auto *log_function = exponential ?
                                 nullptr :
                                 get_or_create_exp_log_f32(
                                     module, type, mode,
                                     detail::NativeExpLogKind::log);
        detail::build_hyperbolic_f32(
            module, function, width, mode, kind,
            exp_half_function, log_function);
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

::llvm::Value *LLVMNativeMath::emit_atan2_f32(
    ::llvm::Module &module, ::llvm::IRBuilder<> &builder,
    ::llvm::Value *y, ::llvm::Value *x,
    LLVMNativeMathMode mode) {
    return cpu::emit_atan2_f32(
        module, builder, y, x, mode);
}

::llvm::Value *LLVMNativeMath::emit_exp_f32(
    ::llvm::Module &module, ::llvm::IRBuilder<> &builder,
    ::llvm::Value *vector, LLVMNativeMathMode mode) {
    return emit_exp_log_f32(
        module, builder, vector, mode,
        detail::NativeExpLogKind::exp);
}

::llvm::Value *LLVMNativeMath::emit_exp2_f32(
    ::llvm::Module &module, ::llvm::IRBuilder<> &builder,
    ::llvm::Value *vector, LLVMNativeMathMode mode) {
    return emit_exp_log_f32(
        module, builder, vector, mode,
        detail::NativeExpLogKind::exp2);
}

::llvm::Value *LLVMNativeMath::emit_exp10_f32(
    ::llvm::Module &module, ::llvm::IRBuilder<> &builder,
    ::llvm::Value *vector, LLVMNativeMathMode mode) {
    return emit_exp_log_f32(
        module, builder, vector, mode,
        detail::NativeExpLogKind::exp10);
}

::llvm::Value *LLVMNativeMath::emit_log_f32(
    ::llvm::Module &module, ::llvm::IRBuilder<> &builder,
    ::llvm::Value *vector, LLVMNativeMathMode mode) {
    return emit_exp_log_f32(
        module, builder, vector, mode,
        detail::NativeExpLogKind::log);
}

::llvm::Value *LLVMNativeMath::emit_log2_f32(
    ::llvm::Module &module, ::llvm::IRBuilder<> &builder,
    ::llvm::Value *vector, LLVMNativeMathMode mode) {
    return emit_exp_log_f32(
        module, builder, vector, mode,
        detail::NativeExpLogKind::log2);
}

::llvm::Value *LLVMNativeMath::emit_log10_f32(
    ::llvm::Module &module, ::llvm::IRBuilder<> &builder,
    ::llvm::Value *vector, LLVMNativeMathMode mode) {
    return emit_exp_log_f32(
        module, builder, vector, mode,
        detail::NativeExpLogKind::log10);
}

::llvm::Value *LLVMNativeMath::emit_pow_f32(
    ::llvm::Module &module, ::llvm::IRBuilder<> &builder,
    ::llvm::Value *base, ::llvm::Value *exponent,
    LLVMNativeMathMode mode) {
    return cpu::emit_pow_f32(
        module, builder, base, exponent, mode);
}

::llvm::Value *LLVMNativeMath::emit_sinh_f32(
    ::llvm::Module &module, ::llvm::IRBuilder<> &builder,
    ::llvm::Value *vector, LLVMNativeMathMode mode) {
    return emit_hyperbolic_f32(
        module, builder, vector, mode,
        detail::NativeHyperbolicKind::sinh);
}

::llvm::Value *LLVMNativeMath::emit_cosh_f32(
    ::llvm::Module &module, ::llvm::IRBuilder<> &builder,
    ::llvm::Value *vector, LLVMNativeMathMode mode) {
    return emit_hyperbolic_f32(
        module, builder, vector, mode,
        detail::NativeHyperbolicKind::cosh);
}

::llvm::Value *LLVMNativeMath::emit_tanh_f32(
    ::llvm::Module &module, ::llvm::IRBuilder<> &builder,
    ::llvm::Value *vector, LLVMNativeMathMode mode) {
    return emit_hyperbolic_f32(
        module, builder, vector, mode,
        detail::NativeHyperbolicKind::tanh);
}

::llvm::Value *LLVMNativeMath::emit_asinh_f32(
    ::llvm::Module &module, ::llvm::IRBuilder<> &builder,
    ::llvm::Value *vector, LLVMNativeMathMode mode) {
    return emit_hyperbolic_f32(
        module, builder, vector, mode,
        detail::NativeHyperbolicKind::asinh);
}

::llvm::Value *LLVMNativeMath::emit_acosh_f32(
    ::llvm::Module &module, ::llvm::IRBuilder<> &builder,
    ::llvm::Value *vector, LLVMNativeMathMode mode) {
    return emit_hyperbolic_f32(
        module, builder, vector, mode,
        detail::NativeHyperbolicKind::acosh);
}

::llvm::Value *LLVMNativeMath::emit_atanh_f32(
    ::llvm::Module &module, ::llvm::IRBuilder<> &builder,
    ::llvm::Value *vector, LLVMNativeMathMode mode) {
    return emit_hyperbolic_f32(
        module, builder, vector, mode,
        detail::NativeHyperbolicKind::atanh);
}

}// namespace luisa::compute::cpu
