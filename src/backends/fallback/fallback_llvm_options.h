#pragma once

#include <llvm/ADT/FloatingPointMode.h>
#include <llvm/Config/llvm-config.h>
#include <llvm/IR/Attributes.h>
#include "../common/llvm_target_options.h"

namespace luisa::compute::fallback {

template<typename Options>
inline void apply_fallback_math_target_options(Options &options, bool enable_fast_math) noexcept {
    if (enable_fast_math) {
        if constexpr (requires { options.UnsafeFPMath = true; }) { options.UnsafeFPMath = true; }
        if constexpr (requires { options.ApproxFuncFPMath = true; }) { options.ApproxFuncFPMath = true; }
        if constexpr (requires { options.NoInfsFPMath = true; }) { options.NoInfsFPMath = true; }
        if constexpr (requires { options.NoNaNsFPMath = true; }) { options.NoNaNsFPMath = true; }
        if constexpr (requires { options.NoSignedZerosFPMath = true; }) { options.NoSignedZerosFPMath = true; }
    }
    luisa::compute::detail::apply_llvm_fp_target_options(
        options, enable_fast_math ? luisa::compute::detail::LLVMFPOpFusion::Fast :
                                     luisa::compute::detail::LLVMFPOpFusion::Standard);
}

inline void apply_fallback_math_function_options(::llvm::Function &function, bool enable_fast_math) noexcept {
    // Dispatch establishes f32 FTZ/DAZ for the complete JIT call tree.
    // Native-math helpers share that environment, but retain their own FMF.
    luisa::compute::detail::apply_llvm_fp_function_options(function, luisa::compute::detail::LLVMFPOpFusion::Standard);
    if (enable_fast_math && !function.isDeclaration()) {
#if LLVM_VERSION_MAJOR >= 23
        auto environment = ::llvm::DenormalFPEnv{
            ::llvm::DenormalMode::getDynamic(),
            ::llvm::DenormalMode::getPreserveSign()};
        ::llvm::AttrBuilder attributes{function.getContext()};
        attributes.addDenormalFPEnvAttr(environment);
        function.addFnAttrs(attributes);
#else
        function.addFnAttr("denormal-fp-math", "dynamic,dynamic");
        function.addFnAttr("denormal-fp-math-f32", "preserve-sign,preserve-sign");
#endif
    }
    if (function.hasFnAttribute("luisa.cpu.native_math")) { return; }
    if (!enable_fast_math) {
        // Imported builtins may carry these legacy fast-math attributes.
        // Clearing instruction FMF alone does not restore precise semantics.
        for (auto name : {"unsafe-fp-math", "no-infs-fp-math", "no-nans-fp-math",
                          "no-signed-zeros-fp-math", "approx-func-fp-math"}) {
            function.removeFnAttr(name);
        }
    }
    for (auto &block : function) {
        for (auto &instruction : block) {
            if (::llvm::isa<::llvm::FPMathOperator>(instruction)) {
                instruction.setFast(enable_fast_math);
            }
        }
    }
}

}// namespace luisa::compute::fallback
