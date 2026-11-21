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
        if constexpr (requires { options.NoSignedZerosFPMath = true; }) { options.NoSignedZerosFPMath = true; }
    }
    luisa::compute::detail::apply_llvm_fp_target_options(
        options, enable_fast_math ? luisa::compute::detail::LLVMFPOpFusion::Fast :
                                     luisa::compute::detail::LLVMFPOpFusion::Standard);
}

inline void apply_fallback_math_function_options(::llvm::Function &function, bool enable_fast_math) noexcept {
    // Remove generic baseline target attributes stamped by offline wrapper bitcode compilation so
    // TargetMachine's actual CPU/subtarget features and TTI inline-compatibility apply uniformly.
    for (auto attr_name : {"target-cpu", "target-features", "tune-cpu"}) {
        function.removeFnAttr(attr_name);
    }
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
    // Never retain `no-infs-fp-math` / `no-nans-fp-math` even in fast-math
    // mode: native math overflow (+inf) and shader guards legitimately produce
    // inf/NaN and must not fold to poison.
    function.removeFnAttr("no-infs-fp-math");
    function.removeFnAttr("no-nans-fp-math");
    if (!enable_fast_math) {
        // Imported builtins may carry these legacy fast-math attributes.
        // Clearing instruction FMF alone does not restore precise semantics.
        for (auto name : {"unsafe-fp-math", "no-signed-zeros-fp-math", "approx-func-fp-math"}) {
            function.removeFnAttr(name);
        }
    }
    ::llvm::FastMathFlags fast_math_flags;
    if (enable_fast_math) {
        // Enable fast reciprocal, contraction, approximate intrinsics,
        // reassociation, and signed-zero elision without `nnan`/`ninf` poison.
        // Native math helpers (e.g. exp overflow -> +inf) and shader guards
        // (e.g. 1 / exp(x) -> 0, 1/0, 0/0) legitimately produce inf/NaN;
        // stamping `nnan`/`ninf` lets LLVM fold overflow selects to poison
        // and silently replace exp(x > 88.72) with exp(0) = 1.0.
        fast_math_flags.setAllowReciprocal(true);
        fast_math_flags.setAllowContract(true);
        fast_math_flags.setApproxFunc(true);
        fast_math_flags.setAllowReassoc(true);
        fast_math_flags.setNoSignedZeros(true);
    }
    for (auto &block : function) {
        for (auto &instruction : block) {
            if (::llvm::isa<::llvm::FPMathOperator>(instruction)) {
                if (enable_fast_math &&
                    instruction.getOpcode() != ::llvm::Instruction::FRem &&
                    !::llvm::isa<::llvm::FCmpInst>(instruction)) {
                    instruction.copyFastMathFlags(fast_math_flags);
                } else {
                    instruction.copyFastMathFlags(::llvm::FastMathFlags{});
                }
            }
        }
    }
}

}// namespace luisa::compute::fallback
