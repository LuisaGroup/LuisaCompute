#pragma once

#include <llvm/IR/Function.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/IntrinsicInst.h>
#include <llvm/IR/Operator.h>
#include <llvm/Target/TargetOptions.h>

namespace luisa::compute::detail {

enum class LLVMFPOpFusion { Strict, Standard, Fast };

template<typename Options = ::llvm::TargetOptions>
inline constexpr bool llvm_has_fp_fusion_target_option = requires(Options &options) { options.AllowFPOpFusion; };

// Upstream LLVM 24 removed these fields, while ROCm LLVM 24 retains them.
// Keep enum lookup dependent as llvm::FPOpFusion itself was also removed.
template<typename Options>
inline void apply_llvm_fp_target_options(Options &options, LLVMFPOpFusion fusion) noexcept {
    if constexpr (requires { options.NoTrappingFPMath = true; }) {
        options.NoTrappingFPMath = true;
    }
    if constexpr (requires { options.AllowFPOpFusion; }) {
        using Fusion = decltype(options.AllowFPOpFusion);
        switch (fusion) {
            case LLVMFPOpFusion::Strict: options.AllowFPOpFusion = Fusion::Strict; break;
            case LLVMFPOpFusion::Standard: options.AllowFPOpFusion = Fusion::Standard; break;
            case LLVMFPOpFusion::Fast: options.AllowFPOpFusion = Fusion::Fast; break;
        }
    }
}

template<typename Options>
inline void enable_llvm_machine_function_splitting(Options &options) noexcept {
    if constexpr (requires { options.FunctionSplitting; }) {
        using Splitting = decltype(options.FunctionSplitting);
        options.FunctionSplitting = Splitting::All;
    } else {
        options.EnableMachineFunctionSplitter = true;
    }
}

// Function attributes and instruction FMF are the portable replacement for
// the removed target options. Only add contraction permission here; callers
// retain their policies for NaNs, infinities, signed zero and approximations.
inline void apply_llvm_fp_function_options(::llvm::Function &function, LLVMFPOpFusion fusion) noexcept {
    if (function.isDeclaration()) { return; }
    function.addFnAttr("no-trapping-math", "true");
    for (auto &block : function) {
        for (auto iterator = block.begin(); iterator != block.end();) {
            auto &instruction = *iterator++;
            if (fusion == LLVMFPOpFusion::Fast && ::llvm::isa<::llvm::FPMathOperator>(instruction)) {
                instruction.setHasAllowContract(true);
            } else if (fusion == LLVMFPOpFusion::Strict) {
                // The old Strict target option disabled implicit fmuladd
                // contraction, but still honored explicit contract FMF.
                // Expand that intrinsic, preserving FMF and explicit llvm.fma.
                if (auto call = ::llvm::dyn_cast<::llvm::IntrinsicInst>(&instruction);
                    call != nullptr && call->getIntrinsicID() == ::llvm::Intrinsic::fmuladd) {
                    ::llvm::IRBuilder<> builder{call};
                    builder.setFastMathFlags(call->getFastMathFlags());
                    builder.SetCurrentDebugLocation(call->getDebugLoc());
                    builder.setDefaultFPMathTag(call->getMetadata(::llvm::LLVMContext::MD_fpmath));
                    auto product = builder.CreateFMul(call->getArgOperand(0), call->getArgOperand(1));
                    auto result = builder.CreateFAdd(product, call->getArgOperand(2));
                    result->takeName(call);
                    call->replaceAllUsesWith(result);
                    call->eraseFromParent();
                }
            }
        }
    }
}

}// namespace luisa::compute::detail
