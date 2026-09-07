#pragma once

#include <llvm/Config/llvm-config.h>
#include <llvm/Target/TargetOptions.h>

namespace luisa::compute::fallback {

inline void apply_fallback_math_target_options(
    ::llvm::TargetOptions &options, bool enable_fast_math) noexcept {
    if (enable_fast_math) {
#if LLVM_VERSION_MAJOR <= 21
        options.UnsafeFPMath = true;
        options.ApproxFuncFPMath = true;
#endif
        options.NoInfsFPMath = true;
        options.NoNaNsFPMath = true;
        options.NoSignedZerosFPMath = true;
    }
    options.NoTrappingFPMath = true;
    options.AllowFPOpFusion = enable_fast_math ?
        ::llvm::FPOpFusion::Fast : ::llvm::FPOpFusion::Standard;
}

}// namespace luisa::compute::fallback
