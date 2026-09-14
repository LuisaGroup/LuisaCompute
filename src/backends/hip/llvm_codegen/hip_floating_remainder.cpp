#include "hip_floating_remainder.h"

#include <luisa/core/logging.h>
#include <llvm/IR/Module.h>

namespace luisa::compute::hip {

llvm::Value *emit_hip_floating_remainder(
    llvm::IRBuilder<> &builder, llvm::Module &module,
    llvm::Value *dividend, llvm::Value *divisor) noexcept {
    auto type = dividend->getType();
    auto scalar_type = type->getScalarType();
    LUISA_ASSERT(type == divisor->getType() &&
                     (scalar_type->isHalfTy() || scalar_type->isFloatTy() || scalar_type->isDoubleTy()),
                 "HIP floating remainder requires matching f16/f32/f64 operands.");
    auto name = scalar_type->isHalfTy() ? "__ocml_fmod_f16" :
                scalar_type->isFloatTy() ? "__ocml_fmod_f32" : "__ocml_fmod_f64";
    auto function = module.getFunction(name);
    auto signature = llvm::FunctionType::get(
        scalar_type, {scalar_type, scalar_type}, false);
    LUISA_ASSERT(function != nullptr && function->getFunctionType() == signature,
                 "HIP native remainder '{}' is absent or has an incompatible ABI.", name);

    // Native OCML retains exponent-stepped range reduction and residual
    // correction even with fast math. Expose that body before IPO instead of
    // leaving FRem for late generic expansion, which misses the native RCP
    // implementation. Ordinary compiler profitability decides inlining.
    if (auto vector_type = llvm::dyn_cast<llvm::FixedVectorType>(type)) {
        llvm::Value *result = llvm::PoisonValue::get(type);
        for (auto lane = 0u; lane < vector_type->getNumElements(); lane++) {
            auto x = builder.CreateExtractElement(dividend, lane);
            auto y = builder.CreateExtractElement(divisor, lane);
            auto remainder = builder.CreateCall(function, {x, y});
            result = builder.CreateInsertElement(result, remainder, lane);
        }
        return result;
    }
    return builder.CreateCall(function, {dividend, divisor});
}

}// namespace luisa::compute::hip
