#include "hip_floating_remainder.h"

#include <luisa/core/logging.h>
#include <llvm/IR/Constants.h>
#include <llvm/IR/Intrinsics.h>
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

    // Guard non-finite dividends/divisors explicitly via v_cmp_class so Inf/NaN
    // never enter __ocml_fmod_*'s frexp exponent-stepping loop when constant
    // divisors (such as 100000.0f in procedural noise) are specialized inline.
    constexpr unsigned non_finite_mask = llvm::fcNan | llvm::fcInf;
    auto quiet_nan = llvm::ConstantFP::getQNaN(scalar_type);
    auto emit_lane = [&](llvm::Value *x, llvm::Value *y) noexcept -> llvm::Value * {
        auto x_non_finite = builder.CreateIntrinsic(
            llvm::Intrinsic::is_fpclass, {scalar_type},
            {x, builder.getInt32(non_finite_mask)});
        auto safe_x = builder.CreateSelect(
            x_non_finite, llvm::ConstantFP::getZero(scalar_type), x);
        auto remainder = builder.CreateCall(function, {safe_x, y});
        return builder.CreateSelect(x_non_finite, quiet_nan, remainder);
    };

    // Native OCML retains exponent-stepped range reduction and residual
    // correction even with fast math. Expose that body before IPO instead of
    // leaving FRem for late generic expansion, which misses the native RCP
    // implementation. Ordinary compiler profitability decides inlining.
    if (auto vector_type = llvm::dyn_cast<llvm::FixedVectorType>(type)) {
        llvm::Value *result = llvm::PoisonValue::get(type);
        for (auto lane = 0u; lane < vector_type->getNumElements(); lane++) {
            auto x = builder.CreateExtractElement(dividend, lane);
            auto y = builder.CreateExtractElement(divisor, lane);
            auto remainder = emit_lane(x, y);
            result = builder.CreateInsertElement(result, remainder, lane);
        }
        return result;
    }
    return emit_lane(dividend, divisor);
}

}// namespace luisa::compute::hip
