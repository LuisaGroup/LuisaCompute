#include "llvm_native_math_internal.h"

// The common-domain reduction is local to the fast tier. The sin/cos and
// lower tan polynomial coefficients are retained from the audited precise
// SLEEF adaptation (Boost Software License 1.0); see LICENSE.SLEEF.txt and
// LLVM_NATIVE_MATH_PROVENANCE.md.

namespace luisa::compute::cpu::detail {

namespace {

class FastTrigF32IRBuilder final : public FastF32IRBuilder {

private:
    NativeTrigKind _kind;
    ::llvm::Function *_precise_function;

private:
    [[nodiscard]] ::llvm::Value *_sin_cos_polynomial(
        ::llvm::Value *reduced) {
        auto &b = builder();
        auto *square = b.CreateFMul(reduced, reduced);
        ::llvm::Value *polynomial = f32(2.6083159809786593542e-6);
        polynomial = mla(
            polynomial, square, f32(-0.00019810690719168633223));
        polynomial = mla(
            polynomial, square, f32(0.0083330785855650901794));
        polynomial = mla(
            polynomial, square, f32(-0.16666659712791442871));
        return b.CreateFAdd(
            b.CreateFMul(square, b.CreateFMul(polynomial, reduced)),
            reduced);
    }

    [[nodiscard]] ::llvm::Value *_tan_polynomial(
        ::llvm::Value *reduced, ::llvm::Value *quadrant) {
        auto &b = builder();
        auto *odd = b.CreateICmpEQ(
            b.CreateAnd(quadrant, i32(1u)), i32(1u));
        auto *flip_bits = b.CreateSelect(
            odd, i32(0x80000000u), i32(0u));
        auto *x = bits_float(b.CreateXor(
            float_bits(reduced), flip_bits));
        auto *square = b.CreateFMul(x, x);

        // SLEEF-derived low-order coefficients retained from the audited
        // precise polynomial. The least-significant high-order term is
        // omitted; the complete fast-domain error is checked independently.
        ::llvm::Value *polynomial = f32(0.0033198499586433172);
        polynomial = mla(
            polynomial, square, f32(0.024299807846546173));
        polynomial = mla(
            polynomial, square, f32(0.053449530154466629));
        polynomial = mla(
            polynomial, square, f32(0.13338300585746765));
        polynomial = mla(
            polynomial, square, f32(0.33333185315132141));
        auto *result = mla(
            square, b.CreateFMul(polynomial, x), x);
        return b.CreateSelect(
            odd, b.CreateFDiv(f32(1.0), result), result);
    }

    [[nodiscard]] ::llvm::Value *_build_common(
        ::llvm::Value *input, ::llvm::Value *common) {
        auto &b = builder();
        auto *safe = b.CreateSelect(common, input, f32(0.0));
        ::llvm::Value *q_float;
        if (_kind == NativeTrigKind::cos) {
            q_float = b.CreateFAdd(
                b.CreateFMul(
                    round_nearest(b.CreateFAdd(
                        b.CreateFMul(safe, f32(0.31830988618379067154)),
                        f32(-0.5))),
                    f32(2.0)),
                f32(1.0));
        } else {
            q_float = round_nearest(b.CreateFMul(
                safe, f32(_kind == NativeTrigKind::tan ?
                              0.63661977236758134308 :
                              0.31830988618379067154)));
        }
        auto *quadrant = b.CreateFPToSI(q_float, int_vector());
        auto *q_as_float = b.CreateSIToFP(quadrant, float_vector());
        auto half_period = _kind == NativeTrigKind::sin ?
                               3.1415926535897932385 :
                               1.5707963267948966192;
        auto period_hi = _kind == NativeTrigKind::sin ?
                             3.140625 :
                             1.5703125;
        // Keep period_hi coarse enough that q * period_hi is exact over the
        // common domain. The two residual terms recover the discarded bits
        // without relying on target-specific extended precision.
        auto period_tail = _kind == NativeTrigKind::sin ?
                               5.126565838509123e-12 :
                               2.5632829192545614e-12;
        auto *reduced = mla(f32(-period_hi), q_as_float, safe);
        reduced = mla(
            f32(-(half_period - period_hi)), q_as_float, reduced);
        reduced = mla(f32(-period_tail), q_as_float, reduced);

        ::llvm::Value *result;
        if (_kind == NativeTrigKind::tan) {
            result = _tan_polynomial(reduced, quadrant);
        } else {
            auto flip_mask = _kind == NativeTrigKind::cos ? 2u : 1u;
            auto flip_value = _kind == NativeTrigKind::cos ? 0u : 1u;
            auto *flip = b.CreateICmpEQ(
                b.CreateAnd(quadrant, i32(flip_mask)),
                i32(flip_value));
            reduced = bits_float(b.CreateXor(
                float_bits(reduced), b.CreateSelect(
                                         flip, i32(0x80000000u),
                                         i32(0u))));
            result = _sin_cos_polynomial(reduced);
        }

        if (_kind != NativeTrigKind::cos) {
            auto *tiny = is_subnormal(input);
            auto *negative_zero = b.CreateICmpEQ(
                float_bits(input), i32(0x80000000u));
            result = b.CreateSelect(
                b.CreateOr(tiny, negative_zero), input, result);
        }
        return result;
    }

public:
    FastTrigF32IRBuilder(
        ::llvm::Module &module, ::llvm::Function *function,
        uint32_t width, NativeTrigKind kind,
        ::llvm::Function *precise_function)
        : FastF32IRBuilder{module, function, width},
          _kind{kind},
          _precise_function{precise_function} {}

    void build(::llvm::Function *function) {
        auto &b = builder();
        auto *input = function->getArg(0u);
        input->setName("x");

        // Cody-Waite reduction is used only where a three-term f32 split
        // retains the declared bound. Larger finite inputs, infinities, and
        // NaNs take the precise SLEEF-derived reduction path, including its
        // Payne-Hanek handling for large finite arguments.
        auto *common = b.CreateFCmpOLT(abs(input), f32(128.0));
        auto *common_result = _build_common(input, common);
        auto *all_common = b.CreateAndReduce(common);
        auto *common_block = b.GetInsertBlock();
        auto *slow = ::llvm::BasicBlock::Create(
            module().getContext(), "large.reduce", function);
        auto *merge = ::llvm::BasicBlock::Create(
            module().getContext(), "result", function);
        b.CreateCondBr(all_common, merge, slow);

        b.SetInsertPoint(slow);
        auto *precise = b.CreateCall(_precise_function, {input});
        auto *mixed_result = b.CreateSelect(
            common, common_result, precise);
        auto *slow_block = b.GetInsertBlock();
        b.CreateBr(merge);

        b.SetInsertPoint(merge);
        auto *result = b.CreatePHI(float_vector(), 2u, "fast.result");
        result->addIncoming(common_result, common_block);
        result->addIncoming(mixed_result, slow_block);
        b.CreateRet(result);
    }
};

}// namespace

void build_fast_trig_f32(
    ::llvm::Module &module, ::llvm::Function *function, uint32_t width,
    NativeTrigKind kind, ::llvm::Function *precise_function) {
    FastTrigF32IRBuilder{
        module, function, width, kind, precise_function}
        .build(function);
}

}// namespace luisa::compute::cpu::detail
