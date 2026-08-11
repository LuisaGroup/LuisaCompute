#include "llvm_native_math_internal.h"

namespace luisa::compute::cpu::detail {

namespace {

class FastExpLogF32IRBuilder final : public FastF32IRBuilder {

private:
    [[nodiscard]] ::llvm::Value *_pow2i(::llvm::Value *exponent) {
        return bits_float(builder().CreateShl(
            builder().CreateAdd(exponent, i32(127u)), i32(23u)));
    }

    void _build_exp(::llvm::Function *function) {
        auto &b = builder();
        auto *input = function->getArg(0u);
        input->setName("x");

        // The fast tier deliberately flushes results below the least normal
        // f32 value. Inputs are neutralized before FP-to-int conversion.
        auto *in_range = b.CreateAnd(
            b.CreateFCmpOGE(
                input, f32(-87.336544750553108986)),
            b.CreateFCmpOLE(
                input, f32(88.722839052068353053)));
        auto *safe = b.CreateSelect(in_range, input, f32(0.0));
        auto *q_float = round_nearest(b.CreateFMul(
            safe, f32(1.4426950408889634074)));
        auto *q = b.CreateFPToSI(q_float, int_vector());
        auto *q_as_float = b.CreateSIToFP(q, float_vector());
        auto *reduced = mla(
            q_as_float, f32(-0.69314718055994530942), safe);

        // exp(r) Maclaurin polynomial through r^4. Nearest-integer range
        // reduction guarantees |r| <= ln(2)/2.
        auto *polynomial = mla(
            f32(1.0 / 24.0), reduced, f32(1.0 / 6.0));
        polynomial = mla(polynomial, reduced, f32(0.5));
        auto *result = b.CreateFAdd(
            f32(1.0),
            b.CreateFAdd(
                reduced,
                b.CreateFMul(
                    b.CreateFMul(reduced, reduced), polynomial)));

        // q can be 128 only at the top of the finite f32 exp range. Build
        // 2^127 first and apply the final factor after the polynomial so a
        // representable near-maximum result is not prematurely overflowed.
        auto *q_is_128 = b.CreateICmpSGT(q, i32(127u));
        auto *scale_exponent = b.CreateSelect(
            q_is_128, i32(127u), q);
        result = b.CreateFMul(result, _pow2i(scale_exponent));
        result = b.CreateSelect(
            q_is_128, b.CreateFMul(result, f32(2.0)), result);
        result = b.CreateSelect(
            is_subnormal(result), f32(0.0), result);

        result = b.CreateSelect(
            b.CreateFCmpOLT(
                input, f32(-87.336544750553108986)),
            f32(0.0), result);
        result = b.CreateSelect(
            b.CreateFCmpOGT(
                input, f32(88.722839052068353053)),
            positive_infinity(), result);
        result = b.CreateSelect(is_nan(input), quiet_nan(), result);
        b.CreateRet(result);
    }

    void _build_log(::llvm::Function *function) {
        auto &b = builder();
        auto *input = function->getArg(0u);
        input->setName("x");
        auto *bits = float_bits(input);
        auto *absolute_bits = b.CreateAnd(bits, i32(0x7fffffffu));
        auto *exponent = b.CreateSub(
            b.CreateAnd(b.CreateLShr(bits, i32(23u)), i32(0xffu)),
            i32(127u));
        auto *mantissa = bits_float(b.CreateOr(
            b.CreateAnd(bits, i32(0x007fffffu)),
            i32(0x3f800000u)));
        auto *upper = b.CreateFCmpOGT(
            mantissa, f32(1.4142135623730950488));
        mantissa = b.CreateSelect(
            upper, b.CreateFMul(mantissa, f32(0.5)), mantissa);
        exponent = b.CreateAdd(
            exponent, b.CreateSelect(upper, i32(1u), i32(0u)));

        auto *x = b.CreateFDiv(
            b.CreateFSub(mantissa, f32(1.0)),
            b.CreateFAdd(mantissa, f32(1.0)));
        auto *x2 = b.CreateFMul(x, x);
        // log(m) = 2 (x + x^3/3 + x^5/5 + ...), where
        // x=(m-1)/(m+1) and |x| <= (sqrt(2)-1)/(sqrt(2)+1).
        auto *polynomial = mla(
            f32(1.0 / 5.0), x2, f32(1.0 / 3.0));
        polynomial = mla(polynomial, x2, f32(1.0));
        auto *result = b.CreateFAdd(
            b.CreateFMul(
                f32(0.69314718055994530942),
                b.CreateSIToFP(exponent, float_vector())),
            b.CreateFMul(f32(2.0), b.CreateFMul(x, polynomial)));

        auto *zero = b.CreateICmpEQ(absolute_bits, i32(0u));
        auto *subnormal = b.CreateAnd(
            b.CreateICmpNE(absolute_bits, i32(0u)),
            b.CreateICmpULT(absolute_bits, i32(0x00800000u)));
        result = b.CreateSelect(
            b.CreateOr(zero, subnormal), negative_infinity(), result);
        result = b.CreateSelect(
            b.CreateFCmpOEQ(input, positive_infinity()),
            positive_infinity(), result);
        result = b.CreateSelect(
            b.CreateOr(
                b.CreateFCmpOLT(input, f32(0.0)), is_nan(input)),
            quiet_nan(), result);
        b.CreateRet(result);
    }

public:
    FastExpLogF32IRBuilder(
        ::llvm::Module &module, ::llvm::Function *function,
        uint32_t width)
        : FastF32IRBuilder{module, function, width} {}

    void build(::llvm::Function *function, bool logarithm) {
        if (logarithm) {
            _build_log(function);
        } else {
            _build_exp(function);
        }
    }
};

}// namespace

void build_fast_exp_log_f32(
    ::llvm::Module &module, ::llvm::Function *function, uint32_t width,
    bool logarithm) {
    FastExpLogF32IRBuilder{module, function, width}.build(
        function, logarithm);
}

}// namespace luisa::compute::cpu::detail
