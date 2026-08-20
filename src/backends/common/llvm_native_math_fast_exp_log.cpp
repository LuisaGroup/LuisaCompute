#include "llvm_native_math_internal.h"

namespace luisa::compute::cpu::detail {

namespace {

class FastExpLogF32IRBuilder final : public FastF32IRBuilder {

private:
    [[nodiscard]] ::llvm::Value *_pow2i(::llvm::Value *exponent) {
        return bits_float(builder().CreateShl(
            builder().CreateAdd(exponent, i32(127u)), i32(23u)));
    }

    [[nodiscard]] ::llvm::Value *_exp_reduced(
        ::llvm::Value *reduced) {
        auto &b = builder();
        // exp(r) Maclaurin polynomial through r^4. Nearest-integer range
        // reduction guarantees |r| <= ln(2)/2.
        auto *polynomial = mla(
            f32(1.0 / 24.0), reduced, f32(1.0 / 6.0));
        polynomial = mla(polynomial, reduced, f32(0.5));
        return b.CreateFAdd(
            f32(1.0),
            b.CreateFAdd(
                reduced,
                b.CreateFMul(
                    b.CreateFMul(reduced, reduced), polynomial)));
    }

    [[nodiscard]] ::llvm::Value *_scale_exp(
        ::llvm::Value *result, ::llvm::Value *q) {
        auto &b = builder();
        // q can be 128 at the top of each finite output range. Build 2^127
        // first so representable near-maximum results do not overflow early.
        auto *q_is_128 = b.CreateICmpSGT(q, i32(127u));
        auto *scale_exponent = b.CreateSelect(
            q_is_128, i32(127u), q);
        result = b.CreateFMul(result, _pow2i(scale_exponent));
        return b.CreateSelect(
            q_is_128, b.CreateFMul(result, f32(2.0)), result);
    }

    [[nodiscard]] ::llvm::Value *_finish_exp(
        ::llvm::Value *input, ::llvm::Value *result,
        double lower, double upper, bool upper_inclusive) {
        auto &b = builder();
        result = b.CreateSelect(
            is_subnormal(result), f32(0.0), result);
        result = b.CreateSelect(
            b.CreateFCmpOLT(input, f32(lower)),
            f32(0.0), result);
        auto *overflow = upper_inclusive ?
                             b.CreateFCmpOGE(input, f32(upper)) :
                             b.CreateFCmpOGT(input, f32(upper));
        result = b.CreateSelect(
            overflow, positive_infinity(), result);
        return b.CreateSelect(is_nan(input), quiet_nan(), result);
    }

    void _build_exp(::llvm::Function *function, bool half_scale) {
        auto &b = builder();
        auto *input = function->getArg(0u);
        input->setName("x");

        // The fast tier deliberately flushes results below the least normal
        // f32 value. Inputs are neutralized before FP-to-int conversion.
        auto *lower = half_scale ?
                          f32(-86.643397569993164) :
                          f32(-87.336544750553108986);
        auto *upper = half_scale ?
                          f32(89.4159862326283) :
                          f32(88.722839052068353053);
        auto *in_range = b.CreateAnd(
            b.CreateFCmpOGE(
                input, lower),
            b.CreateFCmpOLE(
                input, upper));
        auto *safe = b.CreateSelect(in_range, input, f32(0.0));
        auto *q_float = round_nearest(b.CreateFMul(
            safe, f32(1.4426950408889634074)));
        auto *q = b.CreateFPToSI(q_float, int_vector());
        auto *q_as_float = b.CreateSIToFP(q, float_vector());
        auto *reduced = mla(
            q_as_float, f32(-0.69314718055994530942), safe);

        auto *scale_exponent = half_scale ?
                                   b.CreateSub(q, i32(1u)) :
                                   q;
        auto *result = _scale_exp(
            _exp_reduced(reduced), scale_exponent);
        result = _finish_exp(
            input, result,
            half_scale ? -86.643397569993164 :
                         -87.336544750553108986,
            half_scale ? 89.4159862326283 :
                         88.722839052068353053,
            false);
        b.CreateRet(result);
    }

    void _build_exp2(::llvm::Function *function) {
        auto &b = builder();
        auto *input = function->getArg(0u);
        input->setName("x");
        auto *in_range = b.CreateAnd(
            b.CreateFCmpOGE(input, f32(-126.0)),
            b.CreateFCmpOLE(input, f32(128.0)));
        auto *safe = b.CreateSelect(in_range, input, f32(0.0));
        auto *q_float = round_nearest(safe);
        auto *q = b.CreateFPToSI(q_float, int_vector());
        auto *reduced = b.CreateFMul(
            b.CreateFSub(safe, q_float),
            f32(0.69314718055994530942));
        auto *result = _scale_exp(_exp_reduced(reduced), q);
        result = _finish_exp(
            input, result, -126.0, 128.0, true);
        b.CreateRet(result);
    }

    void _build_exp10(::llvm::Function *function) {
        auto &b = builder();
        auto *input = function->getArg(0u);
        input->setName("x");
        auto *in_range = b.CreateAnd(
            b.CreateFCmpOGE(
                input, f32(-37.929779453661631102)),
            b.CreateFCmpOLE(
                input, f32(38.531839419103623894)));
        auto *safe = b.CreateSelect(in_range, input, f32(0.0));
        auto *q_float = round_nearest(b.CreateFMul(
            safe, f32(3.3219280948873623479)));
        auto *q = b.CreateFPToSI(q_float, int_vector());
        auto *reduced = mla(
            q_float, f32(-0.30102999566398119521), safe);
        reduced = b.CreateFMul(
            reduced, f32(2.3025850929940456840));
        auto *result = _scale_exp(_exp_reduced(reduced), q);
        result = _finish_exp(
            input, result,
            -37.929779453661631102,
            38.531839419103623894, false);
        b.CreateRet(result);
    }

    struct LogReduction {
        ::llvm::Value *mantissa_log;
        ::llvm::Value *exponent;
    };

    [[nodiscard]] LogReduction _reduce_log(
        ::llvm::Value *input) {
        auto &b = builder();
        auto *bits = float_bits(input);
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
        return {
            b.CreateFMul(f32(2.0), b.CreateFMul(x, polynomial)),
            b.CreateSIToFP(exponent, float_vector())};
    }

    [[nodiscard]] ::llvm::Value *_finish_log(
        ::llvm::Value *input, ::llvm::Value *result) {
        auto &b = builder();
        auto *absolute_bits = b.CreateAnd(
            float_bits(input), i32(0x7fffffffu));
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
        return result;
    }

    void _build_log(::llvm::Function *function,
                    NativeExpLogKind kind) {
        auto &b = builder();
        auto *input = function->getArg(0u);
        input->setName("x");
        auto reduction = _reduce_log(input);
        ::llvm::Value *result = nullptr;
        switch (kind) {
            case NativeExpLogKind::log:
                result = mla(
                    reduction.exponent,
                    f32(0.69314718055994530942),
                    reduction.mantissa_log);
                break;
            case NativeExpLogKind::log2:
                result = mla(
                    reduction.mantissa_log,
                    f32(1.4426950408889634074),
                    reduction.exponent);
                break;
            case NativeExpLogKind::log10:
                result = b.CreateFAdd(
                    b.CreateFMul(
                        reduction.exponent,
                        f32(0.30102999566398119521)),
                    b.CreateFMul(
                        reduction.mantissa_log,
                        f32(0.43429448190325182765)));
                break;
            default: break;
        }
        result = _finish_log(input, result);
        b.CreateRet(result);
    }

public:
    FastExpLogF32IRBuilder(
        ::llvm::Module &module, ::llvm::Function *function,
        uint32_t width)
        : FastF32IRBuilder{module, function, width} {}

    void build(::llvm::Function *function, NativeExpLogKind kind) {
        switch (kind) {
            case NativeExpLogKind::exp:
                _build_exp(function, false);
                break;
            case NativeExpLogKind::exp_half:
                _build_exp(function, true);
                break;
            case NativeExpLogKind::exp2:
                _build_exp2(function);
                break;
            case NativeExpLogKind::exp10:
                _build_exp10(function);
                break;
            case NativeExpLogKind::log:
            case NativeExpLogKind::log2:
            case NativeExpLogKind::log10:
                _build_log(function, kind);
                break;
        }
    }
};

}// namespace

void build_fast_exp_log_f32(
    ::llvm::Module &module, ::llvm::Function *function, uint32_t width,
    NativeExpLogKind kind) {
    FastExpLogF32IRBuilder{module, function, width}.build(
        function, kind);
}

}// namespace luisa::compute::cpu::detail
