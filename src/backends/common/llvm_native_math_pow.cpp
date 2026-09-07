#include "llvm_native_math.h"
#include "llvm_native_math_internal.h"

// The precise magnitude path adapts SLEEF's logkf/expkf float-pair
// construction (Copyright Naoki Shibata and contributors 2010-2025,
// Boost Software License 1.0). See LICENSE.SLEEF.txt. Exceptional-value and
// domain repair are kept explicit so neither tier relies on LLVM fast-math
// assumptions or target-specific vector-library semantics.

namespace luisa::compute::cpu::detail {

namespace {

class PowF32IRBuilder {

private:
    struct FloatPair {
        ::llvm::Value *high;
        ::llvm::Value *low;
    };

private:
    ::llvm::Module &_module;
    ::llvm::IRBuilder<> _builder;
    ::llvm::FixedVectorType *_float_vector;
    ::llvm::FixedVectorType *_int_vector;
    LLVMNativeMathMode _mode;
    ::llvm::Function *_fast_log_function;
    ::llvm::Function *_fast_exp_function;

private:
    [[nodiscard]] ::llvm::Constant *_f32(double value) const {
        auto *scalar = ::llvm::ConstantFP::get(
            ::llvm::Type::getFloatTy(_module.getContext()), value);
        return ::llvm::ConstantVector::getSplat(
            _float_vector->getElementCount(), scalar);
    }

    [[nodiscard]] ::llvm::Constant *_i32(uint32_t value) const {
        auto *scalar = ::llvm::ConstantInt::get(
            ::llvm::Type::getInt32Ty(_module.getContext()), value);
        return ::llvm::ConstantVector::getSplat(
            _int_vector->getElementCount(), scalar);
    }

    [[nodiscard]] ::llvm::Value *_mla(
        ::llvm::Value *x, ::llvm::Value *y, ::llvm::Value *z) {
        return _builder.CreateFAdd(
            _builder.CreateFMul(x, y), z);
    }

    [[nodiscard]] ::llvm::Value *_float_bits(::llvm::Value *value) {
        return _builder.CreateBitCast(value, _int_vector);
    }

    [[nodiscard]] ::llvm::Value *_bits_float(::llvm::Value *value) {
        return _builder.CreateBitCast(value, _float_vector);
    }

    [[nodiscard]] ::llvm::Value *_upper(::llvm::Value *value) {
        return _bits_float(_builder.CreateAnd(
            _float_bits(value), _i32(0xfffff000u)));
    }

    [[nodiscard]] ::llvm::Value *_abs(::llvm::Value *value) {
        return _bits_float(_builder.CreateAnd(
            _float_bits(value), _i32(0x7fffffffu)));
    }

    [[nodiscard]] ::llvm::Value *_quiet_nan() {
        return _bits_float(_i32(0x7fc00000u));
    }

    [[nodiscard]] ::llvm::Value *_positive_infinity() {
        return _bits_float(_i32(0x7f800000u));
    }

    [[nodiscard]] ::llvm::Value *_round_nearest(::llvm::Value *value) {
        auto *sign = _builder.CreateAnd(
            _float_bits(value), _i32(0x80000000u));
        auto *bias = _bits_float(_builder.CreateOr(
            _i32(0x4b000000u), sign));
        return _builder.CreateFSub(
            _builder.CreateFAdd(value, bias), bias);
    }

    [[nodiscard]] ::llvm::Value *_pow2i(::llvm::Value *exponent) {
        return _bits_float(_builder.CreateShl(
            _builder.CreateAdd(exponent, _i32(127u)), _i32(23u)));
    }

    [[nodiscard]] ::llvm::Value *_ldexp2(
        ::llvm::Value *value, ::llvm::Value *exponent) {
        auto *half = _builder.CreateAShr(exponent, _i32(1u));
        value = _builder.CreateFMul(value, _pow2i(half));
        return _builder.CreateFMul(
            value,
            _pow2i(_builder.CreateSub(exponent, half)));
    }

    [[nodiscard]] FloatPair _normalize(FloatPair value) {
        auto *sum = _builder.CreateFAdd(value.high, value.low);
        return {
            sum,
            _builder.CreateFAdd(
                _builder.CreateFSub(value.high, sum), value.low)};
    }

    [[nodiscard]] FloatPair _scale(
        FloatPair value, ::llvm::Value *scale) {
        return {
            _builder.CreateFMul(value.high, scale),
            _builder.CreateFMul(value.low, scale)};
    }

    [[nodiscard]] FloatPair _add2(
        ::llvm::Value *x, ::llvm::Value *y) {
        auto *sum = _builder.CreateFAdd(x, y);
        auto *v = _builder.CreateFSub(sum, x);
        auto *error = _builder.CreateFAdd(
            _builder.CreateFSub(
                x, _builder.CreateFSub(sum, v)),
            _builder.CreateFSub(y, v));
        return {sum, error};
    }

    [[nodiscard]] FloatPair _add2(
        FloatPair x, ::llvm::Value *y) {
        auto result = _add2(x.high, y);
        result.low = _builder.CreateFAdd(result.low, x.low);
        return result;
    }

    [[nodiscard]] FloatPair _add(
        FloatPair x, FloatPair y) {
        auto *sum = _builder.CreateFAdd(x.high, y.high);
        auto *error = _builder.CreateFAdd(
            _builder.CreateFAdd(
                _builder.CreateFSub(x.high, sum), y.high),
            _builder.CreateFAdd(x.low, y.low));
        return {sum, error};
    }

    [[nodiscard]] FloatPair _add2(
        FloatPair x, FloatPair y) {
        auto *sum = _builder.CreateFAdd(x.high, y.high);
        auto *v = _builder.CreateFSub(sum, x.high);
        auto *error = _builder.CreateFAdd(
            _builder.CreateFAdd(
                _builder.CreateFSub(
                    x.high, _builder.CreateFSub(sum, v)),
                _builder.CreateFSub(y.high, v)),
            _builder.CreateFAdd(x.low, y.low));
        return {sum, error};
    }

    [[nodiscard]] FloatPair _multiply(
        ::llvm::Value *x, ::llvm::Value *y) {
        auto *x_high = _upper(x);
        auto *x_low = _builder.CreateFSub(x, x_high);
        auto *y_high = _upper(y);
        auto *y_low = _builder.CreateFSub(y, y_high);
        auto *product = _builder.CreateFMul(x, y);
        auto *error = _mla(
            x_high, y_high, _builder.CreateFNeg(product));
        error = _mla(x_low, y_high, error);
        error = _mla(x_high, y_low, error);
        error = _mla(x_low, y_low, error);
        return {product, error};
    }

    [[nodiscard]] FloatPair _multiply(
        FloatPair x, ::llvm::Value *y) {
        auto result = _multiply(x.high, y);
        result.low = _mla(x.low, y, result.low);
        return result;
    }

    [[nodiscard]] FloatPair _multiply(
        FloatPair x, FloatPair y) {
        auto result = _multiply(x.high, y.high);
        result.low = _mla(x.high, y.low, result.low);
        result.low = _mla(x.low, y.high, result.low);
        return result;
    }

    [[nodiscard]] FloatPair _divide(
        FloatPair numerator, FloatPair denominator) {
        auto *reciprocal = _builder.CreateFDiv(
            _f32(1.0), denominator.high);
        auto *denominator_high = _upper(denominator.high);
        auto *denominator_low = _builder.CreateFSub(
            denominator.high, denominator_high);
        auto *reciprocal_high = _upper(reciprocal);
        auto *reciprocal_low = _builder.CreateFSub(
            reciprocal, reciprocal_high);
        auto *numerator_high = _upper(numerator.high);
        auto *numerator_low = _builder.CreateFSub(
            numerator.high, numerator_high);
        auto *quotient = _builder.CreateFMul(
            numerator.high, reciprocal);

        ::llvm::Value *denominator_error = _f32(-1.0);
        denominator_error = _mla(
            denominator_high, reciprocal_high, denominator_error);
        denominator_error = _mla(
            denominator_high, reciprocal_low, denominator_error);
        denominator_error = _mla(
            denominator_low, reciprocal_high, denominator_error);
        denominator_error = _mla(
            denominator_low, reciprocal_low, denominator_error);
        denominator_error = _builder.CreateFNeg(denominator_error);

        auto *numerator_error = _mla(
            numerator_high, reciprocal_high,
            _builder.CreateFNeg(quotient));
        numerator_error = _mla(
            numerator_high, reciprocal_low, numerator_error);
        numerator_error = _mla(
            numerator_low, reciprocal_high, numerator_error);
        numerator_error = _mla(
            numerator_low, reciprocal_low, numerator_error);
        numerator_error = _mla(
            quotient, denominator_error, numerator_error);

        auto *correction = _builder.CreateFSub(
            numerator.low,
            _builder.CreateFMul(quotient, denominator.low));
        correction = _mla(reciprocal, correction, numerator_error);
        return {quotient, correction};
    }

    [[nodiscard]] FloatPair _precise_log(
        ::llvm::Value *input) {
        auto *subnormal = _builder.CreateFCmpOLT(
            input, _f32(1.175494350822287508e-38));
        auto *scaled = _builder.CreateSelect(
            subnormal,
            _builder.CreateFMul(
                input, _f32(18446744073709551616.0)),
            input);
        auto *probe = _builder.CreateFMul(
            scaled, _f32(1.3333333333333333333));
        auto *exponent = _builder.CreateSub(
            _builder.CreateAnd(
                _builder.CreateLShr(_float_bits(probe), _i32(23u)),
                _i32(0xffu)),
            _i32(127u));
        auto *mantissa = _bits_float(_builder.CreateAdd(
            _float_bits(scaled),
            _builder.CreateShl(
                _builder.CreateNeg(exponent), _i32(23u))));
        exponent = _builder.CreateSelect(
            subnormal,
            _builder.CreateSub(exponent, _i32(64u)), exponent);

        auto x = _divide(
            _add2(_f32(-1.0), mantissa),
            _add2(_f32(1.0), mantissa));
        auto x2 = _multiply(x, x);

        ::llvm::Value *polynomial =
            _f32(0.240320354700088500976562);
        polynomial = _mla(
            polynomial, x2.high,
            _f32(0.285112679004669189453125));
        polynomial = _mla(
            polynomial, x2.high,
            _f32(0.400007992982864379882812));
        auto coefficient = FloatPair{
            _f32(0.66666662693023681640625),
            _f32(3.69183861259614332084311e-09)};
        auto *exponent_float = _builder.CreateSIToFP(
            exponent, _float_vector);
        auto result = _multiply(
            FloatPair{
                _f32(0.69314718246459960938),
                _f32(-1.904654323148236017e-09)},
            exponent_float);
        result = _add(result, _scale(x, _f32(2.0)));
        auto tail = _multiply(
            _multiply(x2, x),
            _add2(_multiply(x2, polynomial), coefficient));
        return _add(result, tail);
    }

    [[nodiscard]] ::llvm::Value *_precise_exp(FloatPair input) {
        auto *sum = _builder.CreateFAdd(input.high, input.low);
        // A finite float-pair product can overflow its leading component to
        // infinity while the Dekker residual becomes NaN through inf - inf.
        // Classify that lane from the leading component before neutralizing
        // the range-reduction operands.
        auto *classification = _builder.CreateSelect(
            _builder.CreateFCmpORD(sum, sum), sum, input.high);
        auto *in_range = _builder.CreateAnd(
            _builder.CreateFCmpOGE(classification, _f32(-104.0)),
            _builder.CreateFCmpOLE(classification, _f32(100.0)));
        input = {
            _builder.CreateSelect(in_range, input.high, _f32(0.0)),
            _builder.CreateSelect(in_range, input.low, _f32(0.0))};
        auto *safe_sum = _builder.CreateFAdd(input.high, input.low);
        auto *q_float = _round_nearest(_builder.CreateFMul(
            safe_sum, _f32(1.4426950408889634074)));
        auto *q = _builder.CreateFPToSI(q_float, _int_vector);
        auto reduced = _add2(
            input,
            _builder.CreateFMul(
                q_float, _f32(-0.693145751953125)));
        reduced = _add2(
            reduced,
            _builder.CreateFMul(
                q_float, _f32(-1.428606765330187045e-6)));
        reduced = _normalize(reduced);

        ::llvm::Value *polynomial =
            _f32(0.00136324646882712841033936);
        polynomial = _mla(
            polynomial, reduced.high,
            _f32(0.00836596917361021041870117));
        polynomial = _mla(
            polynomial, reduced.high,
            _f32(0.0416710823774337768554688));
        polynomial = _mla(
            polynomial, reduced.high,
            _f32(0.166665524244308471679688));
        polynomial = _mla(
            polynomial, reduced.high,
            _f32(0.499999850988388061523438));

        auto result = _add(
            reduced,
            _multiply(_multiply(reduced, reduced), polynomial));
        result = _add2(
            FloatPair{_f32(1.0), _f32(0.0)}, result);
        auto *value = _ldexp2(
            _builder.CreateFAdd(result.high, result.low), q);
        value = _builder.CreateSelect(
            _builder.CreateFCmpOLT(classification, _f32(-104.0)),
            _f32(0.0), value);
        return _builder.CreateSelect(
            _builder.CreateFCmpOGT(classification, _f32(100.0)),
            _positive_infinity(), value);
    }

    [[nodiscard]] ::llvm::Value *_precise_magnitude(
        ::llvm::Value *base, ::llvm::Value *exponent) {
        auto logarithm = _precise_log(base);
        return _precise_exp(_multiply(logarithm, exponent));
    }

    [[nodiscard]] ::llvm::Value *_fast_magnitude(
        ::llvm::Value *base, ::llvm::Value *exponent) {
        auto *logarithm = _builder.CreateCall(
            _fast_log_function, {base}, "pow.fast.log");
        auto *product = _builder.CreateFMul(
            logarithm, exponent, "pow.fast.exponent");
        return _builder.CreateCall(
            _fast_exp_function, {product}, "pow.fast.exp");
    }

public:
    PowF32IRBuilder(
        ::llvm::Module &module, ::llvm::Function *function,
        uint32_t width, LLVMNativeMathMode mode,
        ::llvm::Function *fast_log_function,
        ::llvm::Function *fast_exp_function)
        : _module{module},
          _builder{module.getContext()},
          _float_vector{::llvm::FixedVectorType::get(
              ::llvm::Type::getFloatTy(module.getContext()), width)},
          _int_vector{::llvm::FixedVectorType::get(
              ::llvm::Type::getInt32Ty(module.getContext()), width)},
          _mode{mode},
          _fast_log_function{fast_log_function},
          _fast_exp_function{fast_exp_function} {
        auto *entry = ::llvm::BasicBlock::Create(
            module.getContext(), "entry", function);
        _builder.SetInsertPoint(entry);
        if (mode == LLVMNativeMathMode::fast) {
            ::llvm::FastMathFlags flags;
            flags.setAllowContract(true);
            _builder.setFastMathFlags(flags);
        }
    }

    void build(::llvm::Function *function) {
        auto *base = function->getArg(0u);
        auto *exponent = function->getArg(1u);
        base->setName("base");
        exponent->setName("exponent");

        auto *base_bits = _float_bits(base);
        auto *absolute_base_bits = _builder.CreateAnd(
            base_bits, _i32(0x7fffffffu));
        auto *absolute_base = _bits_float(absolute_base_bits);
        auto *exponent_bits = _float_bits(exponent);
        auto *absolute_exponent_bits = _builder.CreateAnd(
            exponent_bits, _i32(0x7fffffffu));
        auto *absolute_exponent = _bits_float(
            absolute_exponent_bits);

        auto *base_is_zero = _builder.CreateICmpEQ(
            absolute_base_bits, _i32(0u));
        auto *base_is_infinite = _builder.CreateICmpEQ(
            absolute_base_bits, _i32(0x7f800000u));
        auto *base_is_nan = _builder.CreateICmpUGT(
            absolute_base_bits, _i32(0x7f800000u));
        auto *base_is_finite = _builder.CreateICmpULT(
            absolute_base_bits, _i32(0x7f800000u));
        auto *exponent_is_zero = _builder.CreateICmpEQ(
            absolute_exponent_bits, _i32(0u));
        auto *exponent_is_infinite = _builder.CreateICmpEQ(
            absolute_exponent_bits, _i32(0x7f800000u));
        auto *exponent_is_nan = _builder.CreateICmpUGT(
            absolute_exponent_bits, _i32(0x7f800000u));
        auto *exponent_is_finite = _builder.CreateICmpULT(
            absolute_exponent_bits, _i32(0x7f800000u));
        auto *base_is_negative = _builder.CreateICmpNE(
            _builder.CreateAnd(base_bits, _i32(0x80000000u)),
            _i32(0u));
        auto *exponent_is_negative = _builder.CreateICmpNE(
            _builder.CreateAnd(exponent_bits, _i32(0x80000000u)),
            _i32(0u));

        // FP-to-int is only defined for representable finite operands. Every
        // other lane is neutralized before conversion; f32 values with
        // magnitude >= 2^24 are integral and necessarily even.
        auto *small_exponent = _builder.CreateAnd(
            exponent_is_finite,
            _builder.CreateFCmpOLT(
                absolute_exponent, _f32(16777216.0)));
        auto *safe_integer_exponent = _builder.CreateSelect(
            small_exponent, exponent, _f32(0.0));
        auto *truncated_exponent = _builder.CreateUnaryIntrinsic(
            ::llvm::Intrinsic::trunc, safe_integer_exponent);
        auto *large_integral_exponent = _builder.CreateAnd(
            exponent_is_finite,
            _builder.CreateFCmpOGE(
                absolute_exponent, _f32(16777216.0)));
        auto *exponent_is_integer = _builder.CreateOr(
            large_integral_exponent,
            _builder.CreateAnd(
                small_exponent,
                _builder.CreateFCmpOEQ(
                    truncated_exponent, exponent)));
        auto *integer_exponent = _builder.CreateFPToSI(
            safe_integer_exponent, _int_vector);
        auto *exponent_is_odd = _builder.CreateAnd(
            exponent_is_integer,
            _builder.CreateICmpNE(
                _builder.CreateAnd(integer_exponent, _i32(1u)),
                _i32(0u)));

        auto *core_lane = _builder.CreateAnd(
            exponent_is_finite,
            _builder.CreateAnd(
                base_is_finite,
                _builder.CreateNot(base_is_zero)));
        auto *safe_base = _builder.CreateSelect(
            core_lane, absolute_base, _f32(1.0));
        auto *safe_exponent = _builder.CreateSelect(
            core_lane, exponent, _f32(0.0));
        auto *result = _mode == LLVMNativeMathMode::fast ?
                           _fast_magnitude(safe_base, safe_exponent) :
                           _precise_magnitude(safe_base, safe_exponent);

        // Infinite exponents depend only on |base| relative to one.
        auto *absolute_base_is_one = _builder.CreateFCmpOEQ(
            absolute_base, _f32(1.0));
        auto *infinite_exponent_grows = _builder.CreateXor(
            _builder.CreateFCmpOGT(absolute_base, _f32(1.0)),
            exponent_is_negative);
        auto *infinite_exponent_result = _builder.CreateSelect(
            absolute_base_is_one, _f32(1.0),
            _builder.CreateSelect(
                infinite_exponent_grows,
                _positive_infinity(), _f32(0.0)));
        result = _builder.CreateSelect(
            exponent_is_infinite,
            infinite_exponent_result, result);

        // Zero and infinity bases are reciprocal cases across the sign of y.
        auto *boundary_is_infinite = _builder.CreateXor(
            base_is_infinite, exponent_is_negative);
        auto *boundary_result = _builder.CreateSelect(
            boundary_is_infinite,
            _positive_infinity(), _f32(0.0));
        result = _builder.CreateSelect(
            _builder.CreateOr(base_is_zero, base_is_infinite),
            boundary_result, result);

        // A negative sign is observable only for an odd integral exponent.
        auto *negative_result = _builder.CreateAnd(
            base_is_negative, exponent_is_odd);
        result = _bits_float(_builder.CreateOr(
            _float_bits(result),
            _builder.CreateSelect(
                negative_result, _i32(0x80000000u), _i32(0u))));

        auto *negative_finite_nonzero_base = _builder.CreateAnd(
            base_is_negative,
            _builder.CreateAnd(
                base_is_finite, _builder.CreateNot(base_is_zero)));
        auto *domain_error = _builder.CreateAnd(
            negative_finite_nonzero_base,
            _builder.CreateAnd(
                exponent_is_finite,
                _builder.CreateNot(exponent_is_integer)));
        result = _builder.CreateSelect(
            domain_error, _quiet_nan(), result);
        result = _builder.CreateSelect(
            _builder.CreateFCmpOEQ(exponent, _f32(1.0)),
            base, result);
        result = _builder.CreateSelect(
            _builder.CreateOr(base_is_nan, exponent_is_nan),
            _quiet_nan(), result);

        // C/IEEE identities override NaN propagation for pow(x, +-0) and
        // pow(+1, y). The -1 raised to either infinity case was repaired by
        // the |base| == 1 branch above.
        auto *base_is_positive_one = _builder.CreateFCmpOEQ(
            base, _f32(1.0));
        result = _builder.CreateSelect(
            _builder.CreateOr(
                exponent_is_zero, base_is_positive_one),
            _f32(1.0), result);
        _builder.CreateRet(result);
    }
};

}// namespace

void build_pow_f32(
    ::llvm::Module &module, ::llvm::Function *function, uint32_t width,
    LLVMNativeMathMode mode,
    ::llvm::Function *fast_log_function,
    ::llvm::Function *fast_exp_function) {
    PowF32IRBuilder{
        module, function, width, mode,
        fast_log_function, fast_exp_function}
        .build(function);
}

}// namespace luisa::compute::cpu::detail
