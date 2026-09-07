#include "llvm_native_math.h"
#include "llvm_native_math_internal.h"

#include <array>

// The precise quadrant reduction and polynomial are adapted from SLEEF's
// single-precision xatan2f implementation (Copyright Naoki Shibata and
// contributors 2010-2025, Boost Software License 1.0). See
// LICENSE.SLEEF.txt. The fast body uses a locally derived degree-11 odd
// minimax polynomial and the same audited IEEE repair.

namespace luisa::compute::cpu::detail {

namespace {

class Atan2F32IRBuilder {

private:
    ::llvm::Module &_module;
    ::llvm::IRBuilder<> _builder;
    ::llvm::FixedVectorType *_float_vector;
    ::llvm::FixedVectorType *_int_vector;
    LLVMNativeMathMode _mode;

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

    [[nodiscard]] ::llvm::Value *_float_bits(::llvm::Value *value) {
        return _builder.CreateBitCast(value, _int_vector);
    }

    [[nodiscard]] ::llvm::Value *_bits_float(::llvm::Value *value) {
        return _builder.CreateBitCast(value, _float_vector);
    }

    [[nodiscard]] ::llvm::Value *_abs(::llvm::Value *value) {
        return _bits_float(_builder.CreateAnd(
            _float_bits(value), _i32(0x7fffffffu)));
    }

    [[nodiscard]] ::llvm::Value *_sign_bit(::llvm::Value *value) {
        return _builder.CreateICmpNE(
            _builder.CreateAnd(
                _float_bits(value), _i32(0x80000000u)),
            _i32(0u));
    }

    [[nodiscard]] ::llvm::Value *_copy_sign(
        ::llvm::Value *magnitude, ::llvm::Value *sign) {
        auto *magnitude_bits = _builder.CreateAnd(
            _float_bits(magnitude), _i32(0x7fffffffu));
        auto *sign_bits = _builder.CreateAnd(
            _float_bits(sign), _i32(0x80000000u));
        return _bits_float(_builder.CreateOr(
            magnitude_bits, sign_bits));
    }

    [[nodiscard]] ::llvm::Value *_is_zero(::llvm::Value *value) {
        return _builder.CreateICmpEQ(
            _builder.CreateAnd(
                _float_bits(value), _i32(0x7fffffffu)),
            _i32(0u));
    }

    [[nodiscard]] ::llvm::Value *_is_inf(::llvm::Value *value) {
        return _builder.CreateICmpEQ(
            _builder.CreateAnd(
                _float_bits(value), _i32(0x7fffffffu)),
            _i32(0x7f800000u));
    }

    [[nodiscard]] ::llvm::Value *_is_nan(::llvm::Value *value) {
        return _builder.CreateFCmpUNO(value, value);
    }

    [[nodiscard]] ::llvm::Value *_mla(
        ::llvm::Value *x, ::llvm::Value *y, ::llvm::Value *z) {
        return _builder.CreateFAdd(
            _builder.CreateFMul(x, y), z);
    }

    [[nodiscard]] ::llvm::Value *_invalid_ratio(
        ::llvm::Value *y, ::llvm::Value *x) {
        auto *both_zero = _builder.CreateAnd(
            _is_zero(y), _is_zero(x));
        auto *both_inf = _builder.CreateAnd(
            _is_inf(y), _is_inf(x));
        auto *any_nan = _builder.CreateOr(
            _is_nan(y), _is_nan(x));
        return _builder.CreateOr(
            _builder.CreateOr(both_zero, both_inf), any_nan);
    }

    [[nodiscard]] ::llvm::Value *_finish(
        ::llvm::Value *y, ::llvm::Value *x,
        ::llvm::Value *result) {
        auto *x_negative = _sign_bit(x);
        auto *x_zero = _is_zero(x);
        auto *x_inf = _is_inf(x);
        auto *y_zero = _is_zero(y);
        auto *y_inf = _is_inf(y);

        auto *zero_or_pi = _builder.CreateSelect(
            x_negative, _f32(3.1415926535897932385), _f32(0.0));
        auto *x_special = _builder.CreateSelect(
            x_zero, _f32(1.5707963267948966192), zero_or_pi);
        result = _builder.CreateSelect(
            _builder.CreateOr(x_zero, x_inf), x_special, result);

        auto *both_inf = _builder.CreateAnd(y_inf, x_inf);
        auto *infinite_y = _builder.CreateSelect(
            both_inf,
            _builder.CreateSelect(
                x_negative,
                _f32(2.3561944901923449288),
                _f32(0.78539816339744830962)),
            _f32(1.5707963267948966192));
        result = _builder.CreateSelect(y_inf, infinite_y, result);
        result = _builder.CreateSelect(y_zero, zero_or_pi, result);
        result = _copy_sign(result, y);
        return _builder.CreateSelect(
            _builder.CreateOr(_is_nan(y), _is_nan(x)),
            _bits_float(_i32(0x7fc00000u)), result);
    }

    [[nodiscard]] ::llvm::Value *_precise(
        ::llvm::Value *y, ::llvm::Value *x) {
        auto *absolute_y = _abs(y);
        auto *absolute_x = _abs(x);
        auto *swap = _builder.CreateFCmpOLT(
            absolute_x, absolute_y);
        auto *x_negative = _sign_bit(x);
        auto *q = _builder.CreateSelect(
            x_negative, _f32(-2.0), _f32(0.0));
        q = _builder.CreateSelect(
            swap, _builder.CreateFAdd(q, _f32(1.0)), q);
        auto *numerator = _builder.CreateSelect(
            swap, _builder.CreateFNeg(absolute_x), absolute_y);
        auto *denominator = _builder.CreateSelect(
            swap, absolute_y, absolute_x);
        auto *invalid = _invalid_ratio(y, x);
        numerator = _builder.CreateSelect(
            invalid, _f32(0.0), numerator);
        denominator = _builder.CreateSelect(
            invalid, _f32(1.0), denominator);
        auto *s = _builder.CreateFDiv(numerator, denominator);
        auto *s2 = _builder.CreateFMul(s, s);
        constexpr std::array coefficients{
            0.0028236389625817537308,
            -0.015956902876496315002,
            0.042504988610744476318,
            -0.074890092015266418457,
            0.10634793341159820557,
            -0.14202736318111419678,
            0.19992695748805999756,
            -0.33333101868629455566,
        };
        ::llvm::Value *polynomial = _f32(coefficients.front());
        for (auto i = size_t{1u}; i < coefficients.size(); i++) {
            polynomial = _mla(
                polynomial, s2, _f32(coefficients[i]));
        }
        auto *result = _mla(
            s, _builder.CreateFMul(s2, polynomial), s);
        result = _mla(
            q, _f32(1.5707963267948966192), result);
        result = _builder.CreateSelect(
            x_negative, _builder.CreateFNeg(result), result);
        return _finish(y, x, result);
    }

    [[nodiscard]] ::llvm::Value *_fast(
        ::llvm::Value *y, ::llvm::Value *x) {
        auto *absolute_y = _abs(y);
        auto *absolute_x = _abs(x);
        auto *swap = _builder.CreateFCmpOLT(
            absolute_x, absolute_y);
        auto *numerator = _builder.CreateSelect(
            swap, absolute_x, absolute_y);
        auto *denominator = _builder.CreateSelect(
            swap, absolute_y, absolute_x);
        auto *invalid = _invalid_ratio(y, x);
        numerator = _builder.CreateSelect(
            invalid, _f32(0.0), numerator);
        denominator = _builder.CreateSelect(
            invalid, _f32(1.0), denominator);
        auto *ratio = _builder.CreateFDiv(numerator, denominator);
        auto *ratio2 = _builder.CreateFMul(ratio, ratio);
        // Remez-derived odd minimax polynomial on [-1, 1]. A dense
        // double-precision audit bounds approximation error by 1.663e-6;
        // f32 Horner evaluation stays below 1.756e-6 before quadrant repair.
        ::llvm::Value *polynomial = _f32(-0.011719135567545891);
        polynomial = _mla(
            polynomial, ratio2, _f32(0.05264735221862793));
        polynomial = _mla(
            polynomial, ratio2, _f32(-0.11642648279666901));
        polynomial = _mla(
            polynomial, ratio2, _f32(0.19354037940502167));
        polynomial = _mla(
            polynomial, ratio2, _f32(-0.33262282609939575));
        polynomial = _mla(
            polynomial, ratio2, _f32(0.9999772310256958));
        auto *angle = _builder.CreateFMul(ratio, polynomial);
        auto *magnitude = _builder.CreateSelect(
            swap,
            _builder.CreateFSub(
                _f32(1.5707963267948966192), angle),
            angle);
        magnitude = _builder.CreateSelect(
            _sign_bit(x),
            _builder.CreateFSub(
                _f32(3.1415926535897932385), magnitude),
            magnitude);
        return _finish(y, x, magnitude);
    }

public:
    Atan2F32IRBuilder(
        ::llvm::Module &module, ::llvm::Function *function,
        uint32_t width, LLVMNativeMathMode mode)
        : _module{module},
          _builder{module.getContext()},
          _float_vector{::llvm::FixedVectorType::get(
              ::llvm::Type::getFloatTy(module.getContext()), width)},
          _int_vector{::llvm::FixedVectorType::get(
              ::llvm::Type::getInt32Ty(module.getContext()), width)},
          _mode{mode} {
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
        auto *y = function->getArg(0u);
        auto *x = function->getArg(1u);
        y->setName("y");
        x->setName("x");
        _builder.CreateRet(
            _mode == LLVMNativeMathMode::fast ?
                _fast(y, x) :
                _precise(y, x));
    }
};

}// namespace

void build_atan2_f32(
    ::llvm::Module &module, ::llvm::Function *function, uint32_t width,
    LLVMNativeMathMode mode) {
    Atan2F32IRBuilder{module, function, width, mode}.build(function);
}

}// namespace luisa::compute::cpu::detail
