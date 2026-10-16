#include "llvm_native_math_internal.h"

#include <array>

// The precise inverse-trigonometric approximations are adapted from SLEEF's
// single-precision implementations (Copyright Naoki Shibata and contributors
// 2010-2025, Boost Software License 1.0). See LICENSE.SLEEF.txt.

namespace luisa::compute::cpu {

namespace {

enum struct InverseTrigKind : uint8_t {
    asin,
    acos,
    atan,
};

class InverseTrigF32IRBuilder {

private:
    ::llvm::Module &_module;
    ::llvm::IRBuilder<> _builder;
    ::llvm::FixedVectorType *_float_vector;
    ::llvm::FixedVectorType *_int_vector;

private:
    [[nodiscard]] ::llvm::Constant *_f32(double x) const {
        auto *scalar = ::llvm::ConstantFP::get(
            ::llvm::Type::getFloatTy(_module.getContext()), x);
        return ::llvm::ConstantVector::getSplat(
            _float_vector->getElementCount(), scalar);
    }

    [[nodiscard]] ::llvm::Constant *_i32(uint32_t x) const {
        auto *scalar = ::llvm::ConstantInt::get(
            ::llvm::Type::getInt32Ty(_module.getContext()), x);
        return ::llvm::ConstantVector::getSplat(
            _int_vector->getElementCount(), scalar);
    }

    [[nodiscard]] ::llvm::Value *_mla(
        ::llvm::Value *x, ::llvm::Value *y, ::llvm::Value *z) {
        return _builder.CreateFAdd(_builder.CreateFMul(x, y), z);
    }

    [[nodiscard]] ::llvm::Value *_float_bits(::llvm::Value *x) {
        return _builder.CreateBitCast(x, _int_vector);
    }

    [[nodiscard]] ::llvm::Value *_bits_float(::llvm::Value *x) {
        return _builder.CreateBitCast(x, _float_vector);
    }

    [[nodiscard]] ::llvm::Value *_abs(::llvm::Value *x) {
        return _bits_float(_builder.CreateAnd(
            _float_bits(x), _i32(0x7fffffffu)));
    }

    [[nodiscard]] ::llvm::Value *_copy_sign(
        ::llvm::Value *magnitude, ::llvm::Value *sign) {
        return _bits_float(_builder.CreateOr(
            _builder.CreateAnd(
                _float_bits(magnitude), _i32(0x7fffffffu)),
            _builder.CreateAnd(
                _float_bits(sign), _i32(0x80000000u))));
    }

    [[nodiscard]] ::llvm::Value *_asin_polynomial(
        ::llvm::Value *x2) {
        ::llvm::Value *u = _f32(0.04197454825);
        u = _mla(u, x2, _f32(0.02424046025));
        u = _mla(u, x2, _f32(0.04547423869));
        u = _mla(u, x2, _f32(0.07495029271));
        return _mla(u, x2, _f32(0.1666677296));
    }

    void _build_asin_acos(
        ::llvm::Function *function, bool cosine) {
        auto *input = function->getArg(0u);
        input->setName("x");
        auto *absolute = _abs(input);
        auto *small = _builder.CreateFCmpOLT(
            absolute, _f32(0.5));
        auto *x2 = _builder.CreateSelect(
            small, _builder.CreateFMul(input, input),
            _builder.CreateFMul(
                _builder.CreateFSub(_f32(1.0), absolute),
                _f32(0.5)));
        auto *root = _builder.CreateUnaryIntrinsic(
            ::llvm::Intrinsic::sqrt, x2);
        auto *x = _builder.CreateSelect(small, absolute, root);
        if (cosine) {
            x = _builder.CreateSelect(
                _builder.CreateFCmpOEQ(absolute, _f32(1.0)),
                _f32(0.0), x);
        }
        auto *u = _builder.CreateFMul(
            _asin_polynomial(x2),
            _builder.CreateFMul(x, x2));
        ::llvm::Value *result;
        if (cosine) {
            auto *signed_x = _copy_sign(x, input);
            auto *signed_u = _copy_sign(u, input);
            auto *middle = _builder.CreateFSub(
                _f32(1.5707963267948966192),
                _builder.CreateFAdd(signed_x, signed_u));
            auto *edge = _builder.CreateFMul(
                _builder.CreateFAdd(x, u), _f32(2.0));
            result = _builder.CreateSelect(small, middle, edge);
            result = _builder.CreateSelect(
                _builder.CreateAnd(
                    _builder.CreateNot(small),
                    _builder.CreateFCmpOLT(input, _f32(0.0))),
                _builder.CreateFSub(
                    _f32(3.1415926535897932385), result),
                result);
        } else {
            auto *approximation = _builder.CreateFAdd(x, u);
            auto *edge = _mla(
                approximation, _f32(-2.0),
                _f32(1.5707963267948966192));
            result = _builder.CreateSelect(
                small, approximation, edge);
            result = _copy_sign(result, input);
        }
        _builder.CreateRet(result);
    }

    void _build_atan(::llvm::Function *function) {
        auto *input = function->getArg(0u);
        input->setName("x");
        auto *absolute = _abs(input);
        auto *large = _builder.CreateFCmpOGT(
            absolute, _f32(1.0));
        auto *s = _builder.CreateSelect(
            large,
            _builder.CreateFDiv(_f32(1.0), absolute), absolute);
        auto *t = _builder.CreateFMul(s, s);
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
        ::llvm::Value *u = _f32(coefficients.front());
        for (auto i = size_t{1u}; i < coefficients.size(); i++) {
            u = _mla(u, t, _f32(coefficients[i]));
        }
        auto *result = _mla(
            s, _builder.CreateFMul(t, u), s);
        result = _builder.CreateSelect(
            large,
            _builder.CreateFSub(
                _f32(1.5707963267948966192), result),
            result);
        result = _copy_sign(result, input);
        _builder.CreateRet(result);
    }

public:
    InverseTrigF32IRBuilder(
        ::llvm::Module &module, ::llvm::Function *function,
        uint32_t width)
        : _module{module},
          _builder{module.getContext()},
          _float_vector{::llvm::FixedVectorType::get(
              ::llvm::Type::getFloatTy(module.getContext()), width)},
          _int_vector{::llvm::FixedVectorType::get(
              ::llvm::Type::getInt32Ty(module.getContext()), width)} {
        auto *entry = ::llvm::BasicBlock::Create(
            module.getContext(), "entry", function);
        _builder.SetInsertPoint(entry);
    }

    void build(::llvm::Function *function, InverseTrigKind kind) {
        switch (kind) {
            case InverseTrigKind::asin:
                _build_asin_acos(function, false);
                break;
            case InverseTrigKind::acos:
                _build_asin_acos(function, true);
                break;
            case InverseTrigKind::atan:
                _build_atan(function);
                break;
        }
    }
};

}// namespace

namespace detail {

void build_precise_inverse_trig_f32(
    ::llvm::Module &module, ::llvm::Function *function, uint32_t width,
    NativeInverseTrigKind kind) {
    auto precise_kind = kind == NativeInverseTrigKind::asin ?
                            InverseTrigKind::asin :
                        kind == NativeInverseTrigKind::acos ? InverseTrigKind::acos :
                                                              InverseTrigKind::atan;
    InverseTrigF32IRBuilder{module, function, width}.build(
        function, precise_kind);
}

}// namespace detail

}// namespace luisa::compute::cpu
