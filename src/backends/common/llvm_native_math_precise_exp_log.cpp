#include "llvm_native_math_internal.h"

// The precise exp/log approximations are adapted from SLEEF's single-precision
// implementations (Copyright Naoki Shibata and contributors 2010-2025,
// Boost Software License 1.0). See LICENSE.SLEEF.txt.

namespace luisa::compute::cpu {

namespace {

class ExpLogF32IRBuilder {

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

    [[nodiscard]] ::llvm::Value *_round_nearest(::llvm::Value *x) {
        auto *sign = _builder.CreateAnd(
            _float_bits(x), _i32(0x80000000u));
        auto *bias = _bits_float(_builder.CreateOr(
            _i32(0x4b000000u), sign));
        return _builder.CreateFSub(
            _builder.CreateFAdd(x, bias), bias);
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

    [[nodiscard]] ::llvm::Value *_exp_reduced(
        ::llvm::Value *reduced) {
        ::llvm::Value *polynomial =
            _f32(0.00019852761761285364628);
        polynomial = _mla(
            polynomial, reduced,
            _f32(0.0013930435525253415108));
        polynomial = _mla(
            polynomial, reduced,
            _f32(0.0083333607763051986694));
        polynomial = _mla(
            polynomial, reduced,
            _f32(0.041666485369205474854));
        polynomial = _mla(
            polynomial, reduced,
            _f32(0.16666667163372039795));
        polynomial = _mla(polynomial, reduced, _f32(0.5));
        return _builder.CreateFAdd(
            _f32(1.0),
            _builder.CreateFAdd(
                reduced,
                _builder.CreateFMul(
                    _builder.CreateFMul(reduced, reduced),
                    polynomial)));
    }

    [[nodiscard]] ::llvm::Value *_finish_exp(
        ::llvm::Value *input, ::llvm::Value *result,
        double lower, double upper, bool upper_inclusive) {
        result = _builder.CreateSelect(
            _builder.CreateFCmpOLT(input, _f32(lower)),
            _f32(0.0), result);
        auto *overflow = upper_inclusive ?
                             _builder.CreateFCmpOGE(input, _f32(upper)) :
                             _builder.CreateFCmpOGT(input, _f32(upper));
        result = _builder.CreateSelect(
            overflow, _bits_float(_i32(0x7f800000u)), result);
        return _builder.CreateSelect(
            _builder.CreateFCmpUNO(input, input),
            _bits_float(_i32(0x7fc00000u)), result);
    }

    void _build_exp(::llvm::Function *function) {
        auto *input = function->getArg(0u);
        input->setName("x");
        auto *safe_range = _builder.CreateAnd(
            _builder.CreateFCmpOGE(input, _f32(-104.0)),
            _builder.CreateFCmpOLE(input, _f32(100.0)));
        auto *safe = _builder.CreateSelect(
            safe_range, input, _f32(0.0));
        auto *q_float = _round_nearest(_builder.CreateFMul(
            safe, _f32(1.4426950408889634074)));
        auto *q = _builder.CreateFPToSI(q_float, _int_vector);
        auto *q_as_float = _builder.CreateSIToFP(q, _float_vector);
        auto *reduced = _mla(
            q_as_float, _f32(-0.693145751953125), safe);
        reduced = _mla(
            q_as_float, _f32(-1.428606765330187045e-6), reduced);

        auto *result = _exp_reduced(reduced);
        result = _ldexp2(result, q);
        result = _finish_exp(
            input, result, -104.0, 100.0, false);
        _builder.CreateRet(result);
    }

    void _build_exp2(::llvm::Function *function) {
        auto *input = function->getArg(0u);
        input->setName("x");
        auto *safe_range = _builder.CreateAnd(
            _builder.CreateFCmpOGE(input, _f32(-150.0)),
            _builder.CreateFCmpOLE(input, _f32(128.0)));
        auto *safe = _builder.CreateSelect(
            safe_range, input, _f32(0.0));
        auto *q_float = _round_nearest(safe);
        auto *q = _builder.CreateFPToSI(q_float, _int_vector);
        auto *reduced = _builder.CreateFSub(safe, q_float);
        auto *natural_reduced = _builder.CreateFMul(
            reduced, _f32(0.693145751953125));
        natural_reduced = _mla(
            reduced, _f32(1.428606765330187045e-6),
            natural_reduced);

        auto *result = _ldexp2(
            _exp_reduced(natural_reduced), q);
        result = _finish_exp(
            input, result, -150.0, 128.0, true);
        _builder.CreateRet(result);
    }

    void _build_exp10(::llvm::Function *function) {
        auto *input = function->getArg(0u);
        input->setName("x");
        auto *safe_range = _builder.CreateAnd(
            _builder.CreateFCmpOGE(input, _f32(-50.0)),
            _builder.CreateFCmpOLE(
                input, _f32(38.531839419103623894)));
        auto *safe = _builder.CreateSelect(
            safe_range, input, _f32(0.0));
        auto *q_float = _round_nearest(_builder.CreateFMul(
            safe, _f32(3.3219280948873623479)));
        auto *q = _builder.CreateFPToSI(q_float, _int_vector);
        auto *reduced = _mla(
            q_float, _f32(-0.301025390625), safe);
        reduced = _mla(
            q_float, _f32(-4.605038981e-6), reduced);
        auto *natural_reduced = _builder.CreateFMul(
            reduced, _f32(2.3025851249694824219));
        natural_reduced = _mla(
            reduced, _f32(-3.1705172516493593157e-8),
            natural_reduced);

        auto *result = _ldexp2(
            _exp_reduced(natural_reduced), q);
        result = _finish_exp(
            input, result, -50.0,
            38.531839419103623894, false);
        _builder.CreateRet(result);
    }

    struct LogReduction {
        ::llvm::Value *x;
        ::llvm::Value *x2;
        ::llvm::Value *exponent;
    };

    [[nodiscard]] LogReduction _reduce_log(
        ::llvm::Value *input) {
        auto *subnormal = _builder.CreateAnd(
            _builder.CreateFCmpOGT(input, _f32(0.0)),
            _builder.CreateFCmpOLT(
                input, _f32(1.175494350822287508e-38)));
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
        auto *log_exponent = _builder.CreateSelect(
            subnormal,
            _builder.CreateSub(exponent, _i32(64u)), exponent);

        auto *x = _builder.CreateFDiv(
            _builder.CreateFSub(mantissa, _f32(1.0)),
            _builder.CreateFAdd(_f32(1.0), mantissa));
        auto *x2 = _builder.CreateFMul(x, x);
        return {
            x,
            x2,
            _builder.CreateSIToFP(log_exponent, _float_vector)};
    }

    [[nodiscard]] ::llvm::Value *_finish_log(
        ::llvm::Value *input, ::llvm::Value *result) {
        result = _builder.CreateSelect(
            _builder.CreateFCmpOEQ(input, _f32(0.0)),
            _bits_float(_i32(0xff800000u)), result);
        result = _builder.CreateSelect(
            _builder.CreateFCmpOEQ(
                input, _bits_float(_i32(0x7f800000u))),
            _bits_float(_i32(0x7f800000u)), result);
        result = _builder.CreateSelect(
            _builder.CreateOr(
                _builder.CreateFCmpOLT(input, _f32(0.0)),
                _builder.CreateFCmpUNO(input, input)),
            _bits_float(_i32(0x7fc00000u)), result);
        return result;
    }

    void _build_log(::llvm::Function *function,
                    detail::NativeExpLogKind kind) {
        auto *input = function->getArg(0u);
        input->setName("x");
        auto reduction = _reduce_log(input);
        ::llvm::Value *result = nullptr;
        switch (kind) {
            case detail::NativeExpLogKind::log: {
                ::llvm::Value *polynomial =
                    _f32(0.23928284645080566406);
                polynomial = _mla(
                    polynomial, reduction.x2,
                    _f32(0.28518211841583251953));
                polynomial = _mla(
                    polynomial, reduction.x2,
                    _f32(0.40000587701797485352));
                polynomial = _mla(
                    polynomial, reduction.x2,
                    _f32(0.6666666865348815918));
                polynomial = _mla(
                    polynomial, reduction.x2, _f32(2.0));
                result = _mla(
                    reduction.exponent,
                    _f32(0.69314718055994528623),
                    _builder.CreateFMul(
                        reduction.x, polynomial));
                break;
            }
            case detail::NativeExpLogKind::log2: {
                // SLEEF's u35 direct log2 polynomial. Keeping the exponent
                // in base two avoids the accuracy loss of ln(x) * log2(e).
                ::llvm::Value *polynomial = _f32(0.4374088347);
                polynomial = _mla(
                    polynomial, reduction.x2,
                    _f32(0.5764843822));
                polynomial = _mla(
                    polynomial, reduction.x2,
                    _f32(0.9618024230));
                result = _mla(
                    reduction.x, _f32(2.885390043),
                    reduction.exponent);
                result = _mla(
                    _builder.CreateFMul(
                        reduction.x2, reduction.x),
                    polynomial, result);
                break;
            }
            case detail::NativeExpLogKind::log10: {
                // Direct base-10 form of the same atanh reduction, using
                // the audited SLEEF single-precision coefficients.
                ::llvm::Value *polynomial = _f32(0.1314289868);
                polynomial = _mla(
                    polynomial, reduction.x2,
                    _f32(0.1735493541));
                polynomial = _mla(
                    polynomial, reduction.x2,
                    _f32(0.2895309627));
                result = _mla(
                    reduction.x, _f32(0.868588984),
                    _builder.CreateFMul(
                        reduction.exponent, _f32(0.30103001)));
                result = _mla(
                    _builder.CreateFMul(
                        reduction.x2, reduction.x),
                    polynomial, result);
                break;
            }
            default: break;
        }
        result = _finish_log(input, result);
        _builder.CreateRet(result);
    }

public:
    ExpLogF32IRBuilder(
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

    void build(::llvm::Function *function,
               detail::NativeExpLogKind kind) {
        switch (kind) {
            case detail::NativeExpLogKind::exp:
                _build_exp(function);
                break;
            case detail::NativeExpLogKind::exp2:
                _build_exp2(function);
                break;
            case detail::NativeExpLogKind::exp10:
                _build_exp10(function);
                break;
            case detail::NativeExpLogKind::log:
            case detail::NativeExpLogKind::log2:
            case detail::NativeExpLogKind::log10:
                _build_log(function, kind);
                break;
        }
    }
};

}// namespace

namespace detail {

void build_precise_exp_log_f32(
    ::llvm::Module &module, ::llvm::Function *function, uint32_t width,
    NativeExpLogKind kind) {
    ExpLogF32IRBuilder{module, function, width}.build(
        function, kind);
}

}// namespace detail

}// namespace luisa::compute::cpu
