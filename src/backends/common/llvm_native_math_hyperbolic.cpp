#include "llvm_native_math.h"
#include "llvm_native_math_internal.h"

// The range splits and stable identities follow the same high-level strategy
// as SLEEF's single-precision hyperbolic functions (Copyright Naoki Shibata
// and contributors 2010-2026, Boost Software License 1.0). The local Taylor
// coefficients below are independently derived from the defining power
// series; no SLEEF polynomial coefficient is copied. See
// LLVM_NATIVE_MATH_PROVENANCE.md and LICENSE.SLEEF.txt.

namespace luisa::compute::cpu::detail {

namespace {

class HyperbolicF32IRBuilder final : public FastF32IRBuilder {

private:
    LLVMNativeMathMode _mode;
    ::llvm::Function *_exp_half_function;
    ::llvm::Function *_log_function;

private:
    [[nodiscard]] bool _precise() const noexcept {
        return _mode == LLVMNativeMathMode::precise;
    }

    [[nodiscard]] ::llvm::Value *_exp_half(::llvm::Value *value) {
        return builder().CreateCall(
            _exp_half_function, {value}, "exp.half");
    }

    [[nodiscard]] ::llvm::Value *_log(::llvm::Value *value) {
        return builder().CreateCall(
            _log_function, {value}, "log");
    }

    [[nodiscard]] ::llvm::Value *_sqrt(::llvm::Value *value) {
        return builder().CreateUnaryIntrinsic(
            ::llvm::Intrinsic::sqrt, value);
    }

    [[nodiscard]] ::llvm::Value *_sinh_polynomial(
        ::llvm::Value *x) {
        auto &b = builder();
        auto *x2 = b.CreateFMul(x, x);
        ::llvm::Value *polynomial = nullptr;
        if (_precise()) {
            polynomial = f32(1.0 / 6227020800.0);
            polynomial = mla(polynomial, x2, f32(1.0 / 39916800.0));
            polynomial = mla(polynomial, x2, f32(1.0 / 362880.0));
            polynomial = mla(polynomial, x2, f32(1.0 / 5040.0));
        } else {
            polynomial = f32(1.0 / 5040.0);
        }
        polynomial = mla(polynomial, x2, f32(1.0 / 120.0));
        polynomial = mla(polynomial, x2, f32(1.0 / 6.0));
        polynomial = mla(polynomial, x2, f32(1.0));
        return b.CreateFMul(x, polynomial);
    }

    [[nodiscard]] ::llvm::Value *_cosh_polynomial(
        ::llvm::Value *x) {
        auto &b = builder();
        auto *x2 = b.CreateFMul(x, x);
        ::llvm::Value *polynomial = nullptr;
        if (_precise()) {
            polynomial = f32(1.0 / 479001600.0);
            polynomial = mla(polynomial, x2, f32(1.0 / 3628800.0));
            polynomial = mla(polynomial, x2, f32(1.0 / 40320.0));
        } else {
            polynomial = f32(1.0 / 720.0);
        }
        if (_precise()) {
            polynomial = mla(polynomial, x2, f32(1.0 / 720.0));
        }
        polynomial = mla(polynomial, x2, f32(1.0 / 24.0));
        polynomial = mla(polynomial, x2, f32(0.5));
        polynomial = mla(polynomial, x2, f32(1.0));
        return polynomial;
    }

    [[nodiscard]] ::llvm::Value *_asinh_polynomial(
        ::llvm::Value *x) {
        auto &b = builder();
        auto *x2 = b.CreateFMul(x, x);
        ::llvm::Value *polynomial = nullptr;
        if (_precise()) {
            polynomial = f32(231.0 / 13312.0);
            polynomial = mla(polynomial, x2, f32(-63.0 / 2816.0));
        } else {
            polynomial = f32(35.0 / 1152.0);
        }
        if (_precise()) {
            polynomial = mla(polynomial, x2, f32(35.0 / 1152.0));
        }
        polynomial = mla(polynomial, x2, f32(-5.0 / 112.0));
        polynomial = mla(polynomial, x2, f32(3.0 / 40.0));
        polynomial = mla(polynomial, x2, f32(-1.0 / 6.0));
        polynomial = mla(polynomial, x2, f32(1.0));
        return b.CreateFMul(x, polynomial);
    }

    [[nodiscard]] ::llvm::Value *_acosh_polynomial(
        ::llvm::Value *delta) {
        auto &b = builder();
        ::llvm::Value *polynomial = nullptr;
        if (_precise()) {
            polynomial = f32(46189.0 / 5637144576.0);
            polynomial = mla(
                polynomial, delta, f32(-12155.0 / 637534208.0));
            polynomial = mla(
                polynomial, delta, f32(6435.0 / 142606336.0));
            polynomial = mla(
                polynomial, delta, f32(-143.0 / 1310720.0));
        } else {
            polynomial = f32(231.0 / 851968.0);
        }
        if (_precise()) {
            polynomial = mla(
                polynomial, delta, f32(231.0 / 851968.0));
        }
        polynomial = mla(
            polynomial, delta, f32(-63.0 / 90112.0));
        polynomial = mla(
            polynomial, delta, f32(35.0 / 18432.0));
        polynomial = mla(
            polynomial, delta, f32(-5.0 / 896.0));
        polynomial = mla(
            polynomial, delta, f32(3.0 / 160.0));
        polynomial = mla(
            polynomial, delta, f32(-1.0 / 12.0));
        polynomial = mla(polynomial, delta, f32(1.0));
        return b.CreateFMul(
            _sqrt(b.CreateFMul(f32(2.0), delta)), polynomial);
    }

    [[nodiscard]] ::llvm::Value *_atanh_polynomial(
        ::llvm::Value *x) {
        auto &b = builder();
        auto *x2 = b.CreateFMul(x, x);
        ::llvm::Value *polynomial = nullptr;
        if (_precise()) {
            polynomial = f32(1.0 / 21.0);
            polynomial = mla(polynomial, x2, f32(1.0 / 19.0));
            polynomial = mla(polynomial, x2, f32(1.0 / 17.0));
            polynomial = mla(polynomial, x2, f32(1.0 / 15.0));
            polynomial = mla(polynomial, x2, f32(1.0 / 13.0));
            polynomial = mla(polynomial, x2, f32(1.0 / 11.0));
        } else {
            polynomial = f32(1.0 / 9.0);
        }
        if (_precise()) {
            polynomial = mla(polynomial, x2, f32(1.0 / 9.0));
        }
        polynomial = mla(polynomial, x2, f32(1.0 / 7.0));
        polynomial = mla(polynomial, x2, f32(1.0 / 5.0));
        polynomial = mla(polynomial, x2, f32(1.0 / 3.0));
        polynomial = mla(polynomial, x2, f32(1.0));
        return b.CreateFMul(x, polynomial);
    }

    [[nodiscard]] ::llvm::Value *_finish_nan(
        ::llvm::Value *input, ::llvm::Value *result) {
        return builder().CreateSelect(
            is_nan(input), quiet_nan(), result);
    }

    [[nodiscard]] ::llvm::Value *_build_sinh_cosh(
        ::llvm::Value *input, bool cosine) {
        auto &b = builder();
        auto *magnitude = abs(input);
        auto *small_limit = f32(_precise() ? 1.0 : 0.5);
        auto *small = b.CreateFCmpOLE(magnitude, small_limit);
        auto *polynomial = cosine ?
                               _cosh_polynomial(magnitude) :
                               _sinh_polynomial(magnitude);
        auto *half_exponential = _exp_half(magnitude);
        auto *inverse_half = b.CreateFDiv(
            f32(0.25), half_exponential);
        auto *general = cosine ?
                            b.CreateFAdd(half_exponential, inverse_half) :
                            b.CreateFSub(half_exponential, inverse_half);
        auto *result = b.CreateSelect(small, polynomial, general);
        result = b.CreateSelect(
            b.CreateFCmpOGT(magnitude, f32(89.4159862326283)),
            positive_infinity(), result);
        if (!cosine) { result = copy_sign(result, input); }
        return _finish_nan(input, result);
    }

    [[nodiscard]] ::llvm::Value *_build_tanh(
        ::llvm::Value *input) {
        auto &b = builder();
        auto *magnitude = abs(input);
        auto *small_limit = f32(_precise() ? 1.0 : 0.5);
        auto *small = b.CreateFCmpOLE(magnitude, small_limit);
        auto *sinh_polynomial = _sinh_polynomial(magnitude);
        auto *cosh_polynomial = _cosh_polynomial(magnitude);
        auto *small_result = b.CreateFDiv(
            sinh_polynomial, cosh_polynomial);

        // Saturation also prevents inf/inf in lanes whose mathematical
        // result has already rounded to one.
        auto *general_range = b.CreateFCmpOLE(magnitude, f32(9.0));
        auto *safe_magnitude = b.CreateSelect(
            general_range, magnitude, f32(0.0));
        auto *half_exponential = _exp_half(safe_magnitude);
        auto *inverse_half = b.CreateFDiv(
            f32(0.25), half_exponential);
        auto *general = b.CreateFDiv(
            b.CreateFSub(half_exponential, inverse_half),
            b.CreateFAdd(half_exponential, inverse_half));
        auto *result = b.CreateSelect(
            small, small_result,
            b.CreateSelect(general_range, general, f32(1.0)));
        result = copy_sign(result, input);
        return _finish_nan(input, result);
    }

    [[nodiscard]] ::llvm::Value *_build_asinh(
        ::llvm::Value *input) {
        auto &b = builder();
        auto *magnitude = abs(input);
        auto *small = b.CreateFCmpOLE(magnitude, f32(0.25));
        auto *polynomial = _asinh_polynomial(magnitude);
        auto *large = b.CreateFCmpOGT(magnitude, f32(4096.0));
        auto *quadratic_input = b.CreateSelect(
            large, f32(1.0), magnitude);
        auto *root = _sqrt(b.CreateFAdd(
            b.CreateFMul(quadratic_input, quadratic_input), f32(1.0)));
        auto *medium_argument = b.CreateFAdd(magnitude, root);
        auto *log_argument = b.CreateSelect(
            large, magnitude, medium_argument);
        auto *general = _log(log_argument);
        general = b.CreateFAdd(
            general, b.CreateSelect(large, f32(0.69314718055994530942),
                                    f32(0.0)));
        auto *result = b.CreateSelect(small, polynomial, general);
        result = copy_sign(result, input);
        return _finish_nan(input, result);
    }

    [[nodiscard]] ::llvm::Value *_build_acosh(
        ::llvm::Value *input) {
        auto &b = builder();
        auto *valid = b.CreateFCmpOGE(input, f32(1.0));
        auto *delta = b.CreateFSub(input, f32(1.0));
        auto *near = b.CreateAnd(
            valid, b.CreateFCmpOLE(delta, f32(0.5)));
        auto *safe_delta = b.CreateSelect(near, delta, f32(0.0));
        auto *near_result = _acosh_polynomial(safe_delta);

        auto *large = b.CreateFCmpOGT(input, f32(4096.0));
        auto *medium = b.CreateAnd(valid, b.CreateNot(large));
        auto *quadratic_input = b.CreateSelect(
            medium, input, f32(1.0));
        auto *root = _sqrt(b.CreateFMul(
            b.CreateFSub(quadratic_input, f32(1.0)),
            b.CreateFAdd(quadratic_input, f32(1.0))));
        auto *medium_argument = b.CreateFAdd(input, root);
        auto *log_argument = b.CreateSelect(
            large, input,
            b.CreateSelect(valid, medium_argument, f32(1.0)));
        auto *general = _log(log_argument);
        general = b.CreateFAdd(
            general, b.CreateSelect(large, f32(0.69314718055994530942),
                                    f32(0.0)));
        auto *result = b.CreateSelect(near, near_result, general);
        result = b.CreateSelect(valid, result, quiet_nan());
        return _finish_nan(input, result);
    }

    [[nodiscard]] ::llvm::Value *_build_atanh(
        ::llvm::Value *input) {
        auto &b = builder();
        auto *magnitude = abs(input);
        auto *valid = b.CreateFCmpOLE(magnitude, f32(1.0));
        auto *small_limit = f32(_precise() ? 0.5 : 0.25);
        auto *small = b.CreateAnd(
            valid, b.CreateFCmpOLE(magnitude, small_limit));
        auto *polynomial = _atanh_polynomial(magnitude);
        auto *safe_magnitude = b.CreateSelect(
            valid, magnitude, f32(0.0));
        auto *ratio = b.CreateFDiv(
            b.CreateFAdd(f32(1.0), safe_magnitude),
            b.CreateFSub(f32(1.0), safe_magnitude));
        auto *general = b.CreateFMul(f32(0.5), _log(ratio));
        auto *result = b.CreateSelect(small, polynomial, general);
        result = copy_sign(result, input);
        result = b.CreateSelect(valid, result, quiet_nan());
        return _finish_nan(input, result);
    }

public:
    HyperbolicF32IRBuilder(
        ::llvm::Module &module, ::llvm::Function *function,
        uint32_t width, LLVMNativeMathMode mode,
        ::llvm::Function *exp_half_function,
        ::llvm::Function *log_function)
        : FastF32IRBuilder{module, function, width},
          _mode{mode},
          _exp_half_function{exp_half_function},
          _log_function{log_function} {}

    void build(::llvm::Function *function, NativeHyperbolicKind kind) {
        auto *input = function->getArg(0u);
        input->setName("x");
        ::llvm::Value *result = nullptr;
        switch (kind) {
            case NativeHyperbolicKind::sinh:
                result = _build_sinh_cosh(input, false);
                break;
            case NativeHyperbolicKind::cosh:
                result = _build_sinh_cosh(input, true);
                break;
            case NativeHyperbolicKind::tanh:
                result = _build_tanh(input);
                break;
            case NativeHyperbolicKind::asinh:
                result = _build_asinh(input);
                break;
            case NativeHyperbolicKind::acosh:
                result = _build_acosh(input);
                break;
            case NativeHyperbolicKind::atanh:
                result = _build_atanh(input);
                break;
        }
        builder().CreateRet(result);
    }
};

}// namespace

void build_hyperbolic_f32(
    ::llvm::Module &module, ::llvm::Function *function,
    uint32_t width, LLVMNativeMathMode mode,
    NativeHyperbolicKind kind,
    ::llvm::Function *exp_half_function,
    ::llvm::Function *log_function) {
    HyperbolicF32IRBuilder{
        module, function, width, mode,
        exp_half_function, log_function}
        .build(function, kind);
}

}// namespace luisa::compute::cpu::detail
