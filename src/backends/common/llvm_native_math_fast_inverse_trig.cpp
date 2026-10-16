#include "llvm_native_math_internal.h"

namespace luisa::compute::cpu::detail {

namespace {

class FastInverseTrigF32IRBuilder final : public FastF32IRBuilder {

private:
    [[nodiscard]] ::llvm::Value *_asin_series(
        ::llvm::Value *x, ::llvm::Value *x2) {
        // Maclaurin asin(x) = x + x^3/6 + 3x^5/40 + 5x^7/112 + ...
        // Range transformation below guarantees |x| <= 1/2.
        auto *polynomial = mla(
            f32(5.0 / 112.0), x2, f32(3.0 / 40.0));
        polynomial = mla(
            polynomial, x2, f32(1.0 / 6.0));
        return builder().CreateFAdd(
            x, builder().CreateFMul(
                   builder().CreateFMul(x, x2), polynomial));
    }

    void _build_asin_acos(
        ::llvm::Function *function, bool cosine) {
        auto &b = builder();
        auto *input = function->getArg(0u);
        input->setName("x");
        auto *absolute = abs(input);
        auto *valid = b.CreateFCmpOLE(absolute, f32(1.0));
        auto *safe_absolute = b.CreateSelect(
            valid, absolute, f32(0.0));
        auto *small = b.CreateFCmpOLT(
            safe_absolute, f32(0.5));
        auto *x2 = b.CreateSelect(
            small,
            b.CreateFMul(safe_absolute, safe_absolute),
            b.CreateFMul(
                b.CreateFSub(f32(1.0), safe_absolute),
                f32(0.5)));
        auto *root = b.CreateUnaryIntrinsic(
            ::llvm::Intrinsic::sqrt, x2);
        auto *x = b.CreateSelect(small, safe_absolute, root);
        auto *approximation = _asin_series(x, x2);

        ::llvm::Value *result;
        if (cosine) {
            auto *middle = b.CreateFSub(
                f32(1.5707963267948966192),
                copy_sign(approximation, input));
            auto *edge = b.CreateFMul(
                approximation, f32(2.0));
            edge = b.CreateSelect(
                b.CreateFCmpOLT(input, f32(0.0)),
                b.CreateFSub(f32(3.1415926535897932385), edge),
                edge);
            result = b.CreateSelect(small, middle, edge);
        } else {
            auto *edge = b.CreateFSub(
                f32(1.5707963267948966192),
                b.CreateFMul(approximation, f32(2.0)));
            result = copy_sign(
                b.CreateSelect(small, approximation, edge), input);
            result = b.CreateSelect(
                is_subnormal(input), input, result);
        }
        result = b.CreateSelect(valid, result, quiet_nan());
        b.CreateRet(result);
    }

    void _build_atan(::llvm::Function *function) {
        auto &b = builder();
        auto *input = function->getArg(0u);
        input->setName("x");
        auto *absolute = abs(input);
        auto *large = b.CreateFCmpOGT(
            absolute, f32(2.4142135623730950488));
        auto *middle = b.CreateFCmpOGT(
            absolute, f32(0.41421356237309504880));

        // atan(a) identities reduce every finite a >= 0 to |z| <= tan(pi/8):
        //   atan(a) = pi/2 + atan(-1/a), a > tan(3pi/8)
        //   atan(a) = pi/4 + atan((a-1)/(a+1)), a > tan(pi/8).
        auto *numerator = b.CreateSelect(
            large, f32(-1.0),
            b.CreateSelect(
                middle, b.CreateFSub(absolute, f32(1.0)), absolute));
        auto *denominator = b.CreateSelect(
            large, absolute,
            b.CreateSelect(
                middle, b.CreateFAdd(absolute, f32(1.0)), f32(1.0)));
        auto *z = b.CreateFDiv(numerator, denominator);
        auto *offset = b.CreateSelect(
            large, f32(1.5707963267948966192),
            b.CreateSelect(
                middle, f32(0.78539816339744830962), f32(0.0)));

        auto *z2 = b.CreateFMul(z, z);
        // Alternating atan series through z^9. On the reduced interval the
        // first omitted term bounds the exact-series truncation error.
        ::llvm::Value *polynomial = f32(1.0 / 9.0);
        polynomial = mla(polynomial, z2, f32(-1.0 / 7.0));
        polynomial = mla(polynomial, z2, f32(1.0 / 5.0));
        polynomial = mla(polynomial, z2, f32(-1.0 / 3.0));
        auto *reduced = mla(
            z2, b.CreateFMul(z, polynomial), z);
        auto *result = copy_sign(
            b.CreateFAdd(offset, reduced), input);
        result = b.CreateSelect(
            b.CreateOr(
                is_subnormal(input),
                b.CreateFCmpOEQ(absolute, f32(0.0))),
            input, result);
        result = b.CreateSelect(is_nan(input), quiet_nan(), result);
        b.CreateRet(result);
    }

public:
    FastInverseTrigF32IRBuilder(
        ::llvm::Module &module, ::llvm::Function *function,
        uint32_t width)
        : FastF32IRBuilder{module, function, width} {}

    void build(
        ::llvm::Function *function, NativeInverseTrigKind kind) {
        switch (kind) {
            case NativeInverseTrigKind::asin:
                _build_asin_acos(function, false);
                break;
            case NativeInverseTrigKind::acos:
                _build_asin_acos(function, true);
                break;
            case NativeInverseTrigKind::atan:
                _build_atan(function);
                break;
        }
    }
};

}// namespace

void build_fast_inverse_trig_f32(
    ::llvm::Module &module, ::llvm::Function *function, uint32_t width,
    NativeInverseTrigKind kind) {
    FastInverseTrigF32IRBuilder{module, function, width}.build(
        function, kind);
}

}// namespace luisa::compute::cpu::detail
