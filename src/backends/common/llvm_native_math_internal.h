#pragma once

#include <cstdint>

#include <llvm/IR/Constants.h>
#include <llvm/IR/DerivedTypes.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/Module.h>

namespace luisa::compute::cpu {
enum struct LLVMNativeMathMode : uint8_t;
}// namespace luisa::compute::cpu

namespace luisa::compute::cpu::detail {

enum struct NativeTrigKind : uint8_t {
    sin,
    cos,
    tan,
};

enum struct NativeInverseTrigKind : uint8_t {
    asin,
    acos,
    atan,
};

enum struct NativeExpLogKind : uint8_t {
    exp,
    exp2,
    exp10,
    log,
    log2,
    log10,
};

// Shared target-independent fixed-vector primitives used by the fast tier.
// The precise SLEEF-derived builders deliberately retain their original
// operation ordering in their responsible translation units.
class FastF32IRBuilder {

private:
    ::llvm::Module &_module;
    ::llvm::IRBuilder<> _builder;
    ::llvm::FixedVectorType *_float_vector;
    ::llvm::FixedVectorType *_int_vector;

public:
    FastF32IRBuilder(::llvm::Module &module, ::llvm::Function *function,
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
        ::llvm::FastMathFlags flags;
        flags.setAllowContract(true);
        _builder.setFastMathFlags(flags);
    }

    [[nodiscard]] auto &module() noexcept { return _module; }
    [[nodiscard]] auto &builder() noexcept { return _builder; }
    [[nodiscard]] auto *float_vector() const noexcept {
        return _float_vector;
    }
    [[nodiscard]] auto *int_vector() const noexcept {
        return _int_vector;
    }

    [[nodiscard]] ::llvm::Constant *f32(double value) const {
        auto *scalar = ::llvm::ConstantFP::get(
            ::llvm::Type::getFloatTy(_module.getContext()), value);
        return ::llvm::ConstantVector::getSplat(
            _float_vector->getElementCount(), scalar);
    }

    [[nodiscard]] ::llvm::Constant *i32(uint32_t value) const {
        auto *scalar = ::llvm::ConstantInt::get(
            ::llvm::Type::getInt32Ty(_module.getContext()), value);
        return ::llvm::ConstantVector::getSplat(
            _int_vector->getElementCount(), scalar);
    }

    [[nodiscard]] ::llvm::Value *mla(
        ::llvm::Value *x, ::llvm::Value *y, ::llvm::Value *z) {
        return _builder.CreateFAdd(_builder.CreateFMul(x, y), z);
    }

    [[nodiscard]] ::llvm::Value *float_bits(::llvm::Value *value) {
        return _builder.CreateBitCast(value, _int_vector);
    }

    [[nodiscard]] ::llvm::Value *bits_float(::llvm::Value *value) {
        return _builder.CreateBitCast(value, _float_vector);
    }

    [[nodiscard]] ::llvm::Value *abs(::llvm::Value *value) {
        return bits_float(_builder.CreateAnd(
            float_bits(value), i32(0x7fffffffu)));
    }

    [[nodiscard]] ::llvm::Value *copy_sign(
        ::llvm::Value *magnitude, ::llvm::Value *sign) {
        auto *magnitude_bits = _builder.CreateAnd(
            float_bits(magnitude), i32(0x7fffffffu));
        auto *sign_bits = _builder.CreateAnd(
            float_bits(sign), i32(0x80000000u));
        return bits_float(_builder.CreateOr(magnitude_bits, sign_bits));
    }

    [[nodiscard]] ::llvm::Value *round_nearest(::llvm::Value *value) {
        auto *bias = copy_sign(f32(8388608.0), value);
        return _builder.CreateFSub(
            _builder.CreateFAdd(value, bias), bias);
    }

    [[nodiscard]] ::llvm::Value *quiet_nan() {
        return bits_float(i32(0x7fc00000u));
    }

    [[nodiscard]] ::llvm::Value *positive_infinity() {
        return bits_float(i32(0x7f800000u));
    }

    [[nodiscard]] ::llvm::Value *negative_infinity() {
        return bits_float(i32(0xff800000u));
    }

    [[nodiscard]] ::llvm::Value *is_nan(::llvm::Value *value) {
        return _builder.CreateFCmpUNO(value, value);
    }

    [[nodiscard]] ::llvm::Value *is_subnormal(::llvm::Value *value) {
        auto *absolute_bits = _builder.CreateAnd(
            float_bits(value), i32(0x7fffffffu));
        return _builder.CreateAnd(
            _builder.CreateICmpNE(absolute_bits, i32(0u)),
            _builder.CreateICmpULT(absolute_bits, i32(0x00800000u)));
    }
};

void build_fast_trig_f32(::llvm::Module &module,
                         ::llvm::Function *function, uint32_t width,
                         NativeTrigKind kind,
                         ::llvm::Function *precise_function);

void build_precise_trig_f32(::llvm::Module &module,
                            ::llvm::Function *function, uint32_t width,
                            NativeTrigKind kind);

[[nodiscard]] ::llvm::GlobalVariable *get_trig_reduction_table(
    ::llvm::Module &module);

void build_fast_inverse_trig_f32(
    ::llvm::Module &module, ::llvm::Function *function, uint32_t width,
    NativeInverseTrigKind kind);

void build_precise_inverse_trig_f32(
    ::llvm::Module &module, ::llvm::Function *function, uint32_t width,
    NativeInverseTrigKind kind);

void build_atan2_f32(::llvm::Module &module,
                     ::llvm::Function *function, uint32_t width,
                     LLVMNativeMathMode mode);

void build_fast_exp_log_f32(::llvm::Module &module,
                            ::llvm::Function *function, uint32_t width,
                            NativeExpLogKind kind);

void build_precise_exp_log_f32(::llvm::Module &module,
                               ::llvm::Function *function,
                               uint32_t width, NativeExpLogKind kind);

void build_pow_f32(::llvm::Module &module,
                   ::llvm::Function *function, uint32_t width,
                   LLVMNativeMathMode mode,
                   ::llvm::Function *fast_log_function,
                   ::llvm::Function *fast_exp_function);

}// namespace luisa::compute::cpu::detail
