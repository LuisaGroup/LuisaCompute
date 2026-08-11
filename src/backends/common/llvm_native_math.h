#pragma once

#include <cstdint>

#include <llvm/IR/IRBuilder.h>

namespace llvm {
class Module;
class Value;
}// namespace llvm

namespace luisa::compute::cpu {

enum struct LLVMNativeMathMode : uint8_t {
    precise,
    fast,
};

// Emits calls to module-local, target-independent fixed-vector math bodies.
// Scalar values deliberately stay outside this layer: callers must issue one
// scalar operation for uniform/scalar semantics instead of broadcasting it.
class LLVMNativeMath {

public:
    [[nodiscard]] static ::llvm::Value *emit_sin_f32(
        ::llvm::Module &module, ::llvm::IRBuilder<> &builder,
        ::llvm::Value *vector, LLVMNativeMathMode mode);

    [[nodiscard]] static ::llvm::Value *emit_cos_f32(
        ::llvm::Module &module, ::llvm::IRBuilder<> &builder,
        ::llvm::Value *vector, LLVMNativeMathMode mode);

    [[nodiscard]] static ::llvm::Value *emit_tan_f32(
        ::llvm::Module &module, ::llvm::IRBuilder<> &builder,
        ::llvm::Value *vector, LLVMNativeMathMode mode);

    [[nodiscard]] static ::llvm::Value *emit_asin_f32(
        ::llvm::Module &module, ::llvm::IRBuilder<> &builder,
        ::llvm::Value *vector, LLVMNativeMathMode mode);

    [[nodiscard]] static ::llvm::Value *emit_acos_f32(
        ::llvm::Module &module, ::llvm::IRBuilder<> &builder,
        ::llvm::Value *vector, LLVMNativeMathMode mode);

    [[nodiscard]] static ::llvm::Value *emit_atan_f32(
        ::llvm::Module &module, ::llvm::IRBuilder<> &builder,
        ::llvm::Value *vector, LLVMNativeMathMode mode);

    [[nodiscard]] static ::llvm::Value *emit_exp_f32(
        ::llvm::Module &module, ::llvm::IRBuilder<> &builder,
        ::llvm::Value *vector, LLVMNativeMathMode mode);

    [[nodiscard]] static ::llvm::Value *emit_exp2_f32(
        ::llvm::Module &module, ::llvm::IRBuilder<> &builder,
        ::llvm::Value *vector, LLVMNativeMathMode mode);

    [[nodiscard]] static ::llvm::Value *emit_exp10_f32(
        ::llvm::Module &module, ::llvm::IRBuilder<> &builder,
        ::llvm::Value *vector, LLVMNativeMathMode mode);

    [[nodiscard]] static ::llvm::Value *emit_log_f32(
        ::llvm::Module &module, ::llvm::IRBuilder<> &builder,
        ::llvm::Value *vector, LLVMNativeMathMode mode);

    [[nodiscard]] static ::llvm::Value *emit_log2_f32(
        ::llvm::Module &module, ::llvm::IRBuilder<> &builder,
        ::llvm::Value *vector, LLVMNativeMathMode mode);

    [[nodiscard]] static ::llvm::Value *emit_log10_f32(
        ::llvm::Module &module, ::llvm::IRBuilder<> &builder,
        ::llvm::Value *vector, LLVMNativeMathMode mode);
};

}// namespace luisa::compute::cpu
