#pragma once

#include <llvm/IR/IRBuilder.h>

namespace luisa::compute::hip {

// Resolve the target's range-reduction operation before IPO, with the native
// scalar ABI at each lane. This is not x - trunc(x / y) * y.
[[nodiscard]] llvm::Value *emit_hip_floating_remainder(
    llvm::IRBuilder<> &builder, llvm::Module &module,
    llvm::Value *dividend, llvm::Value *divisor) noexcept;

}// namespace luisa::compute::hip
