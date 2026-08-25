#pragma once

#include <cstdint>
#include <string>
#include <unordered_map>

#include "../schedule/schedule_ir.h"

namespace llvm {
class LLVMContext;
class Type;
class VectorType;
}// namespace llvm

namespace luisa::compute::simd {

// Maps Luisa/Schedule IR values to fixed-width LLVM register types. The
// expression representation keeps cohort-uniform values scalar; the state
// representation widens them because different dynamic cohorts may spill
// different values to the same static slot.
class LLVMValueLayout {

private:
    ::llvm::LLVMContext &_context;
    uint32_t _width{0u};
    std::unordered_map<const Type *, ::llvm::Type *> _uniform_types;
    std::unordered_map<const Type *, ::llvm::Type *> _varying_types;
    std::string _error;

private:
    [[nodiscard]] ::llvm::Type *_uniform_type(const Type *type) noexcept;
    [[nodiscard]] ::llvm::Type *_varying_type(const Type *type) noexcept;
    [[nodiscard]] ::llvm::Type *_fail(const Type *type,
                                      const char *reason) noexcept;

public:
    LLVMValueLayout(::llvm::LLVMContext &context, uint32_t width) noexcept;

    [[nodiscard]] ::llvm::Type *expression_type(
        const schedule::Value &value) noexcept;
    [[nodiscard]] ::llvm::Type *state_type(
        const schedule::Value &value) noexcept;
    [[nodiscard]] ::llvm::VectorType *mask_type() noexcept;

    [[nodiscard]] uint32_t width() const noexcept { return _width; }
    [[nodiscard]] bool succeeded() const noexcept { return _error.empty(); }
    [[nodiscard]] const std::string &error() const noexcept { return _error; }
};

}// namespace luisa::compute::simd
