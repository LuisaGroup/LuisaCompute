#pragma once

#include <cstddef>
#include <cstdint>

#include <luisa/core/stl/string.h>
#include <luisa/core/stl/vector.h>
#include <luisa/ast/function.h>
#include <luisa/xir/module.h>
#include <luisa/xir/op.h>

namespace lc::spirv {

namespace xir = luisa::compute::xir;

enum class SpirvXIRDialectSupport : uint8_t {
    SUPPORTED,
    SEMANTIC_NO_OP,
    UNSUPPORTED,
    UNKNOWN,
};

struct SpirvXIRDialectOpSupport {
    SpirvXIRDialectSupport support{SpirvXIRDialectSupport::UNKNOWN};
    luisa::string_view reason;

    [[nodiscard]] constexpr bool accepted() const noexcept {
        return support == SpirvXIRDialectSupport::SUPPORTED ||
               support == SpirvXIRDialectSupport::SEMANTIC_NO_OP;
    }
    [[nodiscard]] constexpr bool known() const noexcept {
        return support != SpirvXIRDialectSupport::UNKNOWN;
    }
};

// These overloads are the single operation-level support matrix for the
// native XIR-to-SPIR-V dialect. Unknown enum values fail closed. Tests walk
// each current contiguous enum range, assert its classification counts, and
// verify that the first out-of-range value remains rejected.
[[nodiscard]] SpirvXIRDialectOpSupport
spirv_xir_dialect_support(xir::AllocaOp op) noexcept;
[[nodiscard]] SpirvXIRDialectOpSupport
spirv_xir_dialect_support(xir::ArithmeticOp op) noexcept;
[[nodiscard]] SpirvXIRDialectOpSupport
spirv_xir_dialect_support(xir::AtomicOp op) noexcept;
[[nodiscard]] SpirvXIRDialectOpSupport
spirv_xir_dialect_support(xir::CastOp op) noexcept;
[[nodiscard]] SpirvXIRDialectOpSupport
spirv_xir_dialect_support(xir::ResourceQueryOp op) noexcept;
[[nodiscard]] SpirvXIRDialectOpSupport
spirv_xir_dialect_support(xir::ResourceReadOp op) noexcept;
[[nodiscard]] SpirvXIRDialectOpSupport
spirv_xir_dialect_support(xir::ResourceWriteOp op) noexcept;
[[nodiscard]] SpirvXIRDialectOpSupport
spirv_xir_dialect_support(xir::ThreadGroupOp op) noexcept;
[[nodiscard]] SpirvXIRDialectOpSupport
spirv_xir_dialect_support(xir::RayQueryObjectReadOp op) noexcept;
[[nodiscard]] SpirvXIRDialectOpSupport
spirv_xir_dialect_support(xir::RayQueryObjectWriteOp op) noexcept;
[[nodiscard]] SpirvXIRDialectOpSupport
spirv_xir_dialect_support(xir::DerivedSpecialRegisterTag tag) noexcept;
[[nodiscard]] SpirvXIRDialectOpSupport
spirv_xir_dialect_support(xir::DerivedInstructionTag tag) noexcept;

struct SpirvXIRDialectDiagnostic {
    const xir::Function *function{nullptr};
    const xir::BasicBlock *block{nullptr};
    const xir::Instruction *instruction{nullptr};
    luisa::string message;
};

struct SpirvXIRDialectValidationResult {
    luisa::vector<SpirvXIRDialectDiagnostic> diagnostics;

    [[nodiscard]] bool succeeded() const noexcept {
        return diagnostics.empty();
    }
    [[nodiscard]] explicit operator bool() const noexcept {
        return succeeded();
    }
};

enum class SpirvXIRKernelABIStatus : uint8_t {
    SUCCESS,
    NULL_MODULE,
    AST_FUNCTION_IS_NOT_KERNEL,
    KERNEL_DEFINITION_COUNT_MISMATCH,
    BLOCK_SIZE_MISMATCH,
    ARGUMENT_COUNT_MISMATCH,
    ARGUMENT_TYPE_MISMATCH,
    ARGUMENT_KIND_MISMATCH,
};

struct SpirvXIRKernelABIValidationResult {
    SpirvXIRKernelABIStatus status{SpirvXIRKernelABIStatus::SUCCESS};
    size_t argument_index{~size_t{0u}};
    luisa::string diagnostic;

    [[nodiscard]] bool succeeded() const noexcept {
        return status == SpirvXIRKernelABIStatus::SUCCESS;
    }
    [[nodiscard]] explicit operator bool() const noexcept {
        return succeeded();
    }
};

// compile_spirv_xir receives both the source AST function (for persisted
// runtime ABI metadata) and the transformed XIR module (for code emission).
// Validate that this pair still describes one identical kernel before either
// side is used by a different subsystem.
[[nodiscard]] SpirvXIRKernelABIValidationResult
validate_spirv_xir_kernel_abi(
    luisa::compute::Function ast_kernel,
    const xir::Module *module) noexcept;

// Validates whole-module ordinary XIR well-formedness and the narrower native
// SPIR-V contract for exactly the kernel-reachable definitions that emission
// consumes. The returned diagnostics are non-fatal and therefore useful as
// unit-test oracles; the production handoff turns the first one into an
// explicit shader-compilation error.
[[nodiscard]] SpirvXIRDialectValidationResult
validate_spirv_xir_codegen_dialect(const xir::Module *module) noexcept;

}// namespace lc::spirv
