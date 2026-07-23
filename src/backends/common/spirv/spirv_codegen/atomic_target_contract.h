#pragma once

#include <luisa/core/stl/memory.h>
#include <luisa/core/stl/string.h>
#include <luisa/core/stl/vector.h>

#include "target_features.h"

namespace luisa::compute::xir {
class Function;
class Instruction;
}// namespace luisa::compute::xir

namespace lc::spirv {

struct SpirvAtomicTargetContractDiagnostic {
    const luisa::compute::xir::Function *function{nullptr};
    const luisa::compute::xir::Instruction *instruction{nullptr};
    luisa::string message;
};

struct SpirvAtomicTargetContractResult {
    luisa::vector<SpirvAtomicTargetContractDiagnostic> diagnostics;

    [[nodiscard]] bool succeeded() const noexcept {
        return diagnostics.empty();
    }
    [[nodiscard]] explicit operator bool() const noexcept {
        return succeeded();
    }
};

// Checks target-dependent atomic lowering decisions over the canonical
// kernel-reachable function order. Dialect validation owns XIR shape and
// target-independent representation errors; this boundary owns Vulkan feature
// availability that cannot be inferred from a partially emitted module.
[[nodiscard]] SpirvAtomicTargetContractResult
validate_spirv_atomic_target_contract(
    luisa::span<const luisa::compute::xir::Function *const> functions,
    const SpirvTargetFeatures &features) noexcept;

}// namespace lc::spirv
