#pragma once

#include <luisa/core/stl/string.h>
#include <luisa/core/stl/vector.h>

namespace luisa::compute::xir {
class FunctionDefinition;
class BasicBlock;
class Instruction;
}// namespace luisa::compute::xir

namespace lc::spirv {

struct SpirvRayQueryLifetimeDiagnostic {
    const luisa::compute::xir::BasicBlock *block{nullptr};
    const luisa::compute::xir::Instruction *instruction{nullptr};
    luisa::string message;
};

struct SpirvRayQueryLifetimeValidationResult {
    luisa::vector<SpirvRayQueryLifetimeDiagnostic> diagnostics;

    [[nodiscard]] bool succeeded() const noexcept {
        return diagnostics.empty();
    }
    [[nodiscard]] explicit operator bool() const noexcept {
        return succeeded();
    }
};

// Validates opaque ray-query representation and lifetime rules over the
// ordinary reachable prefix of the function's SPIR-V structural closure.
// True orphans are not emitted, while disconnected structured-role payloads
// have a separate flat/no-ray-query dialect contract.
[[nodiscard]] SpirvRayQueryLifetimeValidationResult
validate_spirv_ray_query_lifetimes(
    const luisa::compute::xir::FunctionDefinition *function) noexcept;

}// namespace lc::spirv
