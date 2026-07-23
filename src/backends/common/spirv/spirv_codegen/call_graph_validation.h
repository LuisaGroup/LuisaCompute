#pragma once

#include <luisa/core/stl/string.h>
#include <luisa/core/stl/vector.h>

namespace luisa::compute::xir {
class Module;
class Function;
class BasicBlock;
class Instruction;
}// namespace luisa::compute::xir

namespace lc::spirv {

struct SpirvReachableCallGraphDiagnostic {
    const luisa::compute::xir::Function *function{nullptr};
    const luisa::compute::xir::BasicBlock *block{nullptr};
    const luisa::compute::xir::Instruction *instruction{nullptr};
    luisa::string message;
};

struct SpirvReachableCallGraphValidationResult {
    luisa::vector<SpirvReachableCallGraphDiagnostic> diagnostics;
    // Canonical callee-before-caller order for the exact kernel-reachable
    // structural function graph. Consumers that plan module-wide state must
    // use this list instead of independently guessing reachability. The list
    // is empty when validation fails: a partial function order is not a sound
    // emission or planning boundary.
    luisa::vector<const luisa::compute::xir::Function *>
        functions_post_order;

    [[nodiscard]] bool succeeded() const noexcept {
        return diagnostics.empty();
    }
    [[nodiscard]] explicit operator bool() const noexcept {
        return succeeded();
    }
};

// Validates and freezes the same kernel-reachable function graph consumed by
// native SPIR-V usage analysis. Function operands in the active structural
// closure form edges; true-orphan instructions and unreachable callable
// definitions do not participate in emission and therefore do not participate
// here. Structural-closure planning is part of this non-fatal boundary: a
// reachable function with an invalid closure produces a diagnostic rather than
// a partial order that could later reach an emitter assertion.
[[nodiscard]] SpirvReachableCallGraphValidationResult
validate_spirv_reachable_call_graph(
    const luisa::compute::xir::Module *module) noexcept;

}// namespace lc::spirv
