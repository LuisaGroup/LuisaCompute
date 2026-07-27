#pragma once

#include <cstdint>

#include <luisa/core/stl/memory.h>
#include <luisa/core/stl/string.h>
#include <luisa/core/stl/vector.h>

#include "target_features.h"

namespace luisa::compute {
class Type;
namespace xir {
class Function;
class Instruction;
}// namespace xir
}// namespace luisa::compute

namespace lc::spirv {

// Logical bool has no StorageBuffer representation. Keep the recursive type
// query shared by atomic planning and physical buffer binding so both choose
// the same word-storage/coherency ABI.
[[nodiscard]] bool spirv_type_contains_bool(
    const luisa::compute::Type *type) noexcept;

struct SpirvAtomicBufferPlanOptions {
    // Null means target-independent dialect planning: assume every optional
    // native float32 buffer-atomic feature is available, while retaining
    // representation constraints that no target can remove (notably float
    // compare-exchange's integer-word implementation).
    const SpirvTargetFeatures *target_features{nullptr};
};

struct SpirvAtomicBufferAssignment {
    const luisa::compute::Type *buffer_type;
    SpirvAtomicBufferStoragePlan storage;
};

struct SpirvAtomicBufferPlanDiagnostic {
    const luisa::compute::xir::Function *function{nullptr};
    const luisa::compute::xir::Instruction *instruction{nullptr};
    const luisa::compute::Type *buffer_type{nullptr};
    luisa::string message;
};

struct SpirvAtomicBufferModulePlan {
    luisa::vector<SpirvAtomicBufferAssignment> assignments;
    luisa::vector<SpirvAtomicBufferPlanDiagnostic> diagnostics;

    [[nodiscard]] bool succeeded() const noexcept {
        return diagnostics.empty();
    }
    [[nodiscard]] explicit operator bool() const noexcept {
        return succeeded();
    }
};

// Plans one physical pointer representation for every Buffer<T> containing a
// surviving atomic access. The input function order must be the canonical
// kernel-reachable order returned by call-graph/module usage analysis.
[[nodiscard]] SpirvAtomicBufferModulePlan
plan_spirv_atomic_buffers(
    luisa::span<const luisa::compute::xir::Function *const> functions,
    SpirvAtomicBufferPlanOptions options = {}) noexcept;

}// namespace lc::spirv
