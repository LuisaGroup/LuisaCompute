#pragma once

#include <luisa/core/stl/string.h>
#include <luisa/core/stl/vector.h>

#include "bindless_usage.h"
#include "target_features.h"

namespace luisa::compute {
class Type;
namespace xir {
class Function;
class Instruction;
}// namespace xir
}// namespace luisa::compute

namespace lc::spirv {

struct SpirvRuntimeTargetPlan {
    // Reachable optimized XIR is authoritative: it is the only input walked
    // by native SPIR-V emission. The broader AST propagated-builtin set may
    // contain dead operations and is deliberately absent from this contract.
    SpirvBindlessResourceUsage bindless_resources;
    bool uses_semantic_ray_query{false};
    bool uses_subgroup_extended_types{false};
    bool uses_shader_device_clock{false};
    bool uses_buffer_device_address{false};
    SpirvTargetFeatureMask required_features{};
};

struct SpirvRuntimeTargetDiagnostic {
    const luisa::compute::xir::Function *function{nullptr};
    const luisa::compute::xir::Instruction *instruction{nullptr};
    SpirvTargetFeatureMask feature{};
    luisa::string message;
};

struct SpirvRuntimeTargetPlanResult {
    SpirvRuntimeTargetPlan plan;
    SpirvTargetFeatureMask missing_features{};
    luisa::vector<SpirvRuntimeTargetDiagnostic> diagnostics;

    [[nodiscard]] bool succeeded() const noexcept {
        return diagnostics.empty();
    }
    [[nodiscard]] explicit operator bool() const noexcept {
        return succeeded();
    }
};

// Returns true only for the scalar widths whose use by a GroupNonUniform
// value operation is gated by VkPhysicalDeviceVulkan12Features::
// shaderSubgroupExtendedTypes. Generic scalar arithmetic/storage features are
// accounted independently by the emitted SPIR-V capabilities.
[[nodiscard]] bool spirv_subgroup_type_requires_extended_types(
    const luisa::compute::Type *type) noexcept;

// Freezes every target requirement that has runtime meaning but cannot be
// reconstructed from the optimized SPIR-V capability set. `xir_bindless`
// comes from the canonical reachable instruction analysis and is the exact
// descriptor plan consumed by binding.
[[nodiscard]] SpirvRuntimeTargetPlanResult plan_spirv_runtime_target_contract(
    luisa::span<const luisa::compute::xir::Function *const> functions,
    SpirvBindlessResourceUsage xir_bindless,
    const SpirvTargetFeatures &features) noexcept;

}// namespace lc::spirv
