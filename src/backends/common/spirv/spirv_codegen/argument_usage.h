#pragma once

#include <cstddef>

#include <luisa/ast/usage.h>
#include <luisa/core/stl/memory.h>
#include <luisa/core/stl/unordered_map.h>
#include <luisa/core/stl/vector.h>
#include <luisa/xir/op.h>

namespace luisa::compute::xir {
class Module;
class Function;
class Argument;
class CallInst;
}// namespace luisa::compute::xir

namespace lc::spirv {

[[nodiscard]] bool
spirv_resource_query_requires_accel_traversal_descriptor(
    luisa::compute::xir::ResourceQueryOp op) noexcept;

[[nodiscard]] bool
spirv_resource_query_requires_accel_instance_buffer(
    luisa::compute::xir::ResourceQueryOp op) noexcept;

struct SpirvFunctionArgumentAnalysis {
    luisa::compute::Usage usage{luisa::compute::Usage::NONE};
    // Usage::READ covers both roles at the public AST/runtime boundary, but
    // Vulkan descriptors must distinguish them. Traversal needs the
    // UniformConstant AS handle; instance property reads/writes need the
    // separate kernel-bound instance-record buffer.
    bool requires_accel_traversal_descriptor{false};
    bool requires_accel_instance_buffer{false};
    // Mixed-layout bindless buffer size/read/write and all bindless device-
    // address queries need a per-array local metadata descriptor. Typed
    // size/read/write use their slot record; texture-only and unused arrays
    // need no metadata descriptor.
    bool requires_bindless_buffer_metadata{false};
    // Direct and bindless buffer-address queries require the runtime metadata
    // record to carry a proven VkDeviceAddress for the exact logical view.
    bool requires_buffer_device_address{false};
    // Volatile direct-buffer accesses require a Coherent storage-buffer
    // declaration in addition to the Volatile memory operand and fence. Keep
    // this per argument so unrelated buffers of the same element type retain
    // their ordinary cache policy.
    bool requires_buffer_coherence{false};
};

using SpirvFunctionArgumentAnalysisMap = luisa::unordered_map<
    const luisa::compute::xir::Function *,
    luisa::vector<SpirvFunctionArgumentAnalysis>>;

// Calls in the exact structural closure of the analyzed function domain.
// When kernel_reachable_only is enabled, callers outside the kernel-rooted
// semantic call graph are excluded even if their definitions remain
// physically owned by the module.
using SpirvFunctionCallSiteList = luisa::vector<
    const luisa::compute::xir::CallInst *>;

struct SpirvFunctionArgumentAnalysisOptions {
    // Restrict the result to functions reachable from a kernel through the
    // exact SPIR-V structural closure. Function operands in orphaned blocks
    // then cannot affect a live callable's resource-flow proof.
    bool kernel_reachable_only{false};
};

struct SpirvFunctionArgumentAnalysisStatistics {
    size_t structural_closure_count{0u};
    size_t instruction_scan_count{0u};
    size_t call_dependency_count{0u};
    size_t worklist_pop_count{0u};
    size_t dependency_visit_count{0u};
};

// A surviving callable buffer/bindless argument need not become an SPIR-V
// function parameter when every reachable call forwards the same kernel
// resource. In that case the descriptor (and all of its metadata side
// channels) is already a module-level kernel binding. This map records the
// unique origin proved by the resource-flow fixed point. Missing entries are
// deliberately "unknown or conflicting", never an alias proof.
using SpirvReadonlyResourceOriginMap = luisa::unordered_map<
    const luisa::compute::xir::Argument *,
    const luisa::compute::xir::Argument *>;

[[nodiscard]] SpirvFunctionArgumentAnalysisMap
analyze_spirv_function_argument_usage(
    const luisa::compute::xir::Module *module,
    SpirvFunctionArgumentAnalysisStatistics *statistics = nullptr,
    SpirvFunctionArgumentAnalysisOptions options = {},
    SpirvFunctionCallSiteList *call_sites = nullptr) noexcept;

[[nodiscard]] SpirvReadonlyResourceOriginMap
analyze_spirv_readonly_resource_origins(
    const luisa::compute::xir::Module *module,
    const SpirvFunctionArgumentAnalysisMap &usage) noexcept;

// `call_sites` must be the complete structural index produced together with
// `usage`; omitting an incoming edge would make a unique-origin proof unsound.
// The separate module overload remains the safe standalone API.
[[nodiscard]] SpirvReadonlyResourceOriginMap
analyze_spirv_readonly_resource_origins_from_call_sites(
    const SpirvFunctionArgumentAnalysisMap &usage,
    luisa::span<const luisa::compute::xir::CallInst *const>
        call_sites) noexcept;

[[nodiscard]] luisa::compute::Usage
spirv_function_argument_usage_of(
    const SpirvFunctionArgumentAnalysisMap &analysis,
    const luisa::compute::xir::Function *function,
    const luisa::compute::xir::Argument *argument) noexcept;

[[nodiscard]] bool
spirv_function_argument_requires_accel_traversal_descriptor(
    const SpirvFunctionArgumentAnalysisMap &analysis,
    const luisa::compute::xir::Function *function,
    const luisa::compute::xir::Argument *argument) noexcept;

[[nodiscard]] bool
spirv_function_argument_requires_accel_instance_buffer(
    const SpirvFunctionArgumentAnalysisMap &analysis,
    const luisa::compute::xir::Function *function,
    const luisa::compute::xir::Argument *argument) noexcept;

[[nodiscard]] bool
spirv_function_argument_requires_bindless_buffer_metadata(
    const SpirvFunctionArgumentAnalysisMap &analysis,
    const luisa::compute::xir::Function *function,
    const luisa::compute::xir::Argument *argument) noexcept;

[[nodiscard]] bool
spirv_function_argument_requires_buffer_device_address(
    const SpirvFunctionArgumentAnalysisMap &analysis,
    const luisa::compute::xir::Function *function,
    const luisa::compute::xir::Argument *argument) noexcept;

[[nodiscard]] bool
spirv_function_argument_requires_buffer_coherence(
    const SpirvFunctionArgumentAnalysisMap &analysis,
    const luisa::compute::xir::Function *function,
    const luisa::compute::xir::Argument *argument) noexcept;

}// namespace lc::spirv
