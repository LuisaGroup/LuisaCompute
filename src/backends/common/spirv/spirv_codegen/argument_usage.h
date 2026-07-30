#pragma once

#include <luisa/ast/usage.h>
#include <luisa/core/stl/unordered_map.h>
#include <luisa/core/stl/vector.h>
#include <luisa/xir/op.h>

namespace luisa::compute::xir {
class Module;
class Function;
class Argument;
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
    // Bindless buffer size/read/write operations need a per-array local
    // metadata descriptor; texture-only and unused arrays do not.
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
    const luisa::compute::xir::Module *module) noexcept;

[[nodiscard]] SpirvReadonlyResourceOriginMap
analyze_spirv_readonly_resource_origins(
    const luisa::compute::xir::Module *module,
    const SpirvFunctionArgumentAnalysisMap &usage) noexcept;

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
