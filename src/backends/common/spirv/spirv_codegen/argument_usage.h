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
};

using SpirvFunctionArgumentAnalysisMap = luisa::unordered_map<
    const luisa::compute::xir::Function *,
    luisa::vector<SpirvFunctionArgumentAnalysis>>;

[[nodiscard]] SpirvFunctionArgumentAnalysisMap
analyze_spirv_function_argument_usage(
    const luisa::compute::xir::Module *module) noexcept;

[[nodiscard]] luisa::compute::Usage
spirv_function_argument_usage_of(
    const SpirvFunctionArgumentAnalysisMap &analysis,
    const luisa::compute::xir::Function *function,
    const luisa::compute::xir::Argument *argument,
    luisa::compute::Usage fallback =
        luisa::compute::Usage::NONE) noexcept;

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

}// namespace lc::spirv
