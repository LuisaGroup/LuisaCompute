#pragma once

#include <cstdint>

namespace lc::spirv {

using SpirvKernelArgumentRoleMask = uint32_t;

// Exact native runtime roles for one kernel argument. These values cross the
// SPIR-V codegen/Vulkan serialization boundary; keep them stable and compose
// them as a bitmask. A zero mask is a valid exact plan for an unused argument.
namespace kernel_argument_role {
inline constexpr SpirvKernelArgumentRoleMask none = 0u;
inline constexpr SpirvKernelArgumentRoleMask accel_traversal = 1u << 0u;
inline constexpr SpirvKernelArgumentRoleMask accel_instance = 1u << 1u;
inline constexpr SpirvKernelArgumentRoleMask accel_known_mask =
    accel_traversal | accel_instance;
}// namespace kernel_argument_role

}// namespace lc::spirv
