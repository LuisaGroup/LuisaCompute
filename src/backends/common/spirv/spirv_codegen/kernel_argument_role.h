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
inline constexpr SpirvKernelArgumentRoleMask buffer_device_address = 1u << 2u;
inline constexpr SpirvKernelArgumentRoleMask accel_known_mask =
    accel_traversal | accel_instance;
inline constexpr SpirvKernelArgumentRoleMask buffer_known_mask =
    buffer_device_address;
inline constexpr SpirvKernelArgumentRoleMask bindless_known_mask =
    buffer_device_address;
inline constexpr SpirvKernelArgumentRoleMask known_mask =
    accel_known_mask | buffer_known_mask | bindless_known_mask;
}// namespace kernel_argument_role

}// namespace lc::spirv
