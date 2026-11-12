#pragma once
#include <volk.h>
#include <luisa/core/stl/memory.h>
#include <luisa/core/stl/vector.h>
#include <luisa/runtime/command_list.h>
#include <luisa/runtime/rhi/resource.h>
#include "memory_policy_plan.h"

namespace lc::vk {
class Device;

// Backend-internal surface of VK_EXT_memory_decompression.
//
// The extension exposes no host-side entry point: its only commands are
// vkCmdDecompressMemoryEXT and vkCmdDecompressMemoryIndirectCountEXT, both of
// which record into a command buffer. The synchronous helpers below are the
// host-callable flavour ("decompress now"): they record the extension command as
// an extension command of a Luisa stream, submit it and wait, so the command
// still participates in the backend's normal resource-barrier and command-reorder
// bookkeeping instead of bypassing it with a raw queue submission.
//
// The whole surface is gated on
// `Device::optional_extension_enabled(OptionalExtension::MEMORY_DECOMPRESSION)`
// and fails closed with LUISA_ERROR otherwise. It is an optional capability:
// no pre-existing backend path depends on it.
namespace memory_decompression {

// Whether this logical device can execute decompression commands: the extension
// is negotiated and the backend can express its 64-bit pipeline stage/access
// bits, which requires synchronization2.
[[nodiscard]] bool available(const Device *device) noexcept;

// Decompression methods reported by the physical device
// (VkPhysicalDeviceMemoryDecompressionPropertiesEXT::decompressionMethods).
[[nodiscard]] VkMemoryDecompressionMethodFlagsEXT supported_methods(
    const Device *device) noexcept;

// VkPhysicalDeviceMemoryDecompressionPropertiesEXT::maxDecompressionIndirectCount.
[[nodiscard]] uint64_t max_indirect_count(const Device *device) noexcept;

// A decompression region addressed through backend buffer handles. The compressed
// range is read from `source_buffer`, the decompressed range is written to
// `destination_buffer`. VUID-VkDecompressMemoryRegionEXT-srcAddress-11764/11765
// require both buffers to be created with
// VK_BUFFER_USAGE_2_MEMORY_DECOMPRESSION_BIT_EXT.
struct Region {
    uint64_t source_buffer{luisa::compute::invalid_resource_handle};
    uint64_t source_offset{0u};
    uint64_t compressed_size{0u};
    uint64_t destination_buffer{luisa::compute::invalid_resource_handle};
    uint64_t destination_offset{0u};
    uint64_t decompressed_size{0u};
};

// Creates a storage buffer usable as a decompression source or destination: it
// carries VK_BUFFER_USAGE_2_MEMORY_DECOMPRESSION_BIT_EXT, indirect-buffer usage
// and a shader device address.
//
// That usage bit is bit 32, so it only exists in the 64-bit
// VkBufferUsageFlags2 mask, which VkBufferCreateInfo can only receive through
// VkBufferUsageFlags2CreateInfoKHR (VK_KHR_maintenance5 / Vulkan 1.4). The
// helper fails closed with LUISA_ERROR when the device does not offer that
// channel instead of creating a buffer that violates the extension's valid
// usage.
[[nodiscard]] luisa::compute::BufferCreationInfo create_buffer(
    Device *device, size_t byte_size) noexcept;

// Records the command into `list` without submitting it. The caller owns both
// the list and the stream it is dispatched on.
void append_decompress_memory(
    luisa::compute::CommandList &list, Device *device,
    luisa::span<const Region> regions,
    VkMemoryDecompressionMethodFlagsEXT method) noexcept;
void append_decompress_memory_indirect(
    luisa::compute::CommandList &list, Device *device,
    uint64_t indirect_commands_buffer, uint64_t indirect_count_buffer,
    uint32_t max_decompression_count, uint32_t stride,
    VkMemoryDecompressionMethodFlagsEXT method) noexcept;

// Records and submits; returns before the device executed the command.
void dispatch_decompress_memory(
    Device *device, uint64_t stream, luisa::span<const Region> regions,
    VkMemoryDecompressionMethodFlagsEXT method) noexcept;
void dispatch_decompress_memory_indirect(
    Device *device, uint64_t stream,
    uint64_t indirect_commands_buffer, uint64_t indirect_count_buffer,
    uint32_t max_decompression_count, uint32_t stride,
    VkMemoryDecompressionMethodFlagsEXT method) noexcept;

// Records, submits and waits for completion: the host-callable decompression
// entry point of the backend.
void decompress_memory(
    Device *device, uint64_t stream, luisa::span<const Region> regions,
    VkMemoryDecompressionMethodFlagsEXT method) noexcept;
void decompress_memory_indirect(
    Device *device, uint64_t stream,
    uint64_t indirect_commands_buffer, uint64_t indirect_count_buffer,
    uint32_t max_decompression_count, uint32_t stride,
    VkMemoryDecompressionMethodFlagsEXT method) noexcept;

}// namespace memory_decompression

}// namespace lc::vk
