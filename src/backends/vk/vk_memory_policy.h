#pragma once
#include <volk.h>
#include "buffer.h"
#include "memory_policy_plan.h"
#include "vk_allocator.h"
#include <luisa/ast/type.h>
#include <luisa/core/stl/memory.h>
#include <luisa/runtime/rhi/pixel.h>
#include <luisa/runtime/rhi/resource.h>

namespace lc::vk {
class Device;

// Backend-internal surface of the optional device-memory policy extensions.
//
// None of the public LuisaCompute entry points can express an allocation-time
// memory policy or an image tiling policy:
//
//   * `Device::create_buffer` always creates a VMA suballocated buffer with
//     undefined contents,
//   * `Device::create_texture` has no tiling-policy argument,
//   * `DeviceInterface` has no decompression command.
//
// The three helpers below are the backend-internal entry points that do express
// them. They are ordinary `Device *` free functions so they need no
// `Device::extension` registration; wiring them behind a
// `luisa::compute::DeviceExtension` would be a one-line addition in the
// extension table of `Device::_init_device`.
//
// Every helper keeps the pre-existing behaviour as its fallback: when the
// relevant extension is unsupported, switched off through
// VulkanDeviceConfigExt, or the logical device was imported, the helper either
// produces the same result the old path produces or fails closed with an
// explicit error.

// Vulkan usage mask of a general purpose backend storage buffer. Shared by the
// VMA-backed `DefaultBuffer` path and by `PolicyBuffer` so the two can never
// diverge.
[[nodiscard]] VkBufferUsageFlags storage_buffer_usage_flags(
    Device *device, bool used_as_accel, VkBufferUsageFlagBits extra_bit) noexcept;

// A backend buffer whose VkDeviceMemory is allocated by
// `VkAllocator::allocate_native_buffer` instead of a shared VMA block, so the
// optional allocation-time flags (VK_MEMORY_ALLOCATE_ZERO_INITIALIZE_BIT_EXT,
// VK_BUFFER_USAGE_2_* via VkBufferUsageFlags2CreateInfoKHR) can be requested.
class PolicyBuffer final : public Buffer {
    VkBuffer _vk_buffer{};
    VkDeviceMemory _memory{};
    VkDeviceSize _allocation_size{};
    bool _driver_zeroed{false};

public:
    PolicyBuffer(Device *device, size_t byte_size, VkBufferUsageFlags usage,
                 VkBufferUsageFlags2KHR usage2, bool host_visible,
                 MemoryInitPolicy init) noexcept;
    ~PolicyBuffer() noexcept override;
    PolicyBuffer(PolicyBuffer const &) = delete;
    PolicyBuffer(PolicyBuffer &&) = delete;
    [[nodiscard]] VkBuffer vk_buffer() const noexcept override { return _vk_buffer; }
    [[nodiscard]] VkDeviceMemory device_memory() const noexcept { return _memory; }
    [[nodiscard]] VkDeviceSize allocation_size() const noexcept { return _allocation_size; }
    // True when VK_MEMORY_ALLOCATE_ZERO_INITIALIZE_BIT_EXT was used, i.e. the
    // driver zeroed the memory before it became accessible.
    [[nodiscard]] bool driver_zeroed() const noexcept { return _driver_zeroed; }
};

// Whether driver-side zero initialization is available on this device.
[[nodiscard]] bool zero_initialize_device_memory_available(
    const Device *device) noexcept;

struct ZeroInitializedBufferCreation {
    luisa::compute::BufferCreationInfo buffer{};
    // True when the driver zeroed the memory
    // (VK_MEMORY_ALLOCATE_ZERO_INITIALIZE_BIT_EXT). When it is false and
    // `host_visible` was requested, the backend zeroed the memory from the host
    // instead, so the contents are zero in both cases.
    bool driver_zeroed{false};
};

// Creates a storage buffer whose contents are zero-initialized.
//
// Fast path: allocate the memory with VK_MEMORY_ALLOCATE_ZERO_INITIALIZE_BIT_EXT
// (VK_EXT_zero_initialize_device_memory) for a memory type that may carry the
// flag. Fallback: the pre-existing dedicated allocation plus an explicit
// host-side zero fill when `host_visible` is requested. With `host_visible`
// false and no usable flag the contents stay undefined, which is exactly what
// the generic buffer path provides today, and `driver_zeroed` reports it.
[[nodiscard]] ZeroInitializedBufferCreation create_zero_initialized_buffer(
    Device *device, const luisa::compute::Type *element, size_t elem_count,
    bool host_visible = false) noexcept;

// Creates a texture with an explicit VK_EXT_image_tiling_control tiling policy.
// `detail::ImageTilingPolicy::kDefault` reproduces `Device::create_texture`
// exactly; a non-default policy is applied when the extension is enabled and the
// image is optimally tiled, and silently degrades to the default arrangement
// otherwise.
[[nodiscard]] luisa::compute::ResourceCreationInfo create_texture_with_tiling_policy(
    Device *device, luisa::compute::PixelFormat format, uint dimension,
    uint width, uint height, uint depth, uint mipmap_levels,
    bool simultaneous_access, bool allow_raster_target,
    detail::ImageTilingPolicy tiling_policy) noexcept;

}// namespace lc::vk
