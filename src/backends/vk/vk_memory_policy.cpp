#include "vk_memory_policy.h"
#include "device.h"
#include "texture.h"
#include "log.h"
#include <luisa/core/logging.h>
#include <luisa/core/stl/memory.h>
#include <cstring>
#include <limits>

namespace lc::vk {

namespace vk_memory_policy_detail {

// Vulkan usage mask shared by every backend owned storage buffer: shader
// read/write, transfer in both directions, indirect arguments, vertex/index
// input, plus the two optional bits the device may add.
[[nodiscard]] VkBufferUsageFlags make_storage_buffer_usage(
    bool device_address, bool acceleration_structure_input,
    VkBufferUsageFlagBits extra_bit) noexcept {
    return static_cast<VkBufferUsageFlags>(
        extra_bit |
        VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
        VK_BUFFER_USAGE_INDEX_BUFFER_BIT |
        VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
        VK_BUFFER_USAGE_TRANSFER_DST_BIT |
        VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT |
        VK_BUFFER_USAGE_2_VERTEX_BUFFER_BIT |
        // VK_BUFFER_USAGE_2_INDEX_BUFFER_BIT aliases the plain index bit.
        (device_address ? VK_BUFFER_USAGE_2_SHADER_DEVICE_ADDRESS_BIT : 0u) |
        (acceleration_structure_input ? VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR : 0u));
}

    // Validates the device before the `Buffer` base initializer dereferences it.
    [[nodiscard]] bool device_address_capable(Device *device) noexcept {
        LUISA_ASSERT(device != nullptr,
                     "A Vulkan policy buffer needs a live device.");
        return device->enable_device_address();
    }

    // Physical size of a backend buffer: the final storage word must be fully
    // backed so that a word-addressed descriptor may name it. `DefaultBuffer`
    // pads identically and `storage_buffer_descriptor_range` asserts that the
    // buffer proves at least the padded range, so a dedicated allocation must
    // not be the only buffer flavour that skips the padding.
    [[nodiscard]] size_t word_addressable_size(size_t logical_size) noexcept {
        LUISA_ASSERT(logical_size <= std::numeric_limits<size_t>::max() - 3u,
                     "Vulkan buffer size {} cannot be rounded to a storage word.",
                     logical_size);
        return (logical_size + 3u) & ~size_t{3u};
    }

}// namespace vk_memory_policy_detail

VkBufferUsageFlags storage_buffer_usage_flags(
    Device *device, bool used_as_accel, VkBufferUsageFlagBits extra_bit) noexcept {
    LUISA_ASSERT(device != nullptr, "A Vulkan buffer needs a live device.");
    return vk_memory_policy_detail::make_storage_buffer_usage(
        device->enable_device_address(),
        device->enable_raytracing() && used_as_accel, extra_bit);
}

PolicyBuffer::PolicyBuffer(
    Device *device, size_t byte_size, VkBufferUsageFlags usage,
    VkBufferUsageFlags2KHR usage2, bool host_visible,
    MemoryInitPolicy init) noexcept
    : Buffer{device, byte_size,
             vk_memory_policy_detail::word_addressable_size(byte_size),
             vk_memory_policy_detail::device_address_capable(device)} {
    LUISA_ASSERT(device != nullptr, "A Vulkan buffer needs a live device.");
    // The dedicated allocation backs the *padded* range, so the base class can
    // prove every byte a word-addressed descriptor may name, exactly like the
    // VMA-backed `DefaultBuffer`.
    auto allocation = device->allocator().allocate_native_buffer(
        addressable_byte_size(), usage, usage2, host_visible, init);
    _vk_buffer = allocation.buffer;
    _memory = allocation.memory;
    _allocation_size = allocation.allocation_size;
    _driver_zeroed = allocation.driver_zeroed;
}

PolicyBuffer::~PolicyBuffer() noexcept {
    if (_vk_buffer == VK_NULL_HANDLE && _memory == VK_NULL_HANDLE) { return; }
    NativeBufferAllocation allocation{
        .buffer = _vk_buffer,
        .memory = _memory,
        .allocation_size = _allocation_size,
        .driver_zeroed = _driver_zeroed};
    device()->allocator().destroy_native_buffer(allocation);
    _vk_buffer = VK_NULL_HANDLE;
    _memory = VK_NULL_HANDLE;
}

bool zero_initialize_device_memory_available(const Device *device) noexcept {
    return device != nullptr &&
           device->optional_extension_enabled(
               detail::OptionalExtension::ZERO_INITIALIZE_DEVICE_MEMORY);
}

ZeroInitializedBufferCreation create_zero_initialized_buffer(
    Device *device, const luisa::compute::Type *element, size_t elem_count,
    bool host_visible) noexcept {
    using namespace luisa::compute;
    LUISA_ASSERT(device != nullptr,
                 "Zero-initialized buffer creation needs a live Vulkan device.");
    LUISA_ASSERT(element != nullptr,
                 "Zero-initialized buffer creation needs an element type.");
    LUISA_ASSERT(!element->is_custom() && elem_count > 0u &&
                     elem_count <= std::numeric_limits<size_t>::max() / element->size(),
                 "Cannot create a zero-initialized Vulkan buffer for element '{}' "
                 "and {} elements.",
                 element->description(), elem_count);
    auto byte_size = element->size() * elem_count;
    auto usage = storage_buffer_usage_flags(device, false, static_cast<VkBufferUsageFlagBits>(0u));
    auto *buffer = new PolicyBuffer(
        device, byte_size, usage, 0u, host_visible,
        MemoryInitPolicy::kZeroInitialized);
    if (!buffer->driver_zeroed() && host_visible) {
        // The driver could not (or was not allowed to) zero the memory, so the
        // zero-content contract is preserved from the host instead. Host visible
        // memory is the only memory the backend may map here. The whole
        // allocation is cleared, including the padded final storage word, so a
        // word-addressed read of the last element cannot observe stale bytes.
        auto vk_device = device->logic_device();
        void *mapped = nullptr;
        VK_CHECK_RESULT(vkMapMemory(
            vk_device, buffer->device_memory(), 0u, VK_WHOLE_SIZE, 0u, &mapped));
        memset(mapped, 0, static_cast<size_t>(buffer->allocation_size()));
        VkMappedMemoryRange range{
            .sType = VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE,
            .pNext = nullptr,
            .memory = buffer->device_memory(),
            .offset = 0u,
            .size = VK_WHOLE_SIZE};
        VK_CHECK_RESULT(vkFlushMappedMemoryRanges(vk_device, 1u, &range));
        vkUnmapMemory(vk_device, buffer->device_memory());
    }
    BufferCreationInfo info{};
    info.handle = reinterpret_cast<uint64_t>(buffer);
    info.native_handle = buffer->vk_buffer();
    info.element_stride = element->size();
    info.total_size_bytes = buffer->byte_size();
    return {.buffer = info, .driver_zeroed = buffer->driver_zeroed()};
}

luisa::compute::ResourceCreationInfo create_texture_with_tiling_policy(
    Device *device, luisa::compute::PixelFormat format, uint dimension,
    uint width, uint height, uint depth, uint mipmap_levels,
    bool simultaneous_access, bool allow_raster_target,
    detail::ImageTilingPolicy tiling_policy) noexcept {
    LUISA_ASSERT(device != nullptr,
                 "Tiling-controlled texture creation needs a live Vulkan device.");
    auto *texture = new Texture(
        device, dimension, format,
        uint3(width, height, depth), mipmap_levels,
        simultaneous_access, allow_raster_target, tiling_policy);
    return luisa::compute::ResourceCreationInfo{
        .handle = reinterpret_cast<uint64_t>(texture),
        .native_handle = texture->vk_image()};
}

}// namespace lc::vk
