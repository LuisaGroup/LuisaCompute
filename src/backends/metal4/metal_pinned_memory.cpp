#include <luisa/core/logging.h>

#include "metal_device.h"
#include "metal_buffer.h"
#include "metal_dstorage.h"
#include "metal_pinned_memory.h"

namespace luisa::compute::metal {

MetalPinnedMemoryExt::MetalPinnedMemoryExt(MetalDevice *device) noexcept
    : _device{device} {}

DeviceInterface *MetalPinnedMemoryExt::device() const noexcept {
    return _device;
}

namespace {

[[nodiscard]] size_t luisa_metal_pinned_memory_ext_get_buffer_size(const Type *elem_type, size_t n) noexcept {
    LUISA_ASSERT(elem_type == nullptr || !elem_type->is_custom(),
                 "Custom types are not supported in pinned memory.");
    auto stride = elem_type == nullptr || elem_type == Type::of<void>() ?
                      1u :
                      elem_type->size();
    return stride * n;
}

[[nodiscard]] size_t luisa_metal_pinned_memory_ext_get_buffer_options(const PinnedMemoryOption &option) noexcept {
    auto buffer_options = MTL::ResourceHazardTrackingModeTracked | MTL::ResourceStorageModeShared;
    if (option.write_combined) { buffer_options |= MTL::ResourceCPUCacheModeWriteCombined; }
    return buffer_options;
}

}// namespace

BufferCreationInfo MetalPinnedMemoryExt::_pin_host_memory(
    const Type *elem_type, size_t elem_count,
    void *host_ptr, const PinnedMemoryOption &option) noexcept {
    auto buffer_size = luisa_metal_pinned_memory_ext_get_buffer_size(elem_type, elem_count);
    MetalPinnedMemory memory{_device->handle(), host_ptr, buffer_size, option.write_combined};
    LUISA_ASSERT(memory.valid(), "Failed to pin host memory.");
    auto elem_stride = buffer_size / elem_count;
    LUISA_ASSERT(memory.device_buffer_offset() % elem_stride == 0u, "Invalid buffer offset.");
    auto offset_count = memory.device_buffer_offset() / elem_stride;
    auto info = _device->create_buffer(elem_type, offset_count + elem_count, memory.device_buffer());
    info.native_handle = memory.device_buffer()->contents();
    return info;
}

BufferCreationInfo MetalPinnedMemoryExt::_allocate_pinned_memory(
    const Type *elem_type, size_t elem_count,
    const PinnedMemoryOption &option) noexcept {
    auto buffer_size = luisa_metal_pinned_memory_ext_get_buffer_size(elem_type, elem_count);
    auto buffer_options = luisa_metal_pinned_memory_ext_get_buffer_options(option);
    auto buffer = NS::TransferPtr(_device->handle()->newBuffer(buffer_size, buffer_options));
    auto info = _device->create_buffer(elem_type, elem_count, buffer.get());
    // Pinned memory extension expects native_handle to be the host pointer
    info.native_handle = buffer->contents();
    return info;
}

}// namespace luisa::compute::metal
