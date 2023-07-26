#include <luisa/core/logging.h>

#include "metal_device.h"
#include "metal_buffer.h"
#include "metal_pinned_memory.h"

namespace luisa::compute::metal {

MetalPinnedMemoryExt::MetalPinnedMemoryExt(MetalDevice *device) noexcept
    : _device{device} {}

DeviceInterface *MetalPinnedMemoryExt::device() const noexcept {
    return _device;
}

BufferCreationInfo MetalPinnedMemoryExt::_pin_host_memory(
    const Type *elem_type, size_t elem_count,
    void *host_ptr, const PinnedMemoryOption &option) noexcept {
    LUISA_ERROR_WITH_LOCATION("Not implemented.");
}

BufferCreationInfo MetalPinnedMemoryExt::_allocate_pinned_memory(
    const Type *elem_type, size_t elem_count,
    const PinnedMemoryOption &option) noexcept {
    LUISA_ERROR_WITH_LOCATION("Not implemented.");
}

}// namespace luisa::compute::metal
