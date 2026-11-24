#include <luisa/core/logging.h>

#include "pinned_memory_ext.h"
#include "device.h"
#include "upload_buffer.h"
#include "readback_buffer.h"

namespace lc::vk {

BufferCreationInfo VkPinnedMemoryExt::_pin_host_memory(
    const Type *elem_type, size_t elem_count,
    void *host_ptr, const PinnedMemoryOption &option) noexcept {
    LUISA_ERROR("VK backend can not pin host memory.");
    return BufferCreationInfo::make_invalid();
}

BufferCreationInfo VkPinnedMemoryExt::_allocate_pinned_memory(
    const Type *elem_type, size_t elem_count,
    const PinnedMemoryOption &option) noexcept {
    BufferCreationInfo info{};
    info.element_stride = (elem_type == Type::of<void>()) ? 1 : elem_type->size();
    auto size_bytes = info.element_stride * elem_count;
    if (option.write_combined) {
        auto ptr = new UploadBuffer(
            _device, size_bytes,
            option.prefer_host ? AccessType::kHostSequentialUpload : AccessType::kUpload);
        info.handle = reinterpret_cast<uint64_t>(ptr);
        info.native_handle = ptr->mapped_ptr();
    } else if (option.host_cached_upload) {
        auto ptr = new UploadBuffer(_device, size_bytes, AccessType::kHostCachedUpload);
        info.handle = reinterpret_cast<uint64_t>(ptr);
        info.native_handle = ptr->mapped_ptr();
    } else {
        auto ptr = new ReadbackBuffer(_device, size_bytes);
        info.handle = reinterpret_cast<uint64_t>(ptr);
        info.native_handle = ptr->mapped_ptr();
    }
    info.total_size_bytes = size_bytes;

    return info;
}
void VkPinnedMemoryExt::flush_range(
    uint64_t buffer_handle,
    uint64_t begin,
    uint64_t end) const noexcept {
    auto ptr = reinterpret_cast<Buffer *>(buffer_handle);
    ptr->flush_range(begin, end);
}
DeviceInterface *VkPinnedMemoryExt::device() const noexcept {
    return _device;
}
}// namespace lc::vk