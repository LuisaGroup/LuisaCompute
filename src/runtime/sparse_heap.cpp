#include <luisa/runtime/sparse_heap.h>
#include <luisa/runtime/device.h>
#include <luisa/runtime/rhi/device_interface.h>

namespace luisa::compute {
SparseBufferHeap::SparseBufferHeap(DeviceInterface *device, size_t byte_size) noexcept
    : Resource{device, Tag::SPARSE_BUFFER_HEAP, device->allocate_sparse_buffer_heap(byte_size)} {
}
SparseBufferHeap::~SparseBufferHeap() noexcept {
    if (*this) { device()->deallocate_sparse_buffer_heap(handle()); };
}
SparseTextureHeap::SparseTextureHeap(DeviceInterface *device, size_t byte_size) noexcept
    : Resource{device, Tag::SPARSE_TEXTURE_HEAP, device->allocate_sparse_texture_heap(byte_size)} {
}
SparseTextureHeap::~SparseTextureHeap() noexcept {
    if (*this) { device()->deallocate_sparse_texture_heap(handle()); };
}
SparseBufferHeap Device::allocate_sparse_buffer_heap(size_t byte_size) {
    return SparseBufferHeap{_impl.get(), byte_size};
}
SparseTextureHeap Device::allocate_sparse_texture_heap(size_t byte_size) {
    return SparseTextureHeap{_impl.get(), byte_size};
}
}// namespace luisa::compute