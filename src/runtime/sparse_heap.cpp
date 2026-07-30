#include <luisa/runtime/sparse_heap.h>
#include <luisa/runtime/device.h>
#include <luisa/runtime/rhi/device_interface.h>
#include <luisa/core/logging.h>

#include "sparse_heap_provenance.h"

namespace luisa::compute {

namespace detail {

LUISA_RUNTIME_API void check_sparse_heap_provenance(
    const Resource &resource,
    const Resource &heap) noexcept {
    auto status = validate_sparse_heap_provenance(
        heap.valid(), resource.device(), heap.device());
    LUISA_ASSERT(
        status == SparseHeapProvenanceStatus::SUCCESS,
        "Sparse resource and heap are incompatible: {}.",
        sparse_heap_provenance_status_name(status));
}

}// namespace detail

SparseBufferHeap::SparseBufferHeap(DeviceInterface *device, size_t byte_size) noexcept
    : Resource{device, Tag::SPARSE_BUFFER_HEAP, device->allocate_sparse_buffer_heap(byte_size)} {
}
SparseBufferHeap::~SparseBufferHeap() noexcept {
    if (*this) { device()->deallocate_sparse_buffer_heap(handle()); }
}
SparseTextureHeap::SparseTextureHeap(DeviceInterface *device, size_t byte_size) noexcept
    : Resource{device, Tag::SPARSE_TEXTURE_HEAP, device->allocate_sparse_texture_heap(byte_size)} {
}
SparseTextureHeap::~SparseTextureHeap() noexcept {
    if (*this) { device()->deallocate_sparse_texture_heap(handle()); }
}
SparseBufferHeap Device::allocate_sparse_buffer_heap(size_t byte_size) noexcept {
    return SparseBufferHeap{_impl.get(), byte_size};
}
SparseTextureHeap Device::allocate_sparse_texture_heap(size_t byte_size) noexcept {
    return SparseTextureHeap{_impl.get(), byte_size};
}
}// namespace luisa::compute
