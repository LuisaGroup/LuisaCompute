//
// Created by Mike on 7/30/2021.
//

#include <runtime/heap.h>
#include <backends/cuda/cuda_heap.h>
#include <backends/cuda/cuda_device.h>

namespace luisa::compute::cuda {

CUDAHeap::CUDAHeap(CUDADevice *device, size_t capacity) noexcept
    : _device{device} {
    CUmemPoolProps props{
        .allocType = CU_MEM_ALLOCATION_TYPE_PINNED,
        .handleTypes = CU_MEM_HANDLE_TYPE_NONE,
        .location = CU_MEM_LOCATION_TYPE_DEVICE,
        .win32SecurityAttributes = nullptr,
        .reserved = {}};
    LUISA_CHECK_CUDA(cuMemPoolCreate(&_handle, &props));
    LUISA_CHECK_CUDA(cuMemPoolSetAttribute(_handle, CU_MEMPOOL_ATTR_RELEASE_THRESHOLD, &capacity));
    _items.resize(Heap::slot_count);
    LUISA_CHECK_CUDA(cuMemAllocAsync(&_desc_array, sizeof(Item) * Heap::slot_count, nullptr));
}

CUDAHeap::~CUDAHeap() noexcept {
    for (auto b : _active_buffers) { delete b; }
    LUISA_CHECK_CUDA(cuMemFreeAsync(_desc_array, nullptr));
    LUISA_CHECK_CUDA(cuMemPoolDestroy(_handle));
}

CUDABuffer *CUDAHeap::allocate_buffer(size_t size, size_t index) noexcept {
    //    auto buffer_ptr = _device->with_locked([d = _device->handle().device(), handle = _handle, size]{
    //        CUmemoryPool pool = nullptr;
    //        CUdeviceptr buffer = 0u;
    //        LUISA_CHECK_CUDA(cuDeviceGetMemPool(&pool, d));
    //        LUISA_CHECK_CUDA(cuDeviceSetMemPool(d, handle));
    //        LUISA_CHECK_CUDA(cuMemAllocAsync(&buffer, size, nullptr));
    //        LUISA_CHECK_CUDA(cuDeviceSetMemPool(d, pool));
    //        return buffer;
    //    });
    CUdeviceptr buffer_ptr = 0u;
    LUISA_CHECK_CUDA(cuMemAllocFromPoolAsync(&buffer_ptr, size, _handle, nullptr));
    auto buffer = new CUDABuffer{this, index};
    std::scoped_lock lock{_mutex};
    _items[index].buffer = buffer_ptr;
    _dirty = true;
    _active_buffers.emplace(buffer);
    return buffer;
}

void CUDAHeap::destroy_buffer(CUDABuffer *buffer) noexcept {
    auto index = buffer->index();
    auto address = _items[index].buffer;
    delete buffer;
    LUISA_CHECK_CUDA(cuMemFreeAsync(address, nullptr));
    std::scoped_lock lock{_mutex};
    _items[index].buffer = 0u;
    _active_buffers.erase(buffer);
}

size_t CUDAHeap::memory_usage() const noexcept {
    size_t usage = 0u;
    LUISA_CHECK_CUDA(cuMemPoolGetAttribute(_handle, CU_MEMPOOL_ATTR_USED_MEM_CURRENT, &usage));
    return usage;
}

CUdeviceptr CUDAHeap::descriptor_array() const noexcept {
    std::scoped_lock lock{_mutex};
    if (_dirty) {
        LUISA_CHECK_CUDA(cuMemcpyHtoDAsync(_desc_array, _items.data(), sizeof(Item) * Heap::slot_count, nullptr));
        _dirty = false;
    }
    return _desc_array;
}

}// namespace luisa::compute::cuda
