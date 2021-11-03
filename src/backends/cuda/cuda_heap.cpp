//
// Created by Mike on 7/30/2021.
//

#include <runtime/bindless_array.h>
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
    LUISA_CHECK_CUDA(cuMemAllocAsync(&_desc_array, sizeof(Item) * Heap::slot_count, nullptr));
    _items.resize(Heap::slot_count);
}

CUDAHeap::~CUDAHeap() noexcept {
    for (auto b : _active_buffers) { delete b; }
    for (auto t : _active_textures) {
        LUISA_CHECK_CUDA(cuMipmappedArrayDestroy(t->mip_array()));
        LUISA_CHECK_CUDA(cuTexObjectDestroy(t->handle()));
        delete t;
    }
    LUISA_CHECK_CUDA(cuMemFreeAsync(_desc_array, nullptr));
    LUISA_CHECK_CUDA(cuMemPoolDestroy(_handle));
}

CUDABuffer *CUDAHeap::allocate_buffer(size_t index, size_t size) noexcept {
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

CUDATexture *CUDAHeap::allocate_texture(size_t index, PixelFormat format, uint dim, uint3 size, uint mip_levels, Sampler sampler) noexcept {

    CUDA_ARRAY3D_DESCRIPTOR array_desc{};
    array_desc.Width = size.x;
    array_desc.Height = size.y;
    array_desc.Depth = size.z;
    switch (format) {
        case PixelFormat::R8SInt:
            array_desc.Format = CU_AD_FORMAT_SIGNED_INT8;
            array_desc.NumChannels = 1;
            break;
        case PixelFormat::R8UInt:
        case PixelFormat::R8UNorm:
            array_desc.Format = CU_AD_FORMAT_UNSIGNED_INT8;
            array_desc.NumChannels = 1;
            break;
        case PixelFormat::RG8SInt:
            array_desc.Format = CU_AD_FORMAT_SIGNED_INT8;
            array_desc.NumChannels = 2;
            break;
        case PixelFormat::RG8UInt:
        case PixelFormat::RG8UNorm:
            array_desc.Format = CU_AD_FORMAT_UNSIGNED_INT8;
            array_desc.NumChannels = 2;
            break;
        case PixelFormat::RGBA8SInt:
            array_desc.Format = CU_AD_FORMAT_SIGNED_INT8;
            array_desc.NumChannels = 4;
            break;
        case PixelFormat::RGBA8UInt:
        case PixelFormat::RGBA8UNorm:
            array_desc.Format = CU_AD_FORMAT_UNSIGNED_INT8;
            array_desc.NumChannels = 4;
            break;
        case PixelFormat::R16SInt:
            array_desc.Format = CU_AD_FORMAT_SIGNED_INT16;
            array_desc.NumChannels = 1;
            break;
        case PixelFormat::R16UInt:
        case PixelFormat::R16UNorm:
            array_desc.Format = CU_AD_FORMAT_UNSIGNED_INT16;
            array_desc.NumChannels = 1;
            break;
        case PixelFormat::RG16SInt:
            array_desc.Format = CU_AD_FORMAT_SIGNED_INT16;
            array_desc.NumChannels = 2;
            break;
        case PixelFormat::RG16UInt:
        case PixelFormat::RG16UNorm:
            array_desc.Format = CU_AD_FORMAT_UNSIGNED_INT16;
            array_desc.NumChannels = 2;
            break;
        case PixelFormat::RGBA16SInt:
            array_desc.Format = CU_AD_FORMAT_SIGNED_INT16;
            array_desc.NumChannels = 4;
            break;
        case PixelFormat::RGBA16UInt:
        case PixelFormat::RGBA16UNorm:
            array_desc.Format = CU_AD_FORMAT_UNSIGNED_INT16;
            array_desc.NumChannels = 4;
            break;
        case PixelFormat::R32SInt:
            array_desc.Format = CU_AD_FORMAT_SIGNED_INT32;
            array_desc.NumChannels = 1;
            break;
        case PixelFormat::R32UInt:
            array_desc.Format = CU_AD_FORMAT_UNSIGNED_INT32;
            array_desc.NumChannels = 1;
            break;
        case PixelFormat::RG32SInt:
            array_desc.Format = CU_AD_FORMAT_SIGNED_INT32;
            array_desc.NumChannels = 2;
            break;
        case PixelFormat::RG32UInt:
            array_desc.Format = CU_AD_FORMAT_UNSIGNED_INT32;
            array_desc.NumChannels = 2;
            break;
        case PixelFormat::RGBA32SInt:
            array_desc.Format = CU_AD_FORMAT_SIGNED_INT32;
            array_desc.NumChannels = 4;
            break;
        case PixelFormat::RGBA32UInt:
            array_desc.Format = CU_AD_FORMAT_UNSIGNED_INT32;
            array_desc.NumChannels = 4;
            break;
        case PixelFormat::R16F:
            array_desc.Format = CU_AD_FORMAT_HALF;
            array_desc.NumChannels = 1;
            break;
        case PixelFormat::RG16F:
            array_desc.Format = CU_AD_FORMAT_HALF;
            array_desc.NumChannels = 2;
            break;
        case PixelFormat::RGBA16F:
            array_desc.Format = CU_AD_FORMAT_HALF;
            array_desc.NumChannels = 4;
            break;
        case PixelFormat::R32F:
            array_desc.Format = CU_AD_FORMAT_FLOAT;
            array_desc.NumChannels = 1;
            break;
        case PixelFormat::RG32F:
            array_desc.Format = CU_AD_FORMAT_FLOAT;
            array_desc.NumChannels = 2;
            break;
        case PixelFormat::RGBA32F:
            array_desc.Format = CU_AD_FORMAT_FLOAT;
            array_desc.NumChannels = 4;
            break;
    }
    CUmipmappedArray array_handle{nullptr};
    LUISA_CHECK_CUDA(cuMipmappedArrayCreate(&array_handle, &array_desc, mip_levels));

    CUDA_RESOURCE_DESC resource_desc{};
    resource_desc.resType = CU_RESOURCE_TYPE_MIPMAPPED_ARRAY;
    resource_desc.res.mipmap.hMipmappedArray = array_handle;

    CUDA_TEXTURE_DESC texture_desc{};
    switch (sampler.address()) {
        case Sampler::Address::EDGE:
            texture_desc.addressMode[0] = CU_TR_ADDRESS_MODE_CLAMP;
            texture_desc.addressMode[1] = CU_TR_ADDRESS_MODE_CLAMP;
            texture_desc.addressMode[2] = CU_TR_ADDRESS_MODE_CLAMP;
            break;
        case Sampler::Address::REPEAT:
            texture_desc.addressMode[0] = CU_TR_ADDRESS_MODE_WRAP;
            texture_desc.addressMode[1] = CU_TR_ADDRESS_MODE_WRAP;
            texture_desc.addressMode[2] = CU_TR_ADDRESS_MODE_WRAP;
            break;
        case Sampler::Address::MIRROR:
            texture_desc.addressMode[0] = CU_TR_ADDRESS_MODE_MIRROR;
            texture_desc.addressMode[1] = CU_TR_ADDRESS_MODE_MIRROR;
            texture_desc.addressMode[2] = CU_TR_ADDRESS_MODE_MIRROR;
            break;
        case Sampler::Address::ZERO:
            texture_desc.addressMode[0] = CU_TR_ADDRESS_MODE_BORDER;
            texture_desc.addressMode[1] = CU_TR_ADDRESS_MODE_BORDER;
            texture_desc.addressMode[2] = CU_TR_ADDRESS_MODE_BORDER;
            break;
    }
    switch (sampler.filter()) {
        case Sampler::Filter::POINT:
            texture_desc.filterMode = CU_TR_FILTER_MODE_POINT;
            texture_desc.mipmapFilterMode = CU_TR_FILTER_MODE_POINT;
            break;
        case Sampler::Filter::BILINEAR:
            texture_desc.filterMode = CU_TR_FILTER_MODE_LINEAR;
            texture_desc.mipmapFilterMode = CU_TR_FILTER_MODE_POINT;
            break;
        case Sampler::Filter::TRILINEAR:
            texture_desc.filterMode = CU_TR_FILTER_MODE_LINEAR;
            texture_desc.mipmapFilterMode = CU_TR_FILTER_MODE_LINEAR;
            texture_desc.maxMipmapLevelClamp = static_cast<float>(mip_levels - 1u);
            break;
        case Sampler::Filter::ANISOTROPIC:
            texture_desc.filterMode = CU_TR_FILTER_MODE_LINEAR;
            texture_desc.mipmapFilterMode = CU_TR_FILTER_MODE_LINEAR;
            texture_desc.maxAnisotropy = 16;
            texture_desc.maxMipmapLevelClamp = static_cast<float>(mip_levels - 1u);
            break;
    }
    switch (format) {
        case PixelFormat::R8SInt:
        case PixelFormat::R8UInt:
        case PixelFormat::RG8SInt:
        case PixelFormat::RG8UInt:
        case PixelFormat::RGBA8SInt:
        case PixelFormat::RGBA8UInt:
        case PixelFormat::R16SInt:
        case PixelFormat::R16UInt:
        case PixelFormat::RG16SInt:
        case PixelFormat::RG16UInt:
        case PixelFormat::RGBA16SInt:
        case PixelFormat::RGBA16UInt:
        case PixelFormat::R32SInt:
        case PixelFormat::R32UInt:
        case PixelFormat::RG32SInt:
        case PixelFormat::RG32UInt:
        case PixelFormat::RGBA32SInt:
        case PixelFormat::RGBA32UInt:
            texture_desc.flags = CU_TRSF_NORMALIZED_COORDINATES | CU_TRSF_READ_AS_INTEGER;
            break;
        case PixelFormat::R8UNorm:
        case PixelFormat::RG8UNorm:
        case PixelFormat::RGBA8UNorm:
        case PixelFormat::R16UNorm:
        case PixelFormat::RG16UNorm:
        case PixelFormat::RGBA16UNorm:
        case PixelFormat::R16F:
        case PixelFormat::RG16F:
        case PixelFormat::RGBA16F:
        case PixelFormat::R32F:
        case PixelFormat::RG32F:
        case PixelFormat::RGBA32F:
            texture_desc.flags = CU_TRSF_NORMALIZED_COORDINATES;
            break;
    }
    CUtexObject texture_handle;
    LUISA_CHECK_CUDA(cuTexObjectCreate(&texture_handle, &resource_desc, &texture_desc, nullptr));

    auto texture = new CUDATexture{this, index, array_handle, dim};
    std::scoped_lock lock{_mutex};
    _items[index].texture = texture_handle;
    _dirty = true;
    _active_textures.emplace(texture);
    return texture;
}

void CUDAHeap::destroy_texture(CUDATexture *texture) noexcept {
    auto index = texture->index();
    auto texture_handle = _items[index].texture;
    LUISA_CHECK_CUDA(cuMipmappedArrayDestroy(texture->mip_array()));
    LUISA_CHECK_CUDA(cuTexObjectDestroy(texture_handle));
    delete texture;
    std::scoped_lock lock{_mutex};
    _items[index].texture = 0u;
    _active_textures.erase(texture);
}

}// namespace luisa::compute::cuda