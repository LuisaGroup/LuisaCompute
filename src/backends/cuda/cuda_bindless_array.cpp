//
// Created by Mike on 7/30/2021.
//

#include <runtime/bindless_array.h>
#include <backends/cuda/cuda_stream.h>
#include <backends/cuda/cuda_device.h>
#include <backends/cuda/cuda_bindless_array.h>

namespace luisa::compute::cuda {

[[nodiscard]] static auto cuda_texture_descriptor(Sampler sampler) noexcept {
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
            texture_desc.maxMipmapLevelClamp = 999.0f;
            break;
        case Sampler::Filter::ANISOTROPIC:
            texture_desc.filterMode = CU_TR_FILTER_MODE_LINEAR;
            texture_desc.mipmapFilterMode = CU_TR_FILTER_MODE_LINEAR;
            texture_desc.maxAnisotropy = 16;
            texture_desc.maxMipmapLevelClamp = 999.0f;
            break;
    }
    texture_desc.flags = CU_TRSF_NORMALIZED_COORDINATES;
    return texture_desc;
}

void CUDABindlessArray::_retain(luisa::unordered_map<uint64_t, size_t> &resources, uint64_t r) noexcept {
    if (auto iter = resources.try_emplace(r, 1u); !iter.second) {
        iter.first->second++;
    }
}

void CUDABindlessArray::_release(luisa::unordered_map<uint64_t, size_t> &resources, uint64_t r) noexcept {
    if (auto iter = resources.find(r); iter == resources.end()) [[unlikely]] {
        LUISA_WARNING_WITH_LOCATION(
            "Removing non-existent resource in bindless array");
    } else {
        if (--iter->second == 0u) {
            resources.erase(iter);
        }
    }
}

bool CUDABindlessArray::has_buffer(CUdeviceptr buffer) const noexcept {
    return _buffers.contains(buffer);
}

bool CUDABindlessArray::has_array(CUDAMipmapArray *array) const noexcept {
    return _arrays.contains(reinterpret_cast<uint64_t>(array));
}

void CUDABindlessArray::remove_buffer(size_t index) noexcept {
    auto buffer = _slots[index].buffer;
    if (buffer != 0u) [[likely]] {
        _release_buffer(_slots[index].origin);
        _slots[index].buffer = 0u;
        _slots[index].origin = 0u;
    }
}

void CUDABindlessArray::remove_tex2d(size_t index) noexcept {
    if (auto tex = _slots[index].tex2d) [[likely]] {
        LUISA_CHECK_CUDA(cuTexObjectDestroy(tex));
        auto iter = _tex_to_array.find(tex);
        _release_array(iter->second);
        _tex_to_array.erase(iter);
        _slots[index].tex2d = 0u;
    }
}

void CUDABindlessArray::remove_tex3d(size_t index) noexcept {
    if (auto tex = _slots[index].tex3d) [[likely]] {
        LUISA_CHECK_CUDA(cuTexObjectDestroy(tex));
        auto iter = _tex_to_array.find(tex);
        _release_array(iter->second);
        _tex_to_array.erase(iter);
        _slots[index].tex3d = 0u;
    }
}

void CUDABindlessArray::emplace_buffer(size_t index, CUdeviceptr buffer, size_t offset) noexcept {
    if (auto o = _slots[index].origin) [[unlikely]] {
        _release_buffer(o);
    }
    _slots[index].buffer = buffer + offset;
    _slots[index].origin = buffer;
    _retain_buffer(buffer);
}

void CUDABindlessArray::emplace_tex2d(size_t index, CUDAMipmapArray *array, Sampler sampler) noexcept {
    CUDA_RESOURCE_DESC res_desc{};
    res_desc.resType = CU_RESOURCE_TYPE_MIPMAPPED_ARRAY;
    res_desc.res.mipmap.hMipmappedArray = array->handle();
    auto tex_desc = cuda_texture_descriptor(sampler);
    CUtexObject texture;
    LUISA_CHECK_CUDA(cuTexObjectCreate(&texture, &res_desc, &tex_desc, nullptr));
    if (auto t = _slots[index].tex2d) [[unlikely]] {
        auto iter = _tex_to_array.find(t);
        _release_array(iter->second);
        _tex_to_array.erase(iter);
    }
    _slots[index].tex2d = texture;
    _tex_to_array.emplace(texture, array);
    _retain_array(array);
}

void CUDABindlessArray::emplace_tex3d(size_t index, CUDAMipmapArray *array, Sampler sampler) noexcept {
    CUDA_RESOURCE_DESC res_desc{};
    res_desc.resType = CU_RESOURCE_TYPE_MIPMAPPED_ARRAY;
    res_desc.res.mipmap.hMipmappedArray = array->handle();
    auto tex_desc = cuda_texture_descriptor(sampler);
    CUtexObject texture;
    LUISA_CHECK_CUDA(cuTexObjectCreate(&texture, &res_desc, &tex_desc, nullptr));
    if (auto t = _slots[index].tex3d; t != 0u) [[unlikely]] {
        auto iter = _tex_to_array.find(t);
        _release_array(iter->second);
        _tex_to_array.erase(iter);
    }
    _slots[index].tex3d = texture;
    _tex_to_array.emplace(texture, array);
    _retain_array(array);
}

CUDABindlessArray::~CUDABindlessArray() noexcept {
    for (auto item : _tex_to_array) {
        LUISA_CHECK_CUDA(cuTexObjectDestroy(item.first));
    }
    LUISA_CHECK_CUDA(cuMemFreeAsync(_handle, nullptr));
}

CUDABindlessArray::CUDABindlessArray(CUdeviceptr handle, size_t capacity) noexcept
    : _handle{handle}, _slots(capacity, Item{}) {}

}// namespace luisa::compute::cuda
