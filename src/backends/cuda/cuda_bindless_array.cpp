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
    if (auto buffer = _buffer_slots[index]) [[likely]] {
        _release_buffer(_buffer_handles[index]);
        _buffer_slots[index] = 0u;
        _buffer_handles[index] = 0u;
    }
}

void CUDABindlessArray::remove_tex2d(size_t index) noexcept {
    if (auto tex = _tex2d_slots[index]) [[likely]] {
        LUISA_CHECK_CUDA(cuTexObjectDestroy(tex));
        auto iter = _tex_to_array.find(tex);
        _release_array(iter->second);
        _tex_to_array.erase(iter);
        _tex2d_slots[index] = 0u;
    }
}

void CUDABindlessArray::remove_tex3d(size_t index) noexcept {
    if (auto tex = _tex3d_slots[index]) [[likely]] {
        LUISA_CHECK_CUDA(cuTexObjectDestroy(tex));
        auto iter = _tex_to_array.find(tex);
        _release_array(iter->second);
        _tex_to_array.erase(iter);
        _tex3d_slots[index] = 0u;
    }
}

void CUDABindlessArray::emplace_buffer(size_t index, CUdeviceptr buffer, size_t offset) noexcept {
    if (auto o = _buffer_handles[index]) [[unlikely]] {
        _release_buffer(o);
    }
    _buffer_handles[index] = buffer;
    _buffer_slots[index] = buffer + offset;
    _buffer_dirty_range.mark(index);
    _retain_buffer(buffer);
}

void CUDABindlessArray::emplace_tex2d(size_t index, CUDAMipmapArray *array, Sampler sampler) noexcept {
    CUDA_RESOURCE_DESC res_desc{};
    if (array->levels() == 1u) {
        res_desc.resType = CU_RESOURCE_TYPE_ARRAY;
        res_desc.res.array.hArray = reinterpret_cast<CUarray>(array->handle());
    } else {
        res_desc.resType = CU_RESOURCE_TYPE_MIPMAPPED_ARRAY;
        res_desc.res.mipmap.hMipmappedArray = reinterpret_cast<CUmipmappedArray>(array->handle());
    }
    auto tex_desc = cuda_texture_descriptor(sampler);
    CUtexObject texture;
    LUISA_CHECK_CUDA(cuTexObjectCreate(&texture, &res_desc, &tex_desc, nullptr));
    if (auto t = _tex2d_slots[index]) [[unlikely]] {
        LUISA_CHECK_CUDA(cuTexObjectDestroy(t));
        auto iter = _tex_to_array.find(t);
        _release_array(iter->second);
        _tex_to_array.erase(iter);
    }
    _tex2d_slots[index] = texture;
    _tex2d_sizes[index] = {
        static_cast<unsigned short>(array->size().x),
        static_cast<unsigned short>(array->size().y)};
    _tex_to_array.emplace(texture, array);
    _tex2d_dirty_range.mark(index);
    _retain_array(array);
}

void CUDABindlessArray::emplace_tex3d(size_t index, CUDAMipmapArray *array, Sampler sampler) noexcept {
    CUDA_RESOURCE_DESC res_desc{};
    if (array->levels() == 1u) {
        res_desc.resType = CU_RESOURCE_TYPE_ARRAY;
        res_desc.res.array.hArray = reinterpret_cast<CUarray>(array->handle());
    } else {
        res_desc.resType = CU_RESOURCE_TYPE_MIPMAPPED_ARRAY;
        res_desc.res.mipmap.hMipmappedArray = reinterpret_cast<CUmipmappedArray>(array->handle());
    }
    auto tex_desc = cuda_texture_descriptor(sampler);
    CUtexObject texture;
    LUISA_CHECK_CUDA(cuTexObjectCreate(&texture, &res_desc, &tex_desc, nullptr));
    if (auto t = _tex3d_slots[index]) [[unlikely]] {
        LUISA_CHECK_CUDA(cuTexObjectDestroy(t));
        auto iter = _tex_to_array.find(t);
        _release_array(iter->second);
        _tex_to_array.erase(iter);
    }
    auto s = array->size();
    _tex3d_slots[index] = texture;
    _tex3d_sizes[index] = {
        static_cast<unsigned short>(s.x),
        static_cast<unsigned short>(s.y),
        static_cast<unsigned short>(s.z),
        0u};
    _tex_to_array.emplace(texture, array);
    _tex3d_dirty_range.mark(index);
    _retain_array(array);
}

CUDABindlessArray::~CUDABindlessArray() noexcept {
    for (auto item : _tex_to_array) {
        LUISA_CHECK_CUDA(cuTexObjectDestroy(item.first));
    }
    LUISA_CHECK_CUDA(cuMemFree(_handle._buffer_slots));
}

CUDABindlessArray::CUDABindlessArray(size_t capacity) noexcept
    : _buffer_handles(capacity, 0u),
      _buffer_slots(capacity, 0u),
      _tex2d_slots(capacity, 0u),
      _tex3d_slots(capacity, 0u),
      _tex2d_sizes(capacity, std::array<uint16_t, 2u>{}),
      _tex3d_sizes(capacity, std::array<uint16_t, 4u>{}) {

    constexpr auto align = [](size_t x) noexcept -> size_t {
        static constexpr auto alignment = 16u;
        return (x + alignment - 1u) / alignment * alignment;
    };

    auto buffer_slots_offset = align(0u);
    auto tex2d_slots_offset = align(buffer_slots_offset + sizeof(CUdeviceptr) * capacity);
    auto tex3d_slots_offset = align(tex2d_slots_offset + sizeof(CUtexObject) * capacity);
    auto tex2d_sizes_offset = align(tex3d_slots_offset + sizeof(CUtexObject) * capacity);
    auto tex3d_sizes_offset = align(tex2d_sizes_offset + sizeof(std::array<uint16_t, 2u>) * capacity);
    auto buffer_size = align(tex3d_sizes_offset + sizeof(std::array<uint16_t, 4u>) * capacity);

    CUdeviceptr buffer;
    LUISA_CHECK_CUDA(cuMemAlloc(&buffer, buffer_size));
    _handle._buffer_slots = buffer + buffer_slots_offset;
    _handle._tex2d_slots = buffer + tex2d_slots_offset;
    _handle._tex3d_slots = buffer + tex3d_slots_offset;
    _handle._tex2d_sizes = buffer + tex2d_sizes_offset;
    _handle._tex3d_sizes = buffer + tex3d_sizes_offset;
}

void CUDABindlessArray::upload(CUDAStream *stream) const noexcept {
    constexpr auto align = [](size_t x) noexcept -> size_t {
        static constexpr auto alignment = 16u;
        return (x + alignment - 1u) / alignment * alignment;
    };
    auto buffer_slots_upload_offset = align(0u);
    auto tex2d_slots_upload_offset = align(buffer_slots_upload_offset + sizeof(CUdeviceptr) * _buffer_dirty_range.size());
    auto tex3d_slots_upload_offset = align(tex2d_slots_upload_offset + sizeof(CUtexObject) * _tex2d_dirty_range.size());
    auto tex2d_sizes_upload_offset = align(tex3d_slots_upload_offset + sizeof(CUtexObject) * _tex3d_dirty_range.size());
    auto tex3d_sizes_upload_offset = align(tex2d_sizes_upload_offset + sizeof(std::array<uint16_t, 2u>) * _tex2d_dirty_range.size());
    auto upload_buffer_size = align(tex3d_sizes_upload_offset + sizeof(std::array<uint16_t, 4u>) * _tex3d_dirty_range.size());
    if (upload_buffer_size != 0u) {
        auto upload_buffer = stream->upload_pool().allocate(upload_buffer_size);
        constexpr auto do_upload = []<typename T>(
                                       CUdeviceptr device_buffer, const T *host_buffer,
                                       std::byte *upload_buffer, DirtyRange range) noexcept {
            if (!range.empty()) {
                auto size_bytes = sizeof(T) * range.size();
                std::memcpy(upload_buffer, host_buffer + range.offset(), size_bytes);
                LUISA_CHECK_CUDA(cuMemcpyHtoD(
                    device_buffer + sizeof(T) * range.offset(),
                    upload_buffer, size_bytes));
            }
        };
        do_upload(_handle._buffer_slots, _buffer_slots.data(), upload_buffer.address() + buffer_slots_upload_offset, _buffer_dirty_range);
        do_upload(_handle._tex2d_slots, _tex2d_slots.data(), upload_buffer.address() + tex2d_slots_upload_offset, _tex2d_dirty_range);
        do_upload(_handle._tex3d_slots, _tex3d_slots.data(), upload_buffer.address() + tex3d_slots_upload_offset, _tex3d_dirty_range);
        do_upload(_handle._tex2d_sizes, _tex2d_sizes.data(), upload_buffer.address() + tex2d_sizes_upload_offset, _tex2d_dirty_range);
        do_upload(_handle._tex3d_sizes, _tex3d_sizes.data(), upload_buffer.address() + tex3d_sizes_upload_offset, _tex3d_dirty_range);
        LUISA_CHECK_CUDA(cuLaunchHostFunc(
            stream->handle(),
            [](void *user_data) noexcept {
                auto context = static_cast<CUDARingBuffer::RecycleContext *>(user_data);
                context->recycle();
            },
            CUDARingBuffer::RecycleContext::create(
                upload_buffer, &stream->upload_pool())));
    }
    _buffer_dirty_range.clear();
    _tex2d_dirty_range.clear();
    _tex3d_dirty_range.clear();
}

}// namespace luisa::compute::cuda
