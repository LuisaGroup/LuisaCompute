#include <algorithm>
#include <bit>
#include <mutex>

#include <luisa/core/stl/memory.h>

#include "cuda_device.h"
#include "cuda_error.h"
#include "cuda_texture.h"

namespace luisa::compute::cuda {

CUDA_TEXTURE_DESC cuda_texture_descriptor(Sampler sampler) noexcept {
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
        case Sampler::Filter::LINEAR_POINT:
            texture_desc.filterMode = CU_TR_FILTER_MODE_LINEAR;
            texture_desc.mipmapFilterMode = CU_TR_FILTER_MODE_POINT;
            break;
        case Sampler::Filter::LINEAR_LINEAR:
            texture_desc.filterMode = CU_TR_FILTER_MODE_LINEAR;
            texture_desc.mipmapFilterMode = CU_TR_FILTER_MODE_LINEAR;
            break;
        case Sampler::Filter::ANISOTROPIC:
            texture_desc.filterMode = CU_TR_FILTER_MODE_LINEAR;
            texture_desc.mipmapFilterMode = CU_TR_FILTER_MODE_LINEAR;
            texture_desc.maxAnisotropy = 16;
            break;
    }
    texture_desc.maxMipmapLevelClamp = 999.0f;
    texture_desc.flags = CU_TRSF_NORMALIZED_COORDINATES;
    return texture_desc;
}

CUtexObject create_cuda_texture_object(const CUDATexture *texture, Sampler sampler) noexcept {
    LUISA_ASSERT(texture != nullptr, "Null CUDA texture.");
    CUDA_RESOURCE_DESC res_desc{};
    if (texture->levels() == 1u) {
        res_desc.resType = CU_RESOURCE_TYPE_ARRAY;
        res_desc.res.array.hArray = reinterpret_cast<CUarray>(texture->handle());
    } else {
        res_desc.resType = CU_RESOURCE_TYPE_MIPMAPPED_ARRAY;
        res_desc.res.mipmap.hMipmappedArray = reinterpret_cast<CUmipmappedArray>(texture->handle());
    }
    auto tex_desc = cuda_texture_descriptor(sampler);
    if (is_srgb(texture->format())) { tex_desc.flags |= CU_TRSF_SRGB; }
    CUtexObject tex_obj{};
    LUISA_CHECK_CUDA(cuTexObjectCreate(&tex_obj, &res_desc, &tex_desc, nullptr));
    return tex_obj;
}

CUDATexture::~CUDATexture() noexcept {
    if (auto *cache = _sampler_cache.exchange(nullptr, std::memory_order_acq_rel)) {
        auto destroy_cached = [&]() noexcept {
            for (auto t : cache->objects) {
                if (t != 0ull) {
                    LUISA_CHECK_CUDA(cuTexObjectDestroy(t));
                }
            }
        };
        if (_device != nullptr) {
            _device->with_handle(destroy_cached);
            _device->deallocate_sampler_table(cache->device_table);
        } else {
            destroy_cached();
        }
        delete_with_allocator(cache);
    }
    for (auto i = 0u; i < _levels; i++) {
        if (_mip_surfaces[i]) { LUISA_CHECK_CUDA(cuSurfObjectDestroy(_mip_surfaces[i])); }
        LUISA_CHECK_CUDA(cuArrayDestroy(_mip_arrays[i]));
    }
    if (_levels > 1u) {
        LUISA_CHECK_CUDA(cuMipmappedArrayDestroy(reinterpret_cast<CUmipmappedArray>(_base_array)));
    }
}

CUarray CUDATexture::level(uint32_t i) const noexcept {
    LUISA_ASSERT(i < _levels,
                 "Invalid level {} for texture with {} level(s).",
                 i, _levels);
    return _mip_arrays[i];
}

uint64_t CUDATexture::encoded_storage(uint32_t level) const noexcept {
    LUISA_ASSERT(level < _levels,
                 "Invalid level {} for texture with {} level(s).",
                 level, _levels);
    auto w = static_cast<uint16_t>(std::max(static_cast<uint32_t>(_size[0]) >> level, 1u));
    auto h = static_cast<uint16_t>(std::max(static_cast<uint32_t>(_size[1]) >> level, 1u));
    auto d = static_cast<uint16_t>(std::max(static_cast<uint32_t>(_size[2]) >> level, 1u));
    return CUDASurface::encode_storage(
        w, h, d, static_cast<uint8_t>(storage()), static_cast<uint8_t>(level));
}

CUDATexture::SamplerCache *CUDATexture::_get_or_create_sampler_cache() const noexcept {
    if (auto *cache = _sampler_cache.load(std::memory_order_acquire)) {
        return cache;
    }
    auto *fresh = new_with_allocator<SamplerCache>();
    SamplerCache *expected = nullptr;
    if (_sampler_cache.compare_exchange_strong(
            expected, fresh, std::memory_order_acq_rel, std::memory_order_acquire)) {
        return fresh;
    }
    delete_with_allocator(fresh);
    return expected;
}

CUtexObject CUDATexture::tex_object(Sampler sampler) const noexcept {
    auto *cache = _get_or_create_sampler_cache();
    const auto code = sampler.code();
    std::scoped_lock lock{cache->mutex};
    if (cache->objects[code] == 0ull) {
        auto create_one = [&]() noexcept {
            cache->objects[code] = create_cuda_texture_object(this, sampler);
        };
        if (_device != nullptr) {
            _device->with_handle(create_one);
        } else {
            create_one();
        }
    }
    return cache->objects[code];
}

CUdeviceptr CUDATexture::sampler_table(uint16_t sampler_mask, CUstream /*stream*/) const noexcept {
    if (sampler_mask == 0u) {
        return 0ull;
    }
    auto *cache = _get_or_create_sampler_cache();
    std::scoped_lock lock{cache->mutex};
    if (cache->device_table == 0ull) {
        LUISA_ASSERT(_device != nullptr, "CUDATexture sampler_table requires owning CUDADevice.");
        cache->device_table = _device->allocate_sampler_table();
    }
    const auto missing = static_cast<uint16_t>(sampler_mask & ~cache->uploaded_mask);
    if (missing != 0u) {
        auto populate_and_upload = [&]() noexcept {
            uint16_t m = missing;
            while (m != 0u) {
                const auto code = static_cast<uint32_t>(std::countr_zero(m));
                m &= static_cast<uint16_t>(m - 1u);
                if (cache->objects[code] == 0ull) {
                    cache->objects[code] = create_cuda_texture_object(this, Sampler::decode(code));
                }
            }
            LUISA_CHECK_CUDA(cuMemcpyHtoD(
                cache->device_table, cache->objects.data(), sizeof(cache->objects)));
        };
        if (_device != nullptr) {
            _device->with_handle(populate_and_upload);
        } else {
            populate_and_upload();
        }
        cache->uploaded_mask = static_cast<uint16_t>(cache->uploaded_mask | missing);
    }
    return cache->device_table;
}

CUDASurface CUDATexture::binding(uint32_t level, uint16_t sampler_mask,
                                 bool needs_surface, CUstream stream) const noexcept {
    if (needs_surface) {
        LUISA_ASSERT(!is_block_compressed(format()),
                     "Block compressed textures cannot be used as CUDA surfaces.");
    }
    const auto packed = encoded_storage(level);
    const CUsurfObject surf_handle = needs_surface ? _mip_surfaces[level] : 0ull;
    if (sampler_mask == 0u) {
        return CUDASurface{surf_handle, packed, 0ull, 0ull};
    }
    if (std::has_single_bit(sampler_mask)) {
        const auto code = static_cast<uint32_t>(std::countr_zero(sampler_mask));
        const auto single = tex_object(Sampler::decode(code));
        return CUDASurface{surf_handle, packed, 0ull, single};
    }
    const auto table = sampler_table(sampler_mask, stream);
    return CUDASurface{surf_handle, packed, table, 0ull};
}

namespace detail {

[[nodiscard]] auto create_array_from_mipmapped_array(CUmipmappedArray mipmapped_array, uint32_t level) noexcept {
    CUarray array;
    LUISA_CHECK_CUDA(cuMipmappedArrayGetLevel(&array, mipmapped_array, level));
    return array;
}

[[nodiscard]] auto create_surface_from_array(CUarray array) noexcept {
    CUDA_RESOURCE_DESC res_desc{};
    res_desc.resType = CU_RESOURCE_TYPE_ARRAY;
    res_desc.res.array.hArray = array;
    CUsurfObject surf;
    LUISA_CHECK_CUDA(cuSurfObjectCreate(&surf, &res_desc));
    return surf;
}

}// namespace detail

CUDATexture::CUDATexture(CUDADevice *device, uint64_t array, uint3 size,
                         PixelFormat format, uint32_t levels) noexcept
    : _base_array{array},
      _size{static_cast<uint16_t>(size.x),
            static_cast<uint16_t>(size.y),
            static_cast<uint16_t>(size.z)},
      _format{static_cast<uint8_t>(format)},
      _levels{static_cast<uint8_t>(levels)},
      _device{device} {
    auto is_bc = is_block_compressed(format);
    if (_levels == 1u) {// not mip-mapped
        _mip_arrays[0] = reinterpret_cast<CUarray>(_base_array);
        if (!is_bc) { _mip_surfaces[0] = detail::create_surface_from_array(_mip_arrays[0]); }
    } else {
        for (auto i = 0u; i < _levels; i++) {
            _mip_arrays[i] = detail::create_array_from_mipmapped_array(
                reinterpret_cast<CUmipmappedArray>(_base_array), i);
            if (!is_bc) { _mip_surfaces[i] = detail::create_surface_from_array(_mip_arrays[i]); }
        }
    }
}

void CUDATexture::set_name(luisa::string &&name) noexcept {
    // currently do nothing
}

}// namespace luisa::compute::cuda
