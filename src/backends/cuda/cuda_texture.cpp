#include <mutex>

#include "cuda_error.h"
#include "cuda_texture.h"

namespace luisa::compute::cuda {

CUDATexture::~CUDATexture() noexcept {
    for (auto i = 0u; i < _levels; i++) {
        LUISA_CHECK_CUDA(cuSurfObjectDestroy(_mip_surfaces[i]));
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

CUDASurface CUDATexture::surface(uint32_t level) const noexcept {
    LUISA_ASSERT(level < _levels,
                 "Invalid level {} for texture with {} level(s).",
                 level, _levels);
    LUISA_ASSERT(!is_block_compressed(format()),
                 "Block compressed textures cannot be used as CUDA surfaces.");
    return CUDASurface{_mip_surfaces[level], to_underlying(storage())};
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

CUDATexture::CUDATexture(uint64_t array, uint3 size,
                         PixelFormat format, uint32_t levels) noexcept
    : _base_array{array},
      _size{static_cast<uint16_t>(size.x),
            static_cast<uint16_t>(size.y),
            static_cast<uint16_t>(size.z)},
      _format{static_cast<uint8_t>(format)},
      _levels{static_cast<uint8_t>(levels)} {
    if (_levels == 1u) {// not mip-mapped
        _mip_arrays[0] = reinterpret_cast<CUarray>(_base_array);
        _mip_surfaces[0] = detail::create_surface_from_array(reinterpret_cast<CUarray>(_base_array));
    } else {
        for (auto i = 0u; i < _levels; i++) {
            _mip_arrays[i] = detail::create_array_from_mipmapped_array(
                reinterpret_cast<CUmipmappedArray>(_base_array), i);
            _mip_surfaces[i] = detail::create_surface_from_array(_mip_arrays[i]);
        }
    }
}

void CUDATexture::set_name(luisa::string &&name) noexcept {
    // currently do nothing
}

}// namespace luisa::compute::cuda

