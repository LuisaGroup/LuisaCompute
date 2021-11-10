//
// Created by Mike on 2021/11/6.
//

#include <mutex>

#include <backends/cuda/cuda_error.h>
#include <backends/cuda/cuda_mipmap_array.h>

namespace luisa::compute::cuda {

CUDAMipmapArray::~CUDAMipmapArray() noexcept {
    for (auto s : _surfaces) {
        if (s != 0u) {
            LUISA_CHECK_CUDA(cuSurfObjectDestroy(s));
        }
    }
    if (_levels == 1u) {
        LUISA_CHECK_CUDA(cuArrayDestroy(reinterpret_cast<CUarray>(_array)));
    } else {
        LUISA_CHECK_CUDA(cuMipmappedArrayDestroy(reinterpret_cast<CUmipmappedArray>(_array)));
    }
}

CUarray CUDAMipmapArray::level(uint32_t i) const noexcept {
    if (i >= _levels) [[unlikely]] {
        LUISA_ERROR_WITH_LOCATION(
            "Invalid level {} for texture with {} level(s).",
            i, _levels);
    }
    if (_levels == 1u) {// not mipmapped
        return reinterpret_cast<CUarray>(_array);
    }
    CUarray array;
    LUISA_CHECK_CUDA(cuMipmappedArrayGetLevel(&array, reinterpret_cast<CUmipmappedArray>(_array), i));
    return array;
}

CUDASurface CUDAMipmapArray::surface(uint32_t level) const noexcept {
    auto handle = [this, level] {
        std::scoped_lock lock{_mutex};
        if (auto s = _surfaces[level]; s != 0u) { return s; }
        CUarray mipmap = this->level(level);
        LUISA_VERBOSE_WITH_LOCATION("Getting CUDA array at level {}.", level);
        CUDA_RESOURCE_DESC res_desc{};
        res_desc.resType = CU_RESOURCE_TYPE_ARRAY;
        res_desc.res.array.hArray = mipmap;
        CUsurfObject surf;
        LUISA_CHECK_CUDA(cuSurfObjectCreate(&surf, &res_desc));
        _surfaces[level] = surf;
        return surf;
    }();
    auto [storage, shift, channels] = [ps = pixel_format_to_storage(format())] {
        switch (ps) {
            case PixelStorage::BYTE1: return std::make_tuple(CUDASurface::Storage::BYTE, 0u, 1u);
            case PixelStorage::BYTE2: return std::make_tuple(CUDASurface::Storage::BYTE, 1u, 2u);
            case PixelStorage::BYTE4: return std::make_tuple(CUDASurface::Storage::BYTE, 2u, 4u);
            case PixelStorage::SHORT1: return std::make_tuple(CUDASurface::Storage::SHORT, 1u, 1u);
            case PixelStorage::SHORT2: return std::make_tuple(CUDASurface::Storage::SHORT, 2u, 2u);
            case PixelStorage::SHORT4: return std::make_tuple(CUDASurface::Storage::SHORT, 3u, 4u);
            case PixelStorage::INT1: return std::make_tuple(CUDASurface::Storage::INT, 2u, 1u);
            case PixelStorage::INT2: return std::make_tuple(CUDASurface::Storage::INT, 3u, 2u);
            case PixelStorage::INT4: return std::make_tuple(CUDASurface::Storage::INT, 4u, 4u);
            case PixelStorage::HALF1: return std::make_tuple(CUDASurface::Storage::HALF, 1u, 1u);
            case PixelStorage::HALF2: return std::make_tuple(CUDASurface::Storage::HALF, 2u, 2u);
            case PixelStorage::HALF4: return std::make_tuple(CUDASurface::Storage::HALF, 3u, 4u);
            case PixelStorage::FLOAT1: return std::make_tuple(CUDASurface::Storage::INT, 2u, 1u);
            case PixelStorage::FLOAT2: return std::make_tuple(CUDASurface::Storage::INT, 3u, 2u);
            case PixelStorage::FLOAT4: return std::make_tuple(CUDASurface::Storage::INT, 4u, 4u);
            default: LUISA_ERROR_WITH_LOCATION("Invalid pixel storage.");
        }
    }();
    return CUDASurface{handle, storage, static_cast<uint16_t>(shift), static_cast<uint16_t>(channels)};
}

CUDAMipmapArray::CUDAMipmapArray(uint64_t array, PixelFormat format, uint32_t levels) noexcept
    : _array{array}, _format{static_cast<uint16_t>(format)}, _levels{static_cast<uint16_t>(levels)} {}

}// namespace luisa::compute::cuda