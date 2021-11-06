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
    LUISA_CHECK_CUDA(cuMipmappedArrayDestroy(_array));
}

CUsurfObject CUDAMipmapArray::surface(uint32_t level) const noexcept {
    if (auto s = _surfaces[level]; s != 0u) { return s; }
    CUarray mipmap;
    LUISA_CHECK_CUDA(cuMipmappedArrayGetLevel(&mipmap, _array, level));
    CUDA_RESOURCE_DESC res_desc;
    res_desc.resType = CU_RESOURCE_TYPE_ARRAY;
    res_desc.res.array.hArray = mipmap;
    CUsurfObject surface;
    LUISA_CHECK_CUDA(cuSurfObjectCreate(&surface, &res_desc));
    _surfaces[level] = surface;
    return surface;
}

CUDAMipmapArray::CUDAMipmapArray(CUmipmappedArray array) noexcept
    : _array{array} {}

}// namespace luisa::compute::cuda