//
// Created by Mike on 2021/11/6.
//

#pragma once

#include <array>
#include <cuda.h>

#include <core/spin_mutex.h>
#include <runtime/rhi/pixel.h>

namespace luisa::compute::cuda {

/**
 * @brief Struct of surface on CUDA
 * 
 */
struct alignas(16) CUDASurface {
    CUsurfObject handle;
    PixelStorage storage;
};

static_assert(sizeof(CUDASurface) == 16u);

/**
 * @brief Mipmap array on CUDA
 * 
 */
class CUDAMipmapArray {

public:
    static constexpr auto max_level_count = 14u;

private:
    uint64_t _array;
    mutable std::array<CUsurfObject, max_level_count> _surfaces{};
    uint16_t _format;
    uint16_t _levels;
    mutable spin_mutex _mutex;

public:
    /**
     * @brief Construct a new CUDAMipmapArray object
     * 
     * @param array handle of array on CUDA
     * @param format pixel format
     * @param levels mipmap level
     */
    CUDAMipmapArray(uint64_t array, PixelFormat format, uint32_t levels) noexcept;
    ~CUDAMipmapArray() noexcept;
    /**
     * @brief Return handle of array
     * 
     * @return handle of array
     */
    [[nodiscard]] auto handle() const noexcept { return _array; }
    /**
     * @brief Return pixel format
     * 
     * @return pixel format
     */
    [[nodiscard]] auto format() const noexcept { return static_cast<PixelFormat>(_format); }
    /**
     * @brief Return mipmap level
     * 
     * @return mipmap level
     */
    [[nodiscard]] auto levels() const noexcept { return static_cast<size_t>(_levels); }
    /**
     * @brief Return array at given level
     * 
     * @param i given level
     * @return CUarray 
     */
    [[nodiscard]] CUarray level(uint32_t i) const noexcept;
    /**
     * @brief Return surface at given level
     * 
     * @param level given level
     * @return CUDASurface 
     */
    [[nodiscard]] CUDASurface surface(uint32_t level) const noexcept;
    /**
     * @brief Return size of mipmap array
     * 
     * @return uint3 (width, height, depth)
     */
    [[nodiscard]] uint3 size() const noexcept;
};

static_assert(sizeof(CUDAMipmapArray) == 128u);

}// namespace luisa::compute::cuda
