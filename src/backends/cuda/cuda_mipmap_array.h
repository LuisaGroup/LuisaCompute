//
// Created by Mike on 2021/11/6.
//

#pragma once

#include <array>
#include <cuda.h>

#include <core/spin_mutex.h>
#include <runtime/pixel.h>

namespace luisa::compute::cuda {

struct alignas(16) CUDASurface {
    CUsurfObject handle;
    PixelStorage storage;
};

static_assert(sizeof(CUDASurface) == 16u);

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
    CUDAMipmapArray(uint64_t array, PixelFormat format, uint32_t levels) noexcept;
    ~CUDAMipmapArray() noexcept;
    [[nodiscard]] auto handle() const noexcept { return _array; }
    [[nodiscard]] auto format() const noexcept { return static_cast<PixelFormat>(_format); }
    [[nodiscard]] auto levels() const noexcept { return static_cast<size_t>(_levels); }
    [[nodiscard]] CUarray level(uint32_t i) const noexcept;
    [[nodiscard]] CUDASurface surface(uint32_t level) const noexcept;
};

static_assert(sizeof(CUDAMipmapArray) == 128u);

}// namespace luisa::compute::cuda
