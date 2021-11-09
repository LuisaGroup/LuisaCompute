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
    enum struct Storage : uint32_t {
        BYTE,
        SHORT,
        INT,
        HALF,
        FLOAT
    };
    CUsurfObject handle;
    Storage storage;
    uint16_t pixel_size_shift;
    uint16_t channel_count;
};

static_assert(sizeof(CUDASurface) == 16u);

class CUDAMipmapArray {

public:
    static constexpr auto max_level_count = 14u;

private:
    CUmipmappedArray _array;
    mutable std::array<CUsurfObject, max_level_count> _surfaces{};
    PixelFormat _format;
    mutable spin_mutex _mutex;

public:
    CUDAMipmapArray(CUmipmappedArray array, PixelFormat format) noexcept;
    ~CUDAMipmapArray() noexcept;
    [[nodiscard]] auto handle() const noexcept { return _array; }
    [[nodiscard]] auto format() const noexcept { return _format; }
    [[nodiscard]] CUsurfObject surface(uint32_t level) const noexcept;
};

static_assert(sizeof(CUDAMipmapArray) == 128u);

}// namespace luisa::compute::cuda
