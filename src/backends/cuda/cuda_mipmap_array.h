//
// Created by Mike on 2021/11/6.
//

#pragma once

#include <array>
#include <cuda.h>

#include <core/spin_mutex.h>

namespace luisa::compute::cuda {

class CUDAMipmapArray {

public:
    static constexpr auto max_level_count = 14u;

private:
    CUmipmappedArray _array;
    mutable std::array<CUsurfObject, max_level_count> _surfaces{};
    mutable spin_mutex _mutex;

public:
    explicit CUDAMipmapArray(CUmipmappedArray array) noexcept;
    ~CUDAMipmapArray() noexcept;
    [[nodiscard]] auto handle() const noexcept { return _array; }
    [[nodiscard]] CUsurfObject surface(uint32_t level) const noexcept;
};

static_assert(sizeof(CUDAMipmapArray) == 128u);

}
