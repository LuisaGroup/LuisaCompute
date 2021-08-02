//
// Created by Mike on 8/1/2021.
//

#pragma once

#include <cuda.h>
#include <core/basic_types.h>

namespace luisa::compute::cuda {

class CUDAHeap;

class CUDATexture {

private:
    union {
        CUtexObject _handle;
        size_t _index;
    };
    union {
        CUarray _array;
        CUmipmappedArray _mip_array;
    };
    CUDAHeap *_heap{nullptr};
    uint _dimension{};

public:
    explicit CUDATexture(CUtexObject handle, CUarray array, uint dim) noexcept
        : _handle{handle}, _array{array}, _dimension{dim} {}
    explicit CUDATexture(CUDAHeap *heap, size_t index, CUmipmappedArray mip_array, uint dim) noexcept
        : _index{index}, _mip_array{mip_array}, _heap{heap}, _dimension{dim} {}
    [[nodiscard]] uint64_t handle() const noexcept;
    [[nodiscard]] auto array() const noexcept { return _array; }
    [[nodiscard]] auto mip_array() const noexcept { return _mip_array; }
    [[nodiscard]] auto heap() const noexcept { return _heap; }
    [[nodiscard]] auto index() const noexcept { return _index; }
    [[nodiscard]] auto dimension() const noexcept { return _dimension; }
};

}// namespace luisa::compute::cuda
