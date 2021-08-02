//
// Created by Mike on 7/30/2021.
//

#pragma once

#include <backends/cuda/cuda_error.h>

namespace luisa::compute::cuda {

class CUDAHeap;

class CUDABuffer {

private:
    union {
        CUdeviceptr _handle;
        size_t _index;
    };
    CUDAHeap *_heap{nullptr};

public:
    explicit CUDABuffer(CUdeviceptr handle = 0u) noexcept
        : _handle{handle} {}
    CUDABuffer(CUDAHeap *heap, size_t index) noexcept
        : _index{index}, _heap{heap} {}
    [[nodiscard]] CUdeviceptr handle() const noexcept;
    [[nodiscard]] auto index() const noexcept { return _index; }
    [[nodiscard]] auto heap() const noexcept { return _heap; }
};

}// namespace luisa::compute::cuda
