//
// Created by Mike on 7/30/2021.
//

#pragma once

#include <vector>
#include <unordered_set>

#include <cuda.h>

#include <core/spin_mutex.h>
#include <backends/cuda/cuda_error.h>
#include <backends/cuda/cuda_buffer.h>

namespace luisa::compute::cuda {

class CUDADevice;

class CUDAHeap {

    friend class CUDABuffer;

public:
    struct Item {
        CUtexObject texture;
        CUsurfObject surface;
        CUdeviceptr buffer;
    };

private:
    CUDADevice *_device;
    CUmemoryPool _handle;
    CUdeviceptr _desc_array;
    std::vector<Item> _items;
    std::unordered_set<CUDABuffer *> _active_buffers;
    spin_mutex _mutex;
    bool _dirty{true};

public:
    CUDAHeap(CUDADevice *device, size_t capacity) noexcept;
    ~CUDAHeap() noexcept;
    [[nodiscard]] CUDABuffer *allocate_buffer(size_t size, size_t index) noexcept;
    void destroy_buffer(CUDABuffer *buffer) noexcept;
    [[nodiscard]] size_t memory_usage() const noexcept;
};

}// namespace luisa::compute::cuda
