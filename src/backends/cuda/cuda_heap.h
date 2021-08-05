//
// Created by Mike on 7/30/2021.
//

#pragma once

#include <vector>
#include <unordered_set>

#include <cuda.h>

#include <util/spin_mutex.h>
#include <runtime/texture.h>
#include <backends/cuda/cuda_error.h>
#include <backends/cuda/cuda_buffer.h>
#include <backends/cuda/cuda_texture.h>

namespace luisa::compute::cuda {

class CUDADevice;

class CUDAHeap {

    friend class CUDABuffer;
    friend class CUDATexture;

public:
    struct Item {
        CUtexObject texture{0u};
        CUdeviceptr buffer{0u};
    };

private:
    CUDADevice *_device;
    CUmemoryPool _handle;
    CUdeviceptr _desc_array;
    std::vector<Item> _items;
    std::unordered_set<CUDABuffer *> _active_buffers;
    std::unordered_set<CUDATexture *> _active_textures;
    mutable spin_mutex _mutex;
    mutable bool _dirty{true};

public:
    CUDAHeap(CUDADevice *device, size_t capacity) noexcept;
    ~CUDAHeap() noexcept;
    [[nodiscard]] CUDABuffer *allocate_buffer(size_t index, size_t size) noexcept;
    void destroy_buffer(CUDABuffer *buffer) noexcept;
    [[nodiscard]] CUDATexture *allocate_texture(size_t index, PixelFormat format, uint dim, uint3 size, uint mip_levels, TextureSampler sampler) noexcept;
    void destroy_texture(CUDATexture *texture) noexcept;
    [[nodiscard]] size_t memory_usage() const noexcept;
    [[nodiscard]] CUdeviceptr descriptor_array() const noexcept;
};

}// namespace luisa::compute::cuda
