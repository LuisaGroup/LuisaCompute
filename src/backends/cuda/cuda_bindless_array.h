//
// Created by Mike on 7/30/2021.
//

#pragma once

#include <cuda.h>

#include <core/spin_mutex.h>
#include <core/stl.h>
#include <core/dirty_range.h>
#include <runtime/rhi/sampler.h>
#include <runtime/rhi/command.h>
#include <backends/common/resource_tracker.h>
#include <backends/cuda/cuda_error.h>
#include <backends/cuda/cuda_mipmap_array.h>

namespace luisa::compute::cuda {

class CUDADevice;
class CUDAStream;
class CUDACommandEncoder;

/**
 * @brief Bindless array of CUDA
 * 
 */
class CUDABindlessArray {

public:
    struct Slot {
        uint64_t buffer;
        size_t size;
        uint64_t tex2d;
        uint64_t tex3d;
    };

private:
    CUdeviceptr _handle{};
    luisa::vector<CUtexObject> _tex2d_slots;
    luisa::vector<CUtexObject> _tex3d_slots;
    ResourceTracker _texture_tracker;

public:
    explicit CUDABindlessArray(size_t capacity) noexcept;
    ~CUDABindlessArray() noexcept;
    [[nodiscard]] auto handle() const noexcept { return _handle; }
    void update(CUDACommandEncoder &encoder, BindlessArrayUpdateCommand *cmd) noexcept;
};

}// namespace luisa::compute::cuda
