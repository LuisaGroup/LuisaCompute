//
// Created by Mike on 7/30/2021.
//

#pragma once

#include <cuda.h>

#include <luisa/core/spin_mutex.h>
#include <luisa/core/stl.h>
#include <luisa/core/dirty_range.h>
#include <luisa/runtime/rhi/sampler.h>
#include <luisa/runtime/rhi/command.h>
#include <backends/common/resource_tracker.h>
#include <backends/cuda/cuda_error.h>
#include <backends/cuda/cuda_texture.h>

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

    using Binding = CUdeviceptr;

private:
    CUdeviceptr _handle{};
    luisa::vector<CUtexObject> _tex2d_slots;
    luisa::vector<CUtexObject> _tex3d_slots;
    ResourceTracker _texture_tracker;
    luisa::string _name;
    spin_mutex _mutex;

public:
    explicit CUDABindlessArray(size_t capacity) noexcept;
    ~CUDABindlessArray() noexcept;
    [[nodiscard]] auto handle() const noexcept { return _handle; }
    void update(CUDACommandEncoder &encoder, BindlessArrayUpdateCommand *cmd) noexcept;
    [[nodiscard]] auto binding() const noexcept { return _handle; }
    void set_name(luisa::string &&name) noexcept;
};

}// namespace luisa::compute::cuda

