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
    /**
     * @brief Slot struct on device
     * 
     */
    struct SlotSOA {
        CUdeviceptr _buffer_slots;
        CUdeviceptr _tex2d_slots;
        CUdeviceptr _tex3d_slots;
        CUdeviceptr _tex2d_sizes;
        CUdeviceptr _tex3d_sizes;
    };

private:
    SlotSOA _handle{};
    DirtyRange _buffer_dirty_range;
    DirtyRange _tex2d_dirty_range;
    DirtyRange _tex3d_dirty_range;
    luisa::vector<CUdeviceptr> _buffer_slots;
    luisa::vector<CUtexObject> _tex2d_slots;
    luisa::vector<CUtexObject> _tex3d_slots;
    luisa::vector<std::array<uint16_t, 2u>> _tex2d_sizes;
    luisa::vector<std::array<uint16_t, 4u>> _tex3d_sizes;
    luisa::vector<uint64_t> _buffer_resources;
    luisa::unordered_map<CUtexObject, uint64_t> _texture_resources;

public:
    explicit CUDABindlessArray(size_t capacity) noexcept;
    ~CUDABindlessArray() noexcept;
    [[nodiscard]] auto handle() const noexcept { return _handle; }
    void update(CUDACommandEncoder &encoder, BindlessArrayUpdateCommand *cmd) noexcept;
};

}// namespace luisa::compute::cuda
