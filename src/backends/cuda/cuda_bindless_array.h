//
// Created by Mike on 7/30/2021.
//

#pragma once

#include <cuda.h>

#include <core/spin_mutex.h>
#include <core/stl.h>
#include <core/dirty_range.h>
#include <runtime/sampler.h>
#include <runtime/resource_tracker.h>
#include <backends/cuda/cuda_error.h>
#include <backends/cuda/cuda_mipmap_array.h>

namespace luisa::compute::cuda {

class CUDADevice;
class CUDAStream;

class CUDABindlessArray {

public:
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
    ResourceTracker _resource_tracker;
    luisa::vector<uint64_t> _buffer_resources;
    luisa::unordered_map<CUtexObject, uint64_t> _texture_resources;

public:
    explicit CUDABindlessArray(size_t capacity) noexcept;
    ~CUDABindlessArray() noexcept;
    [[nodiscard]] auto handle() const noexcept { return _handle; }
    void emplace_buffer(size_t index, uint64_t buffer, size_t offset) noexcept;
    void emplace_tex2d(size_t index, CUDAMipmapArray *array, Sampler sampler) noexcept;
    void emplace_tex3d(size_t index, CUDAMipmapArray *array, Sampler sampler) noexcept;
    void remove_buffer(size_t index) noexcept;
    void remove_tex2d(size_t index) noexcept;
    void remove_tex3d(size_t index) noexcept;
    [[nodiscard]] bool uses_buffer(uint64_t handle) const noexcept;
    [[nodiscard]] bool uses_texture(uint64_t handle) const noexcept;
    void upload(CUDAStream *stream) noexcept;
};

}// namespace luisa::compute::cuda
