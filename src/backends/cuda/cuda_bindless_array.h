//
// Created by Mike on 7/30/2021.
//

#pragma once

#include <cuda.h>

#include <core/spin_mutex.h>
#include <core/stl.h>
#include <core/dirty_range.h>
#include <runtime/sampler.h>
#include <runtime/command.h>
#include <backends/cuda/cuda_error.h>
#include <backends/cuda/cuda_mipmap_array.h>

namespace luisa::compute::cuda {

class CUDADevice;
class CUDAStream;

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
    /**
     * @brief Construct a new CUDABindlessArray object
     * 
     * @param capacity capacity of bindless array
     */
    explicit CUDABindlessArray(size_t capacity) noexcept;
    ~CUDABindlessArray() noexcept;
    /**
     * @brief Return SlotSOA handle
     * 
     * @return SlotSOA
     */
    [[nodiscard]] auto handle() const noexcept { return _handle; }
    /**
     * @brief Emplace a buffer
     * 
     * @param index place to emplace
     * @param buffer handle of buffer
     * @param offset offset of buffer
     */
    void emplace_buffer(size_t index, uint64_t buffer, size_t offset) noexcept;
    /**
     * @brief Emplace a 2D texture
     * 
     * @param index place to emplace
     * @param array address of 2D texture
     * @param sampler sampler of texture
     */
    void emplace_tex2d(size_t index, CUDAMipmapArray *array, Sampler sampler) noexcept;
    /**
     * @brief Emplace a 3D texture
     * 
     * @param index index to emplace
     * @param array address of 3D texture
     * @param sampler sampler of texture
     */
    void emplace_tex3d(size_t index, CUDAMipmapArray *array, Sampler sampler) noexcept;
    /**
     * @brief Remove buffer
     * 
     * @param index place to remove
     */
    void remove_buffer(size_t index) noexcept;
    /**
     * @brief Remove 2D texture
     * 
     * @param index place to remove
     */
    void remove_tex2d(size_t index) noexcept;
    /**
     * @brief Remove 3D texture
     * 
     * @param index place to remove
     */
    void remove_tex3d(size_t index) noexcept;
    /**
     * @brief If resource is used
     * 
     * @param handle handle of resource
     * @return true 
     * @return false 
     */
    [[nodiscard]] bool uses_resource(uint64_t handle) const noexcept;
    /**
     * @brief Upload bindless array to CUDA device
     * 
     * @param stream CUDAStream
     */
    void upload(CUDAStream *stream) noexcept;

    void update(CUDAStream *stream, luisa::span<const BindlessArrayUpdateCommand::Modification> mods) noexcept;
};

}// namespace luisa::compute::cuda
