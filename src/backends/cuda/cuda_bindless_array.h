//
// Created by Mike on 7/30/2021.
//

#pragma once

#include <cuda.h>

#include <core/spin_mutex.h>
#include <core/allocator.h>
#include <runtime/sampler.h>
#include <backends/cuda/cuda_error.h>
#include <backends/cuda/cuda_mipmap_array.h>

namespace luisa::compute::cuda {

class CUDADevice;

class CUDABindlessArray {

public:
    struct Item {
        CUdeviceptr buffer{};
        CUdeviceptr origin{};
        CUtexObject tex2d{};
        CUtexObject tex3d{};
    };

private:
    CUdeviceptr _handle;
    CUevent _event{};
    luisa::vector<Item> _slots;
    luisa::unordered_map<uint64_t, size_t> _buffers;
    luisa::unordered_map<uint64_t, size_t> _arrays;
    luisa::unordered_map<CUtexObject, CUDAMipmapArray *> _tex_to_array;
    bool _dirty{true};

public:
    static void _retain(luisa::unordered_map<uint64_t, size_t> &resources, uint64_t r) noexcept;
    static void _release(luisa::unordered_map<uint64_t, size_t> &resources, uint64_t r) noexcept;
    void _retain_buffer(CUdeviceptr buffer) noexcept { _retain(_buffers, buffer); }
    void _release_buffer(CUdeviceptr buffer) noexcept { _release(_buffers, buffer); }
    void _retain_array(CUDAMipmapArray *array) noexcept { _retain(_arrays, reinterpret_cast<uint64_t>(array)); }
    void _release_array(CUDAMipmapArray *array) noexcept { _release(_arrays, reinterpret_cast<uint64_t>(array)); }

public:
    CUDABindlessArray(CUdeviceptr handle, size_t capacity) noexcept;
    ~CUDABindlessArray() noexcept;
    [[nodiscard]] auto handle() const noexcept { return _handle; }
    void encode_update(CUstream stream) noexcept;
    void emplace_buffer(size_t index, CUdeviceptr buffer, size_t offset) noexcept;
    void emplace_tex2d(size_t index, CUDAMipmapArray *array, Sampler sampler) noexcept;
    void emplace_tex3d(size_t index, CUDAMipmapArray *array, Sampler sampler) noexcept;
    void remove_buffer(size_t index) noexcept;
    void remove_tex2d(size_t index) noexcept;
    void remove_tex3d(size_t index) noexcept;
    [[nodiscard]] bool has_buffer(CUdeviceptr buffer) const noexcept;
    [[nodiscard]] bool has_array(CUDAMipmapArray *array) const noexcept;
};

}// namespace luisa::compute::cuda
