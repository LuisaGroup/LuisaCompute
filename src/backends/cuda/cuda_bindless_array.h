#pragma once

#include <cuda.h>

#include <luisa/core/spin_mutex.h>
#include <luisa/core/stl.h>
#include <luisa/runtime/rhi/sampler.h>
#include <luisa/runtime/rhi/command.h>
#include "../common/resource_tracker.h"
#include "cuda_error.h"
#include "cuda_texture.h"

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

    struct Binding {
        CUdeviceptr slots;
        size_t capacity;
    };

private:
    static constexpr size_t sampler_variant_count = 16u;
    struct SamplerTable {
        CUdeviceptr device_ptr{0ull};
        std::array<CUtexObject, sampler_variant_count> objects{};
    };

    CUDADevice *_device{nullptr};
    CUdeviceptr _handle{};
    luisa::vector<SamplerTable> _tex2d_slots;
    luisa::vector<SamplerTable> _tex3d_slots;
    ResourceTracker _texture_tracker;
    luisa::string _name;
    spin_mutex _mutex;

public:
    CUDABindlessArray(CUDADevice *device, size_t capacity) noexcept;
    ~CUDABindlessArray() noexcept;
    [[nodiscard]] auto handle() const noexcept { return _handle; }
    void update(CUDACommandEncoder &encoder, BindlessArrayUpdateCommand *cmd) noexcept;
    [[nodiscard]] auto binding() const noexcept { return Binding{_handle, _tex2d_slots.size()}; }
    void set_name(luisa::string &&name) noexcept;
};

}// namespace luisa::compute::cuda

