//
// Created by Mike on 2021/12/2.
//

#pragma once

#include <vector>

#include <cuda.h>
#include <optix.h>

#include <rtx/accel.h>
#include <core/dirty_range.h>

namespace luisa::compute::cuda {

class CUDAMesh;
class CUDADevice;
class CUDAStream;

class CUDAAccel {

private:
    OptixTraversableHandle _handle{};
    std::vector<CUDAMesh *> _instance_meshes;
    std::vector<float4x4> _instance_transforms;
    std::vector<CUdeviceptr> _resource_buffers;
    CUdeviceptr _instance_buffer{};
    size_t _instance_buffer_size{};
    CUdeviceptr _bvh_buffer{};
    size_t _bvh_buffer_size{};
    CUdeviceptr _update_buffer{};
    size_t _update_buffer_size{};
    DirtyRange _dirty_range{};
    AccelBuildHint _build_hint;

private:
    [[nodiscard]] OptixBuildInput _make_build_input() const noexcept;

public:
    explicit CUDAAccel(AccelBuildHint hint) noexcept;
    ~CUDAAccel() noexcept;
    void add_instance(CUDAMesh *mesh, float4x4 transform) noexcept;
    void set_transform(size_t index, float4x4 transform) noexcept;

    void build(CUDADevice *device, CUDAStream *stream) noexcept;
    void update(CUDADevice *device, CUDAStream *stream) noexcept;
    [[nodiscard]] bool uses_buffer(CUdeviceptr handle) const noexcept;
};

}
