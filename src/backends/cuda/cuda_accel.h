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

class CUDAAccel {

private:
    OptixTraversableHandle _handle{};
    std::vector<CUDAMesh *> _instance_meshes;
    std::vector<float4x4> _instance_transforms;
    std::vector<OptixInstance> _instances;
    CUdeviceptr _instance_buffer{};
    CUdeviceptr _bvh_buffer{};
    size_t _bvh_buffer_size{};
    CUdeviceptr _update_buffer{};
    size_t _update_buffer_size{};
    CUevent _update_event{};
    DirtyRange _dirty_range{};
    AccelBuildHint _build_hint;

public:
    CUDAAccel(AccelBuildHint hint) noexcept;
    ~CUDAAccel() noexcept;
    void add_instance(CUDAMesh *mesh, float4x4 transform) noexcept;
    void set_transform(size_t index, float4x4 transform) noexcept;
};

}
