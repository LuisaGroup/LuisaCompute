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
class CUDAHeap;
class CUDADevice;
class CUDAStream;

class CUDAAccel {

private:
    OptixTraversableHandle _handle{};
    luisa::vector<CUDAMesh *> _instance_meshes;
    luisa::vector<float4x4> _instance_transforms;
    luisa::bitvector<> _instance_visibilities;
    luisa::unordered_set<uint64_t> _resources;
    CUdeviceptr _instance_buffer{};
    size_t _instance_buffer_size{};
    CUdeviceptr _bvh_buffer{};
    size_t _bvh_buffer_size{};
    size_t _update_buffer_size{};
    DirtyRange _dirty_range{};
    AccelBuildHint _build_hint;
    CUDAHeap *_heap{nullptr};

private:
    [[nodiscard]] OptixBuildInput _make_build_input() const noexcept;

public:
    explicit CUDAAccel(AccelBuildHint hint) noexcept;
    ~CUDAAccel() noexcept;
    void add_instance(CUDAMesh *mesh, float4x4 transform, bool visible) noexcept;
    void set_instance(size_t index, CUDAMesh *mesh, float4x4 transform, bool visible) noexcept;
    void set_visibility(size_t index, bool visible) noexcept;
    void pop_instance() noexcept;
    void set_transform(size_t index, float4x4 transform) noexcept;
    void build(CUDADevice *device, CUDAStream *stream) noexcept;
    void update(CUDADevice *device, CUDAStream *stream) noexcept;
    [[nodiscard]] auto handle() const noexcept { return _handle; }
    [[nodiscard]] bool uses_resource(uint64_t handle) const noexcept;
};

}
