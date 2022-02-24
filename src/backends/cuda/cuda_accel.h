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

/**
 * @brief Acceleration structure of CUDA
 * 
 */
class CUDAAccel {

public:
    /**
     * @brief Binding struct of device
     * 
     */
    struct alignas(16) Binding {
        OptixTraversableHandle handle;
        CUdeviceptr instances;
    };

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
    /**
     * @brief Construct a new CUDAAccel object
     * 
     * @param hint build hint
     */
    explicit CUDAAccel(AccelBuildHint hint) noexcept;
    ~CUDAAccel() noexcept;
    /**
     * @brief Add an instance to accel
     * 
     * @param mesh mesh to be added
     * @param transform mesh's transform
     * @param visible mesh's visibility
     */
    void add_instance(CUDAMesh *mesh, float4x4 transform, bool visible) noexcept;
    /**
     * @brief Set an instance
     * 
     * @param index place to set
     * @param mesh new mesh
     * @param transform new transform
     * @param visible new visibility
     */
    void set_instance(size_t index, CUDAMesh *mesh, float4x4 transform, bool visible) noexcept;
    /**
     * @brief Set visibility
     * 
     * @param index place to set
     * @param visible new visibility
     */
    void set_visibility(size_t index, bool visible) noexcept;
    /**
     * @brief Pop the latest instance
     * 
     */
    void pop_instance() noexcept;
    /**
     * @brief Set transform
     * 
     * @param index place to set
     * @param transform new transform
     */
    void set_transform(size_t index, float4x4 transform) noexcept;
    /**
     * @brief Build the accel structure on device
     * 
     * @param device CUDADevice
     * @param stream CUDAStream
     */
    void build(CUDADevice *device, CUDAStream *stream) noexcept;
    /**
     * @brief Update the accel structure on device
     * 
     * @param device CUDADevice
     * @param stream CUDAStream
     */
    void update(CUDADevice *device, CUDAStream *stream) noexcept;
    /**
     * @brief Return OptixTraversableHandle
     * 
     * @return OptixTraversableHandle
     */
    [[nodiscard]] auto handle() const noexcept { return _handle; }
    /**
     * @brief Return the handle of instance buffer
     * 
     * @return handle of instance buffer
     */
    [[nodiscard]] auto instance_buffer() const noexcept { return _instance_buffer; }
    /**
     * @brief If resource is used
     * 
     * @param handle handle of resource
     * @return true 
     * @return false 
     */
    [[nodiscard]] bool uses_resource(uint64_t handle) const noexcept;
};

}// namespace luisa::compute::cuda
