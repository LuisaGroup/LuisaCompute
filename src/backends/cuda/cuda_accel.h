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
    CUdeviceptr _instance_buffer{};
    size_t _instance_buffer_size{};
    CUdeviceptr _bvh_buffer{};
    size_t _bvh_buffer_size{};
    size_t _update_buffer_size{};
    CUDAHeap *_heap{nullptr};
    uint _instance_count{};
    AccelBuildHint _build_hint;

public:
    /**
     * @brief Construct a new CUDAAccel object
     * 
     * @param hint build hint
     */
    explicit CUDAAccel(AccelBuildHint hint) noexcept;
    ~CUDAAccel() noexcept;

    /**
     * @brief Build (or rebuild) the acceleration structure
     *
     * @param device pointer to the CUDA device
     * @param stream pointer to the CUDA stream
     * @param mesh_handles handles of the meshes to emplace in the acceleration structure
     * @param requests update requests from the host
     */
    void build(CUDADevice *device, CUDAStream *stream,
               luisa::span<const uint64_t> mesh_handles,
               luisa::span<const AccelUpdateRequest> requests) noexcept;
    /**
     * @brief Update the acceleration structure
     * 
     * @param device pointer to the CUDA device
     * @param stream pointer to the CUDA stream
     * @param requests update requests from the host
     */
    void update(CUDADevice *device, CUDAStream *stream,
                luisa::span<const AccelUpdateRequest> requests) noexcept;
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
};

}// namespace luisa::compute::cuda
