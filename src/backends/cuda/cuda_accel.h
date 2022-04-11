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
    CUDAHeap *_heap{nullptr};
    CUdeviceptr _instance_buffer{};
    size_t _instance_buffer_size{};
    CUdeviceptr _bvh_buffer{};
    size_t _bvh_buffer_size{};
    size_t _update_buffer_size{};
    luisa::vector<const CUDAMesh *> _meshes;
    luisa::vector<uint64_t> _mesh_handles;
    AccelUsageHint _build_hint;

private:
    void _build(CUDADevice *device, CUDAStream *stream, CUstream cuda_stream) noexcept;
    void _update(CUDADevice *device, CUDAStream *stream, CUstream cuda_stream) noexcept;

public:
    /**
     * @brief Construct a new CUDAAccel object
     * 
     * @param hint build hint
     */
    CUDAAccel(AccelUsageHint hint, CUDAHeap *heap) noexcept;
    ~CUDAAccel() noexcept;

    /**
     * @brief Build (or rebuild) the acceleration structure
     *
     * @param device pointer to the CUDA device
     * @param stream pointer to the CUDA stream
     * @param command command to build the acceleration structure
     */
    void build(CUDADevice *device, CUDAStream *stream, const AccelBuildCommand *command) noexcept;
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
