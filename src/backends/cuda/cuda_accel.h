//
// Created by Mike on 2021/12/2.
//

#pragma once

#include <vector>

#include <cuda.h>

#include <runtime/rtx/accel.h>
#include <core/dirty_range.h>
#include <backends/cuda/optix_api.h>

namespace luisa::compute::cuda {

class CUDAMesh;
class CUDAHeap;
class CUDADevice;
class CUDAStream;
class CUDACommandEncoder;

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
        optix::TraversableHandle handle;
        CUdeviceptr instances;
    };

private:
    optix::TraversableHandle _handle{};
    CUdeviceptr _instance_buffer{};
    size_t _instance_buffer_size{};
    CUdeviceptr _bvh_buffer{};
    size_t _bvh_buffer_size{};
    size_t _update_buffer_size{};
    luisa::vector<const CUDAMesh *> _meshes;
    luisa::vector<uint64_t> _mesh_handles;
    AccelOption _build_hint;

private:
    void _build(CUDADevice *device, CUDAStream *stream, CUstream cuda_stream) noexcept;
    void _update(CUDADevice *device, CUDAStream *stream, CUstream cuda_stream) noexcept;

public:
    explicit CUDAAccel(const AccelOption &option) noexcept;
    ~CUDAAccel() noexcept;
    void build(CUDACommandEncoder &encoder, AccelBuildCommand *command) noexcept;
    [[nodiscard]] auto handle() const noexcept { return _handle; }
    [[nodiscard]] auto instance_buffer() const noexcept { return _instance_buffer; }
};

}// namespace luisa::compute::cuda
