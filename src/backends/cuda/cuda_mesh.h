//
// Created by Mike on 2021/12/2.
//

#pragma once

#include <cuda.h>

#include <runtime/rtx/mesh.h>
#include <backends/cuda/optix_api.h>

namespace luisa::compute::cuda {

class CUDADevice;
class CUDAStream;
class CUDACommandEncoder;

/**
 * @brief Mesh of CUDA
 * 
 */
class CUDAMesh {

private:
    optix::TraversableHandle _handle{};
    CUdeviceptr _bvh_buffer_handle{};
    size_t _bvh_buffer_size{};
    size_t _update_buffer_size{};
    CUdeviceptr _vertex_buffer;
    size_t _vertex_stride;
    size_t _vertex_count;
    CUdeviceptr _triangle_buffer;
    size_t _triangle_count;
    AccelOption _build_hint;

private:
    [[nodiscard]] optix::BuildInput _make_build_input() const noexcept;

public:
    explicit CUDAMesh(const AccelOption &option) noexcept;
    ~CUDAMesh() noexcept;
    void build(CUDACommandEncoder &encoder, MeshBuildCommand *command) noexcept;
    [[nodiscard]] auto handle() const noexcept { return _handle; }
};

}// namespace luisa::compute::cuda
