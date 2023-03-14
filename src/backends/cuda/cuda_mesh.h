//
// Created by Mike on 2021/12/2.
//

#pragma once

#include <cuda.h>

#include <runtime/rtx/mesh.h>
#include <backends/cuda/optix_api.h>
#include <backends/cuda/cuda_primitive.h>

namespace luisa::compute::cuda {

class CUDADevice;
class CUDAStream;
class CUDACommandEncoder;

/**
 * @brief Mesh of CUDA
 * 
 */
class CUDAMesh final : public CUDAPrimitive {

private:
    CUdeviceptr _bvh_buffer_handle{};
    size_t _bvh_buffer_size{};
    size_t _update_buffer_size{};
    CUdeviceptr _vertex_buffer{};
    size_t _vertex_buffer_size{};
    size_t _vertex_stride{};
    CUdeviceptr _triangle_buffer{};
    size_t _triangle_buffer_size{};

private:
    [[nodiscard]] optix::BuildInput _make_build_input() const noexcept;
    void _build(CUDACommandEncoder &encoder) noexcept;
    void _update(CUDACommandEncoder &encoder) noexcept;

public:
    explicit CUDAMesh(const AccelOption &option) noexcept;
    ~CUDAMesh() noexcept override;
    void build(CUDACommandEncoder &encoder, MeshBuildCommand *command) noexcept;
};

}// namespace luisa::compute::cuda
