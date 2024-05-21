#pragma once

#include <cuda.h>

#include <luisa/runtime/rtx/mesh.h>
#include "optix_api.h"
#include "cuda_primitive.h"

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
    CUdeviceptr _vertex_buffer{};
    size_t _vertex_buffer_size{};
    size_t _vertex_stride{};
    CUdeviceptr _triangle_buffer{};
    size_t _triangle_buffer_size{};

private:
    [[nodiscard]] optix::BuildInput _make_build_input() const noexcept override;

public:
    explicit CUDAMesh(const AccelOption &option) noexcept;
    ~CUDAMesh() noexcept override = default;
    void build(CUDACommandEncoder &encoder,
               MeshBuildCommand *command) noexcept;
};

}// namespace luisa::compute::cuda

