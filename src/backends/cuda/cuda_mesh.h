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
    std::vector<CUdeviceptr> _per_frame_vertex_buffer{};
    size_t _vertex_buffer_size{};
    size_t _vertex_stride{};
    CUdeviceptr _triangle_buffer{};
    size_t _triangle_buffer_size{};

private:
    [[nodiscard]] optix::BuildInput _make_build_input() noexcept override;

public:
    explicit CUDAMesh(const AccelOption &option) noexcept;
    ~CUDAMesh() noexcept override = default;
    void build(CUDACommandEncoder &encoder,
               MeshBuildCommand *command) noexcept;
};

class CUDAAnimatedMesh final {
private:
    optix::TraversableHandle _handle{};
    CUdeviceptr _matrix_buffer{};
    optix::TraversableHandle _mesh_handle{};
    CUdeviceptr _motion_transform_buffer{};

private:
    MotionOption _option;

public:
    explicit CUDAAnimatedMesh(const MotionOption &option) noexcept;
    ~CUDAAnimatedMesh() noexcept = default;
    void build(CUDACommandEncoder &encoder,
               AnimatedMeshBuildCommand *command) noexcept;
    
public:
    [[nodiscard]] auto pointer_to_handle() const noexcept { return &_handle; }
};

}// namespace luisa::compute::cuda
