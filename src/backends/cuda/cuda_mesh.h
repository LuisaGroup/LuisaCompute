//
// Created by Mike on 2021/12/2.
//

#pragma once

#include <cuda.h>
#include <optix.h>

#include <rtx/mesh.h>

namespace luisa::compute::cuda {

class CUDADevice;
class CUDAStream;
class CUDAHeap;

/**
 * @brief Mesh of CUDA
 * 
 */
class CUDAMesh {

private:
    OptixTraversableHandle _handle{};
    CUdeviceptr _bvh_buffer_handle{};
    size_t _bvh_buffer_size{};
    size_t _update_buffer_size{};
    CUdeviceptr _vertex_buffer;
    size_t _vertex_stride;
    size_t _vertex_count;
    CUdeviceptr _triangle_buffer;
    size_t _triangle_count;
    AccelUsageHint _build_hint;
    CUDAHeap *_heap{nullptr};

private:
    [[nodiscard]] OptixBuildInput _make_build_input() const noexcept;

public:
    /**
     * @brief Construct a new CUDAMesh object
     * 
     * @param v_buffer handle of vertex buffer
     * @param v_offset offset of vertex buffer
     * @param v_stride stride of vertex buffer
     * @param v_count count of vertices
     * @param t_buffer handle of triangle buffer
     * @param t_offset offset of triangle buffer
     * @param t_count count of triangles
     * @param hint build hint
     */
    CUDAMesh(CUdeviceptr v_buffer, size_t v_offset, size_t v_stride, size_t v_count,
             CUdeviceptr t_buffer, size_t t_offset, size_t t_count, AccelUsageHint hint) noexcept;
    /**
     * @brief Destruct the CUDAMesh object
     */
    ~CUDAMesh() noexcept;
    /**
     * @brief Build mesh on CUDA
     * 
     * @param device CUDADeivce
     * @param stream CUDAStream
     * @param command command to build the mesh
     */
    void build(CUDADevice *device, CUDAStream *stream, const MeshBuildCommand *command) noexcept;
    /**
     * @brief Return Optix handle
     * 
     * @return OptixTraversableHandle
     */
    [[nodiscard]] auto handle() const noexcept { return _handle; }
};

}// namespace luisa::compute::cuda
