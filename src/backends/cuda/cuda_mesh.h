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

class CUDAMesh {

private:
    OptixTraversableHandle _handle{};
    CUdeviceptr _buffer{};
    size_t _buffer_size{};
    CUdeviceptr _update_buffer{};
    size_t _update_buffer_size{};
    CUdeviceptr _vertex_buffer_origin{};
    CUdeviceptr _vertex_buffer;
    size_t _vertex_stride;
    size_t _vertex_count;
    CUdeviceptr _triangle_buffer_origin{};
    CUdeviceptr _triangle_buffer;
    size_t _triangle_count;
    AccelBuildHint _build_hint;

private:
    void _initialize_build_parameters(OptixBuildInput *build_input, OptixAccelBuildOptions *build_options) const noexcept;

public:
    CUDAMesh(CUdeviceptr v_buffer, size_t v_offset, size_t v_stride, size_t v_count,
             CUdeviceptr t_buffer, size_t t_offset, size_t t_count, AccelBuildHint hint) noexcept;
    void build(CUDADevice *device, CUDAStream *stream) noexcept;
    void update(CUDADevice *device, CUDAStream *stream) noexcept;
    [[nodiscard]] auto vertex_buffer_handle() const noexcept { return _vertex_buffer_origin; }
    [[nodiscard]] auto triangle_buffer_handle() const noexcept { return _triangle_buffer_origin; }
    ~CUDAMesh() noexcept;
};

}// namespace luisa::compute::cuda
