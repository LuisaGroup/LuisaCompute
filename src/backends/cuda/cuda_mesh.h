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

class CUDAMesh {

private:
    OptixTraversableHandle _handle{};
    uint64_t _bvh_buffer_handle{};
    size_t _bvh_buffer_size{};
    size_t _update_buffer_size{};
    uint64_t _vertex_buffer_handle{};
    CUdeviceptr _vertex_buffer;
    size_t _vertex_stride;
    size_t _vertex_count;
    uint64_t _triangle_buffer_handle{};
    CUdeviceptr _triangle_buffer;
    size_t _triangle_count;
    AccelBuildHint _build_hint;
    CUDAHeap *_heap{nullptr};

private:
    [[nodiscard]] OptixBuildInput _make_build_input() const noexcept;

public:
    CUDAMesh(CUdeviceptr v_buffer, size_t v_offset, size_t v_stride, size_t v_count,
             CUdeviceptr t_buffer, size_t t_offset, size_t t_count, AccelBuildHint hint) noexcept;
    void build(CUDADevice *device, CUDAStream *stream) noexcept;
    void update(CUDADevice *device, CUDAStream *stream) noexcept;
    [[nodiscard]] auto handle() const noexcept { return _handle; }
    [[nodiscard]] auto vertex_buffer_handle() const noexcept { return _vertex_buffer_handle; }
    [[nodiscard]] auto triangle_buffer_handle() const noexcept { return _triangle_buffer_handle; }
    ~CUDAMesh() noexcept;
};

}// namespace luisa::compute::cuda
