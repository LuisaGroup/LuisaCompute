//
// Created by Mike on 2021/12/2.
//

#include <cuda.h>
#include <optix_stubs.h>

#include <backends/cuda/cuda_error.h>
#include <backends/cuda/cuda_mesh.h>
#include <backends/cuda/cuda_device.h>

namespace luisa::compute::cuda {

CUDAMesh::CUDAMesh(
    CUdeviceptr v_buffer, size_t v_offset, size_t v_stride, size_t v_count,
    CUdeviceptr t_buffer, size_t t_offset, size_t t_count, AccelUsageHint hint) noexcept
    : _vertex_buffer_handle{v_buffer},
      _vertex_buffer{CUDAHeap::buffer_address(v_buffer) + v_offset},
      _vertex_stride{v_stride}, _vertex_count{v_count},
      _triangle_buffer_handle{t_buffer},
      _triangle_buffer{CUDAHeap::buffer_address(t_buffer) + t_offset},
      _triangle_count{t_count}, _build_hint{hint} {}

inline OptixBuildInput CUDAMesh::_make_build_input() const noexcept {
    OptixBuildInput build_input{};
    static const auto geometry_flag = static_cast<uint32_t>(OPTIX_GEOMETRY_FLAG_DISABLE_ANYHIT);
    build_input.type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
    build_input.triangleArray.flags = &geometry_flag;
    build_input.triangleArray.vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3;
    build_input.triangleArray.vertexBuffers = &_vertex_buffer;
    build_input.triangleArray.vertexStrideInBytes = _vertex_stride;
    build_input.triangleArray.numVertices = _vertex_count;
    build_input.triangleArray.indexBuffer = _triangle_buffer;
    build_input.triangleArray.indexFormat = OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
    build_input.triangleArray.indexStrideInBytes = sizeof(Triangle);
    build_input.triangleArray.numIndexTriplets = _triangle_count;
    build_input.triangleArray.numSbtRecords = 1u;
    return build_input;
}

[[nodiscard]] inline auto make_build_options(AccelUsageHint hint, OptixBuildOperation op) noexcept {
    OptixAccelBuildOptions build_options{};
    build_options.operation = op;
    switch (hint) {
        case AccelUsageHint::FAST_TRACE:
            build_options.buildFlags = OPTIX_BUILD_FLAG_ALLOW_COMPACTION |
                                       OPTIX_BUILD_FLAG_ALLOW_UPDATE |
                                       OPTIX_BUILD_FLAG_PREFER_FAST_TRACE;
            break;
        case AccelUsageHint::FAST_UPDATE:
            build_options.buildFlags = OPTIX_BUILD_FLAG_ALLOW_COMPACTION |
                                       OPTIX_BUILD_FLAG_ALLOW_UPDATE;
            break;
        case AccelUsageHint::FAST_BUILD:
            build_options.buildFlags = OPTIX_BUILD_FLAG_ALLOW_UPDATE |
                                       OPTIX_BUILD_FLAG_PREFER_FAST_BUILD;
            break;
    }
    return build_options;
}

void CUDAMesh::build(CUDADevice *device, CUDAStream *stream, const MeshBuildCommand *command) noexcept {

    auto build_input = _make_build_input();
    if (_handle != 0u && command->request() == AccelBuildRequest::PREFER_UPDATE) {
        auto build_options = make_build_options(
            _build_hint, OPTIX_BUILD_OPERATION_UPDATE);
        auto update_buffer = _heap->allocate(_update_buffer_size);
        auto cuda_stream = stream->handle();
        LUISA_CHECK_OPTIX(optixAccelBuild(
            device->handle().optix_context(), cuda_stream,
            &build_options, &build_input, 1u,
            CUDAHeap::buffer_address(update_buffer),
            _update_buffer_size,
            CUDAHeap::buffer_address(_bvh_buffer_handle),
            _bvh_buffer_size,
            &_handle, nullptr, 0u));
        stream->emplace_callback(
            CUDAHeap::BufferFreeContext::create(
                _heap, update_buffer));
        return;
    }

    Clock clock;
    OptixAccelBufferSizes sizes;
    auto build_options = make_build_options(
        _build_hint, OPTIX_BUILD_OPERATION_BUILD);
    LUISA_CHECK_OPTIX(optixAccelComputeMemoryUsage(
        device->handle().optix_context(), &build_options,
        &build_input, 1u, &sizes));
    LUISA_INFO(
        "Computed mesh memory usage in {} ms: "
        "temp = {}, temp_update = {}, output = {}.",
        clock.toc(),
        sizes.tempSizeInBytes, sizes.tempUpdateSizeInBytes, sizes.outputSizeInBytes);
    _update_buffer_size = sizes.tempUpdateSizeInBytes;
    static constexpr auto align = [](size_t x) noexcept -> size_t {
        static constexpr auto alignment = OPTIX_ACCEL_BUFFER_BYTE_ALIGNMENT;
        return (x + alignment - 1u) / alignment * alignment;
    };
    _heap = device->heap();
    auto cuda_stream = stream->handle();
    if (_build_hint == AccelUsageHint::FAST_BUILD) {// no compaction
        if (_bvh_buffer_size < sizes.outputSizeInBytes) {
            stream->emplace_callback(
                CUDAHeap::BufferFreeContext::create(
                    _heap, _bvh_buffer_handle));
            _bvh_buffer_handle = _heap->allocate(sizes.outputSizeInBytes);
            _bvh_buffer_size = sizes.outputSizeInBytes;
        }
        auto temp_buffer = _heap->allocate(sizes.tempSizeInBytes);
        LUISA_CHECK_OPTIX(optixAccelBuild(
            device->handle().optix_context(), cuda_stream,
            &build_options, &build_input, 1,
            CUDAHeap::buffer_address(temp_buffer), sizes.tempSizeInBytes,
            CUDAHeap::buffer_address(_bvh_buffer_handle),
            _bvh_buffer_size,
            &_handle, nullptr, 0u));
        stream->emplace_callback(
            CUDAHeap::BufferFreeContext::create(
                _heap, temp_buffer));
    } else {// with compaction
        auto temp_buffer_offset = align(0u);
        auto output_buffer_offset = align(temp_buffer_offset + sizes.tempSizeInBytes);
        auto compacted_size_buffer_offset = align(output_buffer_offset + sizes.outputSizeInBytes);
        auto build_buffer_size = compacted_size_buffer_offset + sizeof(size_t);
        auto build_buffer = _heap->allocate(build_buffer_size);
        auto build_buffer_address = CUDAHeap::buffer_address(build_buffer);

        auto compacted_size_buffer = build_buffer_address + compacted_size_buffer_offset;
        auto temp_buffer = build_buffer_address + temp_buffer_offset;
        auto output_buffer = build_buffer_address + output_buffer_offset;

        OptixAccelEmitDesc emit_desc{};
        emit_desc.type = OPTIX_PROPERTY_TYPE_COMPACTED_SIZE;
        emit_desc.result = compacted_size_buffer;
        LUISA_CHECK_OPTIX(optixAccelBuild(
            device->handle().optix_context(), cuda_stream,
            &build_options, &build_input, 1,
            temp_buffer, sizes.tempSizeInBytes,
            output_buffer, sizes.outputSizeInBytes,
            &_handle, &emit_desc, 1u));
        size_t compacted_size;
        LUISA_CHECK_CUDA(cuMemcpyDtoHAsync(&compacted_size, compacted_size_buffer, sizeof(size_t), cuda_stream));
        LUISA_CHECK_CUDA(cuStreamSynchronize(cuda_stream));
        LUISA_INFO("Compacted size: {}.", compacted_size);

        if (_bvh_buffer_size < compacted_size) {
            stream->emplace_callback(
                CUDAHeap::BufferFreeContext::create(
                    _heap, _bvh_buffer_handle));
            _bvh_buffer_size = compacted_size;
            _bvh_buffer_handle = _heap->allocate(_bvh_buffer_size);
        }
        LUISA_CHECK_OPTIX(optixAccelCompact(
            device->handle().optix_context(),
            cuda_stream, _handle,
            CUDAHeap::buffer_address(_bvh_buffer_handle),
            _bvh_buffer_size, &_handle));
        stream->emplace_callback(
            CUDAHeap::BufferFreeContext::create(
                _heap, build_buffer));
    }
}

CUDAMesh::~CUDAMesh() noexcept {
    if (_heap != nullptr) {
        _heap->free(_bvh_buffer_handle);
    }
}

}// namespace luisa::compute::cuda
