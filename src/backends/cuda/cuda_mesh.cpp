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
    CUdeviceptr v_buffer, size_t v_stride, size_t v_count,
    CUdeviceptr t_buffer, size_t t_count, AccelBuildHint hint) noexcept
    : _vertex_buffer{v_buffer}, _vertex_stride{v_stride}, _vertex_count{v_count},
      _triangle_buffer{t_buffer}, _triangle_count{t_count}, _build_hint{hint} {}

void CUDAMesh::_initialize_build_parameters(OptixBuildInput *build_input, OptixAccelBuildOptions *build_options) const noexcept {
    static auto geometry_flag = static_cast<uint32_t>(OPTIX_GEOMETRY_FLAG_DISABLE_ANYHIT);
    build_input->type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
    build_input->triangleArray.flags = &geometry_flag;
    build_input->triangleArray.vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3;
    build_input->triangleArray.vertexBuffers = &_vertex_buffer;
    build_input->triangleArray.vertexStrideInBytes = _vertex_stride;
    build_input->triangleArray.numVertices = _vertex_count;
    build_input->triangleArray.indexBuffer = _triangle_buffer;
    build_input->triangleArray.indexFormat = OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
    build_input->triangleArray.indexStrideInBytes = sizeof(Triangle);
    build_input->triangleArray.numIndexTriplets = _triangle_count;
    build_input->triangleArray.numSbtRecords = 1u;
    switch (_build_hint) {
        case AccelBuildHint::FAST_TRACE:
            build_options->buildFlags = OPTIX_BUILD_FLAG_ALLOW_COMPACTION |
                                        OPTIX_BUILD_FLAG_ALLOW_UPDATE |
                                        OPTIX_BUILD_FLAG_PREFER_FAST_TRACE;
            break;
        case AccelBuildHint::FAST_UPDATE:
            build_options->buildFlags = OPTIX_BUILD_FLAG_ALLOW_COMPACTION |
                                        OPTIX_BUILD_FLAG_ALLOW_UPDATE;
            break;
        case AccelBuildHint::FAST_REBUILD:
            build_options->buildFlags = OPTIX_BUILD_FLAG_ALLOW_UPDATE |
                                        OPTIX_BUILD_FLAG_PREFER_FAST_BUILD;
            break;
    }
}

void CUDAMesh::build(CUDADevice *device, CUDAStream *stream) noexcept {

    OptixBuildInput build_input{};
    OptixAccelBuildOptions build_options{};
    _initialize_build_parameters(&build_input, &build_options);
    build_options.operation = OPTIX_BUILD_OPERATION_BUILD;

    auto optix_context = device->handle().optix_context();

    Clock clock;
    OptixAccelBufferSizes sizes;
    LUISA_CHECK_OPTIX(optixAccelComputeMemoryUsage(
        optix_context, &build_options, &build_input, 1u, &sizes));
    LUISA_INFO(
        "Computed mesh memory usage in {} ms: "
        "temp = {}, temp_update = {}, output = {}.",
        clock.toc(),
        sizes.tempSizeInBytes, sizes.tempUpdateSizeInBytes, sizes.outputSizeInBytes);
    if (_update_buffer_size < sizes.tempUpdateSizeInBytes) {
        LUISA_CHECK_CUDA(cuMemFree(_update_buffer));
        _update_buffer = 0u;
        _update_buffer_size = sizes.tempUpdateSizeInBytes;
    }

    auto align = [](size_t x) noexcept -> size_t {
        static constexpr auto alignment = OPTIX_ACCEL_BUFFER_BYTE_ALIGNMENT;
        return (x + alignment - 1u) / alignment * alignment;
    };
    if (_build_hint == AccelBuildHint::FAST_REBUILD) {// no compaction
        auto temp_buffer_offset = align(sizes.outputSizeInBytes);
        auto build_buffer_size = align(temp_buffer_offset + sizes.tempSizeInBytes);
        if (_buffer == 0u || _buffer_size < build_buffer_size) {
            LUISA_CHECK_CUDA(cuMemFree(_buffer));
            LUISA_CHECK_CUDA(cuMemAlloc(&_buffer, build_buffer_size));
            _buffer_size = build_buffer_size;
        }
        auto temp_buffer = _buffer + temp_buffer_offset;
        LUISA_CHECK_OPTIX(optixAccelBuild(
            optix_context, stream->handle(),
            &build_options, &build_input, 1,
            temp_buffer, sizes.tempSizeInBytes,
            _buffer, sizes.outputSizeInBytes,
            &_handle, nullptr, 0u));
    } else {// with compaction

        auto temp_buffer_offset = align(0u);
        auto output_buffer_offset = align(temp_buffer_offset + sizes.tempSizeInBytes);
        auto compacted_size_buffer_offset = align(output_buffer_offset + sizes.outputSizeInBytes);
        auto build_buffer_size = compacted_size_buffer_offset + sizeof(size_t);

        CUdeviceptr build_buffer;
        LUISA_CHECK_CUDA(cuMemAlloc(&build_buffer, build_buffer_size));
        auto compacted_size_buffer = build_buffer + compacted_size_buffer_offset;
        auto temp_buffer = build_buffer + temp_buffer_offset;
        auto output_buffer = build_buffer + output_buffer_offset;

        OptixAccelEmitDesc emit_desc{};
        emit_desc.type = OPTIX_PROPERTY_TYPE_COMPACTED_SIZE;
        emit_desc.result = compacted_size_buffer;
        LUISA_CHECK_OPTIX(optixAccelBuild(
            optix_context, stream->handle(),
            &build_options, &build_input, 1,
            temp_buffer, sizes.tempSizeInBytes,
            output_buffer, sizes.outputSizeInBytes,
            &_handle, &emit_desc, 1u));
        size_t compacted_size;
        LUISA_CHECK_CUDA(cuMemcpyDtoHAsync(&compacted_size, compacted_size_buffer, sizeof(size_t), stream->handle()));
        LUISA_CHECK_CUDA(cuStreamSynchronize(stream->handle()));
        LUISA_INFO("Compacted size: {}.", compacted_size);

        // do compaction...
        if (_buffer == 0u || _buffer_size < compacted_size) {
            LUISA_CHECK_CUDA(cuMemFree(_buffer));
            _buffer_size = compacted_size;
            LUISA_CHECK_CUDA(cuMemAlloc(&_buffer, _buffer_size));
        }
        LUISA_CHECK_OPTIX(optixAccelCompact(
            optix_context, stream->handle(),
            _handle, _buffer, _buffer_size, &_handle));
        LUISA_CHECK_CUDA(cuMemFree(build_buffer));
    }
}

void CUDAMesh::update(CUDADevice *device, CUDAStream *stream) noexcept {

    if (_update_buffer == 0u) {
        LUISA_CHECK_CUDA(cuMemAlloc(&_update_buffer, _update_buffer_size));
    }
    OptixBuildInput build_input{};
    OptixAccelBuildOptions build_options{};
    _initialize_build_parameters(&build_input, &build_options);
    build_options.operation = OPTIX_BUILD_OPERATION_UPDATE;

    auto optix_context = device->handle().optix_context();
    LUISA_CHECK_OPTIX(optixAccelBuild(
        optix_context, stream->handle(),
        &build_options, &build_input, 1u,
        _update_buffer, _update_buffer_size,
        _buffer, _buffer_size,
        &_handle, nullptr, 0u));
}

CUDAMesh::~CUDAMesh() noexcept {
    LUISA_CHECK_CUDA(cuMemFree(_buffer));
    LUISA_CHECK_CUDA(cuMemFree(_update_buffer));
}

}// namespace luisa::compute::cuda
