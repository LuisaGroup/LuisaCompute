//
// Created by Mike on 2021/12/2.
//

#include <cuda.h>

#include <core/clock.h>
#include <backends/cuda/cuda_error.h>
#include <backends/cuda/cuda_buffer.h>
#include <backends/cuda/cuda_mesh.h>
#include <backends/cuda/cuda_command_encoder.h>
#include <backends/cuda/cuda_stream.h>
#include <backends/cuda/cuda_device.h>

namespace luisa::compute::cuda {

CUDAMesh::CUDAMesh(const AccelOption &option) noexcept
    : CUDAPrimitive{Tag::MESH, option} {}

CUDAMesh::~CUDAMesh() noexcept {
    if (_bvh_buffer_handle) {
        LUISA_CHECK_CUDA(cuMemFree(_bvh_buffer_handle));
    }
}

inline optix::BuildInput CUDAMesh::_make_build_input() const noexcept {
    optix::BuildInput build_input{};
    static const auto geometry_flag = static_cast<uint32_t>(optix::GEOMETRY_FLAG_DISABLE_ANYHIT);
    build_input.type = optix::BUILD_INPUT_TYPE_TRIANGLES;
    build_input.triangleArray.flags = &geometry_flag;
    build_input.triangleArray.vertexFormat = optix::VERTEX_FORMAT_FLOAT3;
    build_input.triangleArray.vertexBuffers = &_vertex_buffer;
    build_input.triangleArray.vertexStrideInBytes = _vertex_stride;
    build_input.triangleArray.numVertices = _vertex_buffer_size / _vertex_stride;
    build_input.triangleArray.indexBuffer = _triangle_buffer;
    build_input.triangleArray.indexFormat = optix::INDICES_FORMAT_UNSIGNED_INT3;
    build_input.triangleArray.indexStrideInBytes = sizeof(Triangle);
    build_input.triangleArray.numIndexTriplets = _triangle_buffer_size / sizeof(Triangle);
    build_input.triangleArray.numSbtRecords = 1u;
    return build_input;
}

void CUDAMesh::_build(CUDACommandEncoder &encoder) noexcept {

    auto build_input = _make_build_input();
    auto cuda_stream = encoder.stream()->handle();
    auto optix_ctx = encoder.stream()->device()->handle().optix_context();
    auto build_options = make_optix_build_options(option(), optix::BUILD_OPERATION_BUILD);

    Clock clock;
    optix::AccelBufferSizes sizes{};
    LUISA_CHECK_OPTIX(optix::api().accelComputeMemoryUsage(
        optix_ctx, &build_options,
        &build_input, 1u, &sizes));
    LUISA_INFO(
        "Computed mesh memory usage in {} ms: "
        "temp = {}, temp_update = {}, output = {}.",
        clock.toc(),
        sizes.tempSizeInBytes, sizes.tempUpdateSizeInBytes, sizes.outputSizeInBytes);
    _update_buffer_size = sizes.tempUpdateSizeInBytes;
    static constexpr auto align = [](size_t x) noexcept -> size_t {
        static constexpr auto alignment = optix::ACCEL_BUFFER_BYTE_ALIGNMENT;
        return (x + alignment - 1u) / alignment * alignment;
    };

    auto output_handle = static_cast<optix::TraversableHandle>(0u);
    if (option().allow_compaction) {// with compaction
        auto temp_buffer_offset = align(0u);
        auto output_buffer_offset = align(temp_buffer_offset + sizes.tempSizeInBytes);
        auto compacted_size_buffer_offset = align(output_buffer_offset + sizes.outputSizeInBytes);
        auto build_buffer_size = compacted_size_buffer_offset + sizeof(size_t);
        auto build_buffer = 0ull;
        LUISA_CHECK_CUDA(cuMemAllocAsync(&build_buffer, build_buffer_size, cuda_stream));
        auto compacted_size_buffer = build_buffer + compacted_size_buffer_offset;
        auto temp_buffer = build_buffer + temp_buffer_offset;
        auto output_buffer = build_buffer + output_buffer_offset;

        optix::AccelEmitDesc emit_desc{};
        emit_desc.type = optix::PROPERTY_TYPE_COMPACTED_SIZE;
        emit_desc.result = compacted_size_buffer;
        LUISA_CHECK_OPTIX(optix::api().accelBuild(
            optix_ctx, cuda_stream,
            &build_options, &build_input, 1, temp_buffer, sizes.tempSizeInBytes,
            output_buffer, sizes.outputSizeInBytes, &output_handle, &emit_desc, 1u));
        size_t compacted_size;
        LUISA_CHECK_CUDA(cuMemcpyDtoHAsync(&compacted_size, compacted_size_buffer, sizeof(size_t), cuda_stream));
        LUISA_CHECK_CUDA(cuStreamSynchronize(cuda_stream));
        LUISA_INFO("CUDAMesh compaction sizes: before = {}B, after = {}B, ratio = {}.",
                   sizes.outputSizeInBytes, compacted_size,
                   compacted_size / static_cast<double>(sizes.outputSizeInBytes));

        if (_bvh_buffer_size < compacted_size) {
            _bvh_buffer_size = compacted_size;
            if (_bvh_buffer_handle) { LUISA_CHECK_CUDA(cuMemFreeAsync(_bvh_buffer_handle, cuda_stream)); }
            LUISA_CHECK_CUDA(cuMemAllocAsync(&_bvh_buffer_handle, _bvh_buffer_size, cuda_stream));
        }
        LUISA_CHECK_OPTIX(optix::api().accelCompact(
            optix_ctx, cuda_stream, output_handle,
            _bvh_buffer_handle, _bvh_buffer_size, &output_handle));
        LUISA_CHECK_CUDA(cuMemFreeAsync(build_buffer, cuda_stream));
    } else {// without compaction
        if (_bvh_buffer_size < sizes.outputSizeInBytes) {
            _bvh_buffer_size = sizes.outputSizeInBytes;
            if (_bvh_buffer_handle) { LUISA_CHECK_CUDA(cuMemFreeAsync(_bvh_buffer_handle, cuda_stream)); }
            LUISA_CHECK_CUDA(cuMemAllocAsync(&_bvh_buffer_handle, _bvh_buffer_size, cuda_stream));
        }
        auto temp_buffer = 0ull;
        LUISA_CHECK_CUDA(cuMemAllocAsync(&temp_buffer, sizes.tempSizeInBytes, cuda_stream));
        LUISA_CHECK_OPTIX(optix::api().accelBuild(
            optix_ctx, cuda_stream, &build_options, &build_input, 1,
            temp_buffer, sizes.tempSizeInBytes, _bvh_buffer_handle,
            _bvh_buffer_size, &output_handle, nullptr, 0u));
        LUISA_CHECK_CUDA(cuMemFreeAsync(temp_buffer, cuda_stream));
    }
    // update handle
    LUISA_ASSERT(output_handle != 0ull, "OptiX BVH build failed.");
    _set_handle(output_handle);
}

void CUDAMesh::_update(CUDACommandEncoder &encoder) noexcept {
    auto build_input = _make_build_input();
    auto build_options = make_optix_build_options(option(), optix::BUILD_OPERATION_UPDATE);
    auto cuda_stream = encoder.stream()->handle();
    auto update_buffer = 0ull;
    LUISA_CHECK_CUDA(cuMemAllocAsync(&update_buffer, _update_buffer_size, cuda_stream));
    auto output_handle = static_cast<optix::TraversableHandle>(0u);
    LUISA_CHECK_OPTIX(optix::api().accelBuild(
        encoder.stream()->device()->handle().optix_context(), cuda_stream,
        &build_options, &build_input, 1u, update_buffer, _update_buffer_size,
        _bvh_buffer_handle, _bvh_buffer_size, &output_handle, nullptr, 0u));
    LUISA_CHECK_CUDA(cuMemFreeAsync(update_buffer, cuda_stream));
    if (output_handle) { _set_handle(output_handle); }
}

void CUDAMesh::build(CUDACommandEncoder &encoder, MeshBuildCommand *command) noexcept {

    auto vertex_buffer = reinterpret_cast<const CUDABuffer *>(command->vertex_buffer());
    auto triangle_buffer = reinterpret_cast<const CUDABuffer *>(command->triangle_buffer());

    auto requires_build =
        // not built yet
        handle() == 0u ||
        // not allowed to update
        !option().allow_update ||
        // user wants to force build
        command->request() == AccelBuildRequest::FORCE_BUILD ||
        // buffers changed
        vertex_buffer->handle() + command->vertex_buffer_offset() != _vertex_buffer ||
        command->vertex_buffer_size() != _vertex_buffer_size ||
        command->vertex_stride() != _vertex_stride ||
        triangle_buffer->handle() + command->triangle_buffer_offset() != _triangle_buffer ||
        command->triangle_buffer_size() != _triangle_buffer_size;

    // update buffers
    _vertex_buffer = vertex_buffer->handle() + command->vertex_buffer_offset();
    _vertex_buffer_size = command->vertex_buffer_size();
    _vertex_stride = command->vertex_stride();
    _triangle_buffer = triangle_buffer->handle() + command->triangle_buffer_offset();
    _triangle_buffer_size = command->triangle_buffer_size();

    // build or update
    if (requires_build) {
        _build(encoder);
    } else {
        _update(encoder);
    }
}

}// namespace luisa::compute::cuda
