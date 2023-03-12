//
// Created by Mike on 2021/12/2.
//

#include <cuda.h>

#include <backends/cuda/cuda_error.h>
#include <backends/cuda/cuda_mesh.h>
#include <backends/cuda/cuda_device.h>

namespace luisa::compute::cuda {

CUDAMesh::CUDAMesh(const AccelOption &option) noexcept {

}

CUDAMesh::~CUDAMesh() noexcept {
    if (_bvh_buffer_handle) {
        LUISA_CHECK_CUDA(cuMemFree(_bvh_buffer_handle));
    }
}

void CUDAMesh::build(CUDACommandEncoder &encoder, MeshBuildCommand *command) noexcept {

}

//
//CUDAMesh::CUDAMesh(
//    CUdeviceptr v_buffer, size_t v_offset, size_t v_stride, size_t v_count,
//    CUdeviceptr t_buffer, size_t t_offset, size_t t_count, AccelUsageHint hint) noexcept
//    : _vertex_buffer{v_buffer + v_offset},
//      _vertex_stride{v_stride}, _vertex_count{v_count},
//      _triangle_buffer{t_buffer + t_offset},
//      _triangle_count{t_count}, _build_hint{hint} {}
//
//inline optix::BuildInput CUDAMesh::_make_build_input() const noexcept {
//    optix::BuildInput build_input{};
//    static const auto geometry_flag = static_cast<uint32_t>(optix::GEOMETRY_FLAG_DISABLE_ANYHIT);
//    build_input.type = optix::BUILD_INPUT_TYPE_TRIANGLES;
//    build_input.triangleArray.flags = &geometry_flag;
//    build_input.triangleArray.vertexFormat = optix::VERTEX_FORMAT_FLOAT3;
//    build_input.triangleArray.vertexBuffers = &_vertex_buffer;
//    build_input.triangleArray.vertexStrideInBytes = _vertex_stride;
//    build_input.triangleArray.numVertices = _vertex_count;
//    build_input.triangleArray.indexBuffer = _triangle_buffer;
//    build_input.triangleArray.indexFormat = optix::INDICES_FORMAT_UNSIGNED_INT3;
//    build_input.triangleArray.indexStrideInBytes = sizeof(Triangle);
//    build_input.triangleArray.numIndexTriplets = _triangle_count;
//    build_input.triangleArray.numSbtRecords = 1u;
//    return build_input;
//}
//
//[[nodiscard]] inline auto cuda_mesh_build_options(AccelUsageHint hint, optix::BuildOperation op) noexcept {
//    optix::AccelBuildOptions build_options{};
//    build_options.operation = op;
//    switch (hint) {
//        case AccelUsageHint::FAST_TRACE:
//            build_options.buildFlags = optix::BUILD_FLAG_ALLOW_COMPACTION |
//                                       optix::BUILD_FLAG_PREFER_FAST_TRACE;
//            break;
//        case AccelUsageHint::FAST_UPDATE:
//            build_options.buildFlags = optix::BUILD_FLAG_ALLOW_COMPACTION |
//                                       optix::BUILD_FLAG_ALLOW_UPDATE;
//            break;
//        case AccelUsageHint::FAST_BUILD:
//            build_options.buildFlags = optix::BUILD_FLAG_PREFER_FAST_BUILD;
//            break;
//    }
//    return build_options;
//}
//
//void CUDAMesh::build(CUDADevice *device, CUDAStream *stream, const MeshBuildCommand *command) noexcept {
//
//    auto build_input = _make_build_input();
//    if (_handle != 0u && _build_hint == AccelUsageHint::FAST_UPDATE &&
//        command->request() == AccelBuildRequest::PREFER_UPDATE) {
//        auto build_options = cuda_mesh_build_options(
//            _build_hint, optix::BUILD_OPERATION_UPDATE);
//        auto cuda_stream = stream->handle();
//        auto update_buffer = 0ull;
//        LUISA_CHECK_CUDA(cuMemAllocAsync(&update_buffer, _update_buffer_size, cuda_stream));
//        LUISA_CHECK_OPTIX(optix::api().accelBuild(
//            device->handle().optix_context(), cuda_stream,
//            &build_options, &build_input, 1u, update_buffer, _update_buffer_size,
//            _bvh_buffer_handle, _bvh_buffer_size, &_handle, nullptr, 0u));
//        LUISA_CHECK_CUDA(cuMemFreeAsync(update_buffer, cuda_stream));
//        return;
//    }
//
//    Clock clock;
//    optix::AccelBufferSizes sizes{};
//    auto build_options = cuda_mesh_build_options(
//        _build_hint, optix::BUILD_OPERATION_BUILD);
//    LUISA_CHECK_OPTIX(optix::api().accelComputeMemoryUsage(
//        device->handle().optix_context(), &build_options,
//        &build_input, 1u, &sizes));
//    LUISA_INFO(
//        "Computed mesh memory usage in {} ms: "
//        "temp = {}, temp_update = {}, output = {}.",
//        clock.toc(),
//        sizes.tempSizeInBytes, sizes.tempUpdateSizeInBytes, sizes.outputSizeInBytes);
//    _update_buffer_size = sizes.tempUpdateSizeInBytes;
//    static constexpr auto align = [](size_t x) noexcept -> size_t {
//        static constexpr auto alignment = optix::ACCEL_BUFFER_BYTE_ALIGNMENT;
//        return (x + alignment - 1u) / alignment * alignment;
//    };
//    auto cuda_stream = stream->handle();
//    if (_build_hint == AccelUsageHint::FAST_BUILD) {// no compaction
//        if (_bvh_buffer_size < sizes.outputSizeInBytes) {
//            _bvh_buffer_size = sizes.outputSizeInBytes;
//            if (_bvh_buffer_handle) { LUISA_CHECK_CUDA(cuMemFreeAsync(_bvh_buffer_handle, cuda_stream)); }
//            LUISA_CHECK_CUDA(cuMemAllocAsync(&_bvh_buffer_handle, _bvh_buffer_size, cuda_stream));
//        }
//        auto temp_buffer = 0ull;
//        LUISA_CHECK_CUDA(cuMemAllocAsync(&temp_buffer, sizes.tempSizeInBytes, cuda_stream));
//        LUISA_CHECK_OPTIX(optix::api().accelBuild(
//            device->handle().optix_context(), cuda_stream,
//            &build_options, &build_input, 1, temp_buffer, sizes.tempSizeInBytes,
//            _bvh_buffer_handle, _bvh_buffer_size, &_handle, nullptr, 0u));
//        LUISA_CHECK_CUDA(cuMemFreeAsync(temp_buffer, cuda_stream));
//    } else {// with compaction
//        auto temp_buffer_offset = align(0u);
//        auto output_buffer_offset = align(temp_buffer_offset + sizes.tempSizeInBytes);
//        auto compacted_size_buffer_offset = align(output_buffer_offset + sizes.outputSizeInBytes);
//        auto build_buffer_size = compacted_size_buffer_offset + sizeof(size_t);
//        auto build_buffer = 0ull;
//        LUISA_CHECK_CUDA(cuMemAllocAsync(&build_buffer, build_buffer_size, cuda_stream));
//        auto compacted_size_buffer = build_buffer + compacted_size_buffer_offset;
//        auto temp_buffer = build_buffer + temp_buffer_offset;
//        auto output_buffer = build_buffer + output_buffer_offset;
//
//        optix::AccelEmitDesc emit_desc{};
//        emit_desc.type = optix::PROPERTY_TYPE_COMPACTED_SIZE;
//        emit_desc.result = compacted_size_buffer;
//        LUISA_CHECK_OPTIX(optix::api().accelBuild(
//            device->handle().optix_context(), cuda_stream,
//            &build_options, &build_input, 1, temp_buffer, sizes.tempSizeInBytes,
//            output_buffer, sizes.outputSizeInBytes, &_handle, &emit_desc, 1u));
//        size_t compacted_size;
//        LUISA_CHECK_CUDA(cuMemcpyDtoHAsync(&compacted_size, compacted_size_buffer, sizeof(size_t), cuda_stream));
//        LUISA_CHECK_CUDA(cuStreamSynchronize(cuda_stream));
//        LUISA_INFO("CUDAMesh compaction sizes: before = {}B, after = {}B, ratio = {}.",
//                   sizes.outputSizeInBytes, compacted_size,
//                   compacted_size / static_cast<double>(sizes.outputSizeInBytes));
//
//        if (_bvh_buffer_size < compacted_size) {
//            _bvh_buffer_size = compacted_size;
//            if (_bvh_buffer_handle) { LUISA_CHECK_CUDA(cuMemFreeAsync(_bvh_buffer_handle, cuda_stream)); }
//            LUISA_CHECK_CUDA(cuMemAllocAsync(&_bvh_buffer_handle, _bvh_buffer_size, cuda_stream));
//        }
//        LUISA_CHECK_OPTIX(optix::api().accelCompact(
//            device->handle().optix_context(), cuda_stream, _handle,
//            _bvh_buffer_handle, _bvh_buffer_size, &_handle));
//        LUISA_CHECK_CUDA(cuMemFreeAsync(build_buffer, cuda_stream));
//    }
//}
//

}// namespace luisa::compute::cuda
