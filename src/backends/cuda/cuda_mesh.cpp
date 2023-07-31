#include <cuda.h>

#include "cuda_error.h"
#include "cuda_buffer.h"
#include "cuda_mesh.h"
#include "cuda_command_encoder.h"
#include "cuda_stream.h"
#include "cuda_device.h"

namespace luisa::compute::cuda {

CUDAMesh::CUDAMesh(const AccelOption &option) noexcept
    : CUDAPrimitive{Tag::MESH, option} {}

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

void CUDAMesh::build(CUDACommandEncoder &encoder, MeshBuildCommand *command) noexcept {

    auto vertex_buffer = reinterpret_cast<const CUDABuffer *>(command->vertex_buffer());
    auto triangle_buffer = reinterpret_cast<const CUDABuffer *>(command->triangle_buffer());
    LUISA_ASSERT(command->vertex_buffer_offset() + command->vertex_buffer_size() <= vertex_buffer->size_bytes(),
                 "Vertex buffer offset + size exceeds buffer size {}.");
    LUISA_ASSERT(command->triangle_buffer_offset() + command->triangle_buffer_size() <= triangle_buffer->size_bytes(),
                 "Triangle buffer offset + size exceeds buffer size {}.");

    std::scoped_lock lock{_mutex};

    auto requires_build =
        // not built yet
        _handle == 0u ||
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

