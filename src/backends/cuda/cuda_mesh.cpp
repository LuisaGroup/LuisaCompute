#include "cuda_error.h"
#include "cuda_buffer.h"
#include "cuda_command_encoder.h"
#include "cuda_stream.h"
#include "cuda_device.h"
#include "cuda_mesh.h"

namespace luisa::compute::cuda {

CUDAMesh::CUDAMesh(const AccelOption &option) noexcept
    : CUDAPrimitive{Tag::MESH, option} {}

inline optix::BuildInput CUDAMesh::_make_build_input() noexcept {
    optix::BuildInput build_input{};
    static const auto geometry_flag = optix::GEOMETRY_FLAG_DISABLE_ANYHIT |
                                      optix::GEOMETRY_FLAG_DISABLE_TRIANGLE_FACE_CULLING;
    build_input.type = optix::BUILD_INPUT_TYPE_TRIANGLES;
    build_input.triangleArray.flags = &geometry_flag;
    build_input.triangleArray.vertexFormat = optix::VERTEX_FORMAT_FLOAT3;
    _per_frame_vertex_buffer.clear();
    _per_frame_vertex_buffer.resize(option().motion_option.num_keys);
    size_t per_frame_buffer_size = _vertex_buffer_size / option().motion_option.num_keys;
    for (size_t i = 0; i < option().motion_option.num_keys; ++i) {
        _per_frame_vertex_buffer[i] = _vertex_buffer + i * per_frame_buffer_size;
    }
    build_input.triangleArray.vertexBuffers = _per_frame_vertex_buffer.data();
    build_input.triangleArray.vertexStrideInBytes = _vertex_stride;
    build_input.triangleArray.numVertices = per_frame_buffer_size / _vertex_stride;
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
        vertex_buffer->device_address() + command->vertex_buffer_offset() != _vertex_buffer ||
        command->vertex_buffer_size() != _vertex_buffer_size ||
        command->vertex_stride() != _vertex_stride ||
        triangle_buffer->device_address() + command->triangle_buffer_offset() != _triangle_buffer ||
        command->triangle_buffer_size() != _triangle_buffer_size;

    // update buffers
    _vertex_buffer = vertex_buffer->device_address() + command->vertex_buffer_offset();
    _vertex_buffer_size = command->vertex_buffer_size();
    _vertex_stride = command->vertex_stride();
    _triangle_buffer = triangle_buffer->device_address() + command->triangle_buffer_offset();
    _triangle_buffer_size = command->triangle_buffer_size();

    // build or update
    if (requires_build) {
        _build(encoder);
    } else {
        _update(encoder);
    }
}

CUDAAnimatedMesh::CUDAAnimatedMesh(const MotionOption &option) noexcept
    : _option(option) {}

void CUDAAnimatedMesh::build(CUDACommandEncoder &encoder, AnimatedMeshBuildCommand *command) noexcept {
    auto cuda_stream = encoder.stream()->handle();
    auto optix_ctx = encoder.stream()->device()->handle().optix_context();
    
    auto matrix_buffer = reinterpret_cast<const CUDABuffer *>(command->matrix_buffer());
    _mesh_handle = command->mesh_handle();

    optix::MatrixMotionTransform motion_transform{};
    motion_transform.child = _mesh_handle;
    motion_transform.motionOptions = optix::MotionOptions{.numKeys = _option.num_keys, .flags = static_cast<uint16_t>(_option.flag), .timeBegin = _option.time_begin, .timeEnd = _option.time_end};
    
    size_t transform_size_in_bytes = sizeof(optix::MatrixMotionTransform) + (_option.num_keys - 2) * 12 * sizeof(float);
    LUISA_CHECK_CUDA(cuMemAllocAsync(&_motion_transform_buffer, transform_size_in_bytes, cuda_stream));
    size_t motion_transform_stub_size_in_bytes = sizeof(optix::MatrixMotionTransform) - 2 * 12 * sizeof(float);
    LUISA_ASSERT(sizeof(optix::MatrixMotionTransform) > 2 * 12 * sizeof(float), "Wrong memory layout!");
    LUISA_CHECK_CUDA(cuMemcpyHtoD(_motion_transform_buffer, &motion_transform, motion_transform_stub_size_in_bytes));
    LUISA_CHECK_CUDA(cuMemcpyDtoD(_motion_transform_buffer + motion_transform_stub_size_in_bytes, matrix_buffer->device_address(), _option.num_keys * 12 * sizeof(float)));
    LUISA_CHECK_OPTIX(optix::api().convertPointerToTraversableHandle(optix_ctx, _motion_transform_buffer, optix::TRAVERSABLE_TYPE_MATRIX_MOTION_TRANSFORM, &_handle));
}

}// namespace luisa::compute::cuda
