#include <luisa/core/logging.h>
#include <luisa/runtime/rtx/triangle.h>
#include "metal_command_encoder.h"
#include "metal_buffer.h"
#include "metal_mesh.h"

namespace luisa::compute::metal {

MetalMesh::MetalMesh(MTL::Device *device, const AccelOption &option) noexcept
    : MetalPrimitive{device, option} {}

MetalMesh::~MetalMesh() noexcept {
    if (_descriptor) { _descriptor->release(); }
}

void MetalMesh::_do_add_resources(luisa::vector<MTL::Resource *> &resources) const noexcept {
    auto mesh_desc = _descriptor->geometryDescriptors()
                         ->object<MTL::AccelerationStructureTriangleGeometryDescriptor>(0u);
    resources.emplace_back(mesh_desc->vertexBuffer());
    resources.emplace_back(mesh_desc->indexBuffer());
}

void MetalMesh::build(MetalCommandEncoder &encoder, MeshBuildCommand *command) noexcept {

    std::scoped_lock lock{mutex()};

    auto vertex_buffer = reinterpret_cast<MetalBuffer *>(command->vertex_buffer());
    auto vertex_buffer_handle = vertex_buffer->handle();
    auto vertex_buffer_offset = command->vertex_buffer_offset();
    auto vertex_buffer_size = command->vertex_buffer_size();
    auto vertex_stride = command->vertex_stride();
    LUISA_ASSERT(vertex_buffer_size % vertex_stride == 0u, "Invalid vertex buffer size.");

    auto triangle_buffer = reinterpret_cast<MetalBuffer *>(command->triangle_buffer());
    auto triangle_buffer_handle = triangle_buffer->handle();
    auto triangle_buffer_offset = command->triangle_buffer_offset();
    auto triangle_buffer_size = command->triangle_buffer_size();
    constexpr auto triangle_stride = sizeof(Triangle);
    LUISA_ASSERT(triangle_buffer_size % triangle_stride == 0u, "Invalid triangle buffer size.");

    auto geometry_buffers_changed = [&](auto desc) noexcept {
        return desc->vertexBuffer() != vertex_buffer_handle ||
               desc->vertexBufferOffset() != vertex_buffer_offset ||
               desc->vertexStride() != vertex_stride ||
               desc->indexBuffer() != triangle_buffer_handle ||
               desc->indexBufferOffset() != triangle_buffer_offset ||
               desc->triangleCount() * triangle_stride != triangle_buffer_size;
    };

    // check if build is needed
    using GeometryDescriptor = MTL::AccelerationStructureTriangleGeometryDescriptor;
    auto requires_build = handle() == nullptr ||
                          !option().allow_update ||
                          command->request() == AccelBuildRequest::FORCE_BUILD ||
                          _descriptor == nullptr ||
                          geometry_buffers_changed(_descriptor->geometryDescriptors()
                                                       ->object<GeometryDescriptor>(0u));

    if (requires_build) {
        if (_descriptor) { _descriptor->release(); }
        auto geom_desc = GeometryDescriptor::descriptor();
        geom_desc->setVertexBuffer(vertex_buffer_handle);
        geom_desc->setVertexBufferOffset(vertex_buffer_offset);
        geom_desc->setVertexStride(vertex_stride);
        geom_desc->setVertexFormat(MTL::AttributeFormatFloat3);
        geom_desc->setIndexBuffer(triangle_buffer_handle);
        geom_desc->setIndexBufferOffset(triangle_buffer_offset);
        geom_desc->setIndexType(MTL::IndexTypeUInt32);
        geom_desc->setTriangleCount(triangle_buffer_size / triangle_stride);
        geom_desc->setOpaque(true);
        geom_desc->setAllowDuplicateIntersectionFunctionInvocation(true);
        geom_desc->setIntersectionFunctionTableOffset(0u);
        auto geom_desc_object = static_cast<NS::Object *>(geom_desc);
        auto geom_desc_array = NS::Array::array(&geom_desc_object, 1u);
        _descriptor = MTL::PrimitiveAccelerationStructureDescriptor::alloc()->init();
        _descriptor->setGeometryDescriptors(geom_desc_array);
        _descriptor->setUsage(usage());
        _do_build(encoder, _descriptor);
    } else {
        _do_update(encoder, _descriptor);
    }
}

}// namespace luisa::compute::metal

