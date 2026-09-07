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
    if (_compatibility_descriptor) {
        _compatibility_descriptor->release();
    }
    if (_vertex_buffer) { _vertex_buffer->release(); }
    if (_triangle_buffer) { _triangle_buffer->release(); }
}

void MetalMesh::_do_add_resources(luisa::vector<MTL::Resource *> &resources) const noexcept {
    LUISA_ASSERT(_descriptor != nullptr ||
                     _compatibility_descriptor != nullptr,
                 "Mesh not built.");
    resources.emplace_back(_vertex_buffer);
    resources.emplace_back(_triangle_buffer);
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

    // check if build is needed
    using GeometryDescriptor = MTL4::AccelerationStructureTriangleGeometryDescriptor;
    using MotionGeometryDescriptor = MTL4::AccelerationStructureMotionTriangleGeometryDescriptor;
    auto geometry_buffers_changed =
        _vertex_buffer != vertex_buffer_handle ||
        _triangle_buffer != triangle_buffer_handle ||
        _vertex_buffer_offset != vertex_buffer_offset ||
        _vertex_buffer_size != vertex_buffer_size ||
        _vertex_stride != vertex_stride ||
        _triangle_buffer_offset != triangle_buffer_offset ||
        _triangle_buffer_size != triangle_buffer_size;
    auto address_driven =
        encoder.stream()->supports_address_driven_acceleration_structures();
    auto descriptor_missing = address_driven ?
                                  _descriptor == nullptr :
                                  _compatibility_descriptor == nullptr;
    auto requires_build = handle() == nullptr ||
                          !option().allow_update ||
                          command->request() == AccelBuildRequest::FORCE_BUILD ||
                          descriptor_missing ||
                          geometry_buffers_changed;

    if (requires_build) {
        if (_vertex_buffer != vertex_buffer_handle) {
            vertex_buffer_handle->retain();
            if (_vertex_buffer) { _vertex_buffer->release(); }
            _vertex_buffer = vertex_buffer_handle;
        }
        if (_triangle_buffer != triangle_buffer_handle) {
            triangle_buffer_handle->retain();
            if (_triangle_buffer) { _triangle_buffer->release(); }
            _triangle_buffer = triangle_buffer_handle;
        }
        _vertex_buffer_offset = vertex_buffer_offset;
        _vertex_buffer_size = vertex_buffer_size;
        _vertex_stride = vertex_stride;
        _triangle_buffer_offset = triangle_buffer_offset;
        _triangle_buffer_size = triangle_buffer_size;
        if (address_driven) {
            if (_descriptor) { _descriptor->release(); }
            _descriptor =
                MTL4::PrimitiveAccelerationStructureDescriptor::alloc()->init();
            _descriptor->setUsage(usage());
            _set_motion_options(_descriptor);
            if (option().motion) {
                auto geom_desc = NS::TransferPtr(
                    MotionGeometryDescriptor::alloc()->init());
                geom_desc->setVertexStride(vertex_stride);
                geom_desc->setVertexFormat(MTL::AttributeFormatFloat3);
                geom_desc->setIndexBuffer(MTL4::BufferRange{
                    triangle_buffer_handle->gpuAddress() +
                        triangle_buffer_offset,
                    triangle_buffer_size});
                geom_desc->setIndexType(MTL::IndexTypeUInt32);
                geom_desc->setTriangleCount(
                    triangle_buffer_size / triangle_stride);
                geom_desc->setOpaque(true);
                geom_desc->setAllowDuplicateIntersectionFunctionInvocation(
                    true);
                geom_desc->setIntersectionFunctionTableOffset(0u);
                auto object = static_cast<NS::Object *>(geom_desc.get());
                _descriptor->setGeometryDescriptors(
                    NS::Array::array(&object, 1u));
            } else {
                auto geom_desc = NS::TransferPtr(
                    GeometryDescriptor::alloc()->init());
                geom_desc->setVertexBuffer(MTL4::BufferRange{
                    vertex_buffer_handle->gpuAddress() +
                        vertex_buffer_offset,
                    vertex_buffer_size});
                geom_desc->setVertexStride(vertex_stride);
                geom_desc->setVertexFormat(MTL::AttributeFormatFloat3);
                geom_desc->setIndexBuffer(MTL4::BufferRange{
                    triangle_buffer_handle->gpuAddress() +
                        triangle_buffer_offset,
                    triangle_buffer_size});
                geom_desc->setIndexType(MTL::IndexTypeUInt32);
                geom_desc->setTriangleCount(
                    triangle_buffer_size / triangle_stride);
                geom_desc->setOpaque(true);
                geom_desc->setAllowDuplicateIntersectionFunctionInvocation(
                    true);
                geom_desc->setIntersectionFunctionTableOffset(0u);
                auto object = static_cast<NS::Object *>(geom_desc.get());
                _descriptor->setGeometryDescriptors(
                    NS::Array::array(&object, 1u));
            }
        } else {
            if (_compatibility_descriptor) {
                _compatibility_descriptor->release();
            }
            _compatibility_descriptor =
                MTL::PrimitiveAccelerationStructureDescriptor::alloc()->init();
            _compatibility_descriptor->setUsage(usage());
            _set_motion_options(_compatibility_descriptor);
            if (option().motion) {
                auto geom_desc = NS::TransferPtr(
                    MTL::AccelerationStructureMotionTriangleGeometryDescriptor::alloc()->init());
                luisa::vector<NS::Object *> keyframes;
                keyframes.reserve(motion_keyframe_count());
                auto pitch = vertex_buffer_size /
                             motion_keyframe_count();
                for (auto i = 0u; i < motion_keyframe_count(); i++) {
                    auto data = MTL::MotionKeyframeData::data();
                    data->setBuffer(vertex_buffer_handle);
                    data->setOffset(vertex_buffer_offset + i * pitch);
                    keyframes.emplace_back(data);
                }
                geom_desc->setVertexBuffers(NS::Array::array(
                    keyframes.data(), keyframes.size()));
                geom_desc->setVertexStride(vertex_stride);
                geom_desc->setVertexFormat(MTL::AttributeFormatFloat3);
                geom_desc->setIndexBuffer(triangle_buffer_handle);
                geom_desc->setIndexBufferOffset(triangle_buffer_offset);
                geom_desc->setIndexType(MTL::IndexTypeUInt32);
                geom_desc->setTriangleCount(
                    triangle_buffer_size / triangle_stride);
                geom_desc->setOpaque(true);
                geom_desc->setAllowDuplicateIntersectionFunctionInvocation(
                    true);
                geom_desc->setIntersectionFunctionTableOffset(0u);
                auto object = static_cast<NS::Object *>(geom_desc.get());
                _compatibility_descriptor->setGeometryDescriptors(
                    NS::Array::array(&object, 1u));
            } else {
                auto geom_desc = NS::TransferPtr(
                    MTL::AccelerationStructureTriangleGeometryDescriptor::alloc()->init());
                geom_desc->setVertexBuffer(vertex_buffer_handle);
                geom_desc->setVertexBufferOffset(vertex_buffer_offset);
                geom_desc->setVertexStride(vertex_stride);
                geom_desc->setVertexFormat(MTL::AttributeFormatFloat3);
                geom_desc->setIndexBuffer(triangle_buffer_handle);
                geom_desc->setIndexBufferOffset(triangle_buffer_offset);
                geom_desc->setIndexType(MTL::IndexTypeUInt32);
                geom_desc->setTriangleCount(
                    triangle_buffer_size / triangle_stride);
                geom_desc->setOpaque(true);
                geom_desc->setAllowDuplicateIntersectionFunctionInvocation(
                    true);
                geom_desc->setIntersectionFunctionTableOffset(0u);
                auto object = static_cast<NS::Object *>(geom_desc.get());
                _compatibility_descriptor->setGeometryDescriptors(
                    NS::Array::array(&object, 1u));
            }
        }
    }
    if (address_driven && option().motion) {
        auto geom_desc = _descriptor->geometryDescriptors()
                             ->object<MotionGeometryDescriptor>(0u);
        luisa::vector<MTL4::BufferRange> keyframes;
        keyframes.reserve(motion_keyframe_count());
        auto pitch = vertex_buffer_size / motion_keyframe_count();
        for (auto i = 0u; i < motion_keyframe_count(); i++) {
            keyframes.emplace_back(
                vertex_buffer_handle->gpuAddress() +
                    vertex_buffer_offset + i * pitch,
                pitch);
        }
        auto address = encoder.upload(
            keyframes.data(), keyframes.size() * sizeof(MTL4::BufferRange));
        geom_desc->setVertexBuffers(MTL4::BufferRange{
            address, keyframes.size() * sizeof(MTL4::BufferRange)});
    }
    encoder.use_resource(vertex_buffer_handle);
    encoder.use_resource(triangle_buffer_handle);
    if (requires_build) {
        if (address_driven) { _do_build(encoder, _descriptor); }
        else { _do_build(encoder, _compatibility_descriptor); }
    } else {
        if (address_driven) { _do_update(encoder, _descriptor); }
        else { _do_update(encoder, _compatibility_descriptor); }
    }
}

}// namespace luisa::compute::metal
