//
// Created by Mike Smith on 2021/7/22.
//

#import <backends/metal/metal_stream.h>
#import <backends/metal/metal_mesh.h>

namespace luisa::compute::metal {

MetalMesh::MetalMesh(
    id<MTLBuffer> v_buffer, size_t v_offset, size_t v_stride,
    id<MTLBuffer> t_buffer, size_t t_offset, size_t t_count, AccelBuildHint hint) noexcept {

    if (v_offset != 0u || t_offset != 0u) [[unlikely]] {
        LUISA_WARNING_WITH_LOCATION(
            "Metal seems to have trouble with non-zero-offset "
            "vertex and/or index buffers in mesh.");
    }
    auto mesh_desc = [MTLAccelerationStructureTriangleGeometryDescriptor descriptor];
    mesh_desc.vertexBuffer = v_buffer;
    mesh_desc.vertexBufferOffset = v_offset;
    mesh_desc.vertexStride = v_stride;
    mesh_desc.indexBuffer = t_buffer;
    mesh_desc.indexBufferOffset = t_offset;
    mesh_desc.indexType = MTLIndexTypeUInt32;
    mesh_desc.triangleCount = t_count;
    mesh_desc.opaque = YES;
    _descriptor = [MTLPrimitiveAccelerationStructureDescriptor descriptor];
    _descriptor.geometryDescriptors = @[mesh_desc];
    switch (hint) {
        case AccelBuildHint::FAST_TRACE: _descriptor.usage = MTLAccelerationStructureUsageNone; break;
        case AccelBuildHint::FAST_UPDATE: _descriptor.usage = MTLAccelerationStructureUsageRefit; break;
        case AccelBuildHint::FAST_REBUILD: _descriptor.usage = MTLAccelerationStructureUsagePreferFastBuild; break;
    }
}

id<MTLCommandBuffer> MetalMesh::build(MetalStream *stream, id<MTLCommandBuffer> command_buffer) noexcept {

    auto device = command_buffer.device;
    auto sizes = [device accelerationStructureSizesWithDescriptor:_descriptor];
    _update_buffer_size = sizes.refitScratchBufferSize;
    _handle = [device newAccelerationStructureWithSize:sizes.accelerationStructureSize];
    auto scratch_buffer = [device newBufferWithLength:sizes.buildScratchBufferSize
                                              options:MTLResourceStorageModePrivate |
                                                      MTLResourceHazardTrackingModeUntracked];
    auto command_encoder = [command_buffer accelerationStructureCommandEncoder];
    [command_encoder buildAccelerationStructure:_handle
                                     descriptor:_descriptor
                                  scratchBuffer:scratch_buffer
                            scratchBufferOffset:0u];

    if (_descriptor.usage != MTLAccelerationStructureUsagePreferFastBuild) {
        auto pool = &stream->download_host_buffer_pool();
        auto compacted_size_buffer = pool->allocate(sizeof(uint));
        [command_encoder writeCompactedAccelerationStructureSize:_handle
                                                        toBuffer:compacted_size_buffer.handle()
                                                          offset:compacted_size_buffer.offset()];
        [command_encoder endEncoding];
        stream->dispatch(command_buffer);
        [command_buffer waitUntilCompleted];
        auto compacted_size = *reinterpret_cast<const uint *>(
            static_cast<const std::byte *>(compacted_size_buffer.handle().contents) +
            compacted_size_buffer.offset());
        pool->recycle(compacted_size_buffer);

        auto accel_before_compaction = _handle;
        _handle = [device newAccelerationStructureWithSize:compacted_size];
        command_buffer = stream->command_buffer();
        command_encoder = [command_buffer accelerationStructureCommandEncoder];
        [command_encoder copyAndCompactAccelerationStructure:accel_before_compaction
                                     toAccelerationStructure:_handle];
    }
    [command_encoder endEncoding];
    return command_buffer;
}

id<MTLCommandBuffer> MetalMesh::update(
    MetalStream *stream,
    id<MTLCommandBuffer> command_buffer) noexcept {

    auto device = command_buffer.device;
    if (_update_buffer == nullptr || _update_buffer.length < _update_buffer_size) {
        _update_buffer = [device newBufferWithLength:_update_buffer_size
                                             options:MTLResourceStorageModePrivate];
    }
    auto command_encoder = [command_buffer accelerationStructureCommandEncoder];
    [command_encoder refitAccelerationStructure:_handle
                                     descriptor:_descriptor
                                    destination:_handle
                                  scratchBuffer:_update_buffer
                            scratchBufferOffset:0u];
    [command_encoder endEncoding];
    return command_buffer;
}

id<MTLBuffer> MetalMesh::vertex_buffer() const noexcept {
    auto geom_desc = static_cast<const MTLAccelerationStructureTriangleGeometryDescriptor *>(
        _descriptor.geometryDescriptors[0]);
    return geom_desc.vertexBuffer;
}

id<MTLBuffer> MetalMesh::triangle_buffer() const noexcept {
    auto geom_desc = static_cast<const MTLAccelerationStructureTriangleGeometryDescriptor *>(
        _descriptor.geometryDescriptors[0]);
    return geom_desc.indexBuffer;
}

}
