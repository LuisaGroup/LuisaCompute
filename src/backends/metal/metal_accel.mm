//
// Created by Mike Smith on 2021/7/22.
//

#import <backends/metal/metal_accel.h>
#import <backends/metal/metal_device.h>

namespace luisa::compute::metal {

MetalAccel::MetalAccel(AccelBuildHint hint) noexcept {
    _descriptor = [MTLInstanceAccelerationStructureDescriptor descriptor];
    switch (hint) {
        case AccelBuildHint::FAST_TRACE: _descriptor.usage = MTLAccelerationStructureUsageNone; break;
        case AccelBuildHint::FAST_UPDATE: _descriptor.usage = MTLAccelerationStructureUsageRefit; break;
        case AccelBuildHint::FAST_REBUILD: _descriptor.usage = MTLAccelerationStructureUsagePreferFastBuild; break;
    }
}

id<MTLCommandBuffer> MetalAccel::build(
    MetalStream *stream,
    id<MTLCommandBuffer> command_buffer,
    MetalSharedBufferPool *pool) noexcept {

    // build instance buffer
    auto device = [command_buffer device];
    auto instance_buffer_size = _instance_meshes.size() * sizeof(MTLAccelerationStructureInstanceDescriptor);
    if (_instance_buffer == nullptr || _instance_buffer.length < instance_buffer_size) {
        _instance_buffer = [device newBufferWithLength:instance_buffer_size
                                               options:MTLResourceStorageModePrivate |
                                                       MTLResourceHazardTrackingModeUntracked];
        _instance_buffer_host = [device newBufferWithLength:instance_buffer_size
                                                    options:MTLResourceStorageModeShared |
                                                            MTLResourceHazardTrackingModeUntracked |
                                                            MTLResourceOptionCPUCacheModeWriteCombined];
    } else if (id<MTLCommandBuffer> last = _instance_buffer_copy_command) {
        [last waitUntilCompleted];
    }
    auto instances = static_cast<MTLAccelerationStructureInstanceDescriptor *>(_instance_buffer_host.contents);
    for (auto i = 0u; i < _instance_meshes.size(); i++) {
        instances[i].mask = 0xffffffffu;
        instances[i].accelerationStructureIndex = i;
        instances[i].intersectionFunctionTableOffset = 0u;
        instances[i].options = MTLAccelerationStructureInstanceOptionOpaque;
        auto t = _instance_transforms[i];
        for (auto c = 0; c < 4; c++) {
            instances[i].transformationMatrix[c] = {t[c].x, t[c].y, t[c].z};
        }
    }
    auto blit_encoder = [command_buffer blitCommandEncoder];
    [blit_encoder copyFromBuffer:_instance_buffer_host
                    sourceOffset:0u
                        toBuffer:_instance_buffer
               destinationOffset:0u
                            size:instance_buffer_size];
    [blit_encoder endEncoding];
    _instance_buffer_copy_command = command_buffer;

    // to avoid dead locks
    stream->dispatch(command_buffer);
    command_buffer = stream->command_buffer();

    // build accel and (possibly) compact
    _resources.clear();
    auto meshes = [[NSMutableArray<id<MTLAccelerationStructure>> alloc] init];
    for (auto mesh : _instance_meshes) {
        [meshes addObject:mesh->handle()];
        _resources.emplace_back(mesh->handle());
        _resources.emplace_back(mesh->vertex_buffer());
        _resources.emplace_back(mesh->triangle_buffer());
    }
    _resources.emplace_back(_instance_buffer);

    // sort resources...
    std::sort(_resources.begin(), _resources.end());
    _resources.erase(
        std::unique(
            _resources.begin(), _resources.end()),
        _resources.end());

    LUISA_VERBOSE_WITH_LOCATION(
        "Building accel with reference to {} resource(s).",
        _resources.size());

    _descriptor.instancedAccelerationStructures = meshes;
    _descriptor.instanceCount = _instance_meshes.size();
    _descriptor.instanceDescriptorBuffer = _instance_buffer;
    auto sizes = [device accelerationStructureSizesWithDescriptor:_descriptor];
    _update_scratch_size = sizes.refitScratchBufferSize;
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
        auto compacted_size_buffer = pool->allocate();
        [command_encoder writeCompactedAccelerationStructureSize:_handle
                                                        toBuffer:compacted_size_buffer.handle()
                                                          offset:compacted_size_buffer.offset()];
        [command_encoder endEncoding];
        auto compacted_size = 0u;
        auto p_compacted_size = &compacted_size;
        [command_buffer addCompletedHandler:^(id<MTLCommandBuffer>) {
          *p_compacted_size = *reinterpret_cast<const uint *>(
              static_cast<const std::byte *>(compacted_size_buffer.handle().contents) + compacted_size_buffer.offset());
          pool->recycle(compacted_size_buffer);
        }];

        stream->dispatch(command_buffer);
        [command_buffer waitUntilCompleted];
        auto accel_before_compaction = _handle;
        _handle = [device newAccelerationStructureWithSize:compacted_size];
        command_buffer = stream->command_buffer();
        command_encoder = [command_buffer accelerationStructureCommandEncoder];
        [command_encoder copyAndCompactAccelerationStructure:accel_before_compaction
                                     toAccelerationStructure:_handle];
    }
    [command_encoder endEncoding];

    _dirty_range.clear();
    return command_buffer;
}

id<MTLCommandBuffer> MetalAccel::update(
    MetalStream *stream,
    id<MTLCommandBuffer> command_buffer) noexcept {

    if (!_dirty_range.empty()) {// some instances have been updated...
        // wait for last copy
        if (id<MTLCommandBuffer> last = _instance_buffer_copy_command) {
            [last waitUntilCompleted];
        }
        using Instance = MTLAccelerationStructureInstanceDescriptor;
        auto instances = static_cast<Instance *>(_instance_buffer_host.contents);
        for (auto i = 0u; i < _dirty_range.size(); i++) {
            auto index = i + _dirty_range.offset();
            auto t = _instance_transforms[index];
            for (auto c = 0; c < 4; c++) {
                instances[index].transformationMatrix[c] = {t[c].x, t[c].y, t[c].z};
            }
        }
        auto blit_encoder = [command_buffer blitCommandEncoder];
        [blit_encoder copyFromBuffer:_instance_buffer_host
                        sourceOffset:_dirty_range.offset() * sizeof(Instance)
                            toBuffer:_instance_buffer
                   destinationOffset:_dirty_range.offset() * sizeof(Instance)
                                size:_dirty_range.size() * sizeof(Instance)];
        [blit_encoder endEncoding];
        _instance_buffer_copy_command = command_buffer;

        // commit the command buffer and start a new one to avoid dead locks
        stream->dispatch(command_buffer);
        command_buffer = stream->command_buffer();
    }

    // update accel
    auto device = [command_buffer device];
    if (_update_buffer == nullptr || _update_buffer.length < _update_scratch_size) {
        _update_buffer = [device newBufferWithLength:_update_scratch_size
                                             options:MTLResourceStorageModePrivate];
    }
    auto command_encoder = [command_buffer accelerationStructureCommandEncoder];
    [command_encoder refitAccelerationStructure:_handle
                                     descriptor:_descriptor
                                    destination:_handle
                                  scratchBuffer:_update_buffer
                            scratchBufferOffset:0u];
    [command_encoder endEncoding];

    _dirty_range.clear();
    return command_buffer;
}

void MetalAccel::add_instance(MetalMesh *mesh, float4x4 transform) noexcept {
    _instance_meshes.emplace_back(mesh);
    _instance_transforms.emplace_back(transform);
}

void MetalAccel::set_transform(size_t index, float4x4 transform) noexcept {
    _instance_transforms[index] = transform;
    _dirty_range.mark(index);
}

bool MetalAccel::uses_resource(id<MTLResource> r) const noexcept {
    return std::binary_search(_resources.cbegin(), _resources.cend(), r);
}

}
