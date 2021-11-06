//
// Created by Mike Smith on 2021/7/22.
//

#import <backends/metal/metal_accel.h>
#import <backends/metal/metal_device.h>

namespace luisa::compute::metal {

id<MTLCommandBuffer> MetalAccel::build(
    MetalStream *stream,
    id<MTLCommandBuffer> command_buffer,
    AccelBuildHint hint,
    std::span<const uint64_t> mesh_handles,
    std::span<const float4x4> transforms,
    MetalSharedBufferPool *pool) noexcept {

    // build instance buffer
    auto instance_buffer_size = mesh_handles.size() * sizeof(MTLAccelerationStructureInstanceDescriptor);
    _instance_buffer = [_device->handle() newBufferWithLength:instance_buffer_size
                                                      options:MTLResourceStorageModePrivate | MTLResourceHazardTrackingModeUntracked];
    _instance_buffer_host = [_device->handle() newBufferWithLength:instance_buffer_size
                                                           options:MTLResourceStorageModeShared | MTLResourceHazardTrackingModeUntracked | MTLResourceOptionCPUCacheModeWriteCombined];
    auto instances = static_cast<MTLAccelerationStructureInstanceDescriptor *>(_instance_buffer_host.contents);
    for (auto i = 0u; i < mesh_handles.size(); i++) {
        instances[i].mask = 0xffffffffu;
        instances[i].accelerationStructureIndex = i;
        instances[i].intersectionFunctionTableOffset = 0u;
        instances[i].options = MTLAccelerationStructureInstanceOptionOpaque;
        auto t = transforms[i];
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
    [command_buffer enqueue];
    [command_buffer commit];// to avoid dead locks...
    _last_update = command_buffer;

    // build accel and (possibly) compact
    command_buffer = [[command_buffer commandQueue] commandBuffer];
    auto descriptor = [MTLInstanceAccelerationStructureDescriptor descriptor];

    _resources.clear();
    auto meshes = [[NSMutableArray<id<MTLAccelerationStructure>> alloc] init];
    for (auto mesh_handle : mesh_handles) {
        auto mesh = reinterpret_cast<MetalMesh *>(mesh_handle);
        [meshes addObject:mesh->handle()];
        _resources.emplace_back(mesh->handle());
        _resources.emplace_back(mesh->vertex_buffer());
        _resources.emplace_back(mesh->triangle_buffer());
    }
    _resources.emplace_back(_instance_buffer);
    std::sort(_resources.begin(), _resources.end());
    _resources.erase(std::unique(_resources.begin(), _resources.end()), _resources.end());

    LUISA_VERBOSE_WITH_LOCATION(
        "Building accel with reference to {} resource(s).",
        _resources.size());

    descriptor.instancedAccelerationStructures = meshes;
    descriptor.instanceCount = mesh_handles.size();
    descriptor.instanceDescriptorBuffer = _instance_buffer;
    _descriptor = descriptor;
    switch (hint) {
        case AccelBuildHint::FAST_TRACE: _descriptor.usage = MTLAccelerationStructureUsageNone; break;
        case AccelBuildHint::FAST_UPDATE: _descriptor.usage = MTLAccelerationStructureUsageRefit; break;
        case AccelBuildHint::FAST_REBUILD: _descriptor.usage = MTLAccelerationStructureUsagePreferFastBuild; break;
    }
    auto sizes = [_device->handle() accelerationStructureSizesWithDescriptor:_descriptor];
    _update_scratch_size = sizes.refitScratchBufferSize;
    _handle = [_device->handle() newAccelerationStructureWithSize:sizes.accelerationStructureSize];
    auto scratch_buffer = [_device->handle() newBufferWithLength:sizes.buildScratchBufferSize
                                                         options:MTLResourceStorageModePrivate | MTLResourceHazardTrackingModeUntracked];
    auto command_encoder = [command_buffer accelerationStructureCommandEncoder];
    [command_encoder buildAccelerationStructure:_handle
                                     descriptor:_descriptor
                                  scratchBuffer:scratch_buffer
                            scratchBufferOffset:0u];
    if (hint != AccelBuildHint::FAST_REBUILD) {
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
        _handle = [_device->handle() newAccelerationStructureWithSize:compacted_size];
        command_buffer = stream->command_buffer();
        command_encoder = [command_buffer accelerationStructureCommandEncoder];
        [command_encoder copyAndCompactAccelerationStructure:accel_before_compaction
                                     toAccelerationStructure:_handle];
    }
    [command_encoder endEncoding];
    return command_buffer;
}

id<MTLCommandBuffer> MetalAccel::update(
    MetalStream *stream,
    id<MTLCommandBuffer> command_buffer,
    std::span<const float4x4> transforms,
    size_t first) noexcept {

    if (!transforms.empty()) {
        // wait until last update finishes
        if (auto last_update = _last_update;
            last_update != nullptr) {
            [last_update waitUntilCompleted];
            _last_update = nullptr;
        }
        // now we can safely modify the instance buffer...
        using Instance = MTLAccelerationStructureInstanceDescriptor;
        auto instances = static_cast<Instance *>(_instance_buffer_host.contents);
        for (auto i = 0u; i < transforms.size(); i++) {
            auto t = transforms[i];
            for (auto c = 0; c < 4; c++) {
                instances[i + first].transformationMatrix[c] = {t[c].x, t[c].y, t[c].z};
            }
        }
        auto blit_encoder = [command_buffer blitCommandEncoder];
        [blit_encoder copyFromBuffer:_instance_buffer_host
                        sourceOffset:first * sizeof(Instance)
                            toBuffer:_instance_buffer
                   destinationOffset:first * sizeof(Instance)
                                size:transforms.size() * sizeof(Instance)];
        [blit_encoder endEncoding];
        _last_update = command_buffer;

        // commit the command buffer and start a new one to avoid dead locks
        stream->dispatch(command_buffer);
        command_buffer = stream->command_buffer();
    }

    // update accel
    if (_update_buffer == nullptr || _update_buffer.length < _update_scratch_size) {
        _update_buffer = [_device->handle() newBufferWithLength:_update_scratch_size
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

}
