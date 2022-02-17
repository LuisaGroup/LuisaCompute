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
    MetalStream *stream, id<MTLCommandBuffer> command_buffer) noexcept {

    // build instance buffer
    auto device = [command_buffer device];
    auto instance_buffer_size = _instance_meshes.size() * sizeof(MTLAccelerationStructureInstanceDescriptor);
    if (_instance_buffer == nullptr || _instance_buffer.length < instance_buffer_size) {
        _instance_buffer = [device newBufferWithLength:instance_buffer_size
                                               options:MTLResourceStorageModePrivate];
    }
    _dirty_range.clear();

    auto pool = &stream->upload_host_buffer_pool();
    auto instance_buffer = pool->allocate(instance_buffer_size);
    if (instance_buffer.is_pooled()) {
        [command_buffer addCompletedHandler:^(id<MTLCommandBuffer>) {
          pool->recycle(instance_buffer);
        }];
    }

    auto instances = reinterpret_cast<MTLAccelerationStructureInstanceDescriptor *>(
        static_cast<std::byte *>([instance_buffer.handle() contents]) +
        instance_buffer.offset());
    for (auto i = 0u; i < _instance_meshes.size(); i++) {
        instances[i].mask = _instance_visibilities[i] ? ~0u : 0u;
        instances[i].accelerationStructureIndex = i;
        instances[i].intersectionFunctionTableOffset = 0u;
        instances[i].options = MTLAccelerationStructureInstanceOptionOpaque |
                               MTLAccelerationStructureInstanceOptionDisableTriangleCulling;
        auto t = _instance_transforms[i];
        for (auto c = 0; c < 4; c++) {
            instances[i].transformationMatrix[c] = {t[c].x, t[c].y, t[c].z};
        }
    }
    auto blit_encoder = [command_buffer blitCommandEncoder];
    [blit_encoder copyFromBuffer:instance_buffer.handle()
                    sourceOffset:instance_buffer.offset()
                        toBuffer:_instance_buffer
               destinationOffset:0u
                            size:instance_buffer_size];
    [blit_encoder endEncoding];

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
        std::unique(_resources.begin(), _resources.end()),
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
                                              options:MTLResourceStorageModePrivate];
    auto command_encoder = [command_buffer accelerationStructureCommandEncoder];
    [command_encoder buildAccelerationStructure:_handle
                                     descriptor:_descriptor
                                  scratchBuffer:scratch_buffer
                            scratchBufferOffset:0u];
    [command_encoder useResources:_resources.data()
                            count:_resources.size()
                            usage:MTLResourceUsageRead];
    [command_encoder endEncoding];

    //    if (_descriptor.usage != MTLAccelerationStructureUsagePreferFastBuild) {
    //
    //        auto download_pool = &stream->download_host_buffer_pool();
    //        auto compacted_size_buffer = download_pool->allocate(sizeof(uint));
    //        auto compaction_encoder = [command_buffer accelerationStructureCommandEncoder];
    //        [compaction_encoder writeCompactedAccelerationStructureSize:_handle
    //                                                           toBuffer:compacted_size_buffer.handle()
    //                                                             offset:compacted_size_buffer.offset()];
    //        [command_encoder useResources:_resources.data()
    //                                count:_resources.size()
    //                                usage:MTLResourceUsageRead];
    //        [compaction_encoder endEncoding];
    //        stream->dispatch(command_buffer);
    //        [command_buffer waitUntilCompleted];
    //
    //        auto compacted_size = *reinterpret_cast<const uint *>(
    //            static_cast<const std::byte *>(compacted_size_buffer.handle().contents) +
    //            compacted_size_buffer.offset());
    //        download_pool->recycle(compacted_size_buffer);
    //
    //        LUISA_INFO(
    //            "Accel size: before = {}, after = {}.",
    //            sizes.accelerationStructureSize, compacted_size);
    //
    //        auto accel_before_compaction = _handle;
    //        _handle = [device newAccelerationStructureWithSize:compacted_size];
    //        command_buffer = stream->command_buffer();
    //        compaction_encoder = [command_buffer accelerationStructureCommandEncoder];
    //        [compaction_encoder copyAndCompactAccelerationStructure:accel_before_compaction
    //                                        toAccelerationStructure:_handle];
    //        [compaction_encoder useResources:_resources.data()
    //                                   count:_resources.size()
    //                                   usage:MTLResourceUsageRead];
    //        [compaction_encoder endEncoding];
    //    }
    return command_buffer;
}

id<MTLCommandBuffer> MetalAccel::update(
    MetalStream *stream,
    id<MTLCommandBuffer> command_buffer) noexcept {

    if (!_dirty_range.empty()) {// some instances have been updated...

        using Instance = MTLAccelerationStructureInstanceDescriptor;
        auto dirty_instance_buffer_size = _dirty_range.size() * sizeof(Instance);

        auto pool = &stream->upload_host_buffer_pool();
        auto dirty_instance_buffer = pool->allocate(dirty_instance_buffer_size);
        if (dirty_instance_buffer.is_pooled()) {
            [command_buffer addCompletedHandler:^(id<MTLCommandBuffer>) {
              pool->recycle(dirty_instance_buffer);
            }];
        }

        auto instances = reinterpret_cast<Instance *>(
            static_cast<std::byte *>([dirty_instance_buffer.handle() contents]) +
            dirty_instance_buffer.offset());
        for (auto i = 0u; i < _dirty_range.size(); i++) {
            auto index = i + _dirty_range.offset();
            instances[i].mask = _instance_visibilities[i] ? ~0u : 0u;
            instances[i].accelerationStructureIndex = index;
            instances[i].intersectionFunctionTableOffset = 0u;
            instances[i].options = MTLAccelerationStructureInstanceOptionOpaque;
            auto t = _instance_transforms[index];
            for (auto c = 0; c < 4; c++) {
                instances[i].transformationMatrix[c] = {t[c].x, t[c].y, t[c].z};
            }
        }
        auto blit_encoder = [command_buffer blitCommandEncoder];
        [blit_encoder copyFromBuffer:dirty_instance_buffer.handle()
                        sourceOffset:dirty_instance_buffer.offset()
                            toBuffer:_instance_buffer
                   destinationOffset:_dirty_range.offset() * sizeof(Instance)
                                size:dirty_instance_buffer_size];
        [blit_encoder endEncoding];
        _dirty_range.clear();
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
    return command_buffer;
}

void MetalAccel::add_instance(MetalMesh *mesh, float4x4 transform, bool visible) noexcept {
    _instance_meshes.emplace_back(mesh);
    _instance_transforms.emplace_back(transform);
    _instance_visibilities.push_back(visible);
    _resource_handles.emplace(reinterpret_cast<uint64_t>(mesh));
    _resource_handles.emplace(reinterpret_cast<uint64_t>((__bridge void *)(mesh->vertex_buffer())));
    _resource_handles.emplace(reinterpret_cast<uint64_t>((__bridge void *)(mesh->triangle_buffer())));
}

void MetalAccel::set_transform(size_t index, float4x4 transform) noexcept {
    _instance_transforms[index] = transform;
    _dirty_range.mark(index);
}

void MetalAccel::set_visibility(size_t index, bool visible) noexcept {
    _instance_visibilities[index] = visible;
    _dirty_range.mark(index);
}

bool MetalAccel::uses_resource(uint64_t r) const noexcept {
    return _resource_handles.count(r) != 0u;
}

void MetalAccel::pop_instance() noexcept {
    _instance_meshes.pop_back();
    _instance_transforms.pop_back();
    _instance_visibilities.pop_back();
}

void MetalAccel::set_instance(size_t index, MetalMesh *mesh, float4x4 transform, bool visible) noexcept {
    _instance_meshes[index] = mesh;
    _instance_transforms[index] = transform;
    _instance_visibilities[index] = visible;
    _resource_handles.emplace(reinterpret_cast<uint64_t>(mesh));
    _resource_handles.emplace(reinterpret_cast<uint64_t>((__bridge void *)(mesh->vertex_buffer())));
    _resource_handles.emplace(reinterpret_cast<uint64_t>((__bridge void *)(mesh->triangle_buffer())));
}

}
