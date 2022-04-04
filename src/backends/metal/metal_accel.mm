//
// Created by Mike Smith on 2021/7/22.
//

#import <backends/metal/metal_accel.h>
#import <backends/metal/metal_device.h>

namespace luisa::compute::metal {

MetalAccel::MetalAccel(id<MTLComputePipelineState> update_shader, AccelBuildHint hint) noexcept
    : _update_shader{update_shader} {
    _descriptor = [MTLInstanceAccelerationStructureDescriptor descriptor];
    switch (hint) {
        case AccelBuildHint::FAST_TRACE: _descriptor.usage = MTLAccelerationStructureUsageNone; break;
        case AccelBuildHint::FAST_UPDATE: _descriptor.usage = MTLAccelerationStructureUsageRefit; break;
        case AccelBuildHint::FAST_REBUILD: _descriptor.usage = MTLAccelerationStructureUsagePreferFastBuild; break;
    }
}

id<MTLCommandBuffer> MetalAccel::build(
    MetalStream *stream, id<MTLCommandBuffer> command_buffer,
    luisa::span<const uint64_t> mesh_handles,
    luisa::span<const AccelUpdateRequest> requests) noexcept {

    // create instance buffer
    auto device = [command_buffer device];
    auto instance_buffer_size = mesh_handles.size() *
                                sizeof(MTLAccelerationStructureInstanceDescriptor);
    if (_instance_buffer == nullptr || _instance_buffer.length < instance_buffer_size) {
        _instance_buffer = [device newBufferWithLength:next_pow2(instance_buffer_size)
                                               options:MTLResourceStorageModePrivate];
    }

    // process host update requests
    _process_update_requests(stream, command_buffer, requests);

    // build accel and (possibly) compact
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

    // sort resources...
    std::sort(_resources.begin(), _resources.end());
    _resources.erase(
        std::unique(_resources.begin(), _resources.end()),
        _resources.end());
    LUISA_VERBOSE_WITH_LOCATION(
        "Building accel with reference to {} resource(s).",
        _resources.size());
    _descriptor.instancedAccelerationStructures = meshes;
    _descriptor.instanceCount = mesh_handles.size();
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
    return command_buffer;
}

id<MTLCommandBuffer> MetalAccel::update(
    MetalStream *stream, id<MTLCommandBuffer> command_buffer,
    luisa::span<const AccelUpdateRequest> requests) noexcept {
    _process_update_requests(stream, command_buffer, requests);
    auto device = [command_buffer device];
    if (_update_buffer == nullptr || _update_buffer.length < _update_scratch_size) {
        _update_buffer = [device newBufferWithLength:next_pow2(_update_scratch_size)
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

void MetalAccel::_process_update_requests(
    MetalStream *stream, id<MTLCommandBuffer> command_buffer,
    luisa::span<const AccelUpdateRequest> requests) noexcept {
    if (auto n = static_cast<uint>(requests.size())) {
        auto pool = &stream->upload_host_buffer_pool();
        auto request_buffer = pool->allocate(requests.size() * sizeof(AccelUpdateRequest));
        auto ptr = static_cast<std::byte *>([request_buffer.handle() contents]) +
                   request_buffer.offset();
        std::memcpy(ptr, requests.data(), requests.size_bytes());
        auto encoder = [command_buffer computeCommandEncoderWithDispatchType:MTLDispatchTypeConcurrent];
        constexpr auto threads_per_group = 256u;
        auto groups = (n + threads_per_group - 1u) / threads_per_group;
        [encoder setComputePipelineState:_update_shader];
        [encoder setBuffer:_instance_buffer offset:0u atIndex:0u];
        [encoder setBuffer:request_buffer.handle() offset:request_buffer.offset() atIndex:1u];
        [encoder setBytes:&n length:sizeof(uint) atIndex:2u];
        [encoder dispatchThreadgroups:MTLSizeMake(groups, 1u, 1u)
                threadsPerThreadgroup:MTLSizeMake(threads_per_group, 1u, 1u)];
        [encoder endEncoding];
        if (request_buffer.is_pooled()) {
            [command_buffer addCompletedHandler:^(id<MTLCommandBuffer>) {
              pool->recycle(request_buffer);
            }];
        }
    }
}

}
