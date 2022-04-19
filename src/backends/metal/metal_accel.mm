//
// Created by Mike Smith on 2021/7/22.
//

#import <backends/metal/metal_accel.h>
#import <backends/metal/metal_device.h>

namespace luisa::compute::metal {

MetalAccel::MetalAccel(id<MTLComputePipelineState> update_shader, AccelUsageHint hint) noexcept
    : _update_shader{update_shader} {
    _descriptor = [MTLInstanceAccelerationStructureDescriptor descriptor];
    switch (hint) {
        case AccelUsageHint::FAST_TRACE: _descriptor.usage = MTLAccelerationStructureUsageNone; break;
        case AccelUsageHint::FAST_UPDATE: _descriptor.usage = MTLAccelerationStructureUsageRefit; break;
        case AccelUsageHint::FAST_BUILD: _descriptor.usage = MTLAccelerationStructureUsagePreferFastBuild; break;
    }
}

id<MTLCommandBuffer> MetalAccel::build(MetalStream *stream, id<MTLCommandBuffer> command_buffer,
                                       uint instance_count, AccelBuildRequest request,
                                       luisa::span<const AccelBuildCommand::Modification> mods) noexcept {

    // allocate the instance buffer if necessary
    auto device = [command_buffer device];
    auto instance_buffer_size = instance_count * sizeof(MTLAccelerationStructureInstanceDescriptor);
    if (_instance_buffer == nullptr || _instance_buffer.length < instance_buffer_size) {
        auto instance_buffer = [device newBufferWithLength:next_pow2(instance_buffer_size)
                                                   options:MTLResourceStorageModePrivate];
        if (_instance_buffer != nullptr) {// copy old instance buffer to the new one
            auto blit_encoder = [command_buffer blitCommandEncoder];
            [blit_encoder copyFromBuffer:_instance_buffer
                            sourceOffset:0u
                                toBuffer:instance_buffer
                       destinationOffset:0u
                                    size:_instance_buffer.length];
            [blit_encoder endEncoding];
        }
        _instance_buffer = instance_buffer;
    }

    // fire the update shader, so we can process the build request on CPU simultaneously
    if (auto n = static_cast<uint>(mods.size())) {
        auto pool = &stream->upload_host_buffer_pool();
        auto request_buffer = pool->allocate(mods.size_bytes());
        auto ptr = static_cast<std::byte *>([request_buffer.handle() contents]);
        std::memcpy(ptr + request_buffer.offset(), mods.data(), mods.size_bytes());
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

    // check if requires build
    auto requires_build = request == AccelBuildRequest::FORCE_BUILD ||
                          _meshes.size() != instance_count ||
                          _handle == nullptr;

    // update the mesh list
    _meshes.resize(instance_count);
    for (auto &&mod : mods) {
        if (mod.flags & AccelBuildCommand::Modification::flag_mesh) {
            _meshes[mod.index] = reinterpret_cast<const MetalMesh *>(mod.mesh);
            requires_build = true;
        }
    }

    // check if any mesh resource has changed
    if (!requires_build) {
        for (auto i = 0u; i < instance_count; i++) {
            if ((requires_build |= _meshes[i]->handle() != _mesh_handles[i])) {
                break;
            }
        }
    }

    // now we truly know if we need to build
    if (requires_build) {
        _mesh_handles = [NSMutableArray<id<MTLAccelerationStructure>> arrayWithCapacity:instance_count];
        _resources.clear();
        for (auto mesh : _meshes) {
            [_mesh_handles addObject:mesh->handle()];
            _resources.emplace(Resource{mesh->handle()});
            _resources.emplace(Resource{mesh->vertex_buffer()});
            _resources.emplace(Resource{mesh->triangle_buffer()});
        }
        _resources.emplace(Resource{_instance_buffer});
        _descriptor.instancedAccelerationStructures = _mesh_handles;
        _descriptor.instanceCount = instance_count;
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
        for (auto resource : _resources) {
            [command_encoder useResource:resource.handle
                                   usage:MTLResourceUsageRead];
        }
        [command_encoder endEncoding];
        // TODO: compaction?
        return command_buffer;
    }

    // update is adequate
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

}
