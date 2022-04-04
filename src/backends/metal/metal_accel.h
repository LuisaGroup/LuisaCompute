//
// Created by Mike Smith on 2021/7/22.
//

#pragma once

#import <vector>
#import <Metal/Metal.h>

#import <core/stl.h>
#import <core/dirty_range.h>
#import <rtx/accel.h>
#import <backends/metal/metal_mesh.h>

namespace luisa::compute::metal {

class MetalDevice;
class MetalStream;

class MetalAccel {

private:
    id<MTLComputePipelineState> _update_shader;
    id<MTLAccelerationStructure> _handle{nullptr};
    id<MTLBuffer> _instance_buffer{nullptr};
    id<MTLBuffer> _update_buffer{nullptr};
    MTLInstanceAccelerationStructureDescriptor *_descriptor{nullptr};
    size_t _update_scratch_size{};
    luisa::vector<id<MTLResource>> _resources;
    luisa::unordered_set<uint64_t> _resource_handles;

private:
    void _process_update_requests(
        MetalStream *stream, id<MTLCommandBuffer> command_buffer,
        luisa::span<const AccelUpdateRequest> requests) noexcept;

public:
    MetalAccel(id<MTLComputePipelineState> update_shader, AccelBuildHint hint) noexcept;
    [[nodiscard]] auto handle() const noexcept { return _handle; }
    [[nodiscard]] id<MTLCommandBuffer> build(
        MetalStream *stream, id<MTLCommandBuffer> command_buffer,
        luisa::span<const uint64_t> meshes,
        luisa::span<const AccelUpdateRequest> requests) noexcept;
    [[nodiscard]] id<MTLCommandBuffer> update(
        MetalStream *stream, id<MTLCommandBuffer> command_buffer,
        luisa::span<const AccelUpdateRequest> requests) noexcept;
    [[nodiscard]] auto instance_buffer() const noexcept { return _instance_buffer; }
    [[nodiscard]] auto descriptor() const noexcept { return _descriptor; }
    [[nodiscard]] auto resources() noexcept { return luisa::span{_resources}; }
};

}// namespace luisa::compute::metal
