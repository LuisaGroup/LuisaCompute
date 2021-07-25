//
// Created by Mike Smith on 2021/7/22.
//

#pragma once

#import <vector>

#import <Metal/Metal.h>
#import <rtx/accel.h>

namespace luisa::compute::metal {

class MetalDevice;
class MetalSharedBufferPool;

class MetalAccel {

private:
    MetalDevice *_device;
    id<MTLAccelerationStructure> _handle{nullptr};
    id<MTLBuffer> _instance_buffer{nullptr};
    id<MTLBuffer> _instance_buffer_host{nullptr};
    id<MTLBuffer> _update_buffer{nullptr};
    MTLAccelerationStructureDescriptor *_descriptor{nullptr};
    MTLAccelerationStructureSizes _sizes{};
    __weak id<MTLCommandBuffer> _last_update{nullptr};

public:
    explicit MetalAccel(MetalDevice *device) noexcept
        : _device{device} {}
    [[nodiscard]] auto handle() const noexcept { return _handle; }
    [[nodiscard]] id<MTLCommandBuffer> build(
        id<MTLCommandBuffer> command_buffer,
        AccelBuildHint hint,
        std::span<const uint64_t> mesh_handles,
        std::span<const float4x4> transforms,
        MetalSharedBufferPool *pool) noexcept;
    [[nodiscard]] id<MTLCommandBuffer> update(
        id<MTLCommandBuffer> command_buffer,
        std::span<const float4x4> transforms,
        size_t first) noexcept;
};

}
