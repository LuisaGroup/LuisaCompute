//
// Created by Mike Smith on 2021/7/22.
//

#pragma once

#import <vector>

#import <Metal/Metal.h>
#import <rtx/accel.h>

namespace luisa::compute::metal {

class MetalDevice;
class MetalStream;
class MetalSharedBufferPool;

class MetalAccel {

private:
    MetalDevice *_device;
    id<MTLAccelerationStructure> _handle{nullptr};
    id<MTLBuffer> _instance_buffer{nullptr};
    id<MTLBuffer> _instance_buffer_host{nullptr};
    id<MTLBuffer> _update_buffer{nullptr};
    MTLInstanceAccelerationStructureDescriptor *_descriptor{nullptr};
    size_t _update_scratch_size{};
    __weak id<MTLCommandBuffer> _last_update{nullptr};
    std::vector<id<MTLResource>> _resources;
    std::vector<id<MTLHeap>> _heaps;

public:
    explicit MetalAccel(MetalDevice *device) noexcept : _device{device} {}
    [[nodiscard]] auto handle() const noexcept { return _handle; }
    [[nodiscard]] id<MTLCommandBuffer> build(
        MetalStream *stream,
        id<MTLCommandBuffer> command_buffer,
        AccelBuildHint hint,
        std::span<const uint64_t> mesh_handles,
        std::span<const float4x4> transforms,
        MetalSharedBufferPool *pool) noexcept;
    [[nodiscard]] id<MTLCommandBuffer> update(
        MetalStream *stream,
        id<MTLCommandBuffer> command_buffer,
        std::span<const float4x4> transforms,
        size_t first) noexcept;
    [[nodiscard]] auto instance_buffer() const noexcept { return _instance_buffer; }
    [[nodiscard]] auto descriptor() const noexcept { return _descriptor; }
    [[nodiscard]] auto resources() noexcept { return std::span{_resources}; }
    [[nodiscard]] auto heaps() noexcept { return std::span{_heaps}; }
};

}// namespace luisa::compute::metal
