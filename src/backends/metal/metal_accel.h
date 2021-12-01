//
// Created by Mike Smith on 2021/7/22.
//

#pragma once

#import <vector>
#import <Metal/Metal.h>

#import <core/allocator.h>
#import <core/dirty_range.h>
#import <rtx/accel.h>
#import <backends/metal/metal_mesh.h>

namespace luisa::compute::metal {

class MetalDevice;
class MetalStream;
class MetalSharedBufferPool;

class MetalAccel {

private:
    id<MTLAccelerationStructure> _handle{nullptr};
    id<MTLBuffer> _instance_buffer{nullptr};
    id<MTLBuffer> _instance_buffer_host{nullptr};
    id<MTLBuffer> _update_buffer{nullptr};
    __weak id<MTLCommandBuffer> _instance_buffer_copy_command{nullptr};
    MTLInstanceAccelerationStructureDescriptor *_descriptor{nullptr};
    size_t _update_scratch_size{};
    luisa::vector<MetalMesh *> _instance_meshes;
    luisa::vector<float4x4> _instance_transforms;
    luisa::vector<id<MTLResource>> _resources;// sorted
    DirtyRange _dirty_range;

public:
    explicit MetalAccel(AccelBuildHint hint) noexcept;
    [[nodiscard]] auto handle() const noexcept { return _handle; }
    [[nodiscard]] id<MTLCommandBuffer> build(
        MetalStream *stream,
        id<MTLCommandBuffer> command_buffer,
        MetalSharedBufferPool *pool) noexcept;
    [[nodiscard]] id<MTLCommandBuffer> update(
        MetalStream *stream,
        id<MTLCommandBuffer> command_buffer) noexcept;
    [[nodiscard]] auto instance_buffer_host() const noexcept { return _instance_buffer_host; }
    [[nodiscard]] auto instance_buffer() const noexcept { return _instance_buffer; }
    [[nodiscard]] auto descriptor() const noexcept { return _descriptor; }
    [[nodiscard]] auto resources() noexcept { return std::span{_resources}; }

    void add_instance(MetalMesh *mesh, float4x4 transform) noexcept;
    void set_transform(size_t index, float4x4 transform) noexcept;
    [[nodiscard]] bool uses_resource(id<MTLResource> resource) const noexcept;

    [[nodiscard]] auto dirty_range() const noexcept { return _dirty_range; }
    void clear_dirty_range() noexcept { _dirty_range.clear(); }
};

}// namespace luisa::compute::metal
