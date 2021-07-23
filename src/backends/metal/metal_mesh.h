//
// Created by Mike Smith on 2021/7/22.
//

#pragma once

#import <Metal/Metal.h>
#import <rtx/mesh.h>

namespace luisa::compute::metal {

class MetalMesh {

private:
    id<MTLDevice> _device{nullptr};
    id<MTLAccelerationStructure> _handle{nullptr};
    id<MTLBuffer> _update_buffer{nullptr};
    MTLPrimitiveAccelerationStructureDescriptor *_descriptor{nullptr};
    MTLAccelerationStructureSizes _sizes{};

public:
    explicit MetalMesh(id<MTLDevice> device) noexcept
        : _device{device} {}
    [[nodiscard]] auto handle() const noexcept { return _handle; }
    id<MTLCommandBuffer> build(
        id<MTLCommandBuffer> command_buffer,
        AccelBuildHint hint,
        id<MTLBuffer> v_buffer, size_t v_offset, size_t v_stride,
        id<MTLBuffer> t_buffer, size_t t_offset, size_t t_count) noexcept;
    id<MTLCommandBuffer> update(id<MTLCommandBuffer> command_buffer);
};

}// namespace luisa::compute::metal
