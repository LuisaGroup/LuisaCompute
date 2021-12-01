//
// Created by Mike Smith on 2021/7/22.
//

#pragma once

#import <Metal/Metal.h>
#import <rtx/mesh.h>

namespace luisa::compute::metal {

class MetalStream;
class MetalSharedBufferPool;

class MetalMesh {

private:
    id<MTLAccelerationStructure> _handle{nullptr};
    id<MTLBuffer> _update_buffer{nullptr};
    MTLPrimitiveAccelerationStructureDescriptor *_descriptor{nullptr};
    size_t _update_buffer_size{0u};

public:
    MetalMesh(id<MTLBuffer> v_buffer, size_t v_offset, size_t v_stride,
              id<MTLBuffer> t_buffer, size_t t_offset, size_t t_count, AccelBuildHint hint) noexcept;
    [[nodiscard]] auto handle() const noexcept { return _handle; }
    [[nodiscard]] id<MTLCommandBuffer> build(
        MetalStream *stream,
        id<MTLCommandBuffer> command_buffer,
        MetalSharedBufferPool *pool) noexcept;
    [[nodiscard]] id<MTLCommandBuffer> update(
        MetalStream *stream,
        id<MTLCommandBuffer> command_buffer);
    [[nodiscard]] auto descriptor() const noexcept { return _descriptor; }
    [[nodiscard]] id<MTLBuffer> vertex_buffer() const noexcept;
    [[nodiscard]] id<MTLBuffer> triangle_buffer() const noexcept;
};

}// namespace luisa::compute::metal
