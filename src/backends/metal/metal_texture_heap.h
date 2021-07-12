//
// Created by Mike Smith on 2021/7/1.
//

#pragma once

#import <Metal/Metal.h>

#import <core/spin_mutex.h>
#import <runtime/texture_heap.h>

namespace luisa::compute::metal {

class MetalDevice;

class MetalTextureHeap {

private:
    MetalDevice *_device;
    id<MTLHeap> _handle;
    id<MTLBuffer> _buffer{nullptr};
    id<MTLBuffer> _device_buffer{nullptr};
    id<MTLArgumentEncoder> _encoder{nullptr};
    mutable spin_mutex _mutex;
    mutable bool _dirty{false};

private:
    [[nodiscard]] static MTLHeapDescriptor *_heap_descriptor(size_t size) noexcept;

public:
    MetalTextureHeap(MetalDevice *device, size_t size) noexcept;
    [[nodiscard]] id<MTLTexture> allocate_texture(MTLTextureDescriptor *desc, uint32_t index, TextureSampler sampler) noexcept;
    [[nodiscard]] auto handle() const noexcept { return _handle; }
    [[nodiscard]] auto desc_buffer() const noexcept { return _device_buffer; }
    void encode_update(id<MTLCommandBuffer> cmd_buf) const noexcept;
};

}
