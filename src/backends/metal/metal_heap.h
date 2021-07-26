//
// Created by Mike Smith on 2021/7/1.
//

#pragma once

#import <Metal/Metal.h>

#import <core/spin_mutex.h>
#import <runtime/heap.h>

namespace luisa::compute::metal {

class MetalDevice;
class MetalStream;

class MetalTextureHeap {

private:
    MetalDevice *_device;
    id<MTLHeap> _handle;
    id<MTLBuffer> _buffer{nullptr};
    id<MTLBuffer> _device_buffer{nullptr};
    id<MTLArgumentEncoder> _encoder{nullptr};
    std::array<id<MTLSamplerState>, 16u> _samplers{};
    id<MTLEvent> _event{nullptr};
    mutable uint64_t _event_value{0u};
    mutable __weak id<MTLCommandBuffer> _last_update{nullptr};
    mutable spin_mutex _mutex;
    mutable bool _dirty{true};
    static constexpr auto slot_size = 32u;

private:
    [[nodiscard]] static MTLHeapDescriptor *_heap_descriptor(size_t size) noexcept;

public:
    MetalTextureHeap(MetalDevice *device, size_t size) noexcept;
    [[nodiscard]] id<MTLTexture> allocate_texture(MTLTextureDescriptor *desc, uint32_t index, TextureSampler sampler) noexcept;
    [[nodiscard]] id<MTLBuffer> allocate_buffer(size_t size_bytes, uint32_t index_in_heap) noexcept;
    [[nodiscard]] auto handle() const noexcept { return _handle; }
    [[nodiscard]] auto desc_buffer() const noexcept { return _device_buffer; }
    [[nodiscard]] id<MTLCommandBuffer> encode_update(MetalStream *stream, id<MTLCommandBuffer> cmd_buf) const noexcept;
};

}
