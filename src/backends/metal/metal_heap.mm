//
// Created by Mike Smith on 2021/7/1.
//

#include <backends/metal/metal_device.h>
#include <backends/metal/metal_heap.h>

namespace luisa::compute::metal {

MTLHeapDescriptor *MetalHeap::_heap_descriptor(size_t size) noexcept {
    auto desc = [[MTLHeapDescriptor alloc] init];
    desc.size = size;
    desc.hazardTrackingMode = MTLHazardTrackingModeUntracked;
    desc.type = MTLHeapTypeAutomatic;
    desc.storageMode = MTLStorageModePrivate;
    return desc;
}

MetalHeap::MetalHeap(MetalDevice *device, size_t size) noexcept
    : _device{device},
      _handle{[device->handle() newHeapWithDescriptor:_heap_descriptor(size)]},
      _event{[device->handle() newEvent]} {

    static constexpr auto src = @"#include <metal_stdlib>\n"
                                 "struct alignas(16) HeapItem {\n"
                                 "  metal::texture2d<float> handle2d;\n"
                                 "  metal::texture3d<float> handle3d;\n"
                                 "  metal::sampler sampler;\n"
                                 "  device const void *buffer;\n"
                                 "};\n"
                                 "[[kernel]] void k(device const HeapItem *heap) {}\n";
    auto library = [_device->handle() newLibraryWithSource:src options:nullptr error:nullptr];
    auto function = [library newFunctionWithName:@"k"];
    _encoder = [function newArgumentEncoderWithBufferIndex:0];
    if (auto enc_size = _encoder.encodedLength; enc_size != slot_size) {
        LUISA_ERROR_WITH_LOCATION(
            "Invalid heap texture encoded size: {} (expected {}).",
            enc_size, slot_size);
    }
    _buffer = [_device->handle() newBufferWithLength:_encoder.encodedLength * Heap::slot_count
                                             options:MTLResourceOptionCPUCacheModeWriteCombined
                                                     | MTLResourceStorageModeShared];
    _device_buffer = [_device->handle() newBufferWithLength:_buffer.length
                                                    options:MTLResourceStorageModePrivate];
    for (auto i = 0u; i < 16u; i++) {
        auto sampler = Sampler::decode(i);
        auto desc = [[MTLSamplerDescriptor alloc] init];
        desc.supportArgumentBuffers = YES;
        desc.normalizedCoordinates = YES;
        switch (sampler.address()) {
            case Sampler::Address::EDGE:
                desc.sAddressMode = MTLSamplerAddressModeClampToEdge;
                desc.tAddressMode = MTLSamplerAddressModeClampToEdge;
                desc.rAddressMode = MTLSamplerAddressModeClampToEdge;
                break;
            case Sampler::Address::REPEAT:
                desc.sAddressMode = MTLSamplerAddressModeRepeat;
                desc.tAddressMode = MTLSamplerAddressModeRepeat;
                desc.rAddressMode = MTLSamplerAddressModeRepeat;
                break;
            case Sampler::Address::MIRROR:
                desc.sAddressMode = MTLSamplerAddressModeMirrorRepeat;
                desc.tAddressMode = MTLSamplerAddressModeMirrorRepeat;
                desc.rAddressMode = MTLSamplerAddressModeMirrorRepeat;
                break;
            case Sampler::Address::ZERO:
                desc.sAddressMode = MTLSamplerAddressModeClampToZero;
                desc.tAddressMode = MTLSamplerAddressModeClampToZero;
                desc.rAddressMode = MTLSamplerAddressModeClampToZero;
                break;
        }
        switch (sampler.filter()) {
            case Sampler::Filter::POINT:
                desc.mipFilter = MTLSamplerMipFilterNearest;
                desc.minFilter = MTLSamplerMinMagFilterNearest;
                desc.magFilter = MTLSamplerMinMagFilterNearest;
                break;
            case Sampler::Filter::BILINEAR:
                desc.mipFilter = MTLSamplerMipFilterNearest;
                desc.minFilter = MTLSamplerMinMagFilterLinear;
                desc.magFilter = MTLSamplerMinMagFilterLinear;
                break;
            case Sampler::Filter::TRILINEAR:
                desc.mipFilter = MTLSamplerMipFilterLinear;
                desc.minFilter = MTLSamplerMinMagFilterLinear;
                desc.magFilter = MTLSamplerMinMagFilterLinear;
                break;
            case Sampler::Filter::ANISOTROPIC:
                desc.mipFilter = MTLSamplerMipFilterLinear;
                desc.minFilter = MTLSamplerMinMagFilterLinear;
                desc.magFilter = MTLSamplerMinMagFilterLinear;
                desc.maxAnisotropy = 16u;
                break;
        }
        _samplers[i] = [device->handle() newSamplerStateWithDescriptor:desc];
    }
}

id<MTLTexture> MetalHeap::allocate_texture(MTLTextureDescriptor *desc) noexcept {
    return [_handle newTextureWithDescriptor:desc];
}

id<MTLCommandBuffer> MetalHeap::encode_update(
    MetalStream *stream,
    id<MTLCommandBuffer> cmd_buf) const noexcept {

    std::scoped_lock lock{_mutex};
    if (_dirty) {
        if (auto last = _last_update;
            last != nullptr) {
            [last waitUntilCompleted];
        }
        _last_update = cmd_buf;
        _dirty = false;
        auto blit_encoder = [cmd_buf blitCommandEncoder];
        [blit_encoder copyFromBuffer:_buffer
                        sourceOffset:0u
                            toBuffer:_device_buffer
                   destinationOffset:0u
                                size:_buffer.length];
        [blit_encoder endEncoding];
        [cmd_buf encodeSignalEvent:_event
                             value:++_event_value];
        // create a new command buffer to avoid dead locks
        stream->dispatch(cmd_buf);
        cmd_buf = stream->command_buffer();
    }
    [cmd_buf encodeWaitForEvent:_event
                          value:_event_value];
    return cmd_buf;
}

void MetalHeap::emplace_buffer(uint32_t index, uint64_t buffer_handle) noexcept {
    auto buffer = _device->buffer(buffer_handle);
    std::scoped_lock lock{_mutex};
    [_encoder setArgumentBuffer:_buffer offset:slot_size * index];
    [_encoder setBuffer:buffer offset:0u atIndex:3u];
    _active_buffers.emplace(buffer_handle);
    _dirty = true;
}

void MetalHeap::emplace_texture(uint32_t index, uint64_t texture_handle, Sampler sampler) noexcept {
    auto sampler_state = _samplers[sampler.code()];
    auto texture = _device->texture(texture_handle);
    std::scoped_lock lock{_mutex};
    [_encoder setArgumentBuffer:_buffer offset:slot_size * index];
    [_encoder setTexture:texture atIndex:texture.textureType == MTLTextureType2D ? 0u : 1u];
    [_encoder setSamplerState:sampler_state atIndex:2u];
    _active_textures.emplace(texture_handle);
    _dirty = true;
}

void MetalHeap::destroy_buffer(uint64_t b) noexcept {
    std::scoped_lock lock{_mutex};
    _active_buffers.erase(b);
}

void MetalHeap::destroy_texture(uint64_t t) noexcept {
    std::scoped_lock lock{_mutex};
    _active_textures.erase(t);
}

id<MTLBuffer> MetalHeap::allocate_buffer(size_t size_bytes) noexcept {
    return [_handle newBufferWithLength:size_bytes
                                options:MTLResourceStorageModePrivate
                                        | MTLResourceHazardTrackingModeDefault];
}

}
