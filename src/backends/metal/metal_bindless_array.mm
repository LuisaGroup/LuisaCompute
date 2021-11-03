//
// Created by Mike Smith on 2021/7/1.
//

#include <backends/metal/metal_device.h>
#include <backends/metal/metal_bindless_array.h>

namespace luisa::compute::metal {

MetalBindlessArray::MetalBindlessArray(MetalDevice *device, size_t size) noexcept
    : _device{device},
      _event{[device->handle() newEvent]} {

    static constexpr auto src = @"#include <metal_stdlib>\n"
                                 "struct alignas(16) BindlessItem {\n"
                                 "  device const void *buffer;\n"
                                 "  metal::ushort tex2d_sampler;\n"
                                 "  metal::ushort tex3d_sampler;\n"
                                 "  metal::texture2d<float> handle2d;\n"
                                 "  metal::texture3d<float> handle3d;\n"
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
    _buffer = [_device->handle() newBufferWithLength:_encoder.encodedLength * size
                                             options:MTLResourceOptionCPUCacheModeWriteCombined
                                                     | MTLResourceStorageModeShared];
    _device_buffer = [_device->handle() newBufferWithLength:_buffer.length
                                                    options:MTLResourceStorageModePrivate];
}

id<MTLCommandBuffer> MetalBindlessArray::encode_update(
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

void MetalBindlessArray::emplace_buffer(uint32_t index, uint64_t buffer_handle) noexcept {
    auto buffer = _device->buffer(buffer_handle);
    std::scoped_lock lock{_mutex};
    [_encoder setArgumentBuffer:_buffer offset:slot_size * index];
    [_encoder setBuffer:buffer offset:0u atIndex:0u];
    _resources.emplace(MetalBindlessReousrce{buffer});
    _dirty = true;
}

void MetalBindlessArray::emplace_tex2d(uint32_t index, uint64_t texture_handle, Sampler sampler) noexcept {
    auto texture = _device->texture(texture_handle);
    auto sampler = static_cast<uint16_t>(sampler.code());
    std::scoped_lock lock{_mutex};
    [_encoder setArgumentBuffer:_buffer offset:slot_size * index];
    [_encoder setTexture:texture atIndex:3u];
    std::memcpy([_encoder constantDataAtIndex: 1u], &sampler, sizeof(sampler));
    _resources.emplace(MetalBindlessReousrce{texture});
    _dirty = true;
}

void MetalBindlessArray::emplace_tex2d(uint32_t index, uint64_t texture_handle, Sampler sampler) noexcept {
    auto texture = _device->texture(texture_handle);
    auto sampler = static_cast<uint16_t>(sampler.code());
    std::scoped_lock lock{_mutex};
    [_encoder setArgumentBuffer:_buffer offset:slot_size * index];
    [_encoder setTexture:texture atIndex:4u];
    std::memcpy([_encoder constantDataAtIndex: 2u], &sampler, sizeof(sampler));
    _resources.emplace(MetalBindlessReousrce{texture});
    _dirty = true;
}

}
