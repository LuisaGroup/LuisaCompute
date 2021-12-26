//
// Created by Mike Smith on 2021/7/1.
//

#import <backends/metal/metal_device.h>
#import <backends/metal/metal_bindless_array.h>

namespace luisa::compute::metal {

MetalBindlessArray::MetalBindlessArray(MetalDevice *device, size_t size) noexcept
    : _buffer_encoder{device->bindless_array_buffer_encoder()},
      _tex2d_encoder{device->bindless_array_tex2d_encoder()},
      _tex3d_encoder{device->bindless_array_tex3d_encoder()},
      _buffer_slots(size, MetalBindlessResource{nullptr}),
      _tex2d_slots(size, MetalBindlessResource{nullptr}),
      _tex3d_slots(size, MetalBindlessResource{nullptr}) {
    auto align = [](auto size) noexcept {
        static constexpr auto alignment = 16u;
        return (size + alignment - 1u) / alignment * alignment;
    };
    _buffer_encoding_offset = 0u;
    _tex2d_encoding_offset = align(_buffer_encoding_offset + buffer_slot_size * size);
    _tex3d_encoding_offset = align(_tex2d_encoding_offset + tex2d_slot_size * size);
    _sampler_encoding_offset = align(_tex3d_encoding_offset + tex3d_slot_size * size);
    auto buffer_size = align(_sampler_encoding_offset + sampler_slot_size * size);
    _host_buffer = [device->handle() newBufferWithLength:buffer_size
                                                 options:MTLResourceCPUCacheModeDefaultCache |
                                                         MTLResourceStorageModeShared |
                                                         MTLResourceHazardTrackingModeUntracked];
    _device_buffer = [device->handle() newBufferWithLength:buffer_size
                                                   options:MTLResourceStorageModePrivate];
}

namespace detail {
[[nodiscard]] inline static auto resource_to_handle(id<MTLResource> resource) noexcept {
    return reinterpret_cast<uint64_t>((__bridge void *)(resource));
}
}

void MetalBindlessArray::emplace_buffer(size_t index, uint64_t buffer_handle, size_t offset) noexcept {
    if (auto p = _buffer_slots[index]; p.handle != nullptr) {
        _tracker.release_buffer(
            detail::resource_to_handle(p.handle));
    }
    [_buffer_encoder setArgumentBuffer:_host_buffer
                                offset:_buffer_encoding_offset + buffer_slot_size * index];
    [_buffer_encoder setBuffer:(__bridge id<MTLBuffer>)(reinterpret_cast<void *>(buffer_handle))
                        offset:offset
                       atIndex:0u];
    _tracker.retain_buffer(buffer_handle);
    _buffer_dirty_range.mark(index);
}

void MetalBindlessArray::emplace_tex2d(size_t index, uint64_t texture_handle, Sampler sampler) noexcept {
    if (auto p = _tex2d_slots[index]; p.handle != nullptr) {
        _tracker.release_texture(
            detail::resource_to_handle(p.handle));
    }
    [_tex2d_encoder setArgumentBuffer:_host_buffer
                               offset:_tex2d_encoding_offset + tex2d_slot_size * index];
    [_tex2d_encoder setTexture:(__bridge id<MTLTexture>)(reinterpret_cast<void *>(texture_handle))
                       atIndex:0u];
    auto ptr = static_cast<uint8_t *>([_host_buffer contents]) +
               _sampler_encoding_offset + sampler_slot_size * index;
    *ptr = static_cast<uint8_t>((*ptr & 0xf0u) | sampler.code());
    _tracker.retain_texture(texture_handle);
    _tex2d_dirty_range.mark(index);
    _sampler_dirty_range.mark(index);
}

void MetalBindlessArray::emplace_tex3d(size_t index, uint64_t texture_handle, Sampler sampler) noexcept {
    if (auto p = _tex3d_slots[index]; p.handle != nullptr) {
        _tracker.release_texture(
            detail::resource_to_handle(p.handle));
    }
    [_tex3d_encoder setArgumentBuffer:_host_buffer
                               offset:_tex3d_encoding_offset + tex3d_slot_size * index];
    [_tex3d_encoder setTexture:(__bridge id<MTLTexture>)(reinterpret_cast<void *>(texture_handle))
                       atIndex:0u];
    auto ptr = static_cast<uint8_t *>([_host_buffer contents]) +
               _sampler_encoding_offset + sampler_slot_size * index;
    *ptr = static_cast<uint8_t>((*ptr & 0x0fu) | (sampler.code() << 4u));
    _tracker.retain_texture(texture_handle);
    _tex3d_dirty_range.mark(index);
    _sampler_dirty_range.mark(index);
}

void MetalBindlessArray::remove_buffer(size_t index) noexcept {
    if (auto &&p = _buffer_slots[index]; p.handle != nullptr) {
        _tracker.release_buffer(
            detail::resource_to_handle(p.handle));
        p.handle = nullptr;
    }
}

void MetalBindlessArray::remove_tex2d(size_t index) noexcept {
    if (auto &&p = _tex2d_slots[index]; p.handle != nullptr) {
        _tracker.release_texture(
            detail::resource_to_handle(p.handle));
        p.handle = nullptr;
    }
}

void MetalBindlessArray::remove_tex3d(size_t index) noexcept {
    if (auto &&p = _tex3d_slots[index]; p.handle != nullptr) {
        _tracker.release_texture(
            detail::resource_to_handle(p.handle));
        p.handle = nullptr;
    }
}

void MetalBindlessArray::update(MetalStream *stream, id<MTLCommandBuffer> command_buffer) noexcept {
    auto command_encoder = [command_buffer blitCommandEncoder];
    auto update_slots = [command_encoder, stream, command_buffer, this](DirtyRange range, size_t slot_size, size_t encoding_offset) mutable {
        if (!range.empty()) {
            auto pool = &stream->upload_ring_buffer();
            auto temp_buffer = pool->allocate(range.size() * slot_size);
            std::memcpy(
                static_cast<uint8_t *>([temp_buffer.handle() contents]) + temp_buffer.offset(),
                static_cast<uint8_t *>([_host_buffer contents]) + encoding_offset + range.offset() * slot_size,
                range.size() * slot_size);
            [command_encoder copyFromBuffer:temp_buffer.handle()
                               sourceOffset:temp_buffer.offset()
                                   toBuffer:_device_buffer
                          destinationOffset:encoding_offset + range.offset() * slot_size
                                       size:range.size() * slot_size];
            [command_buffer addCompletedHandler:^(id<MTLCommandBuffer>) {
              pool->recycle(temp_buffer);
            }];
        }
    };
    [command_encoder endEncoding];
    update_slots(_buffer_dirty_range, buffer_slot_size, _buffer_encoding_offset);
    update_slots(_tex2d_dirty_range, tex2d_slot_size, _tex2d_encoding_offset);
    update_slots(_tex3d_dirty_range, tex3d_slot_size, _tex3d_encoding_offset);
    update_slots(_sampler_dirty_range, sampler_slot_size, _sampler_encoding_offset);
    _buffer_dirty_range.clear();
    _tex2d_dirty_range.clear();
    _tex3d_dirty_range.clear();
    _sampler_dirty_range.clear();
    _tracker.commit();
}

}
