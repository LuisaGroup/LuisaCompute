//
// Created by Mike Smith on 2021/7/1.
//

#import <backends/metal/metal_device.h>
#import <backends/metal/metal_bindless_array.h>

namespace luisa::compute::metal {

MetalBindlessArray::MetalBindlessArray(MetalDevice *device, size_t size) noexcept
    : _encoder{device->bindless_array_encoder()},
      _buffer_slots(size, MetalBindlessResource{nullptr}),
      _tex2d_slots(size, MetalBindlessResource{nullptr}),
      _tex3d_slots(size, MetalBindlessResource{nullptr}) {
    auto buffer_size = slot_size * size;
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
        _tracker.release_buffer(detail::resource_to_handle(p.handle));
    }
    [_encoder setArgumentBuffer:_host_buffer offset:slot_size * index];
    [_encoder setBuffer:(__bridge id<MTLBuffer>)(reinterpret_cast<void *>(buffer_handle))
                 offset:offset
                atIndex:0u];
    _tracker.retain_buffer(buffer_handle);
    _dirty_range.mark(index);
}

void MetalBindlessArray::emplace_tex2d(size_t index, uint64_t texture_handle, Sampler sampler) noexcept {
    if (auto p = _tex2d_slots[index]; p.handle != nullptr) {
        _tracker.release_texture(detail::resource_to_handle(p.handle));
    }
    [_encoder setArgumentBuffer:_host_buffer offset:slot_size * index];
    *static_cast<uint *>([_encoder constantDataAtIndex:1u]) = sampler.code();
    [_encoder setTexture:(__bridge id<MTLTexture>)(reinterpret_cast<void *>(texture_handle))
                 atIndex:3u];
    _tracker.retain_texture(texture_handle);
    _dirty_range.mark(index);
}

void MetalBindlessArray::emplace_tex3d(size_t index, uint64_t texture_handle, Sampler sampler) noexcept {
    if (auto p = _tex3d_slots[index]; p.handle != nullptr) {
        _tracker.release_texture(detail::resource_to_handle(p.handle));
    }
    [_encoder setArgumentBuffer:_host_buffer offset:slot_size * index];
    *static_cast<uint *>([_encoder constantDataAtIndex:2u]) = sampler.code();
    [_encoder setTexture:(__bridge id<MTLTexture>)(reinterpret_cast<void *>(texture_handle))
                 atIndex:4u];
    _tracker.retain_texture(texture_handle);
    _dirty_range.mark(index);
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
    if (!_dirty_range.empty()) {
        auto command_encoder = [command_buffer blitCommandEncoder];
        auto pool = &stream->upload_host_buffer_pool();
        auto temp_buffer = pool->allocate(_dirty_range.size() * slot_size);
        std::memcpy(
            static_cast<uint8_t *>([temp_buffer.handle() contents]) + temp_buffer.offset(),
            static_cast<const uint8_t *>([_host_buffer contents]) + slot_size * _dirty_range.offset(),
            _dirty_range.size() * slot_size);
        [command_encoder copyFromBuffer:temp_buffer.handle()
                           sourceOffset:temp_buffer.offset()
                               toBuffer:_device_buffer
                      destinationOffset:_dirty_range.offset() * slot_size
                                   size:_dirty_range.size() * slot_size];
        [command_buffer addCompletedHandler:^(id<MTLCommandBuffer>) {
          pool->recycle(temp_buffer);
        }];
        [command_encoder endEncoding];
    }
    _dirty_range.clear();
    _tracker.commit();
}

}
