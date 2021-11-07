//
// Created by Mike Smith on 2021/7/1.
//

#include <backends/metal/metal_device.h>
#include <backends/metal/metal_bindless_array.h>

namespace luisa::compute::metal {

MetalBindlessArray::MetalBindlessArray(MetalDevice *device, size_t size) noexcept
    : _encoder{device->bindless_array_encoder()},
      _buffer_slots(size, MetalBindlessResource{nullptr}),
      _tex2d_slots(size, MetalBindlessResource{nullptr}),
      _tex3d_slots(size, MetalBindlessResource{nullptr}) {
    _buffer = [device->handle() newBufferWithLength:_encoder.encodedLength * size
                                             options:MTLResourceCPUCacheModeDefaultCache |
                                                     MTLResourceStorageModeShared |
                                                     MTLResourceHazardTrackingModeUntracked];
    _device_buffer = [device->handle() newBufferWithLength:_buffer.length
                                                    options:MTLResourceStorageModePrivate];
}

void MetalBindlessArray::emplace_buffer(size_t index, uint64_t buffer_handle, size_t offset) noexcept {
    auto buffer = (__bridge id<MTLBuffer>)(reinterpret_cast<void *>(buffer_handle));
    if (auto &&p = _buffer_slots[index]; p.handle != nullptr) { _resources.erase(p); }
    [_encoder setArgumentBuffer:_buffer offset:slot_size * index];
    [_encoder setBuffer:buffer offset:offset atIndex:0u];
    _retain(buffer);
}

void MetalBindlessArray::emplace_tex2d(size_t index, uint64_t texture_handle, Sampler sampler) noexcept {
    auto texture = (__bridge id<MTLTexture>)(reinterpret_cast<void *>(texture_handle));
    auto sampler_code = static_cast<uint16_t>(sampler.code());
    if (auto &&p = _tex2d_slots[index]; p.handle != nullptr) { _resources.erase(p); }
    [_encoder setArgumentBuffer:_buffer offset:slot_size * index];
    [_encoder setTexture:texture atIndex:3u];
    std::memcpy([_encoder constantDataAtIndex:1u], &sampler_code, sizeof(sampler));
    _retain(texture);
}

void MetalBindlessArray::emplace_tex3d(size_t index, uint64_t texture_handle, Sampler sampler) noexcept {
    auto texture = (__bridge id<MTLTexture>)(reinterpret_cast<void *>(texture_handle));
    auto sampler_code = static_cast<uint16_t>(sampler.code());
    if (auto &&p = _tex3d_slots[index]; p.handle != nullptr) { _resources.erase(p); }
    [_encoder setArgumentBuffer:_buffer offset:slot_size * index];
    [_encoder setTexture:texture atIndex:4u];
    std::memcpy([_encoder constantDataAtIndex:2u], &sampler_code, sizeof(sampler));
    _retain(texture);
}

void MetalBindlessArray::remove_buffer(size_t index) noexcept {
    if (auto &&p = _buffer_slots[index]; p.handle != nullptr) {
        _release(p.handle);
        p.handle = nullptr;
    }
}

void MetalBindlessArray::remove_tex2d(size_t index) noexcept {
    if (auto &&p = _tex2d_slots[index]; p.handle != nullptr) {
        _release(p.handle);
        p.handle = nullptr;
    }
}

void MetalBindlessArray::remove_tex3d(size_t index) noexcept {
    if (auto &&p = _tex3d_slots[index]; p.handle != nullptr) {
        _release(p.handle);
        p.handle = nullptr;
    }
}

void MetalBindlessArray::_retain(id<MTLResource> r) noexcept {
    if (auto iter = _resources.try_emplace(MetalBindlessResource{r}, 1u);
        !iter.second) {
        iter.first->second++;
    }
}

void MetalBindlessArray::_release(id<MTLResource> r) noexcept {
    MetalBindlessResource resource{r};
    if (auto iter = _resources.find(resource);
        iter != _resources.end()) [[likely]] {
        if (--iter->second == 0u) {
            _resources.erase(iter);
        }
    } else [[unlikely]] {
        LUISA_WARNING_WITH_LOCATION(
            "Removing non-existent resource in bindless array.");
    }
}

bool MetalBindlessArray::has_buffer(uint64_t handle) const noexcept {
    auto buffer = (__bridge id<MTLBuffer>)(reinterpret_cast<void *>(handle));
    return _resources.contains(MetalBindlessResource{buffer});
}

bool MetalBindlessArray::has_texture(uint64_t handle) const noexcept {
    auto texture = (__bridge id<MTLTexture>)(reinterpret_cast<void *>(handle));
    return _resources.contains(MetalBindlessResource{texture});
}

}
