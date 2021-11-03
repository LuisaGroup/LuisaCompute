//
// Created by Mike Smith on 2021/7/1.
//

#include <backends/metal/metal_device.h>
#include <backends/metal/metal_bindless_array.h>

namespace luisa::compute::metal {

MetalBindlessArray::MetalBindlessArray(MetalDevice *device, size_t size) noexcept
    : _device{device},
      _event{[device->handle() newEvent]},
      _buffer_slots(size, MetalBindlessResource{nullptr}),
      _tex2d_slots(size, MetalBindlessResource{nullptr}),
      _tex3d_slots(size, MetalBindlessResource{nullptr}) {

    static constexpr auto src = @"#include <metal_stdlib>\n"
                                 "struct alignas(16) BindlessItem {\n"
                                 "  device const void *buffer;\n"
                                 "  metal::ushort sampler2d;\n"
                                 "  metal::ushort sampler3d;\n"
                                 "  metal::texture2d<float> handle2d;\n"
                                 "  metal::texture3d<float> handle3d;\n"
                                 "};\n"
                                 "[[kernel]] void k(device const BindlessItem *array) {}\n";
    auto library = [_device->handle() newLibraryWithSource:src options:nullptr error:nullptr];
    auto function = [library newFunctionWithName:@"k"];
    _encoder = [function newArgumentEncoderWithBufferIndex:0];
    if (auto enc_size = _encoder.encodedLength; enc_size != slot_size) {
        LUISA_ERROR_WITH_LOCATION(
            "Invalid bindless array encoded size: {} (expected {}).",
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

void MetalBindlessArray::emplace_buffer(size_t index, uint64_t buffer_handle) noexcept {
    auto buffer = _device->buffer(buffer_handle);
    std::scoped_lock lock{_mutex};
    if (auto &&p = _buffer_slots[index]; p.handle != nullptr) { _resources.erase(p); }
    [_encoder setArgumentBuffer:_buffer offset:slot_size * index];
    [_encoder setBuffer:buffer offset:0u atIndex:0u];
    _retain(buffer);
    _dirty = true;
}

void MetalBindlessArray::emplace_tex2d(size_t index, uint64_t texture_handle, Sampler sampler) noexcept {
    auto texture = _device->texture(texture_handle);
    auto sampler_code = static_cast<uint16_t>(sampler.code());
    std::scoped_lock lock{_mutex};
    if (auto &&p = _tex2d_slots[index]; p.handle != nullptr) { _resources.erase(p); }
    [_encoder setArgumentBuffer:_buffer offset:slot_size * index];
    [_encoder setTexture:texture atIndex:3u];
    std::memcpy([_encoder constantDataAtIndex: 1u], &sampler_code, sizeof(sampler));
    _retain(texture);
    _dirty = true;
}

void MetalBindlessArray::emplace_tex3d(size_t index, uint64_t texture_handle, Sampler sampler) noexcept {
    auto texture = _device->texture(texture_handle);
    auto sampler_code = static_cast<uint16_t>(sampler.code());
    std::scoped_lock lock{_mutex};
    if (auto &&p = _tex3d_slots[index]; p.handle != nullptr) { _resources.erase(p); }
    [_encoder setArgumentBuffer:_buffer offset:slot_size * index];
    [_encoder setTexture:texture atIndex:4u];
    std::memcpy([_encoder constantDataAtIndex: 2u], &sampler_code, sizeof(sampler));
    _retain(texture);
    _dirty = true;
}

void MetalBindlessArray::remove_buffer(size_t index) noexcept {
    std::scoped_lock lock{_mutex};
    if (auto &&p = _buffer_slots[index]; p.handle != nullptr) {
        _release(p.handle);
        p.handle = nullptr;
    }
}

void MetalBindlessArray::remove_tex2d(size_t index) noexcept {
    std::scoped_lock lock{_mutex};
    if (auto &&p = _tex2d_slots[index]; p.handle != nullptr) {
        _release(p.handle);
        p.handle = nullptr;
    }
}

void MetalBindlessArray::remove_tex3d(size_t index) noexcept {
    std::scoped_lock lock{_mutex};
    if (auto &&p = _tex3d_slots[index]; p.handle != nullptr) {
        _release(p.handle);
        p.handle = nullptr;
    }
}

void MetalBindlessArray::_retain(id<MTLResource> r) noexcept {
    MetalBindlessResource resource{r};
    if (auto iter = _resources.find(resource);
        iter != _resources.end()) {
        iter->second++;
    } else {
        _resources.emplace(resource, 1u);
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
    auto buffer = _device->buffer(handle);
    std::scoped_lock lock{_mutex};
    return _resources.count(MetalBindlessResource{buffer}) != 0u;
}

bool MetalBindlessArray::has_texture(uint64_t handle) const noexcept {
    auto texture = _device->texture(handle);
    std::scoped_lock lock{_mutex};
    return _resources.count(MetalBindlessResource{texture}) != 0u;
}

}
