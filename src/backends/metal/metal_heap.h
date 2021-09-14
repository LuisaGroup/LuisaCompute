//
// Created by Mike Smith on 2021/7/1.
//

#pragma once

#import <unordered_set>
#import <Metal/Metal.h>

#import <core/spin_mutex.h>
#import <core/allocator.h>
#import <runtime/heap.h>

namespace luisa::compute::metal {

class MetalDevice;
class MetalStream;

class MetalHeap {

private:
    MetalDevice *_device;
    id<MTLHeap> _handle;
    id<MTLBuffer> _buffer{nullptr};
    id<MTLBuffer> _device_buffer{nullptr};
    id<MTLArgumentEncoder> _encoder{nullptr};
    std::array<id<MTLSamplerState>, 16u> _samplers{};
    id<MTLEvent> _event{nullptr};
    luisa::unordered_set<uint64_t> _active_buffers;
    luisa::unordered_set<uint64_t> _active_textures;
    mutable uint64_t _event_value{0u};
    mutable __weak id<MTLCommandBuffer> _last_update{nullptr};
    mutable spin_mutex _mutex;
    mutable bool _dirty{true};
    static constexpr auto slot_size = 32u;

private:
    [[nodiscard]] static MTLHeapDescriptor *_heap_descriptor(size_t size) noexcept;

public:
    MetalHeap(MetalDevice *device, size_t size) noexcept;
    [[nodiscard]] id<MTLTexture> allocate_texture(MTLTextureDescriptor *desc) noexcept;
    [[nodiscard]] id<MTLBuffer> allocate_buffer(size_t size_bytes) noexcept;
    void emplace_buffer(uint32_t index, uint64_t buffer_handle) noexcept;
    void emplace_texture(uint32_t index, uint64_t texture_handle, TextureSampler sampler) noexcept;
    void destroy_buffer(uint64_t b) noexcept;
    void destroy_texture(uint64_t t) noexcept;
    [[nodiscard]] auto handle() const noexcept { return _handle; }
    [[nodiscard]] auto desc_buffer() const noexcept { return _device_buffer; }
    [[nodiscard]] id<MTLCommandBuffer> encode_update(MetalStream *stream, id<MTLCommandBuffer> cmd_buf) const noexcept;

    template<typename F>
    void traverse_textures(F &&f) noexcept {
        std::scoped_lock lock{_mutex};
        for (auto t : _active_textures) {
            std::invoke(std::forward<F>(f), t);
        }
    }

    template<typename F>
    void traverse_buffers(F &&f) noexcept {
        std::scoped_lock lock{_mutex};
        for (auto b : _active_buffers) {
            std::invoke(std::forward<F>(f), b);
        }
    }
};

}// namespace luisa::compute::metal
