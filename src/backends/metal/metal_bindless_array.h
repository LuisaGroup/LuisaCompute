//
// Created by Mike Smith on 2021/7/1.
//

#pragma once

#import <unordered_set>
#import <Metal/Metal.h>

#import <core/spin_mutex.h>
#import <core/allocator.h>
#import <runtime/bindless_array.h>

namespace luisa::compute::metal {

class MetalDevice;
class MetalStream;

struct MetalBindlessReousrce {
    id<MTLResource> handle;
    [[nodiscard]] auto operator<(const MetalBindlessReousrce &rhs) const noexcept {
        return handle < rhs.handle;
    }
};

class MetalBindlessArray {

private:
    MetalDevice *_device;
    id<MTLBuffer> _buffer{nullptr};
    id<MTLBuffer> _device_buffer{nullptr};
    id<MTLArgumentEncoder> _encoder{nullptr};
    std::array<id<MTLSamplerState>, 16u> _samplers{};
    id<MTLEvent> _event{nullptr};
    luisa::set<MetalBindlessReousrce> _resources;
    mutable uint64_t _event_value{0u};
    mutable __weak id<MTLCommandBuffer> _last_update{nullptr};
    mutable spin_mutex _mutex;
    mutable bool _dirty{true};
    static constexpr auto slot_size = 32u;

public:
    MetalBindlessArray(MetalDevice *device, size_t size) noexcept;
    [[nodiscard]] id<MTLTexture> allocate_texture(MTLTextureDescriptor *desc) noexcept;
    [[nodiscard]] id<MTLBuffer> allocate_buffer(size_t size_bytes) noexcept;
    void emplace_buffer(uint32_t index, uint64_t buffer_handle) noexcept;
    void emplace_tex2d(uint32_t index, uint64_t texture_handle, Sampler sampler) noexcept;
    void emplace_tex3d(uint32_t index, uint64_t texture_handle, Sampler sampler) noexcept;
    [[nodiscard]] auto desc_buffer() const noexcept { return _device_buffer; }
    [[nodiscard]] id<MTLCommandBuffer> encode_update(MetalStream *stream, id<MTLCommandBuffer> cmd_buf) const noexcept;

    template<typename F>
    decltype(auto) traverse(F &&f) const noexcept {
        for (auto &&r : _resources) { f(r.handle); }
    }
};

}// namespace luisa::compute::metal
