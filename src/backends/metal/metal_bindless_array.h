//
// Created by Mike Smith on 2021/7/1.
//

#pragma once

#import <unordered_set>
#import <Metal/Metal.h>

#import <core/spin_mutex.h>
#import <core/stl.h>
#import <core/dirty_range.h>
#import <runtime/resource_tracker.h>
#import <runtime/bindless_array.h>
#import <backends/metal/metal_ring_buffer.h>

namespace luisa::compute::metal {

class MetalDevice;
class MetalStream;

struct MetalBindlessResource {
    id<MTLResource> handle;
    [[nodiscard]] auto operator<(const MetalBindlessResource &rhs) const noexcept {
        return handle < rhs.handle;
    }
};

class MetalBindlessArray {

public:
    static constexpr auto buffer_slot_size = 8u;
    static constexpr auto tex2d_slot_size = 8u;
    static constexpr auto tex3d_slot_size = 8u;
    static constexpr auto sampler_slot_size = 1u;

private:
    id<MTLBuffer> _host_buffer{nullptr};
    id<MTLBuffer> _device_buffer{nullptr};
    size_t _buffer_encoding_offset{};
    size_t _tex2d_encoding_offset{};
    size_t _tex3d_encoding_offset{};
    size_t _sampler_encoding_offset{};
    id<MTLArgumentEncoder> _buffer_encoder{nullptr};
    id<MTLArgumentEncoder> _tex2d_encoder{nullptr};
    id<MTLArgumentEncoder> _tex3d_encoder{nullptr};
    id<MTLArgumentEncoder> _sampler_encoder{nullptr};
    DirtyRange _buffer_dirty_range;
    DirtyRange _tex2d_dirty_range;
    DirtyRange _tex3d_dirty_range;
    DirtyRange _sampler_dirty_range;
    ResourceTracker _tracker;
    luisa::vector<MetalBindlessResource> _buffer_slots;
    luisa::vector<MetalBindlessResource> _tex2d_slots;
    luisa::vector<MetalBindlessResource> _tex3d_slots;

public:
    MetalBindlessArray(MetalDevice *device, size_t size) noexcept;
    void emplace_buffer(size_t index, uint64_t buffer_handle, size_t offset) noexcept;
    void emplace_tex2d(size_t index, uint64_t texture_handle, Sampler sampler) noexcept;
    void emplace_tex3d(size_t index, uint64_t texture_handle, Sampler sampler) noexcept;
    void remove_buffer(size_t index) noexcept;
    void remove_tex2d(size_t index) noexcept;
    void remove_tex3d(size_t index) noexcept;
    [[nodiscard]] auto buffer_encoding_offset() const noexcept { return _buffer_encoding_offset; }
    [[nodiscard]] auto tex2d_encoding_offset() const noexcept { return _tex2d_encoding_offset; }
    [[nodiscard]] auto tex3d_encoding_offset() const noexcept { return _tex3d_encoding_offset; }
    [[nodiscard]] auto sampler_encoding_offset() const noexcept { return _sampler_encoding_offset; }
    [[nodiscard]] bool has_buffer(uint64_t handle) const noexcept { return _tracker.uses_buffer(handle); }
    [[nodiscard]] bool has_texture(uint64_t handle) const noexcept { return _tracker.uses_texture(handle); }
    [[nodiscard]] auto desc_buffer_device() const noexcept { return _device_buffer; }
    [[nodiscard]] auto desc_buffer_host() const noexcept { return _host_buffer; }
    void update(MetalStream *stream, id<MTLCommandBuffer> command_buffer) noexcept;

    template<typename F>
    void traverse_resources(const F &f) const noexcept {
        _tracker.traverse_buffers([&f](auto handle) noexcept {
            f((__bridge id<MTLResource>)(reinterpret_cast<void *>(handle)));
        });
        _tracker.traverse_textures([&f](auto handle) noexcept {
            f((__bridge id<MTLResource>)(reinterpret_cast<void *>(handle)));
        });
    }
};

}// namespace luisa::compute::metal
