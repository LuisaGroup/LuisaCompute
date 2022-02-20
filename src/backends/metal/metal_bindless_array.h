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
#import <backends/metal/metal_host_buffer_pool.h>

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
    static constexpr auto slot_size = 32u;
    static constexpr auto buffer_slot_size = 8u;
    static constexpr auto tex2d_slot_size = 8u;
    static constexpr auto tex3d_slot_size = 8u;
    static constexpr auto sampler_slot_size = 1u;

private:
    id<MTLBuffer> _host_buffer{nullptr};
    id<MTLBuffer> _device_buffer{nullptr};
    id<MTLArgumentEncoder> _encoder{nullptr};
    DirtyRange _dirty_range;
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
    [[nodiscard]] bool has_buffer(uint64_t handle) const noexcept { return _tracker.uses_buffer(handle); }
    [[nodiscard]] bool has_texture(uint64_t handle) const noexcept { return _tracker.uses_texture(handle); }
    [[nodiscard]] auto handle() const noexcept { return _device_buffer; }
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
