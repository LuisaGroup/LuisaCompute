//
// Created by Mike Smith on 2021/7/1.
//

#pragma once

#import <unordered_set>
#import <Metal/Metal.h>

#import <core/spin_mutex.h>
#import <core/allocator.h>
#import <core/dirty_range.h>
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
    static constexpr auto slot_size = 32u;

private:
    id<MTLBuffer> _buffer{nullptr};
    id<MTLBuffer> _device_buffer{nullptr};
    DirtyRange _dirty_range;
    id<MTLArgumentEncoder> _encoder{nullptr};
    luisa::map<MetalBindlessResource, size_t /* count */> _resources;
    luisa::vector<MetalBindlessResource> _buffer_slots;
    luisa::vector<MetalBindlessResource> _tex2d_slots;
    luisa::vector<MetalBindlessResource> _tex3d_slots;

private:
    void _retain(id<MTLResource> r) noexcept;
    void _release(id<MTLResource> r) noexcept;

public:
    MetalBindlessArray(MetalDevice *device, size_t size) noexcept;
    void emplace_buffer(size_t index, uint64_t buffer_handle, size_t offset) noexcept;
    void emplace_tex2d(size_t index, uint64_t texture_handle, Sampler sampler) noexcept;
    void emplace_tex3d(size_t index, uint64_t texture_handle, Sampler sampler) noexcept;
    void remove_buffer(size_t index) noexcept;
    void remove_tex2d(size_t index) noexcept;
    void remove_tex3d(size_t index) noexcept;
    [[nodiscard]] bool has_buffer(uint64_t handle) const noexcept;
    [[nodiscard]] bool has_texture(uint64_t handle) const noexcept;
    [[nodiscard]] auto desc_buffer() const noexcept { return _device_buffer; }
    [[nodiscard]] auto desc_buffer_host() const noexcept { return _buffer; }

    [[nodiscard]] auto dirty_range() const noexcept { return _dirty_range; }
    void clear_dirty_range() noexcept { _dirty_range.clear(); }

    template<typename F>
    decltype(auto) traverse(F &&f) const noexcept {
        for (auto &&r : _resources) { f(r.first.handle); }
    }
};

}// namespace luisa::compute::metal
