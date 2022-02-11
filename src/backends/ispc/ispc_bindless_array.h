//
// Created by Mike Smith on 2022/2/11.
//

#pragma once

#include <core/dirty_range.h>
#include <core/thread_pool.h>
#include <runtime/sampler.h>
#include <runtime/resource_tracker.h>
#include <backends/ispc/ispc_texture.h>

namespace luisa::compute::ispc {

class ISPCBindlessArray {

public:
    // for device use
    struct Item {
        const void *buffer;
        ISPCTexture::Handle tex2d;
        ISPCTexture::Handle tex3d;
        uint16_t sampler2d;
        uint16_t sampler3d;
    };

    // for resource tracking
    struct Slot {
        const void *buffer{nullptr};
        size_t buffer_offset{};
        Sampler sampler2d{};
        Sampler sampler3d{};
        const ISPCTexture *tex2d{nullptr};
        const ISPCTexture *tex3d{nullptr};
    };

private:
    luisa::vector<Slot> _slots;
    luisa::vector<Item> _items;
    DirtyRange _dirty;
    ResourceTracker _tracker;
    luisa::vector<uint64_t> _buffer_resources;

public:
    explicit ISPCBindlessArray(size_t capacity) noexcept;
    void emplace_buffer(size_t index, const void *buffer, size_t offset) noexcept;
    void emplace_tex2d(size_t index, const ISPCTexture *tex, Sampler sampler) noexcept;
    void emplace_tex3d(size_t index, const ISPCTexture *tex, Sampler sampler) noexcept;
    void remove_buffer(size_t index) noexcept;
    void remove_tex2d(size_t index) noexcept;
    void remove_tex3d(size_t index) noexcept;
    void update(ThreadPool &pool) noexcept;
    [[nodiscard]] auto handle() const noexcept { return _items.data(); }
    [[nodiscard]] bool uses_buffer(const void *buffer) const noexcept;
    [[nodiscard]] bool uses_texture(const ISPCTexture *texture) const noexcept;
};

}// namespace luisa::compute::ispc
