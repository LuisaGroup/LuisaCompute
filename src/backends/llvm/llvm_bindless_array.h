//
// Created by Mike Smith on 2022/2/11.
//

#pragma once

#include <core/dirty_range.h>
#include <core/thread_pool.h>
#include <runtime/sampler.h>
#include <runtime/resource_tracker.h>
#include <backends/llvm/llvm_texture.h>

namespace luisa::compute::llvm {

class LLVMBindlessArray {

public:
    struct alignas(16) Item {
        const std::byte *buffer;
        const LLVMTexture *tex2d;
        const LLVMTexture *tex3d;
        uint sampler2d;
        uint sampler3d;
    };

    struct Slot {
        const void *buffer{nullptr};
        size_t buffer_offset{};
        Sampler sampler2d{};
        Sampler sampler3d{};
        const LLVMTexture *tex2d{nullptr};
        const LLVMTexture *tex3d{nullptr};
    };

private:
    luisa::vector<Slot> _slots;
    luisa::vector<Item> _items;
    DirtyRange _dirty;
    ResourceTracker _tracker;

public:
    explicit LLVMBindlessArray(size_t capacity) noexcept;
    void emplace_buffer(size_t index, const void *buffer, size_t offset) noexcept;
    void emplace_tex2d(size_t index, const LLVMTexture *tex, Sampler sampler) noexcept;
    void emplace_tex3d(size_t index, const LLVMTexture *tex, Sampler sampler) noexcept;
    void remove_buffer(size_t index) noexcept;
    void remove_tex2d(size_t index) noexcept;
    void remove_tex3d(size_t index) noexcept;
    void update(ThreadPool &pool) noexcept;
    [[nodiscard]] bool uses_resource(uint64_t handle) const noexcept;
    [[nodiscard]] auto handle() const noexcept { return _items.data(); }
};

}// namespace luisa::compute::llvm
