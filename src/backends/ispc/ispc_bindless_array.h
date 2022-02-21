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

/**
 * @brief Bindless array class
 * 
 */
class ISPCBindlessArray {

public:
    /**
     * @brief An item of bindless array on the device
     * 
     */
    struct Item {
        const void *buffer;
        ISPCTexture::Handle tex2d;
        ISPCTexture::Handle tex3d;
        uint16_t sampler2d;
        uint16_t sampler3d;
    };

    /**
     * @brief Bindless array on the device
     * 
     */
    struct Handle {
        const Item *items;
    };

    /**
     * @brief Used for resource tracking
     * 
     */
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

public:
    /**
     * @brief Construct a new ISPCBindlessArray object
     * 
     * @param capacity the capacity of bindless array
     */
    explicit ISPCBindlessArray(size_t capacity) noexcept;
    /**
     * @brief Emplace a buffer
     * 
     * @param index place to emplace
     * @param buffer handle of buffer
     * @param offset buffer's offset
     */
    void emplace_buffer(size_t index, const void *buffer, size_t offset) noexcept;
    /**
     * @brief Emplace a 2D texture
     * 
     * @param index place to emplace
     * @param tex handle of 2D texture
     * @param sampler texture's sampler
     */
    void emplace_tex2d(size_t index, const ISPCTexture *tex, Sampler sampler) noexcept;
    /**
     * @brief Emplace a 3D texture
     * 
     * @param index place to emplace
     * @param tex handle of 3D texture
     * @param sampler texture's sampler
     */
    void emplace_tex3d(size_t index, const ISPCTexture *tex, Sampler sampler) noexcept;
    /**
     * @brief Remove a buffer
     * 
     * @param index place to remove
     */
    void remove_buffer(size_t index) noexcept;
    /**
     * @brief Remove a 2D texture
     * 
     * @param index place to remove
     */
    void remove_tex2d(size_t index) noexcept;
    /**
     * @brief Remove a 3D texture
     * 
     * @param index place to remove
     */
    void remove_tex3d(size_t index) noexcept;
    /**
     * @brief Update bindless array
     * 
     * @param pool thread pool
     */
    void update(ThreadPool &pool) noexcept;
    /**
     * @brief Get the handle for device usage
     * 
     * @return device handle
     */
    [[nodiscard]] auto handle() const noexcept { return Handle{_items.data()}; }
    /**
     * @brief If buffer is used in this array
     * 
     * @param buffer buffer to be tested
     * @return true 
     * @return false 
     */
    [[nodiscard]] bool uses_buffer(const void *buffer) const noexcept;
    /**
     * @brief If texture is used in this array
     * 
     * @param texture texture to be tested
     * @return true 
     * @return false 
     */
    [[nodiscard]] bool uses_texture(const ISPCTexture *texture) const noexcept;
};

}// namespace luisa::compute::ispc
