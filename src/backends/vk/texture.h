#pragma once
#include <atomic>

#include "resource.h"
#include <volk.h>
#include "vk_allocator.h"
#include <luisa/core/mathematics.h>
#include <luisa/core/stl/memory.h>
#include <luisa/core/logging.h>
#include <luisa/core/spin_mutex.h>
#include <luisa/runtime/rhi/pixel.h>
#include <luisa/runtime/depth_format.h>

// X11 headers define None as a macro, undef it her
#ifdef None
#undef None
#endif

namespace lc::vk {
struct NativeImageState {
    VkImage image;
    VkFormat format;
    uint3 size;
    uint mip_levels;
    uint dimension;
    bool simultaneous_access;

private:
    luisa::shared_ptr<std::atomic_size_t> _expiration_counter;
    mutable luisa::spin_mutex _layout_mtx;
    mutable vstd::fixed_vector<VkImageLayout, 1> _layouts;

public:
    NativeImageState(
        VkImage image, VkFormat format, uint dimension, uint3 size,
        uint mip_levels, bool simultaneous_access,
        luisa::shared_ptr<std::atomic_size_t> expiration_counter);
    ~NativeImageState() noexcept;
    [[nodiscard]] VkImageLayout layout(uint level) const;
    void set_layout(uint level, VkImageLayout layout) const;
};

class Texture : public Resource {
    VkImage _vk_img;
    union {
        VmaAllocation _allocation;
        VkDeviceMemory _allocated_memory;
    };
    compute::PixelFormat _format;
    uint3 _size;
    uint _mip;
    uint _dimension;
    bool _contained : 1 {true};
    bool _simultaneous_access : 1 {false};
    bool _external_allocation : 1 {false};
    luisa::shared_ptr<NativeImageState> _native_state;
    VkMemoryRequirements _memory_requirements{};
    VkSparseImageMemoryRequirements _sparse_memory_requirements{};

    void _acquire_native_state(VkFormat format);
public:
    VkDeviceMemory external_device_memory() const { return _allocated_memory; }
    bool is_external_allocation() const { return _external_allocation; }
    static VkImageAspectFlags get_aspect_from_format(VkFormat format);
    auto simultaneous_access() const { return _simultaneous_access; }
    auto dimension() const { return _dimension; }
    Texture(Device *device);
    // external
    Texture(
        Device *device,
        VkImage external_image,
        uint dimension,
        compute::PixelFormat format,
        uint3 size,
        uint mip,
        bool simultaneous_access,
        VkDeviceMemory external_memory = nullptr);
    Texture(
        Device *device,
        VkImage external_image,
        uint dimension,
        VkFormat format,
        uint3 size,
        uint mip,
        bool simultaneous_access,
        VkDeviceMemory external_memory = nullptr);
    Texture(
        Device *device,
        uint dimension,
        compute::PixelFormat format,
        uint3 size,
        uint mip,
        bool simultaneous_access,
        bool allow_raster_target);
    Texture(
        Device *device,
        compute::DepthFormat format,
        uint2 size);
    ~Texture();
    void init_as_sparse(
        uint dimension,
        compute::PixelFormat format,
        uint3 size,
        uint mip,
        bool simultaneous_access);
    VkImageAspectFlags get_aspect() const {
        return get_aspect_from_format(_native_state->format);
    }
    [[nodiscard]] uint3 tile_size() const noexcept {
        auto granularity =
            _sparse_memory_requirements.formatProperties.imageGranularity;
        return {granularity.width, granularity.height, granularity.depth};
    }
    [[nodiscard]] auto sparse_block_size() const noexcept {
        return _memory_requirements.alignment;
    }
    [[nodiscard]] auto const &memory_requirements() const noexcept {
        return _memory_requirements;
    }
    [[nodiscard]] auto const &sparse_memory_requirements() const noexcept {
        return _sparse_memory_requirements;
    }
    auto size() const { return _size; }
    [[nodiscard]] uint3 mip_extent(uint level) const {
        LUISA_ASSERT(level < _mip,
                     "Vulkan sparse image mip {} is outside [0, {}).",
                     level, _mip);
        return luisa::max(_size >> level, 1u);
    }

    auto mip() const { return _mip; }
    auto vk_image() const { return _vk_img; }
    auto const &native_state() const { return _native_state; }
    auto format() const {
        return _format;
    }
    auto depth_format() const {
        if (luisa::to_underlying(_format) <= 65535u) return compute::DepthFormat::None;
        return static_cast<compute::DepthFormat>(luisa::to_underlying(_format) & 65535u);
    }
    auto layout(uint level) const {
        return _native_state->layout(level);
    }
    auto set_layout(uint level, VkImageLayout layout) const {
        _native_state->set_layout(level, layout);
    }
    bool allow_uav() const {
        return !is_srgb(_format) && (luisa::to_underlying(_format) <= 65535u);
    }
    static VkFormat to_vk_format(compute::PixelFormat format);
    Tag tag() const override { return Tag::kTexture; }
};
struct TexView {
    Texture const *tex;
    uint level;
    TexView() : tex(nullptr), level(0) {}
    TexView(Texture const *tex) : tex(tex), level(0) {}
    TexView(Texture const *tex, uint level) : tex(tex), level(level) {}
};
}// namespace lc::vk
