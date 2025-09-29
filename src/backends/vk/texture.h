#pragma once
#include "resource.h"
#include <volk.h>
#include "vk_allocator.h"
#include <luisa/runtime/rhi/pixel.h>
#include <luisa/runtime/depth_format.h>
namespace lc::vk {
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
    mutable luisa::spin_mutex _layout_mtx;
    mutable vstd::fixed_vector<VkImageLayout, 1> _layouts;
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
        return get_aspect_from_format(to_vk_format(_format));
    }
    static uint2 tex2d_tile_size(luisa::compute::PixelStorage storage);
    static uint3 tex3d_tile_size(luisa::compute::PixelStorage storage);
    uint3 tile_size() const {
        if (luisa::to_underlying(_format) > 65535u) return {};// depth
        if (_dimension <= 2) {
            return make_uint3(tex2d_tile_size(luisa::compute::pixel_format_to_storage(_format)), 1);
        } else {
            return tex3d_tile_size(luisa::compute::pixel_format_to_storage(_format));
        }
    }
    auto size() const { return _size; }

    auto mip() const { return _mip; }
    auto vk_image() const { return _vk_img; }
    auto format() const {
        return _format;
    }
    auto depth_format() const {
        if (luisa::to_underlying(_format) <= 65535u) return compute::DepthFormat::None;
        return static_cast<compute::DepthFormat>(luisa::to_underlying(_format) & 65535u);
    }
    auto layout(uint level) const {
        std::lock_guard lck{_layout_mtx};
        return _layouts[level];
    }
    auto set_layout(uint level, VkImageLayout layout) const {
        std::lock_guard lck{_layout_mtx};
        _layouts[level] = layout;
    }
    bool allow_uav() const {
        return !is_srgb(_format) && (luisa::to_underlying(_format) <= 65535u);
    }
    static VkFormat to_vk_format(compute::PixelFormat format);
    Tag tag() const override { return Tag::Texture; }
};
struct TexView {
    Texture const *tex;
    uint level;
    TexView() : tex(nullptr), level(0) {}
    TexView(Texture const *tex) : tex(tex), level(0) {}
    TexView(Texture const *tex, uint level) : tex(tex), level(level) {}
};
}// namespace lc::vk