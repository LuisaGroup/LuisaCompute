#pragma once
#include "resource.h"
#include <vulkan/vulkan.h>
#include "vk_allocator.h"
#include <luisa/runtime/rhi/pixel.h>
namespace lc::vk {
class Texture : public Resource {
    AllocatedImage _img;
    compute::PixelFormat _format;
    uint _dimension;
    bool _simultaneous_access;
    vstd::vector<VkImageLayout> _layouts;
public:
    Texture(
        Device *device,
        uint dimension,
        compute::PixelFormat format,
        uint3 size,
        uint mip,
        bool simultaneous_access,
        bool allow_raster_target);
    ~Texture();
    auto vk_image() const { return _img.image; }
    auto dimension() const { return _dimension; }
    auto format() const { return _format; }
    auto layout(uint level) const { return _layouts[level]; }
    static VkFormat to_vk_format(compute::PixelFormat format);
};
struct TexView {
    Texture const *tex;
    uint level;
    TexView() : tex(nullptr), level(0) {}
    TexView(Texture const *tex) : tex(tex), level(0) {}
    TexView(Texture const *tex, uint level) : tex(tex), level(level) {}
};
}// namespace lc::vk