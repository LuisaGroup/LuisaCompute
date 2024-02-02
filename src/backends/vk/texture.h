#pragma once
#include "resource.h"
#include <vulkan/vulkan.h>
#include "vk_allocator.h"
#include <luisa/runtime/rhi/pixel.h>
namespace lc::vk {
class Texture : public Resource {
    AllocatedImage _img;
    bool _simultaneous_access;
public:
    Texture(
        Device *device,
        uint dimension,
        compute::PixelFormat format,
        uint3 size,
        uint mip,
        bool simultaneous_access);
    ~Texture();
    auto vk_image() const { return _img.image; }

    static VkFormat to_vk_format(compute::PixelFormat format);
};
}// namespace lc::vk