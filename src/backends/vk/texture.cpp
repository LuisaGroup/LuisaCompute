#include "texture.h"
#include "device.h"
#include "log.h"
namespace lc::vk {
using namespace luisa::compute;
Texture::Texture(Device *device)
    : Resource(device) {
    _allocation = nullptr;
}
Texture::Texture(
    Device *device,
    VkImage external_image,
    uint dimension,
    VkFormat format,
    uint3 size,
    uint mip,
    bool simultaneous_access,
    VkDeviceMemory external_memory)
    : Resource(device),
      _vk_img(external_image),
      _format(
          static_cast<compute::PixelFormat>(static_cast<uint>(format) | (1u << 31u))),
      _size(size),
      _mip(mip),
      _dimension(dimension),
      _contained{false},
      _simultaneous_access(simultaneous_access) {
    if (external_memory) {
        _allocated_memory = external_memory;
        _external_allocation = true;
    } else {
        _allocation = nullptr;
    }
}

Texture::Texture(
    Device *device,
    uint dimension,
    PixelFormat format,
    uint3 size,
    uint mip,
    bool simultaneous_access,
    bool allow_raster_target)
    : Resource(device),
      _format(format),
      _size(size),
      _mip(mip),
      _dimension(dimension),
      _simultaneous_access(simultaneous_access) {
    auto allocation = device->allocator().allocate_image(
        [&]() {
            switch (dimension) {
                case 1:
                    return VK_IMAGE_TYPE_1D;
                case 2:
                    return VK_IMAGE_TYPE_2D;
                case 3:
                    return VK_IMAGE_TYPE_3D;
            };
        }(),
        to_vk_format(format),
        size,
        mip,
        VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
            VK_IMAGE_USAGE_TRANSFER_DST_BIT |
            VK_IMAGE_USAGE_SAMPLED_BIT |
            (allow_raster_target ? VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT : 0) |
            (is_srgb(format) ? 0 : VK_IMAGE_USAGE_STORAGE_BIT));
    _vk_img = allocation.image;
    _allocation = allocation.allocation;
    _layouts.resize(mip);
}

VkImageAspectFlags Texture::get_aspect_from_format(VkFormat format) {
    switch (format) {
        case VK_FORMAT_D32_SFLOAT:
        case VK_FORMAT_X8_D24_UNORM_PACK32:
        case VK_FORMAT_D16_UNORM:
            return VK_IMAGE_ASPECT_DEPTH_BIT;
        case VK_FORMAT_S8_UINT:
            return VK_IMAGE_ASPECT_STENCIL_BIT;
        case VK_FORMAT_D16_UNORM_S8_UINT:
        case VK_FORMAT_D24_UNORM_S8_UINT:
        case VK_FORMAT_D32_SFLOAT_S8_UINT:
            return VK_IMAGE_ASPECT_STENCIL_BIT | VK_IMAGE_ASPECT_DEPTH_BIT;
        default:
            return VK_IMAGE_ASPECT_COLOR_BIT;
    }
}
Texture::Texture(
    Device *device,
    compute::DepthFormat format,
    uint2 size)
    : Resource(device),
      _format(static_cast<compute::PixelFormat>(static_cast<uint>(format) | (1u << 16u))),
      _size(make_uint3(size, 1)),
      _mip(1),
      _dimension(2),
      _simultaneous_access(false) {

    auto allocation = device->allocator().allocate_image(
        VK_IMAGE_TYPE_2D,
        to_vk_format(static_cast<compute::PixelFormat>(static_cast<uint>(format) | (1u << 16u))),
        make_uint3(size, 1),
        1,
        VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
            VK_IMAGE_USAGE_TRANSFER_DST_BIT |
            VK_IMAGE_USAGE_SAMPLED_BIT |
            VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT);
    _vk_img = allocation.image;
    _allocation = allocation.allocation;
    _layouts.resize(1);
}
Texture::~Texture() {
    if (_external_allocation) {
        vkDestroyImage(device()->logic_device(), _vk_img, Device::alloc_callbacks());
        vkFreeMemory(device()->logic_device(), _allocated_memory, Device::alloc_callbacks());
    } else if (_allocation)
        device()->allocator().destroy_image({_vk_img, _allocation});
    else if (_contained)
        vkDestroyImage(device()->logic_device(), _vk_img, Device::alloc_callbacks());
}

void Texture::init_as_sparse(
    uint dimension,
    compute::PixelFormat format,
    uint3 size,
    uint mip,
    bool simultaneous_access) {
    auto img_type = [&]() {
        switch (dimension) {
            case 1:
                return VK_IMAGE_TYPE_1D;
            case 2:
                return VK_IMAGE_TYPE_2D;
            case 3:
                return VK_IMAGE_TYPE_3D;
        };
    }();
    VkImageCreateInfo img_create_info{
        .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
        .flags = VK_IMAGE_CREATE_SPARSE_BINDING_BIT | VK_IMAGE_CREATE_SPARSE_RESIDENCY_BIT,
        .imageType = img_type,
        .format = to_vk_format(format),
        .extent = VkExtent3D{size.x, size.y, size.z},
        .mipLevels = mip,
        .arrayLayers = 1,
        .samples = VK_SAMPLE_COUNT_1_BIT,
        .tiling = VK_IMAGE_TILING_OPTIMAL,
        .usage = VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
        .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED};
    if (!is_srgb(format)) {
        img_create_info.usage |= VK_IMAGE_USAGE_STORAGE_BIT;
    }
    VK_CHECK_RESULT(vkCreateImage(device()->logic_device(), &img_create_info, Device::alloc_callbacks(), &_vk_img));
    _format = format;
    _size = size;
    _mip = mip;
    _dimension = dimension;
    _simultaneous_access = simultaneous_access;
    _layouts.resize(mip);
    // TODO
}
uint2 Texture::tex2d_tile_size(luisa::compute::PixelStorage storage) {
    auto size = pixel_storage_size(storage, is_block_compressed(storage) ? uint3(4, 4, 1) : uint3(1));
    switch (size) {
        case 1:
            return {256, 256};
        case 2:
            return {256, 128};
        case 4:
            return {128, 128};
        case 8:
            return {128, 64};
        case 16:
            return {64, 64};
        default:
            LUISA_ERROR("Invalid format.");
            return {};
    }
}
uint3 Texture::tex3d_tile_size(luisa::compute::PixelStorage storage) {
    auto size = pixel_storage_size(storage, is_block_compressed(storage) ? uint3(4, 4, 1) : uint3(1));
    switch (size) {
        case 1:
            return {64, 32, 32};
        case 2:
            return {32, 32, 32};
        case 4:
            return {32, 32, 16};
        case 8:
            return {32, 16, 16};
        case 16:
            return {16, 16, 16};
        default:
            LUISA_ERROR("Invalid format.");
            return {};
    }
}

VkFormat Texture::to_vk_format(PixelFormat format) {
    // native format
    if ((luisa::to_underlying(format) & (1u << 31u)) != 0) {
        return static_cast<VkFormat>(luisa::to_underlying(format) & ((1u << 31u) - 1u));
    }
    // depth
    else if (luisa::to_underlying(format) > 65535u) {
        auto depth_format = static_cast<compute::DepthFormat>(luisa::to_underlying(format) & 65535u);
        switch (depth_format) {
            case compute::DepthFormat::D16:
                return VK_FORMAT_D16_UNORM;
            case compute::DepthFormat::D24S8:
                return VK_FORMAT_D24_UNORM_S8_UINT;
            case compute::DepthFormat::D32:
                return VK_FORMAT_D32_SFLOAT;
            case compute::DepthFormat::D32S8A24:
                return VK_FORMAT_D32_SFLOAT_S8_UINT;
            default:
                return VK_FORMAT_UNDEFINED;
        }
    }

    switch (format) {
        case PixelFormat::R8SInt:
            return VK_FORMAT_R8_SINT;
        case PixelFormat::R8UInt:
            return VK_FORMAT_R8_UINT;
        case PixelFormat::R8UNorm:
            return VK_FORMAT_R8_UNORM;
        case PixelFormat::RG8SInt:
            return VK_FORMAT_R8G8_SINT;
        case PixelFormat::RG8UInt:
            return VK_FORMAT_R8G8_UINT;
        case PixelFormat::RG8UNorm:
            return VK_FORMAT_R8G8_UNORM;
        case PixelFormat::RGBA8SInt:
            return VK_FORMAT_R8G8B8A8_SINT;
        case PixelFormat::RGBA8UInt:
            return VK_FORMAT_R8G8B8A8_UINT;
        case PixelFormat::RGBA8SRGB:
            return VK_FORMAT_R8G8B8A8_SRGB;
        case PixelFormat::RGBA8UNorm:
            return VK_FORMAT_R8G8B8A8_UNORM;

        case PixelFormat::R16SInt:
            return VK_FORMAT_R16_SINT;
        case PixelFormat::R16UInt:
            return VK_FORMAT_R16_UINT;
        case PixelFormat::R16UNorm:
            return VK_FORMAT_R16_UNORM;
        case PixelFormat::RG16SInt:
            return VK_FORMAT_R16G16_SINT;
        case PixelFormat::RG16UInt:
            return VK_FORMAT_R16G16_UINT;
        case PixelFormat::RG16UNorm:
            return VK_FORMAT_R16G16_UNORM;
        case PixelFormat::RGBA16SInt:
            return VK_FORMAT_R16G16B16A16_SINT;
        case PixelFormat::RGBA16UInt:
            return VK_FORMAT_R16G16B16A16_UINT;
        case PixelFormat::RGBA16UNorm:
            return VK_FORMAT_R16G16B16A16_UNORM;

        case PixelFormat::R32SInt:
            return VK_FORMAT_R32_SINT;
        case PixelFormat::R32UInt:
            return VK_FORMAT_R32_UINT;
        case PixelFormat::RG32SInt:
            return VK_FORMAT_R32G32_SINT;
        case PixelFormat::RG32UInt:
            return VK_FORMAT_R32G32_UINT;
        case PixelFormat::RGBA32SInt:
            return VK_FORMAT_R32G32B32A32_SINT;
        case PixelFormat::RGBA32UInt:
            return VK_FORMAT_R32G32B32A32_UINT;
        case PixelFormat::R16F:
            return VK_FORMAT_R16_SFLOAT;
        case PixelFormat::RG16F:
            return VK_FORMAT_R16G16_SFLOAT;
        case PixelFormat::RGBA16F:
            return VK_FORMAT_R16G16B16A16_SFLOAT;
        case PixelFormat::R32F:
            return VK_FORMAT_R32_SFLOAT;
        case PixelFormat::RG32F:
            return VK_FORMAT_R32G32_SFLOAT;
        case PixelFormat::RGBA32F:
            return VK_FORMAT_R32G32B32A32_SFLOAT;
        case PixelFormat::R10G10B10A2UInt:
            return VK_FORMAT_A2R10G10B10_UINT_PACK32;
        case PixelFormat::R10G10B10A2UNorm:
            return VK_FORMAT_A2B10G10R10_UNORM_PACK32;
        case PixelFormat::R11G11B10F:
            return VK_FORMAT_B10G11R11_UFLOAT_PACK32;
        case PixelFormat::BC1UNorm:
            return VK_FORMAT_BC1_RGB_UNORM_BLOCK;
        case PixelFormat::BC2UNorm:
            return VK_FORMAT_BC2_UNORM_BLOCK;
        case PixelFormat::BC3UNorm:
            return VK_FORMAT_BC3_UNORM_BLOCK;
        case PixelFormat::BC4UNorm:
            return VK_FORMAT_BC4_UNORM_BLOCK;
        case PixelFormat::BC5UNorm:
            return VK_FORMAT_BC5_UNORM_BLOCK;
        case PixelFormat::BC6HUF16:
            return VK_FORMAT_BC6H_UFLOAT_BLOCK;
        case PixelFormat::BC7SRGB:
            return VK_FORMAT_BC7_SRGB_BLOCK;
        case PixelFormat::BC7UNorm:
            return VK_FORMAT_BC7_UNORM_BLOCK;
        default:
            return VK_FORMAT_UNDEFINED;
    }
}
}// namespace lc::vk