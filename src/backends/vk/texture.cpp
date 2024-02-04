#include "texture.h"
#include "device.h"
namespace lc::vk {
using namespace luisa::compute;
Texture::Texture(
    Device *device,
    uint dimension,
    PixelFormat format,
    uint3 size,
    uint mip,
    bool simultaneous_access,
    bool allow_raster_target)
    : Resource(device),
      _img(device->allocator().allocate_image(
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
              VK_IMAGE_USAGE_STORAGE_BIT)),
      _format(format),
      _dimension(dimension),
      _simultaneous_access(simultaneous_access) {
    _layouts.resize(mip);
}
Texture::~Texture() {
    device()->allocator().destroy_image(_img);
}

VkFormat Texture::to_vk_format(PixelFormat format) {
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
            return VK_FORMAT_A2R10G10B10_UNORM_PACK32;
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
        case PixelFormat::BC7UNorm:
            return VK_FORMAT_BC7_UNORM_BLOCK;
    }
    return {};
}
}// namespace lc::vk