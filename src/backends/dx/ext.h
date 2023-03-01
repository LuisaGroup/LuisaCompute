#pragma once
#include <vstl/common.h>
#include <backends/common/tex_compress_ext.h>
#include <backends/common/native_resource_ext.h>
#include <backends/dx/d3dx12.h>
using namespace luisa::compute;
namespace toolhub::directx {
class Device;
class DxTexCompressExt final : public TexCompressExt, public vstd::IOperatorNewBase {
public:
    static constexpr size_t BLOCK_SIZE = 16;
    Device *device;
    DxTexCompressExt(Device *device);
    ~DxTexCompressExt();
    Result compress_bc6h(Stream &stream, Image<float> const &src, vstd::vector<std::byte> &result) noexcept override;
    Result compress_bc7(Stream &stream, Image<float> const &src, vstd::vector<std::byte> &result, float alphaImportance) noexcept override;
    Result check_builtin_shader() noexcept override;
};
struct NativeTextureDesc {
    D3D12_RESOURCE_STATES initState;
    // custom_format only for LC's non-supported format, like DXGI_FORMAT_R10G10B10A2_UNORM
    // DXGI_FORMAT_UNKNOWN for default
    DXGI_FORMAT custom_format;
    bool allowUav;
};
class DxNativeResourceExt final : public NativeResourceExt, public vstd::IOperatorNewBase {
public:
    Device *dx_device;

    DxNativeResourceExt(DeviceInterface *lc_device, Device *dx_device);
    ~DxNativeResourceExt() = default;
    BufferCreationInfo register_external_buffer(
        void *external_ptr,
        const Type *element,
        size_t elem_count,
        // D3D12_RESOURCE_STATES const*
        void *custom_data) noexcept override;
    ResourceCreationInfo register_external_image(
        void *external_ptr,
        PixelFormat format, uint dimension,
        uint width, uint height, uint depth,
        uint mipmap_levels,
        // NativeTextureDesc const*
        void *custom_data) noexcept override;
    ResourceCreationInfo register_external_depth_buffer(
        void *external_ptr,
        DepthFormat format,
        uint width,
        uint height,
        // custom data see backends' header
        void *custom_data) noexcept override;
    static PixelFormat ToPixelFormat(GFXFormat f) {
        switch (f) {
            case GFXFormat_R8_SInt:
                return PixelFormat::R8SInt;
            case GFXFormat_R8_UInt:
                return PixelFormat::R8UInt;
            case GFXFormat_R8_UNorm:
                return PixelFormat::R8UNorm;
            case GFXFormat_R8G8_SInt:
                return PixelFormat::RG8SInt;
            case GFXFormat_R8G8_UInt:
                return PixelFormat::RG8UInt;
            case GFXFormat_R8G8_UNorm:
                return PixelFormat::RG8UNorm;
            case GFXFormat_R8G8B8A8_SInt:
                return PixelFormat::RGBA8SInt;
            case GFXFormat_R8G8B8A8_UInt:
                return PixelFormat::RGBA8UInt;
            case GFXFormat_R8G8B8A8_UNorm:
                return PixelFormat::RGBA8UNorm;
            case GFXFormat_R16_SInt:
                return PixelFormat::R16SInt;
            case GFXFormat_R16_UInt:
                return PixelFormat::R16UInt;
            case GFXFormat_R16_UNorm:
                return PixelFormat::R16UNorm;
            case GFXFormat_R16G16_SInt:
                return PixelFormat::RG16SInt;
            case GFXFormat_R16G16_UInt:
                return PixelFormat::RG16UInt;
            case GFXFormat_R16G16_UNorm:
                return PixelFormat::RG16UNorm;
            case GFXFormat_R16G16B16A16_SInt:
                return PixelFormat::RGBA16SInt;
            case GFXFormat_R16G16B16A16_UInt:
                return PixelFormat::RGBA16UInt;
            case GFXFormat_R16G16B16A16_UNorm:
                return PixelFormat::RGBA16UNorm;
            case GFXFormat_R32_SInt:
                return PixelFormat::R32SInt;
            case GFXFormat_R32_UInt:
                return PixelFormat::R32UInt;
            case GFXFormat_R32G32_SInt:
                return PixelFormat::RG32SInt;
            case GFXFormat_R32G32_UInt:
                return PixelFormat::RG32UInt;
            case GFXFormat_R32G32B32A32_SInt:
                return PixelFormat::RGBA32SInt;
            case GFXFormat_R32G32B32A32_UInt:
                return PixelFormat::RGBA32UInt;
            case GFXFormat_R16_Float:
                return PixelFormat::R16F;
            case GFXFormat_R16G16_Float:
                return PixelFormat::RG16F;
            case GFXFormat_R16G16B16A16_Float:
                return PixelFormat::RGBA16F;
            case GFXFormat_R32_Float:
                return PixelFormat::R32F;
            case GFXFormat_R32G32_Float:
                return PixelFormat::RG32F;
            case GFXFormat_R32G32B32A32_Float:
                return PixelFormat::RGBA32F;
            case GFXFormat_BC6H_UF16:
                return PixelFormat::BC6HUF16;
            case GFXFormat_BC7_UNorm:
                return PixelFormat::BC7UNorm;
            case GFXFormat_BC5_UNorm:
                return PixelFormat::BC5UNorm;
            case GFXFormat_BC4_UNorm:
                return PixelFormat::BC4UNorm;
            default:
                return static_cast<PixelFormat>(-1);
        }
    }
};
}// namespace toolhub::directx