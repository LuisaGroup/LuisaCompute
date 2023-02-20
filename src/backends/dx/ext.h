#pragma once
#include <vstl/common.h>
#include <backends/common/tex_compress_ext.h>
#include <backends/common/native_resource_ext.h>
#include <d3d12.h>
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
    bool allowUav;
};
class DxNativeResourceExt final : public NativeResourceExt, public vstd::IOperatorNewBase {
public:
    Device *dx_device;
    DxNativeResourceExt(DeviceInterface *lc_device, Device *dx_device) : NativeResourceExt{lc_device}, dx_device{dx_device} {}
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
};
}// namespace toolhub::directx