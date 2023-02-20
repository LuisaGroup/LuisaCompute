#include "../ext.h"
#include <DXApi/LCDevice.h>
#include <DXRuntime/Device.h>
#include <Resource/RenderTexture.h>
#include <DXApi/LCCmdBuffer.h>
#include <runtime/stream.h>
#include <Resource/ExternalBuffer.h>
#include <Resource/ExternalTexture.h>
#include <Resource/ExternalDepth.h>
namespace toolhub::directx {
// IUtil *LCDevice::get_util() noexcept {
//     if (!util) {
//         util = vstd::create_unique(new DxTexCompressExt(&nativeDevice));
//     }
//     return util.get();
// }
DxTexCompressExt::DxTexCompressExt(Device *device)
    : device(device) {
}
DxTexCompressExt::~DxTexCompressExt() {
}
TexCompressExt::Result DxTexCompressExt::compress_bc6h(Stream &stream, Image<float> const &src, vstd::vector<std::byte> &result) noexcept {
    LCCmdBuffer *cmdBuffer = reinterpret_cast<LCCmdBuffer *>(stream.handle());

    RenderTexture *srcTex = reinterpret_cast<RenderTexture *>(src.handle());
    cmdBuffer->CompressBC(
        srcTex,
        result,
        true,
        0,
        device->defaultAllocator.get(),
        device->maxAllocatorCount);
    return Result::Success;
}

TexCompressExt::Result DxTexCompressExt::compress_bc7(Stream &stream, Image<float> const &src, vstd::vector<std::byte> &result, float alphaImportance) noexcept {
    LCCmdBuffer *cmdBuffer = reinterpret_cast<LCCmdBuffer *>(stream.handle());
    cmdBuffer->CompressBC(
        reinterpret_cast<RenderTexture *>(src.handle()),
        result,
        false,
        alphaImportance,
        device->defaultAllocator.get(),
        device->maxAllocatorCount);
    return Result::Success;
}
TexCompressExt::Result DxTexCompressExt::check_builtin_shader() noexcept {
    LUISA_INFO("start try compile setAccelKernel");
    if (!device->setAccelKernel.Check(device)) return Result::Failed;
    LUISA_INFO("start try compile bc6TryModeG10");
    if (!device->bc6TryModeG10.Check(device)) return Result::Failed;
    LUISA_INFO("start try compile bc6TryModeLE10");
    if (!device->bc6TryModeLE10.Check(device)) return Result::Failed;
    LUISA_INFO("start try compile bc6EncodeBlock");
    if (!device->bc6EncodeBlock.Check(device)) return Result::Failed;
    LUISA_INFO("start try compile bc7TryMode456");
    if (!device->bc7TryMode456.Check(device)) return Result::Failed;
    LUISA_INFO("start try compile bc7TryMode137");
    if (!device->bc7TryMode137.Check(device)) return Result::Failed;
    LUISA_INFO("start try compile bc7TryMode02");
    if (!device->bc7TryMode02.Check(device)) return Result::Failed;
    LUISA_INFO("start try compile bc7EncodeBlock");
    if (!device->bc7EncodeBlock.Check(device)) return Result::Failed;
    return Result::Success;
}
BufferCreationInfo DxNativeResourceExt::register_external_buffer(
    void *external_ptr,
    const Type *element,
    size_t elem_count,
    void *custom_data) noexcept {
    auto res = static_cast<Buffer *>(new ExternalBuffer(
        dx_device,
        reinterpret_cast<ID3D12Resource *>(external_ptr),
        *reinterpret_cast<D3D12_RESOURCE_STATES const *>(custom_data)));
    BufferCreationInfo info;
    info.handle = reinterpret_cast<uint64>(res);
    info.native_handle = res->GetResource();
    info.element_stride = element->size();
    info.total_size_bytes = element->size() * elem_count;
    return info;
}
ResourceCreationInfo DxNativeResourceExt::register_external_image(
    void *external_ptr,
    PixelFormat format, uint dimension,
    uint width, uint height, uint depth,
    uint mipmap_levels,
    void *custom_data) noexcept {
    auto desc = reinterpret_cast<NativeTextureDesc const *>(custom_data);
    auto res = static_cast<TextureBase *>(new ExternalTexture(
        dx_device,
        reinterpret_cast<ID3D12Resource *>(external_ptr),
        desc->initState,
        width,
        height,
        TextureBase::ToGFXFormat(format),
        (TextureDimension)dimension,
        depth,
        mipmap_levels,
        desc->allowUav));
    return {
        reinterpret_cast<uint64_t>(res),
        external_ptr};
}
ResourceCreationInfo DxNativeResourceExt::register_external_depth_buffer(
    void *external_ptr,
    DepthFormat format,
    uint width,
    uint height,
    // custom data see backends' header
    void *custom_data) noexcept {
    auto res = static_cast<TextureBase *>(new ExternalDepth(
        reinterpret_cast<ID3D12Resource *>(external_ptr),
        dx_device,
        width,
        height,
        format,
        *reinterpret_cast<D3D12_RESOURCE_STATES const *>(custom_data)));
    return {
        reinterpret_cast<uint64_t>(res),
        external_ptr};
}
}// namespace toolhub::directx
