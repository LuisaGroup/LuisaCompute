#include <DXApi/ext.h>
#include <DXApi/LCDevice.h>
#include <DXRuntime/Device.h>
#include <Resource/RenderTexture.h>
#include <DXApi/LCCmdBuffer.h>
#include <runtime/stream.h>
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
}// namespace toolhub::directx
