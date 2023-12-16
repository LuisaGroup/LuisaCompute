#include <luisa/core/clock.h>
#include <luisa/core/logging.h>
#include <luisa/core/dynamic_module.h>
#include <luisa/runtime/context.h>
#include <luisa/runtime/device.h>
#include <luisa/runtime/stream.h>
#include <luisa/runtime/event.h>
#include <luisa/runtime/swapchain.h>
#include <luisa/dsl/syntax.h>
#include <luisa/dsl/sugar.h>
#include <luisa/runtime/rtx/accel.h>
#include <luisa/gui/window.h>
#include <luisa/backends/ext/dx_custom_cmd.h>
#include <FidelityFX/host/ffx_fsr3.h>
#include <FidelityFX/host/backends/dx12/ffx_dx12.h>
#include <luisa/vstl/common.h>
#include <iostream>
using namespace luisa;
using namespace luisa::compute;
template<typename T>
struct FuncType;
template<typename Ret, typename... Args>
struct FuncType<Ret(Args...)> {
    using Type = Ret(Args...);
};
template<typename Ret, typename... Args>
struct FuncType<Ret (*)(Args...)> {
    using Type = Ret(Args...);
};
template<typename T>
using FuncType_t = typename FuncType<T>::Type;
#define DEFINE_FUNCPTR(func) vstd::func_ptr_t<decltype(func)> fp_##func;
#define LOAD_FUNCPTR(module, func) fp_##func = module.function<decltype(func)>(#func)
DEFINE_FUNCPTR(ffxGetScratchMemorySizeDX12);
DEFINE_FUNCPTR(ffxGetDeviceDX12);
DEFINE_FUNCPTR(ffxGetInterfaceDX12);
DEFINE_FUNCPTR(ffxGetCommandListDX12);
DEFINE_FUNCPTR(ffxGetResourceDX12);
DEFINE_FUNCPTR(GetFfxResourceDescriptionDX12);
DEFINE_FUNCPTR(ffxGetCommandQueueDX12);
DEFINE_FUNCPTR(ffxGetSwapchainDX12);
DEFINE_FUNCPTR(ffxGetDX12SwapchainPtr);
DEFINE_FUNCPTR(ffxReplaceSwapchainForFrameinterpolationDX12);
DEFINE_FUNCPTR(ffxCreateFrameinterpolationSwapchainDX12);
DEFINE_FUNCPTR(ffxCreateFrameinterpolationSwapchainForHwndDX12);
DEFINE_FUNCPTR(ffxWaitForPresents);
DEFINE_FUNCPTR(ffxRegisterFrameinterpolationUiResourceDX12);
DEFINE_FUNCPTR(ffxGetFrameinterpolationCommandlistDX12);
DEFINE_FUNCPTR(ffxGetFrameinterpolationTextureDX12);
DEFINE_FUNCPTR(ffxSetFrameGenerationConfigToSwapchainDX12);

DEFINE_FUNCPTR(ffxFsr3DispatchFrameGeneration);
DEFINE_FUNCPTR(ffxFsr3ContextCreate);
DEFINE_FUNCPTR(ffxFsr3ContextDispatchUpscale);
DEFINE_FUNCPTR(ffxFsr3SkipPresent);
DEFINE_FUNCPTR(ffxFsr3ContextGenerateReactiveMask);
DEFINE_FUNCPTR(ffxFsr3ConfigureFrameGeneration);
DEFINE_FUNCPTR(ffxFsr3ContextDestroy);
DEFINE_FUNCPTR(ffxFsr3GetUpscaleRatioFromQualityMode);
DEFINE_FUNCPTR(ffxFsr3GetRenderResolutionFromQualityMode);
DEFINE_FUNCPTR(ffxFsr3GetJitterPhaseCount);
DEFINE_FUNCPTR(ffxFsr3GetJitterOffset);
DEFINE_FUNCPTR(ffxFsr3ResourceIsNull);
int main(int argc, char *argv[]) {
    auto dx12_module = DynamicModule::load("ffx_backend_dx12_x64");
    auto fsr3_module = DynamicModule::load("ffx_fsr3_x64");
    LOAD_FUNCPTR(dx12_module, ffxGetScratchMemorySizeDX12);
    LOAD_FUNCPTR(dx12_module, ffxGetDeviceDX12);
    LOAD_FUNCPTR(dx12_module, ffxGetInterfaceDX12);
    LOAD_FUNCPTR(dx12_module, ffxGetCommandListDX12);
    LOAD_FUNCPTR(dx12_module, ffxGetResourceDX12);
    LOAD_FUNCPTR(dx12_module, GetFfxResourceDescriptionDX12);
    LOAD_FUNCPTR(dx12_module, ffxGetCommandQueueDX12);
    LOAD_FUNCPTR(dx12_module, ffxGetSwapchainDX12);
    LOAD_FUNCPTR(dx12_module, ffxGetDX12SwapchainPtr);
    LOAD_FUNCPTR(dx12_module, ffxReplaceSwapchainForFrameinterpolationDX12);
    LOAD_FUNCPTR(dx12_module, ffxCreateFrameinterpolationSwapchainDX12);
    LOAD_FUNCPTR(dx12_module, ffxCreateFrameinterpolationSwapchainForHwndDX12);
    LOAD_FUNCPTR(dx12_module, ffxWaitForPresents);
    LOAD_FUNCPTR(dx12_module, ffxRegisterFrameinterpolationUiResourceDX12);
    LOAD_FUNCPTR(dx12_module, ffxGetFrameinterpolationCommandlistDX12);
    LOAD_FUNCPTR(dx12_module, ffxGetFrameinterpolationTextureDX12);
    LOAD_FUNCPTR(dx12_module, ffxSetFrameGenerationConfigToSwapchainDX12);
    LOAD_FUNCPTR(fsr3_module, ffxFsr3DispatchFrameGeneration);

    LOAD_FUNCPTR(fsr3_module, ffxFsr3ContextCreate);
    LOAD_FUNCPTR(fsr3_module, ffxFsr3ContextDispatchUpscale);
    LOAD_FUNCPTR(fsr3_module, ffxFsr3SkipPresent);
    LOAD_FUNCPTR(fsr3_module, ffxFsr3ContextGenerateReactiveMask);
    LOAD_FUNCPTR(fsr3_module, ffxFsr3ConfigureFrameGeneration);
    LOAD_FUNCPTR(fsr3_module, ffxFsr3ContextDestroy);
    LOAD_FUNCPTR(fsr3_module, ffxFsr3GetUpscaleRatioFromQualityMode);
    LOAD_FUNCPTR(fsr3_module, ffxFsr3GetRenderResolutionFromQualityMode);
    LOAD_FUNCPTR(fsr3_module, ffxFsr3GetJitterPhaseCount);
    LOAD_FUNCPTR(fsr3_module, ffxFsr3GetJitterOffset);
    LOAD_FUNCPTR(fsr3_module, ffxFsr3ResourceIsNull);

    Context context{argv[0]};
    Device device = context.create_device("dx");

    auto scratch_size = fp_ffxGetScratchMemorySizeDX12(1);
    luisa::vector<std::byte> scratch_buffer{scratch_size};
    FfxInterface dx12_interface;
    FFX_VALIDATE(fp_ffxGetInterfaceDX12(
        &dx12_interface,
        reinterpret_cast<ID3D12Device *>(device.impl()->native_handle()),
        scratch_buffer.data(),
        scratch_buffer.size(),
        1));
    FfxFsr3UpscalerMessage msg = [](FfxMsgType type, const wchar_t *message) {
        if (type == FFX_MESSAGE_TYPE_ERROR) {
            std::wcerr << message << '\n';
        } else {
            std::wcout << message << '\n';
        }
    };
    FfxFsr3ContextDescription context_desc{
        .flags = FFX_FSR3_ENABLE_ASYNC_WORKLOAD_SUPPORT,
        .maxRenderSize = FfxDimensions2D{1024u, 1024u},
        .upscaleOutputSize = FfxDimensions2D{1024u, 1024u},
        .displaySize = FfxDimensions2D{1024u, 1024u},
        .backendInterfaceSharedResources = dx12_interface,
        .backendInterfaceUpscaling = dx12_interface,
        .backendInterfaceFrameInterpolation = dx12_interface,
        .fpMessage = msg,
        .backBufferFormat = FfxSurfaceFormat::FFX_SURFACE_FORMAT_R8G8B8A8_UNORM};
    auto fsr3_context = luisa::make_unique<FfxFsr3Context>();
    fp_ffxFsr3ContextCreate(fsr3_context.get(), &context_desc);
    fp_ffxFsr3ContextDestroy(fsr3_context.get());
    return 0;
}