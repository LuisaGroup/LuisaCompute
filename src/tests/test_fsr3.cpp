#include <luisa/core/clock.h>
#include <luisa/core/logging.h>
#include <luisa/core/dynamic_module.h>
#include <luisa/runtime/context.h>
#include <luisa/runtime/device.h>
#include <luisa/runtime/stream.h>
#include <luisa/runtime/event.h>
#include <luisa/runtime/swapchain.h>
#include <luisa/dsl/sugar.h>
#include <luisa/runtime/rtx/accel.h>
#include <luisa/gui/window.h>
#include <luisa/backends/ext/dx_custom_cmd.h>
#include <FidelityFX/host/ffx_fsr3.h>
#include <FidelityFX/host/backends/dx12/ffx_dx12.h>
#include <luisa/vstl/common.h>
#include <iostream>
#include <luisa/backends/ext/dx_config_ext.h>
#include <luisa/backends/ext/native_resource_ext.hpp>
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

struct CameraSetup {
    float near_plane;
    float far_plane;
    float fov;
    bool reset;
};

struct UpscaleSetup {
    CameraSetup cam;
    Image<float> unresolved_color_resource;            // input0
    Image<float> motionvector_resource;                // input1
    Image<float> depthbuffer_resource;                 // input2
    Image<float> reactive_map_resource;                // input3
    Image<float> transparency_and_composition_resource;// input4
    Image<float> resolved_color_resource;              // output
};
class ConfigExt final : public DirectXDeviceConfigExt {
public:
    IDXGIFactory2 *factory;
    void ReadbackDX12Device(
        ID3D12Device *device,
        IDXGIAdapter1 *adapter,
        IDXGIFactory2 *factory) noexcept override {
        this->factory = factory;
    }
};
class FSR3SwapChain {
    FfxSwapchain _swapchain;
public:
    FSR3SwapChain(
        IDXGIFactory *factory,
        uint64_t window_handle, const Stream &stream, uint2 resolution, uint back_buffer_count = 1) {
        DXGI_SWAP_CHAIN_DESC1 swap_chain_desc = {};
        swap_chain_desc.BufferCount = back_buffer_count + 1;
        swap_chain_desc.Width = resolution.x;
        swap_chain_desc.Height = resolution.y;
        swap_chain_desc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
        swap_chain_desc.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT;
        swap_chain_desc.SwapEffect = DXGI_SWAP_EFFECT_FLIP_DISCARD;
        swap_chain_desc.Flags = DXGI_SWAP_CHAIN_FLAG_ALLOW_TEARING;
        swap_chain_desc.SampleDesc.Count = 1;
        DXGI_SWAP_CHAIN_FULLSCREEN_DESC fullscreen_desc{
            .Windowed = true};
        fp_ffxCreateFrameinterpolationSwapchainForHwndDX12(
            reinterpret_cast<HWND>(window_handle),
            &swap_chain_desc,
            &fullscreen_desc,
            reinterpret_cast<ID3D12CommandQueue *>(stream.native_handle()),
            factory,
            _swapchain);
    }
    FSR3SwapChain(FSR3SwapChain const &) = delete;
    FSR3SwapChain(FSR3SwapChain &&rhs) : _swapchain(rhs._swapchain) {
        rhs._swapchain = nullptr;
    }
    FSR3SwapChain &operator=(FSR3SwapChain const &) = delete;
    FSR3SwapChain &operator=(FSR3SwapChain &&rhs) {
        this->~FSR3SwapChain();
        new (std::launder(this)) FSR3SwapChain(std::move(rhs));
        return *this;
    }
    ~FSR3SwapChain() {
        if (_swapchain)
            fp_ffxGetDX12SwapchainPtr(_swapchain)->Release();
    }

    [[nodiscard]] auto swapchain() const { return _swapchain; }
    [[nodiscard]] auto swapchain_dx12() const { return fp_ffxGetDX12SwapchainPtr(_swapchain); }
};
FfxErrorCode present_callback(const FfxPresentCallbackDescription *params) {
    LUISA_INFO("{}", params->isInterpolatedFrame);
    auto pDxCmdList = reinterpret_cast<ID3D12GraphicsCommandList2 *>(params->commandList);
    auto pRtResource = reinterpret_cast<ID3D12Resource *>(params->outputSwapChainBuffer.resource);
    auto pBbResource = reinterpret_cast<ID3D12Resource *>(params->currentBackBuffer.resource);
    D3D12_RESOURCE_BARRIER rt_barriers[2];
    rt_barriers[0] = D3D12_RESOURCE_BARRIER{
        .Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION,
        .Transition = D3D12_RESOURCE_TRANSITION_BARRIER{
            .pResource = pRtResource,
            .Subresource = 0xffffffff,
            .StateBefore = D3D12_RESOURCE_STATE_COMMON,
            .StateAfter = D3D12_RESOURCE_STATE_COPY_DEST}};
    rt_barriers[1] = D3D12_RESOURCE_BARRIER{
        .Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION,
        .Transition = D3D12_RESOURCE_TRANSITION_BARRIER{
            .pResource = pBbResource,
            .Subresource = 0xffffffff,
            .StateBefore = D3D12_RESOURCE_STATE_COMMON,
            .StateAfter = D3D12_RESOURCE_STATE_COPY_SOURCE}};
    pDxCmdList->ResourceBarrier(2, rt_barriers);
    pDxCmdList->CopyResource(pRtResource, pBbResource);
    rt_barriers[0] = D3D12_RESOURCE_BARRIER{
        .Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION,
        .Transition = D3D12_RESOURCE_TRANSITION_BARRIER{
            .pResource = pRtResource,
            .Subresource = 0xffffffff,
            .StateBefore = D3D12_RESOURCE_STATE_COPY_DEST,
            .StateAfter = D3D12_RESOURCE_STATE_COMMON}};
    rt_barriers[1] = D3D12_RESOURCE_BARRIER{
        .Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION,
        .Transition = D3D12_RESOURCE_TRANSITION_BARRIER{
            .pResource = pBbResource,
            .Subresource = 0xffffffff,
            .StateBefore = D3D12_RESOURCE_STATE_COPY_SOURCE,
            .StateAfter = D3D12_RESOURCE_STATE_COMMON}};
    pDxCmdList->ResourceBarrier(2, rt_barriers);
    return FFX_OK;
}
int main(int argc, char *argv[]) {
    constexpr uint display_width = 1024, display_height = 1024;
    constexpr uint2 display_resolution = uint2(display_width, display_height);

    auto dx12_module = DynamicModule::load(
#ifndef NDEBUG
        "ffx_backend_dx12_x64d"
#else
        "ffx_backend_dx12_x64"
#endif
    );
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
    DeviceConfig config{
        .extension = luisa::make_unique<ConfigExt>(),
        .inqueue_buffer_limit = false};
    auto config_ext = static_cast<ConfigExt *>(config.extension.get());
    Device device = context.create_device("dx", &config);
    auto native_res_ext = device.extension<NativeResourceExt>();
    auto stream = device.create_stream(StreamTag::GRAPHICS);
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
        .maxRenderSize = FfxDimensions2D{display_width, display_height},
        .upscaleOutputSize = FfxDimensions2D{display_width, display_height},
        .displaySize = FfxDimensions2D{display_width, display_height},
        .backendInterfaceSharedResources = dx12_interface,
        .backendInterfaceUpscaling = dx12_interface,
        .backendInterfaceFrameInterpolation = dx12_interface,
        .fpMessage = msg,
        .backBufferFormat = FfxSurfaceFormat::FFX_SURFACE_FORMAT_R8G8B8A8_UNORM};
    auto fsr3_context = luisa::make_unique<FfxFsr3Context>();
    FFX_VALIDATE(fp_ffxFsr3ContextCreate(fsr3_context.get(), &context_desc));
    const uint jitter_phase_count = fp_ffxFsr3GetJitterPhaseCount(display_width, display_width);
    UpscaleSetup upload_setup;
    upload_setup.unresolved_color_resource = device.create_image<float>(
        PixelStorage::BYTE4,
        display_resolution);
    upload_setup.motionvector_resource = device.create_image<float>(
        PixelStorage::HALF2,
        display_resolution);
    upload_setup.depthbuffer_resource = device.create_image<float>(
        PixelStorage::SHORT1,
        display_resolution);
    Window window{"fsr3", display_resolution};
    FSR3SwapChain swapchain(
        config_ext->factory,
        window.native_handle(),
        stream,
        display_resolution, 2);
    auto lc_swapchain = native_res_ext->create_native_swapchain(swapchain.swapchain_dx12(), false);
    Image<float> ldr_image = device.create_image<float>(lc_swapchain.backend_storage(), display_resolution);
    Kernel2D clear_kernel = [](ImageVar<float> image) {
        image.write(dispatch_id().xy(), make_float4(0.3f, 0.6f, 0.7f, 1.f));
    };
    auto clear_shader = device.compile(clear_kernel);
    static constexpr uint32_t framebuffer_count = 3;
    TimelineEvent graphics_event = device.create_timeline_event();
    uint64_t frame_index = 0;
    FfxFrameGenerationConfig frame_gen_config{};
    frame_gen_config.frameGenerationEnabled = true;
    frame_gen_config.frameGenerationCallback = fp_ffxFsr3DispatchFrameGeneration;
    frame_gen_config.presentCallback = present_callback;
    frame_gen_config.onlyPresentInterpolated = false;
    frame_gen_config.allowAsyncWorkloads = true;
    frame_gen_config.swapChain = swapchain.swapchain();
    frame_gen_config.HUDLessColor = {};
    FFX_VALIDATE(fp_ffxFsr3ConfigureFrameGeneration(
        fsr3_context.get(),
        &frame_gen_config));
    while (!window.should_close()) {
        if (frame_index >= framebuffer_count) {
            graphics_event.synchronize(frame_index - (framebuffer_count - 1));
        }

        window.poll_events();
        stream
            << clear_shader(ldr_image).dispatch(display_resolution)
            << lc_swapchain.present(ldr_image);
        ++frame_index;
        stream << graphics_event.signal(frame_index);
    }
    // upload_setup.resolved_color_resource = device.create_image<float>(
    //     swap_chain.backend_storage(),
    //     display_resolution);
    FFX_VALIDATE(fp_ffxFsr3ContextDestroy(fsr3_context.get()));
    return 0;
}
#undef LOAD_FUNCPTR
#undef DEFINE_FUNCPTR