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
#include "common/cornell_box.h"
#define TINYOBJLOADER_IMPLEMENTATION
#include "common/tiny_obj_loader.h"
#include "common/projection.hpp"
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

void fsr_assert(FfxErrorCode code) {
    if (code != FFX_OK) [[unlikely]] {
#define FSR2_LOG(x) LUISA_ERROR("FSR error code: {}", #x)
        switch (code) {
            case FFX_ERROR_INVALID_POINTER: FSR2_LOG(FFX_ERROR_INVALID_POINTER); break;
            case FFX_ERROR_INVALID_ALIGNMENT: FSR2_LOG(FFX_ERROR_INVALID_ALIGNMENT); break;
            case FFX_ERROR_INVALID_SIZE: FSR2_LOG(FFX_ERROR_INVALID_SIZE); break;
            case FFX_EOF: FSR2_LOG(FFX_EOF); break;
            case FFX_ERROR_INVALID_PATH: FSR2_LOG(FFX_ERROR_INVALID_PATH); break;
            case FFX_ERROR_EOF: FSR2_LOG(FFX_ERROR_EOF); break;
            case FFX_ERROR_MALFORMED_DATA: FSR2_LOG(FFX_ERROR_MALFORMED_DATA); break;
            case FFX_ERROR_OUT_OF_MEMORY: FSR2_LOG(FFX_ERROR_OUT_OF_MEMORY); break;
            case FFX_ERROR_INCOMPLETE_INTERFACE: FSR2_LOG(FFX_ERROR_INCOMPLETE_INTERFACE); break;
            case FFX_ERROR_INVALID_ENUM: FSR2_LOG(FFX_ERROR_INVALID_ENUM); break;
            case FFX_ERROR_INVALID_ARGUMENT: FSR2_LOG(FFX_ERROR_INVALID_ARGUMENT); break;
            case FFX_ERROR_OUT_OF_RANGE: FSR2_LOG(FFX_ERROR_OUT_OF_RANGE); break;
            case FFX_ERROR_NULL_DEVICE: FSR2_LOG(FFX_ERROR_NULL_DEVICE); break;
            case FFX_ERROR_BACKEND_API_ERROR: FSR2_LOG(FFX_ERROR_BACKEND_API_ERROR); break;
            case FFX_ERROR_INSUFFICIENT_MEMORY: FSR2_LOG(FFX_ERROR_INSUFFICIENT_MEMORY); break;
        }
#undef FSR2_LOG
    }
}
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
        IDXGIFactory2 *factory,
        DirectXAllocator const *allocator,
        luisa::BinaryIO const *shaderIo,
        IDxcCompiler3 *dxcCompiler,
        IDxcLibrary *dxcLibrary,
        IDxcUtils *dxcUtils,
        ID3D12DescriptorHeap *shaderDescriptor,
        ID3D12DescriptorHeap *samplerDescriptor) noexcept override {
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
    void dispose() {
        if (_swapchain)
            fp_ffxGetDX12SwapchainPtr(_swapchain)->Release();
        _swapchain = nullptr;
    }
    ~FSR3SwapChain() {
        dispose();
    }

    [[nodiscard]] auto swapchain() const { return _swapchain; }
    [[nodiscard]] auto swapchain_dx12() const { return fp_ffxGetDX12SwapchainPtr(_swapchain); }
};
FfxErrorCode present_callback(const FfxPresentCallbackDescription *params) {
    // LUISA_INFO("Present: {}", params->isInterpolatedFrame);
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
///////////////////////////// Path Tracing
struct Onb {
    float3 tangent;
    float3 binormal;
    float3 normal;
};
// clang-format off
LUISA_STRUCT(Onb, tangent, binormal, normal){
    [[nodiscard]] Float3 to_world(Expr<float3> v) const noexcept {
        return v.x * tangent + v.y * binormal + v.z * normal;
    }
};
// clang-format on
class FSRCommand : public DXCustomCmd {
    luisa::vector<ResourceUsage> resource_usages;
    FfxFsr3Context *context;
    mutable FfxFsr3DispatchUpscaleDescription dispatch_params{};

    [[nodiscard]] luisa::span<const ResourceUsage> get_resource_usages() const noexcept override {
        return resource_usages;
    }
    template<typename T>
    FfxResource get_image_resource(
        Image<T> const &image,
        const wchar_t *name = nullptr,
        FfxResourceStates state = FFX_RESOURCE_STATE_COMPUTE_READ) {
        Argument::Texture arg{image.handle(), 0};
        switch (state) {
            case FFX_RESOURCE_STATE_UNORDERED_ACCESS:
                resource_usages.emplace_back(arg, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
                break;
            case FFX_RESOURCE_STATE_COMPUTE_READ:
                resource_usages.emplace_back(arg, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
                break;
            case FFX_RESOURCE_STATE_COPY_SRC:
                resource_usages.emplace_back(arg, D3D12_RESOURCE_STATE_COPY_SOURCE);
                break;
            case FFX_RESOURCE_STATE_COPY_DEST:
                resource_usages.emplace_back(arg, D3D12_RESOURCE_STATE_COPY_DEST);
                break;
            case FFX_RESOURCE_STATE_GENERIC_READ:
                resource_usages.emplace_back(arg, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE | D3D12_RESOURCE_STATE_COPY_SOURCE);
                break;
        }
        auto res = reinterpret_cast<ID3D12Resource *>(image.native_handle());
        luisa::wstring name_str;
        if (name) {
            name_str = name_str;
        }
        return fp_ffxGetResourceDX12(res, fp_GetFfxResourceDescriptionDX12(res), name_str.data(), state);
    }

public:
    [[nodiscard]] StreamTag stream_tag() const noexcept override {
        return StreamTag::COMPUTE;
    }
    FSRCommand(
        FfxFsr3Context *context,
        UpscaleSetup *upscale_setup,
        float2 jitter,
        float sharpness,
        float delta_time)
        : context{context} {
        dispatch_params.color = get_image_resource(upscale_setup->unresolved_color_resource, L"FSR2_InputColor");
        dispatch_params.depth = get_image_resource(upscale_setup->depthbuffer_resource, L"FSR2_InputDepth");
        dispatch_params.motionVectors = get_image_resource(upscale_setup->motionvector_resource, L"FSR2_InputMotionVectors");
        dispatch_params.exposure = FfxResource{};//get_empty_resource(L"FSR2_InputExposure");

        if (upscale_setup->reactive_map_resource) {
            dispatch_params.reactive = get_image_resource(upscale_setup->reactive_map_resource, L"FSR2_InputReactiveMap");
        } else {
            dispatch_params.reactive = FfxResource{};//get_empty_resource(L"FSR2_EmptyInputReactiveMap");
        }

        if (upscale_setup->transparency_and_composition_resource) {
            dispatch_params.transparencyAndComposition = get_image_resource(upscale_setup->transparency_and_composition_resource, L"FSR2_TransparencyAndCompositionMap");
        } else {
            dispatch_params.transparencyAndComposition = FfxResource{};//get_empty_resource(L"FSR2_EmptyTransparencyAndCompositionMap");
        }
        uint2 render_size = upscale_setup->unresolved_color_resource.size();
        dispatch_params.upscaleOutput = get_image_resource(upscale_setup->resolved_color_resource, L"FSR2_OutputUpscaledColor", FFX_RESOURCE_STATE_UNORDERED_ACCESS);
        dispatch_params.jitterOffset.x = jitter.x;
        dispatch_params.jitterOffset.y = jitter.y;
        dispatch_params.motionVectorScale.x = -(float)render_size.x;
        dispatch_params.motionVectorScale.y = -(float)render_size.y;
        dispatch_params.reset = upscale_setup->cam.reset;
        dispatch_params.enableSharpening = sharpness > 1e-5;
        dispatch_params.sharpness = sharpness;
        dispatch_params.frameTimeDelta = delta_time;
        dispatch_params.preExposure = 1.0f;
        dispatch_params.renderSize.width = render_size.x;
        dispatch_params.renderSize.height = render_size.y;
        // depth inverted
        dispatch_params.cameraNear = upscale_setup->cam.far_plane;
        dispatch_params.cameraFar = upscale_setup->cam.near_plane;
        dispatch_params.cameraFovAngleVertical = upscale_setup->cam.fov;
    }
    void execute(
        IDXGIAdapter1 *adapter,
        IDXGIFactory2 *dxgi_factory,
        ID3D12Device *device,
        ID3D12GraphicsCommandList4 *command_list) const noexcept override {
        dispatch_params.commandList = fp_ffxGetCommandListDX12(command_list);
        fsr_assert(fp_ffxFsr3ContextDispatchUpscale(context, &dispatch_params));
    }
};

int main(int argc, char *argv[]) {
    bool use_framegen = false;
    if (argc > 1) {
        if (luisa::string(argv[1]) == "--frame_gen") {
            use_framegen = true;
        }
    }
    if (use_framegen) {
        LUISA_INFO("Frame gen enabled.");
    } else {
        LUISA_INFO("Frame gen disabled.");
    }
    constexpr uint display_width = 1024, display_height = 1024;
    constexpr uint2 display_resolution = uint2(display_width, display_height);
    auto fsr3_module = DynamicModule::load("ffx_fsr3_x64");
    auto dx12_module = DynamicModule::load(
#ifndef NDEBUG
        "ffx_backend_dx12_x64d"
#else
        "ffx_backend_dx12_x64"
#endif
    );
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
    struct FfxInterfacePack {
        FfxInterface ptr;
        luisa::vector<std::byte> scratch_buffer;
    };
    Context context{argv[0]};
    DeviceConfig config{
        .extension = luisa::make_unique<ConfigExt>(),
        .inqueue_buffer_limit = false};
    auto config_ext = static_cast<ConfigExt *>(config.extension.get());
    Device device = context.create_device("dx", &config);
    FfxInterfacePack interfaces[3];
    int effects_count[] = {1, 1, 2};
    for (auto i : vstd::range(vstd::array_count(interfaces))) {
        auto size = fp_ffxGetScratchMemorySizeDX12(effects_count[i]);
        auto &v = interfaces[i];
        v.scratch_buffer.resize(size);
        fsr_assert(fp_ffxGetInterfaceDX12(
            &v.ptr,
            reinterpret_cast<ID3D12Device *>(device.impl()->native_handle()),
            v.scratch_buffer.data(),
            v.scratch_buffer.size(),
            effects_count[i]));
    }

    auto native_res_ext = device.extension<NativeResourceExt>();
    auto stream = device.create_stream();
    FfxFsr3UpscalerMessage msg = [](FfxMsgType type, const wchar_t *message) {
        if (type == FFX_MESSAGE_TYPE_ERROR) {
            std::wcerr << message << '\n';
        } else {
            std::wcout << message << '\n';
        }
    };
    FfxFsr3ContextDescription context_desc{
        .flags = FFX_FSR3_ENABLE_DEPTH_INVERTED | FFX_FSR3_ENABLE_ASYNC_WORKLOAD_SUPPORT,
        .maxRenderSize = FfxDimensions2D{display_width, display_height},
        .upscaleOutputSize = FfxDimensions2D{display_width, display_height},
        .displaySize = FfxDimensions2D{display_width, display_height},
        .backendInterfaceSharedResources = interfaces[0].ptr,
        .backendInterfaceUpscaling = interfaces[1].ptr,
        .backendInterfaceFrameInterpolation = interfaces[2].ptr,
        .fpMessage = msg,
        .backBufferFormat = FfxSurfaceFormat::FFX_SURFACE_FORMAT_R8G8B8A8_UNORM};
    auto fsr3_context = luisa::make_unique<FfxFsr3Context>();
    auto dispose_fsr3_context = vstd::scope_exit([&]() {
        fsr_assert(fp_ffxFsr3ContextDestroy(fsr3_context.get()));
    });
    fsr_assert(fp_ffxFsr3ContextCreate(fsr3_context.get(), &context_desc));
    const uint jitter_phase_count = fp_ffxFsr3GetJitterPhaseCount(display_width, display_width);
    UpscaleSetup upscale_setup;
    upscale_setup.unresolved_color_resource = device.create_image<float>(
        PixelStorage::BYTE4,
        display_resolution);
    upscale_setup.resolved_color_resource = device.create_image<float>(
        PixelStorage::BYTE4,
        display_resolution);
    upscale_setup.motionvector_resource = device.create_image<float>(
        PixelStorage::HALF2,
        display_resolution);
    upscale_setup.depthbuffer_resource = device.create_image<float>(
        PixelStorage::SHORT1,
        display_resolution);
    Window window{"fsr3", display_resolution};
    FSR3SwapChain swapchain(
        config_ext->factory,
        window.native_handle(),
        stream,
        display_resolution, 3);
    auto lc_swapchain = native_res_ext->create_native_swapchain(swapchain.swapchain_dx12(), false);
    static constexpr uint32_t framebuffer_count = 3;
    TimelineEvent graphics_event = device.create_timeline_event();
    uint64_t frame_index = 0;
    FfxFrameGenerationConfig frame_gen_config{};
    frame_gen_config.frameGenerationEnabled = use_framegen;
    frame_gen_config.frameGenerationCallback = fp_ffxFsr3DispatchFrameGeneration;
    frame_gen_config.presentCallback = present_callback;
    frame_gen_config.onlyPresentInterpolated = false;
    frame_gen_config.allowAsyncWorkloads = true;
    frame_gen_config.swapChain = swapchain.swapchain();
    frame_gen_config.HUDLessColor = {};
    fsr_assert(fp_ffxFsr3ConfigureFrameGeneration(
        fsr3_context.get(),
        &frame_gen_config));
    ///////////////////////////// Path Tracing
    // load the Cornell Box scene
    tinyobj::ObjReaderConfig obj_reader_config;
    obj_reader_config.triangulate = true;
    obj_reader_config.vertex_color = false;
    tinyobj::ObjReader obj_reader;
    if (!obj_reader.ParseFromString(obj_string, "", obj_reader_config)) {
        luisa::string_view error_message = "unknown error.";
        if (auto &&e = obj_reader.Error(); !e.empty()) { error_message = e; }
        LUISA_ERROR_WITH_LOCATION("Failed to load OBJ file: {}", error_message);
    }
    if (auto &&e = obj_reader.Warning(); !e.empty()) {
        LUISA_WARNING_WITH_LOCATION("{}", e);
    }

    auto &&p = obj_reader.GetAttrib().vertices;
    luisa::vector<float3> vertices;
    vertices.reserve(p.size() / 3u);
    for (uint i = 0u; i < p.size(); i += 3u) {
        vertices.emplace_back(make_float3(
            p[i + 0u], p[i + 1u], p[i + 2u]));
    }
    LUISA_INFO(
        "Loaded mesh with {} shape(s) and {} vertices.",
        obj_reader.GetShapes().size(), vertices.size());

    BindlessArray heap = device.create_bindless_array();
    Buffer<float3> vertex_buffer = device.create_buffer<float3>(vertices.size());
    stream << vertex_buffer.copy_from(vertices.data());
    luisa::vector<Mesh> meshes;
    luisa::vector<Buffer<Triangle>> triangle_buffers;
    for (auto &&shape : obj_reader.GetShapes()) {
        uint index = static_cast<uint>(meshes.size());
        auto const &t = shape.mesh.indices;
        uint triangle_count = t.size() / 3u;
        LUISA_INFO(
            "Processing shape '{}' at index {} with {} triangle(s).",
            shape.name, index, triangle_count);
        luisa::vector<uint> indices;
        indices.reserve(t.size());
        for (tinyobj::index_t i : t) { indices.emplace_back(i.vertex_index); }
        Buffer<Triangle> &triangle_buffer = triangle_buffers.emplace_back(device.create_buffer<Triangle>(triangle_count));
        Mesh &mesh = meshes.emplace_back(device.create_mesh(vertex_buffer, triangle_buffer));
        heap.emplace_on_update(index, triangle_buffer);
        stream << triangle_buffer.copy_from(indices.data())
               << mesh.build();
    }
    uint cube_inst = static_cast<uint>(meshes.size() - 3u);
    uint tall_inst = static_cast<uint>(meshes.size() - 2u);

    Accel accel = device.create_accel({});
    for (Mesh &m : meshes) {
        accel.emplace_back(m, make_float4x4(1.0f));
    }

    Constant materials{
        make_float3(0.725f, 0.710f, 0.680f),// floor
        make_float3(0.725f, 0.710f, 0.680f),// ceiling
        make_float3(0.725f, 0.710f, 0.680f),// back wall
        make_float3(0.140f, 0.450f, 0.091f),// right wall
        make_float3(0.630f, 0.065f, 0.050f),// left wall
        make_float3(0.725f, 0.710f, 0.680f),// short box
        make_float3(0.725f, 0.710f, 0.680f),// tall box
        make_float3(0.000f, 0.000f, 0.000f),// light
    };

    Callable linear_to_srgb = [](Var<float3> x) noexcept {
        return clamp(select(1.055f * pow(x, 1.0f / 2.4f) - 0.055f,
                            12.92f * x,
                            x <= 0.00031308f),
                     0.0f, 1.0f);
    };

    Callable tea = [](UInt v0, UInt v1) noexcept {
        UInt s0 = def(0u);
        for (uint n = 0u; n < 4u; n++) {
            s0 += 0x9e3779b9u;
            v0 += ((v1 << 4) + 0xa341316cu) ^ (v1 + s0) ^ ((v1 >> 5u) + 0xc8013ea4u);
            v1 += ((v0 << 4) + 0xad90777du) ^ (v0 + s0) ^ ((v0 >> 5u) + 0x7e95761eu);
        }
        return v0;
    };

    Kernel2D make_sampler_kernel = [&](ImageUInt seed_image) noexcept {
        UInt2 p = dispatch_id().xy();
        UInt state = tea(p.x, p.y);
        seed_image.write(p, make_uint4(state));
    };

    Callable lcg = [](UInt &state) noexcept {
        constexpr uint lcg_a = 1664525u;
        constexpr uint lcg_c = 1013904223u;
        state = lcg_a * state + lcg_c;
        return cast<float>(state & 0x00ffffffu) *
               (1.0f / static_cast<float>(0x01000000u));
    };

    Callable make_onb = [](const Float3 &normal) noexcept {
        Float3 binormal = normalize(ite(
            abs(normal.x) > abs(normal.z),
            make_float3(-normal.y, normal.x, 0.0f),
            make_float3(0.0f, -normal.z, normal.y)));
        Float3 tangent = normalize(cross(binormal, normal));
        return def<Onb>(tangent, binormal, normal);
    };

    static constexpr float fov = radians(27.8f);
    static constexpr float3 cam_origin = make_float3(-0.01f, 0.995f, 5.0f);

    Callable cosine_sample_hemisphere = [](Float2 u) noexcept {
        Float r = sqrt(u.x);
        Float phi = 2.0f * constants::pi * u.y;
        return make_float3(r * cos(phi), r * sin(phi), sqrt(1.0f - u.x));
    };

    Callable balanced_heuristic = [](Float pdf_a, Float pdf_b) noexcept {
        return pdf_a / max(pdf_a + pdf_b, 1e-4f);
    };
    Callable aces_tonemapping = [](Float3 x) noexcept {
        static constexpr float a = 2.51f;
        static constexpr float b = 0.03f;
        static constexpr float c = 2.43f;
        static constexpr float d = 0.59f;
        static constexpr float e = 0.14f;
        return clamp((x * (a * x + b)) / (x * (c * x + d) + e), 0.0f, 1.0f);
    };
    Kernel2D raytracing_kernel = [&](ImageFloat image, ImageUInt seed_image, ImageFloat depth_image, ImageFloat mv_tex, AccelVar accel, Float4x4 inverse_vp, Float4x4 vp, Float2 jitter, BufferVar<float4x4> last_obj_mats, Float2 motion_vectors_scale) noexcept {
        set_block_size(16u, 16u, 1u);
        UInt2 coord = dispatch_id().xy();
        UInt2 resolution = dispatch_size().xy();
        UInt state = seed_image.read(coord).x;
        Float3 radiance;
        Float2 mv;
        Float depth_value;
        const uint spp = 512;
        for (auto i : dynamic_range(spp)) {
            Float2 pixel = (make_float2(coord) + make_float2(lcg(state), lcg(state)) + jitter * -1.0f) / make_float2(resolution.x.cast<float>(), resolution.y.cast<float>()) * 2.0f - 1.0f;
            Float4 dst_pos = inverse_vp * make_float4(pixel, 0.5f, 1.0f);
            dst_pos /= dst_pos.w;
            Var<Ray> ray = make_ray(cam_origin, normalize(dst_pos.xyz() - cam_origin), 0.3f, 1000.0f);
            Float3 beta = def(make_float3(1.0f));
            Float pdf_bsdf = def(0.0f);
            // trace
            Var<TriangleHit> hit = accel.trace_closest(ray);
            $if (!hit->miss()) {
                Var<Triangle> triangle = heap->buffer<Triangle>(hit.inst).read(hit.prim);
                Float3 p0 = vertex_buffer->read(triangle.i0);
                Float3 p1 = vertex_buffer->read(triangle.i1);
                Float3 p2 = vertex_buffer->read(triangle.i2);
                Float3 local_pos = hit->interpolate(p0, p1, p2);
                Float3 world_pos = (accel.instance_transform(hit.inst) * make_float4(local_pos, 1.f)).xyz();
                Float4 proj_pos = vp * make_float4(world_pos, 1.0f);
                depth_value += proj_pos.z / proj_pos.w;
                Float4x4 last_obj_mat = last_obj_mats.read(hit.inst);
                Float4 last_world_pos = last_obj_mat * make_float4(local_pos, 1.f);
                Float4 last_proj_pos = vp * last_world_pos;
                Float2 last_uv = (last_proj_pos.xy() / last_proj_pos.w) * 0.5f + 0.5f;
                Float2 curr_uv = (proj_pos.xy() / proj_pos.w) * 0.5f + 0.5f;
                mv = curr_uv - last_uv;
                mv *= motion_vectors_scale;
                UInt color_seed = tea(hit.inst, hit.prim);
                Float3 cur_radiance;
                cur_radiance.x = lcg(color_seed);
                cur_radiance.y = lcg(color_seed);
                cur_radiance.z = lcg(color_seed);
                cur_radiance = aces_tonemapping(cur_radiance);
                radiance += cur_radiance;
            };
        };
        radiance /= float(spp);
        seed_image.write(coord, make_uint4(state));
        image.write(coord, make_float4(radiance, 1.f));
        depth_image.write(coord, make_float4(depth_value));
        mv_tex.write(coord, make_float4(mv, 0.f, 0.f));
    };
    constexpr uint tex_heap_index = 0;
    Kernel2D bilinear_upscaling_kernel = [&](Var<BindlessArray> heap, ImageVar<float> img) {
        UInt2 coord = dispatch_id().xy();
        Float2 uv = (make_float2(coord) + 0.5f) / make_float2(dispatch_size().xy());
        img.write(coord, make_float4(heap.tex2d(tex_heap_index).sample(uv).xyz(), 1.0f));
    };
    Kernel1D set_obj_mat_kernel = [&](BufferVar<float4x4> buffer, AccelVar accel) {
        UInt coord = dispatch_id().x;
        buffer.write(coord, accel.instance_transform(coord));
    };
    ///////////////////////////// Path Tracing
    Clock clk;
    float sharpness = 0.05f;
    double delta_time{};
    double time = 0;
    CommandList cmdlist;
    Image<uint> seed_image = device.create_image<uint>(PixelStorage::INT1, display_resolution);
    auto raytracing_shader = device.compile(raytracing_kernel);
    auto make_sampler_shader = device.compile(make_sampler_kernel);
    auto set_obj_mat_shader = device.compile(set_obj_mat_kernel);
    stream << make_sampler_shader(seed_image).dispatch(display_resolution);
    float4x4 view = inverse(translation(cam_origin) * rotation(make_float3(1, 0, 0), pi));
    float4x4 proj = perspective_lh(fov, 1, 1000.0f, 0.3f);
    float4x4 view_proj = proj * view;
    float4x4 inverse_vp = inverse(view_proj);
    Buffer<float4x4> last_obj_mat = device.create_buffer<float4x4>(meshes.size());
    stream << heap.update() << accel.build() << set_obj_mat_shader(last_obj_mat, accel).dispatch(meshes.size());
    float2 mv_scale{1, 1};
    while (!window.should_close()) {
        if (frame_index >= framebuffer_count) {
            graphics_event.synchronize(frame_index - (framebuffer_count - 1));
        }
        float4x4 t = translation(make_float3(0.f, 0.25f + (sin(time * 0.003f) * 0.5f + 0.5f) * 0.5f, 0.f));
        accel.set_transform_on_update(tall_inst, t);
        t = rotation(make_float3(0, 1, 0), time * 0.001f);
        accel.set_transform_on_update(cube_inst, t);
        window.poll_events();
        delta_time = clk.toc();
        LUISA_INFO("{}", delta_time);
        time += delta_time;
        clk.tic();
        float2 jitter;
        fsr_assert(fp_ffxFsr3GetJitterOffset(&jitter.x, &jitter.y, frame_index, jitter_phase_count));
        cmdlist
            << raytracing_shader(
                   upscale_setup.unresolved_color_resource,
                   seed_image,
                   upscale_setup.depthbuffer_resource,
                   upscale_setup.motionvector_resource,
                   accel,
                   inverse_vp,
                   view_proj, jitter,
                   last_obj_mat,
                   mv_scale)
                   .dispatch(display_resolution)
            << set_obj_mat_shader(last_obj_mat, accel).dispatch(meshes.size())
            << accel.build();

        cmdlist << luisa::make_unique<FSRCommand>(
            fsr3_context.get(),
            &upscale_setup,
            jitter,
            sharpness,
            std::max<float>(delta_time, 1e-5f));
        stream << cmdlist.commit();
        ++frame_index;
        stream << graphics_event.signal(frame_index);
        stream << lc_swapchain.present(upscale_setup.resolved_color_resource);
    }
    stream << synchronize();
    swapchain.dispose();
    return 0;
}
#undef LOAD_FUNCPTR
#undef DEFINE_FUNCPTR