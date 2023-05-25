#include <core/clock.h>
#include <core/logging.h>
#include <runtime/context.h>
#include <runtime/device.h>
#include <runtime/stream.h>
#include <runtime/event.h>
#include <dsl/syntax.h>
#include <dsl/sugar.h>
#include <runtime/rtx/accel.h>
#include <gui/window.h>
#include <backends/dx/dx_custom_cmd.h>
// Make sure FSR2 is under this dir
#include <core/magic_enum.h>
#include <ffx_fsr2.h>
#include <dx12/ffx_fsr2_dx12.h>
// #include <ffx_fsr2_interface.h>
using namespace luisa;
using namespace luisa::compute;
struct CameraSetup {
    float4x4 camera_view;
    float4x4 camera_proj;
    float4x4 inv_camera_view;
    float4 cam_pos;
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
void fsr2_message(FfxFsr2MsgType type, const wchar_t *message) {
    luisa::wstring_view ws{message};
    luisa::vector<char> s;
    s.reserve(ws.size());
    for (auto &&i : ws) {
        s.push_back(i);
    }
    s.push_back(0);
    switch (type) {
        case FFX_FSR2_MESSAGE_TYPE_WARNING:
            LUISA_WARNING("FSR: {}", luisa::string_view{s.data(), s.size()});
            break;
        case FFX_FSR2_MESSAGE_TYPE_ERROR:
            LUISA_ERROR("FSR: {}", luisa::string_view{s.data(), s.size()});
            break;
    }
}
class FSRCommand : public DXCustomCmd {
public:
    FfxFsr2Context *context;
    UpscaleSetup *upscale_setup;
    float2 jitter;
    float sharpness;
    float delta_time;
    FSRCommand(
        FfxFsr2Context *context,
        UpscaleSetup *upscale_setup,
        float2 jitter,
        float sharpness,
        float delta_time)
        : context{context},
          upscale_setup{upscale_setup},
          jitter{jitter},
          sharpness{sharpness},
          delta_time{delta_time} {}
    StreamTag stream_tag() const noexcept override {
        return StreamTag::COMPUTE;
    }
    template<typename T>
    FfxResource get_image_resource(
        FfxFsr2Context *context,
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
        return ffxGetResourceDX12(context, reinterpret_cast<ID3D12Resource *>(image.native_handle()), name, state);
    }
    void execute(
        IDXGIAdapter1 *adapter,
        IDXGIFactory4 *dxgi_factory,
        ID3D12Device *device,
        ID3D12GraphicsCommandList4 *command_list) noexcept override {
        FfxFsr2DispatchDescription dispatch_params{};
        dispatch_params.commandList = ffxGetCommandListDX12(command_list);
        dispatch_params.color = get_image_resource(context, upscale_setup->unresolved_color_resource, L"FSR2_InputColor");
        dispatch_params.depth = get_image_resource(context, upscale_setup->depthbuffer_resource, L"FSR2_InputDepth");
        dispatch_params.motionVectors = get_image_resource(context, upscale_setup->motionvector_resource, L"FSR2_InputMotionVectors");
        dispatch_params.exposure = ffxGetResourceDX12(context, nullptr, L"FSR2_InputExposure");
        if (upscale_setup->reactive_map_resource) {
            dispatch_params.reactive = get_image_resource(context, upscale_setup->reactive_map_resource, L"FSR2_InputReactiveMap");
        } else {
            dispatch_params.reactive = ffxGetResourceDX12(context, nullptr, L"FSR2_EmptyInputReactiveMap");
        }

        if (upscale_setup->transparency_and_composition_resource) {
            dispatch_params.transparencyAndComposition = get_image_resource(context, upscale_setup->transparency_and_composition_resource, L"FSR2_TransparencyAndCompositionMap");
        } else {
            dispatch_params.transparencyAndComposition = ffxGetResourceDX12(context, nullptr, L"FSR2_EmptyTransparencyAndCompositionMap");
        }

        dispatch_params.output = get_image_resource(context, upscale_setup->resolved_color_resource, L"FSR2_OutputUpscaledColor", FFX_RESOURCE_STATE_UNORDERED_ACCESS);
        dispatch_params.jitterOffset.x = jitter.x;
        dispatch_params.jitterOffset.y = jitter.y;
        uint2 render_size = upscale_setup->unresolved_color_resource.size();
        uint2 display_size = upscale_setup->resolved_color_resource.size();
        dispatch_params.motionVectorScale.x = (float)render_size.x;
        dispatch_params.motionVectorScale.y = (float)render_size.y;
        dispatch_params.reset = upscale_setup->cam.reset;
        dispatch_params.enableSharpening = sharpness > 1e-6f;
        dispatch_params.sharpness = sharpness;
        dispatch_params.frameTimeDelta = delta_time;
        dispatch_params.preExposure = 1.0f;
        dispatch_params.renderSize.width = render_size.x;
        dispatch_params.renderSize.height = render_size.y;
        dispatch_params.cameraFar = upscale_setup->cam.far_plane;
        dispatch_params.cameraNear = upscale_setup->cam.near_plane;
        dispatch_params.cameraFovAngleVertical = upscale_setup->cam.fov;
        fsr_assert(ffxFsr2ContextDispatch(context, &dispatch_params));
    }
};

int main(int argc, char *argv[]) {
    log_level_info();

    Context context{argv[0]};
    Device device = context.create_device("dx");
    Stream stream = device.create_stream(StreamTag::GRAPHICS);
    constexpr uint32_t display_width = 1920, display_height = 1080;
    constexpr uint32_t render_width = display_width / 2, render_height = display_height / 2;
    const uint2 display_resolution{display_width, display_height};
    const uint2 render_resolution{render_width, render_height};
    Kernel2D clear_kernel = [](ImageVar<float> image) {
        image.write(dispatch_id().xy(), make_float4(0, 0, 0, 0));
    };
    Shader2D<Image<float>> clear_shader = device.compile(clear_kernel);
    FfxFsr2Context fsr2_context;
    FfxFsr2ContextDescription fsr2_desc{
        .flags = FFX_FSR2_ENABLE_DEPTH_INVERTED,
        .maxRenderSize = FfxDimensions2D{display_width, display_height},
        .displaySize = FfxDimensions2D{display_width, display_height},
        .device = device.impl()->native_handle(),
        .fpMessage = fsr2_message};
    auto scratch_size = ffxFsr2GetScratchMemorySizeDX12();
    luisa::vector<std::byte> scratch_buffer{scratch_size};
    fsr_assert(ffxFsr2GetInterfaceDX12(
        &fsr2_desc.callbacks,
        reinterpret_cast<ID3D12Device *>(device.impl()->native_handle()),
        scratch_buffer.data(),
        scratch_buffer.size()));
    fsr_assert(ffxFsr2ContextCreate(&fsr2_context, &fsr2_desc));
    Window window{"path tracing", display_resolution};
    SwapChain swap_chain{device.create_swapchain(
        window.native_handle(),
        stream,
        display_resolution,
        false, false, 3)};
    UpscaleSetup upload_setup;
    upload_setup.unresolved_color_resource = device.create_image<float>(
        PixelStorage::BYTE4,
        render_resolution);
    upload_setup.motionvector_resource = device.create_image<float>(
        PixelStorage::HALF2,
        render_resolution);
    upload_setup.depthbuffer_resource = device.create_image<float>(
        PixelStorage::SHORT1,
        render_resolution);
    upload_setup.resolved_color_resource = device.create_image<float>(
        swap_chain.backend_storage(),
        display_resolution);
    auto clear = [&](Image<float> const &image) {
        return clear_shader(image).dispatch(image.size());
    };
    float delta_time{};
    Clock clk;
    while (!window.should_close()) {
        window.poll_events();
        CommandList cmdlist{};
        cmdlist << clear(upload_setup.unresolved_color_resource)
                << clear(upload_setup.motionvector_resource)
                << clear(upload_setup.depthbuffer_resource)
                << clear(upload_setup.resolved_color_resource)
                << luisa::make_unique<FSRCommand>(
                       &fsr2_context,
                       &upload_setup,
                       float2{},
                       0.0f,
                       std::max<float>(delta_time, 1e-5f));
        cmdlist.add_callback([&]() {
            delta_time = clk.toc();
            clk.tic();
        });
        stream << cmdlist.commit() << swap_chain.present(upload_setup.resolved_color_resource);
    }
    stream << synchronize();
    fsr_assert(ffxFsr2ContextDestroy(&fsr2_context));
}