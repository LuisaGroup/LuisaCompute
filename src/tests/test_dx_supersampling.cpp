#include <luisa/core/clock.h>
#include <luisa/core/logging.h>
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
// Make sure FSR2 is under this dir
#include <luisa/core/magic_enum.h>
#ifdef ENABLE_FSR
#include <ffx_fsr2.h>
#include <dx12/ffx_fsr2_dx12.h>
#else
#include <xess/xess_d3d12.h>
#endif
#include "common/cornell_box.h"
#define TINYOBJLOADER_IMPLEMENTATION
#include "common/tiny_obj_loader.h"
#include "common/projection.hpp"
// #include <ffx_fsr2_interface.h>
using namespace luisa;
using namespace luisa::compute;
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
///////////////////////////// Path Tracing
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
#ifdef ENABLE_FSR
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
    mutable FfxFsr2DispatchDescription dispatch_params{};
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
    FSRCommand(
        FfxFsr2Context *context,
        UpscaleSetup *upscale_setup,
        float2 jitter,
        float sharpness,
        float delta_time)
        : context{context} {
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
        uint2 render_size = upscale_setup->unresolved_color_resource.size();
        dispatch_params.output = get_image_resource(context, upscale_setup->resolved_color_resource, L"FSR2_OutputUpscaledColor", FFX_RESOURCE_STATE_UNORDERED_ACCESS);
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
    StreamTag stream_tag() const noexcept override {
        return StreamTag::COMPUTE;
    }
    void execute(
        IDXGIAdapter1 *adapter,
        IDXGIFactory2 *dxgi_factory,
        ID3D12Device *device,
        ID3D12GraphicsCommandList4 *command_list) const noexcept override {
        dispatch_params.commandList = ffxGetCommandListDX12(command_list);
        fsr_assert(ffxFsr2ContextDispatch(context, &dispatch_params));
    }
};
#else
void xess_assert(xess_result_t code) {
    if (code != XESS_RESULT_SUCCESS) [[unlikely]] {
#define XESS_LOG(x) LUISA_ERROR("FSR error code: {}", #x)
        switch (code) {
            case XESS_RESULT_WARNING_NONEXISTING_FOLDER: XESS_LOG(XESS_RESULT_WARNING_NONEXISTING_FOLDER); break;
            case XESS_RESULT_WARNING_OLD_DRIVER: XESS_LOG(XESS_RESULT_WARNING_OLD_DRIVER); break;
            case XESS_RESULT_ERROR_UNSUPPORTED_DEVICE: XESS_LOG(XESS_RESULT_ERROR_UNSUPPORTED_DEVICE); break;
            case XESS_RESULT_ERROR_UNSUPPORTED_DRIVER: XESS_LOG(XESS_RESULT_ERROR_UNSUPPORTED_DRIVER); break;
            case XESS_RESULT_ERROR_UNINITIALIZED: XESS_LOG(XESS_RESULT_ERROR_UNINITIALIZED); break;
            case XESS_RESULT_ERROR_INVALID_ARGUMENT: XESS_LOG(XESS_RESULT_ERROR_INVALID_ARGUMENT); break;
            case XESS_RESULT_ERROR_DEVICE_OUT_OF_MEMORY: XESS_LOG(XESS_RESULT_ERROR_DEVICE_OUT_OF_MEMORY); break;
            case XESS_RESULT_ERROR_DEVICE: XESS_LOG(XESS_RESULT_ERROR_DEVICE); break;
            case XESS_RESULT_ERROR_NOT_IMPLEMENTED: XESS_LOG(XESS_RESULT_ERROR_NOT_IMPLEMENTED); break;
            case XESS_RESULT_ERROR_INVALID_CONTEXT: XESS_LOG(XESS_RESULT_ERROR_INVALID_CONTEXT); break;
            case XESS_RESULT_ERROR_OPERATION_IN_PROGRESS: XESS_LOG(XESS_RESULT_ERROR_OPERATION_IN_PROGRESS); break;
            case XESS_RESULT_ERROR_UNSUPPORTED: XESS_LOG(XESS_RESULT_ERROR_UNSUPPORTED); break;
            case XESS_RESULT_ERROR_CANT_LOAD_LIBRARY: XESS_LOG(XESS_RESULT_ERROR_CANT_LOAD_LIBRARY); break;
            case XESS_RESULT_ERROR_UNKNOWN: XESS_LOG(XESS_RESULT_ERROR_UNKNOWN); break;
        }
#undef XESS_LOG
    }
}
// From official sample
struct XessJitter {
    luisa::vector<float2> halton_samples;
    size_t jitter_index = 0;
    float get_corput(uint index, uint base) {
        float result = 0;
        float bk = 1.f;
        while (index > 0) {
            bk /= (float)base;
            result += (float)(index % base) * bk;
            index /= base;
        }
        return result;
    }

    void generate_halton(uint base1, uint base2,
                         uint start_index, uint count, float offset1, float offset2) {
        halton_samples.reserve(count);
        for (uint a = start_index; a < count + start_index; ++a) {
            halton_samples.emplace_back(get_corput(a, base1) + offset1, get_corput(a, base2) + offset2);
        }
    }

    XessJitter() {
        jitter_index = 0;
        halton_samples.clear();
        generate_halton(2, 3, 1, 32, -0.5f, -0.5f);
    }

    void reset() {
        jitter_index = 0;
    }

    void frame_move() {
        jitter_index = (jitter_index + 1) % halton_samples.size();
    }
    float2 get_jitter_values() const {
        return halton_samples[jitter_index];
    }
};
class XessCommand : public DXCustomCmd {
public:
    xess_context_handle_t xess_context{};
    xess_d3d12_execute_params_t exec_params{};
    ID3D12Resource *get_resource(Image<float> const &img, D3D12_RESOURCE_STATES state) {
        resource_usages.emplace_back(Argument::Texture{img.handle(), 0}, state);
        return reinterpret_cast<ID3D12Resource *>(img.native_handle());
    }
    XessCommand(
        xess_context_handle_t xess_context,
        UpscaleSetup *upscale_setup,
        float2 jitter,
        float sharpness,
        float delta_time) : xess_context{xess_context} {
        exec_params.pColorTexture = get_resource(upscale_setup->unresolved_color_resource, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
        exec_params.pVelocityTexture = get_resource(upscale_setup->motionvector_resource, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
        exec_params.pDepthTexture = get_resource(upscale_setup->depthbuffer_resource, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
        exec_params.pOutputTexture = get_resource(upscale_setup->resolved_color_resource, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        exec_params.jitterOffsetX = jitter.x;
        exec_params.jitterOffsetY = jitter.y;
        exec_params.exposureScale = 1;
        exec_params.resetHistory = upscale_setup->cam.reset;
        auto input_size = upscale_setup->unresolved_color_resource.size();
        exec_params.inputWidth = input_size.x;
        exec_params.inputHeight = input_size.y;
    }
    StreamTag stream_tag() const noexcept override {
        return StreamTag::COMPUTE;
    }
    void execute(
        IDXGIAdapter1 *adapter,
        IDXGIFactory2 *dxgi_factory,
        ID3D12Device *device,
        ID3D12GraphicsCommandList4 *command_list) const noexcept override {
        xess_assert(xessD3D12Execute(xess_context, command_list, &exec_params));
    }
};
#endif
int main(int argc, char *argv[]) {
    log_level_info();

    Context context{argv[0]};
    Device device = context.create_device("dx");
    Stream stream = device.create_stream(StreamTag::GRAPHICS);
    Kernel2D clear_kernel = [](ImageVar<float> image) {
        image.write(dispatch_id().xy(), make_float4(0, 0, 0, 0));
    };
    Shader2D<Image<float>> clear_shader = device.compile(clear_kernel);
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
        Float2 pixel = (make_float2(coord) + 0.5f + jitter * -1.0f) / make_float2(resolution.x.cast<float>(), resolution.y.cast<float>()) * 2.0f - 1.0f;

        Float3 radiance;
        Float2 mv;
        Float depth_value;
        Float4 dst_pos = inverse_vp * make_float4(pixel, 0.5f, 1.0f);
        dst_pos /= dst_pos.w;
        Var<Ray> ray = make_ray(cam_origin, normalize(dst_pos.xyz() - cam_origin), 0.3f, 1000.0f);
        Float3 beta = def(make_float3(1.0f));
        Float pdf_bsdf = def(0.0f);
        // trace
        Var<TriangleHit> hit = accel.trace_closest(ray);
        $if(!hit->miss()) {
            Var<Triangle> triangle = heap->buffer<Triangle>(hit.inst).read(hit.prim);
            Float3 p0 = vertex_buffer->read(triangle.i0);
            Float3 p1 = vertex_buffer->read(triangle.i1);
            Float3 p2 = vertex_buffer->read(triangle.i2);
            Float3 local_pos = hit->interpolate(p0, p1, p2);
            Float3 world_pos = (accel.instance_transform(hit.inst) * make_float4(local_pos, 1.f)).xyz();
            Float4 proj_pos = vp * make_float4(world_pos, 1.0f);
            depth_value = proj_pos.z / proj_pos.w;
            Float4x4 last_obj_mat = last_obj_mats.read(hit.inst);
            Float4 last_world_pos = last_obj_mat * make_float4(local_pos, 1.f);
            Float4 last_proj_pos = vp * last_world_pos;
            Float2 last_uv = (last_proj_pos.xy() / last_proj_pos.w) * 0.5f + 0.5f;
            Float2 curr_uv = (proj_pos.xy() / proj_pos.w) * 0.5f + 0.5f;
            mv = curr_uv - last_uv;
            mv *= motion_vectors_scale;
            UInt color_seed = tea(hit.inst, hit.prim);
            radiance.x = lcg(color_seed);
            radiance.y = lcg(color_seed);
            radiance.z = lcg(color_seed);
            radiance = aces_tonemapping(radiance);
        };
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
    uint render_width;
    uint render_height;
    constexpr uint display_width = 1024, display_height = 1024;
    render_width = display_width / 2;
    render_height = display_height / 2;
    const uint2 display_resolution{display_width, display_height};
    luisa::string window_name;
#ifdef ENABLE_FSR
    FfxFsr2Context fsr2_context;
    FfxFsr2ContextDescription fsr2_desc{
        // depth is inverted
        .flags = FFX_FSR2_ENABLE_DEPTH_INVERTED,
        .maxRenderSize = FfxDimensions2D{render_width, render_height},
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
    const uint jitter_phase_count = ffxFsr2GetJitterPhaseCount(render_width, display_width);
    window_name = "FSR2";
#else
    XessJitter xess_jitter;
    xess_context_handle_t xess_context;
    xess_d3d12_init_params_t xess_init_params{};
    xess_assert(xessD3D12CreateContext(reinterpret_cast<ID3D12Device *>(device.impl()->native_handle()), &xess_context));
    xess_init_params.outputResolution = {display_width, display_height};
    xess_init_params.qualitySetting = XESS_QUALITY_SETTING_PERFORMANCE;
    xess_init_params.initFlags = XESS_INIT_FLAG_INVERTED_DEPTH;
    xess_2d_t output_res{display_width, display_height};
    xess_2d_t input_res{};
    xessGetInputResolution(xess_context, &output_res, xess_init_params.qualitySetting, &input_res);
    render_width = input_res.x;
    render_height = input_res.y;
    xess_assert(xessD3D12Init(xess_context, &xess_init_params));
    window_name = "XeSS";
#endif
    const uint2 render_resolution{render_width, render_height};

    auto raytracing_shader = device.compile(raytracing_kernel);
    auto make_sampler_shader = device.compile(make_sampler_kernel);
    auto bilinear_upscaling_shader = device.compile(bilinear_upscaling_kernel);
    auto set_obj_mat_shader = device.compile(set_obj_mat_kernel);
    Image<uint> seed_image = device.create_image<uint>(PixelStorage::INT1, render_resolution);
    Buffer<float4x4> last_obj_mat = device.create_buffer<float4x4>(meshes.size());

    stream << make_sampler_shader(seed_image).dispatch(render_resolution);
    float4x4 view = inverse(translation(cam_origin) * rotation(make_float3(1, 0, 0), pi));
    float4x4 proj = perspective_lh(fov, 1, 1000.0f, 0.3f);
    float4x4 view_proj = proj * view;
    float4x4 inverse_vp = inverse(view_proj);
    Window window{window_name, display_resolution};
    Swapchain swap_chain{device.create_swapchain(
        window.native_handle(),
        stream,
        display_resolution,
        false)};
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
    heap.emplace_on_update(tex_heap_index, upload_setup.unresolved_color_resource, Sampler::linear_point_edge());
    float delta_time{};
    Clock clk;
    float sharpness = 0.05f;
    uint frame_count = 0;
    stream << heap.update() << accel.build() << set_obj_mat_shader(last_obj_mat, accel).dispatch(meshes.size());
    bool use_super_sampling = true;
    float2 mv_scale{};
    bool reset = true;
    window.set_key_callback([&](Key key, KeyModifiers modifiers, Action action) {
        if (action != Action::ACTION_PRESSED) return;
        switch (key) {
            case Key::KEY_SPACE:
                use_super_sampling = !use_super_sampling;
                reset = true;
                break;
        }
    });
    double time = 0;
    while (!window.should_close()) {
        float4x4 t = translation(make_float3(0.f, 0.25f + (sin(time * 0.003f) * 0.5f + 0.5f) * 0.5f, 0.f));
        accel.set_transform_on_update(tall_inst, t);
        t = rotation(make_float3(0, 1, 0), time * 0.001f);
        accel.set_transform_on_update(cube_inst, t);
        window.poll_events();
        delta_time = clk.toc();
        time += delta_time;
        clk.tic();
        CommandList cmdlist{};
        float2 jitter;
        upload_setup.cam.near_plane = 0.3f;
        upload_setup.cam.far_plane = 1000.f;
        upload_setup.cam.fov = fov;
        upload_setup.cam.reset = reset;
        reset = false;
#ifdef ENABLE_FSR
        fsr_assert(ffxFsr2GetJitterOffset(&jitter.x, &jitter.y, frame_count, jitter_phase_count));
        mv_scale = float2{1, 1};
#else
        jitter = xess_jitter.get_jitter_values();
        xess_jitter.frame_move();
        mv_scale = float2{-1, -1} * make_float2(render_resolution);
#endif
        frame_count++;
        if (!use_super_sampling) {
            jitter = float2{};
        }
        cmdlist
            << raytracing_shader(
                   upload_setup.unresolved_color_resource,
                   seed_image,
                   upload_setup.depthbuffer_resource,
                   upload_setup.motionvector_resource,
                   accel,
                   inverse_vp,
                   view_proj, jitter,
                   last_obj_mat,
                   mv_scale)
                   .dispatch(render_resolution)
            << set_obj_mat_shader(last_obj_mat, accel).dispatch(meshes.size())
            << accel.build();

        if (use_super_sampling) {
#ifdef ENABLE_FSR
            cmdlist << luisa::make_unique<FSRCommand>(
                &fsr2_context,
                &upload_setup,
                jitter,
                sharpness,
                std::max<float>(delta_time, 1e-5f));
#else
            cmdlist << luisa::make_unique<XessCommand>(
                xess_context,
                &upload_setup,
                jitter,
                sharpness,
                std::max<float>(delta_time, 1e-5f));
//TODO
#endif

        } else {
            cmdlist << bilinear_upscaling_shader(heap, upload_setup.resolved_color_resource).dispatch(upload_setup.resolved_color_resource.size());
        }
        stream << cmdlist.commit() << swap_chain.present(upload_setup.resolved_color_resource);
    }
    stream << synchronize();
#ifdef ENABLE_FSR
    fsr_assert(ffxFsr2ContextDestroy(&fsr2_context));
#else
    xess_assert(xessDestroyContext(xess_context));
#endif
}
