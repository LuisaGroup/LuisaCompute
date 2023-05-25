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
#include <tests/common/cornell_box.h>
#define TINYOBJLOADER_IMPLEMENTATION
#include <tests/common/tiny_obj_loader.h>
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
        dispatch_params.cameraNear = upscale_setup->cam.far_plane;// depth is inverted
        dispatch_params.cameraFar = upscale_setup->cam.near_plane;// depth is inverted
        dispatch_params.cameraFovAngleVertical = upscale_setup->cam.fov;
        fsr_assert(ffxFsr2ContextDispatch(context, &dispatch_params));
    }
};

int main(int argc, char *argv[]) {
    log_level_info();

    Context context{argv[0]};
    Device device = context.create_device("dx");
    Stream stream = device.create_stream(StreamTag::GRAPHICS);
    constexpr uint32_t display_width = 1024, display_height = 1024;
    constexpr uint32_t render_width = display_width / 2, render_height = display_height / 2;
    const uint2 display_resolution{display_width, display_height};
    const uint2 render_resolution{render_width, render_height};
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
        std::vector<tinyobj::index_t> const &t = shape.mesh.indices;
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
               << mesh.build()
               << [indices = std::move(indices)]() {};
    }

    Accel accel = device.create_accel({});
    for (Mesh &m : meshes) {
        accel.emplace_back(m, make_float4x4(1.0f));
    }
    stream << heap.update()
           << accel.build()
           << synchronize();

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
    Callable generate_ray = [](Float2 p) noexcept {
        static constexpr float3 origin = make_float3(-0.01f, 0.995f, 5.0f);
        Float3 pixel = origin + make_float3(p * tan(0.5f * fov), -1.0f);
        Float3 direction = normalize(pixel - origin);
        return make_ray(origin, direction);
    };

    Callable cosine_sample_hemisphere = [](Float2 u) noexcept {
        Float r = sqrt(u.x);
        Float phi = 2.0f * constants::pi * u.y;
        return make_float3(r * cos(phi), r * sin(phi), sqrt(1.0f - u.x));
    };

    Callable balanced_heuristic = [](Float pdf_a, Float pdf_b) noexcept {
        return pdf_a / max(pdf_a + pdf_b, 1e-4f);
    };

    Kernel2D raytracing_kernel = [&](ImageFloat image, ImageUInt seed_image, ImageFloat depth_image, AccelVar accel, UInt2 resolution, Float2 jitter) noexcept {
        set_block_size(16u, 16u, 1u);
        UInt2 coord = dispatch_id().xy();
        Float frame_size = min(resolution.x, resolution.y).cast<float>();
        UInt state = seed_image.read(coord).x;
        Float2 pixel = (make_float2(coord) + make_float2(0.5) + jitter) / frame_size * 2.0f - 1.0f;
        Float3 radiance = def(make_float3(0.0f));
        Float depth_value = 0.0f;
        Var<Ray> ray = generate_ray(pixel * make_float2(1.0f, -1.0f));
        Float3 beta = def(make_float3(1.0f));
        Float pdf_bsdf = def(0.0f);
        constexpr float3 light_position = make_float3(-0.24f, 1.98f, 0.16f);
        constexpr float3 light_u = make_float3(-0.24f, 1.98f, -0.22f) - light_position;
        constexpr float3 light_v = make_float3(0.23f, 1.98f, 0.16f) - light_position;
        constexpr float3 light_emission = make_float3(17.0f, 12.0f, 4.0f);
        Float light_area = length(cross(light_u, light_v));
        Float3 light_normal = normalize(cross(light_u, light_v));
        $for(depth, 10u) {
            // trace
            Var<TriangleHit> hit = accel.trace_closest(ray);
            $if(hit->miss()) { $break; };
            $if(depth == 0) {
                depth_value = hit.committed_ray_t * abs(dot(ray->direction(), make_float3(0, 0, -1.f)));
            };
            Var<Triangle> triangle = heap->buffer<Triangle>(hit.inst).read(hit.prim);
            Float3 p0 = vertex_buffer->read(triangle.i0);
            Float3 p1 = vertex_buffer->read(triangle.i1);
            Float3 p2 = vertex_buffer->read(triangle.i2);
            Float3 p = hit->interpolate(p0, p1, p2);
            Float3 n = normalize(cross(p1 - p0, p2 - p0));
            Float cos_wi = dot(-ray->direction(), n);
            $if(cos_wi < 1e-4f) { $break; };

            // hit light
            $if(hit.inst == static_cast<uint>(meshes.size() - 1u)) {
                $if(depth == 0u) {
                    radiance += light_emission;
                }
                $else {
                    Float pdf_light = length_squared(p - ray->origin()) / (light_area * cos_wi);
                    Float mis_weight = balanced_heuristic(pdf_bsdf, pdf_light);
                    radiance += mis_weight * beta * light_emission;
                };
                $break;
            };

            // sample light
            Float ux_light = lcg(state);
            Float uy_light = lcg(state);
            Float3 p_light = light_position + ux_light * light_u + uy_light * light_v;
            Float3 pp = offset_ray_origin(p, n);
            Float3 pp_light = offset_ray_origin(p_light, light_normal);
            Float d_light = distance(pp, pp_light);
            Float3 wi_light = normalize(pp_light - pp);
            Var<Ray> shadow_ray = make_ray(offset_ray_origin(pp, n), wi_light, 0.f, d_light);
            Bool occluded = accel.trace_any(shadow_ray);
            Float cos_wi_light = dot(wi_light, n);
            Float cos_light = -dot(light_normal, wi_light);
            Float3 albedo = materials.read(hit.inst);
            $if(!occluded & cos_wi_light > 1e-4f & cos_light > 1e-4f) {
                Float pdf_light = (d_light * d_light) / (light_area * cos_light);
                Float pdf_bsdf = cos_wi_light * inv_pi;
                Float mis_weight = balanced_heuristic(pdf_light, pdf_bsdf);
                Float3 bsdf = albedo * inv_pi * cos_wi_light;
                radiance += beta * bsdf * mis_weight * light_emission / max(pdf_light, 1e-4f);
            };

            // sample BSDF
            Var<Onb> onb = make_onb(n);
            Float ux = lcg(state);
            Float uy = lcg(state);
            Float3 new_direction = onb->to_world(cosine_sample_hemisphere(make_float2(ux, uy)));
            ray = make_ray(pp, new_direction);
            beta *= albedo;
            pdf_bsdf = cos_wi * inv_pi;

            // rr
            Float l = dot(make_float3(0.212671f, 0.715160f, 0.072169f), beta);
            $if(l == 0.0f) { $break; };
            Float q = max(l, 0.05f);
            Float r = lcg(state);
            $if(r >= q) { $break; };
            beta *= 1.0f / q;
        };
        seed_image.write(coord, make_uint4(state));
        $if(any(dsl::isnan(radiance))) { radiance = make_float3(0.0f); };
        image.write(dispatch_id().xy(), make_float4(clamp(radiance * 1.5f, 0.0f, 30.0f), 1.0f));
        depth_image.write(dispatch_id().xy(), make_float4(depth_value));
    };
    auto raytracing_shader = device.compile(raytracing_kernel);
    auto make_sampler_shader = device.compile(make_sampler_kernel);
    Image<uint> seed_image = device.create_image<uint>(PixelStorage::INT1, render_resolution);
    stream << make_sampler_shader(seed_image).dispatch(render_resolution);
    ///////////////////////////// Path Tracing
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
    Window window{"FSR2 Path Tracing", display_resolution};
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
    const uint jitter_phase_count = ffxFsr2GetJitterPhaseCount(render_width, display_width);
    float sharpness = 0;
    uint frame_count = 0;
    stream << clear(upload_setup.unresolved_color_resource);
    while (!window.should_close()) {
        window.poll_events();
        delta_time = clk.toc();
        clk.tic();
        CommandList cmdlist{};
        float2 jitter;
        upload_setup.cam.near_plane = 1e-3f;
        upload_setup.cam.far_plane = 1000.f;
        upload_setup.cam.fov = fov;
        upload_setup.cam.reset = (frame_count == 0);
        fsr_assert(ffxFsr2GetJitterOffset(&jitter.x, &jitter.y, frame_count++, jitter_phase_count));
        cmdlist << clear(upload_setup.motionvector_resource)
                << clear(upload_setup.depthbuffer_resource)
                //     [&](ImageFloat image, ImageUInt seed_image, ImageFloat depth_image, AccelVar accel, UInt2 resolution, Float2 jitter) noexcept {

                << raytracing_shader(upload_setup.unresolved_color_resource, seed_image, upload_setup.depthbuffer_resource, accel, render_resolution, jitter)
                       .dispatch(render_resolution)

                << luisa::make_unique<FSRCommand>(
                       &fsr2_context,
                       &upload_setup,
                       jitter,
                       sharpness,
                       std::max<float>(delta_time, 1e-5f));
        stream << cmdlist.commit() << swap_chain.present(upload_setup.resolved_color_resource);
    }
    stream << synchronize();
    fsr_assert(ffxFsr2ContextDestroy(&fsr2_context));
}