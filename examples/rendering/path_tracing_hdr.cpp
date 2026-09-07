#include <filesystem>
#include <iostream>
#include <string_view>
#include <luisa/backends/ext/dx_hdr_ext.hpp>
#include <stb/stb_image_write.h>

#include "common/reference_compare.h"
#include "common/path_tracing_sample_plan.h"

#include <luisa/luisa-compute.h>
#include <luisa/dsl/sugar.h>

#include "cornell_box.h"

#define TINYOBJLOADER_IMPLEMENTATION
#include "tiny_obj_loader.h"

using namespace luisa;
using namespace luisa::compute;

struct Onb {
    float3 tangent;
    float3 binormal;
    float3 normal;
};

LUISA_STRUCT(Onb, tangent, binormal, normal) {
    [[nodiscard]] Float3 to_world(Expr<float3> v) const noexcept {
        return v.x * tangent + v.y * binormal + v.z * normal;
    }
};

int main(int argc, char *argv[]) {

    log_level_verbose();

    Context context{argv[0]};
    if (argc <= 1) {
        LUISA_INFO("Usage: {} <backend>. <backend>: cuda, dx, metal, vk, hip, fallback", argv[0]);
        exit(1);
    }

    auto opts = luisa::ref::ExampleOptions::parse(argc, argv);
    if (!opts.valid()) {
        LUISA_WARNING("Invalid command line: {}", opts.error_message);
        return 1;
    }

    Device device = context.create_device(argv[1]);

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
    Stream stream = device.create_stream(opts.offline ? StreamTag::COMPUTE : StreamTag::GRAPHICS);
    Buffer<float3> vertex_buffer = device.create_buffer<float3>(vertices.size());
    stream << vertex_buffer.copy_from(luisa::span{vertices});
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
        stream << triangle_buffer.copy_from(luisa::span{indices})
               << mesh.build();
    }

    Accel accel = device.create_accel({});
    for (Mesh &m : meshes) {
        accel.emplace_back(m, make_float4x4(1.0f));
    }
    stream << heap.update()
           << accel.build()
           << synchronize();

    // Constant materials{
    //     make_float3(0.725f, 0.710f, 0.680f),// floor
    //     make_float3(0.725f, 0.710f, 0.680f),// ceiling
    //     make_float3(0.725f, 0.710f, 0.680f),// back wall
    //     make_float3(0.140f, 0.450f, 0.091f),// right wall
    //     make_float3(0.630f, 0.065f, 0.050f),// left wall
    //     make_float3(0.725f, 0.710f, 0.680f),// short box
    //     make_float3(0.725f, 0.710f, 0.680f),// tall box
    //     make_float3(0.000f, 0.000f, 0.000f),// light
    // };

    float3 materials_array[] = {
        make_float3(0.725f, 0.710f, 0.680f),// floor
        make_float3(0.725f, 0.710f, 0.680f),// ceiling
        make_float3(0.725f, 0.710f, 0.680f),// back wall
        make_float3(0.140f, 0.450f, 0.091f),// right wall (green)
        make_float3(0.630f, 0.065f, 0.050f),// left wall (red)
        make_float3(0.725f, 0.710f, 0.680f),// short box
        make_float3(0.725f, 0.710f, 0.680f),// tall box
        make_float3(0.000f, 0.000f, 0.000f),// light (emissive, not used directly)
    };
    auto materials = device.create_buffer<float3>(8);
    stream << materials.copy_from(luisa::span{materials_array, std::size(materials_array)});
    Callable linear_to_srgb = [&](Var<float3> x) noexcept {
        return saturate(select(1.055f * pow(x, 1.0f / 2.4f) - 0.055f,
                               12.92f * x,
                               x <= 0.00031308f));
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

    Callable generate_ray = [](Float2 p) noexcept {
        static constexpr float fov = radians(27.8f);
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

    auto max_spp_per_dispatch = device.backend_name() == "metal" || device.backend_name() == "fallback" ? 1u : 64u;

    Kernel2D raytracing_kernel = [&](ImageFloat image, ImageUInt seed_image, AccelVar accel, UInt2 resolution, UInt dispatch_spp) noexcept {
        set_block_size(16u, 16u, 1u);
        UInt2 coord = dispatch_id().xy();
        Float frame_size = min(resolution.x, resolution.y).cast<float>();
        UInt state = seed_image.read(coord).x;
        Float3 radiance = def(make_float3(0.0f));
        $for (i, dispatch_spp) {
            Float rx = lcg(state);
            Float ry = lcg(state);
            Float2 pixel = (make_float2(coord) + make_float2(rx, ry)) / frame_size * 2.0f - 1.0f;
            Var<Ray> ray = generate_ray(pixel * make_float2(1.0f, -1.0f));
            Float3 beta = def(make_float3(1.0f));
            Float pdf_bsdf = def(0.0f);
            constexpr float3 light_position = make_float3(-0.24f, 1.98f, 0.16f);
            constexpr float3 light_u = make_float3(-0.24f, 1.98f, -0.22f) - light_position;
            constexpr float3 light_v = make_float3(0.23f, 1.98f, 0.16f) - light_position;
            constexpr float3 light_emission = make_float3(17.0f, 12.0f, 4.0f);
            Float light_area = length(cross(light_u, light_v));
            Float3 light_normal = normalize(cross(light_u, light_v));
            $for (depth, 10u) {
                // trace
                Var<TriangleHit> hit = accel.intersect(ray, {});
                reorder_shader_execution();
                $if (hit->miss()) { $break; };
                Var<Triangle> triangle = heap->buffer<Triangle>(hit.inst).read(hit.prim);
                Float3 p0 = vertex_buffer->read(triangle.i0);
                Float3 p1 = vertex_buffer->read(triangle.i1);
                Float3 p2 = vertex_buffer->read(triangle.i2);
                Float3 p = triangle_interpolate(hit.bary, p0, p1, p2);
                Float3 n = normalize(cross(p1 - p0, p2 - p0));
                Float cos_wo = dot(-ray->direction(), n);
                $if (cos_wo < 1e-4f) { $break; };

                // hit light
                $if (hit.inst == static_cast<uint>(meshes.size() - 1u)) {
                    $if (depth == 0u) {
                        radiance += light_emission;
                    }
                    $else {
                        Float pdf_light = length_squared(p - ray->origin()) / (light_area * cos_wo);
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
                Bool occluded = accel.intersect_any(shadow_ray, {});
                Float cos_wi_light = dot(wi_light, n);
                Float cos_light = -dot(light_normal, wi_light);
                Float3 albedo = materials->read(hit.inst);
                $if (!occluded & cos_wi_light > 1e-4f & cos_light > 1e-4f) {
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
                Float3 wi_local = cosine_sample_hemisphere(make_float2(ux, uy));
                Float cos_wi = abs(wi_local.z);
                Float3 new_direction = onb->to_world(wi_local);
                ray = make_ray(pp, new_direction);
                pdf_bsdf = cos_wi * inv_pi;
                beta *= albedo;// * cos_wi * inv_pi / pdf_bsdf => * 1.f

                // rr
                Float l = dot(make_float3(0.212671f, 0.715160f, 0.072169f), beta);
                $if (l == 0.0f) { $break; };
                Float q = max(l, 0.05f);
                Float r = lcg(state);
                $if (r >= q) { $break; };
                beta *= 1.0f / q;
            };
        };
        radiance /= dispatch_spp.cast<float>();
        seed_image.write(coord, make_uint4(state));
        $if (any(dsl::isnan(radiance))) { radiance = make_float3(0.0f); };
        image.write(dispatch_id().xy(), make_float4(clamp(radiance, 0.0f, 30.0f), dispatch_spp.cast<float>()));
    };

    Kernel2D accumulate_kernel = [&](ImageFloat accum_image, ImageFloat curr_image) noexcept {
        UInt2 p = dispatch_id().xy();
        Float4 accum = accum_image.read(p);
        Float4 curr = curr_image.read(p);
        accum_image.write(p, accum + make_float4(curr.xyz() * curr.w, curr.w));
    };

    // http://filmicworlds.com/blog/filmic-tonemapping-with-piecewise-power-curves/
    Callable filmic_aces = [](Float3 x) noexcept {
        constexpr float A = 0.22;
        constexpr float B = 0.30;
        constexpr float C = 0.10;
        constexpr float D = 0.20;
        constexpr float E = 0.01;
        constexpr float F = 0.30;
        x = max(x, make_float3(0.f));
        return ((x * (A * x + C * B) + D * E) / (x * (A * x + B) + D * F)) - E / F;
    };

    Callable rec709_to_rec2020 = [](Float3 color) {
        float3x3 conversion =
            transpose(make_float3x3(
                0.627402, 0.329292, 0.043306,
                0.069095, 0.919544, 0.011360,
                0.016394, 0.088028, 0.895578));
        return conversion * color;
    };

    Callable linear_to_st2084 = [](Float3 color) noexcept {
        Float m1 = 2610.f / 4096.f / 4.f;
        Float m2 = 2523.f / 4096.f * 128.f;
        Float c1 = 3424.f / 4096.f;
        Float c2 = 2413.f / 4096.f * 32.f;
        Float c3 = 2392.f / 4096.f * 32.f;
        Float3 cp = pow(abs(color), m1);
        return pow((c1 + c2 * cp) / (1.0f + c3 * cp), m2);
    };

    Kernel2D clear_kernel = [](ImageFloat image) noexcept {
        image.write(dispatch_id().xy(), make_float4(0.f));
    };

    Kernel2D hdr2ldr_kernel = [&](ImageFloat hdr_image, ImageFloat ldr_image, Float scale, Int mode, Float3 white_point, Bool aces) noexcept {
        UInt2 coord = dispatch_id().xy();
        Float4 hdr = hdr_image.read(coord);
        Float3 ldr = hdr.xyz() / hdr.w * scale;
        $if (aces) {
            ldr = (filmic_aces(ldr) / filmic_aces(11.2f * white_point)) * white_point;
        };
        $switch (mode) {
            // sRGB
            $case (0) {
                ldr = linear_to_srgb(ldr);
            };
            // 10-bit
            $case (1) {

                const float st2084max = 10000.0;
                const float hdrScalar = 80.0f / st2084max;

                // The HDR scene is in Rec.709, but the display is Rec.2020
                ldr = rec709_to_rec2020(ldr);

                // Apply the ST.2084 curve to the scene.
                ldr = linear_to_st2084(ldr * hdrScalar);
            };
            // 16-bit
            $case (2) {
                // LINEAR
            };
        };
        ldr_image.write(coord, make_float4(ldr, 1.f));
    };

    ShaderOption o{.enable_debug_info = false};
    auto clear_shader = device.compile(clear_kernel, o);
    auto hdr2ldr_shader = device.compile(hdr2ldr_kernel, o);
    auto accumulate_shader = device.compile(accumulate_kernel, o);
    auto raytracing_shader = device.compile(raytracing_kernel, ShaderOption{.name = "path_tracing"});
    auto make_sampler_shader = device.compile(make_sampler_kernel, o);

    static constexpr uint2 resolution = make_uint2(1024);
    Image<float> framebuffer = device.create_image<float>(PixelStorage::HALF4, resolution);
    Image<float> accum_image = device.create_image<float>(PixelStorage::FLOAT4, resolution);
    CommandList cmd_list;
    Image<uint> seed_image = device.create_image<uint>(PixelStorage::INT1, resolution);
    cmd_list << clear_shader(accum_image).dispatch(resolution)
             << make_sampler_shader(seed_image).dispatch(resolution);

    std::unique_ptr<Window> window;
    std::optional<Swapchain> swap_chain;
    bool use_aces = false;
    float3 white_point{1.0f};
    float scale = 1.0f;

    if (!opts.offline) {
        window = std::make_unique<Window>("path tracing", resolution);

        if (device.backend_name() == "dx") {
            constexpr bool use_hdr10 = true;
            constexpr PixelStorage hdr_storage = use_hdr10 ? PixelStorage::R10G10B10A2 : PixelStorage::HALF4;
            constexpr DXHDRExt::ColorSpace hdr_color_space = use_hdr10 ? DXHDRExt::ColorSpace::RGB_FULL_G2084_NONE_P2020 : DXHDRExt::ColorSpace::RGB_FULL_G10_NONE_P709;
            auto dx_hdr_ext = device.extension<DXHDRExt>();
            swap_chain.emplace(dx_hdr_ext->create_swapchain(
                stream,
                DXHDRExt::DXSwapchainOption{
                    .window = window->native_handle(),
                    .size = make_uint2(resolution),
                    .storage = dx_hdr_ext->device_support_hdr() ? hdr_storage : PixelStorage::BYTE4,
                    .wants_vsync = false,
                }));
            dx_hdr_ext->set_color_space(*swap_chain, dx_hdr_ext->device_support_hdr() ? hdr_color_space : DXHDRExt::ColorSpace::RGB_FULL_G22_NONE_P709);
            if (dx_hdr_ext->device_support_hdr()) {
                auto display_data = dx_hdr_ext->get_display_data(window->native_handle());
                LUISA_INFO("Display data: \nbits_per_color: {}\ncolor_space: {}\nred_primary: {}\ngreen_primary: {}\nblue_primary: {}\nwhite_point: {}\nmin_luminance: {}\nmax_luminance: {}\nmax_full_frame_luminance: {}", display_data.bits_per_color, luisa::to_string(display_data.color_space), display_data.red_primary, display_data.green_primary, display_data.blue_primary, display_data.white_point, display_data.min_luminance, display_data.max_luminance, display_data.max_full_frame_luminance);
                white_point = make_float3(display_data.max_full_frame_luminance / 80.0f);
                use_aces = true;
            }
        } else {
            swap_chain.emplace(device.create_swapchain(
                stream,
                SwapchainOption{
                    .display = window->native_display(),
                    .window = window->native_handle(),
                    .size = make_uint2(resolution),
                    .wants_hdr = true,
                    .wants_vsync = false,
                    .back_buffer_count = 8,
                }));
        }
    }

    Image<float> ldr_image = device.create_image<float>(
        (!opts.offline && swap_chain.has_value()) ? swap_chain->backend_storage() : PixelStorage::BYTE4,
        resolution);
    double last_time = 0.0;
    uint64_t frame_count = 0u;
    Clock clock;

    auto sample_plan = luisa::ref::PathTracingSamplePassPlan{
        .total_spp = opts.spp == 0u ? luisa::ref::DEFAULT_PATH_TRACING_SPP : opts.spp,
        .max_spp_per_dispatch = max_spp_per_dispatch,
        .infinite = !opts.offline,
    };
    while (sample_plan.has_next(frame_count) && (opts.offline || !window->should_close())) {
        auto dispatch_spp = sample_plan.next_dispatch_spp(frame_count);
        int mode = 0;
        if (!opts.offline && swap_chain.has_value()) {
            if (swap_chain->backend_storage() == PixelStorage::R10G10B10A2) {
                mode = 1;
            } else if (swap_chain->backend_storage() == PixelStorage::HALF4) {
                mode = 2;
            }
        }
        cmd_list << raytracing_shader(framebuffer, seed_image, accel, resolution, dispatch_spp)
                        .dispatch(resolution)
                 << accumulate_shader(accum_image, framebuffer)
                        .dispatch(resolution);
        if (!opts.offline && swap_chain.has_value()) {
            cmd_list << hdr2ldr_shader(accum_image, ldr_image, scale, mode, white_point, use_aces).dispatch(resolution);
            stream << cmd_list.commit()
                   << swap_chain->present(ldr_image) << synchronize();
            window->poll_events();
        } else {
            stream << cmd_list.commit() << synchronize();
        }
        double dt = clock.toc() - last_time;
        LUISA_INFO("dt = {:.2f}ms ({:.2f} spp/s)", dt, dispatch_spp / dt * 1000);
        last_time = clock.toc();
        frame_count += dispatch_spp;
    }
    luisa::vector<std::array<uint8_t, 4u>> host_image(resolution.x * resolution.y);
    stream << hdr2ldr_shader(accum_image, ldr_image, scale, 0, white_point, false).dispatch(resolution)
           << ldr_image.copy_to(luisa::span{host_image})
           << synchronize();
    LUISA_INFO("FPS: {}", frame_count / clock.toc() * 1000);
    stbi_write_png("test_path_tracing_hdr.png", resolution.x, resolution.y, 4, host_image.data(), 0);
    if (opts.offline) {
        if (opts.compare_path) {
            auto result = luisa::ref::compare_with_reference_file(
                reinterpret_cast<const uint8_t *>(host_image.data()),
                resolution.x, resolution.y, 4,
                *opts.compare_path);
            LUISA_INFO("Reference comparison: {} ({})", result.passed ? "PASSED" : "FAILED", result.message);
            if (!result.passed) { return 1; }
        }
    }
    return 0;
}
