#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <memory>
#include <optional>
#include <string_view>

#include <luisa/core/clock.h>
#include <luisa/core/logging.h>
#include <luisa/runtime/context.h>
#include <luisa/runtime/device.h>
#include <luisa/runtime/stream.h>
#include <luisa/runtime/event.h>
#include <luisa/runtime/swapchain.h>
#include <luisa/runtime/rtx/mesh.h>
#include <luisa/runtime/rtx/accel.h>
#include <luisa/dsl/sugar.h>
#include <luisa/ir/ast2ir.h>
#if LUISA_RENDERING_USE_XIR_TO_AST
#include <luisa/xir/translators/ast2xir.h>
#include <luisa/xir/translators/xir2ast.h>
#endif
#include <stb/stb_image_write.h>
#include "common/reference_compare.h"
#include <luisa/gui/window.h>
#include "cornell_box.h"

#define TINYOBJLOADER_IMPLEMENTATION
#include "tiny_obj_loader.h"

using namespace luisa;
using namespace luisa::compute;

#if LUISA_RENDERING_USE_XIR_TO_AST
namespace {

[[nodiscard]] auto build_xir_to_ast_kernel(const Function &function) noexcept {
    auto module = xir::ast_to_xir_translate(function, {});
    auto config = xir::XIR2ASTConfig{.bound_arguments = function.bound_arguments()};
    for (auto *f : module->function_list()) {
        if (f->derived_function_tag() == xir::DerivedFunctionTag::KERNEL) {
            return xir::xir_to_ast_translate(*static_cast<xir::FunctionDefinition *>(f), config);
        }
    }
    LUISA_ERROR_WITH_LOCATION("XIR-to-AST translation did not produce a kernel definition.");
}

}// namespace
#endif

struct Onb {
    float3 tangent;
    float3 binormal;
    float3 normal;
};

// clang-format off
LUISA_STRUCT(Onb, tangent, binormal, normal) {
    [[nodiscard]] auto to_world(Expr<float3> v) const noexcept {
        return v.x * tangent + v.y * binormal + v.z * normal;
    }
};
// clang-format on

int main(int argc, char *argv[]) {

    log_level_verbose();

    Context context{argv[0]};
    if (argc <= 1) {
        LUISA_INFO("Usage: {} <backend> [--offline] [-c <reference.png>]. <backend>: cuda, dx, cpu, metal", argv[0]);
        exit(1);
    }
    bool force_offline = false;
    uint user_spp = 0u;
    std::optional<std::filesystem::path> compare_path;
    for (int i = 2; i < argc; i++) {
        if (std::string_view{argv[i]} == "--offline") {
            force_offline = true;
        } else if ((std::string_view{argv[i]} == "--compare" || std::string_view{argv[i]} == "-c") && i + 1 < argc) {
            compare_path = std::filesystem::path{argv[++i]};
            force_offline = true;
            force_offline = true;
        } else if (std::string_view{argv[i]} == "--spp" && i + 1 < argc) {
            user_spp = static_cast<uint>(std::atoi(argv[++i]));
        }
    }
    Device device = context.create_device(argv[1]);

    // load the Cornell Box scene
    tinyobj::ObjReaderConfig obj_reader_config;
    obj_reader_config.triangulate = true;
    obj_reader_config.vertex_color = false;
    tinyobj::ObjReader obj_reader;
    if (!obj_reader.ParseFromString(obj_string, "", obj_reader_config)) {
        std::string_view error_message = "unknown error.";
        if (auto &&e = obj_reader.Error(); !e.empty()) { error_message = e; }
        LUISA_ERROR_WITH_LOCATION("Failed to load OBJ file: {}", error_message);
    }
    if (auto &&e = obj_reader.Warning(); !e.empty()) {
        LUISA_WARNING_WITH_LOCATION("{}", e);
    }

    auto &&p = obj_reader.GetAttrib().vertices;
    std::vector<float3> vertices;
    vertices.reserve(p.size() / 3u);
    for (auto i = 0u; i < p.size(); i += 3u) {
        vertices.emplace_back(float3{
            p[i + 0u],
            p[i + 1u],
            p[i + 2u]});
    }
    LUISA_INFO(
        "Loaded mesh with {} shape(s) and {} vertices.",
        obj_reader.GetShapes().size(), vertices.size());

    BindlessArray heap = device.create_bindless_array();
    Stream stream = device.create_stream(force_offline ? StreamTag::COMPUTE : StreamTag::GRAPHICS);
    auto vertex_buffer = device.create_buffer<float3>(vertices.size());
    stream << vertex_buffer.copy_from(luisa::span{vertices});
    std::vector<Mesh> meshes;
    std::vector<Buffer<Triangle>> triangle_buffers;
    for (auto &&shape : obj_reader.GetShapes()) {
        auto index = static_cast<uint>(meshes.size());
        auto &&t = shape.mesh.indices;
        auto triangle_count = t.size() / 3u;
        LUISA_INFO(
            "Processing shape '{}' at index {} with {} triangle(s).",
            shape.name, index, triangle_count);
        std::vector<uint> indices;
        indices.reserve(t.size());
        for (auto i : t) { indices.emplace_back(i.vertex_index); }
        auto &&triangle_buffer = triangle_buffers.emplace_back(device.create_buffer<Triangle>(triangle_count));
        auto &&mesh = meshes.emplace_back(device.create_mesh(vertex_buffer, triangle_buffer));
        heap.emplace_on_update(index, triangle_buffer);
        stream << triangle_buffer.copy_from(luisa::span{indices})
               << mesh.build();
    }

    auto accel = device.create_accel({});
    for (auto &&m : meshes) {
        accel.emplace_back(m, make_float4x4(1.0f));
    }
    stream << heap.update()
           << accel.build()
           << synchronize();

    float3 materials_array[] = {
        make_float3(0.725f, 0.71f, 0.68f),
        make_float3(0.725f, 0.71f, 0.68f),
        make_float3(0.725f, 0.71f, 0.68f),
        make_float3(0.14f, 0.45f, 0.091f),
        make_float3(0.63f, 0.065f, 0.05f),
        make_float3(0.725f, 0.71f, 0.68f),
        make_float3(0.725f, 0.71f, 0.68f),
        make_float3(0.0f),
    };
    auto materials = device.create_buffer<float3>(8);
    stream << materials.copy_from(luisa::span{materials_array, std::size(materials_array)});

    auto linear_to_srgb = [](Var<float3> x) noexcept {
        return clamp(select(1.055f * pow(x, 1.0f / 2.4f) - 0.055f,
                            12.92f * x,
                            x <= 0.00031308f),
                     0.0f, 1.0f);
    };

    auto tea = [](UInt v0, UInt v1) noexcept {
        auto s0 = def(0u);
        for (auto n = 0u; n < 4u; n++) {
            s0 += 0x9e3779b9u;
            v0 += ((v1 << 4) + 0xa341316cu) ^ (v1 + s0) ^ ((v1 >> 5u) + 0xc8013ea4u);
            v1 += ((v0 << 4) + 0xad90777du) ^ (v0 + s0) ^ ((v0 >> 5u) + 0x7e95761eu);
        }
        return v0;
    };

    Kernel2D make_sampler_kernel = [&](ImageUInt seed_image) noexcept {
        auto p = dispatch_id().xy();
        auto state = tea(p.x, p.y);
        seed_image.write(p, make_uint4(state));
    };

    auto lcg = [](UInt &state) noexcept {
        constexpr auto lcg_a = 1664525u;
        constexpr auto lcg_c = 1013904223u;
        state = lcg_a * state + lcg_c;
        return cast<float>(state & 0x00ffffffu) *
               (1.0f / static_cast<float>(0x01000000u));
    };

    auto make_onb = [](const Float3 &normal) noexcept {
        auto binormal = normalize(ite(
            abs(normal.x) > abs(normal.z),
            make_float3(-normal.y, normal.x, 0.0f),
            make_float3(0.0f, -normal.z, normal.y)));
        auto tangent = normalize(cross(binormal, normal));
        return def<Onb>(tangent, binormal, normal);
    };

    auto generate_ray = [](Float2 p) noexcept {
        static constexpr auto fov = radians(27.8f);
        static constexpr auto origin = make_float3(-0.01f, 0.995f, 5.0f);
        auto pixel = origin + make_float3(p * tan(0.5f * fov), -1.0f);
        auto direction = normalize(pixel - origin);
        return make_ray(origin, direction);
    };

    auto cosine_sample_hemisphere = [](Float2 u) noexcept {
        auto r = sqrt(u.x);
        auto phi = 2.0f * constants::pi * u.y;
        return make_float3(r * cos(phi), r * sin(phi), sqrt(1.0f - u.x));
    };

    auto balanced_heuristic = [](Float pdf_a, Float pdf_b) noexcept {
        return pdf_a / max(pdf_a + pdf_b, 1e-4f);
    };

    auto spp_per_dispatch = device.backend_name() == "metal" || device.backend_name() == "cpu" || device.backend_name() == "fallback" ? 1u : 64u;

    Kernel2D raytracing_kernel = [&](ImageFloat image, ImageUInt seed_image, AccelVar accel, UInt2 resolution) noexcept {
        set_block_size(16u, 16u, 1u);
        auto &&heap_ref = heap;
        auto &&vertex_buffer_ref = vertex_buffer;
        auto &&materials_ref = materials;
        auto coord = dispatch_id().xy();
        auto frame_size = min(resolution.x, resolution.y).cast<float>();
        auto state = seed_image.read(coord).x;
        auto rx = lcg(state);
        auto ry = lcg(state);
        auto pixel = (make_float2(coord) + make_float2(rx, ry)) / frame_size * 2.0f - 1.0f;
        auto radiance = def(make_float3(0.0f));
        $for (i, spp_per_dispatch) {
            auto ray = generate_ray(pixel * make_float2(1.0f, -1.0f));
            auto beta = def(make_float3(1.0f));
            auto pdf_bsdf = def(0.0f);
            constexpr auto light_position = make_float3(-0.24f, 1.98f, 0.16f);
            constexpr auto light_u = make_float3(-0.24f, 1.98f, -0.22f) - light_position;
            constexpr auto light_v = make_float3(0.23f, 1.98f, 0.16f) - light_position;
            constexpr auto light_emission = make_float3(17.0f, 12.0f, 4.0f);
            auto light_area = length(cross(light_u, light_v));
            auto light_normal = normalize(cross(light_u, light_v));
            $for (depth, 10u) {

                // trace
                auto hit = accel.intersect(ray, {});
                reorder_shader_execution();
                $if (hit->miss()) { $break; };
                auto triangle = heap_ref->buffer<Triangle>(hit.inst).read(hit.prim);
                auto p0 = vertex_buffer_ref->read(triangle.i0);
                auto p1 = vertex_buffer_ref->read(triangle.i1);
                auto p2 = vertex_buffer_ref->read(triangle.i2);
                auto p = triangle_interpolate(hit.bary, p0, p1, p2);
                auto n = normalize(cross(p1 - p0, p2 - p0));
                auto cos_wo = dot(-ray->direction(), n);
                $if (cos_wo < 1e-4f) { $break; };
                auto albedo = materials_ref->read(hit.inst);

                // hit light
                $if (hit.inst == static_cast<uint>(meshes.size() - 1u)) {
                    $if (depth == 0u) {
                        radiance += light_emission;
                    }
                    $else {
                        auto pdf_light = length_squared(p - ray->origin()) / (light_area * cos_wo);
                        auto mis_weight = balanced_heuristic(pdf_bsdf, pdf_light);
                        radiance += mis_weight * beta * light_emission;
                    };
                    $break;
                };

                // sample light
                auto ux_light = lcg(state);
                auto uy_light = lcg(state);
                auto p_light = light_position + ux_light * light_u + uy_light * light_v;
                auto pp = offset_ray_origin(p, n);
                auto pp_light = offset_ray_origin(p_light, light_normal);
                auto d_light = distance(pp, pp_light);
                auto wi_light = normalize(pp_light - pp);
                auto shadow_ray = make_ray(offset_ray_origin(pp, n), wi_light, 0.f, d_light);
                auto occluded = accel.intersect_any(shadow_ray, {});
                auto cos_wi_light = dot(wi_light, n);
                auto cos_light = -dot(light_normal, wi_light);
                $if (!occluded & cos_wi_light > 1e-4f & cos_light > 1e-4f) {
                    auto pdf_light = (d_light * d_light) / (light_area * cos_light);
                    auto pdf_bsdf = cos_wi_light * inv_pi;
                    auto mis_weight = balanced_heuristic(pdf_light, pdf_bsdf);
                    auto bsdf = albedo * inv_pi * cos_wi_light;
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
                auto l = dot(make_float3(0.212671f, 0.715160f, 0.072169f), beta);
                $if (l == 0.0f) { $break; };
                auto q = max(l, 0.05f);
                auto r = lcg(state);
                $if (r >= q) { $break; };
                beta *= 1.0f / q;
            };
        };
        radiance /= static_cast<float>(spp_per_dispatch);
        seed_image.write(coord, make_uint4(state));
        $if (any(dsl::isnan(radiance))) { radiance = make_float3(0.0f); };
        image.write(dispatch_id().xy(), make_float4(clamp(radiance, 0.0f, 30.0f), 1.0f));
    };

    Kernel2D accumulate_kernel = [&](ImageFloat accum_image, ImageFloat curr_image) noexcept {
        auto p = dispatch_id().xy();
        auto accum = accum_image.read(p);
        auto curr = curr_image.read(p).xyz();
        accum_image.write(p, accum + make_float4(curr, 1.f));
    };

    auto aces_tonemapping = [](Float3 x) noexcept {
        static constexpr auto a = 2.51f;
        static constexpr auto b = 0.03f;
        static constexpr auto c = 2.43f;
        static constexpr auto d = 0.59f;
        static constexpr auto e = 0.14f;
        return clamp((x * (a * x + b)) / (x * (c * x + d) + e), 0.0f, 1.0f);
    };

    Kernel2D clear_kernel = [](ImageFloat image) noexcept {
        image.write(dispatch_id().xy(), make_float4(0.0f));
    };

    Kernel2D hdr2ldr_kernel = [&](ImageFloat hdr_image, ImageFloat ldr_image, Float scale) noexcept {
        //        Shared<float> s1{13u};
        //        Shared<float> s2{1024u};
        //        s2[thread_x()] = 1.f;
        //        sync_block();
        auto coord = dispatch_id().xy();
        auto hdr = hdr_image.read(coord);
        auto ldr = linear_to_srgb(clamp(hdr.xyz() / hdr.w * scale, 0.0f, 1.0f));
        ldr_image.write(coord, make_float4(ldr, 1.0f));
    };

    auto clear_shader = device.compile(clear_kernel);
    auto hdr2ldr_shader = device.compile(hdr2ldr_kernel);
    auto accumulate_shader = device.compile(accumulate_kernel);
#if LUISA_RENDERING_USE_XIR_TO_AST
    auto raytracing_ast = build_xir_to_ast_kernel(raytracing_kernel.function()->function());
    auto make_sampler_ast = build_xir_to_ast_kernel(make_sampler_kernel.function()->function());
    auto raytracing_shader = device.compile(Kernel<2, Image<float>, Image<uint>, Accel, uint2>{raytracing_ast});
    auto make_sampler_shader = device.compile(Kernel<2, Image<uint>>{make_sampler_ast});
#else
    auto raytracing_ir = AST2IR::build_kernel(raytracing_kernel.function()->function());
    auto make_sampler_ir = AST2IR::build_kernel(make_sampler_kernel.function()->function());
    auto raytracing_shader = device.compile<2, Image<float>, Image<uint>, Accel, uint2>(raytracing_ir->get());
    auto make_sampler_shader = device.compile<2, Image<uint>>(make_sampler_ir->get());
#endif

    static constexpr auto resolution = make_uint2(1024u);
    auto framebuffer = device.create_image<float>(PixelStorage::HALF4, resolution);
    auto accum_image = device.create_image<float>(PixelStorage::FLOAT4, resolution);
    std::vector<std::array<uint8_t, 4u>> host_image(resolution.x * resolution.y);
    CommandList cmd_list;
    auto seed_image = device.create_image<uint>(PixelStorage::INT1, resolution);
    cmd_list << clear_shader(accum_image).dispatch(resolution)
             << make_sampler_shader(seed_image).dispatch(resolution);

    std::unique_ptr<Window> window;
    std::optional<Swapchain> swap_chain;
    if (!force_offline) {
        window = std::make_unique<Window>("path tracing", resolution);
        swap_chain.emplace(device.create_swapchain(
            stream,
            SwapchainOption{
                .display = window->native_display(),
                .window = window->native_handle(),
                .size = resolution,
                .wants_hdr = false,
                .wants_vsync = false,
                .back_buffer_count = 2,
            }));
    }
    auto ldr_image = device.create_image<float>(
        (!force_offline && swap_chain.has_value()) ? swap_chain->backend_storage() : PixelStorage::BYTE4,
        resolution);
    auto last_time = 0.0;
    auto frame_count = 0u;
    Clock clock;
    bool infinite_render = !force_offline;
    uint total_spp = force_offline ? (user_spp == 0u ? 1024u : user_spp) : 0u;

    while (infinite_render || frame_count < total_spp) {
        cmd_list << raytracing_shader(framebuffer, seed_image, accel, resolution)
                        .dispatch(resolution)
                 << accumulate_shader(accum_image, framebuffer)
                        .dispatch(resolution);
        if (!force_offline && swap_chain.has_value()) {
            cmd_list << hdr2ldr_shader(accum_image, ldr_image, 2.0f).dispatch(resolution);
            stream << cmd_list.commit()
                   << swap_chain->present(ldr_image);
            if (window->should_close()) { break; }
            window->poll_events();
        } else {
            stream << cmd_list.commit();
        }
        auto dt = clock.toc() - last_time;
        last_time = clock.toc();
        frame_count += spp_per_dispatch;
        LUISA_INFO("time: {} ms", dt);
    }
    stream << hdr2ldr_shader(accum_image, ldr_image, 2.0f).dispatch(resolution)
            << ldr_image.copy_to(luisa::span{host_image})
            << synchronize();

    LUISA_INFO("FPS: {}", frame_count / clock.toc() * 1000);
    stbi_write_png("test_path_tracing.png", resolution.x, resolution.y, 4, host_image.data(), 0);
    if (force_offline) {
        if (compare_path) {
            auto result = luisa::ref::compare_with_reference_file(
                reinterpret_cast<const uint8_t *>(host_image.data()),
                resolution.x, resolution.y, 4,
                *compare_path);
            LUISA_INFO("Reference comparison: {} ({})", result.passed ? "PASSED" : "FAILED", result.message);
            if (!result.passed) { return 1; }
        }
    }
    return 0;
}
