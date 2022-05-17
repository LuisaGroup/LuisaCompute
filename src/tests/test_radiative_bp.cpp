//
// Created by ChenXin on 2022/2/22.
//

#include <iostream>

#include <runtime/context.h>
#include <runtime/device.h>
#include <runtime/stream.h>
#include <runtime/event.h>
#include <dsl/sugar.h>
#include <rtx/accel.h>
#include <gui/window.h>
#include <gui/framerate.h>
#include <tests/cornell_box.h>
#include <stb/stb_image_write.h>
#include <stb/stb_image.h>

#include <dsl/printer.h>

#define TINYOBJLOADER_IMPLEMENTATION
#include <tests/tiny_obj_loader.h>

using namespace luisa;
using namespace luisa::compute;

struct Material {
    float3 albedo;
    float3 emission;
};

struct Onb {
    float3 tangent;
    float3 binormal;
    float3 normal;
};

// clang-format off
LUISA_STRUCT(Material, albedo, emission) {};
LUISA_STRUCT(Onb, tangent, binormal, normal) {
    [[nodiscard]] auto to_world(Expr<float3> v) const noexcept {
        return v.x * tangent + v.y * binormal + v.z * normal;
    }
};
// clang-format on

int main(int argc, char *argv[]) {
    log_level_info();

    Context context{argv[0]};
    if(argc <= 1){
        LUISA_INFO("Usage: {} <backend>. <backend>: cuda, dx, ispc, metal", argv[0]);
        exit(1);
    }
    auto device = context.create_device(argv[1]);
    Printer printer{device};

    std::filesystem::path output_dir{"frames"};
    std::filesystem::remove_all(output_dir);
    std::filesystem::create_directories(output_dir);

    // settings
    static constexpr uint differentiable_index = 4u;
    static constexpr uint max_depth = 10u;
    static constexpr float learning_rate = pi;
    static constexpr uint spp_per_dispatch = 256u;

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

    auto heap = device.create_bindless_array();
    auto stream = device.create_stream();
    stream << printer.reset();
    auto vertex_buffer = device.create_buffer<float3>(vertices.size());
    stream << vertex_buffer.copy_from(vertices.data());
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
        heap.emplace(index, triangle_buffer);
        stream << triangle_buffer.copy_from(indices.data())
               << mesh.build();
    }
    stream << heap.update();

    auto accel = device.create_accel();
    for (auto &&m : meshes) {
        accel.emplace_back(m, make_float4x4(1.0f));
    }
    stream << accel.build();

    std::vector<Material> materials;
    materials.reserve(accel.size());
    materials.emplace_back(Material{make_float3(0.725f, 0.71f, 0.68f), make_float3(0.0f)});// floor
    materials.emplace_back(Material{make_float3(0.725f, 0.71f, 0.68f), make_float3(0.0f)});// ceiling
    materials.emplace_back(Material{make_float3(0.725f, 0.71f, 0.68f), make_float3(0.0f)});// back wall
    materials.emplace_back(Material{make_float3(0.14f, 0.45f, 0.091f), make_float3(0.0f)});// right wall
    materials.emplace_back(Material{make_float3(0.63f, 0.065f, 0.05f), make_float3(0.0f)});// left wall
    materials.emplace_back(Material{make_float3(0.725f, 0.71f, 0.68f), make_float3(0.0f)});// short box
    materials.emplace_back(Material{make_float3(0.725f, 0.71f, 0.68f), make_float3(0.0f)});// tall box
    materials.emplace_back(Material{make_float3(0.0f), make_float3(17.0f, 12.0f, 4.0f)});  // light

    std::vector<Buffer<Material>> material_buffer;
    material_buffer.emplace_back(device.create_buffer<Material>(materials.size()));
    material_buffer.emplace_back(device.create_buffer<Material>(materials.size()));
    // right cbox
    stream << material_buffer[0].copy_from(materials.data())
           << synchronize();
    LUISA_INFO("target albedo = ({}, {}, {})",
               materials[4].albedo.x, materials[4].albedo.y, materials[4].albedo.z);
    // left wall green cbox
    materials[4] = materials[3];
    stream << material_buffer[1].copy_from(materials.data())
           << synchronize();
    LUISA_INFO("start with albedo = ({}, {}, {})",
               materials[4].albedo.x, materials[4].albedo.y, materials[4].albedo.z);

    Callable linear_to_srgb = [](Var<float3> x) noexcept {
        return clamp(select(1.055f * pow(x, 1.0f / 2.4f) - 0.055f,
                            12.92f * x,
                            x <= 0.00031308f),
                     0.0f, 1.0f);
    };

    Callable tea = [](UInt v0, UInt v1) noexcept {
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

    Callable lcg = [](UInt &state) noexcept {
        constexpr auto lcg_a = 1664525u;
        constexpr auto lcg_c = 1013904223u;
        state = lcg_a * state + lcg_c;
        return cast<float>(state & 0x00ffffffu) * (1.0f / static_cast<float>(0x01000000u));
    };

    Callable make_onb = [](const Float3 &normal) noexcept {
        auto binormal = normalize(ite(
            abs(normal.x) > abs(normal.z),
            make_float3(-normal.y, normal.x, 0.0f),
            make_float3(0.0f, -normal.z, normal.y)));
        auto tangent = normalize(cross(binormal, normal));
        return def<Onb>(tangent, binormal, normal);
    };

    Callable generate_ray = [](Float2 p) noexcept {
        static constexpr auto fov = radians(27.8f);
        static constexpr auto origin = make_float3(-0.01f, 0.995f, 5.0f);
        auto pixel = origin + make_float3(p * tan(0.5f * fov), -1.0f);
        auto direction = normalize(pixel - origin);
        return make_ray(origin, direction);
    };

    Callable cosine_sample_hemisphere = [](Float2 u) noexcept {
        auto r = sqrt(u.x);
        auto phi = 2.0f * constants::pi * u.y;
        return make_float3(r * cos(phi), r * sin(phi), sqrt(1.0f - u.x));
    };

    Callable balanced_heuristic = [](Float pdf_a, Float pdf_b) noexcept {
        return pdf_a / max(pdf_a + pdf_b, 1e-4f);
    };

    auto raytracing_kernel_template = [&](size_t material_buffer_id, ImageFloat &image, ImageUInt &seed_image, AccelVar &accel, UInt2 resolution) noexcept {
        set_block_size(8u, 8u, 1u);
        auto coord = dispatch_id().xy();
        auto frame_size = min(resolution.x, resolution.y).cast<float>();
        auto state = seed_image.read(coord).x;
        auto rx = lcg(state);
        auto ry = lcg(state);
        auto pixel = (make_float2(coord) + make_float2(rx, ry)) / frame_size * 2.0f - 1.0f;
        auto ray = generate_ray(pixel * make_float2(1.0f, -1.0f));
        auto radiance = def(make_float3(0.0f));
        auto beta = def(make_float3(1.0f));
        auto pdf_bsdf = def(0.0f);
        constexpr auto light_position = make_float3(-0.24f, 1.98f, 0.16f);
        constexpr auto light_u = make_float3(-0.24f, 1.98f, -0.22f) - light_position;
        constexpr auto light_v = make_float3(0.23f, 1.98f, 0.16f) - light_position;
        constexpr auto light_emission = make_float3(17.0f, 12.0f, 4.0f);
        auto light_area = length(cross(light_u, light_v));
        auto light_normal = normalize(cross(light_u, light_v));

        $for(depth, max_depth) {
            // trace
            auto hit = accel.trace_closest(ray);
            $if(hit->miss()) { $break; };
            auto triangle = heap.buffer<Triangle>(hit.inst).read(hit.prim);
            auto p0 = vertex_buffer.read(triangle.i0);
            auto p1 = vertex_buffer.read(triangle.i1);
            auto p2 = vertex_buffer.read(triangle.i2);
            auto p = hit->interpolate(p0, p1, p2);
            auto n = normalize(cross(p1 - p0, p2 - p0));
            auto cos_wi = dot(-ray->direction(), n);
            $if(cos_wi < 1e-4f) { $break; };

            auto material = material_buffer[material_buffer_id].read(hit.inst);

            // hit light
            $if(hit.inst == static_cast<uint>(meshes.size() - 1u)) {
                $if(depth == 0u) {
                    radiance += light_emission;
                }
                $else {
                    auto pdf_light = length_squared(p - ray->origin()) / (light_area * cos_wi);
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
            auto shadow_ray = make_ray(pp, wi_light, 0.f, d_light - 1e-3f);
            auto occluded = accel.trace_any(shadow_ray);
            auto cos_wi_light = dot(wi_light, n);
            auto cos_light = -dot(light_normal, wi_light);
            $if(!occluded & cos_wi_light > 1e-4f & cos_light > 1e-4f) {
                auto pdf_light = (d_light * d_light) / (light_area * cos_light);
                auto pdf_bsdf = cos_wi_light * inv_pi;
                auto mis_weight = balanced_heuristic(pdf_light, pdf_bsdf);
                auto bsdf = material.albedo * inv_pi * cos_wi_light;
                radiance += beta * bsdf * mis_weight * light_emission / max(pdf_light, 1e-4f);
            };

            // sample BSDF
            auto onb = make_onb(n);
            auto ux = lcg(state);
            auto uy = lcg(state);
            auto new_direction = onb->to_world(cosine_sample_hemisphere(make_float2(ux, uy)));
            ray = make_ray(pp, new_direction);
            beta *= material.albedo;
            pdf_bsdf = cos_wi * inv_pi;

            // rr
            auto l = dot(make_float3(0.212671f, 0.715160f, 0.072169f), beta);
            $if(l == 0.0f) { $break; };
            auto q = max(l, 0.05f);
            auto r = lcg(state);
            $if(r >= q) { $break; };
            beta *= 1.0f / q;
        };
        seed_image.write(coord, make_uint4(state));
        $if(any(isnan(radiance))) { radiance = make_float3(0.0f); };
        image.write(dispatch_id().xy(), make_float4(clamp(radiance, 0.0f, 30.0f), 1.0f));
    };

    Kernel2D radiative_kernel = [&](ImageFloat rendered_image, ImageFloat target_image, ImageFloat grad_image,
                                    ImageUInt seed_image, AccelVar accel, UInt2 resolution) noexcept {
        set_block_size(8u, 8u, 1u);
        auto coord = dispatch_id().xy();
        auto frame_size = min(resolution.x, resolution.y).cast<float>();
        auto state = seed_image.read(coord).x;
        auto rx = lcg(state);
        auto ry = lcg(state);
        auto pixel = (make_float2(coord) + make_float2(rx, ry)) / frame_size * 2.0f - 1.0f;
        auto ray = generate_ray(pixel * make_float2(1.0f, -1.0f));
        auto grad = def(make_float3(0.0f));
        auto beta = def(make_float3(1.0f));
        auto pdf_bsdf = def(0.0f);
        constexpr auto light_position = make_float3(-0.24f, 1.98f, 0.16f);
        constexpr auto light_u = make_float3(-0.24f, 1.98f, -0.22f) - light_position;
        constexpr auto light_v = make_float3(0.23f, 1.98f, 0.16f) - light_position;
        constexpr auto light_emission = make_float3(17.0f, 12.0f, 4.0f);
        auto light_area = length(cross(light_u, light_v));
        auto light_normal = normalize(cross(light_u, light_v));

        // calculate L2 loss
        auto rendered = rendered_image.read(coord);
        auto target = target_image.read(coord);
        beta *= 2.0f * (rendered - target).xyz();

        $for(depth, max_depth) {
            // trace
            auto hit = accel.trace_closest(ray);
            $if(hit->miss()) { $break; };
            auto triangle = heap.buffer<Triangle>(hit.inst).read(hit.prim);
            auto p0 = vertex_buffer.read(triangle.i0);
            auto p1 = vertex_buffer.read(triangle.i1);
            auto p2 = vertex_buffer.read(triangle.i2);
            auto p = hit->interpolate(p0, p1, p2);
            auto n = normalize(cross(p1 - p0, p2 - p0));
            auto cos_wi = dot(-ray->direction(), n);
            $if(cos_wi < 1e-4f) { $break; };

            auto material = material_buffer[1u].read(hit.inst);

            // hit light
            $if(hit.inst == static_cast<uint>(meshes.size() - 1u)) {
                $break;
            };

            $if(hit.inst == differentiable_index) {
                auto bsdf_diff = make_float3(1.0f) * inv_pi * cos_wi;
                auto Li = make_float3(1.0f);
                grad += beta * Li * bsdf_diff;
            };

            // sample BSDF
            auto onb = make_onb(n);
            auto ux = lcg(state);
            auto uy = lcg(state);
            auto new_direction = onb->to_world(cosine_sample_hemisphere(make_float2(ux, uy)));
            auto pp = offset_ray_origin(p, n);
            ray = make_ray(pp, new_direction);
            pdf_bsdf = cos_wi * inv_pi;
            beta *= material.albedo;

            // rr
            auto l = dot(make_float3(0.212671f, 0.715160f, 0.072169f), beta);
            $if(l == 0.0f) { $break; };
            auto q = max(l, 0.05f);
            auto r = lcg(state);
            $if(r >= q) { $break; };
            beta *= 1.0f / q;
        };
        seed_image.write(coord, make_uint4(state));
        $if(any(isnan(grad))) { grad = make_float3(0.0f); };
        grad_image.write(dispatch_id().xy(), make_float4(grad, 1.0f));
    };

    Kernel2D accumulate_kernel = [&](ImageFloat accum_image, ImageFloat curr_image) noexcept {
        auto p = dispatch_id().xy();
        auto accum = accum_image.read(p);
        auto curr = curr_image.read(p).xyz();
        auto t = 1.0f / (accum.w + 1.0f);
        accum_image.write(p, make_float4(lerp(accum.xyz(), curr, t), accum.w + 1.0f));
    };

    Callable aces_tonemapping = [](Float3 x) noexcept {
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

    Kernel2D copy_kernel = [](ImageUInt from, ImageUInt to) noexcept {
        auto coord = dispatch_id().xy();
        to.write(coord, from.read(coord));
    };

    Kernel2D hdr2ldr_kernel = [&](ImageFloat hdr_image, ImageFloat ldr_image, Float scale) noexcept {
        auto coord = dispatch_id().xy();
        auto hdr = hdr_image.read(coord);
        auto ldr = linear_to_srgb(aces_tonemapping(hdr.xyz() * scale));
        ldr_image.write(coord, make_float4(ldr, 1.0f));
    };

    auto clear_shader = device.compile(clear_kernel);
    auto hdr2ldr_shader = device.compile(hdr2ldr_kernel);
    auto accumulate_shader = device.compile(accumulate_kernel);
    auto raytracing_shader_0 = device.compile<2>(
        [&](ImageFloat image, ImageUInt seed_image, AccelVar accel, UInt2 resolution) noexcept {
            raytracing_kernel_template(0, image, seed_image, accel, resolution);
        });
    auto raytracing_shader_1 = device.compile<2>(
        [&](ImageFloat image, ImageUInt seed_image, AccelVar accel, UInt2 resolution) noexcept {
            raytracing_kernel_template(1, image, seed_image, accel, resolution);
        });
    auto radiative_shader = device.compile(radiative_kernel);
    auto make_sampler_shader = device.compile(make_sampler_kernel);
    auto copy_shader = device.compile(copy_kernel);

    static constexpr auto resolution = make_uint2(1024u);
    auto seed_image = device.create_image<uint>(PixelStorage::INT1, resolution);
    auto seed_bank_image = device.create_image<uint>(PixelStorage::INT1, resolution);
    auto rendered_image = device.create_image<float>(PixelStorage::FLOAT4, resolution);
    auto buffer_image = device.create_image<float>(PixelStorage::FLOAT4, resolution);
    auto ldr_image = device.create_image<float>(PixelStorage::BYTE4, resolution);
    std::vector<std::array<uint8_t, 4u>> host_image(resolution.x * resolution.y);

    auto target_image = device.create_image<float>(PixelStorage::FLOAT4, resolution);
    auto grad_image = device.create_image<float>(PixelStorage::FLOAT4, resolution);
    std::vector<float> grad_vec(4u * resolution.x * resolution.y);

    // pipeline
    Clock clock;
    stream << make_sampler_shader(seed_image).dispatch(resolution) << synchronize();

    // render target image
    stream << clear_shader(target_image).dispatch(resolution);
    for (auto spp = 0u; spp < spp_per_dispatch; ++spp) {
        stream << raytracing_shader_0(buffer_image, seed_image, accel, resolution)
                      .dispatch(resolution)
               << accumulate_shader(target_image, buffer_image).dispatch(resolution)
               << synchronize();
    }

    Framerate framerate;
    Window window{"Display", resolution};
    window.set_key_callback([&](int key, int action) noexcept {
        if (action == GLFW_PRESS && key == GLFW_KEY_ESCAPE) {
            window.set_should_close();
        }
    });
    auto frame_count = 0u;
    auto output_count = 0u;
    window.run([&] {
        auto command_buffer = stream.command_buffer();

        command_buffer << clear_shader(rendered_image).dispatch(resolution)
                       << clear_shader(grad_image).dispatch(resolution)
                       << copy_shader(seed_image, seed_bank_image).dispatch(resolution);

        // render
        for (auto spp = 0u; spp < spp_per_dispatch; ++spp) {
            command_buffer << raytracing_shader_1(buffer_image, seed_image,
                                                  accel, resolution)
                                  .dispatch(resolution)
                           << accumulate_shader(rendered_image, buffer_image).dispatch(resolution)
                           << synchronize();
        }

        // bp
        for (auto spp = 0u; spp < spp_per_dispatch; ++spp) {
            command_buffer << radiative_shader(rendered_image, target_image, buffer_image,
                                               seed_bank_image, accel, resolution)
                                  .dispatch(resolution)
                           << accumulate_shader(grad_image, buffer_image).dispatch(resolution)
                           << synchronize();
        }
        command_buffer << grad_image.copy_to(grad_vec.data());
        // adjust material params
        float3 grad = make_float3(0.f);
        for (auto x = 0u; x < resolution.x; ++x)
            for (auto y = 0u; y < resolution.y; ++y) {
                auto index = (x * resolution.x + y) * 4u;
                grad += {grad_vec[index + 0u],
                         grad_vec[index + 1u],
                         grad_vec[index + 2u]};
            }
        grad /= float(resolution.x * resolution.y);
        auto &albedo = materials[differentiable_index].albedo;
        albedo -= learning_rate * grad;
        command_buffer << material_buffer[1].copy_from(materials.data())
                       << printer.retrieve();

        LUISA_INFO("grad = ({}, {}, {}), "
                   "albedo = ({}, {}, {})",
                   grad.x, grad.y, grad.z,
                   albedo.x, albedo.y, albedo.z);

        // display
        command_buffer << hdr2ldr_shader(rendered_image, ldr_image, 1.0f).dispatch(resolution)
                       << ldr_image.copy_to(host_image.data())
                       << commit();
        stream << synchronize();
        auto file_name = output_dir / luisa::format("{:06}.png", output_count++);
        stbi_write_png(file_name.string().c_str(), resolution.x, resolution.y, 4, host_image.data(), 0);
        window.set_background(host_image.data(), resolution);
        framerate.record(spp_per_dispatch);
        frame_count += spp_per_dispatch;
        ImGui::Begin("Console", nullptr, ImGuiWindowFlags_AlwaysAutoResize);
        ImGui::Text("Frames: %u", frame_count);
        ImGui::Text("FPS: %.1f", framerate.report());
        ImGui::End();
    });

    stbi_write_png("test_radiative_bp.png", resolution.x, resolution.y, 4, host_image.data(), 0);
}
