//
// Created by Mike Smith on 2021/6/23.
//

#include <iostream>

#include <runtime/context.h>
#include <runtime/device.h>
#include <runtime/stream.h>
#include <runtime/event.h>
#include <dsl/syntax.h>
#include <rtx/accel.h>
#include <tests/fake_device.h>

#include <opencv2/opencv.hpp>

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

LUISA_STRUCT(Material, albedo, emission)
LUISA_STRUCT(Onb, tangent, binormal, normal);

int main(int argc, char *argv[]) {

    log_level_verbose();

    Context context{argv[0]};

#if defined(LUISA_BACKEND_METAL_ENABLED)
    auto device = context.create_device("metal", 1u);
#elif defined(LUISA_BACKEND_DX_ENABLED)
    auto device = context.create_device("dx");
#else
    auto device = FakeDevice::create(context);
#endif

    // load the Cornell Box scene
    tinyobj::ObjReaderConfig obj_reader_config;
    obj_reader_config.triangulate = true;
    obj_reader_config.vertex_color = false;
    tinyobj::ObjReader obj_reader;
    if (!obj_reader.ParseFromFile("src/tests/cornell-box.obj", obj_reader_config)) {
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

    auto heap = device.create_heap();
    auto stream = device.create_stream();
    auto vertex_buffer = device.create_buffer<float3>(vertices.size());
    stream << vertex_buffer.copy_from(vertices.data());
    std::vector<Mesh> meshes;
    for (auto &&shape : obj_reader.GetShapes()) {
        auto index = static_cast<uint>(meshes.size());
        auto &&t = shape.mesh.indices;
        auto triangle_count = t.size() / 3u;
        LUISA_INFO(
            "Processing shape '{}' at index {} with {} triangle(s).",
            shape.name, index, triangle_count);
        auto &&mesh = meshes.emplace_back(device.create_mesh());
        std::vector<uint> indices;
        indices.reserve(t.size());
        for (auto i : t) { indices.emplace_back(i.vertex_index); }
        auto triangle_buffer = heap.create_buffer<Triangle>(index, triangle_count);
        stream << triangle_buffer.copy_from(indices.data())
               << commit()// buffers from heap are not hazard-tracked
               << mesh.build(AccelBuildHint::FAST_TRACE, vertex_buffer, triangle_buffer);
    }

    std::vector<uint64_t> instances;
    std::vector<float4x4> transforms;
    for (auto &&m : meshes) {
        instances.emplace_back(m.handle());
        transforms.emplace_back(make_float4x4(1.0f));
    }
    auto accel = device.create_accel();
    stream << accel.build(AccelBuildHint::FAST_TRACE, instances, transforms);

    std::vector<Material> materials;
    materials.reserve(instances.size());
    materials.emplace_back(Material{make_float3(0.725f, 0.71f, 0.68f), make_float3(0.0f)});// floor
    materials.emplace_back(Material{make_float3(0.725f, 0.71f, 0.68f), make_float3(0.0f)});// ceiling
    materials.emplace_back(Material{make_float3(0.725f, 0.71f, 0.68f), make_float3(0.0f)});// back wall
    materials.emplace_back(Material{make_float3(0.14f, 0.45f, 0.091f), make_float3(0.0f)});// right wall
    materials.emplace_back(Material{make_float3(0.63f, 0.065f, 0.05f), make_float3(0.0f)});// left wall
    materials.emplace_back(Material{make_float3(0.725f, 0.71f, 0.68f), make_float3(0.0f)});// short box
    materials.emplace_back(Material{make_float3(0.725f, 0.71f, 0.68f), make_float3(0.0f)});// tall box
    materials.emplace_back(Material{make_float3(0.0f), make_float3(17.0f, 12.0f, 4.0f)});  // light
    auto material_buffer = device.create_buffer<Material>(materials.size());
    stream << material_buffer.copy_from(materials.data());

    Callable linear_to_srgb = [](Var<float3> x) noexcept {
        return clamp(select(1.055f * pow(x, 1.0f / 2.4f) - 0.055f,
                            12.92f * x,
                            x <= 0.00031308f),
                     0.0f, 1.0f);
    };

    Callable tea = [](UInt v0, UInt v1) noexcept {
        Var s0 = 0u;
        for (auto n = 0u; n < 4u; n++) {
            s0 += 0x9e3779b9u;
            v0 += ((v1 << 4) + 0xa341316cu) ^ (v1 + s0) ^ ((v1 >> 5u) + 0xc8013ea4u);
            v1 += ((v0 << 4) + 0xad90777du) ^ (v0 + s0) ^ ((v0 >> 5u) + 0x7e95761eu);
        }
        return v0;
    };

    Kernel2D make_sampler_kernel = [&](ImageUInt state_image) noexcept {
        Var p = dispatch_id().xy();
        Var state = tea(p.x, p.y);
        state_image.write(p, make_uint4(state));
    };

    auto lcg = [](compute::detail::Expr<uint> state) noexcept {
        constexpr auto lcg_a = 1664525u;
        constexpr auto lcg_c = 1013904223u;
        state = lcg_a * state + lcg_c;
        return cast<float>(state & 0x00ffffffu) * (1.0f / static_cast<float>(0x01000000u));
    };

    Callable make_onb = [](Float3 normal) noexcept {
        Var binormal = normalize(ite(
            abs(normal.x) > abs(normal.z),
            make_float3(-normal.y, normal.x, 0.0f),
            make_float3(0.0f, -normal.z, normal.y)));
        Var tangent = normalize(cross(binormal, normal));
        Var<Onb> onb{tangent, binormal, normal};
        return onb;
    };

    Callable transform_to_world = [](Var<Onb> onb, Float3 v) noexcept {
        return v.x * onb.tangent + v.y * onb.binormal + v.z * onb.normal;
    };

    Callable generate_ray = [](Float2 p) noexcept {
        static constexpr auto fov = radians(27.8f);
        static constexpr auto origin = make_float3(-0.01f, 0.995f, 5.0f);
        Var pixel = origin + make_float3(p * tan(0.5f * fov), -1.0f);
        Var direction = normalize(pixel - origin);
        return make_ray(origin, direction);
    };

    Callable cosine_sample_hemisphere = [](Float2 u) noexcept {
        Var r = sqrt(u.x);
        Var phi = 2.0f * constants::pi * u.y;
        return make_float3(r * cos(phi), r * sin(phi), sqrt(1.0f - u.x));
    };

    Callable balanced_heuristic = [](Float pdf_a, Float pdf_b) noexcept {
        return pdf_a / max(pdf_a + pdf_b, 1e-4f);
    };

    Kernel2D raytracing_kernel = [&](ImageFloat image, ImageUInt state_image, AccelVar accel) noexcept {
        set_block_size(8u, 8u, 1u);
        Var coord = dispatch_id().xy();
        Var frame_size = min(dispatch_size().x, dispatch_size().y).cast<float>();
        Var state = state_image.read(coord).x;
        Var rx = lcg(state);
        Var ry = lcg(state);
        Var pixel = (make_float2(coord) + make_float2(rx, ry)) / frame_size * 2.0f - 1.0f;
        Var ray = generate_ray(pixel * make_float2(1.0f, -1.0f));
        Var radiance = make_float3(0.0f);
        Var beta = make_float3(1.0f);
        Var pdf_bsdf = 0.0f;
        constexpr auto light_position = make_float3(-0.24f, 1.98f, 0.16f);
        constexpr auto light_u = make_float3(-0.24f, 1.98f, -0.22f) - light_position;
        constexpr auto light_v = make_float3(0.23f, 1.98f, 0.16f) - light_position;
        constexpr auto light_emission = make_float3(17.0f, 12.0f, 4.0f);
        auto light_area = length(cross(light_u, light_v));
        auto light_normal = normalize(cross(light_u, light_v));

        for (auto depth : range(5u)) {

            // trace
            Var hit = accel.trace_closest(ray);
            if_(miss(hit), break_);
            Var triangle = heap.buffer<Triangle>(hit.inst).read(hit.prim);
            Var p0 = vertex_buffer[triangle.i0];
            Var p1 = vertex_buffer[triangle.i1];
            Var p2 = vertex_buffer[triangle.i2];
            Var p = interpolate(hit, p0, p1, p2);
            Var n = normalize(cross(p1 - p0, p2 - p0));
            Var cos_wi = dot(-direction(ray), n);
            if_(cos_wi < 1e-4f, break_);
            Var material = material_buffer[hit.inst];

            // hit light
            if_(hit.inst == static_cast<uint>(meshes.size() - 1u), [&] {
                if_(depth == 0u, [&] {
                    radiance += light_emission;
                }).else_([&] {
                    Var pdf_light = length_squared(p - origin(ray)) / (light_area * cos_wi);
                    Var mis_weight = balanced_heuristic(pdf_bsdf, pdf_light);
                    radiance += mis_weight * beta * light_emission;
                });
                break_();
            });

            // sample light
            Var ux_light = lcg(state);
            Var uy_light = lcg(state);
            Var p_light = light_position + ux_light * light_u + uy_light * light_v;
            Var d_light = distance(p, p_light);
            Var wi_light = normalize(p_light - p);
            Var shadow_ray = make_ray_robust(p, n, wi_light, 0.0f, d_light - 1e-3f);
            Var occluded = accel.trace_any(shadow_ray);
            Var cos_wi_light = dot(wi_light, n);
            Var cos_light = -dot(light_normal, wi_light);
            if_(!occluded && cos_wi_light > 1e-4f && cos_light > 1e-4f, [&] {
                Var pdf_light = (d_light * d_light) / (light_area * cos_light);
                Var pdf_bsdf = cos_wi_light * inv_pi;
                Var mis_weight = balanced_heuristic(pdf_light, pdf_bsdf);
                Var bsdf = material.albedo * inv_pi * cos_wi_light;
                radiance += beta * bsdf * mis_weight * light_emission / max(pdf_light, 1e-4f);
            });

            // sample BSDF
            Var onb = make_onb(n);
            Var ux = lcg(state);
            Var uy = lcg(state);
            Var new_direction = transform_to_world(onb, cosine_sample_hemisphere(make_float2(ux, uy)));
            ray = make_ray_robust(p, n, new_direction);
            beta *= material.albedo;
            pdf_bsdf = cos_wi * inv_pi;

            // rr
            Var l = dot(make_float3(0.212671f, 0.715160f, 0.072169f), beta);
            if_(l == 0.0f, break_);
            Var q = max(l, 0.05f);
            Var r = lcg(state);
            if_(r >= q, break_);
            beta *= 1.0f / q;
        }
        state_image.write(coord, make_uint4(state));
        if_(isnan(radiance.x) || isnan(radiance.y) || isnan(radiance.z), [&] { radiance = make_float3(0.0f); });
        image.write(coord, make_float4(clamp(radiance, 0.0f, 30.0f), 1.0f));
    };

    Kernel2D accumulate_kernel = [&](ImageFloat accum_image, ImageFloat curr_image) noexcept {
        Var p = dispatch_id().xy();
        Var accum = accum_image.read(p);
        Var curr = curr_image.read(p).xyz();
        Var t = 1.0f / (accum.w + 1.0f);
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

    Kernel2D hdr2ldr_kernel = [&](ImageFloat hdr_image, ImageFloat ldr_image, Float scale) noexcept {
        Var coord = dispatch_id().xy();
        Var hdr = hdr_image.read(coord);
        Var ldr = linear_to_srgb(aces_tonemapping(hdr.zyx() * scale));
        ldr_image.write(coord, make_float4(ldr, 1.0f));
    };

    auto clear_shader = device.compile(clear_kernel);
    auto hdr2ldr_shader = device.compile(hdr2ldr_kernel);
    auto accumulate_shader = device.compile(accumulate_kernel);
    auto raytracing_shader = device.compile(raytracing_kernel);
    auto make_sampler_shader = device.compile(make_sampler_kernel);

    static constexpr auto width = 1024u;
    static constexpr auto height = 1024u;
    auto state_image = device.create_image<uint>(PixelStorage::INT1, width, height);
    auto framebuffer = device.create_image<float>(PixelStorage::FLOAT4, width, height);
    auto accum_image = device.create_image<float>(PixelStorage::FLOAT4, width, height);
    auto ldr_image = device.create_image<float>(PixelStorage::BYTE4, width, height);
    cv::Mat cv_accum{width, height, CV_8UC4, cv::Scalar::all(0.0)};
    cv::Mat cv_curr{width, height, CV_8UC4, cv::Scalar::all(0.0)};

    Clock clock;
    clock.tic();
    stream << clear_shader(accum_image).dispatch(width, height)
           << make_sampler_shader(state_image).dispatch(width, height);
    for (auto d = 0u;; d++) {
        auto command_buffer = stream.command_buffer();
        static constexpr auto spp_per_dispatch = 4u;
        for (auto i = 0u; i < spp_per_dispatch; i++) {
            command_buffer << raytracing_shader(framebuffer, state_image, accel).dispatch(width, height)
                           << accumulate_shader(accum_image, framebuffer).dispatch(width, height);
        }
        command_buffer << hdr2ldr_shader(framebuffer, ldr_image, 1.0f).dispatch(width, height)
                       << ldr_image.copy_to(cv_curr.data)
                       << hdr2ldr_shader(accum_image, ldr_image, 1.0f).dispatch(width, height)
                       << ldr_image.copy_to(cv_accum.data)
                       << commit();
        stream << synchronize();
        cv::imshow("Accumulated", cv_accum);
        cv::imshow("Current", cv_curr);
        if (cv::waitKey(1) == 'q') { break; }
        LUISA_INFO("Progress: {}spp | {}s", (d + 1u) * spp_per_dispatch, clock.toc() * 1e-3f);
    }
    cv::imwrite("test_path_tracing.png", cv_accum);
}
