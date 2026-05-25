#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <memory>
#include <numbers>
#include <optional>
#include <string_view>

#include <luisa/luisa-compute.h>
#include <luisa/dsl/sugar.h>
#include <luisa/runtime/rtx/mesh.h>
#include <luisa/runtime/rtx/accel.h>

#include <stb/stb_image_write.h>
#include "common/reference_compare.h"
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
    [[nodiscard]] auto to_world(Expr<float3> v) const noexcept {
        return v.x * tangent + v.y * binormal + v.z * normal;
    }
};

struct alignas(16) PtFrame {
    uint2 coord;
    uint state;
    uint depth;
    float3 ray_origin;
    float3 ray_dir;
    float3 beta;
    float3 radiance;
    float pdf_bsdf;
    uint sample_idx;
};

LUISA_STRUCT(PtFrame, coord, state, depth, ray_origin, ray_dir, beta, radiance, pdf_bsdf, sample_idx){};

namespace {

constexpr uint2 kResolution = make_uint2(1024u, 1024u);
constexpr uint kPathCount = kResolution.x * kResolution.y;
constexpr uint kMaxDepth = 10u;

constexpr uint kCounterBounceA = 0u;
constexpr uint kCounterBounceB = 1u;
constexpr uint kCounterFinalize = 2u;
constexpr uint kCounterTotal = 3u;

}

int main(int argc, char *argv[]) {
    log_level_verbose();
    Context context{argv[0]};
    if (argc <= 1) {
        LUISA_INFO("Usage: {} <backend> [--offline] [--spp N] [-c <reference.png>].", argv[0]);
        exit(1);
    }
    auto opts = luisa::ref::ExampleOptions::parse(argc, argv);
    Device device = context.create_device(argv[1]);

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
        vertices.emplace_back(float3{p[i + 0u], p[i + 1u], p[i + 2u]});
    }
    LUISA_INFO("Loaded mesh with {} shape(s) and {} vertices.", obj_reader.GetShapes().size(), vertices.size());

    BindlessArray heap = device.create_bindless_array();
    Stream stream = device.create_stream(StreamTag::COMPUTE);
    auto vertex_buffer = device.create_buffer<float3>(vertices.size());
    stream << vertex_buffer.copy_from(luisa::span{vertices});
    std::vector<Mesh> meshes;
    std::vector<Buffer<Triangle>> triangle_buffers;
    for (auto &&shape : obj_reader.GetShapes()) {
        auto index = static_cast<uint>(meshes.size());
        auto &&t = shape.mesh.indices;
        auto triangle_count = t.size() / 3u;
        LUISA_INFO("Processing shape '{}' at index {} with {} triangle(s).", shape.name, index, triangle_count);
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
    for (auto &&m : meshes) { accel.emplace_back(m, make_float4x4(1.0f)); }
    stream << heap.update() << accel.build() << synchronize();

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

    auto light_mesh_id = static_cast<uint>(meshes.size() - 1u);

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

    auto lcg = [](UInt &state) noexcept {
        constexpr auto lcg_a = 1664525u;
        constexpr auto lcg_c = 1013904223u;
        state = lcg_a * state + lcg_c;
        return cast<float>(state & 0x00ffffffu) * (1.0f / static_cast<float>(0x01000000u));
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

    Kernel1D reset_counters_kernel = [](BufferVar<uint> counters) noexcept {
        counters->write(dispatch_id().x, 0u);
    };

    Kernel2D make_sampler_kernel = [&](ImageUInt seed_image) noexcept {
        auto p = dispatch_id().xy();
        auto state = tea(p.x, p.y);
        seed_image.write(p, make_uint4(state));
    };

    Kernel2D clear_kernel = [](ImageFloat image) noexcept {
        image.write(dispatch_id().xy(), make_float4(0.0f));
    };

    // C0_setup: thread per path; init frame, enqueue to bounce queue A
    Kernel1D setup_kernel = [&](BufferVar<PtFrame> frames,
                                BufferVar<uint> bounce_q_a,
                                BufferVar<uint> counters,
                                ImageUInt seed_image,
                                UInt sample_idx) noexcept {
        auto path_id = dispatch_id().x;
        $if (path_id >= kPathCount) { $return(); };
        auto coord = make_uint2(path_id % kResolution.x, path_id / kResolution.x);
        auto state = seed_image.read(coord).x;
        auto frame_size = cast<float>(min(kResolution.x, kResolution.y));
        auto rx = lcg(state);
        auto ry = lcg(state);
        auto pixel = (make_float2(coord) + make_float2(rx, ry)) / frame_size * 2.0f - 1.0f;
        auto ray = generate_ray(pixel * make_float2(1.0f, -1.0f));
        Var<PtFrame> frame;
        frame.coord = coord;
        frame.state = state;
        frame.depth = 0u;
        frame.ray_origin = ray->origin();
        frame.ray_dir = ray->direction();
        frame.beta = make_float3(1.0f);
        frame.radiance = make_float3(0.0f);
        frame.pdf_bsdf = 0.0f;
        frame.sample_idx = sample_idx;
        frames->write(path_id, frame);
        auto slot = counters->atomic(kCounterBounceA).fetch_add(1u);
        bounce_q_a->write(slot, path_id);
    };

    // C_bounce: read in_q, do one bounce, decide next continuation
    Kernel1D bounce_kernel = [&](BufferVar<PtFrame> frames,
                                 BufferVar<uint> in_q,
                                 BufferVar<uint> bounce_q_out,
                                 BufferVar<uint> finalize_q,
                                 BufferVar<uint> counters,
                                 UInt out_bounce_counter_idx,
                                 UInt task_count,
                                 AccelVar accel_var) noexcept {
        constexpr auto light_position = make_float3(-0.24f, 1.98f, 0.16f);
        constexpr auto light_u = make_float3(-0.24f, 1.98f, -0.22f) - light_position;
        constexpr auto light_v = make_float3(0.23f, 1.98f, 0.16f) - light_position;
        constexpr auto light_emission = make_float3(17.0f, 12.0f, 4.0f);
        auto &&heap_ref = heap;
        auto &&vertex_buffer_ref = vertex_buffer;
        auto &&materials_ref = materials;
        auto task = dispatch_id().x;
        $if (task >= task_count) { $return(); };
        auto path_id = in_q->read(task);
        auto frame = frames->read(path_id);
        auto state = def(frame.state);
        auto beta = def(frame.beta);
        auto radiance = def(frame.radiance);
        auto pdf_bsdf = def(frame.pdf_bsdf);
        auto depth = def(frame.depth);
        auto ray = make_ray(frame.ray_origin, frame.ray_dir);
        auto exited = def(false);

        auto light_area = length(cross(light_u, light_v));
        auto light_normal = normalize(cross(light_u, light_v));
        auto hit = accel_var.intersect(ray, {});
        $if (hit->miss()) {
            exited = true;
        }
        $else {
            auto triangle = heap_ref->buffer<Triangle>(hit.inst).read(hit.prim);
            auto p0 = vertex_buffer_ref->read(triangle.i0);
            auto p1 = vertex_buffer_ref->read(triangle.i1);
            auto p2 = vertex_buffer_ref->read(triangle.i2);
            auto p_hit = triangle_interpolate(hit.bary, p0, p1, p2);
            auto n = normalize(cross(p1 - p0, p2 - p0));
            auto cos_wo = dot(-ray->direction(), n);
            $if (cos_wo < 1e-4f) {
                exited = true;
            }
            $else {
                auto albedo = materials_ref->read(hit.inst);
                $if (hit.inst == light_mesh_id) {
                    $if (depth == 0u) {
                        radiance += light_emission;
                    }
                    $else {
                        auto pdf_light = length_squared(p_hit - ray->origin()) / (light_area * cos_wo);
                        auto mis_weight = balanced_heuristic(pdf_bsdf, pdf_light);
                        radiance += mis_weight * beta * light_emission;
                    };
                    exited = true;
                }
                $else {
                    auto ux_light = lcg(state);
                    auto uy_light = lcg(state);
                    auto p_light = light_position + ux_light * light_u + uy_light * light_v;
                    auto pp = offset_ray_origin(p_hit, n);
                    auto pp_light = offset_ray_origin(p_light, light_normal);
                    auto d_light = distance(pp, pp_light);
                    auto wi_light = normalize(pp_light - pp);
                    auto shadow_ray = make_ray(offset_ray_origin(pp, n), wi_light, 0.f, d_light);
                    auto occluded = accel_var.intersect_any(shadow_ray, {});
                    auto cos_wi_light = dot(wi_light, n);
                    auto cos_light = -dot(light_normal, wi_light);
                    $if (!occluded & cos_wi_light > 1e-4f & cos_light > 1e-4f) {
                        auto pdf_light = (d_light * d_light) / (light_area * cos_light);
                        auto pdf_b = cos_wi_light * inv_pi;
                        auto mis_weight = balanced_heuristic(pdf_light, pdf_b);
                        auto bsdf = albedo * inv_pi * cos_wi_light;
                        radiance += beta * bsdf * mis_weight * light_emission / max(pdf_light, 1e-4f);
                    };

                    Var<Onb> onb = make_onb(n);
                    Float ux = lcg(state);
                    Float uy = lcg(state);
                    Float3 wi_local = cosine_sample_hemisphere(make_float2(ux, uy));
                    Float cos_wi = abs(wi_local.z);
                    Float3 new_direction = onb->to_world(wi_local);
                    ray = make_ray(pp, new_direction);
                    pdf_bsdf = cos_wi * inv_pi;
                    beta *= albedo;

                    auto l = dot(make_float3(0.212671f, 0.715160f, 0.072169f), beta);
                    $if (l == 0.0f) {
                        exited = true;
                    }
                    $else {
                        auto q = max(l, 0.05f);
                        auto r = lcg(state);
                        $if (r >= q) {
                            exited = true;
                        }
                        $else {
                            beta *= 1.0f / q;
                        };
                    };
                };
            };
        };

        depth = depth + 1u;
        $if (depth >= kMaxDepth) { exited = true; };

        frame.state = state;
        frame.depth = depth;
        frame.ray_origin = ray->origin();
        frame.ray_dir = ray->direction();
        frame.beta = beta;
        frame.radiance = radiance;
        frame.pdf_bsdf = pdf_bsdf;
        frames->write(path_id, frame);
        $if (exited) {
            auto slot = counters->atomic(kCounterFinalize).fetch_add(1u);
            finalize_q->write(slot, path_id);
        }
        $else {
            auto slot = counters->atomic(out_bounce_counter_idx).fetch_add(1u);
            bounce_q_out->write(slot, path_id);
        };
    };

    // C_finalize: write radiance into accum_image and update seed
    Kernel1D finalize_kernel = [&](BufferVar<PtFrame> frames,
                                   BufferVar<uint> in_q,
                                   ImageUInt seed_image,
                                   ImageFloat accum_image,
                                   UInt task_count) noexcept {
        auto task = dispatch_id().x;
        $if (task >= task_count) { $return(); };
        auto path_id = in_q->read(task);
        auto frame = frames->read(path_id);
        auto coord = frame.coord;
        auto radiance = frame.radiance;
        $if (any(dsl::isnan(radiance))) { radiance = make_float3(0.0f); };
        auto prev = accum_image.read(coord);
        accum_image.write(coord, prev + make_float4(radiance, 1.0f));
        seed_image.write(coord, make_uint4(frame.state));
    };

    Kernel2D hdr2ldr_kernel = [&](ImageFloat hdr_image, ImageFloat ldr_image, Float scale) noexcept {
        auto coord = dispatch_id().xy();
        auto hdr = hdr_image.read(coord);
        auto avg = clamp(hdr.xyz() / hdr.w, 0.0f, 30.0f);
        auto ldr = linear_to_srgb(clamp(avg * scale, 0.0f, 1.0f));
        ldr_image.write(coord, make_float4(ldr, 1.0f));
    };

    auto reset_counters = device.compile(reset_counters_kernel);
    auto make_sampler = device.compile(make_sampler_kernel);
    auto clear_shader = device.compile(clear_kernel);
    auto setup = device.compile(setup_kernel);
    auto bounce = device.compile(bounce_kernel);
    auto finalize = device.compile(finalize_kernel);
    auto hdr2ldr = device.compile(hdr2ldr_kernel);

    auto frames_buf = device.create_buffer<PtFrame>(kPathCount);
    auto bounce_q_a = device.create_buffer<uint>(kPathCount);
    auto bounce_q_b = device.create_buffer<uint>(kPathCount);
    auto finalize_q = device.create_buffer<uint>(kPathCount);
    auto counters = device.create_buffer<uint>(kCounterTotal);

    auto seed_image = device.create_image<uint>(PixelStorage::INT1, kResolution);
    auto accum_image = device.create_image<float>(PixelStorage::FLOAT4, kResolution);
    auto ldr_image = device.create_image<float>(PixelStorage::BYTE4, kResolution);
    std::vector<std::array<uint8_t, 4u>> host_image(kResolution.x * kResolution.y);

    static constexpr uint block_x = 256u;
    auto round_up = [](uint x, uint block) { return ((x + block - 1u) / block) * block; };
    uint total_spp = opts.spp == 0u ? 1024u : opts.spp;

    stream << clear_shader(accum_image).dispatch(kResolution)
           << make_sampler(seed_image).dispatch(kResolution)
           << synchronize();

    Clock clock;
    auto t0 = clock.toc();
    luisa::vector<uint> host_counters(kCounterTotal);
    for (uint sample_idx = 0u; sample_idx < total_spp; sample_idx++) {
        stream << reset_counters(counters).dispatch(kCounterTotal);
        stream << setup(frames_buf, bounce_q_a, counters, seed_image, sample_idx).dispatch(round_up(kPathCount, block_x));
        uint active = 0u;
        while (true) {
            stream << counters.copy_to(host_counters.data()) << synchronize();
            uint count = active == 0u ? host_counters[kCounterBounceA] : host_counters[kCounterBounceB];
            if (count == 0u) { break; }
            uint out_idx = active == 0u ? kCounterBounceB : kCounterBounceA;
            uint reset_idx = active == 0u ? kCounterBounceA : kCounterBounceB;
            auto in_q = active == 0u ? bounce_q_a.view() : bounce_q_b.view();
            auto out_q = active == 0u ? bounce_q_b.view() : bounce_q_a.view();
            stream << bounce(frames_buf, in_q, out_q, finalize_q, counters, out_idx, count, accel)
                          .dispatch(round_up(count, block_x));
            stream << reset_counters(counters.view(reset_idx, 1u)).dispatch(1u);
            active = 1u - active;
        }
        uint finalize_count = host_counters[kCounterFinalize];
        if (finalize_count > 0u) {
            stream << finalize(frames_buf, finalize_q, seed_image, accum_image, finalize_count)
                          .dispatch(round_up(finalize_count, block_x));
        }
    }
    stream << synchronize();
    auto fps = total_spp / (clock.toc() - t0) * 1000;
    LUISA_INFO("Wavefront PT: {} samples/s", fps);

    stream << hdr2ldr(accum_image, ldr_image, 2.0f).dispatch(kResolution)
           << ldr_image.copy_to(luisa::span{host_image})
           << synchronize();
    stbi_write_png("test_path_tracing_wavefront.png", kResolution.x, kResolution.y, 4, host_image.data(), 0);
    if (opts.offline && opts.compare_path) {
        auto result = luisa::ref::compare_with_reference_file(
            reinterpret_cast<const uint8_t *>(host_image.data()),
            kResolution.x, kResolution.y, 4,
            *opts.compare_path);
        LUISA_INFO("Reference comparison: {} ({})", result.passed ? "PASSED" : "FAILED", result.message);
        if (!result.passed) { return 1; }
    }
    return 0;
}
