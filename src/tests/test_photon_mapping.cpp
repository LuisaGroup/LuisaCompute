//
// Created by frvdec99 on 2022/9/3.
//

#include <cstdint>
#include <iostream>

#include <core/clock.h>
#include <core/logging.h>
#include <runtime/context.h>
#include <runtime/device.h>
#include <runtime/stream.h>
#include <dsl/sugar.h>
#include <runtime/rtx/accel.h>
#include <gui/window.h>
#include <tests/cornell_box.h>
#include <stb/stb_image_write.h>

#include <random>

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

struct Photon {
    std::array<float, 3> position;
    std::array<float, 3> power;
    std::array<float, 3> in_direction;
    uint nxt;
};

struct heapNode {
    float dis;
    uint index;
};

// clang-format off
LUISA_STRUCT(Material, albedo, emission) {};
LUISA_STRUCT(Onb, tangent, binormal, normal) {
    [[nodiscard]] auto to_world(Expr<float3> v) const noexcept {
        return v.x * tangent + v.y * binormal + v.z * normal;
    }
};
LUISA_STRUCT(Photon, position, power, in_direction, nxt) {};
LUISA_STRUCT(heapNode, dis, index) {};
// clang-format on

int main(int argc, char *argv[]) {
    log_level_info();

    Context context{argv[0]};
    if (argc <= 1) {
        LUISA_INFO("Usage: {} <backend>. <backend>: cuda, dx, ispc, metal", argv[0]);
        exit(1);
    }
    auto device = context.create_device(argv[1]);

    auto lcg = [](UInt &state) noexcept {
        constexpr auto lcg_a = 1664525u;
        constexpr auto lcg_c = 1013904223u;
        state = lcg_a * state + lcg_c;
        return cast<float>(state & 0x00ffffffu) *
               (1.0f / static_cast<float>(0x01000000u));
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

    auto make_onb = [](const Float3 &normal) noexcept {
        auto binormal = normalize(ite(
            abs(normal.x) > abs(normal.z),
            make_float3(-normal.y, normal.x, 0.0f),
            make_float3(0.0f, -normal.z, normal.y)));
        auto tangent = normalize(cross(binormal, normal));
        return def<Onb>(tangent, binormal, normal);
    };

    auto cosine_sample_hemisphere = [](Float2 u) noexcept {
        auto r = sqrt(u.x);
        auto phi = 2.0f * constants::pi * u.y;
        return make_float3(r * cos(phi), r * sin(phi), sqrt(1.0f - u.x));
    };

    //load the Cornell Box scene
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

    auto grid_min = make_float3(1e8);
    auto grid_max = make_float3(-1e8);
    auto &&p = obj_reader.GetAttrib().vertices;
    std::vector<float3> vertices;
    vertices.reserve(p.size() / 3u);
    for (auto i = 0u; i < p.size(); i += 3u) {
        vertices.emplace_back(float3{
            p[i + 0u],
            p[i + 1u],
            p[i + 2u]});

        grid_min.x = min(p[i + 0u], grid_min.x);
        grid_min.y = min(p[i + 1u], grid_min.y);
        grid_min.z = min(p[i + 2u], grid_min.z);

        grid_max.x = max(p[i + 0u], grid_max.x);
        grid_max.y = max(p[i + 1u], grid_max.y);
        grid_max.z = max(p[i + 2u], grid_max.z);
    }

    LUISA_INFO(
        "Loaded mesh with {} shape(s) and {} vertices.",
        obj_reader.GetShapes().size(), vertices.size());

    auto heap = device.create_bindless_array();
    auto stream = device.create_stream(StreamTag::GRAPHICS);
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
        heap.emplace_on_update(index, triangle_buffer);
        stream << triangle_buffer.copy_from(indices.data())
               << mesh.build();
    }

    auto accel = device.create_accel({});
    for (auto &&m : meshes) {
        accel.emplace_back(m, make_float4x4(1.0f));
    }
    stream << heap.update()
           << accel.build()
           << synchronize();

    //TODO
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
    auto material_buffer = device.create_buffer<Material>(materials.size());
    stream << material_buffer.copy_from(materials.data());

    constexpr auto light_position = make_float3(-0.24f, 1.98f, 0.16f);
    constexpr auto light_u = make_float3(-0.24f, 1.98f, -0.22f) - light_position;
    constexpr auto light_v = make_float3(0.23f, 1.98f, 0.16f) - light_position;
    constexpr auto light_emission = make_float3(17.0f, 12.0f, 4.0f);
    auto light_area = length(cross(light_u, light_v));
    auto light_normal = normalize(cross(light_u, light_v));

    constexpr auto photon_number = 1000000u;
    constexpr auto max_depth = 8u;
    // constexpr auto scale = 1000u;

    luisa::vector<uint> seeds(photon_number);
    std::mt19937 rng{std::random_device{}()};
    for (auto i = 0u; i < photon_number; i++) {
        seeds[i] = rng();
    }
    auto seed_buffer = device.create_buffer<uint>(photon_number);
    auto photon_buffer = device.create_buffer<Photon>(photon_number * max_depth);
    auto photon_limit_buffer = device.create_buffer<uint>(1u);

    constexpr auto split_per_dim = 128u;
    auto grid_real_size = grid_max - grid_min;
    auto grid_len = min(min(grid_real_size.x / split_per_dim, grid_real_size.y / split_per_dim), grid_real_size.z / split_per_dim);
    auto grid_size = make_uint3(uint(grid_real_size.x / grid_len) + 1,
                                uint(grid_real_size.y / grid_len) + 1,
                                uint(grid_real_size.z / grid_len) + 1);
    LUISA_INFO("grid_size {} {} {}", grid_size.x, grid_size.y, grid_size.z);
    auto grid_head_buffer = device.create_buffer<uint>(grid_size.x * grid_size.y * grid_size.z);

    LUISA_INFO("grid_len = {}", grid_len);

    stream << seed_buffer.copy_from(seeds.data());

    Kernel1D clear_grid_kernel = [&]() noexcept {
        auto index = static_cast<UInt>(dispatch_x());
        grid_head_buffer->write(index, ~0u);
    };

    Kernel1D photon_tracing_kernel = [&](AccelVar accel) noexcept {
        // random seed
        auto state = seed_buffer->read(dispatch_x());
        auto state2 = dispatch_x();

        // sample light
        auto ux_light = lcg(state2);
        auto uy_light = lcg(state2);
        auto light_p = light_position + ux_light * light_u + uy_light * light_v;

        auto light_pp = offset_ray_origin(light_p, light_normal);
        auto ux_light_dir = lcg(state);
        auto uy_light_dir = lcg(state);
        auto light_onb = make_onb(light_normal);
        auto light_dir = light_onb->to_world(cosine_sample_hemisphere(make_float2(ux_light_dir, uy_light_dir)));
        auto light_ray = make_ray(light_pp, light_dir);
        auto light_cos = dot(light_dir, light_normal);
        auto light_pdf_pos = 1.0f / light_area;
        auto light_pdf_dir = light_cos * inv_pi;
        // auto power = def(light_emission * light_cos / light_pdf_pos / light_pdf_dir) =>
        auto power = def(light_emission / (light_pdf_pos * inv_pi));

        $for(depth, max_depth) {
            // trace
            auto hit = accel.trace_closest(light_ray);
            $if(hit->miss()) { $break; };
            // $if(hit.inst == 0 & hit.prim == 0) { $break; };
            // TODO
            $if(hit.inst == static_cast<uint>(meshes.size() - 1u)) {
                $break;
            };

            auto triangle = heap->buffer<Triangle>(hit.inst).read(hit.prim);
            auto p0 = vertex_buffer->read(triangle.i0);
            auto p1 = vertex_buffer->read(triangle.i1);
            auto p2 = vertex_buffer->read(triangle.i2);
            auto p = hit->interpolate(p0, p1, p2);
            auto n = normalize(cross(p1 - p0, p2 - p0));
            auto cos_wi = dot(-light_ray->direction(), n);
            $if(cos_wi < 1e-4f) { $break; };
            auto material = material_buffer->read(hit.inst);

            // store photons
            auto index = photon_limit_buffer->atomic(0).fetch_add(1u);

            auto grid_index = make_uint3((p - grid_min) / grid_len);
            auto grid_index_m = grid_index.x * grid_size.y * grid_size.z + grid_index.y * grid_size.z + grid_index.z;
            auto link_head = grid_head_buffer->atomic(grid_index_m).exchange(index);

            Var<Photon> photon{p, power, -light_ray->direction(), link_head};
            photon_buffer->write(index, photon);

            // sample BxDF
            auto onb = make_onb(n);
            auto ux = lcg(state);
            auto uy = lcg(state);
            auto new_dir = onb->to_world(cosine_sample_hemisphere(make_float2(ux, uy)));
            auto cos_dir = dot(new_dir, n);
            auto pdf_dir = cos_dir * inv_pi;
            auto pp = offset_ray_origin(p, n);
            light_ray = make_ray(pp, new_dir);
            // power *= material.albedo * inv_pi * cos_dir / pdf_dir =>
            power *= material.albedo;

            // rr
            auto rr_prob = 0.9f;
            auto rr_check = lcg(state);
            $if(rr_check >= rr_prob) { $break; };
            power *= 1.0f / rr_prob;
        };
    };

    constexpr auto photon_radius = 0.007f;
    // constexpr auto kNN_k = 20u;
    // auto heap_buffer = device.create_buffer<heapNode>(resolution.x * resolution.y * kNN_k);

    Callable generate_ray = [](Float2 p) noexcept {
        static constexpr auto fov = radians(27.8f);
        static constexpr auto origin = make_float3(-0.01f, 0.995f, 5.0f);
        auto pixel = origin + make_float3(p * tan(0.5f * fov), -1.0f);
        auto direction = normalize(pixel - origin);
        return make_ray(origin, direction);
    };

    Callable get_index_from_point = [&](Float3 p) noexcept {
        return make_uint3(max((p - grid_min) / grid_len + 0.5f, 0.f));
    };

    Callable get_index_merge = [&](UInt x, UInt y, UInt z) noexcept {
        return x * grid_size.y * grid_size.z + y * grid_size.z + z;
    };

    auto density_estimation_radius = [&](UInt2 coord, Float3 p, Float3 dir, Float r, Var<Material> material) noexcept {
        auto radiance = def(make_float3(0.0f));
        auto p_index = get_index_from_point(p);
        auto photon_sum = def(0);

        $for(x, ite(p_index.x == 0, 0u, p_index.x - 1), min(p_index.x + 1, grid_size.x)) {
            $for(y, ite(p_index.y == 0, 0u, p_index.y - 1), min(p_index.y + 1, grid_size.y)) {
                $for(z, ite(p_index.z == 0, 0u, p_index.z - 1), min(p_index.z + 1, grid_size.z)) {
                    auto grid_index_m = get_index_merge(x, y, z);
                    auto photon_index = grid_head_buffer->read(grid_index_m);
                    $while(photon_index != ~0u) {
                        photon_sum += 1;
                        auto photon = photon_buffer->read(photon_index);
                        auto dis = distance(Float3{photon.position}, p);
                        $if(dis < r) {
                            radiance += material.albedo * inv_pi * Float3{photon.power};
                        };
                        photon_index = photon.nxt;
                    };
                };
            };
        };

        return radiance;
    };

    /*
    Callable density_estimation_kNN = [&](UInt2 coord, Float3 p, Float3 dir, Float &r, Var<Material> material) noexcept {
        auto radiance = def(make_float3(0.0f));
        auto p_index = get_index_from_point(p);

        auto base_offset = (coord.x * resolution.y + coord.y) * kNN_k;
        auto kNN_offset = def(0u);

        $for(x, ite(p_index.x == 0, 0u, p_index.x-1), min(p_index.x+1, grid_size.x)) {
            $for(y, ite(p_index.y == 0, 0u, p_index.y-1), min(p_index.y+1, grid_size.y)) {
                $for(z, ite(p_index.z == 0, 0u, p_index.z-1), min(p_index.z+1, grid_size.z)) {
                    auto grid_index_m = get_index_merge(x, y, z);
                    auto photon_index = grid_head_buffer->read(grid_index_m);
                    $while(photon_index != -1u) {
                        auto photon = photon_buffer->read(photon_index);
                        auto dis = distance(Float3{photon.position}, p);
                        $if( (kNN_offset < kNN_k) | (dis < r) ) {
                            auto insert_index = base_offset + kNN_offset;
                            $for(check_index, base_offset, base_offset + kNN_offset) {
                                auto check_photon = heap_buffer->read(check_index);
                                $if(dis < check_photon.dis){
                                    insert_index = check_index;
                                    $break;
                                };
                            };
                            $if(kNN_offset < kNN_k){ kNN_offset += 1; };
                            auto copy_index = base_offset+kNN_offset-1;
                            $while(copy_index > insert_index){
                                heap_buffer->write(copy_index, heap_buffer->read(copy_index-1));
                                copy_index -= 1;
                            };
                            Var<heapNode> node{dis, photon_index};
                            heap_buffer->write(insert_index, node);
                            r = heap_buffer->read(base_offset+kNN_offset-1).dis;
                        };
                        photon_index = photon.nxt;
                    };
                };
            };
        };
        $for(index_offset, kNN_offset){
            auto real_index = base_offset + index_offset;
            auto photon_index = heap_buffer->read(real_index).index;
            auto photon = photon_buffer->read(photon_index);
            radiance += material.albedo * Float3{photon.power};
        };
        return radiance;
    };
    */

    Callable indirect_illumination = [&](UInt2 coord, Var<Ray> ray, AccelVar accel) {
        auto radiance = def(make_float3(0.0f));
        auto radius = def(photon_radius);

        auto hit = accel.trace_closest(ray);
        // $if(!hit->miss() & (hit.inst != 0 | hit.prim != 0)) {
        $if(!hit->miss()) {
            $if(hit.inst == static_cast<uint>(meshes.size() - 1u)) {
                // radiance = light_emission / light_area;
            }
            $else {
                auto triangle = heap->buffer<Triangle>(hit.inst).read(hit.prim);
                auto p0 = vertex_buffer->read(triangle.i0);
                auto p1 = vertex_buffer->read(triangle.i1);
                auto p2 = vertex_buffer->read(triangle.i2);
                auto p = hit->interpolate(p0, p1, p2);
                auto n = normalize(cross(p1 - p0, p2 - p0));
                auto material = material_buffer->read(hit.inst);
                auto cos_wi = dot(-ray->direction(), n);

                $if(cos_wi > 1e-4f) {
                    radiance = density_estimation_radius(coord, p, -ray->direction(), radius, material);
                    radiance *= inv_pi / (radius * radius * photon_number);

                    // $if(dot(radiance, radiance) > 10000) {
                    //     printer.info_with_location("p : ({}, {}, {}) o ({}, {}, {}) dir ({}, {}, {}) inst {} prim {}",
                    //     p.x, p.y, p.z,
                    //     ray->origin().x, ray->origin().y, ray->origin().z,
                    //     ray->direction().x, ray->direction().y, ray->direction().z,
                    //     hit.inst, hit.prim
                    //     );
                    // };
                };
            };
        };
        return radiance;
    };

    Callable balanced_heuristic = [](Float pdf_a, Float pdf_b) noexcept {
        return pdf_a / max(pdf_a + pdf_b, 1e-4f);
    };

    Kernel2D photon_gathering_kernel = [&](ImageFloat image, ImageUInt seed_image, AccelVar accel, UInt2 resolution) noexcept {
        auto coord = dispatch_id().xy();
        auto frame_size = min(resolution.x, resolution.y).cast<float>();
        auto state = tea(seed_image.read(coord).x, seed_image.read(coord).y);
        auto radiance = def(make_float3(0.0f));

        auto rx = lcg(state);
        auto ry = lcg(state);
        auto pixel = (make_float2(coord) + make_float2(rx, ry)) / frame_size * 2.0f - 1.0f;
        auto ray = generate_ray(pixel * make_float2(1.0f, -1.0f));
        auto radius = def(photon_radius);
        auto hit = accel.trace_closest(ray);
        $if(!hit->miss()) {
            $if(hit.inst == static_cast<uint>(meshes.size() - 1u)) {
                radiance = light_emission;
            }
            $else {
                auto triangle = heap->buffer<Triangle>(hit.inst).read(hit.prim);
                auto p0 = vertex_buffer->read(triangle.i0);
                auto p1 = vertex_buffer->read(triangle.i1);
                auto p2 = vertex_buffer->read(triangle.i2);
                auto n = normalize(cross(p1 - p0, p2 - p0));
                auto p = hit->interpolate(p0, p1, p2);
                auto cos_wi = dot(-ray->direction(), n);
                $if(cos_wi > 1e-4f) {
                    auto material = material_buffer->read(hit.inst);
                    //direct illumination
                    auto ux_light = lcg(state);
                    auto uy_light = lcg(state);
                    auto p_light = light_position + ux_light * light_u + uy_light * light_v;
                    auto pp = offset_ray_origin(p, n);
                    auto pp_light = offset_ray_origin(p_light, light_normal);
                    auto d_light = distance(pp, pp_light);
                    auto wi_light = normalize(pp_light - pp);
                    auto shadow_ray = make_ray(offset_ray_origin(pp, n), wi_light, 0.f, d_light);
                    auto occluded = accel.trace_any(shadow_ray);
                    auto cos_wi_light = dot(wi_light, n);
                    auto cos_light = -dot(light_normal, wi_light);
                    $if(!occluded & cos_wi_light > 1e-4f & cos_light > 1e-4f) {
                        auto pdf_light = (d_light * d_light) / (light_area * cos_light);
                        auto pdf_bsdf = cos_wi_light * inv_pi;
                        auto mis_weight = balanced_heuristic(pdf_light, pdf_bsdf);
                        auto bsdf = material.albedo * inv_pi * cos_wi_light;
                        radiance += bsdf * mis_weight * light_emission / max(pdf_light, 1e-4f);
                    };

                    //indirect illumination
                    auto onb = make_onb(n);
                    auto ux = lcg(state);
                    auto uy = lcg(state);
                    auto new_direction = onb->to_world(cosine_sample_hemisphere(make_float2(ux, uy)));
                    ray = make_ray(pp, new_direction);
                    auto new_cos = dot(new_direction, n);
                    auto pdf_dir = inv_pi * new_cos;

                    auto indirect = indirect_illumination(coord, ray, accel);
                    // radiance += material.albedo * inv_pi * new_cos * indirect / pdf_dir =>
                    radiance += material.albedo * indirect;
                };
            };
        };

        seed_image.write(coord, make_uint4(state));
        $if(any(dsl::isnan(radiance))) { radiance = make_float3(0.0f); };
        image.write(dispatch_id().xy(), make_float4(clamp(radiance, 0.0f, 30.0f), 1.0f));
    };

    Kernel2D accumulate_kernel = [&](ImageFloat accum_image, ImageFloat curr_image) noexcept {
        auto p = dispatch_id().xy();
        auto accum = accum_image.read(p);
        auto curr = curr_image.read(p).xyz();
        accum_image.write(p, accum + make_float4(curr, 1.f));
    };

    Callable linear_to_srgb = [](Var<float3> x) noexcept {
        return clamp(select(1.055f * pow(x, 1.0f / 2.4f) - 0.055f,
                            12.92f * x,
                            x <= 0.00031308f),
                     0.0f, 1.0f);
    };

    Kernel2D hdr2ldr_kernel = [&](ImageFloat hdr_image, ImageFloat ldr_image, Float scale) noexcept {
        auto coord = dispatch_id().xy();
        auto hdr = hdr_image.read(coord);
        auto ldr = linear_to_srgb(hdr.xyz() / hdr.w * scale);
        ldr_image.write(coord, make_float4(ldr, 1.0f));
    };

    Kernel2D clear_kernel = [](ImageFloat image) noexcept {
        image.write(dispatch_id().xy(), make_float4(0.0f));
    };

    Kernel2D make_sampler_kernel = [&](ImageUInt seed_image) noexcept {
        auto p = dispatch_id().xy();
        auto state = tea(p.x, p.y);
        seed_image.write(p, make_uint4(state));
    };

    auto clear_shader = device.compile(clear_kernel);
    auto make_sampler_shader = device.compile(make_sampler_kernel);
    auto clear_grid_shader = device.compile(clear_grid_kernel);
    auto photon_tracing_shader = device.compile(photon_tracing_kernel);
    auto photon_gathering_shader = device.compile(photon_gathering_kernel);
    auto accumulate_shader = device.compile(accumulate_kernel);
    auto hdr2ldr_shader = device.compile(hdr2ldr_kernel);

    static constexpr auto resolution = make_uint2(1024u);
    auto seed_image = device.create_image<uint>(PixelStorage::INT1, resolution);
    auto framebuffer = device.create_image<float>(PixelStorage::HALF4, resolution);
    auto accum_image = device.create_image<float>(PixelStorage::FLOAT4, resolution);
    auto ldr_image = device.create_image<float>(PixelStorage::BYTE4, resolution);
    std::vector<std::array<uint8_t, 4u>> host_image(resolution.x * resolution.y);

    stream << clear_shader(accum_image).dispatch(resolution)
           << make_sampler_shader(seed_image).dispatch(resolution)
           << clear_grid_shader().dispatch(grid_size.x * grid_size.y * grid_size.z)
           << photon_tracing_shader(accel).dispatch(photon_number)
           << synchronize();
    LUISA_INFO("Photon tracing done");

    auto frame_count = 0;

    Window window{"Display", resolution.x, resolution.y, false};
    auto swap_chain{device.create_swapchain(
        window.window_native_handle(),
        stream,
        resolution,
        true, false, 2)};
    Clock clk;
    while (!window.should_close()) {
        auto cmd_list = CommandList::create();
        static constexpr auto spp_per_dispatch = 1u;
        for (auto i = 0u; i < spp_per_dispatch; i++) {
            cmd_list
                << photon_gathering_shader(framebuffer, seed_image, accel, resolution).dispatch(resolution)
                << accumulate_shader(accum_image, framebuffer).dispatch(resolution);
        }
        cmd_list
            << hdr2ldr_shader(accum_image, ldr_image, 1.0f).dispatch(resolution);
        stream << cmd_list.commit()
               << swap_chain.present(ldr_image);
        frame_count += spp_per_dispatch;

        window.pool_event();
    }
    stream << ldr_image.copy_to(host_image.data()) << synchronize();

    stbi_write_png("test_photon_mapping.png", resolution.x, resolution.y, 4, host_image.data(), 0);
}
