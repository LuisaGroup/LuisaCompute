#include <cstdint>
#include <iostream>
#include <random>

#include <stb/stb_image_write.h>

#include <luisa/luisa-compute.h>
#include <luisa/dsl/sugar.h>

#include "common/cornell_box.h"

#define TINYOBJLOADER_IMPLEMENTATION
#include "common/tiny_obj_loader.h"

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
    [[nodiscard]] Float3 to_world(Expr<float3> v) const noexcept {
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
        LUISA_INFO("Usage: {} <backend>. <backend>: cuda, dx, cpu, metal", argv[0]);
        exit(1);
    }
    Device device = context.create_device(argv[1]);
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
    Callable tea = [](UInt v0, UInt v1) noexcept {
        UInt s0 = def(0u);
        for (uint n = 0u; n < 4u; n++) {
            s0 += 0x9e3779b9u;
            v0 += ((v1 << 4) + 0xa341316cu) ^ (v1 + s0) ^ ((v1 >> 5u) + 0xc8013ea4u);
            v1 += ((v0 << 4) + 0xad90777du) ^ (v0 + s0) ^ ((v0 >> 5u) + 0x7e95761eu);
        }
        return v0;
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

    float3 grid_min = make_float3(1e8);
    float3 grid_max = make_float3(-1e8);
    auto &&p = obj_reader.GetAttrib().vertices;
    luisa::vector<float3> vertices;
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

    BindlessArray heap = device.create_bindless_array();
    Stream stream = device.create_stream(StreamTag::GRAPHICS);
    Buffer<float3> vertex_buffer = device.create_buffer<float3>(vertices.size());
    stream << vertex_buffer.copy_from(vertices.data());

    luisa::vector<Mesh> meshes;
    luisa::vector<Buffer<Triangle>> triangle_buffers;

    for (auto &&shape : obj_reader.GetShapes()) {
        uint index = static_cast<uint>(meshes.size());
        auto &&t = shape.mesh.indices;
        uint triangle_count = t.size() / 3u;
        LUISA_INFO(
            "Processing shape '{}' at index {} with {} triangle(s).",
            shape.name, index, triangle_count);
        luisa::vector<uint> indices;
        indices.reserve(t.size());
        for (auto i : t) { indices.emplace_back(i.vertex_index); }
        Buffer<Triangle> &triangle_buffer = triangle_buffers.emplace_back(device.create_buffer<Triangle>(triangle_count));
        Mesh &mesh = meshes.emplace_back(device.create_mesh(vertex_buffer, triangle_buffer));
        heap.emplace_on_update(index, triangle_buffer);
        stream << triangle_buffer.copy_from(indices.data())
               << mesh.build();
    }

    Accel accel = device.create_accel();
    for (Mesh &m : meshes) {
        accel.emplace_back(m, make_float4x4(1.0f));
    }
    stream << heap.update()
           << accel.build()
           << synchronize();

    //TODO
    luisa::vector<Material> materials;
    materials.reserve(accel.size());
    materials.emplace_back(Material{make_float3(0.725f, 0.71f, 0.68f), make_float3(0.0f)});// floor
    materials.emplace_back(Material{make_float3(0.725f, 0.71f, 0.68f), make_float3(0.0f)});// ceiling
    materials.emplace_back(Material{make_float3(0.725f, 0.71f, 0.68f), make_float3(0.0f)});// back wall
    materials.emplace_back(Material{make_float3(0.14f, 0.45f, 0.091f), make_float3(0.0f)});// right wall
    materials.emplace_back(Material{make_float3(0.63f, 0.065f, 0.05f), make_float3(0.0f)});// left wall
    materials.emplace_back(Material{make_float3(0.725f, 0.71f, 0.68f), make_float3(0.0f)});// short box
    materials.emplace_back(Material{make_float3(0.725f, 0.71f, 0.68f), make_float3(0.0f)});// tall box
    materials.emplace_back(Material{make_float3(0.0f), make_float3(17.0f, 12.0f, 4.0f)});  // light
    Buffer<Material> material_buffer = device.create_buffer<Material>(materials.size());
    stream << material_buffer.copy_from(materials.data());

    constexpr float3 light_position = make_float3(-0.24f, 1.98f, 0.16f);
    constexpr float3 light_u = make_float3(-0.24f, 1.98f, -0.22f) - light_position;
    constexpr float3 light_v = make_float3(0.23f, 1.98f, 0.16f) - light_position;
    constexpr float3 light_emission = make_float3(17.0f, 12.0f, 4.0f);
    float light_area = length(cross(light_u, light_v));
    float3 light_normal = normalize(cross(light_u, light_v));

    constexpr uint photon_number = 1000000u;
    constexpr uint max_depth = 8u;
    // constexpr auto scale = 1000u;

    luisa::vector<uint> seeds(photon_number);
    std::mt19937 rng{std::random_device{}()};
    for (uint i = 0u; i < photon_number; i++) {
        seeds[i] = rng();
    }
    Buffer<uint> seed_buffer = device.create_buffer<uint>(photon_number);
    Buffer<Photon> photon_buffer = device.create_buffer<Photon>(photon_number * max_depth);
    Buffer<uint> photon_limit_buffer = device.create_buffer<uint>(1u);

    constexpr uint split_per_dim = 128u;
    float3 grid_real_size = grid_max - grid_min;
    float grid_len = min(min(grid_real_size.x / split_per_dim, grid_real_size.y / split_per_dim), grid_real_size.z / split_per_dim);
    uint3 grid_size = make_uint3(uint(grid_real_size.x / grid_len) + 1,
                                 uint(grid_real_size.y / grid_len) + 1,
                                 uint(grid_real_size.z / grid_len) + 1);
    LUISA_INFO("grid_size {} {} {}", grid_size.x, grid_size.y, grid_size.z);
    Buffer<uint> grid_head_buffer = device.create_buffer<uint>(grid_size.x * grid_size.y * grid_size.z);

    LUISA_INFO("grid_len = {}", grid_len);

    stream << seed_buffer.copy_from(seeds.data());

    Kernel1D clear_grid_kernel = [&]() noexcept {
        UInt index = static_cast<UInt>(dispatch_x());
        grid_head_buffer->write(index, ~0u);
    };

    Kernel1D photon_tracing_kernel = [&](AccelVar accel) noexcept {
        // random seed
        UInt state = seed_buffer->read(dispatch_x());
        UInt state2 = dispatch_x();

        // sample light
        Float ux_light = lcg(state2);
        Float uy_light = lcg(state2);
        Float3 light_p = light_position + ux_light * light_u + uy_light * light_v;

        Float3 light_pp = offset_ray_origin(light_p, light_normal);
        Float ux_light_dir = lcg(state);
        Float uy_light_dir = lcg(state);
        Var<Onb> light_onb = make_onb(light_normal);
        Float3 light_dir = light_onb->to_world(cosine_sample_hemisphere(make_float2(ux_light_dir, uy_light_dir)));
        Var<Ray> light_ray = make_ray(light_pp, light_dir);
        Float light_cos = dot(light_dir, light_normal);
        Float light_pdf_pos = 1.0f / light_area;
        Float light_pdf_dir = light_cos * inv_pi;
        // auto power = def(light_emission * light_cos / light_pdf_pos / light_pdf_dir) =>
        Float3 power = def(light_emission / (light_pdf_pos * inv_pi));

        $for (depth, max_depth) {
            // trace
            Var<TriangleHit> hit = accel.intersect(light_ray, {});
            $if (hit->miss()) { $break; };
            // $if(hit.inst == 0 & hit.prim == 0) { $break; };
            // TODO
            $if (hit.inst == static_cast<uint>(meshes.size() - 1u)) {
                $break;
            };

            Var<Triangle> triangle = heap->buffer<Triangle>(hit.inst).read(hit.prim);
            Float3 p0 = vertex_buffer->read(triangle.i0);
            Float3 p1 = vertex_buffer->read(triangle.i1);
            Float3 p2 = vertex_buffer->read(triangle.i2);
            Float3 p = triangle_interpolate(hit.bary, p0, p1, p2);
            Float3 n = normalize(cross(p1 - p0, p2 - p0));
            Float cos_wi = dot(-light_ray->direction(), n);
            $if (cos_wi < 1e-4f) { $break; };

            Var<Material> material = material_buffer->read(hit.inst);

            // store photons
            UInt index = photon_limit_buffer->atomic(0).fetch_add(1u);

            UInt3 grid_index = make_uint3((p - grid_min) / grid_len);
            UInt grid_index_m = grid_index.x * grid_size.y * grid_size.z + grid_index.y * grid_size.z + grid_index.z;
            UInt link_head = grid_head_buffer->atomic(grid_index_m).exchange(index);

            Var<Photon> photon{p, power, -light_ray->direction(), link_head};
            photon_buffer->write(index, photon);

            // sample BxDF
            Var<Onb> onb = make_onb(n);
            Float ux = lcg(state);
            Float uy = lcg(state);
            Float3 new_dir = onb->to_world(cosine_sample_hemisphere(make_float2(ux, uy)));
            Float cos_dir = dot(new_dir, n);
            Float pdf_dir = cos_dir * inv_pi;
            Float3 pp = offset_ray_origin(p, n);
            light_ray = make_ray(pp, new_dir);
            // power *= material.albedo * inv_pi * cos_dir / pdf_dir =>
            power *= material.albedo;

            // rr
            Float rr_prob = 0.9f;
            Float rr_check = lcg(state);
            $if (rr_check >= rr_prob) { $break; };
            power *= 1.0f / rr_prob;
        };
    };

    constexpr float photon_radius = 0.007f;
    // constexpr auto kNN_k = 20u;
    // auto heap_buffer = device.create_buffer<heapNode>(resolution.x * resolution.y * kNN_k);

    Callable get_index_from_point = [&](Float3 p) noexcept {
        return make_uint3(max((p - grid_min) / grid_len + 0.5f, 0.f));
    };

    Callable get_index_merge = [&](UInt x, UInt y, UInt z) noexcept {
        return x * grid_size.y * grid_size.z + y * grid_size.z + z;
    };

    auto density_estimation_radius = [&](UInt2 coord, Float3 p, Float3 dir, Float r, Var<Material> material) noexcept {
        Float3 radiance = def(make_float3(0.0f));
        UInt3 p_index = get_index_from_point(p);
        UInt photon_sum = def(0);

        $for (x, ite(p_index.x == 0, 0u, p_index.x - 1), min(p_index.x + 1, grid_size.x)) {
            $for (y, ite(p_index.y == 0, 0u, p_index.y - 1), min(p_index.y + 1, grid_size.y)) {
                $for (z, ite(p_index.z == 0, 0u, p_index.z - 1), min(p_index.z + 1, grid_size.z)) {
                    UInt grid_index_m = get_index_merge(x, y, z);
                    UInt photon_index = grid_head_buffer->read(grid_index_m);
                    $while (photon_index != ~0u) {
                        photon_sum += 1;
                        Var<Photon> photon = photon_buffer->read(photon_index);
                        Float dis = distance(Float3{photon.position}, p);
                        $if (dis < r) {
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
        Float3 radiance = def(make_float3(0.0f));
        Float radius = def(photon_radius);

        Var<TriangleHit> hit = accel.intersect(ray, {});
        // $if(!hit->miss() & (hit.inst != 0 | hit.prim != 0)) {
        $if (!hit->miss()) {
            $if (hit.inst == static_cast<uint>(meshes.size() - 1u)) {
                // radiance = light_emission / light_area;
            }
            $else {
                Var<Triangle> triangle = heap->buffer<Triangle>(hit.inst).read(hit.prim);
                Float3 p0 = vertex_buffer->read(triangle.i0);
                Float3 p1 = vertex_buffer->read(triangle.i1);
                Float3 p2 = vertex_buffer->read(triangle.i2);
                Float3 p = triangle_interpolate(hit.bary, p0, p1, p2);
                Float3 n = normalize(cross(p1 - p0, p2 - p0));
                Var<Material> material = material_buffer->read(hit.inst);
                Float cos_wi = dot(-ray->direction(), n);

                $if (cos_wi > 1e-4f) {
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
    Kernel2D photon_gathering_kernel = [&](ImageFloat image, ImageUInt seed_image, AccelVar accel, UInt2 resolution) noexcept {
        UInt2 coord = dispatch_id().xy();
        Float frame_size = min(resolution.x, resolution.y).cast<float>();
        UInt state = seed_image.read(coord).x;
        Float rx = lcg(state);
        Float ry = lcg(state);
        Float2 pixel = (make_float2(coord) + make_float2(rx, ry)) / frame_size * 2.0f - 1.0f;
        Var<Ray> ray = generate_ray(pixel * make_float2(1.0f, -1.0f));
        Float radius = def(photon_radius);
        Var<TriangleHit> hit = accel.intersect(ray, {});
        Float3 radiance = def(make_float3(0.0f));

        $if (!hit->miss()) {
            $if (hit.inst == static_cast<uint>(meshes.size() - 1u)) {
                radiance = light_emission;
            }
            $else {
                Var<Triangle> triangle = heap->buffer<Triangle>(hit.inst).read(hit.prim);
                Float3 p0 = vertex_buffer->read(triangle.i0);
                Float3 p1 = vertex_buffer->read(triangle.i1);
                Float3 p2 = vertex_buffer->read(triangle.i2);
                Float3 p = triangle_interpolate(hit.bary, p0, p1, p2);
                Float3 n = normalize(cross(p1 - p0, p2 - p0));
                Float cos_wi = dot(-ray->direction(), n);
                $if (cos_wi > 1e-4f) {
                    Var<Material> material = material_buffer->read(hit.inst);
                    //direct illumination
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
                    Float3 albedo = material.albedo;
                    $if (!occluded & cos_wi_light > 1e-4f & cos_light > 1e-4f) {
                        Float pdf_light = (d_light * d_light) / (light_area * cos_light);
                        Float pdf_bsdf = cos_wi_light * inv_pi;
                        Float mis_weight = balanced_heuristic(pdf_light, pdf_bsdf);
                        Float3 bsdf = albedo * inv_pi * cos_wi_light;
                        radiance += bsdf * mis_weight * light_emission / max(pdf_light, 1e-4f);
                    };

                    //indirect illumination
                    Var<Onb> onb = make_onb(n);
                    Float ux = lcg(state);
                    Float uy = lcg(state);
                    Float3 new_direction = onb->to_world(cosine_sample_hemisphere(make_float2(ux, uy)));
                    ray = make_ray(pp, new_direction);
                    Float new_cos = dot(new_direction, n);
                    Float pdf_dir = inv_pi * new_cos;

                    Float3 indirect = indirect_illumination(coord, ray, accel);
                    // radiance += material.albedo * inv_pi * new_cos * indirect / pdf_dir =>
                    radiance += material.albedo * indirect;
                };
            };
        };

        seed_image.write(coord, make_uint4(state));
        $if (any(dsl::isnan(radiance))) { radiance = make_float3(0.0f); };
        image.write(dispatch_id().xy(), make_float4(clamp(radiance, 0.0f, 30.0f), 1.0f));
    };

    Kernel2D accumulate_kernel = [&](ImageFloat accum_image, ImageFloat curr_image) noexcept {
        UInt2 p = dispatch_id().xy();
        Float4 accum = accum_image.read(p);
        Float3 curr = curr_image.read(p).xyz();
        accum_image.write(p, accum + make_float4(curr, 1.f));
    };

    Callable linear_to_srgb = [](Var<float3> x) noexcept {
        return clamp(select(1.055f * pow(x, 1.0f / 2.4f) - 0.055f,
                            12.92f * x,
                            x <= 0.00031308f),
                     0.0f, 1.0f);
    };

    Kernel2D hdr2ldr_kernel = [&](ImageFloat hdr_image, ImageFloat ldr_image, Float scale) noexcept {
        UInt2 coord = dispatch_id().xy();
        Float4 hdr = hdr_image.read(coord);
        Float3 ldr = linear_to_srgb(hdr.xyz() / hdr.w * scale);
        ldr_image.write(coord, make_float4(ldr, 1.0f));
    };

    Kernel2D clear_kernel = [](ImageFloat image) noexcept {
        image.write(dispatch_id().xy(), make_float4(0.0f));
    };

    Kernel2D make_sampler_kernel = [&](ImageUInt seed_image) noexcept {
        UInt2 p = dispatch_id().xy();
        UInt state = tea(p.x, p.y);
        seed_image.write(p, make_uint4(state));
    };

    auto clear_shader = device.compile(clear_kernel);
    auto make_sampler_shader = device.compile(make_sampler_kernel);
    auto clear_grid_shader = device.compile(clear_grid_kernel);
    auto photon_tracing_shader = device.compile(photon_tracing_kernel);
    auto photon_gathering_shader = device.compile(photon_gathering_kernel);
    auto accumulate_shader = device.compile(accumulate_kernel);
    auto hdr2ldr_shader = device.compile(hdr2ldr_kernel);

    static constexpr uint2 resolution = make_uint2(1024u);
    Image<uint> seed_image = device.create_image<uint>(PixelStorage::INT1, resolution);
    Image<float> framebuffer = device.create_image<float>(PixelStorage::HALF4, resolution);
    Image<float> accum_image = device.create_image<float>(PixelStorage::FLOAT4, resolution);
    luisa::vector<std::array<uint8_t, 4u>> host_image(resolution.x * resolution.y);

    stream << clear_shader(accum_image).dispatch(resolution)
           << make_sampler_shader(seed_image).dispatch(resolution)
           << clear_grid_shader().dispatch(grid_size.x * grid_size.y * grid_size.z)
           << photon_tracing_shader(accel).dispatch(photon_number)
           << synchronize();
    LUISA_INFO("Photon tracing done");

    uint frame_count = 0;

    Window window{"Display", resolution.x, resolution.y};
    Swapchain swap_chain = device.create_swapchain(
        stream,
        SwapchainOption{
            .display = window.native_display(),
            .window = window.native_handle(),
            .size = resolution,
            .wants_hdr = false,
            .wants_vsync = false,
            .back_buffer_count = 2,
        });
    Image<float> ldr_image = device.create_image<float>(swap_chain.backend_storage(), resolution);
    Clock clk;
    while (!window.should_close()) {
        CommandList cmd_list = CommandList::create();
        static constexpr uint spp_per_dispatch = 1u;
        for (uint i = 0u; i < spp_per_dispatch; i++) {
            cmd_list
                << photon_gathering_shader(framebuffer, seed_image, accel, resolution).dispatch(resolution)
                << accumulate_shader(accum_image, framebuffer).dispatch(resolution);
        }
        cmd_list
            << hdr2ldr_shader(accum_image, ldr_image, 1.0f).dispatch(resolution);
        stream << cmd_list.commit()
               << swap_chain.present(ldr_image);
        frame_count += spp_per_dispatch;

        window.poll_events();
    }
    stream << ldr_image.copy_to(host_image.data()) << synchronize();

    stbi_write_png("test_photon_mapping.png", resolution.x, resolution.y, 4, host_image.data(), 0);
}
