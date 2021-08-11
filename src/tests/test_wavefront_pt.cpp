//
// Created by Mike Smith on 2021/6/23.
//

#include <iostream>

#include <runtime/context.h>
#include <runtime/device.h>
#include <runtime/stream.h>
#include <runtime/event.h>
#include <dsl/sugar.h>
#include <rtx/accel.h>
#include <tests/fake_device.h>
#include <tests/cornell_box.h>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <tests/stb_image_write.h>

#define TINYOBJLOADER_IMPLEMENTATION
#include <tests/tiny_obj_loader.h>

using namespace luisa;
using namespace luisa::compute;

struct MaterialSOA {
    Buffer<float3> color;
    Buffer<bool> emissive;
};

struct LightSOA {
    Buffer<float3> position;
    Buffer<float3> u;
    Buffer<float3> v;
    Buffer<float3> emission;
};

struct RaySOA {
    Buffer<float3> origin;
    Buffer<float3> direction;
};

struct IntersectionSOA {
    Buffer<float3> position;
    Buffer<float3> normal;
    Buffer<float3> wo;
    Buffer<float> distance;
    Buffer<uint> inst_id;
    Buffer<uint> prim_id;
};

struct RayStateSOA {
    Buffer<float3> radiance;
    Buffer<float> pdf_bsdf;
    Buffer<float3> throughput;
    Buffer<uint> random_state;
};

struct LightSampleSOA {
    Buffer<float3> wi;
    Buffer<float3> Li;
    Buffer<float> pdf;
};

struct RayQueue {
    Buffer<uint> indices;
    Buffer<uint> count;
};

LUISA_BINDING_GROUP(MaterialSOA, color, emissive)
LUISA_BINDING_GROUP(LightSOA, position, u, v, emission)
LUISA_BINDING_GROUP(RaySOA, origin, direction)
LUISA_BINDING_GROUP(IntersectionSOA, position, normal, wo, distance, inst_id, prim_id)
LUISA_BINDING_GROUP(RayStateSOA, radiance, pdf_bsdf, throughput, random_state)
LUISA_BINDING_GROUP(LightSampleSOA, wi, Li, pdf)
LUISA_BINDING_GROUP(RayQueue, indices, count)

struct Camera {
    float3 center;
    float3 front;
    float3 up;
    float fov;
};

struct Onb {
    float3 tangent;
    float3 binormal;
    float3 normal;
};

LUISA_STRUCT(Camera, center, front, up, fov)
LUISA_STRUCT(Onb, tangent, binormal, normal)

int main(int argc, char *argv[]) {
//
//    log_level_verbose();
//
//    Context context{argv[0]};
//
//#if defined(LUISA_BACKEND_METAL_ENABLED)
//    auto device = context.create_device("metal", 1u);
//#elif defined(LUISA_BACKEND_DX_ENABLED)
//    auto device = context.create_device("dx");
//#else
//    auto device = FakeDevice::create(context);
//#endif
//
//    // load the Cornell Box scene
//    tinyobj::ObjReaderConfig obj_reader_config;
//    obj_reader_config.triangulate = true;
//    obj_reader_config.vertex_color = false;
//    tinyobj::ObjReader obj_reader;
//    if (!obj_reader.ParseFromString(obj_string, "", obj_reader_config)) {
//        std::string_view error_message = "unknown error.";
//        if (auto &&e = obj_reader.Error(); !e.empty()) { error_message = e; }
//        LUISA_ERROR_WITH_LOCATION("Failed to load OBJ file: {}", error_message);
//    }
//    if (auto &&e = obj_reader.Warning(); !e.empty()) {
//        LUISA_WARNING_WITH_LOCATION("{}", e);
//    }
//
//    auto &&p = obj_reader.GetAttrib().vertices;
//    std::vector<float3> vertices;
//    vertices.reserve(p.size() / 3u);
//    for (auto i = 0u; i < p.size(); i += 3u) {
//        vertices.emplace_back(float3{
//            p[i + 0u],
//            p[i + 1u],
//            p[i + 2u]});
//    }
//    LUISA_INFO(
//        "Loaded mesh with {} shape(s) and {} vertices.",
//        obj_reader.GetShapes().size(), vertices.size());
//
//    auto heap = device.create_heap();
//    auto stream = device.create_stream();
//    auto vertex_buffer = device.create_buffer<float3>(vertices.size());
//    stream << vertex_buffer.copy_from(vertices.data());
//    std::vector<Mesh> meshes;
//    for (auto &&shape : obj_reader.GetShapes()) {
//        auto index = static_cast<uint>(meshes.size());
//        auto &&t = shape.mesh.indices;
//        auto triangle_count = t.size() / 3u;
//        LUISA_INFO(
//            "Processing shape '{}' at index {} with {} triangle(s).",
//            shape.name, index, triangle_count);
//        auto &&mesh = meshes.emplace_back(device.create_mesh());
//        std::vector<uint> indices;
//        indices.reserve(t.size());
//        for (auto i : t) { indices.emplace_back(i.vertex_index); }
//        auto triangle_buffer = heap.create_buffer<Triangle>(index, triangle_count);
//        stream << triangle_buffer.copy_from(indices.data())
//               << commit()// buffers from heap are not hazard-tracked
//               << mesh.build(AccelBuildHint::FAST_TRACE, vertex_buffer, triangle_buffer);
//    }
//
//    std::vector<uint64_t> instances;
//    std::vector<float4x4> transforms;
//    for (auto &&m : meshes) {
//        instances.emplace_back(m.handle());
//        transforms.emplace_back(make_float4x4(1.0f));
//    }
//    auto accel = device.create_accel();
//    stream << accel.build(AccelBuildHint::FAST_TRACE, instances, transforms);
//
//    constexpr auto light_position = make_float3(-0.24f, 1.98f, 0.16f);
//    Light light{
//        .position = make_float3(-0.24f, 1.98f, 0.16f),
//        .u = make_float3(-0.24f, 1.98f, -0.22f) - light_position,
//        .v = make_float3(0.23f, 1.98f, 0.16f) - light_position,
//        .emission = make_float3(17.0f, 12.0f, 4.0f)};
//
//    Camera camera{
//        .center = make_float3(-0.01f, 0.995f, 5.0f),
//        .front = make_float3(0.0f, 0.0f, -1.0f),
//        .up = make_float3(0.0f, 1.0f, 0.0f),
//        .fov = radians(27.8f)};
//
//    std::vector<Material> materials;
//    materials.reserve(instances.size());
//    materials.emplace_back(Material{{0.725f, 0.71f, 0.68f}, false});// floor
//    materials.emplace_back(Material{{0.725f, 0.71f, 0.68f}, false});// ceiling
//    materials.emplace_back(Material{{0.725f, 0.71f, 0.68f}, false});// back wall
//    materials.emplace_back(Material{{0.14f, 0.45f, 0.091f}, false});// right wall
//    materials.emplace_back(Material{{0.63f, 0.065f, 0.05f}, false});// left wall
//    materials.emplace_back(Material{{0.725f, 0.71f, 0.68f}, false});// short box
//    materials.emplace_back(Material{{0.725f, 0.71f, 0.68f}, false});// tall box
//    materials.emplace_back(Material{{17.0f, 12.0f, 4.00f}, true});  // light
//    auto material_buffer = device.create_buffer<Material>(materials.size());
//    stream << material_buffer.copy_from(materials.data());
//
//    Callable linear_to_srgb = [](Var<float3> x) noexcept {
//        return clamp(select(1.055f * pow(x, 1.0f / 2.4f) - 0.055f,
//                            12.92f * x,
//                            x <= 0.00031308f),
//                     0.0f, 1.0f);
//    };
//
//    Callable float3_from_array = [](ArrayFloat<3> v) noexcept {
//        return make_float3(v[0], v[1], v[2]);
//    };
//
//    Callable array_from_float3 = [](Float3 v) noexcept {
//        ArrayFloat<3> a{v.x, v.y, v.z};
//        return a;
//    };
//
//    Callable tea = [](UInt v0, UInt v1) noexcept {
//        Var s0 = 0u;
//        for (auto n = 0u; n < 4u; n++) {
//            s0 += 0x9e3779b9u;
//            v0 += ((v1 << 4) + 0xa341316cu) ^ (v1 + s0) ^ ((v1 >> 5u) + 0xc8013ea4u);
//            v1 += ((v0 << 4) + 0xad90777du) ^ (v0 + s0) ^ ((v0 >> 5u) + 0x7e95761eu);
//        }
//        return v0;
//    };
//
//    Kernel2D initialize_random_state_kernel = [&](BufferVar<RayState> states) noexcept {
//        Var p = dispatch_id().xy();
//        Var index = p.y * dispatch_size_x() + p.x;
//        states[index].random_state = tea(p.x, p.y);
//    };
//
//    Kernel2D initialize_ray_state_kernel = [&](BufferVar<RayState> states,
//                                               BufferUInt indices,
//                                               BufferUInt count) noexcept {
//        Var index = dispatch_x();
//        indices[index] = index;
//        states[index].radiance = array_from_float3(make_float3(0.0f));
//        states[index].throughput = array_from_float3(make_float3(1.0f));
//        $if(index == 0u) { count[0u] = dispatch_size_x() * dispatch_size_y(); };
//    };
//
//    Kernel1D reset_ray_count_kernel = [](BufferUInt c0, BufferUInt c1) noexcept {
//        c0[0] = 0u;
//        c1[0] = 0u;
//    };
//
//    Kernel2D make_sampler_kernel = [&](ImageUInt state_image) noexcept {
//        Var p = dispatch_id().xy();
//        Var state = tea(p.x, p.y);
//        state_image.write(p, make_uint4(state));
//    };
//
//    auto lcg = [](Ref<uint> state) noexcept {
//        constexpr auto lcg_a = 1664525u;
//        constexpr auto lcg_c = 1013904223u;
//        state = lcg_a * state + lcg_c;
//        return cast<float>(state & 0x00ffffffu) * (1.0f / static_cast<float>(0x01000000u));
//    };
//
//    Callable make_onb = [](Float3 normal) noexcept {
//        Var binormal = normalize(ite(
//            abs(normal.x) > abs(normal.z),
//            make_float3(-normal.y, normal.x, 0.0f),
//            make_float3(0.0f, -normal.z, normal.y)));
//        Var tangent = normalize(cross(binormal, normal));
//        Var<Onb> onb{tangent, binormal, normal};
//        return onb;
//    };
//
//    Callable transform_to_world = [](Var<Onb> onb, Float3 v) noexcept {
//        return v.x * onb.tangent + v.y * onb.binormal + v.z * onb.normal;
//    };
//
//    Callable generate_ray = [](Var<Camera> camera, Float2 p) noexcept {
//        Var camera_z = -camera.front;
//        Var camera_y = camera.up;
//        Var camera_x = normalize(cross(camera_y, camera_z));
//        Var camera_to_world = make_float3x3(camera_x, camera_y, camera_z);
//        Var pp = make_float3(p * tan(0.5f * camera.fov), -1.0f);
//        Var pixel = camera.center + camera_to_world * pp;
//        Var direction = normalize(pixel - camera.center);
//        return make_ray(camera.center, direction);
//    };
//
//    Kernel2D generate_rays_kernel = [&](BufferVar<Ray> rays,
//                                        BufferVar<RayState> states,
//                                        Var<Camera> camera) noexcept {
//        Var coord = dispatch_id().xy();
//        Var index = coord.y * dispatch_size_x() + coord.x;
//
//        // generate random number
//        Var state = states[index].random_state;
//        Var rx = lcg(state);
//        Var ry = lcg(state);
//        states[index].random_state = state;
//
//        // generate ray
//        Var frame_size = min(dispatch_size().x, dispatch_size().y).cast<float>();
//        Var p = (make_float2(coord) + make_float2(rx, ry)) / frame_size * 2.0f - 1.0f;
//        Var camera_z = -camera.front;
//        Var camera_y = camera.up;
//        Var camera_x = normalize(cross(camera_y, camera_z));
//        Var camera_to_world = make_float3x3(camera_x, camera_y, camera_z);
//        Var pp = make_float3(p * tan(0.5f * camera.fov), -1.0f);
//        Var pixel = camera.center + camera_to_world * pp;
//        Var direction = normalize(pixel - camera.center);
//        rays[index] = make_ray(camera.center, direction);
//    };
//
//    Kernel1D trace_closest_kernel = [&](BufferVar<Ray> rays,
//                                        BufferUInt ray_count,
//                                        BufferVar<Intersection> hits,
//                                        AccelVar accel,
//                                        HeapVar heap) noexcept {
//        Var index = dispatch_x();
//        $if(index < ray_count[0]) {
//            Var ray = rays[index];
//            Var hit = accel.trace_closest(ray);
//            $if(miss(hit)) {
//                hits[index].distance = -1.0f;
//            }
//            $else {
//                Var triangle = heap.buffer<Triangle>(hit.inst).read(hit.prim);
//                Var p0 = vertex_buffer[triangle.i0];
//                Var p1 = vertex_buffer[triangle.i1];
//                Var p2 = vertex_buffer[triangle.i2];
//                Var p = interpolate(hit, p0, p1, p2);
//                Var n = normalize(cross(p1 - p0, p2 - p0));
//                Var<Intersection> isect;
//                hits[index].position[0] = p.x;
//                hits[index].position[1] = p.y;
//                hits[index].position[2] = p.z;
//                hits[index].inst_id = hit.inst;
//                hits[index].normal[0] = n.x;
//                hits[index].normal[1] = n.y;
//                hits[index].normal[2] = n.z;
//                hits[index].prim_id = hit.prim;
//                hits[index].wo[0] = -ray.direction[0];
//                hits[index].wo[1] = -ray.direction[1];
//                hits[index].wo[2] = -ray.direction[2];
//                hits[index].distance = distance(origin(ray), p);
//            };
//        };
//    };
//
//    Callable cosine_sample_hemisphere = [](Float2 u) noexcept {
//        Var r = sqrt(u.x);
//        Var phi = 2.0f * constants::pi * u.y;
//        return make_float3(r * cos(phi), r * sin(phi), sqrt(1.0f - u.x));
//    };
//
//    Callable balanced_heuristic = [](Float pdf_a, Float pdf_b) noexcept {
//        return pdf_a / max(pdf_a + pdf_b, 1e-4f);
//    };
//
//    Kernel1D sample_light_kernel = [&](BufferVar<RayState> states,
//                                       BufferUInt ray_indices,
//                                       BufferUInt ray_count,
//                                       BufferVar<Intersection> isects,
//                                       BufferVar<LightSample> light_samples,
//                                       Var<Light> light,
//                                       AccelVar accel) noexcept {
//        Var index = dispatch_x();
//        $if(index < ray_count[0]) {
//
//            Var<LightSample> sample;
//            sample.emission[0] = 0.0f;
//            sample.emission[1] = 0.0f;
//            sample.emission[2] = 0.0f;
//            sample.pdf = 0.0f;
//
//            // compute sample
//            Var isect = isects[index];
//            $if(isect.distance > 0.0f) {
//                // generate random numbers
//                Var ray_index = ray_indices[index];
//                Var state = states[ray_index].random_state;
//                Var ux_light = lcg(state);
//                Var uy_light = lcg(state);
//                states[ray_index].random_state = state;
//                // sample light
//                Var p = float3_from_array(isect.position);
//                Var n = float3_from_array(isect.normal);
//                Var p_light = light.position + ux_light * light.u + uy_light * light.v;
//                Var d_light = distance(p, p_light);
//                Var wi_light = normalize(p_light - p);
//                Var light_normal = normalize(cross(light.u, light.v));
//                Var cos_light = -dot(light_normal, wi_light);
//                $if(cos_light > 1e-4f) {
//                    Var shadow_ray = make_ray_robust(p, n, wi_light, d_light - 1e-3f);
//                    Var occluded = accel.trace_any(shadow_ray);
//                    Var light_area = length(cross(light.u, light.v));
//                    $if(!occluded) {
//                        sample.wi = wi_light;
//                        sample.emission = array_from_float3(light.emission);
//                        sample.pdf = (d_light * d_light) / (light_area * cos_light);
//                    };
//                };
//            };
//            light_samples[index] = sample;
//        };
//    };
//
//    Kernel1D evaluate_material_kernel = [&](BufferVar<Intersection> isects,
//                                            BufferVar<LightSample> light_sampels,
//                                            BufferVar<RayState> states,
//                                            BufferUInt ray_indices,
//                                            BufferUInt ray_count,
//                                            BufferVar<Ray> next_rays,
//                                            BufferUInt next_ray_indices,
//                                            BufferUInt next_ray_count,
//                                            BufferVar<Material> materials) noexcept {
//        Var index = dispatch_x();
//        $if(index < ray_count[0]) {
//            Var ray_index = ray_indices[index];
//            Var ray_state = states[ray_index];
//            Var radiance = float3_from_array(ray_state.radiance);
//            Var isect = isects[index];
//            $if(isect.distance > 0.0f) {
//                Var throughput = float3_from_array(ray_state.throughput);
//                Var material = materials[isect.inst_id];
//                Var color = float3_from_array(material.color);
//                $if(material.emissive) {// hits light
//                    radiance += throughput * color;
//                    throughput = make_float3(0.0f);
//                }
//                $else {// hits material
//                    Var random_state = ray_state.random_state;
//                    Var n = float3_from_array(isect.normal);
//                    Var wo = float3_from_array(isect.wo);
//                    Var cos_wo = dot(n, wo);
//                    $if(cos_wo < 1e-4f) {// hits back face
//                        throughput = make_float3(0.0f);
//                    }
//                    $else {// hits front face
//                        Var light_sample = light_sampels[index];
//                        // compute light contribution
//
//                        // sample material
//                        Var f = make_float3();
//                        Var pdf = 0.0f;
//                    };
//                };
//            }
//            $else{
//
//            };
//        };
//    };
//
//    Kernel2D raytracing_kernel = [&](ImageFloat image, ImageUInt state_image, AccelVar accel, Var<Light> light, Var<Camera> camera) noexcept {
//        set_block_size(8u, 8u, 1u);
//
//        Var coord = dispatch_id().xy();
//        Var frame_size = min(dispatch_size().x, dispatch_size().y).cast<float>();
//        Var state = state_image.read(coord).x;
//        Var rx = lcg(state);
//        Var ry = lcg(state);
//        Var pixel = (make_float2(coord) + make_float2(rx, ry)) / frame_size * 2.0f - 1.0f;
//        Var ray = generate_ray(camera, pixel * make_float2(1.0f, -1.0f));
//        Var radiance = make_float3(0.0f);
//        Var beta = make_float3(1.0f);
//        Var pdf_bsdf = 0.0f;
//
//        auto light_area = length(cross(light.u, light.v));
//        auto light_normal = normalize(cross(light.u, light.v));
//
//        for (auto depth : range(10u)) {
//
//            // trace
//            Var hit = accel.trace_closest(ray);
//            if_(miss(hit), break_);
//            Var triangle = heap.buffer<Triangle>(hit.inst).read(hit.prim);
//            Var p0 = vertex_buffer[triangle.i0];
//            Var p1 = vertex_buffer[triangle.i1];
//            Var p2 = vertex_buffer[triangle.i2];
//            Var p = interpolate(hit, p0, p1, p2);
//            Var n = normalize(cross(p1 - p0, p2 - p0));
//            Var cos_wi = dot(-direction(ray), n);
//            if_(cos_wi < 1e-4f, break_);
//            Var material = material_buffer[hit.inst];
//            Var material_color = make_float3(material.color[0], material.color[1], material.color[2]);
//
//            // hit light
//            if_(material.emissive, [&] {
//                if_(depth == 0u, [&] {
//                    radiance += material_color;
//                }).else_([&] {
//                    Var pdf_light = length_squared(p - origin(ray)) / (light_area * cos_wi);
//                    Var mis_weight = balanced_heuristic(pdf_bsdf, pdf_light);
//                    radiance += mis_weight * beta * material_color;
//                });
//                break_();
//            });
//
//            // sample light
//            Var ux_light = lcg(state);
//            Var uy_light = lcg(state);
//            Var p_light = light.position + ux_light * light.u + uy_light * light.v;
//            Var d_light = distance(p, p_light);
//            Var wi_light = normalize(p_light - p);
//            Var shadow_ray = make_ray_robust(p, n, wi_light, d_light - 1e-3f);
//            Var occluded = accel.trace_any(shadow_ray);
//            Var cos_wi_light = dot(wi_light, n);
//            Var cos_light = -dot(light_normal, wi_light);
//            if_(!occluded && cos_wi_light > 1e-4f && cos_light > 1e-4f, [&] {
//                Var pdf_light = (d_light * d_light) / (light_area * cos_light);
//                Var pdf_bsdf = cos_wi_light * inv_pi;
//                Var mis_weight = balanced_heuristic(pdf_light, pdf_bsdf);
//                Var bsdf = material_color * inv_pi * cos_wi_light;
//                radiance += beta * bsdf * mis_weight * light.emission / max(pdf_light, 1e-4f);
//            });
//
//            // sample BSDF
//            Var onb = make_onb(n);
//            Var ux = lcg(state);
//            Var uy = lcg(state);
//            Var new_direction = transform_to_world(onb, cosine_sample_hemisphere(make_float2(ux, uy)));
//            ray = make_ray_robust(p, n, new_direction);
//            beta *= material_color;
//            pdf_bsdf = cos_wi * inv_pi;
//
//            // rr
//            Var l = dot(make_float3(0.212671f, 0.715160f, 0.072169f), beta);
//            if_(l == 0.0f, break_);
//            Var q = max(l, 0.05f);
//            Var r = lcg(state);
//            if_(r >= q, break_);
//            beta *= 1.0f / q;
//        }
//        state_image.write(coord, make_uint4(state));
//        Var old = image.read(coord);
//        if_(isnan(radiance.x) || isnan(radiance.y) || isnan(radiance.z), [&] { radiance = make_float3(0.0f); });
//        Var t = 1.0f / (old.w + 1.0f);
//        Var color = lerp(old.xyz(), clamp(radiance, 0.0f, 30.0f), t);
//        image.write(coord, make_float4(color, old.w + 1.0f));
//    };
//
//    Callable aces_tonemapping = [](Float3 x) noexcept {
//        static constexpr auto a = 2.51f;
//        static constexpr auto b = 0.03f;
//        static constexpr auto c = 2.43f;
//        static constexpr auto d = 0.59f;
//        static constexpr auto e = 0.14f;
//        return clamp((x * (a * x + b)) / (x * (c * x + d) + e), 0.0f, 1.0f);
//    };
//
//    Kernel2D clear_kernel = [](ImageFloat image) noexcept {
//        image.write(dispatch_id().xy(), make_float4(0.0f));
//    };
//
//    Kernel2D hdr2ldr_kernel = [&](ImageFloat hdr_image, ImageFloat ldr_image, Float scale) noexcept {
//        Var coord = dispatch_id().xy();
//        Var hdr = hdr_image.read(coord);
//        Var ldr = linear_to_srgb(aces_tonemapping(hdr.xyz() * scale));
//        ldr_image.write(coord, make_float4(ldr, 1.0f));
//    };
//
//    auto clear_shader = device.compile(clear_kernel);
//    auto hdr2ldr_shader = device.compile(hdr2ldr_kernel);
//    auto raytracing_shader = device.compile(raytracing_kernel);
//    auto make_sampler_shader = device.compile(make_sampler_kernel);
//
//    static constexpr auto width = 1024u;
//    static constexpr auto height = 1024u;
//    auto state_image = device.create_image<uint>(PixelStorage::INT1, width, height);
//    auto ldr_image = device.create_image<float>(PixelStorage::BYTE4, width, height);
//    auto hdr_image = device.create_image<float>(PixelStorage::FLOAT4, width, height);
//    std::vector<uint8_t> pixels(width * height * 4u);
//
//    Clock clock;
//    clock.tic();
//    static constexpr auto spp = 64u;
//    static constexpr auto spp_per_dispatch = 4u;
//    static constexpr auto dispatch_count = spp / spp_per_dispatch;
//    stream << clear_shader(hdr_image).dispatch(width, height)
//           << make_sampler_shader(state_image).dispatch(width, height);
//    for (auto d = 0u; d < dispatch_count; d++) {
//        auto command_buffer = stream.command_buffer();
//        for (auto i = 0u; i < spp_per_dispatch; i++) {
//            command_buffer << raytracing_shader(hdr_image, state_image, accel, light, camera).dispatch(width, height);
//        }
//        command_buffer << commit();
//        LUISA_INFO("Progress: {}/{}", d + 1u, dispatch_count);
//    }
//    stream << hdr2ldr_shader(hdr_image, ldr_image, 1.0f).dispatch(width, height)
//           << ldr_image.copy_to(pixels.data())
//           << synchronize();
//    auto time = clock.toc();
//    LUISA_INFO("Time: {} ms", time);
//    stbi_write_png("test_wavefront_path_tracing.png", width, height, 4, pixels.data(), 0);
}
