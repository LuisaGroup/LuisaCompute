#include <iostream>
#include <luisa/core/clock.h>
#include <luisa/core/logging.h>
#include <luisa/runtime/context.h>
#include <luisa/runtime/device.h>
#include <luisa/runtime/stream.h>
#include <luisa/runtime/event.h>
#include <luisa/runtime/swapchain.h>
#include <luisa/dsl/sugar.h>
#include <luisa/runtime/rtx/accel.h>
#include "common/cornell_box.h"
#include <stb/stb_image_write.h>
#include <stb/stb_image.h>
#include <luisa/gui/window.h>
#include <luisa/backends/ext/denoiser_ext.h>
#define TINYOBJLOADER_IMPLEMENTATION
#include "common/tiny_obj_loader.h"
#define TINYEXR_IMPLEMENTATION
#include "common/tinyexr.h"
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

LUISA_STRUCT(Material, albedo, emission) {};

LUISA_STRUCT(Onb, tangent, binormal, normal) {
    [[nodiscard]] auto to_world(Expr<float3> v) const noexcept {
        return v.x * tangent + v.y * binormal + v.z * normal;
    }
};

int main(int argc, char *argv[]) {

    log_level_verbose();

    Context context{argv[0]};
    Device device = context.create_device("cuda");
    auto denoiser_ext = device.extension<DenoiserExt>();
    auto resolution = make_uint2(1024u);

    //params
    uint optix_examples = 1;//0:cornellbox pathtracing, 1:soane temporal example, 2:aovs example
    //all for online cornell box
    bool seperate_usage = 1; //check the fullpipeline(denoise) or sperately call functions. the temporal denoise mode can only run when this is true
    bool temporal = 1;       // 1: add camera movement, 0: static camera and accumulate samples
    bool flow_validation = 0;// use when temporal on: 1: validation for the flow calculation, 0: normal
    auto channel_count = 4;  //processing buffer channel, for testing.
    LUISA_INFO("{}",getenv("OPTIX_INCLUDE_DIR"));
    luisa::filesystem::path optix= getenv("OPTIX_INCLUDE_DIR");
    auto optix_path = optix.parent_path() / "SDK";//your optix sdk path, for finding the examples

    if (optix_examples) {
        auto datapath = optix_path / "optixDenoiser" / "motiondata";
        luisa::filesystem::path path;
        if (optix_examples == 1) {
            luisa::string beauty = "soane-Beauty-";
            luisa::string id = "001";
            luisa::string suffix = ".exr";
            path = datapath/(beauty + id + suffix);
        }
        if (optix_examples == 2) {
            path = optix_path / "optixDenoiser" / "data" / "beauty.exr";
        }
        std::ifstream str(path.c_str());
        LUISA_ASSERT(static_cast<bool>(str), "{}: image does not exists.", path.string().c_str());
        auto image_width = 0;
        auto image_height = 0;
        auto image_channels = 0;
        float *beauty_file = nullptr;
        const char *err = nullptr;
        LoadEXR(&beauty_file, &image_width, &image_height, path.string().c_str(), &err);
        resolution.x = uint(image_width);
        resolution.y = uint(image_height);
    }
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
        return cast<float>(state & 0x00ffffffu) *
               (1.0f / static_cast<float>(0x01000000u));
    };
    auto lcg_host = [](uint &state) noexcept {
        constexpr auto lcg_a = 1664525u;
        constexpr auto lcg_c = 1013904223u;
        state = lcg_a * state + lcg_c;
        return cast<float>(state & 0x00ffffffu) *
               (1.0f / static_cast<float>(0x01000000u));
    };
    Callable make_onb = [](const Float3 &normal) noexcept {
        auto binormal = normalize(ite(
            abs(normal.x) > abs(normal.z),
            make_float3(-normal.y, normal.x, 0.0f),
            make_float3(0.0f, -normal.z, normal.y)));
        auto tangent = normalize(cross(binormal, normal));
        return def<Onb>(tangent, binormal, normal);
    };

    auto origin = make_float3(-0.01f, 0.995f, 5.0f);
    Callable generate_ray = [](Float2 p, Float3 _origin) noexcept {
        static constexpr auto fov = radians(27.8f);
        auto pixel = _origin + make_float3(p * tan(0.5f * fov), -1.0f);
        auto direction = normalize(pixel - _origin);
        return make_ray(_origin, direction);
    };
    Callable pos_to_pix = [](Float3 pos, Float3 _origin) noexcept {
        static constexpr auto fov = radians(27.8f);
        auto direction = normalize(pos - _origin);
        auto pixel = direction / -direction.z;
        auto p = pixel.xy() / tan(0.5f * fov);
        return p;
    };

    Callable cosine_sample_hemisphere = [](Float2 u) noexcept {
        auto r = sqrt(u.x);
        auto phi = 2.0f * constants::pi * u.y;
        return make_float3(r * cos(phi), r * sin(phi), sqrt(1.0f - u.x));
    };

    Callable balanced_heuristic = [](Float pdf_a, Float pdf_b) noexcept {
        return pdf_a / max(pdf_a + pdf_b, 1e-4f);
    };

    static constexpr auto spp_per_dispatch = 1u;

    Kernel2D raytracing_kernel = [&](ImageFloat image, ImageUInt seed_image, AccelVar accel, UInt2 resolution, Float3 _origin) noexcept {
        set_block_size(8u, 8u, 1u);
        auto coord = dispatch_id().xy();
        auto frame_size = min(resolution.x, resolution.y).cast<float>();
        auto state = seed_image.read(coord).x;
        auto rx = lcg(state);
        auto ry = lcg(state);
        auto pixel = (make_float2(coord) + make_float2(rx, ry)) / frame_size * 2.0f - 1.0f;
        auto radiance = def(make_float3(0.0f));
        $for (i, spp_per_dispatch) {
            auto ray = generate_ray(pixel * make_float2(1.0f, -1.0f), _origin);
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
                auto hit = accel.trace_closest(ray);
                $if (hit->miss()) { $break; };
                auto triangle = heap->buffer<Triangle>(hit.inst).read(hit.prim);
                auto p0 = vertex_buffer->read(triangle.i0);
                auto p1 = vertex_buffer->read(triangle.i1);
                auto p2 = vertex_buffer->read(triangle.i2);
                auto p = hit->interpolate(p0, p1, p2);
                auto n = normalize(cross(p1 - p0, p2 - p0));
                auto cos_wi = dot(-ray->direction(), n);
                $if (cos_wi < 1e-4f) { $break; };
                auto material = material_buffer->read(hit.inst);

                // hit light
                $if (hit.inst == static_cast<uint>(meshes.size() - 1u)) {
                    $if (depth == 0u) {
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
                auto shadow_ray = make_ray(offset_ray_origin(pp, n), wi_light, 0.f, d_light);
                auto occluded = accel.trace_any(shadow_ray);
                auto cos_wi_light = dot(wi_light, n);
                auto cos_light = -dot(light_normal, wi_light);
                $if (!occluded & cos_wi_light > 1e-4f & cos_light > 1e-4f) {
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
    Kernel2D aux_kernel = [&](ImageFloat normal, ImageFloat albedo, ImageFloat flow, ImageUInt seed_image, AccelVar accel, UInt2 resolution, Float3 _origin, Float3 last_origin) noexcept {
        set_block_size(8u, 8u, 1u);
        auto coord = dispatch_id().xy();
        auto frame_size = min(resolution.x, resolution.y).cast<float>();
        auto state = seed_image.read(coord).x;
        auto rx = lcg(state);
        auto ry = lcg(state);
        auto pixel = (make_float2(coord) + make_float2(rx, ry)) / frame_size * 2.0f - 1.0f;
        auto radiance = def(make_float3(0.0f));
        auto ray = generate_ray(pixel * make_float2(1.0f, -1.0f), _origin);
        $for (depth, 1u) {

            // trace
            auto hit = accel.trace_closest(ray);
            $if (hit->miss()) {
                normal.write(coord, make_float4(0.0f, 0.0f, 0.0f, 1.0f));
                albedo.write(coord, make_float4(0.0f, 0.0f, 0.0f, 1.0f));
                flow.write(coord, make_float4(0.0f, 0.0f, 0.0f, 1.0f));
                $break;
            };
            auto triangle = heap->buffer<Triangle>(hit.inst).read(hit.prim);
            auto p0 = vertex_buffer->read(triangle.i0);
            auto p1 = vertex_buffer->read(triangle.i1);
            auto p2 = vertex_buffer->read(triangle.i2);
            auto pos = hit->interpolate(p0, p1, p2);
            auto n = normalize(cross(p1 - p0, p2 - p0));
            auto material = material_buffer->read(hit.inst);
            auto prev_p = pos_to_pix(pos, last_origin);
            auto prev_pixel = prev_p * make_float2(1.0f, -1.0f);
            auto prev_coord = (prev_pixel + 1.0f) / 2.0f * frame_size;

            normal.write(coord, make_float4(n, 1.0f));
            albedo.write(coord, make_float4(material.albedo, 1.0f));
            flow.write(coord, make_float4(make_float2(coord) + make_float2(rx, ry) - prev_coord, 0.f, 1.f));
            $break;
        };
        //seed_image.write(coord, make_uint4(state));
    };
    Kernel2D accumulate_kernel = [&](ImageFloat accum_image, ImageFloat curr_image) noexcept {
        auto p = dispatch_id().xy();
        auto accum = accum_image.read(p);
        auto curr = curr_image.read(p).xyz();
        accum_image.write(p, accum + make_float4(curr, 1.f));
    };
    Kernel2D combine_kernel = [&](ImageFloat image1, ImageFloat image2, ImageFloat output) noexcept {
        auto p = dispatch_id().xy();
        auto v1 = image1.read(p);
        auto v2 = image2.read(p);
        //diagnal
        auto condition = p.y * 1.f / resolution.y + p.x * 1.f / resolution.x > 1.f;
        //auto condition = p.y * 1.f / resolution.y > 0.8f;
        output.write(p, ite(condition, v1, v2));
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

    Kernel2D hdr2ldr_kernel = [&](ImageFloat hdr_image, ImageFloat ldr_image, Float scale, Bool is_hdr) noexcept {
        //        Shared<float> s1{13u};
        //        Shared<float> s2{1024u};
        //        s2[thread_x()] = 1.f;
        //        sync_block();
        auto coord = dispatch_id().xy();
        auto hdr = hdr_image.read(coord);
        auto ldr = hdr.xyz() / hdr.w * scale;
        $if (!is_hdr) {
            ldr = linear_to_srgb(ldr);
        };
        ldr_image.write(coord, make_float4(ldr, 1.0f));
    };
    Kernel2D convert_kernel = [&](ImageFloat image, BufferFloat buffer, UInt in_channel_count, UInt out_channel_count) noexcept {
        auto coord = dispatch_id().xy();
        auto pix = image.read(coord);
        $for (channel, out_channel_count) {
            auto index = coord.x + coord.y * resolution.x;
            buffer.write(index * out_channel_count + channel, pix[channel]);
        };
    };
    Kernel2D deconvert_kernel = [&](BufferFloat buffer, ImageFloat image, UInt in_channel_count) noexcept {
        auto coord = dispatch_id().xy();
        Float4 pix = make_float4(1.0f);
        $for (channel, in_channel_count) {
            auto index = coord.x + coord.y * resolution.x;
            pix[channel] = buffer.read(index * in_channel_count + channel);
        };
        image.write(coord, pix);
    };
    Kernel2D flow_apply_kernel = [&](ImageFloat prev_image, ImageFloat flow, ImageFloat output) noexcept {
        auto coord = dispatch_id().xy();
        auto origin_pix = floor(make_float2(coord) - flow.read(coord).xy());
        $if ((origin_pix.x < 0.f) | (origin_pix.x >= resolution.x) | (origin_pix.y < 0.f) | (origin_pix.y >= resolution.y)) {
            output.write(coord, make_float4(0.f, 0.f, 0.f, 1.f));
        }
        $else {
            output.write(coord, prev_image.read(make_uint2(origin_pix)));
        };
    };

    auto clear_shader = device.compile(clear_kernel);
    auto hdr2ldr_shader = device.compile(hdr2ldr_kernel);
    auto accumulate_shader = device.compile(accumulate_kernel);
    auto aux_shader = device.compile(aux_kernel);
    auto raytracing_shader = device.compile(raytracing_kernel);
    auto make_sampler_shader = device.compile(make_sampler_kernel);
    auto combine_shader = device.compile(combine_kernel);
    auto image_to_buf = device.compile(convert_kernel);
    auto buf_to_image = device.compile(deconvert_kernel);
    auto apply_flow = device.compile(flow_apply_kernel);

    auto framebuffer = device.create_image<float>(PixelStorage::HALF4, resolution);
    auto accum_image = device.create_image<float>(PixelStorage::FLOAT4, resolution);
    std::vector<std::array<uint8_t, 4u>> host_image(resolution.x * resolution.y);
    CommandList cmd_list;
    auto seed_image = device.create_image<uint>(PixelStorage::INT1, resolution);
    cmd_list << clear_shader(accum_image).dispatch(resolution)
             << make_sampler_shader(seed_image).dispatch(resolution);

    Window window{"path tracing", resolution};
    auto swap_chain{device.create_swapchain(
        window.native_handle(),
        stream,
        resolution,
        false, false, 3)};
    auto combined_image = device.create_image<float>(PixelStorage::FLOAT4, resolution);
    auto prev_image = device.create_image<float>(PixelStorage::FLOAT4, resolution);
    auto normal_image = device.create_image<float>(PixelStorage::FLOAT4, resolution);
    auto albedo_image = device.create_image<float>(PixelStorage::FLOAT4, resolution);
    auto hdr_image = device.create_image<float>(PixelStorage::FLOAT4, resolution);
    auto denoised_image = device.create_image<float>(PixelStorage::FLOAT4, resolution);
    auto ldr_image = device.create_image<float>(swap_chain.backend_storage(), resolution);
    auto flow_image = device.create_image<float>(PixelStorage::FLOAT4, resolution);
    auto glossy_image = device.create_image<float>(PixelStorage::FLOAT4, resolution);
    auto diffuse_image = device.create_image<float>(PixelStorage::FLOAT4, resolution);
    auto specular_image = device.create_image<float>(PixelStorage::FLOAT4, resolution);

    auto hdr_buffer = device.create_buffer<float>(hdr_image.view().size_bytes() / 4 * channel_count / sizeof(float));
    auto denoised_buffer = device.create_buffer<float>(denoised_image.view().size_bytes() / 4 * channel_count / sizeof(float));
    auto normal_buffer = device.create_buffer<float>(normal_image.view().size_bytes() / 4 * channel_count / sizeof(float));
    auto albedo_buffer = device.create_buffer<float>(albedo_image.view().size_bytes() / 4 * channel_count / sizeof(float));
    auto flow_buffer = device.create_buffer<float>(flow_image.view().size_bytes() / 4 * channel_count / sizeof(float));
    auto glossy_buffer = device.create_buffer<float>(glossy_image.view().size_bytes() / 4 * channel_count / sizeof(float));
    auto diffuse_buffer = device.create_buffer<float>(diffuse_image.view().size_bytes() / 4 * channel_count / sizeof(float));
    auto specular_buffer = device.create_buffer<float>(specular_image.view().size_bytes() / 4 * channel_count / sizeof(float));
    if (optix_examples) {
        std::vector<std::array<float, 4u>> host_image(resolution.x * resolution.y);

        auto image_width = 0;
        auto image_height = 0;
        if (optix_examples == 1) {
            auto datapath = optix_path / "optixDenoiser" / "motiondata";
            luisa::string beauty_path = "soane-Beauty-";
            luisa::string normal_path = "soane-Normal-";
            luisa::string albedo_path = "soane-BSDF-";
            luisa::string flow_path = "soane-Flow-";
            luisa::string id = "000";
            luisa::string suffix = ".exr";
            DenoiserExt::DenoiserInput data;
            DenoiserExt::DenoiserMode mode{.temporal = true};
            data.beauty = &hdr_buffer;
            data.normal = &normal_buffer;
            data.albedo = &albedo_buffer;
            data.flow = &flow_buffer;

            for (int i = 1; i <= 20; ++i) {
                id[1] = '0' + i / 10;
                id[2] = '0' + i % 10;
                float *beauty_file = nullptr;
                float *normal_file = nullptr;
                float *albedo_file = nullptr;
                float *flow_file = nullptr;
                const char *err = nullptr;
                LoadEXR(&beauty_file, &image_width, &image_height, (datapath / (beauty_path + id + suffix)).string().c_str(), &err);
                LoadEXR(&normal_file, &image_width, &image_height, (datapath / (normal_path + id + suffix)).string().c_str(), &err);
                LoadEXR(&albedo_file, &image_width, &image_height, (datapath / (albedo_path + id + suffix)).string().c_str(), &err);
                LoadEXR(&flow_file, &image_width, &image_height, (datapath / (flow_path + id + suffix)).string().c_str(), &err);
                cmd_list << hdr_image.copy_from(beauty_file)
                         << normal_image.copy_from(normal_file)
                         << albedo_image.copy_from(albedo_file)
                         << flow_image.copy_from(flow_file);
                if (i == 1)
                    cmd_list << clear_shader(flow_image).dispatch(resolution);
                cmd_list << image_to_buf(hdr_image, hdr_buffer, 4, channel_count).dispatch(resolution)
                         << image_to_buf(normal_image, normal_buffer, 4, channel_count).dispatch(resolution)
                         << image_to_buf(albedo_image, albedo_buffer, 4, channel_count).dispatch(resolution)
                         << image_to_buf(flow_image, flow_buffer, 4, channel_count).dispatch(resolution);

                stream << cmd_list.commit();
                if (i == 1) {
                    denoiser_ext->init(stream, mode, data, resolution);
                }
                denoiser_ext->process(stream, data);
                denoiser_ext->get_result(stream, denoised_buffer);
                cmd_list << buf_to_image(denoised_buffer, denoised_image, channel_count).dispatch(resolution);
                cmd_list << hdr2ldr_shader(denoised_image, ldr_image, 0.001f, swap_chain.backend_storage() != PixelStorage::BYTE4).dispatch(resolution);
                stream << cmd_list.commit()
                       << denoised_image.copy_to(host_image.data())
                       << synchronize();
                stream << cmd_list.commit() << swap_chain.present(ldr_image);
                window.poll_events();

                using namespace std::chrono_literals;
                std::this_thread::sleep_for(400ms);

                luisa::string output_file = "soane-output-";
                SaveEXR(
                    reinterpret_cast<float *>(host_image.data()),
                    resolution.x,
                    resolution.y,
                    4,                         // num components
                    static_cast<int32_t>(true),// save_as_fp16
                    (output_file + id + suffix).c_str(),
                    &err);
            }
        } else if (optix_examples == 2) {
            auto datapath= optix_path / "SDK" / "optixDenoiser" / "data";
            auto beauty_path = optix_path / "beauty.exr";
            auto normal_path = optix_path / "normal.exr";
            auto albedo_path = optix_path / "albedo.exr";
            auto glossy_path = optix_path / "glossy.exr";
            auto diffuse_path = optix_path / "diffuse.exr";
            auto specular_path = optix_path / "specular.exr";
            DenoiserExt::DenoiserInput data;
            DenoiserExt::DenoiserMode mode{.temporal = false};
            data.beauty = &hdr_buffer;
            data.normal = &normal_buffer;
            data.albedo = &albedo_buffer;
            luisa::vector<Buffer<float> *> aovs;
            aovs.push_back(&glossy_buffer);
            aovs.push_back(&diffuse_buffer);
            aovs.push_back(&specular_buffer);
            data.aovs = aovs.data();
            data.aov_size = aovs.size();
            float *beauty_file = nullptr;
            float *normal_file = nullptr;
            float *albedo_file = nullptr;
            float *glossy_file = nullptr;
            float *diffuse_file = nullptr;
            float *specular_file = nullptr;
            const char *err = nullptr;
            LoadEXR(&beauty_file, &image_width, &image_height, beauty_path.string().c_str(), &err);
            LoadEXR(&normal_file, &image_width, &image_height, normal_path.string().c_str(), &err);
            LoadEXR(&albedo_file, &image_width, &image_height, albedo_path.string().c_str(), &err);
            LoadEXR(&glossy_file, &image_width, &image_height, glossy_path.string().c_str(), &err);
            LoadEXR(&diffuse_file, &image_width, &image_height, diffuse_path.string().c_str(), &err);
            LoadEXR(&specular_file, &image_width, &image_height, specular_path.string().c_str(), &err);
            cmd_list << hdr_image.copy_from(beauty_file)
                     << normal_image.copy_from(normal_file)
                     << albedo_image.copy_from(albedo_file)
                     << glossy_image.copy_from(glossy_file)
                     << diffuse_image.copy_from(diffuse_file)
                     << specular_image.copy_from(specular_file);
            cmd_list << image_to_buf(hdr_image, hdr_buffer, 4, channel_count).dispatch(resolution)
                     << image_to_buf(normal_image, normal_buffer, 4, channel_count).dispatch(resolution)
                     << image_to_buf(albedo_image, albedo_buffer, 4, channel_count).dispatch(resolution)
                     << image_to_buf(diffuse_image, diffuse_buffer, 4, channel_count).dispatch(resolution)
                     << image_to_buf(glossy_image, glossy_buffer, 4, channel_count).dispatch(resolution)
                     << image_to_buf(specular_image, specular_buffer, 4, channel_count).dispatch(resolution);
            stream << cmd_list.commit();
            denoiser_ext->init(stream, mode, data, resolution);
            denoiser_ext->process(stream, data);
            auto save_image = [&](int index, const char *file_name) {
                denoiser_ext->get_result(stream, denoised_buffer, index);
                cmd_list << buf_to_image(denoised_buffer, denoised_image, channel_count).dispatch(resolution);
                cmd_list << hdr2ldr_shader(denoised_image, ldr_image, 0.001f, swap_chain.backend_storage() != PixelStorage::BYTE4).dispatch(resolution);
                if (index >= 0)
                    cmd_list << accumulate_shader(accum_image, denoised_image).dispatch(resolution);
                stream << cmd_list.commit()
                       << denoised_image.copy_to(host_image.data())
                       << synchronize();
                stream << cmd_list.commit() << swap_chain.present(ldr_image);
                window.poll_events();
                using namespace std::chrono_literals;
                std::this_thread::sleep_for(400ms);
                SaveEXR(
                    reinterpret_cast<float *>(host_image.data()),
                    resolution.x,
                    resolution.y,
                    4,                         // num components
                    static_cast<int32_t>(true),// save_as_fp16
                    file_name,
                    &err);
            };
            save_image(-1, "beauty.exr");
            save_image(0, "glossy.exr");
            save_image(1, "diffuse.exr");
            save_image(2, "specular.exr");
            Kernel2D adjust_kernel = [](ImageFloat image) {
                auto coord = dispatch_id().xy();
                image.write(coord, make_float4(image.read(coord).xyz(), 1.0f));
            };
            auto adjust = device.compile(adjust_kernel);
            stream << cmd_list.commit()
                   << adjust(accum_image).dispatch(resolution)
                   << accum_image.copy_to(host_image.data())
                   << synchronize();
            stream << cmd_list.commit() << swap_chain.present(ldr_image);
            window.poll_events();
            using namespace std::chrono_literals;
            std::this_thread::sleep_for(400ms);
            SaveEXR(
                reinterpret_cast<float *>(host_image.data()),
                resolution.x,
                resolution.y,
                4,                         // num components
                static_cast<int32_t>(true),// save_as_fp16
                "sum.exr",
                &err);
        }
        denoiser_ext->destroy(stream);

        return 0;
    }

    auto last_time = 0.0;
    auto frame_count = 0u;
    Clock clock;

    DenoiserExt::DenoiserInput data;
    DenoiserExt::DenoiserMode mode{.temporal = temporal};
    data.beauty = &hdr_buffer;
    data.normal = &normal_buffer;
    data.albedo = &albedo_buffer;
    if (temporal)
        data.flow = &flow_buffer;

    bool initialize_flag = true;
    cmd_list << aux_shader(normal_image, albedo_image, flow_image, seed_image, accel, resolution, origin, origin).dispatch(resolution);
    cmd_list << image_to_buf(normal_image, normal_buffer, 4, channel_count).dispatch(resolution)
             << image_to_buf(albedo_image, albedo_buffer, 4, channel_count).dispatch(resolution)
             << image_to_buf(flow_image, flow_buffer, 4, channel_count).dispatch(resolution);

    uint state = 42u;

    while (!window.should_close()) {
        cmd_list << raytracing_shader(framebuffer, seed_image, accel, resolution, origin)
                        .dispatch(resolution)
                 << accumulate_shader(accum_image, framebuffer)
                        .dispatch(resolution)
                 << hdr2ldr_shader(accum_image, hdr_image, 1.0f, true).dispatch(resolution);
        cmd_list << image_to_buf(hdr_image, hdr_buffer, 4, channel_count).dispatch(resolution);
        stream << cmd_list.commit() << synchronize();
        if (flow_validation) {
            cmd_list << apply_flow(prev_image, flow_image, denoised_image).dispatch(resolution);
        } else {
            if (!seperate_usage) {
                denoiser_ext->denoise(stream, resolution, hdr_buffer, denoised_buffer, normal_buffer, albedo_buffer, nullptr, 0);
            } else {
                if (seperate_usage) {
                    if (initialize_flag) {
                        initialize_flag = false;
                        denoiser_ext->init(stream, mode, data, resolution);
                    }
                    denoiser_ext->process(stream, data);
                    denoiser_ext->get_result(stream, denoised_buffer);
                }
            }
            cmd_list << buf_to_image(denoised_buffer, denoised_image, channel_count).dispatch(resolution);
        }
        cmd_list << combine_shader(hdr_image, denoised_image, combined_image).dispatch(resolution);
        cmd_list << hdr2ldr_shader(combined_image, ldr_image, 1.0f, swap_chain.backend_storage() != PixelStorage::BYTE4).dispatch(resolution);
        stream << cmd_list.commit() << swap_chain.present(ldr_image);
        window.poll_events();
        auto dt = clock.toc() - last_time;
        frame_count += spp_per_dispatch;
        using namespace std::chrono_literals;
        std::this_thread::sleep_for(400ms);
        LUISA_INFO("time: {} ms", dt);
        last_time = clock.toc();
        if (temporal) {
            auto new_origin = origin;
            auto v = 0.01;
            new_origin.z += -0.1 * v + 2 * v * lcg_host(state) - v;
            new_origin.x += 2 * v * lcg_host(state) - v;
            new_origin.y += 2 * v * lcg_host(state) - v;
            //new_origin.x += v;
            cmd_list << prev_image.copy_from(accum_image);
            cmd_list << clear_shader(accum_image).dispatch(resolution);
            cmd_list << aux_shader(normal_image, albedo_image, flow_image, seed_image, accel, resolution, new_origin, origin).dispatch(resolution);
            cmd_list << image_to_buf(normal_image, normal_buffer, 4, channel_count).dispatch(resolution)
                     << image_to_buf(albedo_image, albedo_buffer, 4, channel_count).dispatch(resolution)
                     << image_to_buf(flow_image, flow_buffer, 4, channel_count).dispatch(resolution);
            origin = new_origin;
        }
    }
    if (seperate_usage)
        denoiser_ext->destroy(stream);
    stream << cmd_list.commit()
           << ldr_image.copy_to(host_image.data())
           << synchronize();

    LUISA_INFO("FPS: {}", frame_count / clock.toc() * 1000);
    stbi_write_png("test_denoiser.png", resolution.x, resolution.y, 4, host_image.data(), 0);
}
