/*
 * Tutorial 02: Cornell Box Path Tracer
 *
 * This tutorial teaches how a larger LuisaCompute program is structured:
 * 1. Load geometry and build a ray tracing acceleration structure.
 * 2. Define helper callables for random numbers, ONB construction, and cosine sampling.
 * 3. Implement a path tracing kernel with next event estimation and Russian Roulette.
 * 4. Progressively accumulate samples and tone-map the HDR image.
 * 5. Render offline to PNG or interactively to a window.
 *
 * Why this tutorial matters:
 * Path tracing combines the LuisaCompute runtime, DSL, images, buffers, callables, and hardware
 * ray tracing into one compact renderer. It is a practical template for real GPU applications.
 */

#include <cstdlib>
#include <memory>
#include <optional>
#include <string_view>
#include <array>

#include <stb/stb_image_write.h>

#include <luisa/luisa-compute.h>
#include <luisa/core/clock.h>
#include <luisa/dsl/sugar.h>
#include <luisa/runtime/swapchain.h>

#ifndef ENABLE_DISPLAY
#ifdef LUISA_ENABLE_GUI
#define ENABLE_DISPLAY 1
#else
#define ENABLE_DISPLAY 0
#endif
#endif

#if ENABLE_DISPLAY
#include <luisa/gui/window.h>
#endif

#include "cornell_box.h"
#define TINYOBJLOADER_IMPLEMENTATION
#include "tiny_obj_loader.h"

using namespace luisa;
using namespace luisa::compute;

// Step 0: Define a tiny helper structure for an orthonormal basis (ONB).
// Why: cosine hemisphere samples are easiest to generate in local coordinates, but rays must be
// traced in world space. The ONB converts local sample directions to world directions.
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

    // Step 1: Create the Context and Device.
    // Why: every LuisaCompute application starts by selecting a backend and constructing the object
    // that owns resources, streams, and compiled shaders for that backend.
    // Parse command-line: any non-flag argument is the backend name.
    // If no backend is given, the first installed backend is selected automatically.
    luisa::string backend;
    bool offline = false;
    uint requested_spp = 0u;
    for (int i = 1; i < argc; i++) {
        if (std::string_view{argv[i]} == "--offline") {
            offline = true;
        } else if (std::string_view{argv[i]} == "--spp" && i + 1 < argc) {
            requested_spp = static_cast<uint>(std::atoi(argv[++i]));
        } else if (backend.empty()) {
            backend = argv[i];
        }
    }

    Context context{argv[0]};
    if (backend.empty()) {
        auto const &backends = context.installed_backends();
        if (backends.empty()) {
            LUISA_ERROR("No backends installed.");
            return 1;
        }
        static constexpr luisa::string_view preferred_backends[] = {
            "cuda", "dx", "metal", "vk", "fallback", "cpu", "remote"};
        for (auto preferred : preferred_backends) {
            for (auto const &candidate : backends) {
                if (candidate == preferred && !context.backend_device_names(candidate).empty()) {
                    backend = candidate;
                    break;
                }
            }
            if (!backend.empty()) { break; }
        }
        if (backend.empty()) {
            for (auto const &candidate : backends) {
                if (!context.backend_device_names(candidate).empty()) {
                    backend = candidate;
                    break;
                }
            }
        }
        if (backend.empty()) {
            LUISA_ERROR("No usable backends installed.");
            return 1;
        }
        LUISA_INFO("No backend specified, auto-selected: {}", backend);
    }

    if (!offline && !ENABLE_DISPLAY) {
        LUISA_WARNING("GUI support is disabled in this build. Falling back to --offline mode.");
        offline = true;
    }

    Device device = context.create_device(backend);
    Stream stream = device.create_stream(offline ? StreamTag::COMPUTE : StreamTag::GRAPHICS);

    static constexpr uint2 resolution = make_uint2(1024u, 1024u);
    uint total_spp = offline ? (requested_spp == 0u ? 64u : requested_spp) : requested_spp;
    bool infinite_render = !offline && total_spp == 0u;
    constexpr char output_file[] = "tutorial_02_path_tracing.png";

    LUISA_INFO("Tutorial 02: loading Cornell Box scene...");

    // Step 2: Load the Cornell Box OBJ and upload the geometry.
    // Why: the acceleration structure needs triangle meshes; the OBJ gives us a compact scene that
    // is familiar in rendering literature and small enough for a tutorial.
    tinyobj::ObjReaderConfig obj_config;
    obj_config.triangulate = true;
    obj_config.vertex_color = false;
    tinyobj::ObjReader obj_reader;
    if (!obj_reader.ParseFromString(obj_string, "", obj_config)) {
        luisa::string_view error_message = "unknown error";
        if (auto &&e = obj_reader.Error(); !e.empty()) {
            error_message = e;
        }
        LUISA_ERROR_WITH_LOCATION("Failed to parse Cornell Box OBJ: {}", error_message);
    }
    if (auto &&warning = obj_reader.Warning(); !warning.empty()) {
        LUISA_WARNING_WITH_LOCATION("{}", warning);
    }

    auto &&positions = obj_reader.GetAttrib().vertices;
    luisa::vector<float3> vertices;
    vertices.reserve(positions.size() / 3u);
    for (uint i = 0u; i < positions.size(); i += 3u) {
        vertices.emplace_back(make_float3(positions[i], positions[i + 1u], positions[i + 2u]));
    }

    Buffer<float3> vertex_buffer = device.create_buffer<float3>(vertices.size());
    BindlessArray heap = device.create_bindless_array(64u);
    luisa::vector<Mesh> meshes;
    luisa::vector<Buffer<Triangle>> triangle_buffers;

    stream << vertex_buffer.copy_from(luisa::span{vertices});

    for (auto &&shape : obj_reader.GetShapes()) {
        luisa::vector<uint> indices;
        indices.reserve(shape.mesh.indices.size());
        for (auto index : shape.mesh.indices) {
            indices.emplace_back(static_cast<uint>(index.vertex_index));
        }
        auto triangle_count = static_cast<uint>(indices.size() / 3u);
        auto &triangle_buffer = triangle_buffers.emplace_back(device.create_buffer<Triangle>(triangle_count));
        auto &mesh = meshes.emplace_back(device.create_mesh(vertex_buffer, triangle_buffer));
        auto slot = static_cast<uint>(meshes.size() - 1u);
        heap.emplace_on_update(slot, triangle_buffer);
        stream << triangle_buffer.copy_from(luisa::span{indices})
               << mesh.build();
    }

    // Step 3: Build a top-level acceleration structure.
    // Why: path tracing is dominated by ray-scene intersection. The Accel object lets LuisaCompute
    // and the backend use hardware or optimized software traversal.
    Accel accel = device.create_accel({});
    for (auto &mesh : meshes) {
        accel.emplace_back(mesh, make_float4x4(1.0f));
    }
    stream << heap.update()
           << accel.build();

    // Step 4: Upload per-shape diffuse colors.
    // Why: the Cornell Box uses simple Lambertian materials, which keeps the tutorial focused on
    // light transport rather than a large BSDF system.
    float3 material_data[] = {
        make_float3(0.725f, 0.710f, 0.680f),
        make_float3(0.725f, 0.710f, 0.680f),
        make_float3(0.725f, 0.710f, 0.680f),
        make_float3(0.140f, 0.450f, 0.091f),
        make_float3(0.630f, 0.065f, 0.050f),
        make_float3(0.725f, 0.710f, 0.680f),
        make_float3(0.725f, 0.710f, 0.680f),
        make_float3(0.000f, 0.000f, 0.000f),
    };
    Buffer<float3> materials = device.create_buffer<float3>(8u);
    stream << materials.copy_from(luisa::span{material_data, std::size(material_data)});

    // Step 5: Create helper callables.
    // Why: putting repeated logic into Callables makes kernels smaller, easier to explain, and
    // easier to reuse.
    Callable linear_to_srgb = [](Float3 x) noexcept {
        return saturate(select(1.055f * pow(x, 1.0f / 2.4f) - 0.055f,
                               12.92f * x,
                               x <= 0.00031308f));
    };

    Callable tea = [](UInt v0, UInt v1) noexcept {
        UInt sum = 0u;
        $for (i, 4u) {
            sum += 0x9e3779b9u;
            v0 += ((v1 << 4u) + 0xa341316cu) ^ (v1 + sum) ^ ((v1 >> 5u) + 0xc8013ea4u);
            v1 += ((v0 << 4u) + 0xad90777du) ^ (v0 + sum) ^ ((v0 >> 5u) + 0x7e95761eu);
        };
        return v0;
    };

    Callable lcg = [](UInt &state) noexcept {
        constexpr uint a = 1664525u;
        constexpr uint c = 1013904223u;
        state = a * state + c;
        return cast<float>(state & 0x00ffffffu) * (1.0f / static_cast<float>(0x01000000u));
    };

    Callable make_onb = [](Float3 normal) noexcept {
        Float3 binormal = normalize(ite(
            abs(normal.x) > abs(normal.z),
            make_float3(-normal.y, normal.x, 0.0f),
            make_float3(0.0f, -normal.z, normal.y)));
        Float3 tangent = normalize(cross(binormal, normal));
        return def<Onb>(tangent, binormal, normal);
    };

    Callable cosine_sample_hemisphere = [](Float2 u) noexcept {
        Float r = sqrt(u.x);
        Float phi = 2.0f * constants::pi * u.y;
        return make_float3(r * cos(phi), r * sin(phi), sqrt(max(1.0f - u.x, 0.0f)));
    };

    Callable balanced_heuristic = [](Float pdf_a, Float pdf_b) noexcept {
        return pdf_a / max(pdf_a + pdf_b, 1e-4f);
    };

    Callable generate_ray = [](Float2 pixel_ndc) noexcept {
        constexpr float fov = radians(27.8f);
        constexpr float3 origin = make_float3(-0.01f, 0.995f, 5.0f);
        Float3 pixel = origin + make_float3(pixel_ndc * tan(0.5f * fov), -1.0f);
        return make_ray(origin, normalize(pixel - origin));
    };

    // Step 6: Initialize one random seed per pixel.
    // Why: Monte Carlo rendering needs decorrelated random numbers across the image.
    Kernel2D make_sampler_kernel = [&](ImageUInt seed_image) noexcept {
        set_block_size(16u, 16u, 1u);
        UInt2 pixel = dispatch_id().xy();
        seed_image.write(pixel, make_uint4(tea(pixel.x, pixel.y), 0u, 0u, 0u));
    };

    // Step 7: Define the main path tracing kernel.
    // Why: this kernel expresses the rendering algorithm directly in the LuisaCompute DSL.
    Kernel2D path_trace_kernel = [&](ImageFloat framebuffer, ImageUInt seed_image, AccelVar scene, UInt2 image_resolution) noexcept {
        set_block_size(16u, 16u, 1u);

        UInt2 pixel = dispatch_id().xy();
        UInt state = seed_image.read(pixel).x;
        Float image_extent = min(image_resolution.x, image_resolution.y).cast<float>();
        Float2 jitter = make_float2(lcg(state), lcg(state));
        Float2 ndc = (make_float2(pixel) + jitter) / image_extent * 2.0f - 1.0f;

        Float3 radiance = def(make_float3(0.0f));
        Var<Ray> ray = generate_ray(ndc * make_float2(1.0f, -1.0f));
        Float3 throughput = def(make_float3(1.0f));
        Float previous_bsdf_pdf = def(0.0f);

        constexpr float3 light_position = make_float3(-0.24f, 1.98f, 0.16f);
        constexpr float3 light_u = make_float3(-0.24f, 1.98f, -0.22f) - light_position;
        constexpr float3 light_v = make_float3(0.23f, 1.98f, 0.16f) - light_position;
        constexpr float3 light_emission = make_float3(17.0f, 12.0f, 4.0f);
        Float light_area = length(cross(light_u, light_v));
        Float3 light_normal = normalize(cross(light_u, light_v));

        // Step 7.1: March a path through the scene.
        // Why: every bounce estimates one piece of the light transport equation.
        $for (depth, 8u) {
            Var<TriangleHit> hit = scene.intersect(ray, {});
            reorder_shader_execution();
            $if (hit->miss()) {
                $break;
            };

            Var<Triangle> triangle = heap->buffer<Triangle>(hit.inst).read(hit.prim);
            Float3 p0 = vertex_buffer->read(triangle.i0);
            Float3 p1 = vertex_buffer->read(triangle.i1);
            Float3 p2 = vertex_buffer->read(triangle.i2);
            Float3 position = triangle_interpolate(hit.bary, p0, p1, p2);
            Float3 normal = normalize(cross(p1 - p0, p2 - p0));
            Float cos_out = dot(-ray->direction(), normal);

            $if (cos_out < 1e-4f) {
                $break;
            };

            // Step 7.2: Handle direct hits on the area light.
            // Why: if the sampled path lands on the light, we must add emitted radiance.
            $if (hit.inst == static_cast<uint>(meshes.size() - 1u)) {
                $if (depth == 0u) {
                    radiance += light_emission;
                }
                $else {
                    Float pdf_light = length_squared(position - ray->origin()) / (light_area * cos_out);
                    Float mis_weight = balanced_heuristic(previous_bsdf_pdf, pdf_light);
                    radiance += throughput * light_emission * mis_weight;
                };
                $break;
            };

            Float3 albedo = materials->read(hit.inst);
            Float3 offset_position = offset_ray_origin(position, normal);

            // Step 7.3: Next Event Estimation (NEE).
            // Why: explicitly sampling the light greatly reduces noise compared to waiting for a
            // random BSDF bounce to hit the light by chance.
            Float2 light_sample = make_float2(lcg(state), lcg(state));
            Float3 sampled_light_position = light_position + light_sample.x * light_u + light_sample.y * light_v;
            Float3 shadow_target = offset_ray_origin(sampled_light_position, light_normal);
            Float shadow_distance = distance(offset_position, shadow_target);
            Float3 wi_light = normalize(shadow_target - offset_position);
            Var<Ray> shadow_ray = make_ray(offset_position, wi_light, 0.0f, shadow_distance);
            Bool occluded = scene.intersect_any(shadow_ray, {});
            Float cos_surface = dot(wi_light, normal);
            Float cos_light = -dot(light_normal, wi_light);

            $if (!occluded & cos_surface > 1e-4f & cos_light > 1e-4f) {
                Float pdf_light = shadow_distance * shadow_distance / (light_area * cos_light);
                Float pdf_bsdf = cos_surface * inv_pi;
                Float mis_weight = balanced_heuristic(pdf_light, pdf_bsdf);
                Float3 bsdf = albedo * inv_pi * cos_surface;
                radiance += throughput * bsdf * mis_weight * light_emission / max(pdf_light, 1e-4f);
            };

            // Step 7.4: Sample a cosine-weighted diffuse bounce.
            // Why: cosine sampling matches Lambert's cosine law and reduces variance.
            Var<Onb> onb = make_onb(normal);
            Float2 bsdf_sample = make_float2(lcg(state), lcg(state));
            Float3 local_direction = cosine_sample_hemisphere(bsdf_sample);
            Float cos_in = abs(local_direction.z);
            Float3 world_direction = onb->to_world(local_direction);
            ray = make_ray(offset_position, world_direction);
            previous_bsdf_pdf = cos_in * inv_pi;
            throughput *= albedo;

            // Step 7.5: Russian Roulette termination.
            // Why: long, low-contribution paths waste work. Russian Roulette keeps the estimator
            // unbiased while focusing effort where it matters.
            Float survival = max(dot(throughput, make_float3(0.2126f, 0.7152f, 0.0722f)), 0.05f);
            Float roulette = lcg(state);
            $if (roulette >= survival) {
                $break;
            };
            throughput *= 1.0f / survival;
        };

        seed_image.write(pixel, make_uint4(state, 0u, 0u, 0u));
        $if (any(dsl::isnan(radiance))) {
            radiance = make_float3(0.0f);
        };
        framebuffer.write(pixel, make_float4(clamp(radiance, 0.0f, 30.0f), 1.0f));
    };

    // Step 8: Define accumulation and tone mapping kernels.
    // Why: path tracing produces noisy HDR samples. We average them over time, then convert the
    // result to display-ready sRGB.
    Kernel2D clear_kernel = [](ImageFloat image) noexcept {
        set_block_size(16u, 16u, 1u);
        image.write(dispatch_id().xy(), make_float4(0.0f));
    };

    Kernel2D accumulate_kernel = [](ImageFloat accum, ImageFloat sample) noexcept {
        set_block_size(16u, 16u, 1u);
        UInt2 pixel = dispatch_id().xy();
        accum.write(pixel, accum.read(pixel) + make_float4(sample.read(pixel).xyz(), 1.0f));
    };

    Kernel2D tonemap_kernel = [&](ImageFloat accum, ImageFloat output, Float exposure) noexcept {
        set_block_size(16u, 16u, 1u);
        UInt2 pixel = dispatch_id().xy();
        Float4 hdr = accum.read(pixel);
        Float3 average = hdr.xyz() / max(hdr.w, 1.0f);
        Float3 mapped = linear_to_srgb(clamp(average * exposure, 0.0f, 1.0f));
        output.write(pixel, make_float4(mapped, 1.0f));
    };

    LUISA_INFO("Compiling path tracing shaders...");
    auto make_sampler_shader = device.compile(make_sampler_kernel);
    auto path_trace_shader = device.compile(path_trace_kernel, ShaderOption{.name = "tutorial_path_tracer"});
    auto clear_shader = device.compile(clear_kernel);
    auto accumulate_shader = device.compile(accumulate_kernel);
    auto tonemap_shader = device.compile(tonemap_kernel);

    // Step 9: Allocate rendering images.
    // Why: we keep a current-sample image, a persistent accumulation image, and an output image.
    Image<float> sample_image = device.create_image<float>(PixelStorage::HALF4, resolution);
    Image<float> accum_image = device.create_image<float>(PixelStorage::FLOAT4, resolution);
    Image<uint> seed_image = device.create_image<uint>(PixelStorage::INT1, resolution);
    Image<float> output_image = device.create_image<float>(PixelStorage::BYTE4, resolution);

    std::optional<Image<float>> display_image;
#if ENABLE_DISPLAY
    std::unique_ptr<Window> window;
    std::optional<Swapchain> swapchain;
    if (!offline) {
        window = std::make_unique<Window>("Tutorial 02 - Path Tracing", resolution, false);
        swapchain.emplace(device.create_swapchain(
            stream,
            SwapchainOption{
                .display = window->native_display(),
                .window = window->native_handle(),
                .size = resolution,
                .wants_hdr = false,
                .wants_vsync = false,
                .back_buffer_count = 2,
            }));
        display_image.emplace(device.create_image<float>(swapchain->backend_storage(), resolution));
    }
#endif

    stream << clear_shader(accum_image).dispatch(resolution)
           << make_sampler_shader(seed_image).dispatch(resolution)
           << synchronize();

    // Step 10: Run the render loop.
    // Why: offline mode stops after a known sample budget; interactive mode keeps refining until
    // the user closes the window.
    Clock clock;
    uint frame_index = 0u;
    while (infinite_render || frame_index < total_spp) {
        stream << path_trace_shader(sample_image, seed_image, accel, resolution).dispatch(resolution)
               << accumulate_shader(accum_image, sample_image).dispatch(resolution);

        if (!offline) {
#if ENABLE_DISPLAY
            stream << tonemap_shader(accum_image, *display_image, 1.8f).dispatch(resolution)
                   << swapchain->present(*display_image);
            window->poll_events();
            if (window->should_close() || window->is_key_down(KEY_ESCAPE)) {
                break;
            }
#endif
        }

        frame_index++;
        if (frame_index % 8u == 0u || offline) {
            LUISA_INFO("Rendered {} sample(s) per pixel so far.", frame_index);
        }
    }

    // Step 11: Tone-map the final image and save it in offline mode.
    // Why: PNG is an LDR format, so we convert from accumulated HDR radiance before writing.
    if (offline) {
        luisa::vector<std::array<uint8_t, 4u>> host_image(resolution.x * resolution.y);
        stream << tonemap_shader(accum_image, output_image, 1.8f).dispatch(resolution)
               << output_image.copy_to(luisa::span{host_image})
               << synchronize();
        stbi_write_png(output_file, static_cast<int>(resolution.x), static_cast<int>(resolution.y), 4, host_image.data(), 0);
        LUISA_INFO("Saved offline path traced image to {} after {} spp.", output_file, frame_index);
    } else {
        stream << synchronize();
        LUISA_INFO("Interactive session ended after {} spp in {:.2f} ms.", frame_index, clock.toc());
    }

    return 0;
}
