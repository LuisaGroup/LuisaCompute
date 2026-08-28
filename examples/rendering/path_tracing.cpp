// Path Tracing Renderer for Cornell Box Scene
// Implements a physically-based path tracer with Multiple Importance Sampling (MIS)
// for direct lighting. Features include:
// - Unidirectional path tracing with BSDF sampling
// - Next Event Estimation (NEE) for direct light sampling
// - Russian Roulette for path termination
// - Cosine-weighted hemisphere sampling for diffuse BSDF
// - Progressive rendering with real-time display

#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <memory>
#include <optional>
#include <string_view>
#include <luisa/backends/ext/dx_hdr_ext.hpp>
#include <stb/stb_image_write.h>

#include "common/reference_compare.h"
#include "common/path_tracing_sample_plan.h"
#include "rendering/path_tracing_test.h"

#include <luisa/luisa-compute.h>
#include <luisa/dsl/sugar.h>
#include <luisa/gui/window.h>

#include "cornell_box.h"
#include <luisa/dsl/sugar.h>
#define TINYOBJLOADER_IMPLEMENTATION
#include "tiny_obj_loader.h"

using namespace luisa;
using namespace luisa::compute;

// Orthonormal Basis (ONB) for shading coordinate system
// Used to transform vectors between local shading space and world space
struct Onb {
    float3 tangent;
    float3 binormal;
    float3 normal;
};

LUISA_STRUCT(Onb, tangent, binormal, normal) {
    // Transform vector from local shading space to world space
    [[nodiscard]] Float3 to_world(Expr<float3> v) const noexcept {
        return v.x * tangent + v.y * binormal + v.z * normal;
    }
};

luisa::ref::PathTracingTestResult luisa::ref::run_path_tracing_test(
    Device &device, const PathTracingTestOptions &opts) {

    // Load the Cornell Box scene from embedded OBJ string
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

    // Extract vertex positions from OBJ data
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

    // Create bindless array for accessing triangle data in shaders
    BindlessArray heap = device.create_bindless_array(65535);
    Stream stream = device.create_stream(opts.offline ? StreamTag::COMPUTE : StreamTag::GRAPHICS);
    Buffer<float3> vertex_buffer = device.create_buffer<float3>(vertices.size());
    stream << vertex_buffer.copy_from(luisa::span{vertices});

    // Build meshes for each shape in the scene
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

    // Build RTX acceleration structure
    Accel accel = device.create_accel({});
    for (Mesh &m : meshes) {
        accel.emplace_back(m, make_float4x4(1.0f));
    }
    stream << heap.update()
           << accel.build();

    // Material definitions for Cornell Box (diffuse albedos)
    // Constant materials{
    //     make_float3(0.725f, 0.710f, 0.680f),// floor
    //     make_float3(0.725f, 0.710f, 0.680f),// ceiling
    //     make_float3(0.725f, 0.710f, 0.680f),// back wall
    //     make_float3(0.140f, 0.450f, 0.091f),// right wall (green)
    //     make_float3(0.630f, 0.065f, 0.050f),// left wall (red)
    //     make_float3(0.725f, 0.710f, 0.680f),// short box
    //     make_float3(0.725f, 0.710f, 0.680f),// tall box
    //     make_float3(0.000f, 0.000f, 0.000f),// light (emissive, not used directly)
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

    // Convert linear RGB to sRGB with proper gamma correction
    Callable linear_to_srgb = [&](Var<float3> x) noexcept {
        return saturate(select(1.055f * pow(x, 1.0f / 2.4f) - 0.055f,
                               12.92f * x,
                               x <= 0.00031308f));
    };

    // TEA (Tiny Encryption Algorithm) for random number generation
    Callable tea = [](UInt v0, UInt v1) noexcept {
        set_name("tea");
        UInt s0 = def(0u);
        for (uint n = 0u; n < 4u; n++) {
            s0 += 0x9e3779b9u;
            v0 += ((v1 << 4) + 0xa341316cu) ^ (v1 + s0) ^ ((v1 >> 5u) + 0xc8013ea4u);
            v1 += ((v0 << 4) + 0xad90777du) ^ (v0 + s0) ^ ((v0 >> 5u) + 0x7e95761eu);
        }
        return v0;
    };

    // Initialize random seed for each pixel using TEA
    Kernel2D make_sampler_kernel = [&](ImageUInt seed_image) noexcept {
        set_name("make_sampler_kernel");
        UInt2 p = dispatch_id().xy();
        UInt state = tea(p.x, p.y);
        seed_image.write(p, make_uint4(state));
    };

    // Linear Congruential Generator (LCG) for pseudo-random numbers
    Callable lcg = [](UInt &state) noexcept {
        set_name("lcg");
        constexpr uint lcg_a = 1664525u;
        constexpr uint lcg_c = 1013904223u;
        state = lcg_a * state + lcg_c;
        return cast<float>(state & 0x00ffffffu) *
               (1.0f / static_cast<float>(0x01000000u));
    };

    // Construct Orthonormal Basis (ONB) from surface normal
    // Uses the standard method for building a tangent frame
    Callable make_onb = [](const Float3 &normal) noexcept {
        set_name("make_onb");
        // Choose binormal direction based on normal's dominant axis
        Float3 binormal = normalize(ite(
            abs(normal.x) > abs(normal.z),
            make_float3(-normal.y, normal.x, 0.0f),
            make_float3(0.0f, -normal.z, normal.y)));
        Float3 tangent = normalize(cross(binormal, normal));
        return def<Onb>(tangent, binormal, normal);
    };

    // Generate primary ray from camera through pixel
    // Uses pinhole camera model with specified FOV
    Callable generate_ray = [](Float2 p) noexcept {
        set_name("generate_ray");
        static constexpr float fov = radians(27.8f);
        static constexpr float3 origin = make_float3(-0.01f, 0.995f, 5.0f);
        Float3 pixel = origin + make_float3(p * tan(0.5f * fov), -1.0f);
        Float3 direction = normalize(pixel - origin);
        return make_ray(origin, direction);
    };

    // Cosine-weighted hemisphere sampling for diffuse BSDF
    // Produces samples proportional to cos(theta) for importance sampling
    Callable cosine_sample_hemisphere = [](Float2 u) noexcept {
        set_name("cosine_sample_hemisphere");
        Float r = sqrt(u.x);
        Float phi = 2.0f * constants::pi * u.y;
        return make_float3(r * cos(phi), r * sin(phi), sqrt(1.0f - u.x));
    };

    // Balanced heuristic for Multiple Importance Sampling (MIS)
    // Combines PDFs from different sampling strategies optimally
    Callable balanced_heuristic = [](Float pdf_a, Float pdf_b) noexcept {
        set_name("balanced_heuristic");
        return pdf_a / max(pdf_a + pdf_b, 1e-4f);
    };

    // Adjust samples per dispatch based on backend capabilities
    auto default_max_spp_per_dispatch =
        device.backend_name() == "metal" ||
                device.backend_name() == "fallback" ?
            1u :
            64u;
    auto max_spp_per_dispatch = opts.max_spp_per_dispatch.value_or(
        default_max_spp_per_dispatch);
    bool infinite_render = opts.spp == 0u && !opts.offline;
    auto sample_plan = luisa::ref::PathTracingSamplePassPlan{
        .total_spp = infinite_render ? 0u : (opts.spp == 0u ? luisa::ref::DEFAULT_PATH_TRACING_SPP : opts.spp),
        .max_spp_per_dispatch = max_spp_per_dispatch,
        .infinite = infinite_render,
    };

    // Main path tracing kernel
    // Implements unidirectional path tracing with NEE and MIS
    Kernel2D raytracing_kernel = [&](ImageFloat image, ImageUInt seed_image, AccelVar accel, UInt2 resolution, UInt dispatch_spp) noexcept {
        set_name("raytracing_kernel");
        set_block_size(16u, 16u, 1u);
        UInt2 coord = dispatch_id().xy();
        Float frame_size = min(resolution.x, resolution.y).cast<float>();
        UInt state = seed_image.read(coord).x;
        Float3 radiance = def(make_float3(0.0f));

        // Light source definition (area light at ceiling)
        $for (i, dispatch_spp) {
            Float rx = lcg(state);
            Float ry = lcg(state);
            Float2 pixel = (make_float2(coord) + make_float2(rx, ry)) / frame_size * 2.0f - 1.0f;
            Var<Ray> ray = generate_ray(pixel * make_float2(1.0f, -1.0f));
            Float3 beta = def(make_float3(1.0f));// Path throughput
            Float pdf_bsdf = def(0.0f);          // BSDF PDF for MIS
            constexpr float3 light_position = make_float3(-0.24f, 1.98f, 0.16f);
            constexpr float3 light_u = make_float3(-0.24f, 1.98f, -0.22f) - light_position;
            constexpr float3 light_v = make_float3(0.23f, 1.98f, 0.16f) - light_position;
            constexpr float3 light_emission = make_float3(17.0f, 12.0f, 4.0f);
            Float light_area = length(cross(light_u, light_v));
            Float3 light_normal = normalize(cross(light_u, light_v));

            // Path tracing loop with maximum depth of 10
            $for (depth, 10u) {
                // Trace ray against scene
                Var<TriangleHit> hit = accel.intersect(ray, {});
                reorder_shader_execution();
                $if (hit->miss()) {
                    $break;
                };
                Var<Triangle> triangle = heap->buffer<Triangle>(hit.inst).read(hit.prim);
                Float3 p0 = vertex_buffer->read(triangle.i0);
                Float3 p1 = vertex_buffer->read(triangle.i1);
                Float3 p2 = vertex_buffer->read(triangle.i2);
                Float3 p = triangle_interpolate(hit.bary, p0, p1, p2);
                Float3 n = normalize(cross(p1 - p0, p2 - p0));

                Float cos_wo = dot(-ray->direction(), n);
                $if (cos_wo < 1e-4f) { $break; };

                // Direct hit on light source
                $if (hit.inst == static_cast<uint>(meshes.size() - 1u)) {
                    $if (depth == 0u) {
                        // Direct view of light - add full emission
                        radiance += light_emission;
                    }
                    $else {
                        // Indirect hit - use MIS weight
                        Float pdf_light = length_squared(p - ray->origin()) / (light_area * cos_wo);
                        Float mis_weight = balanced_heuristic(pdf_bsdf, pdf_light);
                        radiance += mis_weight * beta * light_emission;
                    };
                    $break;
                };

                // Next Event Estimation (NEE): sample light directly
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
                // Add direct lighting contribution if not occluded
                $if (!occluded & cos_wi_light > 1e-4f & cos_light > 1e-4f) {
                    Float pdf_light = (d_light * d_light) / (light_area * cos_light);
                    Float pdf_bsdf = cos_wi_light * inv_pi;
                    Float mis_weight = balanced_heuristic(pdf_light, pdf_bsdf);
                    Float3 bsdf = albedo * inv_pi * cos_wi_light;
                    radiance += beta * bsdf * mis_weight * light_emission / max(pdf_light, 1e-4f);
                };

                // Sample BSDF for next path segment
                Var<Onb> onb = make_onb(n);
                Float ux = lcg(state);
                Float uy = lcg(state);
                Float3 wi_local = cosine_sample_hemisphere(make_float2(ux, uy));
                Float cos_wi = abs(wi_local.z);
                Float3 new_direction = onb->to_world(wi_local);
                ray = make_ray(pp, new_direction);
                pdf_bsdf = cos_wi * inv_pi;
                beta *= albedo;// * cos_wi * inv_pi / pdf_bsdf => * 1.f

                // Russian Roulette path termination
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

    // Accumulation kernel for progressive rendering
    Kernel2D accumulate_kernel = [&](ImageFloat accum_image, ImageFloat curr_image) noexcept {
        set_name("accumulate_kernel");
        UInt2 p = dispatch_id().xy();
        Float4 accum = accum_image.read(p);
        Float4 curr = curr_image.read(p);
        accum_image.write(p, accum + make_float4(curr.xyz() * curr.w, curr.w));
    };

    // Clear image kernel
    Kernel2D clear_kernel = [](ImageFloat image) noexcept {
        set_name("clear_kernel");
        image.write(dispatch_id().xy(), make_float4(0.f));
    };

    // HDR to LDR conversion with tone mapping
    Kernel2D hdr2ldr_kernel = [&](ImageFloat hdr_image, ImageFloat ldr_image, Float scale) noexcept {
        set_name("hdr2ldr_kernel");
        UInt2 coord = dispatch_id().xy();
        Float4 hdr = hdr_image.read(coord);
        Float3 ldr = linear_to_srgb(clamp(
            hdr.xyz() / max(hdr.w, 1.0e-6f) * scale, 0.f, 1.f));
        ldr_image.write(coord, make_float4(ldr, 1.0f));
    };

    // Compile shaders
    ShaderOption o{.enable_debug_info = false};
    auto raytracing_shader = device.compile(raytracing_kernel, ShaderOption{.name = "path_tracing"});
    auto clear_shader = device.compile(clear_kernel, o);
    auto hdr2ldr_shader = device.compile(hdr2ldr_kernel, o);
    auto accumulate_shader = device.compile(accumulate_kernel, o);
    auto make_sampler_shader = device.compile(make_sampler_kernel, o);

    // Create images and window
    static constexpr uint2 resolution = make_uint2(1024u);
    Image<float> framebuffer = device.create_image<float>(PixelStorage::HALF4, resolution);
    Image<float> accum_image = device.create_image<float>(PixelStorage::FLOAT4, resolution);
    luisa::vector<std::array<uint8_t, 4u>> host_image(resolution.x * resolution.y);

    Image<uint> seed_image = device.create_image<uint>(PixelStorage::INT1, resolution);

    stream << clear_shader(accum_image).dispatch(resolution)
           << make_sampler_shader(seed_image).dispatch(resolution);

    // Setup the platform Window and common Luisa Swapchain presentation path.
    // Desktop owns a GLFW window by default; iOS passes a Window that wraps the
    // UIKit-owned CAMetalLayer.
    std::unique_ptr<Window> window;
    Window *active_window = opts.window;
    std::optional<Swapchain> swap_chain;
    if (!opts.offline) {
#if defined(LUISA_PLATFORM_IOS)
        if (active_window == nullptr) {
            return {
                .success = false,
                .error = "Interactive iOS rendering requires a native Window."};
        }
#else
        if (active_window == nullptr) {
            window = std::make_unique<Window>("path tracing", resolution, false);
            active_window = window.get();
        }
#endif
        swap_chain.emplace(device.create_swapchain(
            stream,
            SwapchainOption{
                .display = active_window->native_display(),
                .window = active_window->native_handle(),
                .size = make_uint2(resolution),
                .wants_hdr = false,
                .wants_vsync = false,
                .back_buffer_count = 2,
            }));
    }

    Image<float> ldr_image = device.create_image<float>(
        (!opts.offline && swap_chain.has_value()) ? swap_chain->backend_storage() : PixelStorage::BYTE4,
        resolution);
    double last_time = 0.0;
    uint64_t frame_count = 0u;
    bool snapshot_captured = false;
    Clock clock;

    // Main render loop
    while (sample_plan.has_next(frame_count)) {
        auto dispatch_spp = sample_plan.next_dispatch_spp(frame_count);
        stream << raytracing_shader(framebuffer, seed_image, accel, resolution, dispatch_spp)
                      .dispatch(resolution)
               << accumulate_shader(accum_image, framebuffer)
                      .dispatch(resolution);
        if (!opts.offline && swap_chain.has_value()) {
            stream << hdr2ldr_shader(accum_image, ldr_image, 2.f).dispatch(resolution)
                   << swap_chain->present(ldr_image);
            if (active_window->should_close()) { break; }
            active_window->poll_events();
        }
        double dt = clock.toc() - last_time;
        LUISA_INFO("dt = {:.2f}ms ({:.2f} spp/s)", dt, dispatch_spp / dt * 1000);
        last_time = clock.toc();
        frame_count += dispatch_spp;
        if (opts.progress_callback) {
            opts.progress_callback(frame_count, last_time);
        }
        if (!snapshot_captured && opts.snapshot_spp != 0u &&
            frame_count >= opts.snapshot_spp && opts.snapshot_callback) {
            stream << hdr2ldr_shader(accum_image, ldr_image, 2.f).dispatch(resolution)
                   << ldr_image.copy_to(luisa::span{host_image})
                   << synchronize();
            auto snapshot_elapsed = clock.toc();
            opts.snapshot_callback(
                resolution, frame_count, snapshot_elapsed, host_image);
            snapshot_captured = true;
        }
    }
    stream << hdr2ldr_shader(accum_image, ldr_image, 2.f).dispatch(resolution)
           << ldr_image.copy_to(luisa::span{host_image})
           << synchronize();
    auto elapsed = clock.toc();
    LUISA_INFO("FPS: {}", frame_count / elapsed * 1000);
    return {
        .success = true,
        .resolution = resolution,
        .completed_spp = frame_count,
        .elapsed_ms = elapsed,
        .pixels = std::move(host_image)};
}

#if !defined(LUISA_PATH_TRACING_LIBRARY_ONLY)
int main(int argc, char *argv[]) {

    log_level_verbose();

    Context context{argv[0]};
    if (argc <= 1) {
        LUISA_INFO("Usage: {} <backend> [--offline] [--spp N] [--max-spp-per-dispatch N]. <backend>: cuda, dx, metal, vk, hip, fallback, simd", argv[0]);
        return 1;
    }

    auto opts = luisa::ref::ExampleOptions::parse(argc, argv);
    if (!opts.valid()) {
        LUISA_WARNING("Invalid command line: {}", opts.error_message);
        return 1;
    }

    Device device = context.create_device(argv[1]);
    auto result = luisa::ref::run_path_tracing_test(
        device,
        luisa::ref::PathTracingTestOptions{
            .offline = opts.offline,
            .spp = opts.spp,
            .max_spp_per_dispatch = opts.max_spp_per_dispatch});
    if (!result.success) {
        LUISA_WARNING("Path tracing failed: {}", result.error);
        return 1;
    }
    stbi_write_png(
        "test_path_tracing.png",
        static_cast<int>(result.resolution.x),
        static_cast<int>(result.resolution.y), 4,
        result.pixels.data(), 0);
    if (opts.offline && opts.compare_path) {
        auto comparison = luisa::ref::compare_with_reference_file(
            reinterpret_cast<const uint8_t *>(result.pixels.data()),
            static_cast<int>(result.resolution.x),
            static_cast<int>(result.resolution.y), 4,
            *opts.compare_path);
        LUISA_INFO(
            "Reference comparison: {} ({})",
            comparison.passed ? "PASSED" : "FAILED",
            comparison.message);
        if (!comparison.passed) { return 1; }
    }
    return 0;
}
#endif
