// Procedural-primitive ray-query renderer.
//
// The scene contains a deterministic cloud of spheres represented by AABBs,
// plus a triangle whose intersection is filtered by a surface-candidate hook.
// The procedural-candidate hook performs the exact sphere intersection and
// records the normal as the sphere color.

#include <array>
#include <cstdint>
#include <memory>
#include <optional>

#include <stb/stb_image_write.h>

#include <luisa/luisa-compute.h>
#include <luisa/dsl/sugar.h>

#include "common/reference_compare.h"

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

using namespace luisa;
using namespace luisa::compute;

namespace {

[[nodiscard]] float lcg(uint &state) noexcept {
    constexpr auto multiplier = 1664525u;
    constexpr auto increment = 1013904223u;
    state = multiplier * state + increment;
    return cast<float>(state & 0x00ffffffu) *
           (1.0f / static_cast<float>(0x01000000u));
}

}// namespace

int main(int argc, char *argv[]) {
    log_level_verbose();
    if (argc <= 1) {
        LUISA_INFO("Usage: {} <backend> [--offline] [--spp N] [--compare <reference.png>]", argv[0]);
        return 1;
    }

    auto opts = luisa::ref::ExampleOptions::parse(argc, argv);
    if (!opts.valid()) {
        LUISA_WARNING("Invalid command line: {}", opts.error_message);
        return 1;
    }
    Context context{argv[0]};
    Device device = context.create_device(argv[1]);
#if ENABLE_DISPLAY
    auto interactive = !opts.offline;
#else
    auto interactive = false;
    if (!opts.offline) {
        LUISA_WARNING("GUI support is disabled; rendering the finite offline workload instead.");
    }
#endif
    Stream stream = device.create_stream(interactive ? StreamTag::GRAPHICS : StreamTag::COMPUTE);

    static constexpr uint width = 1280u;
    static constexpr uint height = 720u;
    static constexpr uint sphere_count = 1024u;
    static constexpr float sphere_radius = 0.2f;

    luisa::vector<AABB> aabbs{sphere_count};
    uint state = 0u;
    for (auto &aabb : aabbs) {
        // Function-argument evaluation order is unspecified. Sequence the
        // stateful draws explicitly so the scene is identical across host
        // compilers (notably Apple Clang and GCC).
        auto x = lcg(state) * 2.0f - 1.0f;
        auto y = lcg(state) * 2.0f - 1.0f;
        auto z = lcg(state) * 2.0f - 1.0f;
        auto center = make_float3(x, y, z) * 10.0f;
        auto aabb_max = center + sphere_radius + 1e-3f;
        auto aabb_min = center - sphere_radius - 1e-3f;
        aabb.packed_max = {aabb_max.x, aabb_max.y, aabb_max.z};
        aabb.packed_min = {aabb_min.x, aabb_min.y, aabb_min.z};
    }

    auto aabb_buffer = device.create_buffer<AABB>(sphere_count);
    auto spheres = device.create_procedural_primitive(aabb_buffer.view());

    std::array vertices{
        float3{-0.5f, -0.5f, 0.0f},
        float3{0.5f, -0.5f, 0.0f},
        float3{0.0f, 0.5f, 0.0f}};
    std::array indices{0u, 1u, 2u};
    auto vertex_buffer = device.create_buffer<float3>(vertices.size());
    auto triangle_buffer = device.create_buffer<Triangle>(1u);
    auto mesh = device.create_mesh(vertex_buffer, triangle_buffer);

    auto accel = device.create_accel();
    accel.emplace_back(spheres);
    accel.emplace_back(mesh, scaling(5.0f), 0xffu, false);

    stream << aabb_buffer.copy_from(luisa::span{aabbs})
           << spheres.build()
           << vertex_buffer.copy_from(luisa::span{vertices})
           << triangle_buffer.copy_from(luisa::span{indices})
           << mesh.build()
           << accel.build()
           << synchronize();

    Callable tea = [](UInt v0, UInt v1) noexcept {
        UInt sum = def(0u);
        for (uint round = 0u; round < 4u; round++) {
            sum += 0x9e3779b9u;
            v0 += ((v1 << 4u) + 0xa341316cu) ^ (v1 + sum) ^ ((v1 >> 5u) + 0xc8013ea4u);
            v1 += ((v0 << 4u) + 0xad90777du) ^ (v0 + sum) ^ ((v0 >> 5u) + 0x7e95761eu);
        }
        return v0;
    };

    auto accum_image = device.create_image<float>(PixelStorage::FLOAT4, width, height);
    Kernel2D render_kernel = [&](Float3 camera_position, UInt frame_index) noexcept {
        UInt2 coord = dispatch_id().xy();
        UInt2 size = dispatch_size().xy();
        Float aspect = size.x.cast<float>() / size.y.cast<float>();
        Float2 jitter = make_float2(make_uint2(
                            tea(coord.x, frame_index),
                            tea(coord.y, frame_index))) /
                        static_cast<float>(~0u);
        Float2 p = (make_float2(coord) + jitter) / make_float2(size) * 2.0f - 1.0f;
        static constexpr float fov = radians(45.8f);
        Float3 ray_direction = normalize(make_float3(
            p * tan(0.5f * fov) * make_float2(aspect, 1.0f), -1.0f));
        Var<Ray> ray = make_ray(camera_position, ray_direction);

        Float3 sphere_color = def(make_float3());
        Var<CommittedHit> hit = accel->traverse(ray, {})
                                    .on_surface_candidate([&](SurfaceCandidate &candidate) noexcept {
                                        Var<TriangleHit> candidate_hit = candidate.hit();
                                        Float3 barycentric = make_float3(
                                            1.0f - candidate_hit.bary.x - candidate_hit.bary.y,
                                            candidate_hit.bary);
                                        $if (length(barycentric.xy()) < 0.8f &
                                             length(barycentric.yz()) < 0.8f &
                                             length(barycentric.zx()) < 0.8f) {
                                            candidate.commit();
                                        };
                                    })
                                    .on_procedural_candidate([&](ProceduralCandidate &candidate) noexcept {
                                        Var<ProceduralHit> candidate_hit = candidate.hit();
                                        Var<Ray> candidate_ray = candidate.ray();
                                        Var<AABB> aabb = aabb_buffer->read(candidate_hit.prim);
                                        Float3 center = (aabb->min() + aabb->max()) * 0.5f;
                                        Float3 ray_origin = candidate_ray->origin();
                                        Float3 to_center = center - ray_origin;
                                        Float3 direction = candidate_ray->direction();
                                        Float cos_theta = dot(direction, normalize(to_center));
                                        $if (cos_theta > 0.0f) {
                                            Float center_distance = length(to_center);
                                            Float projected_distance = center_distance * cos_theta;
                                            Float perpendicular_distance = sqrt(
                                                center_distance * center_distance -
                                                projected_distance * projected_distance);
                                            $if (perpendicular_distance <= sphere_radius) {
                                                Float half_chord = sqrt(
                                                    sphere_radius * sphere_radius -
                                                    perpendicular_distance * perpendicular_distance);
                                                Float hit_distance = projected_distance - half_chord;
                                                $if (hit_distance <= candidate_ray->t_max()) {
                                                    Float3 normal = normalize(
                                                        ray_origin + direction * hit_distance - center);
                                                    sphere_color = normal * 0.5f + 0.5f;
                                                };
                                                candidate.commit(hit_distance);
                                            };
                                        };
                                    })
                                    .trace();

        Float3 old_color = accum_image->read(coord).xyz();
        Float3 color = def(make_float3());
        $if (hit->is_procedural()) {
            color = sphere_color;
        }
        $elif (hit->is_triangle()) {
            color = make_float3(1.0f - hit.bary.x - hit.bary.y, hit.bary);
        };
        Float sample_count = cast<float>(frame_index + 1u);
        accum_image->write(coord, make_float4(lerp(old_color, color, 1.0f / sample_count), 1.0f));
    };

    auto clear_shader = device.compile<2>([](ImageFloat image) noexcept {
        image.write(dispatch_id().xy(), make_float4());
    });
    auto blit_shader = device.compile<2>([](ImageFloat source, ImageFloat destination) noexcept {
        auto coord = dispatch_id().xy();
        destination.write(coord, source.read(coord));
    });
    auto render_shader = device.compile(render_kernel);

    std::optional<Swapchain> swap_chain;
#if ENABLE_DISPLAY
    std::unique_ptr<Window> window;
    if (interactive) {
        window = std::make_unique<Window>("Procedural Primitive Ray Query", width, height);
        swap_chain.emplace(device.create_swapchain(
            stream,
            SwapchainOption{
                .display = window->native_display(),
                .window = window->native_handle(),
                .size = make_uint2(width, height),
                .wants_hdr = false,
                .wants_vsync = false,
                .back_buffer_count = 3,
            }));
    }
#endif
    auto ldr_image = device.create_image<float>(
        swap_chain.has_value() ? swap_chain->backend_storage() : PixelStorage::BYTE4,
        width, height);
    luisa::vector<std::array<uint8_t, 4u>> pixels{width * height};
    constexpr float3 camera_position = make_float3(0.0f, 0.0f, 18.0f);
    auto infinite_render = interactive && opts.spp == 0u;
    auto total_spp = infinite_render ? 0u : (opts.spp == 0u ? 1024u : opts.spp);
    auto spp = 0u;
    Clock clock;
    stream << clear_shader(accum_image).dispatch(width, height);
    clock.tic();
    while (infinite_render || spp < total_spp) {
        CommandList commands;
        // One sample per presentation keeps interactive progress visible and
        // bounds the amount of queued ray-tracing work on mobile GPUs.
        commands << render_shader(camera_position, spp).dispatch(width, height)
                 << blit_shader(accum_image, ldr_image).dispatch(width, height);
        stream << commands.commit();
        ++spp;
        if (swap_chain.has_value()) {
            stream << swap_chain->present(ldr_image);
#if ENABLE_DISPLAY
            window->poll_events();
            if (window->should_close()) { break; }
#endif
        }
    }
    stream << ldr_image.copy_to(luisa::span{pixels}) << synchronize();
    auto elapsed_ms = clock.toc();
    LUISA_INFO("Rendered {} spp in {} ms ({:.2f} spp/s).",
               spp, elapsed_ms, static_cast<double>(spp) / elapsed_ms * 1000.0);

    if (stbi_write_png("test_procedural.png", width, height, 4, pixels.data(), 0) == 0) {
        LUISA_WARNING("Failed to write test_procedural.png.");
        return 1;
    }

    if (opts.compare_path) {
        auto result = luisa::ref::compare_with_reference_file(
            reinterpret_cast<const uint8_t *>(pixels.data()),
            width, height, 4, *opts.compare_path);
        LUISA_INFO("Reference comparison: {} ({})",
                   result.passed ? "PASSED" : "FAILED", result.message);
        if (!result.passed) { return 1; }
    }
    return 0;
}
