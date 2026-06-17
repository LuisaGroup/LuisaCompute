// Coroutine Path Tracing Example — Cornell Box
// Full path tracing pipeline inside a coroutine with multiple $suspend
// points matching the persistent-threads pattern from the old codebase.
//
// Usage: coro_path_tracing <backend> [--offline] [--spp N]
//   backend: cuda, dx, cpu, metal, fallback

#include <array>
#include <cstdlib>
#include <memory>
#include <string_view>

#include <stb/stb_image_write.h>

#include <luisa/luisa-compute.h>
#include <luisa/dsl/sugar.h>
#include <luisa/dsl/coro_func.h>
#include <luisa/coro/schedulers/persistent_threads.h>
#include <luisa/coro/schedulers/state_machine.h>
#include <luisa/coro/schedulers/wavefront.h>

#include "common/reference_compare.h"
#include "cornell_box.h"

#define TINYOBJLOADER_IMPLEMENTATION
#include "tiny_obj_loader.h"

using namespace luisa;
using namespace luisa::compute;
using namespace luisa::compute::coro;

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

enum class ExampleCoroSchedulerKind {
    state_machine,
    wavefront,
    persistent,
};

[[nodiscard]] static auto parse_coro_scheduler(int argc, char *argv[]) noexcept {
    auto kind = ExampleCoroSchedulerKind::state_machine;
    for (int i = 2; i < argc; i++) {
        if (argv[i] == nullptr) { break; }
        std::string_view arg{argv[i]};
        if (arg == "--scheduler") {
            if (i + 1 >= argc || argv[i + 1] == nullptr) {
                LUISA_ERROR("Missing value for --scheduler. Expected state_machine, wavefront, or persistent.");
            }
            std::string_view value{argv[++i]};
            if (value == "state_machine" || value == "statemachine" || value == "state") {
                kind = ExampleCoroSchedulerKind::state_machine;
            } else if (value == "wavefront" || value == "wave") {
                kind = ExampleCoroSchedulerKind::wavefront;
            } else if (value == "persistent" || value == "persistent_threads") {
                kind = ExampleCoroSchedulerKind::persistent;
            } else {
                LUISA_ERROR("Unknown coroutine scheduler '{}'. Expected state_machine, wavefront, or persistent.", value);
            }
        }
    }
    return kind;
}

[[nodiscard]] static constexpr auto scheduler_name(ExampleCoroSchedulerKind kind) noexcept {
    switch (kind) {
        case ExampleCoroSchedulerKind::state_machine: return "state_machine";
        case ExampleCoroSchedulerKind::wavefront: return "wavefront";
        case ExampleCoroSchedulerKind::persistent: return "persistent";
    }
    return "unknown";
}

int main(int argc, char *argv[]) {

    log_level_verbose();

    Context context{argv[0]};
    if (argc <= 1) {
        LUISA_INFO("Usage: {} <backend> [--offline] [--spp N] [--scheduler state_machine|wavefront|persistent]", argv[0]);
        exit(1);
    }

    std::string_view backend_name{argv[1]};
    Device device = context.create_device(backend_name);

    auto opts = luisa::ref::ExampleOptions::parse(argc, argv);
    auto scheduler_kind = parse_coro_scheduler(argc, argv);
    bool offline = opts.offline;
#if !ENABLE_DISPLAY
    if (!offline) {
        LUISA_WARNING("GUI support is disabled. Running in offline mode.");
        offline = true;
    }
#endif

    // ─── Load Cornell Box OBJ ────────────────────────────────────────
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

    auto &&attrib = obj_reader.GetAttrib().vertices;
    luisa::vector<float3> vertices;
    vertices.reserve(attrib.size() / 3u);
    for (uint i = 0u; i < attrib.size(); i += 3u) {
        vertices.emplace_back(make_float3(
            attrib[i + 0u], attrib[i + 1u], attrib[i + 2u]));
    }
    LUISA_INFO(
        "Loaded mesh with {} shape(s) and {} vertices.",
        obj_reader.GetShapes().size(), vertices.size());

    BindlessArray heap = device.create_bindless_array();
    Stream stream = device.create_stream(offline ? StreamTag::COMPUTE : StreamTag::GRAPHICS);
    Buffer<float3> vertex_buffer = device.create_buffer<float3>(vertices.size());
    stream << vertex_buffer.copy_from(vertices.data());
    luisa::vector<Mesh> meshes;
    luisa::vector<Buffer<Triangle>> triangle_buffers;
    for (auto &&shape : obj_reader.GetShapes()) {
        uint index = static_cast<uint>(meshes.size());
        auto const &t = shape.mesh.indices;
        uint triangle_count = t.size() / 3u;
        LUISA_INFO(
            "Processing shape '{}' at index {} with {} triangle(s).",
            shape.name, index, triangle_count);
        luisa::vector<uint> indices;
        indices.reserve(t.size());
        for (tinyobj::index_t i : t) { indices.emplace_back(i.vertex_index); }
        auto &triangle_buffer = triangle_buffers.emplace_back(device.create_buffer<Triangle>(triangle_count));
        auto &mesh = meshes.emplace_back(device.create_mesh(vertex_buffer, triangle_buffer));
        heap.emplace_on_update(index, triangle_buffer);
        stream << triangle_buffer.copy_from(indices.data())
               << mesh.build();
    }

    Accel accel = device.create_accel({});
    for (auto &m : meshes) {
        accel.emplace_back(m, make_float4x4(1.0f));
    }
    stream << heap.update()
           << accel.build()
           << synchronize();

    // ─── Materials ───────────────────────────────────────────────────
    Constant materials{
        make_float3(0.725f, 0.710f, 0.680f),// floor
        make_float3(0.725f, 0.710f, 0.680f),// ceiling
        make_float3(0.725f, 0.710f, 0.680f),// back wall
        make_float3(0.140f, 0.450f, 0.091f),// right wall
        make_float3(0.630f, 0.065f, 0.050f),// left wall
        make_float3(0.725f, 0.710f, 0.680f),// short box
        make_float3(0.725f, 0.710f, 0.680f),// tall box
        make_float3(0.000f, 0.000f, 0.000f),// light
    };

    // ─── Callables ───────────────────────────────────────────────────
    Callable linear_to_srgb = [&](Var<float3> x) noexcept {
        return saturate(select(1.055f * pow(x, 1.0f / 2.4f) - 0.055f,
                               12.92f * x,
                               x <= 0.00031308f));
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

    Callable aces_tonemapping = [](Float3 x) noexcept {
        static constexpr float a = 2.51f;
        static constexpr float b = 0.03f;
        static constexpr float c = 2.43f;
        static constexpr float d = 0.59f;
        static constexpr float e = 0.14f;
        return clamp((x * (a * x + b)) / (x * (c * x + d) + e), 0.0f, 1.0f);
    };

    // ─── SPP per dispatch ────────────────────────────────────────────
    auto spp_per_dispatch = (backend_name == "metal" ||
                             backend_name == "cpu" ||
                             backend_name == "fallback") ? 1u : 64u;

    static constexpr uint2 resolution = make_uint2(1024u);
    uint total_cells = resolution.x * resolution.y;
    uint total_spp = opts.spp == 0u ? 1024u : opts.spp;

    Coroutine coro = [&](ImageFloat image, ImageUInt seed_image,
                          AccelVar accel, UInt2 resolution) noexcept {
        UInt2 coord = dispatch_id().xy();
        Float frame_size = min(resolution.x, resolution.y).cast<float>();
        UInt state = seed_image.read(coord).x;
        Float rx = lcg(state);
        Float ry = lcg(state);
        Float2 pixel = (make_float2(coord) + make_float2(rx, ry)) / frame_size * 2.0f - 1.0f;
        Float3 radiance = def(make_float3(0.0f));
        $suspend("per_spp");
        $for (i, spp_per_dispatch) {
            Var<Ray> ray = generate_ray(pixel * make_float2(1.0f, -1.0f));
            Float3 beta = def(make_float3(1.0f));
            Float pdf_bsdf = def(0.0f);
            constexpr float3 light_position = make_float3(-0.24f, 1.98f, 0.16f);
            constexpr float3 light_u = make_float3(-0.24f, 1.98f, -0.22f) - light_position;
            constexpr float3 light_v = make_float3(0.23f, 1.98f, 0.16f) - light_position;
            constexpr float3 light_emission = make_float3(17.0f, 12.0f, 4.0f);
            Float light_area = length(cross(light_u, light_v));
            Float3 light_normal = normalize(cross(light_u, light_v));
            $suspend("per_depth");
            $for (depth, 10u) {
                $suspend("before_tracing");
                Var<TriangleHit> hit = accel.intersect(ray, {});
                reorder_shader_execution();
                $if (hit->miss()) { $break; };
                Var<Triangle> triangle = heap->buffer<Triangle>(hit.inst).read(hit.prim);
                Float3 p0 = vertex_buffer->read(triangle.i0);
                Float3 p1 = vertex_buffer->read(triangle.i1);
                Float3 p2 = vertex_buffer->read(triangle.i2);
                Float3 p = triangle_interpolate(hit.bary, p0, p1, p2);
                Float3 n = normalize(cross(p1 - p0, p2 - p0));
                Float cos_wo = dot(-ray->direction(), n);
                $if (cos_wo < 1e-4f) { $break; };
                $suspend("after_tracing");
                $if (hit.inst == static_cast<uint>(meshes.size() - 1u)) {
                    $if (depth == 0u) {
                        radiance += light_emission;
                    } $else {
                        Float pdf_light = length_squared(p - ray->origin()) / (light_area * cos_wo);
                        Float mis_weight = balanced_heuristic(pdf_bsdf, pdf_light);
                        radiance += mis_weight * beta * light_emission;
                    };
                    $break;
                };
                $suspend("sample_light");
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
                Float3 albedo = materials.read(hit.inst);
                $if (!occluded & cos_wi_light > 1e-4f & cos_light > 1e-4f) {
                    Float pdf_light = (d_light * d_light) / (light_area * cos_light);
                    Float pdf_bsdf = cos_wi_light * inv_pi;
                    Float mis_weight = balanced_heuristic(pdf_light, pdf_bsdf);
                    Float3 bsdf = albedo * inv_pi * cos_wi_light;
                    radiance += beta * bsdf * mis_weight * light_emission / max(pdf_light, 1e-4f);
                };
                $suspend("sample_bsdf");
                Var<Onb> onb = make_onb(n);
                Float ux = lcg(state);
                Float uy = lcg(state);
                Float3 wi_local = cosine_sample_hemisphere(make_float2(ux, uy));
                Float cos_wi = abs(wi_local.z);
                Float3 new_direction = onb->to_world(wi_local);
                ray = make_ray(pp, new_direction);
                pdf_bsdf = cos_wi * inv_pi;
                beta *= albedo;
                $suspend("rr");
                Float l = dot(make_float3(0.212671f, 0.715160f, 0.072169f), beta);
                $if (l == 0.0f) { $break; };
                Float q = max(l, 0.05f);
                Float r = lcg(state);
                $if (r >= q) { $break; };
                beta *= 1.0f / q;
            };
        };
        $suspend("write_film");
        radiance /= static_cast<float>(spp_per_dispatch);
        seed_image.write(coord, make_uint4(state));
        $if (any(dsl::isnan(radiance))) { radiance = make_float3(0.0f); };
        image.write(coord, make_float4(clamp(radiance, 0.0f, 30.0f), 1.0f));
    };

    using Scheduler = CoroScheduler<Image<float>, Image<uint>, Accel, uint2>;
    std::unique_ptr<Scheduler> scheduler;
    switch (scheduler_kind) {
        case ExampleCoroSchedulerKind::state_machine:
            scheduler = std::make_unique<StateMachineCoroScheduler<Image<float>, Image<uint>, Accel, uint2>>(device, coro);
            break;
        case ExampleCoroSchedulerKind::wavefront: {
            WavefrontCoroSchedulerConfig cfg{};
            cfg.thread_count = total_cells;
            scheduler = std::make_unique<WavefrontCoroScheduler<Image<float>, Image<uint>, Accel, uint2>>(device, coro, cfg);
            break;
        }
        case ExampleCoroSchedulerKind::persistent: {
            PersistentThreadsCoroSchedulerConfig cfg{};
            cfg.thread_count = total_cells;
            scheduler = std::make_unique<PersistentThreadsCoroScheduler<Image<float>, Image<uint>, Accel, uint2>>(device, coro, cfg);
            break;
        }
    }
    LUISA_INFO("CoroScheduler: {}", scheduler_name(scheduler_kind));

    // ─── Make sampler shader ─────────────────────────────────────────
    auto make_sampler_shader = device.compile(Kernel2D([&](ImageUInt seed_image) noexcept {
        UInt2 p = dispatch_id().xy();
        UInt state = tea(p.x, p.y);
        seed_image.write(p, make_uint4(state));
    }));

    // ─── Accumulation kernel ─────────────────────────────────────────
    auto accumulate_shader = device.compile(Kernel2D([&](ImageFloat accum_image, ImageFloat curr_image) noexcept {
        UInt2 p = dispatch_id().xy();
        Float4 accum = accum_image.read(p);
        Float3 curr = curr_image.read(p).xyz();
        accum_image.write(p, accum + make_float4(curr, 1.f));
    }));

    // ─── Clear kernel ────────────────────────────────────────────────
    auto clear_shader = device.compile(Kernel2D([](ImageFloat image) noexcept {
        image.write(dispatch_id().xy(), make_float4(0.0f));
    }));

    // ─── HDR-to-LDR kernel ───────────────────────────────────────────
    auto hdr2ldr_shader = device.compile(Kernel2D([&](ImageFloat hdr_image, ImageFloat ldr_image,
                                                       Float scale, Bool is_hdr) noexcept {
        UInt2 coord = dispatch_id().xy();
        Float4 hdr = hdr_image.read(coord);
        Float3 ldr = clamp(hdr.xyz() / hdr.w * scale, 0.0f, 1.0f);
        $if (!is_hdr) {
            ldr = linear_to_srgb(ldr);
        };
        ldr_image.write(coord, make_float4(ldr, 1.0f));
    }));

    // ─── Resources ───────────────────────────────────────────────────
    Image<float> framebuffer = device.create_image<float>(PixelStorage::HALF4, resolution);
    Image<float> accum_image = device.create_image<float>(PixelStorage::FLOAT4, resolution);
    Image<float> ldr_image = device.create_image<float>(PixelStorage::BYTE4, resolution);
    Image<uint> seed_image = device.create_image<uint>(PixelStorage::INT1, resolution);

    // Init accum + seeds
    stream << clear_shader(accum_image).dispatch(resolution)
           << make_sampler_shader(seed_image).dispatch(resolution)
           << synchronize();

    // ─── Render loop ─────────────────────────────────────────────────
    if (offline) {
        auto passes = (total_spp + spp_per_dispatch - 1u) / spp_per_dispatch;
        Clock clock;
        for (uint pass = 0u; pass < passes; ++pass) {
            stream << (*scheduler)(framebuffer, seed_image, accel, resolution).dispatch(resolution)
                   << accumulate_shader(accum_image, framebuffer)
                            .dispatch(resolution)
                   << synchronize();
            LUISA_INFO("Pass {}/{}: {:.1f} ms",
                       pass + 1u, passes, clock.toc());
        }
        double total_ms = clock.toc();

        luisa::vector<std::array<uint8_t, 4u>> host_image(total_cells);
        stream << hdr2ldr_shader(accum_image, ldr_image, 2.0f, false).dispatch(resolution)
               << ldr_image.copy_to(luisa::span{host_image})
               << synchronize();
        stbi_write_png("coro_path_tracing.png",
                       resolution.x, resolution.y, 4,
                       host_image.data(), 0);

        LUISA_INFO("Rendered {} passes in {:.1f} ms total ({:.1f} ms/pass)",
                   passes, total_ms, total_ms / passes);
        if (opts.compare_path) {
            auto result = luisa::ref::compare_with_reference_file(
                reinterpret_cast<const uint8_t *>(host_image.data()),
                resolution.x, resolution.y, 4,
                *opts.compare_path);
            LUISA_INFO("Reference comparison: {} ({})",
                       result.passed ? "PASSED" : "FAILED", result.message);
            if (!result.passed) { return 1; }
        }
    } else {
#if ENABLE_DISPLAY
        // ─── Interactive mode ────────────────────────────────────────
        Window window{"Coroutine Path Tracing", resolution};
        Swapchain swapchain = device.create_swapchain(
            stream,
            SwapchainOption{
                .display = window.native_display(),
                .window = window.native_handle(),
                .size = resolution,
                .wants_hdr = false,
                .wants_vsync = false,
                .back_buffer_count = 3,
            });
        Image<float> ldr_image = device.create_image<float>(
            swapchain.backend_storage(), resolution);

        double last_time = 0.0;
        uint frame_count = 0u;
        Clock clock;

        LUISA_INFO("Interactive mode: {}-SPP/pass, ESC to quit", spp_per_dispatch);

        while (!window.should_close()) {
            stream << (*scheduler)(framebuffer, seed_image, accel, resolution).dispatch(resolution)
                   << accumulate_shader(accum_image, framebuffer)
                            .dispatch(resolution)
                   << hdr2ldr_shader(accum_image, ldr_image, 2.0f, false)
                           .dispatch(resolution)
                   << swapchain.present(ldr_image)
                   << synchronize();
            window.poll_events();
            double dt = clock.toc() - last_time;
            last_time = clock.toc();
            frame_count += spp_per_dispatch;
            LUISA_INFO("spp: {}, time: {} ms, spp/s: {}",
                       frame_count, dt, spp_per_dispatch / dt * 1000.0);
        }

        // Read back final image and save
        luisa::vector<std::array<uint8_t, 4u>> host_image(total_cells);
        stream << ldr_image.copy_to(host_image.data())
               << synchronize();

        LUISA_INFO("FPS: {}", frame_count / clock.toc() * 1000.0);
        stbi_write_png("coro_path_tracing.png",
                       resolution.x, resolution.y, 4,
                       host_image.data(), 0);
#else
        LUISA_ERROR("GUI support is disabled. Use --offline.");
#endif
    }

    return 0;
}
