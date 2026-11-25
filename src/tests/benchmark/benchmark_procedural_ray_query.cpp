// Benchmark for procedural-primitive ray tracing through ray queries.
//
// The point is to compare the *same* LuisaCompute kernel across backends: the
// AST is identical, only the code generation (HLSL/DXC on dx, NVRTC/CUDA on
// cuda, XIR -> SPIR-V on vk) differs.  Every workload keeps the ray-query
// traversal on the critical path, so the numbers are dominated by the
// intersection handler and the sampling math the backends emit.
//
// Workloads
// ---------
//   zoo      the twelve-kind procedural zoo of procedural_zoo.h, one primary
//            ray per pixel (analytic sphere/ellipsoid/box/capsule/cylinder/cone/
//            bicone/tetrahedron/hex prism/octahedron/sphere chain/helix tube)
//   many     4096 procedural spheres, one primary ray per pixel (pure traversal
//            plus one quadratic)
//   shadow   4096 procedural spheres, one primary ray plus four visibility rays
//            per pixel (five traversals per pixel)
//   bounce   4096 procedural spheres, six-bounce path tracing with cosine and
//            GGX-VNDF sampling (six traversals per pixel, heavy ALU)
//
// Usage
// -----
//   benchmark_procedural_ray_query <backend> [--rounds N] [--reps N] [--width W]
//                                  [--height H] [--workload name]
//                                  [--rays-per-pixel-scale F]
// Defaults: rounds 7, reps 4, width 1280, height 720, all workloads.  Each
// measurement dispatches the kernel `reps` times in one batch and reports the
// per-dispatch time, which keeps the host submission overhead out of the number.
// Each measurement warms up twice, then reports min / median / mean milliseconds
// and the effective number of primary-ray-equivalents per second.
// Output lines look like
//   RESULT backend=vk workload=zoo width=640 height=360 rays=1 min_ms=1.234 median_ms=1.345 mean_ms=1.400 Mray_s=171.0

#include "ut/ut.hpp"// boost.ut cfg used by test_device.h
#include "test_device.h"
#include "procedural_zoo.h"

#include <luisa/core/clock.h>
#include <luisa/core/logging.h>
#include <luisa/core/stl/string.h>
#include <luisa/dsl/sugar.h>
#include <luisa/luisa-compute.h>
#include <luisa/runtime/buffer.h>
#include <luisa/runtime/command_list.h>
#include <luisa/runtime/device.h>
#include <luisa/runtime/stream.h>

#include <algorithm>
#include <charconv>
#include <cmath>
#include <numeric>

using namespace luisa;
using namespace luisa::compute;

namespace {

constexpr uint kSphereCount = 4096u;

struct Options {
    uint width{1280u};
    uint height{720u};
    uint rounds{7u};
    uint reps{4u};
    luisa::string workload{"all"};
};

[[nodiscard]] uint parse_uint(const char *text, const char *name) noexcept {
    auto input = luisa::string_view{text};
    uint value{};
    auto r = std::from_chars(input.data(), input.data() + input.size(), value);
    LUISA_ASSERT(r.ec == std::errc{} && r.ptr == input.data() + input.size() && value > 0u,
                 "{} must be a positive integer, got \"{}\"", name, text);
    return value;
}

[[nodiscard]] Options parse_options(int argc, char *argv[]) noexcept {
    Options o;
    for (auto i = 2; i < argc; i++) {
        auto arg = luisa::string_view{argv[i]};
        auto next = [&]() -> const char * {
            LUISA_ASSERT(i + 1 < argc, "missing value for {}", arg);
            return argv[++i];
        };
        if (arg == "--width") {
            o.width = parse_uint(next(), "--width");
        } else if (arg == "--height") {
            o.height = parse_uint(next(), "--height");
        } else if (arg == "--rounds") {
            o.rounds = parse_uint(next(), "--rounds");
        } else if (arg == "--reps") {
            o.reps = parse_uint(next(), "--reps");
        } else if (arg == "--workload") {
            o.workload = next();
        } else {
            LUISA_WARNING("ignoring unknown option \"{}\"", arg);
        }
    }
    return o;
}

struct Statistics {
    double min{};
    double median{};
    double mean{};
};

[[nodiscard]] Statistics summarize(luisa::vector<double> samples) noexcept {
    LUISA_ASSERT(!samples.empty(), "no samples");
    luisa::sort(samples.begin(), samples.end());
    auto sum = std::accumulate(samples.begin(), samples.end(), 0.0);
    return {samples.front(), samples[samples.size() / 2u], sum / static_cast<double>(samples.size())};
}

}// namespace

int main(int argc, char *argv[]) {
    auto dc = luisa::test::create_device_from_ut(argc, argv);
    if (!dc) { return 0; }
    auto &device = dc->device;
    auto backend = luisa::string_view{device.backend_name()};
    if (backend.empty()) { backend = luisa::string_view{argv[1]}; }
    auto options = parse_options(argc, argv);
    auto width = options.width;
    auto height = options.height;
    LUISA_INFO("Procedural ray-query benchmark on {} ({}x{}, {} rounds).", backend, width, height, options.rounds);

    Clock clock;
    Stream stream = device.create_stream();

    // --- scene: the twelve-kind zoo -------------------------------------------
    auto zoo_prims = luisa::test::zoo::host::make_zoo();
    luisa::vector<AABB> zoo_aabbs;
    for (auto &p : zoo_prims) { zoo_aabbs.emplace_back(luisa::test::zoo::host::to_aabb(luisa::test::zoo::host::bounds(p))); }
    auto zoo_prim_buffer = device.create_buffer<luisa::test::zoo::ProcPrim>(zoo_prims.size());
    auto zoo_aabb_buffer = device.create_buffer<AABB>(zoo_aabbs.size());
    auto zoo_accel = device.create_accel();
    auto zoo_procedural = device.create_procedural_primitive(zoo_aabb_buffer.view());
    zoo_accel.emplace_back(zoo_procedural);

    // --- scene: many spheres --------------------------------------------------
    luisa::vector<luisa::test::zoo::ProcPrim> sphere_prims;
    luisa::vector<AABB> sphere_aabbs;
    sphere_prims.reserve(kSphereCount);
    sphere_aabbs.reserve(kSphereCount);
    {
        uint state = 12345u;
        auto next = [&]() noexcept {
            state = state * 1664525u + 1013904223u;
            return static_cast<float>(state >> 8u) * (1.f / 16777216.f);
        };
        for (uint i = 0u; i < kSphereCount; i++) {
            auto r = 0.09f + 0.09f * next();
            auto p = make_float3(next() * 2.f - 1.f, next() * 2.f - 1.f, next() * 2.f - 1.f) * 2.6f;
            auto prim = luisa::test::zoo::host::make_sphere(p, r);
            sphere_prims.emplace_back(prim);
            sphere_aabbs.emplace_back(luisa::test::zoo::host::to_aabb(luisa::test::zoo::host::bounds(prim)));
        }
    }
    auto sphere_prim_buffer = device.create_buffer<luisa::test::zoo::ProcPrim>(sphere_prims.size());
    auto sphere_aabb_buffer = device.create_buffer<AABB>(sphere_aabbs.size());
    auto sphere_accel = device.create_accel();
    auto sphere_procedural = device.create_procedural_primitive(sphere_aabb_buffer.view());
    sphere_accel.emplace_back(sphere_procedural);

    stream << zoo_prim_buffer.copy_from(luisa::span{zoo_prims})
           << zoo_aabb_buffer.copy_from(luisa::span{zoo_aabbs})
           << zoo_procedural.build()
           << zoo_accel.build()
           << sphere_prim_buffer.copy_from(luisa::span{sphere_prims})
           << sphere_aabb_buffer.copy_from(luisa::span{sphere_aabbs})
           << sphere_procedural.build()
           << sphere_accel.build()
           << synchronize();

    const float3 camera = make_float3(0.f, 0.f, 9.f);
    const float3 target = make_float3(0.f);
    auto target_image = device.create_image<float>(PixelStorage::FLOAT4, width, height);
    auto shadow_image = device.create_image<float>(PixelStorage::FLOAT4, width, height);

    auto make_camera_ray = [](Expr<float2> coord, Expr<float2> size) noexcept {
        auto p = (coord + 0.5f) / size * 2.f - 1.f;
        auto aspect = size.x / size.y;
        return normalize(make_float3(p * tan(0.5f * 0.7f) * make_float2(aspect, 1.f), -1.f));
    };

    // --- kernels --------------------------------------------------------------
    auto kernel_zoo = device.compile<2>(
        [&](ImageFloat image, Float3 camera) noexcept {
            auto coord = make_float2(cast<float>(dispatch_id().x), cast<float>(dispatch_id().y));
            auto size = make_float2(cast<float>(dispatch_size().x), cast<float>(dispatch_size().y));
            auto direction = make_camera_ray(coord, size);
            auto t_hit = def(1e30f);
            auto kind_hit = def(0u);
            auto ray = make_ray(camera, direction, 1e-4f, 1e30f);
            zoo_accel->traverse(ray, {})
                .on_procedural_candidate([&](ProceduralCandidate &candidate) noexcept {
                    auto h = candidate.hit();
                    auto prim = zoo_prim_buffer->read(h.prim);
                    auto cr = candidate.ray();
                    auto t = def(cr->t_max());
                    luisa::test::zoo::zoo_hit(cr->origin(), cr->direction(), prim,
                                              cr->t_min(), t_hit, t);
                    $if (t < t_hit) {
                        t_hit = t;
                        kind_hit = prim->kind;
                        candidate.commit(t);
                    };
                })
                .trace();
            auto hue = cast<float>(kind_hit) * (1.f / 12.f);
            auto color = select(make_float3(0.05f, 0.06f, 0.08f),
                                0.5f + 0.5f * cos(6.2831853f * (hue + make_float3(0.f, 0.33f, 0.67f))),
                                t_hit < 1e29f);
            image.write(dispatch_id().xy(), make_float4(color, 1.f));
        });

    auto kernel_many = device.compile<2>(
        [&](ImageFloat image, Float3 camera) noexcept {
            auto coord = make_float2(cast<float>(dispatch_id().x), cast<float>(dispatch_id().y));
            auto size = make_float2(cast<float>(dispatch_size().x), cast<float>(dispatch_size().y));
            auto direction = make_camera_ray(coord, size);
            auto t_hit = def(1e30f);
            auto ray = make_ray(camera, direction, 1e-4f, 1e30f);
            sphere_accel->traverse(ray, {})
                .on_procedural_candidate([&](ProceduralCandidate &candidate) noexcept {
                    auto h = candidate.hit();
                    auto prim = sphere_prim_buffer->read(h.prim);
                    auto cr = candidate.ray();
                    auto t = def(cr->t_max());
                    luisa::test::zoo::zoo_sphere(cr->origin(), cr->direction(),
                                                 prim->center, prim->dims.x, t, cr->t_min());
                    $if (t < t_hit) {
                        t_hit = t;
                        candidate.commit(t);
                    };
                })
                .trace();
            image.write(dispatch_id().xy(), make_float4(make_float3(t_hit * 0.05f), 1.f));
        });

    auto kernel_shadow = device.compile<2>(
        [&](ImageFloat image, Float3 camera) noexcept {
            auto coord = make_float2(cast<float>(dispatch_id().x), cast<float>(dispatch_id().y));
            auto size = make_float2(cast<float>(dispatch_size().x), cast<float>(dispatch_size().y));
            auto direction = make_camera_ray(coord, size);
            auto t_hit = def(1e30f);
            auto p_hit = def(camera);
            auto ray = make_ray(camera, direction, 1e-4f, 1e30f);
            sphere_accel->traverse(ray, {})
                .on_procedural_candidate([&](ProceduralCandidate &candidate) noexcept {
                    auto h = candidate.hit();
                    auto prim = sphere_prim_buffer->read(h.prim);
                    auto cr = candidate.ray();
                    auto t = def(cr->t_max());
                    luisa::test::zoo::zoo_sphere(cr->origin(), cr->direction(),
                                                 prim->center, prim->dims.x, t, cr->t_min());
                    $if (t < t_hit) {
                        t_hit = t;
                        candidate.commit(t);
                    };
                })
                .trace();
            $if (t_hit < 1e29f) { p_hit = camera + direction * t_hit; };
            auto lit = def(0u);
            $for (s, 0u, 4u) {
                auto u = luisa::test::zoo::sampling::sample2(1u, dispatch_id().x + s * 7919u, s);
                auto dir = luisa::test::zoo::sampling::cosine_hemisphere(u, direction);
                auto shadow = make_ray(p_hit + dir * 1e-3f, dir, 1e-3f, 1e30f);
                auto blocked = !sphere_accel->traverse(shadow, {})
                                    .on_procedural_candidate([&](ProceduralCandidate &candidate) noexcept {
                                        auto h = candidate.hit();
                                        auto prim = sphere_prim_buffer->read(h.prim);
                                        auto cr = candidate.ray();
                                        auto t = def(cr->t_max());
                                        luisa::test::zoo::zoo_sphere(cr->origin(), cr->direction(),
                                                                     prim->center, prim->dims.x, t, cr->t_min());
                                        $if (t < cr->t_max()) { candidate.commit(t); };
                                    })
                                    .trace()
                                    ->miss();
                lit = lit + select(0u, 1u, !blocked);
            };
            image.write(dispatch_id().xy(),
                        make_float4(make_float3(cast<float>(lit) * 0.25f), 1.f));
        });

    auto kernel_bounce = device.compile<2>(
        [&](ImageFloat image, Float3 camera) noexcept {
            auto coord = make_float2(cast<float>(dispatch_id().x), cast<float>(dispatch_id().y));
            auto size = make_float2(cast<float>(dispatch_size().x), cast<float>(dispatch_size().y));
            auto origin = def(camera);
            auto direction = def(make_camera_ray(coord, size));
            auto radiance = def(make_float3(0.f));
            auto throughput = def(make_float3(1.f));
            auto done = def(false);
            $for (b, 0u, 6u) {
                $if (!done) {
                    auto t_hit = def(1e30f);
                    auto ray = make_ray(origin, direction, 1e-4f, 1e30f);
                    sphere_accel->traverse(ray, {})
                        .on_procedural_candidate([&](ProceduralCandidate &candidate) noexcept {
                            auto h = candidate.hit();
                            auto prim = sphere_prim_buffer->read(h.prim);
                            auto cr = candidate.ray();
                            auto t = def(cr->t_max());
                            luisa::test::zoo::zoo_sphere(cr->origin(), cr->direction(),
                                                         prim->center, prim->dims.x, t, cr->t_min());
                            $if (t < t_hit) {
                                t_hit = t;
                                candidate.commit(t);
                            };
                        })
                        .trace();
                    $if (t_hit > 1e29f) {
                        radiance = radiance + throughput * 0.6f;
                        done = true;
                    } $else {
                        auto p_hit = origin + direction * t_hit;
                        auto n = normalize(p_hit);
                        auto u = luisa::test::zoo::sampling::sample3(0u, dispatch_id().x + b * 131u, b);
                        auto is_specular = u.z < 0.5f;
                        auto t0 = def(make_float3(1.f, 0.f, 0.f));
                        auto t1 = def(make_float3(0.f, 1.f, 0.f));
                        luisa::test::zoo::sampling::orthonormal_basis(n, t0, t1);
                        auto v_local = make_float3(dot(direction, t0), dot(direction, t1), dot(direction, n));
                        auto h_local = luisa::test::zoo::sampling::ggx_vndf(u.xy(), -v_local, 0.35f);
                        auto h = normalize(t0 * h_local.x + t1 * h_local.y + n * h_local.z);
                        auto specular = normalize(2.f * dot(direction, h) * h - direction);
                        auto diffuse = luisa::test::zoo::sampling::cosine_hemisphere(u.xy(), n);
                        direction = select(diffuse, specular, is_specular);
                        // keep the ray in the upper hemisphere
                        $if (dot(direction, n) < 1e-3f) { direction = diffuse; };
                        throughput = throughput * select(0.62f, 0.9f, is_specular);
                        origin = p_hit + n * 2e-3f;
                    };
                };
            };
            image.write(dispatch_id().xy(), make_float4(radiance, 1.f));
        });

    // --- measurement ----------------------------------------------------------
    auto measure = [&]<typename Dispatch>(luisa::string_view name, double rays_per_pixel,
                                         Dispatch &&dispatch) noexcept {
        auto selected = luisa::string_view{options.workload};
        if (selected != "all" && selected != name) { return; }
        luisa::vector<double> samples;
        samples.reserve(options.rounds);
        for (uint i = 0u; i < 2u; i++) {// warm up
            stream << dispatch() << synchronize();
        }
        for (uint i = 0u; i < options.rounds; i++) {
            CommandList list;
            for (uint r = 0u; r < options.reps; r++) { list << dispatch(); }
            clock.tic();
            stream << list.commit() << synchronize();
            samples.emplace_back(clock.toc() / static_cast<double>(options.reps));
        }
        auto stats = summarize(std::move(samples));
        auto rays = static_cast<double>(width) * static_cast<double>(height) * rays_per_pixel;
        auto mray = rays / (stats.median * 1e3);
        LUISA_INFO("RESULT backend={} workload={} width={} height={} reps={} rays={:.2f} min_ms={:.4f} median_ms={:.4f} mean_ms={:.4f} Mray_s={:.2f}",
                   backend, name, width, height, options.reps, rays_per_pixel,
                   stats.min, stats.median, stats.mean, mray);
    };

    measure("zoo", 1.0, [&]() noexcept {
        return kernel_zoo(target_image, camera).dispatch(width, height);
    });
    measure("many", 1.0, [&]() noexcept {
        return kernel_many(target_image, camera).dispatch(width, height);
    });
    measure("shadow", 5.0, [&]() noexcept {
        return kernel_shadow(shadow_image, camera).dispatch(width, height);
    });
    measure("bounce", 6.0, [&]() noexcept {
        return kernel_bounce(target_image, camera).dispatch(width, height);
    });
    return 0;
}