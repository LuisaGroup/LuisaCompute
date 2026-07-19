#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <limits>
#include <memory>
#include <numbers>
#include <optional>
#include <string_view>

#include <stb/stb_image_write.h>

#include "common/reference_compare.h"
#include "common/coro_scheduler_options.h"

#include <luisa/luisa-compute.h>
#include <luisa/coro/schedulers/persistent_threads.h>
#include <luisa/coro/schedulers/state_machine.h>
#include <luisa/coro/schedulers/wavefront.h>
#include <luisa/core/clock.h>
#include <luisa/core/logging.h>
#include <luisa/dsl/sugar.h>

using namespace luisa;
using namespace luisa::compute;
using namespace luisa::compute::coro;

int main(int argc, char *argv[]) {

    auto opts = luisa::ref::ExampleOptions::parse(argc, argv);
    auto scheduler_kind = luisa::example::parse_coro_scheduler_arg(argc, argv);

    static constexpr int max_ray_depth = 6;
    static constexpr float eps = 1e-4f;
    static constexpr float inf = 1e10f;
    static constexpr float fov = 0.23f;
    static constexpr float dist_limit = 100.0f;
    static constexpr float3 camera_pos = make_float3(0.0f, 0.32f, 3.7f);
    static constexpr float3 light_pos = make_float3(-1.5f, 0.6f, 0.3f);
    static constexpr float3 light_normal = make_float3(1.0f, 0.0f, 0.0f);
    static constexpr float light_radius = 2.0f;

    Context ctx{argv[0]};
    if (argc <= 1) {
        LUISA_INFO("Usage: {} <backend> [--spp N] [--compare ref.png] [--scheduler state_machine|wavefront|persistent]", argv[0]);
        exit(1);
    }
    Device device = ctx.create_device(argv[1]);
    Stream stream = device.create_stream(StreamTag::COMPUTE);

    Callable intersect_light = [](Float3 pos, Float3 d) noexcept {
        Float cos_w = dot(-d, light_normal);
        Float dist = dot(d, light_pos - pos);
        Float D = dist / cos_w;
        Float dist_to_center = distance_squared(light_pos, pos + D * d);
        Bool valid = cos_w > 0.0f & dist > 0.0f & dist_to_center < light_radius * light_radius;
        return ite(valid, D, inf);
    };

    Callable tea = [](UInt v0, UInt v1) noexcept {
        Var s0 = 0u;
        for (uint n = 0u; n < 4u; n++) {
            s0 += 0x9e3779b9u;
            v0 += ((v1 << 4) + 0xa341316cu) ^ (v1 + s0) ^ ((v1 >> 5u) + 0xc8013ea4u);
            v1 += ((v0 << 4) + 0xad90777du) ^ (v0 + s0) ^ ((v0 >> 5u) + 0x7e95761eu);
        }
        return v0;
    };

    Callable rand = [](UInt &state) noexcept {
        constexpr uint lcg_a = 1664525u;
        constexpr uint lcg_c = 1013904223u;
        state = lcg_a * state + lcg_c;
        return cast<float>(state) / cast<float>(std::numeric_limits<uint>::max());
    };

    Callable out_dir = [&rand](Float3 n, UInt &seed) noexcept {
        Float3 u = ite(
            abs(n.y) < 1.0f - eps,
            normalize(cross(n, make_float3(0.0f, 1.0f, 0.0f))),
            make_float3(1.f, 0.f, 0.f));
        Float3 v = cross(n, u);
        Float phi = 2.0f * std::numbers::pi_v<float> * rand(seed);
        Float ay = sqrt(rand(seed));
        Float ax = sqrt(1.0f - ay * ay);
        return ax * (cos(phi) * u + sin(phi) * v) + ay * n;
    };

    Callable make_nested = [](Float f) noexcept {
        static constexpr float freq = 40.0f;
        f *= freq;
        f = ite(f < 0.f, ite(cast<int>(f) % 2 == 0, 1.f - fract(f), fract(f)), f);
        return (f - 0.2f) * (1.0f / freq);
    };

    Callable sdf = [&make_nested](Float3 o) noexcept {
        Float wall = min(o.y + 0.1f, o.z + 0.4f);
        Float sphere = distance(o, make_float3(0.0f, 0.35f, 0.0f)) - 0.36f;
        Float3 q = abs(o - make_float3(0.8f, 0.3f, 0.0f)) - 0.3f;
        Float box = length(max(q, 0.0f)) + min(max(max(q.x, q.y), q.z), 0.0f);
        Float3 O = o - make_float3(-0.8f, 0.3f, 0.0f);
        Float2 d = make_float2(length(make_float2(O.x, O.z)) - 0.3f, abs(O.y) - 0.3f);
        Float cylinder = min(max(d.x, d.y), 0.0f) + length(max(d, 0.0f));
        Float geometry = make_nested(min(min(sphere, box), cylinder));
        Float g = max(geometry, -(0.32f - (o.y * 0.6f + o.z * 0.8f)));
        return min(wall, g);
    };

    Callable sdf_normal = [&sdf](Float3 p) noexcept {
        static constexpr float d = 1e-3f;
        Float3 n = def(make_float3());
        Float sdf_center = sdf(p);
        for (uint i = 0; i < 3; i++) {
            Float3 inc = p;
            inc[i] += d;
            n[i] = (1.0f / d) * (sdf(inc) - sdf_center);
        }
        return normalize(n);
    };

    Coroutine coro = [&](ImageUInt seed_image, ImageFloat accum_image, UInt frame_index) noexcept {
        UInt2 coord = dispatch_id().xy();
        $if (frame_index == 0u) {
            seed_image.write(coord, make_uint4(tea(coord.x, coord.y)));
            accum_image.write(coord, make_float4(make_float3(0.0f), 1.0f));
        };

        $suspend("setup");
        Float2 resolution = make_float2(dispatch_size().xy());
        Float aspect_ratio = resolution.x / resolution.y;
        Float3 pos = def(camera_pos);
        UInt seed = seed_image.read(coord).x;
        Float ux = rand(seed);
        Float uy = rand(seed);
        Float2 uv = make_float2(
            coord.x.cast<float>() + ux,
            (dispatch_size().y - 1u - coord.y).cast<float>() + uy);
        Float3 d = make_float3(
            2.0f * fov * uv / resolution.y - fov * make_float2(aspect_ratio, 1.0f) - 1e-5f,
            -1.0f);
        d = normalize(d);
        Float3 throughput = def(make_float3(1.0f, 1.0f, 1.0f));
        Float hit_light = def(0.0f);

        $for (depth, max_ray_depth) {
            Float closest = def(inf);
            Float3 normal = def(make_float3());
            Float3 c = def(make_float3());
            Float ray_march_dist = def(0.0f);
            $for (j, 100) {
                Float s = sdf(pos + ray_march_dist * d);
                $if (s <= 1e-6f | ray_march_dist >= inf) { $break; };
                ray_march_dist += s;
                $suspend("ray_march");
            };
            ray_march_dist = min(ray_march_dist, inf);
            $if (ray_march_dist < min(dist_limit, closest)) {
                closest = ray_march_dist;
                Float3 hit_pos = pos + d * closest;
                normal = sdf_normal(hit_pos);
                Int t = cast<int>((hit_pos.x + 10.0f) * 1.1f + 0.5f) % 3;
                c = make_float3(0.4f) + make_float3(0.3f, 0.2f, 0.3f) *
                                            ite(t == make_int3(0, 1, 2), 1.0f, 0.0f);
            };
            $suspend("next_hit");
            Float dist_to_light = intersect_light(pos, d);
            $if (dist_to_light < closest) {
                hit_light = 1.0f;
                $break;
            };
            $if (length_squared(normal) == 0.0f) { $break; };
            Float3 hit_pos = pos + closest * d;
            d = out_dir(normal, seed);
            pos = hit_pos + 1e-4f * d;
            throughput *= c;
            $suspend("bounce");
        };
        Float3 accum_color = lerp(
            accum_image.read(coord).xyz(),
            throughput.xyz() * hit_light,
            1.0f / (frame_index.cast<float>() + 1.0f));
        accum_image.write(coord, make_float4(accum_color, 1.0f));
        seed_image.write(coord, make_uint4(seed));
        $suspend("write");
    };

    LUISA_INFO("Coroutine compiled: {} subroutines, {} graph nodes",
               coro.subroutine_count(), coro.graph().node_count());
    LUISA_INFO("Coroutine frame: {} fields, payload {} B, struct {} B",
               coro.frame().frame_field_count(), coro.frame().total_size(),
               coro.frame().frame_type()->size());
    LUISA_INFO("Coroutine frame R/W by subroutine:");
    luisa::example::dump_coro_frame_rw(coro);

    static constexpr uint width = 1280u;
    static constexpr uint height = 720u;
    static constexpr uint interval = 64u;
    static constexpr uint total_cells = width * height;

    using Scheduler = CoroScheduler<Image<uint>, Image<float>, uint>;
    std::unique_ptr<Scheduler> scheduler;
    switch (scheduler_kind) {
        case luisa::example::CoroSchedulerKind::state_machine:
            scheduler = std::make_unique<StateMachineCoroScheduler<Image<uint>, Image<float>, uint>>(device, coro);
            break;
        case luisa::example::CoroSchedulerKind::wavefront: {
            WavefrontCoroSchedulerConfig cfg{};
            scheduler = std::make_unique<WavefrontCoroScheduler<Image<uint>, Image<float>, uint>>(device, coro, cfg);
            break;
        }
        case luisa::example::CoroSchedulerKind::persistent: {
            PersistentThreadsCoroSchedulerConfig cfg{};
            scheduler = std::make_unique<PersistentThreadsCoroScheduler<Image<uint>, Image<float>, uint>>(device, coro, cfg);
            break;
        }
    }
    LUISA_INFO("CoroScheduler: {}", luisa::example::coro_scheduler_name(scheduler_kind));

    uint total_spp = opts.spp == 0u ? 1024u : opts.spp;
    Image<uint> seed_image = device.create_image<uint>(PixelStorage::INT1, width, height);
    Image<float> accum_image = device.create_image<float>(PixelStorage::FLOAT4, width, height);
    Image<float> ldr_image = device.create_image<float>(PixelStorage::BYTE4, width, height);

    Callable linear_to_srgb = [](Var<float3> x) noexcept {
        return clamp(select(1.055f * pow(x, 1.0f / 2.4f) - 0.055f,
                            12.92f * x,
                            x <= 0.00031308f),
                     0.0f, 1.0f);
    };
    Kernel2D hdr2ldr_kernel = [&](ImageFloat hdr_image, ImageFloat ldr, Float scale) noexcept {
        UInt2 coord = dispatch_id().xy();
        Float4 hdr = hdr_image.read(coord);
        Float3 ldr_value = linear_to_srgb(hdr.xyz() / hdr.w * scale);
        ldr.write(coord, make_float4(ldr_value, 1.0f));
    };
    auto hdr2ldr_shader = device.compile(hdr2ldr_kernel);

    Clock clock;
    uint spp_count = 0u;
    uint spp = 0u;
    while (spp < total_spp) {
        for (uint frame = 0u; frame < interval && spp + frame < total_spp; frame++) {
            stream << (*scheduler)(seed_image, accum_image, spp + frame).dispatch(width, height);
            spp_count++;
        }
        spp += interval;
    }
    stream << synchronize();
    double average_fps = spp_count / clock.toc() * 1000.0;
    LUISA_INFO("{} samples/s", average_fps);

    luisa::vector<uint8_t> host_image(width * height * 4u);
    stream << hdr2ldr_shader(accum_image, ldr_image, 2.0f).dispatch(width, height)
           << ldr_image.copy_to(luisa::span{host_image})
           << synchronize();
    if (stbi_write_png("coro_sdf.png", width, height, 4, host_image.data(), 0) == 0) {
        LUISA_ERROR("Failed to write 'coro_sdf.png'.");
        return 1;
    }

    if (opts.out_ref_path) {
        constexpr size_t pixel_count = width * height;
        constexpr size_t float_count = pixel_count * 4u;
        luisa::vector<float> accum_host(float_count);
        stream << accum_image.copy_to(luisa::span{accum_host}) << synchronize();

        for (size_t i = 0; i < accum_host.size(); ++i) {
            if (!std::isfinite(accum_host[i])) {
                LUISA_ERROR("Rendered raw accumulation contains a non-finite value at pixel {}, channel {}: {}.",
                            i / 4u, i % 4u, accum_host[i]);
                return 1;
            }
        }

        if (opts.out_ref_write) {
            std::ofstream ofs(opts.out_ref_path->string(), std::ios::binary);
            if (!ofs) {
                LUISA_ERROR("Failed to open out_ref file '{}' for writing.", opts.out_ref_path->string());
                return 1;
            }
            auto byte_count = static_cast<std::streamsize>(accum_host.size() * sizeof(float));
            ofs.write(reinterpret_cast<const char *>(accum_host.data()), byte_count);
            if (!ofs) {
                LUISA_ERROR("Failed to write complete out_ref file '{}'.", opts.out_ref_path->string());
                return 1;
            }
            ofs.close();
            if (!ofs) {
                LUISA_ERROR("Failed to finalize out_ref file '{}'.", opts.out_ref_path->string());
                return 1;
            }
            LUISA_INFO("Reference written to {} ({} floats, {} pixels)",
                       opts.out_ref_path->string(), accum_host.size(), pixel_count);
        } else {
            constexpr double absolute_tolerance = 1e-5;
            constexpr double relative_tolerance = 1e-4;
            auto expected_size = float_count * sizeof(float);
            std::ifstream ifs(opts.out_ref_path->string(), std::ios::binary | std::ios::ate);
            if (!ifs) {
                LUISA_ERROR("Failed to open out_ref file '{}' for reading; the reference is required.",
                            opts.out_ref_path->string());
                return 1;
            }
            auto file_size = static_cast<std::streamoff>(ifs.tellg());
            if (file_size != static_cast<std::streamoff>(expected_size)) {
                LUISA_ERROR("Reference file size mismatch: got {}, expected {} ({} floats).",
                            file_size, expected_size, float_count);
                return 1;
            }
            ifs.seekg(0, std::ios::beg);
            luisa::vector<float> ref_host(float_count);
            auto byte_count = static_cast<std::streamsize>(expected_size);
            if (!ifs.read(reinterpret_cast<char *>(ref_host.data()), byte_count)) {
                LUISA_ERROR("Failed to read complete out_ref file '{}'.", opts.out_ref_path->string());
                return 1;
            }

            double total_diff = 0.0;
            double max_diff = 0.0;
            for (size_t i = 0; i < float_count; ++i) {
                auto actual = static_cast<double>(accum_host[i]);
                auto expected = static_cast<double>(ref_host[i]);
                if (!std::isfinite(expected)) {
                    LUISA_ERROR("Raw reference contains a non-finite value at pixel {}, channel {}: {}.",
                                i / 4u, i % 4u, expected);
                    return 1;
                }
                auto diff = std::abs(actual - expected);
                auto limit = absolute_tolerance +
                             relative_tolerance * std::max(std::abs(actual), std::abs(expected));
                total_diff += diff;
                max_diff = std::max(max_diff, diff);
                if (diff > limit) {
                    LUISA_ERROR("Raw reference comparison failed at pixel {}, channel {}: got {}, expected {}, "
                                "abs_diff={} exceeds tolerance {}.",
                                i / 4u, i % 4u, actual, expected, diff, limit);
                    return 1;
                }
            }
            auto avg_diff = total_diff / static_cast<double>(float_count);
            LUISA_INFO("Reference comparison (raw float accum): PASSED "
                       "(avg_abs_diff={:.9f}, max_abs_diff={:.9f}, abs_tol={}, rel_tol={}).",
                       avg_diff, max_diff, absolute_tolerance, relative_tolerance);
        }
    }

    if (opts.compare_path) {
        auto result = luisa::ref::compare_with_reference_file(
            reinterpret_cast<const uint8_t *>(host_image.data()),
            width, height, 4,
            *opts.compare_path);
        LUISA_INFO("Reference comparison: {} ({})", result.passed ? "PASSED" : "FAILED", result.message);
        if (!result.passed) { return 1; }
    }
    return 0;
}
