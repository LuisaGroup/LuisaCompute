#include <cmath>
#include <cstdlib>

#include <luisa/luisa-compute.h>
#include <luisa/dsl/sugar.h>
#include <stb/stb_image_write.h>

using namespace luisa;
using namespace luisa::compute;

int main(int argc, char *argv[]) {

    Context context{argv[0]};
    if (argc <= 1) { exit(1); }

    static constexpr uint width = 1280u;
    static constexpr uint height = 720u;
    static constexpr int max_ray_depth = 6;
    static constexpr float eps = 1e-4f;
    static constexpr float inf = 1e10f;
    static constexpr float fov = 0.23f;
    static constexpr float dist_limit = 100.0f;
    static constexpr float3 camera_pos = make_float3(0.0f, 0.32f, 3.7f);
    static constexpr float3 light_pos = make_float3(-1.5f, 0.6f, 0.3f);
    static constexpr float3 light_normal = make_float3(1.0f, 0.0f, 0.0f);
    static constexpr float light_radius = 2.0f;

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
        Float phi = 2.0f * pi * rand(seed);
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

    Callable ray_march = [&sdf](Float3 p, Float3 d) noexcept {
        Float dist = def(0.0f);
        $for (j, 100) {
            Float s = sdf(p + dist * d);
            $if (s <= 1e-6f | dist >= inf) { $break; };
            dist += s;
        };
        return min(dist, inf);
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

    Callable next_hit = [&ray_march, &sdf_normal](Float &closest, Float3 &normal, Float3 &c, Float3 pos, Float3 d) noexcept {
        closest = inf;
        normal = make_float3();
        c = make_float3();
        Float ray_march_dist = ray_march(pos, d);
        $if (ray_march_dist < min(dist_limit, closest)) {
            closest = ray_march_dist;
            Float3 hit_pos = pos + d * closest;
            normal = sdf_normal(hit_pos);
            Int t = cast<int>((hit_pos.x + 10.0f) * 1.1f + 0.5f) % 3;
            c = make_float3(0.4f) + make_float3(0.3f, 0.2f, 0.3f) * ite(t == make_int3(0, 1, 2), 1.0f, 0.0f);
        };
    };

    coroutine::Coroutine coro = [&](ImageUInt seed_image,
                                    ImageFloat accum_image,
                                    UInt frame_index) noexcept {
        Float2 resolution = make_float2(width, height);
        UInt2 coord = dispatch_id().xy();

        $if (frame_index == 0u) {
            seed_image.write(coord, make_uint4(tea(coord.x, coord.y)));
            accum_image.write(coord, make_float4(make_float3(0.0f), 1.0f));
        };

        Float aspect_ratio = resolution.x / resolution.y;
        Float3 pos = def(camera_pos);
        UInt seed = seed_image.read(coord).x;
        Float ux = rand(seed);
        Float uy = rand(seed);
        Float2 uv = make_float2(coord.x + ux, height - 1u - coord.y + uy);
        Float3 d = make_float3(
            2.0f * fov * uv / resolution.y - fov * make_float2(aspect_ratio, 1.0f) - 1e-5f, -1.0f);
        d = normalize(d);
        Float3 throughput = def(make_float3(1.0f, 1.0f, 1.0f));
        Float hit_light = def(0.0f);
        $for (depth, max_ray_depth) {
            $suspend("2");
            Float closest = def(0.0f);
            Float3 normal = def(make_float3());
            Float3 c = def(make_float3());
            next_hit(closest, normal, c, pos, d);
            Float dist_to_light = intersect_light(pos, d);
            $if (dist_to_light < closest) {
                hit_light = 1.0f;
                $break;
            };
            $if (length_squared(normal) == 0.0f) {
                $break;
            };
            Float3 hit_pos = pos + closest * d;
            d = out_dir(normal, seed);
            pos = hit_pos + 1e-4f * d;
            throughput *= c;
        };
        $suspend("3");
        Float3 accum_color = lerp(accum_image.read(coord).xyz(), throughput.xyz() * hit_light, 1.0f / (frame_index + 1.0f));
        accum_image.write(coord, make_float4(accum_color, 1.0f));
        seed_image.write(coord, make_uint4(seed));
    };

    auto device = context.create_device(argv[1]);
    auto stream = device.create_stream();
    constexpr auto resolution = make_uint2(width, height);

    Image<uint> seed_image = device.create_image<uint>(PixelStorage::INT1, width, height);
    Image<float> accum_image = device.create_image<float>(PixelStorage::FLOAT4, width, height);

    coroutine::StateMachineCoroSchedulerConfig config{
        .block_size = make_uint3(64u, 1u, 1u),
        .shared_memory = false,
        .shared_memory_soa = false};
    coroutine::StateMachineCoroScheduler scheduler{device, coro, config};

    auto clear_shader = device.compile<2>([&] {
        auto coord = dispatch_id().xy();
        accum_image->write(coord, make_float4(make_float3(0.0f), 1.0f));
        seed_image->write(coord, make_uint4(coord.y * resolution.y + coord.x));
    });

    luisa::vector<std::byte> host_image(accum_image.view().size_bytes());
    stream << clear_shader().dispatch(resolution) << synchronize();
    Clock clk;
    auto samples = argc > 2 ? static_cast<uint>(std::strtoul(argv[2], nullptr, 10u)) : 128u;
    if (samples == 0u) { samples = 128u; }
    for (auto i = 0u; i < samples; ++i) {
        stream << scheduler(seed_image, accum_image, i).dispatch(width, height)
               << synchronize();
    }
    stream << synchronize();
    auto dt = clk.toc();
    LUISA_INFO("Time: {} ms ({} spp/s)", dt, samples * 1e3 / dt);

    stream << accum_image.copy_to(host_image.data())
           << synchronize();
    auto pixel_count = static_cast<size_t>(resolution.x) * static_cast<size_t>(resolution.y);
    auto pixels = reinterpret_cast<const float *>(host_image.data());
    auto max_rgb = 0.0f;
    auto sum_rgb = 0.0;
    auto nan_count = size_t{0u};
    for (auto i = 0u; i < pixel_count; i++) {
        auto r = pixels[i * 4u + 0u];
        auto g = pixels[i * 4u + 1u];
        auto b = pixels[i * 4u + 2u];
        if (!std::isfinite(r) || !std::isfinite(g) || !std::isfinite(b)) { nan_count++; }
        max_rgb = std::max(max_rgb, std::max(r, std::max(g, b)));
        sum_rgb += static_cast<double>(r) + static_cast<double>(g) + static_cast<double>(b);
    }
    LUISA_INFO("image stats: max_rgb = {}, avg_rgb = {}, non_finite_pixels = {}",
               max_rgb,
               sum_rgb / static_cast<double>(pixel_count * 3u),
               nan_count);
    stbi_write_hdr("test_sdf.hdr", resolution.x, resolution.y, 4, reinterpret_cast<float *>(host_image.data()));
}
