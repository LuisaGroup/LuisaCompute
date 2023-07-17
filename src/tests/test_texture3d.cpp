//
// Created by Mike on 5/24/2023.
//
#include <luisa/core/logging.h>
#include <luisa/core/clock.h>
#include <luisa/runtime/context.h>
#include <luisa/runtime/device.h>
#include <luisa/runtime/stream.h>
#include <luisa/dsl/syntax.h>
#include <luisa/dsl/sugar.h>
#include <luisa/gui/window.h>
#include <luisa/runtime/swapchain.h>

using namespace luisa;
using namespace luisa::compute;

struct PerlinSettings {
    int octave;
    float power;
    float frequency;
};

struct TRay {
    float3 origin;
    float3 direction;
};

struct Bbox {
    float3 min;
    float3 max;
};

LUISA_STRUCT(PerlinSettings, octave, power, frequency){};
LUISA_STRUCT(TRay, origin, direction){};
LUISA_STRUCT(Bbox, min, max){};

// credit: https://github.com/nvpro-samples/vk_mini_samples/tree/main/samples/texture_3d/shaders (Apache License 2.0)
int main(int argc, char *argv[]) {

    Context context{argv[0]};

    if (argc <= 1) {
        LUISA_INFO("Usage: {} <backend>. <backend>: cuda, dx, cpu, metal", argv[0]);
        exit(1);
    }
    Device device = context.create_device(argv[1]);

    static constexpr auto mod289 = [](auto x) noexcept { return x - floor(x * (1.f / 289.f)) * 289.f; };
    static constexpr auto permute = [](auto x) noexcept { return mod289(((x * 34.f) + 1.f) * x); };
    static constexpr auto taylor_inv_sqrt = [](auto r) noexcept { return 1.79284291400159f - 0.85373472095314f * r; };
    static constexpr auto fade = [](auto t) noexcept { return t * t * t * (t * (t * 6.f - 15.f) + 10.f); };

    Callable perlin = [](Float3 P) noexcept -> Float {
        Float3 Pi0 = floor(P); // Integer part for indexing
        Float3 Pi1 = Pi0 + 1.f;// Integer part + 1
        Pi0 = mod289(Pi0);
        Pi1 = mod289(Pi1);
        Float3 Pf0 = fract(P); // Fractional part for interpolation
        Float3 Pf1 = Pf0 - 1.f;// Fractional part - 1.0
        Float4 ix = make_float4(Pi0.x, Pi1.x, Pi0.x, Pi1.x);
        Float4 iy = make_float4(Pi0.yy(), Pi1.yy());
        Float4 iz0 = Pi0.zzzz();
        Float4 iz1 = Pi1.zzzz();

        Float4 ixy = permute(permute(ix) + iy);
        Float4 ixy0 = permute(ixy + iz0);
        Float4 ixy1 = permute(ixy + iz1);

        Float4 gx0 = ixy0 / 7.f;
        Float4 gy0 = fract(floor(gx0) / 7.f) - .5f;
        gx0 = fract(gx0);
        Float4 gz0 = .5f - abs(gx0) - abs(gy0);
        Float4 sz0 = step(gz0, 0.f);
        gx0 -= sz0 * (step(0.f, gx0) - .5f);
        gy0 -= sz0 * (step(0.f, gy0) - .5f);

        Float4 gx1 = ixy1 / 7.f;
        Float4 gy1 = fract(floor(gx1) / 7.f) - .5f;
        gx1 = fract(gx1);
        Float4 gz1 = .5f - abs(gx1) - abs(gy1);
        Float4 sz1 = step(gz1, 0.f);
        gx1 -= sz1 * (step(0.f, gx1) - .5f);
        gy1 -= sz1 * (step(0.f, gy1) - .5f);

        Float3 g000 = make_float3(gx0.x, gy0.x, gz0.x);
        Float3 g100 = make_float3(gx0.y, gy0.y, gz0.y);
        Float3 g010 = make_float3(gx0.z, gy0.z, gz0.z);
        Float3 g110 = make_float3(gx0.w, gy0.w, gz0.w);
        Float3 g001 = make_float3(gx1.x, gy1.x, gz1.x);
        Float3 g101 = make_float3(gx1.y, gy1.y, gz1.y);
        Float3 g011 = make_float3(gx1.z, gy1.z, gz1.z);
        Float3 g111 = make_float3(gx1.w, gy1.w, gz1.w);

        Float4 norm0 = taylor_inv_sqrt(make_float4(
            dot(g000, g000), dot(g010, g010),
            dot(g100, g100), dot(g110, g110)));

        g000 *= norm0.x;
        g010 *= norm0.y;
        g100 *= norm0.z;
        g110 *= norm0.w;
        Float4 norm1 = taylor_inv_sqrt(make_float4(
            dot(g001, g001), dot(g011, g011),
            dot(g101, g101), dot(g111, g111)));
        g001 *= norm1.x;
        g011 *= norm1.y;
        g101 *= norm1.z;
        g111 *= norm1.w;

        Float n000 = dot(g000, Pf0);
        Float n100 = dot(g100, make_float3(Pf1.x, Pf0.yz()));
        Float n010 = dot(g010, make_float3(Pf0.x, Pf1.y, Pf0.z));
        Float n110 = dot(g110, make_float3(Pf1.xy(), Pf0.z));
        Float n001 = dot(g001, make_float3(Pf0.xy(), Pf1.z));
        Float n101 = dot(g101, make_float3(Pf1.x, Pf0.y, Pf1.z));
        Float n011 = dot(g011, make_float3(Pf0.x, Pf1.yz()));
        Float n111 = dot(g111, Pf1);

        Float3 fade_xyz = fade(Pf0);
        Float4 n_z = lerp(make_float4(n000, n100, n010, n110),
                        make_float4(n001, n101, n011, n111),
                        fade_xyz.z);
        Float2 n_yz = lerp(n_z.xy(), n_z.zw(), fade_xyz.y);
        Float n_xyz = lerp(n_yz.x, n_yz.y, fade_xyz.x);
        return 2.2f * n_xyz;
    };

    auto make_perlin_noise = device.compile<3>([&](VolumeFloat out, Var<PerlinSettings> settings) noexcept {
        Float v = def(0.f);
        Float scale = settings.power;
        Float freq = settings.frequency / cast<float>(dispatch_size_x());
        $for(oct, settings.octave) {
            v += perlin(make_float3(dispatch_id()) * freq) / scale;
            freq *= 2.f;
            scale *= settings.power;
        };
        out.write(dispatch_id(), make_float4(v));
    });

    Callable calculate_shading = [](Float3 surface_color,
                                    Float3 view_direction,
                                    Float3 surface_normal,
                                    Float3 light_direction) noexcept {
        Float3 shaded_color = surface_color;
        Float3 world_up_direction = make_float3(0.f, 1.f, 0.f);
        Float3 reflected_light_direction = normalize(reflect(-light_direction, surface_normal));

        // Diffuse + Specular
        Float light_intensity = max(dot(surface_normal, light_direction) +
                                       pow(max(0.f, dot(reflected_light_direction,
                                                        view_direction)),
                                           32.f),
                                   0.f);
        shaded_color *= light_intensity;

        // Ambient term (sky effect)
        Float3 sky_ambient_color = lerp(make_float3(.1f, .1f, .4f),
                                      make_float3(.8f, .6f, .2f),
                                      dot(surface_normal, world_up_direction) * .5f + .5f) *
                                 .2f;
        return shaded_color + sky_ambient_color;
    };

    Callable intersect_cube = [](Var<TRay> ray, Var<Bbox> bbox,
                                 Float3 &p1, Float3 &p2) noexcept {
        Float3 inv_dir = make_float3(1.f) / ray.direction;
        Float3 t_min = (bbox.min - ray.origin) * inv_dir;
        Float3 t_max = (bbox.max - ray.origin) * inv_dir;
        Float3 t1 = min(t_min, t_max);
        Float3 t2 = max(t_min, t_max);
        Float t_near = max(max(t1.x, t1.y), t1.z);
        Float t_far = min(min(t2.x, t2.y), t2.z);
        Bool hit = t_near <= t_far & t_far > 0.f;
        $if(hit) {
            p1 = ray.origin + ray.direction * max(t_near, 0.f);
            p2 = ray.origin + ray.direction * t_far;
        };
        return hit;
    };

    Callable compute_volume_gradient = [](BindlessVar bindless, Float3 p, Float voxel_size) noexcept {
        auto v = [&bindless](auto p) noexcept { return bindless.tex3d(0u).sample(p).x; };
        Float inc = voxel_size * .5f;
        Float dx = v(p - make_float3(inc, 0.f, 0.f)) - v(p + make_float3(inc, 0.f, 0.f));
        Float dy = v(p - make_float3(0.f, inc, 0.f)) - v(p + make_float3(0.f, inc, 0.f));
        Float dz = v(p - make_float3(0.f, 0.f, inc)) - v(p + make_float3(0.f, 0.f, inc));
        return normalize(make_float3(dx, dy, dz));
    };

    Callable ray_marching = [](BindlessVar bindless,
                               Float3 p1, Float3 p2,
                               Int num_steps,
                               Float threshold,
                               Float3 &hit_point) noexcept {
        auto v = [&bindless](auto p) noexcept { return bindless.tex3d(0u).sample(p).x; };
        Float3 step_size = (p2 - p1) / cast<float>(num_steps);
        hit_point = p1;
        Float3 prev_point = hit_point;
        Float value = v(hit_point);
        Float prev_value = value;
        Bool hit = def(false);
        $for(i, num_steps) {
            $if(value > threshold) {
                Float t = clamp((threshold - prev_value) / (value - prev_value), 0.f, 1.f);
                hit_point = lerp(prev_point, hit_point, t);
                hit = true;
                $break;
            };
            prev_value = value;
            prev_point = hit_point;
            hit_point += step_size;
            value = v(hit_point);
        };
        return hit;
    };

    Callable trace = [&](BindlessVar bindless,
                         Float3 camera_pos,
                         Float fov,
                         Float2 jitter) noexcept {
        Float2 size = make_float2(dispatch_size().xy());
        Float2 uv = (make_float2(dispatch_id().xy()) + jitter) / size * 2.f - 1.f;
        Float aspect = size.x / size.y;
        Float scale = tan(radians(fov) * .5f);
        Float2 p = make_float2(uv.x * aspect * scale, uv.y * scale);
        Float3 front = normalize(-camera_pos);
        Float3 right = normalize(cross(front, make_float3(0.f, 1.f, 0.f)));
        Float3 up = normalize(cross(right, front));
        Float3 ray_dir = normalize(p.x * right + p.y * up + front);
        Var<TRay> ray = def<TRay>(camera_pos, ray_dir);

        Float3 p1 = def(make_float3(0.f));
        Float3 p2 = def(make_float3(0.f));
        Var<Bbox> bbox = def<Bbox>(make_float3(-.48f), make_float3(.48f));

        Float3 albedo = make_float3(.5f);
        Float3 color = def(make_float3(.2f, .4f, .6f));
        $if(intersect_cube(ray, bbox, p1, p2)) {
            p1 = p1 - bbox.min / (bbox.max - bbox.min);
            p2 = p2 - bbox.min / (bbox.max - bbox.min);
            Float3 hit_point = def(make_float3(0.f));
            constexpr int steps = 100;
            constexpr float threshold = 5e-3f;
            $if(ray_marching(bindless, p1, p2, steps, threshold, hit_point)) {
                Float volume_size = cast<float>(bindless.tex3d(0u).size().x);
                Float3 normal = -compute_volume_gradient(bindless, hit_point, 1.f / volume_size);
                Float3 to_light = normalize(make_float3(1.f));
                $if(dot(normal, ray.direction) > 0.f & dot(normal, to_light) > 0.f) {
                    color = calculate_shading(albedo, -ray.direction, normal, to_light);
                }
                $else {
                    color = .15f * albedo;
                };
            };
        };
        return color;
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

    auto make_sampler_states = device.compile<2>([&](ImageUInt seed_image) noexcept {
        UInt2 p = dispatch_id().xy();
        UInt state = tea(p.x, p.y);
        seed_image.write(p, make_uint4(state));
    });

    Callable lcg = [](UInt &state) noexcept {
        constexpr uint lcg_a = 1664525u;
        constexpr uint lcg_c = 1013904223u;
        state = lcg_a * state + lcg_c;
        return cast<float>(state & 0x00ffffffu) *
               (1.0f / static_cast<float>(0x01000000u));
    };

    auto render = device.compile<2>([&](ImageFloat accum,
                                        BindlessVar bindless,
                                        Float3 camera_pos,
                                        ImageUInt seeds,
                                        Float fov) noexcept {
        UInt state = seeds.read(dispatch_id().xy()).x;
        Float rx = lcg(state);
        Float ry = lcg(state);
        seeds.write(dispatch_id().xy(), make_uint4(state));
        Float3 color = trace(bindless, camera_pos, fov, make_float2(rx, ry));
        Float4 old = accum.read(dispatch_id().xy());
        Float3 c = lerp(old.xyz(), color, 1.f / (old.w + 1.f));
        accum.write(dispatch_id().xy(), make_float4(c, old.w + 1.f));
    });

    auto clear = device.compile<2>([&](ImageFloat accum) noexcept {
        accum.write(dispatch_id().xy(), make_float4(0.f));
    });

    auto blit = device.compile<2>([&](ImageFloat accum, ImageFloat output) noexcept {
        output.write(dispatch_id().xy(), make_float4(accum.read(dispatch_id().xy()).xyz(), 1.f));
    });

    static constexpr uint volume_size = 256u;
    static constexpr PerlinSettings settings{
        .octave = 4, .power = 1.f, .frequency = 1.f};

    static constexpr uint2 resolution = make_uint2(1024u);

    Stream stream = device.create_stream(StreamTag::GRAPHICS);
    BindlessArray bindless = device.create_bindless_array(1u);
    Volume<float> volume = device.create_volume<float>(PixelStorage::FLOAT1, make_uint3(volume_size));
    Image<uint> seeds = device.create_image<uint>(PixelStorage::INT1, make_uint2(1024u));
    bindless.emplace_on_update(0u, volume, Sampler::linear_point_edge());

    stream << bindless.update()
           << make_perlin_noise(volume, settings)
                  .dispatch(make_uint3(volume_size))
           << make_sampler_states(seeds).dispatch(resolution);

    Window window{"Display", resolution};
    Swapchain swapchain = device.create_swapchain(
        window.native_handle(), stream,
        resolution, false, true, 3u);
    Image<float> accum = device.create_image<float>(PixelStorage::FLOAT4, resolution);
    Image<float> display = device.create_image<float>(swapchain.backend_storage(), resolution);
    bool dirty = true;

    float fov = 30.f;
    float3 camera_pos = make_float3(2.f, -2.f, -2.f);
    Clock clk;
    while (!window.should_close()) {
        window.poll_events();
        float dt = static_cast<float>(clk.toc() * 1e-3f);
        clk.tic();
        float3 front = normalize(camera_pos);
        float3 right = normalize(cross(front, make_float3(0.f, 1.f, 0.f)));
        float3 up = normalize(cross(right, front));
        if (window.is_key_down(KEY_W) && front.y < .95f) {
            float4x4 R = rotation(right, dt);
            camera_pos = make_float3x3(R) * camera_pos;
            dirty = true;
        }
        if (window.is_key_down(KEY_S) && front.y > -.95f) {
            float4x4 R = rotation(right, -dt);
            camera_pos = make_float3x3(R) * camera_pos;
            dirty = true;
        }
        if (window.is_key_down(KEY_A)) {
            float4x4 R = rotation(up, -dt);
            camera_pos = make_float3x3(R) * camera_pos;
            dirty = true;
        }
        if (window.is_key_down(KEY_D)) {
            float4x4 R = rotation(up, dt);
            camera_pos = make_float3x3(R) * camera_pos;
            dirty = true;
        }
        if (window.is_key_down(KEY_MINUS)) {
            fov = clamp(fov * 1.02f, 5.f, 170.f);
            dirty = true;
        }
        if (window.is_key_down(KEY_EQUAL)) {
            fov = clamp(fov / 1.02f, 5.f, 170.f);
            dirty = true;
        }

        CommandList cmds;
        cmds.reserve(3u, 0u);

        if (dirty) {
            cmds << clear(accum).dispatch(resolution);
            dirty = false;
        }
        cmds << render(accum, bindless, camera_pos, seeds, fov).dispatch(resolution)
             << blit(accum, display).dispatch(resolution);
        stream << cmds.commit() << swapchain.present(display);
    }
    stream << synchronize();
}

