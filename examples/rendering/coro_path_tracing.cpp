// Coroutine Path Tracing Example
// Demonstrates path tracing with coroutine schedulers and multiple $suspend
// points. Uses procedural geometry (sphere + ground plane).
//
// Pipeline limitation: local variables cannot cross $suspend boundaries
// in the current coro-split pass. The coroutine acts as a render-loop
// coordinator with suspend barriers, while the path-tracing computation
// runs in a regular Kernel2D.
//
// Usage: coro_path_tracing <backend> [--offline] [--spp N]
//   backend: cuda, dx, cpu, metal, fallback

#include <cstdlib>
#include <string_view>

#include <stb/stb_image_write.h>

#include <luisa/luisa-compute.h>
#include <luisa/dsl/sugar.h>
#include <luisa/dsl/coro_func.h>
#include <luisa/coro/schedulers/state_machine.h>

using namespace luisa;
using namespace luisa::compute;
using namespace luisa::compute::coro;

int main(int argc, char *argv[]) {

    log_level_verbose();

    Context context{argv[0]};
    if (argc <= 1) {
        LUISA_INFO("Usage: {} <backend> [--offline] [--spp N]. <backend>: cuda, dx, cpu, metal, fallback", argv[0]);
        exit(1);
    }

    std::string_view backend_name{argv[1]};
    Device device = context.create_device(backend_name);

    bool offline = false;
    uint total_spp = 64u;
    for (int i = 2; i < argc; i++) {
        if (std::string_view{argv[i]} == "--offline") {
            offline = true;
        } else if (std::string_view{argv[i]} == "--spp" && i + 1 < argc) {
            total_spp = static_cast<uint>(std::atoi(argv[++i]));
        }
    }

    auto spp_per_dispatch = (backend_name == "metal" ||
                             backend_name == "cpu" ||
                             backend_name == "fallback") ? 1u : 64u;
    auto passes = (total_spp + spp_per_dispatch - 1u) / spp_per_dispatch;

    static constexpr uint2 resolution = make_uint2(800u, 600u);
    Stream stream = device.create_stream();

    // Resources
    Buffer<float4> framebuffer = device.create_buffer<float4>(resolution.x * resolution.y);
    Buffer<float4> accum_buffer = device.create_buffer<float4>(resolution.x * resolution.y);
    Buffer<uint>  seed_buffer  = device.create_buffer<uint>(resolution.x * resolution.y);

    // TEA RNG for seeding
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
        constexpr uint a = 1664525u;
        constexpr uint c = 1013904223u;
        state = a * state + c;
        return cast<float>(state & 0x00ffffffu) *
               (1.0f / static_cast<float>(0x01000000u));
    };

    // ─── Shaders ──────────────────────────────────────────────────────
    auto init_seeds_shader = device.compile(Kernel2D([&](BufferUInt seeds) noexcept {
        UInt2 p = dispatch_id().xy();
        UInt2 res = make_uint2(800u, 600u);
        seeds.write(p.y * res.x + p.x, tea(p.x, p.y));
    }));

    // Path tracing kernel (procedural sphere + ground plane)
    auto pathtrace_shader = device.compile(Kernel2D([&](
        BufferFloat4 output, BufferUInt seeds, UInt spp_this_pass) noexcept {

        UInt2 coord = dispatch_id().xy();
        UInt2 res = make_uint2(800u, 600u);
        UInt idx = coord.y * res.x + coord.x;
        Float aspect = res.x.cast<float>() / res.y.cast<float>();
        UInt state = seeds.read(idx);

        Float3 accum = make_float3(0.0f);
        $for (s, spp_this_pass) {
            Float rx = lcg(state);
            Float ry = lcg(state);
            Float2 uv = (make_float2(coord) + make_float2(rx, ry)) /
                            make_float2(res) * 2.0f - 1.0f;

            Float3 origin = make_float3(0.0f, 1.5f, -3.0f);
            Float3 target = make_float3(0.0f, 0.6f, 0.0f);
            Float3 look_dir = normalize(target - origin);
            Float3 right = normalize(cross(look_dir, make_float3(0.0f, 1.0f, 0.0f)));
            Float3 up = cross(right, look_dir);
            Float3 rd = normalize(look_dir +
                                   right * uv.x * aspect * 0.5f +
                                   up * (-uv.y) * 0.5f);

            // Sphere at (0, 0.7, 0) radius 0.7
            Float3 sc = make_float3(0.0f, 0.7f, 0.0f);
            Float3 oc = origin - sc;
            Float a = dot(rd, rd);
            Float b = 2.0f * dot(oc, rd);
            Float c = dot(oc, oc) - 0.49f;
            Float disc = b * b - 4.0f * a * c;
            Float t_sphere = ite(disc < 0.0f, -1.0f,
                                 (-b - sqrt(disc)) / (2.0f * a));
            Float t_ground = ite(abs(rd.y) < 1e-6f, -1.0f, -origin.y / rd.y);

            Bool hg = t_ground > 0.0f;
            Bool hs = t_sphere > 0.0f;
            Bool sc_is = hs & (!hg | t_sphere < t_ground);
            Float t = ite(sc_is, t_sphere, ite(hg, t_ground, -1.0f));

            Float3 color;
            $if (t > 0.0f) {
                Float3 hit_pos = origin + rd * t;
                Float3 normal = ite(sc_is,
                                    normalize(hit_pos - sc),
                                    make_float3(0.0f, 1.0f, 0.0f));
                Float3 albedo = ite(sc_is,
                                    make_float3(0.8f, 0.3f, 0.2f),
                                    make_float3(0.4f, 0.5f, 0.3f));
                Float3 light_dir = normalize(make_float3(1.0f, 2.0f, 1.0f));
                Float ndotl = max(dot(normal, light_dir), 0.0f);
                color = albedo * (0.05f + 0.95f * ndotl);

                // Shadow ray
                Float3 so = hit_pos + normal * 1e-3f;
                Float3 soc = so - sc;
                Float sd_a = dot(light_dir, light_dir);
                Float sd_b = 2.0f * dot(soc, light_dir);
                Float sd_c = dot(soc, soc) - 0.49f;
                Float sd_disc = sd_b * sd_b - 4.0f * sd_a * sd_c;
                Float st = ite(sd_disc < 0.0f, -1.0f,
                               (-sd_b - sqrt(sd_disc)) / (2.0f * sd_a));
                Float sgt = ite(abs(light_dir.y) < 1e-6f, -1.0f, -so.y / light_dir.y);
                $if ((st > 0.0f | (sgt > 0.0f & sgt < 1e6f)) & !sc_is) {
                    color = albedo * 0.05f;
                };
            }
            $else {
                Float sky_t = 0.5f * (rd.y + 1.0f);
                color = lerp(make_float3(1.0f, 1.0f, 1.0f),
                              make_float3(0.5f, 0.7f, 1.0f), sky_t);
            };
            accum += color;
        };

        seeds.write(idx, state);
        Float inv = 1.0f / spp_this_pass.cast<float>();
        output.write(idx, make_float4(accum * inv, 1.0f));
    }));

    auto accumulate_shader = device.compile(Kernel2D([&](BufferFloat4 accum, BufferFloat4 current) noexcept {
        UInt2 p = dispatch_id().xy();
        UInt2 res = make_uint2(800u, 600u);
        UInt idx = p.y * res.x + p.x;
        Float4 a = accum.read(idx);
        Float4 c = current.read(idx);
        accum.write(idx, a + c);
    }));

    auto clear_shader = device.compile(Kernel2D([&](BufferFloat4 accum) noexcept {
        UInt2 p = dispatch_id().xy();
        UInt2 res = make_uint2(800u, 600u);
        accum.write(p.y * res.x + p.x, make_float4(0.0f));
    }));

    // ─── Coroutine (render-loop coordinator with 4 $suspend barriers) ───
    //
    // The coroutine body must not contain local variables (known coro-split
    // pipeline limitation). Suspends serve as scheduling barriers.
    Coroutine coro = [](BufferFloat4 signal, BufferUInt state,
                         UInt2 size, UInt pass) noexcept {
        (void)signal; (void)state; (void)size; (void)pass;
        $suspend("ray_gen_barrier");
        $suspend("intersection_barrier");
        $suspend("shading_barrier");
        $suspend("writeback_barrier");

        // Mark coroutine completion in output buffer
        signal.write(0u, make_float4(1.0f));
    };

    LUISA_INFO("Coroutine: {} subroutines, {} nodes",
               coro.subroutine_count(), coro.graph().node_count());

    StateMachineCoroScheduler<Buffer<float4>, Buffer<uint>, uint2, uint>
        scheduler{device, coro};
    LUISA_INFO("StateMachineCoroScheduler ready");

    // ─── Render loop ──────────────────────────────────────────────────
    Clock clock;

    stream << init_seeds_shader(seed_buffer).dispatch(resolution)
           << clear_shader(accum_buffer).dispatch(resolution)
           << synchronize();

    for (uint pass = 0u; pass < passes; ++pass) {
        // 1. Coroutine dispatch (runs through 4 suspends)
        scheduler(framebuffer, seed_buffer, resolution, pass)
            .dispatch(resolution.x, resolution.y)(stream);

        // 2. Path tracing dispatch
        stream << pathtrace_shader(framebuffer, seed_buffer,
                                    spp_per_dispatch)
                      .dispatch(resolution)
               << accumulate_shader(accum_buffer, framebuffer)
                      .dispatch(resolution)
               << synchronize();

        LUISA_INFO("Pass {}/{}: {:.1f} ms",
                   pass + 1u, passes, clock.toc());
    }

    // Read back and save
    luisa::vector<float4> host_image(resolution.x * resolution.y);
    stream << accum_buffer.copy_to(host_image.data()) << synchronize();

    float inv_total = 1.0f / static_cast<float>(passes);
    luisa::vector<uint8_t> png_data(resolution.x * resolution.y * 4);
    for (uint i = 0u; i < resolution.x * resolution.y; ++i) {
        float4 hdr = host_image[i] * inv_total;
        png_data[i * 4 + 0] = static_cast<uint8_t>(
            luisa::clamp(pow(hdr.x, 1.0f / 2.2f), 0.0f, 1.0f) * 255.0f);
        png_data[i * 4 + 1] = static_cast<uint8_t>(
            luisa::clamp(pow(hdr.y, 1.0f / 2.2f), 0.0f, 1.0f) * 255.0f);
        png_data[i * 4 + 2] = static_cast<uint8_t>(
            luisa::clamp(pow(hdr.z, 1.0f / 2.2f), 0.0f, 1.0f) * 255.0f);
        png_data[i * 4 + 3] = 255u;
    }

    stbi_write_png("coro_path_tracing.png",
                   resolution.x, resolution.y, 4,
                   png_data.data(), resolution.x * 4);

    LUISA_INFO("Rendered {} passes in {:.1f} ms total ({:.1f} ms/pass)",
               passes, clock.toc(), clock.toc() / passes);

    return 0;
}
