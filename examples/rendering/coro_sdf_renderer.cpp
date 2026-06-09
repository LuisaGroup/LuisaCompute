// Coroutine SDF Renderer — procedural SDF sphere-tracing with coroutine scheduling
//
// Demonstrates StateMachineCoroScheduler-based rendering where each pixel's
// SDF ray-march runs in a coroutine strand, with $suspend markers creating
// continuation boundaries for the scheduler's state machine.
//
// SDF ray-marching computation is distributed across $suspend points:
//   strand 0: camera setup, ray initialization
//   $suspend("1"): per-iteration ray-march step (up to 100 iterations)
//   $suspend("2"): shading and progressive accumulation
//
// NOTE: If the pipeline crashes (xir2ast / coro_split), this is a pipeline
// bug — the code structure follows the correct coroutine pattern from
// LuisaCompute-coroutine/src/tests/coro/sdf_renderer.cpp.
//
// No swapchain/GUI required — renders offline to PNG.

#include <luisa/luisa-compute.h>
#include <luisa/dsl/sugar.h>
#include <luisa/coro/schedulers/state_machine.h>
#include <luisa/core/clock.h>
#include <luisa/core/logging.h>
#include <stb/stb_image_write.h>

#include <algorithm>
#include <string_view>

using namespace luisa;
using namespace luisa::compute;
using namespace luisa::compute::coro;

int main(int argc, char *argv[]) {
    uint user_spp = 0u;
    for (int i = 2; i < argc; i++) {
        if (!argv[i]) break;
        auto arg = std::string_view{argv[i]};
        if (arg == "--spp" && i + 1 < argc) {
            user_spp = static_cast<uint>(std::atoi(argv[++i]));
        }
    }

    static constexpr uint width = 800u;
    static constexpr uint height = 600u;
    static constexpr uint default_spp = 64u;
    uint total_spp = user_spp > 0u ? user_spp : default_spp;

    Context ctx{argv[0]};
    if (argc <= 1) {
        LUISA_INFO("Usage: {} <backend> [--spp N]", argv[0]);
        exit(1);
    }
    Device device = ctx.create_device(argv[1]);
    Stream stream = device.create_stream();

    auto accum = device.create_image<float>(PixelStorage::FLOAT4, width, height);

    // Coroutine: per-pixel SDF ray-march, all computation in strand 0.
    // $suspend markers create continuation boundaries for the scheduler
    // but continuations are empty (skip-guard only).
    auto coro = Coroutine<void(Image<float>, uint, uint, uint)>(
        [](Var<Image<float>> accum, Var<uint> w, Var<uint> h, Var<uint> frame) noexcept {
            UInt2 coord = dispatch_id().xy();
            $if (coord.x >= w | coord.y >= h) { return; };

            Float2 resolution = make_float2(w.cast<float>(), h.cast<float>());
            Float fov = 0.23f;
            Float aspect = resolution.x / resolution.y;
            Float2 uv = make_float2(coord.x.cast<float>(), coord.y.cast<float>());
            Float3 ro = make_float3(0.f, 0.32f, 3.7f);
            Float3 rd = normalize(make_float3(
                2.f * fov * uv / resolution.y - fov * make_float2(aspect, 1.f) - 1e-5f,
                -1.f));

            $suspend("setup");
            static constexpr float inf = 1e10f;
            Float t = def(0.f);
            $for (j, 100) {
                Float3 p = ro + t * rd;
                Float ground = p.y + 0.1f;
                Float sphere = distance(p, make_float3(0.f, 0.35f, 0.f)) - 0.36f;
                Float3 q0 = abs(p - make_float3(0.8f, 0.3f, 0.f)) - 0.3f;
                Float box = length(max(q0, 0.f)) + min(max(max(q0.x, q0.y), q0.z), 0.f);
                Float3 o = p - make_float3(-0.8f, 0.3f, 0.f);
                Float2 dcyl = make_float2(
                    length(make_float2(o.x, o.z)) - 0.3f, abs(o.y) - 0.3f);
                Float cylinder = min(max(dcyl.x, dcyl.y), 0.f) + length(max(dcyl, 0.f));
                Float geo = min(min(sphere, box), cylinder);
                Float g = max(geo, -(0.32f - (p.y * 0.6f + p.z * 0.8f)));
                Float s = min(ground, g);
                $if (s <= 1e-6f | t >= inf) { $break; };
                t += s;
                $suspend("step");
            };

            Float3 color;
            $if (t < inf) {
                Float3 hit = ro + t * rd;
                Float3 base = make_float3(0.7f, 0.65f, 0.55f);
                Int stripe = cast<int>(hit.x * 4.f + hit.z * 3.f) & 1;
                base = ite(stripe == 0, base, base * 0.8f);
                color = base * exp(-t * 0.008f) * 0.6f;
            } $else {
                Float sky_t = rd.y * 0.5f + 0.5f;
                color = lerp(make_float3(0.02f, 0.02f, 0.05f),
                             make_float3(0.3f, 0.5f, 0.8f), sky_t);
            };

            $suspend("accumulate");
            $if (frame == 0u) {
                accum.write(coord, make_float4(color, 1.f));
            } $else {
                Float4 prev = accum.read(coord);
                Float wgt = 1.f / (frame.cast<float>() + 1.f);
                Float3 blended = lerp(prev.xyz(), color, wgt);
                accum.write(coord, make_float4(blended, 1.f));
            };
            $suspend("done");
        });

    LUISA_INFO("Coroutine compiled: {} subroutines, {} graph nodes",
               coro.subroutine_count(), coro.graph().node_count());

    StateMachineCoroScheduler<Image<float>, uint, uint, uint> scheduler{device, coro};

    Clock clock;
    for (uint spp = 0u; spp < total_spp; spp++) {
        stream << scheduler(accum, width, height, spp).dispatch(width, height);
    }
    stream << synchronize();
    auto dt = clock.toc();
    LUISA_INFO("Rendered {} spp in {} ms ({:.1f} spp/s)", total_spp, dt,
               total_spp * 1e3 / dt);

    // Download and save PNG
    luisa::vector<float4> host(width * height);
    stream << accum.copy_to(host.data()) << synchronize();

    luisa::vector<uint8_t> pixels(width * height * 4u);
    for (size_t i = 0u; i < width * height; i++) {
        pixels[i * 4u + 0u] = static_cast<uint8_t>(
            std::clamp(host[i].x, 0.f, 1.f) * 255.99f);
        pixels[i * 4u + 1u] = static_cast<uint8_t>(
            std::clamp(host[i].y, 0.f, 1.f) * 255.99f);
        pixels[i * 4u + 2u] = static_cast<uint8_t>(
            std::clamp(host[i].z, 0.f, 1.f) * 255.99f);
        pixels[i * 4u + 3u] = 255u;
    }
    stbi_write_png("coro_sdf.png", static_cast<int>(width), static_cast<int>(height),
                   4, pixels.data(), 0);
    LUISA_INFO("Saved coro_sdf.png");
    return 0;
}
