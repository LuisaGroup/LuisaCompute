#include <filesystem>
#include <memory>
#include <optional>
#include <string_view>

#include "../common/reference_compare.h"

#include <luisa/core/clock.h>
#include <luisa/core/logging.h>
#include <luisa/runtime/context.h>
#include <luisa/runtime/device.h>
#include <luisa/runtime/stream.h>
#include <luisa/runtime/swapchain.h>
#include <luisa/dsl/sugar.h>

using namespace luisa;
using namespace luisa::compute;

struct SceneEvaluation {
    float distance;
    float d1;
    float d2;
    float d3;
    float lazors;
    float doodad;
    float3 p2;
};

LUISA_STRUCT(SceneEvaluation, distance, d1, d2, d3, lazors, doodad, p2) {};

#ifndef ENABLE_DISPLAY
#ifdef LUISA_ENABLE_GUI
#define ENABLE_DISPLAY 1
#endif
#endif

#if ENABLE_DISPLAY
#include <luisa/gui/window.h>
#endif

int main(int argc, char *argv[]) {

    Context context{argv[0]};
    if (argc <= 1) {
        LUISA_INFO("Usage: {} <backend> [--offline] [-c <reference.png>]. <backend>: cuda, dx, cpu, metal", argv[0]);
        exit(1);
    }
    auto opts = luisa::ref::ExampleOptions::parse(argc, argv);
    if (!opts.valid()) {
        LUISA_WARNING("Invalid command line: {}", opts.error_message);
        return 1;
    }
    auto force_offline = opts.offline;
    auto compare_path = opts.compare_path;
#if !ENABLE_DISPLAY
    if (!force_offline) {
        LUISA_ERROR("GUI support is disabled. Use --offline.");
    }
#endif
    Device device = context.create_device(argv[1]);

    Callable comp = [](Float3 p) noexcept {
        p = asin(sin(p) * .9f);
        return length(p) - 1.f;
    };

    Callable erot = [](Float3 p, Float3 ax, Float ro) noexcept {
        return lerp(dot(p, ax) * ax, p, cos(ro)) + sin(ro) * cross(ax, p);
    };

    Callable smin = [](Float a, Float b, Float k) noexcept {
        auto h = max(0.f, k - abs(b - a)) / k;
        return min(a, b) + h * h * h * k / 6.f;
    };

    Callable wrot = [](Float4 p) noexcept {
        return make_float4(dot(p, make_float4(1.f)), p.yzw() + p.zwy() - p.wyz() - p.xxx()) * .5f;
    };

    static constexpr auto bpm = 125.f;
    Callable evaluate_scene = [&comp, &erot, &smin, &wrot](Float3 p, Float t, Float time) noexcept {
        auto evaluated_p2 = erot(p, make_float3(0.f, 1.f, 0.f), t);
        evaluated_p2 = erot(evaluated_p2, make_float3(0.f, 0.f, 1.f), t / 3.f);
        evaluated_p2 = erot(evaluated_p2, make_float3(1.f, 0.f, 0.f), t / 5.f);
        auto bpt = time / 60.f * bpm;
        auto p4 = make_float4(evaluated_p2, 0.f);
        p4 = lerp(p4, wrot(p4), smoothstep(-.5f, .5f, sin(bpt / 4.f)));
        p4 = abs(p4);
        p4 = lerp(p4, wrot(p4), smoothstep(-.5f, .5f, sin(bpt)));
        auto fctr = smoothstep(-.5f, .5f, sin(bpt / 2.f));
        auto fctr2 = smoothstep(.9f, 1.f, sin(bpt / 16.f));
        auto evaluated_doodad = length(max(abs(p4) - lerp(0.05f, 0.07f, fctr), 0.f) + lerp(-0.1f, .2f, fctr)) - lerp(.15f, .55f, fctr * fctr) + fctr2;
        auto repeated_p = p + make_float3(asin(sin(t / 80.f) * .99f) * 80.f, 0.f, 0.f);
        auto evaluated_lazors = length(asin(sin(erot(repeated_p, make_float3(1.f, 0.f, 0.f), t * .2f).yz() * .5f + 1.f)) / .5f) - .1f;
        auto evaluated_d1 = comp(repeated_p);
        auto evaluated_d2 = comp(erot(repeated_p + 5.f, normalize(make_float3(1.f, 3.f, 4.f)), .4f));
        auto evaluated_d3 = comp(erot(repeated_p + 10.f, normalize(make_float3(3.f, 2.f, 1.f)), 1.f));
        auto distance = min(evaluated_doodad, min(evaluated_lazors, .3f - smin(smin(evaluated_d1, evaluated_d2, .05f), evaluated_d3, .05f)));
        return def<SceneEvaluation>(distance, evaluated_d1, evaluated_d2, evaluated_d3, evaluated_lazors, evaluated_doodad, evaluated_p2);
    };

    Callable scene_distance = [&evaluate_scene](Float3 p, Float t, Float time) noexcept {
        Var<SceneEvaluation> evaluation = evaluate_scene(p, t, time);
        return evaluation.distance;
    };

    Callable scene_normal = [&scene_distance](Float3 p, Float t, Float time) noexcept {
        auto precis = ite(length(p) < 1.f, .005f, .01f);
        auto k = make_float3x3(p, p, p) - make_float3x3(precis, 0.f, 0.f, 0.f, precis, 0.f, 0.f, 0.f, precis);
        // Keep the four distance-only probes explicitly sequenced in the recorded AST.
        Float center_distance = scene_distance(p, t, time);
        Float x_distance = scene_distance(k[0], t, time);
        Float y_distance = scene_distance(k[1], t, time);
        Float z_distance = scene_distance(k[2], t, time);
        return normalize(center_distance - make_float3(x_distance, y_distance, z_distance));
    };

    Kernel2D render_kernel = [&](ImageFloat image, Float time) noexcept {
        Float d1, d2, d3;
        Float t;
        Float lazors, doodad;
        Float3 p2;

        auto fragCoord = make_float2(dispatch_id().xy());
        auto iResolution = make_float2(dispatch_size().xy());
        auto uv = (fragCoord - .5f * iResolution) / iResolution.y;

        auto bpt = time / 60.f * bpm;
        auto bp = lerp(pow(sin(fract(bpt) * constants::pi / 2.f), 20.f) + floor(bpt), bpt, .4f);
        t = bp;
        auto cam = normalize(make_float3(.8f + sin(bp * 3.14f / 4.f) * .3f, uv));
        auto init = make_float3(-1.5f + sin(bp * 3.14f) * .2f, 0.f, 0.f) + cam * .2f;
        init = erot(init, make_float3(0.f, 1.f, 0.f), sin(bp * .2f) * .4f);
        init = erot(init, make_float3(0.f, 0.f, 1.f), cos(bp * .2f) * .4f);
        cam = erot(cam, make_float3(0.f, 1.f, 0.f), sin(bp * .2f) * .4f);
        cam = erot(cam, make_float3(0.f, 0.f, 1.f), cos(bp * .2f) * .4f);
        auto p = init;
        auto atten = def(1.f);
        auto tlen = def(0.f);
        auto glo = def(0.f);
        auto fog = def(0.f);
        auto dlglo = def(0.f);
        auto trg = def(false);
        auto dist = def(0.f);
        $for (i, 80) {
            // Only the primary ray sample commits auxiliaries used by glow and material shading.
            Var<SceneEvaluation> primary_evaluation = evaluate_scene(p, t, time);
            dist = primary_evaluation.distance;
            d1 = primary_evaluation.d1;
            d2 = primary_evaluation.d2;
            d3 = primary_evaluation.d3;
            lazors = primary_evaluation.lazors;
            doodad = primary_evaluation.doodad;
            p2 = primary_evaluation.p2;
            auto hit = dist * dist < 1e-6f;
            glo += .2f / (1.f + lazors * lazors * 20.f) * atten;
            dlglo += .2f / (1.f + doodad * doodad * 20.f) * atten;
            $if (hit & ((sin(d3 * 45.f) < -0.4f & (dist != doodad)) | (dist == doodad & sin(pow(length(p2 * p2 * p2), .3f) * 120.f) > .4f)) & dist != lazors) {
                trg = trg | dist == doodad;
                hit = false;
                auto n = scene_normal(p, t, time);
                atten *= 1.f - abs(dot(cam, n)) * .98f;
                cam = reflect(cam, n);
                dist = .1f;
            };
            p += cam * dist;
            tlen += dist;
            fog += dist * atten / 30.f;
            $if (hit) { $break; };
        };
        fog = smoothstep(0.f, 1.f, fog);
        auto lz = lazors == dist;
        auto dl = doodad == dist;
        auto fogcol = lerp(make_float3(.5f, .8f, 1.2f), make_float3(.4f, .6f, .9f), length(uv));
        auto n = scene_normal(p, t, time);
        auto r = reflect(cam, n);
        auto ss = smoothstep(-.3f, .3f, scene_distance(p + make_float3(.3f), t, time)) + .5f;
        auto fact = length(sin(r * (ite(dl, 4.f, 3.f))) * .5f + .5f) / sqrt(3.f) * .7f + .3f;
        auto matcol = lerp(make_float3(.9f, .4f, .3f), make_float3(.3f, .4f, .8f), smoothstep(-1.f, 1.f, sin(d1 * 5.f + time * 2.f)));
        matcol = lerp(matcol, make_float3(.5f, .4f, 1.f), smoothstep(0.f, 1.f, sin(d2 * 5.f + time * 2.f)));
        matcol = ite(dl, lerp(1.f, matcol, .1f) * .2f + .1f, matcol);
        auto col = matcol * fact * ss + pow(fact, 10.f);
        col = ite(lz, 4.f, col);
        auto fragColor = col * atten + glo * glo + fogcol * glo;
        fragColor = lerp(fragColor, fogcol, fog);
        fragColor = ite(dl, fragColor, abs(erot(fragColor, normalize(sin(p * 2.f)), .2f * (1.f - fog))));
        fragColor = ite(trg | dl, fragColor, fragColor + dlglo * dlglo * .1f * make_float3(.4f, .6f, .9f));
        fragColor = sqrt(fragColor);
        auto color = smoothstep(0.f, 1.2f, fragColor);
        image.write(dispatch_id().xy(), make_float4(pow(color, 2.2f), 1.f));
    };

    Kernel2D clear_kernel = [](ImageVar<float> image) noexcept {
        Var coord = dispatch_id().xy();
        Var rg = make_float2(coord) / make_float2(dispatch_size().xy());
        image.write(coord, make_float4(make_float2(0.3f, 0.4f), 0.5f, 1.0f));
    };
    auto clear = device.compile(clear_kernel);
    auto shader = device.compile(render_kernel);

    static constexpr auto width = 1280u;
    static constexpr auto height = 720u;
    Stream stream = device.create_stream(force_offline ? StreamTag::COMPUTE : StreamTag::GRAPHICS);
#if ENABLE_DISPLAY
    std::unique_ptr<Window> window;
    std::optional<Swapchain> swap_chain;
    if (!force_offline) {
        window = std::make_unique<Window>("Display", make_uint2(width, height));
        swap_chain.emplace(device.create_swapchain(
            stream,
            SwapchainOption{
                .display = window->native_display(),
                .window = window->native_handle(),
                .size = window->size(),
                .wants_hdr = false,
                .wants_vsync = false,
                .back_buffer_count = 2,
            }));
    }
#endif
    auto device_image = [&] {
#if ENABLE_DISPLAY
        if (!force_offline) {
            return device.create_image<float>(swap_chain->backend_storage(), width, height);
        }
#endif
        return device.create_image<float>(PixelStorage::BYTE4, width, height);
    }();
    stream << clear(device_image).dispatch(width, height);

    Clock clock;
    if (force_offline) {
        auto time = 0.0f;
        stream << shader(device_image, time).dispatch(width, height);
        luisa::vector<uint8_t> host_image(width * height * 4u);
        stream << device_image.copy_to(luisa::span{host_image}) << synchronize();
        stbi_write_png("test_shader_visuals_present.png", width, height, 4, host_image.data(), 0);
        if (compare_path) {
            auto result = luisa::ref::compare_with_reference_file(
                reinterpret_cast<const uint8_t *>(host_image.data()),
                width, height, 4,
                *compare_path);
            LUISA_INFO("Reference comparison: {} ({})", result.passed ? "PASSED" : "FAILED", result.message);
            if (!result.passed) { return 1; }
        }
    } else {
#if ENABLE_DISPLAY
        while (!window->should_close()) {
            auto time = static_cast<float>(clock.toc() * 1e-3);
            stream << shader(device_image, time).dispatch(width, height)
                   << swap_chain->present(device_image);
            window->poll_events();
        }
#endif
    }
    stream << synchronize();
}
