//
// Created by Mike Smith on 2021/6/25.
//

#include <core/clock.h>
#include <core/logging.h>
#include <runtime/context.h>
#include <runtime/device.h>
#include <runtime/stream.h>
#include <dsl/sugar.h>
#include <gui/window.h>

using namespace luisa;
using namespace luisa::compute;

int main(int argc, char *argv[]) {

    Context context{argv[0]};
    if (argc <= 1) {
        LUISA_INFO("Usage: {} <backend>. <backend>: cuda, dx, ispc, metal", argv[0]);
        exit(1);
    }
    auto device = context.create_device(argv[1]);

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

    Callable reflect = [](Float3 I, Float3 N) noexcept {
        return I - 2.f * dot(N, I) * N;
    };

    Kernel2D render_kernel = [&](ImageFloat image, Float time) noexcept {
        Float d1, d2, d3;
        Float t;
        Float lazors, doodad;
        Float3 p2;
        static constexpr auto bpm = 125.f;
        auto scene = [&](Float3 p) noexcept {
            p2 = erot(p, make_float3(0.f, 1.f, 0.f), t);
            p2 = erot(p2, make_float3(0.f, 0.f, 1.f), t / 3.f);
            p2 = erot(p2, make_float3(1.f, 0.f, 0.f), t / 5.f);
            auto bpt = time / 60.f * bpm;
            auto p4 = make_float4(p2, 0.f);
            p4 = lerp(p4, wrot(p4), smoothstep(-.5f, .5f, sin(bpt / 4.f)));
            p4 = abs(p4);
            p4 = lerp(p4, wrot(p4), smoothstep(-.5f, .5f, sin(bpt)));
            auto fctr = smoothstep(-.5f, .5f, sin(bpt / 2.f));
            auto fctr2 = smoothstep(.9f, 1.f, sin(bpt / 16.f));
            doodad = length(max(abs(p4) - lerp(0.05f, 0.07f, fctr), 0.f) + lerp(-0.1f, .2f, fctr)) - lerp(.15f, .55f, fctr * fctr) + fctr2;
            p.x += asin(sin(t / 80.f) * .99f) * 80.f;
            lazors = length(asin(sin(erot(p, make_float3(1.f, 0.f, 0.f), t * .2f).yz() * .5f + 1.f)) / .5f) - .1f;
            d1 = comp(p);
            d2 = comp(erot(p + 5.f, normalize(make_float3(1.f, 3.f, 4.f)), .4f));
            d3 = comp(erot(p + 10.f, normalize(make_float3(3.f, 2.f, 1.f)), 1.f));
            return min(doodad, min(lazors, .3f - smin(smin(d1, d2, .05f), d3, .05f)));
        };

        auto norm = [&](Float3 p) noexcept {
            auto precis = ite(length(p) < 1.f, .005f, .01f);
            auto k = make_float3x3(p, p, p) - make_float3x3(precis);
            return normalize(scene(p) - make_float3(scene(k[0]), scene(k[1]), scene(k[2])));
        };

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
        $for(i, 80) {
            dist = scene(p);
            auto hit = dist * dist < 1e-6f;
            glo += .2f / (1.f + lazors * lazors * 20.f) * atten;
            dlglo += .2f / (1.f + doodad * doodad * 20.f) * atten;
            $if(hit & ((sin(d3 * 45.f) < -0.4f & (dist != doodad)) | (dist == doodad & sin(pow(length(p2 * p2 * p2), .3f) * 120.f) > .4f)) & dist != lazors) {
                trg = trg | dist == doodad;
                hit = false;
                auto n = norm(p);
                atten *= 1.f - abs(dot(cam, n)) * .98f;
                cam = reflect(cam, n);
                dist = .1f;
            };
            p += cam * dist;
            tlen += dist;
            fog += dist * atten / 30.f;
            $if(hit) { $break; };
        };
        fog = smoothstep(0.f, 1.f, fog);
        auto lz = lazors == dist;
        auto dl = doodad == dist;
        auto fogcol = lerp(make_float3(.5f, .8f, 1.2f), make_float3(.4f, .6f, .9f), length(uv));
        auto n = norm(p);
        auto r = reflect(cam, n);
        auto ss = smoothstep(-.3f, .3f, scene(p + make_float3(.3f))) + .5f;
        auto fact = length(sin(r * (ite(dl != 0.f, 4.f, 3.f))) * .5f + .5f) / sqrt(3.f) * .7f + .3f;
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
    auto device_image = device.create_image<float>(PixelStorage::BYTE4, width, height);
    auto stream = device.create_stream(StreamTag::GRAPHICS);
    stream << clear(device_image).dispatch(width, height);
    Window window{"Display", make_uint2(width, height), false};
    auto swap_chain{device.create_swapchain(
        window.window_native_handle(),
        stream,
        window.size(),
        true, false, 2)};

    Clock clock;
    while (!window.should_close()) {
        auto time = static_cast<float>(clock.toc() * 1e-3);
        stream << shader(device_image, time).dispatch(width, height)
               << swap_chain.present(device_image);
        window.pool_event();
    }
    stream << synchronize();
}
