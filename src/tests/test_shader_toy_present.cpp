//
// Created by Mike Smith on 2021/6/25.
//

#include <iostream>

#define GLFW_INCLUDE_NONE
#define GLFW_EXPOSE_NATIVE_COCOA
#import <GLFW/glfw3.h>
#import <GLFW/glfw3native.h>

#include <runtime/context.h>
#include <runtime/device.h>
#include <runtime/stream.h>
#include <runtime/event.h>
#include <dsl/syntax.h>
#include <gui/framerate.h>

using namespace luisa;
using namespace luisa::compute;

int main(int argc, char *argv[]) {

    Context context{argv[0]};
    auto device = context.create_device("metal");

    Callable palette = [](Float d) noexcept {
        return lerp(make_float3(0.2f, 0.7f, 0.9f), make_float3(1.0f, 0.0f, 1.0f), d);
    };

    Callable rotate = [](Float2 p, Float a) noexcept {
        Var c = cos(a);
        Var s = sin(a);
        return make_float2(dot(p, make_float2(c, s)), dot(p, make_float2(-s, c)));
    };

    Callable map = [&rotate](Float3 p, Float time) noexcept {
        for (auto i = 0u; i < 8u; i++) {
            Var t = time * 0.2f;
            p = make_float3(rotate(p.xz(), t), p.y).xzy();
            p = make_float3(rotate(p.xy(), t * 1.89f), p.z);
            p = make_float3(abs(p.x) - 0.5f, p.y, abs(p.z) - 0.5f);
        }
        return dot(copysign(1.0f, p), p) * 0.2f;
    };

    Callable rm = [&map, &palette](Float3 ro, Float3 rd, Float time) noexcept {
        Var t = 0.0f;
        Var col = make_float3(0.0f);
        Var d = 0.0f;
        for (auto i : range(64)) {
            Var p = ro + rd * t;
            d = map(p, time) * 0.5f;
            if_(d<0.02f | d> 100.0f, [] { break_(); });
            col += palette(length(p) * 0.1f) / (400.0f * d);
            t += d;
        }
        return make_float4(col, 1.0f / (d * 100.0f));
    };

    Kernel2D clear_kernel = [](ImageVar<float> image) noexcept {
        Var coord = dispatch_id().xy();
        Var rg = make_float2(coord) / make_float2(dispatch_size().xy());
        image.write(coord, make_float4(make_float2(0.3f, 0.4f), 0.5f, 1.0f));
    };

    Kernel2D main_kernel = [&rm, &rotate](ImageFloat image, Float time) noexcept {
        Var xy = dispatch_id().xy();
        Var resolution = make_float2(dispatch_size().xy());
        Var uv = (make_float2(xy) - resolution * 0.5f) / resolution.x;
        Var ro = make_float3(rotate(make_float2(0.0f, -50.0f), time), 0.0f).xzy();
        Var cf = normalize(-ro);
        Var cs = normalize(cross(cf, make_float3(0.0f, 1.0f, 0.0f)));
        Var cu = normalize(cross(cf, cs));
        Var uuv = ro + cf * 3.0f + uv.x * cs + uv.y * cu;
        Var rd = normalize(uuv - ro);
        Var col = rm(ro, rd, time);
        image.write(xy, make_float4(pow(col.xyz(), 2.2f) * 10.f, 1.0f));
    };

    Kernel2D<Image<float>, float> k = main_kernel;
    main_kernel = k;
    k = main_kernel;

    auto clear = device.compile(clear_kernel);
    auto shader = device.compile(k);

    static constexpr auto width = 1024u;
    static constexpr auto height = 1024u;
    auto device_image = device.create_image<float>(PixelStorage::HALF4, width, height);
    std::vector<std::array<uint8_t, 4u>> host_image(width * height);

    auto stream = device.create_stream();
    stream << clear(device_image).dispatch(width, height);

    glfwInit();
    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    auto window = glfwCreateWindow(width, height, "ShaderToy Example", nullptr, nullptr);
    auto window_handle = reinterpret_cast<uint64_t>(glfwGetCocoaWindow(window));
    auto swap_chain = device.create_swapchain(window_handle, stream, width, height);

    Clock clock;
    Framerate framerate{32};
    while (!glfwWindowShouldClose(window)) {
        framerate.record();
        LUISA_INFO("FPS: {}", framerate.report());
        auto time = static_cast<float>(clock.toc() * 1e-3);
        stream << shader(device_image, time).dispatch(width, height)
               << swap_chain.present(device_image);
        glfwPollEvents();
    }
    stream << synchronize();
}
