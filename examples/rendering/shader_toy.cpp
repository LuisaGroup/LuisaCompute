// ShaderToy-style Ray Marching Demo
// A real-time ray marching shader inspired by ShaderToy examples.
// Renders an animated 3D scene using signed distance functions (SDF).
//
// Features demonstrated:
// - Ray marching with SDFs
// - Domain repetition and transformations
// - Real-time animation
// - Interactive window display

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
#include <luisa/runtime/event.h>
#include <luisa/runtime/swapchain.h>
#include <luisa/dsl/syntax.h>

using namespace luisa;
using namespace luisa::compute;

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
        LUISA_INFO("Usage: {} <backend> [--offline] [-c <reference.png>]. <backend>: cuda, dx, metal, vk, hip, fallback", argv[0]);
        exit(1);
    }
    auto opts = luisa::ref::ExampleOptions::parse(argc, argv);
    if (!opts.valid()) {
        LUISA_WARNING("Invalid command line: {}", opts.error_message);
        return 1;
    }
#if !ENABLE_DISPLAY
    if (!opts.offline) {
        LUISA_ERROR("GUI support is disabled. Use --offline.");
    }
#endif
    Device device = context.create_device(argv[1]);

    // Color palette for gradient effects
    Callable palette = [](Float d) noexcept {
        return lerp(make_float3(0.2f, 0.7f, 0.9f), make_float3(1.0f, 0.0f, 1.0f), d);
    };

    // 2D rotation matrix
    Callable rotate = [](Float2 p, Float a) noexcept {
        Var c = cos(a);
        Var s = sin(a);
        return make_float2(dot(p, make_float2(c, s)), dot(p, make_float2(-s, c)));
    };

    // Signed distance function for the scene
    // Uses domain repetition and rotation for complex patterns
    Callable map = [&rotate](Float3 p, Float time) noexcept {
        for (uint i = 0u; i < 8u; i++) {
            Var t = time * 0.2f;
            p = make_float3(rotate(p.xz(), t), p.y).xzy();
            p = make_float3(rotate(p.xy(), t * 1.89f), p.z);
            p = make_float3(abs(p.x) - 0.5f, p.y, abs(p.z) - 0.5f);
        }
        return dot(copysign(1.0f, p), p) * 0.2f;
    };

    // Ray marching function
    // Marches along ray until hitting surface or max distance
    Callable rm = [&map, &palette](Float3 ro, Float3 rd, Float time) noexcept {
        Var t = 0.0f;
        Var col = make_float3(0.0f);
        Var d = 0.0f;
        for (UInt i : dynamic_range(64)) {
            Var p = ro + rd * t;
            d = map(p, time) * 0.5f;
            if_(d < 0.02f | d > 100.0f, [] { break_(); });
            col += palette(length(p) * 0.1f) / (400.0f * d);
            t += d;
        }
        return make_float4(col, 1.0f / (d * 100.0f));
    };

    // Clear kernel
    Kernel2D clear_kernel = [](ImageVar<float> image) noexcept {
        Var coord = dispatch_id().xy();
        image.write(coord, make_float4(make_float2(0.3f, 0.4f), 0.5f, 1.0f));
    };

    // Main rendering kernel
    Kernel2D main_kernel = [&rm, &rotate](ImageFloat image, Float time) noexcept {
        Var xy = dispatch_id().xy();
        Var resolution = make_float2(dispatch_size().xy());
        Var uv = (make_float2(xy) - resolution * 0.5f) / resolution.x;
        // Camera setup
        Var ro = make_float3(rotate(make_float2(0.0f, -50.0f), time), 0.0f).xzy();
        Var cf = normalize(-ro);
        Var cs = normalize(cross(cf, make_float3(0.0f, 1.0f, 0.0f)));
        Var cu = normalize(cross(cf, cs));
        Var uuv = ro + cf * 3.0f + uv.x * cs + uv.y * cu;
        Var rd = normalize(uuv - ro);
        // Ray march
        Var col = rm(ro, rd, time);
        Var color = col.xyz();
        Var alpha = col.w;
        Var old = image.read(xy).xyz();
        Var accum = lerp(color, old, alpha);
        image.write(xy, make_float4(accum, 1.0f));
    };

    auto clear = device.compile(clear_kernel);
    auto shader = device.compile(main_kernel);

    static constexpr uint width = 1024u;
    static constexpr uint height = 1024u;
    Stream stream = device.create_stream(opts.offline ? StreamTag::COMPUTE : StreamTag::GRAPHICS);
#if ENABLE_DISPLAY
    std::unique_ptr<Window> window;
    std::optional<Swapchain> swap_chain;
    if (!opts.offline) {
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
    Image<float> device_image = [&] {
#if ENABLE_DISPLAY
        if (!opts.offline) {
            return device.create_image<float>(swap_chain->backend_storage(), width, height);
        }
#endif
        return device.create_image<float>(PixelStorage::BYTE4, width, height);
    }();

    stream << clear(device_image).dispatch(width, height);

    // Animation loop
    Clock clock;
    if (opts.offline) {
        float time = 0.0f;
        stream << shader(device_image, time).dispatch(width, height);
        luisa::vector<uint8_t> host_image(width * height * 4u);
        stream << device_image.copy_to(luisa::span{host_image}) << synchronize();
        stbi_write_png("test_shader_toy.png", width, height, 4, host_image.data(), 0);
        if (opts.compare_path) {
            auto result = luisa::ref::compare_with_reference_file(
                reinterpret_cast<const uint8_t *>(host_image.data()),
                width, height, 4,
                *opts.compare_path);
            LUISA_INFO("Reference comparison: {} ({})", result.passed ? "PASSED" : "FAILED", result.message);
            if (!result.passed) { return 1; }
        }
    } else {
#if ENABLE_DISPLAY
        while (!window->should_close()) {
            window->poll_events();
            float time = static_cast<float>(clock.toc() * 1e-3);
            stream << shader(device_image, time).dispatch(width, height)
                   << swap_chain->present(device_image);
        }
#endif
    }
    stream << synchronize();
}
