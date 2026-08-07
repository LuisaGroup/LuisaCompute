/*
 * Tutorial 06: Multi-Pass Image Processing
 *
 * This tutorial teaches how to build a classic image-processing pipeline in LuisaCompute.
 * We first generate a procedural input image, then blur it with a separable Gaussian,
 * compute Sobel edges, and finally combine everything into a stylized composite with
 * sharpening, colored edge overlays, and a vignette.
 *
 * What you will learn:
 *   1. Why multi-pass pipelines use intermediate images instead of one giant kernel.
 *   2. How separable convolution reduces work while producing the same visual result.
 *   3. How to support both offline exports and interactive presentation from the same pipeline.
 */

#include <cstddef>
#include <cstdint>
#include <cmath>
#include <string_view>

#include <luisa/luisa-compute.h>
#include <luisa/core/clock.h>
#include <luisa/dsl/sugar.h>
#include <luisa/gui/window.h>
#include <luisa/runtime/swapchain.h>
#include <stb/stb_image_write.h>

using namespace luisa;
using namespace luisa::compute;

namespace {

[[nodiscard]] auto has_flag(int argc, char *argv[], std::string_view flag) noexcept {
    for (auto i = 2; i < argc; i++) {
        if (flag == argv[i]) {
            return true;
        }
    }
    return false;
}

void save_png(Stream &stream, Image<float> &image, uint2 resolution, const char *filename) {
    luisa::vector<std::byte> host_pixels(image.view().size_bytes());
    stream << image.copy_to(luisa::span{host_pixels}) << synchronize();
    auto success = stbi_write_png(filename, static_cast<int>(resolution.x), static_cast<int>(resolution.y), 4, host_pixels.data(), 0);
    LUISA_INFO("Saved {} [{}]", filename, success != 0 ? "ok" : "failed");
}

}// namespace

int main(int argc, char *argv[]) {

    log_level_verbose();

    auto offline = has_flag(argc, argv, "--offline") || (argc > 1 && std::string_view{argv[1]} == "--offline");
    luisa::string backend;
    for (int i = 1; i < argc; i++) {
        if (std::string_view{argv[i]} != "--offline" && backend.empty()) {
            backend = argv[i];
        }
    }

    // Step 1: Create the runtime objects that own all GPU resources and compiled shaders.
    Context context{argv[0]};
    if (backend.empty()) {
        auto const &backends = context.installed_backends();
        if (backends.empty()) {
            LUISA_ERROR("No backends installed.");
            return 1;
        }
        static constexpr luisa::string_view preferred_backends[] = {
            "cuda", "dx", "metal", "vk", "hip", "fallback"};
        for (auto preferred : preferred_backends) {
            for (auto const &candidate : backends) {
                if (candidate == preferred && !context.backend_device_names(candidate).empty()) {
                    backend = candidate;
                    break;
                }
            }
            if (!backend.empty()) { break; }
        }
        if (backend.empty()) {
            for (auto const &candidate : backends) {
                if (!context.backend_device_names(candidate).empty()) {
                    backend = candidate;
                    break;
                }
            }
        }
        if (backend.empty()) {
            LUISA_ERROR("No usable backends installed.");
            return 1;
        }
        LUISA_INFO("No backend specified, auto-selected: {}", backend);
    }
    Device device = context.create_device(backend);
    Stream stream = device.create_stream(StreamTag::GRAPHICS);

    static constexpr auto resolution = make_uint2(1024u, 1024u);

    LUISA_INFO("Tutorial 06 - Multi-Pass Image Processing");
    LUISA_INFO("Backend: {}", backend);
    LUISA_INFO("Mode: {}", offline ? "offline" : "interactive");

    // Step 2: Define the procedural source generator.
    // A synthetic pattern is useful in tutorials because it is reproducible and avoids
    // adding external assets. We mix checkerboard structure with circular motifs so the
    // blur and edge passes both have something interesting to process.
    Kernel2D pattern_kernel = [](ImageFloat output) noexcept {
        set_block_size(16u, 16u, 1u);
        auto pixel = dispatch_id().xy();
        auto uv = (make_float2(pixel) + 0.5f) / make_float2(dispatch_size().xy());

        auto checker_index = cast<int>(floor(uv.x * 16.0f) + floor(uv.y * 16.0f));
        auto checker = ite((checker_index & 1) == 0, 0.18f, 0.82f);

        auto center = uv - 0.5f;
        auto radius = length(center);
        auto rings = 0.5f + 0.5f * cos(radius * 80.0f);

        auto circle_a = exp(-pow(length(uv - make_float2(0.32f, 0.38f)) * 8.0f, 2.0f));
        auto circle_b = exp(-pow(length(uv - make_float2(0.70f, 0.58f)) * 10.0f, 2.0f));
        auto stripe = 0.5f + 0.5f * sin((uv.x * 2.0f - uv.y) * 30.0f);

        auto base = lerp(make_float3(0.10f, 0.12f, 0.18f), make_float3(0.90f, 0.75f, 0.30f), checker);
        base += rings * make_float3(0.08f, 0.10f, 0.18f);
        base += circle_a * make_float3(0.15f, 0.30f, 0.75f);
        base += circle_b * make_float3(0.75f, 0.20f, 0.35f);
        base *= 0.85f + 0.15f * stripe;
        output.write(pixel, make_float4(clamp(base, 0.0f, 1.0f), 1.0f));
    };
    auto pattern_shader = device.compile(pattern_kernel);

    // Step 3: Define a separable 9-tap Gaussian blur.
    // Splitting the blur into horizontal and vertical passes reduces the number of
    // texture reads from 81 to 18 per pixel, which is why separable filters are so common.
    Kernel2D blur_horizontal_kernel = [](ImageFloat input, ImageFloat output) noexcept {
        set_block_size(16u, 16u, 1u);
        auto pixel = dispatch_id().xy();
        auto size = dispatch_size().xy();

        Constant weights = {0.02763f, 0.06628f, 0.12383f, 0.18017f, 0.20418f, 0.18017f, 0.12383f, 0.06628f, 0.02763f};
        Constant offsets = {-4, -3, -2, -1, 0, 1, 2, 3, 4};

        Float3 sum = make_float3(0.0f);
        $for (i, 9u) {
            auto sx = clamp(cast<int>(pixel.x) + offsets.read(i), 0, cast<int>(size.x) - 1);
            auto sample_pixel = make_uint2(cast<uint>(sx), pixel.y);
            sum += input.read(sample_pixel).xyz() * weights.read(i);
        };
        output.write(pixel, make_float4(sum, 1.0f));
    };
    auto blur_horizontal_shader = device.compile(blur_horizontal_kernel);

    Kernel2D blur_vertical_kernel = [](ImageFloat input, ImageFloat output) noexcept {
        set_block_size(16u, 16u, 1u);
        auto pixel = dispatch_id().xy();
        auto size = dispatch_size().xy();

        Constant weights = {0.02763f, 0.06628f, 0.12383f, 0.18017f, 0.20418f, 0.18017f, 0.12383f, 0.06628f, 0.02763f};
        Constant offsets = {-4, -3, -2, -1, 0, 1, 2, 3, 4};

        Float3 sum = make_float3(0.0f);
        $for (i, 9u) {
            auto sy = clamp(cast<int>(pixel.y) + offsets.read(i), 0, cast<int>(size.y) - 1);
            auto sample_pixel = make_uint2(pixel.x, cast<uint>(sy));
            sum += input.read(sample_pixel).xyz() * weights.read(i);
        };
        output.write(pixel, make_float4(sum, 1.0f));
    };
    auto blur_vertical_shader = device.compile(blur_vertical_kernel);

    // Step 4: Define Sobel edge detection.
    // Edges describe where intensity changes quickly, which makes them ideal for both
    // analysis and stylized compositing.
    Kernel2D sobel_kernel = [](ImageFloat input, ImageFloat output) noexcept {
        set_block_size(16u, 16u, 1u);
        auto pixel = dispatch_id().xy();
        auto size = dispatch_size().xy();

        auto luminance = [&](Int2 offset) noexcept {
            auto p = make_int2(pixel) + offset;
            p.x = clamp(p.x, 0, cast<int>(size.x) - 1);
            p.y = clamp(p.y, 0, cast<int>(size.y) - 1);
            auto rgb = input.read(make_uint2(cast<uint>(p.x), cast<uint>(p.y))).xyz();
            return dot(rgb, make_float3(0.299f, 0.587f, 0.114f));
        };

        auto gx = -luminance(make_int2(-1, -1)) + luminance(make_int2(1, -1)) -
                  2.0f * luminance(make_int2(-1, 0)) + 2.0f * luminance(make_int2(1, 0)) -
                  luminance(make_int2(-1, 1)) + luminance(make_int2(1, 1));
        auto gy = -luminance(make_int2(-1, -1)) - 2.0f * luminance(make_int2(0, -1)) - luminance(make_int2(1, -1)) +
                  luminance(make_int2(-1, 1)) + 2.0f * luminance(make_int2(0, 1)) + luminance(make_int2(1, 1));

        auto magnitude = clamp(sqrt(gx * gx + gy * gy), 0.0f, 1.0f);
        auto angle = atan2(gy, gx) * (0.5f / pi) + 0.5f;
        output.write(pixel, make_float4(magnitude, angle, 0.0f, 1.0f));
    };
    auto sobel_shader = device.compile(sobel_kernel);

    // Step 5: Define the final composite pass.
    // We use unsharp masking for local contrast, then layer cyan edges and a vignette
    // to make the result look intentionally stylized instead of merely analytical.
    Kernel2D composite_kernel = [](ImageFloat source, ImageFloat blurred, ImageFloat edges, ImageFloat output, Float edge_intensity) noexcept {
        set_block_size(16u, 16u, 1u);
        auto pixel = dispatch_id().xy();
        auto uv = (make_float2(pixel) + 0.5f) / make_float2(dispatch_size().xy());

        auto src = source.read(pixel).xyz();
        auto blur = blurred.read(pixel).xyz();
        auto edge = edges.read(pixel).x;

        auto sharpened = clamp(src + (src - blur) * 1.35f, 0.0f, 1.0f);
        auto edge_color = make_float3(0.00f, 0.90f, 1.00f) * edge * edge_intensity;
        auto composite = sharpened;

        $if (edge > 0.18f) {
            composite = lerp(composite, edge_color + composite * 0.35f, clamp(edge * 0.75f, 0.0f, 1.0f));
        }
        $else {
            composite += edge_color * 0.15f;
        };

        auto radius = length(uv - 0.5f);
        auto vignette = smoothstep(0.82f, 0.20f, radius);
        output.write(pixel, make_float4(clamp(composite * vignette, 0.0f, 1.0f), 1.0f));
    };
    auto composite_shader = device.compile(composite_kernel);

    // Step 6: Allocate the intermediate images.
    // Multi-pass pipelines become much easier to debug when every stage has its own target.
    auto source = device.create_image<float>(PixelStorage::FLOAT4, resolution);
    auto blur_temp = device.create_image<float>(PixelStorage::FLOAT4, resolution);
    auto blurred = device.create_image<float>(PixelStorage::FLOAT4, resolution);
    auto edges = device.create_image<float>(PixelStorage::FLOAT4, resolution);

    // Step 7: Run the static passes once.
    // The pattern, blur, and Sobel stages do not depend on time, so interactive mode only
    // needs to animate the cheap final composite.
    stream << pattern_shader(source).dispatch(resolution)
           << blur_horizontal_shader(source, blur_temp).dispatch(resolution)
           << blur_vertical_shader(blur_temp, blurred).dispatch(resolution)
           << sobel_shader(source, edges).dispatch(resolution)
           << synchronize();

    if (offline) {
        // Step 8A: Offline mode executes the final pass once and writes a PNG.
        auto output = device.create_image<float>(PixelStorage::BYTE4, resolution);
        stream << composite_shader(source, blurred, edges, output, 1.0f).dispatch(resolution);
        save_png(stream, output, resolution, "tutorial_06_image_processing.png");
        return 0;
    }

    // Step 8B: Interactive mode presents the final composite and pulses the edge overlay.
    Window window{"Tutorial 06 - Image Processing", resolution};
    Swapchain swapchain = device.create_swapchain(
        stream,
        SwapchainOption{
            .display = window.native_display(),
            .window = window.native_handle(),
            .size = resolution,
            .wants_vsync = true,
            .back_buffer_count = 3u,
        });
    auto display = device.create_image<float>(swapchain.backend_storage(), resolution);
    window.set_key_callback([&](Key key, KeyModifiers, Action action) noexcept {
        if (action == ACTION_PRESSED && key == KEY_ESCAPE) {
            window.set_should_close();
        }
    });

    Clock clock;
    while (!window.should_close()) {
        window.poll_events();

        auto time = static_cast<float>(clock.toc() * 1e-3);
        auto edge_intensity = 0.35f + 0.65f * (0.5f + 0.5f * std::sin(time));
        stream << composite_shader(source, blurred, edges, display, edge_intensity).dispatch(resolution)
               << swapchain.present(display);
    }

    stream << synchronize();
    return 0;
}
