// Image Processing Pipeline
// Demonstrates a multi-stage image processing pipeline including:
// - Gaussian blur for noise reduction
// - Sobel edge detection
// - Tone mapping and color grading
//
// Features demonstrated:
// - Multi-pass rendering with intermediate buffers
// - Image convolution kernels
// - Separable filter optimization (horizontal + vertical passes)
// - Buffer-to-image and image-to-buffer transfers

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
#include <luisa/runtime/buffer.h>
#include <luisa/runtime/swapchain.h>
#include <luisa/dsl/sugar.h>

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
        LUISA_INFO("Usage: {} <backend> [--offline] [--iterations N] [-c <reference.png>]. <backend>: cuda, dx, metal, vk, hip, fallback", argv[0]);
        exit(1);
    }
    auto opts = luisa::ref::ExampleOptions::parse(argc, argv);
    if (!opts.valid()) {
        LUISA_WARNING("Invalid command line: {}", opts.error_message);
        return 1;
    }
    auto force_offline = opts.offline;
    auto offline_iterations = opts.iterations;
    auto compare_path = opts.compare_path;
#if !ENABLE_DISPLAY
    if (!force_offline) {
        LUISA_ERROR("GUI support is disabled. Use --offline.");
    }
#endif
    Device device = context.create_device(argv[1]);
    LUISA_INFO("Image Processing Pipeline Demo");
    LUISA_INFO("Shows: Gaussian Blur -> Sobel Edge Detection -> Compositing");

    // Image dimensions
    static constexpr uint width = 1024u;
    static constexpr uint height = 1024u;

    Stream stream = device.create_stream(force_offline ? StreamTag::COMPUTE : StreamTag::GRAPHICS);

    // Generate a test pattern (procedural "Lenna-like" image with geometric shapes)
    // In a real application, you would load an actual image here
    Kernel2D generate_pattern = [](ImageFloat image) noexcept {
        set_block_size(16, 16, 1);
        Var uv = make_float2(dispatch_id().xy()) / make_float2(dispatch_size().xy());

        // Checkerboard pattern
        Var checker = sin(uv.x * 20.0f) * sin(uv.y * 20.0f);

        // Concentric circles
        Var center = uv - make_float2(0.5f);
        Var radius = length(center);
        Var circles = sin(radius * 50.0f);

        // Diagonal gradient
        Var diagonal = sin((uv.x + uv.y) * 10.0f);

        // Combine patterns
        Var pattern = checker * 0.3f + circles * 0.4f + diagonal * 0.3f;

        // Add some "features" (simulating a face-like structure)
        Var eye1 = exp(-length(uv - make_float2(0.35f, 0.4f)) * 50.0f);
        Var eye2 = exp(-length(uv - make_float2(0.65f, 0.4f)) * 50.0f);
        Var mouth = exp(-pow(uv.y - 0.6f - 0.1f * pow(uv.x - 0.5f, 2.0f), 2.0f) * 200.0f);

        Var final_val = pattern * 0.5f + 0.5f;
        final_val = final_val + eye1 * 0.3f + eye2 * 0.3f + mouth * 0.2f;

        Var tint = make_float3(
            0.5f + 0.5f * sin(uv.x * 6.2831853f + 0.0f),
            0.5f + 0.5f * sin(uv.y * 6.2831853f + 2.0943951f),
            0.5f + 0.5f * sin((uv.x + uv.y) * 6.2831853f + 4.1887902f));
        Var color = final_val * tint;

        image.write(dispatch_id().xy(), make_float4(color, 1.0f));
    };

    auto pattern_shader = device.compile(generate_pattern);

    // Gaussian blur - Horizontal pass
    // Uses a 9-tap Gaussian kernel with sigma = 2.0
    Kernel2D gaussian_blur_h = [](ImageFloat input, ImageFloat output) noexcept {
        set_block_size(16, 16, 1);
        Var uv = dispatch_id().xy();
        Var size = dispatch_size().xy();

        // Gaussian weights (normalized, sigma=2.0)
        Float weights[9] = {0.006f, 0.028f, 0.084f, 0.168f, 0.224f,
                            0.168f, 0.084f, 0.028f, 0.006f};
        Int offsets[9] = {-4, -3, -2, -1, 0, 1, 2, 3, 4};

        Var sum = make_float3(0.0f);
        for (uint i = 0u; i < 9u; i++) {
            Var sample_uv = make_uint2(
                clamp(cast<uint>(cast<int>(uv.x) + offsets[i]), 0u, size.x - 1u),
                uv.y);
            sum += input.read(sample_uv).xyz() * weights[i];
        }

        output.write(uv, make_float4(sum, 1.0f));
    };

    auto blur_h_shader = device.compile(gaussian_blur_h);

    // Gaussian blur - Vertical pass
    Kernel2D gaussian_blur_v = [](ImageFloat input, ImageFloat output) noexcept {
        set_block_size(16, 16, 1);
        Var uv = dispatch_id().xy();
        Var size = dispatch_size().xy();

        Float weights[9] = {0.006f, 0.028f, 0.084f, 0.168f, 0.224f,
                            0.168f, 0.084f, 0.028f, 0.006f};
        Int offsets[9] = {-4, -3, -2, -1, 0, 1, 2, 3, 4};

        Var sum = make_float3(0.0f);
        for (uint i = 0u; i < 9u; i++) {
            Var sample_uv = make_uint2(
                uv.x,
                clamp(cast<uint>(cast<int>(uv.y) + offsets[i]), 0u, size.y - 1u));
            sum += input.read(sample_uv).xyz() * weights[i];
        }

        output.write(uv, make_float4(sum, 1.0f));
    };

    auto blur_v_shader = device.compile(gaussian_blur_v);

    // Sobel edge detection
    // Computes gradient magnitude using Sobel operators
    Kernel2D sobel_edge = [](ImageFloat input, ImageFloat output) noexcept {
        set_block_size(16, 16, 1);
        Var uv = dispatch_id().xy();
        Var size = dispatch_size().xy();

        // Sample neighbors with clamping
        auto sample = [&](Int2 offset) noexcept {
            Var sample_uv = make_uint2(
                clamp(cast<uint>(cast<int>(uv.x) + offset.x), 0u, size.x - 1u),
                clamp(cast<uint>(cast<int>(uv.y) + offset.y), 0u, size.y - 1u));
            return input.read(sample_uv).x;// Use luminance
        };

        // Sobel X kernel: [-1 0 1; -2 0 2; -1 0 1]
        Var gx = sample(make_int2(-1, -1)) * -1.0f +
                 sample(make_int2(1, -1)) * 1.0f +
                 sample(make_int2(-1, 0)) * -2.0f +
                 sample(make_int2(1, 0)) * 2.0f +
                 sample(make_int2(-1, 1)) * -1.0f +
                 sample(make_int2(1, 1)) * 1.0f;

        // Sobel Y kernel: [-1 -2 -1; 0 0 0; 1 2 1]
        Var gy = sample(make_int2(-1, -1)) * -1.0f +
                 sample(make_int2(0, -1)) * -2.0f +
                 sample(make_int2(1, -1)) * -1.0f +
                 sample(make_int2(-1, 1)) * 1.0f +
                 sample(make_int2(0, 1)) * 2.0f +
                 sample(make_int2(1, 1)) * 1.0f;

        // Gradient magnitude
        Var magnitude = sqrt(gx * gx + gy * gy);

        // Edge direction visualization (optional)
        Var angle = atan2(gy, gx) / 3.14159f * 0.5f + 0.5f;

        // Output: edge intensity in R, angle in G
        output.write(uv, make_float4(magnitude, angle, 0.0f, 1.0f));
    };

    auto sobel_shader = device.compile(sobel_edge);

    // Compositing kernel - combines original, blurred, and edge images
    Kernel2D composite = [](ImageFloat original, ImageFloat blurred,
                            ImageFloat edges, ImageFloat output, Float edge_intensity) noexcept {
        set_block_size(16, 16, 1);
        Var uv = dispatch_id().xy();

        Var orig = original.read(uv);
        Var blur = blurred.read(uv);
        Var edge = edges.read(uv);

        // Unsharp mask: sharpened = original + (original - blurred) * amount
        Var sharpen_amount = 1.5f;
        Var sharpened = orig.xyz() + (orig.xyz() - blur.xyz()) * sharpen_amount;
        sharpened = clamp(sharpened, 0.0f, 1.0f);

        // Edge overlay with color
        Var edge_color = make_float3(0.0f, 0.8f, 1.0f) * edge.x * edge_intensity;

        // Final composite
        Var final_color = lerp(sharpened, edge_color, edge.x * 0.5f);

        // Vignette effect
        Var coord = make_float2(uv) / make_float2(dispatch_size().xy());
        Var center_dist = length(coord - make_float2(0.5f));
        Var vignette = 1.0f - center_dist * 0.5f;

        output.write(uv, make_float4(final_color * vignette, 1.0f));
    };

    auto composite_shader = device.compile(composite);

    // Create intermediate images
    Image<float> source_image = device.create_image<float>(PixelStorage::FLOAT4, width, height);
    Image<float> temp_image = device.create_image<float>(PixelStorage::FLOAT4, width, height);
    Image<float> blurred_image = device.create_image<float>(PixelStorage::FLOAT4, width, height);
    Image<float> edge_image = device.create_image<float>(PixelStorage::FLOAT4, width, height);

    // Setup window
#if ENABLE_DISPLAY
    std::unique_ptr<Window> window;
    std::optional<Swapchain> swap_chain;
    if (!force_offline) {
        window = std::make_unique<Window>("Image Processing Pipeline", make_uint2(width, height));
        swap_chain.emplace(device.create_swapchain(
            stream,
            SwapchainOption{
                .display = window->native_display(),
                .window = window->native_handle(),
                .size = window->size(),
                .wants_vsync = true,
            }));
    }
#endif
    Image<float> display = [&] {
#if ENABLE_DISPLAY
        if (!force_offline) {
            return device.create_image<float>(swap_chain->backend_storage(), window->size());
        }
#endif
        return device.create_image<float>(PixelStorage::BYTE4, width, height);
    }();

    // Generate initial test pattern
    stream << pattern_shader(source_image).dispatch(width, height);

    // Main loop with animated edge intensity
    Clock clock;
    if (force_offline) {
        // Keep one-time pattern generation outside the repeated pipeline
        // timing window.
        stream << synchronize();
        float time = 0.0f;
        float edge_intensity = (sinf(time) + 1.0f) * 0.5f;
        Clock benchmark_clock;
        for (auto iteration = 0u; iteration < offline_iterations; iteration++) {
            stream << blur_h_shader(source_image, temp_image).dispatch(width, height)
                   << blur_v_shader(temp_image, blurred_image).dispatch(width, height)
                   << sobel_shader(source_image, edge_image).dispatch(width, height)
                   << composite_shader(source_image, blurred_image, edge_image,
                                       display, edge_intensity)
                          .dispatch(width, height);
        }
        stream << synchronize();
        auto elapsed_ms = benchmark_clock.toc();
        LUISA_INFO(
            "Offline image pipeline: {} iterations in {:.3f} ms ({:.3f} ms/iteration).",
            offline_iterations, elapsed_ms,
            elapsed_ms / static_cast<double>(offline_iterations));
        luisa::vector<uint8_t> host_image(width * height * 4u);
        stream << display.copy_to(luisa::span{host_image}) << synchronize();
        stbi_write_png("test_image_processing.png", width, height, 4, host_image.data(), 0);
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
        stream << blur_h_shader(source_image, temp_image).dispatch(width, height)
               << blur_v_shader(temp_image, blurred_image).dispatch(width, height)
               << sobel_shader(source_image, edge_image).dispatch(width, height);
        while (!window->should_close()) {
            window->poll_events();

            float time = static_cast<float>(clock.toc() * 1e-3);
            float edge_intensity = (sinf(time) + 1.0f) * 0.5f;// Pulsing effect

            // Run compositing with animated parameters
            stream << composite_shader(source_image, blurred_image, edge_image,
                                       display, edge_intensity)
                          .dispatch(width, height)
                   << swap_chain->present(display);
        }
#endif
    }

    stream << synchronize();
}
