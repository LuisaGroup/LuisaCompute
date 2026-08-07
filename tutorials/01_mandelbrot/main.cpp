/*
 * Tutorial 01: ShaderToy-style Mandelbrot Renderer
 *
 * This tutorial teaches the complete LuisaCompute workflow for a small image-space renderer:
 * 1. Create a Context, choose a backend, and create a Device/Stream.
 * 2. Allocate images that shaders can write into.
 * 3. Define a 2D kernel with the LuisaCompute DSL sugar syntax.
 * 4. Implement the Mandelbrot iteration z = z^2 + c on the GPU.
 * 5. Map iteration counts to colors with a cosine palette.
 * 6. Dispatch the shader, save the result to a PNG, and optionally present it in a window.
 *
 * Why this tutorial matters:
 * The Mandelbrot set is simple enough to understand mathematically, but rich enough to show how
 * LuisaCompute kernels launch one thread per pixel and turn GPU math into an image.
 */

#include <cstdlib>
#include <memory>
#include <optional>
#include <string_view>
#include <array>

#include <stb/stb_image_write.h>

#include <luisa/luisa-compute.h>
#include <luisa/dsl/sugar.h>
#include <luisa/runtime/swapchain.h>
#ifndef ENABLE_DISPLAY
#ifdef LUISA_ENABLE_GUI
#define ENABLE_DISPLAY 1
#else
#define ENABLE_DISPLAY 0
#endif
#endif

#if ENABLE_DISPLAY
#include <luisa/gui/window.h>
#endif

using namespace luisa;
using namespace luisa::compute;

int main(int argc, char *argv[]) {

    log_level_verbose();

    // Step 1: Create the runtime context and pick a backend.
    // Why: the Context knows where LuisaCompute backends live, and the Device is the object that
    // creates GPU resources and compiles kernels for the chosen backend.
    // Parse command-line: any non-flag argument is the backend name.
    // If no backend is given, the first installed backend is selected automatically.
    luisa::string backend;
    bool offline = false;
    for (int i = 1; i < argc; i++) {
        if (std::string_view{argv[i]} == "--offline") {
            offline = true;
        } else if (backend.empty()) {
            backend = argv[i];
        }
    }

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

    if (!offline && !ENABLE_DISPLAY) {
        LUISA_WARNING("GUI support is disabled in this build. Falling back to --offline mode.");
        offline = true;
    }

    Device device = context.create_device(backend);
    Stream stream = device.create_stream(offline ? StreamTag::COMPUTE : StreamTag::GRAPHICS);

    static constexpr uint2 resolution = make_uint2(1024u, 1024u);
    static constexpr uint max_iterations = 100u;
    constexpr char output_file[] = "tutorial_01_mandelbrot.png";

    LUISA_INFO("Tutorial 01: rendering Mandelbrot set at {}x{}.", resolution.x, resolution.y);

    // Step 2: Create images for the result.
    // Why: a kernel needs a writable device resource. We keep one BYTE4 image for PNG output, and
    // in interactive mode we allocate a second image using the swapchain's native storage format.
    Image<float> png_image = device.create_image<float>(PixelStorage::BYTE4, resolution);

    std::optional<Image<float>> display_image;
#if ENABLE_DISPLAY
    std::unique_ptr<Window> window;
    std::optional<Swapchain> swapchain;
    if (!offline) {
        window = std::make_unique<Window>("Tutorial 01 - Mandelbrot", resolution, false);
        swapchain.emplace(device.create_swapchain(
            stream,
            SwapchainOption{
                .display = window->native_display(),
                .window = window->native_handle(),
                .size = resolution,
                .wants_hdr = false,
                .wants_vsync = true,
                .back_buffer_count = 2,
            }));
        display_image.emplace(device.create_image<float>(swapchain->backend_storage(), resolution));
    }
#endif

    // Step 3: Define a cosine palette helper.
    // Why: fractal escape counts are just numbers; a palette turns those numbers into a readable,
    // visually pleasing image so we can see structure in the complex plane.
    Callable cosine_palette = [](Float t) noexcept {
        Float3 a = make_float3(0.50f, 0.50f, 0.50f);
        Float3 b = make_float3(0.50f, 0.50f, 0.50f);
        Float3 c = make_float3(1.00f, 1.00f, 1.00f);
        Float3 d = make_float3(0.00f, 0.10f, 0.20f);
        return a + b * cos(2.0f * constants::pi * (c * t + d));
    };

    // Step 4: Define the Mandelbrot kernel.
    // Why: each GPU thread corresponds to one pixel. The kernel maps the pixel to a complex number
    // c, iterates z = z^2 + c, and colors the pixel based on how quickly the sequence escapes.
    Kernel2D mandelbrot_kernel = [&](ImageFloat image) noexcept {
        set_block_size(16u, 16u, 1u);

        UInt2 pixel = dispatch_id().xy();
        Float2 uv = (make_float2(pixel) + 0.5f) / make_float2(dispatch_size().xy());

        // Step 4.1: Map pixel coordinates to the region of the complex plane we want to inspect.
        // Why: screen coordinates are in [0, 1], but the fractal lives in the complex plane.
        Float2 c = make_float2(
            lerp(-2.2f, 0.8f, uv.x),
            lerp(-1.5f, 1.5f, uv.y));

        // Step 4.2: Start the classic Mandelbrot iteration at z = 0.
        Float2 z = make_float2(0.0f);
        UInt escaped_iteration = def(max_iterations);
        Bool escaped = def(false);

        // Step 4.3: Iterate up to max_iterations.
        // Why: pixels inside the set never escape within the budget, while others do. The escape
        // time produces the familiar boundary detail.
        $for (iteration, max_iterations) {
            Float x = z.x;
            Float y = z.y;
            z = make_float2(x * x - y * y, 2.0f * x * y) + c;

            Float radius_squared = dot(z, z);
            $if (!escaped & radius_squared > 4.0f) {
                escaped = true;
                escaped_iteration = iteration;
            };
            $if (escaped) {
                $break;
            };
        };

        // Step 4.4: Turn iteration count into color.
        // Why: smooth coloring avoids harsh bands and makes the fractal easier to read.
        Float3 color = make_float3(0.0f);
        $if (escaped) {
            Float smooth_iteration = cast<float>(escaped_iteration) + 1.0f -
                                     log2(max(log2(length(z)), 1e-6f));
            Float t = clamp(smooth_iteration / static_cast<float>(max_iterations), 0.0f, 1.0f);
            color = cosine_palette(t);
        }
        $else {
            color = make_float3(0.0f, 0.0f, 0.0f);
        };

        image.write(pixel, make_float4(clamp(color, 0.0f, 1.0f), 1.0f));
    };

    LUISA_INFO("Compiling Mandelbrot kernel...");
    auto mandelbrot_shader = device.compile(mandelbrot_kernel);

    // Step 5: Render once to the PNG image.
    // Why: the fractal is static, so one dispatch is enough for offline output.
    stream << mandelbrot_shader(png_image).dispatch(resolution);

    // Step 6: Read the rendered image back to the CPU and write a PNG file.
    // Why: kernels run on the device; stbi_write_png needs host memory.
    luisa::vector<std::array<uint8_t, 4u>> host_image(resolution.x * resolution.y);
    stream << png_image.copy_to(luisa::span{host_image})
           << synchronize();
    stbi_write_png(output_file, static_cast<int>(resolution.x), static_cast<int>(resolution.y), 4, host_image.data(), 0);
    LUISA_INFO("Saved Mandelbrot image to {}.", output_file);

    // Step 7: In interactive mode, keep presenting the image in a window.
    // Why: seeing the same result in a swapchain shows the normal real-time rendering path used by
    // graphical applications, even though this tutorial's image itself is static.
    if (!offline) {
        LUISA_INFO("Opening interactive window. Close the window or press ESC to exit.");
#if ENABLE_DISPLAY
        while (!window->should_close()) {
            window->poll_events();
            if (window->is_key_down(KEY_ESCAPE)) {
                break;
            }
            stream << mandelbrot_shader(*display_image).dispatch(resolution)
                   << swapchain->present(*display_image);
        }
        stream << synchronize();
#endif
    }

    return 0;
}
