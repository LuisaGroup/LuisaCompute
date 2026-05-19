#include <filesystem>
#include <memory>
#include <optional>
#include <random>
#include <string_view>

#include "../common/reference_compare.h"

#include "../../src/ext/stb/stb/stb_image.h"

#include <luisa/luisa-compute.h>
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

/*
    "Starship" by @XorDev

    Inspired by the debris from SpaceX's 7th Starship test:
    https://x.com/elonmusk/status/1880040599761596689

    My original twigl version:
    https://x.com/XorDev/status/1880344887033569682

    <512 Chars playlist: shadertoy.com/playlist/N3SyzR
*/

int main(int argc, char *argv[]) {

    Context context{argv[0]};

    if (argc <= 1) {
        LUISA_INFO("Usage: {} <backend> [<image>] [--offline] [-c <reference.png>]. <backend>: cuda, dx, cpu, metal", argv[0]);
        exit(1);
    }
    bool force_offline = false;
    std::optional<std::filesystem::path> compare_path;
    const char *input_image = nullptr;
    for (int i = 2; i < argc; i++) {
        if (std::string_view{argv[i]} == "--offline") {
            force_offline = true;
        } else if ((std::string_view{argv[i]} == "--compare" || std::string_view{argv[i]} == "-c") && i + 1 < argc) {
            compare_path = std::filesystem::path{argv[++i]};
            force_offline = true;
            force_offline = true;
        } else {
            input_image = argv[i];
        }
    }
#if !ENABLE_DISPLAY
    if (!force_offline) {
        LUISA_ERROR("GUI support is disabled. Use --offline.");
    }
#endif
    auto device = context.create_device(argv[1]);
    auto stream = device.create_stream(force_offline ? StreamTag::COMPUTE : StreamTag::GRAPHICS);
    auto image = [&] {
        if (input_image != nullptr) {// load texture from file
            auto image_width = 0, image_height = 0, image_channels = 0;
            auto image_pixels = stbi_load(input_image, &image_width, &image_height, &image_channels, 1);
            LUISA_ASSERT(image_pixels != nullptr, "Failed to load image: {}.", input_image);
            auto texture = device.create_image<float>(PixelStorage::BYTE1, image_width, image_height);
            stream << texture.copy_from(luisa::span{image_pixels, static_cast<size_t>(image_width * image_height)});
            stbi_image_free(image_pixels);
            return texture;
        }
        // generate random texture
        auto size = make_uint2(128u);
        luisa::vector<uint8_t> pixels(size.x * size.y);
        std::mt19937 rng{force_offline ? 42u : std::random_device{}()};
        std::uniform_int_distribution<uint32_t> dist;
        for (auto &x : pixels) { x = static_cast<uint8_t>(dist(rng) & 0xffu); }
        auto texture = device.create_image<float>(PixelStorage::BYTE1, size);
        stream << texture.copy_from(luisa::span{pixels});
        return texture;
    }();

    auto bindless = device.create_bindless_array(1u);
    bindless.emplace_on_update(0u, image, Sampler::linear_point_mirror());
    stream << bindless.update();

    Callable main_image = [](BindlessVar iChannel0, Float iTime, Float2 I) noexcept {
        auto r = make_float2(dispatch_size().xy());
        auto p = make_float2x2(4.f, -3.f, 3.f, 4.f) * ((I + I - r) / r.y);
        auto t = iTime;
        auto T = t + .1f * p.x;
        auto O = def(make_float4());
        $for (j, 50u) {
            auto i = cast<float>(j);
            auto s = iChannel0->tex2d(0u).sample(p / exp(sin(i) + 5.f) + make_float2(t, i) / 8.f).x;
            O += (cos(sin(i) * make_float4(1.f, 2.f, 3.f, 0.f)) + 1.f) *
                 exp(sin(i + .1f * i * T)) /
                 length(max(p, p / make_float2(s * 40.f, 2.f)));
            p += 2.f * cos(i * make_float2(11.f, 9.f) + i * i + T * .2f);
        };
        return tanh(clamp(.01f * p.y * make_float4(0.f, 1.f, 2.f, 3.f) + O * O / 1e4f, -10.f, 10.f));
    };

    auto shader = device.compile<2>([&main_image](ImageFloat output, BindlessVar iChannel0, Float iTime) noexcept {
        auto p = dispatch_id().xy();
        auto i = make_float2(p) + .5f;
        auto o = main_image(iChannel0, iTime, i);
        output.write(p, make_float4(o.xyz(), 1.f));
    });

    auto resolution = make_uint2(1280u, 720u);

#if ENABLE_DISPLAY
    std::unique_ptr<Window> window;
    std::optional<Swapchain> swapchain;
    if (!force_offline) {
        window = std::make_unique<Window>("Starship", resolution);
        SwapchainOption swapchain_option{
            .display = window->native_display(),
            .window = window->native_handle(),
            .size = resolution,
            .wants_hdr = false,
        };
        swapchain.emplace(device.create_swapchain(stream, swapchain_option));
    }
#endif
    auto framebuffer = [&] {
#if ENABLE_DISPLAY
        if (!force_offline) {
            return device.create_image<float>(swapchain->backend_storage(), resolution);
        }
#endif
        return device.create_image<float>(PixelStorage::BYTE4, resolution);
    }();

    Clock clk;
    if (force_offline) {
        auto time = 0.0f;
        stream << shader(framebuffer, bindless, time).dispatch(resolution);
        luisa::vector<uint8_t> host_image(resolution.x * resolution.y * 4u);
        stream << framebuffer.copy_to(luisa::span{host_image}) << synchronize();
        stbi_write_png("test_shader_toy_spacex.png", static_cast<int>(resolution.x), static_cast<int>(resolution.y), 4, host_image.data(), 0);
        if (compare_path) {
            auto result = luisa::ref::compare_with_reference_file(
                reinterpret_cast<const uint8_t *>(host_image.data()),
                static_cast<int>(resolution.x), static_cast<int>(resolution.y), 4,
                *compare_path);
            LUISA_INFO("Reference comparison: {} ({})", result.passed ? "PASSED" : "FAILED", result.message);
            if (!result.passed) { return 1; }
        }
    } else {
#if ENABLE_DISPLAY
        while (!window->should_close()) {
            window->poll_events();
            auto time = static_cast<float>(clk.toc() * 1e-3);
            stream << shader(framebuffer, bindless, time).dispatch(resolution)
                   << swapchain->present(framebuffer);
        }
#endif
    }
    stream.synchronize();
}
