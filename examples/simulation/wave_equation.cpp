// 2D Wave Equation Simulation - Interactive Water Ripples
// Simulates realistic water ripples using the finite difference method.
// Click and drag to create waves, watch them propagate and interfere.
//
// Features demonstrated:
// - Finite difference method for PDEs
// - Three-buffer time integration scheme
// - Interactive mouse input via callbacks
// - Real-time water surface rendering with caustics

#include <algorithm>
#include <filesystem>
#include <memory>
#include <optional>
#include <random>
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
    LUISA_INFO("2D Wave Equation Simulation - Interactive Water Ripples");
    LUISA_INFO("Controls: Click+Drag = Create waves, Space = Reset, ESC = Quit");

    // Simulation parameters - tuned for nice wave propagation
    static constexpr uint width = 512u;
    static constexpr uint height = 512u;
    // c^2 * dt^2 should be < 0.5 for stability (CFL condition)
    static constexpr float c = 0.2f;        // Wave speed (lower = slower waves)
    static constexpr float dt = 1.0f;       // Time step
    static constexpr float damping = 0.995f;// Slight damping for energy loss

    // Create three buffers for time integration
    Image<float> height_prev = device.create_image<float>(PixelStorage::FLOAT1, width, height);
    Image<float> height_curr = device.create_image<float>(PixelStorage::FLOAT1, width, height);
    Image<float> height_next = device.create_image<float>(PixelStorage::FLOAT1, width, height);

    Stream stream = device.create_stream(force_offline ? StreamTag::COMPUTE : StreamTag::GRAPHICS);

    // Clear kernel
    Kernel2D clear_kernel = [](ImageFloat image) noexcept {
        set_block_size(16, 16, 1);
        image.write(dispatch_id().xy(), make_float4(0.0f));
    };
    auto clear = device.compile(clear_kernel);

    // Initialize
    stream << clear(height_prev).dispatch(width, height)
           << clear(height_curr).dispatch(width, height)
           << clear(height_next).dispatch(width, height);

    // Wave equation solver kernel
    Kernel2D wave_step = [&](ImageFloat prev, ImageFloat curr, ImageFloat next) noexcept {
        set_block_size(16, 16, 1);
        Var uv = dispatch_id().xy();
        Var size = dispatch_size().xy();

        Var u_curr = curr.read(uv).x;

        // Sample neighbors with clamping
        auto sample = [&](Int2 offset) noexcept {
            Int2 p = make_int2(uv) + offset;
            p.x = clamp(p.x, 0, cast<int>(size.x) - 1);
            p.y = clamp(p.y, 0, cast<int>(size.y) - 1);
            return curr.read(make_uint2(p)).x;
        };

        Var u_left = sample(make_int2(-1, 0));
        Var u_right = sample(make_int2(1, 0));
        Var u_up = sample(make_int2(0, -1));
        Var u_down = sample(make_int2(0, 1));

        Var laplacian = u_left + u_right + u_up + u_down - 4.0f * u_curr;
        Var u_prev = prev.read(uv).x;
        Var u_next = 2.0f * u_curr - u_prev + (c * c * dt * dt) * laplacian;
        u_next *= damping;
        u_next = clamp(u_next, -2.0f, 2.0f);

        next.write(uv, make_float4(u_next, 0.0f, 0.0f, 1.0f));
    };

    auto wave_shader = device.compile(wave_step);

    // Droplet drop kernel
    Kernel2D drop_droplet = [](ImageFloat height, UInt2 center, Float strength) noexcept {
        set_block_size(16, 16, 1);
        Var uv = dispatch_id().xy();
        Var dx = cast<float>(cast<int>(uv.x) - cast<int>(center.x));
        Var dy = cast<float>(cast<int>(uv.y) - cast<int>(center.y));
        Var dist_sq = dx * dx + dy * dy;
        Var radius = 15.0f;
        Var gaussian = exp(-dist_sq / (radius * radius * 0.5f));
        Var current = height.read(uv).x;
        height.write(uv, make_float4(current + strength * gaussian, 0.0f, 0.0f, 1.0f));
    };

    auto drop_shader = device.compile(drop_droplet);

    // Rendering kernel with caustics effect
    Kernel2D render_kernel = [](ImageFloat height, ImageFloat output, Float time) noexcept {
        set_block_size(16, 16, 1);
        Var uv = dispatch_id().xy();
        Var size = dispatch_size().xy();

        Var h = height.read(uv).x;

        auto sample = [&](Int2 offset) noexcept {
            Int2 p = make_int2(uv) + offset;
            p.x = clamp(p.x, 0, cast<int>(size.x) - 1);
            p.y = clamp(p.y, 0, cast<int>(size.y) - 1);
            return height.read(make_uint2(p)).x;
        };

        Var h_left = sample(make_int2(-2, 0));
        Var h_right = sample(make_int2(2, 0));
        Var h_up = sample(make_int2(0, -2));
        Var h_down = sample(make_int2(0, 2));

        Var slope_x = abs(h_right - h_left);
        Var slope_y = abs(h_down - h_up);
        Var slope = sqrt(slope_x * slope_x + slope_y * slope_y);

        Var deep_color = make_float3(0.0f, 0.1f, 0.3f);
        Var shallow_color = make_float3(0.0f, 0.4f, 0.6f);
        Var foam_color = make_float3(0.9f, 0.95f, 1.0f);
        Var highlight_color = make_float3(0.6f, 0.8f, 1.0f);

        Var t = clamp(h * 0.5f + 0.5f, 0.0f, 1.0f);
        Var base_color = lerp(deep_color, shallow_color, t);

        Var caustics = smoothstep(0.1f, 0.5f, slope);
        base_color = lerp(base_color, highlight_color, caustics * 0.5f);

        Var foam = smoothstep(0.5f, 1.0f, h);
        base_color = lerp(base_color, foam_color, foam * 0.7f);

        Var shimmer = sin(cast<float>(uv.x) * 0.1f + time * 2.0f) * sin(cast<float>(uv.y) * 0.1f + time * 1.5f);
        shimmer = (shimmer + 1.0f) * 0.5f;
        base_color = base_color * (0.9f + 0.1f * shimmer);

        output.write(uv, make_float4(base_color, 1.0f));
    };

    auto render = device.compile(render_kernel);

    // Setup window
    static constexpr uint display_scale = 2u;
    static constexpr uint display_width = width * display_scale;
    static constexpr uint display_height = height * display_scale;

#if ENABLE_DISPLAY
    std::unique_ptr<Window> window;
    if (!force_offline) {
        window = std::make_unique<Window>("Water Ripples - Click and Drag!", make_uint2(display_width, display_height));
    }
#endif

    // Mouse state for interaction
    bool mouse_down = false;
    float2 mouse_pos{0.0f, 0.0f};
    float2 last_mouse_pos{-1.0f, -1.0f};

    // Setup mouse callbacks
#if ENABLE_DISPLAY
    if (!force_offline) {
        window->set_mouse_callback([&mouse_down, &mouse_pos, &last_mouse_pos](MouseButton, Action a, float2 p) noexcept {
            if (a == Action::ACTION_PRESSED || a == Action::ACTION_REPEATED) {
                mouse_down = true;
                last_mouse_pos = make_float2(-1.0f, -1.0f);// Reset for new stroke
            } else {
                mouse_down = false;
            }
            mouse_pos = p;
        });

        window->set_cursor_position_callback([&mouse_down, &mouse_pos](float2 p) noexcept {
            mouse_pos = p;
        });
    }
    std::optional<Swapchain> swap_chain;
    if (!force_offline) {
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
        return device.create_image<float>(PixelStorage::BYTE4, display_width, display_height);
    }();

    // Upscaling kernel
    Kernel2D upscale_kernel = [](ImageFloat input, ImageFloat output) noexcept {
        set_block_size(16, 16, 1);
        Var uv = dispatch_id().xy();
        Var sample_uv = uv / 2u;
        output.write(uv, input.read(sample_uv));
    };
    auto upscale = device.compile(upscale_kernel);

    // Main simulation loop
    Clock clock;
    std::mt19937 rng{force_offline ? 42u : std::random_device{}()};
    float time = 0.0f;
    Image<float> temp_display = device.create_image<float>(PixelStorage::FLOAT4, width, height);
    auto simulate_frame = [&](CommandList &cmdlist) noexcept {
        // Handle mouse interaction - drop droplets when mouse is down
        if (mouse_down) {
            uint sim_x = clamp(cast<uint>(mouse_pos.x / display_scale), 0u, width - 1);
            uint sim_y = clamp(cast<uint>(mouse_pos.y / display_scale), 0u, height - 1);

            float drop_strength = -0.5f;

            if (last_mouse_pos.x >= 0) {
                uint last_x = clamp(cast<uint>(last_mouse_pos.x / display_scale), 0u, width - 1);
                uint last_y = clamp(cast<uint>(last_mouse_pos.y / display_scale), 0u, height - 1);
                int dx = (int)sim_x - (int)last_x;
                int dy = (int)sim_y - (int)last_y;
                int steps = max(abs(dx), abs(dy));
                for (int i = 0; i <= steps && i < 10; i++) {
                    float t = (steps == 0) ? 0.0f : (float)i / steps;
                    uint ix = last_x + cast<uint>(dx * t);
                    uint iy = last_y + cast<uint>(dy * t);
                    cmdlist << drop_shader(height_curr, make_uint2(ix, iy), drop_strength * 0.7f).dispatch(width, height);
                }
            } else {
                cmdlist << drop_shader(height_curr, make_uint2(sim_x, sim_y), drop_strength).dispatch(width, height);
            }

            last_mouse_pos = mouse_pos;
        }

        if (rng() % 120u == 0u) {
            uint rain_x = 50 + rng() % (width - 100);
            uint rain_y = 50 + rng() % (height - 100);
            cmdlist << drop_shader(height_curr, make_uint2(rain_x, rain_y), -0.3f).dispatch(width, height);
        }

        for (int step = 0; step < 4; step++) {
            cmdlist << wave_shader(height_prev, height_curr, height_next).dispatch(width, height);
            std::swap(height_prev, height_curr);
            std::swap(height_curr, height_next);
        }

        time += 0.016f;
        cmdlist << render(height_curr, temp_display, time).dispatch(width, height)
                << upscale(temp_display, display).dispatch(display_width, display_height);
    };
    if (force_offline) {
        static constexpr uint offline_frames = 500u;
        for (uint i = 0u; i < offline_frames; i++) {
            CommandList cmdlist;
            simulate_frame(cmdlist);
            stream << cmdlist.commit();
        }
        luisa::vector<uint8_t> host_image(display_width * display_height * 4u);
        stream << display.copy_to(luisa::span{host_image}) << synchronize();
        static constexpr uint feature_tile_size = 32u;
        luisa::vector<uint8_t> feature_tiles(
            (display_width / feature_tile_size) * (display_height / feature_tile_size), 0u);
        size_t bright_pixel_count = 0u;
        size_t wave_pixel_count = 0u;
        size_t strong_wave_pixel_count = 0u;
        for (auto i = 0u; i < display_width * display_height; i++) {
            auto offset = static_cast<size_t>(i) * 4u;
            auto peak = std::max({host_image[offset + 0u],
                                  host_image[offset + 1u],
                                  host_image[offset + 2u]});
            bright_pixel_count += peak >= 128u;
            auto red = host_image[offset];
            if (red >= 20u) {
                wave_pixel_count++;
                auto x = i % display_width;
                auto y = i / display_width;
                auto tile_x = x / feature_tile_size;
                auto tile_y = y / feature_tile_size;
                feature_tiles[tile_y * (display_width / feature_tile_size) + tile_x] = 1u;
            }
            strong_wave_pixel_count += red >= 40u;
        }
        auto feature_tile_count = static_cast<size_t>(std::count(feature_tiles.cbegin(), feature_tiles.cend(), uint8_t{1u}));
        auto scene_is_valid = bright_pixel_count >= 2000u && bright_pixel_count <= 20000u &&
                              wave_pixel_count >= 2000u && wave_pixel_count <= 20000u &&
                              strong_wave_pixel_count >= 500u && strong_wave_pixel_count <= 10000u &&
                              feature_tile_count >= 8u && feature_tile_count <= 64u;
        if (!scene_is_valid) {
            LUISA_ERROR(
                "Wave output failed feature checks: {} bright pixels (expected 2000-20000), "
                "{} wave pixels (expected 2000-20000), {} strong wave pixels (expected 500-10000), "
                "{} occupied feature tiles (expected 8-64).",
                bright_pixel_count, wave_pixel_count, strong_wave_pixel_count, feature_tile_count);
            return 1;
        }
        if (stbi_write_png("test_wave_equation.png", display_width, display_height, 4, host_image.data(), 0) == 0) {
            LUISA_ERROR("Failed to write test_wave_equation.png.");
            return 1;
        }
        if (compare_path) {
            auto result = luisa::ref::compare_with_reference_file(
                reinterpret_cast<const uint8_t *>(host_image.data()),
                display_width, display_height, 4,
                *compare_path);
            LUISA_INFO("Reference comparison: {} ({})", result.passed ? "PASSED" : "FAILED", result.message);
            if (!result.passed) { return 1; }
        }
    } else {
#if ENABLE_DISPLAY
        while (!window->should_close()) {
            CommandList cmdlist;
            window->poll_events();

            if (window->is_key_down(KEY_ESCAPE)) {
                break;
            }
            if (window->is_key_down(KEY_SPACE)) {
                cmdlist << clear(height_prev).dispatch(width, height)
                        << clear(height_curr).dispatch(width, height)
                        << clear(height_next).dispatch(width, height);
            }
            simulate_frame(cmdlist);
            stream << cmdlist.commit() << swap_chain->present(display);
        }
#endif
    }

    stream << synchronize();
}
