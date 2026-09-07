/*
 * Tutorial 05: 2D Wave Equation
 *
 * This tutorial teaches how to simulate and visualize waves on a height field.
 * The core idea is to keep three time slices of the solution (previous, current,
 * and next), integrate the wave equation with a finite-difference stencil, inject
 * droplets interactively, and shade the resulting height field as stylized water.
 *
 * What you will learn:
 *   1. Why multi-buffer time integration is the simplest way to evolve PDEs on the GPU.
 *   2. How to express local stencils, Gaussian impulses, and slope-based shading in DSL code.
 *   3. How to bridge mouse input on the CPU with GPU droplet injection and offline rendering.
 */

#include <algorithm>
#include <array>
#include <cstddef>
#include <cmath>
#include <cstdint>
#include <random>
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

struct HeightFieldTriple {
    Image<float> prev;
    Image<float> curr;
    Image<float> next;

    HeightFieldTriple(Device &device, uint2 resolution) noexcept
        : prev{device.create_image<float>(PixelStorage::FLOAT1, resolution)},
          curr{device.create_image<float>(PixelStorage::FLOAT1, resolution)},
          next{device.create_image<float>(PixelStorage::FLOAT1, resolution)} {}

    void advance() noexcept {
        std::swap(prev, curr);
        std::swap(curr, next);
    }
};

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

    // Step 1: Set up the LuisaCompute runtime.
    // We use a graphics stream so the same stream can both compute the simulation and present it.
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

    static constexpr auto grid_resolution = make_uint2(512u, 512u);
    static constexpr auto display_resolution = make_uint2(1024u, 1024u);
    static constexpr auto display_scale = 2u;
    static constexpr auto c = 0.2f;
    static constexpr auto dt = 1.0f;
    static constexpr auto damping = 0.995f;
    static constexpr auto c2dt2 = c * c * dt * dt;
    static_assert(c2dt2 < 0.5f);

    LUISA_INFO("Tutorial 05 - 2D Wave Equation");
    LUISA_INFO("Backend: {}", backend);
    LUISA_INFO("Mode: {}", offline ? "offline" : "interactive");
    LUISA_INFO("CFL check: c^2 * dt^2 = {} (< 0.5)", c2dt2);
    if (!offline) {
        LUISA_INFO("Controls: Click + drag = waves, SPACE = reset, ESC = quit");
    }

    // Step 2: Allocate three height buffers.
    // The explicit scheme needs u(t-dt), u(t), and u(t+dt), so storing three images
    // is the most direct way to express the time stepping.
    HeightFieldTriple heights{device, grid_resolution};

    // Step 3: Build a clear kernel.
    // Resetting all height buffers to zero gives us a calm water surface.
    Kernel2D clear_kernel = [](ImageFloat image) noexcept {
        set_block_size(16u, 16u, 1u);
        image.write(dispatch_id().xy(), make_float4(0.0f));
    };
    auto clear_shader = device.compile(clear_kernel);

    auto clear_all = [&] {
        stream << clear_shader(heights.prev).dispatch(grid_resolution)
               << clear_shader(heights.curr).dispatch(grid_resolution)
               << clear_shader(heights.next).dispatch(grid_resolution)
               << synchronize();
        LUISA_INFO("Height field reset.");
    };
    clear_all();

    // Step 4: Define the wave solver.
    // A 5-point Laplacian is enough to show wave propagation and interference clearly.
    Kernel2D wave_step_kernel = [](ImageFloat prev, ImageFloat curr, ImageFloat next) noexcept {
        set_block_size(16u, 16u, 1u);
        auto cell = dispatch_id().xy();
        auto size = dispatch_size().xy();

        auto sample_curr = [&](Int2 offset) noexcept {
            auto p = make_int2(cell) + offset;
            p.x = clamp(p.x, 0, cast<int>(size.x) - 1);
            p.y = clamp(p.y, 0, cast<int>(size.y) - 1);
            return curr.read(make_uint2(cast<uint>(p.x), cast<uint>(p.y))).x;
        };

        auto center = curr.read(cell).x;
        auto laplacian = sample_curr(make_int2(-1, 0)) +
                         sample_curr(make_int2(1, 0)) +
                         sample_curr(make_int2(0, -1)) +
                         sample_curr(make_int2(0, 1)) -
                         4.0f * center;
        auto previous = prev.read(cell).x;
        auto candidate = 2.0f * center - previous + c2dt2 * laplacian;
        candidate *= damping;
        next.write(cell, make_float4(clamp(candidate, -2.0f, 2.0f), 0.0f, 0.0f, 1.0f));
    };
    auto wave_step_shader = device.compile(wave_step_kernel);

    // Step 5: Define a droplet kernel.
    // A Gaussian is smoother than a hard impulse, so it excites the grid without
    // producing blocky artifacts.
    Kernel2D droplet_kernel = [](ImageFloat curr, UInt2 center, Float strength) noexcept {
        set_block_size(16u, 16u, 1u);
        auto cell = dispatch_id().xy();
        auto d = make_float2(make_int2(cell) - make_int2(center));
        auto radius = 14.0f;
        auto gaussian = exp(-dot(d, d) / (radius * radius * 0.5f));
        auto updated = curr.read(cell).x + strength * gaussian;
        curr.write(cell, make_float4(updated, 0.0f, 0.0f, 1.0f));
    };
    auto droplet_shader = device.compile(droplet_kernel);

    // Step 6: Define a render kernel.
    // We derive highlights from the slope of the height field, which reads like
    // cheap caustics and makes the simulation easier to interpret visually.
    Kernel2D render_kernel = [](ImageFloat height, ImageFloat output, Float time) noexcept {
        set_block_size(16u, 16u, 1u);
        auto pixel = dispatch_id().xy();
        auto output_size = dispatch_size().xy();
        auto cell = pixel / display_scale;
        auto sim_size = output_size / display_scale;

        auto sample_height = [&](Int2 offset) noexcept {
            auto p = make_int2(cell) + offset;
            p.x = clamp(p.x, 0, cast<int>(sim_size.x) - 1);
            p.y = clamp(p.y, 0, cast<int>(sim_size.y) - 1);
            return height.read(make_uint2(cast<uint>(p.x), cast<uint>(p.y))).x;
        };

        auto h = sample_height(make_int2(0, 0));
        auto hx = sample_height(make_int2(1, 0)) - sample_height(make_int2(-1, 0));
        auto hy = sample_height(make_int2(0, 1)) - sample_height(make_int2(0, -1));
        auto slope = length(make_float2(hx, hy));
        auto depth = clamp(h * 0.35f + 0.5f, 0.0f, 1.0f);

        auto deep_water = make_float3(0.02f, 0.12f, 0.28f);
        auto shallow_water = make_float3(0.04f, 0.45f, 0.72f);
        auto caustic_tint = make_float3(0.60f, 0.88f, 1.00f);
        auto foam_tint = make_float3(0.90f, 0.96f, 1.00f);

        auto color = lerp(deep_water, shallow_water, depth);
        auto highlight = smoothstep(0.04f, 0.30f, slope);
        color = lerp(color, caustic_tint, highlight * 0.65f);

        auto crest = smoothstep(0.35f, 0.90f, abs(h));
        color = lerp(color, foam_tint, crest * 0.35f);

        auto shimmer = 0.5f + 0.5f * sin(time * 2.5f + cast<float>(pixel.x) * 0.03f + cast<float>(pixel.y) * 0.02f);
        color *= 0.92f + 0.08f * shimmer;
        output.write(pixel, make_float4(color, 1.0f));
    };
    auto render_shader = device.compile(render_kernel);

    auto apply_drop = [&](uint2 center, float strength) {
        stream << droplet_shader(heights.curr, center, strength).dispatch(grid_resolution);
    };

    auto simulate_steps = [&](uint count) {
        for (auto i = 0u; i < count; i++) {
            stream << wave_step_shader(heights.prev, heights.curr, heights.next).dispatch(grid_resolution);
            heights.advance();
        }
    };

    if (offline) {
        // Step 7A: Offline mode injects a few deterministic random droplets, advances
        // the simulation for 200 steps, shades the final frame, and saves a PNG.
        std::mt19937 rng{1337u};
        std::uniform_int_distribution<uint> dist_x{32u, grid_resolution.x - 33u};
        std::uniform_int_distribution<uint> dist_y{32u, grid_resolution.y - 33u};
        std::uniform_real_distribution<float> dist_strength{-0.85f, -0.35f};

        for (auto i = 0u; i < 5u; i++) {
            apply_drop(make_uint2(dist_x(rng), dist_y(rng)), dist_strength(rng));
        }
        simulate_steps(200u);

        auto output = device.create_image<float>(PixelStorage::BYTE4, display_resolution);
        stream << render_shader(heights.curr, output, 3.0f).dispatch(display_resolution);
        save_png(stream, output, display_resolution, "tutorial_05_wave_equation.png");
        return 0;
    }

    // Step 7B: Interactive mode wires mouse motion to droplet injection.
    Window window{"Tutorial 05 - 2D Wave Equation", display_resolution};
    Swapchain swapchain = device.create_swapchain(
        stream,
        SwapchainOption{
            .display = window.native_display(),
            .window = window.native_handle(),
            .size = display_resolution,
            .wants_vsync = true,
            .back_buffer_count = 3u,
        });
    auto display = device.create_image<float>(swapchain.backend_storage(), display_resolution);

    bool mouse_down = false;
    bool reset_requested = false;
    float2 cursor{0.0f};
    float2 last_cursor{-1.0f};

    window.set_mouse_callback([&](MouseButton button, Action action, float2 p) noexcept {
        if (button != MOUSE_BUTTON_LEFT) {
            return;
        }
        cursor = p;
        if (action == ACTION_PRESSED || action == ACTION_REPEATED) {
            mouse_down = true;
            last_cursor = make_float2(-1.0f);
        } else if (action == ACTION_RELEASED) {
            mouse_down = false;
            last_cursor = make_float2(-1.0f);
        }
    });
    window.set_cursor_position_callback([&](float2 p) noexcept {
        cursor = p;
    });
    window.set_key_callback([&](Key key, KeyModifiers, Action action) noexcept {
        if (action != ACTION_PRESSED) {
            return;
        }
        if (key == KEY_SPACE) {
            reset_requested = true;
        } else if (key == KEY_ESCAPE) {
            window.set_should_close();
        }
    });

    Clock clock;

    // Step 8: Run the interactive simulation.
    // Dragging creates a chain of droplets so the interaction feels continuous instead
    // of sampling only isolated mouse positions.
    while (!window.should_close()) {
        window.poll_events();

        if (reset_requested) {
            clear_all();
            reset_requested = false;
        }

        if (mouse_down) {
            auto current = make_float2(
                std::clamp(cursor.x / static_cast<float>(display_scale), 0.0f, static_cast<float>(grid_resolution.x - 1u)),
                std::clamp(cursor.y / static_cast<float>(display_scale), 0.0f, static_cast<float>(grid_resolution.y - 1u)));

            if (last_cursor.x < 0.0f) {
                apply_drop(make_uint2(static_cast<uint>(current.x), static_cast<uint>(current.y)), -0.65f);
            } else {
                auto previous = make_float2(
                    std::clamp(last_cursor.x / static_cast<float>(display_scale), 0.0f, static_cast<float>(grid_resolution.x - 1u)),
                    std::clamp(last_cursor.y / static_cast<float>(display_scale), 0.0f, static_cast<float>(grid_resolution.y - 1u)));
                auto delta = current - previous;
                auto segments = std::max(1, static_cast<int>(std::ceil(length(delta))));
                segments = std::min(segments, 16);
                for (auto i = 0; i <= segments; i++) {
                    auto t = static_cast<float>(i) / static_cast<float>(segments);
                    auto p = lerp(previous, current, t);
                    apply_drop(make_uint2(static_cast<uint>(p.x), static_cast<uint>(p.y)), -0.40f);
                }
            }
            last_cursor = cursor;
        } else {
            last_cursor = make_float2(-1.0f);
        }

        simulate_steps(2u);

        auto time = static_cast<float>(clock.toc() * 1e-3);
        stream << render_shader(heights.curr, display, time).dispatch(display_resolution)
               << swapchain.present(display);
    }

    stream << synchronize();
    return 0;
}
