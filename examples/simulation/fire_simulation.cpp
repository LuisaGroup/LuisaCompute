// Fire and Smoke Particle System Simulation
// Simulates a fire effect using thousands of particles with physics-based motion,
// temperature-based color gradients, and procedural turbulence.
//
// Features demonstrated:
// - Particle system simulation with birth/death cycles
// - Procedural noise for turbulent motion
// - Temperature-based color gradients (black -> red -> orange -> yellow -> white)
// - Additive blending for glow effects
// - Interactive wind influence

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

// Particle structure
struct FireParticle {
    float3 position;  // XYZ position
    float lifetime;   // Remaining lifetime (0 = dead)
    float3 velocity;  // Velocity vector
    float temperature;// Temperature (affects color, 0-1 range)
    float size;       // Particle size
    float pad[3];     // Padding for alignment
};

LUISA_STRUCT(FireParticle, position, lifetime, velocity, temperature, size, pad) {};

int main(int argc, char *argv[]) {

    Context context{argv[0]};
    if (argc <= 1) {
        LUISA_INFO("Usage: {} <backend> [--offline] [-c <reference.png>]. <backend>: cuda, dx, cpu, metal", argv[0]);
        exit(1);
    }
    bool force_offline = false;
    std::optional<std::filesystem::path> compare_path;
    for (int i = 2; i < argc; i++) {
        if (std::string_view{argv[i]} == "--offline") {
            force_offline = true;
        } else if ((std::string_view{argv[i]} == "--compare" || std::string_view{argv[i]} == "-c") && i + 1 < argc) {
            compare_path = std::filesystem::path{argv[++i]};
            force_offline = true;
            force_offline = true;
        }
    }
#if !ENABLE_DISPLAY
    if (!force_offline) {
        LUISA_ERROR("GUI support is disabled. Use --offline.");
    }
#endif
    Device device = context.create_device(argv[1]);
    LUISA_INFO("Fire and Smoke Particle System");
    LUISA_INFO("Controls: SPACE = Toggle wind, R = Reset, ESC = Quit");

    // Simulation parameters
    static constexpr uint n_particles = 65536u;
    static constexpr uint width = 1024u;
    static constexpr uint height = 1024u;
    static constexpr float dt = 0.016f;
    static constexpr float gravity = -2.0f;

    // Create particle buffers
    Buffer<FireParticle> particles = device.create_buffer<FireParticle>(n_particles);

    Stream stream = device.create_stream(force_offline ? StreamTag::COMPUTE : StreamTag::GRAPHICS);

    // Initialize particles
    std::mt19937 rng{force_offline ? 42u : std::random_device{}()};
    luisa::vector<FireParticle> host_particles(n_particles);

    for (uint i = 0u; i < n_particles; i++) {
        float angle = (rng() / float(UINT32_MAX)) * 2.0f * 3.14159f;
        float radius = (rng() / float(UINT32_MAX)) * 0.1f;
        float speed = (rng() / float(UINT32_MAX)) * 2.0f + 1.0f;

        host_particles[i] = FireParticle{
            .position = make_float3(
                radius * cosf(angle),
                (rng() / float(UINT32_MAX)) * 0.5f,
                radius * sinf(angle)),
            .lifetime = (rng() / float(UINT32_MAX)) * 3.0f,
            .velocity = make_float3(
                (rng() / float(UINT32_MAX) - 0.5f) * 0.5f,
                speed,
                (rng() / float(UINT32_MAX) - 0.5f) * 0.5f),
            .temperature = 1.0f,
            .size = (rng() / float(UINT32_MAX)) * 0.05f + 0.02f,
            .pad = {0.0f, 0.0f, 0.0f}};
    }
    stream << particles.copy_from(luisa::span{host_particles}) << synchronize();

    // Particle update kernel
    Kernel1D update_kernel = [&](BufferVar<FireParticle> particle_buf, Float time, Float wind_strength) noexcept {
        set_block_size(256);
        Var idx = dispatch_id().x;
        Var p = particle_buf.read(idx);

        // Only update alive particles
        $if (p.lifetime > 0.0f) {
            // Apply physics
            p.velocity.y += gravity * dt;

            // Simple turbulence using sine waves
            Var turbulence_x = sin(time * 3.0f + p.position.y * 5.0f + cast<float>(idx) * 0.01f) * 0.5f;
            Var turbulence_z = cos(time * 2.5f + p.position.y * 4.0f + cast<float>(idx) * 0.01f) * 0.5f;
            p.velocity.x += turbulence_x * dt;
            p.velocity.z += turbulence_z * dt;

            // Wind effect
            p.velocity.x += wind_strength * dt;

            // Update position
            p.position = p.position + p.velocity * dt;

            // Cool down over time
            p.temperature -= dt * 0.3f;
            p.temperature = max(p.temperature, 0.0f);

            // Decrease lifetime
            p.lifetime -= dt;
        }
        $else {
            // Respawn dead particles at the source
            Var seed = idx + cast<uint>(time * 1000.0f);
            Var seed_f = cast<float>(seed);
            Var angle = (seed_f * 0.01f - floor(seed_f * 0.01f)) * 2.0f * 3.14159f;
            Var radius = (seed_f * 0.0013f - floor(seed_f * 0.0013f)) * 0.1f;
            Var speed = (seed_f * 0.0027f - floor(seed_f * 0.0027f)) * 2.0f + 1.0f;

            p.position = make_float3(
                radius * cos(angle),
                (seed_f * 0.0037f - floor(seed_f * 0.0037f)) * 0.2f,
                radius * sin(angle));
            p.velocity = make_float3(
                ((seed_f * 0.0051f - floor(seed_f * 0.0051f)) - 0.5f) * 0.5f,
                speed,
                ((seed_f * 0.0061f - floor(seed_f * 0.0061f)) - 0.5f) * 0.5f);
            p.lifetime = (seed_f * 0.0043f - floor(seed_f * 0.0043f)) * 2.0f + 1.0f;
            p.temperature = 1.0f;
            p.size = (seed_f * 0.0079f - floor(seed_f * 0.0079f)) * 0.05f + 0.02f;
        };

        particle_buf.write(idx, p);
    };

    auto update_shader = device.compile(update_kernel);

    // Rendering kernel with temperature-based coloring
    Kernel2D render_kernel = [&](BufferVar<FireParticle> particle_buf, ImageFloat image, Float time) noexcept {
        set_block_size(16, 16, 1);
        Var uv = dispatch_id().xy();
        Var size = dispatch_size().xy();

        // Clear with dark background
        image.write(uv, make_float4(0.02f, 0.02f, 0.03f, 1.0f));

        // Accumulate particle contributions
        Var color = make_float3(0.0f);

        // Process particles in chunks for performance
        for (uint i = 0u; i < n_particles; i += 256u) {
            Var p = particle_buf.read(i);

            // Skip dead particles
            $if (p.lifetime > 0.0f) {
                // Project to screen space
                Var screen_x = (p.position.x * 0.5f + 0.5f) * cast<float>(size.x);
                Var screen_y = (1.0f - (p.position.y * 0.5f + 0.5f)) * cast<float>(size.y);

                // Distance from pixel to particle center
                Var dx = cast<float>(uv.x) - screen_x;
                Var dy = cast<float>(uv.y) - screen_y;
                Var dist_sq = dx * dx + dy * dy;
                Var particle_radius = p.size * 500.0f;

                // Gaussian intensity falloff
                Var intensity = exp(-dist_sq / (particle_radius * particle_radius));

                // Temperature-based color gradient
                // Cold (0.0) -> Warm (0.5) -> Hot (1.0)
                Var temp = p.temperature;
                Var particle_color = make_float3(0.0f);

                // Black to red (0.0 - 0.25)
                $if (temp < 0.25f) {
                    Var t = temp * 4.0f;
                    particle_color = make_float3(t, 0.0f, 0.0f);
                }
                $elif (temp < 0.5f) {
                    // Red to orange (0.25 - 0.5)
                    Var t = (temp - 0.25f) * 4.0f;
                    particle_color = make_float3(1.0f, t, 0.0f);
                }
                $elif (temp < 0.75f) {
                    // Orange to yellow (0.5 - 0.75)
                    Var t = (temp - 0.5f) * 4.0f;
                    particle_color = make_float3(1.0f, 1.0f, t);
                }
                $else {
                    // Yellow to white (0.75 - 1.0)
                    Var t = (temp - 0.75f) * 4.0f;
                    particle_color = make_float3(1.0f, 1.0f, 1.0f);
                };

                // Additive blending
                color += particle_color * intensity * 0.5f;
            };
        }

        // Tone mapping and output
        color = min(color, 1.0f);
        // Add slight glow
        color = color + make_float3(0.02f, 0.01f, 0.0f);
        image.write(uv, make_float4(color, 1.0f));
    };

    auto render_shader = device.compile(render_kernel);

    // Setup window and swapchain
#if ENABLE_DISPLAY
    std::unique_ptr<Window> window;
    std::optional<Swapchain> swap_chain;
    if (!force_offline) {
        window = std::make_unique<Window>("Fire Simulation", make_uint2(width, height));
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

    // Main simulation loop
    Clock clock;
    float wind_strength = 0.0f;
    bool wind_enabled = false;

    if (force_offline) {
        static constexpr uint offline_frames = 200u;
        for (uint frame = 0u; frame < offline_frames; frame++) {
            float time = static_cast<float>(frame) * dt;
            stream << update_shader(particles, time, wind_strength).dispatch(n_particles)
                   << render_shader(particles, display, time).dispatch(width, height);
        }
        luisa::vector<uint8_t> host_image(width * height * 4u);
        stream << display.copy_to(luisa::span{host_image}) << synchronize();
        stbi_write_png("test_fire_simulation.png", width, height, 4, host_image.data(), 0);
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
            window->poll_events();

            // Handle input
            if (window->is_key_down(KEY_ESCAPE)) {
                break;
            }
            if (window->is_key_down(KEY_SPACE)) {
                wind_enabled = !wind_enabled;
                wind_strength = wind_enabled ? 1.0f : 0.0f;
            }
            if (window->is_key_down(KEY_R)) {
                // Reset particles
                stream << particles.copy_from(luisa::span{host_particles}) << synchronize();
            }

            float time = static_cast<float>(clock.toc() * 1e-3);

            // Update particles
            stream << update_shader(particles, time, wind_strength).dispatch(n_particles);

            // Render
            stream << render_shader(particles, display, time).dispatch(width, height)
                   << swap_chain->present(display);
        }
#endif
    }

    stream << synchronize();
}
