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

#include <array>
#include <cmath>
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

namespace {

[[nodiscard]] bool fire_particle_is_finite(const FireParticle &p) noexcept {
    return std::isfinite(p.position.x) &&
           std::isfinite(p.position.y) &&
           std::isfinite(p.position.z) &&
           std::isfinite(p.lifetime) &&
           std::isfinite(p.velocity.x) &&
           std::isfinite(p.velocity.y) &&
           std::isfinite(p.velocity.z) &&
           std::isfinite(p.temperature) &&
           std::isfinite(p.size) &&
           std::isfinite(p.pad[0]) &&
           std::isfinite(p.pad[1]) &&
           std::isfinite(p.pad[2]);
}

[[nodiscard]] bool validate_offline_particle(
    uint index,
    const FireParticle &initial,
    const FireParticle &after_first_frame,
    const FireParticle &final,
    float dt,
    float gravity) noexcept {
    static constexpr float tolerance = 1e-4f;
    auto close = [](float a, float b) noexcept {
        return std::abs(a - b) <= tolerance;
    };
    if (!fire_particle_is_finite(after_first_frame) ||
        !fire_particle_is_finite(final)) {
        LUISA_ERROR("Offline fire particle {} contains non-finite state.", index);
        return false;
    }
    if (initial.lifetime <= dt) {
        LUISA_ERROR(
            "Offline fire particle {} is not a valid first-frame integration probe: "
            "initial lifetime {} must exceed dt {}.",
            index, initial.lifetime, dt);
        return false;
    }

    auto first_frame_is_valid =
        close(after_first_frame.lifetime, initial.lifetime - dt) &&
        close(after_first_frame.temperature, initial.temperature - dt * 0.3f) &&
        close(after_first_frame.velocity.y, initial.velocity.y + gravity * dt) &&
        close(after_first_frame.position.x, initial.position.x + after_first_frame.velocity.x * dt) &&
        close(after_first_frame.position.y, initial.position.y + after_first_frame.velocity.y * dt) &&
        close(after_first_frame.position.z, initial.position.z + after_first_frame.velocity.z * dt) &&
        std::abs(after_first_frame.velocity.x - initial.velocity.x) <= 0.5f * dt + tolerance &&
        std::abs(after_first_frame.velocity.z - initial.velocity.z) <= 0.5f * dt + tolerance &&
        close(after_first_frame.size, initial.size) &&
        after_first_frame.pad[0] == 0.0f &&
        after_first_frame.pad[1] == 0.0f &&
        after_first_frame.pad[2] == 0.0f;
    if (!first_frame_is_valid) {
        LUISA_ERROR("Offline fire particle {} failed first-frame integration checks.", index);
        return false;
    }

    auto final_position_is_plausible =
        std::abs(final.position.x) <= 16.0f &&
        std::abs(final.position.y) <= 16.0f &&
        std::abs(final.position.z) <= 16.0f;
    auto final_velocity_is_plausible =
        std::abs(final.velocity.x) <= 8.0f &&
        std::abs(final.velocity.y) <= 8.0f &&
        std::abs(final.velocity.z) <= 8.0f;
    auto final_scalars_are_plausible =
        final.lifetime >= -dt - tolerance && final.lifetime <= 3.0f + tolerance &&
        final.temperature >= -tolerance && final.temperature <= 1.0f + tolerance &&
        final.size >= 0.02f - tolerance && final.size <= 0.07f + tolerance;
    auto padding_is_preserved =
        final.pad[0] == 0.0f &&
        final.pad[1] == 0.0f &&
        final.pad[2] == 0.0f;
    auto position_delta = final.position - after_first_frame.position;
    auto position_evolved = dot(position_delta, position_delta) > 1e-4f;
    auto lifetime_or_temperature_evolved =
        std::abs(final.lifetime - after_first_frame.lifetime) > 1e-3f ||
        std::abs(final.temperature - after_first_frame.temperature) > 1e-3f;
    if (!final_position_is_plausible ||
        !final_velocity_is_plausible ||
        !final_scalars_are_plausible ||
        !padding_is_preserved ||
        !position_evolved ||
        !lifetime_or_temperature_evolved) {
        LUISA_ERROR(
            "Offline fire particle {} failed final-state checks: "
            "position=({}, {}, {}), velocity=({}, {}, {}), lifetime={}, temperature={}, size={}.",
            index,
            final.position.x, final.position.y, final.position.z,
            final.velocity.x, final.velocity.y, final.velocity.z,
            final.lifetime, final.temperature, final.size);
        return false;
    }
    return true;
}

}// namespace

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
    LUISA_INFO("Fire and Smoke Particle System");
    LUISA_INFO("Controls: SPACE = Toggle wind, R = Reset, ESC = Quit");

    // Simulation parameters
    static constexpr uint n_particles = 65536u;
    static constexpr uint width = 1024u;
    static constexpr uint height = 1024u;
    static constexpr float dt = 0.016f;
    static constexpr float gravity = -2.0f;
    // Rendering every particle in every pixel is prohibitively expensive. The
    // visualization intentionally samples one particle out of every 256; the
    // offline state probes below independently cover both sampled and skipped
    // particle indices.
    static constexpr uint render_particle_stride = 256u;
    static_assert(n_particles % render_particle_stride == 0u);
    static constexpr uint rendered_particle_count = n_particles / render_particle_stride;

    // Create particle buffers
    Buffer<FireParticle> particles = device.create_buffer<FireParticle>(n_particles);

    Stream stream = device.create_stream(force_offline ? StreamTag::COMPUTE : StreamTag::GRAPHICS);

    // Initialize particles
    std::mt19937 rng{force_offline ? 42u : std::random_device{}()};
    luisa::vector<FireParticle> host_particles(n_particles);
    auto next_unit_float = [&rng] noexcept {
        return static_cast<float>(rng()) / static_cast<float>(UINT32_MAX);
    };

    for (uint i = 0u; i < n_particles; i++) {
        // Keep the legacy draw order, including z before x for velocity. The
        // old make_float3 call relied on compiler-selected argument evaluation
        // order; naming every draw makes the offline seed portable without
        // changing the checked-in reference image.
        auto angle_sample = next_unit_float();
        auto radius_sample = next_unit_float();
        auto speed_sample = next_unit_float();
        auto height_sample = next_unit_float();
        auto lifetime_sample = next_unit_float();
        auto velocity_z_sample = next_unit_float();
        auto velocity_x_sample = next_unit_float();
        auto size_sample = next_unit_float();
        auto angle = angle_sample * 2.0f * 3.14159f;
        auto radius = radius_sample * 0.1f;
        auto speed = speed_sample * 2.0f + 1.0f;

        host_particles[i] = FireParticle{
            .position = make_float3(
                radius * cosf(angle),
                height_sample * 0.5f,
                radius * sinf(angle)),
            .lifetime = lifetime_sample * 3.0f,
            .velocity = make_float3(
                (velocity_x_sample - 0.5f) * 0.5f,
                speed,
                (velocity_z_sample - 0.5f) * 0.5f),
            .temperature = 1.0f,
            .size = size_sample * 0.05f + 0.02f,
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

        // Draw the documented 1/256 subset for performance.
        for (uint sample_index = 0u; sample_index < rendered_particle_count; sample_index++) {
            auto i = sample_index * render_particle_stride;
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
        static constexpr std::array<uint, 4u> probe_indices{
            0u, 1u,
            render_particle_stride,
            render_particle_stride + 1u};
        std::array<FireParticle, probe_indices.size()> after_first_frame_particles{};
        std::array<FireParticle, probe_indices.size()> final_particles{};
        for (uint frame = 0u; frame < offline_frames; frame++) {
            float time = static_cast<float>(frame) * dt;
            stream << update_shader(particles, time, wind_strength).dispatch(n_particles)
                   << render_shader(particles, display, time).dispatch(width, height);
            if (frame == 0u) {
                for (size_t probe = 0u; probe < probe_indices.size(); probe++) {
                    stream << particles.view(probe_indices[probe], 1u).copy_to(luisa::span{&after_first_frame_particles[probe], 1u});
                }
            }
        }
        luisa::vector<uint8_t> host_image(width * height * 4u);
        stream << display.copy_to(luisa::span{host_image});
        for (size_t probe = 0u; probe < probe_indices.size(); probe++) {
            stream << particles.view(probe_indices[probe], 1u).copy_to(luisa::span{&final_particles[probe], 1u});
        }
        stream << synchronize();
        if (stbi_write_png("test_fire_simulation.png", width, height, 4, host_image.data(), 0) == 0) {
            LUISA_ERROR("Failed to write test_fire_simulation.png.");
            return 1;
        }
        for (size_t probe = 0u; probe < probe_indices.size(); probe++) {
            auto index = probe_indices[probe];
            if (!validate_offline_particle(
                    index,
                    host_particles[index],
                    after_first_frame_particles[probe],
                    final_particles[probe],
                    dt, gravity)) {
                return 1;
            }
        }
        LUISA_INFO(
            "Offline fire state validation: PASSED (rendered indices {}, {}; "
            "non-rendered indices {}, {}).",
            probe_indices[0], probe_indices[2],
            probe_indices[1], probe_indices[3]);
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
