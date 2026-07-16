// N-Body Gravitational Simulation
// Simulates gravitational interactions between thousands of particles representing
// stars or celestial bodies.
//
// Features demonstrated:
// - Particle-based physics simulation
// - Double buffering for position updates
// - Real-time 3D visualization
// - Softening parameter to prevent numerical singularities

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

// Particle structure with position, velocity, and mass
struct Particle {
    float3 position;
    float3 velocity;
    float mass;
    float pad[3];// Padding for alignment
};

LUISA_STRUCT(Particle, position, velocity, mass, pad) {};

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
    LUISA_INFO("N-Body Gravitational Simulation");
    LUISA_INFO("Controls: Mouse drag = Rotate, Scroll/+/- = Zoom, R = Reset, ESC = Quit");

    // Simulation parameters - adjusted for visible results
    static constexpr uint n_particles = 2048u;
    static constexpr uint tile_size = 256u;
    static constexpr float dt = 0.0005f;
    static constexpr float softening = 0.05f;// Prevents division by zero
    static constexpr float G = 0.5f;         // Gravitational constant (scaled down)

    // Create particle buffers (double buffering)
    Buffer<Particle> particles_read = device.create_buffer<Particle>(n_particles);
    Buffer<Particle> particles_write = device.create_buffer<Particle>(n_particles);

    // Initialize particles with random positions in a galaxy-like disk
    std::mt19937 rng{force_offline ? 42u : std::random_device{}()};
    std::uniform_real_distribution<float> dist_radius{0.2f, 1.5f};
    std::uniform_real_distribution<float> dist_angle{0.0f, 2.0f * 3.14159f};
    std::uniform_real_distribution<float> dist_mass{0.5f, 2.0f};

    luisa::vector<Particle> host_particles(n_particles);
    for (uint i = 0u; i < n_particles; i++) {
        float radius = dist_radius(rng);
        float angle = dist_angle(rng);
        float height = (rng() / float(UINT32_MAX) - 0.5f) * 0.15f;

        // Galaxy-like initial configuration
        float3 pos{
            radius * cosf(angle),
            height,
            radius * sinf(angle)};

        // Tangential velocity for orbital motion (balanced with central mass)
        float central_mass = 500.0f;// Effective central mass
        float orbital_speed = sqrtf(G * central_mass / radius);
        float3 vel{
            -orbital_speed * sinf(angle),
            0.0f,
            orbital_speed * cosf(angle)};

        // Add small random perturbation
        vel.x += (rng() / float(UINT32_MAX) - 0.5f) * 0.3f;
        vel.y += (rng() / float(UINT32_MAX) - 0.5f) * 0.1f;
        vel.z += (rng() / float(UINT32_MAX) - 0.5f) * 0.3f;

        host_particles[i] = Particle{
            .position = pos,
            .velocity = vel,
            .mass = dist_mass(rng),
            .pad = {0.0f, 0.0f, 0.0f}};
    }

    Stream stream = device.create_stream(force_offline ? StreamTag::COMPUTE : StreamTag::GRAPHICS);
    stream << particles_read.copy_from(luisa::span{host_particles}) << synchronize();

    // N-Body computation kernel
    Kernel1D nbody_kernel = [&](BufferVar<Particle> read_buf, BufferVar<Particle> write_buf) noexcept {
        set_block_size(tile_size);
        Var idx = dispatch_id().x;
        Var p = read_buf.read(idx);

        // Accumulate gravitational forces
        Var force = make_float3(0.0f);

        // Compute forces from all other particles
        $for (j, n_particles) {
            // Skip self-interaction
            $if (j != idx) {
                Var other = read_buf.read(j);
                Var r = other.position - p.position;
                Var dist_sq = dot(r, r) + softening * softening;
                Var dist = sqrt(dist_sq);
                Var f = G * p.mass * other.mass / dist_sq;
                force += f * r / dist;
            };
        };

        // Update velocity and position using Euler integration
        Var new_vel = p.velocity + force / p.mass * dt;
        Var new_pos = p.position + new_vel * dt;

        // Damping to prevent explosion
        new_vel = new_vel * 0.999f;

        write_buf->write(idx, def<Particle>(new_pos, new_vel, p.mass));
    };

    auto nbody_shader = device.compile(nbody_kernel);

    // Setup window and swapchain
    static constexpr uint width = 1024u;
    static constexpr uint height = 1024u;

#if ENABLE_DISPLAY
    std::unique_ptr<Window> window;
    if (!force_offline) {
        window = std::make_unique<Window>("N-Body Simulation", make_uint2(width, height));
    }
#endif

    // Mouse/keyboard state
    float rot_x = 0.3f;
    float rot_y = 0.0f;
    float zoom = 1.0f;
    bool mouse_down = false;
    float2 last_mouse_pos{0.0f, 0.0f};

#if ENABLE_DISPLAY
    if (!force_offline) {
        window->set_mouse_callback([&mouse_down, &last_mouse_pos](MouseButton, Action a, float2 p) noexcept {
            if (a == Action::ACTION_PRESSED) {
                mouse_down = true;
                last_mouse_pos = p;
            } else if (a == Action::ACTION_RELEASED) {
                mouse_down = false;
            }
        });

        window->set_cursor_position_callback([&mouse_down, &last_mouse_pos, &rot_x, &rot_y](float2 p) noexcept {
            if (mouse_down) {
                float dx = p.x - last_mouse_pos.x;
                float dy = p.y - last_mouse_pos.y;
                rot_y += dx * 0.005f;
                rot_x += dy * 0.005f;
                rot_x = clamp(rot_x, -1.5f, 1.5f);
                last_mouse_pos = p;
            }
        });

        window->set_scroll_callback([&zoom](float2 offset) noexcept {
            // Invert scroll direction (scroll up = zoom in)
            zoom *= (1.0f - offset.y * 0.1f);
            zoom = clamp(zoom, 0.1f, 5.0f);
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
                .wants_hdr = false,
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
        return device.create_image<float>(PixelStorage::BYTE4, make_uint2(width, height));
    }();

    // Clear display kernel
    Kernel2D clear_kernel = [](ImageFloat image) noexcept {
        set_block_size(16, 16, 1);
        image.write(dispatch_id().xy(), make_float4(0.02f, 0.02f, 0.05f, 1.0f));
    };
    auto clear = device.compile(clear_kernel);

    // Particle render kernel - draw particles as points
    Kernel1D render_particles = [&](BufferVar<Particle> particles, ImageFloat image, Float rot_x, Float rot_y, Float zoom, UInt2 image_size) noexcept {
        set_block_size(256);
        Var idx = dispatch_id().x;
        Var p = particles.read(idx);

        // Rotation around Y axis
        Var cos_y = cos(rot_y);
        Var sin_y = sin(rot_y);
        Var x1 = p.position.x * cos_y - p.position.z * sin_y;
        Var z1 = p.position.x * sin_y + p.position.z * cos_y;
        Var y1 = p.position.y;

        // Rotation around X axis
        Var cos_x = cos(rot_x);
        Var sin_x = sin(rot_x);
        Var y2 = y1 * cos_x - z1 * sin_x;
        Var z2 = y1 * sin_x + z1 * cos_x;

        // Perspective projection
        Var view_distance = 4.0f;
        Var distance = view_distance + z2;

        // Skip particles behind camera
        $if (distance > 0.1f) {
            Var scale = 1.5f / distance;
            Var screen_x = (x1 * scale) * cast<float>(image_size.x) * 0.5f + cast<float>(image_size.x) * 0.5f;
            Var screen_y = (y2 * scale) * cast<float>(image_size.y) * 0.5f + cast<float>(image_size.y) * 0.5f;

            // Check bounds
            Int2 ipos = make_int2(cast<int>(screen_x), cast<int>(screen_y));

            $if ((ipos.x >= 0) & (ipos.x < cast<int>(image_size.x)) &
                 (ipos.y >= 0) & (ipos.y < cast<int>(image_size.y))) {

                // Color based on particle index and depth
                Var depth_factor = 1.0f / (1.0f + distance * 0.1f);
                Var r = 0.5f + 0.5f * sin(cast<float>(idx) * 0.1f);
                Var g = 0.5f + 0.5f * sin(cast<float>(idx) * 0.13f + 2.0f);
                Var b = 0.8f + 0.2f * sin(cast<float>(idx) * 0.07f + 4.0f);

                Var color = make_float3(r, g, b) * depth_factor;

                // Draw a larger 5x5 glow around the particle for better visibility
                for (int dy = -2; dy <= 2; dy++) {
                    for (int dx = -2; dx <= 2; dx++) {
                        Int2 offset = make_int2(Int(dx), Int(dy));
                        Int2 p = ipos + offset;
                        $if ((p.x >= 0) & (p.x < cast<int>(image_size.x)) &
                             (p.y >= 0) & (p.y < cast<int>(image_size.y))) {
                            Var d = sqrt(cast<float>(dx * dx + dy * dy));
                            Var intensity = exp(-d * 0.8f) * 0.9f;
                            image.write(make_uint2(cast<uint>(p.x), cast<uint>(p.y)), make_float4(color * intensity, 1.0f));
                        };
                    }
                }
            };
        };
    };

    auto render_shader = device.compile(render_particles);

    // Main simulation loop
    uint frame = 0u;

    if (force_offline) {
        static constexpr uint offline_frames = 100u;
        for (uint i = 0u; i < offline_frames; i++) {
            stream << clear(display).dispatch(width, height)
                   << nbody_shader(particles_read, particles_write).dispatch(n_particles);
            std::swap(particles_read, particles_write);
            stream << render_shader(particles_read, display, rot_x, rot_y, zoom, make_uint2(width, height)).dispatch(n_particles);
            frame++;
        }
        luisa::vector<uint8_t> host_image(width * height * 4u);
        stream << display.copy_to(luisa::span{host_image}) << synchronize();
        stbi_write_png("test_nbody_simulation.png", width, height, 4, host_image.data(), 0);
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

            if (window->is_key_down(KEY_ESCAPE)) {
                break;
            }
            if (window->is_key_down(KEY_R)) {
                rot_x = 0.3f;
                rot_y = 0.0f;
                zoom = 1.0f;
            }
            // Keyboard zoom as alternative to scroll
            if (window->is_key_down(KEY_EQUAL) || window->is_key_down(KEY_KP_ADD)) {
                zoom *= 1.02f;
                zoom = min(zoom, 5.0f);
            }
            if (window->is_key_down(KEY_MINUS) || window->is_key_down(KEY_KP_SUBTRACT)) {
                zoom *= 0.98f;
                zoom = max(zoom, 0.1f);
            }

            // Clear display
            stream << clear(display).dispatch(width, height);

            // Update physics
            stream << nbody_shader(particles_read, particles_write).dispatch(n_particles);
            std::swap(particles_read, particles_write);

            // Render particles
            stream << render_shader(particles_read, display, rot_x, rot_y, zoom, make_uint2(width, height)).dispatch(n_particles)
                   << swap_chain->present(display);

            frame++;
        }
#endif
    }

    stream << synchronize();
}
