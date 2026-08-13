// N-Body Gravitational Simulation
// Simulates gravitational interactions between thousands of particles representing
// stars or celestial bodies.
//
// Features demonstrated:
// - Particle-based physics simulation
// - Double buffering for position updates
// - Real-time 3D visualization
// - Softening parameter to prevent numerical singularities
// - Deterministic depth-resolved particle glow rasterization

#include <algorithm>
#include <filesystem>
#include <memory>
#include <optional>
#include <random>
#include <string_view>

#include "../common/nbody_render_plan.h"
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

struct ParticleProjection {
    int2 pixel;
    float distance;
    uint visible;
};

LUISA_STRUCT(ParticleProjection, pixel, distance, visible) {};

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
    auto force_offline = opts.offline;
    auto compare_path = opts.compare_path;
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
    static_assert(n_particles <= ref::NBodyWinnerEncoding::kMaxParticleCount);

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
    Buffer<uint> particle_winners = device.create_buffer<uint>(width * height);
    Buffer<ParticleProjection> particle_projections = device.create_buffer<ParticleProjection>(n_particles);

    // Positive floating-point distances preserve their ordering when viewed as
    // unsigned integers. The winner key replaces the low mantissa bits with the
    // particle index, so atomic-min is deterministic even when particles overlap.
    Callable encode_winner = [](Float distance, UInt particle_index) noexcept {
        return (as<uint>(distance) & ref::NBodyWinnerEncoding::kDepthMask) |
               particle_index;
    };

    Callable camera_space = [](Float3 position, Float rot_x, Float rot_y) noexcept {
        // Rotation around Y axis
        Var cos_y = cos(rot_y);
        Var sin_y = sin(rot_y);
        Var x1 = position.x * cos_y - position.z * sin_y;
        Var z1 = position.x * sin_y + position.z * cos_y;
        Var y1 = position.y;

        // Rotation around X axis
        Var cos_x = cos(rot_x);
        Var sin_x = sin(rot_x);
        Var y2 = y1 * cos_x - z1 * sin_x;
        Var z2 = y1 * sin_x + z1 * cos_x;

        // Perspective projection
        Var view_distance = 4.0f;
        Var distance = view_distance + z2;
        return compose(make_float2(x1, y2), distance);
    };

    Kernel1D clear_particle_winners = [](BufferUInt winners) noexcept {
        set_block_size(256u);
        winners.write(dispatch_x(), ref::NBodyWinnerEncoding::kInvalid);
    };
    auto clear_winners = device.compile(clear_particle_winners);

    // Rasterize each 5x5 footprint into a packed per-pixel winner. The only
    // contended operation is unsigned integer atomic-min; color is resolved in
    // a separate image-parallel pass after all winners are final.
    Kernel1D rasterize_particles = [&](BufferVar<Particle> particles, BufferVar<ParticleProjection> projections, BufferUInt winners, Float rot_x, Float rot_y, Float zoom, UInt2 image_size) noexcept {
        set_block_size(256u);
        Var idx = dispatch_x();
        Var particle = particles.read(idx);
        Var<ParticleProjection> projection{make_int2(0), 0.0f, 0u};
        Var camera = camera_space(particle.position, rot_x, rot_y);
        Var camera_xy = camera.get<0>();
        Var distance = camera.get<1>();
        Var finite_projection = !luisa::compute::isnan(distance) &
                                !luisa::compute::isinf(distance) &
                                !luisa::compute::isnan(camera_xy.x) &
                                !luisa::compute::isinf(camera_xy.x) &
                                !luisa::compute::isnan(camera_xy.y) &
                                !luisa::compute::isinf(camera_xy.y);

        // Skip particles behind the camera or with invalid projected state.
        $if ((distance > ref::NBodyWinnerEncoding::kMinimumVisibleDistance) & finite_projection) {
            Var scale = 1.5f / distance;
            Var screen_x = (camera_xy.x * scale) * cast<float>(image_size.x) * 0.5f + cast<float>(image_size.x) * 0.5f;
            Var screen_y = (camera_xy.y * scale) * cast<float>(image_size.y) * 0.5f + cast<float>(image_size.y) * 0.5f;
            Var finite_screen = !luisa::compute::isnan(screen_x) &
                                !luisa::compute::isinf(screen_x) &
                                !luisa::compute::isnan(screen_y) &
                                !luisa::compute::isinf(screen_y);
            Var center_in_bounds = (screen_x > -1.0f) &
                                   (screen_x < cast<float>(image_size.x)) &
                                   (screen_y > -1.0f) &
                                   (screen_y < cast<float>(image_size.y));
            $if (finite_screen & center_in_bounds) {
                Int2 ipos = make_int2(cast<int>(screen_x), cast<int>(screen_y));
                projection.pixel = ipos;
                projection.distance = distance;
                projection.visible = 1u;
                Var winner = encode_winner(distance, idx);
                for (int dy = -ref::kNBodyGlowRadius; dy <= ref::kNBodyGlowRadius; dy++) {
                    for (int dx = -ref::kNBodyGlowRadius; dx <= ref::kNBodyGlowRadius; dx++) {
                        Int2 offset = make_int2(Int(dx), Int(dy));
                        Int2 pixel = ipos + offset;
                        $if ((pixel.x >= 0) & (pixel.x < cast<int>(image_size.x)) &
                             (pixel.y >= 0) & (pixel.y < cast<int>(image_size.y))) {
                            Var pixel_index = cast<uint>(pixel.y) * image_size.x + cast<uint>(pixel.x);
                            winners.atomic(pixel_index).fetch_min(winner);
                        };
                    }
                }
            };
        };
        projections.write(idx, projection);
    };
    auto rasterize = device.compile(rasterize_particles);

    Kernel2D resolve_particles = [](BufferVar<ParticleProjection> projections, BufferUInt winners, ImageFloat image, UInt2 image_size) noexcept {
        set_block_size(16u, 16u, 1u);
        Var pixel = dispatch_id().xy();
        Var pixel_index = pixel.y * image_size.x + pixel.x;
        Var winner = winners.read(pixel_index);
        Float4 output = make_float4(0.02f, 0.02f, 0.05f, 1.0f);
        $if (winner != ref::NBodyWinnerEncoding::kInvalid) {
            Var idx = winner & ref::NBodyWinnerEncoding::kParticleIndexMask;
            Var projection = projections.read(idx);
            $if (projection.visible != 0u) {
                Int2 delta = make_int2(cast<int>(pixel.x), cast<int>(pixel.y)) - projection.pixel;
                Var depth_factor = 1.0f / (1.0f + projection.distance * 0.1f);
                Var r = 0.5f + 0.5f * sin(cast<float>(idx) * 0.1f);
                Var g = 0.5f + 0.5f * sin(cast<float>(idx) * 0.13f + 2.0f);
                Var b = 0.8f + 0.2f * sin(cast<float>(idx) * 0.07f + 4.0f);
                Var color = make_float3(r, g, b) * depth_factor;
                Var footprint_distance = sqrt(cast<float>(delta.x * delta.x + delta.y * delta.y));
                Var intensity = exp(-footprint_distance * 0.8f) * 0.9f;
                output = make_float4(color * intensity, 1.0f);
            };
        };
        image.write(pixel, output);
    };
    auto resolve = device.compile(resolve_particles);

    // Main simulation loop
    uint frame = 0u;

    if (force_offline) {
        static constexpr uint offline_frames = 100u;
        for (uint i = 0u; i < offline_frames; i++) {
            stream << clear_winners(particle_winners).dispatch(width * height)
                   << nbody_shader(particles_read, particles_write).dispatch(n_particles);
            std::swap(particles_read, particles_write);
            stream << rasterize(particles_read, particle_projections, particle_winners, rot_x, rot_y, zoom, make_uint2(width, height)).dispatch(n_particles)
                   << resolve(particle_projections, particle_winners, display, make_uint2(width, height)).dispatch(width, height);
            frame++;
        }
        luisa::vector<uint8_t> host_image(width * height * 4u);
        stream << display.copy_to(luisa::span{host_image}) << synchronize();
        static constexpr uint feature_tile_size = 32u;
        luisa::vector<uint8_t> feature_tiles(
            (width / feature_tile_size) * (height / feature_tile_size), 0u);
        size_t bright_pixel_count = 0u;
        size_t active_pixel_count = 0u;
        for (auto i = 0u; i < width * height; i++) {
            auto offset = static_cast<size_t>(i) * 4u;
            auto peak = std::max({host_image[offset + 0u],
                                  host_image[offset + 1u],
                                  host_image[offset + 2u]});
            if (peak >= 32u) { active_pixel_count++; }
            if (peak >= 128u) {
                bright_pixel_count++;
                auto x = i % width;
                auto y = i / width;
                auto tile_x = x / feature_tile_size;
                auto tile_y = y / feature_tile_size;
                feature_tiles[tile_y * (width / feature_tile_size) + tile_x] = 1u;
            }
        }
        auto feature_tile_count = static_cast<size_t>(std::count(feature_tiles.cbegin(), feature_tiles.cend(), uint8_t{1u}));
        auto scene_is_valid = active_pixel_count >= 3000u && active_pixel_count <= 20000u &&
                              bright_pixel_count >= 200u && bright_pixel_count <= 2000u &&
                              feature_tile_count >= 20u && feature_tile_count <= 100u;
        if (!scene_is_valid) {
            LUISA_ERROR(
                "N-body output failed feature checks: {} active pixels (expected 3000-20000), "
                "{} bright pixels (expected 200-2000), {} occupied feature tiles (expected 20-100).",
                active_pixel_count, bright_pixel_count, feature_tile_count);
            return 1;
        }
        if (stbi_write_png("test_nbody_simulation.png", width, height, 4, host_image.data(), 0) == 0) {
            LUISA_ERROR("Failed to write test_nbody_simulation.png.");
            return 1;
        }
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

            // Reset the deterministic per-pixel arbitration buffer.
            stream << clear_winners(particle_winners).dispatch(width * height);

            // Update physics
            stream << nbody_shader(particles_read, particles_write).dispatch(n_particles);
            std::swap(particles_read, particles_write);

            // Rasterize winners, then resolve each pixel exactly once.
            stream << rasterize(particles_read, particle_projections, particle_winners, rot_x, rot_y, zoom, make_uint2(width, height)).dispatch(n_particles)
                   << resolve(particle_projections, particle_winners, display, make_uint2(width, height)).dispatch(width, height)
                   << swap_chain->present(display);

            frame++;
        }
#endif
    }

    stream << synchronize();
}
