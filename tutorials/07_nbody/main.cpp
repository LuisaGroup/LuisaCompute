/*
 * Tutorial 07: N-Body Galaxy Simulation
 *
 * This tutorial teaches how to:
 * - Step 1: Define a GPU-friendly particle struct and register it with LUISA_STRUCT.
 * - Step 2: Initialize a rotating galaxy disk on the CPU and upload it to GPU buffers.
 * - Step 3: Simulate gravitational interaction with a 1D LuisaCompute kernel.
 * - Step 4: Render the particle system with perspective projection and Gaussian splats.
 * - Step 5: Present frames in a window, or run headless with --offline and save a PNG.
 *
 * Usage:
 *   ./tutorial_07_nbody <backend> [--offline]
 *
 * Example:
 *   ./tutorial_07_nbody metal
 *   ./tutorial_07_nbody metal --offline
 */

#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <numbers>

#include <luisa/luisa-compute.h>
#include <luisa/dsl/sugar.h>
#include <luisa/core/clock.h>
#include <luisa/core/logging.h>
#include <stb/stb_image_write.h>

#ifdef LUISA_ENABLE_GUI
#include <luisa/gui/window.h>
#endif

using namespace luisa;
using namespace luisa::compute;

struct alignas(16) Particle {
    float3 position;
    float mass;
    float3 velocity;
    float pad;
};

LUISA_STRUCT(Particle, position, mass, velocity, pad) {};

namespace {

static constexpr uint particle_count = 4096u;
static constexpr uint tile_size = 64u;
static constexpr uint2 resolution = make_uint2(1024u, 1024u);
static constexpr float softening = 0.1f;
static constexpr float gravity_constant = 0.01f;
static constexpr float simulation_dt = 0.01f;

[[nodiscard]] float host_random01(uint &state) noexcept {
    state = 1664525u * state + 1013904223u;
    return static_cast<float>(state & 0x00ffffffu) / static_cast<float>(0x01000000u);
}

[[nodiscard]] float3 camera_position(float yaw, float pitch, float distance) noexcept {
    auto cp = std::cos(pitch);
    return make_float3(
        distance * cp * std::sin(yaw),
        distance * std::sin(pitch),
        distance * cp * std::cos(yaw));
}

}// namespace

int main(int argc, char *argv[]) {

    log_level_verbose();

    luisa::string backend;
    auto offline = false;
    for (auto i = 1; i < argc; i++) {
        if (string_view{argv[i]} == "--offline") {
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

#ifndef LUISA_ENABLE_GUI
    if (!offline) {
        LUISA_ERROR("This tutorial was built without GUI support. Re-run with --offline.");
    }
#endif

    Device device = context.create_device(backend);
    Stream stream = offline ? device.create_stream() : device.create_stream(StreamTag::GRAPHICS);

    LUISA_INFO("Backend: {}, offline mode: {}", backend, offline);

    // Step 1: Create the initial galaxy on the CPU.
    // We place particles in a noisy disk and assign approximate orbital velocities.
    luisa::vector<Particle> host_particles(particle_count);
    auto rng_state = 1u;
    for (auto i = 0u; i < particle_count; i++) {
        auto radius = std::sqrt(host_random01(rng_state)) * 18.0f + 0.25f;
        auto angle = host_random01(rng_state) * 2.0f * std::numbers::pi_v<float>;
        auto height = (host_random01(rng_state) - 0.5f) * 0.8f;
        auto radial_jitter = (host_random01(rng_state) - 0.5f) * 0.4f;
        auto mass = 0.8f + host_random01(rng_state) * 1.6f;

        auto position = make_float3(
            std::cos(angle) * (radius + radial_jitter),
            height,
            std::sin(angle) * (radius + radial_jitter));
        auto tangent = normalize(make_float3(-position.z, 0.0f, position.x));
        auto orbital_speed = std::sqrt(gravity_constant * static_cast<float>(particle_count) * 0.08f / (radius + softening));
        auto velocity = tangent * orbital_speed +
                        make_float3(
                            (host_random01(rng_state) - 0.5f) * 0.06f,
                            (host_random01(rng_state) - 0.5f) * 0.02f,
                            (host_random01(rng_state) - 0.5f) * 0.06f);
        host_particles[i] = Particle{
            .position = position,
            .mass = mass,
            .velocity = velocity,
            .pad = 0.0f};
    }

    Buffer<Particle> particles_a = device.create_buffer<Particle>(particle_count);
    Buffer<Particle> particles_b = device.create_buffer<Particle>(particle_count);
    Buffer<float4> accumulation = device.create_buffer<float4>(resolution.x * resolution.y);
    Image<float> output_image = device.create_image<float>(PixelStorage::BYTE4, resolution);

    stream << particles_a.copy_from(luisa::span{host_particles})
           << particles_b.copy_from(luisa::span{host_particles})
           << synchronize();

    // Step 2: Build the simulation kernel.
    // We keep the implementation simple and readable: each particle sums the force from all others.
    Kernel1D simulate = [&](BufferVar<Particle> src, BufferVar<Particle> dst) noexcept {
        set_block_size(256u, 1u, 1u);

        Var i = dispatch_x();
        Var self = src.read(i);
        Float3 acceleration = make_float3(0.0f);

        $for (tile, particle_count / tile_size) {
            UInt tile_base = tile * tile_size;
            $for (local, tile_size) {
                UInt j = tile_base + local;
                Var other = src.read(j);
                Float3 delta = other.position - self.position;
                Float dist2 = dot(delta, delta) + softening * softening;
                Float inv_dist = rsqrt(dist2);
                Float inv_dist3 = inv_dist * inv_dist * inv_dist;
                $if (j != i) {
                    acceleration = acceleration + delta * (gravity_constant * other.mass * inv_dist3);
                };
            };
        };

        self.velocity = self.velocity + acceleration * simulation_dt;
        self.position = self.position + self.velocity * simulation_dt;
        dst.write(i, self);
    };

    // Step 3: Build a small renderer.
    // We first clear an accumulation buffer, then let one thread per particle splat a glow footprint.
    Callable particle_color = [](UInt index) noexcept {
        Float t = cast<float>(index) * 0.017f;
        return make_float3(
            0.55f + 0.45f * cos(t),
            0.55f + 0.45f * cos(t + 2.1f),
            0.65f + 0.35f * cos(t + 4.2f));
    };

    Kernel1D clear_accumulation = [&](BufferVar<float4> accum) noexcept {
        set_block_size(256u, 1u, 1u);
        Var i = dispatch_x();
        accum.write(i, make_float4(0.0f));
    };

    Kernel1D splat_particles = [&](BufferVar<Particle> particles,
                                   BufferVar<float4> accum,
                                   Float3 camera_pos,
                                   Float3 camera_forward,
                                   Float3 camera_right,
                                   Float3 camera_up) noexcept {
        set_block_size(256u, 1u, 1u);

        Var i = dispatch_x();
        Var p = particles.read(i);
        Float3 rel = p.position - camera_pos;
        Float3 view = make_float3(dot(rel, camera_right), dot(rel, camera_up), dot(rel, camera_forward));

        $if (view.z > 0.1f) {
            $float2 projected = view.xy() / view.z;
            $float focal = 1.35f;
            $float2 pixel = (projected * focal * 0.5f + 0.5f) * make_float2(resolution);
            $int2 center = make_int2(cast<int>(pixel.x), cast<int>(pixel.y));
            $float radius = clamp(8.0f / view.z + sqrt(p.mass) * 1.5f, 1.5f, 10.0f);
            $float3 color = particle_color(i) * (0.35f + 0.5f * p.mass);

            $for (oy, 9u) {
                $for (ox, 9u) {
                    $int x = center.x + cast<int>(ox) - 4;
                    $int y = center.y + cast<int>(oy) - 4;
                    $if (x >= 0 & y >= 0 & x < cast<int>(resolution.x) & y < cast<int>(resolution.y)) {
                        $float2 offset = make_float2(cast<float>(x) + 0.5f, cast<float>(y) + 0.5f) - pixel;
                        $float dist2 = dot(offset, offset);
                        $float sigma2 = radius * radius;
                        $float weight = exp(-dist2 / max(sigma2, 1.0f));
                        $uint pixel_index = cast<uint>(y) * resolution.x + cast<uint>(x);
                        accum.atomic(pixel_index).x.fetch_add(color.x * weight);
                        accum.atomic(pixel_index).y.fetch_add(color.y * weight);
                        accum.atomic(pixel_index).z.fetch_add(color.z * weight);
                        accum.atomic(pixel_index).w.fetch_add(weight);
                    };
                };
            };
        };
    };

    Kernel2D tone_map = [&](BufferVar<float4> accum, ImageFloat image) noexcept {
        set_block_size(16u, 16u, 1u);

        UInt2 coord = dispatch_id().xy();
        UInt pixel_index = coord.y * resolution.x + coord.x;
        Float4 sum = accum.read(pixel_index);
        Float3 color = make_float3(0.005f, 0.01f, 0.02f);

        $if (sum.w > 1e-5f) {
            color = color + sum.xyz() / sum.w;
        }
        $else {
            color = color + make_float3(0.0f);
        };

        color = 1.0f - exp(-color * 2.2f);
        color = pow(clamp(color, 0.0f, 1.0f), 1.0f / 2.2f);
        image.write(coord, make_float4(color, 1.0f));
    };

    auto simulate_shader = device.compile(simulate);
    auto clear_shader = device.compile(clear_accumulation);
    auto splat_shader = device.compile(splat_particles);
    auto tone_map_shader = device.compile(tone_map);

    auto render_frame = [&](Buffer<Particle> &current,
                            Buffer<Particle> &next,
                            float yaw,
                            float pitch) noexcept {
        auto eye = camera_position(yaw, pitch, 42.0f);
        auto forward = normalize(-eye);
        auto right = normalize(cross(forward, make_float3(0.0f, 1.0f, 0.0f)));
        auto up = normalize(cross(right, forward));

        stream << clear_shader(accumulation).dispatch(resolution.x * resolution.y)
               << splat_shader(current, accumulation, eye, forward, right, up).dispatch(particle_count)
               << tone_map_shader(accumulation, output_image).dispatch(resolution)
               << synchronize();
    };

    auto run_simulation_steps = [&](Buffer<Particle> &current, Buffer<Particle> &next, uint step_count) noexcept {
        for (auto i = 0u; i < step_count; i++) {
            stream << simulate_shader(current, next).dispatch(particle_count) << synchronize();
            std::swap(current, next);
        }
    };

    auto &current = particles_a;
    auto &next = particles_b;

    if (offline) {
        // Step 4a: Offline mode runs a fixed number of simulation steps and writes a PNG.
        for (auto i = 0u; i < 500u; i++) {
            run_simulation_steps(current, next, 1u);
        }
        render_frame(current, next, 0.65f, 0.35f);

        luisa::vector<std::array<uint8_t, 4u>> host_image(resolution.x * resolution.y);
        stream << output_image.copy_to(luisa::span{host_image}) << synchronize();
        stbi_write_png("tutorial_07_nbody.png", resolution.x, resolution.y, 4, host_image.data(), 0);
        LUISA_INFO("Saved tutorial_07_nbody.png");
        return 0;
    }

#ifdef LUISA_ENABLE_GUI
    // Step 4b: Interactive mode uses a window + swapchain.
    Window window{"Tutorial 07 - N-Body Galaxy", resolution};
    Swapchain swapchain = device.create_swapchain(
        stream,
        SwapchainOption{
            .display = window.native_display(),
            .window = window.native_handle(),
            .size = resolution,
            .wants_hdr = false,
            .wants_vsync = false,
            .back_buffer_count = 2,
        });
    Image<float> swapchain_image = device.create_image<float>(swapchain.backend_storage(), resolution);

    Kernel2D blit_to_swapchain = [&](ImageFloat src, ImageFloat dst) noexcept {
        set_block_size(16u, 16u, 1u);
        $uint2 coord = dispatch_id().xy();
        dst.write(coord, src.read(coord));
    };
    auto blit_shader = device.compile(blit_to_swapchain);

    auto mouse_pressed = false;
    auto last_mouse = make_float2(0.0f);
    auto drag = make_float2(0.0f);
    auto auto_yaw = 0.0f;
    auto pitch = 0.28f;

    window.set_mouse_callback([&](MouseButton button, Action action, float2 pos) noexcept {
        if (button == MouseButton::MOUSE_BUTTON_1) {
            last_mouse = pos / make_float2(resolution);
            mouse_pressed = action == Action::ACTION_PRESSED || action == Action::ACTION_REPEATED;
        }
    });
    window.set_cursor_position_callback([&](float2 pos) noexcept {
        if (mouse_pressed) {
            auto current_mouse = pos / make_float2(resolution);
            drag += current_mouse - last_mouse;
            last_mouse = current_mouse;
        }
    });
    window.set_key_callback([&](Key key, KeyModifiers, Action action) noexcept {
        if (key == Key::KEY_ESCAPE && action == Action::ACTION_RELEASED) {
            window.set_should_close();
        }
    });

    Clock clock;
    clock.tic();
    while (!window.should_close()) {
        window.poll_events();

        auto dt = static_cast<float>(clock.toc() * 1e-3);
        clock.tic();
        auto_yaw += dt * 0.25f;
        pitch = std::clamp(0.28f - drag.y * 2.5f, -1.1f, 1.1f);
        auto yaw = auto_yaw - drag.x * 6.0f;

        run_simulation_steps(current, next, 1u);
        render_frame(current, next, yaw, pitch);

        stream << blit_shader(output_image, swapchain_image).dispatch(resolution)
               << swapchain.present(swapchain_image)
               << synchronize();
    }
#endif

    return 0;
}
