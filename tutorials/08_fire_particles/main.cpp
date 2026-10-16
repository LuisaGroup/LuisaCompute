/*
 * Tutorial 08: Fire Particles
 *
 * This tutorial teaches how to:
 * - Step 1: Define and register a particle struct for a GPU particle system.
 * - Step 2: Update 65,536 particles fully on the GPU.
 * - Step 3: Add buoyancy, turbulence, cooling, and respawning logic.
 * - Step 4: Render particles as Gaussian splats with a fire color ramp.
 * - Step 5: Toggle wind interactively, or save the last frame with --offline.
 *
 * Usage:
 *   ./tutorial_08_fire_particles <backend> [--offline]
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

struct alignas(16) FireParticle {
    float3 position;
    float lifetime;
    float3 velocity;
    float temperature;
    float size;
    float3 pad;
};

LUISA_STRUCT(FireParticle, position, lifetime, velocity, temperature, size, pad) {};

namespace {

static constexpr uint particle_count = 65536u;
static constexpr uint2 resolution = make_uint2(1024u, 1024u);
static constexpr float simulation_dt = 1.0f / 60.0f;

[[nodiscard]] float host_random01(uint &state) noexcept {
    state = 1103515245u * state + 12345u;
    return static_cast<float>(state & 0x00ffffffu) / static_cast<float>(0x01000000u);
}

[[nodiscard]] float3 fixed_camera_position() noexcept {
    return make_float3(0.0f, 1.45f, 4.2f);
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

    Buffer<FireParticle> particles = device.create_buffer<FireParticle>(particle_count);
    Buffer<float4> accumulation = device.create_buffer<float4>(resolution.x * resolution.y);
    Image<float> output_image = device.create_image<float>(PixelStorage::BYTE4, resolution);

    // Step 1: Seed the particle system from the CPU.
    luisa::vector<FireParticle> host_particles(particle_count);
    auto rng_state = 7u;
    for (auto i = 0u; i < particle_count; i++) {
        auto angle = host_random01(rng_state) * 2.0f * std::numbers::pi_v<float>;
        auto radius = std::sqrt(host_random01(rng_state)) * 0.12f;
        host_particles[i] = FireParticle{
            .position = make_float3(std::cos(angle) * radius, host_random01(rng_state) * 0.2f, std::sin(angle) * radius),
            .lifetime = 0.6f + host_random01(rng_state) * 0.8f,
            .velocity = make_float3(
                (host_random01(rng_state) - 0.5f) * 0.25f,
                1.5f + host_random01(rng_state) * 1.8f,
                (host_random01(rng_state) - 0.5f) * 0.25f),
            .temperature = 0.8f + host_random01(rng_state) * 0.4f,
            .size = 0.018f + host_random01(rng_state) * 0.02f,
            .pad = make_float3(0.0f)};
    }
    stream << particles.copy_from(luisa::span{host_particles}) << synchronize();

    Callable hash_u32 = [](UInt x) noexcept {
        x = (x ^ 61u) ^ (x >> 16u);
        x *= 9u;
        x ^= x >> 4u;
        x *= 0x27d4eb2du;
        x ^= x >> 15u;
        return x;
    };

    Callable random01 = [&hash_u32](UInt seed) noexcept {
        return cast<float>(hash_u32(seed) & 0x00ffffffu) * (1.0f / static_cast<float>(0x01000000u));
    };

    // Step 2: Update particles fully on the GPU.
    Kernel1D update_particles = [&](BufferVar<FireParticle> particle_buffer, Float time, UInt frame_index, Bool wind_enabled) noexcept {
        set_block_size(256u, 1u, 1u);

        Var i = dispatch_x();
        Var p = particle_buffer.read(i);
        Var seed = hash_u32(i ^ (frame_index * 747796405u + 2891336453u));

        $if (p.lifetime <= 0.0f | p.temperature <= 0.01f | p.position.y > 5.5f) {
            Var angle = random01(seed ^ 1u) * 2.0f * pi;
            Var radius = sqrt(random01(seed ^ 2u)) * 0.14f;
            Var jitter = random01(seed ^ 3u) - 0.5f;
            p.position = make_float3(cos(angle) * radius, random01(seed ^ 4u) * 0.08f, sin(angle) * radius);
            p.velocity = make_float3(cos(angle) * 0.15f + jitter * 0.1f,
                                     1.4f + random01(seed ^ 5u) * 2.0f,
                                     sin(angle) * 0.15f + (random01(seed ^ 6u) - 0.5f) * 0.1f);
            p.lifetime = 0.8f + random01(seed ^ 7u) * 1.0f;
            p.temperature = 0.95f + random01(seed ^ 8u) * 0.35f;
            p.size = 0.018f + random01(seed ^ 9u) * 0.028f;
        }
        $else {
            Var noise = sin(p.position.x * 9.0f + time * 7.0f + random01(seed ^ 10u) * 6.2831853f) *
                        cos(p.position.z * 11.0f - time * 5.0f + random01(seed ^ 11u) * 6.2831853f);
            Var swirl = sin(time * 3.0f + p.position.y * 12.0f + cast<float>(i) * 0.0002f);
            Float3 turbulence = make_float3(noise * 0.45f, 0.0f, swirl * 0.35f);
            Float3 buoyancy = make_float3(0.0f, 2.8f * (0.2f + p.temperature), 0.0f);
            Float3 wind = make_float3(0.9f, 0.0f, 0.15f) * ite(wind_enabled, 1.0f, 0.0f);

            p.velocity = p.velocity + (buoyancy + turbulence + wind * (0.25f + 0.75f * random01(seed ^ 12u))) * simulation_dt;
            p.velocity = p.velocity * 0.985f;
            p.position = p.position + p.velocity * simulation_dt;
            p.temperature = max(p.temperature - (0.32f + 0.18f * abs(noise)) * simulation_dt, 0.0f);
            p.lifetime = p.lifetime - (0.55f + 0.2f * random01(seed ^ 13u)) * simulation_dt;
            p.size = max(0.012f, p.size * 0.995f + p.temperature * 0.0015f);
        };

        particle_buffer.write(i, p);
    };

    // Step 3: Render the fire with additive Gaussian splats.
    Callable temperature_to_color = [](Float t) noexcept {
        t = clamp(t, 0.0f, 1.0f);
        Float3 color = make_float3(0.0f);
        $if (t < 0.25f) {
            color = lerp(make_float3(0.0f), make_float3(0.75f, 0.0f, 0.0f), t / 0.25f);
        }
        $elif (t < 0.5f) {
            color = lerp(make_float3(0.75f, 0.0f, 0.0f), make_float3(1.0f, 0.35f, 0.0f), (t - 0.25f) / 0.25f);
        }
        $elif (t < 0.75f) {
            color = lerp(make_float3(1.0f, 0.35f, 0.0f), make_float3(1.0f, 0.9f, 0.1f), (t - 0.5f) / 0.25f);
        }
        $else {
            color = lerp(make_float3(1.0f, 0.9f, 0.1f), make_float3(1.0f), (t - 0.75f) / 0.25f);
        };
        return color;
    };

    Kernel1D clear_accumulation = [&](BufferVar<float4> accum) noexcept {
        set_block_size(256u, 1u, 1u);
        Var i = dispatch_x();
        accum.write(i, make_float4(0.0f));
    };

    Kernel1D render_particles = [&](BufferVar<FireParticle> particle_buffer,
                                    BufferVar<float4> accum,
                                    Float3 camera_pos,
                                    Float3 camera_forward,
                                    Float3 camera_right,
                                    Float3 camera_up) noexcept {
        set_block_size(256u, 1u, 1u);

        Var i = dispatch_x();
        Var p = particle_buffer.read(i);
        Float3 rel = p.position - camera_pos;
        Float3 view = make_float3(dot(rel, camera_right), dot(rel, camera_up), dot(rel, camera_forward));

        $if (view.z > 0.1f) {
            Float2 projected = view.xy() / view.z;
            Float2 pixel = (projected * 1.25f * 0.5f + 0.5f) * make_float2(resolution);
            Int2 center = make_int2(cast<int>(pixel.x), cast<int>(pixel.y));
            Float radius = clamp((p.size * 180.0f) / view.z, 1.5f, 14.0f);
            Float3 color = temperature_to_color(p.temperature) * (0.15f + p.temperature * 0.85f);

            $for (oy, 11u) {
                $for (ox, 11u) {
                    Int x = center.x + cast<int>(ox) - 5;
                    Int y = center.y + cast<int>(oy) - 5;
                    $if (x >= 0 & y >= 0 & x < cast<int>(resolution.x) & y < cast<int>(resolution.y)) {
                        Float2 offset = make_float2(cast<float>(x) + 0.5f, cast<float>(y) + 0.5f) - pixel;
                        Float dist2 = dot(offset, offset);
                        Float sigma2 = radius * radius;
                        Float weight = exp(-dist2 / max(sigma2, 1.0f)) * clamp(p.lifetime, 0.0f, 1.0f);
                        UInt pixel_index = cast<uint>(y) * resolution.x + cast<uint>(x);
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
        Float3 color = make_float3(0.01f, 0.005f, 0.002f);

        $if (sum.w > 1e-5f) {
            color = color + sum.xyz() / sum.w;
        }
        $else {
            color = color + make_float3(0.0f);
        };

        color = 1.0f - exp(-color * 2.7f);
        color = pow(clamp(color, 0.0f, 1.0f), 1.0f / 2.2f);
        image.write(coord, make_float4(color, 1.0f));
    };

    auto update_shader = device.compile(update_particles);
    auto clear_shader = device.compile(clear_accumulation);
    auto render_shader = device.compile(render_particles);
    auto tone_map_shader = device.compile(tone_map);

    auto camera_pos = fixed_camera_position();
    auto camera_forward = normalize(make_float3(0.0f, 1.1f, 0.0f) - camera_pos);
    auto camera_right = normalize(cross(camera_forward, make_float3(0.0f, 1.0f, 0.0f)));
    auto camera_up = normalize(cross(camera_right, camera_forward));

    auto render_frame = [&] {
        stream << clear_shader(accumulation).dispatch(resolution.x * resolution.y)
               << render_shader(particles, accumulation, camera_pos, camera_forward, camera_right, camera_up).dispatch(particle_count)
               << tone_map_shader(accumulation, output_image).dispatch(resolution)
               << synchronize();
    };

    auto wind_enabled = false;
    auto frame_index = 0u;
    auto time_seconds = 0.0f;

    if (offline) {
        for (auto i = 0u; i < 200u; i++) {
            stream << update_shader(particles, time_seconds, frame_index, wind_enabled).dispatch(particle_count) << synchronize();
            frame_index++;
            time_seconds += simulation_dt;
        }
        render_frame();

        luisa::vector<std::array<uint8_t, 4u>> host_image(resolution.x * resolution.y);
        stream << output_image.copy_to(luisa::span{host_image}) << synchronize();
        stbi_write_png("tutorial_08_fire_particles.png", resolution.x, resolution.y, 4, host_image.data(), 0);
        LUISA_INFO("Saved tutorial_08_fire_particles.png");
        return 0;
    }

#ifdef LUISA_ENABLE_GUI
    Window window{"Tutorial 08 - Fire Particles", resolution};
    window.set_key_callback([&](Key key, KeyModifiers, Action action) noexcept {
        if (action == Action::ACTION_RELEASED) {
            if (key == Key::KEY_SPACE) {
                wind_enabled = !wind_enabled;
                LUISA_INFO("Wind toggled: {}", wind_enabled);
            } else if (key == Key::KEY_ESCAPE) {
                window.set_should_close();
            }
        }
    });

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
        UInt2 coord = dispatch_id().xy();
        dst.write(coord, src.read(coord));
    };
    auto blit_shader = device.compile(blit_to_swapchain);

    Clock clock;
    clock.tic();
    while (!window.should_close()) {
        window.poll_events();

        auto dt = static_cast<float>(clock.toc() * 1e-3);
        clock.tic();
        auto substeps = std::max(1u, static_cast<uint>(std::ceil(dt / simulation_dt)));
        for (auto step = 0u; step < substeps; step++) {
            stream << update_shader(particles, time_seconds, frame_index, wind_enabled).dispatch(particle_count) << synchronize();
            frame_index++;
            time_seconds += simulation_dt;
        }

        render_frame();
        stream << blit_shader(output_image, swapchain_image).dispatch(resolution)
               << swapchain.present(swapchain_image)
               << synchronize();
    }
#endif

    return 0;
}
