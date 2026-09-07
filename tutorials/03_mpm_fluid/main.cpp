/*
 * Tutorial 03: MPM3D Fluid Simulation
 *
 * This tutorial teaches a complete particle-grid simulation pipeline with LuisaCompute:
 * 1. Create particle and grid buffers.
 * 2. Clear the Eulerian grid each substep.
 * 3. Scatter particle mass and momentum to the grid (P2G) with B-spline weights.
 * 4. Update grid velocities with gravity and boundary conditions.
 * 5. Gather updated velocities back to particles (G2P).
 * 6. Visualize the particles by projecting the 3D fluid into a 2D image.
 * 7. Run interactively in a window or offline and save the final frame to PNG.
 *
 * Why this tutorial matters:
 * The Material Point Method is a classic hybrid simulation technique. It shows how LuisaCompute
 * can express both physics kernels and visualization kernels in the same DSL-based program.
 */

#include <algorithm>
#include <array>
#include <cstdlib>
#include <memory>
#include <optional>
#include <random>
#include <string_view>

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

    // Step 1: Create the Context and Device.
    // Why: just like the previous tutorials, the Context locates backends and the Device owns the
    // resources and shaders used by this simulation.
    // Parse command-line: any non-flag argument is the backend name.
    // If no backend is given, the first installed backend is selected automatically.
    luisa::string backend;
    bool offline = false;
    uint frame_limit = 0u;
    for (int i = 1; i < argc; i++) {
        if (std::string_view{argv[i]} == "--offline") {
            offline = true;
        } else if (std::string_view{argv[i]} == "--frames" && i + 1 < argc) {
            frame_limit = static_cast<uint>(std::atoi(argv[++i]));
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

    if (offline && frame_limit == 0u) {
        frame_limit = 100u;
    }

    if (!offline && !ENABLE_DISPLAY) {
        LUISA_WARNING("GUI support is disabled in this build. Falling back to --offline mode.");
        offline = true;
    }

    Device device = context.create_device(backend);
    Stream stream = device.create_stream(offline ? StreamTag::COMPUTE : StreamTag::GRAPHICS);

    // Step 2: Define simulation constants.
    // Why: MPM alternates between particles and a background grid, so the grid size, particle
    // count, time step, and material constants define the entire simulation budget and behavior.
    static constexpr int n_grid = 64;
    static constexpr uint n_particles = 50000u;
    static constexpr uint substeps_per_frame = 20u;
    static constexpr float dx = 1.0f / static_cast<float>(n_grid);
    static constexpr float dt = 8.0e-5f;
    static constexpr float particle_density = 1.0f;
    static constexpr float particle_volume = (dx * 0.5f) * (dx * 0.5f) * (dx * 0.5f);
    static constexpr float particle_mass = particle_density * particle_volume;
    static constexpr float gravity = 9.8f;
    static constexpr int boundary = 3;
    static constexpr float bulk_modulus = 1800.0f;
    static constexpr float viscosity = 0.08f;

    static constexpr uint resolution = 1024u;
    constexpr char output_file[] = "tutorial_03_mpm_fluid.png";

    LUISA_INFO("Tutorial 03: simulating {} particles on a {}^3 grid.", n_particles, n_grid);

    // Step 3: Allocate particle and grid buffers.
    // Why: particles carry the Lagrangian state, while the grid is the temporary Eulerian staging
    // area where forces and boundary conditions are easier to apply.
    Buffer<float3> particle_positions = device.create_buffer<float3>(n_particles);
    Buffer<float3> particle_velocities = device.create_buffer<float3>(n_particles);
    Buffer<float3x3> particle_C = device.create_buffer<float3x3>(n_particles);
    Buffer<float> particle_J = device.create_buffer<float>(n_particles);
    Buffer<float4> grid = device.create_buffer<float4>(static_cast<size_t>(n_grid) * n_grid * n_grid);

    Image<float> display = device.create_image<float>(offline ? PixelStorage::BYTE4 : PixelStorage::FLOAT4, make_uint2(resolution));
    std::optional<Image<float>> present_image;
#if ENABLE_DISPLAY
    std::unique_ptr<Window> window;
    std::optional<Swapchain> swapchain;
    if (!offline) {
        window = std::make_unique<Window>("Tutorial 03 - MPM Fluid", make_uint2(resolution), false);
        swapchain.emplace(device.create_swapchain(
            stream,
            SwapchainOption{
                .display = window->native_display(),
                .window = window->native_handle(),
                .size = make_uint2(resolution),
                .wants_hdr = false,
                .wants_vsync = false,
                .back_buffer_count = 2,
            }));
        display = device.create_image<float>(PixelStorage::BYTE4, make_uint2(resolution));
        present_image.emplace(device.create_image<float>(swapchain->backend_storage(), make_uint2(resolution)));
    }
#endif

    // Step 4: Define small math helpers.
    // Why: index conversion, matrix outer products, and traces are repeated in multiple kernels.
    auto grid_index = [](Int3 coord) noexcept {
        Int3 p = clamp(coord, 0, n_grid - 1);
        return cast<uint>(p.x + p.y * n_grid + p.z * n_grid * n_grid);
    };

    auto outer_product = [](Float3 a, Float3 b) noexcept {
        return make_float3x3(
            make_float3(a.x * b.x, a.y * b.x, a.z * b.x),
            make_float3(a.x * b.y, a.y * b.y, a.z * b.y),
            make_float3(a.x * b.z, a.y * b.z, a.z * b.z));
    };

    auto trace = [](Float3x3 m) noexcept {
        return m[0][0] + m[1][1] + m[2][2];
    };

    Callable project_to_screen = [](Float3 p0) noexcept {
        constexpr float phi = radians(28.0f);
        constexpr float theta = radians(32.0f);
        Float3 p = p0 - make_float3(0.5f);
        Float c = cos(phi);
        Float s = sin(phi);
        Float C = cos(theta);
        Float S = sin(theta);
        Float rotated_x = p.x * c + p.z * s;
        Float rotated_z = p.z * c - p.x * s;
        return make_float2(rotated_x, p.y * C + rotated_z * S) + 0.5f;
    };

    // Step 5: Clear the grid at the start of each substep.
    // Why: the grid stores temporary momentum and mass, so each simulation substep starts from zero.
    auto clear_grid = device.compile<3>([&] {
        set_block_size(8u, 8u, 1u);
        UInt index = grid_index(make_int3(dispatch_id().xyz()));
        grid->write(index, make_float4(0.0f));
    });

    // Step 6: Particle-to-grid transfer (P2G).
    // Why: particles carry the fluid state, but forces and collisions are easier to handle on a
    // regular grid. Quadratic B-spline weights distribute each particle to a 3x3x3 stencil.
    auto particle_to_grid = device.compile<1>([&] {
        set_block_size(64u, 1u, 1u);
        UInt particle = dispatch_id().x;

        Float3 Xp = particle_positions->read(particle) / dx;
        Int3 base = make_int3(Xp - 0.5f);
        Float3 fx = Xp - make_float3(base);

        std::array w{
            0.5f * (1.5f - fx) * (1.5f - fx),
            0.75f - (fx - 1.0f) * (fx - 1.0f),
            0.5f * (fx - 0.5f) * (fx - 0.5f)};

        Float J = particle_J->read(particle);
        Float compression = max(1.0f - J, 0.0f);
        Float pressure = bulk_modulus * compression;
        Float stress = -4.0f * dt * particle_volume * pressure / (dx * dx);
        Float3 velocity = particle_velocities->read(particle);
        Float3x3 affine = make_float3x3(
                              make_float3(stress, 0.f, 0.f),
                              make_float3(0.f, stress, 0.f),
                              make_float3(0.f, 0.f, stress)) +
                          viscosity * particle_mass * particle_C->read(particle);

        $for (ii, 27u) {
            UInt i = ii % 3u;
            UInt j = ii / 3u % 3u;
            UInt k = ii / 9u;
            Int3 offset = make_int3(cast<int>(i), cast<int>(j), cast<int>(k));
            Float3 dpos = (make_float3(offset) - fx) * dx;
            Float weight =
                ite(i == 0u, w[0].x, ite(i == 1u, w[1].x, w[2].x)) *
                ite(j == 0u, w[0].y, ite(j == 1u, w[1].y, w[2].y)) *
                ite(k == 0u, w[0].z, ite(k == 1u, w[1].z, w[2].z));
            Float3 momentum = weight * (particle_mass * velocity + affine * dpos);
            UInt node = grid_index(base + offset);
            grid->atomic(node).x.fetch_add(momentum.x);
            grid->atomic(node).y.fetch_add(momentum.y);
            grid->atomic(node).z.fetch_add(momentum.z);
            grid->atomic(node).w.fetch_add(weight * particle_mass);
        };
    });

    // Step 7: Update grid velocities.
    // Why: after particles deposit momentum, the grid update is where we divide by mass, apply
    // gravity, and enforce boundary conditions that keep the fluid inside the simulation domain.
    auto grid_update = device.compile<3>([&] {
        set_block_size(8u, 8u, 1u);
        Int3 coord = make_int3(dispatch_id().xyz());
        UInt index = grid_index(coord);
        Float4 state = grid->read(index);
        Float3 velocity = state.xyz();
        Float mass = state.w;

        $if (mass > 0.0f) {
            velocity /= mass;
            velocity.y -= gravity * dt;

            $if ((coord.x < boundary & velocity.x < 0.0f) | (coord.x > n_grid - boundary & velocity.x > 0.0f)) {
                velocity.x = 0.0f;
            };
            $if ((coord.y < boundary & velocity.y < 0.0f) | (coord.y > n_grid - boundary & velocity.y > 0.0f)) {
                velocity.y = 0.0f;
            };
            $if ((coord.z < boundary & velocity.z < 0.0f) | (coord.z > n_grid - boundary & velocity.z > 0.0f)) {
                velocity.z = 0.0f;
            };
        }
        $else {
            velocity = make_float3(0.0f);
        };

        grid->write(index, make_float4(velocity, mass));
    });

    // Step 8: Grid-to-particle transfer (G2P).
    // Why: the grid now contains updated velocities, so we gather them back to particles and move
    // particle positions forward in time.
    auto grid_to_particle = device.compile<1>([&] {
        set_block_size(64u, 1u, 1u);
        UInt particle = dispatch_id().x;

        Float3 Xp = particle_positions->read(particle) / dx;
        Int3 base = make_int3(Xp - 0.5f);
        Float3 fx = Xp - make_float3(base);

        std::array w{
            0.5f * (1.5f - fx) * (1.5f - fx),
            0.75f - (fx - 1.0f) * (fx - 1.0f),
            0.5f * (fx - 0.5f) * (fx - 0.5f)};

        Float3 new_velocity = def(make_float3(0.0f));
        Float3x3 new_C = def(make_float3x3(0.0f));

        $for (ii, 27u) {
            UInt i = ii % 3u;
            UInt j = ii / 3u % 3u;
            UInt k = ii / 9u;
            Int3 offset = make_int3(cast<int>(i), cast<int>(j), cast<int>(k));
            Float3 dpos = (make_float3(offset) - fx) * dx;
            Float weight =
                ite(i == 0u, w[0].x, ite(i == 1u, w[1].x, w[2].x)) *
                ite(j == 0u, w[0].y, ite(j == 1u, w[1].y, w[2].y)) *
                ite(k == 0u, w[0].z, ite(k == 1u, w[1].z, w[2].z));
            UInt node = grid_index(base + offset);
            Float3 grid_velocity = grid->read(node).xyz();
            new_velocity += weight * grid_velocity;
            new_C += 4.0f * weight * outer_product(grid_velocity, dpos) / (dx * dx);
        };

        Float3 position = particle_positions->read(particle) + new_velocity * dt;
        position = clamp(position, make_float3(0.02f), make_float3(0.98f));

        particle_positions->write(particle, position);
        particle_velocities->write(particle, new_velocity);
        particle_C->write(particle, new_C);
        particle_J->write(particle, max(0.6f, particle_J->read(particle) * (1.0f + dt * trace(new_C))));
    });

    // Step 9: Visualization kernels.
    // Why: simulation data is not directly visible, so we render projected particles into a 2D
    // image that can be presented or written to disk.
    auto clear_display = device.compile<2>([&] {
        set_block_size(16u, 16u, 1u);
        UInt2 pixel = dispatch_id().xy();
        Float2 uv = make_float2(pixel) / static_cast<float>(resolution);
        Float3 background = lerp(make_float3(0.04f, 0.08f, 0.12f), make_float3(0.12f, 0.18f, 0.24f), uv.y);
        display->write(pixel, make_float4(background, 1.0f));
    });

    auto draw_particles = device.compile<1>([&] {
        set_block_size(64u, 1u, 1u);
        UInt particle = dispatch_id().x;
        Float3 position = particle_positions->read(particle);
        Float2 projected = project_to_screen(position);
        Float speed = length(particle_velocities->read(particle));
        Float3 color = lerp(make_float3(0.20f, 0.45f, 0.95f), make_float3(0.85f, 0.95f, 1.00f), clamp(speed * 3.0f, 0.0f, 1.0f));
        Int2 center = make_int2(projected * static_cast<float>(resolution));

        $for (offset_index, 9u) {
            Int2 offset = make_int2(cast<int>(offset_index % 3u) - 1, cast<int>(offset_index / 3u) - 1);
            Int2 pixel = center + offset;
            $if (pixel.x >= 0 & pixel.x < static_cast<int>(resolution) & pixel.y >= 0 & pixel.y < static_cast<int>(resolution)) {
                display->write(make_uint2(cast<uint>(pixel.x), resolution - 1u - cast<uint>(pixel.y)), make_float4(color, 1.0f));
            };
        };
    });

    Kernel2D blit_kernel = [](ImageFloat src, ImageFloat dst) noexcept {
        set_block_size(16u, 16u, 1u);
        UInt2 pixel = dispatch_id().xy();
        dst.write(pixel, src.read(pixel));
    };
    auto blit_to_present = device.compile(blit_kernel);

    // Step 10: Initialize particles.
    // Why: a compact blob near the upper-left of the domain gives the fluid room to fall, splash,
    // and spread under gravity.
    luisa::vector<float3> initial_positions(n_particles);
    luisa::vector<float3> initial_velocities(n_particles, make_float3(0.0f));
    luisa::vector<float3x3> initial_C(n_particles, make_float3x3(0.0f));
    luisa::vector<float> initial_J(n_particles, 1.0f);
    std::mt19937 rng{std::random_device{}()};
    std::uniform_real_distribution<float> uniform(0.0f, 1.0f);
    for (uint i = 0u; i < n_particles; i++) {
        float x = 0.15f + uniform(rng) * 0.25f;
        float y = 0.25f + uniform(rng) * 0.45f;
        float z = 0.20f + uniform(rng) * 0.25f;
        initial_positions[i] = make_float3(x, y, z);
    }
    stream << particle_positions.copy_from(luisa::span{initial_positions})
           << particle_velocities.copy_from(luisa::span{initial_velocities})
           << particle_C.copy_from(luisa::span{initial_C})
           << particle_J.copy_from(luisa::span{initial_J})
           << synchronize();

    auto run_substeps = [&](CommandList &commands) noexcept {
        for (uint i = 0u; i < substeps_per_frame; i++) {
            commands << clear_grid().dispatch(n_grid, n_grid, n_grid)
                     << particle_to_grid().dispatch(n_particles)
                     << grid_update().dispatch(n_grid, n_grid, n_grid)
                     << grid_to_particle().dispatch(n_particles);
        }
    };

    // Step 11: Main simulation loop.
    // Why: offline mode advances a fixed number of frames before saving the last image, while
    // interactive mode advances one frame at a time and presents the visualization live.
    uint frame_index = 0u;
    if (offline) {
        for (; frame_index < frame_limit; frame_index++) {
            CommandList commands;
            run_substeps(commands);
            commands << clear_display().dispatch(resolution, resolution)
                     << draw_particles().dispatch(n_particles);
            stream << commands.commit();
            if ((frame_index + 1u) % 10u == 0u || frame_index + 1u == frame_limit) {
                LUISA_INFO("Simulated {} / {} offline frame(s).", frame_index + 1u, frame_limit);
            }
        }

        luisa::vector<std::array<uint8_t, 4u>> host_image(static_cast<size_t>(resolution) * resolution);
        stream << display.copy_to(luisa::span{host_image})
               << synchronize();
        stbi_write_png(output_file, static_cast<int>(resolution), static_cast<int>(resolution), 4, host_image.data(), 0);
        LUISA_INFO("Saved final offline MPM frame to {}.", output_file);
    } else {
#if ENABLE_DISPLAY
        while (!window->should_close()) {
            window->poll_events();
            if (window->is_key_down(KEY_ESCAPE)) {
                break;
            }

            CommandList commands;
            run_substeps(commands);
            commands << clear_display().dispatch(resolution, resolution)
                     << draw_particles().dispatch(n_particles)
                     << blit_to_present(display, *present_image).dispatch(resolution, resolution);
            stream << commands.commit()
                   << swapchain->present(*present_image);

            frame_index++;
            if (frame_index % 30u == 0u) {
                LUISA_INFO("Simulated {} interactive frame(s).", frame_index);
            }
        }
#endif
        stream << synchronize();
    }

    return 0;
}
