// Material Point Method (MPM) simulation in 3D.
//
// This is a 3D extension of test_mpm88.cpp, simulating deformable
// elastic materials using the Moving Least Squares MPM approach.
//
// Key features:
// - 3D grid with 64^3 resolution
// - Neo-Hookean elasticity model
// - 27-point (3x3x3) quadratic B-spline interpolation
// - Isometric projection for visualization
// - Real-time FPS display

#include <cmath>
#include <cstdlib>
#include <memory>
#include <optional>
#include <random>
#include <fstream>
#include <chrono>
#include <filesystem>
#include <string_view>

#include "../common/reference_compare.h"
#include <luisa/dsl/syntax.h>
#include <luisa/dsl/sugar.h>
#include <luisa/runtime/buffer.h>
#include <luisa/runtime/device.h>
#include <luisa/runtime/stream.h>
#include <luisa/core/logging.h>
#include <luisa/core/clock.h>
#include <luisa/runtime/context.h>
#include <luisa/runtime/swapchain.h>

#ifndef ENABLE_DISPLAY
#ifdef LUISA_ENABLE_GUI
#define ENABLE_DISPLAY 1
#endif
#endif

#if ENABLE_DISPLAY
#include <luisa/gui/window.h>
#include <luisa/gui/framerate.h>
#endif

int main(int argc, char *argv[]) {

    using namespace luisa;
    using namespace luisa::compute;

    auto sqr = [](auto x) noexcept { return x * x; };

    // Initialize compute context
    Context context{argv[0]};
    if (argc <= 1) {
        LUISA_INFO("Usage: {} <backend> [--offline] [-c <reference.png>] [--frames N]. <backend>: cuda, dx, metal, vk, hip, fallback", argv[0]);
        exit(1);
    }

    auto opts = luisa::ref::ExampleOptions::parse(argc, argv);
    if (!opts.valid()) {
        LUISA_WARNING("Invalid command line: {}", opts.error_message);
        return 1;
    }
    auto force_offline = opts.offline;
    auto compare_path = opts.compare_path;
    uint user_frames = 0u;
    for (int i = 2; i < argc; i++) {
        if (std::string_view{argv[i]} == "--frames") {
            if (i + 1 >= argc || argv[i + 1] == nullptr) {
                LUISA_WARNING("Invalid command line: Missing value for --frames.");
                return 1;
            }
            std::string_view value{argv[++i]};
            auto parsed_value = luisa::ref::parse_uint32_option_value(value);
            if (!parsed_value) {
                LUISA_WARNING("Invalid command line: Invalid unsigned integer for --frames: '{}'.", value);
                return 1;
            }
            user_frames = *parsed_value;
        }
    }
    // Default to 200 frames in offline mode if not specified
    if (force_offline && user_frames == 0u) { user_frames = 200u; }
#if !ENABLE_DISPLAY
    if (!force_offline) {
        LUISA_ERROR("GUI support is disabled. Use --offline.");
    }
#endif

    Device device = context.create_device(argv[1]);

    // Simulation parameters
    static constexpr int n_grid = 64;   // Grid resolution per dimension
    static constexpr uint n_steps = 25u;// Substeps per frame

    static constexpr uint n_particles = n_grid * n_grid * n_grid / 4u;  // 1/4 of grid cells filled
    static constexpr float dx = 1.f / n_grid;                           // Grid cell size
    static constexpr float dt = 8e-5f;                                  // Time step (smaller than 2D for stability)
    static constexpr float p_rho = 1.f;                                 // Particle density
    static constexpr float p_vol = (dx * .5f) * (dx * .5f) * (dx * .5f);// Particle volume
    static constexpr float p_mass = p_rho * p_vol;                      // Particle mass
    static constexpr float gravity = 9.8f;                              // Gravitational acceleration
    static constexpr int bound = 3;                                     // Boundary thickness
    static constexpr float E = 400.f;                                   // Young's modulus

    static constexpr uint resolution = 1024u;// Display resolution

    // Particle state buffers
    Buffer<float3> x = device.create_buffer<float3>(n_particles);    // Positions
    Buffer<float3> v = device.create_buffer<float3>(n_particles);    // Velocities
    Buffer<float3x3> C = device.create_buffer<float3x3>(n_particles);// Affine momentum
    Buffer<float> J = device.create_buffer<float>(n_particles);      // Volume

    // Grid buffer: stores (vx, vy, vz, mass) for each cell
    Buffer<float> grid = device.create_buffer<float>(n_grid * n_grid * n_grid * 4);

    // Setup graphics
    Stream stream = device.create_stream(force_offline ? StreamTag::COMPUTE : StreamTag::GRAPHICS);
#if ENABLE_DISPLAY
    std::unique_ptr<Window> window;
    std::optional<Swapchain> swap_chain;
    if (!force_offline) {
        window = std::make_unique<Window>("MPM3D", resolution, resolution);
        swap_chain.emplace(device.create_swapchain(
            stream,
            SwapchainOption{
                .display = window->native_display(),
                .window = window->native_handle(),
                .size = make_uint2(resolution),
                .wants_hdr = false,
                .wants_vsync = false,
                .back_buffer_count = 8,
            }));
    }
#endif
    Image<float> display = [&] {
#if ENABLE_DISPLAY
        if (!force_offline) {
            return device.create_image<float>(swap_chain->backend_storage(), make_uint2(resolution));
        }
#endif
        return device.create_image<float>(PixelStorage::BYTE4, make_uint2(resolution));
    }();

    Kernel2D save_display_kernel = [](ImageFloat src, BufferFloat4 dst, UInt width) noexcept {
        UInt2 p = dispatch_id().xy();
        dst.write(p.y * width + p.x, src.read(p));
    };
    auto save_display_shader = device.compile(save_display_kernel);

    // Helper: compute 1D grid index from 3D coordinates
    auto index = [](Int3 xyz) noexcept {
        auto p = clamp(xyz, 0, n_grid - 1);
        return p.x + p.y * n_grid + p.z * n_grid * n_grid;
    };

    // Helper: compute 3D outer product
    auto outer_product = [](Float3 a, Float3 b) noexcept {
        return make_float3x3(
            make_float3(a[0] * b[0], a[1] * b[0], a[2] * b[0]),
            make_float3(a[0] * b[1], a[1] * b[1], a[2] * b[1]),
            make_float3(a[0] * b[2], a[1] * b[2], a[2] * b[2]));
    };

    // Helper: compute matrix trace
    auto trace = [](Float3x3 m) noexcept { return m[0][0] + m[1][1] + m[2][2]; };

    // Kernel: Clear grid (velocity + mass)
    auto clear_grid = device.compile<3>([&] {
        set_block_size(8, 8, 1);
        UInt idx = index(dispatch_id().xyz());
        for (int i = 0; i < 4; ++i)
            grid->write(idx * 4 + i, 0.f);
    });

    // Kernel: Particle to Grid transfer (P2G)
    // Uses 3x3x3 = 27-point quadratic B-spline stencil
    auto point_to_grid = device.compile<1>([&] {
        set_block_size(64, 1, 1);
        UInt p = dispatch_id().x;

        // Particle position in grid coordinates
        Float3 Xp = x->read(p) / dx;
        Int3 base = make_int3(Xp - 0.5f);
        Float3 fx = Xp - make_float3(base);

        // Quadratic B-spline weights
        std::array w{0.5f * sqr(1.5f - fx),
                     0.75f - sqr(fx - 1.0f),
                     0.5f * sqr(fx - 0.5f)};

        // Neo-Hookean stress
        Float stress = -4.f * dt * E * p_vol * (J->read(p) - 1.f) / sqr(dx);
        Float3x3 affine = make_float3x3(stress, 0.f, 0.f,
                                        0.f, stress, 0.f,
                                        0.f, 0.f, stress) +
                          p_mass * C->read(p);
        Float3 vp = v->read(p);

        // Scatter to 3x3x3 neighboring cells
        $for (ii, 27u) {
            UInt i = ii % 3u;
            UInt j = ii / 3u % 3u;
            UInt k = ii / 3u / 3u;
            Int3 offset = make_int3(cast<int>(i), cast<int>(j), cast<int>(k));

            Float3 dpos = (make_float3(offset) - fx) * dx;
            Float wi = ite(i == 0u, w[0].x, ite(i == 1u, w[1].x, w[2].x));
            Float wj = ite(j == 0u, w[0].y, ite(j == 1u, w[1].y, w[2].y));
            Float wk = ite(k == 0u, w[0].z, ite(k == 1u, w[1].z, w[2].z));
            Float weight = wi * wj * wk;
            Float3 vadd = weight * (p_mass * vp + affine * dpos);
            UInt idx = index(base + offset);

            for (int c = 0; c < 3; ++c) {
                grid->atomic(idx * 4u + c).fetch_add(vadd[c]);
            }
            grid->atomic(idx * 4u + 3u).fetch_add(weight * p_mass);
        };
    });

    // Kernel: Grid velocity update
    auto simulate_grid = device.compile<3>([&] {
        set_block_size(8, 8, 1);
        Int3 coord = make_int3(dispatch_id().xyz());
        UInt i = index(coord);

        // Read velocity and mass
        Float4 v_and_m;
        for (int idx = 0; idx < 4; ++idx)
            v_and_m[idx] = grid->read(i * 4 + idx);
        Float3 v = v_and_m.xyz();
        Float m = v_and_m.w;

        // Normalize by mass
        v = ite(m > 0.f, v / m, v);

        // Apply gravity
        v.y -= dt * gravity;

        // Sticky boundary conditions (all 6 faces)
        v = ite((coord < bound && v < 0.f) || (coord > n_grid - bound && v > 0.f), 0.f, v);

        // Write back
        auto r = make_float4(v, m);
        for (int idx = 0; idx < 4; ++idx)
            grid->write(i * 4 + idx, r[idx]);
    });

    // Kernel: Grid to Particle transfer (G2P)
    auto grid_to_point = device.compile<1>([&] {
        set_block_size(64, 1, 1);
        UInt p = dispatch_id().x;

        Float3 Xp = x->read(p) / dx;
        Int3 base = make_int3(Xp - 0.5f);
        Float3 fx = Xp - make_float3(base);

        std::array w{0.5f * sqr(1.5f - fx),
                     0.75f - sqr(fx - 1.0f),
                     0.5f * sqr(fx - 0.5f)};

        Float3 new_v = def(make_float3(0.f));
        Float3x3 new_C = def(make_float3x3(0.f));

        // Gather from 3x3x3 neighbors
        for (uint ii = 0; ii < 27; ii++) {
            int3 offset = make_int3(ii % 3, ii / 3 % 3, ii / 3 / 3);
            int i = offset.x;
            int j = offset.y;
            int k = offset.z;

            Float3 dpos = (make_float3(offset) - fx) * dx;
            Float weight = w[i].x * w[j].y * w[k].z;
            UInt idx = index(base + offset);

            Float3 g_v;
            for (int i = 0; i < 3; ++i)
                g_v[i] = grid->read(idx * 4 + i);

            new_v += weight * g_v;
            new_C = new_C + 4.f * weight * outer_product(g_v, dpos) / sqr(dx);
        }

        // Update particle state
        v->write(p, new_v);
        x->write(p, x->read(p) + new_v * dt);
        J->write(p, J->read(p) * (1.f + dt * trace(new_C)));
        C->write(p, new_C);
    });

    auto substep = [&](CommandList &cmd_list) noexcept {
        cmd_list << clear_grid().dispatch(n_grid, n_grid, n_grid)
                 << point_to_grid().dispatch(n_particles)
                 << simulate_grid().dispatch(n_grid, n_grid, n_grid)
                 << grid_to_point().dispatch(n_particles);
    };

    // Initialize particles in a cube
    auto init = [&](Stream &stream) noexcept {
        luisa::vector<float3> x_init(n_particles);
        std::default_random_engine random{force_offline ? 42u : std::random_device{}()};
        std::uniform_real_distribution<float> uniform;
        for (uint i = 0; i < n_particles; i++) {
            float rx = uniform(random);
            float ry = uniform(random);
            float rz = uniform(random);
            x_init[i] = make_float3(rx * .4f + .2f, ry * .4f + .2f, rz * .4f + .2f);
        }
        luisa::vector<float3> v_init(n_particles, make_float3(0.f));
        luisa::vector<float> J_init(n_particles, 1.f);
        luisa::vector<float3x3> C_init(n_particles, make_float3x3(0.f));
        stream << x.copy_from(luisa::span{x_init})
               << v.copy_from(luisa::span{v_init})
               << J.copy_from(luisa::span{J_init})
               << C.copy_from(luisa::span{C_init})
               << synchronize();
    };

    // Clear display kernel
    auto clear_display = device.compile<2>([&] {
        display->write(dispatch_id().xy(),
                       make_float4(.1f, .2f, .3f, 1.f));
    });

    // Isometric projection parameters
    static constexpr float phi = radians(28);
    static constexpr float theta = radians(32);

    // Isometric projection transform
    auto T = [&](Float3 a0) noexcept {
        Float3 a = a0 - 0.5f;
        Float c = cos(phi);
        Float s = sin(phi);
        Float C = cos(theta);
        Float S = sin(theta);
        // Rotate around Y axis
        a.x = a.x * c + a.z * s;
        a.z = a.z * c - a.x * s;
        // Project to 2D
        return make_float2(a.x, a.y * C + a.z * S) + 0.5f;
    };

    // Draw particles with isometric projection
    auto draw_particles = device.compile<1>([&] {
        UInt p = dispatch_id().x;
        Float2 basepos = T(x->read(p));
        for (int i = -1; i <= 1; i++) {
            for (int j = -1; j <= 1; j++) {
                Int2 pos = make_int2(basepos * static_cast<float>(resolution)) + make_int2(i, j);
                $if (pos.x >= 0 & pos.x < resolution & pos.y >= 0 & pos.y < resolution) {
                    display->write(make_uint2(cast<uint>(pos.x), resolution - 1u - pos.y),
                                   make_float4(.4f, .6f, .6f, 1.f));
                };
            }
        }
    });

    // Run simulation
    init(stream);
    luisa::vector<std::array<uint8_t, 4u>> host_image(resolution * resolution);
    uint frame_count = 0u;
    Clock clock;
    if (force_offline) {
        for (uint f = 0u; f < user_frames; f++) {
            CommandList cmd_list;
            for (uint i = 0u; i < n_steps; i++) { substep(cmd_list); }
            cmd_list << clear_display().dispatch(resolution, resolution)
                     << draw_particles().dispatch(n_particles);
            stream << cmd_list.commit();
            frame_count++;
        }
        stream << synchronize();
        Buffer<float4> readback_buffer = device.create_buffer<float4>(resolution * resolution);
        luisa::vector<float4> host_float_image(resolution * resolution);
        stream << save_display_shader(display, readback_buffer, resolution).dispatch(resolution, resolution)
               << readback_buffer.copy_to(luisa::span{host_float_image})
               << synchronize();
        auto finite_output = true;
        for (uint i = 0u; i < resolution * resolution; i++) {
            auto pixel = host_float_image[i];
            auto pixel_is_finite = std::isfinite(pixel.x) &&
                                   std::isfinite(pixel.y) &&
                                   std::isfinite(pixel.z) &&
                                   std::isfinite(pixel.w);
            finite_output &= pixel_is_finite;
            if (!pixel_is_finite) {
                host_image[i] = {0u, 0u, 0u, 0u};
                continue;
            }
            host_image[i] = {
                static_cast<uint8_t>(std::clamp(pixel.x, 0.f, 1.f) * 255.f + 0.5f),
                static_cast<uint8_t>(std::clamp(pixel.y, 0.f, 1.f) * 255.f + 0.5f),
                static_cast<uint8_t>(std::clamp(pixel.z, 0.f, 1.f) * 255.f + 0.5f),
                static_cast<uint8_t>(std::clamp(pixel.w, 0.f, 1.f) * 255.f + 0.5f),
            };
        }
        stbi_write_png("test_mpm3d.png", resolution, resolution, 4, host_image.data(), 0);
        if (compare_path) {
            // P2G performs millions of unordered floating-point atomics per
            // substep. Their scheduling changes individual particle pixels
            // across backends even when the simulated distribution agrees.
            // Keep full-resolution image metrics as useful diagnostics, but
            // gate this simulation on resolution-independent foreground
            // occupancy and spatial moments. These tolerances are far above
            // observed atomic-order drift while still rejecting blank output,
            // missing evolution, wrong gravity, and materially bad indexing.
            // They admit 5% occupancy drift, a 2.5%-of-image (25.6 px here)
            // centroid shift, and 10% covariance-shape drift. On the audited
            // Vulkan result, the 16x16 density total variation was 0.0169 and
            // cosine similarity was 0.99946; limits of 0.06 and 0.99 leave
            // scheduler headroom without accepting redistributed mass.
            static constexpr std::array<uint8_t, 3u> background{
                26u, 51u, 77u};
            static constexpr luisa::ref::ForegroundMomentThresholds thresholds{
                .max_relative_count_error = 0.05,
                .max_centroid_distance = 0.025,
                .max_relative_covariance_error = 0.10,
                .max_density_total_variation = 0.06,
                .min_density_cosine_similarity = 0.99,
            };
            auto pixel_diagnostics = luisa::ref::compare_with_reference_file(
                reinterpret_cast<const uint8_t *>(host_image.data()),
                resolution, resolution, 4,
                *compare_path);
            auto distribution =
                luisa::ref::compare_foreground_moments_with_reference_file(
                    reinterpret_cast<const uint8_t *>(host_image.data()),
                    resolution, resolution, 4, *compare_path, background,
                    thresholds, 4u);
            auto passed = finite_output && distribution.passed;
            LUISA_INFO(
                "Reference comparison: {} (finite output={}; MPM3D foreground distribution: {}; full-resolution diagnostics only: {})",
                passed ? "PASSED" : "FAILED", finite_output,
                distribution.message, pixel_diagnostics.message);
            if (!passed) { return 1; }
        }
        LUISA_INFO("Saved offline rendering to test_mpm3d.png ({} frames)", user_frames);
    } else {
#if ENABLE_DISPLAY
        Framerate fps;
        while (!window->should_close()) {
            fps.record(1u);
            LUISA_INFO("FPS: {}", fps.report());
            CommandList cmd_list;
            for (uint i = 0u; i < n_steps; i++) { substep(cmd_list); }
            cmd_list << clear_display().dispatch(resolution, resolution)
                     << draw_particles().dispatch(n_particles);
            stream << cmd_list.commit() << swap_chain->present(display);
            window->poll_events();
            frame_count++;
        }
#endif
        stream << synchronize();
    }
}
