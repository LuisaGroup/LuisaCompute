// Voxel Ray Tracer
// A real-time ray tracer through a 3D voxel grid using ray-box intersection
// and Digital Differential Analyzer (DDA) traversal. Renders a procedural
// voxel scene with ambient occlusion and simple lighting.
//
// Features demonstrated:
// - 3D voxel grid storage and access
// - Ray-box intersection testing
// - DDA algorithm for voxel traversal
// - Ambient occlusion approximation
// - Interactive camera controls

#include <luisa/core/clock.h>
#include <luisa/core/logging.h>
#include <luisa/runtime/context.h>
#include <luisa/runtime/device.h>
#include <luisa/runtime/stream.h>
#include <luisa/runtime/buffer.h>
#include <luisa/runtime/swapchain.h>
#include <luisa/dsl/sugar.h>
#include <luisa/gui/window.h>
#include <stb/stb_image_write.h>
#include <filesystem>
#include <memory>
#include <optional>
#include <string_view>

#include "common/reference_compare.h"

using namespace luisa;
using namespace luisa::compute;

// Maximum ray traversal steps
static constexpr uint max_steps = 256u;
static constexpr uint voxel_grid_size = 64u;

// Note: luisa::compute::Ray is already defined in the framework

int main(int argc, char *argv[]) {

    Context context{argv[0]};
    if (argc <= 1) {
        LUISA_INFO("Usage: {} <backend>. <backend>: cuda, dx, cpu, metal", argv[0]);
        exit(1);
    }
    Device device = context.create_device(argv[1]);
    LUISA_INFO("Voxel Ray Tracer");
    LUISA_INFO("Controls: Arrow Keys = Rotate, W/S = Zoom, R = Reset view");

    auto opts = luisa::ref::ExampleOptions::parse(argc, argv);
    if (!opts.valid()) {
        LUISA_WARNING("Invalid command line: {}", opts.error_message);
        return 1;
    }
    auto force_offline = opts.offline;
    auto compare_path = opts.compare_path;

    // Image dimensions
    static constexpr uint width = 1024u;
    static constexpr uint height = 1024u;

    Stream stream = device.create_stream(force_offline ? StreamTag::COMPUTE : StreamTag::GRAPHICS);

    // Create voxel grid buffer (1 = solid, 0 = empty)
    // Using a 64x64x64 grid
    static constexpr uint grid_size = 64u;
    static constexpr uint total_voxels = grid_size * grid_size * grid_size;
    Buffer<uint> voxel_grid = device.create_buffer<uint>(total_voxels);

    // Initialize voxel grid with procedural scene
    luisa::vector<uint> host_voxels(total_voxels, 0u);

    // Helper to set voxel
    auto set_voxel = [&](int x, int y, int z, uint type) {
        if (x >= 0 && x < (int)grid_size &&
            y >= 0 && y < (int)grid_size &&
            z >= 0 && z < (int)grid_size) {
            host_voxels[(z * grid_size + y) * grid_size + x] = type;
        }
    };

    // Create a procedural scene: terrain with some structures
    for (int x = 0; x < (int)grid_size; x++) {
        for (int z = 0; z < (int)grid_size; z++) {
            // Base terrain using simple sine waves
            int terrain_height = 10 + (int)(sin(x * 0.15f) * 4.0f +
                                            cos(z * 0.15f) * 4.0f +
                                            sin(x * 0.05f + z * 0.05f) * 8.0f);
            terrain_height = clamp(terrain_height, 2, (int)grid_size - 2);

            // Fill terrain
            for (int y = 0; y < terrain_height; y++) {
                uint type = 1;                        // Ground
                if (y >= terrain_height - 2) type = 2;// Grass top
                if (y < 5) type = 3;                  // Stone bottom
                set_voxel(x, y, z, type);
            }

            // Trees (random placement)
            if ((x * 7 + z * 13) % 47 == 0 && terrain_height > 15) {
                // Tree trunk
                int tree_height = 5 + (x + z) % 4;
                for (int h = 0; h < tree_height; h++) {
                    set_voxel(x, terrain_height + h, z, 4);// Wood
                }
                // Tree leaves
                for (int lx = -2; lx <= 2; lx++) {
                    for (int ly = -1; ly <= 2; ly++) {
                        for (int lz = -2; lz <= 2; lz++) {
                            if (abs(lx) + abs(ly) + abs(lz) <= 3) {
                                set_voxel(x + lx, terrain_height + tree_height + ly, z + lz, 5);// Leaves
                            }
                        }
                    }
                }
            }
        }
    }

    // Add a floating sphere
    int sphere_cx = 32, sphere_cy = 45, sphere_cz = 32;
    int sphere_radius = 8;
    for (int x = -sphere_radius; x <= sphere_radius; x++) {
        for (int y = -sphere_radius; y <= sphere_radius; y++) {
            for (int z = -sphere_radius; z <= sphere_radius; z++) {
                if (x * x + y * y + z * z <= sphere_radius * sphere_radius) {
                    set_voxel(sphere_cx + x, sphere_cy + y, sphere_cz + z, 6);// Magic orb
                }
            }
        }
    }

    stream << voxel_grid.copy_from(luisa::span{host_voxels}) << synchronize();

    // Main rendering kernel
    Kernel2D render_kernel = [&](ImageFloat image, Float3 cam_pos,
                                 Float2 cam_rot, BufferVar<uint> voxels) noexcept {
        set_block_size(16, 16, 1);
        Var uv = dispatch_id().xy();
        Var size = dispatch_size().xy();

        // Normalized device coordinates
        Var ndc = (make_float2(uv) / make_float2(size)) * 2.0f - 1.0f;
        ndc.y = -ndc.y;// Flip Y

        // Camera setup
        Var aspect = cast<float>(size.x) / cast<float>(size.y);
        Var fov = 0.8f;

        // Ray direction in camera space
        Var ray_dir_cam = normalize(make_float3(
            ndc.x * aspect * fov,
            ndc.y * fov,
            -1.0f));

        // Apply camera rotation
        Var cos_y = cos(cam_rot.x);
        Var sin_y = sin(cam_rot.x);
        Var cos_x = cos(cam_rot.y);
        Var sin_x = sin(cam_rot.y);

        // Rotate around Y (yaw)
        Var x1 = ray_dir_cam.x * cos_y - ray_dir_cam.z * sin_y;
        Var z1 = ray_dir_cam.x * sin_y + ray_dir_cam.z * cos_y;

        // Rotate around X (pitch)
        Var y2 = ray_dir_cam.y * cos_x - z1 * sin_x;
        Var z2 = ray_dir_cam.y * sin_x + z1 * cos_x;

        Var ray_dir = normalize(make_float3(x1, y2, z2));
        Var ray_origin = cam_pos;

        // Grid bounds
        Var grid_min = make_float3(0.0f);
        Var grid_max = make_float3(cast<float>(grid_size));

        // Ray-box intersection (slab method)
        Var tmin = (grid_min - ray_origin) / ray_dir;
        Var tmax = (grid_max - ray_origin) / ray_dir;

        Var tmin_new = min(tmin, tmax);
        Var tmax_new = max(tmin, tmax);

        Var t_enter = max(max(tmin_new.x, tmin_new.y), tmin_new.z);
        Var t_exit = min(min(tmax_new.x, tmax_new.y), tmax_new.z);

        // Sky gradient (default color)
        Var sky_color = lerp(
            make_float3(0.5f, 0.7f, 1.0f),
            make_float3(0.1f, 0.2f, 0.4f),
            ray_dir.y * 0.5f + 0.5f);

        Var final_color = sky_color;

        // Check if ray hits grid
        $if (t_enter <= t_exit & t_exit > 0.0f) {
            // Start point
            Var t = ite(t_enter > 0.0f, t_enter, 0.0f);
            Var p = ray_origin + ray_dir * (t + 0.001f);// Small offset to avoid self-intersection

            // Current voxel coordinates
            Var vx = floor(p.x);
            Var vy = floor(p.y);
            Var vz = floor(p.z);

            // Step direction
            Var step_x = ite(ray_dir.x > 0.0f, 1, -1);
            Var step_y = ite(ray_dir.y > 0.0f, 1, -1);
            Var step_z = ite(ray_dir.z > 0.0f, 1, -1);

            // Delta T per step
            Var delta_tx = abs(1.0f / ray_dir.x);
            Var delta_ty = abs(1.0f / ray_dir.y);
            Var delta_tz = abs(1.0f / ray_dir.z);

            // Initial T to next boundary
            Var next_tx = ite(ray_dir.x > 0.0f,
                              (vx + 1.0f - p.x) / ray_dir.x,
                              (vx - p.x) / ray_dir.x);
            Var next_ty = ite(ray_dir.y > 0.0f,
                              (vy + 1.0f - p.y) / ray_dir.y,
                              (vy - p.y) / ray_dir.y);
            Var next_tz = ite(ray_dir.z > 0.0f,
                              (vz + 1.0f - p.z) / ray_dir.z,
                              (vz - p.z) / ray_dir.z);

            // Traverse
            Var hit = def(false);
            Var hit_color = make_float3(1.0f);
            Var hit_step = def(0u);

            $for (step, max_steps) {
                // Check current voxel
                Var ix = cast<int>(vx);
                Var iy = cast<int>(vy);
                Var iz = cast<int>(vz);

                // Check bounds
                $if ((ix < 0) | (ix >= cast<int>(grid_size)) |
                     (iy < 0) | (iy >= cast<int>(grid_size)) |
                     (iz < 0) | (iz >= cast<int>(grid_size))) {
                    $break;
                };

                // Read voxel
                Var voxel_idx = (iz * cast<int>(grid_size) + iy) * cast<int>(grid_size) + ix;
                Var voxel_type = voxels.read(voxel_idx);

                // Hit!
                $if (voxel_type > 0u) {
                    // Material colors based on voxel type
                    $if (voxel_type == 1u) {
                        hit_color = make_float3(0.6f, 0.4f, 0.3f);// Dirt
                    }
                    $elif (voxel_type == 2u) {
                        hit_color = make_float3(0.2f, 0.7f, 0.2f);// Grass
                    }
                    $elif (voxel_type == 3u) {
                        hit_color = make_float3(0.5f, 0.5f, 0.5f);// Stone
                    }
                    $elif (voxel_type == 4u) {
                        hit_color = make_float3(0.4f, 0.25f, 0.1f);// Wood
                    }
                    $elif (voxel_type == 5u) {
                        hit_color = make_float3(0.1f, 0.5f, 0.1f);// Leaves
                    }
                    $else {
                        hit_color = make_float3(0.8f, 0.3f, 0.9f);// Magic orb
                    };

                    hit_step = step;
                    hit = true;
                    $break;
                };

                // Step to next voxel
                $if (next_tx < next_ty & next_tx < next_tz) {
                    vx = vx + cast<float>(step_x);
                    next_tx = next_tx + delta_tx;
                }
                $else {
                    $if (next_ty < next_tz) {
                        vy = vy + cast<float>(step_y);
                        next_ty = next_ty + delta_ty;
                    }
                    $else {
                        vz = vz + cast<float>(step_z);
                        next_tz = next_tz + delta_tz;
                    };
                };

                // Max distance check
                $if (t > 200.0f) {
                    $break;
                };
            };

            // Compute lighting if we hit something
            $if (hit) {
                // Simple ambient occlusion based on traversal steps
                Var ao = 1.0f - (cast<float>(hit_step) / cast<float>(max_steps)) * 0.5f;

                // Simple directional light
                Var light_dir = normalize(make_float3(0.5f, 1.0f, 0.3f));
                Var normal = normalize(make_float3(
                    ite(next_tx < next_ty & next_tx < next_tz, cast<float>(-step_x), 0.0f),
                    ite((next_ty <= next_tx | next_tx >= next_tz) & next_ty < next_tz, cast<float>(-step_y), 0.0f),
                    ite(next_tz <= next_ty & (next_tx >= next_ty | next_tz <= next_tx), cast<float>(-step_z), 0.0f)));

                Var diff = max(dot(normal, light_dir), 0.0f);
                final_color = hit_color * (0.3f + 0.7f * diff) * ao;
            };
        };

        image.write(uv, make_float4(final_color, 1.0f));
    };

    auto render = device.compile(render_kernel);

    // Setup window
    std::unique_ptr<Window> window;
    std::optional<Swapchain> swap_chain;
    if (!force_offline) {
        window = std::make_unique<Window>("Voxel Ray Tracer", make_uint2(width, height));
        swap_chain.emplace(device.create_swapchain(
            stream,
            SwapchainOption{
                .display = window->native_display(),
                .window = window->native_handle(),
                .size = window->size(),
                .wants_vsync = true,
            }));
    }
    Image<float> display = device.create_image<float>(
        (!force_offline && swap_chain.has_value()) ? swap_chain->backend_storage() : PixelStorage::BYTE4,
        make_uint2(width, height));

    // Camera state
    float3 cam_pos{32.0f, 40.0f, 80.0f};
    float2 cam_rot{0.0f, -0.3f};

    if (force_offline) {
        luisa::vector<std::array<uint8_t, 4u>> host_image(width * height);
        stream << render(display, cam_pos, cam_rot, voxel_grid).dispatch(width, height)
               << display.copy_to(luisa::span{host_image})
               << synchronize();
        stbi_write_png("test_voxel_raytracer.png", width, height, 4, host_image.data(), 0);
        if (compare_path) {
            auto result = luisa::ref::compare_with_reference_file(
                reinterpret_cast<const uint8_t *>(host_image.data()),
                width, height, 4,
                *compare_path);
            LUISA_INFO("Reference comparison: {} ({})", result.passed ? "PASSED" : "FAILED", result.message);
            if (!result.passed) { return 1; }
        }
    } else {
        // Main loop
        while (!window->should_close()) {
            window->poll_events();

            // Reset view
            if (window->is_key_down(KEY_R)) {
                cam_pos = make_float3(32.0f, 40.0f, 80.0f);
                cam_rot = make_float2(0.0f, -0.3f);
            }

            // Camera controls
            if (window->is_key_down(KEY_LEFT)) {
                cam_rot.x -= 0.02f;
            }
            if (window->is_key_down(KEY_RIGHT)) {
                cam_rot.x += 0.02f;
            }
            if (window->is_key_down(KEY_UP)) {
                cam_rot.y = min(cam_rot.y + 0.02f, 1.5f);
            }
            if (window->is_key_down(KEY_DOWN)) {
                cam_rot.y = max(cam_rot.y - 0.02f, -1.5f);
            }
            if (window->is_key_down(KEY_W)) {
                cam_pos.z -= 1.0f;
            }
            if (window->is_key_down(KEY_S)) {
                cam_pos.z += 1.0f;
            }

            // Render
            stream << render(display, cam_pos, cam_rot, voxel_grid).dispatch(width, height)
                   << swap_chain->present(display);
        }

        stream << synchronize();
    }
    return 0;
}
