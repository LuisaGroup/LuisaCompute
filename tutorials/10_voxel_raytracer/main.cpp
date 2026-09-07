/*
 * Tutorial 10: Voxel Ray Tracer
 *
 * This tutorial teaches how to build a complete voxel renderer with LuisaCompute.
 *
 * You will learn how to:
 *  - Store a 64^3 voxel world inside a linear Buffer<uint>.
 *  - Generate procedural terrain entirely on the GPU with a Kernel3D.
 *  - Intersect rays against a voxel volume with the AABB slab test.
 *  - Traverse grid cells with Digital Differential Analyzer (DDA) stepping.
 *  - Shade different voxel materials and orbit a camera around the scene.
 *  - Render interactively to a window or save a headless PNG with --offline.
 *
 * The key idea is that voxels are axis-aligned cubes arranged on an integer grid,
 * so the fastest traversal strategy is not sphere marching but exact cell-by-cell
 * stepping. DDA tells us which voxel boundary a ray reaches next, letting us walk
 * through the grid in strict front-to-back order until we hit a solid cell.
 */

#include <array>
#include <memory>
#include <optional>
#include <string_view>

#if __has_include(<stb/stb_image_write.h>)
#include <stb/stb_image_write.h>
#elif __has_include(<stb_image_write.h>)
#include <stb_image_write.h>
#endif

#include <luisa/luisa-compute.h>
#include <luisa/dsl/sugar.h>

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

struct CameraRay {
    float3 origin;
    float3 direction;
};

struct RayBoxHit {
    uint hit;
    float t_enter;
    float t_exit;
    float3 normal;
};

struct VoxelHit {
    uint hit;
    float t;
    uint voxel_type;
    float3 normal;
};

LUISA_STRUCT(CameraRay, origin, direction) {};
LUISA_STRUCT(RayBoxHit, hit, t_enter, t_exit, normal) {};
LUISA_STRUCT(VoxelHit, hit, t, voxel_type, normal) {};

namespace {

static constexpr uint voxel_grid_size = 64u;
static constexpr uint image_width = 1024u;
static constexpr uint image_height = 1024u;
static constexpr uint max_dda_steps = voxel_grid_size * 3u;

static constexpr uint voxel_empty = 0u;
static constexpr uint voxel_grass = 1u;
static constexpr uint voxel_dirt = 2u;
static constexpr uint voxel_stone = 3u;
static constexpr uint voxel_wood = 4u;
static constexpr uint voxel_leaf = 5u;

}// namespace

int main(int argc, char *argv[]) {

    log_level_verbose();

    // Step 0: Parse command line.
    // LuisaCompute backends are loaded by name, while the optional --offline
    // flag disables the window and saves a PNG instead, which is useful on CI
    // or remote machines. If no backend is given, the first installed backend
    // is selected automatically.
    luisa::string backend;
    bool force_offline = false;
    for (int i = 1; i < argc; i++) {
        if (std::string_view{argv[i]} == "--offline") {
            force_offline = true;
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
    Device device = context.create_device(backend);

#if !ENABLE_DISPLAY
    if (!force_offline) {
        LUISA_WARNING("GUI support is disabled in this build. Falling back to --offline mode.");
        force_offline = true;
    }
#endif

    LUISA_INFO("Tutorial 10 - Voxel Ray Tracer");
    LUISA_INFO("Controls: Arrow Keys = orbit camera, W/S = zoom");

    Stream stream = device.create_stream(force_offline ? StreamTag::COMPUTE : StreamTag::GRAPHICS);
    Buffer<uint> voxel_grid = device.create_buffer<uint>(voxel_grid_size * voxel_grid_size * voxel_grid_size);

    // Step 1: Define tiny helper callables.
    // A voxel grid is stored linearly in memory, so we flatten (x, y, z) to a
    // single integer index. This is exactly how 3D textures are often emulated
    // with buffers when explicit integer addressing is desired.
    Callable flatten_index = [](UInt3 cell) noexcept {
        return cell.x + voxel_grid_size * (cell.y + voxel_grid_size * cell.z);
    };

    // Step 2: Build a procedural terrain height function.
    // We add a few sine waves together. This is a cheap way to make rolling
    // hills: each wave contributes a frequency, and their sum creates larger
    // variation than a single sinusoid could provide alone.
    Callable terrain_height = [](UInt x, UInt z) noexcept {
        Float fx = cast<float>(x);
        Float fz = cast<float>(z);
        Float hills = 18.0f +
                      6.0f * sin(fx * 0.18f) +
                      5.0f * cos(fz * 0.16f) +
                      7.0f * sin((fx + fz) * 0.08f);
        return clamp(cast<int>(hills), 6, static_cast<int>(voxel_grid_size) - 2);
    };

    // Step 3: Create a deterministic hash for tree placement.
    // We want "random-looking" trees without storing any extra data. A hash
    // turns integer coordinates into a pseudo-random number that is stable from
    // frame to frame, so the same world is regenerated every time.
    Callable hash2d = [](UInt x, UInt z) noexcept {
        UInt h = x * 1973u + z * 9277u + 0x68bc21ebu;
        h ^= h >> 9u;
        h *= 0x85ebca6bu;
        h ^= h >> 13u;
        h *= 0xc2b2ae35u;
        h ^= h >> 16u;
        return h;
    };

    Callable tree_exists = [&terrain_height, &hash2d](UInt x, UInt z) noexcept {
        UInt seed = hash2d(x, z);
        Int h = terrain_height(x, z);
        return h > 16 & (seed % 29u == 0u);
    };

    Callable tree_height = [&hash2d](UInt x, UInt z) noexcept {
        return 4u + hash2d(x, z) % 3u;
    };

    // Step 4: Generate the voxel world with a Kernel3D.
    // Each thread owns exactly one voxel. That makes procedural generation easy:
    // classify the current cell as terrain, trunk, canopy, or empty space.
    Kernel3D generate_voxels = [&](BufferUInt voxels) noexcept {
        UInt3 cell = dispatch_id().xyz();
        UInt index = flatten_index(cell);

        Int ix = cast<int>(cell.x);
        Int iy = cast<int>(cell.y);
        Int iz = cast<int>(cell.z);
        Int h = terrain_height(cell.x, cell.z);
        UInt material = voxel_empty;

        // Step 4.1: Fill the terrain column.
        // The top layer is grass, a few layers underneath are dirt, and the
        // deeper foundation is stone. This mirrors how layered terrain is often
        // represented in block worlds.
        $if (iy < h) {
            $if (iy == h - 1) {
                material = voxel_grass;
            }
            $elif (iy >= h - 4) {
                material = voxel_dirt;
            }
            $else {
                material = voxel_stone;
            };
        };

        // Step 4.2: Scan nearby tree roots.
        // Because each voxel is generated independently, a leaf voxel needs to
        // know whether a tree exists close by. We only inspect a small 5x5 area,
        // because our canopy radius is bounded.
        $for (dz, 5u) {
            $for (dx, 5u) {
                Int root_x = ix + cast<int>(dx) - 2;
                Int root_z = iz + cast<int>(dz) - 2;

                $if (root_x >= 0 & root_x < static_cast<int>(voxel_grid_size) &
                     root_z >= 0 & root_z < static_cast<int>(voxel_grid_size)) {
                    UInt ux = cast<uint>(root_x);
                    UInt uz = cast<uint>(root_z);

                    $if (tree_exists(ux, uz)) {
                        Int root_h = terrain_height(ux, uz);
                        Int trunk_h = cast<int>(tree_height(ux, uz));
                        Int canopy_y = root_h + trunk_h;

                        // Tree trunks are thin vertical columns of wood.
                        $if (ix == root_x & iz == root_z & iy >= root_h & iy < root_h + trunk_h) {
                            material = voxel_wood;
                        }
                        $else {
                            // Leaves are modeled as a small Manhattan-distance blob.
                            // This produces a chunky block canopy that reads well in a
                            // voxel renderer and is cheap to evaluate.
                            Int canopy_dx = ix - root_x;
                            Int canopy_dy = iy - canopy_y;
                            Int canopy_dz = iz - root_z;
                            Bool inside_canopy = abs(canopy_dx) + abs(canopy_dy) + abs(canopy_dz) <= 3 &
                                                 canopy_dy >= -1 & canopy_dy <= 2;
                            $if (inside_canopy & material == voxel_empty) {
                                material = voxel_leaf;
                            };
                        };
                    };
                };
            };
        };

        voxels.write(index, material);
    };

    // Step 5: Intersect the ray with the voxel world's bounding box.
    // The slab method computes the entry/exit time interval along x/y/z and then
    // intersects those intervals. If all three overlap, the ray entered the box.
    Callable intersect_grid_box = [](Float3 ray_origin, Float3 ray_direction) noexcept {
        Float3 safe_dir = make_float3(
            ite(abs(ray_direction.x) < 1e-4f, ite(ray_direction.x >= 0.0f, 1e-4f, -1e-4f), ray_direction.x),
            ite(abs(ray_direction.y) < 1e-4f, ite(ray_direction.y >= 0.0f, 1e-4f, -1e-4f), ray_direction.y),
            ite(abs(ray_direction.z) < 1e-4f, ite(ray_direction.z >= 0.0f, 1e-4f, -1e-4f), ray_direction.z));
        Float3 inv_dir = 1.0f / safe_dir;

        Float3 box_min = make_float3(0.0f);
        Float3 box_max = make_float3(static_cast<float>(voxel_grid_size));
        Float3 t0 = (box_min - ray_origin) * inv_dir;
        Float3 t1 = (box_max - ray_origin) * inv_dir;
        Float3 t_near = min(t0, t1);
        Float3 t_far = max(t0, t1);

        Float t_enter = max(max(t_near.x, t_near.y), t_near.z);
        Float t_exit = min(min(t_far.x, t_far.y), t_far.z);

        Float3 normal = make_float3(0.0f);
        $if (t_near.x >= t_near.y & t_near.x >= t_near.z) {
            normal = make_float3(ite(ray_direction.x > 0.0f, -1.0f, 1.0f), 0.0f, 0.0f);
        }
        $elif (t_near.y >= t_near.z) {
            normal = make_float3(0.0f, ite(ray_direction.y > 0.0f, -1.0f, 1.0f), 0.0f);
        }
        $else {
            normal = make_float3(0.0f, 0.0f, ite(ray_direction.z > 0.0f, -1.0f, 1.0f));
        };

        UInt hit = ite(t_exit >= max(t_enter, 0.0f), 1u, 0u);
        return def<RayBoxHit>(hit, t_enter, t_exit, normal);
    };

    // Step 6: Walk the voxel grid with DDA.
    // Once the ray is inside the box, DDA keeps track of the next x/y/z grid
    // boundary. The smallest boundary time is the next voxel crossed by the ray.
    Callable trace_voxels = [&flatten_index, &intersect_grid_box](BufferUInt voxels, Float3 ray_origin, Float3 ray_direction) noexcept {
        auto box = intersect_grid_box(ray_origin, ray_direction);
        auto result = def<VoxelHit>(0u, 0.0f, voxel_empty, make_float3(0.0f));

        $if (box.hit != 0u) {
            Float3 safe_dir = make_float3(
                ite(abs(ray_direction.x) < 1e-4f, ite(ray_direction.x >= 0.0f, 1e-4f, -1e-4f), ray_direction.x),
                ite(abs(ray_direction.y) < 1e-4f, ite(ray_direction.y >= 0.0f, 1e-4f, -1e-4f), ray_direction.y),
                ite(abs(ray_direction.z) < 1e-4f, ite(ray_direction.z >= 0.0f, 1e-4f, -1e-4f), ray_direction.z));

            Float travel_t = max(box.t_enter, 0.0f);
            Float3 position = ray_origin + ray_direction * (travel_t + 1e-4f);
            auto floored = floor(position);
            Int3 cell = make_int3(cast<int>(floored.x), cast<int>(floored.y), cast<int>(floored.z));
            cell = max(cell, make_int3(0));
            cell = min(cell, make_int3(static_cast<int>(voxel_grid_size - 1u)));

            Int3 step = make_int3(
                ite(ray_direction.x >= 0.0f, 1, -1),
                ite(ray_direction.y >= 0.0f, 1, -1),
                ite(ray_direction.z >= 0.0f, 1, -1));

            Float3 delta_t = abs(1.0f / safe_dir);
            Float3 next_boundary = make_float3(
                ite(step.x > 0, cast<float>(cell.x + 1), cast<float>(cell.x)),
                ite(step.y > 0, cast<float>(cell.y + 1), cast<float>(cell.y)),
                ite(step.z > 0, cast<float>(cell.z + 1), cast<float>(cell.z)));
            Float3 next_t = (next_boundary - position) / safe_dir;
            Float3 current_normal = box.normal;

            $for (step_index, max_dda_steps) {
                Bool outside = cell.x < 0 | cell.x >= static_cast<int>(voxel_grid_size) |
                               cell.y < 0 | cell.y >= static_cast<int>(voxel_grid_size) |
                               cell.z < 0 | cell.z >= static_cast<int>(voxel_grid_size);
                $if (outside | travel_t > box.t_exit) {
                    $break;
                };

                UInt3 ucell = make_uint3(cast<uint>(cell.x), cast<uint>(cell.y), cast<uint>(cell.z));
                UInt voxel_type = voxels.read(flatten_index(ucell));
                $if (voxel_type != voxel_empty) {
                    result = def<VoxelHit>(1u, travel_t, voxel_type, current_normal);
                    $break;
                };

                $if (next_t.x <= next_t.y & next_t.x <= next_t.z) {
                    travel_t = next_t.x;
                    next_t.x += delta_t.x;
                    cell.x += step.x;
                    current_normal = make_float3(cast<float>(-step.x), 0.0f, 0.0f);
                }
                $elif (next_t.y <= next_t.z) {
                    travel_t = next_t.y;
                    next_t.y += delta_t.y;
                    cell.y += step.y;
                    current_normal = make_float3(0.0f, cast<float>(-step.y), 0.0f);
                }
                $else {
                    travel_t = next_t.z;
                    next_t.z += delta_t.z;
                    cell.z += step.z;
                    current_normal = make_float3(0.0f, 0.0f, cast<float>(-step.z));
                };
            };
        };

        return result;
    };

    // Step 7: Map material IDs to colors.
    // Using a callable keeps the shading logic separate from traversal, which is
    // exactly how larger renderers separate geometry from materials.
    Callable voxel_color = [](UInt voxel_type) noexcept {
        Float3 color = make_float3(0.0f);
        $if (voxel_type == voxel_grass) {
            color = make_float3(0.30f, 0.68f, 0.24f);
        }
        $elif (voxel_type == voxel_dirt) {
            color = make_float3(0.55f, 0.35f, 0.18f);
        }
        $elif (voxel_type == voxel_stone) {
            color = make_float3(0.52f, 0.54f, 0.58f);
        }
        $elif (voxel_type == voxel_wood) {
            color = make_float3(0.45f, 0.28f, 0.14f);
        }
        $else {
            color = make_float3(0.15f, 0.42f, 0.16f);
        };
        return color;
    };

    // Step 8: Generate a camera ray.
    // We use an orbit camera: yaw rotates around the vertical axis, pitch tilts
    // the camera up/down, and distance controls zoom. A basis (forward/right/up)
    // turns screen-space coordinates into a world-space ray.
    Callable generate_camera_ray = [](UInt2 pixel, UInt2 resolution, Float2 rotation, Float distance) noexcept {
        Float2 uv = (make_float2(pixel) + 0.5f) / make_float2(resolution) * 2.0f - 1.0f;
        uv.y = -uv.y;

        Float aspect = cast<float>(resolution.x) / cast<float>(resolution.y);
        Float fov_scale = tan(radians(55.0f) * 0.5f);

        Float yaw = rotation.x;
        Float pitch = rotation.y;
        Float3 target = make_float3(32.0f, 20.0f, 32.0f);
        Float3 origin = target + distance * make_float3(
                                                cos(pitch) * sin(yaw),
                                                sin(pitch),
                                                cos(pitch) * cos(yaw));

        Float3 forward = normalize(target - origin);
        Float3 right = normalize(cross(forward, make_float3(0.0f, 1.0f, 0.0f)));
        Float3 up = normalize(cross(right, forward));

        Float3 direction = normalize(forward +
                                     uv.x * aspect * fov_scale * right +
                                     uv.y * fov_scale * up);
        return def<CameraRay>(origin, direction);
    };

    // Step 9: Combine all pieces in the final render kernel.
    // For each pixel we shoot a ray, trace it through the voxel world, and then
    // apply simple Lambert lighting plus a sky gradient. This keeps the tutorial
    // focused on grid traversal instead of advanced material models.
    Kernel2D render_kernel = [&](ImageFloat image, BufferUInt voxels, Float2 camera_rotation, Float camera_distance) noexcept {
        set_block_size(16u, 16u, 1u);

        UInt2 pixel = dispatch_id().xy();
        UInt2 resolution = dispatch_size().xy();
        auto ray = generate_camera_ray(pixel, resolution, camera_rotation, camera_distance);
        auto hit = trace_voxels(voxels, ray.origin, ray.direction);

        Float horizon = clamp(ray.direction.y * 0.5f + 0.5f, 0.0f, 1.0f);
        Float3 sky = lerp(make_float3(0.72f, 0.88f, 1.00f),
                          make_float3(0.16f, 0.30f, 0.55f),
                          horizon);
        Float3 color = sky;

        $if (hit.hit != 0u) {
            Float3 base = voxel_color(hit.voxel_type);
            Float3 light_dir = normalize(make_float3(-0.45f, 0.85f, -0.30f));
            Float diffuse = max(dot(hit.normal, light_dir), 0.0f);
            Float ambient = 0.22f;

            // Fog slightly brightens distant hits and helps explain depth.
            Float fog = exp(-hit.t * 0.03f);
            Float3 lit = base * (ambient + 0.78f * diffuse);
            color = lerp(sky, lit, fog);
        };

        image.write(pixel, make_float4(clamp(color, 0.0f, 1.0f), 1.0f));
    };

    auto terrain_shader = device.compile(generate_voxels);
    auto render_shader = device.compile(render_kernel);

    // Step 10: Run terrain generation once.
    // Because the world is static, we only need to fill the voxel buffer during
    // startup. The renderer then reuses the same buffer every frame.
    stream << terrain_shader(voxel_grid).dispatch(voxel_grid_size, voxel_grid_size, voxel_grid_size)
           << synchronize();

#if ENABLE_DISPLAY
    std::unique_ptr<Window> window;
    std::optional<Swapchain> swap_chain;
    if (!force_offline) {
        window = std::make_unique<Window>("Tutorial 10 - Voxel Ray Tracer", make_uint2(image_width, image_height));
        swap_chain.emplace(device.create_swapchain(
            stream,
            SwapchainOption{
                .display = window->native_display(),
                .window = window->native_handle(),
                .size = make_uint2(image_width, image_height),
                .wants_vsync = true,
            }));
    }
    Image<float> output = device.create_image<float>(
        (!force_offline && swap_chain.has_value()) ? swap_chain->backend_storage() : PixelStorage::BYTE4,
        image_width, image_height);
#else
    Image<float> output = device.create_image<float>(PixelStorage::BYTE4, image_width, image_height);
#endif

    float yaw = 0.75f;
    float pitch = -0.35f;
    float distance = 78.0f;

    if (force_offline) {
        // Step 11a: Headless rendering path.
        // Copying a BYTE4 image back to the host gives us PNG-ready bytes.
        luisa::vector<std::array<uint8_t, 4u>> host_image(image_width * image_height);
        stream << render_shader(output, voxel_grid, make_float2(yaw, pitch), distance).dispatch(image_width, image_height)
               << output.copy_to(luisa::span{host_image})
               << synchronize();
        stbi_write_png("tutorial_10_voxel_raytracer.png", static_cast<int>(image_width), static_cast<int>(image_height), 4, host_image.data(), 0);
        LUISA_INFO("Saved offline render to tutorial_10_voxel_raytracer.png");
    } else {
#if ENABLE_DISPLAY
        // Step 11b: Interactive rendering path.
        // The camera is updated on the CPU, then the GPU consumes the new orbit
        // parameters every frame.
        while (!window->should_close()) {
            window->poll_events();

            if (window->is_key_down(KEY_LEFT)) {
                yaw -= 0.02f;
            }
            if (window->is_key_down(KEY_RIGHT)) {
                yaw += 0.02f;
            }
            if (window->is_key_down(KEY_UP)) {
                pitch = clamp(pitch + 0.02f, -1.15f, 0.75f);
            }
            if (window->is_key_down(KEY_DOWN)) {
                pitch = clamp(pitch - 0.02f, -1.15f, 0.75f);
            }
            if (window->is_key_down(KEY_W)) {
                distance = max(distance - 1.0f, 28.0f);
            }
            if (window->is_key_down(KEY_S)) {
                distance = min(distance + 1.0f, 120.0f);
            }

            stream << render_shader(output, voxel_grid, make_float2(yaw, pitch), distance).dispatch(image_width, image_height)
                   << swap_chain->present(output);
        }
        stream << synchronize();
#endif
    }

    return 0;
}
