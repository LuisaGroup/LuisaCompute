/*
 * Tutorial 11: Schwarzschild Black Hole Renderer
 *
 * This tutorial teaches how to approximate gravitational lensing with a pure
 * compute shader pipeline in LuisaCompute.
 *
 * You will learn how to:
 *  - Set up physically inspired black-hole parameters.
 *  - Orbit a camera around the singularity.
 *  - Ray march light paths with Euler integration under a bending force.
 *  - Detect accretion-disk crossings on the XZ plane.
 *  - Shade the disk with temperature gradients and Doppler beaming.
 *  - Generate a procedural starfield and composite the final image.
 *
 * We do not solve the full general-relativistic geodesic equation here. Instead,
 * we use a compact, tutorial-friendly approximation where ray direction is bent
 * by an inverse-cubic acceleration toward the black hole. This is not physically
 * exact, but it captures the visual intuition: rays passing closer to the black
 * hole bend more strongly, causing lensing, rings, and warped background stars.
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

struct DiskHit {
    uint hit;
    float alpha;
    float3 color;
};

LUISA_STRUCT(CameraRay, origin, direction) {};
LUISA_STRUCT(DiskHit, hit, alpha, color) {};

namespace {

static constexpr uint image_width = 1024u;
static constexpr uint image_height = 1024u;
static constexpr uint max_march_steps = 300u;

static constexpr float bh_mass = 1.0f;
static constexpr float bh_radius = 2.0f;
static constexpr float photon_sphere = 3.0f;
static constexpr float accretion_inner = 6.0f;
static constexpr float accretion_outer = 15.0f;

}// namespace

int main(int argc, char *argv[]) {

    log_level_verbose();

    // Step 0: Parse arguments.
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

    LUISA_INFO("Tutorial 11 - Schwarzschild Black Hole");
    LUISA_INFO("Controls: left mouse drag = orbit camera, scroll = zoom");

    Stream stream = device.create_stream(force_offline ? StreamTag::COMPUTE : StreamTag::GRAPHICS);

    // Step 1: Generate camera rays from orbit parameters.
    // The camera circles the black hole at a controllable radius. Yaw changes the
    // azimuth around the Y axis, pitch changes elevation, and zoom changes orbit
    // radius. The camera always looks back toward the origin.
    Callable generate_camera_ray = [](UInt2 pixel, UInt2 resolution, Float yaw, Float pitch, Float distance) noexcept {
        Float2 uv = (make_float2(pixel) + 0.5f) / make_float2(resolution) * 2.0f - 1.0f;
        uv.y = -uv.y;

        Float aspect = cast<float>(resolution.x) / cast<float>(resolution.y);
        Float fov_scale = tan(radians(40.0f) * 0.5f);

        Float3 origin = distance * make_float3(
                                       cos(pitch) * sin(yaw),
                                       sin(pitch),
                                       cos(pitch) * cos(yaw));
        Float3 forward = normalize(-origin);
        Float3 right = normalize(cross(forward, make_float3(0.0f, 1.0f, 0.0f)));
        Float3 up = normalize(cross(right, forward));

        Float3 direction = normalize(forward +
                                     uv.x * aspect * fov_scale * right +
                                     uv.y * fov_scale * up);
        return def<CameraRay>(origin, direction);
    };

    // Step 2: Build a procedural starfield.
    // A skybox texture is unnecessary here. Instead we hash the ray direction to
    // create sparse bright points with slight color variation.
    Callable starfield = [](Float3 direction) noexcept {
        Float n1 = fract(sin(dot(direction, make_float3(127.1f, 311.7f, 191.999f))) * 43758.5453f);
        Float n2 = fract(sin(dot(direction, make_float3(269.5f, 183.3f, 246.1f))) * 18341.1357f);
        Float n3 = fract(sin(dot(direction, make_float3(419.2f, 371.9f, 128.4f))) * 9717.1230f);

        Float sparkle = ite(n1 > 0.9985f, 1.0f, 0.0f);
        sparkle = ite(n2 > 0.9992f, max(sparkle, 0.7f), sparkle);
        sparkle = ite(n3 > 0.9995f, max(sparkle, 0.5f), sparkle);

        Float3 tint = make_float3(1.0f,
                                  0.92f + 0.08f * n2,
                                  0.82f + 0.18f * n3);
        Float3 nebula = make_float3(0.01f, 0.015f, 0.03f) * pow(max(direction.y * 0.5f + 0.5f, 0.0f), 3.0f);
        return nebula + tint * sparkle;
    };

    // Step 3: Shade the accretion disk.
    // The disk lies on the XZ plane (y = 0). We detect when a ray segment crosses
    // that plane, compute the radial distance, and then derive a simple thermal
    // color. Inner radii are hotter, so they become whiter; outer radii cool down
    // toward orange-red. Doppler beaming brightens the side moving toward us.
    Callable shade_disk = [](Float3 intersection, Float3 ray_direction) noexcept {
        Float radius = length(make_float2(intersection.x, intersection.z));
        Var<DiskHit> result = def<DiskHit>(0u, 0.0f, make_float3(0.0f));

        $if (radius >= accretion_inner & radius <= accretion_outer) {
            Float radial = clamp((radius - accretion_inner) / (accretion_outer - accretion_inner), 0.0f, 1.0f);
            Float heat = 1.0f - radial;

            Float inner_fade = smoothstep(accretion_inner, accretion_inner + 1.2f, radius);
            Float outer_fade = 1.0f - smoothstep(accretion_outer - 1.5f, accretion_outer, radius);
            Float edge = inner_fade * outer_fade;

            Float3 warm = lerp(make_float3(0.85f, 0.18f, 0.05f),
                               make_float3(1.0f, 0.72f, 0.18f),
                               sqrt(heat));
            Float3 hot = lerp(warm, make_float3(1.0f, 0.97f, 0.90f), heat * heat);

            Float3 orbital_velocity = normalize(make_float3(-intersection.z, 0.0f, intersection.x));
            Float orbital_speed = sqrt(bh_mass / max(radius, accretion_inner));
            Float doppler = dot(orbital_velocity, -ray_direction);
            Float beaming = pow(max(0.25f, 1.0f + doppler * orbital_speed * 2.0f), 3.0f);

            // Gravitational redshift dims light emitted closer to the event horizon.
            Float gravitational_redshift = sqrt(max(0.05f, 1.0f - bh_radius / max(radius, bh_radius + 0.05f)));
            Float emissive = (0.4f + 1.8f * heat) * beaming * gravitational_redshift * edge;

            result = def<DiskHit>(1u,
                                  clamp(edge * (0.20f + 0.80f * heat), 0.0f, 1.0f),
                                  hot * emissive);
        };

        return result;
    };

    // Step 4: Render the black hole with Euler integration.
    // Every step bends the ray toward the origin using
    //     a = -1.5 * GM / r^3 * position
    // which is a compact approximation of Schwarzschild-style lensing. We then
    // advance the ray by an adaptive step size so far-away regions march faster
    // while near-horizon regions receive finer sampling.
    Kernel2D render_kernel = [&](ImageFloat image, Float yaw, Float pitch, Float distance) noexcept {
        set_block_size(16u, 16u, 1u);

        UInt2 pixel = dispatch_id().xy();
        UInt2 resolution = dispatch_size().xy();
        Var<CameraRay> camera_ray = generate_camera_ray(pixel, resolution, yaw, pitch, distance);

        Float3 position = camera_ray.origin;
        Float3 direction = camera_ray.direction;

        Float3 previous_position = position;
        Float3 disk_color = make_float3(0.0f);
        Float disk_alpha = 0.0f;
        Float ring_glow = 0.0f;
        UInt swallowed = 0u;

        $for (step, max_march_steps) {
            Float radius = length(position);

            $if (radius <= bh_radius) {
                swallowed = 1u;
                $break;
            };

            // Rays near the photon sphere are repeatedly bent, so we accumulate a
            // soft glow there to hint at the bright relativistic ring.
            ring_glow = max(ring_glow, exp(-abs(radius - photon_sphere) * 5.0f));

            Float step_size = clamp(radius * 0.04f, 0.03f, 0.25f);
            Float safe_radius = max(radius, bh_radius * 1.05f);
            Float3 acceleration = (-1.5f * bh_mass / (safe_radius * safe_radius * safe_radius)) * position;
            Float3 next_direction = normalize(direction + acceleration * step_size);
            Float3 next_position = position + next_direction * step_size;

            // Step 4.1: Detect disk crossings by looking for a sign change in y.
            // A ray segment that goes from y>0 to y<0 (or vice versa) crossed the
            // XZ plane somewhere in between.
            Bool crossed_disk = (position.y > 0.0f & next_position.y <= 0.0f) |
                                (position.y < 0.0f & next_position.y >= 0.0f);
            $if (crossed_disk & disk_alpha == 0.0f) {
                Float t = abs(position.y) / max(abs(position.y) + abs(next_position.y), 1e-4f);
                Float3 disk_position = lerp(position, next_position, t);
                Var<DiskHit> hit = shade_disk(disk_position, next_direction);
                $if (hit.hit != 0u) {
                    disk_alpha = hit.alpha;
                    disk_color = hit.color;
                };
            };

            previous_position = position;
            position = next_position;
            direction = next_direction;
        };

        // Step 5: Composite the final image.
        // Escaping rays sample the bent starfield direction. Rays swallowed by the
        // event horizon contribute only black, while the accretion disk is blended
        // over the background because it emits light on its own.
        Float3 background = ite(swallowed != 0u, make_float3(0.0f), starfield(direction));
        Float3 photon_ring = make_float3(1.0f, 0.88f, 0.70f) * ring_glow * 0.25f;
        Float3 color = background + photon_ring;
        color = lerp(color, disk_color, disk_alpha);

        // Simple tone mapping compresses HDR highlights into the displayable range.
        color = color / (1.0f + color);
        color = clamp(color, 0.0f, 1.0f);
        image.write(pixel, make_float4(color, 1.0f));
    };

    auto render_shader = device.compile(render_kernel);

#if ENABLE_DISPLAY
    std::unique_ptr<Window> window;
    std::optional<Swapchain> swap_chain;
    if (!force_offline) {
        window = std::make_unique<Window>("Tutorial 11 - Schwarzschild Black Hole", make_uint2(image_width, image_height));
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

    float yaw = 0.0f;
    float pitch = 0.18f;
    float zoom = 28.0f;

    if (force_offline) {
        // Step 6a: Offline mode writes a single PNG.
        luisa::vector<std::array<uint8_t, 4u>> host_image(image_width * image_height);
        stream << render_shader(output, yaw, pitch, zoom).dispatch(image_width, image_height)
               << output.copy_to(luisa::span{host_image})
               << synchronize();
        stbi_write_png("tutorial_11_black_hole.png", static_cast<int>(image_width), static_cast<int>(image_height), 4, host_image.data(), 0);
        LUISA_INFO("Saved offline render to tutorial_11_black_hole.png");
    } else {
#if ENABLE_DISPLAY
        // Step 6b: Interactive orbit controls.
        bool dragging = false;
        float2 last_cursor = make_float2(0.0f);

        window->set_mouse_callback([&dragging, &last_cursor](MouseButton button, Action action, float2 position) noexcept {
            if (button == MOUSE_BUTTON_LEFT) {
                dragging = action == ACTION_PRESSED;
                last_cursor = position;
            }
        });

        window->set_cursor_position_callback([&dragging, &last_cursor, &yaw, &pitch](float2 position) noexcept {
            if (!dragging) {
                last_cursor = position;
                return;
            }
            float2 delta = position - last_cursor;
            yaw += delta.x * 0.005f;
            pitch = clamp(pitch + delta.y * 0.005f, -1.1f, 1.1f);
            last_cursor = position;
        });

        window->set_scroll_callback([&zoom](float2 offset) noexcept {
            zoom = clamp(zoom * (1.0f - offset.y * 0.08f), 10.0f, 80.0f);
        });

        while (!window->should_close()) {
            window->poll_events();
            stream << render_shader(output, yaw, pitch, zoom).dispatch(image_width, image_height)
                   << swap_chain->present(output);
        }
        stream << synchronize();
#endif
    }

    return 0;
}
