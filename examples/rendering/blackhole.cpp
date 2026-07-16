// Black Hole Renderer - Interstellar Style
// Renders a Schwarzschild black hole with gravitational lensing and an accretion disk.
// Features relativistic effects including light bending, Doppler beaming, and gravitational redshift.
//
// Features demonstrated:
// - Gravitational lensing via ray tracing with curved light paths
// - Accretion disk with temperature-based emission
// - Doppler shifting (redshift/blueshift) from orbital motion
// - Interactive camera controls

#include <luisa/core/clock.h>
#include <luisa/core/logging.h>
#include <luisa/runtime/context.h>
#include <luisa/runtime/device.h>
#include <luisa/runtime/stream.h>
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

// Black hole parameters
static constexpr float bh_mass = 1.0f;         // Solar masses (scaled)
static constexpr float bh_radius = 2.0f;       // Schwarzschild radius (Rs = 2GM/c²)
static constexpr float photon_sphere = 3.0f;   // 1.5 * Rs
static constexpr float accretion_inner = 6.0f; // Inner edge of accretion disk
static constexpr float accretion_outer = 15.0f;// Outer edge of accretion disk

int main(int argc, char *argv[]) {

    Context context{argv[0]};
    if (argc <= 1) {
        LUISA_INFO("Usage: {} <backend>. <backend>: cuda, dx, cpu, metal", argv[0]);
        exit(1);
    }
    Device device = context.create_device(argv[1]);
    LUISA_INFO("Black Hole Renderer - Interstellar Style");

    // Parse --offline flag
    bool force_offline = false;
    std::optional<std::filesystem::path> compare_path;
    for (int i = 2; i < argc; i++) {
        if (std::string_view{argv[i]} == "--offline") {
            force_offline = true;
        } else if ((std::string_view{argv[i]} == "--compare" || std::string_view{argv[i]} == "-c") && i + 1 < argc) {
            compare_path = std::filesystem::path{argv[++i]};
            force_offline = true;
        }
    }

    if (!force_offline) {
        LUISA_INFO("Controls: Left drag = Orbit, Right drag = Roll, Scroll/+/- = Zoom, ESC = Quit");
    } else {
        LUISA_INFO("Running in offline mode");
    }

    // Image dimensions
    static constexpr uint width = 1024u;
    static constexpr uint height = 1024u;

    Stream stream = device.create_stream(force_offline ? StreamTag::COMPUTE : StreamTag::GRAPHICS);

    // Camera state
    float rot_x = 0.2f;// Tilt angle
    float rot_y = 0.0f;// Rotation around black hole
    float zoom = 30.0f;// Camera distance
    bool left_mouse_down = false;
    bool right_mouse_down = false;
    float2 last_mouse_pos{0.0f, 0.0f};

    // Roll rotation for right mouse drag (rotate around image plane/view direction)
    float roll_angle = 0.0f;

    // Setup window and swapchain conditionally
    std::unique_ptr<Window> window;
    std::optional<Swapchain> swap_chain;
    if (!force_offline) {
        window = std::make_unique<Window>("Black Hole - Interstellar Style", make_uint2(width, height));

        window->set_mouse_callback([&left_mouse_down, &right_mouse_down, &last_mouse_pos](MouseButton button, Action a, float2 p) noexcept {
            if (a == Action::ACTION_PRESSED) {
                if (button == MOUSE_BUTTON_LEFT) {
                    left_mouse_down = true;
                } else if (button == MOUSE_BUTTON_RIGHT) {
                    right_mouse_down = true;
                }
                last_mouse_pos = p;
            } else if (a == Action::ACTION_RELEASED) {
                if (button == MOUSE_BUTTON_LEFT) {
                    left_mouse_down = false;
                } else if (button == MOUSE_BUTTON_RIGHT) {
                    right_mouse_down = false;
                }
            }
        });

        window->set_cursor_position_callback([&left_mouse_down, &right_mouse_down, &last_mouse_pos, &rot_x, &rot_y, &roll_angle](float2 p) noexcept {
            if (left_mouse_down) {
                // Left mouse: orbit around black hole
                float dx = p.x - last_mouse_pos.x;
                float dy = p.y - last_mouse_pos.y;
                rot_y += dx * 0.005f;
                rot_x += dy * 0.005f;
                rot_x = clamp(rot_x, -1.0f, 1.0f);
                last_mouse_pos = p;
            } else if (right_mouse_down) {
                // Right mouse: roll/rotate around view direction (image plane rotation)
                float dx = p.x - last_mouse_pos.x;
                float dy = p.y - last_mouse_pos.y;
                // Roll based on circular motion around center of screen
                roll_angle += (dx * 0.005f + dy * 0.005f);
                last_mouse_pos = p;
            }
        });

        window->set_scroll_callback([&zoom](float2 offset) noexcept {
            zoom *= (1.0f - offset.y * 0.1f);
            zoom = clamp(zoom, 10.0f, 100.0f);
        });

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

    // Black hole rendering kernel with smooth edges
    Kernel2D blackhole_kernel = [&](ImageFloat image, Float rot_x, Float rot_y, Float roll_angle, Float cam_distance, Float time) noexcept {
        set_block_size(16, 16, 1);
        Var uv = dispatch_id().xy();
        Var size = dispatch_size().xy();

        // Normalized coordinates (-1 to 1)
        Var ndc = (make_float2(uv) / make_float2(size)) * 2.0f - 1.0f;
        ndc.y = -ndc.y;// Flip Y
        Var aspect = cast<float>(size.x) / cast<float>(size.y);
        ndc.x *= aspect;

        // Camera setup
        Var fov = 0.6f;

        // Ray origin (camera position)
        Var cam_pos = make_float3(
            cam_distance * sin(rot_y) * cos(rot_x),
            cam_distance * sin(rot_x),
            cam_distance * cos(rot_y) * cos(rot_x));

        // Look at origin
        Var forward = normalize(-cam_pos);
        Var right = normalize(cross(forward, make_float3(0.0f, 1.0f, 0.0f)));
        Var up = cross(right, forward);

        // Apply roll rotation around view direction
        Var cos_roll = cos(roll_angle);
        Var sin_roll = sin(roll_angle);
        Var new_right = right * cos_roll + up * sin_roll;
        Var new_up = -right * sin_roll + up * cos_roll;
        right = new_right;
        up = new_up;

        // Ray direction through pixel
        Var ray_dir = normalize(forward + ndc.x * right * fov + ndc.y * up * fov);

        // Ray marching with gravitational bending
        Var pos = cam_pos;
        Var dir = ray_dir;
        Var color = make_float3(0.0f);
        Var hit_disk = def(false);
        Var disk_alpha = def(0.0f);
        Var disk_color = def(make_float3(0.0f));

        // Background starfield (procedural)
        auto get_star_color = [&](Float3 rd) noexcept {
            // Multiple octaves of noise for better star distribution
            Var star1 = fract(sin(rd.x * 123.456f + rd.y * 234.567f + rd.z * 345.678f) * 43758.5453f);
            Var star2 = fract(sin(rd.x * 456.789f + rd.y * 567.890f + rd.z * 678.901f) * 23421.1234f);
            Var star3 = fract(sin(rd.x * 789.012f + rd.y * 890.123f + rd.z * 901.234f) * 54321.6789f);

            // Different brightness thresholds for variety
            Var brightness = ite(star1 > 0.997f, 0.9f, 0.0f);
            brightness = ite((star2 > 0.998f) & (brightness < 0.1f), 0.7f, brightness);
            brightness = ite((star3 > 0.999f) & (brightness < 0.1f), 0.5f, brightness);

            // Subtle twinkle
            Var twinkle = 0.9f + 0.1f * sin(rd.x * 5.0f + time * 0.5f + rd.y * 3.0f);

            // Star color variation (blue-white to yellow-white)
            Var star_color = make_float3(
                1.0f,
                0.95f + 0.05f * star1,
                0.8f + 0.2f * star2);

            return star_color * brightness * twinkle;
        };

        // Gravitational ray marching
        static constexpr int max_steps = 300;
        static constexpr float dt = 0.25f;

        // Track previous position for disk intersection
        Var prev_pos = pos;
        Var prev_y = cam_pos.y;

        // We need to sample the disk at the correct depth order
        // The disk has two sides: back (hit first) and front (hit second)
        // We only want to show the front side that's facing the camera
        Var back_disk_sampled = def(false);
        Var back_disk_color = def(make_float3(0.0f));
        Var back_disk_alpha = def(0.0f);

        $for (step, max_steps) {
            // Distance from black hole center
            Var r = length(pos);

            // Check if we hit the event horizon
            $if (r < bh_radius * 1.01f) {
                $break;// Absorbed by black hole
            };

            // Check for accretion disk intersection (crossing the XZ plane)
            // The disk is a thin sheet at y=0
            Var crossed_plane = (prev_y > 0.0f & pos.y <= 0.0f) | (prev_y < 0.0f & pos.y >= 0.0f);

            $if (crossed_plane & r > accretion_inner & r < accretion_outer * 1.1f) {
                // We crossed the disk plane - interpolate to find exact intersection
                Var t_cross = abs(prev_y) / (abs(prev_y) + abs(pos.y));
                Var intersect_pos = prev_pos * (1.0f - t_cross) + pos * t_cross;
                Var intersect_r = length(intersect_pos);

                // Smooth edge falloff for inner and outer boundaries
                Var inner_falloff = smoothstep(accretion_inner * 0.9f, accretion_inner, intersect_r);
                Var outer_falloff = smoothstep(accretion_outer * 1.1f, accretion_outer, intersect_r);
                Var edge_smooth = inner_falloff * outer_falloff;

                $if (edge_smooth > 0.01f) {
                    // Orbital velocity at this radius (Keplerian)
                    Var orbital_speed = sqrt(bh_mass / max(intersect_r, 0.1f));

                    // Disk temperature profile (T ~ r^(-3/4))
                    Var temp = 2.0f * pow(accretion_outer / max(intersect_r, 0.1f), 0.75f);

                    // Doppler effect from orbital motion
                    Var orbital_angle = atan2(intersect_pos.z, intersect_pos.x);
                    Var orbital_dir = make_float3(-sin(orbital_angle), 0.0f, cos(orbital_angle));

                    // Doppler shift: approaching = blueshift (brighter), receding = redshift (dimmer)
                    Var doppler = dot(orbital_dir, dir);
                    Var beaming = pow(1.0f + doppler * orbital_speed * 2.0f, 2.0f);

                    // Temperature shifted by Doppler
                    Var shifted_temp = temp * (1.0f + doppler * orbital_speed);

                    // Blackbody color approximation with smooth gradients
                    // Hot = white/blue, Cool = orange/red
                    Var t_clamped = clamp(shifted_temp, 0.0f, 3.0f);
                    Var disk_r = ite(t_clamped > 1.0f, 1.0f, t_clamped * 0.8f + 0.2f);
                    Var disk_g = ite(t_clamped > 1.5f, 1.0f, t_clamped * 0.6f);
                    Var disk_b = ite(t_clamped > 2.0f, 1.0f, t_clamped * 0.4f);

                    Var local_disk_color = make_float3(disk_r, disk_g, disk_b);

                    // === FLOWING TEXTURE FOR ACCRETION DISK ===
                    // Create swirling pattern that rotates with time
                    Var normalized_r = (intersect_r - accretion_inner) / (accretion_outer - accretion_inner);
                    Var spiral_angle = orbital_angle * 3.0f + normalized_r * 10.0f - time * 0.5f;

                    // Multiple spiral arms
                    Var arm_pattern = sin(spiral_angle) * 0.5f + 0.5f;
                    Var fine_detail = sin(spiral_angle * 2.0f + time) * 0.3f + 0.7f;

                    // Radial variations (clumps of matter falling in)
                    Var radial_wave = sin(normalized_r * 20.0f + time * 0.3f) * 0.5f + 0.5f;

                    // Turbulence/noise-like variation
                    Var turbulence = fract(sin(orbital_angle * 7.0f + normalized_r * 13.0f + time * 0.2f) * 43758.5453f);
                    turbulence = turbulence * 0.4f + 0.6f;

                    // Combine all texture layers
                    Var texture_intensity = arm_pattern * fine_detail * radial_wave * turbulence;
                    texture_intensity = pow(texture_intensity, 0.7f);// Adjust contrast

                    // Apply texture as brightness variation
                    local_disk_color *= (0.6f + 0.8f * texture_intensity);

                    // Gravitational redshift
                    Var redshift = sqrt(max(0.001f, 1.0f - bh_radius / max(intersect_r, bh_radius * 1.01f)));
                    local_disk_color *= redshift;

                    // Apply intensity with Doppler beaming and edge smoothing
                    Var intensity = beaming * edge_smooth * 2.0f;
                    Var alpha = intensity * 0.5f;

                    // Determine if this is the front or back side of the disk
                    // Front side: ray is coming from camera toward disk (moving toward y=0)
                    // Back side: ray has passed through disk (moving away from y=0)
                    // We want to show the side that's closer to the camera
                    Var moving_toward_plane = (cam_pos.y > 0.0f & pos.y < prev_y) | (cam_pos.y<0.0f & pos.y> prev_y);

                    $if (moving_toward_plane) {
                        // This is the front side (closer to camera) - use it
                        disk_color = local_disk_color;
                        disk_alpha = alpha;
                        hit_disk = true;
                    }
                    $else {
                        // This is the back side - only use it if we haven't hit front yet
                        // and if it's not behind the event horizon from our view
                        $if (!hit_disk) {
                            back_disk_color = local_disk_color;
                            back_disk_alpha = alpha;
                            back_disk_sampled = true;
                        };
                    };
                };
            };

            // Store previous position for next iteration
            prev_pos = pos;
            prev_y = pos.y;

            // Gravitational acceleration (simplified)
            Var r_safe = max(r, bh_radius * 0.5f);
            Var accel = -1.5f * bh_mass / (r_safe * r_safe * r_safe) * pos;

            // Update direction (bending)
            Var new_dir = dir + accel * dt;
            Var new_dir_len = length(new_dir);
            dir = ite(new_dir_len > 0.001f, new_dir / new_dir_len, dir);

            // Step forward
            pos = pos + dir * dt;
        };

        // If we didn't hit the front side but hit the back side, use back side
        // This handles cases where camera is inside the disk radius looking outward
        $if (!hit_disk & back_disk_sampled) {
            disk_color = back_disk_color;
            disk_alpha = back_disk_alpha;
        };

        // Background starfield
        color = get_star_color(dir);

        // Blend disk on top
        color = lerp(color, disk_color, disk_alpha);

        // Add lensing glow around black hole
        Var to_bh = normalize(-cam_pos);
        Var cos_angle = dot(dir, to_bh);
        Var glow = exp(-(1.0f - cos_angle) * 30.0f) * 0.4f;

        // Check if ray passes near photon sphere
        Var impact_param = length(cross(pos, dir)) / max(length(pos), 0.001f);
        Var near_photon_sphere = smoothstep(photon_sphere * 1.5f, photon_sphere, impact_param);

        color += make_float3(1.0f, 0.9f, 0.7f) * glow * near_photon_sphere;

        // Tone mapping to prevent clipping
        color = color / (color + 1.0f) * 1.5f;

        // Ensure no NaN values
        color = clamp(color, 0.0f, 1.0f);

        image.write(uv, make_float4(color, 1.0f));
    };

    auto blackhole_shader = device.compile(blackhole_kernel);

    if (force_offline) {
        float time = 0.0f;
        stream << blackhole_shader(display, rot_x, rot_y, roll_angle, zoom, time).dispatch(width, height);
        luisa::vector<std::array<uint8_t, 4u>> host_image(width * height);
        stream << display.copy_to(luisa::span{host_image})
               << synchronize();
        stbi_write_png("test_blackhole.png", width, height, 4, host_image.data(), 0);
        if (compare_path) {
            auto result = luisa::ref::compare_with_reference_file(
                reinterpret_cast<const uint8_t *>(host_image.data()),
                width, height, 4,
                *compare_path);
            LUISA_INFO("Reference comparison: {} ({})", result.passed ? "PASSED" : "FAILED", result.message);
            if (!result.passed) { return 1; }
        }
        LUISA_INFO("Saved offline render to test_blackhole.png");
    } else {
        Clock app_clock;
        while (!window->should_close()) {
            window->poll_events();

            if (window->is_key_down(KEY_ESCAPE)) {
                break;
            }

            if (window->is_key_down(KEY_EQUAL) || window->is_key_down(KEY_KP_ADD)) {
                zoom *= 0.98f;
                zoom = max(zoom, 10.0f);
            }
            if (window->is_key_down(KEY_MINUS) || window->is_key_down(KEY_KP_SUBTRACT)) {
                zoom *= 1.02f;
                zoom = min(zoom, 100.0f);
            }

            auto time = static_cast<float>(app_clock.toc() * 1e-3);
            stream << blackhole_shader(display, rot_x, rot_y, roll_angle, zoom, time).dispatch(width, height)
                   << swap_chain->present(display);
        }
    }

    stream << synchronize();
    return 0;
}
