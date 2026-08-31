/*
 * Tutorial 09: Reaction-Diffusion (Gray-Scott)
 *
 * This tutorial teaches how to:
 * - Step 1: Store two evolving chemical fields in ping-pong images.
 * - Step 2: Initialize U and V with a small seed in the center.
 * - Step 3: Implement the Gray-Scott update rule with a 5-point Laplacian.
 * - Step 4: Visualize the pattern in a readable false-color view.
 * - Step 5: Switch presets interactively, or save the coral preset with --offline.
 *
 * Usage:
 *   ./tutorial_09_reaction_diffusion <backend> [--offline]
 */

#include <array>
#include <cstdint>
#include <cstdlib>

#include <luisa/luisa-compute.h>
#include <luisa/dsl/sugar.h>
#include <luisa/core/logging.h>
#include <stb/stb_image_write.h>

#ifdef LUISA_ENABLE_GUI
#include <luisa/gui/window.h>
#endif

using namespace luisa;
using namespace luisa::compute;

namespace {

static constexpr uint2 grid_size = make_uint2(512u, 512u);
static constexpr uint2 display_size = make_uint2(512u, 512u);
static constexpr float du = 0.16f;
static constexpr float dv = 0.08f;
static constexpr float rd_dt = 1.0f;

struct Preset {
    float feed;
    float kill;
    const char *name;
};

static constexpr std::array presets{
    Preset{0.035f, 0.060f, "coral"},
    Preset{0.037f, 0.060f, "fingerprint"},
    Preset{0.030f, 0.062f, "spots"},
    Preset{0.030f, 0.054f, "stripes"}};

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

    Image<float> u_a = device.create_image<float>(PixelStorage::FLOAT4, grid_size);
    Image<float> u_b = device.create_image<float>(PixelStorage::FLOAT4, grid_size);
    Image<float> v_a = device.create_image<float>(PixelStorage::FLOAT4, grid_size);
    Image<float> v_b = device.create_image<float>(PixelStorage::FLOAT4, grid_size);
    Image<float> output_image = device.create_image<float>(PixelStorage::BYTE4, display_size);

    // Step 1: Initialize the Gray-Scott state.
    // U starts at 1, V starts at 0, and we seed a square in the center to trigger pattern formation.
    Kernel2D initialize_fields = [&]() noexcept {
        set_block_size(16u, 16u, 1u);
        $uint2 coord = dispatch_id().xy();
        $uint2 center = grid_size / 2u;
        $bool in_seed = abs(cast<int>(coord.x) - cast<int>(center.x)) < 12 &
                        abs(cast<int>(coord.y) - cast<int>(center.y)) < 12;

        $float u = 1.0f;
        $float v = 0.0f;
        $if (in_seed) {
            u = 0.5f;
            v = 1.0f;
        }
        $else {
            u = 1.0f;
            v = 0.0f;
        };

        u_a->write(coord, make_float4(u));
        u_b->write(coord, make_float4(u));
        v_a->write(coord, make_float4(v));
        v_b->write(coord, make_float4(v));
    };

    Callable sample_scalar = [](ImageFloat image, Int2 coord) noexcept {
        Int2 clamped = clamp(coord, make_int2(0), make_int2(cast<int>(grid_size.x) - 1, cast<int>(grid_size.y) - 1));
        return image.read(make_uint2(clamped)).x;
    };

    // Step 2: Implement one reaction-diffusion update step.
    Kernel2D update_fields = [&]($image<float> u_src,
                                 $image<float> v_src,
                                 $image<float> u_dst,
                                 $image<float> v_dst,
                                 $float feed,
                                 $float kill) noexcept {
        set_block_size(16u, 16u, 1u);

        $uint2 coord = dispatch_id().xy();
        $int2 p = make_int2(cast<int>(coord.x), cast<int>(coord.y));

        $float u = u_src.read(coord).x;
        $float v = v_src.read(coord).x;
        $float lap_u = sample_scalar(u_src, p + make_int2(-1, 0)) +
                       sample_scalar(u_src, p + make_int2(1, 0)) +
                       sample_scalar(u_src, p + make_int2(0, -1)) +
                       sample_scalar(u_src, p + make_int2(0, 1)) -
                       4.0f * u;
        $float lap_v = sample_scalar(v_src, p + make_int2(-1, 0)) +
                       sample_scalar(v_src, p + make_int2(1, 0)) +
                       sample_scalar(v_src, p + make_int2(0, -1)) +
                       sample_scalar(v_src, p + make_int2(0, 1)) -
                       4.0f * v;

        $float reaction = u * v * v;
        $float new_u = clamp(u + (du * lap_u - reaction + feed * (1.0f - u)) * rd_dt, 0.0f, 1.0f);
        $float new_v = clamp(v + (dv * lap_v + reaction - (feed + kill) * v) * rd_dt, 0.0f, 1.0f);

        u_dst.write(coord, make_float4(new_u));
        v_dst.write(coord, make_float4(new_v));
    };

    // Step 3: Visualize the pattern.
    // U contributes cool blue tones, while V produces warmer red/yellow features.
    Kernel2D visualize = [&]($image<float> u_field, $image<float> v_field, ImageFloat image) noexcept {
        set_block_size(16u, 16u, 1u);

        $uint2 coord = dispatch_id().xy();
        $float u = u_field.read(coord).x;
        $float v = v_field.read(coord).x;
        $float3 cool = make_float3(0.08f, 0.18f, 0.9f) * clamp(u, 0.0f, 1.0f);
        $float3 warm = lerp(make_float3(0.2f, 0.0f, 0.0f), make_float3(1.0f, 0.9f, 0.15f), clamp(v * 1.4f, 0.0f, 1.0f));
        $float3 color = cool * (0.35f + 0.65f * (1.0f - v)) + warm * clamp(v * 1.25f, 0.0f, 1.0f);
        color = pow(clamp(color, 0.0f, 1.0f), 1.0f / 2.2f);
        image.write(coord, make_float4(color, 1.0f));
    };

    auto initialize_shader = device.compile(initialize_fields);
    auto update_shader = device.compile(update_fields);
    auto visualize_shader = device.compile(visualize);

    auto reset_fields = [&] {
        stream << initialize_shader().dispatch(grid_size) << synchronize();
    };
    reset_fields();

    auto state_in_a = true;
    auto current_preset = presets[0];

    auto iterate = [&](uint iterations) {
        for (auto i = 0u; i < iterations; i++) {
            if (state_in_a) {
                stream << update_shader(u_a, v_a, u_b, v_b, current_preset.feed, current_preset.kill).dispatch(grid_size) << synchronize();
            } else {
                stream << update_shader(u_b, v_b, u_a, v_a, current_preset.feed, current_preset.kill).dispatch(grid_size) << synchronize();
            }
            state_in_a = !state_in_a;
        }
    };

    auto render_current_state = [&] {
        if (state_in_a) {
            stream << visualize_shader(u_a, v_a, output_image).dispatch(display_size) << synchronize();
        } else {
            stream << visualize_shader(u_b, v_b, output_image).dispatch(display_size) << synchronize();
        }
    };

    if (offline) {
        current_preset = presets[0];
        reset_fields();
        state_in_a = true;
        iterate(5000u);
        render_current_state();

        luisa::vector<std::array<uint8_t, 4u>> host_image(display_size.x * display_size.y);
        stream << output_image.copy_to(luisa::span{host_image}) << synchronize();
        stbi_write_png("tutorial_09_reaction_diffusion.png", display_size.x, display_size.y, 4, host_image.data(), 0);
        LUISA_INFO("Saved tutorial_09_reaction_diffusion.png");
        return 0;
    }

#ifdef LUISA_ENABLE_GUI
    Window window{"Tutorial 09 - Reaction Diffusion", display_size};
    auto reset_requested = false;
    window.set_key_callback([&](Key key, KeyModifiers, Action action) noexcept {
        if (action != Action::ACTION_RELEASED) {
            return;
        }
        switch (key) {
            case Key::KEY_1:
                current_preset = presets[0];
                reset_requested = true;
                LUISA_INFO("Preset: {}", current_preset.name);
                break;
            case Key::KEY_2:
                current_preset = presets[1];
                reset_requested = true;
                LUISA_INFO("Preset: {}", current_preset.name);
                break;
            case Key::KEY_3:
                current_preset = presets[2];
                reset_requested = true;
                LUISA_INFO("Preset: {}", current_preset.name);
                break;
            case Key::KEY_4:
                current_preset = presets[3];
                reset_requested = true;
                LUISA_INFO("Preset: {}", current_preset.name);
                break;
            case Key::KEY_R:
                reset_requested = true;
                break;
            case Key::KEY_ESCAPE:
                window.set_should_close();
                break;
            default:
                break;
        }
    });

    Swapchain swapchain = device.create_swapchain(
        stream,
        SwapchainOption{
            .display = window.native_display(),
            .window = window.native_handle(),
            .size = display_size,
            .wants_hdr = false,
            .wants_vsync = false,
            .back_buffer_count = 2,
        });
    Image<float> swapchain_image = device.create_image<float>(swapchain.backend_storage(), display_size);

    Kernel2D blit_to_swapchain = [&](ImageFloat src, ImageFloat dst) noexcept {
        set_block_size(16u, 16u, 1u);
        $uint2 coord = dispatch_id().xy();
        dst.write(coord, src.read(coord));
    };
    auto blit_shader = device.compile(blit_to_swapchain);

    while (!window.should_close()) {
        window.poll_events();
        if (reset_requested) {
            reset_fields();
            state_in_a = true;
            reset_requested = false;
        }
        iterate(16u);
        render_current_state();

        stream << blit_shader(output_image, swapchain_image).dispatch(display_size)
               << swapchain.present(swapchain_image)
               << synchronize();
    }
#endif

    return 0;
}
