/*
 * Tutorial 04: Conway's Game of Life
 *
 * This tutorial teaches how to build a complete GPU simulation loop in LuisaCompute.
 * We combine three important ideas: ping-pong images for state updates, a compute
 * kernel that applies Conway's rules in parallel, and a second kernel that turns the
 * low-resolution simulation into a larger image suitable for presentation or PNG export.
 *
 * What you will learn:
 *   1. Why ping-pong resources avoid read/write hazards in iterative simulations.
 *   2. How to count neighbors and express branching logic with LuisaCompute DSL sugar.
 *   3. How to support both interactive presentation and a headless "--offline" workflow.
 */

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <random>
#include <string_view>

#include <luisa/luisa-compute.h>
#include <luisa/dsl/sugar.h>
#include <luisa/gui/window.h>
#include <luisa/runtime/swapchain.h>
#include <stb/stb_image_write.h>

using namespace luisa;
using namespace luisa::compute;

namespace {

[[nodiscard]] auto has_flag(int argc, char *argv[], std::string_view flag) noexcept {
    for (auto i = 2; i < argc; i++) {
        if (flag == argv[i]) {
            return true;
        }
    }
    return false;
}

[[nodiscard]] auto pack_alive(bool alive) noexcept {
    auto v = alive ? 255u : 0u;
    return v * 0x00010101u | 0xff000000u;
}

void save_png(Stream &stream, Image<float> &image, uint2 resolution, const char *filename) {
    luisa::vector<std::byte> host_pixels(image.view().size_bytes());
    stream << image.copy_to(luisa::span{host_pixels}) << synchronize();
    auto success = stbi_write_png(filename, static_cast<int>(resolution.x), static_cast<int>(resolution.y), 4, host_pixels.data(), 0);
    LUISA_INFO("Saved {} [{}]", filename, success != 0 ? "ok" : "failed");
}

struct ImagePair {
    Image<uint> front;
    Image<uint> back;

    ImagePair(Device &device, uint2 resolution) noexcept
        : front{device.create_image<uint>(PixelStorage::BYTE4, resolution)},
          back{device.create_image<uint>(PixelStorage::BYTE4, resolution)} {}

    void swap() noexcept { std::swap(front, back); }
};

}// namespace

int main(int argc, char *argv[]) {

    log_level_verbose();

    auto offline = has_flag(argc, argv, "--offline") || (argc > 1 && std::string_view{argv[1]} == "--offline");
    luisa::string backend;
    for (int i = 1; i < argc; i++) {
        if (std::string_view{argv[i]} != "--offline" && backend.empty()) {
            backend = argv[i];
        }
    }

    // Step 1: Create the runtime objects.
    // We do this first because every resource, stream, and shader belongs to a device.
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
    Stream stream = device.create_stream(StreamTag::GRAPHICS);

    static constexpr auto simulation_resolution = make_uint2(256u, 256u);
    static constexpr auto display_resolution = make_uint2(1024u, 1024u);
    static constexpr auto upscale = 4u;

    LUISA_INFO("Tutorial 04 - Conway's Game of Life");
    LUISA_INFO("Backend: {}", backend);
    LUISA_INFO("Mode: {}", offline ? "offline" : "interactive");
    if (!offline) {
        LUISA_INFO("Controls: SPACE = pause/run, R = reset, ESC = quit");
    }

    // Step 2: Allocate a ping-pong pair for the simulation state.
    // Conway's rules read the previous generation and write a new one, so a single
    // image would create hazards. Two images make the data flow explicit and safe.
    ImagePair life_images{device, simulation_resolution};

    // Step 3: Define a tiny helper callable for readability.
    // The simulation stores 0 or 255 in BYTE4 pixels; reading the red channel is enough.
    Callable is_alive = [](ImageUInt state, UInt2 cell) noexcept {
        return state.read(cell).x > 127u;
    };

    // Step 4: Define the update kernel.
    // This kernel counts the eight neighbors of each cell and applies Conway's rules.
    // The toroidal wrap-around keeps the demo simple because every cell always has
    // eight valid neighbors, even at the edges.
    Kernel2D update_kernel = [&](ImageUInt previous, ImageUInt next) noexcept {
        set_block_size(16u, 16u, 1u);
        auto cell = dispatch_id().xy();
        auto size = dispatch_size().xy();
        auto alive = is_alive(previous, cell);
        UInt live_neighbors = 0u;

        $for (oy, 3u) {
            $for (ox, 3u) {
                auto offset = make_int2(cast<int>(ox) - 1, cast<int>(oy) - 1);
                $if (offset.x == 0 & offset.y == 0) {
                    // Skip the center sample because a cell is not its own neighbor.
                }
                $else {
                    auto wrapped = make_int2(cell) + offset + make_int2(size);
                    wrapped.x = wrapped.x % cast<int>(size.x);
                    wrapped.y = wrapped.y % cast<int>(size.y);
                    auto neighbor_cell = make_uint2(cast<uint>(wrapped.x), cast<uint>(wrapped.y));
                    live_neighbors += ite(is_alive(previous, neighbor_cell), 1u, 0u);
                };
            };
        };

        auto survives = alive & (live_neighbors == 2u | live_neighbors == 3u);
        auto is_born = !alive & (live_neighbors == 3u);
        auto next_alive = survives | is_born;
        auto value = ite(next_alive, 255u, 0u);
        next.write(cell, make_uint4(value, value, value, 255u));
    };
    auto update_shader = device.compile(update_kernel);

    // Step 5: Define a display kernel that enlarges the cellular grid by 4x.
    // Keeping the simulation coarse makes the rules easy to see, while a second pass
    // gives us a crisp presentation image for a window or PNG.
    Kernel2D display_kernel = [](ImageUInt state, ImageFloat output) noexcept {
        set_block_size(16u, 16u, 1u);
        auto pixel = dispatch_id().xy();
        auto cell = pixel / upscale;
        auto v = cast<float>(state.read(cell).x) / 255.0f;
        auto color = make_float3(0.05f, 0.07f, 0.10f) + v * make_float3(0.85f, 0.95f, 0.55f);
        output.write(pixel, make_float4(color, 1.0f));
    };
    auto display_shader = device.compile(display_kernel);

    // Step 6: Prepare a reset lambda that creates a random initial state.
    // A 25% alive ratio is sparse enough to evolve interesting structures without
    // immediately saturating the whole board.
    std::mt19937 rng{std::random_device{}()};
    auto reset_simulation = [&] {
        luisa::vector<uint> host_state(simulation_resolution.x * simulation_resolution.y);
        for (auto &pixel : host_state) {
            pixel = pack_alive((rng() % 4u) == 0u);
        }
        stream << life_images.front.copy_from(luisa::span{host_state})
               << life_images.back.copy_from(luisa::span{host_state})
               << synchronize();
        LUISA_INFO("Simulation reset.");
    };
    reset_simulation();

    if (offline) {
        // Step 7A: Offline mode runs a fixed number of generations and writes a PNG.
        // This is useful for CI, documentation screenshots, and headless machines.
        auto output = device.create_image<float>(PixelStorage::BYTE4, display_resolution);
        for (auto generation = 0u; generation < 200u; generation++) {
            stream << update_shader(life_images.front, life_images.back).dispatch(simulation_resolution);
            life_images.swap();
        }
        stream << display_shader(life_images.front, output).dispatch(display_resolution);
        save_png(stream, output, display_resolution, "tutorial_04_game_of_life.png");
        return 0;
    }

    // Step 7B: Interactive mode creates a window and a swapchain-backed image.
    // The window lets us demonstrate a classic render loop driven by user input.
    Window window{"Tutorial 04 - Conway's Game of Life", display_resolution};
    Swapchain swapchain = device.create_swapchain(
        stream,
        SwapchainOption{
            .display = window.native_display(),
            .window = window.native_handle(),
            .size = display_resolution,
            .wants_vsync = true,
            .back_buffer_count = 3u,
        });
    auto display = device.create_image<float>(swapchain.backend_storage(), display_resolution);

    bool paused = false;
    bool reset_requested = false;
    window.set_key_callback([&](Key key, KeyModifiers, Action action) noexcept {
        if (action != ACTION_PRESSED) {
            return;
        }
        if (key == KEY_SPACE) {
            paused = !paused;
            LUISA_INFO("Simulation {}.", paused ? "paused" : "running");
        } else if (key == KEY_R) {
            reset_requested = true;
        } else if (key == KEY_ESCAPE) {
            window.set_should_close();
        }
    });

    // Step 8: Run the simulation loop.
    // The host controls frame pacing and input, while the device updates and renders
    // a whole generation in parallel.
    while (!window.should_close()) {
        window.poll_events();

        if (reset_requested) {
            reset_simulation();
            reset_requested = false;
        }

        if (!paused) {
            stream << update_shader(life_images.front, life_images.back).dispatch(simulation_resolution);
            life_images.swap();
        }

        stream << display_shader(life_images.front, display).dispatch(display_resolution)
               << swapchain.present(display);
    }

    stream << synchronize();
    return 0;
}
