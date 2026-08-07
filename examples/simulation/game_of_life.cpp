// Conway's Game of Life Cellular Automaton
// Implements the classic cellular automaton on the GPU with compute shaders.
// Each pixel represents a cell that evolves based on its neighbors.
//
// Rules:
// - Any live cell with 2-3 live neighbors survives
// - Any dead cell with exactly 3 live neighbors becomes alive
// - All other cells die or stay dead
//
// Features demonstrated:
// - Compute shaders for cellular automata simulation
// - Ping-pong buffer technique for double buffering
// - Window system integration for real-time display
// - Random initial state generation

#include <filesystem>
#include <iostream>
#include <memory>
#include <optional>
#include <random>
#include <string_view>
#include <thread>
#include <chrono>

#include "../common/reference_compare.h"
#include <luisa/core/logging.h>
#include <luisa/runtime/context.h>
#include <luisa/runtime/device.h>
#include <luisa/runtime/stream.h>
#include <luisa/runtime/event.h>
#include <luisa/dsl/sugar.h>
#include <luisa/runtime/swapchain.h>

using namespace luisa;
using namespace luisa::compute;

#ifndef ENABLE_DISPLAY
#ifdef LUISA_ENABLE_GUI
#define ENABLE_DISPLAY 1
#endif
#endif

#if ENABLE_DISPLAY
#include <luisa/gui/window.h>
#endif

// Image pair for ping-pong buffering
// Swaps between two images to avoid read-write conflicts
struct ImagePair {
    Image<uint> prev;
    Image<uint> curr;
    ImagePair(Device &device, PixelStorage storage, uint width, uint height) noexcept
        : prev{device.create_image<uint>(storage, width, height)},
          curr{device.create_image<uint>(storage, width, height)} {}
    void swap() noexcept { std::swap(prev, curr); }
};

int main(int argc, char *argv[]) {

    Context context{argv[0]};
    if (argc <= 1) {
        LUISA_INFO("Usage: {} <backend> [--offline] [-c <reference.png>]. <backend>: cuda, dx, metal, vk, hip, fallback", argv[0]);
        exit(1);
    }
    auto opts = luisa::ref::ExampleOptions::parse(argc, argv);
    if (!opts.valid()) {
        LUISA_WARNING("Invalid command line: {}", opts.error_message);
        return 1;
    }
    auto force_offline = opts.offline;
    auto compare_path = opts.compare_path;
    auto executable_name = std::filesystem::path{argv[0]}.filename().string();
    if (std::string_view{executable_name}.starts_with("test_")) {
        force_offline = true;
    }
#if !ENABLE_DISPLAY
    if (!force_offline) {
        LUISA_ERROR("GUI support is disabled. Use --offline.");
    }
#endif
    Device device = context.create_device(argv[1]);
    LUISA_INFO("Keys: SPACE - Run/Pause, R - Reset, ESC - Quit");

    // Helper to read cell state from image
    Callable read_state = [](ImageUInt prev, UInt2 uv) noexcept {
        return prev.read(uv).x == 255u;
    };

    // Game of Life update kernel
    // Counts live neighbors and applies Conway's rules
    Kernel2D kernel = [&](ImageUInt prev, ImageUInt curr) noexcept {
        set_block_size(16, 16, 1);
        UInt count = def(0u);
        UInt2 uv = dispatch_id().xy();
        UInt2 size = dispatch_size().xy();
        Bool state = read_state(prev, uv);
        Int2 p = make_int2(uv);
        // Check all 8 neighbors
        for (int dy = -1; dy <= 1; dy++) {
            for (int dx = -1; dx <= 1; dx++) {
                if (dx != 0 || dy != 0) {
                    Int2 q = p + make_int2(dx, dy) + make_int2(size);
                    Bool neighbor = read_state(prev, make_uint2(q) % size);
                    count += ite(neighbor, 1, 0);
                }
            }
        }
        // Apply Conway's rules
        Bool c0 = count == 2u;
        Bool c1 = count == 3u;
        curr.write(uv, make_uint4(make_uint3(ite((state & c0) | c1, 255u, 0u)), 255u));
    };
    auto shader = device.compile(kernel);

    // Display kernel: scales up the low-res simulation for display
    Kernel2D display_kernel = [&](ImageUInt in_tex, ImageFloat out_tex) noexcept {
        set_block_size(16, 16, 1);
        UInt2 uv = dispatch_id().xy();
        UInt2 coord = uv / 4u;
        UInt4 value = in_tex.read(coord);
        out_tex.write(uv, make_float4(value) / 255.0f);
    };
    auto display_shader = device.compile(display_kernel);

    // Grid dimensions
    static constexpr uint width = 128u;
    static constexpr uint height = 128u;
    ImagePair image_pair{device, PixelStorage::BYTE4, width, height};

    static constexpr uint display_width = width * 4u;
    static constexpr uint display_height = height * 4u;
    Stream stream = device.create_stream(force_offline ? StreamTag::COMPUTE : StreamTag::GRAPHICS);
    std::mt19937 rng{force_offline ? 42u : std::random_device{}()};
#if ENABLE_DISPLAY
    std::unique_ptr<Window> window;
    std::optional<Swapchain> swap_chain;
    if (!force_offline) {
        window = std::make_unique<Window>("Game of Life", display_width, display_height);
        swap_chain.emplace(device.create_swapchain(
            stream,
            SwapchainOption{
                .display = window->native_display(),
                .window = window->native_handle(),
                .size = window->size(),
            }));
    }
#endif
    Image<float> display = [&] {
#if ENABLE_DISPLAY
        if (!force_offline) {
            return device.create_image<float>(swap_chain->backend_storage(), window->size());
        }
#endif
        return device.create_image<float>(PixelStorage::BYTE4, display_width, display_height);
    }();

    // Initialize with random state (25% chance of being alive)
    luisa::vector<uint> host_image(width * height);
    for (auto &v : host_image) {
        auto x = (rng() % 4u == 0u) * 255u;
        v = x * 0x00010101u | 0xff000000u;
    }
    stream << image_pair.prev.copy_from(luisa::span{host_image}) << synchronize();

    if (force_offline) {
        static constexpr uint offline_frames = 100u;
        for (uint i = 0u; i < offline_frames; i++) {
            stream << shader(image_pair.prev, image_pair.curr).dispatch(width, height)
                   << display_shader(image_pair.curr, display).dispatch(display_width, display_height);
            image_pair.swap();
        }
        luisa::vector<uint8_t> host_image(display_width * display_height * 4u);
        stream << display.copy_to(luisa::span{host_image}) << synchronize();
        stbi_write_png("test_game_of_life.png", display_width, display_height, 4, host_image.data(), 0);
        if (compare_path) {
            auto result = luisa::ref::compare_with_reference_file(
                reinterpret_cast<const uint8_t *>(host_image.data()),
                display_width, display_height, 4,
                *compare_path);
            LUISA_INFO("Reference comparison: {} ({})", result.passed ? "PASSED" : "FAILED", result.message);
            if (!result.passed) { return 1; }
        }
    } else {
#if ENABLE_DISPLAY
        while (!window->should_close()) {
            stream << shader(image_pair.prev, image_pair.curr).dispatch(width, height)
                   << display_shader(image_pair.curr, display).dispatch(display_width, display_height)
                   << swap_chain->present(display);
            image_pair.swap();
            std::this_thread::sleep_for(std::chrono::milliseconds{30});
            window->poll_events();
        }
#endif
    }
    stream << synchronize();
}
