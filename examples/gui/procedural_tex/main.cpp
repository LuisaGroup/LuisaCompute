// Procedural Texture Generator -- example entry point.
//
// Renders a catalog of procedural-noise textures (one algorithm per
// .h/.cpp pair under examples/gui/procedural_tex/, see noise_*.cpp) with
// LuisaCompute kernels. Each algorithm owns an ImGui window with its
// parameter widgets; the texture is regenerated lazily -- only when a
// parameter actually changed -- never per frame.
//
// Headless mode is the default so the example can be verified by a tool:
//   xmake run example_procedural_tex dx
//       generates every texture with default parameters, checks generic
//       invariants (finite, in [0,1], spatially varying) plus each
//       algorithm's own validation, checks run-to-run determinism, and
//       exits non-zero on the first failure.
//   xmake run example_procedural_tex dx --output-dir out
//       additionally writes one PNG per algorithm for eyeballing.
// The GUI loop is opt-in:
//   xmake run example_procedural_tex dx --gui
//       opens one window per algorithm; drag the sliders to re-dispatch.
//       The swapchain render texture is cleared to grey (0.5) every frame.
//       Each window has a Resolution combo (256/512/1024/2048/4096) and a
//       Store button; Store re-generates the texture and writes
//       <algorithm-name>-<resolution>-<count>.png via stb_image_write.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>

#include <luisa/core/clock.h>
#include <luisa/core/logging.h>
#include <luisa/core/stl/filesystem.h>
#include <luisa/core/stl/format.h>
#include <luisa/core/stl/string.h>
#include <luisa/core/stl/vector.h>
#include <luisa/runtime/context.h>
#include <luisa/runtime/device.h>
#include <luisa/runtime/image.h>
#include <luisa/runtime/stream.h>

#include "procedural_texture.h"

#if ENABLE_DISPLAY
#include <luisa/gui/imgui_window.h>
#endif
#include <stb/stb_image_write.h>

using namespace luisa;
using namespace luisa::compute;
using namespace procedural_tex;

namespace {

struct Options {
    luisa::string backend;
    bool gui{false};
    uint32_t resolution{512u};
    luisa::filesystem::path output_dir;
};

void print_usage(const char *argv0) {
    LUISA_INFO("Usage: {} <backend> [--gui] [--resolution <n>] [--output-dir <dir>]", argv0);
    LUISA_INFO("  <backend>: cuda, dx, metal, vk, hip, fallback, simd");
    LUISA_INFO("  default (no --gui): headless self-test, exits non-zero on failure");
    LUISA_INFO("  --gui: open the interactive ImGui windows");
    LUISA_INFO("  --resolution <n>: texture size (default 512, clamped to [64, 4096])");
    LUISA_INFO("  --output-dir <dir>: headless only, write one PNG per algorithm");
}

[[nodiscard]] Options parse_options(int argc, char *argv[]) {
    Options opt;
    if (argc <= 1) {
        print_usage(argv[0]);
        std::exit(1);
    }
    opt.backend = argv[1];
    for (int i = 2; i < argc; ++i) {
        luisa::string_view arg{argv[i]};
        if (arg == "--gui") {
            opt.gui = true;
        } else if (arg == "--resolution" && i + 1 < argc) {
            opt.resolution = static_cast<uint32_t>(std::atoi(argv[++i]));
        } else if (arg == "--output-dir" && i + 1 < argc) {
            opt.output_dir = argv[++i];
        } else {
            LUISA_WARNING("Ignoring unknown argument: {}", arg);
        }
    }
    opt.resolution = std::clamp(opt.resolution, 64u, 4096u);
    return opt;
}

// Headless self-test for one algorithm: generate, read back, validate,
// re-generate and require bit-exact determinism. Returns true on success.
[[nodiscard]] bool headless_check(ProceduralTexture &tex, Image<float> &image,
                                  Stream &stream, uint2 resolution) {
    const auto pixel_count = resolution.x * resolution.y;
    luisa::vector<float4> pixels(pixel_count);
    luisa::vector<float4> pixels_again(pixel_count);

    Clock clock;
    tex.generate(image, stream);
    stream << image.copy_to(luisa::span{pixels}) << synchronize();
    auto first_ms = clock.toc();

    clock.tic();
    tex.generate(image, stream);
    stream << image.copy_to(luisa::span{pixels_again}) << synchronize();
    auto second_ms = clock.toc();

    bool ok = true;
    if (std::memcmp(pixels.data(), pixels_again.data(),
                    pixels.size() * sizeof(float4)) != 0) {
        LUISA_WARNING("[{}] non-deterministic output", tex.name());
        ok = false;
    }
    if (!tex.validate(pixels, resolution)) {
        LUISA_WARNING("[{}] validation failed", tex.name());
        ok = false;
    }
    if (ok) {
        LUISA_INFO("[{}] OK (generate: {:.2f} ms / {:.2f} ms)",
                   tex.name(), first_ms, second_ms);
    }
    return ok;
}

void write_png(const luisa::filesystem::path &path, const luisa::vector<float4> &pixels,
               uint2 resolution) {
    luisa::vector<std::uint8_t> bytes(pixels.size() * 4u);
    for (auto i = 0u; i < pixels.size(); ++i) {
        for (auto c = 0u; c < 4u; ++c) {
            auto v = std::clamp(pixels[i][c], 0.0f, 1.0f);
            bytes[i * 4u + c] = static_cast<std::uint8_t>(v * 255.0f + 0.5f);
        }
    }
    stbi_write_png(path.string().c_str(),
                   static_cast<int>(resolution.x), static_cast<int>(resolution.y),
                   4, bytes.data(), static_cast<int>(resolution.x * 4u));
}

}// namespace

int main(int argc, char *argv[]) {
    auto opt = parse_options(argc, argv);
    Context context{argv[0]};
    auto device = context.create_device(opt.backend.c_str());
    auto stream = device.create_stream(StreamTag::GRAPHICS);
    const auto resolution = make_uint2(opt.resolution, opt.resolution);

    auto textures = create_all_procedural_textures(device);
    if (textures.empty()) {
        LUISA_ERROR("No procedural texture algorithms registered.");
    }
    luisa::vector<Image<float>> images;
    images.reserve(textures.size());
    for (auto i = 0u; i < textures.size(); ++i) {
        images.emplace_back(device.create_image<float>(PixelStorage::FLOAT4, resolution));
    }

    if (!opt.gui) {
        // Headless self-test.
        bool all_ok = true;
        luisa::vector<float4> pixels(resolution.x * resolution.y);
        for (auto i = 0u; i < textures.size(); ++i) {
            all_ok &= headless_check(*textures[i], images[i], stream, resolution);
            if (!opt.output_dir.empty()) {
                stream << images[i].copy_to(luisa::span{pixels}) << synchronize();
                luisa::filesystem::create_directories(opt.output_dir);
                auto file = opt.output_dir / luisa::format("{}.png", textures[i]->name());
                write_png(file, pixels, resolution);
                LUISA_INFO("[{}] wrote {}", textures[i]->name(), file.string());
            }
        }
        if (!all_ok) {
            LUISA_ERROR("Headless self-test FAILED.");
        }
        LUISA_INFO("Headless self-test passed ({} algorithms, {}x{}).",
                   textures.size(), resolution.x, resolution.y);
        return 0;
    }

#if ENABLE_DISPLAY
    // Interactive mode: one window per algorithm. The texture is regenerated
    // only while its window reports a parameter change (lazy update); frames
    // where nothing changed cost zero dispatches.
    ImGuiWindow window{device, stream, "Procedural Textures", {.vsync = true}};

    // Clears the swapchain render texture to grey (0.5) every frame; ImGui
    // draws its windows on top of this background.
    auto clear_to_grey = device.compile<2>([](ImageFloat image) noexcept {
        image.write(dispatch_id().xy(), make_float4(0.5f, 0.5f, 0.5f, 1.0f));
    });

    // Per-window GUI state: the selected resolution (one of kGuiResolutions),
    // the registered ImGui texture id for the current image, and the
    // incremental counter used to name stored PNGs.
    constexpr uint32_t kGuiResolutions[] = {256u, 512u, 1024u, 2048u, 4096u};
    constexpr const char *kGuiResolutionNames =
        "256\0"
        "512\0"
        "1024\0"
        "2048\0"
        "4096\0";
    constexpr auto kGuiResolutionCount = static_cast<int>(std::size(kGuiResolutions));
    struct GuiTextureState {
        int resolution_index;
        uint64_t texture_id;
        uint32_t store_count{0u};
    };
    auto default_resolution_index = [&] {
        for (auto r = 0; r < kGuiResolutionCount; ++r) {
            if (kGuiResolutions[r] == opt.resolution) { return r; }
        }
        return 1;// 512
    }();

    luisa::vector<GuiTextureState> states(textures.size());
    for (auto i = 0u; i < textures.size(); ++i) {
        textures[i]->generate(images[i], stream);
        states[i].resolution_index = default_resolution_index;
        states[i].texture_id =
            window.register_texture(images[i], Sampler::linear_linear_zero());
    }
    stream << synchronize();

    Clock clock;
    while (!window.should_close()) {
        window.prepare_frame();
        // Grey background on the swapchain render texture, every frame.
        if (window.framebuffer()) {
            stream << clear_to_grey(window.framebuffer())
                          .dispatch(window.framebuffer().size());
        }
        for (auto i = 0u; i < textures.size(); ++i) {
            auto &tex = *textures[i];
            auto &state = states[i];
            ImGui::SetNextWindowSize(ImVec2(360.f, 460.f), ImGuiCond_FirstUseEver);
            ImGui::Begin(tex.name().data());
            auto params_changed = tex.draw_ui();

            // Resolution selection: recreates the image and re-registers the
            // ImGui texture whenever the user picks another size.
            auto res_index = state.resolution_index;
            if (ImGui::Combo("Resolution", &res_index, kGuiResolutionNames,
                             kGuiResolutionCount)) {
                state.resolution_index = res_index;
                const auto res =
                    make_uint2(kGuiResolutions[res_index], kGuiResolutions[res_index]);
                window.unregister_texture(state.texture_id);
                images[i] = device.create_image<float>(PixelStorage::FLOAT4, res);
                state.texture_id =
                    window.register_texture(images[i], Sampler::linear_linear_zero());
                params_changed = true;
            }
            const auto res =
                make_uint2(kGuiResolutions[state.resolution_index],
                           kGuiResolutions[state.resolution_index]);

            // Store: re-generate the texture and write
            // <algorithm-name>-<resolution>-<count>.png through stb_image_write.
            if (ImGui::Button("Store")) {
                Clock gen_clock;
                tex.generate(images[i], stream);
                luisa::vector<float4> pixels(res.x * res.y);
                stream << images[i].copy_to(luisa::span{pixels}) << synchronize();
                auto file = luisa::format("{}-{}-{}.png", tex.name(), res.x,
                                          ++state.store_count);
                write_png(file, pixels, res);
                LUISA_INFO("[{}] wrote {} (generated in {:.2f} ms)", tex.name(),
                           file, gen_clock.toc());
            } else if (params_changed) {
                Clock gen_clock;
                tex.generate(images[i], stream);
                LUISA_VERBOSE("[{}] regenerated in {:.2f} ms", tex.name(),
                              gen_clock.toc());
            }

            auto avail = ImGui::GetContentRegionAvail();
            auto side = std::min(avail.x, avail.y);
            ImGui::Image(state.texture_id, ImVec2(side, side));
            ImGui::End();
        }
        window.render_frame();
        static_cast<void>(clock.toc());
        clock.tic();
    }
    return 0;
#else
    LUISA_ERROR("This build has no GUI module; drop --gui or rebuild with lc_enable_gui=true.");
#endif
}
