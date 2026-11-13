/**
 * LuisaCompute NTC Material Viewer
 * Real-time material viewport supporting mip levels, channels, and neural sampling.
 */

#include <libntc/ntc.h>
#include <libntc/buffer_copy.h>

#include <luisa/core/logging.h>
#include <luisa/runtime/context.h>

#ifdef LUISA_ENABLE_GUI
#include <luisa/gui/window.h>
#include <GLFW/glfw3.h>
#endif

#include <stb/stb_image_write.h>

#include <iostream>
#include <filesystem>
#include <string>

using namespace luisa;
using namespace luisa::compute;
using namespace luisa::ntc;

int main(int argc, char *argv[]) {
    luisa::log_level_info();

    std::string ntc_path;
    std::string backend_name = "cuda";
    uint32_t active_mip = 0;
    uint32_t active_channel = 1; // Default to Albedo RGB

    if (const char *env_ntc = std::getenv("NTC_SAMPLE_NTC")) {
        ntc_path = env_ntc;
    } else if (std::filesystem::exists("C:/Users/tomo/Desktop/CVPR2027/demos/output_cli/Tiles135D_py.ntc")) {
        ntc_path = "C:/Users/tomo/Desktop/CVPR2027/demos/output_cli/Tiles135D_py.ntc";
    }

    for (int i = 1; i < argc; ++i) {
        std::string_view arg = argv[i];
        if (arg == "-h" || arg == "--help") {
            std::cout << "Usage: luisa_ntc_viewer --ntc=<path.ntc> [--backend=cuda|dx|vk] [--channel=1] [--mip=0]\n";
            return 0;
        } else if (arg.starts_with("--ntc=")) {
            ntc_path = arg.substr(6);
        } else if (arg.starts_with("--backend=")) {
            backend_name = arg.substr(10);
        } else if (arg.starts_with("--mip=")) {
            active_mip = std::stoi(std::string(arg.substr(6)));
        } else if (arg.starts_with("--channel=")) {
            active_channel = std::stoi(std::string(arg.substr(10)));
        } else if (arg.ends_with(".ntc")) {
            ntc_path = arg;
        }
    }

    if (ntc_path.empty()) {
        std::cout << "Usage: luisa_ntc_viewer --ntc=<path.ntc> [--backend=cuda|dx|vk] [--channel=1] [--mip=0]\n";
        return 1;
    }

    if (!std::filesystem::exists(ntc_path)) {
        LUISA_ERROR_WITH_LOCATION("File not found: {}", ntc_path);
        return 1;
    }

    NtexContainer container;
    std::string err;
    if (!container.load_from_file(ntc_path, err)) {
        LUISA_ERROR_WITH_LOCATION("Failed to load .ntc file: {}", err);
        return 1;
    }

    Context context{argv[0]};
    Device device = context.create_device(backend_name);
    // Note: Creating swapchains requires a stream created with StreamTag::GRAPHICS
    Stream stream = device.create_stream(StreamTag::GRAPHICS);

    InferEngine infer(device, stream);
    if (!infer.load_container(container)) {
        LUISA_ERROR_WITH_LOCATION("Failed to load container into InferEngine.");
        return 1;
    }

    uint32_t max_mips = container.numColorMips > 0 ? container.numColorMips : 1;
    active_mip = std::min(active_mip, max_mips - 1);

    // Find initial texture index matching requested channel
    size_t active_tex_idx = 0;
    for (size_t i = 0; i < container.textures.size(); ++i) {
        if (active_channel >= container.textures[i].firstChannel &&
            active_channel < container.textures[i].firstChannel + container.textures[i].numChannels) {
            active_tex_idx = i;
            break;
        }
    }

    uint2 win_size = make_uint2(
        std::min(container.width, 1024u),
        std::min(container.height, 1024u));

#ifdef LUISA_ENABLE_GUI
    Window window{"LuisaCompute NTC Material Viewer", win_size};
    Swapchain swapchain = device.create_swapchain(
        stream,
        SwapchainOption{
            .display = window.native_display(),
            .window = window.native_handle(),
            .size = win_size,
            .wants_hdr = false,
            .wants_vsync = true,
            .back_buffer_count = 2});

    Image<float> display_image = device.create_image<float>(swapchain.backend_storage(), win_size.x, win_size.y);

    // Bilinear/Nearest blit kernel to convert chosen texture channels to swapchain display image
    Kernel2D blit_kernel = [&](BufferFloat pixels, ImageFloat display, UInt tex_w, UInt tex_h,
                               UInt win_w, UInt win_h, UInt ch_offset, UInt ch_count) {
        UInt2 coord = dispatch_id().xy();
        $if (coord.x < win_w & coord.y < win_h) {
            Float u = (cast<float>(coord.x) + 0.5f) / cast<float>(win_w);
            Float v = (cast<float>(coord.y) + 0.5f) / cast<float>(win_h);
            UInt src_x = clamp(cast<uint>(u * cast<float>(tex_w)), 0u, tex_w - 1u);
            UInt src_y = clamp(cast<uint>(v * cast<float>(tex_h)), 0u, tex_h - 1u);
            UInt base = (src_y * tex_w + src_x) * 16u + ch_offset;
            $if (ch_count >= 3u) {
                Float r = clamp(pixels.read(base + 0u), 0.0f, 1.0f);
                Float g = clamp(pixels.read(base + 1u), 0.0f, 1.0f);
                Float b = clamp(pixels.read(base + 2u), 0.0f, 1.0f);
                display.write(coord, make_float4(r, g, b, 1.0f));
            } $else {
                Float val = clamp(pixels.read(base + 0u), 0.0f, 1.0f);
                display.write(coord, make_float4(val, val, val, 1.0f));
            };
        };
    };

    auto blit_shader = device.compile(blit_kernel);

    Buffer<float> gpu_pixels;
    uint32_t current_decomp_mip = 0xffffffffu;

    auto update_view = [&](uint32_t mip, size_t tex_idx) {
        uint32_t tw = std::max(1u, container.width >> mip);
        uint32_t th = std::max(1u, container.height >> mip);

        if (current_decomp_mip != mip) {
            gpu_pixels = infer.decompress_mip_device(mip, tw, th);
            current_decomp_mip = mip;
        }

        const auto &tex = container.textures[tex_idx];
        stream << blit_shader(gpu_pixels, display_image, tw, th, win_size.x, win_size.y,
                              tex.firstChannel, tex.numChannels)
                      .dispatch(win_size.x, win_size.y)
               << synchronize();

        auto title = luisa::format("LuisaCompute NTC Viewer - [{}/{}: {} (Ch {}-{})] Mip: {} ({}x{})",
                                          tex_idx + 1, container.textures.size(), tex.name,
                                          tex.firstChannel, tex.firstChannel + tex.numChannels - 1,
                                          mip, tw, th);
        glfwSetWindowTitle(window.window(), title.c_str());
        LUISA_INFO("{}", title);
    };

    update_view(active_mip, active_tex_idx);

    window.set_key_callback([&](Key key, KeyModifiers modifiers, Action action) {
        if (action == ACTION_PRESSED || action == ACTION_REPEATED) {
            if (key == KEY_ESCAPE) {
                window.set_should_close();
            } else if (key == KEY_RIGHT || key == KEY_D) {
                active_tex_idx = (active_tex_idx + 1) % container.textures.size();
                update_view(active_mip, active_tex_idx);
            } else if (key == KEY_LEFT || key == KEY_A) {
                active_tex_idx = (active_tex_idx + container.textures.size() - 1) % container.textures.size();
                update_view(active_mip, active_tex_idx);
            } else if (key == KEY_UP || key == KEY_W) {
                if (active_mip > 0) {
                    active_mip--;
                    update_view(active_mip, active_tex_idx);
                }
            } else if (key == KEY_DOWN || key == KEY_S) {
                if (active_mip + 1 < max_mips) {
                    active_mip++;
                    update_view(active_mip, active_tex_idx);
                }
            } else if (key >= KEY_1 && key < KEY_1 + static_cast<int>(container.textures.size())) {
                active_tex_idx = key - KEY_1;
                update_view(active_mip, active_tex_idx);
            }
        }
    });

    LUISA_INFO("Controls: [Left/Right or A/D]: Switch Texture | [Up/Down or W/S]: Switch Mip | [1-{}]: Jump to Texture | [ESC]: Quit",
               container.textures.size());

    while (!window.should_close()) {
        window.poll_events();
        stream << swapchain.present(display_image);
    }
    stream << synchronize();
#else
    uint32_t tw = std::max(1u, container.width >> active_mip);
    uint32_t th = std::max(1u, container.height >> active_mip);
    Buffer<float> gpu_pixels = infer.decompress_mip_device(active_mip, tw, th);
    Image<float> display_image = device.create_image<float>(PixelStorage::BYTE4, tw, th);
    const auto &tex = container.textures[active_tex_idx];
    Kernel2D blit_kernel = [&](BufferFloat pixels, ImageFloat display, UInt width, UInt height, UInt ch_offset) {
        UInt2 coord = dispatch_id().xy();
        $if (coord.x < width & coord.y < height) {
            UInt base = (coord.y * width + coord.x) * 16u + ch_offset;
            Float r = clamp(pixels.read(base + 0u), 0.0f, 1.0f);
            Float g = clamp(pixels.read(base + 1u), 0.0f, 1.0f);
            Float b = clamp(pixels.read(base + 2u), 0.0f, 1.0f);
            display.write(coord, make_float4(r, g, b, 1.0f));
        };
    };
    auto blit_shader = device.compile(blit_kernel);
    stream << blit_shader(gpu_pixels, display_image, tw, th, tex.firstChannel).dispatch(tw, th)
           << synchronize();

    std::vector<uint8_t> screenshot(tw * th * 4);
    stream << copy_from_device(display_image, screenshot.data()) << synchronize();
    stbi_write_png("viewer_output.png", tw, th, 4, screenshot.data(), tw * 4);
    LUISA_INFO("Headless mode: Saved viewer output to 'viewer_output.png' ({}x{})", tw, th);
#endif

    return 0;
}
