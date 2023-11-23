#include <mutex>
#include <random>

#if defined(LUISA_PLATFORM_WINDOWS)
#define GLFW_EXPOSE_NATIVE_WIN32
#elif defined(LUISA_PLATFORM_APPLE)
#define GLFW_EXPOSE_NATIVE_COCOA
#else
#define GLFW_EXPOSE_NATIVE_X11// TODO: other window compositors
#endif

#ifndef GLFW_INCLUDE_NONE
#define GLFW_INCLUDE_NONE
#endif

#include <GLFW/glfw3.h>
#include <GLFW/glfw3native.h>

#include <imgui.h>
#include <imgui_impl_glfw.h>

#include <luisa/core/logging.h>
#include <luisa/core/stl/queue.h>
#include <luisa/runtime/device.h>
#include <luisa/runtime/shader.h>
#include <luisa/runtime/stream.h>
#include <luisa/runtime/image.h>
#include <luisa/runtime/bindless_array.h>
#include <luisa/runtime/swapchain.h>
#include <luisa/dsl/sugar.h>
#include <luisa/gui/imgui_window.h>

namespace luisa::compute {

namespace detail {
[[nodiscard]] inline auto glfw_window_native_handle(GLFWwindow *window) noexcept {
#if defined(LUISA_PLATFORM_WINDOWS)
    return reinterpret_cast<uint64_t>(glfwGetWin32Window(window));
#elif defined(LUISA_PLATFORM_APPLE)
    return reinterpret_cast<uint64_t>(glfwGetCocoaWindow(window));
#else
    return reinterpret_cast<uint64_t>(glfwGetX11Window(window));
#endif
}
}// namespace detail

class ImGuiWindow::Impl {

private:
    class CtxGuard {

    private:
        ImGuiContext *_curr_ctx;
        ImGuiContext *_old_ctx;

    public:
        explicit CtxGuard(ImGuiContext *curr) noexcept
            : _curr_ctx{curr} {
            _old_ctx = ImGui::GetCurrentContext();
            ImGui::SetCurrentContext(_curr_ctx);
        }

        ~CtxGuard() noexcept {
            auto curr_ctx = ImGui::GetCurrentContext();
            LUISA_ASSERT(curr_ctx == nullptr /* destroyed */ || curr_ctx == _curr_ctx,
                         "ImGui context mismatch.");
            ImGui::SetCurrentContext(_old_ctx);
        }

    public:
        CtxGuard(CtxGuard &&) noexcept = delete;
        CtxGuard(const CtxGuard &) noexcept = delete;
        CtxGuard &operator=(CtxGuard &&) noexcept = delete;
        CtxGuard &operator=(const CtxGuard &) noexcept = delete;
    };

private:
    Device &_device;
    Stream &_stream;
    Config _config;
    ImGuiContext *_context;
    GLFWwindow *_main_window;
    Swapchain _main_swapchain;
    Image<float> _main_framebuffer;
    Image<float> _font_texture;
    BindlessArray _texture_array;
    luisa::unordered_map<ImGuiID, Swapchain> _platform_swapchains;
    luisa::unordered_map<ImGuiID, Image<float>> _platform_framebuffers;
    Shader2D<Image<float>, float3> _clear_shader;
    Shader2D<Image<float>, float4, uint2> _simple_shader;

private:
    template<typename F>
    decltype(auto) _with_context(F &&f) noexcept {
        CtxGuard guard{_context};
        return luisa::invoke(std::forward<F>(f));
    }

private:
    void _on_imgui_create_window(ImGuiViewport *vp) noexcept {
        auto glfw_window = static_cast<GLFWwindow *>(vp->PlatformHandle);
        LUISA_ASSERT(glfw_window != nullptr && glfw_window != _main_window,
                     "Invalid GLFW window.");
        auto frame_width = 0, frame_height = 0;
        glfwGetFramebufferSize(glfw_window, &frame_width, &frame_height);
        auto native_handle = detail::glfw_window_native_handle(glfw_window);
        auto sc = _device.create_swapchain(native_handle, _stream,
                                           make_uint2(frame_width, frame_height),
                                           _config.hdr, _config.vsync,
                                           _config.back_buffers);
        auto fb = _device.create_image<float>(sc.backend_storage(), frame_width, frame_height);
        _platform_swapchains[vp->ID] = std::move(sc);
        _platform_framebuffers[vp->ID] = std::move(fb);
    }
    void _on_imgui_destroy_window(ImGuiViewport *vp) noexcept {
        _stream.synchronize();
        if (auto glfw_window = static_cast<GLFWwindow *>(vp->PlatformHandle);
            glfw_window != _main_window) {
            _platform_swapchains.erase(vp->ID);
        }
    }
    void _on_imgui_set_window_size(ImGuiViewport *vp, ImVec2) noexcept {
        _stream.synchronize();
        auto frame_width = 0, frame_height = 0;
        auto glfw_window = static_cast<GLFWwindow *>(vp->PlatformHandle);
        LUISA_ASSERT(glfw_window != nullptr, "Invalid GLFW window.");
        glfwGetFramebufferSize(glfw_window, &frame_width, &frame_height);
        auto native_handle = detail::glfw_window_native_handle(glfw_window);
        auto &sc = glfw_window == _main_window ? _main_swapchain : _platform_swapchains.at(vp->ID);
        auto &fb = glfw_window == _main_window ? _main_framebuffer : _platform_framebuffers.at(vp->ID);
        sc = {};
        fb = {};
        sc = _device.create_swapchain(native_handle, _stream,
                                      make_uint2(frame_width, frame_height),
                                      _config.hdr, _config.vsync,
                                      _config.back_buffers);
        fb = _device.create_image<float>(sc.backend_storage(), frame_width, frame_height);
    }
    void _on_imgui_render_window(ImGuiViewport *vp, void *) noexcept {
        auto &sc = _platform_swapchains.at(vp->ID);
        auto &fb = _platform_framebuffers.at(vp->ID);
        _draw(sc, fb, vp->DrawData);
    }

public:
    Impl(Device &device, Stream &stream, const Config &config) noexcept
        : _device{device},
          _stream{stream},
          _config{config},
          _context{[] {
              IMGUI_CHECKVERSION();
              return ImGui::CreateContext();
          }()},
          _main_window{nullptr} {

        // initialize GLFW
        static std::once_flag once_flag;
        std::call_once(once_flag, [] {
            glfwSetErrorCallback([](int error, const char *description) noexcept {
                if (error != GLFW_NO_ERROR) [[likely]] {
                    LUISA_ERROR("GLFW Error (code = 0x{:08x}): {}.", error, description);
                }
            });
            if (!glfwInit()) [[unlikely]] {
                LUISA_ERROR_WITH_LOCATION("Failed to initialize GLFW.");
            }
        });

        // create main window
        glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
        glfwWindowHint(GLFW_RESIZABLE, config.resizable);
        _main_window = glfwCreateWindow(static_cast<int>(config.size.x),
                                        static_cast<int>(config.size.y),
                                        config.name.c_str(),
                                        nullptr, nullptr);
        LUISA_ASSERT(_main_window != nullptr, "Failed to create GLFW window.");
        glfwSetWindowUserPointer(_main_window, this);

        // create main swapchain
        auto frame_width = 0, frame_height = 0;
        glfwGetFramebufferSize(_main_window, &frame_width, &frame_height);
        auto native_handle = detail::glfw_window_native_handle(_main_window);
        _main_swapchain = _device.create_swapchain(
            native_handle, stream,
            make_uint2(frame_width, frame_height),
            config.hdr, config.vsync, config.back_buffers);
        _main_framebuffer = _device.create_image<float>(
            _main_swapchain.backend_storage(), frame_width, frame_height);

        // TODO: install callbacks

        // imgui config
        _with_context([this] {
            auto &io = ImGui::GetIO();
            io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;// Enable Keyboard Controls
            io.ConfigFlags |= ImGuiConfigFlags_NavEnableGamepad; // Enable Gamepad Controls
            io.ConfigFlags |= ImGuiConfigFlags_DockingEnable;    // Enable Docking
            io.ConfigFlags |= ImGuiConfigFlags_ViewportsEnable;  // Enable Multi-Viewport / Platform Windows

            // styles
            ImGui::StyleColorsDark();
            if (io.ConfigFlags & ImGuiConfigFlags_ViewportsEnable) [[likely]] {
                auto &style = ImGui::GetStyle();
                style.WindowRounding = 0.0f;
                style.Colors[ImGuiCol_WindowBg].w = 1.0f;
            }

            // register glfw window
            ImGui_ImplGlfw_InitForOther(_main_window, true);

            // register renderer (this)
            io.BackendRendererUserData = this;
            io.BackendRendererName = "imgui_impl_luisa";
            io.BackendFlags |= ImGuiBackendFlags_RendererHasVtxOffset;
            io.BackendFlags |= ImGuiBackendFlags_RendererHasViewports;
            if (io.ConfigFlags & ImGuiConfigFlags_ViewportsEnable) [[likely]] {
                auto &platform_io = ImGui::GetPlatformIO();
                static constexpr auto imgui_get_this = [] {
                    return ImGui::GetCurrentContext() ?
                               static_cast<Impl *>(ImGui::GetIO().BackendRendererUserData) :
                               nullptr;
                };
                platform_io.Renderer_CreateWindow = [](ImGuiViewport *vp) noexcept {
                    if (auto self = imgui_get_this()) {
                        self->_on_imgui_create_window(vp);
                    }
                };
                platform_io.Renderer_DestroyWindow = [](ImGuiViewport *vp) noexcept {
                    if (auto self = imgui_get_this()) {
                        self->_on_imgui_destroy_window(vp);
                    }
                };
                platform_io.Renderer_SetWindowSize = [](ImGuiViewport *vp, ImVec2 size) noexcept {
                    if (auto self = imgui_get_this()) {
                        self->_on_imgui_set_window_size(vp, size);
                    }
                };
                platform_io.Renderer_RenderWindow = [](ImGuiViewport *vp, void *user_data) noexcept {
                    if (auto self = imgui_get_this()) {
                        self->_on_imgui_render_window(vp, user_data);
                    }
                };
            }
        });

        // create shaders
        _clear_shader = _device.compile<2>([](ImageFloat fb, Float3 color) noexcept {
            auto tid = dispatch_id().xy();
            fb.write(tid, make_float4(color, 1.f));
        });
        _simple_shader = _device.compile<2>([](ImageFloat fb, Float4 color, UInt2 offset) noexcept {
            auto tid = offset + dispatch_id().xy();
            $if (all(tid < dispatch_size().xy())) {
                auto old = fb.read(tid).xyz();
                auto alpha = color.w;
                fb.write(tid, make_float4(lerp(old, color.xyz(), alpha), 1.f));
            };
        });
    }

    ~Impl() noexcept {
        _stream.synchronize();
        _with_context([] {
            ImGui_ImplGlfw_Shutdown();
            ImGui::DestroyContext();
        });
        LUISA_ASSERT(_platform_swapchains.empty(),
                     "Some ImGui windows are not destroyed.");
        _main_swapchain = {};
        glfwDestroyWindow(_main_window);
    }

public:
    [[nodiscard]] auto handle() const noexcept { return _main_window; }
    [[nodiscard]] auto context() const noexcept { return _context; }
    [[nodiscard]] auto &swapchain() const noexcept { return const_cast<Swapchain &>(_main_swapchain); }
    [[nodiscard]] auto &framebuffer() const noexcept { return const_cast<Image<float> &>(_main_framebuffer); }
    [[nodiscard]] auto should_close() const noexcept {
        return static_cast<bool>(glfwWindowShouldClose(_main_window));
    }
    [[nodiscard]] auto set_should_close(bool b) noexcept {
        glfwSetWindowShouldClose(_main_window, b);
    }

private:
    void _create_font_texture() noexcept {
        auto &io = ImGui::GetIO();
        auto pixels = static_cast<unsigned char *>(nullptr);
        auto width = 0, height = 0;
        io.Fonts->GetTexDataAsAlpha8(&pixels, &width, &height);
        // TODO: mipmaps?
        _font_texture = _device.create_image<float>(PixelStorage::BYTE1, width, height, 1);
        _texture_array = _device.create_bindless_array();
        _texture_array.emplace_on_update(0u, _font_texture, Sampler::linear_point_mirror());
        _stream << _font_texture.copy_from(pixels)
                << _texture_array.update();
        io.Fonts->SetTexID(reinterpret_cast<ImTextureID>(0ull));
    }

    void _draw(Swapchain &sc, Image<float> &fb, ImDrawData *draw_data) noexcept {
        auto total = 0u;
        auto vp = draw_data->OwnerViewport;
        // clear framebuffer if needed
        if (!(vp->Flags & ImGuiViewportFlags_NoRendererClear)) {
            _stream << _clear_shader(fb, make_float3(0.f)).dispatch(fb.size());
        }
        // render imgui draw data to framebuffer
        auto clip_offset = make_float2(draw_data->DisplayPos.x, draw_data->DisplayPos.y);
        auto clip_scale = make_float2(draw_data->FramebufferScale.x, draw_data->FramebufferScale.y);
        auto clip_size = make_float2(draw_data->DisplaySize.x, draw_data->DisplaySize.y) * clip_scale;
        if (all(clip_size > 0.f)) {
            for (auto i = 0u; i < draw_data->CmdListsCount; i++) {
                auto cmd_list = draw_data->CmdLists[i];
                for (auto j = 0u; j < cmd_list->CmdBuffer.Size; j++) {
                    auto cmd = &cmd_list->CmdBuffer[j];
                    // user callback
                    if (auto callback = cmd->UserCallback) {
                        // we ignore ImDrawCallback_ResetRenderState
                        // since we don't have any state to reset
                        if (callback != ImDrawCallback_ResetRenderState) {
                            callback(cmd_list, cmd);
                        }
                        continue;
                    }
                    total++;
                    // render command
                    auto clip_min = max((make_float2(cmd->ClipRect.x, cmd->ClipRect.y) - clip_offset) * clip_scale, 0.f);
                    auto clip_max = min((make_float2(cmd->ClipRect.z, cmd->ClipRect.w) - clip_offset) * clip_scale, clip_size);
                    if (any(clip_max <= clip_min) || cmd->ElemCount == 0) { continue; }
                    LUISA_INFO("clip_min = {}, clip_max = {}", clip_min, clip_max);
                    static std::mt19937 random{std::random_device{}()};
                    std::uniform_real_distribution<float> dist{0.f, 1.f};
                    auto color = make_float4(dist(random), dist(random), dist(random), .2f);
                    _stream << _simple_shader(fb, color, make_uint2(clip_min))
                                   .dispatch(make_uint2(clip_max - clip_min));
                }
            }
        }
        LUISA_INFO("Total Draw Command: {}.", total);
        // present framebuffer to swapchain
        _stream << sc.present(fb);
    }

    void _render() noexcept {
        auto &io = ImGui::GetIO();
        _draw(_main_swapchain, _main_framebuffer, ImGui::GetDrawData());
        if (io.ConfigFlags & ImGuiConfigFlags_ViewportsEnable) {
            ImGui::UpdatePlatformWindows();
            ImGui::RenderPlatformWindowsDefault();
        }
    }

private:
    bool _inside_frame{false};
    ImGuiContext *_old_ctx{nullptr};

public:
    void prepare_frame() noexcept {
        LUISA_ASSERT(!_inside_frame,
                     "Already inside an ImGui frame. "
                     "Did you forget to call ImGuiWindow::render_frame()?");
        _inside_frame = true;
        _old_ctx = ImGui::GetCurrentContext();
        glfwPollEvents();
        ImGui::SetCurrentContext(_context);
        // ImGui checks if the font texture is created in
        // ImGui::NewFrame() so we have to create it here
        if (!_font_texture) { _create_font_texture(); }
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();
    }
    void render_frame() noexcept {
        LUISA_ASSERT(_inside_frame,
                     "Not inside an ImGui frame. "
                     "Did you forget to call ImGuiWindow::prepare_frame()?");
        LUISA_ASSERT(ImGui::GetCurrentContext() == _context,
                     "Invalid ImGui context.");
        ImGui::Render();
        _render();
        ImGui::SetCurrentContext(_old_ctx);
        _old_ctx = nullptr;
        _inside_frame = false;
    }
};

ImGuiWindow::ImGuiWindow(Device &device, Stream &stream, const Config &config) noexcept
    : _impl{luisa::make_unique<Impl>(device, stream, config)} {}

ImGuiWindow::~ImGuiWindow() noexcept = default;

GLFWwindow *ImGuiWindow::handle() const noexcept { return _impl->handle(); }
Swapchain &ImGuiWindow::swapchain() const noexcept { return _impl->swapchain(); }
Image<float> &ImGuiWindow::framebuffer() const noexcept { return _impl->framebuffer(); }
ImGuiContext *ImGuiWindow::context() const noexcept { return _impl->context(); }
bool ImGuiWindow::should_close() const noexcept { return _impl->should_close(); }
void ImGuiWindow::set_should_close(bool b) noexcept { _impl->set_should_close(b); }
void ImGuiWindow::prepare_frame() noexcept { _impl->prepare_frame(); }
void ImGuiWindow::render_frame() noexcept { _impl->render_frame(); }

}// namespace luisa::compute
