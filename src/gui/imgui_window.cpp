#include <mutex>
#include <algorithm>
#include <cmath>
#include <cstdlib>

#if defined(LUISA_PLATFORM_WINDOWS)
#define GLFW_EXPOSE_NATIVE_WIN32
#elif defined(LUISA_PLATFORM_APPLE)
#define GLFW_EXPOSE_NATIVE_COCOA
#else
#if LUISA_ENABLE_WAYLAND
#define GLFW_EXPOSE_NATIVE_WAYLAND
#endif
#define GLFW_EXPOSE_NATIVE_X11// TODO: other window compositors
#endif

#ifndef GLFW_INCLUDE_NONE
#define GLFW_INCLUDE_NONE
#endif

#include <GLFW/glfw3.h>
#include <GLFW/glfw3native.h>

#ifdef Bool// good job!
#undef Bool
#endif

#ifdef True// better!
#undef True
#endif

#ifdef False// best!
#undef False
#endif

#ifdef Always// ...
#undef Always
#endif

#ifdef None// speechless
#undef None
#endif

#ifdef Status// ???
#undef Status
#endif

#include <imgui.h>
#include <imgui_impl_glfw.h>

#include <luisa/core/logging.h>
#include <luisa/core/stl/queue.h>
#include <luisa/core/stl/vector.h>
#include <luisa/core/stl/map.h>
#include <luisa/runtime/device.h>
#include <luisa/runtime/shader.h>
#include <luisa/runtime/stream.h>
#include <luisa/runtime/buffer.h>
#include <luisa/runtime/image.h>
#include <luisa/runtime/bindless_array.h>
#include <luisa/runtime/swapchain.h>
#include <luisa/runtime/rtx/accel.h>
#include <luisa/runtime/raster/raster_shader.h>
#include <luisa/runtime/raster/raster_scene.h>
#include <luisa/runtime/raster/raster_state.h>
#include <luisa/runtime/raster/vertex_attribute.h>
#include <luisa/runtime/raster/app_data.h>
#include <luisa/dsl/sugar.h>
#include <luisa/dsl/raster/raster_kernel.h>
#include <luisa/backends/ext/raster_ext.hpp>
#include <luisa/backends/ext/raster_ext_interface.h>
#include <luisa/gui/imgui_window.h>
#include <luisa/core/stl/memory.h>

namespace luisa::compute::detail {

[[nodiscard]] inline auto glfw_window_native_handle(GLFWwindow *window) noexcept {
#if defined(LUISA_PLATFORM_WINDOWS)
    return reinterpret_cast<uint64_t>(glfwGetWin32Window(window));
#elif defined(LUISA_PLATFORM_APPLE)
    return reinterpret_cast<uint64_t>(glfwGetCocoaWindow(window));
#else
#if LUISA_ENABLE_WAYLAND
    if (glfwGetPlatform() == GLFW_PLATFORM_WAYLAND) {
        return reinterpret_cast<uint64_t>(glfwGetWaylandWindow(window));
    }
#endif
    return reinterpret_cast<uint64_t>(glfwGetX11Window(window));
#endif
}

[[nodiscard]] inline auto glfw_display_native_handle() noexcept -> uint64_t {
#if defined(LUISA_PLATFORM_WINDOWS) || defined(LUISA_PLATFORM_APPLE)
    return 0ull;
#else
#if LUISA_ENABLE_WAYLAND
    if (glfwGetPlatform() == GLFW_PLATFORM_WAYLAND) {
        return reinterpret_cast<uint64_t>(glfwGetWaylandDisplay());
    }
#endif
    return reinterpret_cast<uint64_t>(glfwGetX11Display());
#endif
}

/// Monitor content scale used for GUI DPI awareness. Screen coordinates are
/// physical pixels on Windows/X11, while macOS expresses display density
/// through the framebuffer scale (window coordinates are points) and Wayland
/// leaves scaling to the compositor; both report 1.0 here.
[[nodiscard]] inline float window_content_scale(GLFWwindow *window) noexcept {
#if defined(LUISA_PLATFORM_APPLE)
    static_cast<void>(window);
    return 1.0f;
#else
#if LUISA_ENABLE_WAYLAND
    if (glfwGetPlatform() == GLFW_PLATFORM_WAYLAND) {
        static_cast<void>(window);
        return 1.0f;
    }
#endif
    auto sx = 1.0f;
    auto sy = 1.0f;
    glfwGetWindowContentScale(window, &sx, &sy);
    auto scale = std::max(sx, sy);
    return (scale > 0.0f && std::isfinite(scale)) ? std::clamp(scale, 1.0f, 8.0f) : 1.0f;
#endif
}

/// Framebuffer-to-window size ratio (1.0 on Windows, 2.0 on Retina displays).
[[nodiscard]] inline float window_framebuffer_scale(GLFWwindow *window) noexcept {
    auto ww = 0;
    auto wh = 0;
    auto fw = 0;
    auto fh = 0;
    glfwGetWindowSize(window, &ww, &wh);
    glfwGetFramebufferSize(window, &fw, &fh);
    if (ww <= 0 || wh <= 0 || fw <= 0 || fh <= 0) { return 1.0f; }
    auto scale = std::max(static_cast<float>(fw) / static_cast<float>(ww),
                          static_cast<float>(fh) / static_cast<float>(wh));
    return (scale > 0.0f && std::isfinite(scale)) ? std::clamp(scale, 1.0f, 8.0f) : 1.0f;
}

inline constexpr auto base_font_size = 13.0f;// ImGui's default font size (ProggyClean 13 px)

struct alignas(16u) GUIVertex {
    float px;
    float py;
    float pz;
    uint clip_idx;
    float2 uv;
    uint packed_color;
    uint tex_id;
};

}// namespace luisa::compute::detail

LUISA_STRUCT(luisa::compute::detail::GUIVertex, px, py, pz, clip_idx, uv, packed_color, tex_id) {

    [[nodiscard]] auto p() const noexcept {
        return make_float3(px, py, pz);
    }

    [[nodiscard]] auto color() const noexcept {
        auto r = (packed_color & 0xffu).cast<float>() / 255.f;
        auto g = ((packed_color >> 8u) & 0xffu).cast<float>() / 255.f;
        auto b = ((packed_color >> 16u) & 0xffu).cast<float>() / 255.f;
        auto a = ((packed_color >> 24u) & 0xffu).cast<float>() / 255.f;
        return make_float4(r, g, b, a);
    }
};

namespace luisa::compute::detail {
/// Vertex layout for the rasterization mesh. The member order matches the
/// leading AppData attribute slots (position/normal/tangent/color) consumed by
/// the raster vertex stage, and every slot is a 16-byte float4 so the
/// corresponding `MeshFormat` declares four RGBA32F attributes with no packing
/// ambiguity. A `GUIVertex` is repacked into this layout once per frame.
struct alignas(16u) GUIMeshVertex {
    float4 position;// px, py (screen pixels, top-left origin), unused, unused
    float4 normal;  // clip_min.x, clip_min.y, clip_max.x, clip_max.y
    float4 tangent; // uv.x, uv.y, tex_id (as float), unused
    float4 color;   // per-vertex linear RGBA (not premultiplied)
};
static_assert(sizeof(GUIMeshVertex) == 64u);

/// Vertex-to-pixel varying for the rasterization renderer. Member 0 is the
/// mandatory float4 clip-space position (emitted with w == 1 so every varying
/// interpolates linearly/affine in pixel space). `screen` carries the same
/// top-left pixel coordinate as an ordinary interpolated varying so the pixel
/// stage can apply the scissor rectangle without depending on the
/// backend-specific SV_Position convention; the scissor and texture id are
/// constant across a command's triangles so interpolating them is exact.
struct GUIVarying {
    float4 position;
    float2 uv;
    float4 color;
    float2 clip_min;
    float2 clip_max;
    float tex_id;
    float2 screen;
};
}// namespace luisa::compute::detail

LUISA_STRUCT(luisa::compute::detail::GUIMeshVertex, position, normal, tangent, color) {
    [[nodiscard]] auto pixel() const noexcept {
        return position.xy();
    }
    [[nodiscard]] auto clip_min() const noexcept {
        return normal.xy();
    }
    [[nodiscard]] auto clip_max() const noexcept {
        return make_float2(normal.z, normal.w);
    }
    [[nodiscard]] auto tex_uv() const noexcept {
        return tangent.xy();
    }
    [[nodiscard]] auto tex_id() const noexcept {
        return tangent.z;
    }
    [[nodiscard]] auto rgba() const noexcept {
        return color;
    }
};

LUISA_STRUCT(luisa::compute::detail::GUIVarying, position, uv, color, clip_min, clip_max, tex_id, screen) {};

namespace luisa::compute {

class ImGuiWindow::Impl {
public:
    ImGuiWindow::MouseButtonCallback _mouse_button_callback;
    ImGuiWindow::CursorPositionCallback _cursor_position_callback;
    ImGuiWindow::WindowSizeCallback _window_size_callback;
    ImGuiWindow::KeyCallback _key_callback;
    ImGuiWindow::ScrollCallback _scroll_callback;
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
            LUISA_ASSERT(curr_ctx == _curr_ctx, "ImGui context mismatch.");
            ImGui::SetCurrentContext(_old_ctx);
        }

    public:
        CtxGuard(CtxGuard &&) noexcept = delete;
        CtxGuard(const CtxGuard &) noexcept = delete;
        CtxGuard &operator=(CtxGuard &&) noexcept = delete;
        CtxGuard &operator=(const CtxGuard &) noexcept = delete;
    };

    using Vertex = detail::GUIVertex;

private:
    Device &_device;
    Stream &_stream;
    Config _config;
    float _dpi_scale{1.0f};        // content scale currently applied to style/fonts
    float _dpi_scale_applied{0.0f};// 0 => never applied
    float _dpi_override{0.0f};     // > 0 => LUISA_GUI_DPI_SCALE override (testing)
    float _base_font_size{detail::base_font_size};// style.FontSizeBase before DPI scaling
    bool _owns_default_font{true}; // false once the app supplies its own fonts
    ImGuiStyle _base_style{};      // style snapshot at scale 1 (captured on first frame)
    bool _base_style_valid{false};
    uint64_t _font_texture_id{0u};// bindless id of the font texture (0 => none)
    ImGuiContext *_context;
    GLFWwindow *_main_window;
    Swapchain _main_swapchain;
    Image<float> _main_framebuffer;
    Image<float> _font_texture;
    BindlessArray _texture_array;
    uint _texture_array_offset{0u};
    luisa::queue<uint64_t> _texture_free_slots;
    luisa::unordered_map<uint64_t, std::pair<uint64_t, uint32_t>> _active_textures;
    luisa::map<std::pair<uint64_t, uint32_t>, uint64_t> _registered_images;
    luisa::unordered_map<GLFWwindow *, luisa::unique_ptr<Swapchain>> _platform_swapchains;
    luisa::unordered_map<GLFWwindow *, luisa::unique_ptr<Image<float>>> _platform_framebuffers;

    // for rendering
    Shader2D<Image<float>, float3> _clear_shader;
    Shader2D<Image<float> /* framebuffer */,
             uint2 /* clip base */,
             Accel /* accel */,
             Buffer<Triangle> /* triangles */,
             Buffer<Vertex> /* vertices */,
             BindlessArray /* textures */,
             Buffer<float4> /* clip rectangles */>
        _render_shader;
    Accel _accel;
    uint64_t _mesh_handle{~0ull};
    Buffer<Vertex> _vertex_buffer;
    Buffer<Triangle> _triangle_buffer;
    Buffer<float4> _clip_buffer;

    // Hardware rasterization renderer: replaces the per-frame acceleration
    // structure / mesh build and the per-pixel ray-marched depth peeling of the
    // path above with indexed-free triangle draws (no BVH, no ray query). A
    // `RasterKernel` is compiled once against the fixed GUIMeshVertex layout;
    // the ray-tracing resources stay so the renderer falls back to the legacy
    // path on backends without a working JIT raster pipeline (Vulkan's raster
    // path is AOT-only, see VkRasterExt::create_raster_shader). GUI triangles
    // are emitted in ImGui's back-to-front submission order and composited with
    // hardware premultiplied-alpha blending (dst = src + dst * (1 - src.a)),
    // reproducing the ray path's accumulation without reading the render target
    // inside the pixel stage. The vertex stage emits clip-space positions with
    // w == 1 (affine varyings); the scissor rectangle is applied through an
    // interpolated pixel-coordinate varying because the DSL has no
    // fragment-coordinate builtin and the SV_Position convention differs per
    // backend.
    RasterShader<float2 /* framebuffer size */,
                 float /* Y flip (+1 or -1) */,
                 BindlessArray /* textures */>
        _raster_shader;
    MeshFormat _mesh_format;
    bool _rasterization_enabled{false};
    // +1 or -1: maps ImGui's top-left screen origin to the rasterizer's NDC
    // convention (which framebuffer row a top-left pixel lands on). DX needs
    // +1; Vulkan needs -1 (overridable with LUISA_GUI_RASTER_FLIP).
    float _raster_y_flip{1.0f};
    // a single de-indexed vertex buffer reused every frame (grown to a
    // power-of-two capacity); no index buffer is required.
    Buffer<detail::GUIMeshVertex> _raster_vertex_buffer;
    // persistent host scratch reused across frames to avoid per-frame
    // allocations of the staging vector.
    luisa::vector<detail::GUIMeshVertex> _raster_vertices;

private:
    template<typename F>
    decltype(auto) _with_context(F &&f) noexcept {
        CtxGuard guard{_context};
        return luisa::invoke(std::forward<F>(f));
    }

private:
    /// Content scale to apply, honoring the disable flag and the test override.
    [[nodiscard]] float _content_scale() const noexcept {
        if (!_config.dpi_aware) { return 1.0f; }
        if (_dpi_override > 0.0f) { return _dpi_override; }
        return detail::window_content_scale(_main_window);
    }

    /// Re-bake the default font at the DPI-scaled pixel size. With a legacy
    /// backend (no ImGuiBackendFlags_RendererHasTextures) a font atlas can only
    /// be rebuilt outside ImGui::NewFrame()/EndFrame(), where it is unlocked;
    /// called from prepare_frame() accordingly.
    /// Note: secondary viewport windows share this atlas, so they use the main
    /// window's content scale.
    void _rebuild_font_atlas(float scale) noexcept {
        auto &io = ImGui::GetIO();
        // The application supplied its own fonts: never drop them, scale the
        // existing atlas instead (soft, but correct, on legacy backends).
        if (!_owns_default_font) { return; }
        io.Fonts->ClearFonts();
        auto &style = ImGui::GetStyle();
        // style.FontSizeBase must be set before AddFontDefault() (it picks the
        // bitmap or the vector font from it), and re-asserted afterwards:
        // ClearFonts() and AddFont() notify the surrounding contexts, which
        // writes the previous frame's font size back into the style.
        style.FontSizeBase = _base_font_size * scale;
        style.FontScaleDpi = 1.0f;
        ImFontConfig cfg{};
        // Logical pixel size == baked pixel size, so the text is crisp at the
        // scaled size; AddFontDefault() keeps the bitmap font for scale 1 and
        // picks the scalable vector font for larger sizes.
        cfg.SizePixels = _base_font_size * scale;
        // On platforms where window coordinates are points (macOS Retina),
        // rasterize at the framebuffer density while logical metrics stay
        // unchanged; the renderer applies DisplayFramebufferScale to vertices.
        cfg.RasterizerDensity = detail::window_framebuffer_scale(_main_window);
        io.Fonts->AddFontDefault(&cfg);
        style.FontSizeBase = _base_font_size * scale;
        style.FontScaleDpi = 1.0f;
    }

    void _apply_dpi_scale(float scale) noexcept {
        auto &style = ImGui::GetStyle();
        if (!_base_style_valid) {
            _base_style = style;
            _base_style_valid = true;
            _base_font_size = style.FontSizeBase > 0.0f ? style.FontSizeBase : detail::base_font_size;
            _owns_default_font = ImGui::GetIO().Fonts->Fonts.empty();
        }
        style = _base_style;
        if (scale != 1.0f) { style.ScaleAllSizes(scale); }
        if (_owns_default_font) {
            // The size is baked into the font source (see _rebuild_font_atlas),
            // so no additional global font scale factor is wanted here.
            _rebuild_font_atlas(scale);
        } else {
            style.FontScaleDpi = scale;
        }
        _dpi_scale = scale;
        _dpi_scale_applied = scale;
        LUISA_INFO("GUI DPI scale set to {:.2f} (font size {:.1f} px).", scale, style.FontSizeBase);
    }

    void _update_dpi_scale() noexcept {
        auto scale = _content_scale();
        if (_dpi_scale_applied > 0.0f && scale == _dpi_scale_applied) { return; }
        _apply_dpi_scale(scale);
    }

    /// Grow the window so that a logical size keeps its apparent size when
    /// screen coordinates are physical pixels (Windows/X11). On macOS window
    /// sizes are in points and Wayland handles scaling itself.
    void _apply_initial_window_size(uint2 logical_size) noexcept {
        auto scale = _content_scale();
        if (scale <= 1.0f) { return; }
        auto ww = 0;
        auto wh = 0;
        auto fw = 0;
        auto fh = 0;
        glfwGetWindowSize(_main_window, &ww, &wh);
        glfwGetFramebufferSize(_main_window, &fw, &fh);
        if (ww <= 0 || wh <= 0 || fw != ww || fh != wh) { return; }
        glfwSetWindowSize(_main_window,
                          static_cast<int>(std::lround(logical_size.x * scale)),
                          static_cast<int>(std::lround(logical_size.y * scale)));
    }

private:
    /// Whether this backend can host the JIT raster pipeline that `_raster_shader`
    /// needs. Two conditions must hold: the device must expose the raster
    /// extension at all (CUDA, HIP and the fallback backend expose none), and the
    /// extension must be JIT-capable (Vulkan exposes one, but its
    /// `create_raster_shader` is AOT-only -- VkRasterExt asserts `compile_only`).
    /// Backends failing this test keep the legacy ray-tracing renderer and never
    /// touch a raster resource.
    [[nodiscard]] static bool _rasterization_possible(Device &device) noexcept {
        if (device.extension<RasterExt>() == nullptr) { return false; }
        auto backend = device.backend_name();
        return backend != luisa::string_view{"vk"} &&
               backend != luisa::string_view{"cpu"};
    }

    void _rebuild_swapchain_if_changed(GLFWwindow *window, Swapchain &sc, Image<float> &fb) noexcept {
        auto fw = 0, fh = 0;
        glfwGetFramebufferSize(window, &fw, &fh);
        auto size = make_uint2(fw, fh);
        if (sc && fb && all(fb.size() == size)) { return; }
        if (sc || fb) {
            _stream.synchronize();
            sc = {};
            fb = {};
        }
        if (any(size != 0u)) {
            auto native_display = detail::glfw_display_native_handle();
            auto native_window = detail::glfw_window_native_handle(window);
            auto sc_options = SwapchainOption{
                .display = native_display,
                .window = native_window,
                .size = size,
                .wants_hdr = _config.hdr,
                .wants_vsync = _config.vsync,
                .back_buffer_count = _config.back_buffers};
            sc = _device.create_swapchain(_stream, sc_options);
            // The framebuffer is the color target of the hardware raster renderer
            // only while raster mode is active, so render-target support is
            // requested only then: creating it without the flag would make the
            // backend unable to bind it as an RTV (on DX the device is removed
            // with DXGI_ERROR_INVALID_CALL when the raster pipeline is created),
            // while asking for the flag on a backend that renders through the
            // ray-tracing path is a raster request that backend may not support.
            fb = _device.create_image<float>(sc.backend_storage(), size, 1u, false, _rasterization_enabled);
        }
    }
    void _on_imgui_create_window(ImGuiViewport *vp) noexcept {
        auto glfw_window = static_cast<GLFWwindow *>(vp->PlatformHandle);
        LUISA_ASSERT(glfw_window != nullptr && glfw_window != _main_window,
                     "Invalid GLFW window.");
        auto sc = luisa::make_unique<Swapchain>();
        auto fb = luisa::make_unique<Image<float>>();
        _rebuild_swapchain_if_changed(glfw_window, *sc, *fb);
        _platform_swapchains[glfw_window] = std::move(sc);
        _platform_framebuffers[glfw_window] = std::move(fb);
    }
    void _on_imgui_destroy_window(ImGuiViewport *vp) noexcept {
        _stream.synchronize();
        if (auto glfw_window = static_cast<GLFWwindow *>(vp->PlatformHandle);
            glfw_window != _main_window) {
            _platform_swapchains.erase(glfw_window);
            _platform_framebuffers.erase(glfw_window);
        }
    }
    void _on_imgui_set_window_size(ImGuiViewport *vp, ImVec2) noexcept {
        auto glfw_window = static_cast<GLFWwindow *>(vp->PlatformHandle);
        LUISA_ASSERT(glfw_window != nullptr, "Invalid GLFW window.");
        auto &sc = glfw_window == _main_window ? _main_swapchain : *_platform_swapchains.at(glfw_window);
        auto &fb = glfw_window == _main_window ? _main_framebuffer : *_platform_framebuffers.at(glfw_window);
        _rebuild_swapchain_if_changed(glfw_window, sc, fb);
    }
    void _on_imgui_render_window(ImGuiViewport *vp, void *) noexcept {
        auto glfw_window = static_cast<GLFWwindow *>(vp->PlatformHandle);
        auto &sc = *_platform_swapchains.at(glfw_window);
        auto &fb = *_platform_framebuffers.at(glfw_window);
        _draw(sc, fb, vp->DrawData);
    }

public:
    Impl(Device &device, Stream &stream, luisa::string name, const Config &config) noexcept
        : _device{device},
          _stream{stream},
          _config{config},
          _context{[] {
              IMGUI_CHECKVERSION();
              return ImGui::CreateContext();
          }()},
          _main_window{nullptr} {

        // optional content scale override, does not apply when dpi_aware is disabled
        if (auto env = std::getenv("LUISA_GUI_DPI_SCALE"); env != nullptr && *env != '\0') {
            auto value = std::strtof(env, nullptr);
            if (std::isfinite(value) && value > 0.0f) {
                _dpi_override = std::clamp(value, 1.0f, 8.0f);
            }
        }
        // Decide the renderer *before* creating any device resource, so that a
        // backend without a usable raster pipeline neither gets a raster render
        // target (see _rebuild_swapchain_if_changed) nor a raster pipeline.
        // LUISA_GUI_RASTER=0 forces the legacy ray-tracing path.
        auto raster_requested = config.rasterization;
        if (auto env = std::getenv("LUISA_GUI_RASTER"); env != nullptr && *env != '\0') {
            raster_requested = env[0] != '0';
        }
        _rasterization_enabled = raster_requested && _rasterization_possible(_device);
        if (_rasterization_enabled) {
            // rasterizer NDC-Y convention override for the rasterization renderer
            // (see _raster_y_flip); only meaningful while raster mode is active.
            if (auto env = std::getenv("LUISA_GUI_RASTER_FLIP"); env != nullptr && *env != '\0') {
                auto value = std::strtof(env, nullptr);
                _raster_y_flip = value < 0.0f ? -1.0f : 1.0f;
            }
        }
        LUISA_INFO("GUI hardware rasterization {} on backend '{}'.",
                   _rasterization_enabled ? "enabled" : "disabled",
                   _device.backend_name());

        // initialize GLFW
        static std::once_flag once_flag;
        std::call_once(once_flag, [] {
            glfwSetErrorCallback([](int error, const char *description) noexcept {
                if (error != GLFW_NO_ERROR) [[likely]] {
                    LUISA_WARNING("GLFW Error (code = 0x{:08x}): {}.", error, description);
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
                                        name.c_str(),
                                        nullptr, nullptr);
        LUISA_ASSERT(_main_window != nullptr, "Failed to create GLFW window.");
        glfwSetWindowUserPointer(_main_window, this);
        // apply the monitor content scale to the initial window size
        _dpi_scale = _content_scale();
        _apply_initial_window_size(config.size);
        // TODO: imgui
        glfwSetMouseButtonCallback(_main_window, [](GLFWwindow *window, int button, int action, int mods) noexcept {
            // if (ImGui::GetIO().WantCaptureMouse) {// ImGui is handling the mouse
            //     ImGui_ImplGlfw_MouseButtonCallback(window, button, action, mods);
            // } else {
            auto self = static_cast<Impl *>(glfwGetWindowUserPointer(window));
            auto x = 0.0;
            auto y = 0.0;
            glfwGetCursorPos(self->_main_window, &x, &y);
            if (auto &&cb = self->_mouse_button_callback) {
                cb(static_cast<MouseButton>(button), static_cast<Action>(action),
                   make_float2(static_cast<float>(x), static_cast<float>(y)));
            }
            // }
        });
        glfwSetCursorPosCallback(_main_window, [](GLFWwindow *window, double x, double y) noexcept {
            auto self = static_cast<Impl *>(glfwGetWindowUserPointer(window));
            if (auto &&cb = self->_cursor_position_callback) { cb(make_float2(static_cast<float>(x), static_cast<float>(y))); }
        });
        glfwSetWindowSizeCallback(_main_window, [](GLFWwindow *window, int width, int height) noexcept {
            auto self = static_cast<Impl *>(glfwGetWindowUserPointer(window));
            if (auto &&cb = self->_window_size_callback) { cb(make_uint2(width, height)); }
        });
        glfwSetKeyCallback(_main_window, [](GLFWwindow *window, int key, int scancode, int action, int mods) noexcept {
            // if (ImGui::GetIO().WantCaptureKeyboard) {// ImGui is handling the keyboard
            //     ImGui_ImplGlfw_KeyCallback(window, key, scancode, action, mods);
            // } else {
            auto self = static_cast<Impl *>(glfwGetWindowUserPointer(window));
            if (auto &&cb = self->_key_callback) {
                cb(static_cast<Key>(key), mods, static_cast<Action>(action));
            }
            // }
        });
        glfwSetScrollCallback(_main_window, [](GLFWwindow *window, double dx, double dy) noexcept {
            // if (ImGui::GetIO().WantCaptureMouse) {// ImGui is handling the mouse
            //     ImGui_ImplGlfw_ScrollCallback(window, dx, dy);
            // } else {
            auto self = static_cast<Impl *>(glfwGetWindowUserPointer(window));
            if (auto &&cb = self->_scroll_callback) {
                cb(make_float2(static_cast<float>(dx), static_cast<float>(dy)));
            }
            // }
        });

        // create main swapchain
        _rebuild_swapchain_if_changed(_main_window, _main_swapchain, _main_framebuffer);

        // create texture array
        _texture_array = _device.create_bindless_array();

        // create shaders
        _clear_shader = _device.compile<2>([](ImageFloat fb, Float3 color) noexcept {
            auto tid = dispatch_id().xy();
            fb.write(tid, make_float4(color, 1.f));
        });
        _render_shader = _device.compile<2>([ssaa = config.ssaa](ImageFloat fb, UInt2 offset, AccelVar accel,
                                                                 BufferVar<Triangle> triangles, BufferVar<Vertex> vertices,
                                                                 BindlessVar texture_array, BufferFloat4 clip_rects) noexcept {
            auto tid = offset + dispatch_id().xy();
            $if (all(tid < dispatch_size().xy())) {
                constexpr auto eps = 1e-4f;// slightly offset the center to improve watertightness
                auto offsets = ssaa ?
                                   luisa::vector<float2>{
                                       make_float2(1.f / 3.f + eps, 1.f / 3.f - eps),
                                       make_float2(2.f / 3.f + eps, 1.f / 3.f + eps),
                                       make_float2(2.f / 3.f - eps, 2.f / 3.f + eps),
                                       make_float2(1.f / 3.f - eps, 2.f / 3.f - eps),
                                   } :
                                   luisa::vector<float2>{make_float2(.5f + eps, .5f - eps)};
                auto k = static_cast<float>(1. / static_cast<double>(offsets.size()));
                auto sum = def(make_float3(0.f));
                auto old = fb.read(tid).xyz();
                for (auto offset : offsets) {
                    auto o = make_float3(make_float2(tid) + offset, -1.f);
                    auto d = make_float3(0.f, 0.f, 1.f);
                    auto ray = make_ray(o, d);
                    auto beta = def(1.f);
                    auto depth = def(0u);
                    $while (beta > 1e-3f & depth < 16u) {
                        depth += 1u;
                        auto hit = accel.intersect(ray, {});
                        $if (!hit->is_triangle()) { $break; };
                        auto triangle = triangles->read(hit.prim);
                        auto v0 = vertices->read(triangle.i0);
                        auto v1 = vertices->read(triangle.i1);
                        auto v2 = vertices->read(triangle.i2);
                        auto p = hit->triangle_interpolate(v0->p(), v1->p(), v2->p());
                        auto clip = clip_rects->read(v0.clip_idx);
                        $if (all(p.xy() >= clip.xy() && p.xy() <= clip.zw())) {
                            auto uv = hit->triangle_interpolate(v0.uv, v1.uv, v2.uv);
                            auto c = hit->triangle_interpolate(v0->color(), v1->color(), v2->color());
                            auto tex_id = v0.tex_id;
                            $if (tex_id != 0u) {
                                c *= texture_array->tex2d(v0.tex_id).sample(uv);
                            };
                            sum += k * c.xyz() * beta * c.w;
                            beta *= 1.f - c.w;
                        };
                        // step through the layer
                        auto pp = p + make_float3(0.f, 0.f, depth_peeling_step * .5);
                        ray = make_ray(pp, d);
                    };
                    // accumulate the background
                    sum += k * beta * old;
                }
                fb.write(tid, make_float4(sum, 1.f));
            };
        });

        // Compile the rasterization renderer. The backend was already checked to
        // provide a JIT-capable raster pipeline (see _rasterization_possible), so
        // no raster extension is touched on backends that lack one; if the
        // pipeline cannot be created after all, the flag is cleared and the
        // ray-tracing path above takes over.
        if (_rasterization_enabled) {
            // One RGBA32F attribute per AppData slot the vertex stage reads.
            VertexAttribute raster_attributes[]{
                {VertexAttributeType::Position, PixelFormat::RGBA32F},
                {VertexAttributeType::Normal, PixelFormat::RGBA32F},
                {VertexAttributeType::Tangent, PixelFormat::RGBA32F},
                {VertexAttributeType::Color, PixelFormat::RGBA32F},
            };
            _mesh_format.emplace_vertex_stream(raster_attributes);
            RasterStageKernel raster_vert = [](Var<detail::GUIMeshVertex> v,
                                               Float2 fb_size,
                                               Float y_flip) noexcept {
                Var<detail::GUIVarying> o;
                // Map the packed top-left pixel position to clip space with
                // w == 1 so the rasterizer maps it back to the same pixel and
                // every varying interpolates linearly. y_flip selects the NDC
                // y convention.
                auto p = v->pixel();
                auto ndc_x = p.x / fb_size.x * 2.f - 1.f;
                auto ndc_y = y_flip * (1.f - p.y / fb_size.y * 2.f);
                o.position = make_float4(ndc_x, ndc_y, 0.f, 1.f);
                o.uv = v->tex_uv();
                o.color = v->rgba();
                o.clip_min = v->clip_min();
                o.clip_max = v->clip_max();
                o.tex_id = v->tex_id();
                o.screen = p;
                return o;
            };
            RasterStageKernel raster_pixel = [](Var<detail::GUIVarying> i,
                                                BindlessVar texture_array) noexcept {
                // The interpolated `screen` (top-left pixel center) is clipped
                // against the command's scissor rectangle. SV_Position is not
                // used because its pixel-center convention differs per backend.
                $if (any(i.screen < i.clip_min - .5f) | any(i.screen > i.clip_max - .5f)) {
                    raster_discard();
                };
                auto c = i.color;
                auto tex_id = i.tex_id.cast<uint>();
                $if (tex_id != 0u) {
                    c *= texture_array->tex2d(tex_id).sample(i.uv);
                };
                // Premultiplied output; the blend state composites it over the
                // framebuffer.
                return make_float4(c.xyz() * c.w, c.w);
            };
            RasterKernel<decltype(raster_vert), decltype(raster_pixel)> raster_kernel{raster_vert, raster_pixel};
            _raster_shader = _device.compile(raster_kernel, _mesh_format);
            _rasterization_enabled = static_cast<bool>(_raster_shader);
            if (!_rasterization_enabled) [[unlikely]] {
                LUISA_WARNING_WITH_LOCATION(
                    "Failed to create the GUI raster pipeline on backend '{}'; "
                    "falling back to the ray-tracing path.",
                    _device.backend_name());
            }
        }

        // TODO: install user GLFW callbacks?

        // imgui config
        _with_context([this, &config] {
            auto &io = ImGui::GetIO();
            io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;// Enable Keyboard Controls
            io.ConfigFlags |= ImGuiConfigFlags_NavEnableGamepad; // Enable Gamepad Controls

            if (config.docking) {
                io.ConfigFlags |= ImGuiConfigFlags_DockingEnable;// Enable Docking
            }

            // Wayland does not support querying for window position so multi-viewport is disabled
            if (config.multi_viewport && glfwGetPlatform() != GLFW_PLATFORM_WAYLAND) {
                io.ConfigFlags |= ImGuiConfigFlags_ViewportsEnable;// Enable Multi-Viewport / Platform Windows
            }

            // styles
            ImGui::StyleColorsDark();
            if (io.ConfigFlags & ImGuiConfigFlags_ViewportsEnable) [[likely]] {
                auto &style = ImGui::GetStyle();
                style.WindowRounding = 5.f;
                style.Colors[ImGuiCol_WindowBg].w = .9f;
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
    }

    ~Impl() noexcept {
        _stream.synchronize();
        _with_context([] {
            // to inform ImGui that the renderer is shutdown
            auto &io = ImGui::GetIO();
            io.BackendRendererName = nullptr;
            io.BackendRendererUserData = nullptr;
            io.BackendFlags &= ~(ImGuiBackendFlags_RendererHasVtxOffset | ImGuiBackendFlags_RendererHasViewports);
            ImGui_ImplGlfw_Shutdown();
        });
        ImGui::DestroyContext(_context);
        _stream.synchronize();
        _platform_swapchains.clear();
        _platform_framebuffers.clear();
        _main_swapchain = {};
        _main_framebuffer = {};
        glfwDestroyWindow(_main_window);
        if (_accel) {
            _accel = {};
            _device.impl()->destroy_mesh(_mesh_handle);
        }
    }

public:
    [[nodiscard]] auto handle() const noexcept { return _main_window; }
    [[nodiscard]] auto context() const noexcept { return _context; }
    [[nodiscard]] auto dpi_scale() const noexcept { return _dpi_scale; }
    [[nodiscard]] auto &swapchain() const noexcept { return const_cast<Swapchain &>(_main_swapchain); }
    [[nodiscard]] auto &framebuffer() const noexcept { return const_cast<Image<float> &>(_main_framebuffer); }
    [[nodiscard]] auto should_close() const noexcept {
        return static_cast<bool>(glfwWindowShouldClose(_main_window));
    }
    [[nodiscard]] auto set_should_close(bool b) noexcept {
        glfwSetWindowShouldClose(_main_window, b);
    }
    [[nodiscard]] auto register_texture(const Image<float> &image, Sampler sampler) noexcept {
        return _with_context([&] {
            auto key = std::make_pair(image.uid(), sampler.code());
            if (auto iter = _registered_images.find(key);
                iter != _registered_images.end()) {
                return iter->second;
            }
            auto tex_id = [&] {
                if (!_texture_free_slots.empty()) {
                    auto t = _texture_free_slots.front();
                    _texture_free_slots.pop();
                    return t;
                }
                return static_cast<uint64_t>(++_texture_array_offset);
            }();
            _texture_array.emplace_on_update(tex_id, image, sampler);
            _active_textures.emplace(tex_id, key);
            _registered_images.emplace(key, tex_id);
            // Note: update will be postponed to the next render_frame
            return tex_id;
        });
    }
    void unregister_texture(uint64_t tex_id) noexcept {
        if (auto iter = _active_textures.find(tex_id);
            iter != _active_textures.end()) {
            _texture_array.remove_tex2d_on_update(tex_id);
            _texture_free_slots.emplace(tex_id);
            auto key = iter->second;
            _active_textures.erase(iter);
            _registered_images.erase(key);
        } else {
            LUISA_WARNING_WITH_LOCATION(
                "Unregistering an inactive texture (id = {}). "
                "This operation is ignored.",
                tex_id);
        }
        // Note: update will be postponed to the next render_frame
    }

private:
    void _create_font_texture() noexcept {
        auto &io = ImGui::GetIO();
        auto pixels = static_cast<unsigned char *>(nullptr);
        auto width = 0, height = 0;
        io.Fonts->GetTexDataAsRGBA32(&pixels, &width, &height);
        if (width <= 0 || height <= 0) { return; }
        // TODO: mipmaps?
        if (!_font_texture || any(_font_texture.size() != make_uint2(width, height))) {
            if (_font_texture) { _stream << synchronize(); }
            _font_texture = _device.create_image<float>(PixelStorage::BYTE4, width, height, 1);
        }
        _stream << _font_texture.copy_from(luisa::span{pixels, static_cast<size_t>(width * height * 4)});
        auto tex_id = register_texture(_font_texture, Sampler::linear_point_edge());
        if (_font_texture_id != 0u && _font_texture_id != tex_id) {
            // Register the new texture before unregistering the stale id so the
            // deferred removal can never race the new emplace on the same slot.
            unregister_texture(_font_texture_id);
        }
        _font_texture_id = tex_id;
        io.Fonts->SetTexID(tex_id);
    }

private:
    luisa::vector<Vertex> _vertices;
    luisa::vector<Triangle> _triangles;
    luisa::vector<float4> _clip_rects;

    static constexpr auto depth_peeling_step = 1. / 64.;

    void _build_accel() noexcept {

        // create resources if not created
        AccelOption o{
            .hint = AccelOption::UsageHint::FAST_BUILD,
            .allow_compaction = false,
            .allow_update = false};
        if (!_accel) {
            _accel = _device.create_accel(o);
            _mesh_handle = _device.impl()->create_mesh(o).handle;
            _accel.emplace_back_handle(_mesh_handle, make_float4x4(1.f), 0xffu, true, 0u);
        }
        if (!_vertex_buffer) { _vertex_buffer = _device.create_buffer<Vertex>(std::max(next_pow2(_vertices.size()), 64_k)); }
        if (!_triangle_buffer) { _triangle_buffer = _device.create_buffer<Triangle>(std::max(next_pow2(_triangles.size()), 64_k)); }
        if (!_clip_buffer) { _clip_buffer = _device.create_buffer<float4>(std::max(next_pow2(_clip_rects.size()), static_cast<size_t>(64u))); }

        // resize buffers if insufficient
        if (_vertex_buffer.size() < _vertices.size() ||
            _triangle_buffer.size() < _triangles.size() ||
            _clip_buffer.size() < _clip_rects.size()) {
            _stream.synchronize();
            if (_vertex_buffer.size() < _vertices.size()) {
                _vertex_buffer = {};
                _vertex_buffer = _device.create_buffer<Vertex>(std::max(next_pow2(_vertices.size()), 64_k));
            }
            if (_triangle_buffer.size() < _triangles.size()) {
                _triangle_buffer = {};
                _triangle_buffer = _device.create_buffer<Triangle>(std::max(next_pow2(_triangles.size()), 64_k));
            }
            if (_clip_buffer.size() < _clip_rects.size()) {
                _clip_buffer = {};
                _clip_buffer = _device.create_buffer<float4>(std::max(next_pow2(_clip_rects.size()), static_cast<size_t>(64u)));
            }
        }
        // update the buffers and build the accel
        _stream << _vertex_buffer.view(0u, _vertices.size()).copy_from(luisa::span{_vertices.data(), _vertices.size()})
                << _triangle_buffer.view(0u, _triangles.size()).copy_from(luisa::span{_triangles.data(), _triangles.size()})
                << _clip_buffer.view(0u, _clip_rects.size()).copy_from(luisa::span{_clip_rects.data(), _clip_rects.size()})
                << luisa::make_unique<MeshBuildCommand>(
                       _mesh_handle, AccelBuildRequest::FORCE_BUILD,
                       _vertex_buffer.handle(), 0u,
                       _vertices.size() * sizeof(Vertex), sizeof(Vertex),
                       _triangle_buffer.handle(), 0u,
                       _triangles.size() * sizeof(Triangle))
                << _accel.build(AccelBuildRequest::FORCE_BUILD);
    }

    /// Screen-space rasterization state: no depth test/write and
    /// premultiplied-alpha "over" blending (dst = src + dst * (1 - src.a)), so
    /// the triangles composite over the application-drawn background and over
    /// each other in ImGui's back-to-front submission order.
    [[nodiscard]] static RasterState _raster_state() noexcept {
        return RasterState{
            .fill_mode = FillMode::Solid,
            .cull_mode = CullMode::None,
            .blend_state = BlendState{
                .enable_blend = true,
                .op = BlendOp::Add,
                .prim_op = BlendWeight::One,
                .img_op = BlendWeight::OneMinusPrimAlpha},
            .depth_state = DepthState{},
            .stencil_state = StencilState{},
            .topology = TopologyType::Triangle,
            .front_counter_clockwise = false,
            .depth_clip = false};
    }

    /// Repack the ImGui draw data into the raster mesh layout: a de-indexed
    /// triangle list kept in submission order (the back-to-front order the
    /// blend relies on). Returns false when the frame has no triangles.
    bool _build_raster_mesh(ImDrawData *draw_data) noexcept {
        auto clip_offset = make_float2(draw_data->DisplayPos.x, draw_data->DisplayPos.y);
        auto clip_scale = make_float2(draw_data->FramebufferScale.x, draw_data->FramebufferScale.y);
        auto clip_size = make_float2(draw_data->DisplaySize.x, draw_data->DisplaySize.y) * clip_scale;
        auto transform = [clip_offset, clip_scale](ImVec2 p) noexcept {
            return (make_float2(p.x, p.y) - clip_offset) * clip_scale;
        };
        _raster_vertices.clear();
        if (any(clip_size <= 0.f)) { return false; }
        _raster_vertices.reserve(64_k);
        for (auto i = 0u; i < draw_data->CmdLists.Size; i++) {
            auto cmd_list = draw_data->CmdLists[i];
            for (auto j = 0u; j < cmd_list->CmdBuffer.Size; j++) {
                auto cmd = &cmd_list->CmdBuffer[j];
                // user callback
                if (auto callback = cmd->UserCallback) {
                    // we ignore ImDrawCallback_ResetRenderState since we don't
                    // have any state to reset
                    if (callback != ImDrawCallback_ResetRenderState) {
                        callback(cmd_list, cmd);
                    }
                    continue;
                }
                // render command
                auto clip_min = max((make_float2(cmd->ClipRect.x, cmd->ClipRect.y) - clip_offset) * clip_scale, 0.f);
                auto clip_max = min((make_float2(cmd->ClipRect.z, cmd->ClipRect.w) - clip_offset) * clip_scale, clip_size);
                if (any(clip_max <= clip_min) || cmd->ElemCount == 0) { continue; }
                auto tex_id = [this, cmd] {
                    auto t = cmd->GetTexID();
                    if (t != 0u && !_active_textures.contains(t)) {
                        LUISA_WARNING_WITH_LOCATION(
                            "Using an unregistered texture (id = {}). "
                            "Replaced with a null texture.",
                            t);
                        return 0u;
                    }
                    return static_cast<uint>(t);
                }();
                auto make_vertex = [&](ImDrawVert v) noexcept {
                    auto p = transform(v.pos);
                    auto col = v.col;
                    auto r = static_cast<float>(col & 0xffu) / 255.f;
                    auto g = static_cast<float>((col >> 8u) & 0xffu) / 255.f;
                    auto b = static_cast<float>((col >> 16u) & 0xffu) / 255.f;
                    auto a = static_cast<float>((col >> 24u) & 0xffu) / 255.f;
                    auto mf = detail::GUIMeshVertex{};
                    mf.position = make_float4(p.x, p.y, 0.f, 0.f);
                    mf.normal = make_float4(clip_min.x, clip_min.y, clip_max.x, clip_max.y);
                    mf.tangent = make_float4(v.uv.x, v.uv.y, static_cast<float>(tex_id), 0.f);
                    mf.color = make_float4(r, g, b, a);
                    return mf;
                };
                for (auto t = 0u; t < cmd->ElemCount; t += 3u) {
                    auto i0 = cmd_list->IdxBuffer[cmd->IdxOffset + t + 0u] + cmd->VtxOffset;
                    auto i1 = cmd_list->IdxBuffer[cmd->IdxOffset + t + 1u] + cmd->VtxOffset;
                    auto i2 = cmd_list->IdxBuffer[cmd->IdxOffset + t + 2u] + cmd->VtxOffset;
                    _raster_vertices.emplace_back(make_vertex(cmd_list->VtxBuffer[i0]));
                    _raster_vertices.emplace_back(make_vertex(cmd_list->VtxBuffer[i1]));
                    _raster_vertices.emplace_back(make_vertex(cmd_list->VtxBuffer[i2]));
                }
            }
        }
        return !_raster_vertices.empty();
    }

    /// Upload the frame's raster mesh and issue a single draw. The mesh is a
    /// de-indexed triangle list, so the raster path neither builds a BVH nor
    /// allocates an index buffer. The full-framebuffer viewport keeps the
    /// pixel-stage coordinate reconstruction consistent; the per-command scissor
    /// rectangle is applied inside the pixel stage.
    void _draw_raster(Image<float> &fb) noexcept {
        auto capacity = std::max(next_pow2(_raster_vertices.size()), 64_k);
        if (!_raster_vertex_buffer || _raster_vertex_buffer.size() < capacity) {
            _stream.synchronize();
            _raster_vertex_buffer = _device.create_buffer<detail::GUIMeshVertex>(capacity);
        }
        auto fb_size = fb.size();
        auto fb_size2 = make_float2(fb_size);
        auto vertex_view = VertexBufferView{_raster_vertex_buffer};
        auto meshes = luisa::vector<RasterMesh>{};
        meshes.emplace_back(luisa::span<VertexBufferView const>{&vertex_view, 1u},
                            static_cast<uint>(_raster_vertices.size()), 1u, 0u);
        if (_texture_array.dirty()) { _stream << _texture_array.update(); }
        _stream << _raster_vertex_buffer.view(0u, _raster_vertices.size())
                                                  .copy_from(luisa::span{_raster_vertices.data(), _raster_vertices.size()})
                << _raster_shader(fb_size2, _raster_y_flip, _texture_array)
                       .draw(std::move(meshes), _mesh_format,
                             Viewport{0u, 0u, fb_size.x, fb_size.y},
                             _raster_state(), nullptr, fb);
    }

    void _draw(Swapchain &sc, Image<float> &fb, ImDrawData *draw_data) noexcept {

        auto vp = draw_data->OwnerViewport;
        auto glfw_window = static_cast<GLFWwindow *>(vp->PlatformHandle);
        _rebuild_swapchain_if_changed(glfw_window, sc, fb);

        // skip minimized windows
        if (!sc || !fb || all(fb.size() == 0u)) { return; }

        // clear framebuffer if needed
        if (glfw_window != _main_window &&
            !(vp->Flags & ImGuiViewportFlags_NoRendererClear)) {
            _stream << _clear_shader(fb, make_float3(0.f)).dispatch(fb.size());
        }
        // render imgui draw data to framebuffer
        auto clip_size = make_float2(draw_data->DisplaySize.x, draw_data->DisplaySize.y) *
                         make_float2(draw_data->FramebufferScale.x, draw_data->FramebufferScale.y);
        if (all(clip_size > 0.f)) {
            if (_rasterization_enabled) {
                if (_build_raster_mesh(draw_data)) { _draw_raster(fb); }
            } else {
                _draw_raytraced(fb, draw_data);
            }
        }
        _stream << sc.present(fb);
    }

    /// Legacy ray-tracing renderer: rebuilds the acceleration structure and the
    /// mesh every frame and depth-peels with per-pixel rays. Kept as the
    /// fallback for backends without a working JIT raster pipeline.
    void _draw_raytraced(Image<float> &fb, ImDrawData *draw_data) noexcept {
        auto clip_offset = make_float2(draw_data->DisplayPos.x, draw_data->DisplayPos.y);
        auto clip_scale = make_float2(draw_data->FramebufferScale.x, draw_data->FramebufferScale.y);
        auto clip_size = make_float2(draw_data->DisplaySize.x, draw_data->DisplaySize.y) * clip_scale;
        auto transform = [clip_offset, clip_scale](ImVec2 p) noexcept {
            return (make_float2(p.x, p.y) - clip_offset) * clip_scale;
        };
        _vertices.clear();
        _triangles.clear();
        _clip_rects.clear();
        _vertices.reserve(64_k);
        _triangles.reserve(64_k);
        _clip_rects.reserve(64u);
        auto accum_clip_min = make_float2(std::numeric_limits<float>::max());
        auto accum_clip_max = make_float2(-std::numeric_limits<float>::max());
        for (auto i = 0u; i < draw_data->CmdLists.Size; i++) {
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
                // render command
                auto clip_min = max((make_float2(cmd->ClipRect.x, cmd->ClipRect.y) - clip_offset) * clip_scale, 0.f);
                auto clip_max = min((make_float2(cmd->ClipRect.z, cmd->ClipRect.w) - clip_offset) * clip_scale, clip_size);
                if (any(clip_max <= clip_min) || cmd->ElemCount == 0) { continue; }
                // process the command
                auto clip_idx = static_cast<uint>(_clip_rects.size());
                _clip_rects.emplace_back(make_float4(clip_min, clip_max));
                auto tex_id = [this, cmd] {
                    auto t = cmd->GetTexID();
                    if (t != 0u && !_active_textures.contains(t)) {
                        LUISA_WARNING_WITH_LOCATION(
                            "Using an unregistered texture (id = {}). "
                            "Replaced with a null texture.",
                            t);
                        return 0u;
                    }
                    return static_cast<uint>(t);
                }();
                accum_clip_min = min(accum_clip_min, clip_min);
                accum_clip_max = max(accum_clip_max, clip_max);
                // triangles
                for (auto t = 0u; t < cmd->ElemCount; t += 3u) {
                    auto o = static_cast<uint>(_triangles.size());
                    auto make_vertex = [&](ImDrawVert v) noexcept {
                        auto p = transform(v.pos);
                        // from back to front
                        auto z = (draw_data->TotalIdxCount / 3u - 1u - o) * depth_peeling_step;
                        return Vertex{.px = p.x,
                                      .py = p.y,
                                      .pz = static_cast<float>(z),
                                      .clip_idx = clip_idx,
                                      .uv = make_float2(v.uv.x, v.uv.y),
                                      .packed_color = v.col,
                                      .tex_id = tex_id};
                    };
                    auto i0 = cmd_list->IdxBuffer[cmd->IdxOffset + t + 0u] + cmd->VtxOffset;
                    auto i1 = cmd_list->IdxBuffer[cmd->IdxOffset + t + 1u] + cmd->VtxOffset;
                    auto i2 = cmd_list->IdxBuffer[cmd->IdxOffset + t + 2u] + cmd->VtxOffset;
                    auto v0 = cmd_list->VtxBuffer[i0];
                    auto v1 = cmd_list->VtxBuffer[i1];
                    auto v2 = cmd_list->VtxBuffer[i2];
                    _triangles.emplace_back(Triangle{o * 3u + 0u, o * 3u + 1u, o * 3u + 2u});
                    _vertices.emplace_back(make_vertex(v0));
                    _vertices.emplace_back(make_vertex(v1));
                    _vertices.emplace_back(make_vertex(v2));
                }
            }
        }
        if (!_triangles.empty() && all(accum_clip_max > accum_clip_min)) {
            _build_accel();
            auto clip_min_floor = make_uint2(floor(accum_clip_min));
            auto clip_max_ceil = make_uint2(ceil(accum_clip_max));
            if (_texture_array.dirty()) { _stream << _texture_array.update(); }
            _stream << _render_shader(fb, clip_min_floor, _accel,
                                      _triangle_buffer, _vertex_buffer,
                                      _texture_array, _clip_buffer)
                           .dispatch(clip_max_ceil - clip_min_floor);
        }
    }


    void _render() noexcept {
        auto &io = ImGui::GetIO();
        if (auto draw_data = ImGui::GetDrawData()) {
            _draw(_main_swapchain, _main_framebuffer, draw_data);
        }
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
        // Apply the monitor content scale before ImGui consumes
        // sizes/fonts this frame (may re-bake the font atlas).
        _update_dpi_scale();
        // ImGui checks if the font texture is created in
        // ImGui::NewFrame() so we have to create it here
        if (!ImGui::GetIO().Fonts->IsBuilt() || !_font_texture) { _create_font_texture(); }
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

ImGuiWindow::ImGuiWindow() noexcept = default;

ImGuiWindow::ImGuiWindow(Device &device, Stream &stream,
                         luisa::string name,
                         const Config &config) noexcept
    : ImGuiWindow{} { create(device, stream, std::move(name), config); }

ImGuiWindow::~ImGuiWindow() noexcept = default;

ImGuiWindow::ImGuiWindow(ImGuiWindow &&) noexcept = default;
ImGuiWindow &ImGuiWindow::operator=(ImGuiWindow &&) noexcept = default;

GLFWwindow *ImGuiWindow::handle() const noexcept {
    LUISA_ASSERT(_impl, "ImGuiWindow not created.");
    return _impl->handle();
}

Swapchain &ImGuiWindow::swapchain() const noexcept {
    LUISA_ASSERT(_impl, "ImGuiWindow not created.");
    return _impl->swapchain();
}

Image<float> &ImGuiWindow::framebuffer() const noexcept {
    LUISA_ASSERT(_impl, "ImGuiWindow not created.");
    return _impl->framebuffer();
}

float ImGuiWindow::dpi_scale() const noexcept {
    LUISA_ASSERT(_impl, "ImGuiWindow not created.");
    return _impl->dpi_scale();
}

void ImGuiWindow::create(Device &device, Stream &stream, luisa::string name, const Config &config) noexcept {
    destroy();
    _impl = luisa::make_unique<Impl>(device, stream, std::move(name), config);
}

void ImGuiWindow::destroy() noexcept {
    _impl = nullptr;
}

ImGuiContext *ImGuiWindow::context() const noexcept {
    LUISA_ASSERT(_impl, "ImGuiWindow not created.");
    return _impl->context();
}

namespace detail {
[[nodiscard]] static auto &imgui_context_stack() noexcept {
    static thread_local luisa::vector<ImGuiContext *> stack;
    return stack;
}
}// namespace detail

void ImGuiWindow::push_context() noexcept {
    LUISA_ASSERT(_impl, "ImGuiWindow not created.");
    auto &stack = detail::imgui_context_stack();
    auto curr_ctx = ImGui::GetCurrentContext();
    stack.emplace_back(curr_ctx);
    auto ctx = _impl->context();
    ImGui::SetCurrentContext(ctx);
    detail::imgui_context_stack().emplace_back(ctx);
}

void ImGuiWindow::pop_context() noexcept {
    LUISA_ASSERT(_impl, "ImGuiWindow not created.");
    if (auto &stack = detail::imgui_context_stack();
        !stack.empty() && stack.back() == _impl->context()) {
        stack.pop_back();
        auto ctx = stack.empty() ? nullptr : stack.back();
        ImGui::SetCurrentContext(ctx);
    } else {
        LUISA_WARNING_WITH_LOCATION("Invalid ImGui context stack.");
    }
}

ImGuiWindow &ImGuiWindow::set_mouse_callback(ImGuiWindow::MouseButtonCallback cb) noexcept {
    static_cast<ImGuiWindow::Impl *>(_impl.get())->_mouse_button_callback = std::move(cb);
    return *this;
}

ImGuiWindow &ImGuiWindow::set_cursor_position_callback(ImGuiWindow::CursorPositionCallback cb) noexcept {
    static_cast<ImGuiWindow::Impl *>(_impl.get())->_cursor_position_callback = std::move(cb);
    return *this;
}

ImGuiWindow &ImGuiWindow::set_window_size_callback(ImGuiWindow::WindowSizeCallback cb) noexcept {
    static_cast<ImGuiWindow::Impl *>(_impl.get())->_window_size_callback = std::move(cb);
    return *this;
}

ImGuiWindow &ImGuiWindow::set_key_callback(ImGuiWindow::KeyCallback cb) noexcept {
    static_cast<ImGuiWindow::Impl *>(_impl.get())->_key_callback = std::move(cb);
    return *this;
}

ImGuiWindow &ImGuiWindow::set_scroll_callback(ImGuiWindow::ScrollCallback cb) noexcept {
    static_cast<ImGuiWindow::Impl *>(_impl.get())->_scroll_callback = std::move(cb);
    return *this;
}

uint64_t ImGuiWindow::register_texture(const Image<float> &image, const Sampler &sampler) noexcept {
    LUISA_ASSERT(_impl, "ImGuiWindow not created.");
    return _impl->register_texture(image, sampler);
}

void ImGuiWindow::unregister_texture(uint64_t tex_id) noexcept {
    LUISA_ASSERT(_impl, "ImGuiWindow not created.");
    _impl->unregister_texture(tex_id);
}

bool ImGuiWindow::should_close() const noexcept {
    LUISA_ASSERT(_impl, "ImGuiWindow not created.");
    return _impl->should_close();
}

void ImGuiWindow::set_should_close(bool b) noexcept {
    LUISA_ASSERT(_impl, "ImGuiWindow not created.");
    _impl->set_should_close(b);
}

void ImGuiWindow::prepare_frame() noexcept {
    LUISA_ASSERT(_impl, "ImGuiWindow not created.");
    _impl->prepare_frame();
}

void ImGuiWindow::render_frame() noexcept {
    LUISA_ASSERT(_impl, "ImGuiWindow not created.");
    _impl->render_frame();
}

}// namespace luisa::compute
