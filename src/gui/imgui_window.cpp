#include <mutex>
#include <algorithm>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <cstring>

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
    float2 _font_white_uv{};      // TexUvWhitePixel of the uploaded font atlas
    bool _font_white_uv_opaque{false};// true iff bilinear footprint at _font_white_uv is verified 0xffffffff
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

    enum class RenderMode : uint32_t {
        HARDWARE_RASTER,
        COMPUTE_RASTER,
        RAY_TRACED,
    };

    static constexpr uint compute_tile_size = 16u;

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
    // Tiled compute 2D rasterizer: renders ImGui triangles in exact
    // front-to-back submission order using a compacted list of active tiles,
    // precomputed canonical watertight edge equations, and a single coalesced
    // ByteBuffer upload per frame. Used on backends without a JIT-capable
    // RasterExt (e.g. HIP, CUDA, Vulkan, fallback) so ImGui works without
    // requiring ray-tracing support or per-frame BVH rebuilds.
    Shader2D<Image<float> /* framebuffer */,
             uint /* active_tile_count */,
             uint /* active_grid_x */,
             uint4 /* byte offsets: {tile_tris, clips, tri_hot, tri_extra} */,
             ByteBuffer /* coalesced frame data */,
             BindlessArray /* textures */>
        _compute_render_shader;
    Accel _accel;
    uint64_t _mesh_handle{~0ull};
    Buffer<Vertex> _vertex_buffer;
    Buffer<Triangle> _triangle_buffer;
    Buffer<float4> _clip_buffer;
    ByteBuffer _compute_frame_buffer;

    // Hardware rasterization renderer: replaces the per-frame acceleration
    // structure / mesh build and the per-pixel ray-marched depth peeling of the
    // path above with indexed-free triangle draws (no BVH, no ray query). A
    // `RasterKernel` is compiled once against the fixed GUIMeshVertex layout.
    // GUI triangles are emitted in ImGui's back-to-front submission order and
    // composited with hardware premultiplied-alpha blending
    // (dst = src + dst * (1 - src.a)), reproducing the ray path's accumulation
    // without reading the render target inside the pixel stage. The vertex
    // stage emits clip-space positions with w == 1 (affine varyings); the
    // scissor rectangle is applied through an interpolated pixel-coordinate
    // varying because the DSL has no fragment-coordinate builtin and the
    // SV_Position convention differs per backend.
    RasterShader<float2 /* framebuffer size */,
                 float /* Y flip (+1 or -1) */,
                 BindlessArray /* textures */>
        _raster_shader;
    MeshFormat _mesh_format;
    RenderMode _render_mode{RenderMode::COMPUTE_RASTER};
    // +1 or -1: maps ImGui's top-left screen origin to the rasterizer's NDC
    // convention (which framebuffer row a top-left pixel lands on). DX needs
    // +1; Vulkan needs -1 (overridable with LUISA_GUI_RASTER_FLIP).
    float _raster_y_flip{1.0f};
    // a single de-indexed vertex buffer reused every frame (grown to a
    // power-of-two capacity); no index buffer is required.
    Buffer<detail::GUIMeshVertex> _raster_vertex_buffer;
    // persistent host scratch reused across frames to avoid per-frame
    // allocations of the staging vectors.
    struct TileTriBounds {
        uint16_t tx0;
        uint16_t ty0;
        uint16_t tx1;
        uint16_t ty1;
    };
    struct TileRowSpan {
        uint32_t tile_start;
        uint32_t count;
    };
    // 64-byte (single cache line) hot per-triangle record for compute rasterization.
    // Precomputes canonical signed edge equations (lo.x, lo.y, sx, sy) such that
    // e(q) = sx * (q.y - lo.y) + sy * (q.x - lo.x) is bit-exact with canonical
    // endpoint ordering and shared edges evaluate to exact IEEE-754 negatives.
    struct alignas(16) ComputeTriHot {
        uint clip_idx;
        uint flags_tex;// bits 0..2: top-left mask (e0..e2), bit 3: flat_color, bits 4..31: tex_id
        uint col0;     // packed RGBA8 color of vertex 0
        uint extra_idx;// index into ComputeTriExtra (valid iff !flat_color || tex_id != 0)
        float4 e0;     // opposite v0 (edge p1 -> p2)
        float4 e1;     // opposite v1 (edge p2 -> p0)
        float4 e2;     // opposite v2 (edge p0 -> p1)
    };
    static_assert(sizeof(ComputeTriHot) == 64u);
    static_assert(offsetof(ComputeTriHot, e0) == 16u);
    static_assert(offsetof(ComputeTriHot, e1) == 32u);
    static_assert(offsetof(ComputeTriHot, e2) == 48u);
    // 32-byte cold payload allocated only for textured or non-flat-color triangles.
    struct alignas(16) ComputeTriExtra {
        float2 uv0;
        float2 uv1;
        float2 uv2;
        uint col1;
        uint col2;
    };
    static_assert(sizeof(ComputeTriExtra) == 32u);
    static_assert(offsetof(ComputeTriExtra, uv0) == 0u);
    static_assert(offsetof(ComputeTriExtra, uv2) == 16u);
    static_assert(offsetof(ComputeTriExtra, col1) == 24u);

    luisa::vector<detail::GUIMeshVertex> _raster_vertices;
    luisa::vector<ComputeTriHot> _tri_hot;
    luisa::vector<ComputeTriExtra> _tri_extra;
    luisa::vector<TileTriBounds> _tile_tri_bounds;
    luisa::vector<TileRowSpan> _tile_row_spans;
    luisa::vector<uint> _large_tri_span_ends;
    luisa::vector<uint> _tile_counts;
    luisa::vector<uint2> _active_tiles;
    luisa::vector<uint> _tile_cursors;
    luisa::vector<std::byte> _frame_upload_bytes;

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
    /// Backends failing this test use the tiled compute rasterizer and never
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
            // only while hardware raster mode is active, so render-target support
            // is requested only then: creating it without the flag would make the
            // backend unable to bind it as an RTV (on DX the device is removed
            // with DXGI_ERROR_INVALID_CALL when the raster pipeline is created),
            // while asking for the flag on a backend that renders through compute
            // or ray tracing is a raster request that backend may not support.
            fb = _device.create_image<float>(
                sc.backend_storage(), size, 1u, false,
                _render_mode == RenderMode::HARDWARE_RASTER);
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
        // target (see _rebuild_swapchain_if_changed) nor touches RasterExt, and
        // backends without ray-tracing support (e.g. non-RT HIP architectures)
        // never compile the legacy ray-tracing shader.
        // LUISA_GUI_RASTER=0 forces the legacy ray-tracing path;
        // LUISA_GUI_RASTER=compute forces the tiled compute rasterizer.
        auto raster_requested = config.rasterization;
        auto force_compute_raster = false;
        if (auto env = std::getenv("LUISA_GUI_RASTER"); env != nullptr && *env != '\0') {
            auto env_sv = luisa::string_view{env};
            if (env_sv == "compute") {
                raster_requested = true;
                force_compute_raster = true;
            } else {
                raster_requested = env[0] != '0';
            }
        }
        if (!raster_requested) {
            _render_mode = RenderMode::RAY_TRACED;
        } else if (!force_compute_raster && _rasterization_possible(_device)) {
            _render_mode = RenderMode::HARDWARE_RASTER;
        } else {
            _render_mode = RenderMode::COMPUTE_RASTER;
        }
        if (_render_mode == RenderMode::HARDWARE_RASTER) {
            // rasterizer NDC-Y convention override for the hardware rasterizer
            // (see _raster_y_flip); only meaningful while hardware raster mode is active.
            if (auto env = std::getenv("LUISA_GUI_RASTER_FLIP"); env != nullptr && *env != '\0') {
                auto value = std::strtof(env, nullptr);
                _raster_y_flip = value < 0.0f ? -1.0f : 1.0f;
            }
        }
        auto render_mode_name = [](RenderMode mode) noexcept {
            switch (mode) {
                case RenderMode::HARDWARE_RASTER: return "hardware rasterization";
                case RenderMode::COMPUTE_RASTER: return "compute tiled rasterization";
                case RenderMode::RAY_TRACED: return "ray tracing";
            }
            return "unknown";
        };
        LUISA_INFO("GUI renderer: {} on backend '{}'.",
                   render_mode_name(_render_mode),
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

        // Compile the hardware rasterization renderer when selected. The
        // backend was already checked to provide a JIT-capable raster pipeline
        // (see _rasterization_possible), so no raster extension is touched on
        // backends that lack one; if the pipeline cannot be created after all,
        // fall back to the tiled compute rasterizer.
        if (_render_mode == RenderMode::HARDWARE_RASTER) {
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
            if (!_raster_shader) [[unlikely]] {
                LUISA_WARNING_WITH_LOCATION(
                    "Failed to create the GUI hardware raster pipeline on backend '{}'; "
                    "falling back to the tiled compute rasterizer.",
                    _device.backend_name());
                _render_mode = RenderMode::COMPUTE_RASTER;
            }
        }

        if (_render_mode == RenderMode::COMPUTE_RASTER) {
            _compute_render_shader = _device.compile<2>(
                [ssaa = config.ssaa](ImageFloat fb,
                                     UInt active_tile_count,
                                     UInt active_grid_x,
                                     UInt4 section_offsets,
                                     ByteBufferVar frame_buf,
                                     BindlessVar texture_array) noexcept {
                    set_block_size(compute_tile_size, compute_tile_size, 1u);
                    auto active_idx = block_id().y * active_grid_x + block_id().x;
                    $if (active_idx >= active_tile_count) { $return(); };
                    auto tile_info = frame_buf.read<uint2>(active_idx * 8u);
                    auto next_info = frame_buf.read<uint2>((active_idx + 1u) * 8u);
                    auto packed_xy = tile_info.x;
                    auto range_begin = tile_info.y;
                    auto range_end = next_info.y;
                    auto tile_xy = make_uint2(packed_xy & 0xffffu, packed_xy >> 16u);
                    auto tid = tile_xy * compute_tile_size + thread_id().xy();
                    $if (any(tid >= fb.size())) { $return(); };

                    auto off_tile_tris = section_offsets.x;
                    auto off_clips = section_offsets.y;
                    auto off_tri_hot = section_offsets.z;
                    auto off_tri_extra = section_offsets.w;

                    auto unpack_color = [](Expr<uint> packed) noexcept {
                        return make_float4(
                                   (packed & 0xffu).cast<float>(),
                                   ((packed >> 8u) & 0xffu).cast<float>(),
                                   ((packed >> 16u) & 0xffu).cast<float>(),
                                   (packed >> 24u).cast<float>()) *
                               (1.f / 255.f);
                    };

                    // Exact watertight edge evaluation using precomputed canonical
                    // (lo.x, lo.y, sx, sy) and top-left tie-breaker bit.
                    auto eval_precomp_edge = [](Float4 edge, Float2 q, Bool is_top_left, Float &out_e) noexcept {
                        out_e = edge.z * (q.y - edge.y) + edge.w * (q.x - edge.x);
                        return select(out_e > 0.f, out_e >= 0.f, is_top_left);
                    };

                    auto tid_f = make_float2(tid);
                    if (!ssaa) {
                        auto q = tid_f + make_float2(.5f, .5f);
                        auto sample_sum = def(make_float3(0.f));
                        auto beta = def(1.f);
                        auto idx = def(range_begin);
                        $while (idx < range_end & beta > 1e-3f) {
                            auto tri_idx = frame_buf.read<uint>(off_tile_tris + idx * 4u);
                            idx += 1u;
                            auto tri_off = off_tri_hot + tri_idx * 64u;
                            auto meta = frame_buf.read<uint4>(tri_off);
                            auto e0 = frame_buf.read<float4>(tri_off + 16u);
                            auto e1 = frame_buf.read<float4>(tri_off + 32u);
                            auto e2 = frame_buf.read<float4>(tri_off + 48u);
                            auto clip = frame_buf.read<float4>(off_clips + meta.x * 16u);
                            auto flags_tex = meta.y;
                            auto e0_v = def(0.f);
                            auto e1_v = def(0.f);
                            auto e2_v = def(0.f);
                            auto inside = all(q >= clip.xy()) & all(q < clip.zw()) &
                                          eval_precomp_edge(e0, q, (flags_tex & 1u) != 0u, e0_v) &
                                          eval_precomp_edge(e1, q, (flags_tex & 2u) != 0u, e1_v) &
                                          eval_precomp_edge(e2, q, (flags_tex & 4u) != 0u, e2_v);
                            $if (inside) {
                                auto tex_id = flags_tex >> 4u;
                                auto is_flat = (flags_tex & 8u) != 0u;
                                auto c = def(make_float4(0.f));
                                $if (is_flat & (tex_id == 0u)) {
                                    c = unpack_color(meta.z);
                                } $else {
                                    auto inv_sum = 1.f / max(e0_v + e1_v + e2_v, 1e-20f);
                                    auto w0 = e0_v * inv_sum;
                                    auto w1 = e1_v * inv_sum;
                                    auto w2 = e2_v * inv_sum;
                                    auto extra_off = off_tri_extra + meta.w * 32u;
                                    $if (is_flat) {
                                        c = unpack_color(meta.z);
                                    } $else {
                                        auto cols = frame_buf.read<uint2>(extra_off + 24u);
                                        c = w0 * unpack_color(meta.z) +
                                            w1 * unpack_color(cols.x) +
                                            w2 * unpack_color(cols.y);
                                    };
                                    $if (tex_id != 0u) {
                                        auto uv01 = frame_buf.read<float4>(extra_off);
                                        auto uv2 = frame_buf.read<float2>(extra_off + 16u);
                                        auto uv = w0 * uv01.xy() + w1 * uv01.zw() + w2 * uv2;
                                        c *= texture_array->tex2d(tex_id).sample(uv);
                                    };
                                };
                                sample_sum += c.xyz() * beta * c.w;
                                beta *= 1.f - c.w;
                            };
                        };
                        $if (beta > 0.f) {
                            sample_sum += beta * fb.read(tid).xyz();
                        };
                        fb.write(tid, make_float4(sample_sum, 1.f));
                    } else {
                        // Single-pass 4x SSAA: traverse each tile's triangle list once
                        // so scalar triangle/clip/extra loads are shared across all 4 subpixels.
                        const std::array<float2, 4> kOffsets = {
                            make_float2(1.f / 3.f, 1.f / 3.f),
                            make_float2(2.f / 3.f, 1.f / 3.f),
                            make_float2(2.f / 3.f, 2.f / 3.f),
                            make_float2(1.f / 3.f, 2.f / 3.f),
                        };
                        std::array<Float2, 4> qs = {
                            tid_f + kOffsets[0], tid_f + kOffsets[1],
                            tid_f + kOffsets[2], tid_f + kOffsets[3]};
                        std::array<Float3, 4> ssums = {
                            def(make_float3(0.f)), def(make_float3(0.f)),
                            def(make_float3(0.f)), def(make_float3(0.f))};
                        std::array<Float, 4> betas = {def(1.f), def(1.f), def(1.f), def(1.f)};
                        auto idx = def(range_begin);
                        $while (idx < range_end &
                                (betas[0] > 1e-3f | betas[1] > 1e-3f |
                                 betas[2] > 1e-3f | betas[3] > 1e-3f)) {
                            auto tri_idx = frame_buf.read<uint>(off_tile_tris + idx * 4u);
                            idx += 1u;
                            auto tri_off = off_tri_hot + tri_idx * 64u;
                            auto meta = frame_buf.read<uint4>(tri_off);
                            auto flags_tex = meta.y;
                            auto tex_id = flags_tex >> 4u;
                            auto is_flat = (flags_tex & 8u) != 0u;
                            $if (is_flat & (tex_id == 0u)) {
                                auto e0 = frame_buf.read<float4>(tri_off + 16u);
                                auto e1 = frame_buf.read<float4>(tri_off + 32u);
                                auto e2 = frame_buf.read<float4>(tri_off + 48u);
                                auto clip = frame_buf.read<float4>(off_clips + meta.x * 16u);
                                auto tl0 = (flags_tex & 1u) != 0u;
                                auto tl1 = (flags_tex & 2u) != 0u;
                                auto tl2 = (flags_tex & 4u) != 0u;
                                auto c = unpack_color(meta.z);
                                auto c_premul = c.xyz() * c.w;
                                auto one_minus_a = 1.f - c.w;
                                for (uint s = 0u; s < 4u; s++) {
                                    auto ev0 = def(0.f);
                                    auto ev1 = def(0.f);
                                    auto ev2 = def(0.f);
                                    auto hit = (betas[s] > 1e-3f) &
                                               all(qs[s] >= clip.xy()) & all(qs[s] < clip.zw()) &
                                               eval_precomp_edge(e0, qs[s], tl0, ev0) &
                                               eval_precomp_edge(e1, qs[s], tl1, ev1) &
                                               eval_precomp_edge(e2, qs[s], tl2, ev2);
                                    auto hit_scale = select(0.f, betas[s], hit);
                                    ssums[s] += c_premul * hit_scale;
                                    betas[s] = select(betas[s], betas[s] * one_minus_a, hit);
                                }
                            } $else {
                                auto e0 = frame_buf.read<float4>(tri_off + 16u);
                                auto e1 = frame_buf.read<float4>(tri_off + 32u);
                                auto e2 = frame_buf.read<float4>(tri_off + 48u);
                                auto clip = frame_buf.read<float4>(off_clips + meta.x * 16u);
                                auto tl0 = (flags_tex & 1u) != 0u;
                                auto tl1 = (flags_tex & 2u) != 0u;
                                auto tl2 = (flags_tex & 4u) != 0u;
                                std::array<Float, 4> ev0s = {def(0.f), def(0.f), def(0.f), def(0.f)};
                                std::array<Float, 4> ev1s = {def(0.f), def(0.f), def(0.f), def(0.f)};
                                std::array<Float, 4> ev2s = {def(0.f), def(0.f), def(0.f), def(0.f)};
                                std::array<Bool, 4> hits = {def(false), def(false), def(false), def(false)};
                                for (uint s = 0u; s < 4u; s++) {
                                    hits[s] = (betas[s] > 1e-3f) &
                                              all(qs[s] >= clip.xy()) & all(qs[s] < clip.zw()) &
                                              eval_precomp_edge(e0, qs[s], tl0, ev0s[s]) &
                                              eval_precomp_edge(e1, qs[s], tl1, ev1s[s]) &
                                              eval_precomp_edge(e2, qs[s], tl2, ev2s[s]);
                                }
                                $if (hits[0] | hits[1] | hits[2] | hits[3]) {
                                    auto extra_off = off_tri_extra + meta.w * 32u;
                                    auto c0 = unpack_color(meta.z);
                                    auto c1 = def(make_float4(0.f));
                                    auto c2 = def(make_float4(0.f));
                                    $if (!is_flat) {
                                        auto cols = frame_buf.read<uint2>(extra_off + 24u);
                                        c1 = unpack_color(cols.x);
                                        c2 = unpack_color(cols.y);
                                    };
                                    auto uv01 = def(make_float4(0.f));
                                    auto uv2 = def(make_float2(0.f));
                                    $if (tex_id != 0u) {
                                        uv01 = frame_buf.read<float4>(extra_off);
                                        uv2 = frame_buf.read<float2>(extra_off + 16u);
                                    };
                                    for (uint s = 0u; s < 4u; s++) {
                                        $if (hits[s]) {
                                            auto inv_sum = 1.f / max(ev0s[s] + ev1s[s] + ev2s[s], 1e-20f);
                                            auto w0 = ev0s[s] * inv_sum;
                                            auto w1 = ev1s[s] * inv_sum;
                                            auto w2 = ev2s[s] * inv_sum;
                                            auto c = def(c0);
                                            $if (!is_flat) {
                                                c = w0 * c0 + w1 * c1 + w2 * c2;
                                            };
                                            $if (tex_id != 0u) {
                                                auto uv = w0 * uv01.xy() + w1 * uv01.zw() + w2 * uv2;
                                                c *= texture_array->tex2d(tex_id).sample(uv);
                                            };
                                            ssums[s] += c.xyz() * betas[s] * c.w;
                                            betas[s] *= 1.f - c.w;
                                        };
                                    }
                                };
                            };
                        };
                        auto beta_sum = betas[0] + betas[1] + betas[2] + betas[3];
                        auto total = ssums[0] + ssums[1] + ssums[2] + ssums[3];
                        $if (beta_sum > 0.f) {
                            total += beta_sum * fb.read(tid).xyz();
                        };
                        fb.write(tid, make_float4(.25f * total, 1.f));
                    }
                });
        } else if (_render_mode == RenderMode::RAY_TRACED) {
            _render_shader = _device.compile<2>([ssaa = config.ssaa](ImageFloat fb, UInt2 offset, AccelVar accel,
                                                                     BufferVar<Triangle> triangles, BufferVar<Vertex> vertices,
                                                                     BindlessVar texture_array, BufferFloat4 clip_rects) noexcept {
                auto tid = offset + dispatch_id().xy();
                $if (all(tid < fb.size())) {
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

        // Verify whether the uploaded font atlas's white-pixel UV samples exact
        // opaque white (255, 255, 255, 255) across its 2x2 bilinear footprint
        // (Sampler::linear_point_edge). When verified, solid-color triangles using
        // exclusively TexUvWhitePixel can elide the redundant font texture fetch
        // bit-identically in the compute rasterizer.
        _font_white_uv_opaque = false;
        auto white_uv = io.Fonts->TexUvWhitePixel;
        if (pixels != nullptr &&
            std::isfinite(white_uv.x) && std::isfinite(white_uv.y) &&
            white_uv.x >= 0.f && white_uv.x <= 1.f &&
            white_uv.y >= 0.f && white_uv.y <= 1.f) {
            auto fx = static_cast<double>(white_uv.x) * static_cast<double>(width) - 0.5;
            auto fy = static_cast<double>(white_uv.y) * static_cast<double>(height) - 0.5;
            auto x0 = std::clamp(static_cast<int>(std::floor(fx)), 0, width - 1);
            auto y0 = std::clamp(static_cast<int>(std::floor(fy)), 0, height - 1);
            auto x1 = std::clamp(x0 + 1, 0, width - 1);
            auto y1 = std::clamp(y0 + 1, 0, height - 1);
            auto is_opaque_white = [&](int x, int y) noexcept {
                auto p = pixels + (static_cast<size_t>(y) * static_cast<size_t>(width) + static_cast<size_t>(x)) * 4u;
                return p[0] == 255u && p[1] == 255u && p[2] == 255u && p[3] == 255u;
            };
            if (is_opaque_white(x0, y0) && is_opaque_white(x1, y0) &&
                is_opaque_white(x0, y1) && is_opaque_white(x1, y1)) {
                _font_white_uv = make_float2(white_uv.x, white_uv.y);
                _font_white_uv_opaque = true;
            }
        }
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

    /// Tiled compute 2D rasterizer: packs non-degenerate triangles with
    /// canonicalized winding into a per-tile CSR triangle list in exact
    /// submission order and dispatches `_compute_render_shader`.
    void _draw_compute(Image<float> &fb, ImDrawData *draw_data) noexcept {
        auto fb_size = fb.size();
        if (any(fb_size == 0u)) { return; }
        auto fb_size_f = make_float2(fb_size);
        auto clip_offset = make_float2(draw_data->DisplayPos.x, draw_data->DisplayPos.y);
        auto clip_scale = make_float2(draw_data->FramebufferScale.x, draw_data->FramebufferScale.y);
        auto clip_size = min(make_float2(draw_data->DisplaySize.x, draw_data->DisplaySize.y) * clip_scale, fb_size_f);
        if (any(clip_size <= 0.f)) { return; }
        auto transform = [clip_offset, clip_scale](ImVec2 p) noexcept {
            return (make_float2(p.x, p.y) - clip_offset) * clip_scale;
        };

        auto tiles_x = (fb_size.x + compute_tile_size - 1u) / compute_tile_size;
        auto tiles_y = (fb_size.y + compute_tile_size - 1u) / compute_tile_size;
        LUISA_ASSERT(tiles_x <= 0xffffu && tiles_y <= 0xffffu,
                     "Framebuffer tile dimensions ({}x{}) exceed 16-bit tile bounds.",
                     tiles_x, tiles_y);
        auto num_tiles = static_cast<size_t>(tiles_x) * static_cast<size_t>(tiles_y);

        size_t max_tris = 0u;
        bool has_callback = false;
        for (auto i = 0; i < draw_data->CmdLists.Size; i++) {
            auto cmd_list = draw_data->CmdLists[i];
            for (auto j = 0; j < cmd_list->CmdBuffer.Size; j++) {
                auto cmd = &cmd_list->CmdBuffer[j];
                if (cmd->UserCallback) {
                    if (cmd->UserCallback != ImDrawCallback_ResetRenderState) {
                        has_callback = true;
                    }
                } else if (cmd->ElemCount >= 3u) {
                    max_tris += cmd->ElemCount / 3u;
                }
            }
        }
        if (max_tris == 0u && !has_callback) { return; }

        _tri_hot.resize(max_tris);
        _tri_extra.resize(max_tris);
        _tile_tri_bounds.resize(max_tris);
        _clip_rects.clear();
        _tile_row_spans.clear();
        _large_tri_span_ends.clear();
        _clip_rects.reserve(64u);
        _tile_counts.assign(num_tiles, 0u);

        // Analytic O(height) per-row half-space interval clipper for large triangles
        // (bounding box >= 3x3 tiles). Precomputes linear edge equations in double
        // precision with a conservative margin matching the legacy SAT test so no
        // overlapping tile is ever dropped; emitted row spans are recorded once and
        // replayed directly in Pass 2 without recomputing edge math or reloading vertices.
        struct EdgeRowClipper {
            double c_y0;
            double c_step_y;
            double inv_tile_dy;
            int dy_sign;

            [[nodiscard]] static EdgeRowClipper make(float2 a, float2 b) noexcept {
                constexpr double tile_d = static_cast<double>(compute_tile_size);
                constexpr double eps = 1e-3;
                double ax = static_cast<double>(a.x);
                double ay = static_cast<double>(a.y);
                double dx = static_cast<double>(b.x) - ax;
                double dy = static_cast<double>(b.y) - ay;
                double qy0 = dx > 0.0 ? tile_d : 0.0;
                double c_y0 = dx * (qy0 - ay) + dy * ax + eps;
                double c_step_y = dx * tile_d;
                if (std::abs(dy) <= 1e-12) {
                    return EdgeRowClipper{c_y0, c_step_y, 0.0, 0};
                }
                return EdgeRowClipper{
                    c_y0,
                    c_step_y,
                    1.0 / (tile_d * dy),
                    dy > 0.0 ? 1 : -1};
            }

            // Intersect [rtx0, rtx1] with this edge's valid tile-x range for row ty.
            // Returns false if the clipped interval is empty.
            [[nodiscard]] bool clip_row(double c_row, int &rtx0, int &rtx1) const noexcept {
                if (dy_sign > 0) {
                    double t_max = c_row * inv_tile_dy + 1e-9;
                    if (t_max < static_cast<double>(rtx0)) { return false; }
                    if (t_max < static_cast<double>(rtx1)) {
                        rtx1 = static_cast<int>(t_max);
                    }
                } else if (dy_sign < 0) {
                    double t_min_p1 = c_row * inv_tile_dy - 1e-9;
                    if (t_min_p1 > static_cast<double>(rtx1 + 1)) { return false; }
                    if (t_min_p1 > static_cast<double>(rtx0)) {
                        int min_tx = static_cast<int>(t_min_p1);
                        if (min_tx > rtx0) { rtx0 = min_tx; }
                    }
                } else {
                    if (c_row < 0.0) { return false; }
                }
                return rtx0 <= rtx1;
            }
        };

        auto is_finite_f32 = [](float x) noexcept {
            return (std::bit_cast<uint32_t>(x) & 0x7f800000u) != 0x7f800000u;
        };
        auto make_edge = [](float2 a, float2 b, uint &tl_bit) noexcept {
            bool a_less = (a.y < b.y) || (a.y == b.y && a.x < b.x);
            float2 lo = a_less ? a : b;
            float2 hi = a_less ? b : a;
            float dx = hi.x - lo.x;
            float dy = hi.y - lo.y;
            float sx = a_less ? dx : -dx;
            float sy = a_less ? -dy : dy;
            tl_bit = ((b.y < a.y) || (b.y == a.y && b.x > a.x)) ? 1u : 0u;
            return make_float4(lo.x, lo.y, sx, sy);
        };

        ComputeTriHot *hot_out = _tri_hot.data();
        ComputeTriExtra *extra_out = _tri_extra.data();
        TileTriBounds *bnd_out = _tile_tri_bounds.data();
        uint *tile_counts = _tile_counts.data();
        size_t tri_count = 0u;
        size_t extra_count = 0u;

        for (auto i = 0; i < draw_data->CmdLists.Size; i++) {
            auto cmd_list = draw_data->CmdLists[i];
            const ImDrawIdx *idx_base = cmd_list->IdxBuffer.Data;
            const ImDrawVert *vtx_base = cmd_list->VtxBuffer.Data;
            for (auto j = 0; j < cmd_list->CmdBuffer.Size; j++) {
                auto cmd = &cmd_list->CmdBuffer[j];
                if (auto callback = cmd->UserCallback) {
                    if (callback != ImDrawCallback_ResetRenderState) {
                        callback(cmd_list, cmd);
                    }
                    continue;
                }
                auto clip_min = max((make_float2(cmd->ClipRect.x, cmd->ClipRect.y) - clip_offset) * clip_scale, 0.f);
                auto clip_max = min((make_float2(cmd->ClipRect.z, cmd->ClipRect.w) - clip_offset) * clip_scale, clip_size);
                if (any(clip_max <= clip_min) || cmd->ElemCount < 3u) { continue; }
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
                LUISA_ASSERT(tex_id < (1u << 28u), "Texture id {} exceeds 28-bit compute raster limit.", tex_id);
                const bool cmd_is_opaque_white_font = _font_white_uv_opaque && (tex_id == _font_texture_id);
                const float white_u = _font_white_uv.x;
                const float white_v = _font_white_uv.y;
                const ImDrawIdx *cmd_idx = idx_base + cmd->IdxOffset;
                const ImDrawVert *cmd_vtx = vtx_base + cmd->VtxOffset;
                const auto elem_count = cmd->ElemCount;
                for (auto t = 0u; t + 2u < elem_count; t += 3u) {
                    auto v0 = cmd_vtx[cmd_idx[t + 0u]];
                    auto v1 = cmd_vtx[cmd_idx[t + 1u]];
                    auto v2 = cmd_vtx[cmd_idx[t + 2u]];
                    auto p0 = transform(v0.pos);
                    auto p1 = transform(v1.pos);
                    auto p2 = transform(v2.pos);
                    if (!is_finite_f32(p0.x) || !is_finite_f32(p0.y) ||
                        !is_finite_f32(p1.x) || !is_finite_f32(p1.y) ||
                        !is_finite_f32(p2.x) || !is_finite_f32(p2.y)) {
                        continue;
                    }
                    auto area = (p1.x - p0.x) * (p2.y - p0.y) - (p1.y - p0.y) * (p2.x - p0.x);
                    auto area_abs_bits = std::bit_cast<uint32_t>(area) & 0x7fffffffu;
                    if (area_abs_bits == 0u || area_abs_bits >= 0x7f800000u) { continue; }
                    if (area < 0.f) {
                        std::swap(p1, p2);
                        std::swap(v1, v2);
                    }
                    auto bmin_x = std::max(std::min(p0.x, std::min(p1.x, p2.x)), clip_min.x);
                    auto bmin_y = std::max(std::min(p0.y, std::min(p1.y, p2.y)), clip_min.y);
                    auto bmax_x = std::min(std::max(p0.x, std::max(p1.x, p2.x)), clip_max.x);
                    auto bmax_y = std::min(std::max(p0.y, std::max(p1.y, p2.y)), clip_max.y);
                    if (!(bmin_x < bmax_x && bmin_y < bmax_y)) { continue; }
                    // bmin >= 0 and bmax > 0 are finite and clamped to fb_size_f, so
                    // truncation equals floor and avoids UCRT floorf/ceilf calls.
                    auto tx0 = static_cast<uint>(bmin_x) / compute_tile_size;
                    auto ty0 = static_cast<uint>(bmin_y) / compute_tile_size;
                    auto bx1_i = static_cast<uint>(bmax_x);
                    auto by1_i = static_cast<uint>(bmax_y);
                    auto px1 = static_cast<float>(bx1_i) < bmax_x ? bx1_i : (bx1_i > 0u ? bx1_i - 1u : 0u);
                    auto py1 = static_cast<float>(by1_i) < bmax_y ? by1_i : (by1_i > 0u ? by1_i - 1u : 0u);
                    auto tx1 = std::min(tiles_x - 1u, px1 / compute_tile_size);
                    auto ty1 = std::min(tiles_y - 1u, py1 / compute_tile_size);
                    if (tx0 > tx1 || ty0 > ty1) { continue; }

                    uint eff_tex_id = tex_id;
                    if (cmd_is_opaque_white_font &&
                        v0.uv.x == white_u && v0.uv.y == white_v &&
                        v1.uv.x == white_u && v1.uv.y == white_v &&
                        v2.uv.x == white_u && v2.uv.y == white_v) {
                        eff_tex_id = 0u;
                    }
                    const bool is_flat = (v0.col == v1.col) && (v0.col == v2.col);
                    uint tl0 = 0u, tl1 = 0u, tl2 = 0u;
                    auto e0 = make_edge(p1, p2, tl0);
                    auto e1 = make_edge(p2, p0, tl1);
                    auto e2 = make_edge(p0, p1, tl2);
                    uint extra_idx = 0u;
                    if (!is_flat || eff_tex_id != 0u) {
                        extra_idx = static_cast<uint>(extra_count++);
                        *extra_out++ = ComputeTriExtra{
                            .uv0 = make_float2(v0.uv.x, v0.uv.y),
                            .uv1 = make_float2(v1.uv.x, v1.uv.y),
                            .uv2 = make_float2(v2.uv.x, v2.uv.y),
                            .col1 = v1.col,
                            .col2 = v2.col};
                    }
                    *hot_out++ = ComputeTriHot{
                        .clip_idx = clip_idx,
                        .flags_tex = (eff_tex_id << 4u) | (is_flat ? 8u : 0u) | (tl2 << 2u) | (tl1 << 1u) | tl0,
                        .col0 = v0.col,
                        .extra_idx = extra_idx,
                        .e0 = e0,
                        .e1 = e1,
                        .e2 = e2};
                    *bnd_out++ = TileTriBounds{
                        static_cast<uint16_t>(tx0), static_cast<uint16_t>(ty0),
                        static_cast<uint16_t>(tx1), static_cast<uint16_t>(ty1)};
                    tri_count++;

                    if (tx0 == tx1 && ty0 == ty1) {
                        tile_counts[static_cast<size_t>(ty0) * tiles_x + tx0]++;
                    } else if ((tx1 - tx0) < 2u || (ty1 - ty0) < 2u) {
                        for (auto ty = ty0; ty <= ty1; ty++) {
                            auto *row_counts = tile_counts + static_cast<size_t>(ty) * tiles_x;
                            for (auto tx = tx0; tx <= tx1; tx++) {
                                row_counts[tx]++;
                            }
                        }
                    } else {
                        auto e0 = EdgeRowClipper::make(p0, p1);
                        auto e1 = EdgeRowClipper::make(p1, p2);
                        auto e2 = EdgeRowClipper::make(p2, p0);
                        auto c0 = e0.c_y0 + static_cast<double>(ty0) * e0.c_step_y;
                        auto c1 = e1.c_y0 + static_cast<double>(ty0) * e1.c_step_y;
                        auto c2 = e2.c_y0 + static_cast<double>(ty0) * e2.c_step_y;
                        for (auto ty = ty0; ty <= ty1; ty++) {
                            int rtx0 = static_cast<int>(tx0);
                            int rtx1 = static_cast<int>(tx1);
                            if (e0.clip_row(c0, rtx0, rtx1) &&
                                e1.clip_row(c1, rtx0, rtx1) &&
                                e2.clip_row(c2, rtx0, rtx1)) {
                                auto start = static_cast<uint32_t>(static_cast<size_t>(ty) * tiles_x + static_cast<uint>(rtx0));
                                auto count = static_cast<uint32_t>(rtx1 - rtx0 + 1);
                                _tile_row_spans.emplace_back(TileRowSpan{start, count});
                                auto *dst = tile_counts + start;
                                for (uint32_t k = 0u; k < count; k++) {
                                    dst[k]++;
                                }
                            }
                            c0 += e0.c_step_y;
                            c1 += e1.c_step_y;
                            c2 += e2.c_step_y;
                        }
                        _large_tri_span_ends.emplace_back(static_cast<uint>(_tile_row_spans.size()));
                    }
                }
            }
        }

        _tri_hot.resize(tri_count);
        _tri_extra.resize(extra_count);
        _tile_tri_bounds.resize(tri_count);
        if (tri_count == 0u) { return; }

        _active_tiles.resize(num_tiles + 1u);
        _tile_cursors.resize(num_tiles);
        uint total_refs = 0u;
        uint active_tile_count = 0u;
        uint2 *active_out = _active_tiles.data();
        uint *tile_cursors = _tile_cursors.data();
        for (uint ty = 0u; ty < tiles_y; ty++) {
            const uint y_bits = ty << 16u;
            const uint *row_counts = tile_counts + static_cast<size_t>(ty) * tiles_x;
            uint *row_cursors = tile_cursors + static_cast<size_t>(ty) * tiles_x;
            for (uint tx = 0u; tx < tiles_x; tx++) {
                auto cnt = row_counts[tx];
                if (cnt > 0u) {
                    active_out[active_tile_count++] = make_uint2(y_bits | tx, total_refs);
                    total_refs += cnt;
                    // Start each tile's Pass 2 cursor at range_end and decrement so
                    // tile_tri_indices stores triangles front-to-back (topmost first),
                    // enabling sequential forward shader traversal.
                    row_cursors[tx] = total_refs;
                }
            }
        }
        if (active_tile_count == 0u) { return; }
        active_out[active_tile_count] = make_uint2(0u, total_refs);
        const size_t active_elems = static_cast<size_t>(active_tile_count) + 1u;

        // Pack active tiles, CSR triangle indices, clip rects, hot triangle
        // records, and cold extra records into one 16-byte-aligned staging
        // buffer so the frame requires only a single host-to-device transfer.
        auto align16 = [](size_t n) noexcept { return (n + 15u) & ~size_t{15u}; };
        const size_t active_bytes = active_elems * sizeof(uint2);
        const size_t off_tile_tris = align16(active_bytes);
        const size_t refs_bytes = static_cast<size_t>(total_refs) * sizeof(uint);
        const size_t off_clips = align16(off_tile_tris + refs_bytes);
        const size_t clips_bytes = _clip_rects.size() * sizeof(float4);
        const size_t off_tri_hot = off_clips + clips_bytes;
        const size_t hot_bytes = tri_count * sizeof(ComputeTriHot);
        const size_t off_tri_extra = off_tri_hot + hot_bytes;
        const size_t extra_bytes = extra_count * sizeof(ComputeTriExtra);
        const size_t total_bytes = std::max<size_t>(align16(off_tri_extra + extra_bytes), 16u);
        LUISA_ASSERT(total_bytes <= 0xffffffffull,
                     "Compute GUI frame upload ({} bytes) exceeds 32-bit ByteBuffer offset range.",
                     total_bytes);

        _frame_upload_bytes.resize(total_bytes);
        std::byte *staging = _frame_upload_bytes.data();
        std::memcpy(staging, _active_tiles.data(), active_bytes);
        std::memcpy(staging + off_clips, _clip_rects.data(), clips_bytes);
        std::memcpy(staging + off_tri_hot, _tri_hot.data(), hot_bytes);
        if (extra_bytes > 0u) {
            std::memcpy(staging + off_tri_extra, _tri_extra.data(), extra_bytes);
        }

        // Write Pass 2 tile triangle indices directly into the coalesced staging buffer.
        uint *tile_tri_indices = reinterpret_cast<uint *>(staging + off_tile_tris);
        const TileTriBounds *bounds = _tile_tri_bounds.data();
        const TileRowSpan *spans = _tile_row_spans.data();
        const uint *large_ends = _large_tri_span_ends.data();
        size_t large_idx = 0u;
        size_t span_cursor = 0u;
        auto tri_count_u = static_cast<uint>(tri_count);
        for (auto tri_idx = 0u; tri_idx < tri_count_u; tri_idx++) {
            auto b = bounds[tri_idx];
            if (b.tx0 == b.tx1 && b.ty0 == b.ty1) {
                auto tile_id = static_cast<size_t>(b.ty0) * tiles_x + b.tx0;
                tile_tri_indices[--tile_cursors[tile_id]] = tri_idx;
            } else if ((b.tx1 - b.tx0) < 2u || (b.ty1 - b.ty0) < 2u) {
                for (uint ty = b.ty0; ty <= b.ty1; ty++) {
                    auto *row_cursors = tile_cursors + static_cast<size_t>(ty) * tiles_x;
                    for (uint tx = b.tx0; tx <= b.tx1; tx++) {
                        tile_tri_indices[--row_cursors[tx]] = tri_idx;
                    }
                }
            } else {
                size_t span_end = large_ends[large_idx++];
                for (; span_cursor < span_end; span_cursor++) {
                    auto s = spans[span_cursor];
                    auto *cur = tile_cursors + s.tile_start;
                    for (uint32_t k = 0u; k < s.count; k++) {
                        tile_tri_indices[--cur[k]] = tri_idx;
                    }
                }
            }
        }

        if (!_compute_frame_buffer || _compute_frame_buffer.size_bytes() < total_bytes) {
            if (_compute_frame_buffer) { _stream.synchronize(); }
            _compute_frame_buffer = _device.create_byte_buffer(
                std::max(next_pow2(total_bytes), static_cast<size_t>(256_k)));
        }

        constexpr uint kMaxActiveGridX = 65535u;
        uint active_grid_y = (active_tile_count + kMaxActiveGridX - 1u) / kMaxActiveGridX;
        uint active_grid_x = (active_tile_count + active_grid_y - 1u) / active_grid_y;
        auto active_dispatch = make_uint2(active_grid_x * compute_tile_size,
                                          active_grid_y * compute_tile_size);
        auto section_offsets = make_uint4(
            static_cast<uint>(off_tile_tris),
            static_cast<uint>(off_clips),
            static_cast<uint>(off_tri_hot),
            static_cast<uint>(off_tri_extra));

        if (_texture_array.dirty()) { _stream << _texture_array.update(); }
        _stream << _compute_frame_buffer.view(0u, total_bytes).copy_from(staging)
                << _compute_render_shader(fb, active_tile_count, active_grid_x,
                                          section_offsets, _compute_frame_buffer, _texture_array)
                       .dispatch(active_dispatch);
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
            switch (_render_mode) {
                case RenderMode::HARDWARE_RASTER:
                    if (_build_raster_mesh(draw_data)) { _draw_raster(fb); }
                    break;
                case RenderMode::COMPUTE_RASTER:
                    _draw_compute(fb, draw_data);
                    break;
                case RenderMode::RAY_TRACED:
                    _draw_raytraced(fb, draw_data);
                    break;
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
