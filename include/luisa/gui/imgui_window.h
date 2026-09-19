#pragma once

#include <luisa/runtime/device.h>
#include <luisa/core/dll_export.h>
#include <luisa/core/basic_types.h>
#include <luisa/core/stl/memory.h>
#include <luisa/core/stl/string.h>
#include <luisa/gui/input.h>

struct GLFWwindow;
struct ImGuiContext;

namespace luisa::compute {

class Swapchain;

template<typename T>
class Image;

class Sampler;

class LUISA_GUI_API ImGuiWindow {

public:
    using MouseButtonCallback = luisa::move_only_function<void(MouseButton button, Action action, float2 xy)>;
    using CursorPositionCallback = luisa::move_only_function<void(float2 xy)>;
    using WindowSizeCallback = luisa::move_only_function<void(uint2 size)>;
    using KeyCallback = luisa::move_only_function<void(Key key, KeyModifiers modifiers, Action action)>;
    using ScrollCallback = luisa::move_only_function<void(float2 dxdy)>;
    struct Config {

        uint2 size{800, 600};
        bool resizable{true};
        bool vsync{false};
        bool hdr{false};
        bool ssaa{false};
        /// Render the ImGui draw data with the GPU's fixed-function
        /// rasterizer (triangle draws) instead of the default ray-tracing path
        /// (which rebuilds the acceleration structure and mesh every frame and
        /// depth-peels with per-pixel rays). Falls back to the ray-tracing path
        /// when the backend has no working JIT raster pipeline (e.g. the
        /// AOT-only Vulkan raster path).
        bool rasterization{true};
        bool docking{true};
        bool multi_viewport{true};
        uint back_buffers{2};
        /// Scale the GUI style/fonts (and the initial window size) to the
        /// monitor content scale reported by GLFW. Disable for pixel-exact
        /// rendering independent of the display DPI.
        bool dpi_aware{true};

        [[nodiscard]] static Config make_default() noexcept { return {}; }
    };

private:
    class ContextGuard {

    private:
        ImGuiWindow *_self;

    public:
        explicit ContextGuard(ImGuiWindow *self) noexcept
            : _self{self} { _self->push_context(); }
        ~ContextGuard() noexcept { _self->pop_context(); }
        ContextGuard(const ContextGuard &) noexcept = delete;
        ContextGuard(ContextGuard &&) noexcept = delete;
        ContextGuard &operator=(const ContextGuard &) noexcept = delete;
        ContextGuard &operator=(ContextGuard &&) noexcept = delete;
    };

public:
    class Impl;

private:
    luisa::unique_ptr<Impl> _impl;

public:
    ImGuiWindow() noexcept;
    ImGuiWindow(Device &device, Stream &stream,
                luisa::string name,
                const Config &config = Config::make_default()) noexcept;
    ~ImGuiWindow() noexcept;
    ImGuiWindow(ImGuiWindow &&) noexcept;
    ImGuiWindow &operator=(ImGuiWindow &&) noexcept;
    ImGuiWindow(const ImGuiWindow &) noexcept = delete;
    ImGuiWindow &operator=(const ImGuiWindow &) noexcept = delete;

public:

    ImGuiWindow &set_mouse_callback(MouseButtonCallback cb) noexcept;
    ImGuiWindow &set_cursor_position_callback(CursorPositionCallback cb) noexcept;
    ImGuiWindow &set_window_size_callback(WindowSizeCallback cb) noexcept;
    ImGuiWindow &set_key_callback(KeyCallback cb) noexcept;
    ImGuiWindow &set_scroll_callback(ScrollCallback cb) noexcept;
    void create(Device &device, Stream &stream,
                luisa::string name,
                const Config &config = Config::make_default()) noexcept;
    void destroy() noexcept;

    [[nodiscard]] ImGuiContext *context() const noexcept;
    void push_context() noexcept;
    void pop_context() noexcept;

    [[nodiscard]] GLFWwindow *handle() const noexcept;
    [[nodiscard]] Swapchain &swapchain() const noexcept;
    [[nodiscard]] Image<float> &framebuffer() const noexcept;

    /// Content scale currently applied to the style/fonts
    /// (1.0 when DPI awareness is disabled).
    [[nodiscard]] float dpi_scale() const noexcept;

    [[nodiscard]] auto valid() const noexcept { return _impl != nullptr; }
    [[nodiscard]] explicit operator bool() const noexcept { return valid(); }

    [[nodiscard]] bool should_close() const noexcept;
    void set_should_close(bool b = true) noexcept;

    void prepare_frame() noexcept;// calls glfwPollEvents, ImGui::NewFrame, and other stuff; also makes the context current
    void render_frame() noexcept; // calls ImGui::Render, glfwSwapBuffers, and other stuff; also restores the current context as before prepare_frame

    [[nodiscard]] uint64_t register_texture(const Image<float> &image, const Sampler &sampler) noexcept;
    void unregister_texture(uint64_t tex_id) noexcept;

    template<typename F>
    decltype(auto) with_context(F &&f) noexcept {
        ContextGuard g{this};
        return luisa::invoke(std::forward<F>(f));
    }

    template<typename F>
    void with_frame(F &&f) noexcept {
        prepare_frame();
        luisa::invoke(std::forward<F>(f));
        render_frame();
    }
};

}// namespace luisa::compute
