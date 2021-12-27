//
// Created by Mike Smith on 2021/6/13.
//

#pragma once

#include <span>
#include <array>
#include <functional>

#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <imgui/imgui.h>

#include <core/basic_types.h>

namespace luisa::compute {

class GLTexture;
class GLFWContext;

class Window {

public:
    using MouseButtonCallback = luisa::function<void(int /* button */, int /* action */, float2 /* (x, y) */)>;
    using CursorPositionCallback = luisa::function<void(float2 /* (x, y) */)>;
    using WindowSizeCallback = luisa::function<void(uint2 /* (width, height) */)>;
    using KeyCallback = luisa::function<void(int /* key */, int /* action */)>;
    using ScrollCallback = luisa::function<void(float2 /* (dx, dy) */)>;

private:
    luisa::shared_ptr<GLFWContext> _context;
    GLFWwindow *_handle{nullptr};
    mutable luisa::unique_ptr<GLTexture> _texture;
    MouseButtonCallback _mouse_button_callback;
    CursorPositionCallback _cursor_position_callback;
    WindowSizeCallback _window_size_callback;
    KeyCallback _key_callback;
    ScrollCallback _scroll_callback;
    bool _resizable;

private:
    void _begin_frame() noexcept;
    void _end_frame() noexcept;

public:
    Window(const char *name, uint2 initial_size, bool resizable = false) noexcept;
    Window(Window &&) noexcept = delete;
    Window(const Window &) noexcept = delete;
    Window &operator=(Window &&) noexcept = delete;
    Window &operator=(const Window &) noexcept = delete;
    ~Window() noexcept;

    [[nodiscard]] uint2 size() const noexcept;
    [[nodiscard]] bool should_close() const noexcept;
    [[nodiscard]] auto handle() const noexcept { return _handle; }
    [[nodiscard]] explicit operator bool() const noexcept { return !should_close(); }

    Window &set_mouse_callback(MouseButtonCallback cb) noexcept;
    Window &set_cursor_position_callback(CursorPositionCallback cb) noexcept;
    Window &set_window_size_callback(WindowSizeCallback cb) noexcept;
    Window &set_key_callback(KeyCallback cb) noexcept;
    Window &set_scroll_callback(ScrollCallback cb) noexcept;

    void set_background(const std::array<uint8_t, 4u> *pixels, uint2 size) noexcept;
    void set_background(const float4 *pixels, uint2 size) noexcept;

    void set_should_close() noexcept;
    void set_size(uint2 size) noexcept;

    template<typename F>
    void run(F &&draw) noexcept {
        while (!should_close()) {
            run_one_frame(draw);
        }
    }

    template<typename F>
    void run_one_frame(F &&draw) noexcept {
        _begin_frame();
        std::invoke(std::forward<F>(draw));
        _end_frame();
    }
};

}