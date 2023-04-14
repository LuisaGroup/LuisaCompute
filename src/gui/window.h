#pragma once

#include <core/dll_export.h>
#include <core/stl/string.h>
#include <core/basic_types.h>
#include <core/stl/functional.h>
#include <gui/input.h>

namespace luisa::compute {

class LC_GUI_API Window {

public:
    using MouseButtonCallback = luisa::function<void(MouseButton button, Action action, float2 xy)>;
    using CursorPositionCallback = luisa::function<void(float2 xy)>;
    using WindowSizeCallback = luisa::function<void(uint2 size)>;
    using KeyCallback = luisa::function<void(Key key, KeyModifiers modifiers, Action action)>;
    using ScrollCallback = luisa::function<void(float2 dxdy)>;
    struct IWindowImpl {
        virtual ~IWindowImpl() noexcept = default;
    };

private:
    string _name;
    unique_ptr<IWindowImpl> _impl;
    uint2 _size;

public:
    Window(string name, uint width, uint height) noexcept;
    Window(string name, uint2 size) noexcept
        : Window{std::move(name), size.x, size.y} {}
    ~Window() noexcept;
    Window(const Window &) = delete;
    Window(Window &&) = default;
    Window &operator=(Window &&) noexcept = default;
    Window &operator=(const Window &) noexcept = delete;

    [[nodiscard]] uint64_t native_handle() const noexcept;
    [[nodiscard]] bool should_close() const noexcept;
    void set_should_close(bool should_close) noexcept;
    [[nodiscard]] auto size() const noexcept { return _size; }
    [[nodiscard]] auto name() const noexcept { return string_view{_name}; }

    Window &set_mouse_callback(MouseButtonCallback cb) noexcept;
    Window &set_cursor_position_callback(CursorPositionCallback cb) noexcept;
    Window &set_window_size_callback(WindowSizeCallback cb) noexcept;
    Window &set_key_callback(KeyCallback cb) noexcept;
    Window &set_scroll_callback(ScrollCallback cb) noexcept;
    void poll_events() noexcept;
    [[nodiscard]] bool is_key_down(Key key) const noexcept;
    [[nodiscard]] bool is_mouse_button_down(MouseButton mb) const noexcept;
    [[nodiscard]] explicit operator bool() const noexcept { return !should_close(); }
};

}// namespace luisa::compute
