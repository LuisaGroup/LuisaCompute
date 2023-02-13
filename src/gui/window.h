#pragma once
#include <core/dll_export.h>
#include <core/stl/string.h>
#include <core/basic_types.h>
#include <core/stl/functional.h>
namespace luisa::compute {
class LC_GUI_API Window {
public:
    using MouseButtonCallback = luisa::function<void(int button, int action, float2 xy)>;
    using CursorPositionCallback = luisa::function<void(float2 xy)>;
    using WindowSizeCallback = luisa::function<void(uint2 size)>;
    using KeyCallback = luisa::function<void(int key, int action)>;
    using ScrollCallback = luisa::function<void(float2 dxdy)>;
    struct IWindowImpl {
        virtual ~IWindowImpl() = default;
    };

private:
    string _name;
    unique_ptr<IWindowImpl> _impl;
    uint2 _size;
    bool _vsync;

public:
    Window(
        string name,
        uint width, uint height,
        bool vsync);
    ~Window();
    Window(const Window &) = delete;
    Window(Window &&) = default;
    Window &operator=(Window &&) noexcept;
    Window &operator=(const Window &) noexcept = delete;

    [[nodiscard]] uint64_t window_native_handle() const noexcept;
    [[nodiscard]] bool should_close() const noexcept;
    [[nodiscard]] auto size() const noexcept { return _size; }
    [[nodiscard]] auto name() const noexcept { return string_view{_name}; }

    Window &set_mouse_callback(MouseButtonCallback cb) noexcept;
    Window &set_cursor_position_callback(CursorPositionCallback cb) noexcept;
    Window &set_window_size_callback(WindowSizeCallback cb) noexcept;
    Window &set_key_callback(KeyCallback cb) noexcept;
    Window &set_scroll_callback(ScrollCallback cb) noexcept;
    void pool_event() noexcept;

    [[nodiscard]] explicit operator bool() const noexcept { return !should_close(); }
};
}// namespace luisa::compute