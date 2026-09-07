#pragma once

#include <luisa/core/dll_export.h>
#include <luisa/core/stl/string.h>
#include <luisa/core/basic_types.h>
#include <luisa/core/stl/functional.h>
#include <luisa/gui/input.h>
struct GLFWwindow;
namespace luisa::compute {

class LUISA_GUI_API Window {

public:
    using MouseButtonCallback = luisa::move_only_function<void(MouseButton button, Action action, float2 xy)>;
    using CursorPositionCallback = luisa::move_only_function<void(float2 xy)>;
    using WindowSizeCallback = luisa::move_only_function<void(uint2 size)>;
    using KeyCallback = luisa::move_only_function<void(Key key, KeyModifiers modifiers, Action action)>;
    using ScrollCallback = luisa::move_only_function<void(float2 dxdy)>;
    struct NativeHandle {
        uint64_t window{};
        uint64_t display{};
    };
    using NativeHandleProvider = NativeHandle (*)(
        void *userdata, luisa::string_view name, uint2 size) noexcept;
    struct IWindowImpl {
        MouseButtonCallback mouse_button_callback;
        CursorPositionCallback cursor_position_callback;
        WindowSizeCallback window_size_callback;
        KeyCallback key_callback;
        ScrollCallback scroll_callback;

        virtual ~IWindowImpl() noexcept = default;
        [[nodiscard]] virtual GLFWwindow *window() const noexcept = 0;
        [[nodiscard]] virtual uint64_t native_handle() const noexcept = 0;
        [[nodiscard]] virtual uint64_t native_display() const noexcept = 0;
        [[nodiscard]] virtual bool should_close() const noexcept = 0;
        virtual void set_should_close(bool should_close) noexcept = 0;
        virtual void poll_events() noexcept = 0;
        [[nodiscard]] virtual bool is_key_down(Key key) const noexcept = 0;
        [[nodiscard]] virtual bool is_mouse_button_down(MouseButton button) const noexcept = 0;
    };

private:
    string _name;
    unique_ptr<IWindowImpl> _impl;
    uint2 _size;

public:
    /// Installs a process-wide provider used when an embedding platform owns
    /// window creation. The provider is consulted before GLFW and is required
    /// on iOS. It must outlive every Window created through it.
    static void set_native_handle_provider(
        NativeHandleProvider provider, void *userdata = nullptr) noexcept;
    static void clear_native_handle_provider() noexcept;
    /// Requests cooperative shutdown of all live provider-backed windows.
    static void request_close_all_native_windows() noexcept;
    /// Queues platform input for a provider-backed Window. Events are delivered
    /// on the rendering thread by poll_events(), matching GLFW callback timing.
    static void post_native_mouse_button_event(
        uint64_t native_handle, MouseButton button,
        Action action, float2 xy) noexcept;
    static void post_native_cursor_position_event(
        uint64_t native_handle, float2 xy) noexcept;
    static void post_native_key_event(
        uint64_t native_handle, Key key,
        KeyModifiers modifiers, Action action) noexcept;
    static void post_native_scroll_event(
        uint64_t native_handle, float2 dxdy) noexcept;
    static void post_native_window_size_event(
        uint64_t native_handle, uint2 size) noexcept;

    Window(string name, uint width, uint height, bool resizable = false, bool full_screen = false, bool window_transparent = false) noexcept;
    Window(string name, uint2 size, bool resizable = false, bool full_screen = false, bool window_transparent = false) noexcept
        : Window{std::move(name), size.x, size.y, resizable, full_screen, window_transparent} {}
    /// Wraps a platform-native window or view without taking ownership. This is
    /// useful on platforms such as iOS where the application lifecycle owns the
    /// UIKit view and Luisa only needs its native surface for swapchain creation.
    Window(string name, uint2 size, NativeHandle native_handle) noexcept;
    ~Window() noexcept;
    Window(const Window &) = delete;
    Window(Window &&) = default;
    Window &operator=(Window &&) noexcept = default;
    Window &operator=(const Window &) noexcept = delete;

    [[nodiscard]] GLFWwindow *window() const noexcept;
    [[nodiscard]] uint64_t native_handle() const noexcept;
    [[nodiscard]] uint64_t native_display() const noexcept;
    [[nodiscard]] bool should_close() const noexcept;
    void set_should_close(bool should_close = true) noexcept;
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
