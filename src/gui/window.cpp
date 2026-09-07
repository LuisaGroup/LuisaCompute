#include <luisa/core/platform.h>

#include <array>
#include <atomic>
#include <memory>
#include <mutex>
#include <vector>

#if !defined(LUISA_PLATFORM_IOS)
#if defined(LUISA_PLATFORM_WINDOWS)
#define GLFW_EXPOSE_NATIVE_WIN32
#elif defined(LUISA_PLATFORM_APPLE)
#define GLFW_EXPOSE_NATIVE_COCOA
#else
#if LUISA_ENABLE_WAYLAND
#define GLFW_EXPOSE_NATIVE_WAYLAND
#endif
#define GLFW_EXPOSE_NATIVE_X11
#endif

#ifndef GLFW_INCLUDE_NONE
#define GLFW_INCLUDE_NONE
#endif

#include <GLFW/glfw3.h>
#include <GLFW/glfw3native.h>
#endif

#include <luisa/core/logging.h>
#include <luisa/gui/window.h>

namespace luisa::compute {

namespace detail {

struct NativeInputEvent {
    enum class Type : uint8_t {
        mouse_button,
        cursor_position,
        key,
        scroll,
        window_size,
    };
    Type type{};
    MouseButton mouse_button{MOUSE_BUTTON_UNKNOWN};
    Key key{KEY_UNKNOWN};
    KeyModifiers modifiers{};
    Action action{ACTION_UNKNOWN};
    float2 value{};
    uint2 size{};
};

struct NativeInputState {
    static constexpr auto key_count = static_cast<size_t>(KEY_MENU) + 1u;
    static constexpr auto mouse_button_count =
        static_cast<size_t>(MOUSE_BUTTON_8) + 1u;

    uint64_t native_handle{};
    std::mutex mutex;
    std::vector<NativeInputEvent> events;
    std::array<bool, key_count> keys_down{};
    std::array<bool, mouse_button_count> mouse_buttons_down{};
};

struct NativeWindowHost {
    std::mutex mutex;
    Window::NativeHandleProvider provider{};
    void *userdata{};
    std::vector<std::weak_ptr<std::atomic_bool>> close_states;
    std::vector<std::weak_ptr<NativeInputState>> input_states;
};

[[nodiscard]] NativeWindowHost &native_window_host() noexcept {
    static NativeWindowHost host;
    return host;
}

[[nodiscard]] Window::NativeHandle acquire_native_handle(
    luisa::string_view name, uint2 size) noexcept {
    auto &host = native_window_host();
    Window::NativeHandleProvider provider{};
    void *userdata{};
    {
        std::scoped_lock lock{host.mutex};
        provider = host.provider;
        userdata = host.userdata;
    }
    return provider == nullptr ? Window::NativeHandle{} :
                                 provider(userdata, name, size);
}

void register_native_close_state(
    const std::shared_ptr<std::atomic_bool> &state) noexcept {
    auto &host = native_window_host();
    std::scoped_lock lock{host.mutex};
    host.close_states.emplace_back(state);
}

void register_native_input_state(
    const std::shared_ptr<NativeInputState> &state) noexcept {
    auto &host = native_window_host();
    std::scoped_lock lock{host.mutex};
    auto output = host.input_states.begin();
    for (auto iter = host.input_states.begin();
         iter != host.input_states.end(); ++iter) {
        if (!iter->expired()) { *output++ = *iter; }
    }
    host.input_states.erase(output, host.input_states.end());
    host.input_states.emplace_back(state);
}

[[nodiscard]] std::shared_ptr<NativeInputState> find_native_input_state(
    uint64_t native_handle) noexcept {
    auto &host = native_window_host();
    std::scoped_lock lock{host.mutex};
    auto output = host.input_states.begin();
    std::shared_ptr<NativeInputState> result;
    for (auto iter = host.input_states.begin();
         iter != host.input_states.end(); ++iter) {
        if (auto state = iter->lock()) {
            if (state->native_handle == native_handle) { result = state; }
            *output++ = *iter;
        }
    }
    host.input_states.erase(output, host.input_states.end());
    return result;
}

void post_native_input_event(
    uint64_t native_handle, NativeInputEvent event) noexcept {
    if (auto state = find_native_input_state(native_handle)) {
        std::scoped_lock lock{state->mutex};
        if (event.type == NativeInputEvent::Type::key) {
            auto index = static_cast<int>(event.key);
            if (index >= 0 &&
                static_cast<size_t>(index) < state->keys_down.size()) {
                state->keys_down[index] = event.action != ACTION_RELEASED;
            }
        } else if (event.type == NativeInputEvent::Type::mouse_button) {
            auto index = static_cast<int>(event.mouse_button);
            if (index >= 0 && static_cast<size_t>(index) <
                                  state->mouse_buttons_down.size()) {
                state->mouse_buttons_down[index] =
                    event.action != ACTION_RELEASED;
            }
        }
        state->events.emplace_back(event);
    }
}

#if !defined(LUISA_PLATFORM_IOS)
struct WindowImpl : public Window::IWindowImpl {
    GLFWwindow *handle;
    uint64_t window_handle{};

    WindowImpl(uint2 size, char const *name, bool resizable, bool full_screen, bool window_transparent) noexcept {

        static std::once_flag once_flag;
        std::call_once(once_flag, [] { glfwInit(); });
        glfwDefaultWindowHints();
        glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
        glfwWindowHint(GLFW_RESIZABLE, resizable);
        if (window_transparent) {
            glfwWindowHint(GLFW_TRANSPARENT_FRAMEBUFFER, GLFW_TRUE);
            if (!full_screen) {
                glfwWindowHint(GLFW_DECORATED, GLFW_FALSE);
            }
        }
        handle = glfwCreateWindow(size.x, size.y, name, full_screen ? glfwGetPrimaryMonitor() : nullptr, nullptr);
        LUISA_ASSERT(handle != nullptr, "Failed to create GLFW window '{}'.", name);
#if defined(LUISA_PLATFORM_WINDOWS)
        window_handle = reinterpret_cast<uint64_t>(glfwGetWin32Window(handle));
#elif defined(LUISA_PLATFORM_APPLE)
        window_handle = reinterpret_cast<uint64_t>(glfwGetCocoaWindow(handle));
#else
#if LUISA_ENABLE_WAYLAND
        if (glfwGetPlatform() == GLFW_PLATFORM_WAYLAND) {
            window_handle = reinterpret_cast<uint64_t>(glfwGetWaylandWindow(handle));
        } else {
            window_handle = reinterpret_cast<uint64_t>(glfwGetX11Window(handle));
        }
#else
        window_handle = reinterpret_cast<uint64_t>(glfwGetX11Window(handle));
#endif
#endif
        glfwSetWindowUserPointer(handle, this);
        // TODO: imgui
        glfwSetMouseButtonCallback(handle, [](GLFWwindow *window, int button, int action, int mods) noexcept {
            // if (ImGui::GetIO().WantCaptureMouse) {// ImGui is handling the mouse
            //     ImGui_ImplGlfw_MouseButtonCallback(window, button, action, mods);
            // } else {
            auto self = static_cast<WindowImpl *>(glfwGetWindowUserPointer(window));
            auto x = 0.0;
            auto y = 0.0;
            glfwGetCursorPos(self->handle, &x, &y);
            if (auto &&cb = self->mouse_button_callback) {
                cb(static_cast<MouseButton>(button), static_cast<Action>(action),
                   make_float2(static_cast<float>(x), static_cast<float>(y)));
            }
            // }
        });
        glfwSetCursorPosCallback(handle, [](GLFWwindow *window, double x, double y) noexcept {
            auto self = static_cast<WindowImpl *>(glfwGetWindowUserPointer(window));
            if (auto &&cb = self->cursor_position_callback) { cb(make_float2(static_cast<float>(x), static_cast<float>(y))); }
        });
        glfwSetWindowSizeCallback(handle, [](GLFWwindow *window, int width, int height) noexcept {
            auto self = static_cast<WindowImpl *>(glfwGetWindowUserPointer(window));
            if (auto &&cb = self->window_size_callback) { cb(make_uint2(width, height)); }
        });
        glfwSetKeyCallback(handle, [](GLFWwindow *window, int key, int scancode, int action, int mods) noexcept {
            // if (ImGui::GetIO().WantCaptureKeyboard) {// ImGui is handling the keyboard
            //     ImGui_ImplGlfw_KeyCallback(window, key, scancode, action, mods);
            // } else {
            auto self = static_cast<WindowImpl *>(glfwGetWindowUserPointer(window));
            if (auto &&cb = self->key_callback) {
                cb(static_cast<Key>(key), mods, static_cast<Action>(action));
            }
            // }
        });
        glfwSetScrollCallback(handle, [](GLFWwindow *window, double dx, double dy) noexcept {
            // if (ImGui::GetIO().WantCaptureMouse) {// ImGui is handling the mouse
            //     ImGui_ImplGlfw_ScrollCallback(window, dx, dy);
            // } else {
            auto self = static_cast<WindowImpl *>(glfwGetWindowUserPointer(window));
            if (auto &&cb = self->scroll_callback) {
                cb(make_float2(static_cast<float>(dx), static_cast<float>(dy)));
            }
            // }
        });
        // glfwSetCharCallback(window, ImGui_ImplGlfw_CharCallback);
    }
    ~WindowImpl() noexcept override {
        glfwDestroyWindow(handle);
        // glfwTerminate();
    }
    [[nodiscard]] GLFWwindow *window() const noexcept override { return handle; }
    [[nodiscard]] uint64_t native_handle() const noexcept override { return window_handle; }
    [[nodiscard]] uint64_t native_display() const noexcept override {
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
    [[nodiscard]] bool should_close() const noexcept override {
        return glfwWindowShouldClose(handle);
    }
    void set_should_close(bool should_close) noexcept override {
        glfwSetWindowShouldClose(handle, should_close);
    }
    void poll_events() noexcept override { glfwPollEvents(); }
    [[nodiscard]] bool is_key_down(Key key) const noexcept override {
        return glfwGetKey(handle, static_cast<int>(key)) != GLFW_RELEASE;
    }
    [[nodiscard]] bool is_mouse_button_down(MouseButton button) const noexcept override {
        return glfwGetMouseButton(handle, static_cast<int>(button)) != GLFW_RELEASE;
    }
};
#endif

struct NativeWindowImpl final : public Window::IWindowImpl {
    Window::NativeHandle native;
    std::shared_ptr<std::atomic_bool> close_requested;
    std::shared_ptr<NativeInputState> input_state;

    explicit NativeWindowImpl(Window::NativeHandle handle) noexcept
        : native{handle},
          close_requested{std::make_shared<std::atomic_bool>(false)},
          input_state{std::make_shared<NativeInputState>()} {
        LUISA_ASSERT(native.window != 0u,
                     "A native Window requires a non-null platform handle.");
        input_state->native_handle = native.window;
        register_native_close_state(close_requested);
        register_native_input_state(input_state);
    }
    [[nodiscard]] GLFWwindow *window() const noexcept override { return nullptr; }
    [[nodiscard]] uint64_t native_handle() const noexcept override { return native.window; }
    [[nodiscard]] uint64_t native_display() const noexcept override { return native.display; }
    [[nodiscard]] bool should_close() const noexcept override {
        return close_requested->load(std::memory_order_acquire);
    }
    void set_should_close(bool should_close) noexcept override {
        close_requested->store(should_close, std::memory_order_release);
    }
    void poll_events() noexcept override {
        std::vector<NativeInputEvent> events;
        {
            std::scoped_lock lock{input_state->mutex};
            events.swap(input_state->events);
        }
        for (auto &&event : events) {
            switch (event.type) {
                case NativeInputEvent::Type::mouse_button:
                    if (mouse_button_callback) {
                        mouse_button_callback(
                            event.mouse_button, event.action, event.value);
                    }
                    break;
                case NativeInputEvent::Type::cursor_position:
                    if (cursor_position_callback) {
                        cursor_position_callback(event.value);
                    }
                    break;
                case NativeInputEvent::Type::key:
                    if (key_callback) {
                        key_callback(event.key, event.modifiers, event.action);
                    }
                    break;
                case NativeInputEvent::Type::scroll:
                    if (scroll_callback) { scroll_callback(event.value); }
                    break;
                case NativeInputEvent::Type::window_size:
                    if (window_size_callback) {
                        window_size_callback(event.size);
                    }
                    break;
            }
        }
    }
    [[nodiscard]] bool is_key_down(Key key) const noexcept override {
        auto index = static_cast<int>(key);
        if (index < 0 || static_cast<size_t>(index) >=
                             input_state->keys_down.size()) {
            return false;
        }
        std::scoped_lock lock{input_state->mutex};
        return input_state->keys_down[index];
    }
    [[nodiscard]] bool is_mouse_button_down(
        MouseButton button) const noexcept override {
        auto index = static_cast<int>(button);
        if (index < 0 || static_cast<size_t>(index) >=
                             input_state->mouse_buttons_down.size()) {
            return false;
        }
        std::scoped_lock lock{input_state->mutex};
        return input_state->mouse_buttons_down[index];
    }
};

}// namespace detail

Window::Window(string name, uint width, uint height, bool resizable, bool full_screen, bool window_transparent) noexcept
    : _name{std::move(name)},
      _size{width, height} {
    if (auto native = detail::acquire_native_handle(_name, _size);
        native.window != 0u) {
        _impl = make_unique<detail::NativeWindowImpl>(native);
        return;
    }
#if defined(LUISA_PLATFORM_IOS)
    static_cast<void>(resizable);
    static_cast<void>(full_screen);
    static_cast<void>(window_transparent);
    LUISA_ERROR_WITH_LOCATION(
        "iOS Window creation is owned by UIKit. Construct Window with a "
        "native UIView or CAMetalLayer handle instead.");
#else
    _impl = make_unique<detail::WindowImpl>(_size, _name.c_str(), resizable, full_screen, window_transparent);
#endif
}

Window::Window(string name, uint2 size, NativeHandle native_handle) noexcept
    : _name{std::move(name)},
      _impl{make_unique<detail::NativeWindowImpl>(native_handle)},
      _size{size} {}

Window::~Window() noexcept = default;

void Window::set_native_handle_provider(
    NativeHandleProvider provider, void *userdata) noexcept {
    auto &host = detail::native_window_host();
    std::scoped_lock lock{host.mutex};
    host.provider = provider;
    host.userdata = userdata;
}

void Window::clear_native_handle_provider() noexcept {
    set_native_handle_provider(nullptr, nullptr);
}

void Window::request_close_all_native_windows() noexcept {
    auto &host = detail::native_window_host();
    std::scoped_lock lock{host.mutex};
    auto output = host.close_states.begin();
    for (auto iter = host.close_states.begin();
         iter != host.close_states.end(); ++iter) {
        if (auto state = iter->lock()) {
            state->store(true, std::memory_order_release);
            *output++ = *iter;
        }
    }
    host.close_states.erase(output, host.close_states.end());
}

void Window::post_native_mouse_button_event(
    uint64_t native_handle, MouseButton button,
    Action action, float2 xy) noexcept {
    detail::post_native_input_event(
        native_handle,
        {.type = detail::NativeInputEvent::Type::mouse_button,
         .mouse_button = button,
         .action = action,
         .value = xy});
}

void Window::post_native_cursor_position_event(
    uint64_t native_handle, float2 xy) noexcept {
    detail::post_native_input_event(
        native_handle,
        {.type = detail::NativeInputEvent::Type::cursor_position,
         .value = xy});
}

void Window::post_native_key_event(
    uint64_t native_handle, Key key,
    KeyModifiers modifiers, Action action) noexcept {
    detail::post_native_input_event(
        native_handle,
        {.type = detail::NativeInputEvent::Type::key,
         .key = key,
         .modifiers = modifiers,
         .action = action});
}

void Window::post_native_scroll_event(
    uint64_t native_handle, float2 dxdy) noexcept {
    detail::post_native_input_event(
        native_handle,
        {.type = detail::NativeInputEvent::Type::scroll,
         .value = dxdy});
}

void Window::post_native_window_size_event(
    uint64_t native_handle, uint2 size) noexcept {
    detail::post_native_input_event(
        native_handle,
        {.type = detail::NativeInputEvent::Type::window_size,
         .size = size});
}

GLFWwindow *Window::window() const noexcept {
    return _impl->window();
}

uint64_t Window::native_handle() const noexcept {
    return _impl->native_handle();
}

uint64_t Window::native_display() const noexcept {
    return _impl->native_display();
}

bool Window::should_close() const noexcept {
    return _impl->should_close();
}

Window &Window::set_mouse_callback(Window::MouseButtonCallback cb) noexcept {
    _impl->mouse_button_callback = std::move(cb);
    return *this;
}

Window &Window::set_cursor_position_callback(Window::CursorPositionCallback cb) noexcept {
    _impl->cursor_position_callback = std::move(cb);
    return *this;
}

Window &Window::set_window_size_callback(Window::WindowSizeCallback cb) noexcept {
    _impl->window_size_callback = std::move(cb);
    return *this;
}

Window &Window::set_key_callback(Window::KeyCallback cb) noexcept {
    _impl->key_callback = std::move(cb);
    return *this;
}

Window &Window::set_scroll_callback(Window::ScrollCallback cb) noexcept {
    _impl->scroll_callback = std::move(cb);
    return *this;
}

void Window::poll_events() noexcept {
    _impl->poll_events();
}

void Window::set_should_close(bool should_close) noexcept {
    _impl->set_should_close(should_close);
}

bool Window::is_key_down(Key key) const noexcept {
    return _impl->is_key_down(key);
}

bool Window::is_mouse_button_down(MouseButton mb) const noexcept {
    return _impl->is_mouse_button_down(mb);
}

}// namespace luisa::compute
