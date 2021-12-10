//
// Created by Mike Smith on 2021/6/13.
//

#include <string_view>

#include <glad/glad.h>
#include <imgui/backends/imgui_impl_opengl3.h>

#include <core/allocator.h>
#include <core/logging.h>
#include <gui/imgui_impl_glfw.h>
#include <gui/window.h>

namespace luisa::compute {

namespace detail {

[[nodiscard]] auto gl_error_string(GLenum error) noexcept {
    using namespace std::string_view_literals;
    switch (error) {
        case GL_INVALID_ENUM: return "invalid enum"sv;
        case GL_INVALID_VALUE: return "invalid value"sv;
        case GL_INVALID_OPERATION: return "invalid operation"sv;
        case GL_OUT_OF_MEMORY: return "out of memory"sv;
        default: return "unknown error"sv;
    }
}

}// namespace detail

class GLFWContext {

public:
    GLFWContext() noexcept {
        glfwSetErrorCallback([](int error, const char *message) noexcept {
            LUISA_ERROR_WITH_LOCATION("GLFW error (code = {}): {}", error, message);
        });
        if (!glfwInit()) { LUISA_ERROR_WITH_LOCATION("Failed to initialize GLFW."); }
    }
    GLFWContext(GLFWContext &&) noexcept = delete;
    GLFWContext(const GLFWContext &) noexcept = delete;
    GLFWContext &operator=(GLFWContext &&) noexcept = delete;
    GLFWContext &operator=(const GLFWContext &) noexcept = delete;
    ~GLFWContext() noexcept { glfwTerminate(); }
    [[nodiscard]] static auto retain() noexcept {
        static std::weak_ptr<GLFWContext> instance;
        if (auto p = instance.lock()) { return p; }
        auto p = luisa::make_shared<GLFWContext>();
        instance = p;
        return p;
    }
};

#define CHECK_GL(...)                                              \
    [&] {                                                          \
        __VA_ARGS__;                                               \
        if (auto error = glGetError(); error != GL_NO_ERROR) {     \
            LUISA_ERROR_WITH_LOCATION(                             \
                "OpenGL error: {}.",                               \
                ::luisa::compute::detail::gl_error_string(error)); \
        }                                                          \
    }()

class GLTexture {

private:
    uint32_t _handle{0u};
    bool _is_float4{false};
    uint2 _size{};

public:
    explicit GLTexture() noexcept {
        CHECK_GL(glGenTextures(1, &_handle));
        CHECK_GL(glBindTexture(GL_TEXTURE_2D, _handle));
        CHECK_GL(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR));
        CHECK_GL(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR));
        CHECK_GL(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE));
        CHECK_GL(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE));
    }

    GLTexture(GLTexture &&) noexcept = delete;
    GLTexture(const GLTexture &) noexcept = delete;
    GLTexture &operator=(GLTexture &&) noexcept = delete;
    GLTexture &operator=(const GLTexture &) noexcept = delete;

    ~GLTexture() noexcept { CHECK_GL(glDeleteTextures(1, &_handle)); }

    [[nodiscard]] auto handle() const noexcept { return _handle; }
    [[nodiscard]] auto size() const noexcept { return _size; }

    void load(const std::array<uint8_t, 4u> *pixels, uint2 size) noexcept {
        CHECK_GL(glBindTexture(GL_TEXTURE_2D, _handle));
        if (any(_size != size) || _is_float4) {
            _size = size;
            _is_float4 = false;
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, size.x, size.y, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
        }
        CHECK_GL(glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, _size.x, _size.y, GL_RGBA, GL_UNSIGNED_BYTE, pixels));
    }

    void load(const float4 *pixels, uint2 size) noexcept {
        CHECK_GL(glBindTexture(GL_TEXTURE_2D, _handle));
        if (any(_size != size) || !_is_float4) {
            _size = size;
            _is_float4 = true;
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, size.x, size.y, 0, GL_RGBA, GL_FLOAT, nullptr);
        }
        CHECK_GL(glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, _size.x, _size.y, GL_RGBA, GL_FLOAT, pixels));
    }
};

Window::Window(const char *name, uint2 initial_size, bool resizable) noexcept
    : _context{GLFWContext::retain()},
      _resizable{resizable} {

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 1);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
    glfwWindowHint(GLFW_RESIZABLE, resizable);

    // Create window with graphics context
    _handle = glfwCreateWindow(
        static_cast<int>(initial_size.x),
        static_cast<int>(initial_size.y),
        name, nullptr, nullptr);
    if (_handle == nullptr) {
        const char *error = nullptr;
        glfwGetError(&error);
        LUISA_ERROR_WITH_LOCATION("Failed to create GLFW window: {}.", error);
    }
    glfwMakeContextCurrent(_handle);
    glfwSwapInterval(0);// disable vsync

    if (!gladLoadGL()) { LUISA_ERROR_WITH_LOCATION("Failed to initialize GLAD."); }

    // Setup Dear ImGui context
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGui::StyleColorsDark();
    ImGui_ImplGlfw_InitForOpenGL(_handle);
    ImGui_ImplOpenGL3_Init("#version 330");

    glfwSetWindowUserPointer(_handle, this);
    glfwSetMouseButtonCallback(_handle, [](GLFWwindow *window, int button, int action, int /* mods */) noexcept {
        auto self = static_cast<Window *>(glfwGetWindowUserPointer(window));
        auto x = 0.0;
        auto y = 0.0;
        glfwGetCursorPos(self->handle(), &x, &y);
        if (auto &&cb = self->_mouse_button_callback) {
            cb(button, action, make_float2(static_cast<float>(x), static_cast<float>(y)));
        }
    });
    glfwSetCursorPosCallback(_handle, [](GLFWwindow *window, double x, double y) noexcept {
        auto self = static_cast<Window *>(glfwGetWindowUserPointer(window));
        if (auto &&cb = self->_cursor_position_callback) { cb(make_float2(static_cast<float>(x), static_cast<float>(y))); }
    });
    glfwSetWindowSizeCallback(_handle, [](GLFWwindow *window, int width, int height) noexcept {
        auto self = static_cast<Window *>(glfwGetWindowUserPointer(window));
        if (auto &&cb = self->_window_size_callback) { cb(make_uint2(width, height)); }
    });
    glfwSetKeyCallback(_handle, [](GLFWwindow *window, int key, int /* scancode */, int action, int /* mods */) noexcept {
        auto self = static_cast<Window *>(glfwGetWindowUserPointer(window));
        if (auto &&cb = self->_key_callback) { cb(key, action); }
    });
    glfwSetScrollCallback(_handle, [](GLFWwindow *window, double dx, double dy) noexcept {
        auto self = static_cast<Window *>(glfwGetWindowUserPointer(window));
        if (auto &&cb = self->_scroll_callback) { cb(make_float2(static_cast<float>(dx), static_cast<float>(dy))); }
    });
}

Window::~Window() noexcept {
    glfwMakeContextCurrent(_handle);
    _texture.reset();
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown(_handle);
    ImGui::DestroyContext();
    glfwDestroyWindow(_handle);
}

uint2 Window::size() const noexcept {
    auto width = 0;
    auto height = 0;
    glfwGetWindowSize(_handle, &width, &height);
    return make_uint2(
        static_cast<uint>(width),
        static_cast<uint>(height));
}

bool Window::should_close() const noexcept {
    return glfwWindowShouldClose(_handle);
}

Window &Window::set_mouse_callback(Window::MouseButtonCallback cb) noexcept {
    _mouse_button_callback = std::move(cb);
    return *this;
}

Window &Window::set_cursor_position_callback(Window::CursorPositionCallback cb) noexcept {
    _cursor_position_callback = std::move(cb);
    return *this;
}

Window &Window::set_window_size_callback(Window::WindowSizeCallback cb) noexcept {
    _window_size_callback = std::move(cb);
    return *this;
}

Window &Window::set_key_callback(Window::KeyCallback cb) noexcept {
    _key_callback = std::move(cb);
    return *this;
}

Window &Window::set_scroll_callback(Window::ScrollCallback cb) noexcept {
    _scroll_callback = std::move(cb);
    return *this;
}

void Window::set_background(const std::array<uint8_t, 4u> *pixels, uint2 size) noexcept {
    if (_texture == nullptr) { _texture = luisa::make_unique<GLTexture>(); }
    _texture->load(pixels, size);
}

void Window::set_background(const float4 *pixels, uint2 size) noexcept {
    if (_texture == nullptr) { _texture = luisa::make_unique<GLTexture>(); }
    _texture->load(pixels, size);
}

void Window::set_should_close() noexcept {
    glfwSetWindowShouldClose(_handle, true);
}

void Window::_begin_frame() noexcept {
    if (!should_close()) {
        glfwMakeContextCurrent(_handle);
        glfwPollEvents();
        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame(_handle);
        ImGui::NewFrame();
    }
}

void Window::_end_frame() noexcept {
    if (!should_close()) {
        // background
        if (_texture != nullptr) {
            ImVec2 background_size{
                static_cast<float>(_texture->size().x),
                static_cast<float>(_texture->size().y)};
            ImGui::GetBackgroundDrawList()->AddImage(
                _texture->handle(), {}, background_size);
        }
        // rendering
        ImGui::Render();
        int display_w, display_h;
        glfwGetFramebufferSize(_handle, &display_w, &display_h);
        glViewport(0, 0, display_w, display_h);
        glClearColor(0.3f, 0.5f, 0.7f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
        glfwSwapBuffers(_handle);
    }
}

void Window::set_size(uint2 size) noexcept {
    if (_resizable) {
        glfwSetWindowSize(_handle, static_cast<int>(size.x), static_cast<int>(size.y));
    } else {
        LUISA_WARNING_WITH_LOCATION("Ignoring resize on non-resizable window.");
    }
}

}// namespace luisa::compute
