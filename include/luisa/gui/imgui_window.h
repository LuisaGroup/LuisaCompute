#pragma once

#include <luisa/runtime/device.h>
#include <luisa/core/dll_export.h>
#include <luisa/core/basic_types.h>
#include <luisa/core/stl/memory.h>
#include <luisa/core/stl/string.h>

struct GLFWwindow;
struct ImGuiContext;

namespace luisa::compute {

class Swapchain;

template<typename T>
class Image;

class Sampler;

class LC_GUI_API ImGuiWindow {

public:
    struct Config {

        uint2 size{800, 600};
        bool resizable{true};
        bool vsync{false};
        bool hdr{false};
        bool ssaa{false};
        bool docking{true};
        bool multi_viewport{true};
        uint back_buffers{2};

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
    ImGuiWindow() noexcept = default;
    ImGuiWindow(Device &device, Stream &stream,
                luisa::string name,
                const Config &config = Config::make_default()) noexcept;
    ~ImGuiWindow() noexcept;
    ImGuiWindow(ImGuiWindow &&) noexcept;
    ImGuiWindow &operator=(ImGuiWindow &&) noexcept;
    ImGuiWindow(const ImGuiWindow &) noexcept = delete;
    ImGuiWindow &operator=(const ImGuiWindow &) noexcept = delete;

public:
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
