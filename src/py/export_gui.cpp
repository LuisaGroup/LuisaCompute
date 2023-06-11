#include <pybind11/pybind11.h>
#include <pybind11/functional.h>
#include <pybind11/stl.h>
#include <luisa/gui/window.h>
#include <luisa/runtime/swapchain.h>
#include <luisa/core/stl/optional.h>
#include <luisa/core/clock.h>
#include <luisa/core/logging.h>
#include <luisa/vstl/common.h>

#include "managed_device.h"
#include "py_stream.h"

namespace py = pybind11;
using namespace luisa;
using namespace luisa::compute;

struct PyWindow : public vstd::IOperatorNewBase {
    inline static vstd::unordered_set<vstd::optional<Swapchain> *> swapchains;
    vstd::optional<Window> window;
    vstd::optional<Swapchain> chain;
    enum class KeyState : uint8_t {
        None,
        Down,
        Keep,
        Up
    };
    unordered_map<int, KeyState> mouse_states;
    unordered_map<int, KeyState> key_states;
    float2 size;
    float2 cursur_pos;
    PyWindow() {
    }
    PyWindow(PyWindow const &rhs) = delete;
    PyWindow(PyWindow &&rhs)
        : window{std::move(rhs.window)},
          chain(std::move(rhs.chain)),
          mouse_states(std::move(rhs.mouse_states)),
          key_states(std::move(rhs.key_states)),
          size(std::move(rhs.size)),
          cursur_pos(std::move(rhs.cursur_pos)) {
        swapchains.erase(&rhs.chain);
        swapchains.insert(&chain);
    }
    ~PyWindow() {
        if (chain) {
            swapchains.erase(&chain);
        }
    }
};
void export_gui(py::module &m) {
    static constexpr int kPress = 1;
    static constexpr int kRelease = 0;
    auto get_state = [](auto &&states, int key) {
        auto iter = states.find(key);
        if (iter == states.end()) return (int)PyWindow::KeyState::None;
        return (int)iter->second;
    };
    m.def("delete_all_swapchain", []() {
        for (auto &&i : PyWindow::swapchains) {
            i->destroy();
        }
    });
    py::class_<PyWindow>(m, "PyWindow")
        .def(py::init<>())
        .def("key_event", [get_state](PyWindow &w, int key) {
            return get_state(w.key_states, key);
        })
        .def("mouse_event", [get_state](PyWindow &w, int key) {
            return get_state(w.mouse_states, key);
        })
        .def("cursor_pos", [](PyWindow &w) {
            return w.cursur_pos;
        })
        .def("reset", [](PyWindow &w, ManagedDevice &device, PyStream &stream, string_view name, uint width, uint height, bool vsync) {
            if (w.window) {
                auto sz = w.window->size();
                if (sz.x == width && sz.y == height) return;
                w.window.destroy();
            }
            w.size = float2(width, height);
            w.window.create(string{name}, width, height);
            auto set_action = [](auto &&map, int key, auto &&action) {
                auto iter = map.try_emplace(key, PyWindow::KeyState::None).first;
                if (action == kPress) {
                    switch (iter->second) {
                        case PyWindow::KeyState::None:
                        case PyWindow::KeyState::Up:
                            iter->second = PyWindow::KeyState::Down;
                            break;
                        case PyWindow::KeyState::Down:
                            iter->second = PyWindow::KeyState::Keep;
                            break;
                    }
                } else if (action == kRelease) {
                    switch (iter->second) {
                        case PyWindow::KeyState::None:
                        case PyWindow::KeyState::Down:
                        case PyWindow::KeyState::Keep:
                            iter->second = PyWindow::KeyState::Up;
                            break;
                    }
                }
            };
            w.window->set_mouse_callback([&w, set_action](int button, int action, float2 xy) {
                set_action(w.mouse_states, button, action);
            });
            w.window->set_key_callback([&w, set_action](int key, int modifiers, int action) {
                set_action(w.key_states, key, action);// fixme: add support for modifiers
            });
            w.window->set_cursor_position_callback([&w](float2 cursor_pos) {
                w.cursur_pos = cursor_pos / w.size;
                w.cursur_pos.x = std::clamp(w.cursur_pos.x, 0.0f, 1.0f);
                w.cursur_pos.y = std::clamp(w.cursur_pos.y, 0.0f, 1.0f);
            });
            w.chain = device.device.create_swapchain(w.window->native_handle(), stream.stream(), {width, height}, false, vsync, 2);
            PyWindow::swapchains.emplace(&w.chain);
        })
        .def("should_close", [](PyWindow &w) {
            return w.window->should_close();
        })
        .def("present", [](PyWindow &w, PyStream &stream, uint64_t handle, uint width, uint height, uint level, PixelStorage storage) {
            LUISA_ASSERT(level == 0u, "Only level 0 is supported.");
            ImageView<float> view{handle, storage, level, make_uint2(width, height)};
            stream.execute();
            stream.stream() << w.chain->present(view);
            vector<int> remove_list;
            auto update_state = [&](auto &&states) {
                for (auto &&i : states) {
                    switch (i.second) {
                        case PyWindow::KeyState::Down:
                            i.second = PyWindow::KeyState::Keep;
                            break;
                        case PyWindow::KeyState::Up:
                            i.second = PyWindow::KeyState::None;
                            break;
                    }
                    if (i.second == PyWindow::KeyState::None) {
                        remove_list.emplace_back(i.first);
                    }
                }
            };
            update_state(w.key_states);
            update_state(w.mouse_states);
            for (auto &&i : remove_list) {
                w.key_states.erase(i);
            }
            w.window->poll_events();
        });
    py::class_<Clock>(m, "Clock")
        .def(py::init<>())
        .def("tic", [](Clock &clock) {
            clock.tic();
        })
        .def("toc", [](Clock &clock) {
            return clock.toc();
        });
}
