#ifdef LC_PY_ENABLE_GUI
#include <pybind11/pybind11.h>
#include <pybind11/functional.h>
#include <pybind11/stl.h>
#include <gui/window.h>
#include <runtime/swap_chain.h>
#include <core/stl/optional.h>
#include <vstl/common.h>
#include <py/managed_device.h>
#include <py/py_stream.h>
#include <api/view_export.h>
#include <core/clock.h>
namespace py = pybind11;
using namespace luisa;
using namespace luisa::compute;
struct PyWindow : public vstd::IOperatorNewBase {
    vstd::optional<Window> window;
    SwapChain chain;
};
void export_gui(py::module &m) {
    py::class_<PyWindow>(m, "PyWindow")
        .def(py::init<>())
        .def("reset", [](PyWindow &w, ManagedDevice &device, PyStream &stream, string_view name, uint width, uint height, bool vsync) {
            if (w.window) {
                auto sz = w.window->size();
                if (sz.x == width && sz.y == height) return;
                w.window.Delete();
            }
            w.window.New(string{name}, width, height, vsync);
            w.chain = device.device.create_swapchain(w.window->window_native_handle(), stream.stream(), {width, height}, true, vsync, 2);
        })
        .def("should_close", [](PyWindow &w) {
            return w.window->should_close();
        })
        .def("present", [](PyWindow &w, PyStream &stream, uint64_t handle, uint width, uint height, uint level, PixelStorage storage) {
            auto view = ViewExporter::create_image_view<float>(handle, storage, level, {width, height});
            stream.execute();
            stream.stream() << w.chain.present(view);
            w.window->pool_event();
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
#endif