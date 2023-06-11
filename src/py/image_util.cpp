#include <pybind11/pybind11.h>
#include <pybind11/functional.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <luisa/runtime/image.h>
#include "py_stream.h"
#include <stb/stb_image.h>
#include <stb/stb_image_write.h>
#include <luisa/core/logging.h>
#include <luisa/vstl/string_utility.h>
namespace py = pybind11;
using namespace luisa::compute;
constexpr auto pyref = py::return_value_policy::reference;// object lifetime is managed on C++ side

void export_img(py::module &m) {
    m.def("load_hdr_image", [](const std::string &path) {
        int32_t x, y, channel;
        auto ptr = stbi_loadf(path.c_str(), &x, &y, &channel, 4);
        channel = 4;
        py::capsule free_when_done(ptr, vengine_free);
        return py::array_t<float>(
            {x, y, channel},
            {y * channel * sizeof(float), channel * sizeof(float), sizeof(float)},
            ptr,
            std::move(free_when_done));
    });
    m.def("load_ldr_image", [](const std::string &path) {
        int32_t x, y, channel;
        auto ptr = stbi_load(path.c_str(), &x, &y, &channel, 4);
        channel = 4;
        py::capsule free_when_done(ptr, vengine_free);
        return py::array_t<uint8_t>(
            {x, y, channel},
            {y * channel * sizeof(uint8_t), channel * sizeof(uint8_t), sizeof(uint8_t)},
            ptr,
            std::move(free_when_done));
    });
    m.def("save_hdr_image", [](const std::string &path, const py::buffer &buf, int x, int y) {
        stbi_write_hdr(path.c_str(), x, y, 4, reinterpret_cast<float *>(buf.request().ptr));
    });
    m.def("save_ldr_image", [](const std::string &path, const py::buffer &buf, int x, int y) {
        auto ends_with = [&](vstd::string_view ext) noexcept {
            if (path.size() < ext.size()) { return false; }
            for (auto i : vstd::range(ext.size())) {
                if (::tolower(path[path.size() - ext.size() + i]) != ext[i]) { return false; }
            }
            return true;
        };
        if (ends_with(".png")) {
            stbi_write_png(path.c_str(), x, y, 4, buf.request().ptr, 0);
        } else if (ends_with(".jpg")) {
            stbi_write_jpg(path.c_str(), x, y, 4, buf.request().ptr, 100);
        } else if (ends_with(".bmp")) {
            stbi_write_bmp(path.c_str(), x, y, 4, buf.request().ptr);
        } else if (ends_with(".tga")) {
            stbi_write_tga(path.c_str(), x, y, 4, buf.request().ptr);
        } else {
            LUISA_ERROR("Illegal image file '{}'.", path);
        }
    });
}

