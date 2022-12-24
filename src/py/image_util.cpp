#include <pybind11/pybind11.h>
#include <pybind11/functional.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <runtime/image.h>
#include <py/py_stream.h>
#include <stb/stb_image.h>
#include <stb/stb_image_write.h>
#include <core/logging.h>

namespace py = pybind11;
using namespace luisa::compute;
const auto pyref = py::return_value_policy::reference;// object lifetime is managed on C++ side

void export_img(py::module &m) {
    m.def("load_hdr_image", [](std::string &&path) {
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
    m.def("load_ldr_image", [](std::string &&path) {
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
    m.def("save_hdr_image", [](std::string &&path, py::buffer &&buf, int x, int y) {
        stbi_write_hdr(path.c_str(), x, y, 4, reinterpret_cast<float *>(buf.request().ptr));
    });
    m.def("save_ldr_image", [](std::string &&path, py::buffer &&buf, int x, int y) {
        auto exportFunc = [&](vstd::string_view strv) {
            if (strv == "png"sv) {
                stbi_write_png(path.c_str(), x, y, 4, buf.request().ptr, 0);
            } else if (strv == "jpg"sv) {
                stbi_write_jpg(path.c_str(), x, y, 4, buf.request().ptr, 100);
            } else if (strv == "bmp"sv) {
                stbi_write_bmp(path.c_str(), x, y, 4, buf.request().ptr);
            } else if (strv == "tga"sv) {
                stbi_write_tga(path.c_str(), x, y, 4, buf.request().ptr);
            } else {
                LUISA_ERROR("Illegal image file extension!");
            }
        };
        for (auto &i : vstd::ptr_range(path.data() + path.size() - 1, path.data() - 1, -1)) {
            if (i == '.') {
                vstd::string_view strv(&i + 1, path.data() + path.size());
                exportFunc(strv);
                return;
            }
        }
        LUISA_ERROR("Illegal image file extension!");
    });
}