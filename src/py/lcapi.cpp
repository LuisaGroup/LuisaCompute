// This file exports LuisaCompute functionalities to a python library using pybind11.
// 
// Class:
//   FunctionBuilder
//       define_kernel

#include <pybind11/pybind11.h>
#include <pybind11/functional.h>
#include <luisa-compute.h>
#include <nlohmann/json.hpp>

namespace py = pybind11;
using namespace luisa::compute;
using luisa::compute::detail::FunctionBuilder;

int add(int i, int j) {
    return i + j;
}

PYBIND11_DECLARE_HOLDER_TYPE(T, eastl::shared_ptr<T>);

PYBIND11_MODULE(lcapi, m) {
    m.doc() = "pybind11 example plugin"; // optional module docstring
    m.def("add", &add, "A function that adds two numbers");

    m.def("log_level_verbose", luisa::log_level_verbose);
    m.def("log_level_info", luisa::log_level_info);
    m.def("log_level_warning", luisa::log_level_warning);
    m.def("log_level_error", luisa::log_level_error);


    py::class_<std::filesystem::path>(m, "FsPath")
        .def(py::init<std::string>());
    py::class_<Context>(m, "Context")
        .def(py::init<const std::filesystem::path &>())
        .def("create_device", [](Context& self, std::string_view backend_name){ return self.create_device(backend_name); }); // TODO: support properties
    py::class_<Device>(m, "Device")
        .def("create_stream", &Device::create_stream)
        .def("impl", &Device::impl, py::return_value_policy::reference);
    py::class_<Device::Interface, eastl::shared_ptr<Device::Interface>>(m, "DeviceInterface")
        .def("create_shader", [](Device::Interface& self, Function kernel){return self.create_shader(kernel, {});}); // TODO: support metaoptions
    py::class_<Stream>(m, "Stream")
        .def("synchronize", &Stream::synchronize)
        .def("add", [](Stream& self, ShaderDispatchCommand* cmd){self<<cmd;});



    py::class_<Function>(m, "Function");
    py::class_<FunctionBuilder, eastl::shared_ptr<FunctionBuilder>>(m, "FunctionBuilder")
        .def("define_kernel", &FunctionBuilder::define_kernel<const std::function<void()> &>)
        .def("set_block_size", [](FunctionBuilder& self, uint sx, uint sy, uint sz){self.set_block_size(uint3{sx,sy,sz});})
        .def("function", &FunctionBuilder::function);
    m.def("builder", &FunctionBuilder::current, py::return_value_policy::reference);

    py::class_<ShaderDispatchCommand>(m, "ShaderDispatchCommand")
        .def(py::init<uint64_t, Function>())
        .def("set_dispatch_size", [](ShaderDispatchCommand& self, uint sx, uint sy, uint sz){self.set_dispatch_size(uint3{sx,sy,sz});});
}
