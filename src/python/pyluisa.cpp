//
// Created by Mike Smith on 2021/3/16.
//

#include <pybind11/stl.h>
#include <pybind11/pybind11.h>

#include <ast/interface.h>
#include <ast/function_builder.h>

namespace luisa::compute::python {

}

namespace py = pybind11;

PYBIND11_MODULE(pyluisa, m) {

    auto compute = m.def_submodule("compute");

    // ast
    [&compute] {
        using namespace luisa::compute;
        auto ast = compute.def_submodule("ast");

        auto type = py::class_<Type, std::unique_ptr<Type, py::nodelete>>(ast, "Type");

#define LUISA_MAKE_TYPE_TAG_VALUE(tag) \
    .value(#tag, Type::Tag::tag)
        // clang-format off
        py::enum_<Type::Tag>(type, "Tag")
            LUISA_MAKE_TYPE_TAG_VALUE(BOOL)
            LUISA_MAKE_TYPE_TAG_VALUE(FLOAT)
            LUISA_MAKE_TYPE_TAG_VALUE(INT)
            LUISA_MAKE_TYPE_TAG_VALUE(UINT)
            LUISA_MAKE_TYPE_TAG_VALUE(VECTOR)
            LUISA_MAKE_TYPE_TAG_VALUE(MATRIX)
            LUISA_MAKE_TYPE_TAG_VALUE(ARRAY)
            LUISA_MAKE_TYPE_TAG_VALUE(ATOMIC)
            LUISA_MAKE_TYPE_TAG_VALUE(STRUCTURE);
        // clang-format on
#undef LUISA_MAKE_TYPE_TAG_VALUE

        type.def(py::init([](std::string_view name) {
                return const_cast<Type *>(Type::from(name));
            }))
            .def_static("at", &Type::at)
            .def_property_readonly_static("count", &Type::count)
            .def_property_readonly("hash", &Type::hash)
            .def_property_readonly("index", &Type::index)
            .def_property_readonly("size", &Type::size)
            .def_property_readonly("alignment", &Type::alignment)
            .def_property_readonly("tag", &Type::tag)
            .def_property_readonly("description", &Type::description)
            .def_property_readonly("dimension", &Type::dimension)
            .def_property_readonly("members", [](const Type &t) {
                auto m = t.members();
                return std::vector<const Type *>{m.begin(), m.end()};
            })
            .def_property_readonly("element", &Type::element)
            .def_property_readonly("is_scalar", &Type::is_scalar)
            .def_property_readonly("is_array", &Type::is_array)
            .def_property_readonly("is_vector", &Type::is_vector)
            .def_property_readonly("is_matrix", &Type::is_matrix)
            .def_property_readonly("is_structure", &Type::is_structure)
            .def_property_readonly("is_atomic", &Type::is_atomic)
            .def("__repr__", &Type::description);

        auto function = py::class_<Function>(ast, "Function");
        py::enum_<Function::Tag>(function, "Tag")
            .value("KERNEL", Function::Tag::KERNEL)
            .value("CALLABLE", Function::Tag::CALLABLE);

        // builder methods
        auto builder = py::class_<FunctionBuilder, std::unique_ptr<FunctionBuilder, py::nodelete>>(ast, "Builder");
        builder.def(py::init([](Function::Tag tag) { return FunctionBuilder::create(tag); }))
            .def("__enter__", [](FunctionBuilder &fb) { FunctionBuilder::push(&fb); })
            .def("__exit__", [](FunctionBuilder &fb, py::object &, py::object &, py::object &) { FunctionBuilder::pop(&fb); });
    }();
}
