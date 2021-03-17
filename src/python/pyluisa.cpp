//
// Created by Mike Smith on 2021/3/16.
//

#include <pybind11/stl.h>
#include <pybind11/pybind11.h>

#include <ast/interface.h>
#include <ast/function_builder.h>
#include <compile/cpp_codegen.h>

namespace luisa::compute::python {

}

namespace py = pybind11;

PYBIND11_MODULE(pyluisa, m) {

    auto compute = m.def_submodule("compute");

    // ast
    [ast = compute.def_submodule("ast")] {
        using namespace luisa::compute;

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

        type.def(py::init([](std::string_view name) { return const_cast<Type *>(Type::from(name)); }))
            .def_static("at", &Type::at)
            .def_property_readonly_static("count", &Type::count)
            .def_property_readonly("hash", &Type::hash)
            .def_property_readonly("index", &Type::index)
            .def_property_readonly("size", &Type::size)
            .def_property_readonly("alignment", &Type::alignment)
            .def_property_readonly("tag", &Type::tag)
            .def_property_readonly("description", &Type::description)
            .def_property_readonly("dimension", &Type::dimension)
            .def_property_readonly("members", [](const Type &t) { return std::vector{t.members()}; })
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
            .def("__enter__", [](FunctionBuilder &fb) {
                FunctionBuilder::push(&fb);
                return &fb;
            })
            .def("__exit__", [](FunctionBuilder &fb, py::object &, py::object &, py::object &) { FunctionBuilder::pop(&fb); })
            .def_static("current", &FunctionBuilder::current)
            .def("uid", &FunctionBuilder::uid)
            .def("thread_id", [](FunctionBuilder &fb) -> const void * { return fb.thread_id(); })
            .def("block_id", [](FunctionBuilder &fb) -> const void * { return fb.block_id(); })
            .def("dispatch_id", [](FunctionBuilder &fb) -> const void * { return fb.dispatch_id(); });

        function.def_static("at", &Function::at);
    }();

    // compile
    [compile = compute.def_submodule("compile")] {
        using namespace luisa::compute;
        using namespace luisa::compute::compile;

        py::class_<Codegen::Scratch>(compile, "Scratch")
            .def(py::init())
            .def_property_readonly("view", &Codegen::Scratch::view)
            .def("__repr__", &Codegen::Scratch::view);

        py::class_<CppCodegen>(compile, "CppCodegen")
            .def(py::init<Codegen::Scratch &>())
            .def("emit", &CppCodegen::emit);
    }();
}
