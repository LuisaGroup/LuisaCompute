// This file exports LuisaCompute functionalities to a python library using pybind11.
// 
// Class:
//   FunctionBuilder
//       define_kernel

#include <pybind11/pybind11.h>
#include <pybind11/functional.h>
#include <pybind11/stl.h>
#include <luisa-compute.h>
#include <nlohmann/json.hpp>

namespace py = pybind11;
using namespace luisa::compute;
using luisa::compute::detail::FunctionBuilder;

int add(int i, int j) {
    return i + j;
}

PYBIND11_DECLARE_HOLDER_TYPE(T, eastl::shared_ptr<T>);

const auto pyref = py::return_value_policy::reference; // object lifetime is managed on C++ side

namespace pybind11 { namespace detail {
    template <typename... T>
    struct type_caster<eastl::variant<T...>> : variant_caster<eastl::variant<T...>> {};
}}

// Note: declare pointer & base class;
// use reference policy when python shouldn't destroy returned object

PYBIND11_MODULE(lcapi, m) {
    m.doc() = "pybind11 example plugin"; // optional module docstring
    m.def("add", &add, "A function that adds two numbers");
    // log
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
        .def("impl", &Device::impl, pyref);
    py::class_<Device::Interface, eastl::shared_ptr<Device::Interface>>(m, "DeviceInterface")
        .def("create_shader", [](Device::Interface& self, Function kernel){return self.create_shader(kernel, {});}); // TODO: support metaoptions
    py::class_<Stream>(m, "Stream")
        .def("synchronize", &Stream::synchronize)
        .def("add", [](Stream& self, Command* cmd){self<<cmd;});


    // AST (FunctionBuilder)
    py::class_<Function>(m, "Function");
    py::class_<FunctionBuilder, eastl::shared_ptr<FunctionBuilder>>(m, "FunctionBuilder")
        .def("define_kernel", &FunctionBuilder::define_kernel<const std::function<void()> &>)
        .def("set_block_size", [](FunctionBuilder& self, uint sx, uint sy, uint sz){self.set_block_size(uint3{sx,sy,sz});})

        // .def("thread_id")
        // .def("block_id")
        // .def("dispatch_id")
        // .def("dispatch_size")

        .def("local", &FunctionBuilder::local, pyref)
        // .def("shared")

        .def("literal", &FunctionBuilder::literal, pyref)
        .def("unary", &FunctionBuilder::unary, pyref)
        .def("binary", &FunctionBuilder::binary, pyref)
        // .def("member")
        // .def("swizzle")
        // .def("access")
        // .def("cast")
        // .def("call")

        // .def("break_")
        // .def("continue_")
        // .def("return_")
        // .def("comment_")
        .def("assign", &FunctionBuilder::assign, pyref)

        // // create_expression 内存？
        // .def("if_")
        // .def("loop_") // ???
        // .def("switch_")
        // .def("case_")
        // .def("default_")
        // .def("for_")
        // .def("meta") // ???

        // .def("case_")
        // .def("push_scope")
        // .def("pop_scope")
        .def("function", &FunctionBuilder::function); // returning object
    m.def("builder", &FunctionBuilder::current, pyref);

    py::class_<Expression>(m, "Expression");
    py::class_<LiteralExpr, Expression>(m, "LiteralExpr");
    py::class_<RefExpr, Expression>(m, "RefExpr");
    py::class_<UnaryExpr, Expression>(m, "UnaryExpr");
    py::class_<BinaryExpr, Expression>(m, "BinaryExpr");


    py::enum_<UnaryOp>(m, "UnaryOp")
        .value("PLUS", UnaryOp::PLUS)
        .value("MINUS", UnaryOp::MINUS)
        .value("NOT", UnaryOp::NOT)
        .value("BIT_NOT", UnaryOp::BIT_NOT);

    py::enum_<BinaryOp>(m, "BinaryOp")
        // arithmetic
        .value("ADD", BinaryOp::ADD)
        .value("SUB", BinaryOp::SUB)
        .value("MUL", BinaryOp::MUL)
        .value("DIV", BinaryOp::DIV)
        .value("MOD", BinaryOp::MOD)
        .value("BIT_AND", BinaryOp::BIT_AND)
        .value("BIT_OR", BinaryOp::BIT_OR)
        .value("BIT_XOR", BinaryOp::BIT_XOR)
        .value("SHL", BinaryOp::SHL)
        .value("SHR", BinaryOp::SHR)
        .value("AND", BinaryOp::AND)
        .value("OR", BinaryOp::OR)
        // relational
        .value("LESS", BinaryOp::LESS)
        .value("GREATER", BinaryOp::GREATER)
        .value("LESS_EQUAL", BinaryOp::LESS_EQUAL)
        .value("GREATER_EQUAL", BinaryOp::GREATER_EQUAL)
        .value("EQUAL", BinaryOp::EQUAL)
        .value("NOT_EQUAL", BinaryOp::NOT_EQUAL);


    py::class_<Type>(m, "Type")
        .def_static("from_", &Type::from, pyref);
    // commands
    py::class_<Command>(m, "Command");
    py::class_<ShaderDispatchCommand, Command>(m, "ShaderDispatchCommand")
        .def_static("create", [](uint64_t handle, Function func){return ShaderDispatchCommand::create(handle, func);}, pyref)
        .def("set_dispatch_size", [](ShaderDispatchCommand& self, uint sx, uint sy, uint sz){self.set_dispatch_size(uint3{sx,sy,sz});});
}
