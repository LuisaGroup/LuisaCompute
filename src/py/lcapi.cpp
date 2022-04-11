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

#include "export_op.hpp"
#include "export_vector2.hpp"
#include "export_vector3.hpp"
#include "export_vector4.hpp"

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
        .def("create_shader", [](Device::Interface& self, Function kernel){return self.create_shader(kernel, {});}) // TODO: support metaoptions
        .def("create_buffer", &Device::Interface::create_buffer)
        .def("destroy_shader", &Device::Interface::destroy_shader)
        .def("destroy_buffer", &Device::Interface::destroy_buffer);
    py::class_<Stream>(m, "Stream")
        .def("synchronize", &Stream::synchronize)
        .def("add", [](Stream& self, Command* cmd){self<<cmd;});

    // AST (FunctionBuilder)
    py::class_<Function>(m, "Function");
    py::class_<FunctionBuilder, eastl::shared_ptr<FunctionBuilder>>(m, "FunctionBuilder")
        .def("define_kernel", &FunctionBuilder::define_kernel<const std::function<void()> &>)
        .def("set_block_size", [](FunctionBuilder& self, uint sx, uint sy, uint sz){self.set_block_size(uint3{sx,sy,sz});})

        .def("thread_id", &FunctionBuilder::thread_id, pyref)
        .def("block_id", &FunctionBuilder::block_id, pyref)
        .def("dispatch_id", &FunctionBuilder::dispatch_id, pyref)
        .def("dispatch_size", &FunctionBuilder::dispatch_size, pyref)

        .def("local", &FunctionBuilder::local, pyref)
        // .def("shared")
        // .def("constant")
        .def("buffer_binding", &FunctionBuilder::buffer_binding, pyref)

        .def("literal", &FunctionBuilder::literal, pyref)
        .def("unary", &FunctionBuilder::unary, pyref)
        .def("binary", &FunctionBuilder::binary, pyref)
        .def("member", &FunctionBuilder::member, pyref)
        .def("access", &FunctionBuilder::access, pyref)
        .def("swizzle", &FunctionBuilder::swizzle, pyref)
        // .def("cast")
        .def("call", [](FunctionBuilder& self, const Type *type, CallOp call_op, std::vector<const Expression *> args){return self.call(type, call_op, args);}, pyref)
        .def("call", [](FunctionBuilder& self, const Type *type, Function custom, std::vector<const Expression *> args){return self.call(type, custom, args);}, pyref)
        .def("call", [](FunctionBuilder& self, CallOp call_op, std::vector<const Expression *> args){self.call(call_op, args);})
        .def("call", [](FunctionBuilder& self, Function custom, std::vector<const Expression *> args){self.call(custom, args);})

        .def("break_", &FunctionBuilder::break_)
        .def("continue_", &FunctionBuilder::continue_)
        // .def("return_")
        // .def("comment_")
        .def("assign", &FunctionBuilder::assign, pyref)

        // // create_expression 内存？
        .def("if_", &FunctionBuilder::if_, pyref)
        .def("loop_", &FunctionBuilder::loop_, pyref)
        // .def("switch_")
        // .def("case_")
        // .def("default_")
        // .def("for_")
        // .def("meta") // ???

        // .def("case_")
        .def("push_scope", &FunctionBuilder::push_scope)
        .def("pop_scope", &FunctionBuilder::pop_scope)
        .def("function", &FunctionBuilder::function); // returning object
    m.def("builder", &FunctionBuilder::current, pyref);

    py::class_<Expression>(m, "Expression");
    py::class_<LiteralExpr, Expression>(m, "LiteralExpr");
    py::class_<RefExpr, Expression>(m, "RefExpr");
    py::class_<CallExpr, Expression>(m, "CallExpr");
    py::class_<UnaryExpr, Expression>(m, "UnaryExpr");
    py::class_<BinaryExpr, Expression>(m, "BinaryExpr");
    py::class_<MemberExpr, Expression>(m, "MemberExpr");
    py::class_<AccessExpr, Expression>(m, "AccessExpr");

    py::class_<ScopeStmt>(m, "ScopeStmt"); // not yet exporting base class (Statement)
    py::class_<IfStmt>(m, "IfStmt")
        .def("true_branch", py::overload_cast<>(&IfStmt::true_branch), pyref) // using overload_cast because there's also a const method variant
        .def("false_branch", py::overload_cast<>(&IfStmt::false_branch), pyref);
    py::class_<LoopStmt>(m, "LoopStmt")
        .def("body", py::overload_cast<>(&LoopStmt::body), pyref);

    export_op(m); // UnaryOp, BinaryOp, CallOp. def at export_op.hpp

    py::class_<Type>(m, "Type")
        .def_static("from_", &Type::from, pyref)
        .def("size", &Type::size)
        .def("is_array", &Type::is_array)
        .def("is_vector", &Type::is_vector)
        .def("is_matrix", &Type::is_matrix)
        .def("is_structure", &Type::is_structure)
        .def("is_buffer", &Type::is_buffer)
        .def("is_texture", &Type::is_texture)
        .def("is_bindless_array", &Type::is_bindless_array)
        .def("is_accel", &Type::is_accel)
        .def("element", &Type::element, pyref)
        .def("description", &Type::description)
        .def("dimension", &Type::dimension);
    // commands
    py::class_<Command>(m, "Command");
    py::class_<ShaderDispatchCommand, Command>(m, "ShaderDispatchCommand")
        .def_static("create", [](uint64_t handle, Function func){return ShaderDispatchCommand::create(handle, func);}, pyref)
        .def("set_dispatch_size", [](ShaderDispatchCommand& self, uint sx, uint sy, uint sz){self.set_dispatch_size(uint3{sx,sy,sz});})
        .def("encode_buffer", &ShaderDispatchCommand::encode_buffer)
        .def("encode_texture", &ShaderDispatchCommand::encode_texture)
        .def("encode_uniform", &ShaderDispatchCommand::encode_uniform)
        .def("encode_bindless_array", &ShaderDispatchCommand::encode_bindless_array)
        .def("encode_accel", &ShaderDispatchCommand::encode_accel);
        
    py::class_<BufferUploadCommand, Command>(m, "BufferUploadCommand")
        .def_static("create", [](uint64_t handle, size_t offset_bytes, size_t size_bytes, py::buffer buf){
            return BufferUploadCommand::create(handle, offset_bytes, size_bytes, buf.request().ptr);
        }, pyref);
    py::class_<BufferDownloadCommand, Command>(m, "BufferDownloadCommand")
        .def_static("create", [](uint64_t handle, size_t offset_bytes, size_t size_bytes, py::buffer buf){
            return BufferDownloadCommand::create(handle, offset_bytes, size_bytes, buf.request().ptr);
        }, pyref);

    export_vector2(m);
    export_vector3(m);
    export_vector4(m);
    // TODO export vector operators

    py::class_<float2x2>(m, "float2x2").def("identity", [](){return float2x2();});
    py::class_<float3x3>(m, "float3x3").def("identity", [](){return float3x3();});
    py::class_<float4x4>(m, "float4x4").def("identity", [](){return float4x4();});

    m.def("make_float2x2", [](float a){return make_float2x2(a);});
    m.def("make_float2x2", [](float a, float b, float c, float d){return make_float2x2(a,b,c,d);});
    m.def("make_float2x2", [](float2 a, float2 b){return make_float2x2(a,b);});
    m.def("make_float2x2", [](float2x2 a){return make_float2x2(a);});
    m.def("make_float2x2", [](float3x3 a){return make_float2x2(a);});
    m.def("make_float2x2", [](float4x4 a){return make_float2x2(a);});

    m.def("make_float3x3", [](float a){return make_float3x3(a);});
    m.def("make_float3x3", [](
        float m00, float m01, float m02,
        float m10, float m11, float m12,
        float m20, float m21, float m22)
        {return make_float3x3(m00,m01,m02, m10,m11,m12, m20,m21,m22);});
    m.def("make_float3x3", [](float3 a, float3 b, float3 c){return make_float3x3(a,b,c);});
    m.def("make_float3x3", [](float2x2 a){return make_float3x3(a);});
    m.def("make_float3x3", [](float3x3 a){return make_float3x3(a);});
    m.def("make_float3x3", [](float4x4 a){return make_float3x3(a);});

    m.def("make_float4x4", [](float a){return make_float4x4(a);});
    m.def("make_float4x4", [](
        float m00, float m01, float m02, float m03,
        float m10, float m11, float m12, float m13,
        float m20, float m21, float m22, float m23,
        float m30, float m31, float m32, float m33)
        {return make_float4x4(m00,m01,m02,m03, m10,m11,m12,m13, m20,m21,m22,m23, m30,m31,m32,m33);});
    m.def("make_float4x4", [](float4 a, float4 b, float4 c, float4 d){return make_float4x4(a,b,c,d);});
    m.def("make_float4x4", [](float2x2 a){return make_float4x4(a);});
    m.def("make_float4x4", [](float3x3 a){return make_float4x4(a);});
    m.def("make_float4x4", [](float4x4 a){return make_float4x4(a);});
    // TODO export matrix operators




}
