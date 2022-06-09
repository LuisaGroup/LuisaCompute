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
using AccelModification = AccelBuildCommand::Modification;

void export_op(py::module &m);
void export_vector2(py::module &m);
void export_vector3(py::module &m);
void export_vector4(py::module &m);
void export_matrix(py::module &m);

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
    m.doc() = "LuisaCompute API"; // optional module docstring

    // log
    m.def("log_level_verbose", luisa::log_level_verbose);
    m.def("log_level_info", luisa::log_level_info);
    m.def("log_level_warning", luisa::log_level_warning);
    m.def("log_level_error", luisa::log_level_error);

    // Context, device, stream
    py::class_<std::filesystem::path>(m, "FsPath")
        .def(py::init<std::string>());
    py::class_<Context>(m, "Context")
        .def(py::init<const std::filesystem::path &>())
        .def("create_device", [](Context& self, std::string_view backend_name){ return self.create_device(backend_name); }) // TODO: support properties
        .def("installed_backends", [](Context& self){
            std::vector<std::string> strs;
            for (auto s: self.installed_backends()) strs.push_back(s.c_str());
            return strs;
        });
    py::class_<Device>(m, "Device")
        .def("create_stream", &Device::create_stream, py::arg("present")=false)
        .def("impl", &Device::impl, pyref)
        .def("create_accel", &Device::create_accel);
    py::class_<Device::Interface, eastl::shared_ptr<Device::Interface>>(m, "DeviceInterface")
        .def("create_shader", [](Device::Interface& self, Function kernel){return self.create_shader(kernel, {});}) // TODO: support metaoptions
        .def("destroy_shader", &Device::Interface::destroy_shader)
        .def("create_buffer", &Device::Interface::create_buffer)
        .def("destroy_buffer", &Device::Interface::destroy_buffer)
        .def("create_texture", &Device::Interface::create_texture)
        .def("destroy_texture", &Device::Interface::destroy_texture)
        // .def("create_accel", &Device::Interface::create_accel) // (AccelUsageHint hint = AccelUsageHint::FAST_TRACE)
        // .def("destroy_accel", &Device::Interface::destroy_accel)
        .def("create_mesh", &Device::Interface::create_mesh) // (uint64_t v_buffer, size_t v_offset, size_t v_stride, size_t v_count, uint64_t t_buffer, size_t t_offset, size_t t_count, AccelUsageHint hint)
        .def("destroy_mesh", &Device::Interface::destroy_mesh)
        .def("create_bindless_array", &Device::Interface::create_bindless_array) // size 
        .def("destroy_bindless_array", &Device::Interface::destroy_bindless_array)
        .def("emplace_buffer_in_bindless_array", &Device::Interface::emplace_buffer_in_bindless_array) // arr, i, handle, offset_bytes
        .def("emplace_tex2d_in_bindless_array", &Device::Interface::emplace_tex2d_in_bindless_array) // arr, i, handle, sampler
        .def("emplace_tex3d_in_bindless_array", &Device::Interface::emplace_tex3d_in_bindless_array)
        .def("is_resource_in_bindless_array", &Device::Interface::is_resource_in_bindless_array) // arr, handle -> bool
        .def("remove_buffer_in_bindless_array", &Device::Interface::remove_buffer_in_bindless_array)
        .def("remove_tex2d_in_bindless_array", &Device::Interface::remove_tex2d_in_bindless_array)
        .def("remove_tex3d_in_bindless_array", &Device::Interface::remove_tex3d_in_bindless_array);


    py::class_<Stream>(m, "Stream")
        .def("synchronize", &Stream::synchronize)
        .def("add", [](Stream& self, Command* cmd){self<<cmd;})
        .def("add_callback", [](Stream& self, const std::function<void()>& callback){self<<callback;});


    // AST (FunctionBuilder)
    py::class_<Function>(m, "Function");
    py::class_<FunctionBuilder, eastl::shared_ptr<FunctionBuilder>>(m, "FunctionBuilder")
        .def("define_kernel", &FunctionBuilder::define_kernel<const std::function<void()> &>)
        .def("define_callable", &FunctionBuilder::define_callable<const std::function<void()> &>)
        .def("set_block_size", [](FunctionBuilder& self, uint sx, uint sy, uint sz){self.set_block_size(uint3{sx,sy,sz});})

        .def("thread_id", &FunctionBuilder::thread_id, pyref)
        .def("block_id", &FunctionBuilder::block_id, pyref)
        .def("dispatch_id", &FunctionBuilder::dispatch_id, pyref)
        .def("dispatch_size", &FunctionBuilder::dispatch_size, pyref)

        .def("local", &FunctionBuilder::local, pyref)
        // .def("shared")
        // .def("constant")
        .def("buffer_binding", &FunctionBuilder::buffer_binding, pyref)
        .def("texture_binding", &FunctionBuilder::texture_binding, pyref)
        .def("bindless_array_binding", &FunctionBuilder::bindless_array_binding, pyref)
        .def("accel_binding", &FunctionBuilder::accel_binding, pyref)

        .def("argument", &FunctionBuilder::argument, pyref)
        .def("reference", &FunctionBuilder::reference, pyref)
        .def("buffer", &FunctionBuilder::buffer, pyref)
        .def("texture", &FunctionBuilder::texture, pyref)
        .def("bindless_array", &FunctionBuilder::bindless_array, pyref)
        .def("accel", &FunctionBuilder::accel, pyref)

        .def("literal", &FunctionBuilder::literal, pyref)
        .def("unary", &FunctionBuilder::unary, pyref)
        .def("binary", &FunctionBuilder::binary, pyref)
        .def("member", &FunctionBuilder::member, pyref)
        .def("access", &FunctionBuilder::access, pyref)
        .def("swizzle", &FunctionBuilder::swizzle, pyref)
        .def("cast", &FunctionBuilder::cast, pyref)
        
        .def("call", [](FunctionBuilder& self, const Type *type, CallOp call_op, std::vector<const Expression *> args){return self.call(type, call_op, args);}, pyref)
        .def("call", [](FunctionBuilder& self, const Type *type, Function custom, std::vector<const Expression *> args){return self.call(type, custom, args);}, pyref)
        .def("call", [](FunctionBuilder& self, CallOp call_op, std::vector<const Expression *> args){self.call(call_op, args);})
        .def("call", [](FunctionBuilder& self, Function custom, std::vector<const Expression *> args){self.call(custom, args);})

        .def("break_", &FunctionBuilder::break_)
        .def("continue_", &FunctionBuilder::continue_)
        .def("return_", &FunctionBuilder::return_)
        .def("comment_", [](FunctionBuilder& self, std::string s){ self.comment_(s.c_str()); })
        .def("assign", &FunctionBuilder::assign, pyref)

        .def("if_", &FunctionBuilder::if_, pyref)
        .def("loop_", &FunctionBuilder::loop_, pyref)
        // .def("switch_")
        // .def("case_")
        // .def("default_")
        .def("for_", &FunctionBuilder::for_, pyref)
        // .def("meta") // unused
        .def("function", &FunctionBuilder::function); // returning object
    // current function builder
    m.def("builder", &FunctionBuilder::current, pyref);


    // expression types
    py::class_<Expression>(m, "Expression");
    py::class_<LiteralExpr, Expression>(m, "LiteralExpr");
    py::class_<RefExpr, Expression>(m, "RefExpr");
    py::class_<CallExpr, Expression>(m, "CallExpr");
    py::class_<UnaryExpr, Expression>(m, "UnaryExpr");
    py::class_<BinaryExpr, Expression>(m, "BinaryExpr");
    py::class_<MemberExpr, Expression>(m, "MemberExpr");
    py::class_<AccessExpr, Expression>(m, "AccessExpr");
    py::class_<CastExpr, Expression>(m, "CastExpr");
    // statement types
    py::class_<ScopeStmt>(m, "ScopeStmt") // not yet exporting base class (Statement)
        .def("__enter__", [](ScopeStmt& self){FunctionBuilder::current()->push_scope(&self);})
        .def("__exit__", [](ScopeStmt& self, py::object& e1, py::object& e2, py::object& tb){FunctionBuilder::current()->pop_scope(&self);});
    py::class_<IfStmt>(m, "IfStmt")
        .def("true_branch", py::overload_cast<>(&IfStmt::true_branch), pyref) // using overload_cast because there's also a const method variant
        .def("false_branch", py::overload_cast<>(&IfStmt::false_branch), pyref);
    py::class_<LoopStmt>(m, "LoopStmt")
        .def("body", py::overload_cast<>(&LoopStmt::body), pyref);
    py::class_<ForStmt>(m, "ForStmt")
        .def("body", py::overload_cast<>(&ForStmt::body), pyref);


    // OPs
    export_op(m); // UnaryOp, BinaryOp, CallOp. def at export_op.hpp

    py::class_<Type>(m, "Type")
        .def_static("from_", &Type::from, pyref)
        .def("size", &Type::size)
        .def("alignment", &Type::alignment)
        .def("is_scalar", &Type::is_scalar)
        .def("is_vector", &Type::is_vector)
        .def("is_matrix", &Type::is_matrix)
        .def("is_basic", &Type::is_basic)
        .def("is_array", &Type::is_array)
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
        .def("encode_uniform", [](ShaderDispatchCommand& self, char* buf, size_t size){self.encode_uniform(buf,size);})
        .def("encode_bindless_array", &ShaderDispatchCommand::encode_bindless_array)
        .def("encode_accel", &ShaderDispatchCommand::encode_accel);
    // buffer operation commands
    // Pybind can't deduce argument list of the create function, so using lambda to inform it
    py::class_<BufferUploadCommand, Command>(m, "BufferUploadCommand")
        .def_static("create", [](uint64_t handle, size_t offset_bytes, size_t size_bytes, py::buffer buf){
            return BufferUploadCommand::create(handle, offset_bytes, size_bytes, buf.request().ptr);
        }, pyref);
    py::class_<BufferDownloadCommand, Command>(m, "BufferDownloadCommand")
        .def_static("create", [](uint64_t handle, size_t offset_bytes, size_t size_bytes, py::buffer buf){
            return BufferDownloadCommand::create(handle, offset_bytes, size_bytes, buf.request().ptr);
        }, pyref);
    py::class_<BufferCopyCommand, Command>(m, "BufferCopyCommand")
        .def_static("create", [](uint64_t src, uint64_t dst, size_t src_offset, size_t dst_offset, size_t size){
            return BufferCopyCommand::create(src, dst, src_offset, dst_offset, size);
        }, pyref);
    // texture operation commands
    py::class_<TextureUploadCommand, Command>(m, "TextureUploadCommand")
        .def_static("create", [](uint64_t handle, PixelStorage storage, uint level, uint3 size, py::buffer buf){
            return TextureUploadCommand::create(handle, storage, level, size, buf.request().ptr);
        }, pyref);
    py::class_<TextureDownloadCommand, Command>(m, "TextureDownloadCommand")
        .def_static("create", [](uint64_t handle, PixelStorage storage, uint level, uint3 size, py::buffer buf){
            return TextureDownloadCommand::create(handle, storage, level, size, buf.request().ptr);
        }, pyref);
    py::class_<TextureCopyCommand, Command>(m, "TextureCopyCommand")
        .def_static("create", [](PixelStorage storage, uint64_t src_handle, uint64_t dst_handle, uint src_level, uint dst_level, uint3 size){
            return TextureCopyCommand::create(storage, src_handle, dst_handle, src_level, dst_level, size);
        }, pyref);
    py::class_<BufferToTextureCopyCommand, Command>(m, "BufferToTextureCopyCommand")
        .def_static("create", [](uint64_t buffer, size_t buffer_offset, uint64_t texture, PixelStorage storage, uint level, uint3 size){
            return BufferToTextureCopyCommand::create(buffer, buffer_offset, texture, storage, level, size);
        }, pyref);
    py::class_<TextureToBufferCopyCommand, Command>(m, "TextureToBufferCopyCommand")
        .def_static("create", [](uint64_t buffer, size_t buffer_offset, uint64_t texture, PixelStorage storage, uint level, uint3 size){
            return TextureToBufferCopyCommand::create(buffer, buffer_offset, texture, storage, level, size);
        }, pyref);
    // accel commands
    py::class_<MeshBuildCommand, Command>(m, "MeshBuildCommand")
        .def_static("create", [](uint64_t handle, AccelBuildRequest request,
                    uint64_t vertex_buffer, size_t vertex_buffer_offset, size_t vertex_buffer_size,
                    uint64_t triangle_buffer, size_t triangle_buffer_offset, size_t triangle_buffer_size) {
            return MeshBuildCommand::create(handle, request, vertex_buffer, vertex_buffer_offset, vertex_buffer_size,
                    triangle_buffer, triangle_buffer_offset, triangle_buffer_size);
        }, pyref);
    py::class_<AccelBuildCommand, Command>(m, "AccelBuildCommand")
        .def_static("create", [](uint64_t handle, uint32_t instance_count, AccelBuildRequest request, std::vector<AccelModification> modifications) {
            return AccelBuildCommand::create(handle, instance_count, request, luisa::vector<AccelModification>(modifications.cbegin(), modifications.cend()));
        }, pyref);
    // bindless
    py::class_<BindlessArrayUpdateCommand, Command>(m, "BindlessArrayUpdateCommand")
        .def_static("create", [](uint64_t handle){
            return BindlessArrayUpdateCommand::create(handle);
        }, pyref);


    // vector and matrix types
    export_vector2(m);
    export_vector3(m);
    export_vector4(m);
    // TODO export vector operators
    export_matrix(m);

    // util function for uniform encoding
    m.def("to_bytes", [](LiteralExpr::Value value){
        return luisa::visit([](auto x) noexcept { return py::bytes(std::string(reinterpret_cast<char*>(&x), sizeof(x))); }, value);
    });


    // accel
    py::class_<Accel>(m, "Accel")
        .def("size", &Accel::size)
        .def("handle", [](Accel& self){ return self.handle(); })
        .def("emplace_back", &Accel::emplace_back_mesh_with_handle)
        .def("set", &Accel::set_mesh_with_handle)
        .def("pop_back", &Accel::pop_back)
        .def("set_transform_on_update", &Accel::set_transform_on_update)
        .def("set_visibility_on_update", &Accel::set_visibility_on_update)
        .def("build_command", &Accel::build, pyref);

    py::enum_<AccelUsageHint>(m, "AccelUsageHint")
        .value("FAST_TRACE", AccelUsageHint::FAST_TRACE)
        .value("FAST_UPDATE", AccelUsageHint::FAST_UPDATE)
        .value("FAST_BUILD", AccelUsageHint::FAST_BUILD);

    py::enum_<AccelBuildRequest>(m, "AccelBuildRequest")
        .value("PREFER_UPDATE", AccelBuildRequest::PREFER_UPDATE)
        .value("FORCE_BUILD", AccelBuildRequest::FORCE_BUILD);

    py::class_<AccelModification>(m, "AccelModification")
        .def("set_transform", &AccelModification::set_transform)
        .def("set_visibility", &AccelModification::set_visibility)
        .def("set_mesh", &AccelModification::set_mesh);


    // pixel
    py::enum_<PixelFormat>(m, "PixelFormat")
        .value("R8SInt", PixelFormat::R8SInt)
        .value("R8UInt", PixelFormat::R8UInt)
        .value("R8UNorm", PixelFormat::R8UNorm)
        .value("RG8SInt", PixelFormat::RG8SInt)
        .value("RG8UInt", PixelFormat::RG8UInt)
        .value("RG8UNorm", PixelFormat::RG8UNorm)
        .value("RGBA8SInt", PixelFormat::RGBA8SInt)
        .value("RGBA8UInt", PixelFormat::RGBA8UInt)
        .value("RGBA8UNorm", PixelFormat::RGBA8UNorm)
        .value("R16SInt", PixelFormat::R16SInt)
        .value("R16UInt", PixelFormat::R16UInt)
        .value("R16UNorm", PixelFormat::R16UNorm)
        .value("RG16SInt", PixelFormat::RG16SInt)
        .value("RG16UInt", PixelFormat::RG16UInt)
        .value("RG16UNorm", PixelFormat::RG16UNorm)
        .value("RGBA16SInt", PixelFormat::RGBA16SInt)
        .value("RGBA16UInt", PixelFormat::RGBA16UInt)
        .value("RGBA16UNorm", PixelFormat::RGBA16UNorm)
        .value("R32SInt", PixelFormat::R32SInt)
        .value("R32UInt", PixelFormat::R32UInt)
        .value("RG32SInt", PixelFormat::RG32SInt)
        .value("RG32UInt", PixelFormat::RG32UInt)
        .value("RGBA32SInt", PixelFormat::RGBA32SInt)
        .value("RGBA32UInt", PixelFormat::RGBA32UInt)
        .value("R16F", PixelFormat::R16F)
        .value("RG16F", PixelFormat::RG16F)
        .value("RGBA16F", PixelFormat::RGBA16F)
        .value("R32F", PixelFormat::R32F)
        .value("RG32F", PixelFormat::RG32F)
        .value("RGBA32F", PixelFormat::RGBA32F);

    py::enum_<PixelStorage>(m, "PixelStorage")
        .value("BYTE1", PixelStorage::BYTE1)
        .value("BYTE2", PixelStorage::BYTE2)
        .value("BYTE4", PixelStorage::BYTE4)
        .value("SHORT1", PixelStorage::SHORT1)
        .value("SHORT2", PixelStorage::SHORT2)
        .value("SHORT4", PixelStorage::SHORT4)
        .value("INT1", PixelStorage::INT1)
        .value("INT2", PixelStorage::INT2)
        .value("INT4", PixelStorage::INT4)
        .value("HALF1", PixelStorage::HALF1)
        .value("HALF2", PixelStorage::HALF2)
        .value("HALF4", PixelStorage::HALF4)
        .value("FLOAT1", PixelStorage::FLOAT1)
        .value("FLOAT2", PixelStorage::FLOAT2)
        .value("FLOAT4", PixelStorage::FLOAT4);

    m.def("pixel_storage_channel_count", pixel_storage_channel_count);
    m.def("pixel_storage_to_format_int", pixel_storage_to_format<int>);
    m.def("pixel_storage_to_format_float", pixel_storage_to_format<float>);
    m.def("pixel_storage_size", pixel_storage_size);

    // sampler
    auto m_sampler = py::class_<Sampler>(m, "Sampler")
        .def(py::init<Sampler::Filter, Sampler::Address>());

    py::enum_<Sampler::Filter>(m_sampler, "Filter")
        .value("POINT", Sampler::Filter::POINT)
        .value("LINEAR_POINT", Sampler::Filter::LINEAR_POINT)
        .value("LINEAR_LINEAR", Sampler::Filter::LINEAR_LINEAR)
        .value("ANISOTROPIC", Sampler::Filter::ANISOTROPIC);

    py::enum_<Sampler::Address>(m_sampler, "Address")
        .value("EDGE", Sampler::Address::EDGE)
        .value("REPEAT", Sampler::Address::REPEAT)
        .value("MIRROR", Sampler::Address::MIRROR)
        .value("ZERO", Sampler::Address::ZERO);

}
