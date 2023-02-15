// This file exports LuisaCompute functionalities to a python library using pybind11.
//
// Class:
//   FunctionBuilder
//       define_kernel

#include <filesystem>
#include <pybind11/pybind11.h>
#include <pybind11/functional.h>
#include <pybind11/stl.h>
#include <ast/function.h>
#include <core/logging.h>
#include <runtime/device.h>
#include <runtime/context.h>
#include <runtime/stream.h>
#include <py/py_stream.h>
#include <runtime/command_encoder.h>
#include <runtime/image.h>
#include <runtime/rtx/accel.h>
#include <runtime/rtx/mesh.h>
#include <runtime/rtx/hit.h>
#include <runtime/rtx/ray.h>
#include <runtime/raster/raster_shader.h>
#include <runtime/raster/raster_scene.h>
#include <py/ref_counter.h>
#include <py/managed_accel.h>
#include <py/managed_bindless.h>
#include <ast/ast_evaluator.h>
#include <dsl/struct.h>
#include <core/thread_pool.h>
#include <py/managed_device.h>
namespace py = pybind11;
using namespace luisa;
using namespace luisa::compute;
using luisa::compute::detail::FunctionBuilder;
using AccelModification = AccelBuildCommand::Modification;

void export_op(py::module &m);
void export_vector2(py::module &m);
void export_vector3(py::module &m);
void export_vector4(py::module &m);
void export_matrix(py::module &m);
void export_img(py::module &m);
#ifdef LC_PY_ENABLE_GUI
void export_gui(py::module &m);
#endif
class ManagedMeshFormat {
public:
    MeshFormat format;
    luisa::vector<VertexAttribute> attributes;
};

struct VertexData {
    float3 position;
    float3 normal;
    float4 tangent;
    float4 color;
    std::array<float2, 4> uv;
    uint vertex_id;
    uint instance_id;
};


template<typename T>
class raw_ptr {

private:
    T *_p;

public:
    [[nodiscard]] raw_ptr(T *p) noexcept : _p{p} {}
    [[nodiscard]] T *get() const noexcept { return _p; }
    [[nodiscard]] T *operator->() const noexcept { return _p; }
    [[nodiscard]] T &operator*() const noexcept { return *_p; }
    [[nodiscard]] explicit operator bool() const noexcept { return _p != nullptr; }
};

PYBIND11_DECLARE_HOLDER_TYPE(T, raw_ptr<T>, true)
PYBIND11_DECLARE_HOLDER_TYPE(T, luisa::shared_ptr<T>)
static vstd::vector<std::shared_future<void>> futures;
static vstd::optional<ThreadPool> thread_pool;
static std::filesystem::path output_path;
static size_t device_count = 0;
ManagedDevice::ManagedDevice(Device &&device) noexcept : device(std::move(device)) {
    valid = true;
    if (device_count == 0) {
        if (!RefCounter::current)
            RefCounter::current = vstd::create_unique(new RefCounter());
    }
    device_count++;
}
ManagedDevice::ManagedDevice(ManagedDevice &&v) noexcept : device(std::move(v.device)), valid(false) {
    std::swap(valid, v.valid);
}
ManagedDevice::~ManagedDevice() noexcept {
    device_count--;
    for (auto &&i : futures) {
        i.wait();
    }
    futures.clear();
    if (device_count == 0) {
        thread_pool.Delete();
    }
}
struct IntEval {
    int32_t value;
    bool exist;
};

const auto pyref = py::return_value_policy::reference;// object lifetime is managed on C++ side
// Note: declare pointer & base class;
// use reference policy when python shouldn't destroy returned object
static vstd::vector<ASTEvaluator> analyzer;
PYBIND11_MODULE(lcapi, m) {
    m.doc() = "LuisaCompute API";// optional module docstring

    // log
    m.def("log_level_verbose", luisa::log_level_verbose);
    m.def("log_level_info", luisa::log_level_info);
    m.def("log_level_warning", luisa::log_level_warning);
    m.def("log_level_error", luisa::log_level_error);

    // Context, device, stream
    py::class_<Context>(m, "Context")
        .def(py::init<luisa::string>())
        .def("create_device", [](Context &self, luisa::string_view backend_name) {
            return ManagedDevice(self.create_device(backend_name));
        })// TODO: support properties
        .def("set_shader_path", [](Context &self, std::string const &str) {
            std::filesystem::path p{str};
            auto cp = std::filesystem::canonical(p);
            if (!std::filesystem::is_directory(cp))
                cp = std::filesystem::canonical(cp.parent_path());
            output_path = std::move(cp);
        })
        .def("create_headless_device", [](Context &self, luisa::string_view backend_name) {
            DeviceConfig settings{
                .headless = true};
            return ManagedDevice(self.create_device(backend_name, &settings));
        })// TODO: support properties
        .def("installed_backends", [](Context &self) {
            luisa::vector<luisa::string> strs;
            for (auto s : self.installed_backends()) strs.emplace_back(luisa::string_view(s.data(), s.size()));
            return strs;
        });
    py::class_<ManagedAccel>(m, "Accel")
        .def("size", [](ManagedAccel &accel) {
            return accel.GetAccel().size();
        })
        .def("handle", [](ManagedAccel &accel) { return accel.GetAccel().handle(); })
        .def("emplace_back", [](ManagedAccel &accel, uint64_t vertex_buffer, size_t vertex_buffer_offset, size_t vertex_buffer_size, size_t vertex_stride, uint64_t triangle_buffer, size_t triangle_buffer_offset, size_t triangle_buffer_size, float4x4 transform, bool allow_compact, bool allow_update, bool visible, bool opaque) {
            MeshUpdateCmd cmd;
            cmd.option = {.hint = AccelOption::UsageHint::FAST_BUILD,
                          .allow_compaction = allow_compact,
                          .allow_update = allow_update};
            cmd.vertex_buffer = vertex_buffer;
            cmd.vertex_buffer_offset = vertex_buffer_offset;
            cmd.triangle_buffer_offset = triangle_buffer_offset;
            cmd.vertex_buffer_size = vertex_buffer_size;
            cmd.vertex_stride = vertex_stride;
            cmd.triangle_buffer = triangle_buffer;
            cmd.triangle_buffer_size = triangle_buffer_size;
            accel.emplace(cmd, transform, visible, opaque);
        })
        .def("pop_back", [](ManagedAccel &accel) { accel.pop_back(); })
        .def("set", [](ManagedAccel &accel, size_t index, uint64_t vertex_buffer, size_t vertex_buffer_offset, size_t vertex_buffer_size, size_t vertex_stride, uint64_t triangle_buffer, size_t triangle_buffer_offset, size_t triangle_buffer_size, float4x4 transform, bool allow_compact, bool allow_update, bool visible, bool opaque) {
            MeshUpdateCmd cmd;
            cmd.option = {.hint = AccelOption::UsageHint::FAST_BUILD,
                          .allow_compaction = allow_compact,
                          .allow_update = allow_update};
            cmd.vertex_buffer = vertex_buffer;
            cmd.vertex_buffer_offset = vertex_buffer_offset;
            cmd.triangle_buffer_offset = triangle_buffer_offset;
            cmd.vertex_buffer_size = vertex_buffer_size;
            cmd.vertex_stride = vertex_stride;
            cmd.triangle_buffer = triangle_buffer;
            cmd.triangle_buffer_size = triangle_buffer_size;
            accel.set(index, cmd, transform, visible, opaque);
        })
        .def("set_transform_on_update", [](ManagedAccel &a, size_t index, float4x4 transform) { a.GetAccel().set_transform_on_update(index, transform); })
        .def("set_visibility_on_update", [](ManagedAccel &a, size_t index, bool visible) { a.GetAccel().set_visibility_on_update(index, visible); });
    py::class_<ManagedDevice>(m, "Device")
        .def(
            "create_stream", [](ManagedDevice &self, bool support_window) { return PyStream(self.device, support_window); })
        .def(
            "impl", [](ManagedDevice &s) { return s.device.impl(); }, pyref)
        .def("create_accel", [](ManagedDevice &device, AccelOption::UsageHint hint, bool allow_compact, bool allow_update) {
            return ManagedAccel(device.device.create_accel(AccelOption{
                .hint = hint,
                .allow_compaction = allow_compact,
                .allow_update = allow_update}));
        });
    m.def("get_bindless_handle", [](uint64 handle) {
        return reinterpret_cast<ManagedBindless *>(handle)->GetHandle();
    });
    // py::class_<DeviceInterface::BuiltinBuffer>(m, "BuiltinBuffer")
    //     .def("handle", [](DeviceInterface::BuiltinBuffer &buffer) {
    //         return buffer.handle;
    //     })
    //     .def("size", [](DeviceInterface::BuiltinBuffer &buffer) {
    //         return buffer.size;
    //     });

    py::class_<DeviceInterface, eastl::shared_ptr<DeviceInterface>>(m, "DeviceInterface")
        .def("create_shader", [](DeviceInterface &self, Function kernel, luisa::string_view str) {
            ShaderOption option{
                .enable_fast_math = true,
                .enable_debug_info = false,
                .compile_only = false,
                .name = str};
            return self.create_shader(option, kernel).handle;
        })// TODO: support metaoptions
        .def("save_shader", [](DeviceInterface &self, Function kernel, luisa::string_view str) {
            luisa::string_view str_view;
            luisa::string dst_path_str;
            if (!output_path.empty()) {
                auto dst_path = output_path / std::filesystem::path{str};
                dst_path_str = to_string(dst_path);
                str_view = dst_path_str;
            } else {
                str_view = str;
            }
            ShaderOption option{
                .enable_fast_math = true,
                .enable_debug_info = false,
                .compile_only = true,
                .name = str_view};
            auto useless = self.create_shader(option, kernel);
        })
        .def("save_shader_async", [](DeviceInterface &self, eastl::shared_ptr<FunctionBuilder> const &builder, luisa::string_view str) {
            thread_pool.New();
            futures.emplace_back(thread_pool->async([str = luisa::string{str}, builder, &self]() {
                luisa::string_view str_view;
                luisa::string dst_path_str;
                if (!output_path.empty()) {
                    auto dst_path = output_path / std::filesystem::path{str};
                    dst_path_str = to_string(dst_path);
                    str_view = dst_path_str;
                } else {
                    str_view = str;
                }
                ShaderOption option{
                    .enable_fast_math = true,
                    .enable_debug_info = false,
                    .compile_only = true,
                    .name = str_view};
                auto useless = self.create_shader(option, builder->function());
            }));
        })
        /*
        0: legal shader
        1: vertex return != pixel arg0
        2: illegal v2p type
        3: pixel output larger than 8
        4: pixel output illegal
        5: not callable
        6: illegal vertex first arguments
        */

        .def("check_raster_shader", [](DeviceInterface &self, Function vertex, Function pixel) -> int {
            if (vertex.tag() != Function::Tag::CALLABLE || pixel.tag() != Function::Tag::CALLABLE) return 5;
            auto v2p = vertex.return_type();
            auto vert_args = vertex.arguments();
            if (vert_args.empty() || vert_args[0].type() != Type::of<VertexData>()) {
                return 6;
            }
            auto pixel_args = pixel.arguments();
            if (pixel_args.size() < 1 || v2p != pixel_args[0].type()) {
                return 1;
            }
            auto pos = v2p;
            if (v2p->is_structure()) {
                if (v2p->members().empty()) return 2;
                pos = v2p->members()[0];
            }
            if (pos != Type::of<float4>()) {
                return 2;
            }
            auto legal_ret_type = [&](Type const *type) {
                if (type->is_vector()) {
                    type = type->element();
                }
                return (type->tag() == Type::Tag::INT || type->tag() == Type::Tag::UINT || type->tag() == Type::Tag::FLOAT);
            };
            auto ret = pixel.return_type();
            if (ret) {
                if (ret->is_structure()) {
                    auto mems = ret->members();
                    if (mems.size() > 8) return 3;
                    for (auto &&i : mems) {
                        if (!legal_ret_type(i)) return 4;
                    }
                } else {
                    if (!legal_ret_type(ret)) return 4;
                }
            }
            return 0;
        })
        .def("save_raster_shader", [](DeviceInterface &self, ManagedMeshFormat const &fmt, Function vertex, Function pixel, luisa::string_view str) {
            luisa::string_view str_view;
            luisa::string dst_path_str;
            if (!output_path.empty()) {
                auto dst_path = output_path / std::filesystem::path{str};
                dst_path_str = to_string(dst_path);
                str_view = dst_path_str;
            } else {
                str_view = str;
            }
            self.save_raster_shader(fmt.format, vertex, pixel, str_view, false, true);
        })
        .def("save_raster_shader_async", [](DeviceInterface &self, ManagedMeshFormat const &fmt, eastl::shared_ptr<FunctionBuilder> const &vertex, eastl::shared_ptr<FunctionBuilder> const &pixel, luisa::string_view str) {
            thread_pool.New();
            futures.emplace_back(thread_pool->async([fmt, str = luisa::string{str}, vertex, pixel, &self]() {
                luisa::string_view str_view;
                luisa::string dst_path_str;
                if (!output_path.empty()) {
                    auto dst_path = output_path / std::filesystem::path{str};
                    dst_path_str = to_string(dst_path);
                    str_view = dst_path_str;
                } else {
                    str_view = str;
                }
                self.save_raster_shader(fmt.format, vertex->function(), pixel->function(), str_view, false, true);
            }));
        })
        .def("destroy_shader", &DeviceInterface::destroy_shader)
        .def("create_buffer", [](DeviceInterface &d, const Type *type, size_t size) {
            auto ptr = d.create_buffer(type, size).handle;
            RefCounter::current->AddObject(ptr, {[](DeviceInterface *d, uint64 handle) { d->destroy_buffer(handle); }, &d});
            return ptr;
        })
        // .def("create_dispatch_buffer", [](DeviceInterface &d, uint32_t dimension, size_t size) {
        //     auto ptr = d.create_dispatch_buffer(dimension, size);
        //     RefCounter::current->AddObject(ptr.handle, {[](DeviceInterface *d, uint64 handle) { d->destroy_buffer(handle); }, &d});
        //     return ptr;
        // })
        .def("destroy_buffer", [](DeviceInterface &d, uint64_t handle) {
            RefCounter::current->DeRef(handle);
        })
        .def("create_texture", [](DeviceInterface &d, PixelFormat format, uint dimension, uint width, uint height, uint depth, uint mipmap_levels) {
            auto ptr = d.create_texture(format, dimension, width, height, depth, mipmap_levels).handle;
            RefCounter::current->AddObject(ptr, {[](DeviceInterface *d, uint64 handle) { d->destroy_texture(handle); }, &d});
            return ptr;
        })
        .def("destroy_texture", [](DeviceInterface &d, uint64_t handle) {
            RefCounter::current->DeRef(handle);
        })
        .def("create_bindless_array", [](DeviceInterface &d, size_t slots) {
            return reinterpret_cast<uint64>(new_with_allocator<ManagedBindless>(&d, slots));
        })// size
        .def("destroy_bindless_array", [](DeviceInterface &d, uint64 handle) {
            delete_with_allocator(reinterpret_cast<ManagedBindless *>(handle));
        })
        .def("emplace_buffer_in_bindless_array", [](DeviceInterface &d, uint64_t array, size_t index, uint64_t handle, size_t offset_bytes) {
            reinterpret_cast<ManagedBindless *>(array)->emplace_buffer(index, handle, offset_bytes);
        })// arr, i, handle, offset_bytes
        .def("emplace_tex2d_in_bindless_array", [](DeviceInterface &d, uint64_t array, size_t index, uint64_t handle, Sampler sampler) {
            reinterpret_cast<ManagedBindless *>(array)->emplace_tex2d(index, handle, sampler);
        })// arr, i, handle, sampler
        .def("emplace_tex3d_in_bindless_array", [](DeviceInterface &d, uint64_t array, size_t index, uint64_t handle, Sampler sampler) {
            reinterpret_cast<ManagedBindless *>(array)->emplace_tex3d(index, handle, sampler);
        })
        // .def("is_resource_in_bindless_array", &DeviceInterface::is_resource_in_bindless_array) // arr, handle -> bool
        .def("remove_buffer_in_bindless_array", [](DeviceInterface &d, uint64_t array, size_t index) {
            reinterpret_cast<ManagedBindless *>(array)->remove_buffer(index);
        })
        .def("remove_tex2d_in_bindless_array", [](DeviceInterface &d, uint64_t array, size_t index) {
            reinterpret_cast<ManagedBindless *>(array)->remove_tex2d(index);
        })
        .def("remove_tex3d_in_bindless_array", [](DeviceInterface &d, uint64_t array, size_t index) {
            reinterpret_cast<ManagedBindless *>(array)->remove_tex3d(index);
        });

    py::class_<PyStream>(m, "Stream")
        .def(
            "synchronize", [](PyStream &self) { self.sync(); }, pyref)
        .def(
            "add", [](PyStream &self, Command *cmd) { self.add(cmd); }, pyref)
        .def(
            "add_upload_buffer", [](PyStream &self, py::buffer &&buf) { self.add_upload(std::move(buf)); }, pyref)
        // .def(
        //     "add_readback_buffer", [](PyStream &self, py::buffer &&buf) { self.add_readback(std::move(buf)); }, pyref)
        .def(
            "update_accel", [](PyStream &self, ManagedAccel &accel) {
                accel.update(self);
            })
        .def("update_bindless", [](PyStream &self, uint64 bindless) {
            reinterpret_cast<ManagedBindless *>(bindless)->Update(self);
        })
        .def(
            "execute", [](PyStream &self) { self.execute(); }, pyref);

    // AST (FunctionBuilder)
    py::class_<Function>(m, "Function")
        .def("argument_size", [](Function &func) { return func.arguments().size(); });
    py::class_<IntEval>(m, "IntEval")
        .def("value", [](IntEval &self) { return self.value; })
        .def("exist", [](IntEval &self) { return self.exist; });
    py::class_<FunctionBuilder, eastl::shared_ptr<FunctionBuilder>>(m, "FunctionBuilder")
        .def("define_kernel", &FunctionBuilder::define_kernel<const luisa::function<void()> &>)
        .def("define_callable", &FunctionBuilder::define_callable<const luisa::function<void()> &>)
        .def("set_block_size", [](FunctionBuilder &self, uint sx, uint sy, uint sz) { self.set_block_size(uint3(sx, sy, sz)); })
        .def("try_eval_int", [](FunctionBuilder &self, Expression const *expr) {
            auto eval = analyzer.back().try_eval(expr);
            return visit(
                [&]<typename T>(T const &t) -> IntEval {
                    if constexpr (std::is_same_v<T, int32_t> || std::is_same_v<T, uint32_t>) {
                        return {
                            .value = static_cast<int32_t>(t),
                            .exist = true};
                    } else {
                        return {.value = 0, .exist = false};
                    }
                },
                eval);
        })

        .def("thread_id", &FunctionBuilder::thread_id, pyref)
        .def("block_id", &FunctionBuilder::block_id, pyref)
        .def("dispatch_id", &FunctionBuilder::dispatch_id, pyref)
        .def("kernel_id", &FunctionBuilder::kernel_id, pyref)
        .def("object_id", &FunctionBuilder::object_id, pyref)
        .def("dispatch_size", &FunctionBuilder::dispatch_size, pyref)

        .def("local", &FunctionBuilder::local, pyref)
        .def("shared", &FunctionBuilder::shared, pyref)
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

        .def(
            "literal", [](FunctionBuilder &self, const Type *type, LiteralExpr::Value value) {
                return self.literal(type, std::move(value));
            },
            pyref)
        .def("unary", &FunctionBuilder::unary, pyref)
        .def("binary", &FunctionBuilder::binary, pyref)
        .def("member", &FunctionBuilder::member, pyref)
        .def("access", &FunctionBuilder::access, pyref)
        .def("swizzle", &FunctionBuilder::swizzle, pyref)
        .def("cast", &FunctionBuilder::cast, pyref)

        .def(
            "call", [](FunctionBuilder &self, const Type *type, CallOp call_op, const luisa::vector<const Expression *> &args) { return self.call(type, call_op, std::move(args)); }, pyref)
        .def(
            "call", [](FunctionBuilder &self, const Type *type, Function custom, const luisa::vector<const Expression *> &args) {
                analyzer.back().check_call_ref(custom, args);
                 return self.call(type, custom, std::move(args)); }, pyref)
        .def("call", [](FunctionBuilder &self, CallOp call_op, const luisa::vector<const Expression *> &args) { self.call(call_op, std::move(args)); })
        .def("call", [](FunctionBuilder &self, Function custom, const luisa::vector<const Expression *> &args) {
            analyzer.back().check_call_ref(custom, args);
            self.call(custom, std::move(args));
        })

        .def("break_", &FunctionBuilder::break_)
        .def("continue_", &FunctionBuilder::continue_)
        .def("return_", &FunctionBuilder::return_)
        .def(
            "assign", [](FunctionBuilder &self, Expression const *l, Expression const *r) {
                auto result = analyzer.back().assign(l, r);
                // FIXME: the following checks are not reliable
                auto is_initialized = [&self](const Expression *expr) noexcept {
                    if (expr->tag() == Expression::Tag::REF) {
                        auto v = static_cast<const RefExpr *>(expr)->variable();
                        return v.externally_initialized() ||
                               (to_underlying(self.variable_usage(v.uid())) &
                                to_underlying(Usage::WRITE)) != 0u;
                    }
                    return true;
                };
                auto assign = [&](const Expression *lhs, const Expression *rhs) noexcept {
                    // FIXME: the following checks are not reliable
                    if (!is_initialized(rhs)) [[unlikely]] {
                        if (!is_initialized(lhs)) {
                            return;
                        }
                        LUISA_ERROR_WITH_LOCATION("Cannot assign the value of "
                                                  "an uninitialized variable.");
                    }
                    self.assign(lhs, rhs);
                };
                visit(
                    [&]<typename T>(T const &t) {
                        if constexpr (std::is_same_v<T, monostate>) {
                            assign(l, r);
                        } else {
                            assign(l, self.literal(Type::of<T>(), t));
                        }
                    },
                    result);
            },
            pyref)

        .def("if_", &FunctionBuilder::if_, pyref)
        .def("switch_", &FunctionBuilder::switch_, pyref)
        .def("case_", &FunctionBuilder::case_, pyref)
        .def("loop_", &FunctionBuilder::loop_, pyref)
        // .def("switch_")
        // .def("case_")
        .def("default_", &FunctionBuilder::default_, pyref)
        .def(
            "for_", [](FunctionBuilder &self, const Expression *var, const Expression *condition, const Expression *update) {
                auto ptr = self.for_(var, condition, update);
                analyzer.back().execute_for(ptr);
                return ptr;
            },
            pyref)
        // .def("meta") // unused
        .def("function", &FunctionBuilder::function);// returning object
    // current function builder
    m.def("builder", &FunctionBuilder::current, pyref);
    m.def("begin_analyzer", []() {
        analyzer.emplace_back();
    });
    m.def("end_analyzer", []() {
        analyzer.pop_back();
    });
    m.def("begin_branch", [](bool is_loop) {
        analyzer.back().begin_branch_scope(is_loop);
    });
    m.def("end_branch", []() {
        analyzer.back().end_branch_scope();
    });
    m.def("begin_switch", [](SwitchStmt const *stmt) {
        analyzer.back().begin_switch(stmt);
    });
    m.def("end_switch", []() {
        analyzer.back().end_switch();
    });
    m.def("analyze_condition", [](Expression const *expr) -> int32_t {
        auto result = analyzer.back().try_eval(expr);
        return visit(
            [&]<typename T>(T const &t) -> int32_t {
                if constexpr (std::is_same_v<T, bool>) {
                    if (t) {
                        return 0;// true
                    } else {
                        return 1;// false
                    }
                } else {
                    return 2;// unsure
                }
            },
            result);
    });
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
    py::class_<ScopeStmt>(m, "ScopeStmt")// not yet exporting base class (Statement)
        .def("__enter__", [](ScopeStmt &self) { FunctionBuilder::current()->push_scope(&self); })
        .def("__exit__", [](ScopeStmt &self, py::object &e1, py::object &e2, py::object &tb) { FunctionBuilder::current()->pop_scope(&self); });
    py::class_<IfStmt>(m, "IfStmt")
        .def("true_branch", py::overload_cast<>(&IfStmt::true_branch), pyref)// using overload_cast because there's also a const method variant
        .def("false_branch", py::overload_cast<>(&IfStmt::false_branch), pyref);
    py::class_<SwitchStmt>(m, "SwitchStmt")
        .def("body", py::overload_cast<>(&SwitchStmt::body), pyref);
    py::class_<SwitchCaseStmt>(m, "SwitchCaseStmt")
        .def("body", py::overload_cast<>(&SwitchCaseStmt::body), pyref);
    py::class_<SwitchDefaultStmt>(m, "SwitchDefaultStmt")
        .def("body", py::overload_cast<>(&SwitchDefaultStmt::body), pyref);
    py::class_<LoopStmt>(m, "LoopStmt")
        .def("body", py::overload_cast<>(&LoopStmt::body), pyref);
    py::class_<ForStmt>(m, "ForStmt")
        .def("body", py::overload_cast<>(&ForStmt::body), pyref);

    // OPs
    export_op(m);// UnaryOp, BinaryOp, CallOp. def at export_op.hpp

    py::class_<Type, raw_ptr<Type>>(m, "Type")
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
        .def("is_custom", &Type::is_custom)
        .def("element", &Type::element, pyref)
        .def("description", &Type::description)
        .def("dimension", &Type::dimension)
        .def_static("custom", [](luisa::string_view str) {
            return Type::custom(str);
        });
    // commands
    py::class_<Command>(m, "Command");
    py::class_<ShaderDispatchCommand, Command>(m, "ShaderDispatchCommand");
    py::class_<ComputeDispatchCmdEncoder>(m, "ComputeDispatchCmdEncoder")
        .def_static(
            "create", [](size_t arg_size, uint64_t handle, Function func) { return make_unique<ComputeDispatchCmdEncoder>(arg_size, handle, func).release(); }, pyref)
        .def("set_dispatch_size", [](ComputeDispatchCmdEncoder &self, uint sx, uint sy, uint sz) { self.set_dispatch_size(uint3{sx, sy, sz}); })
        .def("set_dispatch_buffer", [](ComputeDispatchCmdEncoder &self, uint64_t handle) { self.set_dispatch_size(IndirectDispatchArg{handle}); })
        .def("encode_buffer", &ComputeDispatchCmdEncoder::encode_buffer)
        .def("encode_texture", &ComputeDispatchCmdEncoder::encode_texture)
        .def("encode_uniform", [](ComputeDispatchCmdEncoder &self, char *buf, size_t size) { self.encode_uniform(buf, size); })
        .def("encode_bindless_array", &ComputeDispatchCmdEncoder::encode_bindless_array)
        .def("encode_accel", &ComputeDispatchCmdEncoder::encode_accel)
        .def(
            "build", [](ComputeDispatchCmdEncoder &c) { return std::move(c).build().release(); }, pyref);
    // buffer operation commands
    // Pybind can't deduce argument list of the create function, so using lambda to inform it
    py::class_<BufferUploadCommand, Command>(m, "BufferUploadCommand")
        .def_static(
            "create", [](uint64_t handle, size_t offset_bytes, size_t size_bytes, py::buffer const &buf) {
                return BufferUploadCommand::create(handle, offset_bytes, size_bytes, buf.request().ptr).release();
            },
            pyref);
    py::class_<BufferDownloadCommand, Command>(m, "BufferDownloadCommand")
        .def_static(
            "create", [](uint64_t handle, size_t offset_bytes, size_t size_bytes, py::buffer const &buf) {
                return BufferDownloadCommand::create(handle, offset_bytes, size_bytes, buf.request().ptr).release();
            },
            pyref);
    py::class_<BufferCopyCommand, Command>(m, "BufferCopyCommand")
        .def_static(
            "create", [](uint64_t src, uint64_t dst, size_t src_offset, size_t dst_offset, size_t size) {
                return BufferCopyCommand::create(src, dst, src_offset, dst_offset, size).release();
            },
            pyref);
    // texture operation commands
    py::class_<TextureUploadCommand, Command>(m, "TextureUploadCommand")
        .def_static(
            "create", [](uint64_t handle, PixelStorage storage, uint level, uint3 size, py::buffer const &buf) {
                return TextureUploadCommand::create(handle, storage, level, size, buf.request().ptr).release();
            },
            pyref);
    py::class_<TextureDownloadCommand, Command>(m, "TextureDownloadCommand")
        .def_static(
            "create", [](uint64_t handle, PixelStorage storage, uint level, uint3 size, py::buffer const &buf) {
                return TextureDownloadCommand::create(handle, storage, level, size, buf.request().ptr).release();
            },
            pyref);
    py::class_<TextureCopyCommand, Command>(m, "TextureCopyCommand")
        .def_static(
            "create", [](PixelStorage storage, uint64_t src_handle, uint64_t dst_handle, uint src_level, uint dst_level, uint3 size) {
                return TextureCopyCommand::create(storage, src_handle, dst_handle, src_level, dst_level, size).release();
            },
            pyref);
    py::class_<BufferToTextureCopyCommand, Command>(m, "BufferToTextureCopyCommand")
        .def_static(
            "create", [](uint64_t buffer, size_t buffer_offset, uint64_t texture, PixelStorage storage, uint level, uint3 size) {
                return BufferToTextureCopyCommand::create(buffer, buffer_offset, texture, storage, level, size).release();
            },
            pyref);
    py::class_<TextureToBufferCopyCommand, Command>(m, "TextureToBufferCopyCommand")
        .def_static(
            "create", [](uint64_t buffer, size_t buffer_offset, uint64_t texture, PixelStorage storage, uint level, uint3 size) {
                return TextureToBufferCopyCommand::create(buffer, buffer_offset, texture, storage, level, size).release();
            },
            pyref);
    // vector and matrix types
    export_vector2(m);
    export_vector3(m);
    export_vector4(m);
    // TODO export vector operators
    export_matrix(m);

    // util function for uniform encoding
    m.def("to_bytes", [](LiteralExpr::Value value) {
        return luisa::visit([](auto x) noexcept { return py::bytes(std::string(reinterpret_cast<char *>(&x), sizeof(x))); }, value);
    });
    //.def()

    // accel
    /*
    py::class_<AccelWrapper>(m, "Accel")
        .def("size", [](AccelWrapper &a) { return a.accel.size(); })
        .def("handle", [](AccelWrapper &self) { return self.accel.handle(); })
        .def("emplace_back", [](AccelWrapper &accel, uint64_t mesh_handle, float4x4 transform, bool visible) {
            auto sz = accel.accel.size();
            accel.accel.emplace_back_handle(mesh_handle, transform, visible);
            RefCounter::current->SetAccelRef(accel.accel.handle(), sz, mesh_handle);
        })
        .def("set", [](AccelWrapper &accel, size_t index, uint64_t mesh, float4x4 transform, bool visible) {
            accel.accel.set_handle(index, mesh, transform, visible);
            RefCounter::current->SetAccelRef(accel.accel.handle(), index, mesh);
        })
        .def("pop_back", [](AccelWrapper &accel) {
            accel.accel.pop_back();
            auto sz = accel.accel.size();
            RefCounter::current->SetAccelRef(accel.accel.handle(), sz, 0);
        })
        .def("set_transform_on_update", [](AccelWrapper &a, size_t index, float4x4 transform) { a.accel.set_transform_on_update(index, transform); })
        .def("set_visibility_on_update", [](AccelWrapper &a, size_t index, bool visible) { a.accel.set_visibility_on_update(index, visible); })
        .def(
            "build_command", [](AccelWrapper &self, Accel::BuildRequest request) { return self.accel.build(request).release(); }, pyref);
*/
    py::enum_<AccelOption::UsageHint>(m, "AccelUsageHint")
        .value("FAST_TRACE", AccelOption::UsageHint::FAST_TRACE)
        .value("FAST_BUILD", AccelOption::UsageHint::FAST_BUILD);

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
        .value("RGBA32F", PixelFormat::RGBA32F)
        .value("BC4UNorm", PixelFormat::BC4UNorm)
        .value("BC5UNorm", PixelFormat::BC5UNorm)
        .value("BC6HUF16", PixelFormat::BC6HUF16)
        .value("BC7UNorm", PixelFormat::BC7UNorm);

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
        .value("FLOAT4", PixelStorage::FLOAT4)
        .value("BC4", PixelStorage::BC4)
        .value("BC5", PixelStorage::BC5)
        .value("BC6", PixelStorage::BC6)
        .value("BC7", PixelStorage::BC7);

    m.def("pixel_storage_channel_count", pixel_storage_channel_count);
    m.def("pixel_storage_to_format_int", pixel_storage_to_format<int>);
    m.def("pixel_storage_to_format_float", pixel_storage_to_format<float>);
    auto _pixel_storage_size = [](PixelStorage storage) { return pixel_storage_size(storage); };
    m.def("pixel_storage_size", _pixel_storage_size);

    // sampler
    auto m_sampler = py::class_<Sampler>(m, "Sampler")
                         .def(py::init<Sampler::Filter, Sampler::Address>());

    py::enum_<Sampler::Filter>(m, "Filter")
        .value("POINT", Sampler::Filter::POINT)
        .value("LINEAR_POINT", Sampler::Filter::LINEAR_POINT)
        .value("LINEAR_LINEAR", Sampler::Filter::LINEAR_LINEAR)
        .value("ANISOTROPIC", Sampler::Filter::ANISOTROPIC);

    py::enum_<Sampler::Address>(m, "Address")
        .value("EDGE", Sampler::Address::EDGE)
        .value("REPEAT", Sampler::Address::REPEAT)
        .value("MIRROR", Sampler::Address::MIRROR)
        .value("ZERO", Sampler::Address::ZERO);
    export_img(m);
    py::enum_<VertexElementFormat>(m, "VertexElementFormat")
        .value("XYZW8UNorm", VertexElementFormat::XYZW8UNorm)
        .value("XY16UNorm", VertexElementFormat::XY16UNorm)
        .value("XYZW16UNorm", VertexElementFormat::XYZW16UNorm)
        .value("XY16Float", VertexElementFormat::XY16Float)
        .value("XYZW16Float", VertexElementFormat::XYZW16Float)
        .value("X32Float", VertexElementFormat::X32Float)
        .value("XY32Float", VertexElementFormat::XY32Float)
        .value("XYZ32Float", VertexElementFormat::XYZ32Float)
        .value("XYZW32Float", VertexElementFormat::XYZW32Float);
    py::enum_<VertexAttributeType>(m, "VertexAttributeType")
        .value("Position", VertexAttributeType::Position)
        .value("Normal", VertexAttributeType::Normal)
        .value("Tangent", VertexAttributeType::Tangent)
        .value("Color", VertexAttributeType::Color)
        .value("UV0", VertexAttributeType::UV0)
        .value("UV1", VertexAttributeType::UV1)
        .value("UV2", VertexAttributeType::UV2)
        .value("UV3", VertexAttributeType::UV3);
    py::class_<ManagedMeshFormat>(m, "MeshFormat")
        .def(py::init<>())
        .def("add_attribute", [](ManagedMeshFormat &fmt, VertexAttributeType type, VertexElementFormat format) {
            fmt.attributes.emplace_back(VertexAttribute{.type = type, .format = format});
        })
        .def("add_stream", [](ManagedMeshFormat &fmt) {
            fmt.format.emplace_vertex_stream(fmt.attributes);
            fmt.attributes.clear();
        });
#ifdef LC_PY_ENABLE_GUI
    export_gui(m);
#endif
}
