#include <pybind11/pybind11.h>
#include <pybind11/functional.h>
#include <pybind11/stl.h>
#include <runtime/shader.h>
#include <runtime/raster/raster_shader.h>
#include <ast/ast_evaluator.h>
#include <vstl/common.h>
#include <ast/function.h>
#include <ast/function_builder.h>
#include <py/managed_device.h>
#include <py/managed_accel.h>
#include <py/managed_bindless.h>
#include <core/thread_pool.h>

namespace py = pybind11;
using namespace luisa;
using namespace luisa::compute;
constexpr auto pyref = py::return_value_policy::reference;
using luisa::compute::detail::FunctionBuilder;
static vstd::vector<ASTEvaluator> analyzer;
struct IntEval {
    int32_t value;
    bool exist;
};
static size_t device_count = 0;

class ManagedMeshFormat {
public:
    MeshFormat format;
    luisa::vector<VertexAttribute> attributes;
};
static vstd::vector<std::shared_future<void>> futures;
static vstd::optional<ThreadPool> thread_pool;
PYBIND11_DECLARE_HOLDER_TYPE(T, luisa::shared_ptr<T>)
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
        thread_pool.destroy();
    }
}
static std::filesystem::path output_path;

struct VertexData {
    float3 position;
    float3 normal;
    float4 tangent;
    float4 color;
    std::array<float2, 4> uv;
    uint32_t vertex_id;
    uint32_t instance_id;
};

void export_runtime(py::module &m) {
    py::class_<ManagedMeshFormat>(m, "MeshFormat")
        .def(py::init<>())
        .def("add_attribute", [](ManagedMeshFormat &fmt, VertexAttributeType type, VertexElementFormat format) {
            fmt.attributes.emplace_back(VertexAttribute{.type = type, .format = format});
        })
        .def("add_stream", [](ManagedMeshFormat &fmt) {
            fmt.format.emplace_vertex_stream(fmt.attributes);
            fmt.attributes.clear();
        });

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
        .def("emplace_back", [](ManagedAccel &accel, uint64_t vertex_buffer, size_t vertex_buffer_offset, size_t vertex_buffer_size, size_t vertex_stride, uint64_t triangle_buffer, size_t triangle_buffer_offset, size_t triangle_buffer_size, float4x4 transform, bool allow_compact, bool allow_update, int visibility_mask, bool opaque) {
            MeshUpdateCmd cmd;
            cmd.option = {.hint = AccelOption::UsageHint::FAST_TRACE,
                          .allow_compaction = allow_compact,
                          .allow_update = allow_update};
            cmd.vertex_buffer = vertex_buffer;
            cmd.vertex_buffer_offset = vertex_buffer_offset;
            cmd.triangle_buffer_offset = triangle_buffer_offset;
            cmd.vertex_buffer_size = vertex_buffer_size;
            cmd.vertex_stride = vertex_stride;
            cmd.triangle_buffer = triangle_buffer;
            cmd.triangle_buffer_size = triangle_buffer_size;
            accel.emplace(cmd, transform, visibility_mask, opaque);
        })
        .def("emplace_procedural", [](ManagedAccel &accel, uint64_t aabb_buffer, size_t aabb_offset, size_t aabb_count, float4x4 transform, bool allow_compact, bool allow_update, int visibility_mask, bool opaque) {
            ProceduralUpdateCmd cmd;
            cmd.option = {.hint = AccelOption::UsageHint::FAST_TRACE,
                          .allow_compaction = allow_compact,
                          .allow_update = allow_update};
            cmd.aabb_buffer = aabb_buffer;
            cmd.aabb_offset = aabb_offset;
            cmd.aabb_count = aabb_count;
            accel.emplace(cmd, transform, visibility_mask, opaque);
        })
        .def("pop_back", [](ManagedAccel &accel) { accel.pop_back(); })
        .def("set", [](ManagedAccel &accel, size_t index, uint64_t vertex_buffer, size_t vertex_buffer_offset, size_t vertex_buffer_size, size_t vertex_stride, uint64_t triangle_buffer, size_t triangle_buffer_offset, size_t triangle_buffer_size, float4x4 transform, bool allow_compact, bool allow_update, int visibility_mask, bool opaque) {
            MeshUpdateCmd cmd;
            cmd.option = {.hint = AccelOption::UsageHint::FAST_TRACE,
                          .allow_compaction = allow_compact,
                          .allow_update = allow_update};
            cmd.vertex_buffer = vertex_buffer;
            cmd.vertex_buffer_offset = vertex_buffer_offset;
            cmd.triangle_buffer_offset = triangle_buffer_offset;
            cmd.vertex_buffer_size = vertex_buffer_size;
            cmd.vertex_stride = vertex_stride;
            cmd.triangle_buffer = triangle_buffer;
            cmd.triangle_buffer_size = triangle_buffer_size;
            accel.set(index, cmd, transform, visibility_mask, opaque);
        })
        .def("set_procedural", [](ManagedAccel &accel, size_t index, uint64_t aabb_buffer, size_t aabb_offset, size_t aabb_count, float4x4 transform, bool allow_compact, bool allow_update, int visibility_mask, bool opaque) {
            ProceduralUpdateCmd cmd;
            cmd.option = {.hint = AccelOption::UsageHint::FAST_TRACE,
                          .allow_compaction = allow_compact,
                          .allow_update = allow_update};
            cmd.aabb_buffer = aabb_buffer;
            cmd.aabb_offset = aabb_offset;
            cmd.aabb_count = aabb_count;
            accel.set(index, cmd, transform, visibility_mask, opaque);
        })
        .def("set_transform_on_update", [](ManagedAccel &a, size_t index, float4x4 transform) { a.GetAccel().set_transform_on_update(index, transform); })
        .def("set_visibility_on_update", [](ManagedAccel &a, size_t index, int visibility_mask) { a.GetAccel().set_visibility_on_update(index, visibility_mask); });
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

    py::class_<DeviceInterface, luisa::shared_ptr<DeviceInterface>>(m, "DeviceInterface")
        .def("create_shader", [](DeviceInterface &self, Function kernel, luisa::string_view str) {
            ShaderOption option{
                .enable_fast_math = true,
                .enable_debug_info = false,
                .compile_only = false,
                .name = luisa::string{str}};
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
                .name = luisa::string{str_view}};
            auto useless = self.create_shader(option, kernel);
        })
        .def("save_shader_async", [](DeviceInterface &self, luisa::shared_ptr<FunctionBuilder> const &builder, luisa::string_view str) {
            thread_pool.create();
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
                    .name = luisa::string{str_view}};
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
                return (type->is_scalar() && type->tag() != Type::Tag::BOOL);
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
        .def("save_raster_shader_async", [](DeviceInterface &self, ManagedMeshFormat const &fmt, luisa::shared_ptr<FunctionBuilder> const &vertex, luisa::shared_ptr<FunctionBuilder> const &pixel, luisa::string_view str) {
            thread_pool.create();
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
        .def("create_texture", [](DeviceInterface &d, PixelFormat format, uint32_t dimension, uint32_t width, uint32_t height, uint32_t depth, uint32_t mipmap_levels) {
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
    py::class_<Function>(m, "Function")
        .def("argument_size", [](Function &func) { return func.arguments().size(); });
    py::class_<IntEval>(m, "IntEval")
        .def("value", [](IntEval &self) { return self.value; })
        .def("exist", [](IntEval &self) { return self.exist; });
    py::class_<FunctionBuilder, luisa::shared_ptr<FunctionBuilder>>(m, "FunctionBuilder")
        .def("define_kernel", &FunctionBuilder::define_kernel<const luisa::function<void()> &>)
        .def("define_callable", &FunctionBuilder::define_callable<const luisa::function<void()> &>)
        .def("set_block_size", [](FunctionBuilder &self, uint32_t sx, uint32_t sy, uint32_t sz) { self.set_block_size(uint3(sx, sy, sz)); })
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
                visit(
                    [&]<typename T>(T const &t) {
                        if constexpr (std::is_same_v<T, monostate>) {
                            self.assign(l, r);
                        } else {
                            self.assign(l, self.literal(Type::of<T>(), t));
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
}