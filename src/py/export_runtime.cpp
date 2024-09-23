#include <fstream>

#include <pybind11/pybind11.h>
#include <pybind11/functional.h>
#include <pybind11/stl.h>

#include "ast_evaluator.h"
#include "managed_device.h"
#include "managed_accel.h"
#include "managed_bindless.h"

#include <luisa/luisa-compute.h>

namespace py = pybind11;
using namespace luisa;
using namespace luisa::compute;
constexpr auto pyref = py::return_value_policy::reference;
using luisa::compute::detail::FunctionBuilder;
static vstd::vector<luisa::optional<ASTEvaluator>> analyzer;
static luisa::weak_ptr<PyStream::Data> default_stream_data;
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
template <typename T>
struct halfN{
    static constexpr bool value = false;
};
template <size_t n>
struct halfN<luisa::Vector<half, n>>{
    static constexpr bool value = true;
    using Type = bool;
    static constexpr size_t dimension = n;
};
static vstd::vector<luisa::fiber::event> futures;
static vstd::optional<luisa::fiber::scheduler> thread_pool;
PYBIND11_DECLARE_HOLDER_TYPE(T, luisa::shared_ptr<T>)
ManagedDevice::ManagedDevice(Device &&device) noexcept : device(std::move(device)) {
    valid = true;
    if (device_count == 0) {
        if (!RefCounter::current)
            RefCounter::current = vstd::create_unique(new RefCounter());
    }
    device_count++;
}
ManagedDevice::ManagedDevice(ManagedDevice &&v) noexcept : device(std::move(v.device)), valid(v.valid) {
    v.valid = false;
}
ManagedDevice::~ManagedDevice() noexcept {
    if (valid) {
        device_count--;
        for (auto &&i : futures) {
            i.wait();
        }
        futures.clear();

        if (device_count == 0) {
            RefCounter::current = nullptr;
            thread_pool.destroy();
        }
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
struct AtomicAccessChain {
    using Node = luisa::compute::detail::AtomicRefNode;
    Node const *node{};
};

class UserBinaryIO : public BinaryIO {

private:
    std::filesystem::path _path;

public:
    UserBinaryIO() noexcept {

#ifdef LUISA_PLATFORM_WINDOWS
        auto home = getenv("USERPROFILE");
#else
        auto home = getenv("HOME");
#endif
        if (!home) {
            LUISA_WARNING("Failed to get user home directory: environment variable not found.");
        } else {
            std::error_code ec;
            auto p = std::filesystem::canonical(home, ec);
            if (!ec) {
                _path = p / ".luisa";
            } else {
                LUISA_WARNING("Failed to get user home directory: {}.", ec.message());
            }
        }
        if (_path.empty()) {
            LUISA_WARNING("Failed to get user home directory. Using temporary directory instead.");
            _path = std::filesystem::temp_directory_path() / ".luisa";
        }
        std::error_code ec;
        std::filesystem::create_directories(_path, ec);
        if (ec) {
            LUISA_WARNING("Failed to create application data directory at '{}': {}.",
                          _path.string(), ec.message());
        }
    }

public:
    unique_ptr<BinaryStream> read_shader_bytecode(luisa::string_view name) const noexcept override {
        return luisa::make_unique<BinaryFileStream>(luisa::string{name});
    }
    unique_ptr<BinaryStream> read_shader_cache(luisa::string_view name) const noexcept override {
        if (_path.empty()) { return {}; }
        auto path = _path / "cache" / name;
        return luisa::make_unique<BinaryFileStream>(luisa::string{path.string()});
    }
    unique_ptr<BinaryStream> read_internal_shader(luisa::string_view name) const noexcept override {
        if (_path.empty()) { return {}; }
        auto path = _path / "internal" / name;
        return luisa::make_unique<BinaryFileStream>(luisa::string{path.string()});
    }
    filesystem::path write_shader_bytecode(luisa::string_view name, luisa::span<const std::byte> data) const noexcept override {
        std::filesystem::path path{name};
        if (std::ofstream file{path, std::ios::binary}) {
            file.write(reinterpret_cast<const char *>(data.data()), data.size_bytes());
            return path;
        }
        LUISA_WARNING("Failed to write shader bytecode to '{}'.", name);
        return {};
    }
    void clear_shader_cache() const noexcept override {
        if (_path.empty()) { return; }
        auto cache_path = _path / "cache";
        std::error_code ec;
        std::filesystem::remove_all(cache_path, ec);
        if (ec) {
            LUISA_WARNING("Failed to remove cache directory '{}': {}.",
                          cache_path.string(), ec.message());
        }
    }
    filesystem::path write_shader_cache(luisa::string_view name, luisa::span<const std::byte> data) const noexcept override {
        if (_path.empty()) { return {}; }
        auto cache_path = _path / "cache";
        std::error_code ec;
        std::filesystem::create_directories(cache_path, ec);
        if (ec) {
            LUISA_WARNING("Failed to create application cache directory at '{}': {}.",
                          cache_path.string(), ec.message());
            return {};
        }
        auto path = cache_path / name;
        if (std::ofstream file{path, std::ios::binary}) {
            file.write(reinterpret_cast<const char *>(data.data()), data.size_bytes());
            return path;
        }
        LUISA_WARNING("Failed to write shader cache to '{}'.", path.string());
        return {};
    }
    filesystem::path write_internal_shader(luisa::string_view name, luisa::span<const std::byte> data) const noexcept override {
        if (_path.empty()) { return {}; }
        auto internal_path = _path / "internal";
        std::error_code ec;
        std::filesystem::create_directories(internal_path, ec);
        if (ec) {
            LUISA_WARNING("Failed to create application internal data directory at '{}': {}.",
                          internal_path.string(), ec.message());
            return {};
        }
        auto path = internal_path / name;
        if (std::ofstream file{path, std::ios::binary}) {
            file.write(reinterpret_cast<const char *>(data.data()), data.size_bytes());
            return path;
        }
        LUISA_WARNING("Failed to write internal shader to '{}'.", path.string());
        return {};
    }
};

void export_runtime(py::module &m) {
    // py::class_<ManagedMeshFormat>(m, "MeshFormat")
    //     .def(py::init<>())
    //     .def("add_attribute", [](ManagedMeshFormat &fmt, VertexAttributeType type, VertexElementFormat format) {
    //         fmt.attributes.emplace_back(VertexAttribute{.type = type, .format = format});
    //     })
    //     .def("add_stream", [](ManagedMeshFormat &fmt) {
    //         fmt.format.emplace_vertex_stream(fmt.attributes);
    //         fmt.attributes.clear();
    //     });
    py::class_<ResourceCreationInfo>(m, "ResourceCreationInfo")
        .def(py::init<>())
        .def("handle", [](ResourceCreationInfo &self) { return self.handle; })
        .def("native_handle", [](ResourceCreationInfo &self) { return reinterpret_cast<uint64_t>(self.native_handle); });
    py::class_<BufferCreationInfo>(m, "BufferCreationInfo")
        .def(py::init<>())
        .def("element_stride", [](BufferCreationInfo &self) { return self.element_stride; })
        .def("size_bytes", [](BufferCreationInfo &self) { return self.total_size_bytes; })
        .def("handle", [](BufferCreationInfo &self) { return self.handle; })
        .def("native_handle", [](BufferCreationInfo &self) { return reinterpret_cast<uint64_t>(self.native_handle); });
    py::class_<Context>(m, "Context")
        .def(py::init<luisa::string>())
        .def("create_device", [](Context &self, luisa::string_view backend_name) {
            static UserBinaryIO io;
            DeviceConfig config{.binary_io = &io};
            return ManagedDevice(self.create_device(backend_name, &config));
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
        .def("emplace_back", [](ManagedAccel &accel, uint64_t vertex_buffer, size_t vertex_buffer_offset, size_t vertex_buffer_size, size_t vertex_stride, uint64_t triangle_buffer, size_t triangle_buffer_offset, size_t triangle_buffer_size, float4x4 transform, bool allow_compact, bool allow_update, int visibility_mask, bool opaque, uint user_id) {
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
            accel.emplace(cmd, transform, visibility_mask, opaque, user_id);
        })
        .def("emplace_procedural", [](ManagedAccel &accel, uint64_t aabb_buffer, size_t aabb_offset, size_t aabb_count, float4x4 transform, bool allow_compact, bool allow_update, int visibility_mask, bool opaque, uint user_id) {
            ProceduralUpdateCmd cmd;
            cmd.option = {.hint = AccelOption::UsageHint::FAST_TRACE,
                          .allow_compaction = allow_compact,
                          .allow_update = allow_update};
            cmd.aabb_buffer = aabb_buffer;
            cmd.aabb_offset = aabb_offset * sizeof(AABB);
            cmd.aabb_size = aabb_count * sizeof(AABB);
            accel.emplace(cmd, transform, visibility_mask, opaque, user_id);
        })
        .def("pop_back", [](ManagedAccel &accel) { accel.pop_back(); })
        .def("set", [](ManagedAccel &accel, size_t index, uint64_t vertex_buffer, size_t vertex_buffer_offset, size_t vertex_buffer_size, size_t vertex_stride, uint64_t triangle_buffer, size_t triangle_buffer_offset, size_t triangle_buffer_size, float4x4 transform, bool allow_compact, bool allow_update, int visibility_mask, bool opaque, uint user_id) {
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
            accel.set(index, cmd, transform, visibility_mask, opaque, user_id);
        })
        .def("set_procedural", [](ManagedAccel &accel, size_t index, uint64_t aabb_buffer, size_t aabb_offset, size_t aabb_count, float4x4 transform, bool allow_compact, bool allow_update, int visibility_mask, bool opaque, uint user_id) {
            ProceduralUpdateCmd cmd;
            cmd.option = {.hint = AccelOption::UsageHint::FAST_TRACE,
                          .allow_compaction = allow_compact,
                          .allow_update = allow_update};
            cmd.aabb_buffer = aabb_buffer;
            cmd.aabb_offset = aabb_offset * sizeof(AABB);
            cmd.aabb_size = aabb_count * sizeof(AABB);
            accel.set(index, cmd, transform, visibility_mask, opaque, user_id);
        })
        .def("set_transform_on_update", [](ManagedAccel &a, size_t index, float4x4 transform) { a.GetAccel().set_transform_on_update(index, transform); })
        .def("set_visibility_on_update", [](ManagedAccel &a, size_t index, int visibility_mask) { a.GetAccel().set_visibility_on_update(index, visibility_mask); })
        .def("set_user_id", [](ManagedAccel &a, size_t index, uint user_id) { a.GetAccel().set_instance_user_id_on_update(index, user_id); });
    py::class_<ManagedDevice>(m, "Device")
        .def(
            "create_stream", [](ManagedDevice &self, bool support_window) {
                PyStream stream(self.device, support_window);
                if (auto gs = default_stream_data.lock(); gs == nullptr) {
                    default_stream_data = stream.data();
                }
                return stream;
            })
        .def(
            "impl", [](ManagedDevice &s) { return s.device.impl(); }, pyref)
        .def("create_accel", [](ManagedDevice &device, AccelOption::UsageHint hint, bool allow_compact, bool allow_update) {
            return ManagedAccel(device.device.create_accel(AccelOption{
                .hint = hint,
                .allow_compaction = allow_compact,
                .allow_update = allow_update}));
        });
    m.def(
        "get_bindless_handle", [](uint64 handle) {
            return reinterpret_cast<ManagedBindless *>(handle)->GetHandle();
        },
        pyref);
    // py::class_<DeviceInterface::BuiltinBuffer>(m, "BuiltinBuffer")
    //     .def("handle", [](DeviceInterface::BuiltinBuffer &buffer) {
    //         return buffer.handle;
    //     })
    //     .def("size", [](DeviceInterface::BuiltinBuffer &buffer) {
    //         return buffer.size;
    //     });

    py::class_<DeviceInterface, luisa::shared_ptr<DeviceInterface>>(m, "DeviceInterface")
        .def("backend_name", [](DeviceInterface &self) {
            return self.backend_name();
        })
        .def(
            "create_shader", [](DeviceInterface &self, Function kernel) {
                auto handle = self.create_shader({}, kernel).handle;
                RefCounter::current->AddObject(handle, {[](DeviceInterface *d, uint64 handle) { d->destroy_shader(handle); }, &self});
                return handle;
            },
            pyref)// TODO: support metaoptions
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
            futures.emplace_back(
                luisa::fiber::async([str = luisa::string{str}, builder, &self]() {
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
            if (vertex.tag() != Function::Tag::RASTER_STAGE || pixel.tag() != Function::Tag::RASTER_STAGE) return 5;
            auto v2p = vertex.return_type();
            auto vert_args = vertex.arguments();
            if (vert_args.empty() || vert_args[0].type() != Type::of<VertexData>()) {
                return 6;
            }
            auto pixel_args = pixel.arguments();
            if (pixel_args.empty() || v2p != pixel_args[0].type()) {
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
        .def("save_raster_shader", [](DeviceInterface &self, Function vertex, Function pixel, luisa::string_view str) {
            ShaderOption option;
            option.compile_only = true;
            if (!output_path.empty()) {
                auto dst_path = output_path / std::filesystem::path{str};
                option.name = to_string(dst_path);
            } else {
                option.name = str;
            }
            static_cast<void>(static_cast<RasterExt *>(self.extension(RasterExt::name))
                                  ->create_raster_shader(vertex, pixel, option));
        })
        .def("save_raster_shader_async", [](DeviceInterface &self, luisa::shared_ptr<FunctionBuilder> const &vertex, luisa::shared_ptr<FunctionBuilder> const &pixel, luisa::string_view str) {
            thread_pool.create();
            futures.emplace_back(luisa::fiber::async([str = luisa::string{str}, vertex, pixel, &self]() {
                ShaderOption option;
                option.compile_only = true;
                if (!output_path.empty()) {
                    auto dst_path = output_path / std::filesystem::path{str};
                    option.name = to_string(dst_path);
                } else {
                    option.name = str;
                }
                static_cast<void>(static_cast<RasterExt *>(self.extension(RasterExt::name))
                                      ->create_raster_shader(vertex->function(), pixel->function(), option));
            }));
        })
        .def("destroy_shader", [](DeviceInterface &self, uint64_t handle) {
            RefCounter::current->DeRef(handle);
        })
        .def(
            "create_buffer", [](DeviceInterface &d, const Type *type, size_t size) {
                auto info = d.create_buffer(type, size, nullptr);
                RefCounter::current->AddObject(
                    info.handle,
                    {[](DeviceInterface *d, uint64 handle) {
                         if (auto gs = default_stream_data.lock()) {
                             gs->sync();
                         } else {
                             default_stream_data.reset();
                         }
                         d->destroy_buffer(handle);
                     },
                     &d});
                return info;
            },
            pyref)
        .def("import_external_buffer", [](DeviceInterface &d, const Type *type, uint64_t native_address, size_t elem_count) noexcept {
            auto info = d.create_buffer(type, elem_count, reinterpret_cast<void *>(native_address));
            RefCounter::current->AddObject(info.handle, {[](DeviceInterface *d, uint64 handle) {
                if (auto gs = default_stream_data.lock()) {
                    gs->sync();
                } else {
                    default_stream_data.reset();
                }
                 d->destroy_buffer(handle); }, &d});
            return info;
        })
        .def("create_dispatch_buffer", [](DeviceInterface &d, size_t size) {
            auto ptr = d.create_buffer(Type::of<IndirectKernelDispatch>(), size, nullptr);
            RefCounter::current->AddObject(ptr.handle, {[](DeviceInterface *d, uint64 handle) {
                if (auto gs = default_stream_data.lock()) {
                    gs->sync();
                } else {
                    default_stream_data.reset();
                }
                 d->destroy_buffer(handle); }, &d});
            return ptr;
        })
        .def("destroy_buffer", [](DeviceInterface &d, uint64_t handle) {
            RefCounter::current->DeRef(handle);
        })
        .def(
            "create_texture", [](DeviceInterface &d, PixelFormat format, uint32_t dimension, uint32_t width, uint32_t height, uint32_t depth, uint32_t mipmap_levels) {
                auto info = d.create_texture(format, dimension, width, height, depth, mipmap_levels, false, false);
                RefCounter::current->AddObject(info.handle, {[](DeviceInterface *d, uint64 handle) {
                    if (auto gs = default_stream_data.lock()) {
                        gs->sync();
                    } else {
                        default_stream_data.reset();
                    }
                    d->destroy_texture(handle); }, &d});
                return info;
            },
            pyref)
        .def("destroy_texture", [](DeviceInterface &d, uint64_t handle) {
            RefCounter::current->DeRef(handle);
        })
        .def(
            "create_bindless_array", [](DeviceInterface &d, size_t slots) {
                return reinterpret_cast<uint64>(new_with_allocator<ManagedBindless>(&d, slots));
            },
            pyref)// size
        .def("destroy_bindless_array", [](DeviceInterface &d, uint64 handle) {
            if (auto gs = default_stream_data.lock()) {
                gs->sync();
            } else {
                default_stream_data.reset();
            }
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
        .def("update_instance_buffer", [](PyStream &self, ManagedAccel &accel) {
            accel.update_instance_buffer(self);
        })
        .def("update_bindless", [](PyStream &self, uint64 bindless) {
            reinterpret_cast<ManagedBindless *>(bindless)->Update(self);
        })
        .def(
            "execute", [](PyStream &self) { self.execute(); }, pyref);

    m.def("builder", &FunctionBuilder::current, pyref);
    m.def("begin_analyzer", [](bool enabled) {
        analyzer.emplace_back(enabled ? luisa::make_optional<ASTEvaluator>() : luisa::nullopt);
    });
    m.def("end_analyzer", []() {
        analyzer.pop_back();
    });
    m.def("begin_branch", [](bool is_loop) {
        if (auto &&a = analyzer.back()) {
            a->begin_branch_scope(is_loop);
        }
    });
    m.def("end_branch", []() {
        if (auto &&a = analyzer.back()) {
            a->end_branch_scope();
        }
    });
    m.def("begin_switch", [](SwitchStmt const *stmt) {
        if (auto &&a = analyzer.back()) {
            a->begin_switch(stmt);
        }
    });
    m.def("end_switch", []() {
        if (auto &&a = analyzer.back()) {
            a->end_switch();
        }
    });
    m.def("analyze_condition", [](Expression const *expr) -> int32_t {
        ASTEvaluator::Result result;
        if (auto &&a = analyzer.back()) { result = a->try_eval(expr); }
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
        .def("argument_size", [](Function &func) { return func.arguments().size() - func.bound_arguments().size(); });
    py::class_<IntEval>(m, "IntEval")
        .def("value", [](IntEval &self) { return self.value; })
        .def("exist", [](IntEval &self) { return self.exist; });
    py::class_<CallableLibrary>(m, "CallableLibrary")
        .def(py::init<>())
        .def("add_callable", &CallableLibrary::add_callable)
        .def("serialize", [](CallableLibrary &self, luisa::string_view path) {
            auto vec = self.serialize();
            luisa::string path_str{path};
            auto f = fopen(path_str.c_str(), "wb");
            if (f) {
                fwrite(vec.data(), vec.size(), 1, f);
                LUISA_INFO("Save serialized callable with size: {} bytes.", vec.size());
                fclose(f);
            }
        })
        .def("load", [](CallableLibrary &self, luisa::string_view path) {
            BinaryFileStream file_stream{luisa::string{path}};
            luisa::vector<std::byte> vec;
            if (file_stream.valid()) {
                vec.push_back_uninitialized(file_stream.length());
                file_stream.read(vec);
            }
            self.load(vec);
        });
    py::class_<FunctionBuilder, luisa::shared_ptr<FunctionBuilder>>(m, "FunctionBuilder")
        .def("define_kernel", &FunctionBuilder::define_kernel<const luisa::function<void()> &>)
        .def("define_callable", &FunctionBuilder::define_callable<const luisa::function<void()> &>)
        .def("define_raster_stage", &FunctionBuilder::define_raster_stage<const luisa::function<void()> &>)
        .def("set_block_size", [](FunctionBuilder &self, uint32_t sx, uint32_t sy, uint32_t sz) { self.set_block_size(uint3(sx, sy, sz)); })
        .def("dimension", [](FunctionBuilder &self) {
            if (self.block_size().z > 1) {
                return 3;
            } else if (self.block_size().y > 1) {
                return 2;
            }
            return 1;
        })
        .def("try_eval_int", [](FunctionBuilder &self, Expression const *expr) {
            ASTEvaluator::Result eval;
            if (auto &&a = analyzer.back()) { eval = a->try_eval(expr); }
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
        .def("warp_lane_count", &FunctionBuilder::warp_lane_count, pyref)
        .def("warp_lane_id", &FunctionBuilder::warp_lane_id, pyref)
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

        .def("literal", [](
                FunctionBuilder &self,
                 const Type *type,
                  const LiteralExpr::Value::variant_type &value) {
                return luisa::visit(
                    [&self, type]<typename T>(T v) {
                        // we do not allow conversion between vector/matrix/bool types
                        if (type->is_vector() || type->is_matrix() ||
                            type == Type::of<bool>() || type == Type::of<T>()) {
                            return self.literal(type, v);
                        }
                        auto print_v = [&](){
                            if constexpr(std::is_same_v<std::decay_t<T>, half>){
                                return (float)v;
                            } else if constexpr(halfN<T>::value){
                                constexpr auto dim = halfN<T>::dimension;
                                if constexpr(dim == 2){
                                    return float2((float)v.x, (float)v.y);
                                }else if constexpr(dim == 3){
                                    return float3((float)v.x, (float)v.y, (float)v.z);
                                }else{
                                    return float4((float)v.x, (float)v.y, (float)v.z, (float)v.z);
                                }
                            }
                            else {
                                return v;
                            }
                        };
                        if constexpr (is_scalar_v<T>) {
                            // we are less strict here to allow implicit conversion
                            // between integral or between floating-point types,
                            // since python does not distinguish them
                            auto safe_convert = [v = print_v()]<typename U>(U /* for tagged dispatch */) noexcept {
                                auto u = static_cast<U>(v);
                                LUISA_ASSERT(static_cast<T>(u) == v,
                                             "Cannot convert literal value {} to type {}.",
                                             v, Type::of<U>()->description());
                                return u;
                            };
                            switch (type->tag()) {
                                case Type::Tag::INT16: return self.literal(type, safe_convert(short{}));
                                case Type::Tag::UINT16: return self.literal(type, safe_convert(luisa::ushort{}));
                                case Type::Tag::INT32: return self.literal(type, safe_convert(int{}));
                                case Type::Tag::UINT32: return self.literal(type, safe_convert(luisa::uint{}));
                                case Type::Tag::INT64: return self.literal(type, safe_convert(luisa::slong{}));
                                case Type::Tag::UINT64: return self.literal(type, safe_convert(luisa::ulong{}));
                                case Type::Tag::FLOAT16: return self.literal(type, static_cast<luisa::half>(v));
                                case Type::Tag::FLOAT32: return self.literal(type, static_cast<float>(v));
                                case Type::Tag::FLOAT64: return self.literal(type, static_cast<double>(v));
                                default: break;
                            }
                        }

                        LUISA_ERROR_WITH_LOCATION(
                            "Cannot convert literal value {} to type {}.",
                            print_v(), type->description());
                    },
                    LiteralExpr::Value{value});
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
                if (auto &&a = analyzer.back()) { a->check_call_ref(custom, args); }
                return self.call(type, custom, std::move(args));
            },
            pyref)
        .def("call", [](FunctionBuilder &self, CallOp call_op, const luisa::vector<const Expression *> &args) { self.call(call_op, std::move(args)); })
        .def("call", [](FunctionBuilder &self, Function custom, const luisa::vector<const Expression *> &args) {
            if (auto &&a = analyzer.back()) { a->check_call_ref(custom, args); }
            self.call(custom, std::move(args));
        })

        .def("break_", &FunctionBuilder::break_)
        .def("continue_", &FunctionBuilder::continue_)
        .def("return_", &FunctionBuilder::return_)
        .def(
            "assign", [](FunctionBuilder &self, Expression const *l, Expression const *r) {
                ASTEvaluator::Result result;
                if (auto &&a = analyzer.back()) { result = a->assign(l, r); }
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
        .def("ray_query_", &FunctionBuilder::ray_query_, pyref)
        .def("case_", &FunctionBuilder::case_, pyref)
        .def("loop_", &FunctionBuilder::loop_, pyref)
        // .def("switch_")
        // .def("case_")
        .def("default_", &FunctionBuilder::default_, pyref)
        .def(
            "for_", [](FunctionBuilder &self, const Expression *var, const Expression *condition, const Expression *update) {
                auto ptr = self.for_(var, condition, update);
                if (auto &&a = analyzer.back()) { a->execute_for(ptr); }
                return ptr;
            },
            pyref)
        .def("autodiff_", &FunctionBuilder::autodiff_, pyref)
        .def(
            "print_", [](FunctionBuilder &self, luisa::string_view format, const luisa::vector<const Expression *> &args) {
                self.print_(luisa::string{format}, args);
            },
            pyref)
        // .def("meta") // unused
        .def("function", &FunctionBuilder::function);// returning object

    py::class_<AtomicAccessChain>(m, "AtomicAccessChain")
        .def(py::init<>())
        .def(
            "create", [&](AtomicAccessChain &self, RefExpr const *buffer_expr) {
                LUISA_ASSERT(self.node == nullptr, "Re-create chain not allowed");
                self.node = AtomicAccessChain::Node::create(buffer_expr);
            },
            pyref)
        .def(
            "access", [&](AtomicAccessChain &self, Expression const *expr) {
                self.node = self.node->access(expr);
            },
            pyref)
        .def(
            "member", [&](AtomicAccessChain &self, size_t member_index) {
                self.node = self.node->access(member_index);
            },
            pyref)
        .def(
            "operate", [&](AtomicAccessChain &self, CallOp op, const luisa::vector<const Expression *> &args) -> Expression const * {
                return self.node->operate(op, luisa::span<Expression const *const>{args});
            },
            pyref);
}
