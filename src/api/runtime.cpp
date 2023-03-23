//
// Created by Mike Smith on 2021/10/17.
//

#include <luisa-compute.h>
#include <ast/function_builder.h>
#include <api/runtime.h>
#include <runtime/rhi/resource.h>
#include <rust/luisa_compute_api_types/bindings.hpp>
#include <utility>
// #include <ir/ir.hpp>

#define LC_RC_TOMBSTONE 0xdeadbeef

template<class T>
struct RC {

    T *_object;
    std::atomic_uint64_t _ref_count;
    std::function<void(T *)> _deleter;
    uint32_t tombstone;

    RC(T *object, std::function<void(T *)> deleter)
        : _object{object},
          _deleter{deleter},
          _ref_count{1} { tombstone = 0; }

    ~RC() { _deleter(_object); }

    void check() const {
        if (tombstone == LC_RC_TOMBSTONE) {
            LUISA_ERROR_WITH_LOCATION("Object has been destroyed");
        }
    }

    RC *retain() {
        check();
        _ref_count.fetch_add(1, std::memory_order_acquire);
        return this;
    }

    void release() {
        check();
        if (_ref_count.fetch_sub(1, std::memory_order_release) == 0) {
            std::atomic_thread_fence(std::memory_order_acquire);
            tombstone = LC_RC_TOMBSTONE;
            delete this;
        }
    }

    T *object() {
        check();
        return _object;
    }
};

namespace luisa::compute {

struct BufferResource final : public Resource {
    BufferResource(DeviceInterface *device, const ir::CArc<ir::Type> *element, size_t elem_count) noexcept
        : Resource{device, Tag::BUFFER,
                   device->create_buffer(element, elem_count)} {}
};

struct TextureResource final : public Resource {
    TextureResource(DeviceInterface *device,
                    PixelFormat format, uint dimension,
                    uint width, uint height, uint depth,
                    uint mipmap_levels) noexcept
        : Resource{device, Tag::TEXTURE,
                   device->create_texture(format, dimension,
                                          width, height, depth,
                                          mipmap_levels)} {}
};

struct ShaderResource : public Resource {
    ShaderResource(DeviceInterface *device,
                   Function function,
                   const api::ShaderOption &option) noexcept
        : Resource{
              device,
              Tag::SHADER,
              device->create_shader(
                  ShaderOption{
                      .enable_cache = option.enable_cache,
                      .enable_fast_math = option.enable_fast_math,
                      .enable_debug_info = option.enable_debug_info,
                      .compile_only = option.compile_only,
                      .name = luisa::string{option.name}},
                  function)} {}
};

}// namespace luisa::compute

// TODO: rewrite with runtime constructs, e.g., Stream, Event, BindlessArray...

using namespace luisa;
using namespace luisa::compute;

Sampler convert_sampler(api::Sampler sampler);

namespace luisa::compute::detail {

class CommandListConverter {

private:
    api::CommandList _list;
    bool _is_c_api;

    luisa::unique_ptr<Command> convert_one(api::Command cmd) const noexcept {
        auto convert_pixel_storage = [](const api::PixelStorage &storage) noexcept {
            return PixelStorage{(uint32_t)to_underlying(storage)};
        };
        auto convert_uint3 = [](uint32_t array[3]) noexcept {
            return luisa::uint3{array[0], array[1], array[2]};
        };
        auto convert_accel_request = [](api::AccelBuildRequest req) noexcept {
            return AccelBuildRequest{(uint32_t)to_underlying(req)};
        };
        switch (cmd.tag) {
            case api::Command::Tag::BUFFER_COPY: {
                auto c = cmd.BUFFER_COPY._0;
                return make_unique<BufferCopyCommand>(
                    c.src._0, c.dst._0,
                    c.src_offset, c.dst_offset, c.size);
            }
            case api::Command::Tag::BUFFER_UPLOAD: {
                auto c = cmd.BUFFER_UPLOAD._0;
                return make_unique<BufferUploadCommand>(
                    c.buffer._0, c.offset,
                    c.size, c.data);
            }
            case api::Command::Tag::BUFFER_DOWNLOAD: {
                auto c = cmd.BUFFER_DOWNLOAD._0;
                return make_unique<BufferDownloadCommand>(
                    c.buffer._0, c.offset,
                    c.size, c.data);
            }
            case api::Command::Tag::BINDLESS_ARRAY_UPDATE: {
                auto c = cmd.BINDLESS_ARRAY_UPDATE._0;
                auto modifications = luisa::vector<BindlessArrayUpdateCommand::Modification>{};
                for (auto i = 0u; i < c.modifications_count; i++) {
                    auto &&[slot, api_buffer, api_tex2d, api_tex3d] = c.modifications[i];
                    auto convert_op = [](api::BindlessArrayUpdateOperation op) noexcept {
                        return BindlessArrayUpdateCommand::Modification::Operation{(uint)to_underlying(op)};
                    };
                    auto buffer = BindlessArrayUpdateCommand::Modification::Buffer(
                        api_buffer.handle._0,
                        api_buffer.offset,
                        convert_op(api_buffer.op));
                    auto tex2d = BindlessArrayUpdateCommand::Modification::Texture(
                        api_tex2d.handle._0,
                        convert_sampler(api_tex2d.sampler),
                        convert_op(api_tex2d.op));
                    auto tex3d = BindlessArrayUpdateCommand::Modification::Texture(
                        api_tex3d.handle._0,
                        convert_sampler(api_tex3d.sampler),
                        convert_op(api_tex3d.op));
                    modifications.emplace_back(slot, buffer, tex2d, tex3d);
                }// Manually conversion is painful...
                return make_unique<BindlessArrayUpdateCommand>(
                    c.handle._0,
                    modifications);
            }
            case api::Command::Tag::SHADER_DISPATCH: {
                auto c = cmd.SHADER_DISPATCH._0;
                auto first = (std::byte *)c.args;
                auto buffer = luisa::vector<std::byte>(first, first + c.args_count);
                auto dispatch_size = luisa::uint3{c.dispatch_size[0], c.dispatch_size[1], c.dispatch_size[2]};

                return make_unique<ShaderDispatchCommand>(
                    c.shader._0,
                    std::move(buffer),
                    c.args_count,
                    ShaderDispatchCommand::DispatchSize{dispatch_size}
                );
            }
            case api::Command::Tag::BUFFER_TO_TEXTURE_COPY: {
                auto [buffer, buffer_offset, texture, storage, texture_level, texture_size] = cmd.BUFFER_TO_TEXTURE_COPY._0;

                return make_unique<BufferToTextureCopyCommand>(
                    buffer._0,
                    buffer_offset,
                    texture._0,
                    convert_pixel_storage(storage),
                    texture_level,
                    convert_uint3(texture_size));
            }
            case api::Command::Tag::TEXTURE_TO_BUFFER_COPY: {
                auto [buffer, buffer_offset, texture, storage, texture_level, texture_size] = cmd.TEXTURE_TO_BUFFER_COPY._0;
                return make_unique<TextureToBufferCopyCommand>(
                    buffer._0,
                    buffer_offset,
                    texture._0,
                    convert_pixel_storage(storage),
                    texture_level,
                    convert_uint3(texture_size));
            }
            case api::Command::Tag::TEXTURE_UPLOAD: {
                auto [texture, storage, level, size, data] = cmd.TEXTURE_UPLOAD._0;
                return make_unique<TextureUploadCommand>(
                    texture._0,
                    convert_pixel_storage(storage),
                    level,
                    convert_uint3(size),
                    (const void *)data);
            }
            case api::Command::Tag::TEXTURE_DOWNLOAD: {
                auto [texture, storage, level, size, data] = cmd.TEXTURE_DOWNLOAD._0;
                return make_unique<TextureDownloadCommand>(
                    texture._0,
                    convert_pixel_storage(storage),
                    level,
                    convert_uint3(size),
                    (void *)data);
            }
            case api::Command::Tag::TEXTURE_COPY: {
                auto [storage, src, dst, size, src_level, dst_level] = cmd.TEXTURE_COPY._0;
                return make_unique<TextureCopyCommand>(
                    convert_pixel_storage(storage),
                    src._0,
                    dst._0,
                    src_level,
                    dst_level,
                    convert_uint3(size));
            }
            case api::Command::Tag::MESH_BUILD: {
                auto [mesh, request,
                      vertex_buffer, vertex_buffer_offset, vertex_buffer_size, vertex_stride,
                      index_buffer, index_buffer_offset, index_buffer_size, index_stride] = cmd.MESH_BUILD._0;
                LUISA_ASSERT(index_stride == 12, "Index stride must be 12.");
                return make_unique<MeshBuildCommand>(
                    mesh._0,
                    convert_accel_request(request),
                    vertex_buffer._0, vertex_buffer_offset, vertex_buffer_size, vertex_stride,
                    index_buffer._0, index_buffer_offset, index_buffer_size);
            }
            case api::Command::Tag::PROCEDURAL_PRIMITIVE_BUILD: {
                auto [primitive, request, aabb_buffer, aabb_offset, aabb_count] = cmd.PROCEDURAL_PRIMITIVE_BUILD._0;
                return make_unique<ProceduralPrimitiveBuildCommand>(
                    primitive._0,
                    convert_accel_request(request),
                    aabb_buffer._0,
                    aabb_offset,
                    aabb_count);
            }
            case api::Command::Tag::ACCEL_BUILD: {
                auto [accel, request, instance_count, api_modifications, modifications_count, build_accel] = cmd.ACCEL_BUILD._0;
                auto modifications = luisa::vector<AccelBuildCommand::Modification>(modifications_count);
                for (auto i = 0u; i < modifications_count; i++) {
                    auto [index, flags, visibility, mesh, affine] = api_modifications[i];
                    auto modification = AccelBuildCommand::Modification(index);
                    modification.flags = flags.bits;
                    modification.set_visibility(visibility);
                    modification.set_primitive(mesh);
                    std::memcpy(modification.affine, affine, sizeof(float) * 12);

                    modifications.emplace_back(std::move(modification));
                }
                return make_unique<AccelBuildCommand>(
                    accel._0,
                    instance_count,
                    convert_accel_request(request),
                    modifications,
                    build_accel);
            }
            default: LUISA_ERROR_WITH_LOCATION("unimplemented command {}", (int)cmd.tag);
        }
    }

public:
    CommandListConverter(const api::CommandList list, bool is_c_api)
        : _list(list), _is_c_api(is_c_api) {}

    [[nodiscard]] auto convert() const noexcept {
        auto cmd_list = CommandList();
        for (int i = 0; i < _list.commands_count; i++) {
            cmd_list.append(convert_one(_list.commands[i]));
        }
        return cmd_list;
    }
};

}// namespace luisa::compute::detail

template<class T>
T from_ptr(void *ptr) {
    return T{
        ._0 = reinterpret_cast<uint64_t>(ptr)};
}

LUISA_EXPORT_API api::Context luisa_compute_context_create(const char *exe_path) LUISA_NOEXCEPT {
    return from_ptr<api::Context>(new_with_allocator<Context>(exe_path));
}

LUISA_EXPORT_API void luisa_compute_context_destroy(api::Context ctx) LUISA_NOEXCEPT {
    delete_with_allocator(reinterpret_cast<Context *>(ctx._0));
}

inline char *path_to_c_str(const std::filesystem::path &path) LUISA_NOEXCEPT {
    auto s = path.string();
    auto cs = static_cast<char *>(malloc(s.size() + 1u));
    memcpy(cs, s.c_str(), s.size() + 1u);
    return cs;
}

LUISA_EXPORT_API void luisa_compute_free_c_string(char *cs) LUISA_NOEXCEPT { free(cs); }

LUISA_EXPORT_API char *luisa_compute_context_runtime_directory(api::Context ctx) LUISA_NOEXCEPT {
    return path_to_c_str(reinterpret_cast<Context *>(ctx._0)->paths().runtime_directory());
}

LUISA_EXPORT_API char *luisa_compute_context_cache_directory(api::Context ctx) LUISA_NOEXCEPT {
    return path_to_c_str(reinterpret_cast<Context *>(ctx._0)->paths().cache_directory());
}

LUISA_EXPORT_API api::Device luisa_compute_device_create(api::Context ctx,
                                                         const char *name,
                                                         const char *properties) LUISA_NOEXCEPT {
    // TODO: handle properties? or convert it to DeviceConfig?
    auto device = new Device(std::move(reinterpret_cast<Context *>(ctx._0)->create_device(name, nullptr)));
    return from_ptr<api::Device>(new_with_allocator<RC<Device>>(
        device, [](Device *d) { delete d; }));
}

LUISA_EXPORT_API void luisa_compute_device_destroy(api::Device device) LUISA_NOEXCEPT {
    reinterpret_cast<RC<Device> *>(device._0)->release();
}

LUISA_EXPORT_API void luisa_compute_device_retain(api::Device device) LUISA_NOEXCEPT {
    reinterpret_cast<RC<Device> *>(device._0)->retain();
}

LUISA_EXPORT_API void luisa_compute_device_release(api::Device device) LUISA_NOEXCEPT {
    reinterpret_cast<RC<Device> *>(device._0)->release();
}

LUISA_EXPORT_API void *luisa_compute_device_native_handle(api::Device device) LUISA_NOEXCEPT {
    return reinterpret_cast<RC<Device> *>(device._0)->object()->impl()->native_handle();
}

LUISA_EXPORT_API api::CreatedBufferInfo luisa_compute_buffer_create(api::Device device, const ir::CArc<ir::Type> *element, size_t elem_count) LUISA_NOEXCEPT {
    auto d = reinterpret_cast<RC<Device> *>(device._0);
    auto info = d->retain()->object()->impl()->create_buffer(element, elem_count);
    return api::CreatedBufferInfo{
        .resource = api::CreatedResourceInfo{
            .handle = info.handle,
            .native_handle = info.native_handle,
        },
        .element_stride = info.element_stride,
        .total_size_bytes = info.total_size_bytes,
    };
}

LUISA_EXPORT_API void luisa_compute_buffer_destroy(api::Device device, api::Buffer buffer) LUISA_NOEXCEPT {
    auto handle = buffer._0;
    auto d = reinterpret_cast<RC<Device> *>(device._0);
    d->object()->impl()->destroy_buffer(handle);
    d->release();
}

LUISA_EXPORT_API api::CreatedResourceInfo luisa_compute_texture_create(api::Device device,
                                                                       api::PixelFormat format, uint32_t dim,
                                                                       uint32_t w, uint32_t h, uint32_t d,
                                                                       uint32_t mips) LUISA_NOEXCEPT {
    auto dev = reinterpret_cast<RC<Device> *>(device._0);
    auto pixel_format = PixelFormat{(uint8_t)to_underlying(format)};
    auto info = dev->retain()->object()->impl()->create_texture(pixel_format, dim, w, h, d, mips);
    return api::CreatedResourceInfo{
        .handle = info.handle,
        .native_handle = info.native_handle,
    };
}

LUISA_EXPORT_API void luisa_compute_texture_destroy(api::Device device, api::Texture texture) LUISA_NOEXCEPT {
    auto handle = texture._0;
    auto d = reinterpret_cast<RC<Device> *>(device._0);
    d->object()->impl()->destroy_texture(handle);
    d->release();
}

LUISA_EXPORT_API api::CreatedResourceInfo luisa_compute_stream_create(api::Device device, api::StreamTag stream_tag) LUISA_NOEXCEPT {
    auto d = reinterpret_cast<RC<Device> *>(device._0);
    auto tag = StreamTag{(uint8_t)to_underlying(stream_tag)};
    auto info = d->retain()->object()->impl()->create_stream(tag);
    return api::CreatedResourceInfo{
        .handle = info.handle,
        .native_handle = info.native_handle,
    };
}

LUISA_EXPORT_API void luisa_compute_stream_destroy(api::Device device, api::Stream stream) LUISA_NOEXCEPT {
    auto handle = stream._0;
    auto d = reinterpret_cast<RC<Device> *>(device._0);
    d->object()->impl()->destroy_stream(handle);
    d->release();
}

LUISA_EXPORT_API void luisa_compute_stream_synchronize(api::Device device, api::Stream stream) LUISA_NOEXCEPT {
    auto handle = stream._0;
    auto d = reinterpret_cast<RC<Device> *>(device._0);
    d->object()->impl()->synchronize_stream(handle);
}

LUISA_EXPORT_API void luisa_compute_stream_dispatch(api::Device device, api::Stream stream, api::CommandList c_cmd_list) LUISA_NOEXCEPT {
    auto handle = stream._0;
    auto d = reinterpret_cast<RC<Device> *>(device._0);
    auto converter = luisa::compute::detail::CommandListConverter(c_cmd_list, d->object()->impl()->is_c_api());
    d->object()->impl()->dispatch(handle, converter.convert());
}

LUISA_EXPORT_API api::CreatedShaderInfo luisa_compute_shader_create(api::Device device, api::KernelModule m, const api::ShaderOption &option) LUISA_NOEXCEPT {
    auto d = reinterpret_cast<RC<Device> *>(device._0);
    auto ir = reinterpret_cast<const ir::KernelModule *>(m.ptr);

    auto shader_option = ShaderOption{
        .enable_cache = option.enable_cache,
        .enable_fast_math = option.enable_fast_math,
        .enable_debug_info = option.enable_debug_info,
        .compile_only = option.compile_only,
        .name = luisa::string{option.name}};

    auto info = d->retain()->object()->impl()->create_shader(shader_option, ir);
    return api::CreatedShaderInfo{
        .resource = api::CreatedResourceInfo{
            .handle = info.handle,
            .native_handle = info.native_handle,
        },
        .block_size = {info.block_size[0], info.block_size[1], info.block_size[2]},
    };
}

LUISA_EXPORT_API void luisa_compute_shader_destroy(api::Device device, api::Shader shader) LUISA_NOEXCEPT {
    auto handle = shader._0;
    auto d = reinterpret_cast<RC<Device> *>(device._0);
    d->object()->impl()->destroy_shader(handle);
    d->release();
}

LUISA_EXPORT_API api::CreatedResourceInfo luisa_compute_event_create(api::Device device) LUISA_NOEXCEPT {
    auto d = reinterpret_cast<RC<Device> *>(device._0);
    auto info = d->retain()->object()->impl()->create_event();
    return api::CreatedResourceInfo{
        .handle = info.handle,
        .native_handle = info.native_handle,
    };
}

LUISA_EXPORT_API void luisa_compute_event_destroy(api::Device device, api::Event event) LUISA_NOEXCEPT {
    auto handle = event._0;
    auto d = reinterpret_cast<RC<Device> *>(device._0);
    d->object()->impl()->destroy_event(handle);
    d->release();
}

LUISA_EXPORT_API void luisa_compute_event_signal(api::Device device, api::Event event, api::Stream stream) LUISA_NOEXCEPT {
    auto handle = event._0;
    auto stream_handle = stream._0;
    auto d = reinterpret_cast<RC<Device> *>(device._0);
    d->object()->impl()->signal_event(handle, stream_handle);
}

LUISA_EXPORT_API void luisa_compute_event_wait(api::Device device, api::Event event, api::Stream stream) LUISA_NOEXCEPT {
    auto handle = event._0;
    auto stream_handle = stream._0;
    auto d = reinterpret_cast<RC<Device> *>(device._0);
    d->object()->impl()->wait_event(handle, stream_handle);
}

LUISA_EXPORT_API void luisa_compute_event_synchronize(api::Device device, api::Event event) LUISA_NOEXCEPT {
    auto handle = event._0;
    auto d = reinterpret_cast<RC<Device> *>(device._0);
    d->object()->impl()->synchronize_event(handle);
}

LUISA_EXPORT_API api::CreatedResourceInfo luisa_compute_mesh_create(api::Device device, const api::AccelOption &option) LUISA_NOEXCEPT {
    auto d = reinterpret_cast<RC<Device> *>(device._0);
    auto accel_option = AccelOption{
        .hint = AccelOption::UsageHint{(uint32_t)to_underlying(option.hint)},
        .allow_compaction = option.allow_compaction,
        .allow_update = option.allow_update,
    };
    auto info = d->retain()->object()->impl()->create_mesh(accel_option);
    return api::CreatedResourceInfo{
        .handle = info.handle,
        .native_handle = info.native_handle,
    };
}

LUISA_EXPORT_API void luisa_compute_mesh_destroy(api::Device device, api::Mesh mesh) LUISA_NOEXCEPT {
    auto handle = mesh._0;
    auto d = reinterpret_cast<RC<Device> *>(device._0);
    d->object()->impl()->destroy_mesh(handle);
    d->release();
}

LUISA_EXPORT_API api::CreatedResourceInfo luisa_compute_accel_create(api::Device device, const api::AccelOption &option) LUISA_NOEXCEPT {
    auto d = reinterpret_cast<RC<Device> *>(device._0);
    auto accel_option = AccelOption{
        .hint = AccelOption::UsageHint{(uint32_t)to_underlying(option.hint)},
        .allow_compaction = option.allow_compaction,
        .allow_update = option.allow_update,
    };
    auto info = d->retain()->object()->impl()->create_accel(accel_option);
    return api::CreatedResourceInfo{
        .handle = info.handle,
        .native_handle = info.native_handle,
    };
}

LUISA_EXPORT_API void luisa_compute_accel_destroy(api::Device device, api::Accel accel) LUISA_NOEXCEPT {
    auto handle = accel._0;
    auto d = reinterpret_cast<RC<Device> *>(device._0);
    d->object()->impl()->destroy_accel(handle);
    d->release();
}

// api::Command luisa_compute_command_update_mesh(uint64_t handle) LUISA_NOEXCEPT {
//     return (api::Command)MeshUpdateCommand::create(handle);
// }

// api::Command luisa_compute_command_update_accel(uint64_t handle) LUISA_NOEXCEPT {
//     return (api::Command)AccelUpdateCommand::create(handle);
// }

LUISA_EXPORT_API api::PixelStorage luisa_compute_pixel_format_to_storage(api::PixelFormat format) LUISA_NOEXCEPT {
    return static_cast<api::PixelStorage>(
        to_underlying(pixel_format_to_storage(static_cast<PixelFormat>(format))));
}

inline Sampler convert_sampler(api::Sampler sampler) {
    auto address = [&sampler]() noexcept -> Sampler::Address {
        switch (sampler.address) {
            case api::SamplerAddress::ZERO: return Sampler::Address::ZERO;
            case api::SamplerAddress::EDGE: return Sampler::Address::EDGE;
            case api::SamplerAddress::REPEAT: return Sampler::Address::REPEAT;
            case api::SamplerAddress::MIRROR: return Sampler::Address::MIRROR;
            default: LUISA_ERROR_WITH_LOCATION("Invalid sampler address mode {}", static_cast<int>(sampler.address));
        }
    }();
    auto filter = [&sampler]() noexcept -> Sampler::Filter {
        switch (sampler.filter) {
            case api::SamplerFilter::POINT: return Sampler::Filter::POINT;
            case api::SamplerFilter::LINEAR_LINEAR: return Sampler::Filter::LINEAR_LINEAR;
            case api::SamplerFilter::ANISOTROPIC: return Sampler::Filter::ANISOTROPIC;
            case api::SamplerFilter::LINEAR_POINT: return Sampler::Filter::LINEAR_POINT;
            default: LUISA_ERROR_WITH_LOCATION("Invalid sampler filter mode {}", static_cast<int>(sampler.filter));
        }
    }();
    return Sampler(filter, address);
}

LUISA_EXPORT_API api::CreatedResourceInfo luisa_compute_bindless_array_create(api::Device device, size_t n) LUISA_NOEXCEPT {
    auto d = reinterpret_cast<RC<Device> *>(device._0);
    auto info = d->retain()->object()->impl()->create_bindless_array(n);
    return api::CreatedResourceInfo{
        .handle = info.handle,
        .native_handle = info.native_handle,
    };
}

LUISA_EXPORT_API void luisa_compute_bindless_array_destroy(api::Device device, api::BindlessArray array) LUISA_NOEXCEPT {
    auto handle = array._0;
    auto d = reinterpret_cast<RC<Device> *>(device._0);
    d->object()->impl()->destroy_bindless_array(handle);
    d->release();
}