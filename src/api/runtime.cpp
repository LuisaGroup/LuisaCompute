#include <luisa/core/logging.h>
#include <luisa/runtime/device.h>
#include <luisa/runtime/context.h>
#include <luisa/runtime/rhi/command_encoder.h>
#include <luisa/runtime/rtx/triangle.h>
#include <luisa/runtime/rtx/aabb.h>
#include <luisa/api/api.h>

#include <utility>

#define LUISA_RC_TOMBSTONE 0xdeadbeef

// TODO: rewrite with runtime constructs, e.g., Stream, Event, BindlessArray...

using namespace luisa;
using namespace luisa::compute;

[[nodiscard]] Sampler convert_sampler(LCSampler sampler);

namespace luisa::compute::detail {

class CommandListConverter {

private:
    LCCommandList _list;

    static luisa::unique_ptr<Command> convert_one(LCCommand cmd) noexcept {
        auto convert_pixel_storage = [](const LCPixelStorage &storage) noexcept {
            return PixelStorage{(uint32_t)to_underlying(storage)};
        };
        auto convert_uint3 = [](uint32_t array[3]) noexcept {
            return luisa::uint3{array[0], array[1], array[2]};
        };
        auto convert_accel_request = [](LCAccelBuildRequest req) noexcept {
            switch (req) {
                case LC_ACCEL_BUILD_REQUEST_PREFER_UPDATE:
                    return AccelBuildRequest::PREFER_UPDATE;
                case LC_ACCEL_BUILD_REQUEST_FORCE_BUILD:
                    return AccelBuildRequest::FORCE_BUILD;
                default: break;
            }
            LUISA_ERROR_WITH_LOCATION("Unreachable.");
        };
        switch (cmd.tag) {
            case LC_COMMAND_BUFFER_COPY: {
                auto c = cmd.buffer_copy;
                return luisa::make_unique<BufferCopyCommand>(
                    c.src._0, c.dst._0,
                    c.src_offset, c.dst_offset, c.size);
            }
            case LC_COMMAND_BUFFER_UPLOAD: {
                auto c = cmd.buffer_upload;
                return luisa::make_unique<BufferUploadCommand>(
                    c.buffer._0, c.offset,
                    c.size, c.data);
            }
            case LC_COMMAND_BUFFER_DOWNLOAD: {
                auto c = cmd.buffer_download;
                return luisa::make_unique<BufferDownloadCommand>(
                    c.buffer._0, c.offset,
                    c.size, c.data);
            }
            case LC_COMMAND_BINDLESS_ARRAY_UPDATE: {
                auto c = cmd.bindless_array_update;
                auto modifications = luisa::vector<BindlessArrayUpdateCommand::Modification>{};
                modifications.reserve(c.modifications_count);
                for (auto i = 0u; i < c.modifications_count; i++) {
                    auto &&[slot, api_buffer, api_tex2d, api_tex3d] = c.modifications[i];
                    auto convert_op = [](LCBindlessArrayUpdateOperation op) noexcept {
                        switch (op) {
                            case LC_BINDLESS_ARRAY_UPDATE_OPERATION_NONE:
                                return BindlessArrayUpdateCommand::Modification::Operation::NONE;
                            case LC_BINDLESS_ARRAY_UPDATE_OPERATION_EMPLACE:
                                return BindlessArrayUpdateCommand::Modification::Operation::EMPLACE;
                            case LC_BINDLESS_ARRAY_UPDATE_OPERATION_REMOVE:
                                return BindlessArrayUpdateCommand::Modification::Operation::REMOVE;
                            default: break;
                        }
                        // unreachable
                        return BindlessArrayUpdateCommand::Modification::Operation::NONE;
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
                return luisa::make_unique<BindlessArrayUpdateCommand>(
                    c.handle._0, std::move(modifications));
            }
            case LC_COMMAND_SHADER_DISPATCH: {
                auto c = cmd.shader_dispatch;
                auto uniform_size = 0ull;
                for (auto i = 0u; i < c.args_count; i++) {
                    if (c.args[i].tag == LC_ARGUMENT_UNIFORM) {
                        uniform_size += c.args[i].uniform.size;
                    }
                }
                ComputeDispatchCmdEncoder encoder{c.shader._0, c.args_count, uniform_size};
                for (auto i = 0u; i < c.args_count; i++) {
                    auto arg = c.args[i];
                    switch (arg.tag) {
                        case LC_ARGUMENT_BUFFER:
                            encoder.encode_buffer(arg.buffer.buffer._0,
                                                  arg.buffer.offset,
                                                  arg.buffer.size);
                            break;
                        case LC_ARGUMENT_TEXTURE:
                            encoder.encode_texture(arg.texture.texture._0,
                                                   arg.texture.level);
                            break;
                        case LC_ARGUMENT_UNIFORM:
                            encoder.encode_uniform(arg.uniform.data,
                                                   arg.uniform.size);
                            break;
                        case LC_ARGUMENT_BINDLESS_ARRAY:
                            encoder.encode_bindless_array(arg.bindless_array._0);
                            break;
                        case LC_ARGUMENT_ACCEL:
                            encoder.encode_accel(arg.accel._0);
                            break;
                    }
                }
                encoder.set_dispatch_size(make_uint3(c.dispatch_size[0],
                                                     c.dispatch_size[1],
                                                     c.dispatch_size[2]));
                return std::move(encoder).build();
            }
            case LC_COMMAND_BUFFER_TO_TEXTURE_COPY: {
                auto [buffer, buffer_offset, texture, storage, texture_level, texture_size] = cmd.buffer_to_texture_copy;

                return luisa::make_unique<BufferToTextureCopyCommand>(
                    buffer._0,
                    buffer_offset,
                    texture._0,
                    convert_pixel_storage(storage),
                    texture_level,
                    convert_uint3(texture_size));
            }
            case LC_COMMAND_TEXTURE_TO_BUFFER_COPY: {
                auto [buffer, buffer_offset, texture, storage, texture_level, texture_size] = cmd.texture_to_buffer_copy;
                return luisa::make_unique<TextureToBufferCopyCommand>(
                    buffer._0,
                    buffer_offset,
                    texture._0,
                    convert_pixel_storage(storage),
                    texture_level,
                    convert_uint3(texture_size));
            }
            case LC_COMMAND_TEXTURE_UPLOAD: {
                auto [texture, storage, level, size, data] = cmd.texture_upload;
                return luisa::make_unique<TextureUploadCommand>(
                    texture._0,
                    convert_pixel_storage(storage),
                    level,
                    convert_uint3(size),
                    (const void *)data);
            }
            case LC_COMMAND_TEXTURE_DOWNLOAD: {
                auto [texture, storage, level, size, data] = cmd.texture_download;
                return luisa::make_unique<TextureDownloadCommand>(
                    texture._0,
                    convert_pixel_storage(storage),
                    level,
                    convert_uint3(size),
                    (void *)data);
            }
            case LC_COMMAND_TEXTURE_COPY: {
                auto [storage, src, dst, size, src_level, dst_level] = cmd.texture_copy;
                return luisa::make_unique<TextureCopyCommand>(
                    convert_pixel_storage(storage),
                    src._0,
                    dst._0,
                    src_level,
                    dst_level,
                    convert_uint3(size));
            }
            case LC_COMMAND_MESH_BUILD: {
                auto [mesh, request,
                      vertex_buffer, vertex_buffer_offset, vertex_buffer_size, vertex_stride,
                      index_buffer, index_buffer_offset, index_buffer_size, index_stride] = cmd.mesh_build;
                LUISA_ASSERT(index_stride == sizeof(Triangle),
                             "Index stride must be {} (got {}).",
                             sizeof(Triangle), index_stride);
                return luisa::make_unique<MeshBuildCommand>(
                    mesh._0,
                    convert_accel_request(request),
                    vertex_buffer._0, vertex_buffer_offset, vertex_buffer_size, vertex_stride,
                    index_buffer._0, index_buffer_offset, index_buffer_size);
            }
            case LC_COMMAND_PROCEDURAL_PRIMITIVE_BUILD: {
                auto [primitive, request, aabb_buffer, aabb_offset, aabb_count] = cmd.procedural_primitive_build;
                return luisa::make_unique<ProceduralPrimitiveBuildCommand>(
                    primitive._0,
                    convert_accel_request(request),
                    aabb_buffer._0,
                    aabb_offset,
                    aabb_count * sizeof(AABB));
            }
            case LC_COMMAND_ACCEL_BUILD: {
                auto b = cmd.accel_build;
                auto modifications = luisa::vector<AccelBuildCommand::Modification>{};
                modifications.reserve(b.modifications_count);
                for (auto i = 0u; i < b.modifications_count; i++) {
                    auto m = b.modifications[i];
                    AccelBuildCommand::Modification modification{m.index};
#define LUISA_DEPARANTHESES_IMPL(...) __VA_ARGS__
#define LUISA_DEPARANTHESES(...) LUISA_DEPARANTHESES_IMPL __VA_ARGS__
                    constexpr auto flag_primitive = LUISA_DEPARANTHESES(LCAccelBuildModificationFlags_PRIMITIVE);
                    constexpr auto flag_transform = LUISA_DEPARANTHESES(LCAccelBuildModificationFlags_TRANSFORM);
                    constexpr auto flag_opaque_on = LUISA_DEPARANTHESES(LCAccelBuildModificationFlags_OPAQUE_ON);
                    constexpr auto flag_opaque_off = LUISA_DEPARANTHESES(LCAccelBuildModificationFlags_OPAQUE_OFF);
                    constexpr auto flag_visibility = LUISA_DEPARANTHESES(LCAccelBuildModificationFlags_VISIBILITY);
                    constexpr auto flag_user_id = LUISA_DEPARANTHESES(LCAccelBuildModificationFlags_USER_ID);
#undef LUISA_DEPARANTHESES
#undef LUISA_DEPARANTHESES_IMPL
                    if (m.flags.bits & flag_primitive.bits) {
                        modification.set_primitive(m.mesh);
                    }
                    if (m.flags.bits & flag_transform.bits) {
                        modification.set_transform_data(m.affine);
                    }
                    if (m.flags.bits & flag_opaque_on.bits) {
                        modification.set_opaque(true);
                    } else if (m.flags.bits & flag_opaque_off.bits) {
                        modification.set_opaque(false);
                    }
                    if (m.flags.bits & flag_visibility.bits) {
                        modification.set_visibility(m.visibility);
                    }
                    if (m.flags.bits & flag_user_id.bits) {
                        modification.set_user_id(m.user_id);
                    }
                    modifications.emplace_back(modification);
                }
                LUISA_ASSERT(b.modifications_count == modifications.size(),
                             "modifications size mismatch");
                return luisa::make_unique<AccelBuildCommand>(
                    b.accel._0,
                    b.instance_count,
                    convert_accel_request(b.request),
                    std::move(modifications),
                    b.update_instance_buffer_only);
            }
            default:
                LUISA_ERROR_WITH_LOCATION(
                    "unimplemented command {}",
                    luisa::to_underlying(cmd.tag));
        }
    }

    void (*_callback)(uint8_t *) = nullptr;
    uint8_t *_callback_ctx = nullptr;

public:
    CommandListConverter(const LCCommandList list,
                         void (*callback)(uint8_t *),
                         uint8_t *callback_ctx) noexcept
        : _list(list), _callback(callback), _callback_ctx(callback_ctx) {}

    [[nodiscard]] auto convert() const noexcept {
        CommandList cmd_list;
        cmd_list.reserve(_list.commands_count, _callback ? 1u : 0u);
        for (int i = 0; i < _list.commands_count; i++) {
            cmd_list.append(convert_one(_list.commands[i]));
        }
        if (_callback != nullptr) {
            cmd_list.add_callback([cb = _callback, ctx = _callback_ctx]() {
                cb(ctx);
            });
        }
        return cmd_list.commit();
    }
};

}// namespace luisa::compute::detail

template<class T>
T from_ptr(void *ptr) {
    return T{._0 = reinterpret_cast<uint64_t>(ptr)};
}

LUISA_EXPORT_API LCContext luisa_compute_context_create(const char *exe_path) LUISA_NOEXCEPT {
    return from_ptr<LCContext>(new_with_allocator<Context>(exe_path));
}

LUISA_EXPORT_API void luisa_compute_context_destroy(LCContext ctx) LUISA_NOEXCEPT {
    delete_with_allocator(reinterpret_cast<Context *>(ctx._0));
}

inline char *path_to_c_str(const std::filesystem::path &path) LUISA_NOEXCEPT {
    auto s = path.string();
    auto cs = static_cast<char *>(malloc(s.size() + 1u));
    memcpy(cs, s.c_str(), s.size() + 1u);
    return cs;
}

LUISA_EXPORT_API void luisa_compute_free_c_string(char *cs) LUISA_NOEXCEPT { free(cs); }

LUISA_EXPORT_API char *luisa_compute_context_runtime_directory(LCContext ctx) LUISA_NOEXCEPT {
    return path_to_c_str(reinterpret_cast<Context *>(ctx._0)->runtime_directory());
}

LUISA_EXPORT_API LCDevice luisa_compute_device_create(LCContext ctx,
                                                      const char *name,
                                                      const char *properties) LUISA_NOEXCEPT {
    auto device = reinterpret_cast<Context *>(ctx._0)->create_device(name, nullptr);
    std::aligned_storage_t<sizeof(Device), alignof(Device)> storage;
    auto handle = device.impl();
    new (&storage) Device(std::move(device));
    return LCDevice{._0 = reinterpret_cast<uint64_t>(handle)};
}

LUISA_EXPORT_API void luisa_compute_device_destroy(LCDevice device) LUISA_NOEXCEPT {
    auto handle = reinterpret_cast<DeviceInterface *>(device._0)->shared_from_this();
    LUISA_ASSERT(handle.use_count() == 2u, "Should have exactly 2 references.");
    luisa::shared_ptr<DeviceInterface> _copy;
    std::memcpy(&_copy, &handle, sizeof(luisa::shared_ptr<DeviceInterface>));
}

LUISA_EXPORT_API void *luisa_compute_device_native_handle(LCDevice device) LUISA_NOEXCEPT {
    return reinterpret_cast<DeviceInterface *>(device._0)->native_handle();
}

LUISA_EXPORT_API LCCreatedBufferInfo
luisa_compute_buffer_create(LCDevice device, const void *element_, size_t elem_count) LUISA_NOEXCEPT {
    auto element = reinterpret_cast<const ir::CArc<ir::Type> *>(element_);
    auto d = reinterpret_cast<DeviceInterface *>(device._0);
    auto info = d->create_buffer(element, elem_count);
    return LCCreatedBufferInfo{
        .resource = LCCreatedResourceInfo{
            .handle = info.handle,
            .native_handle = info.native_handle,
        },
        .element_stride = info.element_stride,
        .total_size_bytes = info.total_size_bytes,
    };
}

LUISA_EXPORT_API void luisa_compute_buffer_destroy(LCDevice device, LCBuffer buffer) LUISA_NOEXCEPT {
    auto handle = buffer._0;
    auto d = reinterpret_cast<DeviceInterface *>(device._0);
    d->destroy_buffer(handle);
}

LUISA_EXPORT_API LCCreatedResourceInfo luisa_compute_texture_create(LCDevice device,
                                                                    LCPixelFormat format, uint32_t dim,
                                                                    uint32_t w, uint32_t h, uint32_t d,
                                                                    uint32_t mips, bool allow_simultaneous_access) LUISA_NOEXCEPT {
    auto dev = reinterpret_cast<DeviceInterface *>(device._0);
    auto pixel_format = PixelFormat{(uint8_t)to_underlying(format)};
    auto info = dev->create_texture(pixel_format, dim, w, h, d, mips, allow_simultaneous_access);
    return LCCreatedResourceInfo{
        .handle = info.handle,
        .native_handle = info.native_handle,
    };
}

LUISA_EXPORT_API void luisa_compute_texture_destroy(LCDevice device, LCTexture texture) LUISA_NOEXCEPT {
    auto handle = texture._0;
    auto d = reinterpret_cast<DeviceInterface *>(device._0);
    d->destroy_texture(handle);
}

LUISA_EXPORT_API LCCreatedResourceInfo
luisa_compute_stream_create(LCDevice device, LCStreamTag stream_tag) LUISA_NOEXCEPT {
    auto d = reinterpret_cast<DeviceInterface *>(device._0);
    auto tag = StreamTag{(uint8_t)to_underlying(stream_tag)};
    auto info = d->create_stream(tag);
    return LCCreatedResourceInfo{
        .handle = info.handle,
        .native_handle = info.native_handle,
    };
}

LUISA_EXPORT_API void luisa_compute_stream_destroy(LCDevice device, LCStream stream) LUISA_NOEXCEPT {
    auto handle = stream._0;
    auto d = reinterpret_cast<DeviceInterface *>(device._0);
    d->destroy_stream(handle);
}

LUISA_EXPORT_API void luisa_compute_stream_synchronize(LCDevice device, LCStream stream) LUISA_NOEXCEPT {
    auto handle = stream._0;
    auto d = reinterpret_cast<DeviceInterface *>(device._0);
    d->synchronize_stream(handle);
}

LUISA_EXPORT_API void
luisa_compute_stream_dispatch(LCDevice device, LCStream stream, LCCommandList cmd_list, void (*callback)(uint8_t *),
                              uint8_t *callback_ctx) LUISA_NOEXCEPT {
    auto handle = stream._0;
    auto d = reinterpret_cast<DeviceInterface *>(device._0);
    luisa::compute::detail::CommandListConverter converter{cmd_list, callback, callback_ctx};
    auto list = converter.convert().command_list();
    d->dispatch(handle, std::move(list));
}

LUISA_EXPORT_API LCCreatedShaderInfo
luisa_compute_shader_create(LCDevice device, LCKernelModule m, const LCShaderOption *option_) LUISA_NOEXCEPT {
    const auto &option = *option_;
    auto d = reinterpret_cast<DeviceInterface *>(device._0);
    auto ir = reinterpret_cast<const ir::KernelModule *>(m.ptr);

    auto shader_option = ShaderOption{
        .enable_cache = option.enable_cache,
        .enable_fast_math = option.enable_fast_math,
        .enable_debug_info = option.enable_debug_info,
        .compile_only = option.compile_only,
        .name = luisa::string{option.name}};

    auto info = d->create_shader(shader_option, ir);
    return LCCreatedShaderInfo{
        .resource = LCCreatedResourceInfo{
            .handle = info.handle,
            .native_handle = info.native_handle,
        },
        .block_size = {info.block_size[0], info.block_size[1], info.block_size[2]},
    };
}

LUISA_EXPORT_API void luisa_compute_shader_destroy(LCDevice device, LCShader shader) LUISA_NOEXCEPT {
    auto handle = shader._0;
    auto d = reinterpret_cast<DeviceInterface *>(device._0);
    d->destroy_shader(handle);
}

LUISA_EXPORT_API LCCreatedResourceInfo luisa_compute_event_create(LCDevice device) LUISA_NOEXCEPT {
    auto d = reinterpret_cast<DeviceInterface *>(device._0);
    auto info = d->create_event();
    return LCCreatedResourceInfo{
        .handle = info.handle,
        .native_handle = info.native_handle,
    };
}

LUISA_EXPORT_API void luisa_compute_event_destroy(LCDevice device, LCEvent event) LUISA_NOEXCEPT {
    auto handle = event._0;
    auto d = reinterpret_cast<DeviceInterface *>(device._0);
    d->destroy_event(handle);
}

LUISA_EXPORT_API void luisa_compute_event_signal(LCDevice device, LCEvent event, LCStream stream, uint64_t value) LUISA_NOEXCEPT {
    auto handle = event._0;
    auto stream_handle = stream._0;
    auto d = reinterpret_cast<DeviceInterface *>(device._0);
    d->signal_event(handle, stream_handle, value);
}

LUISA_EXPORT_API void luisa_compute_event_wait(LCDevice device, LCEvent event, LCStream stream, uint64_t value) LUISA_NOEXCEPT {
    auto handle = event._0;
    auto stream_handle = stream._0;
    auto d = reinterpret_cast<DeviceInterface *>(device._0);
    d->wait_event(handle, stream_handle, value);
}

LUISA_EXPORT_API void luisa_compute_event_synchronize(LCDevice device, LCEvent event, uint64_t value) LUISA_NOEXCEPT {
    auto handle = event._0;
    auto d = reinterpret_cast<DeviceInterface *>(device._0);
    d->synchronize_event(handle, value);
}

LUISA_EXPORT_API bool luisa_compute_is_event_completed(LCDevice device, LCEvent event, uint64_t value) LUISA_NOEXCEPT {
    auto handle = event._0;
    auto d = reinterpret_cast<DeviceInterface *>(device._0);
    return d->is_event_completed(handle, value);
}

LUISA_EXPORT_API LCCreatedResourceInfo
luisa_compute_mesh_create(LCDevice device, const LCAccelOption *option_) LUISA_NOEXCEPT {
    const auto &option = *option_;
    auto d = reinterpret_cast<DeviceInterface *>(device._0);
    auto accel_option = AccelOption{
        .hint = AccelOption::UsageHint{(uint32_t)to_underlying(option.hint)},
        .allow_compaction = option.allow_compaction,
        .allow_update = option.allow_update,
    };
    auto info = d->create_mesh(accel_option);
    return LCCreatedResourceInfo{
        .handle = info.handle,
        .native_handle = info.native_handle,
    };
}

LUISA_EXPORT_API void luisa_compute_mesh_destroy(LCDevice device, LCMesh mesh) LUISA_NOEXCEPT {
    auto handle = mesh._0;
    auto d = reinterpret_cast<DeviceInterface *>(device._0);
    d->destroy_mesh(handle);
}

LUISA_EXPORT_API LCCreatedResourceInfo
luisa_compute_procedural_primitive_create(LCDevice device, const LCAccelOption *option_) LUISA_NOEXCEPT {
    const auto &option = *option_;
    auto d = reinterpret_cast<DeviceInterface *>(device._0);
    auto accel_option = AccelOption{
        .hint = AccelOption::UsageHint{(uint32_t)to_underlying(option.hint)},
        .allow_compaction = option.allow_compaction,
        .allow_update = option.allow_update,
    };
    auto info = d->create_procedural_primitive(accel_option);
    return LCCreatedResourceInfo{
        .handle = info.handle,
        .native_handle = info.native_handle,
    };
}

LUISA_EXPORT_API void luisa_compute_procedural_primitive_destroy(LCDevice device, LCProceduralPrimitive prim) LUISA_NOEXCEPT {
    auto handle = prim._0;
    auto d = reinterpret_cast<DeviceInterface *>(device._0);
    d->destroy_procedural_primitive(handle);
}

LUISA_EXPORT_API LCCreatedResourceInfo
luisa_compute_accel_create(LCDevice device, const LCAccelOption *option_) LUISA_NOEXCEPT {
    const auto &option = *option_;
    auto d = reinterpret_cast<DeviceInterface *>(device._0);
    auto accel_option = AccelOption{
        .hint = AccelOption::UsageHint{(uint32_t)to_underlying(option.hint)},
        .allow_compaction = option.allow_compaction,
        .allow_update = option.allow_update,
    };
    auto info = d->create_accel(accel_option);
    return LCCreatedResourceInfo{
        .handle = info.handle,
        .native_handle = info.native_handle,
    };
}

LUISA_EXPORT_API void luisa_compute_accel_destroy(LCDevice device, LCAccel accel) LUISA_NOEXCEPT {
    auto handle = accel._0;
    auto d = reinterpret_cast<DeviceInterface *>(device._0);
    d->destroy_accel(handle);
}

LUISA_EXPORT_API LCPixelStorage luisa_compute_pixel_format_to_storage(LCPixelFormat format) LUISA_NOEXCEPT {
    return static_cast<LCPixelStorage>(
        to_underlying(pixel_format_to_storage(static_cast<PixelFormat>(format))));
}

inline Sampler convert_sampler(LCSampler sampler) {
    auto address = [&sampler]() noexcept -> Sampler::Address {
        switch (sampler.address) {
            case LC_SAMPLER_ADDRESS_ZERO:
                return Sampler::Address::ZERO;
            case LC_SAMPLER_ADDRESS_EDGE:
                return Sampler::Address::EDGE;
            case LC_SAMPLER_ADDRESS_REPEAT:
                return Sampler::Address::REPEAT;
            case LC_SAMPLER_ADDRESS_MIRROR:
                return Sampler::Address::MIRROR;
            default:
                LUISA_ERROR_WITH_LOCATION("Invalid sampler address mode {}", static_cast<int>(sampler.address));
        }
    }();
    auto filter = [&sampler]() noexcept -> Sampler::Filter {
        switch (sampler.filter) {
            case LC_SAMPLER_FILTER_POINT:
                return Sampler::Filter::POINT;
            case LC_SAMPLER_FILTER_LINEAR_LINEAR:
                return Sampler::Filter::LINEAR_LINEAR;
            case LC_SAMPLER_FILTER_ANISOTROPIC:
                return Sampler::Filter::ANISOTROPIC;
            case LC_SAMPLER_FILTER_LINEAR_POINT:
                return Sampler::Filter::LINEAR_POINT;
            default:
                LUISA_ERROR_WITH_LOCATION("Invalid sampler filter mode {}", static_cast<int>(sampler.filter));
        }
    }();
    return {filter, address};
}

LUISA_EXPORT_API LCCreatedResourceInfo luisa_compute_bindless_array_create(LCDevice device, size_t n) LUISA_NOEXCEPT {
    auto d = reinterpret_cast<DeviceInterface *>(device._0);
    auto info = d->create_bindless_array(n);
    return LCCreatedResourceInfo{
        .handle = info.handle,
        .native_handle = info.native_handle,
    };
}

LUISA_EXPORT_API void luisa_compute_bindless_array_destroy(LCDevice device, LCBindlessArray array) LUISA_NOEXCEPT {
    auto handle = array._0;
    auto d = reinterpret_cast<DeviceInterface *>(device._0);
    d->destroy_bindless_array(handle);
}

size_t luisa_compute_device_query(LCDevice device, const char *query, char *result, size_t maxlen) LUISA_NOEXCEPT {
    auto d = reinterpret_cast<DeviceInterface *>(device._0);
    auto result_s = d->query(luisa::string_view{query});
    auto len = std::min(result_s.size(), maxlen);
    std::memcpy(result, result_s.data(), len);
    result[len] = '\0';
    return len;
}

LCCreatedSwapchainInfo luisa_compute_swapchain_create(
    LCDevice device, uint64_t window_handle, LCStream stream_handle,
    uint width, uint height, bool allow_hdr, bool vsync, uint back_buffer_size) LUISA_NOEXCEPT {
    auto d = reinterpret_cast<DeviceInterface *>(device._0);
    auto ret = d->create_swapchain(
        window_handle, stream_handle._0,
        width, height, allow_hdr, vsync, back_buffer_size);
    return LCCreatedSwapchainInfo{
        .resource = LCCreatedResourceInfo{
            .handle = ret.handle,
            .native_handle = ret.native_handle,
        },
        .storage = static_cast<LCPixelStorage>(ret.storage),
    };
}

void luisa_compute_swapchain_destroy(LCDevice device, LCSwapchain swapchain) LUISA_NOEXCEPT {
    auto d = reinterpret_cast<DeviceInterface *>(device._0);
    d->destroy_swap_chain(swapchain._0);
}

void luisa_compute_swapchain_present(LCDevice device, LCStream stream, LCSwapchain swapchain,
                                     LCTexture image) LUISA_NOEXCEPT {
    auto d = reinterpret_cast<DeviceInterface *>(device._0);
    d->present_display_in_stream(stream._0, swapchain._0, image._0);
}

LUISA_EXPORT_API void luisa_compute_device_interface_destroy(LCDeviceInterface device) {
    luisa_compute_device_destroy(device.device);
}

LUISA_EXPORT_API LCDeviceInterface luisa_compute_device_interface_create(LCContext ctx, const char *name, const char *config) {
    LCDeviceInterface interface{};
    auto device = luisa_compute_device_create(ctx, name, config);
    interface.device = device;
    interface.destroy_device = luisa_compute_device_interface_destroy;
    interface.create_buffer = luisa_compute_buffer_create;
    interface.destroy_buffer = luisa_compute_buffer_destroy;
    interface.create_texture = luisa_compute_texture_create;
    interface.destroy_texture = luisa_compute_texture_destroy;
    interface.create_bindless_array = luisa_compute_bindless_array_create;
    interface.destroy_bindless_array = luisa_compute_bindless_array_destroy;
    interface.create_event = luisa_compute_event_create;
    interface.destroy_event = luisa_compute_event_destroy;
    interface.wait_event = luisa_compute_event_wait;
    interface.signal_event = luisa_compute_event_signal;
    interface.is_event_completed = luisa_compute_is_event_completed;
    interface.synchronize_event = luisa_compute_event_synchronize;
    interface.create_shader = luisa_compute_shader_create;
    interface.destroy_shader = luisa_compute_shader_destroy;
    interface.create_stream = luisa_compute_stream_create;
    interface.synchronize_stream = luisa_compute_stream_synchronize;
    interface.destroy_stream = luisa_compute_stream_destroy;
    interface.dispatch = luisa_compute_stream_dispatch;
    interface.create_mesh = luisa_compute_mesh_create;
    interface.destroy_mesh = luisa_compute_mesh_destroy;
    interface.create_accel = luisa_compute_accel_create;
    interface.destroy_accel = luisa_compute_accel_destroy;
    interface.create_swapchain = luisa_compute_swapchain_create;
    interface.present_display_in_stream = luisa_compute_swapchain_present;
    interface.destroy_swapchain = luisa_compute_swapchain_destroy;
    interface.create_procedural_primitive = luisa_compute_procedural_primitive_create;
    interface.destroy_procedural_primitive = luisa_compute_procedural_primitive_destroy;
    interface.query = [](LCDevice device, const char *query) -> char * {
        auto d = reinterpret_cast<DeviceInterface *>(device._0);
        auto result_s = d->query(luisa::string_view{query});
        char *result = (char *)malloc(result_s.size() + 1);
        std::memcpy(result, result_s.data(), result_s.size());
        result[result_s.size()] = '\0';
        return result;
    };
    return interface;
}

LUISA_EXPORT_API LCLibInterface luisa_compute_lib_interface() {
    LCLibInterface interface{};
    interface.inner = nullptr;
    interface.set_logger_callback = luisa_compute_set_logger_callback;
    interface.create_context = luisa_compute_context_create;
    interface.destroy_context = luisa_compute_context_destroy;
    interface.create_device = luisa_compute_device_interface_create;
    interface.free_string = luisa_compute_free_c_string;
    return interface;
}
