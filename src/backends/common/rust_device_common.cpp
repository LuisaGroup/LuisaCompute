#include <luisa/rust/ir.hpp>
#include <luisa/rust/api_types.hpp>

namespace luisa::compute::backend {
using namespace luisa::compute::api;
using luisa::compute::ir::CArc;
using luisa::compute::ir::KernelModule;
using luisa::compute::ir::Type;
}// namespace luisa::compute::backend

#include <luisa/core/dynamic_module.h>
#include <luisa/core/logging.h>
#include <luisa/runtime/context.h>
#include <luisa/runtime/rtx/triangle.h>
#include <luisa/ir/ast2ir.h>
#include <luisa/runtime/rtx/aabb.h>
#include "rust_device_common.h"

// must go last to avoid name conflicts
#include <luisa/runtime/rhi/resource.h>

namespace luisa::compute::rust {

class APICommandConverter final : public CommandVisitor {

public:
    class CommandBuffer {

    private:
        luisa::vector<void *> _temp;
        luisa::vector<api::Command> _api_commands;
        CommandList _list;

    public:
        CommandBuffer(luisa::vector<void *> temp,
                      luisa::vector<api::Command> api_commands,
                      CommandList list) noexcept
            : _temp{std::move(temp)},
              _api_commands{std::move(api_commands)},
              _list{std::move(list)} {}

        void on_completion() noexcept {
            for (auto &&callback : _list.callbacks()) { callback(); }
            for (auto p : _temp) {
                luisa::deallocate_with_allocator(
                    static_cast<std::byte *>(p));
            }
        }
    };

private:
    luisa::vector<void *> _temp;
    luisa::vector<api::Command> _converted;

private:
    template<typename T>
    [[nodiscard]] auto _create_temporary(size_t n) noexcept {
        auto ptr = luisa::allocate_with_allocator<T>(n);
        memset(ptr, 0, sizeof(T) * n);
        _temp.emplace_back(ptr);
        return ptr;
    }

    using Tag = api::Command::Tag;

    [[nodiscard]] static auto _convert_pixel_storage(PixelStorage s) noexcept {
        // TODO: might be better to use a lookup table
        return static_cast<api::PixelStorage>(s);
    }

    [[nodiscard]] static auto _convert_accel_build_request(AccelBuildRequest r) noexcept {
        return r == AccelBuildRequest::PREFER_UPDATE ?
                   api::AccelBuildRequest::PREFER_UPDATE :
                   api::AccelBuildRequest::FORCE_BUILD;
    }

public:
    void dispatch(api::DeviceInterface device, api::Stream stream,
                  CommandList &&list) noexcept {

        LUISA_ASSERT(_temp.empty(), "Temporary buffer leak.");
        LUISA_ASSERT(_converted.empty(), "Command buffer leak.");

        _converted.reserve(list.commands().size());
        for (auto &&cmd : list.commands()) { cmd->accept(*this); }
        LUISA_ASSERT(_converted.size() == list.commands().size(),
                     "Command list size mismatch.");

        api::CommandList converted_list{
            .commands = _converted.data(),
            .commands_count = _converted.size(),
        };
        auto ctx = luisa::new_with_allocator<CommandBuffer>(
            std::move(_temp),
            std::move(_converted),
            std::move(list));
        device.dispatch(
            device.device, stream, converted_list,
            [](uint8_t *ctx) noexcept {
                auto cb = reinterpret_cast<CommandBuffer *>(ctx);
                cb->on_completion();
                luisa::delete_with_allocator(cb);
            },
            reinterpret_cast<uint8_t *>(ctx));
    }
    void visit(const BufferUploadCommand *command) noexcept override {
        api::Command converted{.tag = Tag::BUFFER_UPLOAD};
        converted.BUFFER_UPLOAD._0 = api::BufferUploadCommand{
            .buffer = {command->handle()},
            .offset = command->offset(),
            .size = command->size(),
            .data = static_cast<const uint8_t *>(command->data())};
        _converted.emplace_back(converted);
    }
    void visit(const BufferDownloadCommand *command) noexcept override {
        api::Command converted{.tag = Tag::BUFFER_DOWNLOAD};
        converted.BUFFER_DOWNLOAD._0 = api::BufferDownloadCommand{
            .buffer = {command->handle()},
            .offset = command->offset(),
            .size = command->size(),
            .data = static_cast<uint8_t *>(command->data())};
        _converted.emplace_back(converted);
    }
    void visit(const BufferCopyCommand *command) noexcept override {
        api::Command converted{.tag = Tag::BUFFER_COPY};
        converted.BUFFER_COPY._0 = api::BufferCopyCommand{
            .src = {command->src_handle()},
            .src_offset = command->src_offset(),
            .dst = {command->dst_handle()},
            .dst_offset = command->dst_offset(),
            .size = command->size()};
        _converted.emplace_back(converted);
    }
    void visit(const BufferToTextureCopyCommand *command) noexcept override {
        api::Command converted{.tag = Tag::BUFFER_TO_TEXTURE_COPY};
        converted.BUFFER_TO_TEXTURE_COPY._0 = api::BufferToTextureCopyCommand{
            .buffer = {command->buffer()},
            .buffer_offset = command->buffer_offset(),
            .texture = {command->texture()},
            .storage = _convert_pixel_storage(command->storage()),
            .texture_level = command->level(),
            .texture_size = {command->size().x,
                             command->size().y,
                             command->size().z}};
        _converted.emplace_back(converted);
    }
    void visit(const ShaderDispatchCommand *command) noexcept override {
        LUISA_ASSERT(!command->is_indirect(),
                     "Indirect dispatch is not supported.");
        auto n = command->arguments().size();
        auto args = _create_temporary<api::Argument>(n);
        for (size_t i = 0; i < n; i++) {
            auto &&arg = command->arguments()[i];
            switch (arg.tag) {
                case Argument::Tag::BUFFER: {
                    args[i].tag = api::Argument::Tag::BUFFER;
                    args[i].BUFFER._0 = api::BufferArgument{
                        .buffer = {arg.buffer.handle},
                        .offset = arg.buffer.offset,
                        .size = arg.buffer.size};
                    break;
                }
                case Argument::Tag::TEXTURE: {
                    args[i].tag = api::Argument::Tag::TEXTURE;
                    args[i].TEXTURE._0 = api::TextureArgument{
                        .texture = {arg.texture.handle},
                        .level = arg.texture.level};
                    break;
                }
                case Argument::Tag::UNIFORM: {
                    auto data = command->uniform(arg.uniform);
                    args[i].tag = api::Argument::Tag::UNIFORM;
                    args[i].UNIFORM._0 = api::UniformArgument{
                        .data = reinterpret_cast<const uint8_t *>(data.data()),
                        .size = data.size()};
                    break;
                }
                case Argument::Tag::BINDLESS_ARRAY: {
                    args[i].tag = api::Argument::Tag::BINDLESS_ARRAY;
                    args[i].BINDLESS_ARRAY._0 = {arg.bindless_array.handle};
                    break;
                }
                case Argument::Tag::ACCEL: {
                    args[i].tag = api::Argument::Tag::ACCEL;
                    args[i].ACCEL._0 = {arg.accel.handle};
                    break;
                }
                default: LUISA_ERROR_WITH_LOCATION(
                    "Unsupported shader argument type.");
            }
        }
        api::Command converted{.tag = Tag::SHADER_DISPATCH};
        converted.SHADER_DISPATCH._0 = api::ShaderDispatchCommand{
            .shader = {command->handle()},
            .dispatch_size = {command->dispatch_size().x,
                              command->dispatch_size().y,
                              command->dispatch_size().z},
            .args = args,
            .args_count = n};
        _converted.emplace_back(converted);
    }
    void visit(const TextureUploadCommand *command) noexcept override {
        api::Command converted{.tag = Tag::TEXTURE_UPLOAD};
        converted.TEXTURE_UPLOAD._0 = api::TextureUploadCommand{
            .texture = {command->handle()},
            .storage = _convert_pixel_storage(command->storage()),
            .level = command->level(),
            .size = {command->size().x,
                     command->size().y,
                     command->size().z},
            .data = static_cast<const uint8_t *>(command->data())};
        _converted.emplace_back(converted);
    }
    void visit(const TextureDownloadCommand *command) noexcept override {
        api::Command converted{.tag = Tag::TEXTURE_DOWNLOAD};
        converted.TEXTURE_DOWNLOAD._0 = api::TextureDownloadCommand{
            .texture = {command->handle()},
            .storage = _convert_pixel_storage(command->storage()),
            .level = command->level(),
            .size = {command->size().x,
                     command->size().y,
                     command->size().z},
            .data = static_cast<uint8_t *>(command->data())};
        _converted.emplace_back(converted);
    }
    void visit(const TextureCopyCommand *command) noexcept override {
        api::Command converted{.tag = Tag::TEXTURE_COPY};
        converted.TEXTURE_COPY._0 = api::TextureCopyCommand{
            .storage = _convert_pixel_storage(command->storage()),
            .src = {command->src_handle()},
            .dst = {command->dst_handle()},
            .size = {command->size().x,
                     command->size().y,
                     command->size().z},
            .src_level = command->src_level(),
            .dst_level = command->dst_level()};
        _converted.emplace_back(converted);
    }
    void visit(const TextureToBufferCopyCommand *command) noexcept override {
        api::Command converted{.tag = Tag::TEXTURE_TO_BUFFER_COPY};
        converted.TEXTURE_TO_BUFFER_COPY._0 = api::TextureToBufferCopyCommand{
            .buffer = {command->buffer()},
            .buffer_offset = command->buffer_offset(),
            .texture = {command->texture()},
            .storage = _convert_pixel_storage(command->storage()),
            .texture_level = command->level(),
            .texture_size = {command->size().x,
                             command->size().y,
                             command->size().z}};
        _converted.emplace_back(converted);
    }
    void visit(const AccelBuildCommand *command) noexcept override {
        auto n = command->modifications().size();
        auto m = _create_temporary<api::AccelBuildModification>(n);
        for (auto i = 0u; i < n; i++) {
            using Mod = AccelBuildCommand::Modification;
            auto &&mod = command->modifications()[i];
            m[i].index = mod.index;
            if (mod.flags & Mod::flag_primitive) {
                m[i].flags.bits |= api::AccelBuildModificationFlags_PRIMITIVE.bits;
                m[i].mesh = mod.primitive;
            }
            if (mod.flags & Mod::flag_transform) {
                m[i].flags.bits |= api::AccelBuildModificationFlags_TRANSFORM.bits;
                memcpy(m[i].affine, mod.affine, sizeof(float) * 12u);
            }
            if (mod.flags & Mod::flag_opaque_on) {
                m[i].flags.bits |= api::AccelBuildModificationFlags_OPAQUE_ON.bits;
            } else if (mod.flags & Mod::flag_opaque_off) {
                m[i].flags.bits |= api::AccelBuildModificationFlags_OPAQUE_OFF.bits;
            }
            if (mod.flags & Mod::flag_visibility) {
                m[i].flags.bits |= api::AccelBuildModificationFlags_VISIBILITY.bits;
                m[i].visibility = mod.flags >> Mod::flag_vis_mask_offset;
            }
        }
        api::Command converted{.tag = Tag::ACCEL_BUILD};
        converted.ACCEL_BUILD._0 = api::AccelBuildCommand{
            .accel = {command->handle()},
            .request = _convert_accel_build_request(command->request()),
            .instance_count = command->instance_count(),
            .modifications = m,
            .modifications_count = n,
            .update_instance_buffer_only = command->update_instance_buffer_only()};
        _converted.emplace_back(converted);
    }
    void visit(const MeshBuildCommand *command) noexcept override {
        api::Command converted{.tag = Tag::MESH_BUILD};
        converted.MESH_BUILD._0 = api::MeshBuildCommand{
            .mesh = {command->handle()},
            .request = _convert_accel_build_request(command->request()),
            .vertex_buffer = {command->vertex_buffer()},
            .vertex_buffer_offset = command->vertex_buffer_offset(),
            .vertex_buffer_size = command->vertex_buffer_size(),
            .vertex_stride = command->vertex_stride(),
            .index_buffer = {command->triangle_buffer()},
            .index_buffer_offset = command->triangle_buffer_offset(),
            .index_buffer_size = command->triangle_buffer_size(),
            .index_stride = sizeof(Triangle)};
        _converted.emplace_back(converted);
    }
    void visit(const ProceduralPrimitiveBuildCommand *command) noexcept override {
        api::Command converted{.tag = Tag::PROCEDURAL_PRIMITIVE_BUILD};
        converted.PROCEDURAL_PRIMITIVE_BUILD._0 = api::ProceduralPrimitiveBuildCommand{
            .handle = {command->handle()},
            .request = _convert_accel_build_request(command->request()),
            .aabb_buffer = {command->aabb_buffer()},
            .aabb_buffer_offset = command->aabb_buffer_offset(),
            .aabb_count = command->aabb_buffer_size() / sizeof(AABB)};
        _converted.emplace_back(converted);
    }
    void visit(const BindlessArrayUpdateCommand *command) noexcept override {
        auto n = command->modifications().size();
        auto m = _create_temporary<api::BindlessArrayUpdateModification>(n);
        auto convert_op = [](auto op) noexcept {
            using Op = BindlessArrayUpdateCommand::Modification::Operation;
            switch (op) {
                case Op::NONE: return api::BindlessArrayUpdateOperation::NONE;
                case Op::EMPLACE: return api::BindlessArrayUpdateOperation::EMPLACE;
                case Op::REMOVE: return api::BindlessArrayUpdateOperation::REMOVE;
                default: break;
            }
            LUISA_ERROR_WITH_LOCATION("Invalid bindless array update operation.");
        };
        auto convert_sampler = [](Sampler sampler) noexcept {
            api::Sampler s{};
            switch (sampler.filter()) {
                case Sampler::Filter::POINT: s.filter = api::SamplerFilter::POINT; break;
                case Sampler::Filter::LINEAR_POINT: s.filter = api::SamplerFilter::LINEAR_POINT; break;
                case Sampler::Filter::LINEAR_LINEAR: s.filter = api::SamplerFilter::LINEAR_LINEAR; break;
                case Sampler::Filter::ANISOTROPIC: s.filter = api::SamplerFilter::ANISOTROPIC; break;
            }
            switch (sampler.address()) {
                case Sampler::Address::EDGE: s.address = api::SamplerAddress::EDGE; break;
                case Sampler::Address::REPEAT: s.address = api::SamplerAddress::REPEAT; break;
                case Sampler::Address::MIRROR: s.address = api::SamplerAddress::MIRROR; break;
                case Sampler::Address::ZERO: s.address = api::SamplerAddress::ZERO; break;
            }
            return s;
        };
        for (auto i = 0u; i < n; i++) {
            auto &&mod = command->modifications()[i];
            m[i].slot = mod.slot;
            m[i].buffer = api::BindlessArrayUpdateBuffer{
                .op = convert_op(mod.buffer.op),
                .handle = {mod.buffer.handle},
                .offset = mod.buffer.offset_bytes};
            m[i].tex2d = api::BindlessArrayUpdateTexture{
                .op = convert_op(mod.tex2d.op),
                .handle = {mod.tex2d.handle},
                .sampler = convert_sampler(mod.tex2d.sampler)};
            m[i].tex3d = api::BindlessArrayUpdateTexture{
                .op = convert_op(mod.tex3d.op),
                .handle = {mod.tex3d.handle},
                .sampler = convert_sampler(mod.tex3d.sampler)};
        }
        api::Command converted{.tag = Tag::BINDLESS_ARRAY_UPDATE};
        converted.BINDLESS_ARRAY_UPDATE._0 = api::BindlessArrayUpdateCommand{
            .handle = {command->handle()},
            .modifications = m,
            .modifications_count = n};
        _converted.emplace_back(converted);
    }
    void visit(const CustomCommand *command) noexcept override {
        LUISA_ERROR_WITH_LOCATION("Not implemented.");
    }
};

// @Mike-Leo-Smith: fill-in the blanks pls
class RustDevice final : public DeviceInterface {
    api::DeviceInterface device{};
    api::LibInterface lib{};
    luisa::filesystem::path runtime_path;
    DynamicModule dll;

    api::LibInterface (*luisa_compute_lib_interface)();

    api::Context api_ctx{};

public:
    ~RustDevice() noexcept override {
        device.destroy_device(device);
        lib.destroy_context(api_ctx);
    }

    RustDevice(Context &&ctx, luisa::filesystem::path runtime_path, string_view name) noexcept
        : DeviceInterface(std::move(ctx)),
          runtime_path(std::move(runtime_path)) {
        dll = DynamicModule::load(this->runtime_path, "luisa_compute_backend_impl");
        luisa_compute_lib_interface = dll.function<api::LibInterface()>("luisa_compute_lib_interface");
        lib = luisa_compute_lib_interface();
        api_ctx = lib.create_context(this->runtime_path.generic_string().c_str());
        device = lib.create_device(api_ctx, name.data(), nullptr);
        lib.set_logger_callback([](api::LoggerMessage message) {
            luisa::string_view target(message.target);
            luisa::string_view level(message.level);
            luisa::string_view body(message.message);
            if (level == "I") {
                LUISA_INFO("[{}] {}", target, body);
            } else if (level == "W") {
                LUISA_WARNING("[{}] {}", target, body);
            } else if (level == "E") {
                LUISA_ERROR("[{}] {}", target, body);
            } else {
                LUISA_VERBOSE("[{}] {}", target, body);
            }
        });
    }

    void *native_handle() const noexcept override {
        return (void *)device.device._0;
    }

    uint compute_warp_size() const noexcept override {
        LUISA_NOT_IMPLEMENTED();
    }

    BufferCreationInfo create_buffer(const Type *element, size_t elem_count) noexcept override {
        auto type = AST2IR::build_type(element);
        return create_buffer(&type, elem_count);
    }

    BufferCreationInfo create_buffer(const ir::CArc<ir::Type> *element, size_t elem_count) noexcept override {
        api::CreatedBufferInfo buffer = device.create_buffer(device.device, element, elem_count);
        BufferCreationInfo info{};
        info.element_stride = buffer.element_stride;
        info.total_size_bytes = buffer.total_size_bytes;
        info.handle = buffer.resource.handle;
        info.native_handle = buffer.resource.native_handle;
        return info;
    }

    void destroy_buffer(uint64_t handle) noexcept override {
        device.destroy_buffer(device.device, api::Buffer{handle});
    }

    ResourceCreationInfo create_texture(PixelFormat format, uint dimension, uint width, uint height, uint depth,
                                        uint mipmap_levels, bool simultaneous_access) noexcept override {
        api::CreatedResourceInfo texture =
            device.create_texture(device.device, (api::PixelFormat)format, dimension,
                                  width, height, depth, mipmap_levels, simultaneous_access);
        ResourceCreationInfo info{};
        info.handle = texture.handle;
        info.native_handle = texture.native_handle;
        return info;
    }

    void destroy_texture(uint64_t handle) noexcept override {
        device.destroy_texture(device.device, api::Texture{handle});
    }

    ResourceCreationInfo create_bindless_array(size_t size) noexcept override {
        api::CreatedResourceInfo array = device.create_bindless_array(device.device, size);
        ResourceCreationInfo info{};
        info.handle = array.handle;
        info.native_handle = array.native_handle;
        return info;
    }

    void destroy_bindless_array(uint64_t handle) noexcept override {
        device.destroy_bindless_array(device.device, api::BindlessArray{handle});
    }

    ResourceCreationInfo create_stream(StreamTag stream_tag) noexcept override {
        api::CreatedResourceInfo stream = device.create_stream(device.device, (api::StreamTag)stream_tag);
        ResourceCreationInfo info{};
        info.handle = stream.handle;
        info.native_handle = stream.native_handle;
        return info;
    }

    void destroy_stream(uint64_t handle) noexcept override {
        device.destroy_stream(device.device, api::Stream{handle});
    }

    void synchronize_stream(uint64_t stream_handle) noexcept override {
        device.synchronize_stream(device.device, api::Stream{stream_handle});
    }

    void dispatch(uint64_t stream_handle, CommandList &&list) noexcept override {
        APICommandConverter converter;
        converter.dispatch(device, api::Stream{stream_handle}, std::move(list));
    }

    SwapchainCreationInfo
    create_swapchain(uint64_t window_handle, uint64_t stream_handle, uint width, uint height, bool allow_hdr,
                     bool vsync, uint back_buffer_size) noexcept override {
        auto swapchain =
            device.create_swapchain(device.device, window_handle, api::Stream{stream_handle}, width, height,
                                    allow_hdr, vsync, back_buffer_size);
        SwapchainCreationInfo info{};
        info.handle = swapchain.resource.handle;
        info.native_handle = swapchain.resource.native_handle;
        info.storage = (PixelStorage)swapchain.storage;
        return info;
    }

    void destroy_swap_chain(uint64_t handle) noexcept override {
        device.destroy_swapchain(device.device, api::Swapchain{handle});
    }

    void present_display_in_stream(uint64_t stream_handle, uint64_t swapchain_handle,
                                   uint64_t image_handle) noexcept override {
        device.present_display_in_stream(device.device, api::Stream{stream_handle},
                                         api::Swapchain{swapchain_handle}, api::Texture{image_handle});
    }

    ShaderCreationInfo create_shader(const ShaderOption &option, Function kernel) noexcept override {
        auto shader = AST2IR::build_kernel(kernel);
        return create_shader(option, shader->get());
    }

    ShaderCreationInfo
    create_shader(const ShaderOption &option_, const ir::KernelModule *kernel) noexcept override {
        api::ShaderOption option{};
        option.compile_only = option_.compile_only;
        option.enable_cache = option_.enable_cache;
        option.enable_debug_info = option_.enable_debug_info;
        option.enable_fast_math = option_.enable_fast_math;
        auto shader = device.create_shader(device.device, api::KernelModule{(uint64_t)kernel}, &option);
        ShaderCreationInfo info{};
        info.block_size[0] = shader.block_size[0];
        info.block_size[1] = shader.block_size[1];
        info.block_size[2] = shader.block_size[2];
        info.handle = shader.resource.handle;
        info.native_handle = shader.resource.native_handle;
        return info;
    }

    ShaderCreationInfo
    load_shader(luisa::string_view name, luisa::span<const Type *const> arg_types) noexcept override {
        LUISA_NOT_IMPLEMENTED();
    }

    Usage shader_argument_usage(uint64_t handle, size_t index) noexcept override {
        LUISA_NOT_IMPLEMENTED();
    }

    void destroy_shader(uint64_t handle) noexcept override {
        device.destroy_shader(device.device, api::Shader{handle});
    }

    ResourceCreationInfo create_event() noexcept override {
        api::CreatedResourceInfo event = device.create_event(device.device);
        ResourceCreationInfo info{};
        info.handle = event.handle;
        info.native_handle = event.native_handle;
        return info;
    }

    void destroy_event(uint64_t handle) noexcept override {
        device.destroy_event(device.device, api::Event{handle});
    }

    void signal_event(uint64_t handle, uint64_t stream_handle, uint64_t value) noexcept override {
        device.signal_event(device.device, api::Event{handle}, api::Stream{stream_handle}, value);
    }

    void wait_event(uint64_t handle, uint64_t stream_handle, uint64_t value) noexcept override {
        device.wait_event(device.device, api::Event{handle}, api::Stream{stream_handle}, value);
    }

    void synchronize_event(uint64_t handle, uint64_t value) noexcept override {
        device.synchronize_event(device.device, api::Event{handle}, value);
    }

    bool is_event_completed(uint64_t handle, uint64_t value) const noexcept override {
        return device.is_event_completed(device.device, api::Event{handle}, value);
    }

    // make sure that we can convert between the two enums
    static_assert(luisa::to_underlying(AccelOption::UsageHint::FAST_TRACE) ==
                  luisa::to_underlying(api::AccelUsageHint::FAST_TRACE));
    static_assert(luisa::to_underlying(AccelOption::UsageHint::FAST_BUILD) ==
                  luisa::to_underlying(api::AccelUsageHint::FAST_BUILD));

    ResourceCreationInfo create_mesh(const AccelOption &option_) noexcept override {
        api::AccelOption option{};
        option.allow_compaction = option_.allow_compaction;
        option.allow_update = option_.allow_update;
        option.hint = static_cast<api::AccelUsageHint>(option_.hint);
        auto mesh = device.create_mesh(device.device, &option);
        ResourceCreationInfo info{};
        info.handle = mesh.handle;
        info.native_handle = mesh.native_handle;
        return info;
    }

    void destroy_mesh(uint64_t handle) noexcept override {
        device.destroy_mesh(device.device, api::Mesh{handle});
    }

    ResourceCreationInfo create_procedural_primitive(const AccelOption &option_) noexcept override {
        api::AccelOption option{};
        option.allow_compaction = option_.allow_compaction;
        option.allow_update = option_.allow_update;
        option.hint = static_cast<api::AccelUsageHint>(option_.hint);
        auto primitive = device.create_procedural_primitive(device.device, &option);
        ResourceCreationInfo info{};
        info.handle = primitive.handle;
        info.native_handle = primitive.native_handle;
        return info;
    }

    void destroy_procedural_primitive(uint64_t handle) noexcept override {
        device.destroy_procedural_primitive(device.device, api::ProceduralPrimitive{handle});
    }

    ResourceCreationInfo create_accel(const AccelOption &option_) noexcept override {
        api::AccelOption option{};
        option.allow_compaction = option_.allow_compaction;
        option.allow_update = option_.allow_update;
        option.hint = static_cast<api::AccelUsageHint>(option_.hint);
        auto accel = device.create_accel(device.device, &option);
        ResourceCreationInfo info{};
        info.handle = accel.handle;
        info.native_handle = accel.native_handle;
        return info;
    }

    void destroy_accel(uint64_t handle) noexcept override {
        device.destroy_accel(device.device, api::Accel{handle});
    }

    void set_name(luisa::compute::Resource::Tag resource_tag, uint64_t resource_handle,
                  luisa::string_view name) noexcept override {
    }
};

luisa::compute::DeviceInterface *create(luisa::compute::Context &&ctx,
                                        const luisa::compute::DeviceConfig *config,
                                        luisa::string_view name) noexcept {
    auto path = ctx.runtime_directory();
    return luisa::new_with_allocator<luisa::compute::rust::RustDevice>(
        std::move(ctx), std::move(path), "cpu");
}

void destroy(luisa::compute::DeviceInterface *device) noexcept {
    luisa::delete_with_allocator(device);
}
}// namespace luisa::compute::rust
