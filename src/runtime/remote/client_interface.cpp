#include <luisa/runtime/remote/client_interface.h>
#include <luisa/runtime/context.h>
#include <luisa/core/logging.h>
#include "serde.hpp"
#include "device_func.h"
#include <luisa/ast/callable_library.h>
namespace luisa::compute {
ClientInterface::ClientInterface(
    Context ctx,
    ClientCallback *callback) noexcept
    : DeviceInterface(std::move(ctx)),
      _callback(callback) {
    _receive_bytes.reserve(32);
    _send_bytes.reserve(65536);
}
BufferCreationInfo ClientInterface::create_buffer(
    const Type *element,
    size_t elem_count,
    void *external_memory /* nullptr if now imported from external memory */) noexcept {
    _send_bytes.clear();
    LUISA_ASSERT(external_memory == nullptr, "Remote device not support.");
    BufferCreationInfo r{};
    if (element) {
        r.total_size_bytes = elem_count * element->size();
        r.element_stride = element->size();
    } else {
        r.total_size_bytes = elem_count;
        r.element_stride = 1;
    }
    r.native_handle = nullptr;
    r.handle = _flag++;
    SerDe::ser_value(DeviceFunc::CreateBufferAst, _send_bytes);
    SerDe::ser_value(r.handle, _send_bytes);
    SerDe::ser_value(element->description(), _send_bytes);
    SerDe::ser_value(elem_count, _send_bytes);
    _callback->async_send(_send_bytes);
    // TODO
    return r;
}
BufferCreationInfo ClientInterface::create_buffer(const ir::CArc<ir::Type> *element,
                                                  size_t elem_count,
                                                  void *external_memory /* nullptr if now imported from external memory */) noexcept {
    // TODO
    return BufferCreationInfo::make_invalid();
}
void ClientInterface::destroy_buffer(uint64_t handle) noexcept {
    _send_bytes.clear();
    SerDe::ser_value(DeviceFunc::DestroyBuffer, _send_bytes);
    SerDe::ser_value(handle, _send_bytes);
    _callback->async_send(_send_bytes);
}

// texture
ResourceCreationInfo ClientInterface::create_texture(
    PixelFormat format, uint dimension,
    uint width, uint height, uint depth,
    uint mipmap_levels, bool simultaneous_access) noexcept {
    _send_bytes.clear();
    ResourceCreationInfo r{};
    r.handle = _flag++;
    r.native_handle = nullptr;
    SerDe::ser_value(DeviceFunc::CreateTexture, _send_bytes);
    SerDe::ser_value(format, _send_bytes);
    SerDe::ser_value(dimension, _send_bytes);
    SerDe::ser_value(width, _send_bytes);
    SerDe::ser_value(height, _send_bytes);
    SerDe::ser_value(depth, _send_bytes);
    SerDe::ser_value(mipmap_levels, _send_bytes);
    SerDe::ser_value(simultaneous_access, _send_bytes);
    _callback->async_send(_send_bytes);
    return r;
}
void ClientInterface::destroy_texture(uint64_t handle) noexcept {
    _send_bytes.clear();
    SerDe::ser_value(DeviceFunc::DestroyTexture, _send_bytes);
    SerDe::ser_value(handle, _send_bytes);
    _callback->async_send(_send_bytes);
}

// bindless array
ResourceCreationInfo ClientInterface::create_bindless_array(size_t size) noexcept {
    ResourceCreationInfo r{};
    r.handle = _flag++;
    r.native_handle = nullptr;
    _send_bytes.clear();
    SerDe::ser_value(DeviceFunc::CreateBindlessArray, _send_bytes);
    SerDe::ser_value(size, _send_bytes);
    SerDe::ser_value(r.handle, _send_bytes);
    _callback->async_send(_send_bytes);
    return r;
}
void ClientInterface::destroy_bindless_array(uint64_t handle) noexcept {
    _send_bytes.clear();
    SerDe::ser_value(DeviceFunc::DestroyBindlessArray, _send_bytes);
    SerDe::ser_value(handle, _send_bytes);
    _callback->async_send(_send_bytes);
}

// stream
ResourceCreationInfo ClientInterface::create_stream(StreamTag stream_tag) noexcept {
    ResourceCreationInfo r{};
    r.handle = _flag++;
    r.native_handle = nullptr;
    _send_bytes.clear();
    SerDe::ser_value(DeviceFunc::CreateStream, _send_bytes);
    SerDe::ser_value(stream_tag, _send_bytes);
    SerDe::ser_value(r.handle, _send_bytes);
    _callback->async_send(_send_bytes);
    return r;
}
void ClientInterface::destroy_stream(uint64_t handle) noexcept {
    _send_bytes.clear();
    SerDe::ser_value(DeviceFunc::DestroyStream, _send_bytes);
    SerDe::ser_value(handle, _send_bytes);
    _callback->async_send(_send_bytes);
    {
        std::lock_guard lck{_stream_map_mtx};
        _unfinished_stream.erase(handle);
    }
}
void ClientInterface::synchronize_stream(uint64_t stream_handle) noexcept {
    while (true) {
        {
            std::lock_guard lck{_stream_map_mtx};
            auto iter = _unfinished_stream.find(stream_handle);
            if (iter == _unfinished_stream.end() || iter->second.length() == 0) {
                break;
            }
        }
        std::this_thread::yield();
    }
}
void ClientInterface::dispatch(uint64_t stream_handle, CommandList &&list) noexcept {
    DispatchFeedback feedback;
    auto cmds = list.commands();
    _send_bytes.clear();
    SerDe::ser_value(DeviceFunc::Dispatch, _send_bytes);
    SerDe::ser_value(stream_handle, _send_bytes);
    SerDe::ser_value(cmds.size(), _send_bytes);
    for (auto &cmd_base : cmds) {
        SerDe::ser_value(cmd_base->tag(), _send_bytes);
        switch (cmd_base->tag()) {
            case Command::Tag::EBufferUploadCommand: {
                auto cmd = static_cast<BufferUploadCommand const *>(cmd_base.get());
                SerDe::ser_value(cmd->handle(), _send_bytes);
                SerDe::ser_value(cmd->offset(), _send_bytes);
                SerDe::ser_value(cmd->size(), _send_bytes);
                SerDe::ser_array(
                    luisa::span<std::byte const>{
                        reinterpret_cast<std::byte const *>(cmd->data()),
                        cmd->size()},
                    _send_bytes);
            } break;
            case Command::Tag::EBufferDownloadCommand: {
                auto cmd = static_cast<BufferDownloadCommand const *>(cmd_base.get());
                SerDe::ser_value(cmd->handle(), _send_bytes);
                SerDe::ser_value(cmd->offset(), _send_bytes);
                SerDe::ser_value(cmd->size(), _send_bytes);
                feedback.readback_data.emplace_back(cmd->data());
            } break;
            case Command::Tag::EBufferCopyCommand: {
                auto cmd = static_cast<BufferCopyCommand const *>(cmd_base.get());
                SerDe::ser_value(cmd->src_handle(), _send_bytes);
                SerDe::ser_value(cmd->src_offset(), _send_bytes);
                SerDe::ser_value(cmd->dst_handle(), _send_bytes);
                SerDe::ser_value(cmd->dst_offset(), _send_bytes);
                SerDe::ser_value(cmd->size(), _send_bytes);
            } break;
            case Command::Tag::EBufferToTextureCopyCommand: {
                auto cmd = static_cast<BufferToTextureCopyCommand const *>(cmd_base.get());
                SerDe::ser_value(cmd->buffer(), _send_bytes);
                SerDe::ser_value(cmd->buffer_offset(), _send_bytes);
                SerDe::ser_value(cmd->texture(), _send_bytes);
                SerDe::ser_value(cmd->storage(), _send_bytes);
                SerDe::ser_value(cmd->level(), _send_bytes);
                SerDe::ser_value(cmd->texture_offset(), _send_bytes);
                SerDe::ser_value(cmd->size(), _send_bytes);
            } break;
            case Command::Tag::ETextureToBufferCopyCommand: {
                auto cmd = static_cast<TextureToBufferCopyCommand const *>(cmd_base.get());
                SerDe::ser_value(cmd->buffer(), _send_bytes);
                SerDe::ser_value(cmd->buffer_offset(), _send_bytes);
                SerDe::ser_value(cmd->texture(), _send_bytes);
                SerDe::ser_value(cmd->storage(), _send_bytes);
                SerDe::ser_value(cmd->level(), _send_bytes);
                SerDe::ser_value(cmd->texture_offset(), _send_bytes);
                SerDe::ser_value(cmd->size(), _send_bytes);
            } break;
            case Command::Tag::ETextureCopyCommand: {
                auto cmd = static_cast<TextureCopyCommand const *>(cmd_base.get());
                SerDe::ser_value(cmd->storage(), _send_bytes);
                SerDe::ser_value(cmd->src_handle(), _send_bytes);
                SerDe::ser_value(cmd->dst_handle(), _send_bytes);
                SerDe::ser_value(cmd->size(), _send_bytes);
                SerDe::ser_value(cmd->src_level(), _send_bytes);
                SerDe::ser_value(cmd->src_offset(), _send_bytes);
                SerDe::ser_value(cmd->dst_offset(), _send_bytes);
                SerDe::ser_value(cmd->dst_level(), _send_bytes);
            } break;
            case Command::Tag::ETextureUploadCommand: {
                auto cmd = static_cast<TextureUploadCommand const *>(cmd_base.get());
                SerDe::ser_value(cmd->handle(), _send_bytes);
                SerDe::ser_value(cmd->storage(), _send_bytes);
                SerDe::ser_value(cmd->level(), _send_bytes);
                SerDe::ser_value(cmd->size(), _send_bytes);
                SerDe::ser_value(cmd->offset(), _send_bytes);
                SerDe::ser_array(
                    luisa::span<std::byte const>{
                        reinterpret_cast<std::byte const *>(cmd->data()),
                        pixel_storage_size(cmd->storage(), cmd->size())},
                    _send_bytes);
            } break;
            case Command::Tag::ETextureDownloadCommand: {
                auto cmd = static_cast<TextureDownloadCommand const *>(cmd_base.get());
                SerDe::ser_value(cmd->handle(), _send_bytes);
                SerDe::ser_value(cmd->storage(), _send_bytes);
                SerDe::ser_value(cmd->level(), _send_bytes);
                SerDe::ser_value(cmd->size(), _send_bytes);
                SerDe::ser_value(cmd->offset(), _send_bytes);
                feedback.readback_data.emplace_back(cmd->data());
            } break;
            case Command::Tag::EMeshBuildCommand: {
                auto cmd = static_cast<MeshBuildCommand const *>(cmd_base.get());
                SerDe::ser_value(cmd->handle(), _send_bytes);
                SerDe::ser_value(cmd->request(), _send_bytes);
                SerDe::ser_value(cmd->vertex_buffer(), _send_bytes);
                SerDe::ser_value(cmd->vertex_stride(), _send_bytes);
                SerDe::ser_value(cmd->vertex_buffer_offset(), _send_bytes);
                SerDe::ser_value(cmd->vertex_buffer_size(), _send_bytes);
                SerDe::ser_value(cmd->triangle_buffer(), _send_bytes);
                SerDe::ser_value(cmd->triangle_buffer_offset(), _send_bytes);
                SerDe::ser_value(cmd->triangle_buffer_size(), _send_bytes);
            } break;
            case Command::Tag::ECurveBuildCommand: {
                auto cmd = static_cast<CurveBuildCommand const *>(cmd_base.get());
                SerDe::ser_value(cmd->handle(), _send_bytes);
                SerDe::ser_value(cmd->request(), _send_bytes);
                SerDe::ser_value(cmd->basis(), _send_bytes);
                SerDe::ser_value(cmd->cp_count(), _send_bytes);
                SerDe::ser_value(cmd->seg_count(), _send_bytes);
                SerDe::ser_value(cmd->cp_buffer(), _send_bytes);
                SerDe::ser_value(cmd->cp_buffer_offset(), _send_bytes);
                SerDe::ser_value(cmd->cp_stride(), _send_bytes);
                SerDe::ser_value(cmd->seg_buffer(), _send_bytes);
                SerDe::ser_value(cmd->seg_buffer_offset(), _send_bytes);
            } break;
            case Command::Tag::EProceduralPrimitiveBuildCommand: {
                auto cmd = static_cast<ProceduralPrimitiveBuildCommand const *>(cmd_base.get());
                SerDe::ser_value(cmd->handle(), _send_bytes);
                SerDe::ser_value(cmd->request(), _send_bytes);
                SerDe::ser_value(cmd->aabb_buffer(), _send_bytes);
                SerDe::ser_value(cmd->aabb_buffer_offset(), _send_bytes);
                SerDe::ser_value(cmd->aabb_buffer_size(), _send_bytes);
            } break;
            case Command::Tag::EAccelBuildCommand: {
                auto cmd = static_cast<AccelBuildCommand const *>(cmd_base.get());
                SerDe::ser_value(cmd->handle(), _send_bytes);
                SerDe::ser_value(cmd->request(), _send_bytes);
                SerDe::ser_value(cmd->instance_count(), _send_bytes);
                SerDe::ser_array(cmd->modifications(), _send_bytes);
                SerDe::ser_value(cmd->update_instance_buffer_only(), _send_bytes);
            } break;
            case Command::Tag::EBindlessArrayUpdateCommand: {
                auto cmd = static_cast<BindlessArrayUpdateCommand const *>(cmd_base.get());
                SerDe::ser_value(cmd->handle(), _send_bytes);
                SerDe::ser_value(cmd->modifications(), _send_bytes);
            } break;
        }
    }
    _unfinished_stream.try_emplace(stream_handle).first->second.push(std::move(feedback));
    _callback->async_send(_send_bytes);
}

void ClientInterface::set_stream_log_callback(
    uint64_t stream_handle,
    const StreamLogCallback &callback) noexcept {
    LUISA_ERROR("Set log callback not allowed in remote device.");
}

// swap chain
SwapchainCreationInfo ClientInterface::create_swapchain(
    uint64_t window_handle, uint64_t stream_handle,
    uint width, uint height, bool allow_hdr,
    bool vsync, uint back_buffer_size) noexcept {
    LUISA_ERROR("Swapchain not supported");
    return {};
}
void ClientInterface::destroy_swap_chain(uint64_t handle) noexcept {
    LUISA_ERROR("Swapchain not supported");
}
void ClientInterface::present_display_in_stream(uint64_t stream_handle, uint64_t swapchain_handle, uint64_t image_handle) noexcept {
    LUISA_ERROR("Swapchain not supported");
}
// kernel
ShaderCreationInfo ClientInterface::create_shader(const ShaderOption &option, Function kernel) noexcept {
    ShaderCreationInfo r{};
    r.handle = _flag++;
    r.block_size = kernel.block_size();
    CallableLibrary lib{};
    lib.add_callable("##", kernel.shared_builder());
    _send_bytes.clear();
    SerDe::ser_value(DeviceFunc::CreateShaderAst, _send_bytes);
    SerDe::ser_value(r.handle, _send_bytes);
    SerDe::ser_value(option.enable_cache, _send_bytes);
    SerDe::ser_value(option.enable_fast_math, _send_bytes);
    SerDe::ser_value(option.enable_debug_info, _send_bytes);
    SerDe::ser_value(option.compile_only, _send_bytes);
    SerDe::ser_value(option.max_registers, _send_bytes);
    SerDe::ser_value(option.time_trace, _send_bytes);
    SerDe::ser_value(option.name, _send_bytes);
    LUISA_ASSERT(option.native_include.empty(), "Native include not allowed in remote device.");
    auto ser_data = lib.serialize();
    SerDe::ser_array(span<std::byte const>(ser_data), _send_bytes);
    _callback->async_send(_send_bytes);
    return {};
}
ShaderCreationInfo ClientInterface::create_shader(const ShaderOption &option, const ir::KernelModule *kernel) noexcept {
    LUISA_ERROR("Not supported.");
    return {};
}
ShaderCreationInfo ClientInterface::create_shader(const ShaderOption &option, const ir_v2::KernelModule &kernel) noexcept {
    LUISA_ERROR("Not supported.");
    return {};
}
ShaderCreationInfo ClientInterface::load_shader(luisa::string_view name, luisa::span<const Type *const> arg_types) noexcept {
    return {};
}
Usage ClientInterface::shader_argument_usage(uint64_t handle, size_t index) noexcept {
    return {};
}
void ClientInterface::destroy_shader(uint64_t handle) noexcept {}

// event
ResourceCreationInfo ClientInterface::create_event() noexcept {
    return {};
}
void ClientInterface::destroy_event(uint64_t handle) noexcept {}
void ClientInterface::signal_event(uint64_t handle, uint64_t stream_handle, uint64_t fence_value) noexcept {}
void ClientInterface::wait_event(uint64_t handle, uint64_t stream_handle, uint64_t fence_value) noexcept {}
bool ClientInterface::is_event_completed(uint64_t handle, uint64_t fence_value) const noexcept {
    return {};
}
void ClientInterface::synchronize_event(uint64_t handle, uint64_t fence_value) noexcept {}

// accel
ResourceCreationInfo ClientInterface::create_mesh(const AccelOption &option) noexcept {
    return {};
}
void ClientInterface::destroy_mesh(uint64_t handle) noexcept {}

ResourceCreationInfo ClientInterface::create_procedural_primitive(const AccelOption &option) noexcept {
    return {};
}
void ClientInterface::destroy_procedural_primitive(uint64_t handle) noexcept {}

ResourceCreationInfo ClientInterface::create_curve(const AccelOption &option) noexcept {
    return {};
}
void ClientInterface::destroy_curve(uint64_t handle) noexcept {}

ResourceCreationInfo ClientInterface::create_accel(const AccelOption &option) noexcept {
    return {};
}
void ClientInterface::destroy_accel(uint64_t handle) noexcept {}

// query
luisa::string ClientInterface::query(luisa::string_view property) noexcept { return {}; }
DeviceExtension *ClientInterface::extension(luisa::string_view name) noexcept { return nullptr; }
void ClientInterface::set_name(luisa::compute::Resource::Tag resource_tag, uint64_t resource_handle, luisa::string_view name) noexcept {}

// sparse buffer
SparseBufferCreationInfo ClientInterface::create_sparse_buffer(const Type *element, size_t elem_count) noexcept { return {}; }
ResourceCreationInfo ClientInterface::allocate_sparse_buffer_heap(size_t byte_size) noexcept { return {}; }
void ClientInterface::deallocate_sparse_buffer_heap(uint64_t handle) noexcept {}
void ClientInterface::update_sparse_resources(
    uint64_t stream_handle,
    luisa::vector<SparseUpdateTile> &&textures_update) noexcept {}
void ClientInterface::destroy_sparse_buffer(uint64_t handle) noexcept {}

// sparse texture
ResourceCreationInfo ClientInterface::allocate_sparse_texture_heap(size_t byte_size, bool is_compressed_type) noexcept { return ResourceCreationInfo::make_invalid(); }
void ClientInterface::deallocate_sparse_texture_heap(uint64_t handle) noexcept {}
SparseTextureCreationInfo ClientInterface::create_sparse_texture(
    PixelFormat format, uint dimension,
    uint width, uint height, uint depth,
    uint mipmap_levels, bool simultaneous_access) noexcept {
    return SparseTextureCreationInfo::make_invalid();
}
void ClientInterface::destroy_sparse_texture(uint64_t handle) noexcept {}
}// namespace luisa::compute