#include "transient_resource_device.h"
#include <luisa/core/logging.h>
#include <luisa/backends/ext/registry.h>
#include <luisa/backends/ext/raster_cmd.h>
#include <luisa/core/logging.h>

namespace luisa::utils {
#define LUISA_NOT_IMPL_RET   \
    LUISA_NOT_IMPLEMENTED(); \
    vstd::unreachable();     \
    return {}
TransientResourceDevice::TransientResourceDevice(Context &&ctx, DeviceInterface *impl)
    : DeviceInterface{std::move(ctx)}, _impl(impl), _tex_meta_pool(256), _buffer_meta_pool(256), _tex_handle_pool(256), _buffer_fit(std::numeric_limits<uint64_t>::max(), 16) {
}
TransientResourceDevice::~TransientResourceDevice() {
    for (auto &i : _native_resources) {
        _impl->destroy_texture(i->res_info.handle);
    }
    if (_transient_buffer_handle != invalid_resource_handle) {
        _impl->destroy_buffer(_transient_buffer_handle);
    }
}

auto TransientResourceDevice::_allocate_handle(TransientTexDesc const &desc) -> TexResourceHandle * {
    auto res_handle = _tex_handle_pool.create();
    res_handle->res_info = _impl->create_texture(
        desc.format, desc.dimension, desc.width, desc.height, desc.depth, desc.mipmap_levels, nullptr, desc.simultaneous_access(), desc.allow_raster_target());
    res_handle->index = _native_resources.size();
    res_handle->last_frame = _frame_index;
    _native_resources.emplace_back(res_handle);
    return res_handle;
}
void TransientResourceDevice::_deallocate_handle(TexResourceHandle *handle) {
    _tex_dispose_queue.emplace_back(handle->res_info.handle);
    auto &v = _native_resources[handle->index];
    if (handle->index != _native_resources.size() - 1) {
        v = _native_resources.back();
        v->index = handle->index;
    }
    _native_resources.pop_back();
    _tex_handle_pool.destroy(handle);
}

uint64_t TransientResourceDevice::_get_tex_handle(uint64_t handle) {
    auto iter = _tex_desc_to_native.find(reinterpret_cast<TransientTexDesc *>(handle));
    if (!iter) {
        return handle;
    }
    auto v = iter.value().handle->res_info.handle;
    return v;
}

std::pair<uint64_t, size_t> TransientResourceDevice::_get_buffer_handle_offset(uint64_t handle) {
    auto iter = _buffer_desc_to_native.find(reinterpret_cast<size_t *>(handle));
    if (!iter) {
        return {handle, 0};
    }
    return {_transient_buffer_handle, iter.value().offset};
}

void *TransientResourceDevice::get_native_handle(uint64_t handle) {
#ifndef NDEBUG
    if (!_is_committing) [[unlikely]] {
        LUISA_ERROR("Acquiring native handle out of commit scope.");
    }
#endif
    auto iter = _tex_desc_to_native.find(reinterpret_cast<TransientTexDesc *>(handle));
    if (!iter) [[unlikely]] {
        return nullptr;
    }
    return iter.value().handle->res_info.native_handle;
}

void TransientResourceDevice::_mark_tex(uint64_t handle, uint64_t command_index) {
    auto iter = _tex_desc_to_native.find(reinterpret_cast<TransientTexDesc *>(handle));
    if (!iter) return;
    auto &v = iter.value();
    v.start_command_index = std::min<int64_t>(v.start_command_index, command_index);
    v.end_command_index = std::max<int64_t>(v.end_command_index, command_index);
}

void TransientResourceDevice::_mark_buffer(uint64_t handle, uint64_t command_index) {
    auto iter = _buffer_desc_to_native.find(reinterpret_cast<size_t *>(handle));
    if (!iter) return;
    auto &v = iter.value();
    v.start_command_index = std::min<int64_t>(v.start_command_index, command_index);
    v.end_command_index = std::max<int64_t>(v.end_command_index, command_index);
}

void TransientResourceDevice::_preprocess(luisa::span<luisa::unique_ptr<Command> const> commands, CommandList &cmdlist) {
    for (uint64_t idx = 0; idx < commands.size(); ++idx) {
        auto cmd = commands[idx].get();
        switch (cmd->tag()) {

            case Command::Tag::EBufferToTextureCopyCommand: {
                auto c = static_cast<BufferToTextureCopyCommand const *>(cmd);
                _mark_buffer(c->buffer(), idx);
                _mark_tex(c->texture(), idx);
            } break;
            case Command::Tag::EShaderDispatchCommand: {
                auto c = static_cast<ShaderDispatchCommand const *>(cmd);
                for (auto &i : c->arguments()) {
                    switch (i.tag) {
                        case Argument::Tag::BUFFER:
                            _mark_buffer(i.buffer.handle, idx);
                            break;
                        case Argument::Tag::TEXTURE:
                            _mark_tex(i.texture.handle, idx);
                            break;
                    }
                }
            } break;
            case Command::Tag::ETextureUploadCommand: {
                auto c = static_cast<TextureUploadCommand const *>(cmd);
                _mark_tex(c->handle(), idx);
            } break;
            case Command::Tag::ETextureDownloadCommand: {
                auto c = static_cast<TextureUploadCommand const *>(cmd);
                _mark_tex(c->handle(), idx);
            } break;
            case Command::Tag::ETextureCopyCommand: {
                auto c = static_cast<TextureCopyCommand const *>(cmd);
                _mark_tex(c->src_handle(), idx);
                _mark_tex(c->dst_handle(), idx);
            } break;
            case Command::Tag::ETextureToBufferCopyCommand: {
                auto c = static_cast<TextureToBufferCopyCommand const *>(cmd);
                _mark_tex(c->texture(), idx);
                _mark_buffer(c->buffer(), idx);
            } break;
            case Command::Tag::EBufferUploadCommand: {
                auto c = static_cast<BufferUploadCommand const *>(cmd);
                _mark_buffer(c->handle(), idx);
            } break;
            case Command::Tag::EBufferDownloadCommand: {
                auto c = static_cast<BufferDownloadCommand const *>(cmd);
                _mark_buffer(c->handle(), idx);
            } break;
            case Command::Tag::EBufferCopyCommand: {
                auto c = static_cast<BufferCopyCommand const *>(cmd);
                _mark_buffer(c->src_handle(), idx);
                _mark_buffer(c->dst_handle(), idx);
            } break;
            case Command::Tag::EMeshBuildCommand: {
                auto c = static_cast<MeshBuildCommand const *>(cmd);
                _mark_buffer(c->vertex_buffer(), idx);
                _mark_buffer(c->triangle_buffer(), idx);
            } break;
            case Command::Tag::EProceduralPrimitiveBuildCommand: {
                auto c = static_cast<ProceduralPrimitiveBuildCommand const *>(cmd);
                _mark_buffer(c->aabb_buffer(), idx);
            } break;
            case Command::Tag::EBindlessArrayUpdateCommand:
            case Command::Tag::EAccelBuildCommand:
            case Command::Tag::ECurveBuildCommand:
            case Command::Tag::EMotionInstanceBuildCommand:
                break;
            case Command::Tag::ECustomCommand: {
                switch (static_cast<CustomCommand const *>(cmd)->custom_cmd_uuid()) {
                    case to_underlying(CustomCommandUUID::RASTER_CLEAR_DEPTH): {
                        auto c = static_cast<ClearDepthCommand const *>(cmd);
                        _mark_tex(c->handle(), idx);
                    } break;
                    case to_underlying(CustomCommandUUID::RASTER_DRAW_SCENE): {
                        auto c = static_cast<DrawRasterSceneCommand const *>(cmd);
                        for (auto &i : c->arguments()) {
                            if (i.tag == Argument::Tag::TEXTURE) {
                                _mark_tex(i.texture.handle, idx);
                            }
                        }
                        for (auto &i : c->rtv_texs()) {
                            _mark_tex(i.handle, idx);
                        }
                        _mark_tex(c->dsv_tex().handle, idx);
                    } break;
                    case to_underlying(CustomCommandUUID::CUSTOM_DISPATCH): {
                        auto c = static_cast<CustomDispatchCommand const *>(cmd);
                        c->traverse_arguments([&]<typename T>(T const &t, auto usage) {
                            if constexpr (std::is_same_v<T, Argument::Buffer>) {
                                _mark_buffer(t.handle, idx);
                            } else if constexpr (std::is_same_v<T, Argument::Texture>) {
                                _mark_tex(t.handle, idx);
                            }
                        });
                    } break;
                    default:
                        LUISA_ERROR("Unsupported command.");
                }
            } break;
            default: {
                LUISA_ERROR("Unsupported command.");
            } break;
        }
    }
    _buffer_fit.clean_all();
    _command_caches.clear();
    _command_caches.resize(commands.size());
    for (auto &i : _tex_desc_to_native) {
        auto &v = i.second;
        if (v.start_command_index != std::numeric_limits<int64_t>::max() && v.end_command_index != std::numeric_limits<int64_t>::min()) {
            _command_caches[v.start_command_index]._allocate_tex.emplace_back(i.first, &i.second);
            _command_caches[v.end_command_index]._deallocate_tex.emplace_back(i.first, &i.second);
        }
    }
    for (auto &i : _buffer_desc_to_native) {
        auto &v = i.second;
        if (v.start_command_index != std::numeric_limits<int64_t>::max() && v.end_command_index != std::numeric_limits<int64_t>::min()) {
            _command_caches[v.start_command_index]._allocate_buffer.emplace_back(i.first, &i.second);
            _command_caches[v.end_command_index]._deallocate_buffer.emplace_back(i.first, &i.second);
        }
    }
    size_t buffer_size = 0;
    for (auto &caches : _command_caches) {
        for (auto alloc : caches._allocate_tex) {
            auto iter = _ready_texs.find(*alloc.first);
            auto managed_desc = alloc.second;
            if (iter && (!iter.value().empty())) {
                auto &vec = iter.value();
                auto res = vec.back();
                res->last_frame = _frame_index;
                vec.pop_back();
                managed_desc->handle = res;
            } else {
                managed_desc->handle = _allocate_handle(*alloc.first);
            }
        }
        for (auto dealloc : caches._deallocate_tex) {
            auto iter = _ready_texs.emplace(*dealloc.first);
            auto managed_desc = dealloc.second;
            LUISA_DEBUG_ASSERT(managed_desc->handle);
            iter.value().emplace_back(managed_desc->handle);
        }
        for (auto alloc : caches._allocate_buffer) {
            auto node = _buffer_fit.allocate_best_fit(*alloc.first);
            alloc.second->_node = node;
            alloc.second->offset = node->offset();
            buffer_size = std::max<size_t>(buffer_size, node->offset() + node->size());
        }
        for (auto dealloc : caches._deallocate_buffer) {
            _buffer_fit.free(dealloc.second->_node);
        }
    }
    buffer_size = (buffer_size + 65535ull) & (~65535ull);
    if (buffer_size > 0 && ((buffer_size > _transient_buffer_size || buffer_size <= _transient_buffer_size / 2))) {
        _transient_buffer_size = buffer_size;
        if (_transient_buffer_handle != invalid_resource_handle) {
            cmdlist.add_callback([handle = _transient_buffer_handle, impl = this->_impl]() {
                impl->destroy_buffer(handle);
            });
        }
        _transient_buffer_handle = _impl->create_buffer(Type::of<uint4>(), buffer_size / sizeof(uint4), nullptr).handle;
    }
    // steal command handles
    for (uint64_t idx = 0; idx < commands.size(); ++idx) {
        auto cmd = commands[idx].get();
        switch (cmd->tag()) {
            case Command::Tag::EBufferToTextureCopyCommand: {
                auto c = static_cast<BufferToTextureCopyCommand *>(cmd);
                c->set_texture_handle(_get_tex_handle(c->texture()));
                auto bf = _get_buffer_handle_offset(c->buffer());
                c->set_buffer_handle(bf.first);
                c->set_buffer_offset(c->buffer_offset() + bf.second);
            } break;
            case Command::Tag::EShaderDispatchCommand: {
                auto c = static_cast<ShaderDispatchCommand *>(cmd);
                for (auto &i : c->arguments()) {
                    switch (i.tag) {
                        case Argument::Tag::BUFFER: {
                            auto bf = _get_buffer_handle_offset(i.buffer.handle);
                            i.buffer.handle = bf.first;
                            i.buffer.offset += bf.second;
                        } break;
                        case Argument::Tag::TEXTURE: {
                            i.texture.handle = _get_tex_handle(i.texture.handle);
                        } break;
                    }
                }
            } break;
            case Command::Tag::ETextureUploadCommand: {
                auto c = static_cast<TextureUploadCommand *>(cmd);
                c->set_handle(_get_tex_handle(c->handle()));
            } break;
            case Command::Tag::ETextureDownloadCommand: {
                auto c = static_cast<TextureDownloadCommand *>(cmd);
                c->set_handle(_get_tex_handle(c->handle()));
            } break;
            case Command::Tag::ETextureCopyCommand: {
                auto c = static_cast<TextureCopyCommand *>(cmd);
                c->set_src_handle(_get_tex_handle(c->src_handle()));
                c->set_dst_handle(_get_tex_handle(c->dst_handle()));
            } break;
            case Command::Tag::ETextureToBufferCopyCommand: {
                auto c = static_cast<TextureToBufferCopyCommand *>(cmd);
                c->set_texture_handle(_get_tex_handle(c->texture()));
                auto bf = _get_buffer_handle_offset(c->buffer());
                c->set_buffer_handle(bf.first);
                c->set_buffer_offset(bf.second + c->buffer_offset());
            } break;
            case Command::Tag::EBufferUploadCommand: {
                auto c = static_cast<BufferUploadCommand *>(cmd);
                auto bf = _get_buffer_handle_offset(c->handle());
                c->set_handle(bf.first);
                c->set_offset(c->offset() + bf.second);
            } break;
            case Command::Tag::EBufferDownloadCommand: {
                auto c = static_cast<BufferDownloadCommand *>(cmd);
                _mark_buffer(c->handle(), idx);
                auto bf = _get_buffer_handle_offset(c->handle());
                c->set_handle(bf.first);
                c->set_offset(c->offset() + bf.second);
            } break;
            case Command::Tag::EBufferCopyCommand: {
                auto c = static_cast<BufferCopyCommand *>(cmd);
                _mark_buffer(c->src_handle(), idx);
                _mark_buffer(c->dst_handle(), idx);
                auto bf = _get_buffer_handle_offset(c->src_handle());
                c->set_src_handle(bf.first);
                c->set_src_offset(c->src_offset() + bf.second);

                bf = _get_buffer_handle_offset(c->dst_handle());
                c->set_dst_handle(bf.first);
                c->set_dst_offset(c->dst_offset() + bf.second);
            } break;
            case Command::Tag::EMeshBuildCommand: {
                auto c = static_cast<MeshBuildCommand *>(cmd);
                auto bf = _get_buffer_handle_offset(c->vertex_buffer());
                c->set_vertex_buffer(bf.first);
                c->set_vertex_offset(c->vertex_buffer_offset() + bf.second);

                bf = _get_buffer_handle_offset(c->triangle_buffer());
                c->set_triangle_buffer(bf.first);
                c->set_triangle_offset(c->triangle_buffer_offset() + bf.second);
            } break;
            case Command::Tag::EProceduralPrimitiveBuildCommand: {
                auto c = static_cast<ProceduralPrimitiveBuildCommand *>(cmd);
                auto bf = _get_buffer_handle_offset(c->aabb_buffer());
                c->set_aabb_buffer(bf.first);
                c->set_aabb_buffer_offset(c->aabb_buffer_offset() + bf.second);
            } break;
            case Command::Tag::ECustomCommand: {
                switch (static_cast<CustomCommand *>(cmd)->custom_cmd_uuid()) {
                    case to_underlying(CustomCommandUUID::RASTER_CLEAR_DEPTH): {
                        auto c = static_cast<ClearDepthCommand *>(cmd);
                    } break;
                    case to_underlying(CustomCommandUUID::RASTER_DRAW_SCENE): {
                        auto c = static_cast<DrawRasterSceneCommand *>(cmd);
                        for (auto &i : c->arguments()) {
                            if (i.tag == Argument::Tag::TEXTURE) {
                                i.texture.handle = _get_tex_handle(i.texture.handle);
                            }
                        }
                        auto rtv_texs = c->rtv_texs();
                        for (auto &i : rtv_texs) {
                            i.handle = _get_tex_handle(i.handle);
                        }
                        auto dst_tex = c->dsv_tex();
                        dst_tex.handle = _get_tex_handle(dst_tex.handle);
                        c->set_dsv_texs(dst_tex);
                    } break;
                    case to_underlying(CustomCommandUUID::CUSTOM_DISPATCH): {
                        auto c = static_cast<CustomDispatchCommand *>(cmd);
                        c->traverse_arguments([&]<typename T>(T &t, auto usage) {
                            if constexpr (std::is_same_v<T, Argument::Buffer>) {
                                auto bf = _get_buffer_handle_offset(t.handle);
                                t.handle = bf.first;
                                t.offset += bf.second;
                            } else if constexpr (std::is_same_v<T, Argument::Texture>) {
                                t.handle = _get_tex_handle(t.handle);
                            }
                        });
                    } break;
                    default:
                        break;
                }
            } break;
            default:
                break;
        }
    }
}

BufferCreationInfo TransientResourceDevice::create_buffer(const ir::CArc<ir::Type> *element, size_t elem_count, void *external_memory /* nullptr if not imported from external memory */) noexcept {
    LUISA_NOT_IMPL_RET;
}
BufferCreationInfo TransientResourceDevice::create_buffer(const Type *element, size_t elem_count, void *external_memory /* nullptr if not imported from external memory */) noexcept {
    if (_temp_name.empty()) [[unlikely]] {
        LUISA_ERROR("Texture must have name, call set_next_res_name first.");
    }
    if (!_is_managing) [[unlikely]] {
        LUISA_ERROR("Not in the managing scope, you can not allocate a texture .");
    }
    BufferCreationInfo info;
    info.element_stride = element->size();
    info.total_size_bytes = info.element_stride * elem_count;
    auto iter = _buffer_name_to_desc.try_emplace(std::move(_temp_name), vstd::lazy_eval([&]() {
                                                     auto ptr = _buffer_meta_pool.create(info.total_size_bytes);
                                                     _buffer_desc_to_native.emplace(ptr);
                                                     return ptr;
                                                 }));
    luisa::string_view name_view{iter.first.key()};
    if (!iter.second) {
        auto src = iter.first.value();
        if (*src != info.total_size_bytes) [[unlikely]] {
            LUISA_ERROR("Trying to acquire buffer {} with different size.", name_view);
        }
    }
    info.handle = reinterpret_cast<uint64_t>(iter.first.value());
    info.native_handle = nullptr;
    return info;
}
// texture
ResourceCreationInfo TransientResourceDevice::create_texture(
    PixelFormat format, uint dimension,
    uint width, uint height, uint depth,
    uint mipmap_levels, void *external_native_handle,
    bool simultaneous_access, bool allow_raster_target) noexcept {
    if (_temp_name.empty()) [[unlikely]] {
        LUISA_ERROR("Texture must have name, call set_next_res_name first.");
    }
    if (!_is_managing) [[unlikely]] {
        LUISA_ERROR("Not in the managing scope, you can not allocate a texture .");
    }
    TransientTexDesc desc{
        .format = format,
        .dimension = dimension,
        .width = width,
        .height = height,
        .depth = depth,
        .mipmap_levels = mipmap_levels,
        .mask = 0};
    desc.set_simultaneous_access(simultaneous_access);
    desc.set_allow_raster_target(allow_raster_target);
    auto iter =
        _tex_name_to_desc.try_emplace(std::move(_temp_name), vstd::lazy_eval([&]() {
                                          auto ptr = _tex_meta_pool.create(desc);
                                          _tex_desc_to_native.emplace(ptr);
                                          return ptr;
                                      }));
    luisa::string_view name_view{iter.first.key()};
    if (!iter.second) {
        auto src = iter.first.value();
        if (std::memcmp(src, &desc, sizeof(TransientTexDesc)) != 0) [[unlikely]] {
            LUISA_ERROR("Trying to acquire texture {} with different description.", name_view);
        }
    }
    ResourceCreationInfo info;
    info.handle = reinterpret_cast<uint64_t>(iter.first.value());
    info.native_handle = nullptr;
    _temp_name.clear();
    return info;
}
void TransientResourceDevice::begin_managing(CommandList const &cmdlist) {
    LUISA_ASSERT(!_is_managing, "Managing already begin.");
    _is_managing = true;
    managing_cmd_range.first = cmdlist.commands().size();
}
void TransientResourceDevice::dispatch(uint64_t stream_handle, CommandList &&list) noexcept {
    LUISA_ASSERT(_is_managing, "begin_managing not called before dispatch.");
    _is_managing = false;
    managing_cmd_range.second = list.commands().size() - managing_cmd_range.first;
    auto commands = list.commands().subspan(managing_cmd_range.first, managing_cmd_range.second);
    _preprocess(commands, list);
    for (auto &i : _ready_texs) {
        auto &texs = i.second;
        for (size_t idx = 0; idx < texs.size(); ++idx) {
            if (texs[idx]->last_frame - _frame_index <= resource_contain_frame) continue;
            auto &v = texs[idx];
            _deallocate_handle(v);
            if (idx != texs.size() - 1) {
                v = texs.back();
            }
            texs.pop_back();
            --idx;
        }
    }
    if (!_tex_dispose_queue.empty()) {
        list.add_callback([impl = _impl, dsp = std::move(_tex_dispose_queue)]() {
            for (auto &i : dsp) {
                impl->destroy_texture(i);
            }
        });
    }
    _is_committing = true;
    _impl->dispatch(stream_handle, std::move(list));
    _is_committing = false;
    // finalize
    managing_cmd_range = {
        std::numeric_limits<uint64_t>::max(),
        std::numeric_limits<uint64_t>::max()};
    for (auto &i : _tex_desc_to_native) {
        _tex_meta_pool.destroy(i.first);
    }
    if (dump_func) {
        luisa::string dump;
        for (auto &i : _buffer_name_to_desc) {
            auto iter = _buffer_desc_to_native.find(i.second);
            LUISA_DEBUG_ASSERT(iter);
            dump += luisa::format("Buffer {} has sub-range [{}, {}] from command index {} to {}\n", i.first, iter.value().offset, iter.value().offset + *i.second, iter.value().start_command_index, iter.value().end_command_index);
        }
        for (auto &i : _tex_name_to_desc) {
            auto iter = _tex_desc_to_native.find(i.second);
            LUISA_DEBUG_ASSERT(iter);
            dump += luisa::format("Texture {}'s physical resource handle 0x{:X} from command index {} to {}\n", i.first, iter.value().handle->res_info.handle, iter.value().start_command_index, iter.value().end_command_index);
        }
        dump_func(std::move(dump));
    }
    _tex_desc_to_native.clear();
    _tex_name_to_desc.clear();
    _buffer_name_to_desc.clear();
    _temp_name.clear();
    _buffer_meta_pool.destroy_all();
    _buffer_desc_to_native.clear();
    ++_frame_index;
    list.clear();
}

//////////////// Others
void *TransientResourceDevice::native_handle() const noexcept {
    return _impl->native_handle();
}
uint TransientResourceDevice::compute_warp_size() const noexcept {
    return _impl->compute_warp_size();
}
uint64_t TransientResourceDevice::memory_granularity() const noexcept {
    return _impl->memory_granularity();
}
void TransientResourceDevice::destroy_texture(uint64_t handle) noexcept {
    // Managed resource can not be free manually
}
void TransientResourceDevice::destroy_buffer(uint64_t handle) noexcept {
    // Managed resource can not be free manually
}
// bindless array
ResourceCreationInfo TransientResourceDevice::create_bindless_array(size_t size, BindlessSlotType type) noexcept {
    LUISA_NOT_IMPL_RET;
}
void TransientResourceDevice::destroy_bindless_array(uint64_t handle) noexcept {
    LUISA_NOT_IMPLEMENTED();
}

// stream
ResourceCreationInfo TransientResourceDevice::create_stream(StreamTag stream_tag) noexcept {
    LUISA_NOT_IMPL_RET;
}
void TransientResourceDevice::destroy_stream(uint64_t handle) noexcept {
    LUISA_NOT_IMPLEMENTED();
}
void TransientResourceDevice::synchronize_stream(uint64_t stream_handle) noexcept {
    LUISA_NOT_IMPLEMENTED();
}

using StreamLogCallback = luisa::function<void(luisa::string_view)>;
void TransientResourceDevice::set_stream_log_callback(uint64_t stream_handle, const StreamLogCallback &callback) noexcept {
    LUISA_NOT_IMPLEMENTED();
}

// swap chain
SwapchainCreationInfo TransientResourceDevice::create_swapchain(const SwapchainOption &option, uint64_t stream_handle) noexcept {
    LUISA_NOT_IMPL_RET;
}
void TransientResourceDevice::destroy_swap_chain(uint64_t handle) noexcept {
    LUISA_NOT_IMPLEMENTED();
}
void TransientResourceDevice::present_display_in_stream(uint64_t stream_handle, uint64_t swapchain_handle, uint64_t image_handle) noexcept {
    LUISA_NOT_IMPLEMENTED();
}

// kernel
ShaderCreationInfo TransientResourceDevice::create_shader(const ShaderOption &option, Function kernel) noexcept {
    LUISA_NOT_IMPL_RET;
}
ShaderCreationInfo TransientResourceDevice::create_shader(const ShaderOption &option, const ir::KernelModule *kernel) noexcept {
    LUISA_NOT_IMPL_RET;
}
ShaderCreationInfo TransientResourceDevice::create_shader(const ShaderOption &option, const ir_v2::KernelModule &kernel) noexcept {
    LUISA_NOT_IMPL_RET;
}
ShaderCreationInfo TransientResourceDevice::load_shader(luisa::string_view name, luisa::span<const Type *const> arg_types) noexcept {
    LUISA_NOT_IMPL_RET;
}
Usage TransientResourceDevice::shader_argument_usage(uint64_t handle, size_t index) noexcept {
    LUISA_NOT_IMPL_RET;
}
void TransientResourceDevice::destroy_shader(uint64_t handle) noexcept {
    LUISA_NOT_IMPLEMENTED();
}

// event
ResourceCreationInfo TransientResourceDevice::create_event() noexcept {
    LUISA_NOT_IMPL_RET;
}
void TransientResourceDevice::destroy_event(uint64_t handle) noexcept {
    LUISA_NOT_IMPLEMENTED();
}
void TransientResourceDevice::signal_event(uint64_t handle, uint64_t stream_handle, uint64_t fence_value) noexcept {
    LUISA_NOT_IMPLEMENTED();
}
void TransientResourceDevice::wait_event(uint64_t handle, uint64_t stream_handle, uint64_t fence_value) noexcept {
    LUISA_NOT_IMPLEMENTED();
}
bool TransientResourceDevice::is_event_completed(uint64_t handle, uint64_t fence_value) const noexcept {
    LUISA_NOT_IMPLEMENTED();
}
void TransientResourceDevice::synchronize_event(uint64_t handle, uint64_t fence_value) noexcept {
    LUISA_NOT_IMPLEMENTED();
}

// accel
ResourceCreationInfo TransientResourceDevice::create_mesh(const AccelOption &option) noexcept {
    LUISA_NOT_IMPL_RET;
}
void TransientResourceDevice::destroy_mesh(uint64_t handle) noexcept {
    LUISA_NOT_IMPLEMENTED();
}

ResourceCreationInfo TransientResourceDevice::create_procedural_primitive(const AccelOption &option) noexcept {
    LUISA_NOT_IMPL_RET;
}
void TransientResourceDevice::destroy_procedural_primitive(uint64_t handle) noexcept {
    LUISA_NOT_IMPLEMENTED();
}

ResourceCreationInfo TransientResourceDevice::create_curve(const AccelOption &option) noexcept {
    LUISA_NOT_IMPL_RET;
}
void TransientResourceDevice::destroy_curve(uint64_t handle) noexcept {
    LUISA_NOT_IMPLEMENTED();
}

ResourceCreationInfo TransientResourceDevice::create_motion_instance(const AccelMotionOption &option) noexcept {
    LUISA_NOT_IMPL_RET;
}
void TransientResourceDevice::destroy_motion_instance(uint64_t handle) noexcept {
    LUISA_NOT_IMPLEMENTED();
}

ResourceCreationInfo TransientResourceDevice::create_accel(const AccelOption &option) noexcept {
    LUISA_NOT_IMPL_RET;
}
void TransientResourceDevice::destroy_accel(uint64_t handle) noexcept {
    LUISA_NOT_IMPLEMENTED();
}

// query
luisa::string TransientResourceDevice::query(luisa::string_view property) noexcept {
    LUISA_NOT_IMPL_RET;
}
DeviceExtension *TransientResourceDevice::extension(luisa::string_view name) noexcept {
    LUISA_NOT_IMPL_RET;
}
void TransientResourceDevice::set_name(luisa::compute::Resource::Tag resource_tag, uint64_t resource_handle, luisa::string_view name) noexcept {
    LUISA_NOT_IMPLEMENTED();
}
luisa::string_view TransientResourceDevice::get_name(uint64_t resource_handle) const noexcept {
    LUISA_NOT_IMPL_RET;
}

// sparse buffer
SparseBufferCreationInfo TransientResourceDevice::create_sparse_buffer(const Type *element, size_t elem_count) noexcept {
    LUISA_NOT_IMPL_RET;
}
ResourceCreationInfo TransientResourceDevice::allocate_sparse_buffer_heap(size_t byte_size) noexcept {
    LUISA_NOT_IMPL_RET;
}
void TransientResourceDevice::deallocate_sparse_buffer_heap(uint64_t handle) noexcept {
    LUISA_NOT_IMPLEMENTED();
}
void TransientResourceDevice::update_sparse_resources(
    uint64_t stream_handle,
    luisa::vector<SparseUpdateTile> &&textures_update) noexcept {
    LUISA_NOT_IMPLEMENTED();
}
void TransientResourceDevice::destroy_sparse_buffer(uint64_t handle) noexcept {
    LUISA_NOT_IMPLEMENTED();
}

// sparse texture
ResourceCreationInfo TransientResourceDevice::allocate_sparse_texture_heap(size_t byte_size) noexcept {
    LUISA_NOT_IMPL_RET;
}
void TransientResourceDevice::deallocate_sparse_texture_heap(uint64_t handle) noexcept {}
SparseTextureCreationInfo TransientResourceDevice::create_sparse_texture(
    PixelFormat format, uint dimension,
    uint width, uint height, uint depth,
    uint mipmap_levels, bool simultaneous_access) noexcept {
    LUISA_NOT_IMPL_RET;
}
void TransientResourceDevice::destroy_sparse_texture(uint64_t handle) noexcept {
    LUISA_NOT_IMPLEMENTED();
}
TransientResourceDeviceScope::TransientResourceDeviceScope(
    compute::Stream &stream,
    compute::Device &transient_res_device,
    bool log_info) : stream(stream), transient_res_device(transient_res_device) {
    auto device = static_cast<TransientResourceDevice *>(transient_res_device.impl());
    device->begin_managing(cmdlist);
    if (log_info) {
        device->dump_func = [&](luisa::string &&str) {
            LUISA_INFO("\n{}", str);
        };
    }
}
}// namespace luisa::utils
#undef LUISA_NOT_IMPL_RET