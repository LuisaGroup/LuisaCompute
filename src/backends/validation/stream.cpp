#include "device.h"
#include "event.h"
#include "stream.h"
#include "accel.h"
#include "buffer.h"
#include "texture.h"
#include "depth_buffer.h"
#include "bindless_array.h"
#include "mesh.h"
#include "procedural_primitives.h"
#include "shader.h"
#include "swap_chain.h"
#include <core/logging.h>
#include <runtime/raster/raster_scene.h>
namespace lc::validation {
Stream::Stream(uint64_t handle, StreamTag stream_tag) : Resource{handle, Tag::STREAM}, _stream_tag{stream_tag} {}
void Stream::signal(Event *evt) {
    evt->signaled.force_emplace(this, _executed_layer);
}
uint64_t Stream::stream_synced_frame(Stream const *stream) const {
    auto iter = waited_stream.find(stream);
    if (iter == waited_stream.end()) {
        return stream->synced_layer();
    } else {
        return iter->second;
    }
}
void Stream::wait(Event *evt) {
    for (auto &&i : evt->signaled) {
        waited_stream.force_emplace(i.first, i.second);
    }
}
namespace detail {
static vstd::string usage_name(Usage usage) {
    switch (usage) {
        case Usage::READ:
            return "read";
        case Usage::READ_WRITE:
            return "read and written";
        case Usage::WRITE:
            return "written";
        default:
            return "none";
    }
}
}// namespace detail
void Stream::check_compete() {
    for (auto &&iter : res_usages) {
        auto res = iter.first;
        for (auto &&stream_iter : res->info()) {
            auto other_stream = stream_iter.first;
            if (other_stream == this) continue;
            auto synced_frame = stream_synced_frame(other_stream);
            if (stream_iter.second.last_frame > synced_frame) {
                // Texture type
                if (res->non_simultaneous()) {
                    LUISA_ERROR(
                        "Non-simultaneous resource \"{}\" is not allowed to be {} by \"{}\" and {} by \"{}\" simultaneously.",
                        res->get_name(),
                        detail::usage_name(stream_iter.second.usage),
                        other_stream->get_name(),
                        detail::usage_name(iter.second),
                        get_name());
                }
                // others, buffer, etc
                else if ((luisa::to_underlying(stream_iter.second.usage) & luisa::to_underlying(iter.second) & luisa::to_underlying(Usage::WRITE)) != 0) {
                    LUISA_ERROR(
                        "Resource \"{}\" is not allowed to be {} by \"{}\" and {} by \"{}\" simultaneously.",
                        res->get_name(),
                        detail::usage_name(stream_iter.second.usage),
                        other_stream->get_name(),
                        detail::usage_name(iter.second),
                        get_name());
                }
            }
        }
    }
}
void Stream::dispatch() {
    _executed_layer++;
    res_usages.clear();
}
void Stream::mark_shader_dispatch(DeviceInterface *dev, ShaderDispatchCommandBase *cmd, bool contain_bindings) {
    size_t arg_idx = 0;
    auto shader = reinterpret_cast<RWResource *>(cmd->handle());
    auto native_shader = shader->handle();
    auto mark_handle = [&](uint64_t &handle) {
        auto res = reinterpret_cast<RWResource *>(handle);
        res->set(this, dev->shader_arg_usage(native_shader, arg_idx));
        handle = res->handle();
    };
    auto set_arg = [&](Argument &arg) {
        switch (arg.tag) {
            case Argument::Tag::BUFFER: {
                mark_handle(arg.buffer.handle);
            } break;
            case Argument::Tag::TEXTURE: {
                mark_handle(arg.texture.handle);
            } break;
            case Argument::Tag::BINDLESS_ARRAY: {
                mark_handle(arg.bindless_array.handle);
            } break;
            case Argument::Tag::ACCEL: {
                mark_handle(arg.accel.handle);
            } break;
        }
        ++arg_idx;
    };
    if (contain_bindings) {
        for (auto &&i : static_cast<Shader *>(shader)->bound_arguments()) {
            auto arg = luisa::visit(
                [&]<typename T>(T const &a) -> Argument {
                    Argument arg;
                    if constexpr (std::is_same_v<T, Function::BufferBinding>) {
                        arg.tag = Argument::Tag::BUFFER;
                        arg.buffer = a;
                    } else if constexpr (std::is_same_v<T, Function::TextureBinding>) {
                        arg.tag = Argument::Tag::TEXTURE;
                        arg.texture = a;
                    } else if constexpr (std::is_same_v<T, Function::BindlessArrayBinding>) {
                        arg.tag = Argument::Tag::BINDLESS_ARRAY;
                        arg.bindless_array = a;
                    } else if constexpr (std::is_same_v<T, Function::AccelBinding>) {
                        arg.tag = Argument::Tag::ACCEL;
                        arg.accel = a;
                    } else {
                        LUISA_ERROR("Binding Contain unwanted variable.");
                    }
                    return arg;
                },
                i);
            set_arg(arg);
        }
    }
    for (auto &&i : cmd->arguments()) {
        set_arg(const_cast<Argument &>(i));
    }
    swap_handle(cmd->_handle, Usage::READ);
}
void Stream::swap_handle(uint64_t &v, Usage usage) {
    reinterpret_cast<RWResource *>(v)->set(this, usage);
    v = reinterpret_cast<RWResource *>(v)->handle();
};
void Stream::custom(DeviceInterface *dev, Command *cmd) {
    switch (static_cast<CustomCommand *>(cmd)->uuid()) {
        case clear_depth_command_uuid: {
            auto c = static_cast<ClearDepthCommand *>(cmd);
            swap_handle(c->_handle, Usage::WRITE);
        } break;
        case draw_raster_command_uuid: {
            auto c = static_cast<DrawRasterSceneCommand *>(cmd);
            mark_shader_dispatch(dev, c, false);
            if (c->_dsv_tex.handle != invalid_resource_handle) {
                swap_handle(c->_dsv_tex.handle, Usage::READ_WRITE);
            }
            for (auto i : vstd::range(c->_rtv_count)) {
                swap_handle(c->_rtv_texs[i].handle, Usage::WRITE);
            }
            for (auto &&i : c->_scene) {
                for (auto &&vb : i._vertex_buffers) {
                    swap_handle(const_cast<uint64_t &>(vb._handle), Usage::READ);
                }
                luisa::visit(
                    [&]<typename T>(T &t) {
                        if constexpr (std::is_same_v<T, BufferView<uint>>) {
                            swap_handle(t._handle, Usage::READ);
                        }
                    },
                    i._index_buffer);
            }
        } break;
        default:
            LUISA_ERROR("Custom command not supported by validation layer.");
            break;
    }
}
void Stream::dispatch(DeviceInterface *dev, CommandList &cmd_list) {
    _executed_layer++;
    res_usages.clear();
    using CmdTag = luisa::compute::Command::Tag;
    for (auto &&cmd_ptr : cmd_list.commands()) {
        Command *cmd = cmd_ptr.get();
        switch (cmd->tag()) {
            case CmdTag::EBufferUploadCommand: {
                auto c = static_cast<BufferUploadCommand *>(cmd);
                swap_handle(c->_handle, Usage::WRITE);
            } break;
            case CmdTag::EBufferDownloadCommand: {
                auto c = static_cast<BufferDownloadCommand *>(cmd);
                swap_handle(c->_handle, Usage::READ);
            } break;
            case CmdTag::EBufferCopyCommand: {
                auto c = static_cast<BufferCopyCommand *>(cmd);
                swap_handle(c->_src_handle, Usage::READ);
                swap_handle(c->_dst_handle, Usage::WRITE);
            } break;
            case CmdTag::EBufferToTextureCopyCommand: {
                auto c = static_cast<BufferToTextureCopyCommand *>(cmd);
                swap_handle(c->_buffer_handle, Usage::READ);
                swap_handle(c->_texture_handle, Usage::WRITE);
            } break;
            case CmdTag::EShaderDispatchCommand: {
                auto c = static_cast<ShaderDispatchCommand *>(cmd);
                mark_shader_dispatch(dev, c, true);
            } break;
            case CmdTag::ETextureUploadCommand: {
                auto c = static_cast<TextureUploadCommand *>(cmd);
                swap_handle(c->_handle, Usage::WRITE);
            } break;
            case CmdTag::ETextureDownloadCommand: {
                auto c = static_cast<TextureDownloadCommand *>(cmd);
                swap_handle(c->_handle, Usage::READ);

            } break;
            case CmdTag::ETextureCopyCommand: {
                auto c = static_cast<TextureCopyCommand *>(cmd);
                swap_handle(c->_src_handle, Usage::READ);
                swap_handle(c->_dst_handle, Usage::WRITE);
            } break;
            case CmdTag::ETextureToBufferCopyCommand: {
                auto c = static_cast<TextureToBufferCopyCommand *>(cmd);
                swap_handle(c->_texture_handle, Usage::READ);
                swap_handle(c->_buffer_handle, Usage::WRITE);
            } break;
            case CmdTag::EAccelBuildCommand: {
                auto c = static_cast<AccelBuildCommand *>(cmd);
                reinterpret_cast<Accel *>(c->handle())->modify(c->instance_count(), this, c->modifications());
                for (auto &&i : c->_modifications) {
                    if (i.primitive) {
                        i.primitive = reinterpret_cast<RWResource *>(i.primitive)->handle();
                    }
                }
                swap_handle(c->_handle, Usage::WRITE);
            } break;
            case CmdTag::EMeshBuildCommand: {
                auto c = static_cast<MeshBuildCommand *>(cmd);
                auto mesh = reinterpret_cast<Mesh *>(c->handle());
                mesh->vert = reinterpret_cast<Buffer *>(c->_vertex_buffer);
                mesh->index = reinterpret_cast<Buffer *>(c->_triangle_buffer);
                swap_handle(c->_handle, Usage::WRITE);
                swap_handle(c->_vertex_buffer, Usage::READ);
                swap_handle(c->_triangle_buffer, Usage::READ);
            } break;
            case CmdTag::EProceduralPrimitiveBuildCommand: {
                auto c = static_cast<ProceduralPrimitiveBuildCommand *>(cmd);
                auto prim = reinterpret_cast<ProceduralPrimitives *>(c->handle());
                prim->bbox = reinterpret_cast<Buffer *>(c->aabb_buffer());
                swap_handle(c->_aabb_buffer, Usage::READ);
                swap_handle(c->_handle, Usage::WRITE);
            } break;
            case CmdTag::EBindlessArrayUpdateCommand: {
                using Operation = BindlessArrayUpdateCommand::Modification::Operation;
                auto c = static_cast<BindlessArrayUpdateCommand *>(cmd);
                for (auto &&i : c->_modifications) {
                    if (i.buffer.op == Operation::EMPLACE) {
                        i.buffer.handle = reinterpret_cast<RWResource *>(i.buffer.handle)->handle();
                    }
                    if (i.tex2d.op == Operation::EMPLACE) {
                        i.tex2d.handle = reinterpret_cast<RWResource *>(i.tex2d.handle)->handle();
                    }
                    if (i.tex3d.op == Operation::EMPLACE) {
                        i.tex3d.handle = reinterpret_cast<RWResource *>(i.tex3d.handle)->handle();
                    }
                }
                swap_handle(c->_handle, Usage::WRITE);
            } break;
            case CmdTag::ECustomCommand: {
                custom(dev, cmd);
            } break;
        }
        // TODO: resources record
    }
}
void Stream::sync_layer(uint64_t layer) {
    _synced_layer = std::max(_synced_layer, layer);
}
void Stream::sync() {
    _synced_layer = _executed_layer;
}
void Event::sync() {
    for (auto &&i : signaled) {
        i.first->sync_layer(i.second);
    }
    signaled.clear();
}
vstd::string Stream::stream_tag() const {
    switch (_stream_tag) {
        case StreamTag::COMPUTE:
            return "Compute";
        case StreamTag::COPY:
            return "Copy";
        case StreamTag::GRAPHICS:
            return "Graphics";
        default:
            return "Unknown";
    }
}
}// namespace lc::validation