#include "device.h"
#include "event.h"
#include "stream.h"
#include "accel.h"
#include "buffer.h"
#include "texture.h"
#include "depth_buffer.h"
#include "bindless_array.h"
#include "mesh.h"
#include "curve.h"
#include "motion_instance.h"
#include "procedural_primitives.h"
#include "shader.h"
#include "swap_chain.h"
#include <luisa/core/logging.h>
#include <luisa/runtime/raster/raster_scene.h>
#include <luisa/runtime/rtx/aabb.h>
#include <luisa/backends/ext/raster_cmd.h>
#include <luisa/backends/ext/dstorage_cmd.h>

namespace lc::validation {
Stream::Stream(uint64_t handle, StreamTag stream_tag) : RWResource{handle, Tag::STREAM, false}, _stream_tag{stream_tag} {}
std::recursive_mutex stream_global_lock;
void Stream::signal(Event *evt, uint64_t fence) {
    std::lock_guard lck{stream_global_lock};
    evt->signaled.force_emplace(this, Event::Signaled{fence, _executed_layer});
}
uint64_t Stream::stream_synced_frame(Stream *stream) const {
    auto iter = waited_stream.find(stream);
    if (iter == waited_stream.end()) {
        return stream->synced_layer();
    } else {
        return iter->second;
    }
}
void Stream::wait(Event *evt, uint64_t fence) {
    std::lock_guard lck{stream_global_lock};
    for (auto &&i : evt->signaled) {
        if (fence >= i.second.event_fence) {
            waited_stream.force_emplace(i.first, i.second.stream_fence);
        }
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
    std::lock_guard lck{stream_global_lock};
    for (auto &&iter : res_usages) {
        auto res = iter.first;
        for (auto &&stream_iter : res->info()) {
            auto other_stream = RWResource::get<Stream>(stream_iter.first);
            if (!other_stream || other_stream == this) continue;
            auto synced_frame = stream_synced_frame(other_stream);
            if (stream_iter.second.last_frame > synced_frame) {
                // Texture type
                if (res->non_simultaneous()) {
                    LUISA_ERROR(
                        "Non simultaneous-accessible resource {} is not allowed to be {} by {} and {} by {} simultaneously.",
                        res->get_name(),
                        detail::usage_name(stream_iter.second.usage),
                        other_stream->get_name(),
                        detail::usage_name(iter.second.usage),
                        get_name());
                } else {
                    LUISA_WARNING(
                        "Simultaneous-accessible resource {} is used to be {} by {} and {} by {} simultaneously.",
                        res->get_name(),
                        detail::usage_name(stream_iter.second.usage),
                        other_stream->get_name(),
                        detail::usage_name(iter.second.usage),
                        get_name());
                }
            }
        }
    }
}
void Stream::dispatch() {
    std::lock_guard lck{stream_global_lock};
    _executed_layer++;
    res_usages.clear();
}
void Stream::mark_shader_dispatch(DeviceInterface *dev, ShaderDispatchCommandBase *cmd, bool contain_bindings) {
    size_t arg_idx = 0;
    auto shader = RWResource::get<RWResource>(cmd->handle());
    auto mark_handle = [&](uint64_t &handle, Range range) -> std::pair<RWResource *, Usage> {
        auto res = RWResource::get<RWResource>(handle);
        auto usage = dev->shader_argument_usage(cmd->handle(), arg_idx);
        res->set(this, usage, range);
        return {res, usage};
    };
    auto set_arg = [&](Argument &arg) {
        switch (arg.tag) {
            case Argument::Tag::BUFFER: {
                if (arg.buffer.handle == invalid_resource_handle) [[unlikely]] {
                    LUISA_ERROR("Invalid shader dispatch buffer argument.");
                }
                mark_handle(arg.buffer.handle, Range{arg.buffer.offset, arg.buffer.size});
            } break;
            case Argument::Tag::TEXTURE: {
                if (arg.texture.handle == invalid_resource_handle) [[unlikely]] {
                    LUISA_ERROR("Invalid shader dispatch texture argument.");
                }
                auto tex_usage = mark_handle(arg.texture.handle, Range{arg.texture.level, 1});
                if (tex_usage.first->tag() == Resource::Tag::DEPTH_BUFFER && (luisa::to_underlying(tex_usage.second) & luisa::to_underlying(Usage::WRITE)) != 0) {
                    LUISA_ERROR("{} can not be written by kernel.", tex_usage.first->get_name());
                }
            } break;
            case Argument::Tag::BINDLESS_ARRAY: {
                if (arg.bindless_array.handle == invalid_resource_handle) [[unlikely]] {
                    LUISA_ERROR("Invalid shader dispatch bindless argument.");
                }
                mark_handle(arg.bindless_array.handle, Range{});
            } break;
            case Argument::Tag::ACCEL: {
                if (arg.accel.handle == invalid_resource_handle) [[unlikely]] {
                    LUISA_ERROR("Invalid shader dispatch accel argument.");
                }
                mark_handle(arg.accel.handle, Range{});
            } break;
            default:
                break;
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
    Stream::mark_handle(cmd->handle(), Usage::READ, Range{});
}
void Stream::mark_handle(uint64_t v, Usage usage, Range range) {
    if (v != invalid_resource_handle) {
        RWResource::get<RWResource>(v)->set(this, usage, range);
    }
}

class CustomDispatchArgumentVisitor : public CustomDispatchCommand::ArgumentVisitor {

private:
    Stream *_stream;

public:
    explicit CustomDispatchArgumentVisitor(Stream *stream) noexcept : _stream{stream} {}
    void visit(const Argument::Buffer &t, Usage usage) noexcept override {
        _stream->mark_handle(t.handle, usage, Range{t.offset, t.size});
    }
    void visit(const Argument::Texture &t, Usage usage) noexcept override {
        _stream->mark_handle(t.handle, usage, Range{t.level, 1});
    }
    void visit(const Argument::Accel &t, Usage usage) noexcept override {
        _stream->mark_handle(t.handle, usage, Range{});
    }
    void visit(const Argument::BindlessArray &t, Usage usage) noexcept override {
        _stream->mark_handle(t.handle, usage, Range{});
    }
};

void Stream::custom(DeviceInterface *dev, Command *cmd) {
    switch (static_cast<CustomCommand *>(cmd)->uuid()) {
        case to_underlying(CustomCommandUUID::RASTER_CLEAR_DEPTH): {
            auto c = static_cast<ClearDepthCommand *>(cmd);
            mark_handle(c->handle(), Usage::WRITE, Range{});
        } break;
        case to_underlying(CustomCommandUUID::RASTER_DRAW_SCENE): {
            // auto c = static_cast<DrawRasterSceneCommand *>(cmd);
            // mark_shader_dispatch(dev, c, false);
            // if (c->_dsv_tex.handle != invalid_resource_handle) {
            //     mark_handle(c->_dsv_tex.handle, Usage::READ_WRITE, Range{0, 1});
            // }
            // for (auto i : vstd::range(c->_rtv_count)) {
            //     mark_handle(c->_rtv_texs[i].handle, Usage::WRITE, Range{c->_rtv_texs[i].level, 1});
            // }
            // for (auto &&i : c->_scene) {
            //     for (auto &&vb : i._vertex_buffers) {
            //         mark_handle(const_cast<uint64_t &>(vb._handle), Usage::READ, Range{vb._offset, vb._size});
            //     }
            //     luisa::visit(
            //         [&]<typename T>(T &t) {
            //             if constexpr (std::is_same_v<T, BufferView<uint>>) {
            //                 mark_handle(t._handle, Usage::READ, Range{t.offset_bytes(), t.size_bytes()});
            //             }
            //         },
            //         i._index_buffer);
            // }
        } break;
        case to_underlying(CustomCommandUUID::DSTORAGE_READ): {
            auto c = static_cast<DStorageReadCommand *>(cmd);
            auto check_range = [&](uint64_t handle, Range range) -> vstd::optional<std::pair<Range, Range>> {
                auto add_range = vstd::scope_exit([&] {
                    dstorage_range_check.try_emplace(handle).first->second.emplace_back(range);
                });
                auto iter = dstorage_range_check.find(handle);
                if (iter == dstorage_range_check.end()) return {};
                for (auto &&i : iter->second) {
                    if (Range::collide(i, range)) return std::pair<Range, Range>{i, range};
                }
                return {};
            };
            luisa::visit(
                [&](auto t) {
                    mark_handle(t.handle, Usage::READ, Range{});
                },
                c->source());
            auto log_error = [&](uint64_t handle, auto &&check_result) {
                LUISA_ERROR("Resource {} read conflict from range: ({}, {}) to ({}, {})", RWResource::get<RWResource>(handle)->get_name(),
                            check_result->first.min, check_result->first.max,
                            check_result->second.min, check_result->second.max);
            };
            luisa::visit(
                [&]<typename T>(T const &t) {
                    if constexpr (std::is_same_v<DStorageReadCommand::BufferRequest, T>) {
                        mark_handle(t.handle, Usage::WRITE, Range{t.offset_bytes, t.size_bytes});
                        auto check_result = check_range(t.handle, Range{t.offset_bytes, t.size_bytes});
                        if (check_result) [[unlikely]] {
                            log_error(t.handle, check_result);
                        }
                    } else if constexpr (std::is_same_v<DStorageReadCommand::TextureRequest, T>) {
                        mark_handle(t.handle, Usage::WRITE, Range{t.level, 1});
                        auto check_result = check_range(t.handle, Range{t.level, 1});
                        if (check_result) [[unlikely]] {
                            log_error(t.handle, check_result);
                        }
                    }
                },
                c->request());

        } break;
        case to_underlying(CustomCommandUUID::CUSTOM_DISPATCH): {
            auto c = static_cast<CustomDispatchCommand *>(cmd);
            CustomDispatchArgumentVisitor visitor{this};
            c->traverse_arguments(visitor);
        } break;
        default: break;
    }
}
void Stream::dispatch(DeviceInterface *dev, CommandList &cmd_list) {
    std::lock_guard lck{stream_global_lock};
    _executed_layer++;
    res_usages.clear();
    dstorage_range_check.clear();
    using CmdTag = luisa::compute::Command::Tag;
    for (auto &&cmd_ptr : cmd_list.commands()) {
        Command *cmd = cmd_ptr.get();
        switch (cmd->tag()) {
            case CmdTag::EBufferUploadCommand: {
                Device::check_stream(handle(), StreamFunc::Copy);
                auto c = static_cast<BufferUploadCommand *>(cmd);
                mark_handle(c->handle(), Usage::WRITE, Range{c->offset(), c->size()});
            } break;
            case CmdTag::EBufferDownloadCommand: {
                Device::check_stream(handle(), StreamFunc::Copy);
                auto c = static_cast<BufferDownloadCommand *>(cmd);
                mark_handle(c->handle(), Usage::READ, Range{c->offset(), c->size()});
            } break;
            case CmdTag::EBufferCopyCommand: {
                Device::check_stream(handle(), StreamFunc::Copy);
                auto c = static_cast<BufferCopyCommand *>(cmd);
                mark_handle(c->src_handle(), Usage::READ, Range{c->src_offset(), c->size()});
                mark_handle(c->dst_handle(), Usage::WRITE, Range{c->dst_offset(), c->size()});
            } break;
            case CmdTag::EBufferToTextureCopyCommand: {
                Device::check_stream(handle(), StreamFunc::Copy);
                auto c = static_cast<BufferToTextureCopyCommand *>(cmd);
                mark_handle(c->buffer(), Usage::READ, Range{c->buffer_offset(), pixel_storage_size(c->storage(), c->size())});
                mark_handle(c->texture(), Usage::WRITE, Range{c->level(), 1});
            } break;
            case CmdTag::EShaderDispatchCommand: {
                Device::check_stream(handle(), StreamFunc::Compute);
                auto c = static_cast<ShaderDispatchCommand *>(cmd);
                mark_shader_dispatch(dev, c, true);
            } break;
            case CmdTag::ETextureUploadCommand: {
                Device::check_stream(handle(), StreamFunc::Copy);
                auto c = static_cast<TextureUploadCommand *>(cmd);
                mark_handle(c->handle(), Usage::WRITE, Range{c->level(), 1});
            } break;
            case CmdTag::ETextureDownloadCommand: {
                Device::check_stream(handle(), StreamFunc::Copy);
                auto c = static_cast<TextureDownloadCommand *>(cmd);
                mark_handle(c->handle(), Usage::READ, Range{c->level(), 1});

            } break;
            case CmdTag::ETextureCopyCommand: {
                Device::check_stream(handle(), StreamFunc::Copy);
                auto c = static_cast<TextureCopyCommand *>(cmd);
                mark_handle(c->src_handle(), Usage::READ, Range{c->src_level(), 1});
                mark_handle(c->dst_handle(), Usage::WRITE, Range{c->dst_level(), 1});
            } break;
            case CmdTag::ETextureToBufferCopyCommand: {
                Device::check_stream(handle(), StreamFunc::Copy);
                auto c = static_cast<TextureToBufferCopyCommand *>(cmd);
                mark_handle(c->texture(), Usage::READ, Range{c->level(), 1});
                mark_handle(c->buffer(), Usage::WRITE, Range{c->buffer_offset(), pixel_storage_size(c->storage(), c->size())});
            } break;
            case CmdTag::EAccelBuildCommand: {
                Device::check_stream(handle(), StreamFunc::Compute);
                auto c = static_cast<AccelBuildCommand *>(cmd);
                auto accel = RWResource::get<Accel>(c->handle());
                if (c->update_instance_buffer_only()) {
                    if (!accel->init_build) [[unlikely]] {
                        LUISA_ERROR("Accel should been fully build before any other operations.");
                    }
                } else {
                    accel->init_build = true;
                }
                accel->modify(c->instance_count(), this, c->modifications());
                mark_handle(c->handle(), Usage::WRITE, Range{});
            } break;
            case CmdTag::EMeshBuildCommand: {
                Device::check_stream(handle(), StreamFunc::Compute);
                auto c = static_cast<MeshBuildCommand *>(cmd);
                auto mesh = RWResource::get<Mesh>(c->handle());
                mesh->vert = RWResource::get<Buffer>(c->vertex_buffer());
                mesh->index = RWResource::get<Buffer>(c->triangle_buffer());
                mesh->vert_range = Range{c->vertex_buffer_offset(), c->vertex_buffer_size()};
                mesh->index_range = Range{c->triangle_buffer_offset(), c->triangle_buffer_size()};
                mark_handle(c->handle(), Usage::WRITE, Range{});
            } break;
            case CmdTag::EProceduralPrimitiveBuildCommand: {
                Device::check_stream(handle(), StreamFunc::Compute);
                auto c = static_cast<ProceduralPrimitiveBuildCommand *>(cmd);
                auto prim = RWResource::get<ProceduralPrimitives>(c->handle());
                prim->range = Range{c->aabb_buffer_offset(), c->aabb_buffer_size()};
                prim->bbox = RWResource::get<Buffer>(c->aabb_buffer());
                mark_handle(c->handle(), Usage::WRITE, Range{});
            } break;
            case CmdTag::EBindlessArrayUpdateCommand: {
                Device::check_stream(handle(), StreamFunc::Compute);
                auto c = static_cast<BindlessArrayUpdateCommand *>(cmd);
                mark_handle(c->handle(), Usage::WRITE, Range{});
            } break;
            case CmdTag::ECustomCommand: {
                auto custom_cmd = static_cast<CustomCommand *>(cmd);
                switch (custom_cmd->uuid()) {
                    case to_underlying(CustomCommandUUID::RASTER_DRAW_SCENE):
                    case to_underlying(CustomCommandUUID::RASTER_CLEAR_DEPTH):
                        Device::check_stream(handle(), StreamFunc::Graphics, custom_cmd->uuid());
                        break;
                    case to_underlying(CustomCommandUUID::DSTORAGE_READ):
                        Device::check_stream(handle(), StreamFunc::Custom, custom_cmd->uuid());
                        break;
                }
                custom(dev, cmd);
            } break;
            case Command::Tag::ECurveBuildCommand: {
                Device::check_stream(handle(), StreamFunc::Compute);
                auto c = static_cast<CurveBuildCommand *>(cmd);
                auto curve = RWResource::get<Curve>(c->handle());
                curve->cp = RWResource::get<Buffer>(c->cp_buffer());
                curve->seg = RWResource::get<Buffer>(c->seg_buffer());
                curve->cp_range = Range{c->cp_buffer_offset(), c->cp_count() * c->cp_stride()};
                curve->seg_range = Range{c->seg_buffer_offset(), c->seg_count()};
                mark_handle(c->handle(), Usage::WRITE, Range{});
            } break;
            case CmdTag::EMotionInstanceBuildCommand: {
                Device::check_stream(handle(), StreamFunc::Compute);
                auto c = static_cast<MotionInstanceBuildCommand *>(cmd);
                auto motion = RWResource::get<MotionInstance>(c->handle());
                motion->child = RWResource::get<RWResource>(c->child());
                mark_handle(c->handle(), Usage::WRITE, Range{});
            } break;
        }
        // TODO: resources record
    }
}
void Stream::sync_layer(uint64_t layer) {
    std::lock_guard lck{stream_global_lock};
    if (_synced_layer >= layer) return;
    _synced_layer = layer;
    for (auto &&i : waited_stream) {
        i.first->sync_layer(i.second);
    }
    waited_stream.clear();
}
void Stream::sync() {
    sync_layer(_executed_layer);
}
void Event::sync(uint64_t fence) {
    std::lock_guard lck{stream_global_lock};
    vstd::vector<Stream *> removed_stream;
    for (auto &&i : signaled) {
        if (fence >= i.second.event_fence) {
            i.first->sync_layer(i.second.stream_fence);
            removed_stream.emplace_back(i.first);
        }
    }
    if (removed_stream.size() == signaled.size()) {
        signaled.clear();
    } else {
        for (auto &&i : removed_stream) {
            signaled.erase(i);
        }
    }
}
vstd::string Stream::stream_tag() const {
    switch (_stream_tag) {
        case StreamTag::COMPUTE:
            return "compute";
        case StreamTag::COPY:
            return "copy";
        case StreamTag::GRAPHICS:
            return "graphics";
        case StreamTag::CUSTOM:
            return "custom";
        default:
            return "unknown";
    }
}
}// namespace lc::validation
