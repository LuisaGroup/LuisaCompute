#pragma once

#include <runtime/device.h>
#include <core/stl/hash.h>
#include <cstdint>
#include <vstl/common.h>
#include <runtime/command.h>
#include <runtime/buffer.h>
#include <runtime/raster/raster_scene.h>
namespace luisa::compute {
template<typename T>
/*
struct ReorderFuncTable{
    bool is_res_in_bindless(uint64_t bindless_handle, uint64_t resource_handle) const noexcept {}
    Usage get_usage(uint64_t shader_handle, size_t argument_index) const noexcept {}
    size_t aabb_stride() const noexcept {}
    void update_bindless(uint64_t handle, luisa::span<const BindlessArrayUpdateCommand::Modification> modifications) const noexcept {}
}
*/
concept ReorderFuncTable =
    requires(const T t, uint64_t uint64_v, size_t size_v, luisa::span<const BindlessArrayUpdateCommand::Modification> modification) {
        requires(std::is_same_v<bool, decltype(t.is_res_in_bindless(uint64_v, uint64_v))>);
        requires(std::is_same_v<Usage, decltype(t.get_usage(uint64_v, size_v))>);
        requires(std::is_same_v<size_t, decltype(t.aabb_stride())>);
        t.update_bindless(uint64_v, modification);
    };
template<ReorderFuncTable FuncTable, bool supportConcurrentCopy>
class CommandReorderVisitor : public CommandVisitor {
public:
    enum class ResourceRW : uint8_t {
        Read,
        Write
    };
    enum class ResourceType : uint8_t {
        Texture,
        Buffer,
        Mesh,
        Bindless,
        Accel
    };
    struct Range {
        int64_t min;
        int64_t max;
        Range() {
            min = std::numeric_limits<int64_t>::min();
            max = std::numeric_limits<int64_t>::max();
        }
        Range(int64_t value) {
            min = value;
            max = value + 1;
        }
        Range(int64_t min, int64_t size)
            : min(min), max(size + min) {}
        bool collide(Range const &r) const {
            return min < r.max && r.min < max;
        }
        bool operator==(Range const &r) const {
            return min == r.min && max == r.max;
        }
        bool operator!=(Range const &r) const { return !operator==(r); }
    };
    struct RangeHash {
        size_t operator()(Range const &r) const {
            return hash64(this, sizeof(Range), hash64_default_seed);
        }
    };
    struct ResourceView {
        int64_t read_layer = -1;
        int64_t write_layer = -1;
    };
    struct ResourceHandle {
        uint64_t handle;
        ResourceType type;
    };
    struct RangeHandle : public ResourceHandle {
        vstd::unordered_map<Range, ResourceView, RangeHash> views;
    };
    struct NoRangeHandle : public ResourceHandle {
        ResourceView view;
    };
    struct BindlessHandle : public ResourceHandle {
        ResourceView view;
    };

private:
    static Range copy_range(int64_t offset, int64_t size) {
        if constexpr (supportConcurrentCopy) {
            return Range(offset, size);
        } else {
            return Range();
        }
    }
    template<typename Func>
        requires(std::is_invocable_v<Func, CommandReorderVisitor::ResourceView const &>)
    void iterate_map(Func &&func, RangeHandle &handle, Range const &range) {
        for (auto &&r : handle.views) {
            if (r.first.collide(range)) {
                func(r.second);
            }
        }
    }
    vstd::Pool<RangeHandle, true> _range_pool;
    vstd::Pool<NoRangeHandle, true> _no_range_pool;
    vstd::Pool<BindlessHandle, true> _bindless_handle_pool;
    vstd::unordered_map<uint64_t, RangeHandle *> _res_map;
    vstd::unordered_map<uint64_t, NoRangeHandle *> _no_range_resmap;
    vstd::unordered_map<uint64_t, BindlessHandle *> _bindless_map;
    int64_t _bindless_max_layer = -1;
    int64_t _max_mesh_level = -1;
    int64_t _max_accel_read_level = -1;
    int64_t _max_accel_write_level = -1;
    vstd::vector<vstd::fixed_vector<Command const *, 4>> _cmd_lists;
    size_t _layer_count = 0;
    vstd::vector<std::pair<Range, ResourceHandle *>> _dispatch_read_handle;
    vstd::vector<std::pair<Range, ResourceHandle *>> _dispatch_write_handle;
    size_t _arg_idx;
    uint64_t _shader_handle;
    size_t _dispatch_layer;
    bool _use_bindless_in_pass;
    bool _use_accel_in_pass;
    ResourceHandle *get_handle(
        uint64_t target_handle,
        ResourceType target_type) {
        auto func = [&](auto &&map, auto &&pool) {
            auto try_result = map.try_emplace(
                target_handle);
            auto &&value = try_result.first->second;
            if (try_result.second) {
                value = pool.create();
                value->handle = target_handle;
                value->type = target_type;
            }
            return value;
        };
        switch (target_type) {
            case ResourceType::Bindless:
                return func(_bindless_map, _bindless_handle_pool);
            case ResourceType::Mesh:
            case ResourceType::Accel:
                return func(_no_range_resmap, _no_range_pool);
            default:
                return func(_res_map, _range_pool);
        }
    }
    // Texture, Buffer
    size_t get_last_layer_write(RangeHandle *handle, Range range) {
        size_t layer = 0;
        iterate_map(
            [&](auto &&handle) {
                layer = std::max<int64_t>(layer, std::max<int64_t>(handle.read_layer + 1, handle.write_layer + 1));
            },
            *handle,
            range);
        if (_bindless_max_layer >= layer) {
            for (auto &&i : _bindless_map) {
                if (_func_table.is_res_in_bindless(i.first, handle->handle)) {
                    layer = std::max<int64_t>(layer, i.second->view.read_layer + 1);
                }
            }
        }
        return layer;
    }
    // Mesh, Accel
    size_t get_last_layer_write(NoRangeHandle *handle) {
        size_t layer = std::max<int64_t>(handle->view.read_layer + 1, handle->view.write_layer + 1);

        switch (handle->type) {
            case ResourceType::Mesh: {
                auto max_accel_level = std::max(_max_accel_read_level, _max_accel_write_level);
                layer = std::max<int64_t>(layer, max_accel_level + 1);
            } break;
            case ResourceType::Accel: {
                auto max_accel_level = std::max(_max_accel_read_level, _max_accel_write_level);
                layer = std::max<int64_t>(layer, max_accel_level + 1);
                layer = std::max<int64_t>(layer, _max_mesh_level + 1);
            } break;
            default: break;
        }
        return layer;
    }
    // Bindless
    size_t get_last_layer_write(BindlessHandle *handle) {
        return std::max<int64_t>(handle->view.read_layer + 1, handle->view.write_layer + 1);
    }
    size_t get_last_layer_read(RangeHandle *handle, Range range) {
        size_t layer = 0;
        iterate_map(
            [&](auto &&handle) {
                layer = std::max<int64_t>(layer, handle.write_layer + 1);
            },
            *handle,
            range);
        return layer;
    }
    size_t get_last_layer_read(NoRangeHandle *handle) {
        size_t layer = handle->view.write_layer + 1;
        if (handle->type == ResourceType::Accel) {
            layer = std::max<int64_t>(layer, _max_accel_write_level + 1);
        }
        return layer;
    }
    size_t get_last_layer_read(BindlessHandle *handle) {
        return handle->view.write_layer + 1;
    }
    void add_command(Command const *cmd, size_t layer) {
        if (_cmd_lists.size() <= layer) {
            _cmd_lists.resize(layer + 1);
        }
        _layer_count = std::max<int64_t>(_layer_count, layer + 1);
        _cmd_lists[layer].push_back(cmd);
    }
    size_t set_read(
        uint64_t handle,
        Range range,
        ResourceType type) {
        auto src_handle = get_handle(
            handle,
            type);
        return set_read(src_handle, range);
    }
    size_t set_read(
        ResourceHandle *src_handle,
        Range range) {
        size_t layer = 0;
        switch (src_handle->type) {
            case ResourceType::Mesh:
            case ResourceType::Accel: {
                auto handle = static_cast<NoRangeHandle *>(src_handle);
                layer = get_last_layer_read(handle);
                handle->view.read_layer = std::max<int64_t>(layer, handle->view.read_layer);
            } break;
            case ResourceType::Bindless: {
                auto handle = static_cast<BindlessHandle *>(src_handle);
                layer = get_last_layer_read(handle);
                handle->view.read_layer = std::max<int64_t>(layer, handle->view.read_layer);
            } break;
            default: {
                auto handle = static_cast<RangeHandle *>(src_handle);
                layer = get_last_layer_read(handle, range);
                auto ite = handle->views.try_emplace(range);
                if (ite.second)
                    ite.first->second.read_layer = std::max<int64_t>(ite.first->second.read_layer, layer);
                else
                    ite.first->second.read_layer = layer;
            } break;
        }
        return layer;
    }
    size_t set_write(
        ResourceHandle *dst_handle,
        Range range) {
        size_t layer = 0;
        switch (dst_handle->type) {
            case ResourceType::Mesh:
            case ResourceType::Accel: {
                auto handle = static_cast<NoRangeHandle *>(dst_handle);
                layer = get_last_layer_write(handle);
                handle->view.write_layer = layer;
            } break;
            case ResourceType::Bindless: {
                auto handle = static_cast<BindlessHandle *>(dst_handle);
                layer = get_last_layer_write(handle);
                handle->view.write_layer = layer;
            } break;
            default: {
                auto handle = static_cast<RangeHandle *>(dst_handle);
                layer = get_last_layer_write(handle, range);
                auto ite = handle->views.try_emplace(range);
                ite.first->second.write_layer = layer;
            } break;
        }

        return layer;
    }
    size_t set_write(
        uint64_t handle,
        Range range,
        ResourceType type) {
        auto dst_handle = get_handle(
            handle,
            type);
        return set_write(dst_handle, range);
    }
    size_t set_rw(
        uint64_t read_handle,
        Range read_range,
        ResourceType read_type,
        uint64_t write_handle,
        Range write_range,
        ResourceType write_type) {

        size_t layer = 0;
        auto src_handle = get_handle(
            read_handle,
            read_type);
        auto dst_handle = get_handle(
            write_handle,
            write_type);
        luisa::move_only_function<void()> set_read_layer;
        switch (read_type) {
            case ResourceType::Mesh:
            case ResourceType::Accel: {
                auto handle = static_cast<NoRangeHandle *>(src_handle);
                layer = get_last_layer_read(handle);
                set_read_layer = [&]() {
                    auto handle = static_cast<NoRangeHandle *>(src_handle);
                    handle->view.read_layer = std::max<int64_t>(layer, handle->view.read_layer);
                };
            } break;
            case ResourceType::Bindless: {
                auto handle = static_cast<BindlessHandle *>(src_handle);
                layer = get_last_layer_read(handle);
                set_read_layer = [&]() {
                    auto handle = static_cast<BindlessHandle *>(src_handle);
                    handle->view.read_layer = std::max<int64_t>(layer, handle->view.read_layer);
                };
            } break;
            default: {
                auto handle = static_cast<RangeHandle *>(src_handle);
                layer = get_last_layer_read(handle, read_range);
                auto ite = handle->views.try_emplace(read_range);
                if (ite.second) {
                    auto view_ptr = &ite.first->second;
                    set_read_layer = [view_ptr, &layer]() {
                        view_ptr->read_layer = std::max<int64_t>(view_ptr->read_layer, layer);
                    };
                } else {
                    auto view_ptr = &ite.first->second;
                    set_read_layer = [view_ptr, &layer]() {
                        view_ptr->read_layer = layer;
                    };
                }

            } break;
        }

        switch (write_type) {
            case ResourceType::Mesh:
            case ResourceType::Accel: {
                auto handle = static_cast<NoRangeHandle *>(dst_handle);
                layer = std::max<int64_t>(layer, get_last_layer_write(handle));
                handle->view.write_layer = layer;
            } break;
            case ResourceType::Bindless: {
                auto handle = static_cast<BindlessHandle *>(dst_handle);
                layer = std::max<int64_t>(layer, get_last_layer_write(handle));
                handle->view.write_layer = layer;
            } break;
            default: {
                auto handle = static_cast<RangeHandle *>(dst_handle);
                layer = std::max<int64_t>(layer, get_last_layer_write(handle, write_range));
                auto ite = handle->views.try_emplace(write_range);
                ite.first->second.write_layer = layer;
            } break;
        }
        set_read_layer();
        return layer;
    }
    size_t set_mesh(
        uint64_t handle,
        uint64_t vb,
        Range vb_range,
        uint64_t ib,
        Range ib_range) {

        auto vb_handle = get_handle(
            vb,
            ResourceType::Buffer);
        auto mesh_handle = get_handle(
            handle,
            ResourceType::Mesh);
        auto layer = get_last_layer_read(static_cast<RangeHandle *>(vb_handle), vb_range);
        layer = std::max<int64_t>(layer, get_last_layer_write(static_cast<NoRangeHandle *>(mesh_handle)));
        auto set_handle = [](auto &&handle, auto &&range, auto layer) {
            auto ite = handle->views.try_emplace(range);
            if (ite.second)
                ite.first->second.read_layer = layer;
            else
                ite.first->second.read_layer = std::max<int64_t>(layer, ite.first->second.read_layer);
        };
        auto ib_handle = get_handle(
            ib,
            ResourceType::Buffer);
        auto range_handle = static_cast<RangeHandle *>(ib_handle);
        layer = std::max<int64_t>(layer, get_last_layer_read(range_handle, ib_range));
        set_handle(range_handle, ib_range, layer);
        set_handle(static_cast<RangeHandle *>(vb_handle), vb_range, layer);
        static_cast<NoRangeHandle *>(mesh_handle)->view.write_layer = layer;
        _max_mesh_level = std::max<int64_t>(_max_mesh_level, layer);
        return layer;
    }
    size_t set_aabb(
        uint64_t handle,
        uint64_t aabb_buffer,
        Range aabb_range) {
        auto vb_handle = get_handle(
            aabb_buffer,
            ResourceType::Buffer);
        auto mesh_handle = get_handle(
            handle,
            ResourceType::Mesh);
        auto layer = get_last_layer_read(static_cast<RangeHandle *>(vb_handle), aabb_range);
        layer = std::max<int64_t>(layer, get_last_layer_write(static_cast<NoRangeHandle *>(mesh_handle)));
        auto set_handle = [](auto &&handle, auto &&range, auto layer) {
            auto ite = handle->views.try_emplace(range);
            if (ite.second)
                ite.first->second.read_layer = layer;
            else
                ite.first->second.read_layer = std::max<int64_t>(layer, ite.first->second.read_layer);
        };
        set_handle(static_cast<RangeHandle *>(vb_handle), aabb_range, layer);
        static_cast<NoRangeHandle *>(mesh_handle)->view.write_layer = layer;
        _max_mesh_level = std::max<int64_t>(_max_mesh_level, layer);
        return layer;
    }

    void set_read_layer(
        ResourceHandle *src_handle,
        Range range,
        int64_t layer) {
        switch (src_handle->type) {
            case ResourceType::Mesh:
            case ResourceType::Accel: {
                auto handle = static_cast<NoRangeHandle *>(src_handle);
                handle->view.read_layer = std::max<int64_t>(layer, handle->view.read_layer);
            } break;
            case ResourceType::Bindless: {
                auto handle = static_cast<BindlessHandle *>(src_handle);
                handle->view.read_layer = std::max<int64_t>(layer, handle->view.read_layer);
            } break;
            default: {
                auto handle = static_cast<RangeHandle *>(src_handle);
                auto emplace_result = handle->views.try_emplace(range);
                if (emplace_result.second) {
                    emplace_result.first->second.read_layer = layer;
                } else {
                    emplace_result.first->second.read_layer = std::max<int64_t>(emplace_result.first->second.read_layer, layer);
                }
            } break;
        }
    }
    void set_write_layer(
        ResourceHandle *dst_handle,
        Range range,
        int64_t layer) {
        switch (dst_handle->type) {
            case ResourceType::Mesh:
            case ResourceType::Accel: {
                auto handle = static_cast<NoRangeHandle *>(dst_handle);
                handle->view.write_layer = layer;
            } break;
            case ResourceType::Bindless: {
                auto handle = static_cast<BindlessHandle *>(dst_handle);
                handle->view.write_layer = layer;
            } break;
            default: {
                auto handle = static_cast<RangeHandle *>(dst_handle);
                auto emplace_result = handle->views.try_emplace(range);
                emplace_result.first->second.write_layer = layer;
            } break;
        }
    }
    void add_dispatch_handle(
        uint64_t handle,
        ResourceType type,
        Range range,
        bool is_write) {
        if (is_write) {
            auto h = get_handle(
                handle,
                type);
            switch (type) {
                case ResourceType::Accel:
                case ResourceType::Mesh:
                    _dispatch_layer = std::max<int64_t>(_dispatch_layer, get_last_layer_write(static_cast<NoRangeHandle *>(h)));
                    break;
                case ResourceType::Buffer:
                case ResourceType::Texture:
                    _dispatch_layer = std::max<int64_t>(_dispatch_layer, get_last_layer_write(static_cast<RangeHandle *>(h), range));
                    break;
                case ResourceType::Bindless:
                    _dispatch_layer = std::max<int64_t>(_dispatch_layer, get_last_layer_write(static_cast<BindlessHandle *>(h)));
                    break;
            }
            _dispatch_write_handle.emplace_back(range, h);
        } else {
            auto h = get_handle(
                handle,
                type);
            switch (type) {
                case ResourceType::Accel:
                case ResourceType::Mesh:
                    _dispatch_layer = std::max<int64_t>(_dispatch_layer, get_last_layer_read(static_cast<NoRangeHandle *>(h)));
                    break;
                case ResourceType::Buffer:
                case ResourceType::Texture:
                    _dispatch_layer = std::max<int64_t>(_dispatch_layer, get_last_layer_read(static_cast<RangeHandle *>(h), range));
                    break;
                case ResourceType::Bindless:
                    _dispatch_layer = std::max<int64_t>(_dispatch_layer, get_last_layer_read(static_cast<BindlessHandle *>(h)));
                    break;
            }
            _dispatch_read_handle.emplace_back(range, h);
        }
    }
    FuncTable _func_table;
    template<typename... Callbacks>
    void visit(const ShaderDispatchCommandBase *command, uint64_t shader_handle, Callbacks &&...callbacks) noexcept {
        _dispatch_read_handle.clear();
        _dispatch_write_handle.clear();
        _use_bindless_in_pass = false;
        _use_accel_in_pass = false;
        _dispatch_layer = 0;
        _arg_idx = 0;
        _shader_handle = shader_handle;
        using Argument = ShaderDispatchCommandBase::Argument;
        using Tag = Argument::Tag;
        for (auto &&i : command->arguments()) {
            switch (i.tag) {
                case Tag::BUFFER: {
                    auto &&bf = i.buffer;
                    add_dispatch_handle(
                        bf.handle,
                        ResourceType::Buffer,
                        Range(bf.offset, bf.size),
                        ((uint)_func_table.get_usage(_shader_handle, _arg_idx) & (uint)Usage::WRITE) != 0);
                    ++_arg_idx;
                } break;
                case Tag::TEXTURE: {
                    auto &&tex = i.texture;
                    add_dispatch_handle(
                        tex.handle,
                        ResourceType::Texture,
                        Range(tex.level),
                        ((uint)_func_table.get_usage(_shader_handle, _arg_idx) & (uint)Usage::WRITE) != 0);
                    ++_arg_idx;
                } break;
                case Tag::UNIFORM: {
                    ++_arg_idx;
                } break;
                case Tag::BINDLESS_ARRAY: {
                    auto &&arr = i.bindless_array;
                    _use_bindless_in_pass = true;
                    add_dispatch_handle(
                        arr.handle,
                        ResourceType::Bindless,
                        Range(),
                        false);
                    ++_arg_idx;
                } break;
                case Tag::ACCEL: {
                    auto &&acc = i.accel;
                    _use_accel_in_pass = true;
                    add_dispatch_handle(
                        acc.handle,
                        ResourceType::Accel,
                        Range(),
                        false);
                    ++_arg_idx;
                } break;
            }
        }
        if constexpr (sizeof...(callbacks) > 0) {
            auto cb = {(callbacks(), 0)...};
        }
        for (auto &&i : _dispatch_read_handle) {
            set_read_layer(i.second, i.first, _dispatch_layer);
        }
        for (auto &&i : _dispatch_write_handle) {
            set_write_layer(i.second, i.first, _dispatch_layer);
        }
        add_command(command, _dispatch_layer);
        if (_use_bindless_in_pass) {
            _bindless_max_layer = std::max<int64_t>(_bindless_max_layer, _dispatch_layer);
        }
        if (_use_accel_in_pass) {
            _max_accel_read_level = std::max<int64_t>(_max_accel_read_level, _dispatch_layer);
        }
    }

public:
    explicit CommandReorderVisitor(FuncTable &&func_table) noexcept
        : _range_pool(256, true),
          _no_range_pool(256, true),
          _bindless_handle_pool(32, true),
          _func_table(std::forward<FuncTable>(func_table)) {
    }
    ~CommandReorderVisitor() noexcept = default;
    void clear() noexcept {
        for (auto &&i : _res_map) {
            _range_pool.destroy(i.second);
        }
        for (auto &&i : _no_range_resmap) {
            _no_range_pool.destroy(i.second);
        }
        for (auto &&i : _bindless_map) {
            _bindless_handle_pool.destroy(i.second);
        }

        _res_map.clear();
        _no_range_resmap.clear();
        _bindless_map.clear();
        _bindless_max_layer = -1;
        _max_accel_read_level = -1;
        _max_accel_write_level = -1;
        _max_mesh_level = -1;
        luisa::span<typename decltype(_cmd_lists)::value_type> sp(_cmd_lists.data(), _layer_count);
        for (auto &&i : sp) {
            i.clear();
        }
        _layer_count = 0;
    }
    [[nodiscard]] auto command_lists() const noexcept {
        return luisa::span{_cmd_lists.data(), _layer_count};
    }

    // Buffer : resource
    void visit(const BufferUploadCommand *command) noexcept override {
        add_command(command, set_write(command->handle(), copy_range(command->offset(), command->size()), ResourceType::Buffer));
    }
    void visit(const BufferDownloadCommand *command) noexcept override {
        add_command(command, set_read(command->handle(), copy_range(command->offset(), command->size()), ResourceType::Buffer));
    }
    void visit(const BufferCopyCommand *command) noexcept override {
        add_command(command, set_rw(command->src_handle(), copy_range(command->src_offset(), command->size()), ResourceType::Buffer, command->dst_handle(), copy_range(command->dst_offset(), command->size()), ResourceType::Buffer));
    }
    void visit(const BufferToTextureCopyCommand *command) noexcept override {
        auto sz = command->size();
        auto binSize = pixel_storage_size(command->storage(), sz.x, sz.y, sz.z);
        add_command(command, set_rw(command->buffer(), copy_range(command->buffer_offset(), binSize), ResourceType::Buffer, command->texture(), copy_range(command->level(), 1), ResourceType::Texture));
    }

    // Shader : function, read/write multi resources
    void visit(const ShaderDispatchCommand *command) noexcept override {
        visit(command, command->handle(), [&] {
            if (command->is_indirect()) {
                auto &&t = command->indirect_dispatch_size();
                add_dispatch_handle(
                    t.handle,
                    ResourceType::Buffer,
                    Range(),
                    false);
            }
        });
    }
    void visit(const DrawRasterSceneCommand *command) noexcept override {
        auto set_tex_dsl = [&](ShaderDispatchCommandBase::Argument::Texture const &a) {
            add_dispatch_handle(
                a.handle,
                ResourceType::Texture,
                Range(a.level),
                true);
        };
        visit(command, command->handle(), [&] {
            auto &&rtv = command->rtv_texs();
            auto &&dsv = command->dsv_tex();
            for (auto &&i : rtv) {
                set_tex_dsl(i);
            }
            if (dsv.handle != ~0ull) {
                set_tex_dsl(dsv);
            }
            for (auto &&mesh : command->scene()) {
                for (auto &&v : mesh.vertex_buffers()) {
                    add_dispatch_handle(
                        v.handle(),
                        ResourceType::Buffer,
                        Range(v.offset(), v.size()),
                        false);
                }
                auto &&i = mesh.index();
                if (i.index() == 0) {
                    auto idx = luisa::get<0>(i);
                    add_dispatch_handle(
                        idx.handle(),
                        ResourceType::Buffer,
                        Range(idx.offset_bytes(), idx.size_bytes()),
                        false);
                }
            }
        });
    }

    // Texture : resource
    void visit(const TextureUploadCommand *command) noexcept override {
        add_command(command, set_write(command->handle(), copy_range(command->level(), 1), ResourceType::Texture));
    }
    void visit(const TextureDownloadCommand *command) noexcept override {
        add_command(command, set_read(command->handle(), copy_range(command->level(), 1), ResourceType::Texture));
    }
    void visit(const TextureCopyCommand *command) noexcept override {
        add_command(command, set_rw(command->src_handle(), copy_range(command->src_level(), 1), ResourceType::Texture, command->dst_handle(), copy_range(command->dst_level(), 1), ResourceType::Texture));
    }
    void visit(const TextureToBufferCopyCommand *command) noexcept override {
        auto sz = command->size();
        auto binSize = pixel_storage_size(command->storage(), sz.x, sz.y, sz.z);
        add_command(command, set_rw(command->texture(), copy_range(command->level(), 1), ResourceType::Texture, command->buffer(), copy_range(command->buffer_offset(), binSize), ResourceType::Buffer));
    }
    void visit(const ClearDepthCommand *command) noexcept override {
        add_command(command, set_write(command->handle(), Range{}, ResourceType::Texture));
    }

    // BindlessArray : read multi resources
    void visit(const BindlessArrayUpdateCommand *command) noexcept override {
        _func_table.update_bindless(command->handle(), command->modifications());
        add_command(command, set_write(command->handle(), Range(), ResourceType::Bindless));
    }

    // Accel : conclude meshes and their buffer
    void visit(const AccelBuildCommand *command) noexcept override {
        auto layer = set_write(command->handle(), Range(), ResourceType::Accel);
        _max_accel_write_level = std::max<int64_t>(_max_accel_write_level, layer);
        add_command(command, layer);
    }

    // Mesh : conclude vertex and triangle buffers
    void visit(const MeshBuildCommand *command) noexcept override {
        add_command(
            command,
            set_mesh(
                command->handle(),
                command->vertex_buffer(),
                Range(command->vertex_buffer_offset(),
                      command->vertex_buffer_size()),
                command->triangle_buffer(),
                Range(command->triangle_buffer_offset(),
                      command->triangle_buffer_size())));
    }
    void visit(const ProceduralPrimitiveBuildCommand *command) noexcept override {
        auto stride = _func_table.aabb_stride();
        add_command(
            command,
            set_aabb(
                command->handle(),
                command->aabb_buffer(),
                Range(command->aabb_offset() * stride, command->aabb_count() * stride)));
    }

    void visit(const CustomCommand *command) noexcept override {
        _dispatch_read_handle.clear();
        _dispatch_write_handle.clear();
        _use_bindless_in_pass = false;
        _use_accel_in_pass = false;
        _dispatch_layer = 0;
        for (auto &&i : command->resources()) {
            bool is_write = ((uint)i.usage & (uint)Usage::WRITE) != 0;
            luisa::visit(
                [&]<typename T>(T const &res) {
                    if constexpr (std::is_same_v<T, CustomCommand::BufferView>) {
                        add_dispatch_handle(
                            res.handle,
                            ResourceType::Buffer,
                            Range(res.start_byte, res.size_byte),
                            is_write);
                    } else if constexpr (std::is_same_v<T, CustomCommand::TextureView>) {
                        add_dispatch_handle(
                            res.handle,
                            ResourceType::Texture,
                            Range(res.start_mip, res.size_mip),
                            is_write);
                    } else if constexpr (std::is_same_v<T, CustomCommand::MeshView>) {
                        add_dispatch_handle(
                            res.handle,
                            ResourceType::Mesh,
                            Range(),
                            is_write);
                    } else if constexpr (std::is_same_v<T, CustomCommand::AccelView>) {
                        add_dispatch_handle(
                            res.handle,
                            ResourceType::Accel,
                            Range(),
                            is_write);
                    } else {
                        add_dispatch_handle(
                            res.handle,
                            ResourceType::Bindless,
                            Range(),
                            is_write);
                    }
                },
                i.resource_view);
        }
        for (auto &&i : _dispatch_read_handle) {
            set_read_layer(i.second, i.first, _dispatch_layer);
        }
        for (auto &&i : _dispatch_write_handle) {
            set_write_layer(i.second, i.first, _dispatch_layer);
        }
        add_command(command, _dispatch_layer);
        if (_use_bindless_in_pass) {
            _bindless_max_layer = std::max<int64_t>(_bindless_max_layer, _dispatch_layer);
        }
        if (_use_accel_in_pass) {
            _max_accel_read_level = std::max<int64_t>(_max_accel_read_level, _dispatch_layer);
        }
    }
};

}// namespace luisa::compute
