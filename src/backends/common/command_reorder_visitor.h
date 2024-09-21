#pragma once

#include <luisa/runtime/device.h>
#include <luisa/core/stl/hash.h>
#include <cstdint>
#include <luisa/vstl/common.h>
#include <luisa/runtime/rhi/command.h>
#include <luisa/runtime/buffer.h>
#include <luisa/runtime/rhi/argument.h>
#include <luisa/core/logging.h>
#include <luisa/backends/ext/raster_cmd.h>
#include <luisa/vstl/stack_allocator.h>
#include <luisa/vstl/arena_hash_map.h>

namespace luisa::compute {
class ArenaRef {
    vstd::StackAllocator &_allocator;

public:
    ArenaRef(vstd::StackAllocator &allocator) : _allocator(allocator) {}
    ArenaRef(ArenaRef const &) = delete;
    ArenaRef(ArenaRef &&) = default;
    void *allocate(size_t size_bytes) {
        auto handle = _allocator.allocate(size_bytes, 16);
        auto ptr = reinterpret_cast<void *>(handle.handle + handle.offset);
        return ptr;
    }
};
template<typename T>
/*
struct ReorderFuncTable{
    void lock_bindless(uint64_t bindless_handle) const noexcept{}
    void unlock_bindless(uint64_t bindless_handle) const noexcept{}
    bool is_res_in_bindless(uint64_t bindless_handle, uint64_t resource_handle) const noexcept {}
    Usage get_usage(uint64_t shader_handle, size_t argument_index) const noexcept {}
    void update_bindless(uint64_t handle, luisa::span<const BindlessArrayUpdateCommand::Modification> modifications) const noexcept {}
    luisa::span<const Argument> shader_bindings(uint64_t handle) const noexcept {}
}
*/

concept ReorderFuncTable =
    requires(const T t, uint64_t uint64_v, size_t size_v, luisa::span<const BindlessArrayUpdateCommand::Modification> modification,
             CustomDispatchCommand const *cmd) {
        requires(std::is_same_v<bool, decltype(t.is_res_in_bindless(uint64_v, uint64_v))>);
        requires(std::is_same_v<Usage, decltype(t.get_usage(uint64_v, size_v))>);
        t.update_bindless(uint64_v, modification);
        t.lock_bindless(uint64_v);
        t.unlock_bindless(uint64_v);
        requires(std::is_same_v<luisa::span<const Argument>, decltype(t.shader_bindings(uint64_v))>);
    };

template<ReorderFuncTable FuncTable, bool supportConcurrentCopy, size_t fixedVectorSize = 2>
class CommandReorderVisitor : public CommandVisitor {

public:
    enum class ResourceRW : uint8_t {
        Read,
        Write
    };
    enum class ResourceType : uint8_t {
        Texture_Buffer,
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
        using Map = vstd::ArenaHashMap<ArenaRef, Range, ResourceView, RangeHash>;
    private:
        ResourceView max_view;
        Range read_range;
        Range write_range;
        Map views;
        static constexpr uint GIVEUP_SIZE = 16;

    public:
        RangeHandle(
            ArenaRef &&pool) : views(GIVEUP_SIZE, std::move(pool)) {
            read_range.min = std::numeric_limits<int64_t>::max();
            read_range.max = std::numeric_limits<int64_t>::min();
            write_range.min = std::numeric_limits<int64_t>::max();
            write_range.max = std::numeric_limits<int64_t>::min();
        }
        auto get_max_write_layer(Range const &range) {
            int64_t layer = -1;
            if (!range.collide(write_range))
                return layer;
            for (auto &&r : views) {
                if (r.first.collide(range)) {
                    layer = std::max<int64_t>(layer, r.second.write_layer);
                    if (layer >= max_view.write_layer) {
                        return layer;
                    }
                }
            }
            return layer;
        }
        auto get_max_read_layer(Range const &range) {
            int64_t layer = -1;
            if (!range.collide(read_range))
                return layer;
            for (auto &&r : views) {
                if (r.first.collide(range)) {
                    layer = std::max<int64_t>(layer, r.second.read_layer);
                    if (layer >= max_view.read_layer) {
                        return layer;
                    }
                }
            }
            return layer;
        }
        void clear_views() {
            views.clear();
            auto ite = views.try_emplace(read_range);
            auto &value = ite.first.value();
            value.read_layer = max_view.read_layer;
            value.write_layer = max_view.write_layer;
        };
        void emplace_read_layer(Range const &range, int64_t layer) {
            read_range.min = std::min(read_range.min, range.min);
            read_range.max = std::max(read_range.max, range.max);
            max_view.read_layer = std::max(layer, max_view.read_layer);
            if (views.size() >= GIVEUP_SIZE) {
                clear_views();
            } else {
                auto ite = views.try_emplace(range);
                auto &read_layer = ite.first.value().read_layer;
                if (ite.second) {
                    read_layer = layer;
                } else {
                    read_layer = std::max<int64_t>(read_layer, layer);
                }
            }
        }
        void emplace_write_layer(Range const &range, int64_t layer) {
            read_range.min = std::min(read_range.min, range.min);
            read_range.max = std::max(read_range.max, range.max);
            write_range.min = std::min(write_range.min, range.min);
            write_range.max = std::max(write_range.max, range.max);
            max_view.read_layer = std::max(layer, max_view.read_layer);
            max_view.write_layer = std::max(layer, max_view.write_layer);
            if (views.size() >= GIVEUP_SIZE) {
                clear_views();
            } else {
                auto ite = views.try_emplace(range);
                auto &read_layer = ite.first.value().read_layer;
                auto &write_layer = ite.first.value().write_layer;
                if (ite.second) {
                    read_layer = layer;
                    write_layer = layer;
                } else {
                    read_layer = std::max<int64_t>(read_layer, layer);
                    write_layer = std::max<int64_t>(write_layer, layer);
                }
            }
        }
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
    vstd::DefaultMallocVisitor malloc_visitor;
    vstd::StackAllocator _arena;
    vstd::ArenaHashMap<ArenaRef, uint64_t, RangeHandle *> _res_map;
    vstd::ArenaHashMap<ArenaRef, uint64_t, NoRangeHandle *> _no_range_resmap;
    vstd::ArenaHashMap<ArenaRef, uint64_t, BindlessHandle *> _bindless_map;
    vstd::ArenaHashMap<ArenaRef, uint64_t> _write_res_map;
    int64_t _bindless_max_layer = -1;
    int64_t _max_mesh_level = -1;
    int64_t _max_accel_read_level = -1;
    int64_t _max_accel_write_level = -1;
    struct CommandLink {
        Command const *cmd;
        CommandLink const *p_next;
    };
    vstd::vector<CommandLink const *> _cmd_lists;
    vstd::vector<std::pair<Range, ResourceHandle *>> _dispatch_read_handle;
    vstd::vector<std::pair<Range, ResourceHandle *>> _dispatch_write_handle;
    int64_t _dispatch_layer;
    bool _use_bindless_in_pass;
    bool _use_accel_in_pass;
    ResourceHandle *get_handle(
        uint64_t target_handle,
        ResourceType target_type) {
        auto func = [&](auto &&map) {
            auto try_result = map.try_emplace(
                target_handle);
            auto &&value = try_result.first.value();
            using Type = typename std::remove_pointer_t<std::remove_cvref_t<decltype(value)>>;
            if (try_result.second) {
                auto mem = _arena.allocate(sizeof(Type), alignof(Type));
                value = reinterpret_cast<Type *>(mem.handle + mem.offset);
                new (value) Type{};
                value->handle = target_handle;
                value->type = target_type;
            }
            return value;
        };
        switch (target_type) {
            case ResourceType::Bindless:
                return func(_bindless_map);
            case ResourceType::Mesh:
            case ResourceType::Accel:
                return func(_no_range_resmap);
            default: {
                auto try_result = _res_map.try_emplace(
                    target_handle);
                auto &&value = try_result.first.value();
                if (try_result.second) {
                    auto mem = _arena.allocate(sizeof(RangeHandle), alignof(RangeHandle));
                    value = reinterpret_cast<RangeHandle *>(mem.handle + mem.offset);
                    new (value) RangeHandle{ArenaRef{_arena}};
                    value->handle = target_handle;
                    value->type = target_type;
                }
                return value;
            }
        }
    }
    // Texture, Buffer
    int64_t get_last_layer_write(RangeHandle *handle, Range range) {
        int64_t layer = handle->get_max_read_layer(range);
        if (_bindless_max_layer >= layer) {
            for (auto &&i : _bindless_map) {
                _func_table.lock_bindless(i.first);
                if (_func_table.is_res_in_bindless(i.first, handle->handle)) {
                    layer = std::max<int64_t>(layer, i.second->view.read_layer);
                }
                _func_table.unlock_bindless(i.first);
            }
        }
        return layer + 1;
    }
    // Mesh, Accel
    int64_t get_last_layer_write(NoRangeHandle *handle) {
        int64_t layer = std::max<int64_t>(handle->view.read_layer, handle->view.write_layer);

        switch (handle->type) {
            case ResourceType::Mesh: {
                auto max_accel_level = std::max(_max_accel_read_level, _max_accel_write_level);
                layer = std::max<int64_t>(layer, max_accel_level);
            } break;
            case ResourceType::Accel: {
                auto max_accel_level = std::max(_max_accel_read_level, _max_accel_write_level);
                layer = std::max<int64_t>(layer, max_accel_level);
                layer = std::max<int64_t>(layer, _max_mesh_level);
            } break;
            default: break;
        }
        return layer + 1;
    }
    // Bindless
    int64_t get_last_layer_write(BindlessHandle *handle) {
        return std::max<int64_t>(handle->view.read_layer, handle->view.write_layer) + 1;
    }
    int64_t get_last_layer_read(RangeHandle *handle, Range range) {
        int64_t layer = handle->get_max_write_layer(range) + 1;
        return layer;
    }
    int64_t get_last_layer_read(NoRangeHandle *handle) {
        int64_t layer = handle->view.write_layer;
        if (handle->type == ResourceType::Accel) {
            layer = std::max<int64_t>(layer, _max_accel_write_level);
        }
        return layer + 1;
    }
    int64_t get_last_layer_read(BindlessHandle *handle) {
        return handle->view.write_layer + 1;
    }
    void add_command(Command const *cmd, int64_t layer) {
        if (_cmd_lists.size() <= layer) {
            _cmd_lists.resize(layer + 1);
        }
        auto &v = _cmd_lists[layer];
        auto new_cmd_list = _arena.allocate_memory<CommandLink, false>();
        new_cmd_list->cmd = cmd;
        new_cmd_list->p_next = v;
        v = new_cmd_list;
    }
    int64_t set_read(
        uint64_t handle,
        Range range,
        ResourceType type) {
        auto src_handle = get_handle(
            handle,
            type);
        return set_read(src_handle, range);
    }
    int64_t set_read(
        ResourceHandle *src_handle,
        Range range) {
        int64_t layer = 0;
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
                handle->emplace_read_layer(range, layer);
            } break;
        }
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
                handle->emplace_read_layer(range, layer);
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
                handle->emplace_write_layer(range, layer);
                _write_res_map.emplace(dst_handle->handle);
            } break;
        }
    }

    int64_t set_write(
        ResourceHandle *dst_handle,
        Range range) {
        int64_t layer = 0;
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
                handle->emplace_write_layer(range, layer);
                _write_res_map.emplace(dst_handle->handle);
            } break;
        }

        return layer;
    }
    int64_t set_write(
        uint64_t handle,
        Range range,
        ResourceType type) {
        auto dst_handle = get_handle(
            handle,
            type);
        return set_write(dst_handle, range);
    }
    int64_t set_rw(
        uint64_t read_handle,
        Range read_range,
        ResourceType read_type,
        uint64_t write_handle,
        Range write_range,
        ResourceType write_type) {

        int64_t layer = 0;
        auto src_handle = get_handle(
            read_handle,
            read_type);
        auto dst_handle = get_handle(
            write_handle,
            write_type);
        switch (read_type) {
            case ResourceType::Mesh:
            case ResourceType::Accel: {
                auto handle = static_cast<NoRangeHandle *>(src_handle);
                layer = get_last_layer_read(handle);
            } break;
            case ResourceType::Bindless: {
                auto handle = static_cast<BindlessHandle *>(src_handle);
                layer = get_last_layer_read(handle);
            } break;
            default: {
                auto handle = static_cast<RangeHandle *>(src_handle);
                layer = get_last_layer_read(handle, read_range);
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
                handle->emplace_write_layer(write_range, layer);
                _write_res_map.emplace(write_handle);
            } break;
        }
        // set_read_layer
        switch (read_type) {
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
                handle->emplace_read_layer(read_range, layer);
            } break;
        }
        return layer;
    }
    int64_t set_mesh(
        uint64_t handle,
        uint64_t vb,
        Range vb_range,
        uint64_t ib,
        Range ib_range) {

        auto vb_handle = get_handle(
            vb,
            ResourceType::Texture_Buffer);
        auto mesh_handle = get_handle(
            handle,
            ResourceType::Mesh);
        auto layer = get_last_layer_read(static_cast<RangeHandle *>(vb_handle), vb_range);
        layer = std::max<int64_t>(layer, get_last_layer_write(static_cast<NoRangeHandle *>(mesh_handle)));
        auto ib_handle = get_handle(
            ib,
            ResourceType::Texture_Buffer);
        auto range_handle = static_cast<RangeHandle *>(ib_handle);
        layer = std::max<int64_t>(layer, get_last_layer_read(range_handle, ib_range));
        range_handle->emplace_read_layer(ib_range, layer);
        static_cast<RangeHandle *>(vb_handle)->emplace_read_layer(vb_range, layer);
        static_cast<NoRangeHandle *>(mesh_handle)->view.write_layer = layer;
        _max_mesh_level = std::max<int64_t>(_max_mesh_level, layer);
        return layer;
    }
    int64_t set_aabb(
        uint64_t handle,
        uint64_t aabb_buffer,
        Range aabb_range) {
        auto vb_handle = get_handle(
            aabb_buffer,
            ResourceType::Texture_Buffer);
        auto mesh_handle = get_handle(
            handle,
            ResourceType::Mesh);
        auto layer = get_last_layer_read(static_cast<RangeHandle *>(vb_handle), aabb_range);
        layer = std::max<int64_t>(layer, get_last_layer_write(static_cast<NoRangeHandle *>(mesh_handle)));
        static_cast<RangeHandle *>(vb_handle)->emplace_read_layer(aabb_range, layer);
        static_cast<NoRangeHandle *>(mesh_handle)->view.write_layer = layer;
        _max_mesh_level = std::max<int64_t>(_max_mesh_level, layer);
        return layer;
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
                case ResourceType::Texture_Buffer:
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
                case ResourceType::Texture_Buffer:
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
    void visit(const CustomDispatchCommand *command) noexcept {
        _dispatch_read_handle.clear();
        _dispatch_write_handle.clear();
        _use_bindless_in_pass = false;
        _use_accel_in_pass = false;
        _dispatch_layer = 0;

        auto f = [&]<typename T>(T const &t, Usage usage) {
            if constexpr (std::is_same_v<T, Argument::Buffer>) {
                add_dispatch_handle(
                    t.handle,
                    ResourceType::Texture_Buffer,
                    Range(t.offset, t.size),
                    ((uint)usage & (uint)Usage::WRITE) != 0);
            } else if constexpr (std::is_same_v<T, Argument::Texture>) {
                add_dispatch_handle(
                    t.handle,
                    ResourceType::Texture_Buffer,
                    Range(t.level, 1),
                    ((uint)usage & (uint)Usage::WRITE) != 0);

            } else if constexpr (std::is_same_v<T, Argument::BindlessArray>) {
                _use_bindless_in_pass = true;
                {
                    _func_table.lock_bindless(t.handle);
                    auto unlocker = vstd::scope_exit([&] {
                        _func_table.unlock_bindless(t.handle);
                    });
                    for (auto &&res : _write_res_map) {
                        if (_func_table.is_res_in_bindless(t.handle, res)) {
                            add_dispatch_handle(
                                res,
                                ResourceType::Texture_Buffer,
                                Range{},
                                false);
                        }
                    }
                }
                add_dispatch_handle(
                    t.handle,
                    ResourceType::Bindless,
                    Range(),
                    false);
            } else {
                _use_accel_in_pass = true;
                add_dispatch_handle(
                    t.handle,
                    ResourceType::Accel,
                    Range(),
                    false);
            }
        };
        command->traverse_arguments(f);

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

    template<bool contain_binding, typename Callback>
    void visit(const ShaderDispatchCommandBase *command, const Command *cmd_base, uint64_t shader_handle, Callback callback) noexcept {
        _dispatch_read_handle.clear();
        _dispatch_write_handle.clear();
        _use_bindless_in_pass = false;
        _use_accel_in_pass = false;
        _dispatch_layer = 0;
        size_t arg_idx = 0;
        using Argument = ShaderDispatchCommandBase::Argument;
        using Tag = Argument::Tag;
        auto ite_arg = [&](auto &&i) {
            switch (i.tag) {
                case Tag::BUFFER: {
                    auto &&bf = i.buffer;
                    add_dispatch_handle(
                        bf.handle,
                        ResourceType::Texture_Buffer,
                        Range(bf.offset, bf.size),
                        ((uint)_func_table.get_usage(shader_handle, arg_idx) & (uint)Usage::WRITE) != 0);
                    ++arg_idx;
                } break;
                case Tag::TEXTURE: {
                    auto &&tex = i.texture;
                    add_dispatch_handle(
                        tex.handle,
                        ResourceType::Texture_Buffer,
                        Range(tex.level),
                        ((uint)_func_table.get_usage(shader_handle, arg_idx) & (uint)Usage::WRITE) != 0);
                    ++arg_idx;
                } break;
                case Tag::UNIFORM: {
                    ++arg_idx;
                } break;
                case Tag::BINDLESS_ARRAY: {
                    auto &&arr = i.bindless_array;
                    _use_bindless_in_pass = true;
                    {
                        _func_table.lock_bindless(arr.handle);
                        auto unlocker = vstd::scope_exit([&] {
                            _func_table.unlock_bindless(arr.handle);
                        });
                        for (auto &&res : _write_res_map) {
                            if (_func_table.is_res_in_bindless(arr.handle, res)) {
                                add_dispatch_handle(
                                    res,
                                    ResourceType::Texture_Buffer,
                                    Range{},
                                    false);
                            }
                        }
                    }
                    add_dispatch_handle(
                        arr.handle,
                        ResourceType::Bindless,
                        Range(),
                        false);
                    ++arg_idx;
                } break;
                case Tag::ACCEL: {
                    auto &&acc = i.accel;
                    _use_accel_in_pass = true;
                    add_dispatch_handle(
                        acc.handle,
                        ResourceType::Accel,
                        Range(),
                        false);
                    ++arg_idx;
                } break;
            }
        };
        if constexpr (contain_binding) {
            for (auto &&i : _func_table.shader_bindings(shader_handle)) {
                ite_arg(i);
            }
        }
        for (auto &&i : command->arguments()) {
            ite_arg(i);
        }
        callback();
        for (auto &&i : _dispatch_read_handle) {
            set_read_layer(i.second, i.first, _dispatch_layer);
        }
        for (auto &&i : _dispatch_write_handle) {
            set_write_layer(i.second, i.first, _dispatch_layer);
        }
        add_command(cmd_base, _dispatch_layer);
        if (_use_bindless_in_pass) {
            _bindless_max_layer = std::max<int64_t>(_bindless_max_layer, _dispatch_layer);
        }
        if (_use_accel_in_pass) {
            _max_accel_read_level = std::max<int64_t>(_max_accel_read_level, _dispatch_layer);
        }
    }

public:
    explicit CommandReorderVisitor(FuncTable &&func_table) noexcept
        : _arena(65536, &malloc_visitor),
          _res_map(64, ArenaRef{_arena}),
          _no_range_resmap(64, ArenaRef{_arena}),
          _bindless_map(64, ArenaRef{_arena}),
          _write_res_map(64, ArenaRef{_arena}),
          _func_table(std::forward<FuncTable>(func_table)) {
    }
    void clear() noexcept {
        auto re_construct_map = [&]<typename T>(T &t) {
            t.~T();
            new (&t) T(64, ArenaRef{_arena});
        };
        _bindless_max_layer = -1;
        _max_accel_read_level = -1;
        _max_accel_write_level = -1;
        _max_mesh_level = -1;
        _cmd_lists.clear();
        _arena.clear();
        re_construct_map(_res_map);
        re_construct_map(_no_range_resmap);
        re_construct_map(_bindless_map);
        re_construct_map(_write_res_map);
    }
    ~CommandReorderVisitor() noexcept {}
    [[nodiscard]] auto command_lists() const noexcept {
        return luisa::span{_cmd_lists};
    }

    // Buffer : resource
    void visit(const BufferUploadCommand *command) noexcept override {
        add_command(command, set_write(command->handle(), copy_range(command->offset(), command->size()), ResourceType::Texture_Buffer));
    }
    void visit(const BufferDownloadCommand *command) noexcept override {
        add_command(command, set_read(command->handle(), copy_range(command->offset(), command->size()), ResourceType::Texture_Buffer));
    }
    void visit(const BufferCopyCommand *command) noexcept override {
        add_command(command, set_rw(command->src_handle(), copy_range(command->src_offset(), command->size()), ResourceType::Texture_Buffer, command->dst_handle(), copy_range(command->dst_offset(), command->size()), ResourceType::Texture_Buffer));
    }
    void visit(const BufferToTextureCopyCommand *command) noexcept override {
        auto sz = command->size();
        auto bin_size = pixel_storage_size(command->storage(), sz);
        add_command(command, set_rw(command->buffer(), copy_range(command->buffer_offset(), bin_size), ResourceType::Texture_Buffer, command->texture(), copy_range(command->level(), 1), ResourceType::Texture_Buffer));
    }
    // Shader : function, read/write multi resources
    void visit(const ShaderDispatchCommand *command) noexcept override {
        visit<true>(command, command, command->handle(), [&] {
            if (command->is_indirect()) {
                auto &&t = command->indirect_dispatch();
                add_dispatch_handle(
                    t.handle,
                    ResourceType::Texture_Buffer,
                    Range(),
                    false);
            }
        });
    }

    // Texture : resource
    void visit(const TextureUploadCommand *command) noexcept override {
        add_command(command, set_write(command->handle(), copy_range(command->level(), 1), ResourceType::Texture_Buffer));
    }
    void visit(const TextureDownloadCommand *command) noexcept override {
        add_command(command, set_read(command->handle(), copy_range(command->level(), 1), ResourceType::Texture_Buffer));
    }
    void visit(const TextureCopyCommand *command) noexcept override {
        add_command(command, set_rw(command->src_handle(), copy_range(command->src_level(), 1), ResourceType::Texture_Buffer, command->dst_handle(), copy_range(command->dst_level(), 1), ResourceType::Texture_Buffer));
    }
    void visit(const TextureToBufferCopyCommand *command) noexcept override {
        auto sz = command->size();
        auto bin_size = pixel_storage_size(command->storage(), sz);
        add_command(command, set_rw(command->texture(), copy_range(command->level(), 1), ResourceType::Texture_Buffer, command->buffer(), copy_range(command->buffer_offset(), bin_size), ResourceType::Texture_Buffer));
    }
    void visit(const ClearDepthCommand *command) noexcept {
        add_command(command, set_write(command->handle(), Range{}, ResourceType::Texture_Buffer));
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

    void visit(const CurveBuildCommand *) noexcept override { /* TODO */
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
        add_command(
            command,
            set_aabb(
                command->handle(),
                command->aabb_buffer(),
                Range(command->aabb_buffer_offset(), command->aabb_buffer_size())));
    }

    void visit(const DrawRasterSceneCommand *command) noexcept {
        auto set_tex_dsl = [&](ShaderDispatchCommandBase::Argument::Texture const &a) {
            add_dispatch_handle(
                a.handle,
                ResourceType::Texture_Buffer,
                Range(a.level),
                true);
        };
        visit<false>(command, command, command->handle(), [&] {
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
                        ResourceType::Texture_Buffer,
                        Range(v.offset(), v.size()),
                        false);
                }
                auto &&i = mesh.index();
                if (i.index() == 0) {
                    auto idx = luisa::get<0>(i);
                    add_dispatch_handle(
                        idx.handle(),
                        ResourceType::Texture_Buffer,
                        Range(idx.offset_bytes(), idx.size_bytes()),
                        false);
                }
            }
        });
    }

    void visit(const CustomCommand *command) noexcept override {
        switch (command->uuid()) {
            case to_underlying(CustomCommandUUID::RASTER_CLEAR_DEPTH):
                visit(static_cast<ClearDepthCommand const *>(command));
                break;
            case to_underlying(CustomCommandUUID::RASTER_DRAW_SCENE):
                visit(static_cast<DrawRasterSceneCommand const *>(command));
                break;
            case to_underlying(CustomCommandUUID::CUSTOM_DISPATCH):
                visit(static_cast<CustomDispatchCommand const *>(command));
                break;
            default:
                LUISA_ERROR("Custom command not supported by reorder.");
        }
    }

    void visit(const MotionInstanceBuildCommand *) noexcept override {
        LUISA_NOT_IMPLEMENTED();
    }
};

}// namespace luisa::compute