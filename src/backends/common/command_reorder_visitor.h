#pragma once

#include <luisa/runtime/device.h>
#include <luisa/core/stl/hash.h>
#include <cstdint>
#include <limits>
#include <luisa/vstl/common.h>
#include <luisa/runtime/rhi/command.h>
#include <luisa/backends/ext/raster_cmd.h>
#include <luisa/backends/ext/work_graph_cmd.h>
#include <luisa/runtime/buffer.h>
#include <luisa/runtime/rhi/argument.h>
#include <luisa/core/logging.h>
#include <luisa/vstl/stack_allocator.h>
#include <luisa/vstl/arena_hash_map.h>

namespace luisa::compute {
struct ReorderBindlessResourceVisitor {
    void *context{};
    void (*callback)(void *context, uint64_t handle,
                     bool is_buffer) noexcept {};

    void operator()(uint64_t handle, bool is_buffer) const noexcept {
        callback(context, handle, is_buffer);
    }
};

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
    uint64_t canonical_buffer_handle(uint64_t handle) const noexcept {}
    uint64_t canonical_texture_handle(uint64_t handle) const noexcept {}
    void traverse_bindless_resources(
        uint64_t bindless_handle,
        ReorderBindlessResourceVisitor visitor) const noexcept {}
    Usage get_usage(uint64_t shader_handle, size_t argument_index) const noexcept {}
    void update_bindless(uint64_t handle, luisa::span<const BindlessArrayUpdateCommand::Modification> modifications) const noexcept {}
    void update_bindless(uint64_t handle, luisa::span<const BindlessArrayUpdateCommand::BufferModification> modifications) const noexcept {}
    void update_bindless(uint64_t handle, luisa::span<const BindlessArrayUpdateCommand::Texture2DModification> modifications) const noexcept {}
    void update_bindless(uint64_t handle, luisa::span<const BindlessArrayUpdateCommand::Texture3DModification> modifications) const noexcept {}
    luisa::span<const Argument> shader_bindings(uint64_t handle) const noexcept {}
    luisa::span<const Argument> raster_shader_bindings(uint64_t handle) const noexcept {}
}
*/

concept ReorderFuncTable =
    requires(const T t, uint64_t uint64_v, size_t size_v,
             luisa::span<const BindlessArrayUpdateCommand::Modification> modifications,
             luisa::span<const BindlessArrayUpdateCommand::BufferModification> buffer_modifications,
             luisa::span<const BindlessArrayUpdateCommand::Texture2DModification> texture_2d_modifications,
             luisa::span<const BindlessArrayUpdateCommand::Texture3DModification> texture_3d_modifications) {
        requires(std::is_same_v<Usage, decltype(t.get_usage(uint64_v, size_v))>);
        requires(std::is_same_v<uint64_t, decltype(t.canonical_buffer_handle(uint64_v))>);
        requires(std::is_same_v<uint64_t, decltype(t.canonical_texture_handle(uint64_v))>);
        t.update_bindless(uint64_v, modifications);
        t.update_bindless(uint64_v, buffer_modifications);
        t.update_bindless(uint64_v, texture_2d_modifications);
        t.update_bindless(uint64_v, texture_3d_modifications);
        t.traverse_bindless_resources(
            uint64_v, ReorderBindlessResourceVisitor{});
        requires(std::is_same_v<luisa::span<const Argument>, decltype(t.shader_bindings(uint64_v))>);
        requires(std::is_same_v<luisa::span<const Argument>, decltype(t.work_graph_bindings(uint64_v))>);
        requires(std::is_same_v<luisa::span<const Argument>, decltype(t.raster_shader_bindings(uint64_v))>);
    };

template<ReorderFuncTable FuncTable, bool supportConcurrentCopy, size_t fixedVectorSize = 2>
class CommandReorderVisitor : public CommandVisitor {

public:
    enum class ResourceRW : uint8_t {
        Read,
        Write
    };
    enum class ResourceType : uint8_t {
        Buffer,
        Texture,
        Mesh,
        Bindless,
        Accel
    };
    struct Range {
        uint64_t min;
        uint64_t max;

    private:
        struct FromBounds {};
        constexpr Range(uint64_t min, uint64_t max, FromBounds) noexcept
            : min{min}, max{max} {}

    public:
        [[nodiscard]] static constexpr Range empty() noexcept {
            return Range{std::numeric_limits<uint64_t>::max(), 0u, FromBounds{}};
        }
        [[nodiscard]] static constexpr Range whole() noexcept {
            return Range{0u, std::numeric_limits<uint64_t>::max(), FromBounds{}};
        }
        [[nodiscard]] static Range from_offset_size(uint64_t offset, uint64_t size) noexcept {
            LUISA_ASSERT(
                size <= std::numeric_limits<uint64_t>::max() - offset,
                "Resource range overflows: offset={}, size={}.",
                offset, size);
            return Range{offset, offset + size, FromBounds{}};
        }
        [[nodiscard]] static Range single(uint64_t value) noexcept {
            LUISA_ASSERT(
                value != std::numeric_limits<uint64_t>::max(),
                "Cannot represent a resource range starting at UINT64_MAX.");
            return Range{value, value + 1u, FromBounds{}};
        }
        [[nodiscard]] static constexpr Range suffix(uint64_t first) noexcept {
            return Range{first, std::numeric_limits<uint64_t>::max(), FromBounds{}};
        }
        [[nodiscard]] bool collide(Range const &r) const noexcept {
            return min < r.max && r.min < max;
        }
        [[nodiscard]] bool operator==(Range const &r) const noexcept {
            return min == r.min && max == r.max;
        }
        [[nodiscard]] bool operator!=(Range const &r) const noexcept { return !operator==(r); }
    };
    struct RangeHash {
        size_t operator()(Range const &r) const {
            return hash64(&r, sizeof(Range), hash64_default_seed);
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
        Range read_range{Range::empty()};
        Range write_range{Range::empty()};
        Map views;
        static constexpr uint GIVEUP_SIZE = 16;

    public:
        RangeHandle(
            ArenaRef &&pool) : views(GIVEUP_SIZE, std::move(pool)) {}
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
    [[nodiscard]] static constexpr Range whole_range() noexcept {
        return Range::whole();
    }
    [[nodiscard]] static Range buffer_range(size_t offset, size_t size) noexcept {
        return Range::from_offset_size(offset, size);
    }
    [[nodiscard]] static Range copy_buffer_range(size_t offset, size_t size) noexcept {
        if constexpr (supportConcurrentCopy) {
            return buffer_range(offset, size);
        } else {
            return whole_range();
        }
    }
    [[nodiscard]] static Range base_mip_range(uint32_t level) noexcept {
        return Range::single(level);
    }
    [[nodiscard]] static constexpr Range mip_suffix_range(uint32_t base_level) noexcept {
        return Range::suffix(base_level);
    }
    [[nodiscard]] static Range copy_tex_range(uint32_t level) noexcept {
        return base_mip_range(level);
    }
    vstd::DefaultMallocVisitor malloc_visitor;
    vstd::StackAllocator _arena;
    // Buffers and textures occupy distinct native handle namespaces. Equal
    // canonical bit patterns only alias within the same resource kind.
    vstd::ArenaHashMap<ArenaRef, uint64_t, RangeHandle *> _buffer_map;
    vstd::ArenaHashMap<ArenaRef, uint64_t, RangeHandle *> _texture_map;
    vstd::ArenaHashMap<ArenaRef, uint64_t, NoRangeHandle *> _no_range_resmap;
    vstd::ArenaHashMap<ArenaRef, uint64_t, BindlessHandle *> _bindless_map;
    static constexpr uint64_t max_allowed_dispatch_size = 65535ull * 32ull;
    vstd::vector<uint64> _max_dispatch_blocks;
    int64_t _max_mesh_level = -1;
    int64_t _max_accel_read_level = -1;
    int64_t _max_accel_write_level = -1;
    struct CommandLink {
        Command const *cmd;
        CommandLink const *p_next;
    };
    vstd::vector<CommandLink const *> _cmd_lists;
    vstd::vector<CommandLink *> _cmd_list_tails;
    vstd::vector<std::pair<Range, ResourceHandle *>> _dispatch_read_handle;
    vstd::vector<std::pair<Range, ResourceHandle *>> _dispatch_write_handle;
    int64_t _dispatch_layer;
    bool _use_accel_in_pass;
    bool _write_accel_in_pass;
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
        auto range_func = [&](auto &map, uint64_t canonical_handle) {
            auto try_result = map.try_emplace(canonical_handle);
            auto &&value = try_result.first.value();
            if (try_result.second) {
                auto mem = _arena.allocate(sizeof(RangeHandle), alignof(RangeHandle));
                value = reinterpret_cast<RangeHandle *>(mem.handle + mem.offset);
                new (value) RangeHandle{ArenaRef{_arena}};
                value->handle = canonical_handle;
                value->type = target_type;
            }
            return value;
        };
        switch (target_type) {
            case ResourceType::Buffer:
                return range_func(
                    _buffer_map,
                    _func_table.canonical_buffer_handle(target_handle));
            case ResourceType::Texture:
                return range_func(
                    _texture_map,
                    _func_table.canonical_texture_handle(target_handle));
            case ResourceType::Bindless:
                return func(_bindless_map);
            case ResourceType::Mesh:
            case ResourceType::Accel:
                return func(_no_range_resmap);
        }
        LUISA_ASSUME(false);
        return nullptr;
    }
    // Texture, Buffer
    int64_t get_last_layer_write(RangeHandle *handle, Range range) {
        auto layer = std::max(
            handle->get_max_read_layer(range),
            handle->get_max_write_layer(range));
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
        if (static_cast<int64_t>(_cmd_lists.size()) <= layer) {
            _cmd_lists.resize(layer + 1);
            _cmd_list_tails.resize(layer + 1);
        }
        auto &head = _cmd_lists[layer];
        auto &tail = _cmd_list_tails[layer];
        auto new_cmd_list = _arena.allocate_memory<CommandLink, false>();
        new_cmd_list->cmd = cmd;
        new_cmd_list->p_next = nullptr;
        if (tail == nullptr) {
            head = new_cmd_list;
        } else {
            tail->p_next = new_cmd_list;
        }
        tail = new_cmd_list;
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
            ResourceType::Buffer);
        auto mesh_handle = get_handle(
            handle,
            ResourceType::Mesh);
        auto layer = get_last_layer_read(static_cast<RangeHandle *>(vb_handle), vb_range);
        layer = std::max<int64_t>(layer, get_last_layer_write(static_cast<NoRangeHandle *>(mesh_handle)));
        auto ib_handle = get_handle(
            ib,
            ResourceType::Buffer);
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
            ResourceType::Buffer);
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

    void add_texture_dispatch_handles(
        uint64_t handle,
        uint32_t base_level,
        Usage usage) {
        auto bits = static_cast<uint>(usage);
        auto reads = (bits & static_cast<uint>(Usage::READ)) != 0u ||
                     usage == Usage::NONE;
        auto writes = (bits & static_cast<uint>(Usage::WRITE)) != 0u;
        LUISA_ASSERT(reads || writes, "Texture argument has an invalid empty usage mask.");
        // Sampling may select any mip from the bound base level onward. Storage
        // image access targets only the explicitly bound base mip.
        if (reads) {
            add_dispatch_handle(
                handle,
                ResourceType::Texture,
                mip_suffix_range(base_level),
                false);
        }
        if (writes) {
            add_dispatch_handle(
                handle,
                ResourceType::Texture,
                base_mip_range(base_level),
                true);
        }
    }

    void add_bindless_dispatch_handles(
        uint64_t bindless_handle,
        bool writes_buffers,
        bool isolate_resource_states = false) {
        struct TraversalContext {
            CommandReorderVisitor *reorder;
            bool writes_buffers;
            bool isolate_resource_states;
        } context{this, writes_buffers, isolate_resource_states};
        _func_table.traverse_bindless_resources(
            bindless_handle,
            ReorderBindlessResourceVisitor{
                .context = &context,
                .callback = [](void *opaque, uint64_t resource_handle,
                               bool is_buffer) noexcept {
                    auto &state = *static_cast<TraversalContext *>(opaque);
                    state.reorder->add_dispatch_handle(
                        resource_handle,
                        is_buffer ? ResourceType::Buffer : ResourceType::Texture,
                        whole_range(),
                        state.isolate_resource_states ||
                            (is_buffer && state.writes_buffers));
                }});
        // Shader access reads the bindless index/descriptor object itself.
        // Writes normally apply only to the snapshotted buffer resources
        // above. A native state contract also serializes the descriptor
        // snapshot so no update or other access can share its reorder layer.
        add_dispatch_handle(
            bindless_handle,
            ResourceType::Bindless,
            whole_range(),
            isolate_resource_states);
    }

    FuncTable _func_table;
    void visit(const CustomDispatchCommand *command) noexcept {
        _dispatch_read_handle.clear();
        _dispatch_write_handle.clear();
        _use_accel_in_pass = false;
        _write_accel_in_pass = false;
        _dispatch_layer = 0;
        auto isolate_resource_states =
            command->requires_resource_state_isolation();

        auto f = [&]<typename T>(T const &t, Usage usage) {
            if constexpr (std::is_same_v<T, Argument::Buffer>) {
                add_dispatch_handle(
                    t.handle,
                    ResourceType::Buffer,
                    buffer_range(t.offset, t.size),
                    isolate_resource_states ||
                        ((uint)usage & (uint)Usage::WRITE) != 0);
            } else if constexpr (std::is_same_v<T, Argument::Texture>) {
                if (isolate_resource_states) {
                    add_dispatch_handle(
                        t.handle,
                        ResourceType::Texture,
                        base_mip_range(t.level),
                        true);
                } else {
                    add_texture_dispatch_handles(t.handle, t.level, usage);
                }

            } else if constexpr (std::is_same_v<T, Argument::BindlessArray>) {
                add_bindless_dispatch_handles(
                    t.handle,
                    (static_cast<uint>(usage) &
                     static_cast<uint>(Usage::WRITE)) != 0u,
                    isolate_resource_states);
            } else {
                _use_accel_in_pass = true;
                auto is_write =
                    isolate_resource_states ||
                    (static_cast<uint>(usage) &
                     static_cast<uint>(Usage::WRITE)) != 0u;
                _write_accel_in_pass |= is_write;
                add_dispatch_handle(
                    t.handle,
                    ResourceType::Accel,
                    whole_range(),
                    is_write);
            }
        };
        command->traverse_arguments(f);
        auto max_disp_size_vec = command->max_dispatch_size();
        auto max_disp_size = std::max<size_t>(max_disp_size_vec.x, std::max(max_disp_size_vec.y, max_disp_size_vec.z));
        if (_dispatch_layer >= static_cast<int64_t>(_max_dispatch_blocks.size())) {
            _max_dispatch_blocks.resize(_dispatch_layer + 1);
        }
        while (_max_dispatch_blocks[_dispatch_layer] > 0u &&
               (max_disp_size > max_allowed_dispatch_size ||
                _max_dispatch_blocks[_dispatch_layer] >
                    max_allowed_dispatch_size - max_disp_size)) {
            _dispatch_layer++;
            if (_dispatch_layer == _max_dispatch_blocks.size()) {
                _max_dispatch_blocks.emplace_back(0);
                break;
            }
        }
        _max_dispatch_blocks[_dispatch_layer] += max_disp_size;

        for (auto &&i : _dispatch_read_handle) {
            set_read_layer(i.second, i.first, _dispatch_layer);
        }
        for (auto &&i : _dispatch_write_handle) {
            set_write_layer(i.second, i.first, _dispatch_layer);
        }
        add_command(command, _dispatch_layer);
        if (_use_accel_in_pass) {
            if (_write_accel_in_pass) {
                _max_accel_write_level = std::max<int64_t>(_max_accel_write_level, _dispatch_layer);
            } else {
                _max_accel_read_level = std::max<int64_t>(_max_accel_read_level, _dispatch_layer);
            }
        }
    }

    template<typename Callback>
    void visit(const ShaderDispatchCommandBase *command,
               const Command *cmd_base,
               uint64_t shader_handle,
               luisa::span<const Argument> captured_bindings,
               Callback callback) noexcept {
        _dispatch_read_handle.clear();
        _dispatch_write_handle.clear();
        _use_accel_in_pass = false;
        _write_accel_in_pass = false;
        _dispatch_layer = 0;
        size_t arg_idx = 0;
        using Argument = ShaderDispatchCommandBase::Argument;
        using Tag = Argument::Tag;
        auto ite_arg = [&](auto &&i) {
            switch (i.tag) {
                case Tag::BUFFER: {
                    auto &&bf = i.buffer;
                    bool is_write = ((uint)_func_table.get_usage(shader_handle, arg_idx) & (uint)Usage::WRITE) != 0;
                    add_dispatch_handle(
                        bf.handle,
                        ResourceType::Buffer,
                        buffer_range(bf.offset, bf.size),
                        is_write);
                    ++arg_idx;
                } break;
                case Tag::TEXTURE: {
                    auto &&tex = i.texture;
                    add_texture_dispatch_handles(
                        tex.handle,
                        tex.level,
                        _func_table.get_usage(shader_handle, arg_idx));
                    ++arg_idx;
                } break;
                case Tag::UNIFORM: {
                    ++arg_idx;
                } break;
                case Tag::BINDLESS_ARRAY: {
                    auto &&arr = i.bindless_array;
                    auto usage = _func_table.get_usage(
                        shader_handle, arg_idx);
                    add_bindless_dispatch_handles(
                        arr.handle,
                        (static_cast<uint>(usage) &
                         static_cast<uint>(Usage::WRITE)) != 0u);
                    ++arg_idx;
                } break;
                case Tag::ACCEL: {
                    auto &&acc = i.accel;
                    _use_accel_in_pass = true;
                    auto is_write = (static_cast<uint>(_func_table.get_usage(shader_handle, arg_idx)) &
                                     static_cast<uint>(Usage::WRITE)) != 0u;
                    _write_accel_in_pass |= is_write;
                    add_dispatch_handle(
                        acc.handle,
                        ResourceType::Accel,
                        whole_range(),
                        is_write);
                    ++arg_idx;
                } break;
            }
        };
        for (auto &&i : captured_bindings) {
            ite_arg(i);
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
        if (_use_accel_in_pass) {
            if (_write_accel_in_pass) {
                _max_accel_write_level = std::max<int64_t>(_max_accel_write_level, _dispatch_layer);
            } else {
                _max_accel_read_level = std::max<int64_t>(_max_accel_read_level, _dispatch_layer);
            }
        }
    }

public:
    explicit CommandReorderVisitor(FuncTable &&func_table) noexcept
        : _arena(65536, &malloc_visitor),
          _buffer_map(64, ArenaRef{_arena}),
          _texture_map(64, ArenaRef{_arena}),
          _no_range_resmap(64, ArenaRef{_arena}),
          _bindless_map(64, ArenaRef{_arena}),
          _func_table(std::forward<FuncTable>(func_table)) {
    }
    void clear() noexcept {
        auto destroy_map = []<typename T>(T &t) noexcept {
            t.~T();
        };
        _max_accel_read_level = -1;
        _max_accel_write_level = -1;
        _max_mesh_level = -1;
        _dispatch_layer = 0;
        _use_accel_in_pass = false;
        _write_accel_in_pass = false;
        _cmd_lists.clear();
        _cmd_list_tails.clear();
        _max_dispatch_blocks.clear();
        _dispatch_read_handle.clear();
        _dispatch_write_handle.clear();
        destroy_map(_buffer_map);
        destroy_map(_texture_map);
        destroy_map(_no_range_resmap);
        destroy_map(_bindless_map);
        _arena.clear();
        new (&_buffer_map) decltype(_buffer_map)(64, ArenaRef{_arena});
        new (&_texture_map) decltype(_texture_map)(64, ArenaRef{_arena});
        new (&_no_range_resmap) decltype(_no_range_resmap)(64, ArenaRef{_arena});
        new (&_bindless_map) decltype(_bindless_map)(64, ArenaRef{_arena});
    }
    ~CommandReorderVisitor() noexcept {}
    [[nodiscard]] auto command_lists() const noexcept {
        return luisa::span{_cmd_lists};
    }

    // Buffer : resource
    void visit(const BufferUploadCommand *command) noexcept override {
        add_command(command, set_write(command->handle(), copy_buffer_range(command->offset(), command->size()), ResourceType::Buffer));
    }
    void visit(const BufferDownloadCommand *command) noexcept override {
        add_command(command, set_read(command->handle(), copy_buffer_range(command->offset(), command->size()), ResourceType::Buffer));
    }
    void visit(const BufferCopyCommand *command) noexcept override {
        add_command(command, set_rw(command->src_handle(), copy_buffer_range(command->src_offset(), command->size()), ResourceType::Buffer, command->dst_handle(), copy_buffer_range(command->dst_offset(), command->size()), ResourceType::Buffer));
    }
    void visit(const BufferToTextureCopyCommand *command) noexcept override {
        auto sz = command->size();
        auto bin_size = pixel_storage_size(command->storage(), sz);
        add_command(command, set_rw(command->buffer(), copy_buffer_range(command->buffer_offset(), bin_size), ResourceType::Buffer, command->texture(), copy_tex_range(command->level()), ResourceType::Texture));
    }
    // Shader : function, read/write multi resources
    void visit(const ShaderDispatchCommand *command) noexcept override {
        visit(command, command, command->handle(), _func_table.shader_bindings(command->handle()), [&] {
            // Resource dependencies determine the final layer. Register the
            // indirect source before charging this dispatch's workload to
            // that layer; otherwise a dependency can move the command after
            // the accounting has already been applied to an earlier layer.
            if (command->is_indirect()) {
                auto &&indirect = command->indirect_dispatch();
                add_dispatch_handle(
                    indirect.handle,
                    ResourceType::Buffer,
                    whole_range(),
                    false);
            }
            uint64_t max_disp_size = 0;
            if (command->is_multiple_dispatch()) {
                for (auto &&i : command->dispatch_sizes()) {
                    auto dispatch_work = static_cast<uint64_t>(
                        std::max(i.x, std::max(i.y, i.z)));
                    if (dispatch_work >
                        std::numeric_limits<uint64_t>::max() -
                            max_disp_size) {
                        max_disp_size =
                            std::numeric_limits<uint64_t>::max();
                        break;
                    }
                    max_disp_size += dispatch_work;
                }
            } else if (!command->is_indirect()) {
                auto i = command->dispatch_size();
                max_disp_size = std::max(i.x, std::max(i.y, i.z));
            } else {
                auto i = command->indirect_dispatch().max_dispatch_size;
                max_disp_size = i;
            }
            if (_dispatch_layer >= static_cast<int64_t>(_max_dispatch_blocks.size())) {
                _max_dispatch_blocks.resize(_dispatch_layer + 1);
            }
            // An oversized indirect upper bound is allowed to occupy one
            // otherwise-empty layer. Never skip an empty layer merely because
            // the public default bound is UINT32_MAX.
            while (_max_dispatch_blocks[_dispatch_layer] > 0u &&
                   (max_disp_size > max_allowed_dispatch_size ||
                    _max_dispatch_blocks[_dispatch_layer] >
                        max_allowed_dispatch_size - max_disp_size)) {
                _dispatch_layer++;
                if (_dispatch_layer == static_cast<int64_t>(_max_dispatch_blocks.size())) {
                    _max_dispatch_blocks.emplace_back(0);
                    break;
                }
            }
            _max_dispatch_blocks[_dispatch_layer] += max_disp_size;
        });
    }

    // Texture : resource
    void visit(const TextureUploadCommand *command) noexcept override {
        add_command(command, set_write(command->handle(), copy_tex_range(command->level()), ResourceType::Texture));
    }
    void visit(const TextureDownloadCommand *command) noexcept override {
        add_command(command, set_read(command->handle(), copy_tex_range(command->level()), ResourceType::Texture));
    }
    void visit(const TextureCopyCommand *command) noexcept override {
        add_command(command, set_rw(command->src_handle(), copy_tex_range(command->src_level()), ResourceType::Texture, command->dst_handle(), copy_tex_range(command->dst_level()), ResourceType::Texture));
    }
    void visit(const TextureToBufferCopyCommand *command) noexcept override {
        auto sz = command->size();
        auto bin_size = pixel_storage_size(command->storage(), sz);
        add_command(command, set_rw(command->texture(), copy_tex_range(command->level()), ResourceType::Texture, command->buffer(), copy_buffer_range(command->buffer_offset(), bin_size), ResourceType::Buffer));
    }
    void visit(const ClearDepthCommand *command) noexcept {
        add_command(command, set_write(command->handle(), whole_range(), ResourceType::Texture));
    }
    void visit(const ClearRenderTargetCommand *command) noexcept {
        add_command(command, set_write(command->handle(), base_mip_range(command->level()), ResourceType::Texture));
    }

    // BindlessArray : read multi resources
    void visit(const BindlessArrayUpdateCommand *command) noexcept override {
        command->visit_modifications([&](auto &&mods) {
            _func_table.update_bindless(command->handle(), luisa::span{mods});
        });
        add_command(command, set_write(command->handle(), whole_range(), ResourceType::Bindless));
    }

    // Accel : conclude meshes and their buffer
    void visit(const AccelBuildCommand *command) noexcept override {
        auto layer = set_write(command->handle(), whole_range(), ResourceType::Accel);
        _max_accel_write_level = std::max<int64_t>(_max_accel_write_level, layer);
        add_command(command, layer);
    }

    void visit(const CurveBuildCommand *) noexcept override {
        LUISA_ERROR("Curve build commands are not supported by command reordering.");
    }

    // Mesh : conclude vertex and triangle buffers
    void visit(const MeshBuildCommand *command) noexcept override {
        add_command(
            command,
            set_mesh(
                command->handle(),
                command->vertex_buffer(),
                buffer_range(command->vertex_buffer_offset(),
                             command->vertex_buffer_size()),
                command->triangle_buffer(),
                buffer_range(command->triangle_buffer_offset(),
                             command->triangle_buffer_size())));
    }
    void visit(const ProceduralPrimitiveBuildCommand *command) noexcept override {
        add_command(
            command,
            set_aabb(
                command->handle(),
                command->aabb_buffer(),
                buffer_range(command->aabb_buffer_offset(), command->aabb_buffer_size())));
    }

    void visit(const DrawRasterSceneCommand *command) noexcept {
        auto set_tex_dsl = [&](ShaderDispatchCommandBase::Argument::Texture const &a) {
            add_dispatch_handle(
                a.handle,
                ResourceType::Texture,
                base_mip_range(a.level),
                true);
        };
        visit(command, command, command->handle(), _func_table.raster_shader_bindings(command->handle()), [&] {
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
                        buffer_range(v.offset(), v.size()),
                        false);
                }
                auto &&i = mesh.index();
                if (i.index() == 0) {
                    auto idx = luisa::get<0>(i);
                    add_dispatch_handle(
                        idx.handle(),
                        ResourceType::Buffer,
                        buffer_range(idx.offset_bytes(), idx.size_bytes()),
                        false);
                }
            }
        });
    }

    void visit(const CustomCommand *custom_cmd) noexcept override {
        uint64_t uuid_value = custom_cmd->custom_cmd_uuid();
        switch (uuid_value) {
            case to_underlying(CustomCommandUUID::RASTER_CLEAR_DEPTH):
                visit(static_cast<ClearDepthCommand const *>(custom_cmd));
                break;
            case to_underlying(CustomCommandUUID::RASTER_CLEAR_RENDER_TARGET):
                visit(static_cast<ClearRenderTargetCommand const *>(custom_cmd));
                break;
            case to_underlying(CustomCommandUUID::RASTER_DRAW_SCENE):
                visit(static_cast<DrawRasterSceneCommand const *>(custom_cmd));
                break;
            case to_underlying(CustomCommandUUID::CUSTOM_DISPATCH):
                visit(static_cast<CustomDispatchCommand const *>(custom_cmd));
                break;
            case to_underlying(CustomCommandUUID::WORK_GRAPH_DISPATCH):
                visit(static_cast<WorkGraphDispatchCommand const *>(custom_cmd));
                break;
            default:
                LUISA_ERROR("Custom command not supported by reorder.");
        }
    }

    void visit(const WorkGraphDispatchCommand *command) noexcept {
        // Track resource dependencies for work graph dispatch, just like shader dispatch.
        // All arguments come from the program's captured bindings (no per-dispatch arguments).
        // TODO: unify this with the normal shader path;
        //       WorkGraphDispatchCommand should probably inherit from ShaderDispatchCommandBase
        _dispatch_read_handle.clear();
        _dispatch_write_handle.clear();
        _use_bindless_in_pass = false;
        _use_accel_in_pass = false;
        _dispatch_layer = 0;
        size_t arg_idx = 0;
        auto shader_handle = command->handle();
        using Tag = Argument::Tag;

        for (auto &&i : _func_table.work_graph_bindings(shader_handle)) {
            switch (i.tag) {
                case Tag::BUFFER: {
                    auto &&bf = i.buffer;
                    bool is_write = ((uint)_func_table.work_graph_get_usage(shader_handle, arg_idx) & (uint)Usage::WRITE) != 0;
                    Range buffer_range(bf.offset, bf.size);
                    add_dispatch_handle(
                        bf.handle,
                        ResourceType::Texture_Buffer,
                        buffer_range,
                        is_write);
                    ++arg_idx;
                } break;
                case Tag::TEXTURE: {
                    auto &&tex = i.texture;
                    add_dispatch_handle(
                        tex.handle,
                        ResourceType::Texture_Buffer,
                        Range(tex.level),
                        ((uint)_func_table.work_graph_get_usage(shader_handle, arg_idx) & (uint)Usage::WRITE) != 0);
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

    void visit(const MotionInstanceBuildCommand *command) noexcept override {
        // Register as a write to the motion instance handle to ensure
        // it's ordered before AccelBuildCommand that references it.
        add_command(command, set_write(command->handle(), whole_range(), ResourceType::Accel));
    }
};

}// namespace luisa::compute
