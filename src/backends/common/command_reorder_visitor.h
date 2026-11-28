#pragma once

#include <luisa/runtime/device.h>
#include <luisa/core/stl/hash.h>
#include <cstdint>
#include <limits>
#include <memory>
#include <utility>
#include <luisa/vstl/common.h>
#include <luisa/runtime/rhi/command.h>
#include <luisa/backends/ext/raster_cmd.h>
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
        requires(std::is_same_v<luisa::span<const Argument>, decltype(t.raster_shader_bindings(uint64_v))>);
    };

namespace detail {
// Keep the generic callable outside the requires-expression for GCC 13.
struct ReorderBindlessPendingLookupProbe {
    template<typename Contains>
    void operator()(Contains &&contains) const noexcept {
        [[maybe_unused]] bool found = contains(uint64_t{0u}, true);
    }
};
struct ReorderBindlessPendingStateProbe {
    template<typename Contains>
    void operator()(size_t, bool, bool, Contains &&contains) const noexcept {
        [[maybe_unused]] bool found = contains(uint64_t{0u}, true);
    }
};
}// namespace detail

// Optional fast-path interface for backends whose BindlessArray maintains an
// O(1) canonical pending-resource set. When satisfied, CommandReorderVisitor
// avoids traversing the full bindless array (potentially 10k+ resources) on
// every shader dispatch and instead checks only the resources actively tracked
// in this command batch, materializing lazily only when a mid-batch update
// mutates a previously-dispatched bindless array.
template<typename T>
concept ReorderFastBindlessFuncTable =
    ReorderFuncTable<T> &&
    requires(const T t, uint64_t bindless_handle) {
        requires(std::is_same_v<size_t, decltype(t.bindless_pending_count(bindless_handle))>);
        requires(std::is_same_v<bool, decltype(t.bindless_has_pending_buffers(bindless_handle))>);
        t.with_bindless_pending_lookup(
            bindless_handle, detail::ReorderBindlessPendingLookupProbe{});
    };

template<typename T>
concept ReorderSingleLockBindlessFuncTable =
    ReorderFastBindlessFuncTable<T> &&
    requires(const T t, uint64_t bindless_handle) {
        t.with_bindless_pending_state(
            bindless_handle, detail::ReorderBindlessPendingStateProbe{});
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
    struct ResourceView {
        int64_t read_layer = -1;
        int64_t write_layer = -1;
    };
    struct ResourceHandle {
        uint64_t handle;
        ResourceType type;
    };
    struct RangeHandle : public ResourceHandle {
    private:
        ResourceView max_view;
        Range read_range{Range::empty()};
        Range write_range{Range::empty()};
        // Disjoint intervals sorted by `min`, each carrying the highest
        // read/write layer recorded for any access merged into it. Insertions
        // clip colliding intervals at the new range's boundaries: the pieces
        // outside keep their intervals and layers, the inside is replaced by
        // the new range carrying the absorbed maximum layers. That keeps
        // tracking precise for arbitrarily many disjoint ranges and
        // near-optimal for overlapping ones. The previous design kept a
        // bounded hash map of exact ranges and collapsed to a single union
        // view once it overflowed, so any later access touching a previously
        // recorded region was conservatively serialized (a false hazard) and
        // range tracking silently degraded for batches touching more than 16
        // distinct ranges of one resource.
        vstd::fixed_vector<std::pair<Range, ResourceView>, 2> views;
        bool tracked_written{false};
        uint64_t bindless_checked_mask{0u};
        uint64_t bindless_member_mask{0u};

        // First interval that can collide with `range`: the intervals are
        // sorted and disjoint, so interval `i + 1` starts at or after
        // interval `i`'s end and a binary search on `max` suffices.
        [[nodiscard]] size_t _lower_bound(const Range &range) const noexcept {
            size_t lo = 0u, hi = views.size();
            while (lo < hi) {
                auto mid = lo + (hi - lo) / 2u;
                if (views[mid].first.max <= range.min) {
                    lo = mid + 1u;
                } else {
                    hi = mid;
                }
            }
            return lo;
        }
        // Insert the interval for one access, clipping the colliding window.
        // `read_layer` and `write_layer` are this access's own contributions;
        // the colliding intervals contribute their stored layers on top. Every
        // recorded access stays covered by an interval whose layers bound the
        // access's own from above (sound), and pieces of old intervals outside
        // the new range keep their exact layers (precise).
        void _emplace_layer(const Range &range, int64_t read_layer, int64_t write_layer) {
            if (views.empty()) {
                views.emplace_back(range, ResourceView{read_layer, write_layer});
                return;
            }
            if (views.size() == 1u && views[0].first == range) {
                auto &v = views[0].second;
                v.read_layer = std::max(v.read_layer, read_layer);
                v.write_layer = std::max(v.write_layer, write_layer);
                return;
            }
            auto lo = _lower_bound(range);
            auto hi = lo;
            while (hi < views.size() && views[hi].first.collide(range)) {
                auto &&view = views[hi].second;
                read_layer = std::max(read_layer, view.read_layer);
                write_layer = std::max(write_layer, view.write_layer);
                hi++;
            }
            if (hi == lo) {
                views.insert(views.begin() + lo,
                             {range, ResourceView{read_layer, write_layer}});
                return;
            }
            // Copy the boundary entries before mutating the vector, drop the
            // colliding window, and insert the replacement entries (1-3 of
            // them) at its position: the window can be shorter than the
            // replacement (one interval clipped on both sides becomes three
            // entries), so direct index assignment would write out of bounds.
            auto first = views[lo];
            auto last = views[hi - 1u];
            views.erase(views.begin() + lo, views.begin() + hi);
            auto pos = views.begin() + lo;
            if (first.first.min < range.min) {
                pos = views.insert(pos,
                                   std::pair{Range::from_offset_size(first.first.min, range.min - first.first.min),
                                             first.second}) +
                    1;
            }
            pos = views.insert(pos, std::pair{range, ResourceView{read_layer, write_layer}}) + 1;
            if (range.max < last.first.max) {
                views.insert(pos,
                             std::pair{Range::from_offset_size(range.max, last.first.max - range.max),
                                       last.second});
            }
        }

    public:
        RangeHandle() noexcept {}
        [[nodiscard]] bool is_tracked_written() const noexcept { return tracked_written; }
        void mark_tracked_written() noexcept { tracked_written = true; }
        [[nodiscard]] bool has_bindless_membership_cached(uint32_t slot) const noexcept {
            return slot < 64u && (bindless_checked_mask & (uint64_t{1u} << slot)) != 0u;
        }
        [[nodiscard]] bool cached_bindless_member(uint32_t slot) const noexcept {
            return (bindless_member_mask & (uint64_t{1u} << slot)) != 0u;
        }
        void cache_bindless_membership(uint32_t slot, bool is_member) noexcept {
            if (slot < 64u) {
                auto const bit = uint64_t{1u} << slot;
                bindless_checked_mask |= bit;
                if (is_member) {
                    bindless_member_mask |= bit;
                }
            }
        }
        void clear() noexcept {
            max_view = {};
            read_range = Range::empty();
            write_range = Range::empty();
            views.clear();
            tracked_written = false;
            bindless_checked_mask = 0u;
            bindless_member_mask = 0u;
        }
        [[nodiscard]] auto get_max_write_layer(Range const &range) const noexcept {
            if (!range.collide(write_range))
                return int64_t{-1};
            if (views.size() == 1u && views[0].first == range) {
                return views[0].second.write_layer;
            }
            int64_t layer = -1;
            for (auto i = _lower_bound(range); i < views.size(); i++) {
                auto &&[iv, view] = views[i];
                // Sorted: no interval at or after this one collides, and every
                // iterated one does.
                if (iv.min >= range.max) {
                    break;
                }
                layer = std::max<int64_t>(layer, view.write_layer);
                if (layer >= max_view.write_layer) {
                    break;
                }
            }
            return layer;
        }
        [[nodiscard]] auto get_max_read_layer(Range const &range) const noexcept {
            if (!range.collide(read_range))
                return int64_t{-1};
            if (views.size() == 1u && views[0].first == range) {
                return views[0].second.read_layer;
            }
            int64_t layer = -1;
            for (auto i = _lower_bound(range); i < views.size(); i++) {
                auto &&[iv, view] = views[i];
                if (iv.min >= range.max) {
                    break;
                }
                layer = std::max<int64_t>(layer, view.read_layer);
                if (layer >= max_view.read_layer) {
                    break;
                }
            }
            return layer;
        }
        void emplace_read_layer(Range const &range, int64_t layer) {
            read_range.min = std::min(read_range.min, range.min);
            read_range.max = std::max(read_range.max, range.max);
            max_view.read_layer = std::max(layer, max_view.read_layer);
            _emplace_layer(range, layer, -1);
        }
        void emplace_write_layer(Range const &range, int64_t layer) {
            read_range.min = std::min(read_range.min, range.min);
            read_range.max = std::max(read_range.max, range.max);
            write_range.min = std::min(write_range.min, range.min);
            write_range.max = std::max(write_range.max, range.max);
            max_view.read_layer = std::max(layer, max_view.read_layer);
            max_view.write_layer = std::max(layer, max_view.write_layer);
            _emplace_layer(range, layer, layer);
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
    RangeHandle _host_memory_handle;
    // Buffers and textures occupy distinct native handle namespaces. Equal
    // canonical bit patterns only alias within the same resource kind.
    vstd::ArenaHashMap<ArenaRef, uint64_t, RangeHandle *> _buffer_map;
    vstd::ArenaHashMap<ArenaRef, uint64_t, RangeHandle *> _texture_map;
    vstd::ArenaHashMap<ArenaRef, uint64_t, NoRangeHandle *> _no_range_resmap;
    vstd::ArenaHashMap<ArenaRef, uint64_t, BindlessHandle *> _bindless_map;
    struct CachedCapturedEntry {
        ResourceHandle *handle;
        Range range;
        uint64_t raw_handle;
        ResourceType type;
        bool is_write;
    };
    struct CachedCapturedShader {
        CachedCapturedEntry *entries{nullptr};
        uint32_t entry_count{0u};
        uint32_t arg_count{0u};
        uint32_t resource_count{0u};
        bool has_accel{false};
        bool writes_accel{false};
    };
    vstd::ArenaHashMap<ArenaRef, uint64_t, CachedCapturedShader> _captured_shader_cache;
    vstd::vector<CachedCapturedEntry> _tmp_captured_entries;
    static constexpr uint64_t max_allowed_dispatch_size = 65535ull * 32ull;
    vstd::vector<uint64> _max_dispatch_blocks;
    int64_t _max_mesh_level = -1;
    // Highest layer any command of the current batch was placed in.
    int64_t _max_command_layer = -1;
    // Layer of the most recent cluster (mega-geometry) mesh build. Such a
    // build reads its vertex and index data through the raw device addresses
    // stored inside its ClusterBuildInfo records. Those records are device
    // data, so the pass can neither resolve the buffers they name nor record
    // their uploads as ordinary reads. The build is therefore a full ordering
    // point for buffer writes: it lands after every command already recorded
    // in the batch, and every buffer write recorded after it lands in a later
    // layer.
    int64_t _cluster_build_read_level = -1;
    int64_t _max_accel_read_level = -1;
    int64_t _max_accel_write_level = -1;
    // Per-layer intrusive lists of reordered commands. Backends walk each
    // layer from head to tail, so commands must be appended at the tail to
    // preserve the source commit order within a layer. (Prepending used to
    // reverse same-layer order, letting a later upload overtake an earlier
    // in-list consumer of the same resource; see issue #254.)
    struct CommandLink {
        Command const *cmd;
        CommandLink const *p_next;
    };
    vstd::vector<CommandLink const *> _cmd_lists;
    // Tail pointers for O(1) tail append in add_command(). Indexed exactly
    // like _cmd_lists: resized together, cleared together.
    vstd::vector<CommandLink *> _cmd_list_tails;
    vstd::vector<std::pair<Range, ResourceHandle *>> _dispatch_read_handle;
    vstd::vector<std::pair<Range, ResourceHandle *>> _dispatch_write_handle;
    struct ActiveBindlessState {
        uint64_t bindless_handle;
        int64_t read_layer = -1;
        int64_t write_layer = -1;
        uint32_t slot{0u};
        bool has_buffers{true};
        bool has_textures{true};
    };
    struct DispatchBindlessAccess {
        uint64_t bindless_handle;
        bool is_write;
        bool active{false};
        bool has_buffers{false};
        bool has_textures{false};
        uint32_t assigned_slot{~0u};
    };
    vstd::vector<ActiveBindlessState> _active_bindless_states;
    vstd::vector<DispatchBindlessAccess> _dispatch_bindless_accesses;
    vstd::vector<std::pair<RangeHandle *, bool>> _dispatch_bindless_matched_handles;
    vstd::vector<RangeHandle *> _written_buffers;
    vstd::vector<RangeHandle *> _written_textures;
    bool _any_active_bindless_write{false};
    uint32_t _next_bindless_slot{0u};
    size_t _resolved_bindless_buffer_count{0u};
    size_t _resolved_bindless_texture_count{0u};
    // Reordering switch (see set_enabled()). When it is off, add_command()
    // ignores the hazard-derived layer and appends the command to a fresh
    // layer of its own, so the emitted layer lists degenerate into the
    // original submission order with a barrier boundary between every pair of
    // commands. This is the pre-reorder behaviour the backends fall back to.
    bool _enabled{true};
    int64_t _seq_layer{0};
    int64_t _dispatch_layer;
    bool _use_accel_in_pass;
    bool _write_accel_in_pass;
    void mark_write_layer(RangeHandle *handle, Range const &range, int64_t layer) {
        handle->emplace_write_layer(range, layer);
        if (!handle->is_tracked_written()) {
            handle->mark_tracked_written();
            if (handle->type == ResourceType::Buffer) {
                _written_buffers.emplace_back(handle);
            } else if (handle->type == ResourceType::Texture) {
                _written_textures.emplace_back(handle);
            }
        }
    }
    [[nodiscard]] ActiveBindlessState *find_active_bindless(uint64_t bindless_handle) noexcept {
        for (auto &state : _active_bindless_states) {
            if (state.bindless_handle == bindless_handle) {
                return &state;
            }
        }
        return nullptr;
    }
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
                new (value) RangeHandle{};
                value->handle = canonical_handle;
                value->type = target_type;
                if constexpr (ReorderFastBindlessFuncTable<FuncTable>) {
                    if (_any_active_bindless_write) {
                        auto is_buffer = target_type == ResourceType::Buffer;
                        int64_t max_read_layer = -1;
                        int64_t max_write_layer = -1;
                        for (auto const &state : _active_bindless_states) {
                            if (state.write_layer < 0 ||
                                (is_buffer ? !state.has_buffers : !state.has_textures)) {
                                continue;
                            }
                            auto matched = false;
                            _func_table.with_bindless_pending_lookup(
                                state.bindless_handle,
                                [&](auto &&contains) noexcept {
                                    matched = contains(canonical_handle, is_buffer);
                                });
                            value->cache_bindless_membership(state.slot, matched);
                            if (matched) {
                                max_read_layer = std::max<int64_t>(max_read_layer, state.read_layer);
                                max_write_layer = std::max<int64_t>(max_write_layer, state.write_layer);
                            }
                        }
                        if (max_write_layer >= 0) {
                            mark_write_layer(value, whole_range(), max_write_layer);
                        }
                        if (max_read_layer >= 0) {
                            value->emplace_read_layer(whole_range(), max_read_layer);
                        }
                    }
                }
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
    int64_t get_last_layer_write(
        RangeHandle *handle,
        Range range,
        bool check_active_bindless = true) {
        auto layer = std::max(
            handle->get_max_read_layer(range),
            handle->get_max_write_layer(range));
        if constexpr (ReorderFastBindlessFuncTable<FuncTable>) {
            if (check_active_bindless &&
                !_active_bindless_states.empty() &&
                (handle->type == ResourceType::Buffer ||
                 handle->type == ResourceType::Texture)) {
                auto const is_buffer = handle->type == ResourceType::Buffer;
                for (auto const &state : _active_bindless_states) {
                    if (is_buffer ? !state.has_buffers : !state.has_textures) {
                        continue;
                    }
                    auto const state_layer = std::max<int64_t>(state.read_layer, state.write_layer);
                    if (state_layer <= layer) {
                        continue;
                    }
                    bool matched = false;
                    if (handle->has_bindless_membership_cached(state.slot)) {
                        matched = handle->cached_bindless_member(state.slot);
                    } else {
                        _func_table.with_bindless_pending_lookup(
                            state.bindless_handle,
                            [&](auto &&contains) noexcept {
                                matched = contains(handle->handle, is_buffer);
                            });
                        handle->cache_bindless_membership(state.slot, matched);
                    }
                    if (matched) {
                        layer = state_layer;
                    }
                }
            }
        }
        // A cluster build reads buffers this range does not name, so its barrier
        // layer also orders every later buffer write (see
        // _cluster_build_read_level).
        layer = std::max<int64_t>(layer, _cluster_build_read_level);
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
    // Append at the layer tail so hazard-free commands sharing one layer
    // still execute in source order. This only pins intra-layer ordering;
    // layer assignment (and thus the number of layers) is unaffected.
    void add_command(Command const *cmd, int64_t layer) {
        if (!_enabled) {
            layer = _seq_layer++;
        }
        _max_command_layer = std::max<int64_t>(_max_command_layer, layer);
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
                mark_write_layer(handle, range, layer);
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
                mark_write_layer(handle, range, layer);
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
                mark_write_layer(handle, write_range, layer);
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
        Range ib_range,
        bool has_micromap = false,
        uint64_t micromap_data = 0u,
        Range micromap_data_range = {},
        bool has_micromap_index = false,
        uint64_t micromap_index = 0u,
        Range micromap_index_range = {}) {

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
        // Micromap payload: the packed bits are written by the frontend bake
        // dispatch or by a buffer upload, so the build has to be ordered after those
        // writes (otherwise the BLAS reads uninitialised opacity bits).
        if (has_micromap) {
            auto data_handle = static_cast<RangeHandle *>(
                get_handle(micromap_data, ResourceType::Buffer));
            layer = std::max<int64_t>(layer, get_last_layer_read(data_handle, micromap_data_range));
        }
        if (has_micromap_index) {
            auto index_handle = static_cast<RangeHandle *>(
                get_handle(micromap_index, ResourceType::Buffer));
            layer = std::max<int64_t>(layer, get_last_layer_read(index_handle, micromap_index_range));
        }
        range_handle->emplace_read_layer(ib_range, layer);
        if (has_micromap) {
            static_cast<RangeHandle *>(get_handle(micromap_data, ResourceType::Buffer))
                ->emplace_read_layer(micromap_data_range, layer);
            if (has_micromap_index) {
                static_cast<RangeHandle *>(get_handle(micromap_index, ResourceType::Buffer))
                    ->emplace_read_layer(micromap_index_range, layer);
            }
        }
        static_cast<RangeHandle *>(vb_handle)->emplace_read_layer(vb_range, layer);
        static_cast<NoRangeHandle *>(mesh_handle)->view.write_layer = layer;
        _max_mesh_level = std::max<int64_t>(_max_mesh_level, layer);
        return layer;
    }
    // A cluster (mega-geometry) build's only host-visible input is its array of
    // ClusterBuildInfo records (`set_mesh` cannot be used: vertex_buffer() and
    // triangle_buffer() are 0 for a cluster build, and handle 0 would create a
    // phantom buffer dependency).
    int64_t set_cluster_mesh(
        uint64_t handle,
        uint64_t cluster_buffer,
        Range cluster_range) {
        auto cluster_handle = get_handle(
            cluster_buffer,
            ResourceType::Buffer);
        auto mesh_handle = get_handle(
            handle,
            ResourceType::Mesh);
        auto layer = get_last_layer_read(static_cast<RangeHandle *>(cluster_handle), cluster_range);
        layer = std::max<int64_t>(layer, get_last_layer_write(static_cast<NoRangeHandle *>(mesh_handle)));
        // The records reach their vertex and index buffers only through raw
        // device addresses, so those uploads cannot be registered as reads of
        // this build. Order it after every command recorded so far and make it
        // a barrier for the writes that follow - the conservative equivalent of
        // naming the buffers. This is what keeps an upload + build recorded in
        // one batch correct under reordering.
        layer = std::max<int64_t>(layer, _max_command_layer + 1);
        _cluster_build_read_level = std::max<int64_t>(_cluster_build_read_level, layer);
        static_cast<RangeHandle *>(cluster_handle)->emplace_read_layer(cluster_range, layer);
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

    void materialize_bindless_state(uint64_t bindless_handle) noexcept {
        if constexpr (ReorderFastBindlessFuncTable<FuncTable>) {
            for (size_t idx = 0u; idx < _active_bindless_states.size(); ++idx) {
                if (_active_bindless_states[idx].bindless_handle != bindless_handle) {
                    continue;
                }
                auto state = _active_bindless_states[idx];
                _active_bindless_states.erase(_active_bindless_states.begin() + idx);
                vstd::vector<std::pair<uint64_t, bool>> snapshot;
                _func_table.traverse_bindless_resources(
                    bindless_handle,
                    ReorderBindlessResourceVisitor{
                        .context = &snapshot,
                        .callback = [](void *opaque, uint64_t resource_handle,
                                       bool is_buffer) noexcept {
                            static_cast<vstd::vector<std::pair<uint64_t, bool>> *>(opaque)
                                ->emplace_back(resource_handle, is_buffer);
                        }});
                for (auto [resource_handle, is_buffer] : snapshot) {
                    auto *handle = static_cast<RangeHandle *>(get_handle(
                        resource_handle,
                        is_buffer ? ResourceType::Buffer : ResourceType::Texture));
                    if (state.write_layer >= 0) {
                        mark_write_layer(handle, whole_range(), state.write_layer);
                    }
                    if (state.read_layer >= 0) {
                        handle->emplace_read_layer(whole_range(), state.read_layer);
                    }
                }
                _any_active_bindless_write = false;
                for (auto const &remaining : _active_bindless_states) {
                    if (remaining.write_layer >= 0) {
                        _any_active_bindless_write = true;
                        break;
                    }
                }
                break;
            }
        }
    }

    [[nodiscard]] bool bindless_arrays_overlap(
        uint64_t lhs, uint64_t rhs) const noexcept {
        if constexpr (ReorderFastBindlessFuncTable<FuncTable>) {
            if (lhs == rhs) { return true; }
            auto lhs_count = _func_table.bindless_pending_count(lhs);
            auto rhs_count = _func_table.bindless_pending_count(rhs);
            if (lhs_count == 0u || rhs_count == 0u) { return false; }
            auto small_handle = lhs_count <= rhs_count ? lhs : rhs;
            auto large_handle = lhs_count <= rhs_count ? rhs : lhs;
            struct CollectCtx {
                FuncTable const *func_table;
                vstd::vector<std::pair<uint64_t, bool>> items;
            } ctx{&_func_table, {}};
            ctx.items.reserve(std::min(lhs_count, rhs_count));
            _func_table.traverse_bindless_resources(
                small_handle,
                ReorderBindlessResourceVisitor{
                    .context = &ctx,
                    .callback = [](void *opaque, uint64_t resource_handle,
                                   bool is_buffer) noexcept {
                        auto &c = *static_cast<CollectCtx *>(opaque);
                        auto canonical = is_buffer ?
                                             c.func_table->canonical_buffer_handle(resource_handle) :
                                             c.func_table->canonical_texture_handle(resource_handle);
                        c.items.emplace_back(canonical, is_buffer);
                    }});
            auto overlaps = false;
            _func_table.with_bindless_pending_lookup(
                large_handle,
                [&](auto &&contains) noexcept {
                    for (auto [canonical, is_buffer] : ctx.items) {
                        if (contains(canonical, is_buffer)) {
                            overlaps = true;
                            break;
                        }
                    }
                });
            return overlaps;
        } else {
            return false;
        }
    }

    void resolve_dispatch_bindless_accesses() noexcept {
        if constexpr (ReorderFastBindlessFuncTable<FuncTable>) {
            _resolved_bindless_buffer_count = _buffer_map.size();
            _resolved_bindless_texture_count = _texture_map.size();
            if (_dispatch_bindless_accesses.empty()) {
                return;
            }
            for (auto &access : _dispatch_bindless_accesses) {
                ActiveBindlessState *existing = find_active_bindless(access.bindless_handle);
                if (!access.is_write && existing != nullptr) {
                    access.active = true;
                    access.has_buffers = existing->has_buffers;
                    access.has_textures = existing->has_textures;
                    access.assigned_slot = existing->slot;
                    vstd::fixed_vector<RangeHandle *, 16> unchecked_buffers;
                    vstd::fixed_vector<RangeHandle *, 16> unchecked_textures;
                    if (existing->has_buffers) {
                        for (auto *handle : _written_buffers) {
                            if (handle->has_bindless_membership_cached(existing->slot)) {
                                if (handle->cached_bindless_member(existing->slot)) {
                                    _dispatch_layer = std::max<int64_t>(
                                        _dispatch_layer,
                                        get_last_layer_read(handle, whole_range()));
                                }
                            } else {
                                unchecked_buffers.emplace_back(handle);
                            }
                        }
                    }
                    if (existing->has_textures) {
                        for (auto *handle : _written_textures) {
                            if (handle->has_bindless_membership_cached(existing->slot)) {
                                if (handle->cached_bindless_member(existing->slot)) {
                                    _dispatch_layer = std::max<int64_t>(
                                        _dispatch_layer,
                                        get_last_layer_read(handle, whole_range()));
                                }
                            } else {
                                unchecked_textures.emplace_back(handle);
                            }
                        }
                    }
                    if (!unchecked_buffers.empty() || !unchecked_textures.empty()) {
                        _func_table.with_bindless_pending_lookup(
                            access.bindless_handle,
                            [&](auto &&contains) noexcept {
                                for (auto *handle : unchecked_buffers) {
                                    bool matched = contains(handle->handle, true);
                                    handle->cache_bindless_membership(existing->slot, matched);
                                    if (matched) {
                                        _dispatch_layer = std::max<int64_t>(
                                            _dispatch_layer,
                                            get_last_layer_read(handle, whole_range()));
                                    }
                                }
                                for (auto *handle : unchecked_textures) {
                                    bool matched = contains(handle->handle, false);
                                    handle->cache_bindless_membership(existing->slot, matched);
                                    if (matched) {
                                        _dispatch_layer = std::max<int64_t>(
                                            _dispatch_layer,
                                            get_last_layer_read(handle, whole_range()));
                                    }
                                }
                            });
                    }
                } else {
                    auto resolve_locked = [&](size_t pending_count,
                                              bool has_buffers,
                                              bool has_textures,
                                              auto &&contains) noexcept {
                        if (pending_count == 0u) {
                            access.active = false;
                            return;
                        }
                        access.active = true;
                        access.has_buffers = has_buffers;
                        access.has_textures = has_textures;
                        access.assigned_slot =
                            existing != nullptr ? existing->slot : _next_bindless_slot++;
                        if (access.is_write) {
                            _dispatch_layer = std::max<int64_t>(
                                _dispatch_layer, _cluster_build_read_level + 1);
                            auto scan_map = [&](auto const &map, bool is_buffer) noexcept {
                                for (auto &&[canonical, handle] : map) {
                                    if (contains(canonical, is_buffer)) {
                                        auto layer = get_last_layer_write(handle, whole_range(), false);
                                        _dispatch_layer = std::max<int64_t>(_dispatch_layer, layer);
                                        _dispatch_bindless_matched_handles.emplace_back(
                                            handle, true);
                                    }
                                }
                            };
                            if (has_buffers && !_buffer_map.empty()) {
                                scan_map(_buffer_map, true);
                            }
                            if (has_textures && !_texture_map.empty()) {
                                scan_map(_texture_map, false);
                            }
                        } else {
                            auto scan_written = [&](vstd::vector<RangeHandle *> const &list, bool is_buffer) noexcept {
                                for (auto *handle : list) {
                                    bool matched = contains(handle->handle, is_buffer);
                                    handle->cache_bindless_membership(access.assigned_slot, matched);
                                    if (matched) {
                                        _dispatch_layer = std::max<int64_t>(
                                            _dispatch_layer,
                                            get_last_layer_read(handle, whole_range()));
                                    }
                                }
                            };
                            if (has_buffers && !_written_buffers.empty()) {
                                scan_written(_written_buffers, true);
                            }
                            if (has_textures && !_written_textures.empty()) {
                                scan_written(_written_textures, false);
                            }
                        }
                    };
                    if constexpr (ReorderSingleLockBindlessFuncTable<FuncTable>) {
                        _func_table.with_bindless_pending_state(
                            access.bindless_handle, resolve_locked);
                    } else {
                        auto pending_count =
                            _func_table.bindless_pending_count(access.bindless_handle);
                        if (pending_count == 0u) {
                            access.active = false;
                        } else {
                            const bool has_buffers =
                                _func_table.bindless_has_pending_buffers(access.bindless_handle);
                            _func_table.with_bindless_pending_lookup(
                                access.bindless_handle,
                                [&](auto &&contains) noexcept {
                                    resolve_locked(
                                        pending_count, has_buffers, true, contains);
                                });
                        }
                    }
                }
                if (!access.active) {
                    continue;
                }
                for (auto const &state : _active_bindless_states) {
                    if (!(access.has_buffers && state.has_buffers) &&
                        !(access.has_textures && state.has_textures)) {
                        continue;
                    }
                    auto candidate_layer = access.is_write ?
                                               std::max<int64_t>(state.read_layer, state.write_layer) + 1 :
                                               (state.write_layer >= 0 ? state.write_layer + 1 : -1);
                    if (candidate_layer <= _dispatch_layer) {
                        continue;
                    }
                    if (bindless_arrays_overlap(state.bindless_handle, access.bindless_handle)) {
                        _dispatch_layer = candidate_layer;
                    }
                }
            }
        }
    }

    void finalize_dispatch_bindless_accesses() noexcept {
        if constexpr (ReorderFastBindlessFuncTable<FuncTable>) {
            for (auto [handle, is_write] : _dispatch_bindless_matched_handles) {
                if (is_write) {
                    mark_write_layer(handle, whole_range(), _dispatch_layer);
                } else {
                    handle->emplace_read_layer(whole_range(), _dispatch_layer);
                }
            }
            const bool maps_grew =
                _buffer_map.size() > _resolved_bindless_buffer_count ||
                _texture_map.size() > _resolved_bindless_texture_count;
            for (auto const &access : _dispatch_bindless_accesses) {
                if (!access.active) {
                    continue;
                }
                if (access.is_write && maps_grew) {
                    const bool check_buffers =
                        access.has_buffers &&
                        _buffer_map.size() > _resolved_bindless_buffer_count;
                    const bool check_textures =
                        access.has_textures &&
                        _texture_map.size() > _resolved_bindless_texture_count;
                    if (check_buffers || check_textures) {
                        _func_table.with_bindless_pending_lookup(
                            access.bindless_handle,
                            [&](auto &&contains) noexcept {
                                auto apply_map = [&](auto const &map, bool is_buffer) noexcept {
                                    for (auto &&[canonical, handle] : map) {
                                        if (contains(canonical, is_buffer)) {
                                            mark_write_layer(handle, whole_range(), _dispatch_layer);
                                        }
                                    }
                                };
                                if (check_buffers) {
                                    apply_map(_buffer_map, true);
                                }
                                if (check_textures) {
                                    apply_map(_texture_map, false);
                                }
                            });
                    }
                }
                if (access.is_write) {
                    _any_active_bindless_write = true;
                }
                ActiveBindlessState *found = find_active_bindless(access.bindless_handle);
                if (found == nullptr) {
                    _active_bindless_states.emplace_back(ActiveBindlessState{
                        .bindless_handle = access.bindless_handle,
                        .read_layer = _dispatch_layer,
                        .write_layer = access.is_write ? _dispatch_layer : -1,
                        .slot = access.assigned_slot,
                        .has_buffers = access.has_buffers,
                        .has_textures = access.has_textures});
                } else {
                    found->read_layer = std::max<int64_t>(found->read_layer, _dispatch_layer);
                    if (access.is_write) {
                        found->write_layer = std::max<int64_t>(found->write_layer, _dispatch_layer);
                    }
                    found->has_buffers |= access.has_buffers;
                    found->has_textures |= access.has_textures;
                }
            }
        }
    }

    void add_bindless_dispatch_handles(
        uint64_t bindless_handle,
        bool may_write,
        bool isolate_resource_states = false) {
        if constexpr (ReorderFastBindlessFuncTable<FuncTable>) {
            auto is_write = isolate_resource_states || may_write;
            bool found = false;
            for (auto &existing : _dispatch_bindless_accesses) {
                if (existing.bindless_handle == bindless_handle) {
                    existing.is_write |= is_write;
                    found = true;
                    break;
                }
            }
            if (!found) {
                _dispatch_bindless_accesses.emplace_back(
                    DispatchBindlessAccess{bindless_handle, is_write});
            }
        } else {
            struct TraversalContext {
                CommandReorderVisitor *reorder;
                bool may_write;
                bool isolate_resource_states;
            } context{this, may_write, isolate_resource_states};
            _func_table.traverse_bindless_resources(
                bindless_handle,
                ReorderBindlessResourceVisitor{
                    .context = &context,
                    .callback = [](void *opaque, uint64_t resource_handle,
                                   bool is_buffer) noexcept {
                        auto &state = *static_cast<TraversalContext *>(opaque);
                        // A WRITE bindless argument may write ANY snapshot
                        // resource: the DSL marks the whole array variable as
                        // WRITE when the shader writes through any slot, and
                        // tex2d()/tex3d() views support write(). Restricting
                        // the write mark to buffers (is_buffer && may_write)
                        // left bindless texture writes unordered, so a later
                        // direct read could share the dispatch's layer (RAW)
                        // and repeated write dispatches could share one layer
                        // (WAW).
                        state.reorder->add_dispatch_handle(
                            resource_handle,
                            is_buffer ? ResourceType::Buffer : ResourceType::Texture,
                            whole_range(),
                            state.isolate_resource_states || state.may_write);
                    }});
        }
        // Shader access reads the bindless index/descriptor object itself;
        // resource writes are tracked on the snapshotted resources above.
        // A native state contract also serializes the descriptor
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
        _dispatch_bindless_accesses.clear();
        _dispatch_bindless_matched_handles.clear();
        _use_accel_in_pass = false;
        _write_accel_in_pass = false;
        _dispatch_layer = 0;
        // Every argument is tracked with its declared usage - the same
        // contract a native dispatch gets from the shader's compiled
        // usages. A declared READ is a read-only contract: concurrent
        // reads of one range do not race, so dispatches that share a
        // read-only argument (e.g. one input buffer) merge into a single
        // barrier-free layer. A declared WRITE (or READ_WRITE) is an
        // exclusive access over its range, so RAW/WAW/WAR chains still
        // serialize exactly as they do for native dispatches. The
        // declaration itself is the contract.
        auto f = [&]<typename T>(T const &t, Usage usage) {
            if constexpr (std::is_same_v<T, Argument::Buffer>) {
                add_dispatch_handle(
                    t.handle,
                    ResourceType::Buffer,
                    buffer_range(t.offset, t.size),
                    ((uint)usage & (uint)Usage::WRITE) != 0);
            } else if constexpr (std::is_same_v<T, Argument::Texture>) {
                add_texture_dispatch_handles(t.handle, t.level, usage);
            } else if constexpr (std::is_same_v<T, Argument::BindlessArray>) {
                add_bindless_dispatch_handles(
                    t.handle,
                    (static_cast<uint>(usage) &
                     static_cast<uint>(Usage::WRITE)) != 0u,
                    false);
            } else {
                _use_accel_in_pass = true;
                auto is_write =
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
        resolve_dispatch_bindless_accesses();
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
        finalize_dispatch_bindless_accesses();
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
               luisa::span<const Usage> declared_usages,
               bool native_shader,
               Callback callback) noexcept {
        _dispatch_read_handle.clear();
        _dispatch_write_handle.clear();
        _dispatch_bindless_accesses.clear();
        _dispatch_bindless_matched_handles.clear();
        _use_accel_in_pass = false;
        _write_accel_in_pass = false;
        _dispatch_layer = 0;
        size_t arg_idx = 0;
        // A native raster draw has no DSL shader behind its handle, so its
        // usages cannot come from `get_usage` (which would reinterpret the
        // handle as a `Shader`): the command carries the launcher's declaration
        // instead - one entry per *resource* argument, i.e. uniforms excluded,
        // which is why the two index spaces differ. `native_shader` is an
        // explicit discriminator rather than "declared_usages is non-empty", so
        // that a malformed native command cannot fall back into the DSL path.
        size_t resource_idx = 0;
        auto usage_of = [&]() noexcept -> Usage {
            if (!native_shader) {
                return _func_table.get_usage(shader_handle, arg_idx);
            }
            LUISA_ASSERT(resource_idx < declared_usages.size(),
                         "Native raster draw resource argument {} has no "
                         "declared usage ({} declared); a native raster command "
                         "must carry one usage per resource argument.",
                         resource_idx, declared_usages.size());
            return declared_usages[resource_idx];
        };
        using Argument = ShaderDispatchCommandBase::Argument;
        using Tag = Argument::Tag;
        auto ite_arg = [&](auto &&i) {
            switch (i.tag) {
                case Tag::BUFFER: {
                    auto &&bf = i.buffer;
                    bool is_write = ((uint)usage_of() & (uint)Usage::WRITE) != 0;
                    add_dispatch_handle(
                        bf.handle,
                        ResourceType::Buffer,
                        buffer_range(bf.offset, bf.size),
                        is_write);
                    ++arg_idx;
                    ++resource_idx;
                } break;
                case Tag::TEXTURE: {
                    auto &&tex = i.texture;
                    add_texture_dispatch_handles(
                        tex.handle,
                        tex.level,
                        usage_of());
                    ++arg_idx;
                    ++resource_idx;
                } break;
                case Tag::UNIFORM: {
                    ++arg_idx;
                } break;
                case Tag::BINDLESS_ARRAY: {
                    auto &&arr = i.bindless_array;
                    auto usage = usage_of();
                    add_bindless_dispatch_handles(
                        arr.handle,
                        (static_cast<uint>(usage) &
                         static_cast<uint>(Usage::WRITE)) != 0u);
                    ++arg_idx;
                    ++resource_idx;
                } break;
                case Tag::ACCEL: {
                    auto &&acc = i.accel;
                    _use_accel_in_pass = true;
                    auto is_write = (static_cast<uint>(usage_of()) &
                                     static_cast<uint>(Usage::WRITE)) != 0u;
                    _write_accel_in_pass |= is_write;
                    add_dispatch_handle(
                        acc.handle,
                        ResourceType::Accel,
                        whole_range(),
                        is_write);
                    ++arg_idx;
                    ++resource_idx;
                } break;
            }
        };
        if constexpr (ReorderFastBindlessFuncTable<FuncTable>) {
            if (!native_shader && !captured_bindings.empty()) {
                auto cached_iter = _captured_shader_cache.find(shader_handle);
                if (!cached_iter) {
                    _tmp_captured_entries.clear();
                    CachedCapturedShader entry_meta{};
                    auto record_handle = [&](uint64_t raw_handle, ResourceType type, Range range, bool is_write) {
                        auto *h = get_handle(raw_handle, type);
                        if (type == ResourceType::Bindless) {
                            auto is_w = is_write;
                            bool found = false;
                            for (auto &existing : _dispatch_bindless_accesses) {
                                if (existing.bindless_handle == raw_handle) {
                                    existing.is_write |= is_w;
                                    found = true;
                                    break;
                                }
                            }
                            if (!found) {
                                _dispatch_bindless_accesses.emplace_back(
                                    DispatchBindlessAccess{raw_handle, is_w});
                            }
                            _dispatch_layer = std::max<int64_t>(
                                _dispatch_layer,
                                get_last_layer_read(static_cast<BindlessHandle *>(h)));
                            _dispatch_read_handle.emplace_back(whole_range(), h);
                        } else if (is_write) {
                            switch (type) {
                                case ResourceType::Accel:
                                case ResourceType::Mesh:
                                    _dispatch_layer = std::max<int64_t>(_dispatch_layer, get_last_layer_write(static_cast<NoRangeHandle *>(h)));
                                    break;
                                case ResourceType::Buffer:
                                case ResourceType::Texture:
                                    _dispatch_layer = std::max<int64_t>(_dispatch_layer, get_last_layer_write(static_cast<RangeHandle *>(h), range));
                                    break;
                                default: break;
                            }
                            _dispatch_write_handle.emplace_back(range, h);
                        } else {
                            switch (type) {
                                case ResourceType::Accel:
                                case ResourceType::Mesh:
                                    _dispatch_layer = std::max<int64_t>(_dispatch_layer, get_last_layer_read(static_cast<NoRangeHandle *>(h)));
                                    break;
                                case ResourceType::Buffer:
                                case ResourceType::Texture:
                                    _dispatch_layer = std::max<int64_t>(_dispatch_layer, get_last_layer_read(static_cast<RangeHandle *>(h), range));
                                    break;
                                default: break;
                            }
                            _dispatch_read_handle.emplace_back(range, h);
                        }
                        _tmp_captured_entries.emplace_back(CachedCapturedEntry{
                            .handle = h,
                            .range = range,
                            .raw_handle = raw_handle,
                            .type = type,
                            .is_write = is_write});
                    };
                    for (auto &&i : captured_bindings) {
                        switch (i.tag) {
                            case Tag::BUFFER: {
                                auto &&bf = i.buffer;
                                bool is_write = ((uint)usage_of() & (uint)Usage::WRITE) != 0;
                                record_handle(bf.handle, ResourceType::Buffer, buffer_range(bf.offset, bf.size), is_write);
                                ++arg_idx;
                                ++resource_idx;
                            } break;
                            case Tag::TEXTURE: {
                                auto &&tex = i.texture;
                                auto usage = usage_of();
                                auto bits = static_cast<uint>(usage);
                                auto reads = (bits & static_cast<uint>(Usage::READ)) != 0u || usage == Usage::NONE;
                                auto writes = (bits & static_cast<uint>(Usage::WRITE)) != 0u;
                                LUISA_ASSERT(reads || writes, "Texture argument has an invalid empty usage mask.");
                                if (reads) {
                                    record_handle(tex.handle, ResourceType::Texture, mip_suffix_range(tex.level), false);
                                }
                                if (writes) {
                                    record_handle(tex.handle, ResourceType::Texture, base_mip_range(tex.level), true);
                                }
                                ++arg_idx;
                                ++resource_idx;
                            } break;
                            case Tag::UNIFORM: {
                                ++arg_idx;
                            } break;
                            case Tag::BINDLESS_ARRAY: {
                                auto &&arr = i.bindless_array;
                                auto usage = usage_of();
                                bool is_write = (static_cast<uint>(usage) & static_cast<uint>(Usage::WRITE)) != 0u;
                                record_handle(arr.handle, ResourceType::Bindless, whole_range(), is_write);
                                ++arg_idx;
                                ++resource_idx;
                            } break;
                            case Tag::ACCEL: {
                                auto &&acc = i.accel;
                                _use_accel_in_pass = true;
                                entry_meta.has_accel = true;
                                auto is_write = (static_cast<uint>(usage_of()) & static_cast<uint>(Usage::WRITE)) != 0u;
                                _write_accel_in_pass |= is_write;
                                entry_meta.writes_accel |= is_write;
                                record_handle(acc.handle, ResourceType::Accel, whole_range(), is_write);
                                ++arg_idx;
                                ++resource_idx;
                            } break;
                        }
                    }
                    entry_meta.entry_count = static_cast<uint32_t>(_tmp_captured_entries.size());
                    entry_meta.arg_count = static_cast<uint32_t>(arg_idx);
                    entry_meta.resource_count = static_cast<uint32_t>(resource_idx);
                    if (entry_meta.entry_count > 0u) {
                        auto mem = _arena.allocate(
                            sizeof(CachedCapturedEntry) * entry_meta.entry_count,
                            alignof(CachedCapturedEntry));
                        entry_meta.entries = reinterpret_cast<CachedCapturedEntry *>(mem.handle + mem.offset);
                        std::memcpy(entry_meta.entries, _tmp_captured_entries.data(),
                                    sizeof(CachedCapturedEntry) * entry_meta.entry_count);
                    }
                    _captured_shader_cache.try_emplace(shader_handle, entry_meta);
                } else {
                    auto const &cached = cached_iter.value();
                    arg_idx = cached.arg_count;
                    resource_idx = cached.resource_count;
                    if (cached.has_accel) {
                        _use_accel_in_pass = true;
                        _write_accel_in_pass |= cached.writes_accel;
                    }
                    for (uint32_t ei = 0u; ei < cached.entry_count; ++ei) {
                        auto const &e = cached.entries[ei];
                        switch (e.type) {
                            case ResourceType::Buffer:
                            case ResourceType::Texture: {
                                auto *h = static_cast<RangeHandle *>(e.handle);
                                if (e.is_write) {
                                    _dispatch_layer = std::max<int64_t>(_dispatch_layer, get_last_layer_write(h, e.range));
                                    _dispatch_write_handle.emplace_back(e.range, h);
                                } else {
                                    _dispatch_layer = std::max<int64_t>(_dispatch_layer, get_last_layer_read(h, e.range));
                                    _dispatch_read_handle.emplace_back(e.range, h);
                                }
                            } break;
                            case ResourceType::Accel:
                            case ResourceType::Mesh: {
                                auto *h = static_cast<NoRangeHandle *>(e.handle);
                                if (e.is_write) {
                                    _dispatch_layer = std::max<int64_t>(_dispatch_layer, get_last_layer_write(h));
                                    _dispatch_write_handle.emplace_back(e.range, h);
                                } else {
                                    _dispatch_layer = std::max<int64_t>(_dispatch_layer, get_last_layer_read(h));
                                    _dispatch_read_handle.emplace_back(e.range, h);
                                }
                            } break;
                            case ResourceType::Bindless: {
                                bool found = false;
                                for (auto &existing : _dispatch_bindless_accesses) {
                                    if (existing.bindless_handle == e.raw_handle) {
                                        existing.is_write |= e.is_write;
                                        found = true;
                                        break;
                                    }
                                }
                                if (!found) {
                                    _dispatch_bindless_accesses.emplace_back(
                                        DispatchBindlessAccess{e.raw_handle, e.is_write});
                                }
                                auto *h = static_cast<BindlessHandle *>(e.handle);
                                _dispatch_layer = std::max<int64_t>(_dispatch_layer, get_last_layer_read(h));
                                _dispatch_read_handle.emplace_back(whole_range(), h);
                            } break;
                        }
                    }
                }
            } else {
                for (auto &&i : captured_bindings) {
                    ite_arg(i);
                }
            }
        } else {
            for (auto &&i : captured_bindings) {
                ite_arg(i);
            }
        }
        for (auto &&i : command->arguments()) {
            ite_arg(i);
        }
        resolve_dispatch_bindless_accesses();
        callback();
        for (auto &&i : _dispatch_read_handle) {
            set_read_layer(i.second, i.first, _dispatch_layer);
        }
        for (auto &&i : _dispatch_write_handle) {
            set_write_layer(i.second, i.first, _dispatch_layer);
        }
        finalize_dispatch_bindless_accesses();
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
          _buffer_map(256, ArenaRef{_arena}),
          _texture_map(64, ArenaRef{_arena}),
          _no_range_resmap(64, ArenaRef{_arena}),
          _bindless_map(64, ArenaRef{_arena}),
          _captured_shader_cache(32, ArenaRef{_arena}),
          _func_table(std::forward<FuncTable>(func_table)) {
    }
    void clear() noexcept {
        auto destroy_map = []<typename T>(T &t) noexcept {
            // Handle objects are placement-constructed in the arena and
            // RangeHandle additionally owns a heap interval vector, so run
            // their destructors before the arena is reset.
            for (auto &&i : t) {
                std::destroy_at(i.second);
            }
            t.~T();
        };
        _seq_layer = 0;
        _max_accel_read_level = -1;
        _max_accel_write_level = -1;
        _max_mesh_level = -1;
        _max_command_layer = -1;
        _cluster_build_read_level = -1;
        _dispatch_layer = 0;
        _use_accel_in_pass = false;
        _write_accel_in_pass = false;
        _cmd_lists.clear();
        _cmd_list_tails.clear();
        _max_dispatch_blocks.clear();
        _dispatch_read_handle.clear();
        _dispatch_write_handle.clear();
        _active_bindless_states.clear();
        _dispatch_bindless_accesses.clear();
        _dispatch_bindless_matched_handles.clear();
        _written_buffers.clear();
        _written_textures.clear();
        _tmp_captured_entries.clear();
        _any_active_bindless_write = false;
        _next_bindless_slot = 0u;
        _resolved_bindless_buffer_count = 0u;
        _resolved_bindless_texture_count = 0u;
        _host_memory_handle.clear();
        destroy_map(_buffer_map);
        destroy_map(_texture_map);
        destroy_map(_no_range_resmap);
        destroy_map(_bindless_map);
        std::destroy_at(&_captured_shader_cache);
        _arena.clear();
        new (&_buffer_map) decltype(_buffer_map)(256, ArenaRef{_arena});
        new (&_texture_map) decltype(_texture_map)(64, ArenaRef{_arena});
        new (&_no_range_resmap) decltype(_no_range_resmap)(64, ArenaRef{_arena});
        new (&_bindless_map) decltype(_bindless_map)(64, ArenaRef{_arena});
        new (&_captured_shader_cache) decltype(_captured_shader_cache)(32, ArenaRef{_arena});
    }
    // clear() also runs the destructors of the placement-constructed resource
    // handles (RangeHandle owns a heap interval vector), so destruction must
    // go through it instead of only releasing the arena.
    ~CommandReorderVisitor() noexcept {
        clear();
    }
    [[nodiscard]] auto command_lists() const noexcept {
        return luisa::span{_cmd_lists};
    }
    // Enable or disable reordering. It must be set before any command is
    // visited for the current batch (clear() resets the sequence counter but
    // keeps the flag, so backends assign it once per batch). When disabled,
    // command_lists() yields exactly one command per layer in submission
    // order, i.e. the batch is executed without any reordering.
    void set_enabled(bool enabled) noexcept { _enabled = enabled; }
    [[nodiscard]] bool enabled() const noexcept { return _enabled; }

    // Buffer : resource
    [[nodiscard]] static Range host_byte_range(const void *ptr, size_t size) noexcept {
        if (ptr == nullptr || size == 0u) {
            return Range::empty();
        }
        const auto begin = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(ptr));
        const auto clamped = std::min<uint64_t>(size, std::numeric_limits<uint64_t>::max() - begin);
        return Range::from_offset_size(begin, clamped);
    }
    void visit(const BufferUploadCommand *command) noexcept override {
        auto dev_range = copy_buffer_range(command->offset(), command->size());
        auto dev_handle = static_cast<RangeHandle *>(get_handle(command->handle(), ResourceType::Buffer));
        auto layer = get_last_layer_write(dev_handle, dev_range);
        auto h_range = host_byte_range(command->data(), command->size());
        if (h_range.min < h_range.max) {
            layer = std::max<int64_t>(layer, _host_memory_handle.get_max_write_layer(h_range) + 1);
            _host_memory_handle.emplace_read_layer(h_range, layer);
        }
        mark_write_layer(dev_handle, dev_range, layer);
        add_command(command, layer);
    }
    void visit(const BufferDownloadCommand *command) noexcept override {
        auto dev_range = copy_buffer_range(command->offset(), command->size());
        auto dev_handle = static_cast<RangeHandle *>(get_handle(command->handle(), ResourceType::Buffer));
        auto layer = get_last_layer_read(dev_handle, dev_range);
        auto h_range = host_byte_range(command->data(), command->size());
        if (h_range.min < h_range.max) {
            auto host_dep = std::max(
                _host_memory_handle.get_max_read_layer(h_range),
                _host_memory_handle.get_max_write_layer(h_range));
            layer = std::max<int64_t>(layer, host_dep + 1);
            _host_memory_handle.emplace_write_layer(h_range, layer);
        }
        dev_handle->emplace_read_layer(dev_range, layer);
        add_command(command, layer);
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
        visit(command, command, command->handle(), _func_table.shader_bindings(command->handle()),
              luisa::span<const Usage>{}, /*native_shader*/ false, [&] {
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
        auto dev_range = copy_tex_range(command->level());
        auto dev_handle = static_cast<RangeHandle *>(get_handle(command->handle(), ResourceType::Texture));
        auto layer = get_last_layer_write(dev_handle, dev_range);
        auto h_range = host_byte_range(
            command->data(),
            pixel_storage_size(command->storage(), command->size()));
        if (h_range.min < h_range.max) {
            layer = std::max<int64_t>(layer, _host_memory_handle.get_max_write_layer(h_range) + 1);
            _host_memory_handle.emplace_read_layer(h_range, layer);
        }
        mark_write_layer(dev_handle, dev_range, layer);
        add_command(command, layer);
    }
    void visit(const TextureDownloadCommand *command) noexcept override {
        auto dev_range = copy_tex_range(command->level());
        auto dev_handle = static_cast<RangeHandle *>(get_handle(command->handle(), ResourceType::Texture));
        auto layer = get_last_layer_read(dev_handle, dev_range);
        auto h_range = host_byte_range(
            command->data(),
            pixel_storage_size(command->storage(), command->size()));
        if (h_range.min < h_range.max) {
            auto host_dep = std::max(
                _host_memory_handle.get_max_read_layer(h_range),
                _host_memory_handle.get_max_write_layer(h_range));
            layer = std::max<int64_t>(layer, host_dep + 1);
            _host_memory_handle.emplace_write_layer(h_range, layer);
        }
        dev_handle->emplace_read_layer(dev_range, layer);
        add_command(command, layer);
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
        materialize_bindless_state(command->handle());
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
        if (command->is_cluster_build()) {
            // A cluster (mega-geometry) build has no vertex/triangle buffer; its
            // only host-visible input is the array of ClusterBuildInfo records,
            // which the build reads as srcInfosArray. Without this dependency the
            // pass could record the upload/copy that fills those records *after*
            // the build, and the build would read uninitialized records (garbage
            // device addresses for the vertex/index data, or stale geometry).
            // TODO(cluster): also declare the vertex/index buffers, which are
            // only reachable through the records' device addresses.
            add_command(
                command,
                set_cluster_mesh(
                    command->handle(),
                    command->cluster_buffer(),
                    buffer_range(command->cluster_buffer_offset(),
                                 command->cluster_buffer_size())));
            return;
        }
      add_command(
          command,
          set_mesh(
              command->handle(),
              command->vertex_buffer(),
              buffer_range(command->vertex_buffer_offset(),
                           command->vertex_buffer_size()),
              command->triangle_buffer(),
              buffer_range(command->triangle_buffer_offset(),
                           command->triangle_buffer_size()),
              command->has_micromap(),
              command->micromap().data_buffer,
              buffer_range(command->micromap().data_offset,
                           command->micromap().data_size),
              command->micromap().index_stride != 0u,
              command->micromap().index_buffer,
              buffer_range(command->micromap().index_offset,
                           command->micromap().index_size)));
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
        // A native raster draw carries its own per-resource usages (there is no
        // DSL shader behind its handle, so `get_usage`/`raster_shader_bindings`
        // would reinterpret a `NativeRasterShader` as a `Shader`), and it has no
        // captured bindings at all.
        auto native = command->is_native_shader();
        visit(command, command, command->handle(),
              native ? luisa::span<const Argument>{}
                     : _func_table.raster_shader_bindings(command->handle()),
              command->argument_usages(), native, [&] {
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
            case to_underlying(CustomCommandUUID::VK_CUDA_LAUNCH_KERNEL):
                visit(static_cast<CustomDispatchCommand const *>(custom_cmd));
                break;
            case to_underlying(CustomCommandUUID::NATIVE_SHADER_DISPATCH):
                // Native shader dispatch: a plain CustomDispatchCommand that
                // declares its resource usages through traverse_arguments.
                visit(static_cast<CustomDispatchCommand const *>(custom_cmd));
                break;
            case to_underlying(CustomCommandUUID::BATCH_BUFFER_TO_TEXTURE_COPY):
                visit(static_cast<CustomDispatchCommand const *>(custom_cmd));
                break;
            case to_underlying(CustomCommandUUID::STREAM_SIGNAL_EVENT):
                break;
            default:
                LUISA_ERROR("Custom command not supported by reorder.");
        }
    }

    void visit(const MotionInstanceBuildCommand *command) noexcept override {
        // Register as a write to the motion instance handle to ensure
        // it's ordered before AccelBuildCommand that references it.
        add_command(command, set_write(command->handle(), whole_range(), ResourceType::Accel));
    }
};

}// namespace luisa::compute
