// Structural interning of Vulkan descriptor-set and pipeline layouts (plan P1.1).
//
// Why this exists
// ---------------
// Every `Shader` used to create its own VkDescriptorSetLayout objects and its
// own VkPipelineLayout, so two compiles of the same kernel produced two
// structurally identical sets of layout objects (plan §4, F1: 4 descriptor-set
// layouts and 2 pipeline layouts for two identical compiles). Interning turns
// that into one object per *structure* per device. It also gives a later
// "bind once" optimisation (P1.2) the object identity it needs: a bind may only
// be skipped when the bound handle is the same handle, never when two handles
// merely describe the same structure.
//
// What a key covers
// -----------------
// A key is the structure of a create info, never an address:
//   * `DescriptorSetLayoutKey` - the VkDescriptorSetLayoutCreateInfo flags, the
//     ordered (binding number, descriptor type, descriptor count, stage flags,
//     binding flags) tuple of every binding, and the immutable samplers a
//     binding bakes into the layout.
//   * `PipelineLayoutKey` - the VkPipelineLayoutCreateInfo flags, the *ordered*
//     descriptor-set structures (by their interned identity, which is exact:
//     an id is only handed out for a structure no live entry holds) and every
//     push-constant range (stage flags, offset, size).
// Equality and hashing compare field by field, so padding or uninitialised
// bytes can neither split two identical structures nor alias two different ones
// (the container's byte-wise hash is never used on a key).
//
// Ownership contract
// ------------------
//   * The `Device` owns the table (`Device::layout_interner()`) and is the only
//     thing that calls `vkCreate*`/`vkDestroy*` for these objects.
//   * An acquired layout is *borrowed*: the holder must call `release_*`
//     exactly once. The last release hands the native handle to the device,
//     which destroys it exactly once. A release for an unknown or already
//     released id is a no-op, never a double destroy (device teardown drains the
//     table).
//   * `InternedLayout::id` is a table-local, monotonic identity: it is the
//     canonical structural identity inside one device and is what a pipeline
//     layout key compares. `InternedLayout::structural_hash` is the portable
//     fingerprint (built from the structure only, no ids, no addresses) and is
//     what a later PSO/blob cache may persist.
//
// Concurrency
// -----------
// The table is safe for concurrent compilation. The mutex is only held while
// the table itself is touched - never over `vkCreate*` and never over shader
// compilation - so a slow driver create or a long DXC/glslang compile in
// another thread cannot serialize anyone. Two compiles that race on the same
// structure both create a handle; the loser observes the winner's entry and
// destroys its own handle, so a race leaves exactly one live object and cannot
// corrupt the index.
//
// Scope note: this file is header-only except for `LayoutInterner`, which lives
// in layout_intern.cpp next to the Vulkan calls. The key types and the table
// template are deliberately free of Vulkan entry points so the key/equality/
// reference-counting contract can be tested exhaustively on the host
// (src/tests/unit/ext/test_vk_layout_intern.cpp).
#pragma once

#include <volk.h>

#include <cstddef>
#include <cstdint>
#include <limits>
#include <mutex>
#include <type_traits>
#include <unordered_map>

#include <luisa/core/stl/hash.h>
#include <luisa/core/stl/string.h>
#include <luisa/vstl/common.h>

namespace lc::vk::detail {

// Vk non-dispatchable handles are either a pointer or a uint64_t depending on
// VK_USE_64_BIT_PTR_DEFINES. The table stores plain integers (so it can be
// exercised without a device); these two helpers are the only conversions.
template<typename Handle>
[[nodiscard]] inline uint64_t handle_to_uint64(Handle handle) noexcept {
    if constexpr (std::is_pointer_v<Handle>) {
        return static_cast<uint64_t>(reinterpret_cast<uintptr_t>(handle));
    } else {
        return static_cast<uint64_t>(handle);
    }
}

template<typename Handle>
[[nodiscard]] inline Handle uint64_to_handle(uint64_t value) noexcept {
    if constexpr (std::is_pointer_v<Handle>) {
        return reinterpret_cast<Handle>(static_cast<uintptr_t>(value));
    } else {
        return static_cast<Handle>(value);
    }
}

// Mixes one explicitly typed field into a running hash. Nothing is hashed by
// raw memory, so a key's padding bytes are irrelevant.
[[nodiscard]] inline uint64_t hash_layout_field(uint64_t seed, uint64_t value) noexcept {
    return luisa::hash64(&value, sizeof(value), seed);
}

// One binding of a descriptor-set layout. Mirrors every field of the
// VkDescriptorSetLayoutBinding that will be created, plus the binding flags that
// Vulkan takes through the VkDescriptorSetLayoutBindingFlagsCreateInfo chain.
struct DescriptorSetLayoutBindingKey {
    uint32_t binding{0u};
    VkDescriptorType descriptor_type{VK_DESCRIPTOR_TYPE_SAMPLER};
    uint32_t descriptor_count{0u};
    VkShaderStageFlags stage_flags{0u};
    VkDescriptorBindingFlags binding_flags{0u};
    // Immutable samplers are baked into the layout by the specification, so both
    // their values and their order are part of the structure. Empty whenever the
    // binding has no immutable samplers (VkDescriptorSetLayoutBinding::
    // pImmutableSamplers == nullptr). The values are device-owned VkSampler
    // handles; a table is device-local, so comparing them is comparing the same
    // immutable-sampler content.
    vstd::vector<VkSampler> immutable_samplers;

    [[nodiscard]] bool operator==(const DescriptorSetLayoutBindingKey &rhs) const noexcept {
        return binding == rhs.binding &&
               descriptor_type == rhs.descriptor_type &&
               descriptor_count == rhs.descriptor_count &&
               stage_flags == rhs.stage_flags &&
               binding_flags == rhs.binding_flags &&
               immutable_samplers == rhs.immutable_samplers;
    }

    [[nodiscard]] uint64_t structural_hash(uint64_t seed) const noexcept {
        auto hash = hash_layout_field(seed, binding);
        hash = hash_layout_field(hash, static_cast<uint64_t>(descriptor_type));
        hash = hash_layout_field(hash, descriptor_count);
        hash = hash_layout_field(hash, static_cast<uint64_t>(stage_flags));
        hash = hash_layout_field(hash, static_cast<uint64_t>(binding_flags));
        hash = hash_layout_field(hash, immutable_samplers.size());
        for (auto &&sampler : immutable_samplers) {
            hash = hash_layout_field(hash, handle_to_uint64(sampler));
        }
        return hash;
    }
};

// The complete structure of a VkDescriptorSetLayoutCreateInfo.
struct DescriptorSetLayoutKey {
    VkDescriptorSetLayoutCreateFlags flags{0u};
    // Ordered by binding number, exactly as the create info lists them.
    vstd::vector<DescriptorSetLayoutBindingKey> bindings;

    [[nodiscard]] bool operator==(const DescriptorSetLayoutKey &rhs) const noexcept {
        return flags == rhs.flags && bindings == rhs.bindings;
    }

    [[nodiscard]] uint64_t structural_hash() const noexcept {
        auto hash = hash_layout_field(luisa::hash64_default_seed, static_cast<uint64_t>(flags));
        hash = hash_layout_field(hash, bindings.size());
        for (auto &&binding : bindings) {
            hash = binding.structural_hash(hash);
        }
        return hash;
    }
};

struct PushConstantRangeKey {
    VkShaderStageFlags stage_flags{0u};
    uint32_t offset{0u};
    uint32_t size{0u};

    [[nodiscard]] bool operator==(const PushConstantRangeKey &rhs) const noexcept {
        return stage_flags == rhs.stage_flags &&
               offset == rhs.offset &&
               size == rhs.size;
    }

    [[nodiscard]] uint64_t structural_hash(uint64_t seed) const noexcept {
        auto hash = hash_layout_field(seed, static_cast<uint64_t>(stage_flags));
        hash = hash_layout_field(hash, offset);
        return hash_layout_field(hash, size);
    }
};

// The complete structure of a VkPipelineLayoutCreateInfo.
struct PipelineLayoutKey {
    // Interned descriptor-set-layout identities in set order. An id is handed
    // out only for a structure no live entry holds, so comparing ids is exact
    // structural equality - a hash collision can never alias two layouts here.
    vstd::vector<uint64_t> descriptor_set_layouts;
    // The same sets' portable structural fingerprints, in the same order.
    // Deliberately *not* part of equality (the ids already decide it) but the
    // only thing `structural_hash()` is built from, so the portable identity of
    // a pipeline layout never depends on table-local ids.
    vstd::vector<uint64_t> descriptor_set_layout_hashes;
    vstd::vector<PushConstantRangeKey> push_constant_ranges;
    VkPipelineLayoutCreateFlags flags{0u};

    [[nodiscard]] bool operator==(const PipelineLayoutKey &rhs) const noexcept {
        return flags == rhs.flags &&
               descriptor_set_layouts == rhs.descriptor_set_layouts &&
               push_constant_ranges == rhs.push_constant_ranges;
    }

    [[nodiscard]] uint64_t structural_hash() const noexcept {
        auto hash = hash_layout_field(luisa::hash64_default_seed, static_cast<uint64_t>(flags));
        hash = hash_layout_field(hash, descriptor_set_layout_hashes.size());
        for (auto set_hash : descriptor_set_layout_hashes) {
            hash = hash_layout_field(hash, set_hash);
        }
        hash = hash_layout_field(hash, push_constant_ranges.size());
        for (auto &&range : push_constant_ranges) {
            hash = range.structural_hash(hash);
        }
        return hash;
    }
};

// What the table hands out for one interned layout.
struct InternedLayout {
    uint64_t id{0u};             // canonical structural identity (device-local)
    uint64_t handle{0u};         // native handle, owned by the device
    uint64_t structural_hash{0u};// portable structural fingerprint

    [[nodiscard]] explicit operator bool() const noexcept { return handle != 0u; }
};

// Result of one `LayoutInternTable::acquire`.
struct LayoutAcquireResult {
    InternedLayout layout{};
    // Non-zero when this call created a handle that the table refused because an
    // equal key won the insertion race. The caller owns that handle and must
    // destroy it; `layout` is the entry the table kept. A device-level caller
    // (LayoutInterner) consumes it so shader code never sees it.
    uint64_t orphaned_handle{0u};
};

// A structurally shared, reference-counted table of native handles.
//
// `Key` must be copyable and provide `structural_hash()` and `operator==`.
// `CreateFn` is called without the lock held and returns the native handle as a
// uint64_t (0 == failure).
template<typename Key>
class LayoutInternTable {

private:
    struct Slot {
        Key key;
        InternedLayout layout;
        uint32_t refs{0u};
    };
    static constexpr size_t npos = std::numeric_limits<size_t>::max();

    mutable std::mutex _mutex;
    uint64_t _next_id{1u};
    // structural hash -> live slot indices. Collisions are resolved by comparing
    // the complete key, never the hash alone.
    std::unordered_map<uint64_t, vstd::vector<size_t>> _buckets;
    vstd::vector<Slot> _slots;
    vstd::vector<size_t> _free_slots;
    std::unordered_map<uint64_t, size_t> _live_ids;
    size_t _live_count{0u};

    // Called with the lock held.
    [[nodiscard]] size_t find_locked(const Key &key, uint64_t structural_hash) const noexcept {
        auto iter = _buckets.find(structural_hash);
        if (iter == _buckets.end()) { return npos; }
        for (auto index : iter->second) {
            if (_slots[index].key == key) { return index; }
        }
        return npos;
    }

    // Called with the lock held. Unlinks a dead slot from the index.
    void unlink_locked(size_t index, uint64_t structural_hash) noexcept {
        auto iter = _buckets.find(structural_hash);
        if (iter == _buckets.end()) { return; }
        auto &indices = iter->second;
        for (auto i = size_t{0u}; i < indices.size(); ++i) {
            if (indices[i] == index) {
                indices[i] = indices.back();
                indices.pop_back();
                break;
            }
        }
        if (indices.empty()) { _buckets.erase(iter); }
    }

public:
    LayoutInternTable() noexcept = default;
    LayoutInternTable(LayoutInternTable const &) = delete;
    LayoutInternTable(LayoutInternTable &&) = delete;
    LayoutInternTable &operator=(LayoutInternTable const &) = delete;
    LayoutInternTable &operator=(LayoutInternTable &&) = delete;
    ~LayoutInternTable() noexcept = default;

    // Returns the interned entry for `key`, creating the native object through
    // `create` when no live entry matches. Increments the reference count of the
    // returned entry. Returns an empty entry (and leaves the table untouched)
    // when `create` fails.
    template<typename CreateFn>
    [[nodiscard]] LayoutAcquireResult acquire(const Key &key, CreateFn &&create) noexcept {
        auto const structural_hash = key.structural_hash();
        {
            std::lock_guard lock{_mutex};
            if (auto index = find_locked(key, structural_hash); index != npos) {
                auto &slot = _slots[index];
                ++slot.refs;
                return {.layout = slot.layout, .orphaned_handle = 0u};
            }
        }
        // No lock is held from here on: creating the object (and, for a real
        // caller, the shader compilation that led here) must not serialize other
        // threads.
        auto const handle = static_cast<uint64_t>(create(key));
        if (handle == 0u) { return {}; }// failure: the table was never touched
        std::lock_guard lock{_mutex};
        if (auto index = find_locked(key, structural_hash); index != npos) {
            // Another compile interned the identical structure first. The table
            // is already correct; report our own handle as orphaned so it is
            // destroyed instead of leaking.
            auto &slot = _slots[index];
            ++slot.refs;
            return {.layout = slot.layout, .orphaned_handle = handle};
        }
        auto index = size_t{0u};
        if (!_free_slots.empty()) {
            index = _free_slots.back();
            _free_slots.pop_back();
        } else {
            index = _slots.size();
            _slots.emplace_back();
        }
        auto &slot = _slots[index];
        slot.key = key;
        slot.refs = 1u;
        slot.layout = InternedLayout{
            .id = _next_id++,
            .handle = handle,
            .structural_hash = structural_hash};
        _buckets[structural_hash].emplace_back(index);
        _live_ids.emplace(slot.layout.id, index);
        ++_live_count;
        return {.layout = slot.layout, .orphaned_handle = 0u};
    }

    // Takes an *additional* reference on a live entry and returns its native
    // handle (0 when the id is unknown, i.e. the layout was already destroyed or
    // the table was drained). Plan P4: a replay graph keeps the interned layouts
    // its recorded shaders use alive after those shaders are gone, so it has to
    // retain by id - the structural key is no longer available (or needed).
    [[nodiscard]] uint64_t retain(uint64_t id) noexcept {
        std::lock_guard lock{_mutex};
        auto iter = _live_ids.find(id);
        if (iter == _live_ids.end()) { return 0u; }
        auto &slot = _slots[iter->second];
        ++slot.refs;
        return slot.layout.handle;
    }

    // Drops one reference. Returns the native handle the caller must destroy, or
    // 0 when the entry survives (or the id is unknown/already released).
    [[nodiscard]] uint64_t release(uint64_t id) noexcept {
        std::lock_guard lock{_mutex};
        auto iter = _live_ids.find(id);
        if (iter == _live_ids.end()) { return 0u; }
        auto const index = iter->second;
        auto &slot = _slots[index];
        if (slot.refs > 1u) {
            --slot.refs;
            return 0u;
        }
        auto const handle = slot.layout.handle;
        auto const structural_hash = slot.layout.structural_hash;
        _live_ids.erase(iter);
        unlink_locked(index, structural_hash);
        // Drop the key's storage as well: a dead slot keeps no structure alive.
        slot.key = Key{};
        slot.layout = InternedLayout{};
        slot.refs = 0u;
        _free_slots.emplace_back(index);
        --_live_count;
        return handle;
    }

    // Empties the table and appends every live handle to `handles` (device
    // teardown). Ids become unknown, so a later `release` is a no-op.
    void drain(vstd::vector<uint64_t> &handles) noexcept {
        std::lock_guard lock{_mutex};
        handles.reserve(handles.size() + _live_count);
        for (auto &&slot : _slots) {
            if (slot.layout.handle != 0u) {
                handles.emplace_back(slot.layout.handle);
            }
        }
        _buckets.clear();
        _live_ids.clear();
        _slots.clear();
        _free_slots.clear();
        _live_count = 0u;
    }

    [[nodiscard]] uint64_t handle_of(uint64_t id) const noexcept {
        std::lock_guard lock{_mutex};
        auto iter = _live_ids.find(id);
        return iter == _live_ids.end() ? 0u : _slots[iter->second].layout.handle;
    }

    [[nodiscard]] size_t live_layout_count() const noexcept {
        std::lock_guard lock{_mutex};
        return _live_count;
    }

    [[nodiscard]] size_t live_reference_count() const noexcept {
        std::lock_guard lock{_mutex};
        auto count = size_t{0u};
        for (auto &&slot : _slots) { count += slot.refs; }
        return count;
    }
};

// Builds the VkDescriptorSetLayoutCreateInfo a key describes, and owns every
// array that create info points into (bindings, binding flags, immutable
// samplers), so the builder - not the key - has to outlive the create info.
class DescriptorSetLayoutCreateInfoBuilder {

private:
    vstd::vector<VkDescriptorSetLayoutBinding> _bindings;
    vstd::vector<VkDescriptorBindingFlags> _binding_flags;
    vstd::vector<vstd::vector<VkSampler>> _immutable_samplers;
    VkDescriptorSetLayoutBindingFlagsCreateInfo _binding_flags_info{};
    VkDescriptorSetLayoutCreateInfo _create_info{};

public:
    explicit DescriptorSetLayoutCreateInfoBuilder(const DescriptorSetLayoutKey &key) noexcept {
        _bindings.reserve(key.bindings.size());
        _binding_flags.reserve(key.bindings.size());
        _immutable_samplers.resize(key.bindings.size());
        auto has_binding_flags = false;
        for (auto i = size_t{0u}; i < key.bindings.size(); ++i) {
            auto &&source = key.bindings[i];
            auto &samplers = _immutable_samplers[i];
            samplers = source.immutable_samplers;
            _bindings.emplace_back(VkDescriptorSetLayoutBinding{
                .binding = source.binding,
                .descriptorType = source.descriptor_type,
                .descriptorCount = source.descriptor_count,
                .stageFlags = source.stage_flags,
                .pImmutableSamplers = samplers.empty() ? nullptr : samplers.data()});
            _binding_flags.emplace_back(source.binding_flags);
            has_binding_flags |= source.binding_flags != 0u;
        }
        _binding_flags_info = VkDescriptorSetLayoutBindingFlagsCreateInfo{
            .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_BINDING_FLAGS_CREATE_INFO,
            .bindingCount = static_cast<uint32_t>(_binding_flags.size()),
            .pBindingFlags = _binding_flags.empty() ? nullptr : _binding_flags.data()};
        // An all-zero binding-flag chain describes exactly the same layout as no
        // chain at all, so it is only attached when a flag is actually set (which
        // is what this backend did before interning: only bindless sets carry
        // flags).
        _create_info = VkDescriptorSetLayoutCreateInfo{
            .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
            .pNext = has_binding_flags ? &_binding_flags_info : nullptr,
            .flags = key.flags,
            .bindingCount = static_cast<uint32_t>(_bindings.size()),
            .pBindings = _bindings.empty() ? nullptr : _bindings.data()};
    }

    DescriptorSetLayoutCreateInfoBuilder(DescriptorSetLayoutCreateInfoBuilder const &) = delete;
    DescriptorSetLayoutCreateInfoBuilder &operator=(DescriptorSetLayoutCreateInfoBuilder const &) = delete;

    [[nodiscard]] const VkDescriptorSetLayoutCreateInfo &create_info() const noexcept {
        return _create_info;
    }
};

// The device-owned interning facility (layout_intern.cpp). `Device` holds one
// and is the only caller; the Vulkan create/destroy calls for both object kinds
// live behind it.
class LayoutInterner {

private:
    LayoutInternTable<DescriptorSetLayoutKey> _descriptor_set_layouts;
    LayoutInternTable<PipelineLayoutKey> _pipeline_layouts;

public:
    LayoutInterner() noexcept = default;
    LayoutInterner(LayoutInterner const &) = delete;
    LayoutInterner(LayoutInterner &&) = delete;
    LayoutInterner &operator=(LayoutInterner const &) = delete;
    LayoutInterner &operator=(LayoutInterner &&) = delete;
    ~LayoutInterner() noexcept = default;

    // Interns the descriptor-set layout `key` describes. An empty entry means the
    // device refused to create it.
    [[nodiscard]] InternedLayout acquire_descriptor_set_layout(
        const DescriptorSetLayoutKey &key, VkDevice device) noexcept;

    // Interns the pipeline layout `key` describes. Its descriptor sets are
    // resolved from their interned ids; an id that is no longer live makes the
    // call fail closed instead of creating a layout with a dangling set.
    [[nodiscard]] InternedLayout acquire_pipeline_layout(
        const PipelineLayoutKey &key, VkDevice device) noexcept;

    // Releases one reference each. When it was the last one the device destroys
    // the native object and the destroyed handle is returned (0 otherwise).
    uint64_t release_descriptor_set_layout(uint64_t id, VkDevice device) noexcept;
    uint64_t release_pipeline_layout(uint64_t id, VkDevice device) noexcept;
    // Plan P4: takes an additional reference on an already-interned layout by id,
    // so a replay graph can outlive the shaders whose creation acquired it. The
    // returned handle is the same native object and 0 means "no such live entry"
    // (the caller must then refuse to record instead of referencing a layout that
    // is about to be destroyed). Release it with the matching `release_*` above:
    // retain-by-id and release-by-id are symmetric.
    [[nodiscard]] uint64_t retain_descriptor_set_layout(uint64_t id) noexcept {
        return _descriptor_set_layouts.retain(id);
    }
    [[nodiscard]] uint64_t retain_pipeline_layout(uint64_t id) noexcept {
        return _pipeline_layouts.retain(id);
    }

    // Destroys every layout still interned. Called by `~Device` before the
    // logical device goes away; afterwards every release is a no-op.
    void destroy_all(VkDevice device) noexcept;

    // Introspection for diagnostics and tests (never used to make a decision).
    [[nodiscard]] size_t descriptor_set_layout_count() const noexcept {
        return _descriptor_set_layouts.live_layout_count();
    }
    [[nodiscard]] size_t pipeline_layout_count() const noexcept {
        return _pipeline_layouts.live_layout_count();
    }
    [[nodiscard]] size_t descriptor_set_layout_reference_count() const noexcept {
        return _descriptor_set_layouts.live_reference_count();
    }
    [[nodiscard]] size_t pipeline_layout_reference_count() const noexcept {
        return _pipeline_layouts.live_reference_count();
    }
};

}// namespace lc::vk::detail