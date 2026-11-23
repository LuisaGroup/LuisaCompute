// Host-level coverage for the Vulkan layout interning facility (plan P1.1,
// src/backends/vk/layout_intern.h).
//
// The keys, their hashing/equality and the reference-counted interning table are
// deliberately free of Vulkan entry points, so the two properties that must hold
// for the optimisation to be safe are asserted here instead of through fragile
// shader-generation assumptions:
//
//  * positive: two independently built copies of the same structure are equal,
//    hash equally, and intern to the *same* id/handle while the entry is alive -
//    the native object is created once, not once per compile.
//  * negative: a different binding list, binding flags, immutable-sampler
//    content, stage bits, set flags, push-constant range (stage/offset/size) or
//    pipeline-layout flag never aliases the base structure; nor do two pipeline
//    layouts whose descriptor sets are structurally different.
//
// The table tests also cover what the real device path relies on: reference
// counting (destroy exactly once, on the last release), the lost-insertion-race
// path (the loser's handle is reported as orphaned instead of leaking), the
// creation-failure path (nothing is inserted), and teardown drain (later
// releases are no-ops, never a double destroy).

#include "ut/ut.hpp"
#include "layout_intern.h"

#include <array>
#include <atomic>
#include <cstdint>
#include <thread>
#include <utility>

#include <luisa/core/logging.h>

using namespace boost::ut;
using namespace boost::ut::literals;

namespace vk_detail = lc::vk::detail;

namespace {

// Fake VkSampler identities: the immutable-sampler content is compared by value,
// so plain integers are enough (the table never dereferences a handle).
[[nodiscard]] VkSampler fake_sampler(uint64_t value) noexcept {
    return vk_detail::uint64_to_handle<VkSampler>(value);
}

[[nodiscard]] vk_detail::DescriptorSetLayoutBindingKey
make_binding(uint32_t binding, VkDescriptorType type, uint32_t count,
             VkShaderStageFlags stages,
             VkDescriptorBindingFlags flags = 0u) noexcept {
    return vk_detail::DescriptorSetLayoutBindingKey{
        .binding = binding,
        .descriptor_type = type,
        .descriptor_count = count,
        .stage_flags = stages,
        .binding_flags = flags,
        .immutable_samplers = {}};
}

// The "local" descriptor set of a compute shader: one storage buffer in set 0.
[[nodiscard]] vk_detail::DescriptorSetLayoutKey make_local_set_key() noexcept {
    vk_detail::DescriptorSetLayoutKey key;
    key.flags = 0u;
    key.bindings.emplace_back(make_binding(
        0u, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1u, VK_SHADER_STAGE_COMPUTE_BIT));
    return key;
}

// The sampler heap of the same shader: 16 immutable samplers in set 1.
[[nodiscard]] vk_detail::DescriptorSetLayoutKey make_sampler_set_key() noexcept {
    vk_detail::DescriptorSetLayoutKey key;
    key.flags = 0u;
    auto binding = make_binding(
        0u, VK_DESCRIPTOR_TYPE_SAMPLER, 16u, VK_SHADER_STAGE_ALL);
    for (auto i = uint64_t{0u}; i < 16u; ++i) {
        binding.immutable_samplers.emplace_back(fake_sampler(0x1000u + i));
    }
    key.bindings.emplace_back(std::move(binding));
    return key;
}

[[nodiscard]] vk_detail::PipelineLayoutKey
make_pipeline_layout_key(uint64_t set_id, uint64_t set_hash,
                         VkShaderStageFlags stages = VK_SHADER_STAGE_COMPUTE_BIT,
                         uint32_t offset = 0u, uint32_t size = 32u,
                         VkPipelineLayoutCreateFlags flags = 0u) noexcept {
    vk_detail::PipelineLayoutKey key;
    key.flags = flags;
    key.descriptor_set_layouts.emplace_back(set_id);
    key.descriptor_set_layout_hashes.emplace_back(set_hash);
    key.push_constant_ranges.emplace_back(vk_detail::PushConstantRangeKey{
        .stage_flags = stages, .offset = offset, .size = size});
    return key;
}

// Minimal stand-in for the device: hands out increasing fake handles and counts
// them, so "created once" is observable without a Vulkan device.
struct FakeFactory {
    uint64_t next{1u};
    uint64_t created{0u};

    [[nodiscard]] uint64_t operator()(const vk_detail::DescriptorSetLayoutKey &) noexcept {
        ++created;
        return next++;
    }
};

}// namespace

int main(int argc, char *argv[]) {
    boost::ut::detail::cfg::parse_arg_with_fallback(
        argc, const_cast<const char **>(argv));

    "vk_layout_handle_conversion_supports_integer_handles"_test = [] {
        constexpr auto raw = uint64_t{0x1234'5678'9abc'def0ull};
        expect(vk_detail::handle_to_uint64<uint64_t>(raw) == raw);
        expect(vk_detail::uint64_to_handle<uint64_t>(raw) == raw);
    };

    // ---- structural keys: positive -----------------------------------------
    "vk_layout_key_identical_structures_are_equal"_test = [] {
        auto first = make_local_set_key();
        auto second = make_local_set_key();
        expect(first == second)
            << "two copies of the same structure must compare equal";
        expect(first.structural_hash() == second.structural_hash())
            << "equal structures must hash equally";
        expect(first.structural_hash() != 0u);
        // A copy taken through a temporary must not change anything either.
        auto copy = first;
        expect(copy == first);
        expect(copy.structural_hash() == first.structural_hash());
    };

    "vk_layout_key_sampler_set_is_structural"_test = [] {
        auto first = make_sampler_set_key();
        auto second = make_sampler_set_key();
        expect(first == second) << "identical immutable samplers must be equal";
        expect(first.structural_hash() == second.structural_hash());
        LUISA_INFO("local set hash = {:#x}, sampler set hash = {:#x}",
                   make_local_set_key().structural_hash(),
                   first.structural_hash());
    };

    // ---- structural keys: negative (never alias) ---------------------------
    "vk_layout_key_binding_fields_never_alias"_test = [] {
        auto base = make_local_set_key();
        // Different binding number.
        auto other_binding = make_local_set_key();
        other_binding.bindings[0u].binding = 1u;
        expect(other_binding != base) << "a different binding number must not alias";
        expect(other_binding.structural_hash() != base.structural_hash());
        // Different descriptor type.
        auto other_type = make_local_set_key();
        other_type.bindings[0u].descriptor_type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        expect(other_type != base) << "a different descriptor type must not alias";
        expect(other_type.structural_hash() != base.structural_hash());
        // Different descriptor count.
        auto other_count = make_local_set_key();
        other_count.bindings[0u].descriptor_count = 2u;
        expect(other_count != base) << "a different descriptor count must not alias";
        expect(other_count.structural_hash() != base.structural_hash());
        // Different stage bits.
        auto other_stage = make_local_set_key();
        other_stage.bindings[0u].stage_flags =
            VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;
        expect(other_stage != base)
            << "different stage bits must not alias (a raster and a compute "
               "shader cannot share a set layout)";
        expect(other_stage.structural_hash() != base.structural_hash());
        // Extra binding: a different binding *list*.
        auto extra_binding = make_local_set_key();
        extra_binding.bindings.emplace_back(make_binding(
            1u, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1u, VK_SHADER_STAGE_COMPUTE_BIT));
        expect(extra_binding != base) << "a longer binding list must not alias";
        expect(extra_binding.structural_hash() != base.structural_hash());
        // Reordered bindings: the order is part of the structure.
        auto reordered = make_local_set_key();
        reordered.bindings.insert(
            reordered.bindings.begin(),
            make_binding(1u, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1u,
                         VK_SHADER_STAGE_COMPUTE_BIT));
        expect(reordered != extra_binding)
            << "the binding order must be part of the structure, even when the "
               "set of bindings is the same";
        expect(reordered.structural_hash() != extra_binding.structural_hash());
        // Different set-level create flags (e.g. update-after-bind pool).
        auto other_flags = make_local_set_key();
        other_flags.flags = VK_DESCRIPTOR_SET_LAYOUT_CREATE_UPDATE_AFTER_BIND_POOL_BIT;
        expect(other_flags != base) << "a different create flags value must not alias";
        expect(other_flags.structural_hash() != base.structural_hash());
    };

    "vk_layout_key_binding_flags_never_alias"_test = [] {
        auto base = make_local_set_key();
        constexpr auto kinds = std::array<VkDescriptorBindingFlags, 5u>{
            static_cast<VkDescriptorBindingFlags>(VK_DESCRIPTOR_BINDING_UPDATE_AFTER_BIND_BIT),
            static_cast<VkDescriptorBindingFlags>(VK_DESCRIPTOR_BINDING_UPDATE_UNUSED_WHILE_PENDING_BIT),
            static_cast<VkDescriptorBindingFlags>(VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT),
            static_cast<VkDescriptorBindingFlags>(VK_DESCRIPTOR_BINDING_VARIABLE_DESCRIPTOR_COUNT_BIT),
            static_cast<VkDescriptorBindingFlags>(VK_DESCRIPTOR_BINDING_UPDATE_AFTER_BIND_BIT |
                                                  VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT)};
        for (auto flags : kinds) {
            auto other = make_local_set_key();
            other.bindings[0u].binding_flags = flags;
            expect(other != base)
                << "a binding flags value must not alias an unflagged binding";
            expect(other.structural_hash() != base.structural_hash());
        }
        // Two different flag sets must not alias each other either.
        auto update_after_bind = make_local_set_key();
        update_after_bind.bindings[0u].binding_flags =
            VK_DESCRIPTOR_BINDING_UPDATE_AFTER_BIND_BIT;
        auto partially_bound = make_local_set_key();
        partially_bound.bindings[0u].binding_flags =
            VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT;
        expect(update_after_bind != partially_bound);
        expect(update_after_bind.structural_hash() != partially_bound.structural_hash());
    };

    "vk_layout_key_immutable_samplers_never_alias"_test = [] {
        auto base = make_sampler_set_key();
        // Different sampler content.
        auto different_content = make_sampler_set_key();
        different_content.bindings[0u].immutable_samplers[7u] = fake_sampler(0xbeefu);
        expect(different_content != base)
            << "different immutable-sampler content must not alias";
        expect(different_content.structural_hash() != base.structural_hash());
        // Same samplers, different order.
        auto swapped = make_sampler_set_key();
        std::swap(swapped.bindings[0u].immutable_samplers[3u],
                  swapped.bindings[0u].immutable_samplers[4u]);
        expect(swapped != base)
            << "the immutable-sampler order is baked into the layout";
        expect(swapped.structural_hash() != base.structural_hash());
        // One sampler fewer.
        auto shorter = make_sampler_set_key();
        shorter.bindings[0u].immutable_samplers.pop_back();
        expect(shorter != base);
        expect(shorter.structural_hash() != base.structural_hash());
        // No immutable samplers at all.
        auto none = make_sampler_set_key();
        none.bindings[0u].immutable_samplers.clear();
        expect(none != base);
        expect(none.structural_hash() != base.structural_hash());
    };

    "vk_pipeline_layout_key_fields_never_alias"_test = [] {
        auto base = make_pipeline_layout_key(1u, 0x11u);
        expect(base == make_pipeline_layout_key(1u, 0x11u));
        // Different push-constant stage bits.
        expect(base != make_pipeline_layout_key(1u, 0x11u, VK_SHADER_STAGE_ALL))
            << "the push-constant stage mask must be part of the key";
        // Different push-constant offset.
        expect(base != make_pipeline_layout_key(1u, 0x11u, VK_SHADER_STAGE_COMPUTE_BIT, 4u))
            << "the push-constant offset must be part of the key";
        // Different push-constant size.
        expect(base != make_pipeline_layout_key(1u, 0x11u, VK_SHADER_STAGE_COMPUTE_BIT, 0u, 64u))
            << "the push-constant size must be part of the key";
        // Different interned descriptor-set structure.
        expect(base != make_pipeline_layout_key(2u, 0x11u))
            << "a different interned descriptor-set id must not alias";
        // Different pipeline-layout flags.
        expect(base != make_pipeline_layout_key(
                           1u, 0x11u, VK_SHADER_STAGE_COMPUTE_BIT, 0u, 32u,
                           VK_PIPELINE_LAYOUT_CREATE_INDEPENDENT_SETS_BIT_EXT))
            << "the pipeline-layout create flags must be part of the key";
        // Ordered descriptor sets.
        auto two_sets = base;
        two_sets.descriptor_set_layouts.emplace_back(2u);
        two_sets.descriptor_set_layout_hashes.emplace_back(0x22u);
        expect(two_sets != base) << "an extra descriptor set must not alias";
        // Multiple push-constant ranges, in order.
        auto two_ranges = base;
        two_ranges.push_constant_ranges.emplace_back(vk_detail::PushConstantRangeKey{
            .stage_flags = VK_SHADER_STAGE_ALL, .offset = 32u, .size = 16u});
        expect(two_ranges != base) << "an extra push-constant range must not alias";
        auto swapped_ranges = two_ranges;
        std::swap(swapped_ranges.push_constant_ranges[0u],
                  swapped_ranges.push_constant_ranges[1u]);
        expect(swapped_ranges != two_ranges) << "push-constant ranges are ordered";
    };

    "vk_pipeline_layout_hash_is_structural_not_id_based"_test = [] {
        // Two devices (or two table generations) hand out different ids for the
        // same structure. Equality - which is what the table uses - must differ,
        // but the portable fingerprint a PSO/blob cache persists must not.
        auto first = make_pipeline_layout_key(1u, 0x4242u);
        auto second = make_pipeline_layout_key(9u, 0x4242u);
        expect(first != second) << "table-local ids decide table equality";
        expect(first.structural_hash() == second.structural_hash())
            << "the portable hash must not include table-local ids";
        // The set hashes *are* part of it.
        auto other_sets = make_pipeline_layout_key(1u, 0x4343u);
        expect(other_sets.structural_hash() != first.structural_hash())
            << "the descriptor-set structures must change the portable hash";
    };

    // ---- interning table ----------------------------------------------------
    "vk_layout_intern_table_dedups_identical_structures"_test = [] {
        vk_detail::LayoutInternTable<vk_detail::DescriptorSetLayoutKey> table;
        FakeFactory factory;
        auto create = [&factory](const vk_detail::DescriptorSetLayoutKey &key) noexcept {
            return factory(key);
        };
        auto first = table.acquire(make_local_set_key(), create);
        auto second = table.acquire(make_local_set_key(), create);
        expect(first.orphaned_handle == 0u);
        expect(second.orphaned_handle == 0u);
        expect(eq(factory.created, 1u))
            << "an identical structure must not create a second Vulkan object";
        expect(eq(first.layout.id, second.layout.id))
            << "identical structures must intern to the same id";
        expect(eq(first.layout.handle, second.layout.handle))
            << "identical structures must intern to the same native handle";
        expect(eq(first.layout.structural_hash, second.layout.structural_hash));
        expect(eq(table.live_layout_count(), 1u));
        expect(eq(table.live_reference_count(), 2u))
            << "both acquisitions hold a reference";
        expect(eq(table.handle_of(first.layout.id), first.layout.handle));
    };

    "vk_layout_intern_table_distinguishes_different_structures"_test = [] {
        vk_detail::LayoutInternTable<vk_detail::DescriptorSetLayoutKey> table;
        FakeFactory factory;
        auto create = [&factory](const vk_detail::DescriptorSetLayoutKey &key) noexcept {
            return factory(key);
        };
        auto local = table.acquire(make_local_set_key(), create);
        auto sampler = table.acquire(make_sampler_set_key(), create);
        expect(local.layout.id != sampler.layout.id)
            << "different structures must intern to different ids";
        expect(local.layout.handle != sampler.layout.handle)
            << "different structures must not share a Vulkan object";
        expect(local.layout.structural_hash != sampler.layout.structural_hash);
        expect(eq(factory.created, 2u));
        expect(eq(table.live_layout_count(), 2u));

        // A structurally different binding flag set must not be folded into an
        // existing entry, even though everything else matches.
        auto bound_elsewhere = make_local_set_key();
        bound_elsewhere.bindings[0u].binding_flags =
            VK_DESCRIPTOR_BINDING_UPDATE_AFTER_BIND_BIT;
        auto third = table.acquire(bound_elsewhere, create);
        expect(third.layout.id != local.layout.id)
            << "a different binding flags value must not alias a live entry";
        expect(eq(factory.created, 3u));
        expect(eq(table.live_layout_count(), 3u));
    };

    "vk_layout_intern_table_reference_counting_destroys_once"_test = [] {
        vk_detail::LayoutInternTable<vk_detail::DescriptorSetLayoutKey> table;
        FakeFactory factory;
        auto create = [&factory](const vk_detail::DescriptorSetLayoutKey &key) noexcept {
            return factory(key);
        };
        auto first = table.acquire(make_local_set_key(), create);
        auto second = table.acquire(make_local_set_key(), create);
        expect(eq(table.live_reference_count(), 2u));
        // First release keeps the entry alive and destroys nothing.
        expect(eq(table.release(first.layout.id), 0u))
            << "a still-referenced entry must not be destroyed";
        expect(eq(table.live_layout_count(), 1u));
        expect(eq(table.live_reference_count(), 1u));
        expect(eq(table.handle_of(second.layout.id), second.layout.handle))
            << "the surviving reference must keep the same handle";
        // Last release hands the handle back exactly once.
        expect(eq(table.release(second.layout.id), second.layout.handle))
            << "the last release must hand back the native handle";
        expect(eq(table.live_layout_count(), 0u));
        expect(eq(table.live_reference_count(), 0u));
        // A second release of the same id is a no-op, never a double destroy.
        expect(eq(table.release(second.layout.id), 0u))
            << "releasing an already released id must not destroy twice";
        expect(eq(table.release(0u), 0u)) << "id 0 is never a live entry";
        expect(eq(table.release(12345u), 0u)) << "an unknown id must be a no-op";
        // The structure was released, so a later compile creates a fresh object
        // (with a fresh, monotonically larger id).
        auto third = table.acquire(make_local_set_key(), create);
        expect(third.layout.id > second.layout.id)
            << "interned ids are monotonic and never reused";
        expect(third.layout.handle != second.layout.handle);
        expect(eq(factory.created, 2u))
            << "a structure whose last reference was released is created again";
    };

    // Plan P4: a replay graph must keep the interned layouts its recorded shaders
    // use alive *after* those shaders are destroyed, so the table has to offer a
    // retain-by-id. Retain and release must be symmetric: every extra reference is
    // given back exactly once, and the object dies on the last release.
    "vk_layout_intern_table_retain_by_id"_test = [] {
        vk_detail::LayoutInternTable<vk_detail::DescriptorSetLayoutKey> table;
        FakeFactory factory;
        auto create = [&factory](const vk_detail::DescriptorSetLayoutKey &key) noexcept {
            return factory(key);
        };
        auto acquired = table.acquire(make_local_set_key(), create);
        expect(eq(table.live_reference_count(), 1u));
        // The shader's own reference plus the graph's.
        expect(eq(table.retain(acquired.layout.id), acquired.layout.handle))
            << "retain(id) must hand back the same native object, retargeted";
        expect(eq(table.live_reference_count(), 2u));
        expect(eq(table.live_layout_count(), 1u))
            << "retaining must not create a second object for the same structure";
        expect(eq(factory.created, 1u));
        // The shader is destroyed first: the graph's reference keeps the layout.
        expect(eq(table.release(acquired.layout.id), 0u))
            << "the recording's reference must survive the shader's release";
        expect(eq(table.live_layout_count(), 1u));
        expect(eq(table.handle_of(acquired.layout.id), acquired.layout.handle));
        // The graph is destroyed second: now the object goes.
        expect(eq(table.release(acquired.layout.id), acquired.layout.handle))
            << "the last release must destroy exactly once";
        expect(eq(table.live_layout_count(), 0u));
        expect(eq(table.live_reference_count(), 0u));
        // Retaining an id that is no longer live is a defined failure (0), never a
        // resurrected handle: a graph must refuse to record instead.
        expect(eq(table.retain(acquired.layout.id), 0u))
            << "retaining a released id must fail closed";
        expect(eq(table.retain(0u), 0u)) << "id 0 is never a live entry";
        expect(eq(table.retain(12345u), 0u)) << "an unknown id must fail closed";
        // Drain (device teardown) invalidates the ids, so a later release is a
        // no-op rather than a double destroy - the same contract `release` has.
        auto survivor = table.acquire(make_local_set_key(), create);
        expect(eq(table.retain(survivor.layout.id), survivor.layout.handle));
        vstd::vector<uint64_t> drained;
        table.drain(drained);
        expect(eq(drained.size(), 1u)) << "drain hands out every live handle once";
        expect(eq(table.release(survivor.layout.id), 0u))
            << "a release after the drain must not destroy again";
        expect(eq(table.retain(survivor.layout.id), 0u));
    };

    "vk_layout_intern_table_creation_failure_inserts_nothing"_test = [] {
        vk_detail::LayoutInternTable<vk_detail::DescriptorSetLayoutKey> table;
        auto fail = [](const vk_detail::DescriptorSetLayoutKey &) noexcept {
            return uint64_t{0u};// VK_NULL_HANDLE
        };
        auto result = table.acquire(make_local_set_key(), fail);
        expect(result.layout.handle == 0u)
            << "a failed creation must report an empty layout";
        expect(eq(result.layout.handle, 0u));
        expect(eq(result.orphaned_handle, 0u));
        expect(eq(table.live_layout_count(), 0u))
            << "a failed creation must not insert a bogus handle";
        expect(eq(table.live_reference_count(), 0u));
        // A later successful acquisition must not see any residue.
        FakeFactory factory;
        auto create = [&factory](const vk_detail::DescriptorSetLayoutKey &key) noexcept {
            return factory(key);
        };
        auto second = table.acquire(make_local_set_key(), create);
        expect(second.layout.handle != 0u)
            << "the table must stay usable after a failed creation";
        expect(eq(second.layout.id, 1u))
            << "the first *successful* entry gets the first id";
        expect(eq(table.live_layout_count(), 1u));
    };

    "vk_layout_intern_table_lost_race_orphans_the_losers_handle"_test = [] {
        // Two compiles of the same structure race: both create a native object,
        // exactly one is adopted and the other is reported as orphaned so the
        // caller destroys it. Forced deterministically: neither thread may insert
        // before both have created, which is exactly the window a real race has.
        vk_detail::LayoutInternTable<vk_detail::DescriptorSetLayoutKey> table;
        std::atomic<uint64_t> entered{0u};
        std::atomic<uint64_t> created{0u};
        auto create = [&entered, &created](const vk_detail::DescriptorSetLayoutKey &) noexcept {
            entered.fetch_add(1u, std::memory_order_acq_rel);
            while (entered.load(std::memory_order_acquire) != 2u) {
                std::this_thread::yield();
            }
            return created.fetch_add(1u, std::memory_order_relaxed) + 1u;
        };
        vk_detail::LayoutAcquireResult results[2];
        std::thread first{[&] { results[0] = table.acquire(make_local_set_key(), create); }};
        std::thread second{[&] { results[1] = table.acquire(make_local_set_key(), create); }};
        first.join();
        second.join();
        expect(eq(created.load(), 2u))
            << "both racing compiles must have created their own object";
        expect(eq(results[0].layout.handle, results[1].layout.handle))
            << "both racing compiles must end up on the same interned handle";
        expect(eq(results[0].layout.id, results[1].layout.id));
        auto orphans = (results[0].orphaned_handle != 0u ? 1 : 0) +
                       (results[1].orphaned_handle != 0u ? 1 : 0);
        expect(eq(orphans, 1))
            << "exactly one racing handle must be reported as orphaned";
        auto orphan = results[0].orphaned_handle != 0u ?
                          results[0].orphaned_handle :
                          results[1].orphaned_handle;
        expect(orphan != results[0].layout.handle)
            << "the orphaned handle is not the one the table kept";
        expect(eq(table.live_layout_count(), 1u)) << "a race must not duplicate entries";
        expect(eq(table.live_reference_count(), 2u));
    };

    "vk_layout_intern_table_drain_hands_every_handle_over_once"_test = [] {
        vk_detail::LayoutInternTable<vk_detail::DescriptorSetLayoutKey> table;
        FakeFactory factory;
        auto create = [&factory](const vk_detail::DescriptorSetLayoutKey &key) noexcept {
            return factory(key);
        };
        auto local = table.acquire(make_local_set_key(), create);
        auto sampler = table.acquire(make_sampler_set_key(), create);
        vstd::vector<uint64_t> handles;
        table.drain(handles);
        expect(eq(handles.size(), 2u))
            << "teardown must hand over every live handle exactly once";
        auto contains = [&handles](uint64_t handle) {
            for (auto candidate : handles) {
                if (candidate == handle) { return true; }
            }
            return false;
        };
        expect(contains(local.layout.handle));
        expect(contains(sampler.layout.handle));
        expect(eq(table.live_layout_count(), 0u));
        // Everything after the drain is a no-op, so a late shader destructor
        // cannot destroy an already destroyed object.
        expect(eq(table.release(local.layout.id), 0u));
        expect(eq(table.handle_of(sampler.layout.id), 0u));
        vstd::vector<uint64_t> again;
        table.drain(again);
        expect(again.empty());
    };

    "vk_layout_intern_table_pipeline_layout_uses_set_ids"_test = [] {
        vk_detail::LayoutInternTable<vk_detail::DescriptorSetLayoutKey> sets;
        vk_detail::LayoutInternTable<vk_detail::PipelineLayoutKey> pipelines;
        FakeFactory factory;
        auto create_set = [&factory](const vk_detail::DescriptorSetLayoutKey &key) noexcept {
            return factory(key);
        };
        auto pipeline_creates = uint64_t{0u};
        auto create_pipeline = [&pipeline_creates](const vk_detail::PipelineLayoutKey &) noexcept {
            return 0x8000u + ++pipeline_creates;
        };
        auto local = sets.acquire(make_local_set_key(), create_set);
        auto sampler = sets.acquire(make_sampler_set_key(), create_set);
        auto pipeline_key = make_pipeline_layout_key(
            local.layout.id, local.layout.structural_hash);
        pipeline_key.descriptor_set_layouts.emplace_back(sampler.layout.id);
        pipeline_key.descriptor_set_layout_hashes.emplace_back(
            sampler.layout.structural_hash);
        auto first = pipelines.acquire(pipeline_key, create_pipeline);
        auto second = pipelines.acquire(pipeline_key, create_pipeline);
        expect(eq(first.layout.id, second.layout.id))
            << "an identical pipeline layout must be interned once";
        expect(eq(pipelines.live_layout_count(), 1u));
        // Swapping the two sets keeps the same *set of structures* but describes
        // a different pipeline layout: the order matters.
        auto swapped = pipeline_key;
        std::swap(swapped.descriptor_set_layouts[0u], swapped.descriptor_set_layouts[1u]);
        std::swap(swapped.descriptor_set_layout_hashes[0u],
                  swapped.descriptor_set_layout_hashes[1u]);
        auto third = pipelines.acquire(swapped, create_pipeline);
        expect(third.layout.id != first.layout.id)
            << "a reordered descriptor-set list must not alias";
        expect(eq(pipelines.live_layout_count(), 2u));
        expect(eq(pipeline_creates, 2u))
            << "the two distinct pipeline-layout structures must create two objects";
    };

    return 0;
}