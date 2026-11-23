// Per-dispatch descriptor-set-0 updates through VkDescriptorUpdateTemplate
// (plan P2.2). VK_KHR_descriptor_update_template was promoted to core Vulkan
// 1.1, i.e. it is below this backend's 1.2 floor, and all three entry points
// used here are core 1.1 functions (`vkCreateDescriptorUpdateTemplate`,
// `vkUpdateDescriptorSetWithTemplate`, `vkDestroyDescriptorUpdateTemplate`).
//
// Why
// ---
// The backend builds a VkWriteDescriptorSet list for set 0 on every dispatch and
// used to hand it to vkUpdateDescriptorSets (plan §4 F2: 438 updates for one
// 438-dispatch batch). Vulkan 1.2 lets the *shape* of that update - which
// binding, which array element, how many descriptors, which descriptor type, and
// where the payload sits inside a packed byte block - be compiled once into a
// VkDescriptorUpdateTemplate. After that an update is one
// vkUpdateDescriptorSetWithTemplate call that only copies payload bytes into the
// set; the driver no longer re-validates a list of structures.
//
// Rules this file keeps (plan §6)
// ------------------------------
// * Fail closed, per write shape. A list the template ABI cannot represent
//   exactly - a chained pNext (the acceleration-structure write always carries
//   VkWriteDescriptorSetAccelerationStructureKHR), a descriptor type without a
//   spec-defined template payload, a null payload, overlapping ranges, a
//   payload that does not fit - is never forced through a template. The caller
//   then gets the pre-P2.2 call with exactly those writes.
// * The decision is a function of the device capability (latched once at device
//   creation from the negotiated extension, the config switch and the resolved
//   entry points) and of the write list's shape, which is a property of the
//   shader. It never reads a counter and never changes for a given shader
//   between frames.
// * Nothing here changes *what* is written. Every entry carries the write's
//   dstBinding/dstArrayElement/descriptorCount/descriptorType unchanged, the
//   payload is a byte-for-byte copy of the spec-defined structure
//   (VkDescriptorBufferInfo / VkDescriptorImageInfo / VkBufferView) and the
//   payload slot of entry i is aligned to that structure's size, which is what
//   VkDescriptorUpdateTemplateEntry requires of `offset` and `stride`.
// * The packed payload is a stack object of the update call. The cached plan
//   stores offsets and sizes only - never a source pointer - so a plan that
//   outlives the caller's descriptor arena cannot dangle. (The writes themselves
//   still point into that arena, exactly as they did before P2.2; they are read
//   only inside this call.)
// * Templates are cached per descriptor-set-layout owner (the Shader) behind a
//   spin mutex, so two streams recording the same shader cannot race a
//   creation. The owner destroys them before it gives the descriptor-set layout
//   they were built from back to the device, so the destruction order Vulkan
//   requires is a property of the owner's destructor, not of a side table.
//
// Deliberately *not* converted
// ---------------------------
// The indirect-preparation kernel's own set, the bindless-array index/heap
// updates, the swapchain blit and the native shader/raster paths each own a
// different descriptor-set lifetime and bind point; they keep
// vkUpdateDescriptorSets unchanged so this change cannot alter them.
#pragma once

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>

#include <volk.h>

#include <luisa/core/logging.h>
#include <luisa/core/spin_mutex.h>
#include <luisa/core/stl/memory.h>
#include <luisa/core/stl/vector.h>

#include "submit_profile.h"

namespace lc::vk::detail {

// ---------------------------------------------------------------------------
// planning
// ---------------------------------------------------------------------------

// Why a write list could not be turned into a template update. `kOk` is the only
// value that permits the template path; every other value keeps the legacy call.
enum class DescriptorUpdatePlanStatus : uint32_t {
    kOk,
    kNoWrites = 1u,
    kMultipleDescriptorSets,
    kChainedWrite,
    kEmptyDescriptorCount,
    kUnsupportedDescriptorType,
    kNullPayload,
    kOverlappingWrites,
    kPayloadTooLarge,
    kCacheFull,
    kTemplateCreationFailed,
};

[[nodiscard]] inline const char *
descriptor_update_plan_status_name(DescriptorUpdatePlanStatus status) noexcept {
    switch (status) {
        case DescriptorUpdatePlanStatus::kOk: return "ok";
        case DescriptorUpdatePlanStatus::kNoWrites: return "no-writes";
        case DescriptorUpdatePlanStatus::kMultipleDescriptorSets: return "multiple-descriptor-sets";
        case DescriptorUpdatePlanStatus::kChainedWrite: return "chained-write";
        case DescriptorUpdatePlanStatus::kEmptyDescriptorCount: return "empty-descriptor-count";
        case DescriptorUpdatePlanStatus::kUnsupportedDescriptorType: return "unsupported-descriptor-type";
        case DescriptorUpdatePlanStatus::kNullPayload: return "null-payload";
        case DescriptorUpdatePlanStatus::kOverlappingWrites: return "overlapping-writes";
        case DescriptorUpdatePlanStatus::kPayloadTooLarge: return "payload-too-large";
        case DescriptorUpdatePlanStatus::kCacheFull: return "cache-full";
        case DescriptorUpdatePlanStatus::kTemplateCreationFailed: return "template-creation-failed";
    }
    return "unknown";
}

// Upper bound on the packed payload of one set-0 update, and therefore the size
// of the stack block the update call packs into. The backend's descriptor
// interface allows at most 256 local bindings
// (descriptor_interface_max_local_bindings), i.e. at most 256 * 24 B in theory;
// this covers 85 such payloads, and a list that would need more is refused
// (kPayloadTooLarge, counted as a fallback) instead of being silently deferred to
// the heap on the hot path.
inline constexpr size_t descriptor_update_max_payload_size = 2048u;

// Alignment of the packed payload block. Only `std::memcpy` writes into it, so the
// language would not require any particular alignment; declaring the widest
// payload structure's alignment anyway keeps a future typed access (or a
// `reinterpret_cast` in a debug helper) legal at no cost, and makes the
// requirement explicit instead of accidental.
inline constexpr size_t descriptor_update_payload_alignment = 16u;

static_assert(alignof(VkDescriptorBufferInfo) <= descriptor_update_payload_alignment &&
                  alignof(VkDescriptorImageInfo) <= descriptor_update_payload_alignment &&
                  alignof(VkBufferView) <= descriptor_update_payload_alignment,
              "the packed descriptor payload block must hold every payload "
              "structure at its natural alignment");

// The spec-defined payload of one descriptor of `type` inside a template data
// block, or {nullptr, 0} when Vulkan defines none the backend could fill.
struct DescriptorUpdatePayload {
    const void *source{nullptr};
    uint32_t element_size{0u};
};

[[nodiscard]] inline DescriptorUpdatePayload
descriptor_update_payload(VkDescriptorType type,
                          const VkWriteDescriptorSet &write) noexcept {
    switch (type) {
        case VK_DESCRIPTOR_TYPE_SAMPLER:
        case VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER:
        case VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE:
        case VK_DESCRIPTOR_TYPE_STORAGE_IMAGE:
            return DescriptorUpdatePayload{
                write.pImageInfo,
                static_cast<uint32_t>(sizeof(VkDescriptorImageInfo))};
        case VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER:
        case VK_DESCRIPTOR_TYPE_STORAGE_BUFFER:
        case VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC:
        case VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC:
            return DescriptorUpdatePayload{
                write.pBufferInfo,
                static_cast<uint32_t>(sizeof(VkDescriptorBufferInfo))};
        case VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER:
        case VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER:
            return DescriptorUpdatePayload{
                write.pTexelBufferView,
                static_cast<uint32_t>(sizeof(VkBufferView))};
        default:
            // Acceleration structures (always chained through a pNext
            // structure this backend does not model as a payload), inline
            // uniform blocks, input attachments and the "mutable" types have
            // no payload this backend can fill correctly.
            return {};
    }
}

// One set-0 update expressed as template entries plus the placement of each
// payload inside the packed block. Entries are in write order, so entry i always
// describes `writes[i]`.
struct DescriptorUpdatePlan {
    luisa::fixed_vector<VkDescriptorUpdateTemplateEntry, 8> entries;
    // Bytes of entry i's payload (descriptorCount * element size).
    luisa::fixed_vector<size_t, 8> payload_sizes;
    size_t data_size{0u};

    void clear() noexcept {
        entries.clear();
        payload_sizes.clear();
        data_size = 0u;
    }
};

namespace descriptor_update_template_detail {

[[nodiscard]] inline size_t round_up(size_t value, size_t alignment) noexcept {
    LUISA_ASSERT(alignment != 0u, "Descriptor update payload alignment is zero.");
    auto remainder = value % alignment;
    return remainder == 0u ? value : value + (alignment - remainder);
}

[[nodiscard]] inline bool ranges_overlap(uint32_t first_begin, uint32_t first_count,
                                         uint32_t second_begin, uint32_t second_count) noexcept {
    auto first_end = static_cast<uint64_t>(first_begin) + first_count;
    auto second_end = static_cast<uint64_t>(second_begin) + second_count;
    return static_cast<uint64_t>(first_begin) < second_end &&
           static_cast<uint64_t>(second_begin) < first_end;
}

}// namespace descriptor_update_template_detail

// Turns one dispatch's set-0 write list into a template plan. Returns `kOk` only
// when every write is represented exactly; nothing is approximated.
[[nodiscard]] inline DescriptorUpdatePlanStatus
plan_descriptor_update(luisa::span<const VkWriteDescriptorSet> writes,
                       DescriptorUpdatePlan &plan) noexcept {
    using namespace descriptor_update_template_detail;
    // A rejected plan is always empty: a caller that ignores the status can not
    // apply a half-built plan by accident.
    auto reject = [&plan](DescriptorUpdatePlanStatus status) noexcept {
        plan.clear();
        return status;
    };
    plan.clear();
    if (writes.empty()) { return DescriptorUpdatePlanStatus::kNoWrites; }
    auto set = writes.front().dstSet;
    for (auto const &write : writes) {
        // A template entry updates exactly one descriptor set: the set is the
        // call's second parameter, not part of an entry.
        if (write.dstSet != set) {
            return reject(DescriptorUpdatePlanStatus::kMultipleDescriptorSets);
        }
        // A pNext chain (VkWriteDescriptorSetAccelerationStructureKHR, an
        // inline-uniform-block extension structure, ...) has no template
        // representation here, and silently dropping it would write the wrong
        // descriptor.
        if (write.pNext != nullptr) {
            return reject(DescriptorUpdatePlanStatus::kChainedWrite);
        }
        if (write.descriptorCount == 0u) {
            return reject(DescriptorUpdatePlanStatus::kEmptyDescriptorCount);
        }
        auto payload = descriptor_update_payload(write.descriptorType, write);
        if (payload.element_size == 0u) {
            return reject(DescriptorUpdatePlanStatus::kUnsupportedDescriptorType);
        }
        if (payload.source == nullptr) {
            return reject(DescriptorUpdatePlanStatus::kNullPayload);
        }
        // Two entries writing the same binding range would make the result
        // order-dependent; the legacy list applies them in order, a template
        // has no defined order. Refuse instead of guessing.
        for (auto const &existing : plan.entries) {
            if (existing.dstBinding == write.dstBinding &&
                ranges_overlap(existing.dstArrayElement, existing.descriptorCount,
                               write.dstArrayElement, write.descriptorCount)) {
                return reject(DescriptorUpdatePlanStatus::kOverlappingWrites);
            }
        }
        auto payload_size =
            static_cast<uint64_t>(payload.element_size) * write.descriptorCount;
        if (payload_size > descriptor_update_max_payload_size) {
            return reject(DescriptorUpdatePlanStatus::kPayloadTooLarge);
        }
        auto offset = round_up(plan.data_size, payload.element_size);
        if (offset + payload_size > descriptor_update_max_payload_size) {
            return reject(DescriptorUpdatePlanStatus::kPayloadTooLarge);
        }
        // `stride` is the element size: consecutive descriptors of one write
        // are contiguous, which is what the legacy pBufferInfo/pImageInfo array
        // contains as well. Both `offset` and `stride` are therefore multiples
        // of the payload structure's size, as VkDescriptorUpdateTemplateEntry
        // requires.
        plan.entries.emplace_back(VkDescriptorUpdateTemplateEntry{
            write.dstBinding,
            write.dstArrayElement,
            write.descriptorCount,
            write.descriptorType,
            offset,
            static_cast<size_t>(payload.element_size)});
        plan.payload_sizes.emplace_back(static_cast<size_t>(payload_size));
        plan.data_size = offset + static_cast<size_t>(payload_size);
    }
    return DescriptorUpdatePlanStatus::kOk;
}

// True when `writes` is exactly the shape the plan was built from - and still
// eligible for it. A cached plan may only be applied to a list whose bindings,
// array elements, counts and descriptor types are identical, and every property
// the planner would have *refused* must still be absent:
//
//  * a write that grew a pNext chain (the acceleration-structure write always
//    has one). Without this test a shape cached while it had none would match a
//    chained write later and silently drop the chain, i.e. write a descriptor the
//    legacy call would have written differently;
//  * a write whose payload pointer became null, or whose descriptor type has no
//    spec-defined template payload.
//
// The payload *pointer* is deliberately not compared: the bound resources change
// on every dispatch and `pack_descriptor_update` reads the current writes'
// payloads. What is compared is everything that decides how many bytes move and
// where they land - the payload size implied by the type, and the element count
// that multiplies it.
[[nodiscard]] inline bool
descriptor_update_plan_matches(const DescriptorUpdatePlan &plan,
                               luisa::span<const VkWriteDescriptorSet> writes) noexcept {
    if (plan.entries.empty() || plan.entries.size() != writes.size()) {
        return false;
    }
    for (size_t i = 0u; i < writes.size(); ++i) {
        auto const &entry = plan.entries[i];
        auto const &write = writes[i];
        if (write.pNext != nullptr) {
            return false;
        }
        auto payload = descriptor_update_payload(write.descriptorType, write);
        if (payload.element_size == 0u || payload.source == nullptr) {
            return false;
        }
        if (entry.dstBinding != write.dstBinding ||
            entry.dstArrayElement != write.dstArrayElement ||
            entry.descriptorCount != write.descriptorCount ||
            entry.descriptorType != write.descriptorType ||
            entry.stride != payload.element_size ||
            plan.payload_sizes[i] !=
                static_cast<size_t>(payload.element_size) * write.descriptorCount) {
            return false;
        }
    }
    return true;
}

// Copies every payload into the packed block (`destination` holds
// plan.data_size bytes) in the order the plan describes.
inline void pack_descriptor_update(const DescriptorUpdatePlan &plan,
                                   luisa::span<const VkWriteDescriptorSet> writes,
                                   std::byte *destination) noexcept {
    LUISA_ASSERT(plan.entries.size() == writes.size(),
                 "Descriptor update plan covers {} writes but {} were supplied.",
                 plan.entries.size(), writes.size());
    for (size_t i = 0u; i < writes.size(); ++i) {
        auto const &entry = plan.entries[i];
        auto payload = descriptor_update_payload(entry.descriptorType, writes[i]);
        LUISA_ASSERT(payload.source != nullptr && payload.element_size != 0u,
                     "Descriptor update write {} has no template payload.", i);
        std::memcpy(destination + entry.offset, payload.source,
                    plan.payload_sizes[i]);
    }
}

// ---------------------------------------------------------------------------
// entry points and templates
// ---------------------------------------------------------------------------

// The three entry points a template update needs. All of them are core Vulkan
// 1.1, so every device this backend accepts exposes the core names; the KHR
// aliases are accepted because a driver that only implements the extension
// spelling exposes those (the fallback costs nothing).
struct DescriptorUpdateTemplateApi {
    PFN_vkCreateDescriptorUpdateTemplate create{nullptr};
    PFN_vkDestroyDescriptorUpdateTemplate destroy{nullptr};
    PFN_vkUpdateDescriptorSetWithTemplate update{nullptr};
    // The host allocation callbacks every other Vulkan object of this backend is
    // created and destroyed with (`Device::alloc_callbacks()`). The same pointer
    // is handed to vkCreateDescriptorUpdateTemplate and to
    // vkDestroyDescriptorUpdateTemplate: Vulkan requires the callbacks of a
    // destroy call to match the ones the object was created with, and this struct
    // is the only place that can guarantee it. `nullptr` means "Vulkan's default
    // allocator" and is what the host test uses.
    const VkAllocationCallbacks *callbacks{nullptr};

    [[nodiscard]] bool available() const noexcept {
        return create != nullptr && destroy != nullptr && update != nullptr;
    }
};

// Distinct write shapes cached per owner. A shader has one shape today (its
// argument list is static) plus at most the "empty argument block" variant, so
// this bound is never reached in practice; when it is, the affected dispatch
// keeps the legacy call instead of evicting a template another dispatch uses.
inline constexpr size_t descriptor_update_template_cache_capacity = 8u;

// Templates of one descriptor-set layout, keyed by write shape.
//
// The hot path (a dispatch whose shape was already used by this shader) takes no
// lock and allocates nothing: it walks the published pointers, compares the
// shape and packs the payload into a stack buffer. Only a shape that has never
// been seen enters the lock, creates the template and publishes it. A published
// template and its plan are immutable from then on, so a reader needs no lock -
// and `destroy` only runs when the owner is gone (see the file header).
class DescriptorUpdateTemplateCache {

public:
    DescriptorUpdateTemplateCache() noexcept = default;
    ~DescriptorUpdateTemplateCache() noexcept = default;
    DescriptorUpdateTemplateCache(DescriptorUpdateTemplateCache const &) = delete;
    DescriptorUpdateTemplateCache(DescriptorUpdateTemplateCache &&) = delete;
    DescriptorUpdateTemplateCache &operator=(DescriptorUpdateTemplateCache const &) = delete;
    DescriptorUpdateTemplateCache &operator=(DescriptorUpdateTemplateCache &&) = delete;

    [[nodiscard]] size_t size() const noexcept {
        return _published_count.load(std::memory_order_relaxed);
    }

    // Destroys every cached template. Called by the owner's destructor while the
    // device (and therefore the descriptor-set layout the templates name) is
    // still alive. Safe to call twice.
    void destroy(const DescriptorUpdateTemplateApi &api, VkDevice device) noexcept {
        for (auto i = 0u; i < descriptor_update_template_cache_capacity; ++i) {
            _published[i].store(nullptr, std::memory_order_relaxed);
        }
        _published_count.store(0u, std::memory_order_relaxed);
        if (api.destroy == nullptr) {
            _templates.clear();
            return;
        }
        for (auto &cached : _templates) {
            if (cached->handle != VK_NULL_HANDLE) {
                api.destroy(device, cached->handle, api.callbacks);
                cached->handle = VK_NULL_HANDLE;
            }
        }
        _templates.clear();
    }

    // Finds (or creates) the template for this write shape and applies it.
    // `created` reports whether this call performed vkCreateDescriptorUpdateTemplate.
    // `kOk` means the descriptor set was updated through the template.
    [[nodiscard]] DescriptorUpdatePlanStatus
    update(VkDevice device, const DescriptorUpdateTemplateApi &api,
           VkDescriptorSet set, VkDescriptorSetLayout layout,
           VkPipelineBindPoint bind_point,
           luisa::span<const VkWriteDescriptorSet> writes,
           bool &created) noexcept {
        created = false;
        if (!api.available() || layout == VK_NULL_HANDLE || set == VK_NULL_HANDLE) {
            return DescriptorUpdatePlanStatus::kTemplateCreationFailed;
        }
        // Fast path: every template this shader ever published, last one first
        // (a shader has one shape today, so the first probe hits).
        auto published = _published_count.load(std::memory_order_acquire);
        for (auto i = published; i > 0u; --i) {
            auto *cached = _published[i - 1u].load(std::memory_order_acquire);
            if (cached != nullptr &&
                descriptor_update_plan_matches(cached->plan, writes)) {
                return apply(api, device, set, *cached, writes);
            }
        }
        // Slow path: a shape this shader has never used.
        _mutex.lock();
        published = _published_count.load(std::memory_order_relaxed);
        for (auto i = published; i > 0u; --i) {
            auto *cached = _published[i - 1u].load(std::memory_order_relaxed);
            if (cached != nullptr &&
                descriptor_update_plan_matches(cached->plan, writes)) {
                _mutex.unlock();
                return apply(api, device, set, *cached, writes);
            }
        }
        if (published >= descriptor_update_template_cache_capacity) {
            _mutex.unlock();
            return DescriptorUpdatePlanStatus::kCacheFull;
        }
        auto owned = luisa::make_unique<CachedTemplate>();
        auto status = plan_descriptor_update(writes, owned->plan);
        if (status != DescriptorUpdatePlanStatus::kOk) {
            _mutex.unlock();
            return status;
        }
        VkDescriptorUpdateTemplateCreateInfo info{
            .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_UPDATE_TEMPLATE_CREATE_INFO,
            .pNext = nullptr,
            .flags = VkDescriptorUpdateTemplateCreateFlags{0u},
            .descriptorUpdateEntryCount =
                static_cast<uint32_t>(owned->plan.entries.size()),
            .pDescriptorUpdateEntries = owned->plan.entries.data(),
            .templateType =
                VK_DESCRIPTOR_UPDATE_TEMPLATE_TYPE_DESCRIPTOR_SET,
            // For a DESCRIPTOR_SET template Vulkan uses only the layout; the
            // three bind-point fields below are ignored by the driver and
            // carried so the create info describes the update it models.
            .descriptorSetLayout = layout,
            .pipelineBindPoint = bind_point,
            .pipelineLayout = VK_NULL_HANDLE,
            .set = 0u};
        auto result = api.create(device, &info, api.callbacks, &owned->handle);
        if (result != VK_SUCCESS || owned->handle == VK_NULL_HANDLE) {
            _mutex.unlock();
            return DescriptorUpdatePlanStatus::kTemplateCreationFailed;
        }
        auto const *plan = &owned->plan;
        auto handle = owned->handle;
        _templates.emplace_back(std::move(owned));
        auto *published_template = _templates.back().get();
        _published[published].store(published_template, std::memory_order_release);
        _published_count.store(published + 1u, std::memory_order_release);
        _mutex.unlock();
        created = true;
        // The payload of one dispatch lives on the stack of this call and is
        // consumed by the update call below; nothing points into it afterwards.
        alignas(descriptor_update_payload_alignment)
            std::byte packed[kPackedPayloadInlineSize];
        pack_descriptor_update(*plan, writes, packed);
        api.update(device, set, handle, packed);
        return DescriptorUpdatePlanStatus::kOk;
    }

private:
    struct CachedTemplate {
        VkDescriptorUpdateTemplate handle{VK_NULL_HANDLE};
        DescriptorUpdatePlan plan;
    };
    // The payload of one update is a handful of 24-byte structures; a stack
    // buffer avoids any allocation on the hot path. Its size is the planner's
    // limit, so an accepted plan always fits.
    static constexpr size_t kPackedPayloadInlineSize =
        descriptor_update_max_payload_size;
    static_assert(kPackedPayloadInlineSize == descriptor_update_max_payload_size);
    // Templates this shader may publish (see descriptor_update_template_cache_capacity).
    std::array<std::atomic<CachedTemplate *>,
               descriptor_update_template_cache_capacity>
        _published{};
    std::atomic<uint32_t> _published_count{0u};
    mutable luisa::spin_mutex _mutex;
    luisa::vector<luisa::unique_ptr<CachedTemplate>> _templates;

    [[nodiscard]] static DescriptorUpdatePlanStatus
    apply(const DescriptorUpdateTemplateApi &api, VkDevice device,
          VkDescriptorSet set, const CachedTemplate &cached,
          luisa::span<const VkWriteDescriptorSet> writes) noexcept {
        LUISA_ASSERT(cached.plan.data_size <= kPackedPayloadInlineSize,
                     "A cached descriptor update plan exceeds the stack payload "
                     "block ({} > {} bytes).",
                     cached.plan.data_size, kPackedPayloadInlineSize);
        // A plain stack array, deliberately not a container: the hot path must
        // not pay for vector bookkeeping, and the planner bounds data_size.
        alignas(descriptor_update_payload_alignment)
            std::byte packed[kPackedPayloadInlineSize];
        pack_descriptor_update(cached.plan, writes, packed);
        api.update(device, set, cached.handle, packed);
        return DescriptorUpdatePlanStatus::kOk;
    }
};

// One set-0 update, from the caller's point of view.
struct DescriptorSetUpdateRequest {
    VkDevice device{VK_NULL_HANDLE};
    VkDescriptorSet set{VK_NULL_HANDLE};
    VkDescriptorSetLayout layout{VK_NULL_HANDLE};
    VkPipelineBindPoint bind_point{VK_PIPELINE_BIND_POINT_COMPUTE};
    DescriptorUpdateTemplateApi api{};
    // False issues the pre-P2.2 call unconditionally (config switch, a device
    // without the entry points, or a path that intentionally stays legacy).
    bool allow_templates{false};
};

struct DescriptorSetUpdateResult {
    // True when vkUpdateDescriptorSetWithTemplate performed the update.
    bool used_template{false};
    // True when this call also created the template it used.
    bool created_template{false};
    // Why the template path was not taken (kOk when it was).
    DescriptorUpdatePlanStatus status{DescriptorUpdatePlanStatus::kOk};
};

// The single funnel for one descriptor-set update. Either the template path or
// exactly one vkUpdateDescriptorSets call with the same writes - never both, and
// never a partial write.
[[nodiscard]] inline DescriptorSetUpdateResult
update_descriptor_set(const DescriptorSetUpdateRequest &request,
                      DescriptorUpdateTemplateCache &cache,
                      luisa::span<const VkWriteDescriptorSet> writes) noexcept {
    DescriptorSetUpdateResult result;
    if (writes.empty()) {
        // Nothing was ever written for an empty list, on either path; the
        // caller records no update call either (the pre-P2.2 code skipped the
        // native call in this case too).
        result.status = DescriptorUpdatePlanStatus::kNoWrites;
        return result;
    }
    LUISA_ASSERT(request.device != VK_NULL_HANDLE,
                 "Vulkan descriptor update without a device.");
    auto const templates_allowed =
        request.allow_templates && request.api.available();
    if (templates_allowed) {
        result.status = cache.update(request.device, request.api, request.set,
                                     request.layout, request.bind_point,
                                     writes, result.created_template);
        if (result.status == DescriptorUpdatePlanStatus::kOk) {
            result.used_template = true;
            if (result.created_template) {
                submit_profile::record_descriptor_update_template_create();
            }
            submit_profile::record_descriptor_set_update_template();
            return result;
        }
    }
    LUISA_ASSERT(writes.size() <= std::numeric_limits<uint32_t>::max(),
                 "Vulkan descriptor-write count is not representable.");
    vkUpdateDescriptorSets(request.device,
                           static_cast<uint32_t>(writes.size()),
                           writes.data(), 0u, nullptr);
    // `vk_profile_descriptor_set_updates` counts exactly these calls.
    submit_profile::record_descriptor_set_update();
    if (templates_allowed) {
        submit_profile::record_descriptor_update_template_fallback(
            descriptor_update_plan_status_name(result.status));
    }
    return result;
}

}// namespace lc::vk::detail