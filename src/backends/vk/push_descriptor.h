// Set-0 descriptor binding through VK_KHR_push_descriptor (plan P2.3).
//
// Why this exists
// ---------------
// Every compute/RT/raster dispatch of this backend builds its local argument set
// (set 0) fresh, so the classic path pays three host calls per dispatch: one
// vkAllocateDescriptorSets, one vkUpdateDescriptorSets (plan P2.2 can move the
// second one to a cached template, which measured slower on this driver) and one
// vkCmdBindDescriptorSets. VK_KHR_push_descriptor replaces all three with one
// vkCmdPushDescriptorSetKHR that carries exactly the same VkWriteDescriptorSet
// list: no set is allocated, nothing is updated, and set 0 never appears in a
// bind. The global sets (the immutable-sampler heap and up to three bindless
// heaps) are then bound at their real first set, which also means they no longer
// have to be re-bound just because the local set changed (plan P1.2's tracker
// keeps them, see bind_state.h).
//
// What may be pushed
// ------------------
// The decision is taken once per shader at creation (shader.cpp), never per
// dispatch, because the descriptor-set *layout* has to be created with
// VK_DESCRIPTOR_SET_LAYOUT_CREATE_PUSH_DESCRIPTOR_BIT_KHR: a push descriptor set
// is a property of the pipeline layout, so switching mid-frame is impossible by
// construction. `push_descriptor_eligibility` is the pure decision:
//   * the device must have negotiated VK_KHR_push_descriptor *and* resolved
//     vkCmdPushDescriptorSetKHR *and* reported a non-zero maxPushDescriptors;
//   * set 0 must have at least one binding (vkCmdPushDescriptorSetKHR requires
//     descriptorWriteCount > 0, so a layout with no binding cannot be pushed);
//   * the total number of descriptor elements of set 0 must be within
//     VkPhysicalDevicePushDescriptorPropertiesKHR::maxPushDescriptors
//     (VUID-VkDescriptorSetLayoutCreateInfo-flags-00281);
//   * no binding may carry a descriptor-binding flag (update-after-bind,
//     partially-bound, variable count): set 0 is never bindless in this backend,
//     and push descriptors have no "update after bind" notion, so such a shape
//     stays classic instead of being approximated;
//   * no binding may be a dynamic buffer or an inline uniform block
//     (VUID-VkDescriptorSetLayoutCreateInfo-flags-00280 / -02208).
// Anything else keeps the classic layout *from creation*: no dispatch ever
// changes path, and a shader that exceeds the push limit simply allocates and
// binds its set 0 like it did before.
//
// What may be pushed *now*
// ------------------------
// `push_descriptor_set` is the single funnel that records the call. It re-checks
// the write list against the Vulkan validity rules it can see:
//   * every write must carry a descriptor type push descriptors accept;
//   * descriptorCount must be non-zero;
//   * a pNext chain is legal for a small, spec-defined set of blocks: the
//     acceleration-structure write of a hardware ray-tracing shader
//     (VkWriteDescriptorSetAccelerationStructureKHR) is pushed like any other
//     write, and its chain is passed through unchanged;
//   * the sum of descriptorCount of the writes must stay within
//     maxPushDescriptors.
// A refused call records nothing and reports why; the caller treats that as a
// hard error (fail closed) instead of silently falling back mid-frame.
//
// Lifetimes
// ---------
// vkCmdPushDescriptorSetKHR "consumes" the writes when it is recorded: it copies
// the descriptor payload into the command buffer, and the buffers / image views /
// acceleration structures those payloads name must stay alive until the command
// buffer completes - exactly the contract the bound descriptor sets had (the
// command buffer state owns the image views, see CommandBufferState::img_views).
// The writes themselves (including the pNext chain) only have to be valid for the
// duration of the call, which the caller guarantees by building them in the
// command buffer's per-recording descriptor arena.
//
// This header is deliberately free of a Vulkan *device*: the decision is a pure
// function and the recorder takes the entry point as a parameter, so both are
// covered exhaustively on the host (src/tests/unit/ext/test_vk_push_descriptor.cpp).
#pragma once

#include <cstdint>
#include <limits>

#include <volk.h>

#include <luisa/core/logging.h>
#include <luisa/core/stl/memory.h>
#include <luisa/core/stl/vector.h>

#include "submit_profile.h"

namespace lc::vk::detail {

// Why a shader's set 0 does (not) use push descriptors. `kOk` is the only value
// that permits the push layout; every other value keeps the classic set-0 layout
// and the classic allocate/update/bind sequence for that shader.
enum class PushDescriptorStatus : uint32_t {
    kOk,
    kDeviceUnsupported = 1u,
    kEmptySet,
    kDescriptorCountExceedsLimit,
    kBindingFlagsNotPushable,
    kDynamicOrInlineUniformBinding,
    // Record-time refusals (never a creation-time decision).
    kNoWrites,
    kUnsupportedDescriptorType,
    kEmptyDescriptorCount,
    kUnsupportedWriteChain,
    kWriteChainMismatch,
    kPushCountExceedsLimit,
    kOverlappingWrites,
};

[[nodiscard]] inline const char *
push_descriptor_status_name(PushDescriptorStatus status) noexcept {
    switch (status) {
        case PushDescriptorStatus::kOk: return "ok";
        case PushDescriptorStatus::kDeviceUnsupported: return "device-unsupported";
        case PushDescriptorStatus::kEmptySet: return "empty-set";
        case PushDescriptorStatus::kDescriptorCountExceedsLimit: return "descriptor-count-exceeds-limit";
        case PushDescriptorStatus::kBindingFlagsNotPushable: return "binding-flags-not-pushable";
        case PushDescriptorStatus::kDynamicOrInlineUniformBinding: return "dynamic-or-inline-uniform-binding";
        case PushDescriptorStatus::kNoWrites: return "no-writes";
        case PushDescriptorStatus::kUnsupportedDescriptorType: return "unsupported-descriptor-type";
        case PushDescriptorStatus::kEmptyDescriptorCount: return "empty-descriptor-count";
        case PushDescriptorStatus::kUnsupportedWriteChain: return "unsupported-write-chain";
        case PushDescriptorStatus::kWriteChainMismatch: return "write-chain-mismatch";
        case PushDescriptorStatus::kPushCountExceedsLimit: return "push-count-exceeds-limit";
        case PushDescriptorStatus::kOverlappingWrites: return "overlapping-writes";
    }
    return "unknown";
}

// The resolved entry points of the push path. `available()` is false whenever the
// device could not negotiate VK_KHR_push_descriptor or the loader did not resolve
// the entry point, so a caller may pass them on unconditionally.
struct PushDescriptorApi {
    // KHR entry point (or the core 1.4 spelling when a driver drops the name but
    // exposes the core command). Never called unless non-null *and* the shader was
    // created with a push descriptor set layout.
    PFN_vkCmdPushDescriptorSetKHR cmd{nullptr};

    [[nodiscard]] bool available() const noexcept { return cmd != nullptr; }
};

// Everything the creation-time decision depends on. All fields are properties of
// the device, or of the shader's own set-0 layout - never of a measurement or a
// counter, so the decision is reproducible and cannot flip at runtime.
struct PushDescriptorEligibility {
    // Device negotiates the extension, resolved the entry point and reports a
    // non-zero maxPushDescriptors.
    bool device_capable{false};
    uint32_t max_push_descriptors{0u};
    // Bindings of set 0 and the sum of their descriptorCount.
    uint32_t set_binding_count{0u};
    uint64_t set_descriptor_count{0u};
    // Any VkDescriptorBindingFlags on a set-0 binding.
    bool has_binding_flags{false};
    // Any dynamic buffer / inline uniform block binding in set 0.
    bool has_dynamic_or_inline_uniform_binding{false};
};

// One decision per shader, taken at creation.
[[nodiscard]] inline PushDescriptorStatus
push_descriptor_eligibility(const PushDescriptorEligibility &info) noexcept {
    if (!info.device_capable || info.max_push_descriptors == 0u) {
        return PushDescriptorStatus::kDeviceUnsupported;
    }
    if (info.set_binding_count == 0u || info.set_descriptor_count == 0u) {
        return PushDescriptorStatus::kEmptySet;
    }
    if (info.set_descriptor_count > info.max_push_descriptors) {
        return PushDescriptorStatus::kDescriptorCountExceedsLimit;
    }
    if (info.has_binding_flags) {
        return PushDescriptorStatus::kBindingFlagsNotPushable;
    }
    if (info.has_dynamic_or_inline_uniform_binding) {
        return PushDescriptorStatus::kDynamicOrInlineUniformBinding;
    }
    return PushDescriptorStatus::kOk;
}

// Whether push descriptors can express this descriptor type at all. The list is
// the spec-defined set of types vkCmdPushDescriptorSetKHR accepts; dynamic
// buffers and inline uniform blocks are *not* in it, which is why a layout that
// contains them is refused at creation instead of at record time.
[[nodiscard]] inline bool
push_descriptor_type_supported(VkDescriptorType type) noexcept {
    switch (type) {
        case VK_DESCRIPTOR_TYPE_SAMPLER:
        case VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER:
        case VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE:
        case VK_DESCRIPTOR_TYPE_STORAGE_IMAGE:
        case VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER:
        case VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER:
        case VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER:
        case VK_DESCRIPTOR_TYPE_STORAGE_BUFFER:
        case VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT:
        case VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR:
            return true;
        default:
            return false;
    }
}

namespace push_descriptor_detail {

// A pNext chain is passed through unchanged, so the only chains this funnel
// accepts are the ones whose structure Vulkan defines for a descriptor write and
// whose contents it reads immediately: the acceleration-structure block (one
// VkAccelerationStructureKHR) and the inline-uniform-block block (which a push
// layout cannot contain anyway, see push_descriptor_eligibility). Everything else
// is refused instead of being handed to the driver with an unknown sType.
[[nodiscard]] inline bool
write_chain_supported(const VkWriteDescriptorSet &write) noexcept {
    auto const *node = write.pNext;
    if (node == nullptr) { return true; }
    auto const *base = static_cast<const VkBaseInStructure *>(node);
    switch (base->sType) {
        case VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET_ACCELERATION_STRUCTURE_KHR:
            // The write must actually describe an acceleration structure, and the
            // chain must end there (Vulkan reads exactly this one structure).
            return write.descriptorType == VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR &&
                   base->pNext == nullptr;
        default:
            return false;
    }
}

// Two writes that address the same descriptor with overlapping element ranges make
// a push list ambiguous: `vkUpdateDescriptorSets` defines "last write wins" for a
// list, a push descriptor bank does not. The encoder never produces such a list -
// each set-0 binding is written once - but a regression that forgets to consume a
// write list did (plan P2.3 review: it made the driver crash), so the list is
// rejected here instead of being handed over. Writes to *disjoint* array elements
// of one binding stay legal.
[[nodiscard]] inline bool writes_overlap(const VkWriteDescriptorSet &a,
                                         const VkWriteDescriptorSet &b) noexcept {
    if (a.dstBinding != b.dstBinding) { return false; }
    auto const a_end = uint64_t{a.dstArrayElement} + a.descriptorCount;
    auto const b_end = uint64_t{b.dstArrayElement} + b.descriptorCount;
    return a.dstArrayElement < b_end && b.dstArrayElement < a_end;
}

}// namespace push_descriptor_detail

// What one push call needs. `allow_push` is false whenever the shader was created
// with a classic set-0 layout (device without the extension, unsupported shape,
// an internal/native/indirect-preparation path) - the caller then allocates and
// binds its set 0 exactly as before.
struct PushDescriptorRecordRequest {
    VkCommandBuffer command_buffer{VK_NULL_HANDLE};
    VkPipelineBindPoint bind_point{VK_PIPELINE_BIND_POINT_COMPUTE};
    VkPipelineLayout layout{VK_NULL_HANDLE};
    // The set number of the push layout inside `layout`. Always 0 for this
    // backend (set 0 is the only push layout), asserted by the caller.
    uint32_t set{0u};
    uint32_t max_push_descriptors{0u};
    PushDescriptorApi api{};
    bool allow_push{false};
};

struct PushDescriptorRecordResult {
    // True when vkCmdPushDescriptorSetKHR performed the binding.
    bool pushed{false};
    // Set number that was pushed (0) - the caller binds the global sets from
    // `set + 1` on.
    uint32_t set{0u};
    PushDescriptorStatus status{PushDescriptorStatus::kDeviceUnsupported};
};

// The single funnel for one set-0 push. Records exactly one
// vkCmdPushDescriptorSetKHR with the caller's own write list, or nothing at all.
[[nodiscard]] inline PushDescriptorRecordResult
push_descriptor_set(const PushDescriptorRecordRequest &request,
                    luisa::span<const VkWriteDescriptorSet> writes) noexcept {
    PushDescriptorRecordResult result;
    result.set = request.set;
    if (!request.allow_push || !request.api.available()) {
        result.status = PushDescriptorStatus::kDeviceUnsupported;
        return result;
    }
    if (writes.empty()) {
        // vkCmdPushDescriptorSetKHR requires descriptorWriteCount > 0. A push
        // shader always has set-0 bindings (push_descriptor_eligibility rejects an
        // empty set), so an empty list here is a planning error, not a fallback.
        result.status = PushDescriptorStatus::kNoWrites;
        return result;
    }
    LUISA_ASSERT(request.command_buffer != VK_NULL_HANDLE &&
                     request.layout != VK_NULL_HANDLE,
                 "Vulkan push descriptor call without a command buffer or layout.");
    uint64_t descriptor_count = 0u;
    for (auto &&write : writes) {
        if (!push_descriptor_type_supported(write.descriptorType)) {
            result.status = PushDescriptorStatus::kUnsupportedDescriptorType;
            return result;
        }
        if (write.descriptorCount == 0u) {
            result.status = PushDescriptorStatus::kEmptyDescriptorCount;
            return result;
        }
        if (!push_descriptor_detail::write_chain_supported(write)) {
            auto const *base = static_cast<const VkBaseInStructure *>(write.pNext);
            result.status = base != nullptr &&
                                    base->sType == VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET_ACCELERATION_STRUCTURE_KHR ?
                                PushDescriptorStatus::kWriteChainMismatch :
                                PushDescriptorStatus::kUnsupportedWriteChain;
            return result;
        }
        descriptor_count += write.descriptorCount;
    }
    if (request.max_push_descriptors != 0u &&
        descriptor_count > request.max_push_descriptors) {
        result.status = PushDescriptorStatus::kPushCountExceedsLimit;
        return result;
    }
    // A set-0 write list writes every binding once; two overlapping writes to the
    // same binding would leave the pushed content undefined (see writes_overlap).
    // The quadratic scan is over the <= 32 entries a set 0 can have and runs only
    // for multi-write lists, so it costs nothing on the one-descriptor path.
    if (writes.size() > 1u) {
        for (auto i = size_t{0u}; i < writes.size(); ++i) {
            for (auto j = i + 1u; j < writes.size(); ++j) {
                if (push_descriptor_detail::writes_overlap(writes[i], writes[j])) {
                    result.status = PushDescriptorStatus::kOverlappingWrites;
                    return result;
                }
            }
        }
    }
    LUISA_ASSERT(writes.size() <= uint64_t{std::numeric_limits<uint32_t>::max()},
                 "Vulkan push descriptor write count is not representable.");
    request.api.cmd(request.command_buffer, request.bind_point, request.layout,
                    request.set, static_cast<uint32_t>(writes.size()),
                    writes.data());
    // Counter-only: one recorded vkCmdPushDescriptorSetKHR, never an intention.
    submit_profile::record_push_descriptor_call();
    result.pushed = true;
    result.status = PushDescriptorStatus::kOk;
    return result;
}

}// namespace lc::vk::detail