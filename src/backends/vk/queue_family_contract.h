#pragma once

#include <array>
#include <cstdint>

#include <vulkan/vulkan_core.h>

#include <luisa/core/stl/memory.h>

namespace lc::vk::detail {

enum class QueueFamilyContractStatus : uint8_t {
    SUCCESS,
    MISSING_INDEX,
    INDEX_OUT_OF_RANGE,
    EMPTY_FAMILY,
    MISSING_CAPABILITY
};

enum class ExternalDeviceAncestryStatus : uint8_t {
    SUCCESS,
    MISSING_INSTANCE,
    MISSING_PHYSICAL_DEVICE,
    MISSING_LOGICAL_DEVICE,
    QUEUE_WITHOUT_LOGICAL_DEVICE
};

enum class ExternalQueueHandleStatus : uint8_t {
    SUCCESS,
    MISSING_HANDLE
};

enum class SparseBindingQueueStatus : uint8_t {
    SUCCESS,
    MISSING_INDEX,
    INDEX_OUT_OF_RANGE,
    EMPTY_FAMILY,
    MISSING_CAPABILITY
};

struct ExternalDeviceAncestryResult {
    ExternalDeviceAncestryStatus status;

    [[nodiscard]] constexpr explicit operator bool() const noexcept {
        return status == ExternalDeviceAncestryStatus::SUCCESS;
    }
};

struct ExternalQueueHandleResult {
    ExternalQueueHandleStatus status;
    uint32_t role;

    [[nodiscard]] constexpr explicit operator bool() const noexcept {
        return status == ExternalQueueHandleStatus::SUCCESS;
    }
};

struct QueueLockPlan {
    // graphics, compute, copy, sparse-binding
    std::array<uint8_t, 4u> lock_indices{};
    uint8_t lock_count{};
};

struct QueueFamilySharingPlan {
    std::array<uint32_t, 3u> family_indices{};
    uint32_t family_count{};

    [[nodiscard]] constexpr bool concurrent() const noexcept {
        return family_count > 1u;
    }

    [[nodiscard]] constexpr VkSharingMode sharing_mode() const noexcept {
        return concurrent() ? VK_SHARING_MODE_CONCURRENT :
                              VK_SHARING_MODE_EXCLUSIVE;
    }

    [[nodiscard]] constexpr uint32_t create_info_family_count() const noexcept {
        return concurrent() ? family_count : 0u;
    }

    [[nodiscard]] constexpr const uint32_t *create_info_family_indices() const noexcept {
        return concurrent() ? family_indices.data() : nullptr;
    }
};

struct SparseBindingQueueResult {
    SparseBindingQueueStatus status;
    uint32_t family_index;
    VkQueueFlags available_flags;

    [[nodiscard]] constexpr explicit operator bool() const noexcept {
        return status == SparseBindingQueueStatus::SUCCESS;
    }
};

struct QueueFamilyContractResult {
    QueueFamilyContractStatus status;
    uint32_t role;
    uint32_t family_index;
    VkQueueFlags required_flags;
    VkQueueFlags available_flags;

    [[nodiscard]] constexpr explicit operator bool() const noexcept {
        return status == QueueFamilyContractStatus::SUCCESS;
    }
};

[[nodiscard]] constexpr const char *queue_family_role_name(
    uint32_t role) noexcept {
    switch (role) {
        case 0u: return "graphics";
        case 1u: return "compute";
        case 2u: return "copy";
        case 3u: return "sparse-binding";
        default: return "unknown";
    }
}

[[nodiscard]] constexpr const char *queue_family_contract_status_name(
    QueueFamilyContractStatus status) noexcept {
    switch (status) {
        case QueueFamilyContractStatus::SUCCESS: return "success";
        case QueueFamilyContractStatus::MISSING_INDEX: return "missing family index";
        case QueueFamilyContractStatus::INDEX_OUT_OF_RANGE: return "family index out of range";
        case QueueFamilyContractStatus::EMPTY_FAMILY: return "family exposes no queues";
        case QueueFamilyContractStatus::MISSING_CAPABILITY: return "family lacks required queue capabilities";
    }
    return "unknown";
}

[[nodiscard]] constexpr const char *external_device_ancestry_status_name(
    ExternalDeviceAncestryStatus status) noexcept {
    switch (status) {
        case ExternalDeviceAncestryStatus::SUCCESS: return "success";
        case ExternalDeviceAncestryStatus::MISSING_INSTANCE: return "logical device import is missing its instance";
        case ExternalDeviceAncestryStatus::MISSING_PHYSICAL_DEVICE: return "logical device import is missing its physical device";
        case ExternalDeviceAncestryStatus::MISSING_LOGICAL_DEVICE: return "physical device import is missing its logical device";
        case ExternalDeviceAncestryStatus::QUEUE_WITHOUT_LOGICAL_DEVICE: return "queue metadata was supplied without a logical device";
    }
    return "unknown external-device ancestry error";
}

[[nodiscard]] constexpr const char *external_queue_handle_status_name(
    ExternalQueueHandleStatus status) noexcept {
    switch (status) {
        case ExternalQueueHandleStatus::SUCCESS: return "success";
        case ExternalQueueHandleStatus::MISSING_HANDLE: return "missing queue handle";
    }
    return "unknown external-queue handle error";
}

[[nodiscard]] constexpr const char *sparse_binding_queue_status_name(
    SparseBindingQueueStatus status) noexcept {
    switch (status) {
        case SparseBindingQueueStatus::SUCCESS: return "success";
        case SparseBindingQueueStatus::MISSING_INDEX: return "missing family index";
        case SparseBindingQueueStatus::INDEX_OUT_OF_RANGE: return "family index out of range";
        case SparseBindingQueueStatus::EMPTY_FAMILY: return "family exposes no queues";
        case SparseBindingQueueStatus::MISSING_CAPABILITY: return "family lacks sparse-binding capability";
    }
    return "unknown sparse-binding queue error";
}

// Supplying only an instance is supported: the backend can select a physical
// device and create its own logical device under that instance. Once either a
// physical or logical device is imported, however, the complete ancestry is
// required. Queue handles/families never make sense without a logical device.
[[nodiscard]] constexpr ExternalDeviceAncestryResult
validate_external_device_ancestry(
    bool has_instance, bool has_physical_device,
    bool has_logical_device, bool has_queue_metadata) noexcept {
    if (!has_logical_device && has_queue_metadata) {
        return {ExternalDeviceAncestryStatus::QUEUE_WITHOUT_LOGICAL_DEVICE};
    }
    if (!has_physical_device && !has_logical_device) {
        return {ExternalDeviceAncestryStatus::SUCCESS};
    }
    if (!has_instance) {
        return {ExternalDeviceAncestryStatus::MISSING_INSTANCE};
    }
    if (!has_physical_device) {
        return {ExternalDeviceAncestryStatus::MISSING_PHYSICAL_DEVICE};
    }
    if (!has_logical_device) {
        return {ExternalDeviceAncestryStatus::MISSING_LOGICAL_DEVICE};
    }
    return {ExternalDeviceAncestryStatus::SUCCESS};
}

// Queue handles are opaque: the backend cannot prove that queue index zero
// was requested when the imported logical device was created. Require the
// actual handles instead of guessing with vkGetDeviceQueue.
[[nodiscard]] constexpr ExternalQueueHandleResult
validate_external_queue_handles(
    bool imported_device,
    std::array<bool, 3u> has_queue_handles) noexcept {
    if (!imported_device) {
        return {ExternalQueueHandleStatus::SUCCESS, 0u};
    }
    for (auto role = 0u; role < has_queue_handles.size(); ++role) {
        if (!has_queue_handles[role]) {
            return {ExternalQueueHandleStatus::MISSING_HANDLE, role};
        }
    }
    return {ExternalQueueHandleStatus::SUCCESS, 0u};
}

// Vulkan queue submission is externally synchronized per VkQueue, not per
// Luisa stream role. Map equal native queue identities to one host mutex while
// preserving independent locks for genuinely distinct queues.
[[nodiscard]] constexpr QueueLockPlan plan_queue_locks(
    std::array<uint64_t, 4u> queue_identities) noexcept {
    auto plan = QueueLockPlan{};
    for (auto role = 0u; role < queue_identities.size(); ++role) {
        auto found = false;
        for (auto prior = 0u; prior < role; ++prior) {
            if (queue_identities[prior] == queue_identities[role]) {
                plan.lock_indices[role] = plan.lock_indices[prior];
                found = true;
                break;
            }
        }
        if (!found) {
            plan.lock_indices[role] = plan.lock_count++;
        }
    }
    return plan;
}

// Prefer a queue family dedicated to sparse/transfer work, but fall back to
// any sparse-capable family. This keeps sparse binding available on devices
// that expose it only on a graphics/compute family while avoiding needless
// serialization with command submission when a dedicated family exists.
[[nodiscard]] constexpr SparseBindingQueueResult
select_sparse_binding_queue_family(
    luisa::span<const VkQueueFamilyProperties> families) noexcept {
    auto fallback_index = VK_QUEUE_FAMILY_IGNORED;
    auto fallback_flags = VkQueueFlags{};
    for (auto index = 0u; index < families.size(); ++index) {
        auto const &family = families[index];
        if (family.queueCount == 0u ||
            (family.queueFlags & VK_QUEUE_SPARSE_BINDING_BIT) == 0u) {
            continue;
        }
        if (fallback_index == VK_QUEUE_FAMILY_IGNORED) {
            fallback_index = index;
            fallback_flags = family.queueFlags;
        }
        if ((family.queueFlags &
             (VK_QUEUE_GRAPHICS_BIT | VK_QUEUE_COMPUTE_BIT)) == 0u) {
            return {
                SparseBindingQueueStatus::SUCCESS,
                index, family.queueFlags};
        }
    }
    if (fallback_index != VK_QUEUE_FAMILY_IGNORED) {
        return {
            SparseBindingQueueStatus::SUCCESS,
            fallback_index, fallback_flags};
    }
    return {
        SparseBindingQueueStatus::MISSING_CAPABILITY,
        VK_QUEUE_FAMILY_IGNORED, 0u};
}

// Backend-owned resources may be used by any stream role. Preserve the role
// order while deduplicating equal families so Vk*CreateInfo receives exactly
// the participating graphics/compute/copy families.
[[nodiscard]] constexpr QueueFamilySharingPlan plan_queue_family_sharing(
    std::array<uint32_t, 3u> family_indices) noexcept {
    auto plan = QueueFamilySharingPlan{};
    for (auto family : family_indices) {
        auto found = false;
        for (auto i = 0u; i < plan.family_count; ++i) {
            if (plan.family_indices[i] == family) {
                found = true;
                break;
            }
        }
        if (!found) {
            plan.family_indices[plan.family_count++] = family;
        }
    }
    return plan;
}

// vkQueueBindSparse is valid only on a queue whose family advertises
// VK_QUEUE_SPARSE_BINDING_BIT. This is independent of whether the resource is
// concurrently shared with that family.
[[nodiscard]] constexpr SparseBindingQueueResult
validate_sparse_binding_queue_family(
    uint32_t family_index,
    luisa::span<const VkQueueFamilyProperties> families) noexcept {
    if (family_index == VK_QUEUE_FAMILY_IGNORED) {
        return {SparseBindingQueueStatus::MISSING_INDEX, family_index, 0u};
    }
    if (family_index >= families.size()) {
        return {SparseBindingQueueStatus::INDEX_OUT_OF_RANGE, family_index, 0u};
    }
    auto family = families[family_index];
    if (family.queueCount == 0u) {
        return {SparseBindingQueueStatus::EMPTY_FAMILY,
                family_index, family.queueFlags};
    }
    if ((family.queueFlags & VK_QUEUE_SPARSE_BINDING_BIT) == 0u) {
        return {SparseBindingQueueStatus::MISSING_CAPABILITY,
                family_index, family.queueFlags};
    }
    return {SparseBindingQueueStatus::SUCCESS,
            family_index, family.queueFlags};
}

// A GRAPHICS stream records both raster and compute commands, so its family
// must support both capabilities. The imported VkDevice owns queue creation;
// the backend therefore needs the caller's exact family indices instead of
// guessing them again from physical-device preferences.
[[nodiscard]] constexpr QueueFamilyContractResult
validate_external_queue_family_contract(
    bool imported_device,
    std::array<uint32_t, 3u> family_indices,
    luisa::span<const VkQueueFamilyProperties> families) noexcept {
    if (!imported_device) {
        return QueueFamilyContractResult{
            QueueFamilyContractStatus::SUCCESS, 0u, 0u, 0u, 0u};
    }
    constexpr std::array required_flags{
        static_cast<VkQueueFlags>(
            VK_QUEUE_GRAPHICS_BIT | VK_QUEUE_COMPUTE_BIT),
        static_cast<VkQueueFlags>(VK_QUEUE_COMPUTE_BIT),
        static_cast<VkQueueFlags>(VK_QUEUE_TRANSFER_BIT)};
    for (auto role = 0u; role < family_indices.size(); ++role) {
        auto index = family_indices[role];
        if (index == VK_QUEUE_FAMILY_IGNORED) {
            return QueueFamilyContractResult{
                QueueFamilyContractStatus::MISSING_INDEX,
                role, index, required_flags[role], 0u};
        }
        if (index >= families.size()) {
            return QueueFamilyContractResult{
                QueueFamilyContractStatus::INDEX_OUT_OF_RANGE,
                role, index, required_flags[role], 0u};
        }
        auto family = families[index];
        if (family.queueCount == 0u) {
            return QueueFamilyContractResult{
                QueueFamilyContractStatus::EMPTY_FAMILY,
                role, index, required_flags[role], family.queueFlags};
        }
        // Graphics and compute queues implicitly support transfer operations
        // even when VK_QUEUE_TRANSFER_BIT is not reported explicitly.
        auto supports_role = role == 2u ?
                                 (family.queueFlags &
                                  (VK_QUEUE_TRANSFER_BIT |
                                   VK_QUEUE_GRAPHICS_BIT |
                                   VK_QUEUE_COMPUTE_BIT)) != 0u :
                                 (family.queueFlags & required_flags[role]) ==
                                     required_flags[role];
        if (!supports_role) {
            return QueueFamilyContractResult{
                QueueFamilyContractStatus::MISSING_CAPABILITY,
                role, index, required_flags[role], family.queueFlags};
        }
    }
    return QueueFamilyContractResult{
        QueueFamilyContractStatus::SUCCESS, 0u,
        family_indices[0], required_flags[0],
        families[family_indices[0]].queueFlags};
}

}// namespace lc::vk::detail
