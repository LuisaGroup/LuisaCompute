#pragma once

#include <cstddef>
#include <cstdint>

#include <vulkan/vulkan_core.h>

#include <luisa/backends/ext/vk_extension_ext.h>
#include <luisa/core/stl/optional.h>
#include <luisa/core/stl/string.h>
#include <luisa/core/stl/format.h>

// Core Vulkan limits and the SM-count reporting contract (plan P6.1 / P6.2).
//
// One source of truth: a Device fills exactly one VulkanCapabilityStatus at
// creation, and both `Device::query` and
// `VulkanExtensionStatusExt::capability_status()` answer from that copy. The
// query strings and the record therefore cannot disagree - there is no second
// mapping to keep in sync.
//
// Everything here is device-independent value logic (the Vulkan structures
// involved are plain aggregates), so the name mapping, the unknown-value
// semantics, the SM-count contract and the 64-bit formatting are covered on the
// host by src/tests/unit/ext/test_vk_device_capability.cpp; the runtime side
// (that the values are this device's real ones) is covered by
// src/tests/unit/runtime/test_vk_capability_status.cpp.
namespace lc::vk::detail {

using luisa::compute::VulkanCapabilityStatus;
using luisa::compute::VulkanSmCountSource;

// The 14 core limits plan P6.1 exposes. One enumerator per public query key;
// `core_limit_name()` is the authoritative spelling, so a host test can pin the
// key set without a device.
enum class CoreLimit : uint32_t {
    MAX_MEMORY_ALLOCATION_COUNT,
    MIN_SUBGROUP_SIZE,
    MAX_SUBGROUP_SIZE,
    SUBGROUP_OPS,
    MAX_COMPUTE_WORKGROUP_COUNT_X,
    MAX_COMPUTE_WORKGROUP_COUNT_Y,
    MAX_COMPUTE_WORKGROUP_COUNT_Z,
    MAX_COMPUTE_WORKGROUP_SIZE_X,
    MAX_COMPUTE_WORKGROUP_SIZE_Y,
    MAX_COMPUTE_WORKGROUP_SIZE_Z,
    MAX_COMPUTE_SHARED_MEMORY_BYTES,
    NON_COHERENT_ATOM_SIZE,
    MIN_STORAGE_BUFFER_OFFSET_ALIGNMENT,
    MAX_STORAGE_BUFFER_RANGE,
    COUNT
};

inline constexpr size_t core_limit_count = static_cast<size_t>(CoreLimit::COUNT);
static_assert(core_limit_count == 14u,
              "plan P6.1 exposes exactly 14 core limit keys; a new key needs a "
              "new enumerator and an updated P6.1 test, not a silently growing set");

[[nodiscard]] constexpr const char *core_limit_name(CoreLimit limit) noexcept {
    switch (limit) {
        case CoreLimit::MAX_MEMORY_ALLOCATION_COUNT: return "max_memory_allocation_count";
        case CoreLimit::MIN_SUBGROUP_SIZE: return "min_subgroup_size";
        case CoreLimit::MAX_SUBGROUP_SIZE: return "max_subgroup_size";
        case CoreLimit::SUBGROUP_OPS: return "subgroup_ops";
        case CoreLimit::MAX_COMPUTE_WORKGROUP_COUNT_X: return "max_compute_workgroup_count_x";
        case CoreLimit::MAX_COMPUTE_WORKGROUP_COUNT_Y: return "max_compute_workgroup_count_y";
        case CoreLimit::MAX_COMPUTE_WORKGROUP_COUNT_Z: return "max_compute_workgroup_count_z";
        case CoreLimit::MAX_COMPUTE_WORKGROUP_SIZE_X: return "max_compute_workgroup_size_x";
        case CoreLimit::MAX_COMPUTE_WORKGROUP_SIZE_Y: return "max_compute_workgroup_size_y";
        case CoreLimit::MAX_COMPUTE_WORKGROUP_SIZE_Z: return "max_compute_workgroup_size_z";
        case CoreLimit::MAX_COMPUTE_SHARED_MEMORY_BYTES: return "max_compute_shared_memory_bytes";
        case CoreLimit::NON_COHERENT_ATOM_SIZE: return "non_coherent_atom_size";
        case CoreLimit::MIN_STORAGE_BUFFER_OFFSET_ALIGNMENT: return "min_storage_buffer_offset_alignment";
        case CoreLimit::MAX_STORAGE_BUFFER_RANGE: return "max_storage_buffer_range";
        case CoreLimit::COUNT: break;
    }
    return "unknown_core_limit";
}

// Resolves one of the 14 public keys. `luisa::nullopt` for every other name, so
// an unknown property keeps falling through to the backend's other answers
// instead of being reported as "no query for this limit".
[[nodiscard]] inline luisa::optional<CoreLimit>
core_limit_from_name(luisa::string_view name) noexcept {
    for (auto i = 0u; i < core_limit_count; ++i) {
        auto limit = static_cast<CoreLimit>(i);
        if (name == luisa::string_view{core_limit_name(limit)}) {
            return limit;
        }
    }
    return luisa::nullopt;
}

// Builds the report from a physical device's own property structures. Nothing
// is derived or guessed:
//   `limits`                     VkPhysicalDeviceLimits (vkGetPhysicalDeviceProperties)
//   `subgroup_properties`        VkPhysicalDeviceSubgroupProperties, or nullptr when
//                                the device could not be asked (base structure, so
//                                it carries subgroupSize and supportedOperations only)
//   `subgroup_size_control`      VkPhysicalDeviceSubgroupSizeControlProperties, or
//                                nullptr when VK_EXT_subgroup_size_control / Vulkan
//                                1.3 subgroup-size control is unavailable
//
// `subgroup_size_control_available` is set from the two non-zero subgroup-size
// bounds of that structure, so a driver that ignores the pNext chain (leaving it
// zeroed) is reported as "unknown" instead of "min = max = 0".
[[nodiscard]] inline VulkanCapabilityStatus
make_capability_limits(VkPhysicalDeviceLimits const &limits,
                       VkPhysicalDeviceSubgroupProperties const *subgroup_properties,
                       VkPhysicalDeviceSubgroupSizeControlProperties const *subgroup_size_control) noexcept {
    VulkanCapabilityStatus status{};
    status.max_memory_allocation_count = static_cast<uint32_t>(limits.maxMemoryAllocationCount);
    status.max_compute_workgroup_count_x = static_cast<uint32_t>(limits.maxComputeWorkGroupCount[0]);
    status.max_compute_workgroup_count_y = static_cast<uint32_t>(limits.maxComputeWorkGroupCount[1]);
    status.max_compute_workgroup_count_z = static_cast<uint32_t>(limits.maxComputeWorkGroupCount[2]);
    status.max_compute_workgroup_size_x = static_cast<uint32_t>(limits.maxComputeWorkGroupSize[0]);
    status.max_compute_workgroup_size_y = static_cast<uint32_t>(limits.maxComputeWorkGroupSize[1]);
    status.max_compute_workgroup_size_z = static_cast<uint32_t>(limits.maxComputeWorkGroupSize[2]);
    status.max_compute_shared_memory_bytes = static_cast<uint32_t>(limits.maxComputeSharedMemorySize);
    // VkDeviceSize fields stay 64-bit end to end: no narrowing cast, no
    // round-trip through a 32-bit intermediate.
    status.non_coherent_atom_size = static_cast<uint64_t>(limits.nonCoherentAtomSize);
    status.min_storage_buffer_offset_alignment = static_cast<uint64_t>(limits.minStorageBufferOffsetAlignment);
    status.max_storage_buffer_range = static_cast<uint64_t>(limits.maxStorageBufferRange);
    if (subgroup_properties != nullptr && subgroup_properties->subgroupSize != 0u) {
        status.subgroup_ops_available = true;
        status.subgroup_ops = static_cast<uint32_t>(subgroup_properties->supportedOperations);
    }
    if (subgroup_size_control != nullptr &&
        subgroup_size_control->minSubgroupSize != 0u &&
        subgroup_size_control->maxSubgroupSize != 0u) {
        status.subgroup_size_control_available = true;
        status.min_subgroup_size = static_cast<uint32_t>(subgroup_size_control->minSubgroupSize);
        status.max_subgroup_size = static_cast<uint32_t>(subgroup_size_control->maxSubgroupSize);
    }
    return status;
}

// One query answer. An empty string means "this Vulkan device has no value for
// this limit" - the caller must not read it as zero. Only min/max subgroup size
// and subgroup ops can be unknown on a device this backend accepts (the other
// eleven fields are unconditional members of VkPhysicalDeviceLimits).
//
// Integer formatting: the 32-bit fields are formatted from uint32_t and the
// three VkDeviceSize fields from uint64_t, so a value above 2^32 prints in full.
[[nodiscard]] inline luisa::string
query_core_limit(VulkanCapabilityStatus const &capabilities, CoreLimit limit) noexcept {
    switch (limit) {
        case CoreLimit::MAX_MEMORY_ALLOCATION_COUNT:
            return luisa::format("{}", static_cast<uint32_t>(capabilities.max_memory_allocation_count));
        case CoreLimit::MIN_SUBGROUP_SIZE:
            return capabilities.subgroup_size_control_available ?
                       luisa::format("{}", static_cast<uint32_t>(capabilities.min_subgroup_size)) :
                       luisa::string{};
        case CoreLimit::MAX_SUBGROUP_SIZE:
            return capabilities.subgroup_size_control_available ?
                       luisa::format("{}", static_cast<uint32_t>(capabilities.max_subgroup_size)) :
                       luisa::string{};
        case CoreLimit::SUBGROUP_OPS:
            return capabilities.subgroup_ops_available ?
                       luisa::format("{}", static_cast<uint32_t>(capabilities.subgroup_ops)) :
                       luisa::string{};
        case CoreLimit::MAX_COMPUTE_WORKGROUP_COUNT_X:
            return luisa::format("{}", static_cast<uint32_t>(capabilities.max_compute_workgroup_count_x));
        case CoreLimit::MAX_COMPUTE_WORKGROUP_COUNT_Y:
            return luisa::format("{}", static_cast<uint32_t>(capabilities.max_compute_workgroup_count_y));
        case CoreLimit::MAX_COMPUTE_WORKGROUP_COUNT_Z:
            return luisa::format("{}", static_cast<uint32_t>(capabilities.max_compute_workgroup_count_z));
        case CoreLimit::MAX_COMPUTE_WORKGROUP_SIZE_X:
            return luisa::format("{}", static_cast<uint32_t>(capabilities.max_compute_workgroup_size_x));
        case CoreLimit::MAX_COMPUTE_WORKGROUP_SIZE_Y:
            return luisa::format("{}", static_cast<uint32_t>(capabilities.max_compute_workgroup_size_y));
        case CoreLimit::MAX_COMPUTE_WORKGROUP_SIZE_Z:
            return luisa::format("{}", static_cast<uint32_t>(capabilities.max_compute_workgroup_size_z));
        case CoreLimit::MAX_COMPUTE_SHARED_MEMORY_BYTES:
            return luisa::format("{}", static_cast<uint32_t>(capabilities.max_compute_shared_memory_bytes));
        case CoreLimit::NON_COHERENT_ATOM_SIZE:
            return luisa::format("{}", static_cast<uint64_t>(capabilities.non_coherent_atom_size));
        case CoreLimit::MIN_STORAGE_BUFFER_OFFSET_ALIGNMENT:
            return luisa::format("{}", static_cast<uint64_t>(capabilities.min_storage_buffer_offset_alignment));
        case CoreLimit::MAX_STORAGE_BUFFER_RANGE:
            return luisa::format("{}", static_cast<uint64_t>(capabilities.max_storage_buffer_range));
        case CoreLimit::COUNT: break;
    }
    return {};
}

// ---- SM count (plan P6.2) ---------------------------------------------------
//
// Vulkan has no physical SM / compute-unit count query: no
// VkPhysicalDeviceProperties or VkPhysicalDevice*Properties member reports it,
// and this backend will not invent one from a name table, a device-type guess or
// an implicit benchmark. The reported `sm_count` is therefore *only* a value the
// caller supplied as an externally calibrated estimate through
// VulkanDeviceConfigExt::calibrated_sm_count_estimate(), and it is always paired
// with `sm_count_source = calibration_estimate`. With no calibration the answer
// is unknown: an empty `sm_count` and `sm_count_source = unknown`.
//
// Plausibility bound for that input. A single Vulkan physical device's SM count
// is a two-digit number today; a value above this bound is not a device
// attribute but a caller bug (a byte count, a cluster size, a MHz figure), so it
// is rejected - dropped to unknown and logged - instead of being published as an
// attribute nobody can tell apart from a query. The bound is a sanity guard, not
// a claim about any GPU.
inline constexpr uint32_t kMaxCalibratedSmCount = 4096u;

enum class CalibratedSmCountStatus : uint8_t {
    // No calibration input (the getter returned 0, or no config extension).
    ABSENT,
    // A positive, plausible estimate: reported as calibration_estimate.
    CALIBRATION_ESTIMATE,
    // A non-zero value outside the plausibility bound: reported as unknown.
    REJECTED
};

[[nodiscard]] constexpr const char *
calibrated_sm_count_status_name(CalibratedSmCountStatus status) noexcept {
    switch (status) {
        case CalibratedSmCountStatus::ABSENT: return "absent";
        case CalibratedSmCountStatus::CALIBRATION_ESTIMATE: return "calibration_estimate";
        case CalibratedSmCountStatus::REJECTED: return "rejected_out_of_range";
    }
    return "unknown";
}

struct CalibratedSmCountPlan {
    CalibratedSmCountStatus status{CalibratedSmCountStatus::ABSENT};
    // Valid only for CALIBRATION_ESTIMATE; 0 otherwise.
    uint32_t value{0u};
    [[nodiscard]] constexpr explicit operator bool() const noexcept {
        return status == CalibratedSmCountStatus::CALIBRATION_ESTIMATE;
    }
};

// Zero means "no estimate" (the documented default) and stays unknown, exactly
// like an absent getter: zero is not a device attribute either.
[[nodiscard]] constexpr CalibratedSmCountPlan
plan_calibrated_sm_count(uint64_t requested) noexcept {
    if (requested == 0u) {
        return {.status = CalibratedSmCountStatus::ABSENT, .value = 0u};
    }
    if (requested > static_cast<uint64_t>(kMaxCalibratedSmCount)) {
        return {.status = CalibratedSmCountStatus::REJECTED, .value = 0u};
    }
    return {.status = CalibratedSmCountStatus::CALIBRATION_ESTIMATE,
            .value = static_cast<uint32_t>(requested)};
}

// Empty when unknown. A calibration estimate is reported with its source, so a
// consumer that needs a real device attribute can still tell the difference.
[[nodiscard]] inline luisa::string
query_sm_count(VulkanCapabilityStatus const &capabilities) noexcept {
    return capabilities.sm_count_source == VulkanSmCountSource::CALIBRATION_ESTIMATE &&
                   capabilities.sm_count != 0u ?
               luisa::format("{}", static_cast<uint32_t>(capabilities.sm_count)) :
               luisa::string{};
}

[[nodiscard]] inline luisa::string
query_sm_count_source(VulkanCapabilityStatus const &capabilities) noexcept {
    return luisa::string{vulkan_sm_count_source_name(capabilities.sm_count_source)};
}

}// namespace lc::vk::detail