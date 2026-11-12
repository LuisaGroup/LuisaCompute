#pragma once

#include <cstddef>
#include <cstdint>
#include <vulkan/vulkan_core.h>

#include "device_address_command_plan.h"

namespace lc::vk::detail {

// Host-only planning for the optional buffer-to-buffer copy batching paths.
//
// The pre-existing encoder records one vkCmdCopyBuffer / vkCmdCopyBuffer2 per
// LuisaCompute BufferCopyCommand, immediately when the command is visited. Two
// optional extensions can collapse a batch of such copies into far fewer
// commands:
//
//   * VK_KHR_copy_memory_indirect (alias VK_NV_copy_memory_indirect): the copy
//     parameters are read by the device from an array of
//     VkCopyMemoryIndirectCommandKHR records, so one
//     vkCmdCopyMemoryIndirectKHR replaces N copy commands. The command array is
//     a backend-owned, host-writable, device-addressable buffer.
//
//   * VK_KHR_device_address_commands: vkCmdCopyMemoryKHR takes an array of
//     VkDeviceMemoryCopyKHR ranges described by VkDeviceAddress, so one command
//     can cover copies into *different* buffers. Its aliasing rule is much
//     stricter than the indirect one (see below), so it is only selected when
//     the batch happens to satisfy it.
//
// Everything here is `constexpr` and free of device state so the whole decision
// matrix is unit testable without a Vulkan device. Correctness never depends on
// either extension: `plan_buffer_copy_batch` reports an explicit mode, and
// `BufferCopyBatchMode::DIRECT_COPIES` is exactly the pre-existing per-copy
// vkCmdCopyBuffer path.

// ---------------------------------------------------------------------------
// Copy command records
// ---------------------------------------------------------------------------

// VUID-VkCopyMemoryIndirectCommandKHR-srcAddress-10958 / -dstAddress-10959 /
// -size-10960: every address and the size of one indirect copy must be a
// multiple of 4. A copy that does not satisfy this stays on the direct path.
inline constexpr uint64_t copy_command_alignment = 4u;

// VUID-VkCopyMemoryIndirectInfoKHR-copyAddressRange-10943: the record stride
// must be a multiple of 4 and at least sizeof(VkCopyMemoryIndirectCommandKHR).
// The backend packs the records tightly, so stride == record size.
inline constexpr uint64_t indirect_copy_command_stride =
    sizeof(VkCopyMemoryIndirectCommandKHR);
static_assert(indirect_copy_command_stride == 24u,
              "VkCopyMemoryIndirectCommandKHR must stay three 64-bit words.");
static_assert(indirect_copy_command_stride % copy_command_alignment == 0u,
              "The indirect copy record stride must be a multiple of 4.");
// VK_NV_copy_memory_indirect is the vendor spelling of the same command: the
// record layout is identical (the NV structure is a typedef of the KHR one) and
// its stride parameter is a 32-bit value.
static_assert(sizeof(VkCopyMemoryIndirectCommandNV) ==
                  sizeof(VkCopyMemoryIndirectCommandKHR),
              "The NV and KHR indirect copy records must stay identical.");
static_assert(indirect_copy_command_stride <= UINT32_MAX,
              "vkCmdCopyMemoryIndirectNV takes a 32-bit record stride.");

// Below this many copies the single indirect command does not pay for the extra
// host-side packing, so the encoder keeps the direct path.
inline constexpr uint32_t indirect_copy_minimum_batch_size = 4u;

// Records the backend-owned command array holds. A layer with more eligible
// copies is flushed in several indirect commands.
inline constexpr uint32_t indirect_copy_command_capacity = 256u;

// One buffer-to-buffer copy described by absolute device addresses.
struct BufferCopyRange {
    uint64_t src_address{0u};
    uint64_t dst_address{0u};
    uint64_t size{0u};
};

enum class BufferCopyEligibility : uint32_t {
    ELIGIBLE = 0u,
    // size == 0: nothing to copy, so there is nothing to batch.
    EMPTY,
    // VUID-VkCopyMemoryIndirectCommandKHR-size-10960.
    SIZE_MISALIGNED,
    // VUID-VkCopyMemoryIndirectCommandKHR-srcAddress-10958.
    SOURCE_MISALIGNED,
    // VUID-VkCopyMemoryIndirectCommandKHR-dstAddress-10959.
    DESTINATION_MISALIGNED,
    // src/dst address + size does not fit a VkDeviceAddress, so the range cannot
    // be described at all.
    RANGE_OVERFLOW,
};

[[nodiscard]] constexpr const char *
buffer_copy_eligibility_name(BufferCopyEligibility eligibility) noexcept {
    switch (eligibility) {
        case BufferCopyEligibility::ELIGIBLE: return "eligible";
        case BufferCopyEligibility::EMPTY: return "empty copy";
        case BufferCopyEligibility::SIZE_MISALIGNED: return "size is not 4-byte aligned";
        case BufferCopyEligibility::SOURCE_MISALIGNED: return "source address is not 4-byte aligned";
        case BufferCopyEligibility::DESTINATION_MISALIGNED: return "destination address is not 4-byte aligned";
        case BufferCopyEligibility::RANGE_OVERFLOW: return "address range overflows a VkDeviceAddress";
    }
    return "unknown";
}

[[nodiscard]] constexpr BufferCopyEligibility
check_buffer_copy_eligibility(uint64_t src_address, uint64_t dst_address,
                              uint64_t size) noexcept {
    if (size == 0u) { return BufferCopyEligibility::EMPTY; }
    if (size % copy_command_alignment != 0u) {
        return BufferCopyEligibility::SIZE_MISALIGNED;
    }
    if (src_address % copy_command_alignment != 0u) {
        return BufferCopyEligibility::SOURCE_MISALIGNED;
    }
    if (dst_address % copy_command_alignment != 0u) {
        return BufferCopyEligibility::DESTINATION_MISALIGNED;
    }
    if (size > UINT64_MAX - src_address ||
        size > UINT64_MAX - dst_address) {
        return BufferCopyEligibility::RANGE_OVERFLOW;
    }
    return BufferCopyEligibility::ELIGIBLE;
}

// Two half-open address ranges overlap.
[[nodiscard]] constexpr bool address_ranges_overlap(
    uint64_t begin, uint64_t size,
    uint64_t other_begin, uint64_t other_size) noexcept {
    if (size == 0u || other_size == 0u) { return false; }
    return begin < other_begin + other_size &&
           other_begin < begin + size;
}

// VUID-VkCopyMemoryIndirectInfoKHR-copyAddressRange-10945: inside one
// vkCmdCopyMemoryIndirectKHR no memory location may be read by a source region
// and written by a destination region, and no location may be written by more
// than one destination region. Source regions may overlap each other, which is
// what makes the indirect form usable for the common
// "one source range into many destination ranges" flush.
[[nodiscard]] constexpr bool
indirect_copy_ranges_honor_aliasing_rule(
    BufferCopyRange const *ranges, size_t count) noexcept {
    for (auto i = size_t{0u}; i < count; ++i) {
        auto const &a = ranges[i];
        // The rule covers the element itself: a single copy whose source range
        // overlaps its own destination range is not expressible either.
        if (address_ranges_overlap(a.src_address, a.size,
                                   a.dst_address, a.size)) {
            return false;
        }
        for (auto j = i + 1u; j < count; ++j) {
            auto const &b = ranges[j];
            if (address_ranges_overlap(a.dst_address, a.size,
                                       b.dst_address, b.size)) {
                return false;
            }
            if (address_ranges_overlap(a.src_address, a.size,
                                       b.dst_address, b.size) ||
                address_ranges_overlap(b.src_address, b.size,
                                       a.dst_address, a.size)) {
                return false;
            }
        }
    }
    return true;
}

// VUID-VkCopyDeviceMemoryInfoKHR-srcRange-13015: vkCmdCopyMemoryKHR is much
// stricter than its indirect sibling - *every* source range must be disjoint
// from *every* source and destination range of the command. A batch therefore
// only qualifies when all of its ranges are mutually disjoint in both roles,
// which is why this path is selected far less often than the indirect one.
[[nodiscard]] constexpr bool
device_address_copy_ranges_honor_aliasing_rule(
    BufferCopyRange const *ranges, size_t count) noexcept {
    for (auto i = size_t{0u}; i < count; ++i) {
        auto const &a = ranges[i];
        // The rule covers the element itself: a source range may not overlap
        // its own destination range, so an in-place copy cannot use
        // vkCmdCopyMemoryKHR either.
        if (address_ranges_overlap(a.src_address, a.size,
                                   a.dst_address, a.size)) {
            return false;
        }
        for (auto j = i + 1u; j < count; ++j) {
            auto const &b = ranges[j];
            if (address_ranges_overlap(a.src_address, a.size,
                                       b.src_address, b.size) ||
                address_ranges_overlap(a.src_address, a.size,
                                       b.dst_address, b.size) ||
                address_ranges_overlap(a.dst_address, a.size,
                                       b.src_address, b.size) ||
                address_ranges_overlap(a.dst_address, a.size,
                                       b.dst_address, b.size)) {
                return false;
            }
        }
    }
    return true;
}

// ---------------------------------------------------------------------------
// The batch decision
// ---------------------------------------------------------------------------

enum class BufferCopyBatchMode : uint32_t {
    // Pre-existing path: one vkCmdCopyBuffer(2) per copy.
    DIRECT_COPIES = 0u,
    // One vkCmdCopyMemoryKHR carrying VkDeviceMemoryCopyKHR ranges.
    DEVICE_ADDRESS_REGIONS,
    // One vkCmdCopyMemoryIndirectKHR reading VkCopyMemoryIndirectCommandKHR
    // records out of the backend command array.
    INDIRECT_COMMANDS,
};

enum class BufferCopyBatchStatus : uint32_t {
    // The names are unique across the two batch enums of this header, so
    // `using enum` on either of them stays unambiguous.
    BATCHED_INDIRECT_COMMANDS = 0u,
    BATCHED_DEVICE_ADDRESS_REGIONS,
    // Neither participating buffer exposes a shader device address, so neither
    // address based form can describe the copy.
    FALLBACK_NO_DEVICE_ADDRESS,
    // Nothing was accumulated.
    FALLBACK_EMPTY_BATCH,
    // Neither VK_KHR_copy_memory_indirect nor VK_KHR_device_address_commands is
    // enabled, so the pre-existing per-copy path is the only one left.
    FALLBACK_EXTENSIONS_DISABLED,
    // Fewer copies than indirect_copy_minimum_batch_size.
    FALLBACK_BATCH_TOO_SMALL,
    // The backend command array is missing (no device address, allocation
    // failure) or too small for the batch.
    FALLBACK_COMMAND_ARRAY_UNAVAILABLE,
    // The command array's device address is not 4-byte aligned
    // (VUID-VkCopyMemoryIndirectInfoKHR-copyAddressRange-10942).
    FALLBACK_COMMAND_ARRAY_MISALIGNED,
    // The batch violates the aliasing rule of every address based form, so it
    // must be recorded copy by copy.
    FALLBACK_ALIASING,
};

[[nodiscard]] constexpr const char *
buffer_copy_batch_status_name(BufferCopyBatchStatus status) noexcept {
    switch (status) {
        case BufferCopyBatchStatus::BATCHED_INDIRECT_COMMANDS: return "indirect copy commands";
        case BufferCopyBatchStatus::BATCHED_DEVICE_ADDRESS_REGIONS: return "device address copy regions";
        case BufferCopyBatchStatus::FALLBACK_NO_DEVICE_ADDRESS: return "fallback: no buffer device address";
        case BufferCopyBatchStatus::FALLBACK_EMPTY_BATCH: return "fallback: empty batch";
        case BufferCopyBatchStatus::FALLBACK_EXTENSIONS_DISABLED: return "fallback: copy batching extensions disabled";
        case BufferCopyBatchStatus::FALLBACK_BATCH_TOO_SMALL: return "fallback: batch too small";
        case BufferCopyBatchStatus::FALLBACK_COMMAND_ARRAY_UNAVAILABLE: return "fallback: no indirect command array";
        case BufferCopyBatchStatus::FALLBACK_COMMAND_ARRAY_MISALIGNED: return "fallback: misaligned indirect command array";
        case BufferCopyBatchStatus::FALLBACK_ALIASING: return "fallback: copy ranges alias each other";
    }
    return "unknown";
}

struct BufferCopyBatchInput {
    // optional_extension_enabled(COPY_MEMORY_INDIRECT)
    bool indirect_extension_enabled{false};
    // The command pool the batch is recorded into supports one of the queue
    // types in VkPhysicalDeviceCopyMemoryIndirectPropertiesKHR::supportedQueues
    // (VUID-vkCmdCopyMemoryIndirectKHR-commandBuffer-10936).
    bool indirect_queue_supported{false};
    // optional_extension_enabled(DEVICE_ADDRESS_COMMANDS)
    bool device_address_commands_enabled{false};
    // Every buffer of the batch was created with a shader device address.
    bool device_addresses_available{false};
    uint32_t candidate_count{0u};
    uint32_t minimum_batch_size{indirect_copy_minimum_batch_size};
    uint32_t record_capacity{indirect_copy_command_capacity};
    // Device address of the backend indirect command array (0 when there is
    // none).
    uint64_t command_array_address{0u};
    // indirect_copy_ranges_honor_aliasing_rule(batch)
    bool indirect_aliasing_rule_satisfied{true};
    // device_address_copy_ranges_honor_aliasing_rule(batch)
    bool device_address_aliasing_rule_satisfied{true};
};

struct BufferCopyBatchPlan {
    BufferCopyBatchMode mode{BufferCopyBatchMode::DIRECT_COPIES};
    BufferCopyBatchStatus status{
        BufferCopyBatchStatus::FALLBACK_EMPTY_BATCH};
    // Copies covered by the single batched command (0 for DIRECT_COPIES).
    uint32_t batched_copy_count{0u};
    // Copies the encoder must still record one by one.
    uint32_t direct_copy_count{0u};
    // Record stride of the indirect command array (0 unless INDIRECT_COMMANDS).
    uint64_t stride{0u};
    // Bytes of the command array the indirect command reads
    // (VkCopyMemoryIndirectInfoKHR::copyAddressRange.size).
    uint64_t command_array_size_bytes{0u};
    [[nodiscard]] constexpr bool uses_indirect_commands() const noexcept {
        return mode == BufferCopyBatchMode::INDIRECT_COMMANDS;
    }
    [[nodiscard]] constexpr bool uses_device_address_regions() const noexcept {
        return mode == BufferCopyBatchMode::DEVICE_ADDRESS_REGIONS;
    }
    // True when the encoder has to fall back to per-copy commands.
    [[nodiscard]] constexpr bool uses_direct_copies() const noexcept {
        return mode == BufferCopyBatchMode::DIRECT_COPIES;
    }
};

// Decides how one accumulated batch of buffer-to-buffer copies is recorded.
//
// The invariant that makes the fallback equivalent is encoded here and unit
// tested: `batched_copy_count + direct_copy_count == candidate_count` for every
// possible input, so the fallback records exactly the same number of copies as
// the pre-existing path (and the fast path records every candidate exactly once,
// either inside the single batched command or directly).
[[nodiscard]] constexpr BufferCopyBatchPlan
plan_buffer_copy_batch(const BufferCopyBatchInput &input) noexcept {
    constexpr auto all_direct = [](uint32_t count,
                                   BufferCopyBatchStatus status) noexcept {
        return BufferCopyBatchPlan{
            .mode = BufferCopyBatchMode::DIRECT_COPIES,
            .status = status,
            .batched_copy_count = 0u,
            .direct_copy_count = count,
            .stride = 0u,
            .command_array_size_bytes = 0u};
    };
    if (input.candidate_count == 0u) {
        return all_direct(0u, BufferCopyBatchStatus::FALLBACK_EMPTY_BATCH);
    }
    if (!input.device_addresses_available) {
        // Neither address based form can describe the copies, whatever the
        // extensions say.
        return all_direct(
            input.candidate_count,
            BufferCopyBatchStatus::FALLBACK_NO_DEVICE_ADDRESS);
    }
    auto const indirect_requested =
        input.indirect_extension_enabled && input.indirect_queue_supported;
    auto const device_address_requested =
        input.device_address_commands_enabled;
    // The backend command array is the only part the encoder can be missing at
    // runtime (a device without buffer device addresses, an allocation failure).
    // A zero address is an absent array, not an aligned one
    // (VUID-VkDeviceAddressRangeKHR-size-11411: a non-zero size needs a non-zero
    // address).
    auto const command_array_available =
        input.record_capacity != 0u &&
        input.candidate_count <= input.record_capacity &&
        input.command_array_address != 0u;
    auto const command_array_aligned =
        input.command_array_address % copy_command_alignment == 0u;
    auto const batch_worthwhile =
        input.candidate_count >= input.minimum_batch_size;
    // One vkCmdCopyMemoryIndirectKHR first: it covers every copy with one
    // command, tolerates source ranges that overlap each other, and needs no
    // host-side region array.
    if (indirect_requested && command_array_available &&
        command_array_aligned && batch_worthwhile &&
        input.indirect_aliasing_rule_satisfied) {
        return {
            .mode = BufferCopyBatchMode::INDIRECT_COMMANDS,
            .status = BufferCopyBatchStatus::BATCHED_INDIRECT_COMMANDS,
            .batched_copy_count = input.candidate_count,
            .direct_copy_count = 0u,
            .stride = indirect_copy_command_stride,
            .command_array_size_bytes =
                indirect_copy_command_stride *
                static_cast<uint64_t>(input.candidate_count)};
    }
    // Then one vkCmdCopyMemoryKHR, whose ranges must all be mutually disjoint.
    if (device_address_requested && batch_worthwhile &&
        input.device_address_aliasing_rule_satisfied) {
        return {
            .mode = BufferCopyBatchMode::DEVICE_ADDRESS_REGIONS,
            .status = BufferCopyBatchStatus::BATCHED_DEVICE_ADDRESS_REGIONS,
            .batched_copy_count = input.candidate_count,
            .direct_copy_count = 0u,
            .stride = 0u,
            .command_array_size_bytes = 0u};
    }
    // Every copy stays on the pre-existing path. Report the reason that explains
    // why the fastest applicable form was rejected.
    if (!indirect_requested && !device_address_requested) {
        return all_direct(
            input.candidate_count,
            BufferCopyBatchStatus::FALLBACK_EXTENSIONS_DISABLED);
    }
    if (!batch_worthwhile) {
        return all_direct(
            input.candidate_count,
            BufferCopyBatchStatus::FALLBACK_BATCH_TOO_SMALL);
    }
    if (indirect_requested && !device_address_requested) {
        if (!command_array_available) {
            return all_direct(
                input.candidate_count,
                BufferCopyBatchStatus::FALLBACK_COMMAND_ARRAY_UNAVAILABLE);
        }
        if (!command_array_aligned) {
            return all_direct(
                input.candidate_count,
                BufferCopyBatchStatus::FALLBACK_COMMAND_ARRAY_MISALIGNED);
        }
    }
    if (!input.indirect_aliasing_rule_satisfied ||
        !input.device_address_aliasing_rule_satisfied) {
        return all_direct(
            input.candidate_count,
            BufferCopyBatchStatus::FALLBACK_ALIASING);
    }
    // The remaining case is an indirect batch that the device-address form could
    // not replace either, so the array is the only thing left to blame.
    return all_direct(
        input.candidate_count,
        indirect_requested ?
            BufferCopyBatchStatus::FALLBACK_COMMAND_ARRAY_UNAVAILABLE :
            BufferCopyBatchStatus::FALLBACK_EXTENSIONS_DISABLED);
}

// ---------------------------------------------------------------------------
// Record builders
// ---------------------------------------------------------------------------

[[nodiscard]] constexpr VkCopyMemoryIndirectCommandKHR
make_copy_memory_indirect_command(const BufferCopyRange &range) noexcept {
    return VkCopyMemoryIndirectCommandKHR{
        .srcAddress = range.src_address,
        .dstAddress = range.dst_address,
        .size = range.size};
}

// VkCopyMemoryIndirectInfoKHR for a plan that selected INDIRECT_COMMANDS.
// `srcCopyFlags`/`dstCopyFlags` stay 0: VK_ADDRESS_COPY_DEVICE_LOCAL_BIT_KHR is
// only a residency hint, and the backend cannot prove that a user buffer lives
// in device-local memory. VK_ADDRESS_COPY_PROTECTED_BIT_KHR must not be set
// (VUID-...-srcCopyFlags-10940 / -10941).
[[nodiscard]] constexpr VkCopyMemoryIndirectInfoKHR
make_copy_memory_indirect_info(
    const BufferCopyBatchPlan &plan,
    uint64_t command_array_address) noexcept {
    return VkCopyMemoryIndirectInfoKHR{
        .sType = VK_STRUCTURE_TYPE_COPY_MEMORY_INDIRECT_INFO_KHR,
        .pNext = nullptr,
        .srcCopyFlags = 0u,
        .dstCopyFlags = 0u,
        .copyCount = plan.batched_copy_count,
        .copyAddressRange = VkStridedDeviceAddressRangeKHR{
            .address = command_array_address,
            .size = plan.command_array_size_bytes,
            .stride = plan.stride}};
}

// One VkDeviceMemoryCopyKHR region of a vkCmdCopyMemoryKHR batch.
// VUID-VkDeviceMemoryCopyKHR-size-13016: the destination range must be at least
// as large as the source range, so both describe exactly the copied bytes.
[[nodiscard]] constexpr VkDeviceMemoryCopyKHR
make_device_memory_copy_region(const BufferCopyRange &range,
                               bool storage_buffer_usage) noexcept {
    return VkDeviceMemoryCopyKHR{
        .sType = VK_STRUCTURE_TYPE_DEVICE_MEMORY_COPY_KHR,
        .pNext = nullptr,
        .srcRange = VkDeviceAddressRangeKHR{
            .address = range.src_address,
            .size = range.size},
        .srcFlags = device_address_command_flags(storage_buffer_usage),
        .dstRange = VkDeviceAddressRangeKHR{
            .address = range.dst_address,
            .size = range.size},
        .dstFlags = device_address_command_flags(storage_buffer_usage)};
}

// VkCopyDeviceMemoryInfoKHR for a plan that selected DEVICE_ADDRESS_REGIONS.
// The region array itself is recorded from host memory, exactly like the region
// array of vkCmdCopyBuffer, so no device buffer is involved.
[[nodiscard]] constexpr VkCopyDeviceMemoryInfoKHR
make_copy_device_memory_info(
    uint32_t region_count,
    const VkDeviceMemoryCopyKHR *regions) noexcept {
    return VkCopyDeviceMemoryInfoKHR{
        .sType = VK_STRUCTURE_TYPE_COPY_DEVICE_MEMORY_INFO_KHR,
        .pNext = nullptr,
        .regionCount = region_count,
        .pRegions = regions};
}

}// namespace lc::vk::detail
