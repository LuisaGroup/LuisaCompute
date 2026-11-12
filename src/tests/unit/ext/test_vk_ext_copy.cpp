// Test for Vulkan indirect copy and device-address command planning.
// This test covers:
// - the host-visible planning helpers of the extension group above
//   (VK_KHR_copy_memory_indirect / VK_NV_copy_memory_indirect and
//   VK_KHR_device_address_commands)
// - the batch decision matrix of `detail::plan_buffer_copy_batch`: extension
//   enabled/disabled/too small/no device address/no command array/aliasing
// - the record layout, stride and alignment of the indirect command array and
//   of the device-address copy regions
// - the invariant that the fallback records exactly as many copies as the
//   pre-existing per-copy path, i.e. that batching never loses or duplicates a
//   copy
// - the device-address indirect dispatch gate (`vkCmdDispatchIndirect2KHR`
//   versus the pre-existing `vkCmdDispatchIndirect`)
#include "ut/ut.hpp"

#include "copy_command_plan.h"
#include "device_address_command_plan.h"

#include <luisa/backends/ext/vk_config_ext.h>
#include <luisa/backends/ext/vk_extension_ext.h>
#include <luisa/dsl/sugar.h>
#include <luisa/luisa-compute.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <string_view>

using namespace boost::ut;
using namespace boost::ut::literals;
using namespace lc::vk::detail;

namespace vk_ext_copy_test_detail {

// Number of Vulkan commands and number of copies the encoder would record for
// one planned batch. A batched mode is a single command that still covers every
// copy of the batch; the direct mode is one command per copy.
struct RecordedCopies {
    uint32_t command_count;
    uint32_t copy_count;
};

[[nodiscard]] RecordedCopies simulate_recording(
    const BufferCopyBatchPlan &plan) noexcept {
    if (plan.uses_direct_copies()) {
        return {.command_count = plan.direct_copy_count,
                .copy_count = plan.direct_copy_count};
    }
    return {.command_count = 1u,
            .copy_count = plan.batched_copy_count};
}

// A batch input with everything enabled and every constraint satisfied, which
// each test then perturbs in exactly one field.
[[nodiscard]] constexpr BufferCopyBatchInput enabled_input(
    uint32_t candidate_count = 8u) noexcept {
    return BufferCopyBatchInput{
        .indirect_extension_enabled = true,
        .indirect_queue_supported = true,
        .device_address_commands_enabled = true,
        .device_addresses_available = true,
        .candidate_count = candidate_count,
        .minimum_batch_size = indirect_copy_minimum_batch_size,
        .record_capacity = indirect_copy_command_capacity,
        .command_array_address = 0x1'0000u,
        .indirect_aliasing_rule_satisfied = true,
        .device_address_aliasing_rule_satisfied = true};
}

// Fully disjoint copies: legal for both the indirect and the stricter
// device-address form.
inline constexpr std::array<BufferCopyRange, 3u> disjoint_ranges{{
    {.src_address = 0x1000u, .dst_address = 0x9000u, .size = 0x100u},
    {.src_address = 0x2000u, .dst_address = 0xa000u, .size = 0x100u},
    {.src_address = 0x3000u, .dst_address = 0xb000u, .size = 0x100u},
}};

// One source range copied into three different destinations: legal for
// vkCmdCopyMemoryIndirectKHR (source ranges may overlap each other), illegal for
// vkCmdCopyMemoryKHR (VUID-VkCopyDeviceMemoryInfoKHR-srcRange-13015).
inline constexpr std::array<BufferCopyRange, 3u> shared_source_ranges{{
    {.src_address = 0x1000u, .dst_address = 0x9000u, .size = 0x100u},
    {.src_address = 0x1000u, .dst_address = 0xa000u, .size = 0x100u},
    {.src_address = 0x1000u, .dst_address = 0xb000u, .size = 0x100u},
}};

// Two copies into the same destination: neither form is legal.
inline constexpr std::array<BufferCopyRange, 2u> shared_destination_ranges{{
    {.src_address = 0x1000u, .dst_address = 0x9000u, .size = 0x100u},
    {.src_address = 0x2000u, .dst_address = 0x9000u, .size = 0x100u},
}};

// A copy that touches the same memory as source and destination.
inline constexpr std::array<BufferCopyRange, 1u> in_place_ranges{{
    {.src_address = 0x1000u, .dst_address = 0x1040u, .size = 0x100u},
}};

// Switches only VK_KHR_copy_memory_indirect, so one process can compare the
// batched fast path with the pre-existing per-copy path on the same GPU.
class CopyMemoryIndirectConfig final : public luisa::compute::VulkanDeviceConfigExt {
private:
    bool _enabled;

public:
    explicit CopyMemoryIndirectConfig(bool enabled) noexcept
        : _enabled{enabled} {}
    [[nodiscard]] bool enable_copy_memory_indirect() const noexcept override {
        return _enabled;
    }
};

// The planners are pure: the whole decision matrix is evaluable at compile time,
// which is what lets the backend branch on it without any hidden device state.
static_assert(check_buffer_copy_eligibility(0x1000u, 0x2000u, 0x40u) ==
              BufferCopyEligibility::ELIGIBLE);
static_assert(check_buffer_copy_eligibility(0x1000u, 0x2000u, 0x3u) ==
              BufferCopyEligibility::SIZE_MISALIGNED);
static_assert(plan_buffer_copy_batch(
                  vk_ext_copy_test_detail::enabled_input(8u))
                  .uses_indirect_commands());
static_assert(plan_buffer_copy_batch(
                  vk_ext_copy_test_detail::enabled_input(2u))
                  .uses_direct_copies());
static_assert(plan_device_address_indirect_dispatch(
                  true, true, 0x1000u, 0u, dispatch_indirect_command_size)
                  .use_device_address());
static_assert(!plan_device_address_indirect_dispatch(
                  false, true, 0x1000u, 0u, dispatch_indirect_command_size)
                   .use_device_address());
static_assert(indirect_copy_ranges_honor_aliasing_rule(
                  vk_ext_copy_test_detail::disjoint_ranges.data(),
                  vk_ext_copy_test_detail::disjoint_ranges.size()));

}// namespace vk_ext_copy_test_detail

int main(int argc, char *argv[]) {

    boost::ut::detail::cfg::parse_arg_with_fallback(
        argc, const_cast<const char **>(argv));

    // -----------------------------------------------------------------------
    // VK_KHR_copy_memory_indirect: record layout, stride and alignment
    // -----------------------------------------------------------------------
    "indirect_copy_record_layout"_test = [] {
        // VUID-VkCopyMemoryIndirectInfoKHR-copyAddressRange-10943: the stride
        // must be a multiple of 4 and must be at least the record size.
        expect(indirect_copy_command_stride ==
               sizeof(VkCopyMemoryIndirectCommandKHR));
        expect(indirect_copy_command_stride % copy_command_alignment == 0u);
        expect(indirect_copy_command_stride >=
               sizeof(VkCopyMemoryIndirectCommandKHR));
        // The three 64-bit words are packed in declaration order, so the records
        // the backend uploads are exactly what the device reads.
        expect(sizeof(VkCopyMemoryIndirectCommandKHR) == 24u);
        expect(offsetof(VkCopyMemoryIndirectCommandKHR, srcAddress) == 0u);
        expect(offsetof(VkCopyMemoryIndirectCommandKHR, dstAddress) == 8u);
        expect(offsetof(VkCopyMemoryIndirectCommandKHR, size) == 16u);
        // VK_NV_copy_memory_indirect is the vendor spelling of the same
        // command: the record is a typedef of the KHR structure, so the same
        // tightly packed array can be fed to either entry point.
        expect(sizeof(VkCopyMemoryIndirectCommandNV) ==
               sizeof(VkCopyMemoryIndirectCommandKHR));
        // Alignment constants the planner checks against.
        expect(copy_command_alignment == 4u);
        expect(indirect_copy_minimum_batch_size >= 2u);
        expect(indirect_copy_command_capacity >=
               indirect_copy_minimum_batch_size);

        // A record carries the copy parameters verbatim.
        auto record = make_copy_memory_indirect_command(
            BufferCopyRange{.src_address = 0x1000u,
                            .dst_address = 0x2000u,
                            .size = 0x40u});
        expect(record.srcAddress == 0x1000u);
        expect(record.dstAddress == 0x2000u);
        expect(record.size == 0x40u);

        // The info structure describes `copyCount` tightly packed records and
        // never requests the protected copy flags
        // (VUID-VkCopyMemoryIndirectInfoKHR-srcCopyFlags-10940 / -10941).
        auto plan = plan_buffer_copy_batch(
            vk_ext_copy_test_detail::enabled_input(5u));
        expect(plan.uses_indirect_commands());
        auto info = make_copy_memory_indirect_info(plan, 0x1'0000u);
        expect(info.sType == VK_STRUCTURE_TYPE_COPY_MEMORY_INDIRECT_INFO_KHR);
        expect(info.pNext == nullptr);
        expect(info.srcCopyFlags == 0u);
        expect(info.dstCopyFlags == 0u);
        expect(info.copyCount == 5u);
        expect(info.copyAddressRange.address == 0x1'0000u);
        expect(info.copyAddressRange.stride == indirect_copy_command_stride);
        expect(info.copyAddressRange.size ==
               5u * indirect_copy_command_stride);
        // VUID-VkCopyMemoryIndirectInfoKHR-copyCount-10944: copyCount must be
        // less than or equal to size / stride.
        expect(info.copyCount <=
               info.copyAddressRange.size / info.copyAddressRange.stride);
        // VUID-VkCopyMemoryIndirectInfoKHR-copyAddressRange-10942.
        expect(info.copyAddressRange.address % copy_command_alignment == 0u);
    };

    "device_address_region_layout"_test = [] {
        // Both ranges describe exactly the copied bytes, so the destination is
        // never smaller than the source
        // (VUID-VkDeviceMemoryCopyKHR-size-13016).
        auto region = make_device_memory_copy_region(
            BufferCopyRange{.src_address = 0x1000u,
                            .dst_address = 0x2000u,
                            .size = 0x84u},
            true);
        expect(region.sType == VK_STRUCTURE_TYPE_DEVICE_MEMORY_COPY_KHR);
        expect(region.pNext == nullptr);
        expect(region.srcRange.address == 0x1000u);
        expect(region.dstRange.address == 0x2000u);
        expect(region.srcRange.size == 0x84u);
        expect(region.dstRange.size == region.srcRange.size);
        expect(region.srcFlags ==
               VK_ADDRESS_COMMAND_STORAGE_BUFFER_USAGE_BIT_KHR);
        expect(region.dstFlags == region.srcFlags);

        // A range whose backing buffers cannot be classified uses the
        // always-valid UNKNOWN flag instead.
        auto unknown = make_device_memory_copy_region(
            BufferCopyRange{.src_address = 0x1000u,
                            .dst_address = 0x2000u,
                            .size = 0x40u},
            false);
        expect(unknown.srcFlags ==
               VK_ADDRESS_COMMAND_UNKNOWN_STORAGE_BUFFER_USAGE_BIT_KHR);
        expect(unknown.dstFlags == unknown.srcFlags);

        std::array regions{region, unknown};
        auto info = make_copy_device_memory_info(
            static_cast<uint32_t>(regions.size()), regions.data());
        // The chained structures must identify themselves; a zero sType would be
        // VK_STRUCTURE_TYPE_APPLICATION_INFO.
        expect(info.sType == VK_STRUCTURE_TYPE_COPY_DEVICE_MEMORY_INFO_KHR);
        expect(info.pNext == nullptr);
        expect(info.regionCount == 2u);
        expect(info.pRegions == regions.data());
    };

    "address_command_flags"_test = [] {
        // VUID-...-addressFlags-13100: the two storage-buffer flags are mutually
        // exclusive, and the backend never asks for protected memory.
        auto known = device_address_command_flags(true);
        auto unknown = device_address_command_flags(false);
        expect((known & VK_ADDRESS_COMMAND_STORAGE_BUFFER_USAGE_BIT_KHR) != 0u);
        expect((known & VK_ADDRESS_COMMAND_UNKNOWN_STORAGE_BUFFER_USAGE_BIT_KHR) == 0u);
        expect((unknown & VK_ADDRESS_COMMAND_STORAGE_BUFFER_USAGE_BIT_KHR) == 0u);
        expect((unknown & VK_ADDRESS_COMMAND_UNKNOWN_STORAGE_BUFFER_USAGE_BIT_KHR) != 0u);
        for (auto flags : {known, unknown}) {
            expect((flags & VK_ADDRESS_COMMAND_PROTECTED_BIT_KHR) == 0u);
            expect((flags &
                    (VK_ADDRESS_COMMAND_STORAGE_BUFFER_USAGE_BIT_KHR |
                     VK_ADDRESS_COMMAND_UNKNOWN_STORAGE_BUFFER_USAGE_BIT_KHR)) !=
                   0u);
        }
    };

    // -----------------------------------------------------------------------
    // Per-copy eligibility
    // -----------------------------------------------------------------------
    "per_copy_eligibility"_test = [] {
        using enum BufferCopyEligibility;
        // Both addresses and the size must be 4-byte aligned.
        expect(check_buffer_copy_eligibility(0x1000u, 0x2000u, 0x40u) == ELIGIBLE);
        expect(check_buffer_copy_eligibility(0x1000u, 0x2000u, 0x4u) == ELIGIBLE);
        expect(check_buffer_copy_eligibility(0x1000u, 0x2000u, 0x0u) == EMPTY);
        expect(check_buffer_copy_eligibility(0x1000u, 0x2000u, 0x41u) == SIZE_MISALIGNED);
        expect(check_buffer_copy_eligibility(0x1000u, 0x2000u, 0x2u) == SIZE_MISALIGNED);
        expect(check_buffer_copy_eligibility(0x1001u, 0x2000u, 0x40u) == SOURCE_MISALIGNED);
        expect(check_buffer_copy_eligibility(0x1000u, 0x2002u, 0x40u) == DESTINATION_MISALIGNED);
        // An address range that does not fit a VkDeviceAddress cannot be
        // described at all.
        expect(check_buffer_copy_eligibility(
                   UINT64_MAX - 3u, 0x2000u, 0x40u) == RANGE_OVERFLOW);
        expect(check_buffer_copy_eligibility(
                   0x1000u, UINT64_MAX - 3u, 0x40u) == RANGE_OVERFLOW);
        // Every status is named, because the encoder reports it.
        expect(std::string_view{buffer_copy_eligibility_name(ELIGIBLE)} == "eligible");
        expect(std::string_view{buffer_copy_eligibility_name(EMPTY)} != "");
        expect(std::string_view{buffer_copy_eligibility_name(SIZE_MISALIGNED)} != "");
        expect(std::string_view{buffer_copy_eligibility_name(SOURCE_MISALIGNED)} != "");
        expect(std::string_view{buffer_copy_eligibility_name(DESTINATION_MISALIGNED)} != "");
        expect(std::string_view{buffer_copy_eligibility_name(RANGE_OVERFLOW)} != "");
    };

    // -----------------------------------------------------------------------
    // Aliasing rules of the two batched forms
    // -----------------------------------------------------------------------
    "aliasing_rules"_test = [] {
        namespace d = vk_ext_copy_test_detail;
        // Disjoint copies satisfy both rules.
        expect(indirect_copy_ranges_honor_aliasing_rule(
            d::disjoint_ranges.data(), d::disjoint_ranges.size()));
        expect(device_address_copy_ranges_honor_aliasing_rule(
            d::disjoint_ranges.data(), d::disjoint_ranges.size()));
        // One source into many destinations: the indirect rule allows it
        // (VUID-VkCopyMemoryIndirectInfoKHR-copyAddressRange-10945 only forbids
        // destination/destination and source/destination overlap), the device
        // address rule does not (VUID-VkCopyDeviceMemoryInfoKHR-srcRange-13015).
        expect(indirect_copy_ranges_honor_aliasing_rule(
            d::shared_source_ranges.data(), d::shared_source_ranges.size()));
        expect(!device_address_copy_ranges_honor_aliasing_rule(
            d::shared_source_ranges.data(), d::shared_source_ranges.size()));
        // Two copies into one destination: neither form.
        expect(!indirect_copy_ranges_honor_aliasing_rule(
            d::shared_destination_ranges.data(),
            d::shared_destination_ranges.size()));
        expect(!device_address_copy_ranges_honor_aliasing_rule(
            d::shared_destination_ranges.data(),
            d::shared_destination_ranges.size()));
        // In-place copies: neither form.
        expect(!indirect_copy_ranges_honor_aliasing_rule(
            d::in_place_ranges.data(), d::in_place_ranges.size()));
        expect(!device_address_copy_ranges_honor_aliasing_rule(
            d::in_place_ranges.data(), d::in_place_ranges.size()));
        // Empty batches trivially satisfy both.
        expect(indirect_copy_ranges_honor_aliasing_rule(nullptr, 0u));
        expect(device_address_copy_ranges_honor_aliasing_rule(nullptr, 0u));
        // A failing indirect rule always implies a failing device-address rule,
        // which is why a rejected indirect batch never falls through to the
        // stricter form.
        expect(!indirect_copy_ranges_honor_aliasing_rule(
                   d::shared_destination_ranges.data(),
                   d::shared_destination_ranges.size()) &&
               !device_address_copy_ranges_honor_aliasing_rule(
                   d::shared_destination_ranges.data(),
                   d::shared_destination_ranges.size()));
        // Half-open ranges: a range that ends exactly where another begins does
        // not overlap.
        std::array adjacent{
            BufferCopyRange{.src_address = 0x1000u, .dst_address = 0x9000u, .size = 0x100u},
            BufferCopyRange{.src_address = 0x1100u, .dst_address = 0x9100u, .size = 0x100u}};
        expect(!address_ranges_overlap(adjacent[0].src_address, adjacent[0].size,
                                       adjacent[1].src_address, adjacent[1].size));
        expect(indirect_copy_ranges_honor_aliasing_rule(
            adjacent.data(), adjacent.size()));
    };

    // -----------------------------------------------------------------------
    // The batch decision matrix
    // -----------------------------------------------------------------------
    "batch_decision_matrix"_test = [] {
        using enum BufferCopyBatchMode;
        using enum BufferCopyBatchStatus;
        namespace d = vk_ext_copy_test_detail;

        // Fully enabled: one indirect command covers the whole batch.
        auto indirect = plan_buffer_copy_batch(d::enabled_input(8u));
        expect(indirect.mode == INDIRECT_COMMANDS);
        expect(indirect.status == BATCHED_INDIRECT_COMMANDS);
        expect(indirect.uses_indirect_commands());
        expect(!indirect.uses_direct_copies());
        expect(!indirect.uses_device_address_regions());
        expect(indirect.batched_copy_count == 8u);
        expect(indirect.direct_copy_count == 0u);
        expect(indirect.stride == indirect_copy_command_stride);
        expect(indirect.command_array_size_bytes ==
               8u * indirect_copy_command_stride);

        // Neither extension enabled: exactly the pre-existing per-copy path.
        auto none = d::enabled_input(8u);
        none.indirect_extension_enabled = false;
        none.device_address_commands_enabled = false;
        auto direct = plan_buffer_copy_batch(none);
        expect(direct.mode == DIRECT_COPIES);
        expect(direct.status == FALLBACK_EXTENSIONS_DISABLED);
        expect(direct.uses_direct_copies());
        expect(direct.batched_copy_count == 0u);
        expect(direct.direct_copy_count == 8u);
        expect(direct.command_array_size_bytes == 0u);

        // VK_KHR_copy_memory_indirect is switched off/unsupported but
        // VK_KHR_device_address_commands is available: one
        // vkCmdCopyMemoryKHR covers the batch.
        auto addresses_only = d::enabled_input(8u);
        addresses_only.indirect_extension_enabled = false;
        auto regions = plan_buffer_copy_batch(addresses_only);
        expect(regions.mode == DEVICE_ADDRESS_REGIONS);
        expect(regions.status == BATCHED_DEVICE_ADDRESS_REGIONS);
        expect(regions.uses_device_address_regions());
        expect(regions.batched_copy_count == 8u);
        expect(regions.direct_copy_count == 0u);

        // The command pool's queue type is not in supportedQueues
        // (VUID-vkCmdCopyMemoryIndirectKHR-commandBuffer-10936): the indirect
        // command may not be recorded, the device-address form may.
        auto unsupported_queue = d::enabled_input(8u);
        unsupported_queue.indirect_queue_supported = false;
        expect(plan_buffer_copy_batch(unsupported_queue)
                   .mode == DEVICE_ADDRESS_REGIONS);
        unsupported_queue.device_address_commands_enabled = false;
        expect(plan_buffer_copy_batch(unsupported_queue).mode == DIRECT_COPIES);

        // No buffer device address at all: neither address based form works.
        auto no_address = d::enabled_input(8u);
        no_address.device_addresses_available = false;
        auto no_address_plan = plan_buffer_copy_batch(no_address);
        expect(no_address_plan.mode == DIRECT_COPIES);
        expect(no_address_plan.status == FALLBACK_NO_DEVICE_ADDRESS);
        expect(no_address_plan.direct_copy_count == 8u);

        // Too small: below the threshold the batched command is not worth it and
        // every copy stays on the direct path.
        for (auto count = 1u; count < indirect_copy_minimum_batch_size; ++count) {
            auto small = plan_buffer_copy_batch(d::enabled_input(count));
            expect(small.mode == DIRECT_COPIES);
            expect(small.status == FALLBACK_BATCH_TOO_SMALL);
            expect(small.direct_copy_count == count);
            expect(small.batched_copy_count == 0u);
        }
        // The threshold itself is already batched.
        auto at_threshold = plan_buffer_copy_batch(
            d::enabled_input(indirect_copy_minimum_batch_size));
        expect(at_threshold.mode == INDIRECT_COMMANDS);
        expect(at_threshold.batched_copy_count ==
               indirect_copy_minimum_batch_size);

        // Empty batch: nothing to record at all.
        auto empty = plan_buffer_copy_batch(d::enabled_input(0u));
        expect(empty.mode == DIRECT_COPIES);
        expect(empty.status == FALLBACK_EMPTY_BATCH);
        expect(empty.batched_copy_count == 0u);
        expect(empty.direct_copy_count == 0u);

        // No backend command array (allocation failure): the indirect form is
        // unavailable, the device-address form still is.
        auto no_array = d::enabled_input(8u);
        no_array.record_capacity = 0u;
        auto no_array_plan = plan_buffer_copy_batch(no_array);
        expect(no_array_plan.mode == DEVICE_ADDRESS_REGIONS);
        no_array.device_address_commands_enabled = false;
        auto no_array_direct = plan_buffer_copy_batch(no_array);
        expect(no_array_direct.mode == DIRECT_COPIES);
        expect(no_array_direct.status == FALLBACK_COMMAND_ARRAY_UNAVAILABLE);
        expect(no_array_direct.direct_copy_count == 8u);

        // A batch larger than the array: the encoder flushes at capacity, so this
        // can only happen without an array, and it must never truncate a copy.
        auto oversized = d::enabled_input(
            indirect_copy_command_capacity + 1u);
        auto oversized_plan = plan_buffer_copy_batch(oversized);
        // The device-address form still covers the whole batch.
        expect(oversized_plan.mode == DEVICE_ADDRESS_REGIONS);
        oversized.device_address_commands_enabled = false;
        auto oversized_direct = plan_buffer_copy_batch(oversized);
        expect(oversized_direct.mode == DIRECT_COPIES);
        expect(oversized_direct.status == FALLBACK_COMMAND_ARRAY_UNAVAILABLE);
        expect(oversized_direct.direct_copy_count ==
               indirect_copy_command_capacity + 1u);

        // A command array whose device address is not usable
        // (VUID-VkCopyMemoryIndirectInfoKHR-copyAddressRange-10942). The
        // device-address form is switched off here so the array-specific status
        // stays observable.
        auto unaligned_array = d::enabled_input(8u);
        unaligned_array.device_address_commands_enabled = false;
        unaligned_array.command_array_address = 0x1'0002u;
        expect(plan_buffer_copy_batch(unaligned_array)
                   .status == FALLBACK_COMMAND_ARRAY_MISALIGNED);
        // A zero address means "there is no command array at all".
        unaligned_array.command_array_address = 0u;
        expect(plan_buffer_copy_batch(unaligned_array)
                   .status == FALLBACK_COMMAND_ARRAY_UNAVAILABLE);
        // With the extension enabled, an unusable array still leaves the
        // device-address form available.
        auto unaligned_with_addresses = d::enabled_input(8u);
        unaligned_with_addresses.command_array_address = 0x1'0002u;
        expect(plan_buffer_copy_batch(unaligned_with_addresses)
                   .mode == DEVICE_ADDRESS_REGIONS);

        // Aliasing: the indirect form keeps a shared source, the device-address
        // form does not, and a batch that violates both stays direct.
        auto shared_source = d::enabled_input(4u);
        shared_source.device_address_aliasing_rule_satisfied = false;
        expect(plan_buffer_copy_batch(shared_source).mode == INDIRECT_COMMANDS);
        auto shared_destination = d::enabled_input(4u);
        shared_destination.indirect_aliasing_rule_satisfied = false;
        shared_destination.device_address_aliasing_rule_satisfied = false;
        auto aliasing_plan = plan_buffer_copy_batch(shared_destination);
        expect(aliasing_plan.mode == DIRECT_COPIES);
        expect(aliasing_plan.status == FALLBACK_ALIASING);
        expect(aliasing_plan.direct_copy_count == 4u);
        // Without the indirect extension the same batch may still use the
        // device-address form when that rule holds.
        auto regions_ok = d::enabled_input(4u);
        regions_ok.indirect_extension_enabled = false;
        regions_ok.indirect_aliasing_rule_satisfied = false;
        expect(plan_buffer_copy_batch(regions_ok)
                   .mode == DEVICE_ADDRESS_REGIONS);
        regions_ok.device_address_aliasing_rule_satisfied = false;
        expect(plan_buffer_copy_batch(regions_ok).mode == DIRECT_COPIES);

        // Every status is named, because the encoder reports it in its asserts.
        expect(std::string_view{
                   buffer_copy_batch_status_name(
                       BATCHED_INDIRECT_COMMANDS)} ==
               "indirect copy commands");
        expect(std::string_view{
                   buffer_copy_batch_status_name(
                       BATCHED_DEVICE_ADDRESS_REGIONS)} ==
               "device address copy regions");
        expect(std::string_view{
                   buffer_copy_batch_status_name(
                       FALLBACK_COMMAND_ARRAY_UNAVAILABLE)} != "");
        expect(std::string_view{
                   buffer_copy_batch_status_name(
                       FALLBACK_EXTENSIONS_DISABLED)} != "");
    };

    // -----------------------------------------------------------------------
    // The fallback must record exactly the same copies as the direct path
    // -----------------------------------------------------------------------
    "batch_copy_count_invariant"_test = [] {
        namespace d = vk_ext_copy_test_detail;
        auto checked = 0u;
        for (auto indirect_enabled : {false, true}) {
            for (auto queue_supported : {false, true}) {
                for (auto addresses_enabled : {false, true}) {
                    for (auto device_addresses : {false, true}) {
                        for (auto count : {0u, 1u, 2u, 3u, 4u, 8u,
                                           indirect_copy_command_capacity,
                                           indirect_copy_command_capacity + 1u}) {
                            for (auto capacity : {0u, indirect_copy_command_capacity}) {
                                for (auto array_address : {0x1'0001u, 0u, 0x2'0000u}) {
                                    for (auto aliasing : {false, true}) {
                                        auto input = BufferCopyBatchInput{
                                            .indirect_extension_enabled = indirect_enabled,
                                            .indirect_queue_supported = queue_supported,
                                            .device_address_commands_enabled = addresses_enabled,
                                            .device_addresses_available = device_addresses,
                                            .candidate_count = count,
                                            .minimum_batch_size = indirect_copy_minimum_batch_size,
                                            .record_capacity = capacity,
                                            .command_array_address = array_address,
                                            .indirect_aliasing_rule_satisfied = aliasing,
                                            .device_address_aliasing_rule_satisfied = aliasing};
                                        auto plan = plan_buffer_copy_batch(input);
                                        ++checked;
                                        // No copy is ever lost or duplicated.
                                        expect(plan.batched_copy_count +
                                                   plan.direct_copy_count ==
                                               count);
                                        auto recorded = d::simulate_recording(plan);
                                        expect(recorded.copy_count == count);
                                        // A batched plan covers the whole batch in
                                        // one command and keeps nothing direct.
                                        if (!plan.uses_direct_copies()) {
                                            expect(plan.batched_copy_count == count);
                                            expect(plan.direct_copy_count == 0u);
                                            expect(recorded.command_count == 1u);
                                            expect(plan.stride ==
                                                   (plan.uses_indirect_commands() ?
                                                        indirect_copy_command_stride :
                                                        0u));
                                            expect(plan.command_array_size_bytes ==
                                                   plan.batched_copy_count *
                                                       plan.stride);
                                            // Only a sufficient, supported batch is
                                            // ever batched.
                                            expect(count >=
                                                   indirect_copy_minimum_batch_size);
                                            expect(device_addresses);
                                            expect(aliasing);
                                            expect(plan.uses_indirect_commands() ?
                                                       (indirect_enabled &&
                                                        queue_supported &&
                                                        array_address % 4u == 0u &&
                                                        array_address != 0u &&
                                                        capacity >= count) :
                                                       addresses_enabled);
                                        } else {
                                            // The fallback never batches anything,
                                            // so its command count equals the copy
                                            // count of the pre-existing path.
                                            expect(plan.batched_copy_count == 0u);
                                            expect(plan.direct_copy_count == count);
                                            expect(recorded.command_count == count);
                                            expect(plan.command_array_size_bytes == 0u);
                                            expect(plan.stride == 0u);
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        expect(checked > 0u);
    };

    // -----------------------------------------------------------------------
    // VK_KHR_device_address_commands: the indirect dispatch gate
    // -----------------------------------------------------------------------
    "device_address_dispatch_layout"_test = [] {
        // vkCmdDispatchIndirect2KHR reads exactly one VkDispatchIndirectCommand,
        // whose size the SPIR-V side already assumes.
        expect(dispatch_indirect_command_size ==
               sizeof(VkDispatchIndirectCommand));
        expect(dispatch_indirect_command_size == 12u);
        expect(device_address_alignment == 4u);
    };

    "device_address_dispatch_matrix"_test = [] {
        using enum DeviceAddressIndirectDispatchStatus;
        constexpr auto base_address = uint64_t{0x7fff'0000'0000u};
        // Enabled, addressable and 4-byte aligned: vkCmdDispatchIndirect2KHR.
        auto plan = plan_device_address_indirect_dispatch(
            true, true, base_address, 0x20u, dispatch_indirect_command_size);
        expect(plan.use_device_address());
        expect(static_cast<bool>(plan));
        expect(plan.status == DEVICE_ADDRESS);
        expect(plan.range.address == base_address + 0x20u);
        expect(plan.range.size == dispatch_indirect_command_size);
        // A backend-owned storage buffer makes the storage-buffer usage
        // attestable (VUID-...-addressRange-13122).
        expect(plan.address_flags ==
               VK_ADDRESS_COMMAND_STORAGE_BUFFER_USAGE_BIT_KHR);
        auto info = make_dispatch_indirect_2_info(plan);
        expect(info.sType == VK_STRUCTURE_TYPE_DISPATCH_INDIRECT_2_INFO_KHR);
        expect(info.pNext == nullptr);
        expect(info.addressRange.address == plan.range.address);
        expect(info.addressRange.size == plan.range.size);
        expect(info.addressFlags == plan.address_flags);

        // An unknown classification uses the always-valid UNKNOWN flag.
        auto unknown = plan_device_address_indirect_dispatch(
            true, true, base_address, 0u,
            dispatch_indirect_command_size, false);
        expect(unknown.use_device_address());
        expect(unknown.address_flags ==
               VK_ADDRESS_COMMAND_UNKNOWN_STORAGE_BUFFER_USAGE_BIT_KHR);

        // The extension is unsupported, switched off, or the logical device was
        // imported: the pre-existing vkCmdDispatchIndirect stays.
        auto disabled = plan_device_address_indirect_dispatch(
            false, true, base_address, 0u, dispatch_indirect_command_size);
        expect(!disabled.use_device_address());
        expect(!static_cast<bool>(disabled));
        expect(disabled.status == FALLBACK_EXTENSION_DISABLED);
        expect(disabled.range.address == 0u);
        expect(disabled.range.size == 0u);

        // No buffer device address (device without the feature, or a buffer
        // created without VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT).
        expect(plan_device_address_indirect_dispatch(
                   true, false, base_address, 0u,
                   dispatch_indirect_command_size)
                   .status == FALLBACK_NO_DEVICE_ADDRESS);
        // A zero base address is an unavailable address
        // (VUID-...-size-11411), not an aligned one.
        expect(plan_device_address_indirect_dispatch(
                   true, true, 0u, 0u, dispatch_indirect_command_size)
                   .status == FALLBACK_ADDRESS_OVERFLOW);
        // address + offset must not overflow a VkDeviceAddress.
        expect(plan_device_address_indirect_dispatch(
                   true, true, UINT64_MAX - 8u, 16u,
                   dispatch_indirect_command_size)
                   .status == FALLBACK_ADDRESS_OVERFLOW);
        // VUID-...-addressRange-13109: the address must be a multiple of 4.
        expect(plan_device_address_indirect_dispatch(
                   true, true, base_address + 2u, 0u,
                   dispatch_indirect_command_size)
                   .status == FALLBACK_ADDRESS_MISALIGNED);
        expect(plan_device_address_indirect_dispatch(
                   true, true, base_address, 6u,
                   dispatch_indirect_command_size)
                   .status == FALLBACK_ADDRESS_MISALIGNED);
        // VUID-vkCmdDispatchIndirect2KHR-pInfo-13050: the range must hold a whole
        // VkDispatchIndirectCommand.
        expect(plan_device_address_indirect_dispatch(
                   true, true, base_address, 0u,
                   dispatch_indirect_command_size - 1u)
                   .status == FALLBACK_RANGE_TOO_SMALL);
        expect(plan_device_address_indirect_dispatch(
                   true, true, base_address, 0u,
                   dispatch_indirect_command_size)
                   .use_device_address());
        // Every status is named, because the encoder reports it.
        expect(std::string_view{device_address_indirect_dispatch_status_name(
                   DEVICE_ADDRESS)} == "device address");
        expect(std::string_view{device_address_indirect_dispatch_status_name(
                   FALLBACK_EXTENSION_DISABLED)} != "");
        expect(std::string_view{device_address_indirect_dispatch_status_name(
                   FALLBACK_NO_DEVICE_ADDRESS)} != "");
        expect(std::string_view{device_address_indirect_dispatch_status_name(
                   FALLBACK_ADDRESS_OVERFLOW)} != "");
        expect(std::string_view{device_address_indirect_dispatch_status_name(
                   FALLBACK_ADDRESS_MISALIGNED)} != "");
        expect(std::string_view{device_address_indirect_dispatch_status_name(
                   FALLBACK_RANGE_TOO_SMALL)} != "");
    };

    // The address of every prepared dispatch record must stay inside the buffer
    // range the plan describes: the encoder walks the prepared command array one
    // vulkan_command_size block at a time, and each block is larger than the
    // header the device reads.
    "device_address_dispatch_stride"_test = [] {
        constexpr auto base_address = uint64_t{0x1000'0000u};
        constexpr auto record_stride = uint64_t{16u};
        constexpr auto record_count = 4u;
        for (auto index = 0u; index < record_count; ++index) {
            auto plan = plan_device_address_indirect_dispatch(
                true, true, base_address, index * record_stride,
                dispatch_indirect_command_size);
            expect(plan.use_device_address());
            expect(plan.range.address == base_address + index * record_stride);
            expect(plan.range.address % device_address_alignment == 0u);
            expect(plan.range.size == dispatch_indirect_command_size);
            expect(plan.range.size <= record_stride);
        }
    };

    // -----------------------------------------------------------------------
    // On-device verification
    //
    // The planners above are host-only, so this section runs the encoder itself:
    // a batch of disjoint buffer-to-buffer copies must be exact with and without
    // VK_KHR_copy_memory_indirect, the enabled device must report the extension
    // as *active* (the backend only marks it once the indirect command is really
    // recorded), and a batch below the threshold must keep it inactive.
    // A host without the Vulkan backend skips this section.
    // -----------------------------------------------------------------------
    "vk_buffer_copy_batching"_test = [&] {
        namespace d = vk_ext_copy_test_detail;
        using namespace luisa;
        using namespace luisa::compute;
        Context context{argv[0]};
        auto backends = context.installed_backends();
        auto has_vk = std::any_of(
            backends.begin(), backends.end(),
            [](auto &&name) { return name == "vk"; });
        if (!has_vk) {
            LUISA_INFO("The Vulkan backend is not installed; skipping the "
                       "on-device copy batching checks.");
            return;
        }
        // 8 x 1024 uint32 elements: every copy size and offset is a multiple of
        // 4 bytes, so every copy is expressible as an indirect copy command.
        constexpr auto element_count = size_t{8192u};
        constexpr auto fast_path_copies = 8u;
        constexpr auto below_threshold_copies = 2u;// < indirect_copy_minimum_batch_size
        constexpr auto chunk = element_count / fast_path_copies;
        luisa::vector<uint32_t> host_source(element_count);
        for (auto i = size_t{0u}; i < element_count; i++) {
            host_source[i] = static_cast<uint32_t>(i + 1u);
        }
        auto const *extension_name = "VK_KHR_copy_memory_indirect";
        for (auto enabled : {false, true}) {
            DeviceConfig device_config{};
            device_config.extension =
                luisa::make_unique<d::CopyMemoryIndirectConfig>(enabled);
            // Only one Vulkan backend device may be alive at a time, so each
            // configuration is created and destroyed inside its own scope.
            auto device = context.create_device("vk", &device_config);
            auto *status = device.extension<VulkanExtensionStatusExt>();
            auto supported = status->is_supported(extension_name);
            auto negotiated = status->is_enabled(extension_name);
            expect(!status->is_active(extension_name))
                << "no batch has been recorded yet";
            auto source = device.create_buffer<uint32_t>(element_count);
            auto destination = device.create_buffer<uint32_t>(element_count);
            auto stream = device.create_stream();
            auto readback = luisa::vector<uint32_t>(element_count, 0u);
            stream << source.copy_from(luisa::span{host_source});
            // One command list carries the whole batch; a single copy per
            // dispatch could never be batched.
            auto below_threshold = CommandList{};
            for (auto i = 0u; i < below_threshold_copies; i++) {
                below_threshold
                    << destination.view(i * chunk, chunk)
                           .copy_from(source.view(i * chunk, chunk));
            }
            stream << below_threshold.commit() << synchronize();
            // Below the batching threshold the copies stay on the pre-existing
            // one-command-per-copy path, so nothing may report the extension.
            expect(!status->is_active(extension_name))
                << "a batch below the threshold must not use the extension";
            // Above the threshold, all copies are disjoint so the reorder pass
            // keeps them in one layer and the encoder batches them.
            auto batch = CommandList{};
            for (auto i = 0u; i < fast_path_copies; i++) {
                batch << destination.view(i * chunk, chunk)
                             .copy_from(source.view(i * chunk, chunk));
            }
            stream << batch.commit()
                   << destination.copy_to(luisa::span{readback})
                   << synchronize();
            auto mismatch_count = size_t{0u};
            for (auto i = size_t{0u}; i < element_count; i++) {
                mismatch_count += readback[i] == host_source[i] ? 0u : 1u;
            }
            expect(mismatch_count == 0u)
                << "batched buffer copies must be exact";
            auto active = status->is_active(extension_name);
            if (enabled && negotiated) {
                expect(active)
                    << "the recorded indirect copy batch must be reported active";
            } else {
                expect(!active)
                    << "a switched off or unsupported extension must stay inactive";
            }
            // ---- more than one batch in a single command buffer -------------
            // The device reads the records of vkCmdCopyMemoryIndirectKHR when
            // the command executes, i.e. after the whole command buffer was
            // recorded, so every batch of one command buffer must own its own
            // records inside the backend command array. Three dependent copy
            // groups are forced into three barrier layers (each of them reads
            // what the previous one wrote), i.e. three separate batches.
            auto source_b = device.create_buffer<uint32_t>(element_count);
            auto intermediate = device.create_buffer<uint32_t>(element_count);
            auto destination_b = device.create_buffer<uint32_t>(element_count);
            luisa::vector<uint32_t> host_source_b(element_count);
            for (auto i = size_t{0u}; i < element_count; i++) {
                host_source_b[i] = 0x8000'0000u + static_cast<uint32_t>(i);
            }
            auto host_intermediate = luisa::vector<uint32_t>(element_count, 0u);
            auto host_destination_b = luisa::vector<uint32_t>(element_count, 0u);
            stream << source_b.copy_from(luisa::span{host_source_b});
            auto multi_batch = CommandList{};
            for (auto i = 0u; i < fast_path_copies; i++) {
                multi_batch << destination_b.view(i * chunk, chunk)
                                   .copy_from(source.view(i * chunk, chunk));
            }
            for (auto i = 0u; i < fast_path_copies; i++) {
                multi_batch << intermediate.view(i * chunk, chunk)
                                   .copy_from(destination_b.view(i * chunk, chunk));
            }
            for (auto i = 0u; i < fast_path_copies; i++) {
                multi_batch << destination_b.view(i * chunk, chunk)
                                   .copy_from(source_b.view(i * chunk, chunk));
            }
            stream << multi_batch.commit()
                   << destination_b.copy_to(luisa::span{host_destination_b})
                   << intermediate.copy_to(luisa::span{host_intermediate})
                   << synchronize();
            auto multi_batch_mismatch_count = size_t{0u};
            for (auto i = size_t{0u}; i < element_count; i++) {
                multi_batch_mismatch_count +=
                    host_destination_b[i] == host_source_b[i] ? 0u : 1u;
                multi_batch_mismatch_count +=
                    host_intermediate[i] == host_source[i] ? 0u : 1u;
            }
            expect(multi_batch_mismatch_count == 0u)
                << "every batch of one command buffer must keep its own copy "
                   "records";
            // ---- more records than the command array can hold --------------
            // Three dependent groups of 100 copies need 300 records while the
            // array holds indirect_copy_command_capacity. The batches that still
            // fit are recorded as indirect commands, the rest must fall back to
            // one command per copy - and the result must stay exact either way.
            constexpr auto many_copies = 100u;
            constexpr auto many_chunk = 64u;
            constexpr auto many_elements = many_copies * many_chunk;
            static_assert(3u * many_copies > indirect_copy_command_capacity);
            auto host_intermediate_b = luisa::vector<uint32_t>(element_count, 0u);
            auto host_destination_c = luisa::vector<uint32_t>(element_count, 0u);
            auto overflow = CommandList{};
            for (auto i = 0u; i < many_copies; i++) {
                overflow << intermediate.view(i * many_chunk, many_chunk)
                                .copy_from(source.view(i * many_chunk, many_chunk));
            }
            for (auto i = 0u; i < many_copies; i++) {
                overflow << destination_b.view(i * many_chunk, many_chunk)
                                .copy_from(intermediate.view(i * many_chunk, many_chunk));
            }
            for (auto i = 0u; i < many_copies; i++) {
                overflow << intermediate.view(i * many_chunk, many_chunk)
                                .copy_from(source_b.view(i * many_chunk, many_chunk));
            }
            stream << overflow.commit()
                   << intermediate.copy_to(luisa::span{host_intermediate_b})
                   << destination_b.copy_to(luisa::span{host_destination_c})
                   << synchronize();
            auto overflow_mismatch_count = size_t{0u};
            for (auto i = size_t{0u}; i < many_elements; i++) {
                overflow_mismatch_count +=
                    host_intermediate_b[i] == host_source_b[i] ? 0u : 1u;
                overflow_mismatch_count +=
                    host_destination_c[i] == host_source[i] ? 0u : 1u;
            }
            expect(overflow_mismatch_count == 0u)
                << "a batch that no longer fits the command array must fall back "
                   "to per-copy commands, not to stale records";
            LUISA_INFO("buffer copy batching: enabled={}, supported={}, "
                       "negotiated={}, active={}, "
                       "multi-batch mismatches={}, array-overflow mismatches={}",
                       enabled, supported, negotiated, active,
                       multi_batch_mismatch_count, overflow_mismatch_count);
        }
    };

    return 0;
}
