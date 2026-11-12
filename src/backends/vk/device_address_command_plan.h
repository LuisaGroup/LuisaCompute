#pragma once

#include <cstddef>
#include <cstdint>
#include <vulkan/vulkan_core.h>

namespace lc::vk::detail {

// Host-only planning for the VK_KHR_device_address_commands fast paths.
//
// The extension replaces a few VkBuffer-handle commands by VkDeviceAddress
// forms. The backend only uses two of them:
//
//   * vkCmdDispatchIndirect2KHR  - the indirect dispatch of the compute queue
//                                  (see plan_device_address_indirect_dispatch)
//   * vkCmdCopyMemoryKHR         - one command for a batch of buffer-to-buffer
//                                  copies that live in *different* buffers
//                                  (see copy_command_plan.h)
//
// Both are optimizations only: `optional_extension_enabled(DEVICE_ADDRESS_COMMANDS)`
// gates them, every planner below reports an explicit fallback, and the fallback
// is exactly the pre-existing command (vkCmdDispatchIndirect / one
// vkCmdCopyBuffer per copy).
//
// Everything here is `constexpr` and free of device state so the decision matrix
// is unit testable without a Vulkan device.
//
// Not used by this backend: vkCmdBindVertexBuffers3KHR / vkCmdBindIndexBuffer3KHR
// and vkCmdDrawIndirect2KHR. The raster path binds vertex/index buffers with the
// classic commands and draws with non-indirect vkCmdDraw/vkCmdDrawIndexed, so
// there is no device-address variant to substitute.

// ---------------------------------------------------------------------------
// VkAddressCommandFlagsKHR
// ---------------------------------------------------------------------------

// The two "storage buffer usage" flags are mutually exclusive
// (VUID-...-addressFlags-13100):
//
//   * STORAGE_BUFFER_USAGE_BIT_KHR must be set when every buffer backing the
//     range was created with VK_BUFFER_USAGE_STORAGE_BUFFER_BIT
//     (VUID-...-addressRange-13122), and must not be set when none was
//     (VUID-...-addressRange-13123).
//   * UNKNOWN_STORAGE_BUFFER_USAGE_BIT_KHR is the always-valid answer for a
//     range whose backing buffers the backend cannot classify (an imported
//     VkBuffer, for example).
//
// VK_ADDRESS_COMMAND_PROTECTED_BIT_KHR is never needed (the backend never
// creates protected memory) and VK_ADDRESS_COMMAND_FULLY_BOUND_BIT_KHR is only a
// hint, so both stay clear.
[[nodiscard]] constexpr VkAddressCommandFlagsKHR
device_address_command_flags(bool storage_buffer_usage) noexcept {
    return storage_buffer_usage
               ? static_cast<VkAddressCommandFlagsKHR>(
                     VK_ADDRESS_COMMAND_STORAGE_BUFFER_USAGE_BIT_KHR)
               : static_cast<VkAddressCommandFlagsKHR>(
                     VK_ADDRESS_COMMAND_UNKNOWN_STORAGE_BUFFER_USAGE_BIT_KHR);
}

// ---------------------------------------------------------------------------
// vkCmdDispatchIndirect2KHR
// ---------------------------------------------------------------------------

// sizeof(VkDispatchIndirectCommand): the group counts the device reads from the
// address range. VUID-vkCmdDispatchIndirect2KHR-pInfo-13050 requires the range
// size to be at least this large.
inline constexpr uint64_t dispatch_indirect_command_size =
    sizeof(VkDispatchIndirectCommand);
static_assert(dispatch_indirect_command_size == 12u,
              "VkDispatchIndirectCommand must stay the fixed three-word "
              "indirect dispatch header the SPIR-V side already reads.");

// VUID-...-addressRange-13109: addressRange.address must be a multiple of 4.
inline constexpr uint64_t device_address_alignment = 4u;

enum class DeviceAddressIndirectDispatchStatus : uint32_t {
    // Fast path: record vkCmdDispatchIndirect2KHR with the planned range.
    DEVICE_ADDRESS = 0u,
    // VK_KHR_device_address_commands is unsupported on this physical device,
    // switched off through VulkanDeviceConfigExt, or the logical device is
    // imported (its feature chain cannot be attested).
    FALLBACK_EXTENSION_DISABLED,
    // The device does not expose buffer device addresses, or the buffer that
    // holds the dispatch parameters was not created with
    // VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT, so no legal address exists.
    FALLBACK_NO_DEVICE_ADDRESS,
    // base address + offset overflowed a VkDeviceAddress.
    FALLBACK_ADDRESS_OVERFLOW,
    // The resulting address is not 4-byte aligned, so the address range cannot
    // be described.
    FALLBACK_ADDRESS_MISALIGNED,
    // Fewer than sizeof(VkDispatchIndirectCommand) bytes remain in the buffer.
    FALLBACK_RANGE_TOO_SMALL,
};

[[nodiscard]] constexpr const char *
device_address_indirect_dispatch_status_name(
    DeviceAddressIndirectDispatchStatus status) noexcept {
    switch (status) {
        case DeviceAddressIndirectDispatchStatus::DEVICE_ADDRESS:
            return "device address";
        case DeviceAddressIndirectDispatchStatus::FALLBACK_EXTENSION_DISABLED:
            return "fallback: VK_KHR_device_address_commands is not enabled";
        case DeviceAddressIndirectDispatchStatus::FALLBACK_NO_DEVICE_ADDRESS:
            return "fallback: no buffer device address";
        case DeviceAddressIndirectDispatchStatus::FALLBACK_ADDRESS_OVERFLOW:
            return "fallback: device address overflow";
        case DeviceAddressIndirectDispatchStatus::FALLBACK_ADDRESS_MISALIGNED:
            return "fallback: device address is not 4-byte aligned";
        case DeviceAddressIndirectDispatchStatus::FALLBACK_RANGE_TOO_SMALL:
            return "fallback: address range is shorter than the dispatch header";
    }
    return "unknown";
}

// The complete input of the indirect-dispatch decision. `buffer_address` is the
// address of the buffer that holds the indirect dispatch header (the backend's
// prepared command array), `offset` is the byte offset of one
// VkDispatchIndirectCommand inside it, and `available_bytes` is what is left in
// the buffer from `offset` (VkDeviceAddressRangeKHR may not describe a range
// that runs past the buffer, VUID-...-address-11365).
struct DeviceAddressIndirectDispatchPlan {
    DeviceAddressIndirectDispatchStatus status{
        DeviceAddressIndirectDispatchStatus::FALLBACK_EXTENSION_DISABLED};
    // Address range to hand to VkDispatchIndirect2InfoKHR::addressRange.
    VkDeviceAddressRangeKHR range{};
    // Address range flags (see device_address_command_flags).
    VkAddressCommandFlagsKHR address_flags{0u};
    [[nodiscard]] constexpr bool use_device_address() const noexcept {
        return status == DeviceAddressIndirectDispatchStatus::DEVICE_ADDRESS;
    }
    [[nodiscard]] constexpr explicit operator bool() const noexcept {
        return use_device_address();
    }
};

[[nodiscard]] constexpr DeviceAddressIndirectDispatchPlan
plan_device_address_indirect_dispatch(
    bool extension_enabled,
    bool buffer_device_address_capable,
    uint64_t buffer_address,
    uint64_t offset,
    uint64_t available_bytes,
    bool storage_buffer_usage = true) noexcept {
    constexpr auto fallback = [](DeviceAddressIndirectDispatchStatus status) noexcept {
        return DeviceAddressIndirectDispatchPlan{.status = status};
    };
    if (!extension_enabled) {
        return fallback(
            DeviceAddressIndirectDispatchStatus::FALLBACK_EXTENSION_DISABLED);
    }
    if (!buffer_device_address_capable) {
        return fallback(
            DeviceAddressIndirectDispatchStatus::FALLBACK_NO_DEVICE_ADDRESS);
    }
    // VUID-...-size-11411 forbids a non-zero size with a null address, so a
    // zero base address is an unavailable address, not an aligned one.
    if (buffer_address == 0u ||
        offset > UINT64_MAX - buffer_address) {
        return fallback(
            DeviceAddressIndirectDispatchStatus::FALLBACK_ADDRESS_OVERFLOW);
    }
    auto address = buffer_address + offset;
    if (address % device_address_alignment != 0u) {
        return fallback(
            DeviceAddressIndirectDispatchStatus::FALLBACK_ADDRESS_MISALIGNED);
    }
    if (available_bytes < dispatch_indirect_command_size) {
        return fallback(
            DeviceAddressIndirectDispatchStatus::FALLBACK_RANGE_TOO_SMALL);
    }
    return {
        .status = DeviceAddressIndirectDispatchStatus::DEVICE_ADDRESS,
        .range = VkDeviceAddressRangeKHR{
            .address = address,
            .size = dispatch_indirect_command_size},
        .address_flags = device_address_command_flags(storage_buffer_usage)};
}

// The VkDispatchIndirect2InfoKHR handed to vkCmdDispatchIndirect2KHR. Only call
// this for a plan whose `use_device_address()` is true: the caller must have
// checked the gate, because the entry point may not be loaded otherwise.
[[nodiscard]] constexpr VkDispatchIndirect2InfoKHR
make_dispatch_indirect_2_info(
    const DeviceAddressIndirectDispatchPlan &plan) noexcept {
    return VkDispatchIndirect2InfoKHR{
        .sType = VK_STRUCTURE_TYPE_DISPATCH_INDIRECT_2_INFO_KHR,
        .pNext = nullptr,
        .addressRange = plan.range,
        .addressFlags = plan.address_flags};
}

}// namespace lc::vk::detail
