#pragma once

#include <cstdint>

namespace lc::vk::detail {

inline constexpr uint32_t amd_pci_vendor_id = 0x1002u;

struct VulkanFloatAtomicCodegenPolicy {
    bool native_xir_spirv_prefers_software_buffer_float32_rmw{false};

    [[nodiscard]] constexpr uint64_t cache_key() const noexcept {
        return static_cast<uint64_t>(
            native_xir_spirv_prefers_software_buffer_float32_rmw);
    }
};

// AMD's native storage-buffer float32 arithmetic atomics can be substantially
// slower than a uint32 compare-exchange loop under heavy contention. Keep the
// physical-device feature snapshot truthful and express this as an independent
// code-generation policy. Other vendors retain native float atomics when the
// corresponding Vulkan feature is enabled.
[[nodiscard]] constexpr VulkanFloatAtomicCodegenPolicy
plan_vulkan_float_atomic_codegen(uint32_t pci_vendor_id) noexcept {
    return {
        .native_xir_spirv_prefers_software_buffer_float32_rmw =
            pci_vendor_id == amd_pci_vendor_id};
}

}// namespace lc::vk::detail
