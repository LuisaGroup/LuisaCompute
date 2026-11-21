#pragma once

#include <luisa/core/stl/memory.h>
#include <luisa/core/stl/string.h>
#include <luisa/core/stl/vector.h>
#include <luisa/runtime/rhi/device_interface.h>

namespace luisa::compute {

// Negotiated state of one optional Vulkan device extension.
struct VulkanExtensionStatus {
    // Extension name the backend negotiated. For extensions that have a
    // KHR/EXT or EXT/NV twin this is the spelling the physical device exposed.
    luisa::string name;
    // The physical device exposes the extension.
    bool supported{false};
    // VulkanDeviceConfigExt allows it (`enable_*()` returned true) and the
    // device was created by the backend (imported logical devices can never
    // enable optional extensions).
    bool requested{false};
    // The extension is enabled on the logical device, i.e. it is named in
    // VkDeviceCreateInfo and its feature structure carries VK_TRUE.
    bool enabled{false};
    // True when the backend actually routes work through the extension instead
    // of the fallback path. `enabled` alone only means it may do so; for
    // example an extension that only removes API friction stays inactive.
    bool active{false};
};

// Negotiated state of the two cooperative compute extensions.
//
// These are reported HERE and not in `extensions()` because the Vulkan backend
// negotiates them outside the optional-extension registry: each one carries its
// own feature chain, a KHR/NV preference order, prerequisites
// (vulkanMemoryModel + shaderReplicatedComposites for vectors) and, for
// cooperative vector, a driver-enumerated list of supported
// (input, matrix, bias, result, transpose) configurations. A `false` in
// `extensions()` for these two names is therefore *not* evidence that the device
// lacks them -- that is exactly the misreading this struct removes.
struct VulkanCooperativeStatus {
    // ---- VK_KHR_cooperative_matrix (+ VK_NV_cooperative_matrix2 / _matrix) --
    // Extension name enabled on the logical device; empty when none was.
    luisa::string matrix_extension;
    // The extension is named in VkDeviceCreateInfo and its feature structure
    // carries cooperativeMatrix == VK_TRUE.
    bool matrix_enabled{false};
    // VK_NV_cooperative_matrix2 workgroup scope was enabled on top of KHR.
    bool matrix_workgroup_scope{false};
    // `matrix_enabled` NEVER means "emitted": no LuisaCompute or gallery kernel
    // lowers to a cooperative-matrix instruction today. It is reported so a run
    // log can state what the device negotiated instead of guessing.
    // ---- VK_NV_cooperative_vector ------------------------------------------
    // The physical device exposes VK_NV_cooperative_vector with the
    // cooperativeVector feature bit and VK_SHADER_STAGE_COMPUTE_BIT support.
    bool vector_physical_support{false};
    // The SPIR-V capability's prerequisites: vulkanMemoryModel +
    // vulkanMemoryModelDeviceScope, and shaderReplicatedComposites.
    bool vector_memory_model{false};
    bool vector_replicated_composites{false};
    // Both prerequisites held, so the extension and its feature bit were enabled
    // on the logical device: the SPIR-V codegen may emit the NV instructions.
    bool vector_enabled{false};
    // A FLOAT32/FLOAT32/FLOAT32/FLOAT32 with transpose == VK_FALSE configuration
    // was enumerated. This is the ONLY configuration the current codegen can
    // emit (spirv_codegen/instruction.cpp hardcodes the FP32 component types and
    // transpose = false), so `vector_enabled && !vector_fp32_config` means
    // "enabled but not usable by this tree" -- the honest field LCV-1 (phase 9c)
    // needed and the reason a `bool` alone was not enough.
    bool vector_fp32_config{false};
    // Number of configuration entries the driver enumerated, INCLUDING the
    // packed/quantized ones the backend's corruption guard hides from its Info
    // log (those carry VkComponentTypeKHR values above the sanity bound).
    uint32_t vector_config_count{0u};
};

// Runtime view of the Vulkan backend's optional extension negotiation
// (obtained with `Device::extension<VulkanExtensionStatusExt>()`).
//
// This is a read-only diagnostic surface: it makes "is the fast path actually
// in use on this machine?" answerable without parsing backend logs, and it is
// what the Vulkan backend's own enable/disable tests assert on.
class VulkanExtensionStatusExt : public DeviceExtension {
public:
    static constexpr luisa::string_view name = "VulkanExtensionStatusExt";
    // One entry per optional extension the backend knows about, sorted by the
    // backend's stable extension order. Deliberately does NOT include
    // VK_KHR_cooperative_matrix / VK_NV_cooperative_vector: they are not
    // registry entries, so an entry for them would have to invent a
    // `requested`/`active` answer. Use cooperative_status() for those.
    [[nodiscard]] virtual luisa::span<const VulkanExtensionStatus> extensions() const noexcept = 0;
    [[nodiscard]] virtual bool is_supported(luisa::string_view extension) const noexcept = 0;
    [[nodiscard]] virtual bool is_enabled(luisa::string_view extension) const noexcept = 0;
    [[nodiscard]] virtual bool is_active(luisa::string_view extension) const noexcept = 0;
    // Negotiated state of VK_KHR_cooperative_matrix / VK_NV_cooperative_vector.
    // The per-name queries above also answer for these two names
    // (is_supported = physical support, is_requested is not exposed, is_enabled =
    // enabled on the logical device, is_active = the codegen routes work to it),
    // but only this struct distinguishes "enabled" from "usable".
    [[nodiscard]] virtual VulkanCooperativeStatus cooperative_status() const noexcept = 0;

protected:
    ~VulkanExtensionStatusExt() = default;
};

}// namespace luisa::compute
