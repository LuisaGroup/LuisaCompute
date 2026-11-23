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

// Where the reported SM (shader-multiprocessor / compute-unit) count came from.
//
// Vulkan has no physical SM-count query, so the Vulkan backend never reports one
// it did not receive from the caller. `Unknown` is the default and the answer on
// every device created without a calibration input.
enum class VulkanSmCountSource : uint32_t {
    // No value: `VulkanCapabilityStatus::sm_count` is 0 and must not be read as
    // "this device has no SMs".
    UNKNOWN = 0u,
    // The caller's externally calibrated estimate, supplied through
    // `VulkanDeviceConfigExt::calibrated_sm_count_estimate()`. It is an estimate
    // about the device, not a device attribute the backend measured.
    CALIBRATION_ESTIMATE = 1u
};

[[nodiscard]] constexpr luisa::string_view
vulkan_sm_count_source_name(VulkanSmCountSource source) noexcept {
    switch (source) {
        case VulkanSmCountSource::UNKNOWN: return "unknown";
        case VulkanSmCountSource::CALIBRATION_ESTIMATE: return "calibration_estimate";
    }
    return "unknown";
}

// Negotiated state of VK_EXT_descriptor_heap, reported separately from the flat
// extension list because it is the one optional extension whose state a
// consumer needs as a value rather than a name lookup. Every field comes from
// the same OptionalExtensionRegistry entry the flat list reports (and `active`
// from the backend's own active bits), never from a hard-coded constant.
struct VulkanDescriptorHeapStatus {
    // The physical device exposes VK_EXT_descriptor_heap.
    bool supported{false};
    // VulkanDeviceConfigExt::enable_descriptor_heap() returned true.
    bool requested{false};
    // The extension is named in VkDeviceCreateInfo and its feature chain is on.
    bool enabled{false};
    // The backend really routes descriptor storage through the extension.
    bool active{false};
};

// Numeric capability record of the Vulkan backend (plan P6.3).
//
// Every field is answered from exactly the same device properties as the
// matching `Device::query` key, so a report built from this struct and a report
// built from queries cannot disagree: for each limit the query key is
// `Device::query("<field name>")` (e.g. "max_compute_shared_memory_bytes",
// "min_subgroup_size"), and 0 here corresponds to an empty query answer.
//
// Zero/unknown semantics:
//  - MAX_MEMORY_ALLOCATION_COUNT, the six workgroup fields,
//    MAX_COMPUTE_SHARED_MEMORY_BYTES, NON_COHERENT_ATOM_SIZE,
//    MIN_STORAGE_BUFFER_OFFSET_ALIGNMENT and MAX_STORAGE_BUFFER_RANGE are
//    unconditional VkPhysicalDeviceLimits members, so they are always real
//    (Vulkan never reports 0 for them).
//  - MIN_SUBGROUP_SIZE / MAX_SUBGROUP_SIZE are **conditional**: they exist only
//    when Vulkan's subgroup-size-control properties are available
//    (VK_EXT_subgroup_size_control, core in Vulkan 1.3). On a Vulkan 1.2 device
//    without that extension both stay 0, `subgroup_size_control_available` stays
//    false, and `Device::query("min_subgroup_size"/"max_subgroup_size")` returns
//    an empty string. They are never fabricated from
//    VkPhysicalDeviceSubgroupProperties::subgroupSize, which is a single value
//    with no lower/upper bound.
//  - SUBGROUP_OPS is VkPhysicalDeviceSubgroupProperties::supportedOperations (a
//    VkSubgroupFeatureFlags bitmask, reported as an integer). It is 0 with
//    `subgroup_ops_available == false` only when the device could not be asked.
//  - SM_COUNT is 0 unless SM_COUNT_SOURCE says the caller supplied a calibrated
//    estimate (plan P6.2). It is never a value the backend measured.
struct VulkanCapabilityStatus {
    // ---- core limits (plan P6.1) -------------------------------------------
    uint32_t max_memory_allocation_count{0u};
    // 0 = unknown (see SUBGROUP_SIZE_CONTROL_AVAILABLE).
    uint32_t min_subgroup_size{0u};
    uint32_t max_subgroup_size{0u};
    // True when the two subgroup-size bounds above are real.
    bool subgroup_size_control_available{false};
    // VkSubgroupFeatureFlags bitmask; 0 = unknown when SUBGROUP_OPS_AVAILABLE is false.
    uint32_t subgroup_ops{0u};
    bool subgroup_ops_available{false};
    uint32_t max_compute_workgroup_count_x{0u};
    uint32_t max_compute_workgroup_count_y{0u};
    uint32_t max_compute_workgroup_count_z{0u};
    uint32_t max_compute_workgroup_size_x{0u};
    uint32_t max_compute_workgroup_size_y{0u};
    uint32_t max_compute_workgroup_size_z{0u};
    uint32_t max_compute_shared_memory_bytes{0u};
    // VkDeviceSize values keep their full 64-bit range; they are never narrowed.
    uint64_t non_coherent_atom_size{0u};
    uint64_t min_storage_buffer_offset_alignment{0u};
    uint64_t max_storage_buffer_range{0u};
    // ---- descriptor heap ----------------------------------------------------
    VulkanDescriptorHeapStatus descriptor_heap;
    // ---- SM count (plan P6.2) ----------------------------------------------
    // 0 = unknown; only non-zero when sm_count_source is CALIBRATION_ESTIMATE.
    uint32_t sm_count{0u};
    VulkanSmCountSource sm_count_source{VulkanSmCountSource::UNKNOWN};
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
    // Numeric capability record (plan P6.3): the 14 core limits of plan P6.1,
    // the descriptor-heap negotiation state, and the honest SM-count answer of
    // plan P6.2. Returned by value; the returned struct owns no pointer into the
    // device, so it stays valid for the caller's lifetime.
    //
    // Consistency contract: every limit field equals the corresponding
    // `Device::query` answer of the same name (0 <=> an empty answer), because
    // both are read from the one capability record the device captured at
    // creation.
    [[nodiscard]] virtual VulkanCapabilityStatus capability_status() const noexcept = 0;

protected:
    ~VulkanExtensionStatusExt() = default;
};

}// namespace luisa::compute
