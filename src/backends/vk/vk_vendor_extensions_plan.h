#pragma once
#include <cstdint>

namespace lc::vk::detail {

// ===========================================================================
// Vendor extension planning (capability inventory only)
// ===========================================================================
//
// The fix extensions below are all negotiated by
// `detail::OptionalExtensionRegistry` (src/backends/vk/vk_extension_registry.{h,cpp})
// and can be switched on through `VulkanDeviceConfigExt`, but *none* of them can be
// routed through the code this work package is allowed to touch:
//
//   VK_KHR_internally_synchronized_queues  - needs the queue create info
//   VK_ARM_tensors                         - needs a public N-D tensor API
//   VK_ARM_data_graph                      - needs a data-graph queue + session API
//   VK_QCOM_tile_shading                   - needs a tile-shading render-pass path
//   VK_QCOM_tile_memory_heap               - needs the same render-pass path
//   VK_EXT_external_memory_metal           - Apple-only platform interop
//
// This header is the design artifact for them: one entry per extension naming
// (a) the exact entry points and structures the backend would use, (b) the concrete
// blocker in this codebase, with `file:line`, and (c) the change that would unblock
// it. It deliberately contains no Vulkan and no `luisa` includes so the unit test
// can read it on any machine.
//
// ---------------------------------------------------------------------------
// Hard contract
// ---------------------------------------------------------------------------
// None of these extensions may ever be assumed enabled:
//
//   * the *only* legal gate is
//     `Device::optional_extension_enabled(detail::OptionalExtension::X)`, which is
//     false when the extension is unsupported, switched off through
//     `VulkanDeviceConfigExt`, or the logical device was imported,
//   * an entry point of a disabled extension must never be called and a structure
//     of a disabled extension must never be chained,
//   * no fast path below exists yet, so `active` is false for every entry and
//     `vendor_extension_plan_is_inert()` must hold for all of them.
//
// `active` may only be set to true by the change that introduces the real fast
// path, at the same time that the path starts calling
// `Device::mark_optional_extension_active()`.

inline constexpr const char *kVendorExtensionNeverAssumedEnabled =
    "Only Device::optional_extension_enabled(detail::OptionalExtension::X) may "
    "select this path. The extension is unsupported, switched off through "
    "VulkanDeviceConfigExt, or belongs to an imported logical device in the "
    "default case, and no entry point of it may be called and no structure of it "
    "may be chained then.";

enum class VendorExtensionId : uint32_t {
    kInternallySynchronizedQueues = 0u,
    kArmTensors,
    kArmDataGraph,
    kQcomTileShading,
    kQcomTileMemoryHeap,
    kExternalMemoryMetal,
    kCount
};

inline constexpr uint32_t kVendorExtensionCount =
    static_cast<uint32_t>(VendorExtensionId::kCount);

[[nodiscard]] constexpr const char *vendor_extension_id_name(VendorExtensionId id) noexcept {
    switch (id) {
        case VendorExtensionId::kInternallySynchronizedQueues:
            return "VK_KHR_internally_synchronized_queues";
        case VendorExtensionId::kArmTensors: return "VK_ARM_tensors";
        case VendorExtensionId::kArmDataGraph: return "VK_ARM_data_graph";
        case VendorExtensionId::kQcomTileShading: return "VK_QCOM_tile_shading";
        case VendorExtensionId::kQcomTileMemoryHeap: return "VK_QCOM_tile_memory_heap";
        case VendorExtensionId::kExternalMemoryMetal: return "VK_EXT_external_memory_metal";
        case VendorExtensionId::kCount: break;
    }
    return "unknown";
}

struct VendorExtensionPlan {
    VendorExtensionId id{VendorExtensionId::kCount};
    // Exactly the string the registry negotiates
    // (VK_*_EXTENSION_NAME, or the literal name for the platform-gated Metal one).
    const char *extension_name{nullptr};
    // `VulkanDeviceConfigExt` switch that has to return true.
    const char *config_switch{nullptr};
    // Registry identifier the backend has to branch on.
    const char *registry_identifier{nullptr};
    // The single legal gate expression.
    const char *enabled_gate{nullptr};
    // Exact Vulkan commands the backend would call.
    const char *entry_points{nullptr};
    // Exact Vulkan structures/enums it would chain or set.
    const char *structures{nullptr};
    // Feature bits that must be VK_TRUE.
    const char *features{nullptr};
    // (b) The concrete blocker in this codebase, with `file:line`.
    const char *blocker{nullptr};
    // (c) The exact change that would unblock it.
    const char *unblock_requirement{nullptr};
    // True only once the backend really routes work through the extension. Never
    // true in this work package.
    bool active{false};
    // Must stay false: nothing may treat the extension as present.
    bool assumed_enabled{false};
    // Whether a file this work package may edit can reach the extension at all.
    // False for all six: each one needs a file owned by another work package
    // (device creation, queue creation, render-pass/raster path, platform build).
    bool reachable_from_backend{false};
};

// One entry per extension. The strings are the deliverable of this work package:
// they pin the Vulkan surface *and* the exact place the backend is missing it, so
// the follow-up wiring cannot silently drift.
inline constexpr VendorExtensionPlan kVendorExtensionPlans[kVendorExtensionCount] = {
    // -----------------------------------------------------------------------
    // VK_KHR_internally_synchronized_queues
    // -----------------------------------------------------------------------
    {.id = VendorExtensionId::kInternallySynchronizedQueues,
     .extension_name = "VK_KHR_internally_synchronized_queues",
     .config_switch = "VulkanDeviceConfigExt::enable_internally_synchronized_queues()",
     .registry_identifier = "detail::OptionalExtension::INTERNALLY_SYNCHRONIZED_QUEUES",
     .enabled_gate = "device->optional_extension_enabled("
                     "detail::OptionalExtension::INTERNALLY_SYNCHRONIZED_QUEUES)",
     .entry_points = "none (queue creation flag only)",
     .structures = "VkDeviceQueueCreateInfo::flags must include "
                   "VK_DEVICE_QUEUE_CREATE_INTERNALLY_SYNCHRONIZED_BIT_KHR",
     .features = "VkPhysicalDeviceInternallySynchronizedQueuesFeaturesKHR::"
                 "internallySynchronizedQueues",
     .blocker =
         "Queue creation is in src/backends/vk/VulkanDevice.cpp:263-340, which "
         "fills VkDeviceQueueCreateInfo without ever setting ::flags, before "
         "vkCreateDevice at VulkanDevice.cpp:394. Every host-side queue contract "
         "in src/backends/vk/device.h:89-96 (the _graphics/_compute/_copy/_sparse "
         "spin mutexes) and src/backends/vk/queue_family_contract.h assumes "
         "external synchronization, and an imported logical device cannot attest "
         "the queue flags it was created with "
         "(VulkanDeviceConfigExt::ExternalDevice carries no flags), so both would "
         "have to change together.",
     .unblock_requirement =
         "Set VK_DEVICE_QUEUE_CREATE_INTERNALLY_SYNCHRONIZED_BIT_KHR on every "
         "backend-owned VkDeviceQueueCreateInfo in VulkanDevice.cpp, make each "
         "*_queue_mtx acquisition conditional on the gate, and keep the flag off "
         "for imported devices (the import contract cannot attest it).",
     .active = false,
     .assumed_enabled = false,
     .reachable_from_backend = false},

    // -----------------------------------------------------------------------
    // VK_ARM_tensors
    // -----------------------------------------------------------------------
    {.id = VendorExtensionId::kArmTensors,
     .extension_name = "VK_ARM_tensors",
     .config_switch = "VulkanDeviceConfigExt::enable_tensor_arm()",
     .registry_identifier = "detail::OptionalExtension::TENSOR_ARM",
     .enabled_gate = "device->optional_extension_enabled("
                     "detail::OptionalExtension::TENSOR_ARM)",
     .entry_points = "vkCreateTensorARM, vkDestroyTensorARM, "
                     "vkCreateTensorViewARM, vkDestroyTensorViewARM, "
                     "vkGetTensorMemoryRequirementsARM, vkBindTensorMemoryARM, "
                     "vkGetDeviceTensorMemoryRequirementsARM, vkCmdCopyTensorARM, "
                     "vkGetPhysicalDeviceExternalTensorPropertiesARM",
     .structures = "VkTensorCreateInfoARM, VkTensorDescriptionARM, "
                   "VkTensorViewCreateInfoARM, VkBindTensorMemoryInfoARM, "
                   "VkTensorMemoryBarrierARM, VkTensorDependencyInfoARM, "
                   "VkMemoryDedicatedAllocateInfoTensorARM, and "
                   "VkPhysicalDeviceDescriptorHeapTensorPropertiesARM for heap "
                   "tensors (tensorDescriptorSize/tensorDescriptorAlignment)",
     .features = "VkPhysicalDeviceTensorFeaturesARM::tensors (plus "
                 "shaderTensorAccess/shaderStorageTensorArrayDynamicIndexing as "
                 "needed by the generated SPIR-V)",
     .blocker =
         "There is no public LuisaCompute tensor API: the DSL type system has no "
         "tensor tag (include/luisa/ast/type.h:339-397), "
         "include/luisa/runtime/rhi/device_interface.h:124-140 has no create_tensor "
         "entry next to create_buffer/create_texture, and "
         "include/luisa/runtime/rhi/command.h has no tensor copy/barrier command, "
         "so the backend has nothing to bind an ARM tensor to. The SPIR-V route "
         "(src/backends/common/spirv/spirv_codegen/) also emits no "
         "OpTypeTensorARM, and VkPhysicalDeviceDescriptorHeapTensorPropertiesARM "
         "is only reachable through DescriptorHeap::query_limits() "
         "(src/backends/vk/vk_descriptor_heap.cpp), which refuses a tensor region "
         "unless that structure was queried.",
     .unblock_requirement =
         "A new device extension interface (for example "
         "include/luisa/backends/ext/vk_tensor_ext.h) exposing create/destroy, "
         "memory requirements, view creation and a tensor copy command, plus "
         "OpTypeTensorARM support in the SPIR-V route and tensor regions in the "
         "descriptor-heap plan.",
     .active = false,
     .assumed_enabled = false,
     .reachable_from_backend = false},

    // -----------------------------------------------------------------------
    // VK_ARM_data_graph
    // -----------------------------------------------------------------------
    {.id = VendorExtensionId::kArmDataGraph,
     .extension_name = "VK_ARM_data_graph",
     .config_switch = "VulkanDeviceConfigExt::enable_data_graph_arm()",
     .registry_identifier = "detail::OptionalExtension::DATA_GRAPH_ARM",
     .enabled_gate = "device->optional_extension_enabled("
                     "detail::OptionalExtension::DATA_GRAPH_ARM)",
     .entry_points = "vkCreateDataGraphPipelinesARM, "
                     "vkCreateDataGraphPipelineSessionARM, "
                     "vkGetDataGraphPipelineSessionBindPointRequirementsARM, "
                     "vkGetDataGraphPipelineSessionMemoryRequirementsARM, "
                     "vkBindDataGraphPipelineSessionMemoryARM, "
                     "vkDestroyDataGraphPipelineSessionARM, "
                     "vkCmdDispatchDataGraphARM, "
                     "vkGetDataGraphPipelineAvailablePropertiesARM, "
                     "vkGetDataGraphPipelinePropertiesARM, "
                     "vkGetPhysicalDeviceQueueFamilyDataGraphPropertiesARM, "
                     "vkGetPhysicalDeviceQueueFamilyDataGraphProcessingEnginePropertiesARM",
     .structures = "VkDataGraphPipelineCreateInfoARM, "
                   "VkDataGraphPipelineShaderModuleCreateInfoARM, "
                   "VkDataGraphPipelineSessionCreateInfoARM, "
                   "VkDataGraphPipelineSessionBindPointRequirementARM, "
                   "VkBindDataGraphPipelineSessionMemoryInfoARM, "
                   "VkDataGraphPipelineDispatchInfoARM, "
                   "VkDataGraphProcessingEngineCreateInfoARM, "
                   "VkQueueFamilyDataGraphPropertiesARM",
     .features = "VkPhysicalDeviceDataGraphFeaturesARM::dataGraph; the queue must "
                 "advertise VK_QUEUE_DATA_GRAPH_BIT_ARM and the barrier code needs "
                 "VK_PIPELINE_STAGE_2_DATA_GRAPH_BIT_ARM with "
                 "VK_ACCESS_2_DATA_GRAPH_READ/WRITE_BIT_ARM",
     .blocker =
         "The backend only creates graphics/compute/copy/sparse queues "
         "(src/backends/vk/VulkanDevice.cpp:246-394 with the queue-family "
         "contract in src/backends/vk/queue_family_contract.h:59), so there is no "
         "VK_QUEUE_DATA_GRAPH_BIT_ARM queue to dispatch on, and "
         "src/backends/vk/resource_barrier.cpp has no data-graph stage/access "
         "entry. The pipeline+session object model (bind points, session memory, "
         "processing engines) has no counterpart in "
         "include/luisa/runtime/rhi/device_interface.h, and the extension also "
         "requires VK_KHR_extended_flags, VK_KHR_maintenance5 and "
         "VK_KHR_deferred_host_operations, which the registry does not chain.",
     .unblock_requirement =
         "A queue role for VK_QUEUE_DATA_GRAPH_BIT_ARM in the queue-family plan, "
         "an extension interface owning data-graph pipelines and sessions, a "
         "VkCmdDispatchDataGraphARM command in the stream encoder, and the "
         "data-graph stage/access bits in the barrier tracker.",
     .active = false,
     .assumed_enabled = false,
     .reachable_from_backend = false},

    // -----------------------------------------------------------------------
    // VK_QCOM_tile_shading
    // -----------------------------------------------------------------------
    {.id = VendorExtensionId::kQcomTileShading,
     .extension_name = "VK_QCOM_tile_shading",
     .config_switch = "VulkanDeviceConfigExt::enable_tile_shading_qcom()",
     .registry_identifier = "detail::OptionalExtension::TILE_SHADING_QCOM",
     .enabled_gate = "device->optional_extension_enabled("
                     "detail::OptionalExtension::TILE_SHADING_QCOM)",
     .entry_points = "vkCmdDispatchTileQCOM, vkCmdBeginPerTileExecutionQCOM, "
                     "vkCmdEndPerTileExecutionQCOM",
     .structures = "VkRenderPassTileShadingCreateInfoQCOM chained into the render "
                   "pass, VkSubpassDescriptionFlagBits::"
                   "VK_SUBPASS_DESCRIPTION_TILE_SHADING_APRON_BIT_QCOM, "
                   "VkPerTileBeginInfoQCOM, VkPerTileEndInfoQCOM, "
                   "VkDispatchTileInfoQCOM, and the access bits "
                   "VK_ACCESS_2_SHADER_TILE_ATTACHMENT_READ_BIT_QCOM / "
                   "VK_ACCESS_2_SHADER_TILE_ATTACHMENT_WRITE_BIT_QCOM",
     .features = "VkPhysicalDeviceTileShadingFeaturesQCOM::tileShading plus "
                 "tileShadingPerTileDispatch / tileShadingPerTileDraw / "
                 "tileShadingFragmentStage, and the attachment support bits the "
                 "raster path needs",
     .blocker =
         "Tile shading is a render-pass feature and the raster path has exactly "
         "one VkRenderPass, created from a single-subpass "
         "VkRenderPassCreateInfo in src/backends/vk/raster_pipeline.cpp:122-135 and "
         "begun with vkCmdBeginRenderPass from src/backends/vk/stream.cpp:4982 "
         "(raster_pipeline.cpp:303 documents that the raster path never uses "
         "vkCmdBeginRendering): there is no per-tile execution scope, no apron "
         "subpass flag and no dispatch point in "
         "src/backends/vk/native_raster_shader.cpp. "
         "src/backends/vk/resource_barrier.cpp has no "
         "VK_ACCESS_2_SHADER_TILE_ATTACHMENT_* mapping, and the extension depends "
         "on VK_QCOM_tile_properties, which the optional-extension registry does "
         "not negotiate at all (no OptionalExtension entry).",
     .unblock_requirement =
         "Chain VkRenderPassTileShadingCreateInfoQCOM and the apron subpass flag "
         "in the raster pipeline, record vkCmdBeginPerTileExecutionQCOM / "
         "vkCmdEndPerTileExecutionQCOM / vkCmdDispatchTileQCOM in the raster "
         "encoder, add the tile-attachment access bits to the barrier tracker, and "
         "negotiate the VK_QCOM_tile_properties dependency.",
     .active = false,
     .assumed_enabled = false,
     .reachable_from_backend = false},

    // -----------------------------------------------------------------------
    // VK_QCOM_tile_memory_heap
    // -----------------------------------------------------------------------
    {.id = VendorExtensionId::kQcomTileMemoryHeap,
     .extension_name = "VK_QCOM_tile_memory_heap",
     .config_switch = "VulkanDeviceConfigExt::enable_tile_memory_heap_qcom()",
     .registry_identifier = "detail::OptionalExtension::TILE_MEMORY_HEAP_QCOM",
     .enabled_gate = "device->optional_extension_enabled("
                     "detail::OptionalExtension::TILE_MEMORY_HEAP_QCOM)",
     .entry_points = "vkCmdBindTileMemoryQCOM",
     .structures = "VkPhysicalDeviceTileMemoryHeapPropertiesQCOM, "
                   "VkTileMemoryRequirementsQCOM, VkTileMemoryBindInfoQCOM, "
                   "VkMemoryHeap::flags with VK_MEMORY_HEAP_TILE_MEMORY_BIT_QCOM, "
                   "VK_BUFFER_USAGE_TILE_MEMORY_BIT_QCOM / "
                   "VK_IMAGE_USAGE_TILE_MEMORY_BIT_QCOM on the bound resources",
     .features = "VkPhysicalDeviceTileMemoryHeapFeaturesQCOM::tileMemoryHeap",
     .blocker =
       "vkCmdBindTileMemoryQCOM is only valid inside a tile-shading render pass, "
       "so it is blocked by the same missing raster path as VK_QCOM_tile_shading "
       "(src/backends/vk/raster_pipeline.cpp:122-135). In addition the memory "
         "type selection in src/backends/vk/vk_allocator.cpp:204-219 only admits "
         "VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT / HOST_VISIBLE_BIT candidates and "
         "never a VK_MEMORY_HEAP_TILE_MEMORY_BIT_QCOM heap, and the public "
         "create_buffer/create_texture signatures have no tile-memory usage "
         "channel.",
     .unblock_requirement =
         "Admit tile-memory heaps in the allocator's memory-type selection, add a "
         "tile-memory usage flag to the resource creation path, and record "
         "vkCmdBindTileMemoryQCOM from the tile-shading raster encoder (i.e. after "
         "VK_QCOM_tile_shading is wired).",
     .active = false,
     .assumed_enabled = false,
     .reachable_from_backend = false},

    // -----------------------------------------------------------------------
    // VK_EXT_external_memory_metal
    // -----------------------------------------------------------------------
    {.id = VendorExtensionId::kExternalMemoryMetal,
     // `VK_EXT_EXTERNAL_MEMORY_METAL_EXTENSION_NAME` is only declared when
     // VK_USE_PLATFORM_METAL_EXT is defined (the extension has platform="metal"),
     // and the vendored volk header declares only its VkStructureType values and
     // handle-type bits, so the literal name is spelled out here.
     .extension_name = "VK_EXT_external_memory_metal",
     .config_switch = "VulkanDeviceConfigExt::enable_external_memory_metal()",
     .registry_identifier = "detail::OptionalExtension::EXTERNAL_MEMORY_METAL",
     .enabled_gate = "device->optional_extension_enabled("
                     "detail::OptionalExtension::EXTERNAL_MEMORY_METAL)",
     .entry_points = "vkGetMemoryMetalHandleEXT, vkGetMemoryMetalHandlePropertiesEXT",
     .structures = "VkImportMemoryMetalHandleInfoEXT, "
                   "VkMemoryMetalHandlePropertiesEXT, "
                   "VkMemoryGetMetalHandleInfoEXT, and the handle types "
                   "VK_EXTERNAL_MEMORY_HANDLE_TYPE_MTLBUFFER_BIT_EXT / "
                   "MTLTEXTURE_BIT_EXT / MTLHEAP_BIT_EXT",
     .features = "no feature structure (nofeatures=\"true\"); the handle/compatible "
                 "handle queries carry the capability",
     .blocker =
         "The extension is platform=\"metal\": every one of its structures and "
         "commands is only declared when VK_USE_PLATFORM_METAL_EXT is defined, and "
         "src/ext/volk/include/vulkan/vulkan_core.h:1402-1404 declares only the "
         "three VkStructureType values while lines 5457-5459 declare only the three "
         "handle-type bits - never the two commands the backend would call. It is "
         "also only usable on an Apple platform whose driver exposes Metal "
         "interop, while LuisaCompute's external-memory contract "
         "(DeviceInterface::create_buffer(..., external_memory) and "
         "create_texture(..., external_native_handle, ...) in "
         "include/luisa/runtime/rhi/device_interface.h:124-140) has no Metal-handle "
         "channel, and the Metal backends (src/backends/metal, "
         "src/backends/metal4) never go through Vulkan.",
     .unblock_requirement =
         "An Apple-only Vulkan build with VK_USE_PLATFORM_METAL_EXT (volk "
         "regenerated with the Metal platform), plus a public Metal-handle "
         "import/export extension interface mapping the borrowed MTLBuffer/"
         "MTLTexture to the existing external-memory creation path.",
     .active = false,
     .assumed_enabled = false,
     .reachable_from_backend = false},
};

[[nodiscard]] constexpr const char *cstr_non_empty(const char *text) noexcept {
    return (text != nullptr && text[0] != '\0') ? text : nullptr;
}

// The invariant every entry has to satisfy: it is inert (never active, never
// assumed present), it cannot be reached from the code of this work package, and
// it documents both its Vulkan surface and its blocker.
[[nodiscard]] constexpr bool vendor_extension_plan_is_inert(
    const VendorExtensionPlan &plan) noexcept {
    return !plan.active && !plan.assumed_enabled && !plan.reachable_from_backend &&
           cstr_non_empty(plan.extension_name) != nullptr &&
           cstr_non_empty(plan.config_switch) != nullptr &&
           cstr_non_empty(plan.registry_identifier) != nullptr &&
           cstr_non_empty(plan.enabled_gate) != nullptr &&
           cstr_non_empty(plan.entry_points) != nullptr &&
           cstr_non_empty(plan.structures) != nullptr &&
           cstr_non_empty(plan.features) != nullptr &&
           cstr_non_empty(plan.blocker) != nullptr &&
           cstr_non_empty(plan.unblock_requirement) != nullptr;
}

[[nodiscard]] constexpr const VendorExtensionPlan *vendor_extension_plan(
    VendorExtensionId id) noexcept {
    for (uint32_t i = 0u; i < kVendorExtensionCount; i++) {
        if (kVendorExtensionPlans[i].id == id) { return &kVendorExtensionPlans[i]; }
    }
    return nullptr;
}

// True when every entry of the table satisfies the contract above. The backend may
// assert this once at startup; the unit test checks it exhaustively.
[[nodiscard]] constexpr bool vendor_extension_plans_are_inert() noexcept {
    for (uint32_t i = 0u; i < kVendorExtensionCount; i++) {
        if (!vendor_extension_plan_is_inert(kVendorExtensionPlans[i])) { return false; }
        // A duplicate identifier (or a missing one) would make the lookup above
        // ambiguous.
        if (kVendorExtensionPlans[i].id != static_cast<VendorExtensionId>(i)) {
            return false;
        }
    }
    return true;
}

}// namespace lc::vk::detail
