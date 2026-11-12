# Vulkan Backend Optional Extensions

The Vulkan backend uses a large, growing set of *optional* device extensions.
Every one of them is an optimization or an extra capability: when an extension is
missing, switched off, or the logical device is imported, the backend falls back
to the code path it used before the extension existed. This document describes
the negotiation machinery, how to enable/disable individual extensions, and what
each extension actually changes.

```
VulkanDeviceConfigExt::enable_*()          physical device extension list
                 |                                    |
                 v                                    v
        +--------------------------------------------------+
        |  detail::OptionalExtensionRegistry                |
        |  1. select(): supported & requested               |
        |  2. one vkGetPhysicalDeviceFeatures2 probe        |
        |  3. resolve(): enabled = requested && feature ok   |
        |  4. device_feature_chain(): pNext for vkCreateDevice
        +--------------------------------------------------+
                 |                                    |
                 v                                    v
   VkDeviceCreateInfo::ppEnabledExtensionNames   feature pNext chain
                 |                                    |
                 +----------------+-------------------+
                                  v
                        backend fast paths, gated by
                   Device::optional_extension_enabled(...)
                                  |
                                  v
                 mark_optional_extension_active(...)
                                  |
                                  v
                  VulkanExtensionStatusExt (runtime query)
```

## Negotiation

`src/backends/vk/vk_extension_registry.h` holds one
`detail::OptionalExtension` identifier and one `OptionalExtensionDescriptor` per
extension. Each descriptor records:

| Field | Meaning |
| --- | --- |
| `name` / `alias_name` | Preferred extension name and acceptable twin (`KHR`/`EXT`, `EXT`/`NV`, ...). Whichever spelling the physical device exposes is the one enabled. |
| `requires_extension` | Another extension that must end up enabled, evaluated order-independently (present extensions need `VK_KHR_swapchain`, ray-tracing helpers need `VK_KHR_acceleration_structure`, ...). |
| `requested` | Reads the matching `VulkanDeviceConfigExt::enable_*()` switch. |
| `feature_structure_type` / `make_feature` / `check_and_enable` | The `VkPhysicalDevice*Features*` structure the extension requires, how to build it, and which bits the backend needs. `check_and_enable` both reports support and turns exactly those bits on. `alias_feature_structure_type` covers the few KHR retrofits (`VK_EXT/VK_KHR_opacity_micromap`, `VK_EXT/VK_NV_ray_tracing_invocation_reorder`, `VK_KHR/VK_NV_copy_memory_indirect`) whose two spellings define *different* `VkStructureType` values. |

`OptionalExtensionRegistry::select()` matches the descriptors against the
physical device's extension list, one `vkGetPhysicalDeviceFeatures2` call probes
every candidate's feature structure, and `resolve()` decides what is enabled. A
feature structure only enters the `vkCreateDevice` pNext chain when its
extension is enabled — chaining a structure of a disabled extension is a
validation error.

A handful of optional extensions are **instance** extensions
(`VK_KHR_surface_maintenance1`). Those descriptors set `instance_level`, are
matched against the instance extension list the backend actually enabled, never
enter `VkDeviceCreateInfo`, and additionally require `VK_KHR_swapchain` so they
only light up when a surface exists.

An **imported logical device** (see `VulkanDeviceConfigExt::ExternalDevice`)
never enables an optional extension: Vulkan cannot report which extensions or
feature bits a borrowed `VkDevice` was created with, so nothing is assumed.

## Switching extensions off

`include/luisa/backends/ext/vk_config_ext.h` exposes one virtual switch per
extension, all defaulting to `true`:

```cpp
class NoFastPaths : public VulkanDeviceConfigExt {
public:
    [[nodiscard]] bool enable_unified_image_layouts() const noexcept override { return false; }
    [[nodiscard]] bool enable_descriptor_heap() const noexcept override { return false; }
    // ...
};

DeviceConfig config;
config.extension = luisa::make_unique<NoFastPaths>();
auto device = context.create_device("vk", &config);
```

Without a `VulkanDeviceConfigExt` the backend behaves exactly as if every switch
returned `true` (the physical device still decides what is possible).

`VK_KHR_robustness2` is special: it is a debugging aid, so the backend only ever
requests it in non-`NDEBUG` builds.

## Querying the result

`include/luisa/backends/ext/vk_extension_ext.h` exposes
`VulkanExtensionStatusExt`:

```cpp
auto ext = device.extension<VulkanExtensionStatusExt>();
for (auto const &status : ext->extensions()) {
    // status.name / supported / requested / enabled / active
}
bool fast = ext->is_active("VK_KHR_unified_image_layouts");
```

* `supported` — the physical device exposes the extension.
* `requested` — `VulkanDeviceConfigExt` allows it.
* `enabled` — it is on the logical device (name + `VK_TRUE` feature bits).
* `active` — the backend actually routes work through it. "Enabled" only means
  it *may*; a module calls `Device::mark_optional_extension_active()` at the
  point where it really takes the fast path. An extension can be enabled and
  never active (for example a maintenance extension whose only effect is
  removing a restriction the backend never hit) — that is expected, not a bug.

The backend also logs one line per negotiated extension at device creation.

## Extension-by-extension

### Resources, descriptors, GPU-driven rendering

| Extension | What the backend does | Fallback |
| --- | --- | --- |
| `VK_EXT_descriptor_heap` | `src/backends/vk/vk_descriptor_heap.*` implements the heap model: device-address sampler/resource heaps sized from `VkPhysicalDeviceDescriptorHeapPropertiesEXT`, host-side `vkWriteResourceDescriptorsEXT`/`vkWriteSamplerDescriptorsEXT` with staging upload, a first-fit slot allocator, and `vkCmdBind*HeapEXT`/`vkCmdPushDataEXT` recording helpers. It is an opt-in foundation: the default bindless path still uses update-after-bind descriptor sets because switching it also needs `VkShaderDescriptorSetAndBindingMappingInfoEXT` on every shader stage. `detail::plan_bindless_descriptor_heap_migration()` records the exact follow-up. | Descriptor sets (unchanged). |
| `VK_KHR_device_address_commands` | `stream.cpp` addresses indirect dispatch through device addresses instead of a `VkBuffer` handle. | `vkCmdDispatchIndirect` with the buffer handle. |
| `VK_KHR_unified_image_layouts` | The barrier tracker may keep every image in `VK_IMAGE_LAYOUT_GENERAL`, so inter-frame layout transitions disappear; only availability/visibility barriers remain. | The previous layout choice (specific layouts, except `simultaneous_access` images). |
| `VK_KHR_copy_memory_indirect` | A batch of same-shaped buffer copies becomes one `vkCmdCopyMemoryIndirectKHR` driven by a device-side command array. | One `vkCmdCopyBuffer` per copy. |
| `VK_EXT_memory_decompression` | Backend entry points for `vkCmdDecompressMemoryEXT` / `vkCmdDecompressMemoryIndirectCountEXT` (GDeflate, `EXT` spelling only — the `NV` alias has its own command signature and is rejected), recorded as extension commands of a Luisa stream so normal barrier/reorder bookkeeping applies. Fails closed with `LUISA_ERROR` when unavailable. **Reach:** `lc::vk::memory_decompression::*`, i.e. backend-internal; no `luisa::compute` API exposes it yet, so the extension reports enabled-but-inactive. | No decompression command exists. |
| `VK_EXT_zero_initialize_device_memory` | Dedicated allocations may request `VK_MEMORY_ALLOCATE_ZERO_INITIALIZE_BIT_EXT` (honouring the spec's protected-memory and memory-import constraints, and excluding lazily-allocated types). When it cannot be requested, the backend zeroes the range itself for host-visible memory, so the zero-content contract never depends on the extension. **Reach:** `lc::vk::create_zero_initialized_buffer()` / `create_texture_with_tiling_policy()`, i.e. backend-internal; `Device::create_buffer` still uses the VMA path, so the extension is enabled-but-inactive today. | Undefined contents, as before. |
| `VK_KHR_internally_synchronized_queues` | Capability only: queue creation is centralized and the switch would have to gate the backend's queue mutexes. Documented in `vk_vendor_extensions_plan.h`. | Backend queue mutexes. |
| `VK_KHR_extended_flags` | Capability only: the backend's flag values all fit 32 bits. | n/a |
| `VK_EXT_image_tiling_control` | `VkImageTilingControlCreateInfoEXT` is chained for a non-default tiling policy; `DEFAULT` is never chained because the spec defines it as "structure omitted". | Plain `VkImageCreateInfo`. |
| `VK_EXT_texture_compression_astc_3d` | 3D ASTC formats are mapped onto `VK_FORMAT_ASTC_{4x4x4,5x5x5,6x6x6}_{UNORM,SRGB}_BLOCK_EXT` when a 2D ASTC `PixelFormat` is used for a 3D texture; 8x8/10x10/12x12 have no 3D counterpart and keep failing. Note the block footprint differs (a 2D `NxN` block is 128 bit / 16 texels, a 3D `NxNxN` block is 128 bit / 64 texels), so host-side byte-size helpers must not treat the two as interchangeable. | `VK_FORMAT_UNDEFINED` → the pre-existing "unsupported pixel format" error (fail-closed). |

### Presentation and frame pacing

| Extension | What the backend does | Fallback |
| --- | --- | --- |
| `VK_KHR_present_id2` / `VK_KHR_present_wait2` | Present ids are assigned per present and gated for chaining / `vkWaitForPresent2KHR`. The actual chaining happens in the swapchain `present` path; the planning is host-tested. | Queue-idle synchronization. |
| `VK_KHR_present_mode_fifo_latest_ready` | Tear-free "latest ready frame" mode selected when the surface reports it and vsync was requested. | The previously selected present mode. |
| `VK_KHR_surface_maintenance1` / `VK_KHR_swapchain_maintenance1` | Present-mode/scaling capability queries and `VkSwapchainPresentModesCreateInfoKHR`/scaling create-info gating. | `vkGetPhysicalDeviceSurfacePresentModesKHR` and the previous create-info. |
| `VK_EXT_present_timing` | Negotiated and reported, but the present chain cannot carry `VkPresentTimingsInfoEXT` until the public API delivers per-frame timing; see the in-code `kPublicApiDeliversPresentTiming`. | No timing feedback. |
| `VK_EXT_multisampled_render_to_swapchain` | Negotiated; wiring needs the raster MSAA path. | Explicit MSAA attachments. |

### Shader capabilities

`VK_KHR_shader_fma`, `VK_EXT_shader_64bit_indexing`,
`VK_EXT_shader_uniform_buffer_unsized_array`, `VK_EXT_shader_long_vector`,
`VK_EXT_shader_subgroup_partitioned`, `VK_EXT_shader_split_barrier`,
`VK_EXT_shader_ocp_microscaling_types`, `VK_KHR_shader_constant_data` and
`VK_KHR_shader_abort` are appended to the persistent
`lc::spirv::SpirvTargetFeatureMask` ABI
(`src/backends/common/spirv/spirv_codegen/target_feature_mask.h`) and populated
from `Device::enabled_spirv_target_features()`. A shader artifact can therefore
record that it requires one of these capabilities, and the artifact cache keys
on it. They stay `active=false` until a code generator emission site uses them,
so no kernel changes behaviour today.

### Ray tracing

`VK_EXT_ray_tracing_invocation_reorder` (cross-vendor SER),
`VK_KHR_opacity_micromap`, `VK_NV_ray_tracing_linear_swept_spheres`,
`VK_AMDX_dense_geometry_format` and `VK_NV_partitioned_acceleration_structure`
are negotiated. Opacity micromaps get a real build path
(`src/backends/vk/opacity_micromap.*`: micromap allocation, `vkBuildMicromapsEXT`,
`VkAccelerationStructureTrianglesOpacityMicromapEXT` build input) which is
currently unreachable from the public accel API, so the extension reports
enabled-but-inactive. The others are capability-only; each blocker is recorded in
code with the exact entry points that would be required.

### Rasterization

`VK_EXT_primitive_restart_index`, `VK_KHR_depth_clamp_zero_one`,
`VK_EXT_custom_resolve`, `VK_EXT_fragment_density_map_offset`,
`VK_KHR_pipeline_library_group_handles` and
`VK_EXT_multisampled_render_to_swapchain` are planned in
`src/backends/vk/raster_extension_plan.h`, which decides between dynamic and
static pipeline state, maps depth-clamp ranges, builds the render-pass end info
and rounds fragment-density offsets to the reported granularity. Adopting a plan
requires a field on the raster command/state API; until then the static pipeline
state and the previous render pass are used.

### Maintenance

`VK_KHR_maintenance8/9/10` (and 11/12 where exposed) are negotiated but change
nothing by themselves: each relaxation was analysed against this backend and
either does not apply (for example every barrier already uses
`VK_QUEUE_FAMILY_IGNORED` because resources are created concurrently shared) or
needs a property query the backend does not make yet. The analysis is recorded
in `src/backends/vk/resource_barrier_contract.h` (and referenced from
`resource_barrier.cpp`). `VK_KHR_maintenance5` is enabled whenever the
physical device supports it, because `VkBufferUsageFlags2CreateInfoKHR` is the
only channel for the 64-bit usage masks newer optional features need.

### Vendor extensions

`VK_ARM_tensors`, `VK_ARM_data_graph`, `VK_QCOM_tile_shading`,
`VK_QCOM_tile_memory_heap` and `VK_EXT_external_memory_metal` have no public
LuisaCompute surface (tensor type, data-graph queue role, tile render pass,
Metal handle channel). `src/backends/vk/vk_vendor_extensions_plan.h` carries one
inert plan per extension: the entry points and structures the backend would use,
the concrete blocker with file:line, the unblocking requirement, and the
invariant that none of them is ever assumed enabled.

## Tests

| Target | Covers |
| --- | --- |
| `test_vk_extension_registry` | Name/alias resolution, supported/requested/enabled matrix, probe chain contents, alias `VkStructureType`, prerequisite handling, imported devices, `force_enable`, status strings. No GPU needed. |
| `test_vk_ext_layouts` | Unified image layout policy truth table, the barrier plan's hazard/layout separation (`ImageBarrierPlan` keeps identical stage/access masks while the layout transition disappears), the layout-limiting rule, and the maintenance analysis invariants. The end-to-end proof that barriers are still emitted is the image workload in `test_vk_optional_extensions`. |
| `test_vk_ext_memory` | Zero-initialize allocation matrix, decompression region validation, tiling control, ASTC 3D mapping, flag width invariant. |
| `test_vk_ext_copy` | Indirect-copy batching decisions and command layout, device-address indirect dispatch planning. |
| `test_vk_ext_present` | Instance attestation, present-mode selection, maintenance1 create-info gating, present-id assignment, id2/timing gating. |
| `test_vk_ext_raster` | Dynamic-vs-static state decisions, depth-clamp mapping, render-pass end info, density-map offset rounding. |
| `test_vk_ext_shader_caps` | Capability matrix, new `SpirvTargetFeatureMask` bits, no collisions. |
| `test_vk_ext_rt` | Micromap build-input planning, SER applicability, geometry flags. |
| `test_vk_ext_misc` | Descriptor-heap planner, descriptor offsets, bindless migration artifact, vendor-plan invariants. |
| `test_vk_optional_extensions` (GPU) | Creates the device twice — once with every switch off, once with defaults — runs a compute + bindless + image workload in both, and asserts `enabled ⊆ supported`, `enabled ⊆ requested`, `active ⊆ enabled`, plus that the unified-layout fast path is genuinely active when enabled. |

Run the host tests with `xmake run <target>` and the device test with
`bin/release/test_vk_optional_extensions.exe vk`.
