#pragma once
#include <vulkan/vulkan_core.h>
#include <luisa/core/stl/filesystem.h>
#include <luisa/core/stl/string.h>
#include <luisa/runtime/rhi/device_interface.h>
#include <luisa/backends/ext/vk_custom_cmd.h>
struct IDxcCompiler3;
struct IDxcLibrary;
struct IDxcUtils;
namespace luisa::compute {

// Volk's default dispatch tables are process-global. Consequently, at most
// one Vulkan backend Device may be alive in a process, and Vulkan device
// enumeration/instance initialization must not run while that Device is
// alive. The first backend Vulkan operation also pins one loader identity;
// later operations must request the same default or custom loader.
class VulkanDeviceConfigExt : public DeviceConfigExt {
public:
    // Every non-null native handle below is borrowed. The backend does not
    // destroy the imported instance, physical device, logical device, or
    // queues. The caller must keep those handles, their complete Vulkan
    // ancestry, and the Vulkan loader that created them valid until the Luisa
    // Vulkan Device has been destroyed. In particular, destroy the Luisa
    // Device before destroying or invalidating any imported handle.
    struct ExternalDevice {
        struct RequiredFeatures {
            // Vulkan exposes physical-device support but cannot report which
            // feature bits were enabled when an existing VkDevice was
            // created. These fields are caller attestations for features the
            // backend unconditionally uses.
            bool timeline_semaphore{false};
            bool synchronization2{false};
        };
        VkInstance instance{};
        // Required when instance is non-null. This is the effective API
        // version supplied when the borrowed instance was created, not the
        // physical-device apiVersion. The backend uses Vulkan 1.3 core
        // commands and therefore rejects versions below VK_API_VERSION_1_3.
        // Borrowed instances are currently compute-only because Vulkan cannot
        // query their enabled surface-extension list after creation.
        uint32_t api_version{};
        VkPhysicalDevice physical_device{};
        VkDevice device{};
        // All three handles are required when device is non-null. A handle may
        // be reused for multiple roles; the backend serializes equal handles
        // through one host mutex. The caller must externally synchronize any
        // direct queue access performed outside the backend.
        VkQueue graphics_queue{};
        VkQueue compute_queue{};
        VkQueue copy_queue{};
        // Required when device is non-null. Vulkan provides no query from a
        // VkQueue handle back to its family, and backend-owned resources must
        // use the exact participating families for concurrent sharing.
        uint32_t graphics_queue_family_index{VK_QUEUE_FAMILY_IGNORED};
        uint32_t compute_queue_family_index{VK_QUEUE_FAMILY_IGNORED};
        uint32_t copy_queue_family_index{VK_QUEUE_FAMILY_IGNORED};
        // Required when device is non-null. Each true value asserts that the
        // corresponding feature was enabled on that logical device, not only
        // supported by physical_device.
        RequiredFeatures required_features{};
    };
    struct VulkanLibPath {
        // lib_path is a loader search directory, not a complete library path;
        // lib_name may be empty to use the backend's platform candidate list.
        // An empty pair selects Volk's default loader. The first Vulkan
        // backend operation pins that choice for the loaded backend's
        // lifetime. A later custom request must have the same lib_path after
        // absolute/canonical normalization and the same exact lib_name.
        // Calling Context::backend_device_names("vk") first therefore pins
        // the default loader and makes a later custom request invalid.
        luisa::filesystem::path lib_path;
        luisa::string lib_name;
    };
    VulkanDeviceConfigExt() noexcept = default;
    ~VulkanDeviceConfigExt() noexcept = default;
    [[nodiscard]] virtual ExternalDevice create_external_device() noexcept {
        return {};
    }
    [[nodiscard]] virtual bool enable_bindless_feature() const noexcept {
        return true;
    }
    // Requested update-after-bind heap capacity for the three global bindless
    // heaps. Return 0 to use the backend default (262144 on desktop). Mobile
    // builds can lower this to reduce maxUpdateAfterBindDescriptorsInAllPools
    // pressure.
    [[nodiscard]] virtual uint32_t requested_bindless_heap_capacity() const noexcept {
        return 0u;
    }
    // Minimum Vulkan API version the backend must honor during device
    // capability negotiation. Return VK_API_VERSION_1_2 to force the legacy
    // barrier/copy paths (the Android-compatible floor), VK_API_VERSION_1_3 to
    // require synchronization2/copy_commands2, or 0 for "auto" (use whatever
    // the physical device exposes).
    [[nodiscard]] virtual uint32_t min_api_version() const noexcept {
        return 0u;
    }
    // Command reordering merges consecutive commands whose resource accesses
    // do not alias into one barrier-free layer, so several small dispatches
    // can overlap on the GPU instead of being serialized by a pipeline
    // barrier per command. Return false to submit the batch in strict order
    // instead (a barrier boundary between every pair of commands), which is
    // what LUISA_DISABLE_COMMAND_REORDER=1 forces process-wide. The backend
    // queries this once per batch, so an override may change at runtime.
    [[nodiscard]] virtual bool enable_command_reorder() const noexcept {
        return true;
    }
    [[nodiscard]] virtual bool enable_raytracing_feature() const noexcept {
        return true;
    }
    // Answer every acceleration-structure request (create_mesh / create_accel
    // and their build commands) with the software two-level LBVH of
    // `luisa-fallback-rtx` (src/backends/common/rtx) and never touch
    // VK_KHR_acceleration_structure, even on a device that could trace in
    // hardware. When it is false the backend uses the hardware path whenever
    // the physical device exposes it, and falls back to the software one only
    // when it does not, so the default behaviour is unchanged.
    //
    // A fallen-back device compiles a ray-tracing shader through the
    // compatibility HLSL route (the native XIR-to-SPIR-V route has no software
    // traversal): the shader's acceleration-structure argument is bound to the
    // fallback's shared acceleration buffer and instance buffer instead of a
    // VkAccelerationStructureKHR. Ray queries, motion blur, curves and
    // procedural primitives are not implemented by the fallback and fail
    // closed. `LUISA_VK_FALLBACK_RTX=1` forces it for tests that cannot hand a
    // `VulkanDeviceConfigExt` to `Context::create_device`.
    [[nodiscard]] virtual bool use_fallback_rtx() const noexcept { return false; }
    [[nodiscard]] virtual bool enable_interop_feature() const noexcept {
        return true;
    }
    [[nodiscard]] virtual bool enable_device_address_feature() const noexcept {
        return true;
    }
    [[nodiscard]] virtual bool enable_surface_feature() const noexcept {
        return true;
    }
    [[nodiscard]] virtual bool enable_motion_blur() const noexcept {
        return true;
    }
    // ---------------------------------------------------------------------
    // Optional extension switches
    //
    // Every switch below gates one *optional* Vulkan device extension. The
    // backend enables an extension only when the switch returns true *and* the
    // physical device exposes it, so returning true never makes an unsupported
    // device fail. Returning false is always safe: the backend keeps a
    // fallback path and only loses the optimization or the extra capability.
    //
    // The negotiated result (supported / requested / enabled) is queryable at
    // runtime through VulkanExtensionStatusExt
    // (<luisa/backends/ext/vk_extension_ext.h>) and is logged once at device
    // creation.
    // ---------------------------------------------------------------------
    // Descriptor heaps (VK_EXT_descriptor_heap): device-address based
    // descriptor storage that can replace update-after-bind descriptor sets.
    [[nodiscard]] virtual bool enable_descriptor_heap() const noexcept {
        return true;
    }
    // Device-address command variants (VK_KHR_device_address_commands):
    // indirect draw/dispatch and copies addressed by VkDeviceAddress instead of
    // VkBuffer handles.
    [[nodiscard]] virtual bool enable_device_address_commands() const noexcept {
        return true;
    }
    // Unified image layouts (VK_KHR_unified_image_layouts): VK_IMAGE_LAYOUT_GENERAL
    // keeps full performance, so the barrier tracker can stop switching layouts.
    [[nodiscard]] virtual bool enable_unified_image_layouts() const noexcept {
        return true;
    }
    // Indirect memory/image copies (VK_KHR_copy_memory_indirect) whose
    // parameters are read from a device buffer at execution time.
    [[nodiscard]] virtual bool enable_copy_memory_indirect() const noexcept {
        return true;
    }
    // GPU memory decompression (VK_EXT_memory_decompression).
    [[nodiscard]] virtual bool enable_memory_decompression() const noexcept {
        return true;
    }
    // Zero-initialized device memory (VK_EXT_zero_initialize_device_memory).
    [[nodiscard]] virtual bool enable_zero_initialize_device_memory() const noexcept {
        return true;
    }
    // Internally synchronized queues (VK_KHR_internally_synchronized_queues).
    [[nodiscard]] virtual bool enable_internally_synchronized_queues() const noexcept {
        return true;
    }
    // 64-bit flag types (VK_KHR_extended_flags).
    [[nodiscard]] virtual bool enable_extended_flags() const noexcept {
        return true;
    }
    // Per-image tiling policy selection (VK_EXT_image_tiling_control).
    [[nodiscard]] virtual bool enable_image_tiling_control() const noexcept {
        return true;
    }
    // Present id/wait v2 (VK_KHR_present_id2 / VK_KHR_present_wait2).
    [[nodiscard]] virtual bool enable_present_id2() const noexcept {
        return true;
    }
    [[nodiscard]] virtual bool enable_present_wait2() const noexcept {
        return true;
    }
    // Tear-free "latest ready frame" present mode
    // (VK_KHR_present_mode_fifo_latest_ready).
    [[nodiscard]] virtual bool enable_present_mode_fifo_latest_ready() const noexcept {
        return true;
    }
    // Surface/swapchain maintenance1 capability queries and creation options
    // (VK_KHR_surface_maintenance1 / VK_KHR_swapchain_maintenance1).
    [[nodiscard]] virtual bool enable_surface_maintenance1() const noexcept {
        return true;
    }
    [[nodiscard]] virtual bool enable_swapchain_maintenance1() const noexcept {
        return true;
    }
    // Frame pacing feedback (VK_EXT_present_timing).
    [[nodiscard]] virtual bool enable_present_timing() const noexcept {
        return true;
    }
    // Multisampled render directly to the swapchain
    // (VK_EXT_multisampled_render_to_swapchain).
    [[nodiscard]] virtual bool enable_multisampled_render_to_swapchain() const noexcept {
        return true;
    }
    // Shader capability extensions. The backend only advertises them to the
    // code generator when the corresponding extension is enabled.
    [[nodiscard]] virtual bool enable_shader_fma() const noexcept {
        return true;
    }
    [[nodiscard]] virtual bool enable_shader_64bit_indexing() const noexcept {
        return true;
    }
    [[nodiscard]] virtual bool enable_shader_uniform_buffer_unsized_array() const noexcept {
        return true;
    }
    [[nodiscard]] virtual bool enable_shader_long_vector() const noexcept {
        return true;
    }
    [[nodiscard]] virtual bool enable_shader_subgroup_partitioned() const noexcept {
        return true;
    }
    [[nodiscard]] virtual bool enable_shader_split_barrier() const noexcept {
        return true;
    }
    [[nodiscard]] virtual bool enable_shader_ocp_microscaling_types() const noexcept {
        return true;
    }
    [[nodiscard]] virtual bool enable_shader_constant_data() const noexcept {
        return true;
    }
    [[nodiscard]] virtual bool enable_shader_abort() const noexcept {
        return true;
    }
    // Cross-vendor shader execution reordering
    // (VK_EXT_ray_tracing_invocation_reorder).
    [[nodiscard]] virtual bool enable_ray_tracing_invocation_reorder() const noexcept {
        return true;
    }
    // Opacity micromaps (VK_KHR_opacity_micromap; the EXT alias is accepted
    // when the KHR name is unavailable).
    [[nodiscard]] virtual bool enable_opacity_micromap() const noexcept {
        return true;
    }
    // Shader-provided attachment resolve (VK_EXT_custom_resolve).
    [[nodiscard]] virtual bool enable_custom_resolve() const noexcept {
        return true;
    }
    // Fragment density map offsets (VK_EXT_fragment_density_map_offset).
    [[nodiscard]] virtual bool enable_fragment_density_map_offset() const noexcept {
        return true;
    }
    // Dynamic primitive restart index (VK_EXT_primitive_restart_index).
    [[nodiscard]] virtual bool enable_primitive_restart_index() const noexcept {
        return true;
    }
    // 3D ASTC compressed textures (VK_EXT_texture_compression_astc_3d).
    [[nodiscard]] virtual bool enable_texture_compression_astc_3d() const noexcept {
        return true;
    }
    // Pipeline library group handles
    // (VK_KHR_pipeline_library_group_handles).
    [[nodiscard]] virtual bool enable_pipeline_library_group_handles() const noexcept {
        return true;
    }
    // Vulkan/Metal external memory interop (VK_EXT_external_memory_metal).
    [[nodiscard]] virtual bool enable_external_memory_metal() const noexcept {
        return true;
    }
    // D3D-style [0,1] depth clamp range (VK_KHR_depth_clamp_zero_one).
    [[nodiscard]] virtual bool enable_depth_clamp_zero_one() const noexcept {
        return true;
    }
    // Maintenance extensions. Each one only removes API friction the backend
    // previously worked around; disabling it restores the workaround.
    [[nodiscard]] virtual bool enable_maintenance8() const noexcept {
        return true;
    }
    [[nodiscard]] virtual bool enable_maintenance9() const noexcept {
        return true;
    }
    [[nodiscard]] virtual bool enable_maintenance10() const noexcept {
        return true;
    }
    [[nodiscard]] virtual bool enable_maintenance11() const noexcept {
        return true;
    }
    [[nodiscard]] virtual bool enable_maintenance12() const noexcept {
        return true;
    }
    // Robustness2 (VK_KHR_robustness2 / VK_EXT_robustness2) is a debugging aid:
    // the backend only considers it in non-NDEBUG builds, where nullDescriptor
    // and out-of-bounds access are turned into well-defined behaviour.
    [[nodiscard]] virtual bool enable_robustness2() const noexcept {
        return true;
    }
    // Vendor extensions. They are only usable on the matching hardware, and
    // the backend treats "unsupported" as "feature absent".
    [[nodiscard]] virtual bool enable_tensor_arm() const noexcept {
        return true;
    }
    [[nodiscard]] virtual bool enable_data_graph_arm() const noexcept {
        return true;
    }
    [[nodiscard]] virtual bool enable_tile_shading_qcom() const noexcept {
        return true;
    }
    [[nodiscard]] virtual bool enable_tile_memory_heap_qcom() const noexcept {
        return true;
    }
    [[nodiscard]] virtual bool enable_partitioned_acceleration_structure() const noexcept {
        return true;
    }
    [[nodiscard]] virtual bool enable_ray_tracing_linear_swept_spheres() const noexcept {
        return true;
    }
    [[nodiscard]] virtual bool enable_dense_geometry_format() const noexcept {
        return true;
    }
    // A non-null result is a one-shot borrowed primary command buffer for this
    // acquisition. It must be in the initial state, be recordable for the
    // stream role's queue family, and remain alive together with its command
    // pool until the stream has completed the submission. The backend begins
    // and ends it, but never resets, frees, or reuses it. Return a fresh buffer
    // on every call; return null to use a backend-owned recyclable buffer.
    virtual VkCommandBuffer borrow_command_buffer(
        StreamTag stream_tag) noexcept { return nullptr; }
    // For imported handles, this must identify the loader through which their
    // instance ancestry was created. Mismatches are rejected before the
    // backend installs instance/device dispatch entry points.
    virtual VulkanLibPath external_vulkan_lib_path() noexcept { return {}; }
    virtual bool execute_command_buffer(VkCommandBuffer cmd_buffer) noexcept { return false; }
    // Returning false delegates the operation to the backend. Returning true
    // is a completion contract, not merely an ownership hint:
    //
    // - signal_semaphore()/wait_semaphore() must have submitted the requested
    //   Vulkan queue operation before returning, on the supplied queue, with
    //   independent forward progress after the callback returns.
    // - sync_semaphore() must not return true until the requested timeline
    //   value is actually complete.
    //
    // The backend holds its canonical mutex for `queue` while invoking the
    // queue callbacks. Implementations inherit Vulkan's external-
    // synchronization responsibility for any additional direct access to the
    // same VkQueue and must not recursively acquire that backend mutex.
    virtual bool signal_semaphore(VkQueue queue, VkSemaphore _semaphore, uint64_t index) noexcept { return false; }
    virtual bool wait_semaphore(VkQueue queue, VkSemaphore _semaphore, uint64_t index) noexcept { return false; }
    virtual bool sync_semaphore(VkSemaphore _semaphore, uint64_t index) noexcept { return false; }
    virtual bool load_dxc() const noexcept { return true; }
    // Initializes a client-owned Vulkan dispatch table with the exact loader
    // used by the backend. For Volk clients, this callback is the place to
    // call volkInitializeCustom(handler). That only loads loader-level entry
    // points: clients that call volkLoadDeviceTable() must additionally call
    // volkLoadInstanceOnly(instance) from readback_vulkan_device() first so
    // their private vkGetDeviceProcAddr is resolved for this VkInstance.
    virtual void init_volk(PFN_vkGetInstanceProcAddr handler) noexcept {}
    virtual luisa::vector<luisa::string> extra_instance_exts() noexcept { return {}; }
    virtual luisa::vector<luisa::string> extra_device_exts() noexcept { return {}; }
    virtual void get_defragment_function(luisa::move_only_function<void()> &&defragment_func) {}
    virtual void readback_vulkan_device(
        VkInstance instance,
        VkPhysicalDevice physical_device,
        VkDevice device,
        VkAllocationCallbacks *alloc_callback,
        VkPipelineCacheHeaderVersionOne const &pso_meta,
        VkQueue graphics_queue,
        VkQueue compute_queue,
        VkQueue copy_queue,
        uint32_t graphics_queue_family_index,
        uint32_t compute_queue_family_index,
        uint32_t copy_queue_family_index,
        IDxcCompiler3 *dxc_compiler,
        IDxcLibrary *dxc_library,
        IDxcUtils *dxc_utils) noexcept {}
    // before_states() is expanded against the bindless descriptor snapshot at
    // command-list entry; after_states() is expanded against the final
    // snapshot after all updates in that list. A bindless-array entry covers
    // both its descriptor-index buffer and every encoded member. The declared
    // image layout is applied independently to every mip and must not be
    // VK_IMAGE_LAYOUT_UNDEFINED when the snapshot contains images.
    //
    // A newly imported native image has no queryable current layout, so the
    // backend initially tracks it as VK_IMAGE_LAYOUT_UNDEFINED. The importer
    // must publish the real external layout/access state through
    // before_states() before preserving or consuming existing image contents.
    virtual luisa::span<VKCustomCmd::ResourceUsage const> before_states(uint64_t stream_handle) noexcept { return {}; }
    virtual luisa::span<VKCustomCmd::ResourceUsage const> after_states(uint64_t stream_handle) noexcept { return {}; }
    // Return a valid VkPhysicalDeviceXXXFeatures pNext chain containing only
    // feature structures not owned by the Vulkan backend. The backend rejects
    // repeated sTypes and backend-owned/promoted-alias collisions before
    // creating the logical device.
    virtual void *device_feature_settings() noexcept { return nullptr; }
};
}// namespace luisa::compute
