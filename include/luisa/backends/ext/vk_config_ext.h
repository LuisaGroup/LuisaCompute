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
    [[nodiscard]] virtual bool enable_raytracing_feature() const noexcept {
        return true;
    }
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
