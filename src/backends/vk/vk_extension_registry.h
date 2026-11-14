#pragma once

#include <cstddef>
#include <cstdint>

#include <vulkan/vulkan_core.h>

#include <luisa/core/stl/memory.h>
#include <luisa/core/stl/string.h>
#include <luisa/core/stl/vector.h>

namespace luisa::compute {
class VulkanDeviceConfigExt;
}// namespace luisa::compute

namespace lc::vk::detail {

// Every *optional* Vulkan device extension the backend knows how to negotiate.
//
// "Optional" means the backend keeps a fallback path when the extension is
// unavailable or switched off through VulkanDeviceConfigExt. Required
// extensions (timeline semaphore, synchronization2, ...) are negotiated
// separately in device.cpp and are not part of this registry.
//
// The identifiers are stable because VulkanExtensionStatusExt exposes them by
// name to users, and backend modules branch on the negotiated state.
enum class OptionalExtension : uint32_t {
    DESCRIPTOR_HEAP,
    DEVICE_ADDRESS_COMMANDS,
    UNIFIED_IMAGE_LAYOUTS,
    COPY_MEMORY_INDIRECT,
    MEMORY_DECOMPRESSION,
    ZERO_INITIALIZE_DEVICE_MEMORY,
    INTERNALLY_SYNCHRONIZED_QUEUES,
    EXTENDED_FLAGS,
    IMAGE_TILING_CONTROL,
    PRESENT_ID2,
    PRESENT_WAIT2,
    PRESENT_MODE_FIFO_LATEST_READY,
    SURFACE_MAINTENANCE1,
    SWAPCHAIN_MAINTENANCE1,
    PRESENT_TIMING,
    MULTISAMPLED_RENDER_TO_SWAPCHAIN,
    SHADER_FMA,
    SHADER_64BIT_INDEXING,
    SHADER_UNIFORM_BUFFER_UNSIZED_ARRAY,
    SHADER_LONG_VECTOR,
    SHADER_SUBGROUP_PARTITIONED,
    SHADER_SPLIT_BARRIER,
    SHADER_OCP_MICROSCALING_TYPES,
    SHADER_CONSTANT_DATA,
    SHADER_ABORT,
    RAY_TRACING_INVOCATION_REORDER,
    OPACITY_MICROMAP,
    CUSTOM_RESOLVE,
    FRAGMENT_DENSITY_MAP_OFFSET,
    PRIMITIVE_RESTART_INDEX,
    TEXTURE_COMPRESSION_ASTC_3D,
    PIPELINE_LIBRARY_GROUP_HANDLES,
    EXTERNAL_MEMORY_METAL,
    DEPTH_CLAMP_ZERO_ONE,
    MAINTENANCE8,
    MAINTENANCE9,
    MAINTENANCE10,
    MAINTENANCE11,
    MAINTENANCE12,
    ROBUSTNESS2,
    TENSOR_ARM,
    DATA_GRAPH_ARM,
    TILE_SHADING_QCOM,
    TILE_MEMORY_HEAP_QCOM,
    PARTITIONED_ACCELERATION_STRUCTURE,
    RAY_TRACING_LINEAR_SWEPT_SPHERES,
    DENSE_GEOMETRY_FORMAT,
    SHADER_MIXED_FLOAT_DOT_PRODUCT,
    COUNT
};

inline constexpr size_t optional_extension_count =
    static_cast<size_t>(OptionalExtension::COUNT);

// Storage reserved for one VkPhysicalDevice*Features* structure. The largest
// structure currently negotiated (VkPhysicalDeviceTileShadingFeaturesQCOM) is
// 72 bytes; the bound is asserted in vk_extension_registry.cpp so a future
// extension with a bigger structure fails to compile instead of overflowing.
inline constexpr size_t optional_extension_feature_storage_size = 128u;

// Negotiated state of one optional extension.
struct OptionalExtensionState {
    OptionalExtension id{OptionalExtension::COUNT};
    // Name used for this negotiation. Aliases (KHR/EXT twins) are resolved to
    // whichever name the physical device actually exposes.
    const char *name{nullptr};
    bool supported_by_device{false};
    bool requested_by_config{false};
    bool enabled{false};
};

// Static description of one optional extension. Everything here is immutable;
// the mutable negotiation result lives in OptionalExtensionRegistry.
struct OptionalExtensionDescriptor {
    OptionalExtension id{OptionalExtension::COUNT};
    // Preferred extension name.
    const char *name{nullptr};
    // Acceptable alternative name (KHR/EXT twins, NV/EXT twins), or nullptr.
    const char *alias_name{nullptr};
    // Another *optional* extension that must end up enabled for this one to be
    // usable, or nullptr. Evaluated after every entry has been resolved, so
    // table order does not matter.
    const char *requires_extension{nullptr};
    // Whether VulkanDeviceConfigExt allows this extension. Never nullptr.
    bool (*requested)(const luisa::compute::VulkanDeviceConfigExt &config) noexcept;
    // VkPhysicalDevice*Features* structure required by the extension, or
    // VK_STRUCTURE_TYPE_MAX_ENUM when the extension has no feature structure
    // (in which case make_feature/check_and_enable are nullptr).
    VkStructureType feature_structure_type{VK_STRUCTURE_TYPE_MAX_ENUM};
    // Placement-constructs the feature structure inside `storage` with every
    // bit cleared and returns its address.
    void *(*make_feature)(void *storage) noexcept;
    // Returns false when the physical device does not support every feature bit
    // the backend needs; otherwise sets exactly those bits to VK_TRUE and
    // returns true. Optional bits are never set.
    bool (*check_and_enable)(void *feature) noexcept;
    // sType to use when the *alias* name is the one the physical device
    // exposes. Vulkan aliases usually reuse one value (which is why this
    // defaults to VK_STRUCTURE_TYPE_MAX_ENUM meaning "same as
    // feature_structure_type"), but a few KHR retrofits do not:
    // VK_EXT/VK_KHR_opacity_micromap, VK_EXT/VK_NV_ray_tracing_invocation_reorder
    // and VK_KHR/VK_NV_copy_memory_indirect have distinct values, and probing
    // with the wrong one silently reports "feature unavailable".
    VkStructureType alias_feature_structure_type{VK_STRUCTURE_TYPE_MAX_ENUM};
    // Instance-level extension: matched against the instance extension list
    // passed to `select()`, not against the physical device's device extension
    // list. Only a handful of optional extensions are instance-level
    // (VK_KHR_surface_maintenance1); the rest are device extensions.
    bool instance_level{false};
};

// Negotiates the optional device extensions of the Vulkan backend.
//
// Usage, in this order:
//
//   OptionalExtensionRegistry registry;
//   registry.select(supported_device_extensions, config, imported_device);
//   if (auto chain = registry.feature_query_chain()) {
//       VkPhysicalDeviceFeatures2 features2{..., .pNext = chain};
//       vkGetPhysicalDeviceFeatures2(physical_device, &features2);
//   }
//   registry.resolve();                       // final supported/requested verdict
//   feature_next = registry.device_feature_chain(feature_next);
//   registry.append_enabled_names(device_extension_names);
//
// `select()` must not be called twice; `resolve()` is idempotent.
class OptionalExtensionRegistry {

public:
    OptionalExtensionRegistry() noexcept;
    ~OptionalExtensionRegistry() noexcept;
    OptionalExtensionRegistry(OptionalExtensionRegistry const &) = delete;
    OptionalExtensionRegistry(OptionalExtensionRegistry &&) noexcept;
    OptionalExtensionRegistry &operator=(OptionalExtensionRegistry const &) = delete;
    OptionalExtensionRegistry &operator=(OptionalExtensionRegistry &&) noexcept;

    // Matches every descriptor against the names the physical device exposes
    // and against VulkanDeviceConfigExt. An imported logical device can never
    // enable an optional extension: Vulkan cannot report which extensions or
    // features the borrowed device was created with, so `supported` stays
    // false for it and every switch is reported as "not enabled".
    void select(luisa::span<const luisa::string> supported_device_extensions,
                luisa::span<const luisa::string> enabled_instance_extensions,
                const luisa::compute::VulkanDeviceConfigExt *config,
                bool imported_device) noexcept;
    // Convenience overload for a device-only naming set (host tests, imported
    // devices): instance-level entries then stay unsupported.
    void select(luisa::span<const luisa::string> supported_device_extensions,
                const luisa::compute::VulkanDeviceConfigExt *config,
                bool imported_device) noexcept {
        select(supported_device_extensions, {}, config, imported_device);
    }

    // Head of the pNext chain handed to vkGetPhysicalDeviceFeatures2. Only
    // entries that are supported *and* requested are chained, so a driver that
    // never heard of an extension is never asked about it. nullptr when the
    // chain is empty.
    [[nodiscard]] void *feature_query_chain() const noexcept;

    // Records an extension the backend decided to enable for a reason outside
    // this registry (swapchain, ray tracing pipeline, external memory, ...), so
    // descriptor `requires_extension` checks can see it.
    void note_enabled_extension(luisa::string_view name) noexcept;

    // Applies the probed feature bits: an entry is enabled when it is supported,
    // requested and every required feature bit is VK_TRUE.
    void resolve() noexcept;

    // Prepends the feature structures of every enabled entry to `tail` and
    // returns the new head. The returned pointers are owned by the registry,
    // which must outlive vkCreateDevice.
    [[nodiscard]] void *device_feature_chain(void *tail) noexcept;

    // Names of the enabled extensions, ready for
    // VkDeviceCreateInfo::ppEnabledExtensionNames.
    [[nodiscard]] luisa::span<const char *const> enabled_extension_names() const noexcept;

    [[nodiscard]] bool supported(OptionalExtension ext) const noexcept;
    [[nodiscard]] bool requested(OptionalExtension ext) const noexcept;
    [[nodiscard]] bool enabled(OptionalExtension ext) const noexcept;

    [[nodiscard]] OptionalExtensionState const &state(OptionalExtension ext) const noexcept;
    [[nodiscard]] luisa::span<const OptionalExtensionState> states() const noexcept;

    // Enables an extension the backend needs unconditionally, provided the
    // negotiation can prove the device supports it: the entry must be exposed by
    // the physical device and either needs no feature structure or already
    // passed its feature-bit probe. Returns false (and leaves the registry
    // untouched) otherwise, in particular for an extension the configuration
    // switched off: `select()` never probes those, so enabling the name without
    // its feature bits would produce an invalid logical device.
    bool force_enable(OptionalExtension ext) noexcept;

    // One human-readable line per extension, used by the device log and by the
    // status extension.
    [[nodiscard]] luisa::string status_string() const noexcept;

private:
    struct Entry {
        OptionalExtensionDescriptor desc;
        // Extension name actually used for this negotiation (primary or alias).
        const char *name{nullptr};
        bool supported{false};
        bool requested{false};
        bool enabled{false};
        alignas(16) std::byte storage[optional_extension_feature_storage_size]{};
        void *feature{nullptr};
    };
    luisa::vector<Entry> _entries;
    // Extensions the backend enabled outside this registry, so descriptor
    // `requires_extension` checks can see them.
    luisa::vector<luisa::string> _registered_extensions;
    // Enabled names handed to VkDeviceCreateInfo.
    luisa::vector<const char *> _enabled_names;
    // Flattened view used by the public status extension.
    OptionalExtensionState _states[optional_extension_count]{};
    uint64_t _supported_bits{0u};
    uint64_t _requested_bits{0u};
    uint64_t _enabled_bits{0u};
    void *_query_head{nullptr};
    void *_device_head{nullptr};

    void refresh_states() noexcept;
};

[[nodiscard]] const char *optional_extension_name(OptionalExtension ext) noexcept;

// Resolves a name (primary or alias) to the registry identifier. Returns
// OptionalExtension::COUNT when the name is not handled by the registry.
[[nodiscard]] OptionalExtension optional_extension_from_name(
    luisa::string_view name) noexcept;

}// namespace lc::vk::detail
