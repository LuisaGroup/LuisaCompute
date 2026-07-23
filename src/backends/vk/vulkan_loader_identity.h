#pragma once

#include <cstdint>

#include <luisa/core/stl/string.h>

namespace lc::vk::detail {

// Volk's default dispatch tables are process-global. Once an instance has
// been created through one Vulkan loader, replacing vkGetInstanceProcAddr
// would make those tables and the live handles disagree about their loader.
enum class VulkanLoaderSource : uint8_t {
    DEFAULT_LOADER,
    CUSTOM_LOADER
};

struct VulkanLoaderIdentityView {
    VulkanLoaderSource source{VulkanLoaderSource::DEFAULT_LOADER};
    // CUSTOM_LOADER identities use a normalized absolute search directory
    // and the exact requested library name. Both strings are empty for
    // DEFAULT_LOADER.
    luisa::string_view search_path{};
    luisa::string_view library_name{};
};

enum class VulkanLoaderInitializationStatus : uint8_t {
    INITIALIZE,
    REUSE,
    SOURCE_MISMATCH,
    SEARCH_PATH_MISMATCH,
    LIBRARY_NAME_MISMATCH
};

struct VulkanLoaderInitializationPlan {
    VulkanLoaderInitializationStatus status;

    [[nodiscard]] constexpr explicit operator bool() const noexcept {
        return status == VulkanLoaderInitializationStatus::INITIALIZE ||
               status == VulkanLoaderInitializationStatus::REUSE;
    }

    [[nodiscard]] constexpr bool should_initialize() const noexcept {
        return status == VulkanLoaderInitializationStatus::INITIALIZE;
    }
};

// `identity_pinned` is deliberately independent of a custom module handle:
// volkInitialize() owns the default loader internally and therefore leaves a
// client-side DynamicModule empty even though initialization has completed.
[[nodiscard]] constexpr VulkanLoaderInitializationPlan
plan_vulkan_loader_initialization(
    bool identity_pinned,
    VulkanLoaderIdentityView pinned,
    VulkanLoaderIdentityView requested) noexcept {
    if (!identity_pinned) {
        return {VulkanLoaderInitializationStatus::INITIALIZE};
    }
    if (pinned.source != requested.source) {
        return {VulkanLoaderInitializationStatus::SOURCE_MISMATCH};
    }
    if (pinned.search_path != requested.search_path) {
        return {VulkanLoaderInitializationStatus::SEARCH_PATH_MISMATCH};
    }
    if (pinned.library_name != requested.library_name) {
        return {VulkanLoaderInitializationStatus::LIBRARY_NAME_MISMATCH};
    }
    return {VulkanLoaderInitializationStatus::REUSE};
}

[[nodiscard]] constexpr const char *
vulkan_loader_initialization_status_name(
    VulkanLoaderInitializationStatus status) noexcept {
    switch (status) {
        case VulkanLoaderInitializationStatus::INITIALIZE: return "initialize";
        case VulkanLoaderInitializationStatus::REUSE: return "reuse";
        case VulkanLoaderInitializationStatus::SOURCE_MISMATCH: return "default/custom source mismatch";
        case VulkanLoaderInitializationStatus::SEARCH_PATH_MISMATCH: return "custom search-path mismatch";
        case VulkanLoaderInitializationStatus::LIBRARY_NAME_MISMATCH: return "custom library-name mismatch";
    }
    return "unknown Vulkan loader identity error";
}

[[nodiscard]] constexpr const char *vulkan_loader_source_name(
    VulkanLoaderSource source) noexcept {
    switch (source) {
        case VulkanLoaderSource::DEFAULT_LOADER: return "default";
        case VulkanLoaderSource::CUSTOM_LOADER: return "custom";
    }
    return "unknown";
}

}// namespace lc::vk::detail
