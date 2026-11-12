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
    // backend's stable extension order.
    [[nodiscard]] virtual luisa::span<const VulkanExtensionStatus> extensions() const noexcept = 0;
    [[nodiscard]] virtual bool is_supported(luisa::string_view extension) const noexcept = 0;
    [[nodiscard]] virtual bool is_enabled(luisa::string_view extension) const noexcept = 0;
    [[nodiscard]] virtual bool is_active(luisa::string_view extension) const noexcept = 0;

protected:
    ~VulkanExtensionStatusExt() = default;
};

}// namespace luisa::compute
