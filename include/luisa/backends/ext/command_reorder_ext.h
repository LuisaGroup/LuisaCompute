#pragma once
#include <luisa/core/stl/string.h>
#include <luisa/runtime/rhi/device_interface.h>

namespace luisa::compute {

// Runtime control for the backend command-reordering pass, implemented by the
// Vulkan and DirectX backends (obtained with `Device::extension<CommandReorderExt>()`).
//
// Reordering groups consecutive commands whose resource accesses do not alias
// into a single barrier-free layer, so a batch of small dispatches can overlap
// on the GPU instead of being serialized by a barrier between every pair of
// commands. Turning it off makes the backend submit the batch in strict order,
// which is the baseline to A/B compare against.
//
// The switch is sampled when a backend starts a command batch: a change applies
// to the next submission and never disturbs a batch that is already recorded.
// It is seeded from the device-config extension
// (`VulkanDeviceConfigExt::enable_command_reorder()` /
// `DirectXDeviceConfigExt::EnableCommandReorder()`), and
// `LUISA_DISABLE_COMMAND_REORDER=1` forces it off for the whole process.
class CommandReorderExt : public DeviceExtension {
public:
    static constexpr luisa::string_view name = "CommandReorderExt";

    [[nodiscard]] virtual bool command_reorder_enabled() const noexcept = 0;
    virtual void set_command_reorder_enabled(bool enabled) noexcept = 0;

protected:
    ~CommandReorderExt() = default;
};

}// namespace luisa::compute
