#pragma once
#include "env_flag.h"
#include <atomic>

namespace luisa::compute {

// Command-reordering switch shared by the Vulkan and DirectX backends.
//
// The reorder pass (see command_reorder_visitor.h) groups consecutive commands
// whose resource accesses do not alias into one barrier-free layer, so a batch
// of small dispatches can overlap on the GPU instead of being serialized by a
// barrier between every pair of commands. Disabling it makes a backend emit the
// batch in strict submission order, which is what the A/B benchmarks measure.
//
// Three inputs, in increasing precedence:
// - the device-config extension value (`VulkanDeviceConfigExt::
//   enable_command_reorder()` / `DirectXDeviceConfigExt::EnableCommandReorder()`)
//   seeds the switch during device construction;
// - `CommandReorderExt::set_command_reorder_enabled()` changes it at runtime;
// - `LUISA_DISABLE_COMMAND_REORDER=1` forces it off process-wide and cannot be
//   re-enabled by either of the above.
//
// Backends read the switch once per command batch, so a change takes effect
// from the next submission and never disturbs a batch that is being recorded.
class CommandReorderSwitch {
public:
    void seed(bool enabled) noexcept {
        _enabled.store(enabled, std::memory_order_relaxed);
    }
    void set_enabled(bool enabled) noexcept {
        _enabled.store(enabled, std::memory_order_relaxed);
    }
    [[nodiscard]] bool enabled() const noexcept {
        // The kill switch exists to profile or CI-run a backend without code
        // changes, so it is resolved once instead of on every submitted batch.
        static const auto env_disabled = detail::env_flag("LUISA_DISABLE_COMMAND_REORDER");
        return !env_disabled && _enabled.load(std::memory_order_relaxed);
    }

private:
    std::atomic<bool> _enabled{true};
};

}// namespace luisa::compute
