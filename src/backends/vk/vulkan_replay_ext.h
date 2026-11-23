// Backend-private implementation of `VulkanReplayExt` (plan P4, narrow v1).
//
// The engine holds no state of its own: every graph lives in
// `Device::replay_registry()`, which is also what `destroy_buffer`,
// `destroy_stream` and `~Device` invalidate/drain. That split is deliberate - it
// keeps invalidation correct even when the caller never asks for the extension.
//
// Everything this class does is observable through `Device::query`
// (`vk_profile_replay_*`): one `record()` moves `replay_records` (and the ordinary
// encode counters of the batch it encoded), one `replay()` moves
// `replay_submits` + exactly one `queue_submits`, and a refusal moves
// `replay_rejections` / `replay_update_refusals` without creating anything.
#pragma once

#include <luisa/backends/ext/vk_replay_ext.h>

namespace lc::vk {

class Device;

class VkReplayExtImpl final : public luisa::compute::VulkanReplayExt {
private:
    Device *_device;

public:
    explicit VkReplayExtImpl(Device *device) noexcept
        : _device{device} {}

    [[nodiscard]] luisa::compute::VulkanReplayCapabilities
    capabilities() const noexcept override;

    [[nodiscard]] luisa::compute::VulkanReplayRecordResult
    record(uint64_t stream_handle, luisa::compute::CommandList &&list) noexcept override;

    [[nodiscard]] luisa::compute::VulkanReplaySubmitResult
    replay(uint64_t replay_handle) noexcept override;

    [[nodiscard]] luisa::compute::VulkanReplayUpdateStatus
    update(uint64_t replay_handle, luisa::compute::CommandList &&list) noexcept override;

    [[nodiscard]] luisa::compute::VulkanReplayStatus
    destroy(uint64_t replay_handle) noexcept override;

    [[nodiscard]] luisa::compute::VulkanReplayGraphStatus
    status(uint64_t replay_handle) const noexcept override;

    [[nodiscard]] size_t live_replay_count() const noexcept override;
};

}// namespace lc::vk