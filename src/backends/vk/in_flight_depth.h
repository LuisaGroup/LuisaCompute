#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <string_view>

// In-flight depth of a Vulkan stream (plan P1.5).
//
// A stream submits one command batch per `Event::_signal`, and every batch
// carries the next value of the stream's timeline semaphore. Before recording a
// new batch the host throttles itself so that no more than
// `max_frames_in_flight` batches are still outstanding on the GPU:
//
//   F = last reserved fence value (== number of batches submitted so far)
//   if F > depth:  wait until value (F - depth) has completed, then submit F+1
//
// With depth = 2 this is exactly the historical hard-coded rule
// (`if (last_fence() > 2) sync(last_fence() - 2)` in `Stream::dispatch` and its
// present twin), which is why depth 2 must stay bit-identical: the subtraction
// is only evaluated under `F > depth`, so it can never underflow, and the target
// it produces is the value the old expression produced. Depth 3 allows exactly
// one additional batch in flight before the host throttles.
//
// The plan is device-independent (no Vulkan handle, no device entry point), so
// the arithmetic - including the compatibility envelope above - is covered on
// the host by src/tests/unit/ext/test_vk_in_flight_depth.cpp.
namespace lc::vk::detail {

// Backend default when the DeviceConfig carries no VulkanDeviceConfigExt, or
// when the extension keeps the getter at its default. Must match
// VulkanDeviceConfigExt::max_frames_in_flight()'s default return value.
inline constexpr uint32_t kDefaultFramesInFlight = 3u;

enum class FramesInFlightStatus : uint8_t {
    SUCCESS,
    ZERO_DEPTH
};

[[nodiscard]] constexpr const char *
frames_in_flight_status_name(FramesInFlightStatus status) noexcept {
    switch (status) {
        case FramesInFlightStatus::SUCCESS: return "success";
        case FramesInFlightStatus::ZERO_DEPTH:
            return "max_frames_in_flight is zero";
    }
    return "unknown";
}

struct FramesInFlightPlan {
    FramesInFlightStatus status;
    // Valid only when `status == SUCCESS`; 0 otherwise.
    uint32_t depth;
    [[nodiscard]] constexpr explicit operator bool() const noexcept {
        return status == FramesInFlightStatus::SUCCESS;
    }
};

// Validates the configured depth. Zero is rejected instead of being read as
// "unlimited": a caller that wants no host throttling disables it through
// `DeviceConfig::inqueue_buffer_limit`, which is the existing switch, and a
// depth of zero would otherwise produce a target of `F - 0 == F`, i.e. a wait
// for the batch that is about to be submitted. Any value >= 1 is accepted; the
// Device additionally checks it against the physical device's timeline value
// window, because every outstanding batch spends one timeline value.
[[nodiscard]] constexpr FramesInFlightPlan
plan_frames_in_flight(uint32_t requested_depth) noexcept {
    if (requested_depth == 0u) {
        return {.status = FramesInFlightStatus::ZERO_DEPTH, .depth = 0u};
    }
    return {.status = FramesInFlightStatus::SUCCESS, .depth = requested_depth};
}

struct InFlightThrottlePlan {
    // Whether the host has to consider throttling before this submission.
    bool enter;
    // The timeline value that must have completed before submitting. Only
    // meaningful when `enter` is true. A plan never produces the target 0: the
    // engine only emits it under `last_fence > depth`, which implies
    // `last_fence - depth >= 1`, and every submitted batch carries a value >= 1.
    uint64_t target;
};

// `last_fence` is the value of the previously submitted batch (0 before the
// first one), `depth` must have gone through `plan_frames_in_flight`.
[[nodiscard]] constexpr InFlightThrottlePlan
plan_in_flight_throttle(uint64_t last_fence, uint32_t depth) noexcept {
    // A depth of 0 is unreachable through the Device (it fails closed at
    // create), but the guard keeps a hypothetical caller from computing a
    // nonsensical target: it degrades to "no throttle" instead of waiting for a
    // value that equals or exceeds `last_fence`, which would never be satisfied
    // and would deadlock the host.
    if (depth == 0u) {
        return {.enter = false, .target = 0u};
    }
    // The engine of the depth rule. `last_fence <= depth` is the "there is room
    // for one more batch" case, and it is also what makes the subtraction safe
    // (unsigned underflow is impossible because the difference is only formed
    // when it is at least 1). For depth 2 this is byte for byte the historical
    // condition `last_fence > 2` and the historical target `last_fence - 2`.
    auto depth64 = static_cast<uint64_t>(depth);
    if (last_fence <= depth64) {
        return {.enter = false, .target = 0u};
    }
    return {.enter = true, .target = last_fence - depth64};
}

// Upper bound of the configurable busy-spin window. A budget above it buys
// nothing (the poll would outlive the wait it is meant to cover) and a negative
// or otherwise malformed value must never be able to turn into one.
inline constexpr uint64_t kMaxSpinBudgetUs = 100'000ull;
// A `LUISA_INTRIN_PAUSE` retires in roughly 10 ns on this class of CPU, i.e.
// ~100 per microsecond. The exact ratio does not matter: the budget only decides
// how long a thread polls before it blocks, never whether it observes the
// counter.
inline constexpr uint64_t kPausesPerUs = 100ull;

// Parses an optional microsecond spin budget from an environment variable value
// into a `LUISA_INTRIN_PAUSE` iteration count. Accepts strict non-empty ASCII
// unsigned decimal digits (including "0" to disable spinning); rejects negative
// values, whitespace, and trailing garbage by falling back to `default_us`.
[[nodiscard]] constexpr uint32_t parse_spin_pause_iterations(
    std::string_view env_value, uint64_t default_us) noexcept {
    auto clamped_default = std::min(default_us, kMaxSpinBudgetUs);
    if (env_value.empty()) {
        return static_cast<uint32_t>(clamped_default * kPausesPerUs);
    }
    uint64_t parsed = 0u;
    for (char ch : env_value) {
        if (ch < '0' || ch > '9') {
            return static_cast<uint32_t>(clamped_default * kPausesPerUs);
        }
        auto digit = static_cast<uint64_t>(ch - '0');
        if (parsed <= kMaxSpinBudgetUs) {
            parsed = parsed * 10ull + digit;
            if (parsed > kMaxSpinBudgetUs) {
                parsed = kMaxSpinBudgetUs;
            }
        }
    }
    return static_cast<uint32_t>(parsed * kPausesPerUs);
}

[[nodiscard]] constexpr uint32_t parse_spin_pause_iterations(
    const char *env_value, uint64_t default_us) noexcept {
    return parse_spin_pause_iterations(
        env_value == nullptr ? std::string_view{} : std::string_view{env_value},
        default_us);
}

// A cached descriptor-set bucket whose layout was destroyed still owns native
// sets in the state's pools; once this many of them piled up, rewinding the pool
// cursors is no longer equivalent to a reset and the pools must really be reset.
inline constexpr size_t kDescriptorPoolOrphanResetThreshold = 1024u;

// Whether `CommandBufferState::reset_native` has to call `vkResetDescriptorPool`
// instead of only rewinding the cache cursors: an allocation outside the cache,
// an overflow pool that was touched (the cache does not record which pool a set
// came from), or enough orphaned sets to make the leak visible.
[[nodiscard]] constexpr bool should_reset_descriptor_pools(
    bool has_uncached_desc_allocs,
    size_t active_desc_pool_index,
    size_t orphaned_desc_sets) noexcept {
    return has_uncached_desc_allocs ||
           active_desc_pool_index != 0u ||
           orphaned_desc_sets >= kDescriptorPoolOrphanResetThreshold;
}

}// namespace lc::vk::detail