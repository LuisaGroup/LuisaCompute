#pragma once

#include <cstdint>
#include <limits>

namespace lc::vk::detail {

enum class TimelineValueIncrementStatus : uint8_t {
    SUCCESS,
    VALUE_OVERFLOW
};

struct TimelineValueIncrementPlan {
    TimelineValueIncrementStatus status;
    uint64_t value;

    [[nodiscard]] constexpr explicit operator bool() const noexcept {
        return status == TimelineValueIncrementStatus::SUCCESS;
    }
};

[[nodiscard]] constexpr TimelineValueIncrementPlan
plan_timeline_value_increment(
    uint64_t previous_value,
    uint64_t increment) noexcept {
    if (increment >
        std::numeric_limits<uint64_t>::max() - previous_value) {
        return {
            .status = TimelineValueIncrementStatus::VALUE_OVERFLOW};
    }
    return {
        .status = TimelineValueIncrementStatus::SUCCESS,
        .value = previous_value + increment};
}

// Vulkan limits every pending timeline-semaphore value to a device-specific
// window around both the current counter and every other pending operation.
// The backend submits signals monotonically, so all of its outstanding values
// lie in the closed interval [current_value, tracked_signal_value]. Checking
// the interval endpoints is therefore sufficient for a new signal or wait.
enum class TimelineSemaphoreValueStatus : uint8_t {
    SUCCESS,
    ZERO_MAX_VALUE_DIFFERENCE,
    TRACKED_SIGNAL_BEHIND_CURRENT,
    TRACKED_SIGNAL_RANGE_EXCEEDED,
    SIGNAL_VALUE_NOT_INCREASING,
    WAIT_VALUE_AHEAD_OF_TRACKED_SIGNAL,
    MAX_VALUE_DIFFERENCE_EXCEEDED
};

[[nodiscard]] constexpr const char *
timeline_semaphore_value_status_name(
    TimelineSemaphoreValueStatus status) noexcept {
    switch (status) {
        case TimelineSemaphoreValueStatus::SUCCESS: return "success";
        case TimelineSemaphoreValueStatus::ZERO_MAX_VALUE_DIFFERENCE: return "maxTimelineSemaphoreValueDifference is zero";
        case TimelineSemaphoreValueStatus::TRACKED_SIGNAL_BEHIND_CURRENT: return "tracked signal is behind the current GPU counter";
        case TimelineSemaphoreValueStatus::TRACKED_SIGNAL_RANGE_EXCEEDED: return "tracked outstanding signal exceeds maxTimelineSemaphoreValueDifference";
        case TimelineSemaphoreValueStatus::SIGNAL_VALUE_NOT_INCREASING: return "signal value is not strictly increasing";
        case TimelineSemaphoreValueStatus::WAIT_VALUE_AHEAD_OF_TRACKED_SIGNAL: return "wait value is ahead of every submitted signal";
        case TimelineSemaphoreValueStatus::MAX_VALUE_DIFFERENCE_EXCEEDED: return "value exceeds maxTimelineSemaphoreValueDifference";
    }
    return "unknown";
}

struct TimelineSemaphoreValuePlan {
    TimelineSemaphoreValueStatus status;
    bool already_satisfied;

    [[nodiscard]] constexpr explicit operator bool() const noexcept {
        return status == TimelineSemaphoreValueStatus::SUCCESS;
    }
};

[[nodiscard]] constexpr TimelineSemaphoreValuePlan
plan_timeline_semaphore_signal(
    uint64_t current_value,
    uint64_t tracked_signal_value,
    uint64_t signal_value,
    uint64_t max_value_difference) noexcept {
    // Per the Vulkan spec, maxTimelineSemaphoreValueDifference == 0 means
    // "no limit". Normalize it to the largest representable window so
    // conformant drivers that report 0 are not rejected.
    if (max_value_difference == 0u) {
        max_value_difference = std::numeric_limits<uint64_t>::max();
    }
    if (current_value > tracked_signal_value) {
        return {
            .status = TimelineSemaphoreValueStatus::
                TRACKED_SIGNAL_BEHIND_CURRENT};
    }
    if (tracked_signal_value - current_value >
        max_value_difference) {
        return {
            .status = TimelineSemaphoreValueStatus::
                TRACKED_SIGNAL_RANGE_EXCEEDED};
    }
    if (signal_value <= tracked_signal_value) {
        return {
            .status = TimelineSemaphoreValueStatus::
                SIGNAL_VALUE_NOT_INCREASING};
    }
    if (signal_value - current_value > max_value_difference) {
        return {
            .status = TimelineSemaphoreValueStatus::
                MAX_VALUE_DIFFERENCE_EXCEEDED};
    }
    return {
        .status = TimelineSemaphoreValueStatus::SUCCESS,
        .already_satisfied = false};
}

[[nodiscard]] constexpr TimelineSemaphoreValuePlan
plan_timeline_semaphore_wait(
    uint64_t current_value,
    uint64_t tracked_signal_value,
    uint64_t wait_value,
    uint64_t max_value_difference) noexcept {
    // maxTimelineSemaphoreValueDifference == 0 means "no limit".
    if (max_value_difference == 0u) {
        max_value_difference = std::numeric_limits<uint64_t>::max();
    }
    if (current_value > tracked_signal_value) {
        return {
            .status = TimelineSemaphoreValueStatus::
                TRACKED_SIGNAL_BEHIND_CURRENT};
    }
    // Do not submit an already-satisfied wait. Apart from avoiding redundant
    // queue work, this keeps a very old logical wait out of Vulkan's bounded
    // pending-value window.
    if (wait_value <= current_value) {
        return {
            .status = TimelineSemaphoreValueStatus::SUCCESS,
            .already_satisfied = true};
    }
    if (tracked_signal_value - current_value >
        max_value_difference) {
        return {
            .status = TimelineSemaphoreValueStatus::
                TRACKED_SIGNAL_RANGE_EXCEEDED};
    }
    if (wait_value > tracked_signal_value) {
        return {
            .status = TimelineSemaphoreValueStatus::
                WAIT_VALUE_AHEAD_OF_TRACKED_SIGNAL};
    }
    // wait_value lies between the current counter and the highest outstanding
    // signal, so the tracked interval check above bounds it against both.
    return {
        .status = TimelineSemaphoreValueStatus::SUCCESS,
        .already_satisfied = false};
}

enum class InternalTimelineWaitStatus : uint8_t {
    SUCCESS,
    GPU_SIGNAL_AHEAD_OF_LOGICAL_FENCE
};

struct InternalTimelineWaitPlan {
    InternalTimelineWaitStatus status;
    uint64_t wait_value;

    [[nodiscard]] constexpr explicit operator bool() const noexcept {
        return status == InternalTimelineWaitStatus::SUCCESS;
    }
};

// Logical stream fences also cover callback-only and skipped-present work.
// Such host-only values must never be submitted as Vulkan semaphore waits.
[[nodiscard]] constexpr InternalTimelineWaitPlan
plan_internal_timeline_wait(
    uint64_t logical_fence,
    uint64_t gpu_signal_fence) noexcept {
    if (gpu_signal_fence > logical_fence) {
        return {
            .status = InternalTimelineWaitStatus::
                GPU_SIGNAL_AHEAD_OF_LOGICAL_FENCE};
    }
    return {
        .status = InternalTimelineWaitStatus::SUCCESS,
        .wait_value = gpu_signal_fence};
}

}// namespace lc::vk::detail
