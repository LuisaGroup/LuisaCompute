#pragma once

#include <cstdint>

namespace lc::vk::detail {

enum class CommandBufferOwnership : uint8_t {
    BACKEND,
    BORROWED,
    // Plan P4: recorded once into a dedicated pool without the reset bit, then
    // submitted N times (VK_COMMAND_BUFFER_USAGE_SIMULTANEOUS_USE_BIT) and never
    // reset, re-begun or recycled. The backend frees the native buffer exactly
    // once, when the replay graph that owns it is destroyed.
    PERSISTENT
};

struct CommandBufferRetirementPlan {
    bool reset_native_buffer;
    bool recycle_native_buffer;
    bool free_native_buffer;
};

[[nodiscard]] constexpr CommandBufferRetirementPlan
plan_command_buffer_retirement(
    CommandBufferOwnership ownership) noexcept {
    auto backend_owned = ownership == CommandBufferOwnership::BACKEND;
    // A PERSISTENT buffer is never reset and never recycled (it is resubmitted as
    // it is), but the backend still owns the native object and frees it exactly
    // once - through the replay graph, not through `~CommandBuffer`.
    return {
        .reset_native_buffer = backend_owned,
        .recycle_native_buffer = backend_owned,
        .free_native_buffer = backend_owned ||
                              ownership == CommandBufferOwnership::PERSISTENT};
}

// Whether `CommandBuffer::release_native_batch()` may hand this buffer (and its
// state, and therefore the command pool the buffer was allocated from) to a replay
// graph. Only a PERSISTENT buffer qualifies:
//   * a BACKEND buffer is owned by the stream's retirement FIFO or by
//     `~CommandBuffer`, which both free it against the same pool - releasing it
//     would produce a second free of the same native handle;
//   * a BORROWED buffer belongs to the config ext's pool, which the backend created
//     neither for this recording nor for a resubmission, so it must never be freed
//     here at all (`free_native_buffer` is false for it by design).
// This is the sole acquisition path of a recording (`Stream::_begin_batch_record`
// with PERSISTENT), stated as a predicate so a test can assert it without a device.
[[nodiscard]] constexpr bool may_release_native_batch(
    CommandBufferOwnership ownership) noexcept {
    return ownership == CommandBufferOwnership::PERSISTENT;
}

}// namespace lc::vk::detail
