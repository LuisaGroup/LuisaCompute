#pragma once

#include <cstdint>

namespace lc::vk::detail {

enum class CommandBufferOwnership : uint8_t {
    BACKEND,
    BORROWED
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
    return {
        .reset_native_buffer = backend_owned,
        .recycle_native_buffer = backend_owned,
        .free_native_buffer = backend_owned};
}

}// namespace lc::vk::detail
