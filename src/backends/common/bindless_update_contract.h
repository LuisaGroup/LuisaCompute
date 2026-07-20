#pragma once

#include <cstddef>
#include <cstdint>

#include <luisa/runtime/rhi/command.h>

namespace lc::bindless_update_detail {

using Operation =
    luisa::compute::BindlessArrayUpdateCommand::Operation;

[[nodiscard]] constexpr bool valid_operation(
    Operation operation) noexcept {
    switch (operation) {
        case Operation::NONE:
        case Operation::EMPLACE:
        case Operation::REMOVE: return true;
    }
    return false;
}

[[nodiscard]] constexpr bool changes_slot(
    Operation operation) noexcept {
    return operation == Operation::EMPLACE ||
           operation == Operation::REMOVE;
}

[[nodiscard]] constexpr bool slot_in_bounds(
    size_t slot, size_t slot_count) noexcept {
    return slot < slot_count;
}

}// namespace lc::bindless_update_detail
