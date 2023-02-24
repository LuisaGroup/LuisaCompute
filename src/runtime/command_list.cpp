//
// Created by Mike Smith on 2021/3/18.
//

#include <runtime/command.h>
#include <runtime/command_list.h>
#include <core/logging.h>

namespace luisa::compute {

CommandList::~CommandList() noexcept {
    if (!empty()) [[unlikely]] {
        LUISA_ERROR_WITH_LOCATION(
            "Destructing non-empty command list. "
            "Did you forget to commit?");
    }
}

void CommandList::reserve(size_t command_size, size_t callback_size) noexcept {
    if (command_size) { _commands.reserve(command_size); }
    if (callback_size) { _callbacks.reserve(callback_size); }
}

void CommandList::clear() noexcept {
    _commands.clear();
    _callbacks.clear();
}

CommandList &CommandList::append(unique_ptr<Command> &&cmd) noexcept {
    _commands.emplace_back(std::move(cmd));
    return *this;
}

CommandList &CommandList::append(move_only_function<void()> &&callback) noexcept {
    _callbacks.emplace_back(std::move(callback));
    return *this;
}

CommandList &CommandList::operator<<(luisa::unique_ptr<Command> &&cmd) noexcept {
    return append(std::move(cmd));
}

CommandList &CommandList::operator<<(luisa::move_only_function<void()> &&callback) noexcept {
    return append(std::move(callback));
}

std::pair<CommandList::CommandContainer,
          CommandList::CallbackContainer>
CommandList::steal() &&noexcept {
    return std::make_pair(std::move(_commands), std::move(_callbacks));
}

CommandList CommandList::create(size_t reserved_command_size, size_t reserved_callback_size) noexcept {
    CommandList list{};
    list.reserve(reserved_command_size, reserved_callback_size);
    return list;
}

}// namespace luisa::compute
