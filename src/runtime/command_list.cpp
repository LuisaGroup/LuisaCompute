//
// Created by Mike Smith on 2021/3/18.
//

#include <runtime/command_list.h>

namespace luisa::compute {

void CommandList::_recycle() noexcept {
    if (!_commands.empty() && owner) {
        for (auto cmd : _commands) {
            cmd->recycle();
        }
    }
}

void CommandList::append(Command *cmd) noexcept {
    for (; cmd != nullptr; cmd = cmd->next()) {
        _commands.emplace_back(cmd);
    }
}

CommandList::CommandList(CommandList &&another) noexcept = default;

CommandList &CommandList::operator=(CommandList &&rhs) noexcept {
    if (&rhs != this) [[likely]] {
        _recycle();
        _commands = std::move(rhs._commands);
    }
    return *this;
}

CommandList::~CommandList() noexcept { _recycle(); }

}// namespace luisa::compute
