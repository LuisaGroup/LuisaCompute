//
// Created by Mike Smith on 2021/3/18.
//

#include <runtime/command.h>
#include <runtime/command_list.h>
#include <core/logging.h>
namespace luisa::compute {

CommandList &CommandList::append(luisa::unique_ptr<Command> &&cmd) noexcept {
    _commands.emplace_back(std::move(cmd));
    return *this;
}
CommandList::~CommandList() noexcept {
    if (!empty()) [[unlikely]] {
        LUISA_ERROR_WITH_LOCATION(
            "Destructing non-empty command list. "
            "Did you forget to commit?");
    }
}
luisa::vector<luisa::unique_ptr<Command>> CommandList::steal_commands() &&noexcept {
    luisa::vector<luisa::unique_ptr<Command>> cmds;
    cmds.swap(_commands);
    return cmds;
}
void CommandList::reserve(size_t size) noexcept {
    _commands.reserve(size);
}
}// namespace luisa::compute
