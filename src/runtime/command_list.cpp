//
// Created by Mike Smith on 2021/3/18.
//

#include <runtime/command.h>
#include <runtime/command_list.h>
#include <core/logging.h>
namespace luisa::compute {

CommandList &CommandList::operator<<(luisa::unique_ptr<Command> &&cmd) noexcept {
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
CommandList &CommandList::operator<<(luisa::move_only_function<void()> &&callback) noexcept {
    _callbacks.emplace_back(std::move(callback));
}
void CommandList::reserve(size_t size) noexcept {
    _commands.reserve(size);
}
void CommandList::clear() noexcept {
    _commands.clear();
    _callbacks.clear();
}
}// namespace luisa::compute
