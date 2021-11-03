//
// Created by Mike Smith on 2021/3/18.
//

#include <runtime/command_list.h>

namespace luisa::compute {

void CommandList::_recycle() noexcept {
    while (_head != nullptr) {
        auto cmd = _head;
        _head = _head->next();
        cmd->recycle();
    }
}

void CommandList::append(Command *cmd) noexcept {
    if (cmd != nullptr) {
        if (cmd->next() != nullptr) [[unlikely]] {
            LUISA_ERROR_WITH_LOCATION(
                "Adding composed command to command list is not allowed.");
        }
        if (_head == nullptr) { _head = cmd; }
        if (_tail != nullptr) { _tail->set_next(cmd); }
        _tail = cmd;
    }
}

CommandList::CommandList(CommandList &&another) noexcept
    : _head{another._head},
      _tail{another._tail} {
    another._head = nullptr;
    another._tail = nullptr;
}

CommandList &CommandList::operator=(CommandList &&rhs) noexcept {
    if (&rhs != this) [[likely]] {
        _recycle();
        _head = rhs._head;
        _tail = rhs._tail;
        rhs._head = nullptr;
        rhs._tail = nullptr;
    }
    return *this;
}

CommandList::~CommandList() noexcept { _recycle(); }

}// namespace luisa::compute
