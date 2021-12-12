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
        if (_head == nullptr) { _head = cmd; }
        if (_tail != nullptr) { _tail->set_next(cmd); }
        for (_tail = cmd; _tail->next() != nullptr; _tail = _tail->next()) {}
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

auto CommandList::remove_all() noexcept {
    std::vector<Command *> ans;
    while (_head != nullptr) {
        ans.push_back(_head);
        _head = _head->next();
        ans.back()->set_next(nullptr);
    }
    _tail = nullptr;
    return ans;
}
std::vector<Command *> CommandList::get_all() const noexcept {
    std::vector<Command *> ans;
    auto ptr = _head;
    while (ptr != nullptr) {
        ans.push_back(ptr);
        ptr = ptr->next();
    }
    return ans;
}

}// namespace luisa::compute
