//
// Created by Mike Smith on 2021/3/18.
//

#include <runtime/command_buffer.h>

namespace luisa::compute {

void CommandBuffer::_recycle() noexcept {
    while (_head != nullptr) {
        auto cmd = _head;
        _head = _head->next();
        cmd->recycle();
    }
}

void CommandBuffer::append(Command *cmd) noexcept {
    if (_head == nullptr) { _head = cmd; }
    _tail = _tail == nullptr ? cmd->tail() : _tail->set_next(cmd);
}

CommandBuffer::CommandBuffer(CommandBuffer &&another) noexcept
    : _head{another._head},
      _tail{another._tail} {
    another._head = nullptr;
    another._tail = nullptr;
}

CommandBuffer &CommandBuffer::operator=(CommandBuffer &&rhs) noexcept {
    if (&rhs != this) {
        _recycle();
        _head = rhs._head;
        _tail = rhs._tail;
        rhs._head = nullptr;
        rhs._tail = nullptr;
    }
    return *this;
}

CommandBuffer::~CommandBuffer() noexcept { _recycle(); }

}// namespace luisa::compute
