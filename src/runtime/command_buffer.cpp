//
// Created by Mike Smith on 2021/7/20.
//

#include <runtime/command_buffer.h>
#include <runtime/stream.h>

namespace luisa::compute {

CommandBuffer::CommandBuffer(Stream *stream) noexcept
    : _stream{stream} {}

CommandBuffer::CommandBuffer(CommandBuffer &&another) noexcept
    : _stream{another._stream},
      _command_list{std::move(another._command_list)} { another._stream = nullptr; }

CommandBuffer::~CommandBuffer() noexcept {
    if (!_command_list.empty()) {
        LUISA_ERROR_WITH_LOCATION(
            "Destructing non-empty command buffer. "
            "Did you forget to commit?");
    }
}

CommandBuffer &CommandBuffer::operator<<(Command *cmd) &noexcept {
    _command_list.append(cmd);
    return *this;
}

void CommandBuffer::_commit() &noexcept {
    if (!_command_list.empty()) {
        _stream->_dispatch(std::move(_command_list));
    }
}

CommandBuffer &CommandBuffer::operator<<(Event::Signal signal) &noexcept {
    _commit();
    *_stream << signal;
    return *this;
}

CommandBuffer &CommandBuffer::operator<<(Event::Wait wait) &noexcept {
    _commit();
    *_stream << wait;
    return *this;
}

CommandBuffer &CommandBuffer::operator<<(SwapChain::Present p) &noexcept {
    _commit();
    *_stream << p;
    return *this;
}

CommandBuffer &CommandBuffer::operator<<(CommandBuffer::Commit) &noexcept {
    _commit();
    return *this;
}

CommandBuffer &CommandBuffer::operator<<(CommandBuffer::Synchronize) &noexcept {
    synchronize();
    return *this;
}

void CommandBuffer::synchronize() &noexcept {
    _commit();
    _stream->synchronize();
}

CommandBuffer &CommandBuffer::operator<<(luisa::move_only_function<void()> &&f) &noexcept {
    _commit();
    *_stream << std::move(f);
    return *this;
}

}// namespace luisa::compute
