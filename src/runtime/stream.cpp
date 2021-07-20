//
// Created by Mike Smith on 2021/3/18.
//

#include <utility>
#include <runtime/stream.h>

namespace luisa::compute {

void Stream::_dispatch(CommandList command_buffer) noexcept {
    _device->dispatch(_handle, std::move(command_buffer));
}

Stream::Delegate Stream::operator<<(Command *cmd) noexcept {
    return Delegate{this} << cmd;
}

Stream::Stream(Stream &&s) noexcept
    : _device{std::move(s._device)},
      _handle{s._handle} {}

Stream::~Stream() noexcept { _destroy(); }

Stream &Stream::operator=(Stream &&rhs) noexcept {
    if (this != &rhs) {
        _destroy();
        _device = std::move(rhs._device);
        _handle = rhs._handle;
    }
    return *this;
}

void Stream::_synchronize() noexcept { _device->synchronize_stream(_handle); }

Stream &Stream::operator<<(Event::Signal signal) noexcept {
    _device->signal_event(signal.handle, _handle);
    return *this;
}

Stream &Stream::operator<<(Event::Wait wait) noexcept {
    _device->wait_event(wait.handle, _handle);
    return *this;
}

void Stream::_destroy() noexcept {
    _synchronize();
    if (*this) { _device->destroy_stream(_handle); }
}

Stream &Stream::operator<<(Stream::Synchronize) noexcept {
    _synchronize();
    return *this;
}

Stream::Delegate::~Delegate() noexcept { _commit(); }

Stream::Delegate::Delegate(Stream *s) noexcept : _stream{s} {}

void Stream::Delegate::_commit() noexcept {
    if (!_command_list.empty()) {
        _stream->_dispatch(std::move(_command_list));
    }
}

Stream::Delegate::Delegate(Stream::Delegate &&s) noexcept
    : _stream{s._stream},
      _command_list{std::move(s._command_list)} { s._stream = nullptr; }

Stream::Delegate &&Stream::Delegate::operator<<(Command *cmd) &&noexcept {
    _command_list.append(cmd);
    return std::move(*this);
}

Stream::Delegate &&Stream::Delegate::operator<<(Event::Signal signal) &&noexcept {
    _commit();
    *_stream << signal;
    return std::move(*this);
}

Stream::Delegate &&Stream::Delegate::operator<<(Event::Wait wait) &&noexcept {
    _commit();
    *_stream << wait;
    return std::move(*this);
}

Stream::Delegate &&Stream::Delegate::operator<<(Stream::Synchronize) &&noexcept {
    _commit();
    *_stream << Synchronize{};
    return std::move(*this);
}

Stream::CommandBuffer::CommandBuffer(Stream *stream) noexcept
    : _stream{stream} {}

Stream::CommandBuffer::CommandBuffer(Stream::CommandBuffer &&another) noexcept
    : _stream{another._stream},
      _command_list{std::move(another._command_list)} { another._stream = nullptr; }

Stream::CommandBuffer::~CommandBuffer() noexcept {
    if (!_command_list.empty()) {
        LUISA_ERROR_WITH_LOCATION(
            "Destructing non-empty command buffer. "
            "Did you forget to commit?");
    }
}

Stream::CommandBuffer &Stream::CommandBuffer::operator<<(Command *cmd) &noexcept {
    _command_list.append(cmd);
    return *this;
}

void Stream::CommandBuffer::_commit() &noexcept {
    if (!_command_list.empty()) {
        _stream->_dispatch(std::move(_command_list));
    }
}

Stream::CommandBuffer &Stream::CommandBuffer::operator<<(Event::Signal signal) &noexcept {
    _commit();
    *_stream << signal;
    return *this;
}

Stream::CommandBuffer &Stream::CommandBuffer::operator<<(Event::Wait wait) &noexcept {
    _commit();
    *_stream << wait;
    return *this;
}

Stream::CommandBuffer &Stream::CommandBuffer::operator<<(Stream::CommandBuffer::Commit) &noexcept {
    _commit();
    return *this;
}

}// namespace luisa::compute
