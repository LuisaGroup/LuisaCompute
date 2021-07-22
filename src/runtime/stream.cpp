//
// Created by Mike Smith on 2021/3/18.
//

#include <utility>
#include <runtime/stream.h>

namespace luisa::compute {

Stream Device::create_stream() noexcept {
    return _create<Stream>();
}

void Stream::_dispatch(CommandList command_buffer) noexcept {
    _device->dispatch(_handle, std::move(command_buffer));
}

Stream::Delegate Stream::operator<<(Command *cmd) noexcept {
    return Delegate{this} << cmd;
}

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

}// namespace luisa::compute
