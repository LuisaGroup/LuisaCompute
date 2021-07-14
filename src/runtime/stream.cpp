//
// Created by Mike Smith on 2021/3/18.
//

#include <utility>
#include <runtime/stream.h>

namespace luisa::compute {

void Stream::_dispatch(CommandBuffer command_buffer) noexcept {
    _device->dispatch(_handle, std::move(command_buffer));
}

Stream::Delegate Stream::operator<<(CommandHandle cmd) noexcept {
    Delegate delegate{this};
    delegate << std::move(cmd);
    return delegate;
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

void Stream::synchronize() noexcept { _device->synchronize_stream(_handle); }

Stream &Stream::operator<<(Event::Signal signal) noexcept {
    _device->signal_event(signal.handle, _handle);
    return *this;
}

Stream &Stream::operator<<(Event::Wait wait) noexcept {
    _device->wait_event(wait.handle, _handle);
    return *this;
}

void Stream::_destroy() noexcept {
    if (*this) { _device->destroy_stream(_handle); }
}

Stream::Delegate::~Delegate() noexcept { _commit(); }

Stream::Delegate &Stream::Delegate::operator<<(CommandHandle cmd) noexcept {
    _command_buffer.append(std::move(cmd));
    return *this;
}

Stream::Delegate::Delegate(Stream *s) noexcept : _stream{s} {}

void Stream::Delegate::_commit() noexcept {
    if (_stream != nullptr && !_command_buffer.empty()) {
        LUISA_VERBOSE_WITH_LOCATION(
            "Commit {} command{} to stream #{}.",
            _command_buffer.size(),
            _command_buffer.size() == 1u ? "" : "s",
            _stream->_handle);
        _stream->_dispatch(std::move(_command_buffer));
        _stream = nullptr;
    }
}

Stream::Delegate::Delegate(Stream::Delegate &&s) noexcept
    : _stream{s._stream},
      _command_buffer{std::move(s._command_buffer)} { s._stream = nullptr; }

Stream &Stream::Delegate::operator<<(Event::Signal signal) noexcept {
    auto &&stream = *_stream;
    _commit();
    return stream << signal;
}

Stream &Stream::Delegate::operator<<(Event::Wait wait) noexcept {
    auto &&stream = *_stream;
    _commit();
    return stream << wait;
}

}// namespace luisa::compute
