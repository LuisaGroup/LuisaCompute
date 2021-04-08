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

Stream &Stream::operator<<(std::function<void()> f) noexcept {
    _dispatch(std::move(f));
    return *this;
}

Stream::Stream(Stream &&s) noexcept
    : _device{s._device}, _handle{s._handle} { s._device = nullptr; }

Stream::~Stream() noexcept {
    if (_device != nullptr) {
        _device->dispose_stream(_handle);
    }
}

Stream &Stream::operator=(Stream &&rhs) noexcept {
    if (this != &rhs) {
        _device->dispose_stream(_handle);
        _device = rhs._device;
        _handle = rhs._handle;
        rhs._device = nullptr;
    }
    return *this;
}

void Stream::synchronize() noexcept { _device->synchronize_stream(_handle); }

void Stream::_dispatch(std::function<void()> f) noexcept {
    _device->dispatch(_handle, std::move(f));
}

Stream::Delegate::~Delegate() noexcept { _commit(); }

Stream::Delegate &Stream::Delegate::operator<<(CommandHandle cmd) noexcept {
    _command_buffer.append(std::move(cmd));
    return *this;
}

Stream &Stream::Delegate::operator<<(std::function<void()> f) noexcept {
    auto s = _stream;
    _commit();
    return *s << std::move(f);
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

Stream::Delegate &Stream::Delegate::operator=(Stream::Delegate &&rhs) {
    if (this != &rhs) {
        _commit();
        _stream = rhs._stream;
        _command_buffer = std::move(rhs._command_buffer);
        rhs._stream = nullptr;
    }
    return *this;
}

}// namespace luisa::compute
