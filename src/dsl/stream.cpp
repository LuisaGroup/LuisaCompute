//
// Created by Mike Smith on 2021/3/18.
//

#include <dsl/stream.h>

#include <utility>

namespace luisa::compute::dsl {

void Stream::_dispatch(CommandBuffer command_buffer, std::function<void()> callback) noexcept {
    _device->dispatch(_handle, std::move(command_buffer), std::move(callback));
}

Stream::Delegate Stream::operator<<(CommandHandle cmd) noexcept {
    Delegate delegate{this};
    delegate << std::move(cmd);
    return delegate;
}

Stream::Delegate Stream::operator<<(std::function<void()> f) noexcept {
    Delegate delegate{this};
    delegate << std::move(f);
    return delegate;
}

void Stream::operator<<(SynchronizeToken) { _device->synchronize_stream(_handle); }

Stream::Stream(Stream &&s) noexcept
    : _device{s._device}, _handle{s._handle} { s._device = nullptr; }

Stream::~Stream() noexcept {
    if (_device != nullptr) { _device->dispose_stream(_handle); }
}

Stream &Stream::operator=(Stream &&rhs) noexcept {
    _device = rhs._device;
    _handle = rhs._handle;
    rhs._device = nullptr;
    return *this;
}

Stream::Delegate::~Delegate() noexcept { _commit(); }

Stream::Delegate &Stream::Delegate::operator<<(CommandHandle cmd) noexcept {
    if (_callback) { _commit(); }
    _command_buffer.append(std::move(cmd));
    return *this;
}

Stream::Delegate &Stream::Delegate::operator<<(std::function<void()> f) noexcept {
    if (_callback) {
        _callback = [f = std::move(f), callback = std::move(_callback)] {
            callback();
            f();
        };
    } else { _callback = std::move(f); }
    return *this;
}

Stream::Delegate::Delegate(Stream *s) noexcept : _stream{s} {}

void Stream::Delegate::operator<<(Stream::SynchronizeToken) noexcept {
    _commit();
    *_stream << SynchronizeToken{};
}

void Stream::Delegate::_commit() noexcept {
    if (!_command_buffer.empty() || _callback) {
        LUISA_VERBOSE_WITH_LOCATION(
            "Commit {} command{} to stream #{} with{} callback.",
            _command_buffer.size(),
            _command_buffer.size() == 1u ? "" : "s",
            _stream->_handle,
            _callback ? "" : "out");
        _stream->_dispatch(std::move(_command_buffer), std::move(_callback));
        _command_buffer = {};
        _callback = {};
    }
}

}// namespace luisa::compute
