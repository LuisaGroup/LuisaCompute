//
// Created by Mike Smith on 2021/3/18.
//

#include <runtime/stream.h>

#include <utility>

namespace luisa::compute {

void Stream::_dispatch(CommandBuffer cb) noexcept { _device->_dispatch(_handle, std::move(cb)); }

Stream::Delegate Stream::operator<<(std::unique_ptr<Command> cmd) noexcept {
    Delegate delegate{this};
    delegate << std::move(cmd);
    return delegate;
}

Stream &Stream::operator<<(std::function<void()> f) noexcept {
    _device->_dispatch(_handle, std::move(f));
    return *this;
}

void Stream::operator<<(SynchronizeToken) { _device->_synchronize_stream(_handle); }

Stream::Stream(Stream &&s) noexcept
    : _device{s._device}, _handle{s._handle} { s._device = nullptr; }

Stream::~Stream() noexcept {
    if (_device != nullptr) { _device->_dispose_stream(_handle); }
}

Stream &Stream::operator=(Stream &&rhs) noexcept {
    _device = rhs._device;
    _handle = rhs._handle;
    rhs._device = nullptr;
    return *this;
}

Stream::Delegate::~Delegate() noexcept { _commit(); }

Stream::Delegate &Stream::Delegate::operator<<(std::unique_ptr<Command> cmd) noexcept {
    _cb.append(std::move(cmd));
    return *this;
}

Stream &Stream::Delegate::operator<<(std::function<void()> f) noexcept {
    auto &&stream = *_stream;
    _commit();
    return stream << std::move(f);
}

Stream::Delegate::Delegate(Stream *s) noexcept : _stream{s} {}

void Stream::Delegate::operator<<(Stream::SynchronizeToken) noexcept {
    auto &stream = *_stream;
    _commit();
    stream << SynchronizeToken{};
}

void Stream::Delegate::_commit() noexcept {
    if (_stream != nullptr && !_cb.empty()) {
        LUISA_VERBOSE_WITH_LOCATION(
            "Commit {} command{} to stream #{}.",
            _cb.size(), _cb.size() == 1u ? "" : "s", _stream->_handle);
        _stream->_dispatch(std::move(_cb));
        _stream = nullptr;
    }
}

}// namespace luisa::compute
