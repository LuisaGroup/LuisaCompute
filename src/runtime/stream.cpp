#include <utility>

#include <luisa/core/logging.h>
#include <luisa/core/magic_enum.h>
#include <luisa/runtime/device.h>
#include <luisa/runtime/stream.h>

namespace luisa::compute {

Stream Device::create_stream(StreamTag stream_tag) noexcept {
    return _create<Stream>(stream_tag);
}

void Stream::_dispatch(CommandList &&list) noexcept {
    _check_is_valid();
    if (!list.empty()) {
#ifndef NDEBUG
        for (auto &&i : list.commands()) {
            auto cmd_tag = luisa::to_underlying(i->stream_tag());
            auto s_tag = luisa::to_underlying(_stream_tag);
            LUISA_ASSERT(cmd_tag >= s_tag,
                         "Command of type {} in stream of type {} not allowed!",
                         to_string(i->stream_tag()), to_string(_stream_tag));
        }
#endif
        device()->dispatch(handle(), std::move(list));
    }
}

Stream::Delegate Stream::operator<<(luisa::unique_ptr<Command> &&cmd) noexcept {
    // No Delegate{this}<< here, may boom GCC
    Delegate delegate{this};
    return std::move(delegate) << std::move(cmd);
}

void Stream::_synchronize() noexcept {
    _check_is_valid();
    device()->synchronize_stream(handle());
}

Stream::Stream(DeviceInterface *device, StreamTag stream_tag) noexcept
    : Stream{device, stream_tag, device->create_stream(stream_tag)} {}

Stream::Stream(DeviceInterface *device, StreamTag stream_tag, const ResourceCreationInfo &handle) noexcept
    : Resource{device, Tag::STREAM, handle},
      _stream_tag(stream_tag) {}

Stream::Delegate::Delegate(Stream *s) noexcept : _stream{s} {}

Stream::Delegate::~Delegate() noexcept {
    _commit();
}

void Stream::Delegate::_commit() noexcept {
    if (_stream != nullptr && !_command_list.empty()) {
        *_stream << _command_list.commit();
    }
}

Stream::Delegate::Delegate(Stream::Delegate &&s) noexcept
    : _stream{s._stream},
      _command_list{std::move(s._command_list)} {
    s._stream = nullptr;
}

Stream::Delegate Stream::Delegate::operator<<(luisa::unique_ptr<Command> &&cmd) && noexcept {
    if (!_command_list.callbacks().empty()) { _commit(); }
    _command_list.append(std::move(cmd));
    return std::move(*this);
}

Stream &Stream::Delegate::operator<<(CommandList::Commit &&commit) && noexcept {
    _commit();
    return *_stream << std::move(commit);
}

Stream::Delegate Stream::Delegate::operator<<(luisa::move_only_function<void()> &&f) && noexcept {
    _command_list.add_callback(std::move(f));
    return std::move(*this);
}

Stream &Stream::Delegate::operator<<(Stream::Synchronize &&) && noexcept {
    _commit();
    return *_stream << Stream::Synchronize{};
}

Stream &Stream::Delegate::operator<<(Stream::Commit &&) && noexcept {
    _commit();
    return *_stream;
}

Stream::Delegate Stream::operator<<(luisa::move_only_function<void()> &&f) noexcept {
    // No Delegate{this}<< here, may boom GCC
    Delegate delegate{this};
    return std::move(delegate) << std::move(f);
}

Stream &Stream::operator<<(CommandList::Commit &&commit) noexcept {
    _dispatch(std::move(commit).command_list());
    return *this;
}

Stream &Stream::operator<<(Stream::Synchronize &&) noexcept {
    _synchronize();
    return *this;
}

Stream::~Stream() noexcept {
    if (*this) { device()->destroy_stream(handle()); }
}

}// namespace luisa::compute
