//
// Created by Mike Smith on 2021/3/18.
//

#include <utility>
#include <runtime/device.h>
#include <runtime/stream.h>
#include <core/logging.h>

namespace luisa::compute {

Stream Device::create_stream(StreamTag stream_tag) noexcept {
    return _create<Stream>(stream_tag);
}

void Stream::_dispatch(CommandList &&list) noexcept {
#ifndef NDEBUG
    for (auto &&i : list) {
        if (static_cast<uint32_t>(i->stream_tag()) < static_cast<uint32_t>(_stream_tag)) {
            auto kNames = {
                "graphics",
                "compute",
                "copy"};
            LUISA_ERROR(
                "Command of type {} in stream of type {} not allowed!",
                kNames.begin()[static_cast<uint32_t>(i->stream_tag())],
                kNames.begin()[static_cast<uint32_t>(_stream_tag)]);
        }
    }
#endif
    if (_callbacks.empty()) {
        device()->dispatch(handle(), std::move(list));
    } else {
        device()->dispatch(handle(), std::move(list), std::move(_callbacks));
    }
}

Stream::Delegate Stream::operator<<(luisa::unique_ptr<Command> &&cmd) noexcept {
    return Delegate{this} << std::move(cmd);
}

void Stream::_synchronize() noexcept { device()->synchronize_stream(handle()); }

Stream &Stream::operator<<(Event::Signal &&signal) noexcept {
    device()->signal_event(signal.handle, handle());
    return *this;
}
Stream &Stream::operator<<(Event::Wait &&wait) noexcept {
    device()->wait_event(wait.handle, handle());
    return *this;
}

Stream::Stream(DeviceInterface *device, StreamTag stream_tag) noexcept
    : Resource{device, Tag::STREAM, device->create_stream(stream_tag)}, _stream_tag(stream_tag) {}

Stream::Delegate::Delegate(Stream *s) noexcept : _stream{s} {}
Stream::Delegate::~Delegate() noexcept { _commit(); }

void Stream::Delegate::_commit() noexcept {
    if (!_command_list.empty()) [[likely]] {
        _stream->_dispatch(std::move(_command_list));
    }
}

Stream::Delegate::Delegate(Stream::Delegate &&s) noexcept
    : _stream{s._stream},
      _command_list{std::move(s._command_list)} { s._stream = nullptr; }

Stream::Delegate &&Stream::Delegate::operator<<(luisa::unique_ptr<Command> &&cmd) &&noexcept {
    _command_list.append(std::move(cmd));
    return std::move(*this);
}

Stream::Delegate &&Stream::Delegate::operator<<(CommandList::Commit &&commit) &&noexcept {
    if (!commit.cmd_list.empty()) [[likely]] {
        _stream->_dispatch(std::move(commit.cmd_list));
    }
}

Stream::Delegate &&Stream::Delegate::operator<<(Event::Signal &&signal) &&noexcept {
    _commit();
    *_stream << std::move(signal);
    return std::move(*this);
}

Stream::Delegate &&Stream::Delegate::operator<<(Event::Wait &&wait) &&noexcept {
    _commit();
    *_stream << std::move(wait);
    return std::move(*this);
}

Stream::Delegate &&Stream::Delegate::operator<<(SwapChain::Present &&p) &&noexcept {
    _commit();
    *_stream << std::move(p);
    return std::move(*this);
}

Stream::Delegate &&Stream::Delegate::operator<<(luisa::move_only_function<void()> &&f) &&noexcept {
    *_stream << std::move(f);
    return std::move(*this);
}

Stream &Stream::operator<<(SwapChain::Present &&p) noexcept {
#ifndef NDEBUG
    if (_stream_tag != StreamTag::GRAPHICS) {
        LUISA_ERROR("Present only allowed in stream of graphics type!");
    }
#endif
    device()->present_display_in_stream(handle(), p.chain->handle(), p.frame.handle());
    return *this;
}

Stream &Stream::operator<<(luisa::move_only_function<void()> &&f) noexcept {
    _callbacks.emplace_back(std::move(f));
    return *this;
}

Stream &Stream::operator<<(CommandList::Commit &&commit) noexcept {
    if (!commit.cmd_list.empty()) [[likely]] {
        _dispatch(std::move(commit.cmd_list));
    }
}

}// namespace luisa::compute
