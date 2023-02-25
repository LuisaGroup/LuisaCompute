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
    if (!list.empty()) {
#ifndef NDEBUG
        constexpr luisa::string_view tag_names[]{"graphics", "compute", "copy"};
        for (auto &&i : list.commands()) {
            auto cmd_tag = luisa::to_underlying(i->stream_tag());
            auto s_tag = luisa::to_underlying(_stream_tag);
            LUISA_ASSERT(cmd_tag >= s_tag,
                         "Command of type {} in stream of type {} not allowed!",
                         tag_names[cmd_tag], tag_names[s_tag]);
        }
#endif
        device()->dispatch(handle(), std::move(list));
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
    *_stream << _command_list.commit();
}

Stream::Delegate::Delegate(Stream::Delegate &&s) noexcept
    : _stream{s._stream},
      _command_list{std::move(s._command_list)} { s._stream = nullptr; }

Stream::Delegate &&Stream::Delegate::operator<<(luisa::unique_ptr<Command> &&cmd) &&noexcept {
    if (!_command_list.callbacks().empty()) { _commit(); }
    _command_list.append(std::move(cmd));
    return std::move(*this);
}

Stream &Stream::Delegate::operator<<(CommandList::Commit &&commit) &&noexcept {
    _commit();
    return *_stream << std::move(commit);
}

Stream &Stream::Delegate::operator<<(Event::Signal &&signal) &&noexcept {
    _commit();
    return *_stream << std::move(signal);
}

Stream &Stream::Delegate::operator<<(Event::Wait &&wait) &&noexcept {
    _commit();
    return *_stream << std::move(wait);
}

Stream &Stream::Delegate::operator<<(SwapChain::Present &&p) &&noexcept {
    _commit();
    return *_stream << std::move(p);
}

Stream::Delegate &&Stream::Delegate::operator<<(luisa::move_only_function<void()> &&f) &&noexcept {
    _command_list.append(std::move(f));
    return std::move(*this);
}

Stream &Stream::Delegate::operator<<(Stream::Synchronize &&) &&noexcept {
    _commit();
    return *_stream << Stream::Synchronize{};
}

Stream &Stream::Delegate::operator<<(Stream::Commit &&) &&noexcept {
    _commit();
    return *_stream;
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

Stream::Delegate Stream::operator<<(luisa::move_only_function<void()> &&f) noexcept {
    return Delegate{this} << std::move(f);
}

Stream &Stream::operator<<(CommandList::Commit &&commit) noexcept {
    _dispatch(std::move(commit._list));
    return *this;
}

Stream &Stream::operator<<(Stream::Synchronize &&) noexcept {
    _synchronize();
    return *this;
}

}// namespace luisa::compute
