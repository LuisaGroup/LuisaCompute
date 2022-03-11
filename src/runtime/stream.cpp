//
// Created by Mike Smith on 2021/3/18.
//

#include <utility>
#include <runtime/device.h>
#include <runtime/stream.h>

namespace luisa::compute {

Stream Device::create_stream() noexcept {
    return _create<Stream>();
}

void Stream::_dispatch(CommandList list) noexcept {
    if (auto size = list.size();
        size > 1u && device()->requires_command_reordering()) {
        auto commands = list.steal_commands();
        for (auto command : commands) {
            command->accept(*reorder_visitor);
        }
        auto lists = reorder_visitor->command_lists();
        device()->dispatch(handle(), lists);
        reorder_visitor->clear();
        for (auto command : commands) {
            command->recycle();
        }


    } else {
        device()->dispatch(handle(), list);
    }
}
void Stream::_dispatch(CommandList list, luisa::move_only_function<void()> &&func) noexcept {
    if (auto size = list.size();
        size > 1u && device()->requires_command_reordering()) {
        auto commands = list.steal_commands();
        for (auto command : commands) {
            command->accept(*reorder_visitor);
        }
        auto lists = reorder_visitor->command_lists();
        device()->dispatch(handle(), lists, std::move(func));
        reorder_visitor->clear();
        for (auto command : commands) {
            command->recycle();
        }

    } else {
        device()->dispatch(handle(), list, std::move(func));
    }
}


Stream::Delegate Stream::operator<<(Command *cmd) noexcept {
    return Delegate{this} << cmd;
}

void Stream::_synchronize() noexcept { device()->synchronize_stream(handle()); }

Stream &Stream::operator<<(Event::Signal signal) noexcept {
    device()->signal_event(signal.handle, handle());
    return *this;
}
Stream &Stream::operator<<(Event::Wait wait) noexcept {
    device()->wait_event(wait.handle, handle());
    return *this;
}


Stream &Stream::operator<<(Stream::Synchronize) noexcept {
    _synchronize();
    return *this;
}

Stream::Stream(Device::Interface *device) noexcept
    : Resource{device, Tag::STREAM, device->create_stream()},
      reorder_visitor{luisa::make_unique<CommandReorderVisitor>(device)} {}

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

Stream::Delegate &&Stream::Delegate::operator<<(CommandBuffer::Commit) &&noexcept {
    _commit();
    return std::move(*this);
}

}// namespace luisa::compute
