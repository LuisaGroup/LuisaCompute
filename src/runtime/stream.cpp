//
// Created by Mike Smith on 2021/3/18.
//

#include <utility>

#include <runtime/device.h>
#include <runtime/stream.h>
#include <runtime/command_graph.h>

namespace luisa::compute {

Stream Device::create_stream() noexcept {
    return _create<Stream>();
}

void Stream::_dispatch(CommandList list) noexcept {
    if (auto size = list.size();
        size > 1u && device()->requires_command_reordering()) {
        auto commands = list.steal_commands();
        Clock clock;
        // CommandGraph graph{device()};
        // for (auto cmd : commands) { graph.add(cmd); }
        // auto lists = graph.schedule();
        // LUISA_INFO(
        //     "Reordered {} commands into {} list(s) in {} ms.",
        //     commands.size(), lists.size(), clock.toc());
        // device()->dispatch(handle(), lists);
        // Clock clock;
        for (auto command : commands) {
            command->accept(*reorder_visitor);
        }
        auto lists = reorder_visitor->command_lists();
        LUISA_INFO(
            "Reordered {} commands into {} list(s) in {} ms.",
            commands.size(), lists.size(), clock.toc());
        device()->dispatch(handle(), lists);
        reorder_visitor->clear();
        for (auto cmd : commands) { cmd->recycle(); }
    } else {
        device()->dispatch(handle(), list);
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

Stream &Stream::operator<<(CommandBuffer::Synchronize) noexcept {
    _synchronize();
    return *this;
}

Stream::Stream(Device::Interface *device) noexcept
    : Resource{device, Tag::STREAM, device->create_stream()},
      reorder_visitor{luisa::make_unique<CommandReorderVisitor>(device)} {}

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

Stream::Delegate &&Stream::Delegate::operator<<(CommandBuffer::Synchronize s) &&noexcept {
    _commit();
    *_stream << s;
    return std::move(*this);
}

Stream::Delegate &&Stream::Delegate::operator<<(SwapChain::Present p) &&noexcept {
    _commit();
    *_stream << p;
    return std::move(*this);
}

Stream::Delegate &&Stream::Delegate::operator<<(CommandBuffer::Commit) &&noexcept {
    _commit();
    return std::move(*this);
}

Stream::Delegate &&Stream::Delegate::operator<<(luisa::move_only_function<void()> &&f) &&noexcept {
    _commit();
    *_stream << std::move(f);
    return std::move(*this);
}

Stream &Stream::operator<<(SwapChain::Present p) noexcept {
    device()->present_display_in_stream(handle(), p.chain->handle(), p.frame.handle());
    return *this;
}

Stream &Stream::operator<<(luisa::move_only_function<void()> &&f) noexcept {
    device()->dispatch(handle(), std::move(f));
    return *this;
}

}// namespace luisa::compute
